"""
attn_models.py  -- attention at three loci, each with a matched non-attention control
=====================================================================================
EXTENDS fusion_models.py; NEVER edits it. Every pre-existing encoder/fusion at
readout="gap" is DELEGATED to fusion_models verbatim, so the reference arms
(cnn_gru__concat, cnn__concat, cnn_gru__xattn) run the SAME code path as the completed
fusion study -- a cross-study bridge, not a re-implementation. `__main__` asserts
param-count equality for those three.

The completed study tested attention at ONE locus (xattn: CGM queries -> MEAL tokens).
Attention over the CGM SEQUENCE ITSELF was never tested. Three loci here, each isolating
ONE component against a matched control (see PLAN_attention.md):

  readout   `+attnpool`      vs `concat` (GAP)    -- learned vs uniform pooling weights.
            Additive (Bahdanau) attention pooling is a STRICT GENERALIZATION of GAP
            (GAP = the uniform-weight special case), which is what makes this a clean
            one-component test. Split into two layers so the weights are extractable.
  encoder   `cnn_sa`         vs `cnn`             -- a pre-LN TRANSFORMER BLOCK in place of
            a 2nd conv block, behind an IDENTICAL Conv1D(d,5) front-end. Honest scope: the
            treatment is the whole block (MHSA + LN + residuals + GELU FFN + ~3.1k extra
            params), NOT self-attention in isolation. Claimed as such.
  cross     `coattn`         vs `coattn_ctl`      -- the reverse direction (meal tokens
            query the CGM curve) vs an IDENTICALLY-SHAPED meal SELF-attention branch.
            Same parameter count, same 80-d head input, same pooled-meal representation;
            the ONLY difference is whether that branch may read the CGM curve. `__main__`
            asserts the two param counts are equal.
            (`coattn` vs `xattn` is NOT matched -- it also widens the head 56->80 -- so
            that contrast is SECONDARY, decomposed as ctl-xattn = "extra meal branch"
            and coattn-ctl = "the direction".)
  encoder   `sa`             vs `cnn`             -- SECONDARY, not matched: differs in
            projection, positional encoding, locality, normalization, residuals, FFN and
            capacity all at once. An architectural alternative, not a component test.

Positional encoding is FIXED SINUSOIDAL (0 trainable params), locked a priori: a learned
table would add T*d = 1440 free params (~18% of the model) for 913 training samples and
would confound "attention helps" with "more capacity helps".

Keras-3-safe: every raw-tf op lives in a custom Layer.call; multi-output compiles with a
LIST of losses. Output order is ALWAYS [out_30,out_60,out_120]. CGM-encoder layers are
prefixed "cgmenc_" so fusion_ssl.transfer would still work.
=====================================================================================
"""
import tensorflow as tf
from tensorflow.keras import layers, models

import fusion_models as FM
from lightdl_models import _gated_fusion, _heads, _compile  # noqa: F401

HORIZONS = FM.HORIZONS
ENCODERS = FM.ENCODERS + ("sa", "cnn_sa")          # + pure / conv-front self-attention
FUSIONS = FM.FUSIONS + ("coattn", "coattn_ctl")    # + bidirectional cross-modal & its control
READOUTS = ("gap", "attn")

POS_MODE = "sincos"        # LOCKED: fixed sinusoidal, 0 params (see module docstring)
FF_MULT = 2                # transformer FFN expansion
D_ATT = 16                 # additive-attention scoring width for the pooling readout


# ----------------------------- custom layers -----------------------------
class PosEncode(layers.Layer):
    """Add a positional signal to (b,T,d). mode='sincos' (fixed, 0 params) | 'learned'."""
    def __init__(self, mode=POS_MODE, **kw):
        super().__init__(**kw); self.mode = str(mode)

    def build(self, input_shape):
        if self.mode == "learned":
            self.pe = self.add_weight(shape=(1, int(input_shape[1]), int(input_shape[-1])),
                                      initializer="zeros", trainable=True, name="pe")

    def call(self, x):
        if self.mode == "learned":
            return x + self.pe
        T, d = tf.shape(x)[1], tf.shape(x)[-1]
        pos = tf.cast(tf.range(T)[:, None], x.dtype)                     # (T,1)
        i = tf.cast(tf.range(d)[None, :], x.dtype)                       # (1,d)
        ang = pos / tf.pow(tf.constant(10000.0, x.dtype),
                           (2.0 * tf.floor(i / 2.0)) / tf.cast(d, x.dtype))
        even = tf.equal(tf.math.floormod(tf.range(d), 2), 0)             # (d,)
        pe = tf.where(even[None, :], tf.sin(ang), tf.cos(ang))           # (T,d)
        return x + pe[None]

    def get_config(self):
        c = super().get_config(); c["mode"] = self.mode; return c


class AttnScores(layers.Layer):
    """Additive (Bahdanau) attention scores over time -> softmax weights (b,T,1).
    Kept SEPARATE from the weighted sum so the profile is directly extractable for the
    'where on the 60-min window does the model look' figure (attention_profile_model)."""
    def __init__(self, d_att=D_ATT, **kw):
        super().__init__(**kw); self.d_att = int(d_att)

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(shape=(d, self.d_att), initializer="glorot_uniform", name="W")
        self.b = self.add_weight(shape=(self.d_att,), initializer="zeros", name="b")
        self.v = self.add_weight(shape=(self.d_att,), initializer="glorot_uniform", name="v")

    def call(self, x):
        e = tf.tanh(tf.tensordot(x, self.W, [[2], [0]]) + self.b)        # (b,T,a)
        s = tf.tensordot(e, self.v, [[2], [0]])                          # (b,T)
        return tf.nn.softmax(s, axis=1)[:, :, None]                      # (b,T,1), sums to 1

    def get_config(self):
        c = super().get_config(); c["d_att"] = self.d_att; return c


class WeightedSum(layers.Layer):
    """Attention-weighted temporal pooling: sum_t w_t * x_t. GAP is w_t == 1/T."""
    def call(self, inputs):
        x, w = inputs
        return tf.reduce_sum(x * w, axis=1)


# ----------------------------- transformer block -----------------------------
def _sa_block(x, d, n_heads, name):
    """Pre-LN transformer block: MHSA + GELU FFN, both residual. Pre-LN (not post-LN) so
    it trains without a warmup schedule -- there is no LR schedule in this protocol."""
    h = layers.LayerNormalization(name=f"{name}_ln1")(x)
    att = layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(1, d // n_heads),
                                    name=f"{name}_mha")(h, h)            # SELF-attention
    x = layers.Add(name=f"{name}_res1")([x, att])
    h = layers.LayerNormalization(name=f"{name}_ln2")(x)
    f = layers.Dense(d * FF_MULT, activation="gelu", name=f"{name}_ff1")(h)
    f = layers.Dense(d, name=f"{name}_ff2")(f)
    return layers.Add(name=f"{name}_res2")([x, f])


# ----------------------------- CGM encoders -----------------------------
def _encode_cgm(ci, encoder, d, n_heads):
    """(T,1) CGM -> token sequence (T,d). Existing encoders delegate to fusion_models."""
    if encoder in FM.ENCODERS:
        return FM._encode_cgm(ci, encoder, d)              # verbatim: same code path
    if encoder == "sa":
        x = layers.Dense(d, name="cgmenc_sa_proj")(ci)     # no locality prior at all
    elif encoder == "cnn_sa":
        x = layers.Conv1D(d, 5, padding="same", activation="relu",
                          name="cgmenc_cnn")(ci)           # SAME front-end as `cnn`
    else:
        raise ValueError(f"unknown encoder {encoder}")
    x = PosEncode(POS_MODE, name="cgmenc_pos")(x)
    return _sa_block(x, d, n_heads, "cgmenc_sablk")


# ----------------------------- readout -----------------------------
def _pool(seq, readout, prefix="cgm"):
    if readout == "gap":
        return layers.GlobalAveragePooling1D(name=f"{prefix}_gap")(seq)
    w = AttnScores(D_ATT, name=f"{prefix}_attnpool_scores")(seq)
    return WeightedSum(name=f"{prefix}_attnpool")([seq, w])


# ----------------------------- fusion -----------------------------
def _fuse(seq, meal_ctx, meal_tokens, pt_ctx, a, fusion, d, n_heads, readout):
    if readout == "gap" and fusion in FM.FUSIONS:
        return FM._fuse(seq, meal_ctx, meal_tokens, pt_ctx, a, fusion, d, n_heads)

    if fusion in ("concat", "gate"):
        h = layers.Dense(d, activation="relu", name="cgm_proj")(_pool(seq, readout))
        z = layers.Concatenate(name="fuse_concat")([h, meal_ctx, pt_ctx, a])
        return _gated_fusion(z, "glu") if fusion == "gate" else z

    if fusion == "film":
        gamma = layers.Dense(d, name="film_gamma")(meal_ctx)
        beta = layers.Dense(d, name="film_beta")(meal_ctx)
        seq_mod = FM.FiLMSeq(name="film_seq")([seq, gamma, beta])
        h = layers.Dense(d, activation="relu", name="cgm_proj")(_pool(seq_mod, readout))
        return layers.Concatenate(name="fuse_concat")([h, pt_ctx, a])

    if fusion in ("xattn", "coattn", "coattn_ctl"):
        # direction 1 (identical to fusion_models' xattn): CGM timesteps query MEAL tokens
        q = layers.Dense(d, name="xattn_q")(seq)
        c2m = layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(1, d // n_heads),
                                        name="xattn_mha")(q, meal_tokens, meal_tokens)
        xc = layers.LayerNormalization(name="xattn_ln")(layers.Add(name="xattn_res")([q, c2m]))
        h = layers.Dense(d, activation="relu", name="cgm_proj")(_pool(xc, readout))
        parts = [h, pt_ctx, a]
        if fusion in ("coattn", "coattn_ctl"):
            # direction 2: the MEAL tokens attend, then pool into the head.
            #   coattn      -> keys/values are the CGM curve   (bidirectional: the treatment)
            #   coattn_ctl  -> keys/values are the meal tokens (meal SELF-attention: control)
            # Identical shapes, identical parameter count, identical 80-d head input. The
            # ONLY difference is whether this branch can see the CGM curve, so the pair
            # isolates the DIRECTION rather than "one more branch + a wider head".
            kv = q if fusion == "coattn" else meal_tokens
            m2c = layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(1, d // n_heads),
                                            name="coattn_mha")(meal_tokens, kv, kv)
            xm = layers.LayerNormalization(name="coattn_ln")(
                layers.Add(name="coattn_res")([meal_tokens, m2c]))
            parts.insert(1, layers.GlobalAveragePooling1D(name="coattn_gap")(xm))
        return layers.Concatenate(name="fuse_concat")(parts)

    raise ValueError(f"unknown fusion {fusion}")


# ----------------------------- dispatcher -----------------------------
def build(config, timesteps, n_dyn, n_static, d=24, d_a=8, n_heads=4, lr=1e-3,
          dyn_mode="gated", readout="gap"):
    """config = '<encoder>__<fusion>' (e.g. 'cnn_sa__coattn'); readout in {gap,attn}.
    Inputs [cgm(T,1), dyn(T,n_dyn), static(n_static)]; outputs [out_30,out_60,out_120]."""
    encoder, fusion = config.split("__")
    assert encoder in ENCODERS and fusion in FUSIONS, config
    assert readout in READOUTS, readout
    ci = layers.Input(shape=(timesteps, 1), name="cgm_input")
    di = layers.Input(shape=(timesteps, n_dyn), name="dyn_input")
    si = layers.Input(shape=(n_static,), name="static_input")

    seq = _encode_cgm(ci, encoder, d, n_heads)
    meal_ctx, meal_tokens, pt_ctx = FM._contexts(si, d)
    a = FM._activity(di, d_a, dyn_mode)
    z = _fuse(seq, meal_ctx, meal_tokens, pt_ctx, a, fusion, d, n_heads, readout)

    hh = layers.Dense(32, activation="relu", name="head_dense")(z)
    hh = layers.Dropout(0.3, name="head_dropout")(hh)
    model = models.Model([ci, di, si], _heads(hh, aux=False),
                         name=f"attn_{encoder}_{fusion}_{readout}_{dyn_mode}")
    return _compile(model, aux=False, lr=lr)


def build_cgm_encoder(config, timesteps, d=24, n_heads=4):
    """Standalone CGM encoder (same cgmenc_* layers), for latency/param accounting."""
    encoder = config.split("__")[0]
    ci = layers.Input(shape=(timesteps, 1), name="cgm_input")
    return models.Model(ci, _encode_cgm(ci, encoder, d, n_heads), name=f"cgmenc_{encoder}")


# ----------------------------- attention-weight extraction -----------------------------
SCORES_LAYER = "cgm_attnpool_scores"


def has_attnpool(model):
    return any(l.name == SCORES_LAYER for l in model.layers)


# Receptive field of position t in the ENCODER STATE sequence the readout pools over.
# This governs how far an attention weight can honestly be read back to raw minutes.
RECEPTIVE_FIELD = {
    "cnn": ("local", 7),        # Conv1D(k=5,'same') -> Conv1D(k=3,'same'): state t sees t+/-3
    "cnn_sa": ("global", None),  # a self-attention block mixes all T positions
    "sa": ("global", None),
    "cnn_gru": ("cumulative", None),   # GRU state t summarizes minutes 0..t -- NOT minute t
    "gru": ("cumulative", None),
    "bilstm": ("global", None),
    "tcn": ("local", None),
}


def attention_profile_model(model):
    """Model mapping the same inputs -> (b,T,1) softmax pooling weights, or None if this
    architecture uses GAP. Weights sum to 1 over the T encoder STATES; the GAP null is the
    uniform profile 1/T.

    READ THIS BEFORE INTERPRETING THE OUTPUT. These are weights over ENCODER STATES, not
    over raw CGM minutes. Only for `cnn` is the mapping close to positional (receptive
    field 7 min, so state t ~ minutes t+/-3). For `cnn_gru` the state at t is a CUMULATIVE
    summary of minutes 0..t, so "mass on the last 10 states" does NOT mean "the last 10
    minutes carried the prediction" -- a late state already contains the early curve.
    Attention weights here are extractable, not identifiable explanations. The faithful
    input-level measurement is the occlusion sweep in attn_run.py (`attn_occlusion_*.csv`),
    which perturbs the raw window and measures the actual R2 loss."""
    if not has_attnpool(model):
        return None
    return tf.keras.Model(model.inputs, model.get_layer(SCORES_LAYER).output)


# ----------------------------- self-test -----------------------------
if __name__ == "__main__":
    T, NDYN, NSTAT = 60, 3, 17

    # 1) delegation must reproduce fusion_models EXACTLY for the reference arms
    print("delegation check (attn_models vs fusion_models, readout=gap):")
    for cfg in ("cnn_gru__concat", "cnn__concat", "cnn_gru__xattn"):
        tf.keras.backend.clear_session()
        a = build(cfg, T, NDYN, NSTAT).count_params()
        tf.keras.backend.clear_session()
        b = FM.build(cfg, T, NDYN, NSTAT).count_params()
        assert a == b, f"{cfg}: {a} != {b} -- delegation drifted, cross-study bridge broken"
        print(f"  {cfg:26s} {a:,}  OK")

    # 1b) PRIMARY-3 is only a direction test if treatment and control are the same size
    tf.keras.backend.clear_session()
    ca = build("cnn_gru__coattn", T, NDYN, NSTAT).count_params()
    tf.keras.backend.clear_session()
    cc = build("cnn_gru__coattn_ctl", T, NDYN, NSTAT).count_params()
    assert ca == cc, (f"coattn {ca:,} != coattn_ctl {cc:,} -- the primary-3 control is NOT "
                      f"parameter-matched, so the contrast would confound direction with size")
    print(f"  coattn == coattn_ctl       {ca:,}  OK (matched control)")

    # 2) param table for every config this study fits
    print(f"\n{'config':30s} {'params':>8s}")
    for cfg, ro in [("cnn_gru__concat", "gap"), ("cnn_gru__concat", "attn"),
                    ("cnn__concat", "gap"), ("cnn__concat", "attn"),
                    ("cnn_sa__concat", "gap"), ("sa__concat", "gap"),
                    ("cnn_gru__xattn", "gap"), ("cnn_gru__coattn_ctl", "gap"),
                    ("cnn_gru__coattn", "gap"), ("cnn_sa__coattn", "gap")]:
        tf.keras.backend.clear_session()
        m = build(cfg, T, NDYN, NSTAT, readout=ro)
        nm = cfg + ("+attnpool" if ro == "attn" else "")
        print(f"{nm:30s} {m.count_params():>8,}")

    # 3) the attention profile must exist, be (b,T,1) and sum to 1
    import numpy as np
    tf.keras.backend.clear_session()
    m = build("cnn_gru__concat", T, NDYN, NSTAT, readout="attn")
    pm = attention_profile_model(m)
    w = pm.predict([np.zeros((4, T, 1), "float32"), np.zeros((4, T, NDYN), "float32"),
                    np.zeros((4, NSTAT), "float32")], verbose=0)
    print(f"\nprofile shape {w.shape}, row sums {np.round(w.sum(axis=1).ravel(), 5)} (must be 1.0)")
    assert w.shape == (4, T, 1) and np.allclose(w.sum(axis=1), 1.0, atol=1e-4)
    print("attn_models self-test OK")
