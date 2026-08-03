"""
fusion_models.py  -- modular lightweight multimodal builders (encoder x fusion)
===============================================================================
The RF ablation's design brief, made into architecture (the brief is RF-ablation-
CONDITIONAL -- see PLAN_fusion.md):
  * CGM trajectory and MEAL are the co-equal carriers -> fuse them by CONDITIONING
    the CGM *sequence* on the MEAL, before temporal pooling (FiLM / cross-attention),
    so the mechanism changes HOW THE CURVE IS READ, not just a pooled vector.
  * activity/HR was non-additive (RF)              -> the dyn branch is separable: a
    learned scalar gate (`act_gate`), and a hard `no_dyn` control + a frozen test-time
    dyn-ablation (in fusion_run.py) make the effect IDENTIFIABLE (the raw scalar alone
    is not -- up/downstream weights can compensate).
  * demographics/time add little                   -> person+time are a compact side
    context concatenated downstream, never used to condition the CGM curve.

Mechanism contrast (all fusions see CGM + MEAL + person/time + gated activity; they
differ ONLY in how MEAL combines with CGM):
  concat : pooled-CGM concatenated with meal context (additive)
  gate   : feature-wise GLU over that concatenation  (NOT a scalar modality gate)
  film   : MEAL -> (gamma,beta) modulate the CGM SEQUENCE per-timestep, THEN pool
  xattn  : CGM timesteps (queries) attend to MEAL tokens (keys/values), THEN pool

Param budgets are reported, not claimed equal (encoders differ inherently); the fusion
HEAD adds only a small, reported delta over concat.

Keras-3-safe: every raw-tf op lives in a custom Layer.call; multi-output compiles with
a LIST of losses. Output order is ALWAYS [out_30,out_60,out_120]. CGM-encoder layers are
prefixed "cgmenc_" so fusion_ssl.py transfers pretrained encoder weights by name.
===============================================================================
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from lightdl_models import _tcn_block, _gated_fusion, _heads, _compile  # noqa: F401

HORIZONS = (30, 60, 120)
ENCODERS = ("bilstm", "gru", "cnn", "cnn_gru", "tcn")
FUSIONS = ("concat", "gate", "film", "xattn")

# static-feature layout (verified vs static_names): person 0-3, meal 4-8, time 9-16
PERSON = (0, 4)
MEAL = (4, 9)
TIME = (9, 17)
N_MEAL_TOKENS = 4          # meal (5 feats) -> K learned tokens for cross-attention
GATE_OFF_LOGIT = -20.0     # sigmoid(-20) ~ 0 : a hard "no activity" gate


# ----------------------------- custom layers -----------------------------
class SliceCols(layers.Layer):
    """Select a contiguous static-feature block [start:end] (raw slice in call)."""
    def __init__(self, start, end, **kw):
        super().__init__(**kw); self.start = int(start); self.end = int(end)

    def call(self, x):
        return x[:, self.start:self.end]

    def get_config(self):
        c = super().get_config(); c["start"] = self.start; c["end"] = self.end; return c


class ScalarGate(layers.Layer):
    """Multiply a branch by sigmoid(scalar). One learned number in [0,1]. Set
    trainable_gate=False with init_logit=GATE_OFF_LOGIT for a hard `no_dyn` control that
    keeps the dyn input connected but contributes ~0."""
    def __init__(self, init_logit=0.0, trainable_gate=True, **kw):
        super().__init__(**kw); self.init_logit = float(init_logit); self.trainable_gate = bool(trainable_gate)

    def build(self, input_shape):
        self.g = self.add_weight(shape=(), initializer=tf.keras.initializers.Constant(self.init_logit),
                                 trainable=self.trainable_gate, name="gate_logit")

    def call(self, x):
        return x * tf.sigmoid(self.g)

    def gate_value(self):
        return float(tf.sigmoid(self.g).numpy())

    def get_config(self):
        c = super().get_config(); c["init_logit"] = self.init_logit; c["trainable_gate"] = self.trainable_gate
        return c


class FiLMSeq(layers.Layer):
    """Feature-wise linear modulation of a SEQUENCE: seq*gamma + beta, broadcast over
    time (gamma,beta are per-channel, shared across timesteps)."""
    def call(self, inputs):
        seq, gamma, beta = inputs                     # (b,T,d), (b,d), (b,d)
        return seq * gamma[:, None, :] + beta[:, None, :]


# ----------------------------- CGM encoders (return a token sequence) -----------------------------
def _encode_cgm(ci, encoder, d):
    """Map (T,1) CGM -> sequence (T', d). Named cgmenc_* for SSL weight transfer."""
    if encoder == "bilstm":
        return layers.Bidirectional(layers.LSTM(d // 2, return_sequences=True),
                                    name="cgmenc_bilstm")(ci)
    if encoder == "gru":
        return layers.GRU(d, return_sequences=True, name="cgmenc_gru")(ci)
    if encoder == "cnn":
        x = layers.Conv1D(d, 5, padding="same", activation="relu", name="cgmenc_cnn1")(ci)
        return layers.Conv1D(d, 3, padding="same", activation="relu", name="cgmenc_cnn2")(x)
    if encoder == "cnn_gru":
        x = layers.Conv1D(d, 5, padding="same", activation="relu", name="cgmenc_cnn")(ci)
        return layers.GRU(d, return_sequences=True, name="cgmenc_gru")(x)
    if encoder == "tcn":
        x = ci
        for dil in (1, 2, 4, 8):
            x = _tcn_block(x, d, 3, dil, f"cgmenc_tcn_d{dil}")
        return x
    raise ValueError(f"unknown encoder {encoder}")


# ----------------------------- static context (meal isolated) -----------------------------
def _contexts(si, d):
    """meal_ctx (b,d) and meal_tokens (b,K,d) for CONDITIONING; pt_ctx (b,d) = person+time
    side context concatenated downstream (never conditions the CGM curve)."""
    meal = SliceCols(MEAL[0], MEAL[1], name="slice_meal")(si)
    pt = layers.Concatenate(name="slice_persontime")(
        [SliceCols(PERSON[0], PERSON[1], name="slice_person")(si),
         SliceCols(TIME[0], TIME[1], name="slice_time")(si)])
    meal_ctx = layers.Dense(d, activation="relu", name="meal_ctx")(meal)
    pt_ctx = layers.Dense(d, activation="relu", name="pt_ctx")(pt)
    meal_tokens = layers.Reshape((N_MEAL_TOKENS, d), name="meal_tokens")(
        layers.Dense(N_MEAL_TOKENS * d, activation="relu", name="meal_tok_dense")(meal))
    return meal_ctx, meal_tokens, pt_ctx


def _activity(di, d_a, dyn_mode):
    """Encode dyn channels then a learned scalar suppression gate. dyn_mode:
    'gated' (learned) | 'off' (hard ~0 gate, non-trainable = the no_dyn control)."""
    a = layers.GRU(8, name="act_gru")(di)
    a = layers.Dense(d_a, activation="relu", name="act_dense")(a)
    if dyn_mode == "off":
        return ScalarGate(init_logit=GATE_OFF_LOGIT, trainable_gate=False, name="act_gate")(a)
    return ScalarGate(name="act_gate")(a)


# ----------------------------- fusion mechanisms -----------------------------
def _fuse(seq, meal_ctx, meal_tokens, pt_ctx, a, fusion, d, n_heads):
    if fusion in ("concat", "gate"):
        h = layers.Dense(d, activation="relu", name="cgm_proj")(
            layers.GlobalAveragePooling1D(name="cgm_gap")(seq))
        z = layers.Concatenate(name="fuse_concat")([h, meal_ctx, pt_ctx, a])
        return _gated_fusion(z, "glu") if fusion == "gate" else z
    if fusion == "film":
        gamma = layers.Dense(d, name="film_gamma")(meal_ctx)
        beta = layers.Dense(d, name="film_beta")(meal_ctx)
        seq_mod = FiLMSeq(name="film_seq")([seq, gamma, beta])      # condition the CURVE
        h = layers.Dense(d, activation="relu", name="cgm_proj")(
            layers.GlobalAveragePooling1D(name="cgm_gap")(seq_mod))
        return layers.Concatenate(name="fuse_concat")([h, pt_ctx, a])
    if fusion == "xattn":
        q = layers.Dense(d, name="xattn_q")(seq)                    # (b,T,d)
        attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(1, d // n_heads),
                                         name="xattn_mha")(q, meal_tokens, meal_tokens)
        x = layers.LayerNormalization(name="xattn_ln")(layers.Add(name="xattn_res")([q, attn]))
        h = layers.Dense(d, activation="relu", name="cgm_proj")(
            layers.GlobalAveragePooling1D(name="cgm_gap")(x))
        return layers.Concatenate(name="fuse_concat")([h, pt_ctx, a])
    raise ValueError(f"unknown fusion {fusion}")


# ----------------------------- dispatcher -----------------------------
def build(config, timesteps, n_dyn, n_static, d=24, d_a=8, n_heads=4, lr=1e-3, dyn_mode="gated"):
    """config = '<encoder>__<fusion>' (e.g. 'cnn_gru__film'). dyn_mode='off' => no_dyn
    control. Inputs [cgm(T,1), dyn(T,n_dyn), static(n_static)]; outputs [out_30,60,120]."""
    encoder, fusion = config.split("__")
    assert encoder in ENCODERS and fusion in FUSIONS, config
    ci = layers.Input(shape=(timesteps, 1), name="cgm_input")
    di = layers.Input(shape=(timesteps, n_dyn), name="dyn_input")
    si = layers.Input(shape=(n_static,), name="static_input")

    seq = _encode_cgm(ci, encoder, d)
    meal_ctx, meal_tokens, pt_ctx = _contexts(si, d)
    a = _activity(di, d_a, dyn_mode)
    z = _fuse(seq, meal_ctx, meal_tokens, pt_ctx, a, fusion, d, n_heads)

    hh = layers.Dense(32, activation="relu", name="head_dense")(z)
    hh = layers.Dropout(0.3, name="head_dropout")(hh)
    model = models.Model([ci, di, si], _heads(hh, aux=False),
                         name=f"fusion_{encoder}_{fusion}_{dyn_mode}")
    return _compile(model, aux=False, lr=lr)


def build_cgm_encoder(config, timesteps, d=24):
    """Standalone CGM encoder (same cgmenc_* layers) for SSL pretraining."""
    encoder = config.split("__")[0]
    ci = layers.Input(shape=(timesteps, 1), name="cgm_input")
    seq = _encode_cgm(ci, encoder, d)
    return models.Model(ci, seq, name=f"cgmenc_{encoder}")


def transfer_encoder(full_model, encoder_model):
    """Copy pretrained cgmenc_* weights from a standalone encoder into a full model."""
    src = {l.name: l for l in encoder_model.layers if l.name.startswith("cgmenc_")}
    n = 0
    for l in full_model.layers:
        if l.name in src:
            l.set_weights(src[l.name].get_weights()); n += 1
    return n


if __name__ == "__main__":
    T, NDYN, NSTAT = 60, 3, 17
    print(f"{'config':26s} params")
    for enc in ENCODERS:
        for fus in FUSIONS:
            m = build(f"{enc}__{fus}", T, NDYN, NSTAT)
            print(f"{enc+'__'+fus:26s} {m.count_params():,}")
    m = build("cnn_gru__film", T, NDYN, NSTAT, dyn_mode="off")
    print(f"{'cnn_gru__film (no_dyn)':26s} {m.count_params():,}")
    full = build("cnn_gru__film", T, NDYN, NSTAT)
    print("encoder transfer copied layers:", transfer_encoder(full, build_cgm_encoder("cnn_gru__film", T)))
