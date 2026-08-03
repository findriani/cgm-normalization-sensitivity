"""
lightdl_models.py  -- lightweight Keras-3-safe model builders for the 4 ideas
======================================================================
All raw tf ops live inside custom Layer.call (Keras 3 forbids them on symbolic
tensors during functional construction). Every builder returns a COMPILED model.

Builders:
  build_tabular_attention(...)  -- idea #1 (mini FT-Transformer)
  build_highres_tcn(...)        -- idea #2 (dilated depthwise-separable TCN)
  build_current_multimodal(...) -- reference: paper's late-simple fusion (binned)

Toggles:
  aux=True    -- idea #3: extra heads predict (y - pre-meal baseline); loss weight AUX_LAMBDA
  gate="glu"/"se" -- idea #4: lightweight gated fusion (only for multi-branch models)

Output order is ALWAYS [out_30,out_60,out_120] (+ [aux_30,aux_60,aux_120] if aux),
so callers feed targets in that order.
======================================================================
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

AUX_LAMBDA = 0.3
HORIZONS = (30, 60, 120)


# ----------------------------- custom layers -----------------------------
class FeatureTokenizer(layers.Layer):
    """FT-Transformer numerical tokenizer: each scalar feature -> d-dim token."""
    def __init__(self, d_token, **kw):
        super().__init__(**kw); self.d_token = int(d_token)

    def build(self, input_shape):
        n = int(input_shape[-1])
        self.W = self.add_weight(shape=(n, self.d_token), initializer="glorot_uniform", name="W")
        self.b = self.add_weight(shape=(n, self.d_token), initializer="zeros", name="b")

    def call(self, x):                       # x: (batch, n_feat)
        return x[:, :, None] * self.W[None] + self.b[None]   # (batch, n_feat, d)

    def get_config(self):
        c = super().get_config(); c["d_token"] = self.d_token; return c


class AddCLSToken(layers.Layer):
    """Prepend a learned CLS token to a (batch, tokens, d) sequence."""
    def build(self, input_shape):
        self.cls = self.add_weight(shape=(1, 1, int(input_shape[-1])),
                                   initializer="zeros", name="cls")

    def call(self, x):
        b = tf.shape(x)[0]
        cls = tf.tile(self.cls, [b, 1, 1])
        return tf.concat([cls, x], axis=1)


class TakeToken(layers.Layer):
    """Select one token (default CLS at index 0) from (batch, tokens, d)."""
    def __init__(self, index=0, **kw):
        super().__init__(**kw); self.index = int(index)

    def call(self, x):
        return x[:, self.index]

    def get_config(self):
        c = super().get_config(); c["index"] = self.index; return c


# ----------------------------- shared helpers -----------------------------
def _heads(h, aux):
    outs = [layers.Dense(1, name=f"out_{k}")(h) for k in HORIZONS]
    if aux:
        outs += [layers.Dense(1, name=f"aux_{k}")(h) for k in HORIZONS]
    return outs


def _compile(model, aux, lr):
    # Keras 3 does NOT broadcast a single loss string across multiple outputs when
    # loss_weights is a list -- pass one loss per output (matches len(outputs)).
    n_out = len(model.outputs)
    losses = ["mse"] * n_out
    loss_weights = [1.0, 1.0, 1.0] + ([AUX_LAMBDA] * 3 if aux else [])
    assert len(loss_weights) == n_out, f"weights {len(loss_weights)} != outputs {n_out}"
    model.compile(optimizer=optimizers.Adam(lr), loss=losses, loss_weights=loss_weights)
    return model


def _gated_fusion(x, mode):
    """Idea #4: lightweight gated fusion. mode in {None,'glu','se'}."""
    if mode is None:
        return x
    dim = int(x.shape[-1])
    if mode == "glu":
        g = layers.Dense(dim, activation="sigmoid", name="glu_gate")(x)
        v = layers.Dense(dim, name="glu_value")(x)
        return layers.Multiply(name="glu_out")([v, g])
    if mode == "se":
        r = max(2, dim // 4)
        s = layers.Dense(r, activation="relu", name="se_squeeze")(x)
        s = layers.Dense(dim, activation="sigmoid", name="se_excite")(s)
        return layers.Multiply(name="se_scale")([x, s])
    raise ValueError(f"unknown gate mode {mode}")


def _tcn_block(x, filters, kernel, dilation, name):
    """Dilated depthwise-separable conv + residual (lightweight)."""
    y = layers.SeparableConv1D(filters, kernel, padding="same", dilation_rate=dilation,
                               activation="relu", name=f"{name}_sconv")(x)
    if int(x.shape[-1]) != filters:
        x = layers.Conv1D(filters, 1, padding="same", name=f"{name}_proj")(x)
    return layers.Add(name=f"{name}_res")([x, y])


# ----------------------------- idea #1: tabular attention -----------------------------
def build_tabular_attention(n_features, d_token=16, n_heads=4, ff_mult=2,
                            aux=False, lr=1e-3):
    inp = layers.Input(shape=(n_features,), name="feat_input")
    tok = FeatureTokenizer(d_token, name="tokenizer")(inp)          # (b,F,d)
    seq = AddCLSToken(name="add_cls")(tok)                          # (b,F+1,d)
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=max(1, d_token // n_heads),
                                     name="mha")(seq, seq)
    x = layers.LayerNormalization(name="ln1")(layers.Add(name="res1")([seq, attn]))
    ff = layers.Dense(d_token * ff_mult, activation="gelu", name="ff1")(x)
    ff = layers.Dense(d_token, name="ff2")(ff)
    x = layers.LayerNormalization(name="ln2")(layers.Add(name="res2")([x, ff]))
    cls = TakeToken(0, name="take_cls")(x)                          # (b,d)
    h = layers.Dense(32, activation="relu", name="head_dense")(cls)
    h = layers.Dropout(0.3, name="head_dropout")(h)
    model = models.Model(inp, _heads(h, aux), name="tabular_attention")
    return _compile(model, aux, lr)


# ----------------------------- idea #2: high-res TCN -----------------------------
def build_highres_tcn(timesteps, n_dyn, n_static, filters=16, dilations=(1, 2, 4, 8),
                      aux=False, gate=None, lr=1e-3):
    ci = layers.Input(shape=(timesteps, 1), name="cgm_input")
    di = layers.Input(shape=(timesteps, n_dyn), name="dyn_input")
    si = layers.Input(shape=(n_static,), name="static_input")

    x = ci
    for d in dilations:
        x = _tcn_block(x, filters, 3, d, f"cgm_tcn_d{d}")
    cgm_repr = layers.GlobalAveragePooling1D(name="cgm_gap")(x)

    xd = di
    for d in (1, 2):
        xd = _tcn_block(xd, 8, 3, d, f"dyn_tcn_d{d}")
    dyn_repr = layers.GlobalAveragePooling1D(name="dyn_gap")(xd)

    st = layers.Dense(32, activation="relu", name="stat_dense1")(si)
    st = layers.Dense(16, activation="relu", name="stat_dense2")(st)

    fused = layers.Concatenate(name="fusion_concat")([cgm_repr, dyn_repr, st])
    fused = _gated_fusion(fused, gate)
    h = layers.Dense(32, activation="relu", name="fusion_dense")(fused)
    h = layers.Dropout(0.3, name="fusion_dropout")(h)
    model = models.Model([ci, di, si], _heads(h, aux), name="highres_tcn")
    return _compile(model, aux, lr)


# ----------------------------- reference: paper's late-simple fusion -----------------------------
def build_current_multimodal(timesteps, n_dyn, n_static, aux=False, lr=1e-3):
    ci = layers.Input(shape=(timesteps, 1), name="cgm_input")
    di = layers.Input(shape=(timesteps, n_dyn), name="dyn_input")
    si = layers.Input(shape=(n_static,), name="static_input")
    c = layers.Bidirectional(layers.LSTM(32), name="cgm_bilstm32")(ci)
    c = layers.Dense(32, activation="relu", name="cgm_dense32")(c)
    d = layers.GRU(16, name="dyn_gru16")(di)
    d = layers.Dense(16, activation="relu", name="dyn_dense16")(d)
    s = layers.Dense(32, activation="relu", name="stat_dense32")(si)
    s = layers.Dense(16, activation="relu", name="stat_dense16")(s)
    fused = layers.Concatenate(name="fusion_concat")([c, d, s])
    h = layers.Dense(32, activation="relu", name="fusion_dense32")(fused)
    h = layers.Dropout(0.3, name="fusion_dropout")(h)
    model = models.Model([ci, di, si], _heads(h, aux), name="current_multimodal")
    return _compile(model, aux, lr)
