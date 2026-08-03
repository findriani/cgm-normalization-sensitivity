"""
phase2_multimodal.py  -- leakage-free version of Tables 4 & 5 (+ gated / attention)
======================================================================
Faithful to the RA's part2_multimodal_fusion.ipynb. Only data loading changed:
absolute-mg/dL CGM, per-fold leakage-free normalization (via
common.get_fold), same fold splits.

Covers:
  2.1  modality ablation (7 configs)            -> table_part2_2.1_ablation.csv
  2.2  fusion strategies (early/simple/deep)     -> table_part2_2.2_fusion_strategies.csv
  2.3  gated fusion (entropy-regularized)        -> table_part2_2.3_gated_fusion.csv
  2.4  temporal attention (static->CGM)          -> table_part2_2.4_temporal_attention.csv

Run in Colab:  !python phase2_multimodal.py
======================================================================
"""
import random
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from common import (get_fold, compute_metrics_per_horizon, METRIC_COLS,
                              N_FOLDS, TIMESTEPS, N_STATIC, DIAG, RANDOM_STATE)

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)
N_CGM, N_DYN = 1, 3


# ---------- backbone encoders (Part 1/2 protocol) ----------
def cgm_backbone_bilstm(cgm_in):
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False), name="cgm_bilstm32")(cgm_in)
    return layers.Dense(32, activation="relu", name="cgm_dense32")(x)

def cgm_backbone_cnn(cgm_in):
    x = layers.Conv1D(16, 3, padding="same", activation="relu", name="cgm_conv1")(cgm_in)
    x = layers.Conv1D(32, 3, padding="same", activation="relu", name="cgm_conv2")(x)
    x = layers.GlobalMaxPooling1D(name="cgm_pool")(x)
    return layers.Dense(32, activation="relu", name="cgm_dense32")(x)

def get_cgm_backbone(name, cgm_in):
    return cgm_backbone_bilstm(cgm_in) if name == "bilstm" else cgm_backbone_cnn(cgm_in)

def dyn_backbone(dyn_in):
    x = layers.GRU(16, return_sequences=False, name="dyn_gru16")(dyn_in)
    return layers.Dense(16, activation="relu", name="dyn_dense16")(x)

def static_backbone(stat_in):
    x = layers.Dense(32, activation="relu", name="stat_dense32")(stat_in)
    return layers.Dense(16, activation="relu", name="stat_dense16")(x)


# ---------- generic per-fold runner ----------
def _slice(fd, split):
    X, S, y = fd[split]
    return X[:, :, 0:1], X[:, :, 1:], S, y

def run_folds(build_fn, exp_id, use=("cgm", "dyn", "stat"),
              batch_size=32, max_epochs=100, patience=10, extra=None):
    """extra: optional callable(model, inputs_test, test_idx) -> DataFrame for gate/attn dumps."""
    rows, extras = [], []
    for fold_idx in range(N_FOLDS):
        fd = get_fold(fold_idx)
        (cgm_tr, dyn_tr, s_tr, y_tr) = _slice(fd, "train")
        (cgm_va, dyn_va, s_va, y_va) = _slice(fd, "val")
        (cgm_te, dyn_te, s_te, y_te) = _slice(fd, "test")

        def pick(cgm, dyn, s):
            out = []
            if "cgm" in use: out.append(cgm)
            if "dyn" in use: out.append(dyn)
            if "stat" in use: out.append(s)
            return out

        Xtr, Xva, Xte = pick(cgm_tr, dyn_tr, s_tr), pick(cgm_va, dyn_va, s_va), pick(cgm_te, dyn_te, s_te)
        model = build_fn()
        cb = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        model.fit(Xtr, [y_tr[:, 0], y_tr[:, 1], y_tr[:, 2]],
                  validation_data=(Xva, [y_va[:, 0], y_va[:, 1], y_va[:, 2]]),
                  epochs=max_epochs, batch_size=batch_size, callbacks=[cb], verbose=0)
        p30, p60, p120 = model.predict(Xte, verbose=0)
        y_pred = np.stack([p30.squeeze(-1), p60.squeeze(-1), p120.squeeze(-1)], axis=-1)
        m = compute_metrics_per_horizon(y_te, y_pred)
        m["fold"] = fold_idx; m["exp_id"] = exp_id
        rows.append(m)
        print(f"  fold {fold_idx}: NRMSE60={m['NRMSE_60']:.4f} R2_60={m['R2_60']:.4f} MARD60={m['MARD_60']:.2f}")
        if extra is not None:
            extras.append(extra(model, Xte, fd["idx"][2]))
        tf.keras.backend.clear_session()
    df = pd.DataFrame(rows)
    df.to_csv(f"results_part2_{exp_id}.csv", index=False)
    extra_df = pd.concat(extras, ignore_index=True) if extras else None
    return df, extra_df


def summarize(df):
    s = df[METRIC_COLS].describe().loc[["mean", "std"]]
    return s


# ---------- 2.1 modality ablation ----------
def build_late_fusion(use_cgm, use_dyn, use_stat, backbone="bilstm"):
    inputs, reprs = [], []
    if use_cgm:
        ci = layers.Input(shape=(TIMESTEPS, N_CGM), name="cgm_input"); inputs.append(ci); reprs.append(get_cgm_backbone(backbone, ci))
    if use_dyn:
        di = layers.Input(shape=(TIMESTEPS, N_DYN), name="dyn_input"); inputs.append(di); reprs.append(dyn_backbone(di))
    if use_stat:
        si = layers.Input(shape=(N_STATIC,), name="static_input"); inputs.append(si); reprs.append(static_backbone(si))
    fused = reprs[0] if len(reprs) == 1 else layers.Concatenate(name="fusion_concat")(reprs)
    x = layers.Dense(32, activation="relu", name="fusion_dense32")(fused)
    x = layers.Dropout(0.3, name="fusion_dropout")(x)
    outs = [layers.Dense(1, name=f"out_{h}")(x) for h in (30, 60, 120)]
    model = models.Model(inputs=inputs, outputs=outs, name="late_fusion")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
    return model

ABLATION = [
    ("2.1a_cgm_only", ("cgm",), True, False, False),
    ("2.1b_dyn_only", ("dyn",), False, True, False),
    ("2.1c_stat_only", ("stat",), False, False, True),
    ("2.1d_cgm_dyn", ("cgm", "dyn"), True, True, False),
    ("2.1e_cgm_stat", ("cgm", "stat"), True, False, True),
    ("2.1f_dyn_stat", ("dyn", "stat"), False, True, True),
    ("2.1g_full", ("cgm", "dyn", "stat"), True, True, True),
]

def run_ablation():
    rows = []
    for exp_id, use, uc, ud, us in ABLATION:
        print(f"=== 2.1 {exp_id} ===")
        df, _ = run_folds(lambda uc=uc, ud=ud, us=us: build_late_fusion(uc, ud, us), exp_id, use=use)
        s = summarize(df)
        rows.append({"exp_id": exp_id, "use_cgm": uc, "use_dynamic": ud, "use_static": us,
                     **{f"{m}_mean": s.loc["mean", m] for m in METRIC_COLS},
                     **{f"{m}_std": s.loc["std", m] for m in METRIC_COLS}})
    pd.DataFrame(rows).set_index("exp_id").to_csv("table_part2_2.1_ablation.csv")
    print("Saved table_part2_2.1_ablation.csv")


# ---------- 2.2 fusion strategies ----------
def build_22a_early(backbone="bilstm"):
    ci = layers.Input(shape=(TIMESTEPS, N_CGM), name="cgm_input")
    di = layers.Input(shape=(TIMESTEPS, N_DYN), name="dyn_input")
    si = layers.Input(shape=(N_STATIC,), name="static_input")
    xt = layers.Concatenate(axis=-1, name="concat_cgm_dyn")([ci, di])
    tr = get_cgm_backbone(backbone, xt)
    sr = static_backbone(si)
    fused = layers.Concatenate(name="fusion_concat")([tr, sr])
    x = layers.Dense(32, activation="relu", name="fusion_dense32")(fused)
    x = layers.Dropout(0.3, name="fusion_dropout")(x)
    outs = [layers.Dense(1, name=f"out_{h}")(x) for h in (30, 60, 120)]
    model = models.Model(inputs=[ci, di, si], outputs=outs, name="fusion_early")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse"); return model

def _late(backbone, deep):
    ci = layers.Input(shape=(TIMESTEPS, N_CGM), name="cgm_input")
    di = layers.Input(shape=(TIMESTEPS, N_DYN), name="dyn_input")
    si = layers.Input(shape=(N_STATIC,), name="static_input")
    fused = layers.Concatenate(name="fusion_concat")([get_cgm_backbone(backbone, ci), dyn_backbone(di), static_backbone(si)])
    if deep:
        x = layers.Dense(64, activation="relu", name="fusion_dense64")(fused)
        x = layers.Dense(32, activation="relu", name="fusion_dense32")(x)
    else:
        x = layers.Dense(32, activation="relu", name="fusion_dense32")(fused)
    x = layers.Dropout(0.3, name="fusion_dropout")(x)
    outs = [layers.Dense(1, name=f"out_{h}")(x) for h in (30, 60, 120)]
    model = models.Model(inputs=[ci, di, si], outputs=outs, name=f"fusion_{'deep' if deep else 'simple'}")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse"); return model

def run_fusion():
    specs = [("2.2a_early", build_22a_early),
             ("2.2b_late_simple", lambda: _late("bilstm", False)),
             ("2.2c_late_deep", lambda: _late("bilstm", True))]
    rows = []
    for exp_id, fn in specs:
        print(f"=== 2.2 {exp_id} ===")
        df, _ = run_folds(fn, exp_id, use=("cgm", "dyn", "stat"))
        s = summarize(df)
        rows.append({"exp_id": exp_id, **{f"{m}_mean": s.loc["mean", m] for m in METRIC_COLS},
                     **{f"{m}_std": s.loc["std", m] for m in METRIC_COLS}})
    pd.DataFrame(rows).set_index("exp_id").to_csv("table_part2_2.2_fusion_strategies.csv")
    print("Saved table_part2_2.2_fusion_strategies.csv")


# ---------- 2.3 gated fusion ----------
class EntropyGatingLayer(layers.Layer):
    def __init__(self, lambda_entropy=0.01, **kw):
        super().__init__(**kw); self.lambda_entropy = lambda_entropy
    def call(self, gate_logits):
        w = tf.nn.softmax(gate_logits, axis=-1)
        ent = -tf.reduce_sum(w * tf.math.log(tf.clip_by_value(w, 1e-8, 1.0)), axis=-1)
        self.add_loss(-self.lambda_entropy * tf.reduce_mean(ent))
        return w

def build_gated(backbone="bilstm", lambda_entropy=0.01):
    ci = layers.Input(shape=(TIMESTEPS, N_CGM), name="cgm_input")
    di = layers.Input(shape=(TIMESTEPS, N_DYN), name="dyn_input")
    si = layers.Input(shape=(N_STATIC,), name="static_input")
    if backbone == "bilstm":
        cgm_repr = layers.Bidirectional(layers.LSTM(32), name="cgm_bilstm")(ci)
    else:
        x = layers.Conv1D(16, 3, padding="same", activation="relu")(ci)
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
        cgm_repr = layers.GlobalMaxPooling1D()(x)
    dyn_repr = layers.GRU(16, name="dyn_gru")(di)
    sr = layers.Dense(32, activation="relu", name="static_dense1")(si)
    sr = layers.Dense(16, activation="relu", name="static_dense2")(sr)
    gate_logits = layers.Dense(3, name="gate_logits")(layers.Concatenate(name="gate_input")([sr, cgm_repr]))
    gw = EntropyGatingLayer(lambda_entropy, name="gate_weights")(gate_logits)
    w_cgm = layers.Lambda(lambda w: tf.expand_dims(w[:, 0], -1), name="w_cgm")(gw)
    w_dyn = layers.Lambda(lambda w: tf.expand_dims(w[:, 1], -1), name="w_dyn")(gw)
    w_stat = layers.Lambda(lambda w: tf.expand_dims(w[:, 2], -1), name="w_stat")(gw)
    fused = layers.Concatenate(name="fusion_concat")([
        layers.Multiply()([cgm_repr, w_cgm]), layers.Multiply()([dyn_repr, w_dyn]), layers.Multiply()([sr, w_stat])])
    x = layers.Dense(32, activation="relu", name="fusion_dense32")(fused)
    x = layers.Dropout(0.3, name="fusion_dropout")(x)
    outs = [layers.Dense(1, name=f"out_{h}")(x) for h in (30, 60, 120)]
    model = models.Model(inputs=[ci, di, si], outputs=outs, name="gated_fusion")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse"); return model

def run_gated():
    print("=== 2.3 gated fusion ===")
    def dump(model, Xte, test_idx):
        gm = models.Model(inputs=model.inputs, outputs=model.get_layer("gate_weights").output)
        gw = gm.predict(Xte, verbose=0)
        return pd.DataFrame({"cgm_weight": gw[:, 0], "dyn_weight": gw[:, 1],
                             "static_weight": gw[:, 2], "diagnosis": DIAG[test_idx]})
    df, gates = run_folds(lambda: build_gated(), "2.3_gated", use=("cgm", "dyn", "stat"), extra=dump)
    s = summarize(df)
    pd.DataFrame([{"exp_id": "2.3_gated", **{f"{m}_mean": s.loc["mean", m] for m in METRIC_COLS},
                   **{f"{m}_std": s.loc["std", m] for m in METRIC_COLS}}]).set_index("exp_id").to_csv("table_part2_2.3_gated_fusion.csv")
    if gates is not None:
        gates.groupby("diagnosis")[["cgm_weight", "dyn_weight", "static_weight"]].agg(["mean", "std"]).to_csv("table_part2_2.3_gate_by_diagnosis.csv")
    print("Saved table_part2_2.3_gated_fusion.csv")


# ---------- 2.4 temporal attention (static->CGM) ----------
class StaticGuidedAttention(layers.Layer):
    """Dot-product attention where the static+dynamic context queries the CGM
    sequence. Same math as the RA's notebook, but the tf ops run inside a Layer
    (Keras 3 forbids raw tf ops on symbolic tensors during model construction).
    Returns (attended_vector (b,h), attn_weights (b,T))."""
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.query_dense = layers.Dense(self.hidden_dim, name="attn_query")

    def call(self, inputs):
        cgm_seq, context = inputs                              # (b,T,h), (b,c)
        q = tf.expand_dims(self.query_dense(context), axis=1)  # (b,1,h)
        scores = tf.matmul(q, cgm_seq, transpose_b=True)       # (b,1,T)
        w = tf.nn.softmax(scores, axis=-1)                     # (b,1,T)
        ctx = tf.matmul(w, cgm_seq)                            # (b,1,h)
        return tf.squeeze(ctx, axis=1), tf.squeeze(w, axis=1)  # (b,h), (b,T)

    def get_config(self):
        cfg = super().get_config(); cfg["hidden_dim"] = self.hidden_dim; return cfg


def build_temporal_attention(backbone="bilstm"):
    ci = layers.Input(shape=(TIMESTEPS, N_CGM), name="cgm_input")
    di = layers.Input(shape=(TIMESTEPS, N_DYN), name="dyn_input")
    si = layers.Input(shape=(N_STATIC,), name="static_input")
    if backbone == "bilstm":
        cgm_enc = layers.Bidirectional(layers.LSTM(32, return_sequences=True), name="cgm_bilstm32")(ci)
    else:
        x = layers.Conv1D(16, 3, padding="same", activation="relu")(ci)
        cgm_enc = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    dyn_repr = layers.GRU(16, name="dyn_gru16")(di)
    sr = layers.Dense(32, activation="relu", name="static_dense32")(si)
    sr = layers.Dense(16, activation="relu", name="static_dense16")(sr)
    context = layers.Concatenate(name="context")([sr, dyn_repr])
    h = cgm_enc.shape[-1]
    attended, _attn = StaticGuidedAttention(h, name="temporal_attention")([cgm_enc, context])
    combined = layers.Concatenate(name="fusion_concat")([attended, dyn_repr, sr])
    x = layers.Dense(32, activation="relu", name="fusion_dense")(combined)
    x = layers.Dropout(0.3, name="fusion_dropout")(x)
    outs = [layers.Dense(1, name=f"out_{h_}")(x) for h_ in (30, 60, 120)]
    model = models.Model(inputs=[ci, di, si], outputs=outs, name="temporal_attn_static")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse"); return model

def run_attention():
    print("=== 2.4 temporal attention ===")
    df, _ = run_folds(lambda: build_temporal_attention(), "2.4_attn", use=("cgm", "dyn", "stat"))
    s = summarize(df)
    pd.DataFrame([{"exp_id": "2.4_attn", **{f"{m}_mean": s.loc["mean", m] for m in METRIC_COLS},
                   **{f"{m}_std": s.loc["std", m] for m in METRIC_COLS}}]).set_index("exp_id").to_csv("table_part2_2.4_temporal_attention.csv")
    print("Saved table_part2_2.4_temporal_attention.csv")


def main():
    run_ablation()
    run_fusion()
    run_gated()
    run_attention()
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
