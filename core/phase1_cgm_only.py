"""
phase1_cgm_only.py  -- leakage-free version of Table 2 / Fig 2 (31 CGM-only models)
======================================================================
Identical architectures and training protocol to the RA's tahap1_dexcom.py.
ONLY the data path changed: CGM is now absolute mg/dL, normalized per fold
(fit on train only) via common.get_fold, then channel 0 is sliced.

Run in Colab:  !python phase1_cgm_only.py
Outputs: results_tahap1_<exp>.csv (per fold) and table1_tahap1_cgm_only.csv
======================================================================
"""
import os, random
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from common import (get_fold, compute_metrics_per_horizon, METRIC_COLS,
                              N_FOLDS, TIMESTEPS, RANDOM_STATE)

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)
N_CGM = 1  # phase 1 uses CGM channel only


def build_model_from_config(config, input_timesteps, input_features):
    inputs = layers.Input(shape=(input_timesteps, input_features), name="cgm_input")
    x = inputs
    mt = config["model_type"]

    if mt in ["LSTM", "BiLSTM", "GRU", "BiGRU"]:
        for i, units in enumerate(config["hidden_units"]):
            rs = i < len(config["hidden_units"]) - 1
            if mt == "LSTM":
                x = layers.LSTM(units, return_sequences=rs, name=f"lstm_{i+1}")(x)
            elif mt == "BiLSTM":
                x = layers.Bidirectional(layers.LSTM(units, return_sequences=rs), name=f"bilstm_{i+1}")(x)
            elif mt == "GRU":
                x = layers.GRU(units, return_sequences=rs, name=f"gru_{i+1}")(x)
            elif mt == "BiGRU":
                x = layers.Bidirectional(layers.GRU(units, return_sequences=rs), name=f"bigru_{i+1}")(x)
    elif mt == "CNN":
        for i, c in enumerate(config["conv_layers"]):
            x = layers.Conv1D(c["filters"], c["kernel_size"], dilation_rate=c.get("dilation_rate", 1),
                              padding="same", activation="relu", name=f"conv1d_{i+1}")(x)
        x = (layers.GlobalAveragePooling1D(name="global_avgpool")(x)
             if config.get("pooling", "max") == "avg"
             else layers.GlobalMaxPooling1D(name="global_maxpool")(x))
    elif mt == "HYBRID":
        for i, c in enumerate(config["cnn_layers"]):
            x = layers.Conv1D(c["filters"], c["kernel_size"], dilation_rate=c.get("dilation_rate", 1),
                              padding="same", activation="relu", name=f"hyb_conv1d_{i+1}")(x)
        rt, ru = config["rnn_type"], config["rnn_units"]
        for i, units in enumerate(ru):
            rs = i < len(ru) - 1
            if rt == "LSTM":
                x = layers.LSTM(units, return_sequences=rs, name=f"hyb_lstm_{i+1}")(x)
            elif rt == "BiLSTM":
                x = layers.Bidirectional(layers.LSTM(units, return_sequences=rs), name=f"hyb_bilstm_{i+1}")(x)
            elif rt == "GRU":
                x = layers.GRU(units, return_sequences=rs, name=f"hyb_gru_{i+1}")(x)
            elif rt == "BiGRU":
                x = layers.Bidirectional(layers.GRU(units, return_sequences=rs), name=f"hyb_bigru_{i+1}")(x)
    else:
        raise ValueError(f"Unknown model_type {mt}")

    for i, units in enumerate(config.get("dense_units", [32])):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
    dr = float(config.get("dropout", 0.0))
    if dr > 0.0:
        x = layers.Dropout(dr, name="dropout")(x)

    outs = [layers.Dense(1, name=f"out_{h}")(x) for h in (30, 60, 120)]
    model = models.Model(inputs=inputs, outputs=outs, name=f"model_{config['exp_id']}")
    model.compile(optimizer=optimizers.Adam(learning_rate=config.get("learning_rate", 1e-3)), loss="mse")
    return model


# --- exact 31 configs from tahap1_dexcom.py ---
EXPERIMENT_CONFIGS = {
    "1.1a": {"exp_id": "1.1a", "family": "1.1", "model_type": "LSTM", "hidden_units": [16], "dense_units": [32], "dropout": 0.3, "description": "LSTM 16 1 layer"},
    "1.1b": {"exp_id": "1.1b", "family": "1.1", "model_type": "LSTM", "hidden_units": [32], "dense_units": [32], "dropout": 0.3, "description": "LSTM 32 1 layer"},
    "1.1c": {"exp_id": "1.1c", "family": "1.1", "model_type": "LSTM", "hidden_units": [48], "dense_units": [32], "dropout": 0.3, "description": "LSTM 48 1 layer"},
    "1.1d": {"exp_id": "1.1d", "family": "1.1", "model_type": "LSTM", "hidden_units": [32, 16], "dense_units": [32], "dropout": 0.3, "description": "LSTM 32 then 16"},
    "1.1e": {"exp_id": "1.1e", "family": "1.1", "model_type": "LSTM", "hidden_units": [48, 24], "dense_units": [32], "dropout": 0.3, "description": "LSTM 48 then 24"},
    "1.1f": {"exp_id": "1.1f", "family": "1.1", "model_type": "LSTM", "hidden_units": [64, 32], "dense_units": [32], "dropout": 0.3, "description": "LSTM 64 then 32"},
    "1.1g": {"exp_id": "1.1g", "family": "1.1", "model_type": "LSTM", "hidden_units": [48, 32, 16], "dense_units": [32], "dropout": 0.3, "description": "LSTM 48 then 32 then 16"},
    "1.2a": {"exp_id": "1.2a", "family": "1.2", "model_type": "BiLSTM", "hidden_units": [16], "dense_units": [32], "dropout": 0.3, "description": "BiLSTM 16 1 layer"},
    "1.2b": {"exp_id": "1.2b", "family": "1.2", "model_type": "BiLSTM", "hidden_units": [24], "dense_units": [32], "dropout": 0.3, "description": "BiLSTM 24 1 layer"},
    "1.2c": {"exp_id": "1.2c", "family": "1.2", "model_type": "BiLSTM", "hidden_units": [32], "dense_units": [32], "dropout": 0.3, "description": "BiLSTM 32 1 layer"},
    "1.2d": {"exp_id": "1.2d", "family": "1.2", "model_type": "BiLSTM", "hidden_units": [24, 12], "dense_units": [32], "dropout": 0.3, "description": "BiLSTM 24 then 12"},
    "1.3a": {"exp_id": "1.3a", "family": "1.3", "model_type": "GRU", "hidden_units": [16], "dense_units": [32], "dropout": 0.3, "description": "GRU 16 1 layer"},
    "1.3b": {"exp_id": "1.3b", "family": "1.3", "model_type": "GRU", "hidden_units": [32], "dense_units": [32], "dropout": 0.3, "description": "GRU 32 1 layer"},
    "1.3c": {"exp_id": "1.3c", "family": "1.3", "model_type": "GRU", "hidden_units": [48], "dense_units": [32], "dropout": 0.3, "description": "GRU 48 1 layer"},
    "1.3d": {"exp_id": "1.3d", "family": "1.3", "model_type": "BiGRU", "hidden_units": [16], "dense_units": [32], "dropout": 0.3, "description": "BiGRU 16 1 layer"},
    "1.3e": {"exp_id": "1.3e", "family": "1.3", "model_type": "BiGRU", "hidden_units": [24], "dense_units": [32], "dropout": 0.3, "description": "BiGRU 24 1 layer"},
    "1.3f": {"exp_id": "1.3f", "family": "1.3", "model_type": "GRU", "hidden_units": [32, 16], "dense_units": [32], "dropout": 0.3, "description": "GRU 32 then 16"},
    "1.4a": {"exp_id": "1.4a", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 16, "kernel_size": 5}], "pooling": "max", "dense_units": [32], "dropout": 0.2, "description": "Conv1D 16 k5 max pool"},
    "1.4b": {"exp_id": "1.4b", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 16, "kernel_size": 5}], "pooling": "avg", "dense_units": [32], "dropout": 0.2, "description": "Conv1D 16 k5 avg pool"},
    "1.4c": {"exp_id": "1.4c", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 32, "kernel_size": 5}], "pooling": "max", "dense_units": [32], "dropout": 0.2, "description": "Conv1D 32 k5 max pool"},
    "1.4d": {"exp_id": "1.4d", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 32, "kernel_size": 5}], "pooling": "avg", "dense_units": [32], "dropout": 0.2, "description": "Conv1D 32 k5 avg pool"},
    "1.4e": {"exp_id": "1.4e", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 16, "kernel_size": 3}, {"filters": 32, "kernel_size": 3}], "pooling": "max", "dense_units": [64, 32], "dropout": 0.2, "description": "Conv1D 16 32 dua layer max pool"},
    "1.4f": {"exp_id": "1.4f", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 16, "kernel_size": 3}, {"filters": 32, "kernel_size": 3}], "pooling": "avg", "dense_units": [64, 32], "dropout": 0.2, "description": "Conv1D 16 32 dua layer avg pool"},
    "1.4g": {"exp_id": "1.4g", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 16, "kernel_size": 5}, {"filters": 32, "kernel_size": 3}, {"filters": 48, "kernel_size": 3}], "pooling": "max", "dense_units": [64, 32], "dropout": 0.2, "description": "Conv1D 16 32 48 tiga layer max pool"},
    "1.4h": {"exp_id": "1.4h", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 16, "kernel_size": 5}, {"filters": 32, "kernel_size": 3}, {"filters": 48, "kernel_size": 3}], "pooling": "avg", "dense_units": [64, 32], "dropout": 0.2, "description": "Conv1D 16 32 48 tiga layer avg pool"},
    "1.4i": {"exp_id": "1.4i", "family": "1.4", "model_type": "CNN", "conv_layers": [{"filters": 24, "kernel_size": 3, "dilation_rate": 1}, {"filters": 24, "kernel_size": 3, "dilation_rate": 2}], "pooling": "avg", "dense_units": [32], "dropout": 0.2, "description": "Dilated Conv1D 24 24"},
    "1.5a": {"exp_id": "1.5a", "family": "1.5", "model_type": "HYBRID", "cnn_layers": [{"filters": 16, "kernel_size": 3}], "rnn_type": "GRU", "rnn_units": [32], "dense_units": [32], "dropout": 0.2, "description": "Conv1D 16 k3 lalu GRU 32"},
    "1.5b": {"exp_id": "1.5b", "family": "1.5", "model_type": "HYBRID", "cnn_layers": [{"filters": 16, "kernel_size": 3}], "rnn_type": "LSTM", "rnn_units": [32], "dense_units": [32], "dropout": 0.2, "description": "Conv1D 16 k3 lalu LSTM 32"},
    "1.5c": {"exp_id": "1.5c", "family": "1.5", "model_type": "HYBRID", "cnn_layers": [{"filters": 16, "kernel_size": 3}], "rnn_type": "LSTM", "rnn_units": [48], "dense_units": [32], "dropout": 0.2, "description": "Conv1D 16 k3 lalu LSTM 48"},
    "1.5d": {"exp_id": "1.5d", "family": "1.5", "model_type": "HYBRID", "cnn_layers": [{"filters": 32, "kernel_size": 5}], "rnn_type": "BiLSTM", "rnn_units": [24], "dense_units": [32], "dropout": 0.2, "description": "Conv1D 32 k5 lalu BiLSTM 24"},
    "1.5e": {"exp_id": "1.5e", "family": "1.5", "model_type": "HYBRID", "cnn_layers": [{"filters": 16, "kernel_size": 3}, {"filters": 32, "kernel_size": 3}], "rnn_type": "GRU", "rnn_units": [48], "dense_units": [32], "dropout": 0.2, "description": "Conv1D 16 32 lalu GRU 48"},
}


def run_single_experiment(exp_id, batch_size=32, max_epochs=100, patience=10):
    config = EXPERIMENT_CONFIGS[exp_id]
    print(f"[{exp_id}] {config['description']}")
    rows = []
    for fold_idx in range(N_FOLDS):
        fd = get_fold(fold_idx)
        Xtr, _, ytr = fd["train"]; Xva, _, yva = fd["val"]; Xte, _, yte = fd["test"]
        # CGM channel only
        Xtr, Xva, Xte = Xtr[:, :, 0:1], Xva[:, :, 0:1], Xte[:, :, 0:1]

        model = build_model_from_config(config, TIMESTEPS, N_CGM)
        cb = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        model.fit(Xtr, [ytr[:, 0], ytr[:, 1], ytr[:, 2]],
                  validation_data=(Xva, [yva[:, 0], yva[:, 1], yva[:, 2]]),
                  epochs=max_epochs, batch_size=batch_size, callbacks=[cb], verbose=0)
        p30, p60, p120 = model.predict(Xte, verbose=0)
        y_pred = np.stack([p30.squeeze(-1), p60.squeeze(-1), p120.squeeze(-1)], axis=-1)
        m = compute_metrics_per_horizon(yte, y_pred)
        m["fold"] = fold_idx; m["exp_id"] = exp_id
        rows.append(m)
        print(f"  fold {fold_idx}: NRMSE60={m['NRMSE_60']:.4f} R2_60={m['R2_60']:.4f} MARD60={m['MARD_60']:.2f}")
        tf.keras.backend.clear_session()
    df = pd.DataFrame(rows)
    df.to_csv(f"results_tahap1_{exp_id}.csv", index=False)
    return df


def main():
    for exp_id in sorted(EXPERIMENT_CONFIGS.keys()):
        run_single_experiment(exp_id)

    # assemble Table 1 (all horizons) with parameter counts
    rows = []
    for exp_id in sorted(EXPERIMENT_CONFIGS.keys()):
        cfg = EXPERIMENT_CONFIGS[exp_id]
        df = pd.read_csv(f"results_tahap1_{exp_id}.csv")
        model = build_model_from_config(cfg, TIMESTEPS, N_CGM)
        row = {"exp_id": exp_id, "family": cfg["family"], "model_type": cfg["model_type"],
               "description": cfg["description"], "params": int(model.count_params())}
        for m in METRIC_COLS:
            row[f"{m}_mean"] = float(df[m].mean()); row[f"{m}_std"] = float(df[m].std())
        rows.append(row); tf.keras.backend.clear_session()
    table1 = pd.DataFrame(rows).sort_values(["family", "exp_id"]).reset_index(drop=True)
    table1.to_csv("table1_tahap1_cgm_only.csv", index=False)
    print("\nSaved table1_tahap1_cgm_only.csv")
    print(table1[["exp_id", "model_type", "params", "NRMSE_60_mean", "R2_60_mean", "R2_60_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
