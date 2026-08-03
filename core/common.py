"""
common.py
======================================================================
Shared loader + metrics + per-fold normalization for the leakage-free version.

Differences vs the original pipeline (this is the whole point of the fix):
  * Loads dexcom_binned_prediction_raw.npz  -> CGM in ABSOLUTE mg/dL
    (level preserved), instead of the per-subject de-meaned file.
  * Normalization is applied PER FOLD, fit on TRAINING participants only
    (FoldNormalizer, global/level-preserving), instead of baked into the
    file over the whole dataset. This removes both the leakage and the
    baseline-destruction the reviewer flagged.
  * Reuses the SAME fold splits (fold_splits_tahap1.json) so results are
    directly comparable to the paper's participant partitioning.

Metrics are identical to the RA's code:
  NRMSE_h = RMSE_h / (max(y_h) - min(y_h))   [per-horizon range normalization]
  MARD    = mean(|y-yhat| / max(|y|,eps)) * 100
  R2      = sklearn r2_score
======================================================================
"""

import os
import json
import numpy as np
from sklearn.metrics import r2_score

from foldwise_normalizer import FoldNormalizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_NPZ = os.path.join(HERE, "dexcom_binned_prediction_raw.npz")
SPLIT_JSON = os.path.join(HERE, "fold_splits_tahap1.json")

RANDOM_STATE = 42
HORIZONS = [30, 60, 120]
METRIC_COLS = [f"{m}_{h}" for m in ("NRMSE", "R2", "MARD") for h in HORIZONS]

# ---- load leakage-free (raw, level-preserving) data ----
_d = np.load(DATA_NPZ, allow_pickle=True)
X_RAW = _d["X"].astype("float32")          # (n, 12, 4) absolute: CGM mg/dL, HR, Cal, METs
STATIC_RAW = _d["static"].astype("float32")  # (n, 17) native units
Y = _d["y"].astype("float32")              # (n, 3) mg/dL at 30/60/120
PID = _d["participant_id"]
DIAG = _d["diagnosis"]

N_SAMPLES, TIMESTEPS, N_CH = X_RAW.shape
N_STATIC = STATIC_RAW.shape[1]

# per-horizon range for NRMSE (computed on all targets, in mg/dL; targets are never normalized)
Y_MIN = Y.min(axis=0)
Y_MAX = Y.max(axis=0)
Y_RANGE = Y_MAX - Y_MIN

# ---- fold splits (train/val/test indices; participant-level) ----
# OPTIONAL: only the original version (phase1/2/3, via get_fold) uses these fixed
# splits. The light-DL pipeline builds its own stratified splits, so a missing
# split file must NOT break importing the metrics below.
if os.path.exists(SPLIT_JSON):
    with open(SPLIT_JSON, "r") as f:
        FOLD_SPLITS = json.load(f)
    N_FOLDS = len(FOLD_SPLITS)
    # sanity: indices must be valid for this array
    _all_idx = []
    for k, s in FOLD_SPLITS.items():
        _all_idx += s["train_idx"] + s["val_idx"] + s["test_idx"]
    assert max(_all_idx) < N_SAMPLES, "fold indices exceed npz length -- data/order mismatch!"
else:
    FOLD_SPLITS = None
    N_FOLDS = 0


def nrmse(y_true, y_pred, horizon_index):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = float(Y_RANGE[horizon_index])
    return float(rmse / denom) if denom != 0.0 else float("nan")


def mard(y_true, y_pred):
    eps = 1e-6
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)) * 100.0)


def compute_metrics_per_horizon(y_true, y_pred):
    """y_true, y_pred: (n, 3) in mg/dL. Returns dict of NRMSE/R2/MARD per horizon."""
    out = {}
    for i, h in enumerate(HORIZONS):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[f"NRMSE_{h}"] = nrmse(yt, yp, i)
        out[f"R2_{h}"] = float(r2_score(yt, yp))
        out[f"MARD_{h}"] = mard(yt, yp)
    return out


def get_fold(fold_idx):
    """
    Return normalized arrays for one fold. Normalization is FIT ON TRAIN ONLY
    (leakage-free) and preserves absolute level (global standardization).

    Returns dict with keys 'train','val','test', each a tuple
    (X_norm (n,12,4), static_norm (n,17), y (n,3)), plus 'idx' = (tr,va,te).
    Slice channels downstream: X[:, :, 0:1]=CGM, X[:, :, 1:]=dynamic.
    """
    if FOLD_SPLITS is None:
        raise RuntimeError(f"{SPLIT_JSON} not found -- get_fold() needs the original "
                           "fixed splits (upload fold_splits_tahap1.json). The light-DL "
                           "pipeline does not use this function.")
    s = FOLD_SPLITS[str(fold_idx)]
    tr = np.array(s["train_idx"])
    va = np.array(s["val_idx"])
    te = np.array(s["test_idx"])

    nrm = FoldNormalizer().fit(X_RAW[tr], STATIC_RAW[tr])
    Xtr, Str = nrm.transform(X_RAW[tr], STATIC_RAW[tr])
    Xva, Sva = nrm.transform(X_RAW[va], STATIC_RAW[va])
    Xte, Ste = nrm.transform(X_RAW[te], STATIC_RAW[te])

    return {
        "train": (Xtr.astype("float32"), Str.astype("float32"), Y[tr]),
        "val": (Xva.astype("float32"), Sva.astype("float32"), Y[va]),
        "test": (Xte.astype("float32"), Ste.astype("float32"), Y[te]),
        "idx": (tr, va, te),
    }


if __name__ == "__main__":
    print(f"Loaded data: X={X_RAW.shape}, static={STATIC_RAW.shape}, y={Y.shape}")
    print(f"CGM abs mg/dL range: [{X_RAW[:,:,0].min():.0f}, {X_RAW[:,:,0].max():.0f}]")
    print(f"y_range (per horizon): {Y_RANGE.round(1)}")
    print(f"folds: {N_FOLDS}")
    f0 = get_fold(0)
    print("fold0 train X", f0["train"][0].shape, "| test CGM mean (nonzero=level kept):",
          round(float(f0["test"][0][:, :, 0].mean()), 3))
