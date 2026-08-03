"""
build_data.py
======================================================================
Regenerate CORRECTED, leakage-free, level-preserving prediction data.

WHY THIS EXISTS
---------------
The original normalization (see glucose_prediction_preprocessing_revision_2.py)
had two defects flagged in review:

  1. LEAKAGE. Normalization statistics (per-subject CGM/HR mean-std,
     activity min-max, and global static mean-std) were computed over the
     ENTIRE dataset -- including participants that later land in the held-out
     CV fold. A test participant was scaled using their own held-out records.

  2. LEVEL DESTRUCTION. CGM was z-scored PER SUBJECT, which subtracts each
     participant's own mean and thereby deletes the absolute glucose baseline
     from the CGM branch. Because ~50% of the 60-min target variance is
     between-subject, this stripped exactly the signal that lets CGM predict
     absolute mg/dL -- while the static branch kept HbA1c (a level proxy).
     Result: CGM looked useless (R2<0) and static looked dominant, as an
     ARTIFACT of preprocessing rather than a property of the data.

WHAT THIS SCRIPT DOES
---------------------
  * Recovers the RAW absolute values by exactly inverting the stored
    normalization parameters (normalization_params.pkl). No re-extraction
    from the original CGMacros CSVs is needed.
  * Verifies the inversion round-trips (re-applying the original transform to
    the recovered raw values must reproduce the stored normalized arrays).
  * Assigns a FIXED, reproducible 5-fold GroupKFold split (by participant) and
    stores it in the file, so every future experiment uses identical folds.
  * Saves raw, level-preserving .npz files. NORMALIZATION IS NOT BAKED IN --
    it is deferred to foldwise_normalizer.py, applied per fold at train time
    (fit on TRAIN participants only) to remain leakage-free.

NOTE: no models are trained here. This only regenerates the input data.
======================================================================
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

PARAMS_FILE = os.path.join(HERE, "normalization_params.pkl")

# (input normalized file, key into normalization_params.pkl)
FILES = [
    ("dexcom_raw_prediction.npz",    "dexcom_raw"),
    ("dexcom_binned_prediction.npz", "dexcom_binned"),
    ("libre_raw_prediction.npz",     "libre_raw"),
    ("libre_binned_prediction.npz",  "libre_binned"),
]

# static feature layout (17 features)
STATIC_NAMES = ["age", "gender", "bmi", "hba1c",
                "calories", "carbs", "protein", "fat", "fiber",
                "hour_sin", "hour_cos", "is_morning", "is_evening", "is_weekend",
                "is_breakfast", "is_lunch", "is_dinner"]
# indices that were globally normalized in the original pipeline
STATIC_ZSCORE = {"age": 0, "bmi": 2, "hba1c": 3}
STATIC_MACROS = {"calories": 4, "carbs": 5, "protein": 6, "fat": 7, "fiber": 8}

N_SPLITS = 5


def invert_timeseries(Xn, pid, per_subject):
    """Recover raw absolute time series (CGM mg/dL, HR bpm, Cal, METs)."""
    X = Xn.astype(float).copy()
    for subj in np.unique(pid):
        m = pid == subj
        p = per_subject[str(subj)]
        # ch0 CGM: per-subject z-score  ->  raw = norm*std + mean
        if p["cgm_std"] > 0:
            X[m, :, 0] = Xn[m, :, 0] * p["cgm_std"] + p["cgm_mean"]
        # ch1 HR: per-subject z-score
        if p["hr_std"] > 0:
            X[m, :, 1] = Xn[m, :, 1] * p["hr_std"] + p["hr_mean"]
        # ch2 Calories(Activity): per-subject min-max  ->  raw = norm*(max-min)+min
        if p["cal_max"] > p["cal_min"]:
            X[m, :, 2] = Xn[m, :, 2] * (p["cal_max"] - p["cal_min"]) + p["cal_min"]
        # ch3 METs: per-subject min-max
        if p["mets_max"] > p["mets_min"]:
            X[m, :, 3] = Xn[m, :, 3] * (p["mets_max"] - p["mets_min"]) + p["mets_min"]
    return X


def invert_static(static_n, glob):
    """Recover raw static features (native units). Binary/cyclical stay as-is."""
    S = static_n.astype(float).copy()
    for name, idx in STATIC_ZSCORE.items():
        g = glob[name]
        S[:, idx] = static_n[:, idx] * g["std"] + g["mean"]
    for name, idx in STATIC_MACROS.items():
        g = glob[name]
        v = static_n[:, idx] * g["std"] + g["mean"]
        if g.get("log_transformed", False):
            v = np.expm1(v)                      # undo log1p
        S[:, idx] = v
    return S


def reapply_original_norm(X_raw, static_raw, pid, per_subject, glob):
    """Re-apply the ORIGINAL normalization to raw values (for round-trip check)."""
    Xn = X_raw.astype(float).copy()
    for subj in np.unique(pid):
        m = pid == subj
        p = per_subject[str(subj)]
        if p["cgm_std"] > 0:
            Xn[m, :, 0] = (X_raw[m, :, 0] - p["cgm_mean"]) / p["cgm_std"]
        if p["hr_std"] > 0:
            Xn[m, :, 1] = (X_raw[m, :, 1] - p["hr_mean"]) / p["hr_std"]
        if p["cal_max"] > p["cal_min"]:
            Xn[m, :, 2] = (X_raw[m, :, 2] - p["cal_min"]) / (p["cal_max"] - p["cal_min"])
        if p["mets_max"] > p["mets_min"]:
            Xn[m, :, 3] = (X_raw[m, :, 3] - p["mets_min"]) / (p["mets_max"] - p["mets_min"])
    Sn = static_raw.astype(float).copy()
    for name, idx in STATIC_ZSCORE.items():
        g = glob[name]
        Sn[:, idx] = (static_raw[:, idx] - g["mean"]) / g["std"]
    for name, idx in STATIC_MACROS.items():
        g = glob[name]
        v = np.log1p(static_raw[:, idx]) if g.get("log_transformed", False) else static_raw[:, idx]
        Sn[:, idx] = (v - g["mean"]) / g["std"]
    return Xn, Sn


def make_folds(pid, n_splits=N_SPLITS):
    """Deterministic participant-level fold ids (GroupKFold is order-deterministic)."""
    fold = np.full(len(pid), -1, dtype=int)
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(pid))
    for k, (_, te) in enumerate(gkf.split(dummy, dummy, groups=pid)):
        fold[te] = k
    assert (fold >= 0).all()
    return fold


def main():
    with open(PARAMS_FILE, "rb") as f:
        all_params = pickle.load(f)

    for fname, key in FILES:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"SKIP (missing): {fname}")
            continue

        d = np.load(path, allow_pickle=True)
        Xn = d["X"].astype(float)
        static_n = d["static"].astype(float)
        y = d["y"].astype(float)                # already raw mg/dL -- keep
        pid = d["participant_id"]
        diagnosis = d["diagnosis"]

        P = all_params[key]
        per_subject = {str(k): v for k, v in P["per_subject"].items()}
        glob = P["global"]

        # --- recover raw absolute values ---
        X_raw = invert_timeseries(Xn, pid, per_subject)
        static_raw = invert_static(static_n, glob)

        # --- verify inversion is exact (re-normalize -> compare to stored) ---
        Xn_chk, Sn_chk = reapply_original_norm(X_raw, static_raw, pid, per_subject, glob)
        err_X = np.nanmax(np.abs(Xn_chk - Xn))
        err_S = np.nanmax(np.abs(Sn_chk - static_n))

        # --- fixed folds ---
        fold = make_folds(pid)

        out = os.path.join(OUT_DIR, fname.replace("_prediction.npz", "_prediction_raw.npz"))
        np.savez_compressed(
            out,
            X=X_raw.astype(np.float32),          # absolute units (CGM mg/dL, HR bpm, Cal, METs)
            static=static_raw.astype(np.float32),  # native units (HbA1c %, BMI, age, macros g/kcal, ...)
            y=y.astype(np.float32),              # mg/dL, unchanged
            participant_id=pid,
            diagnosis=diagnosis,
            fold=fold,                           # fixed 5-fold GroupKFold assignment
            static_names=np.array(STATIC_NAMES),
        )

        cgm = X_raw[:, :, 0]
        hba1c = static_raw[:, 3]
        bmi = static_raw[:, 2]
        age = static_raw[:, 0]
        print(f"\n=== {fname}  (key={key}) ===")
        print(f"  round-trip max err   X={err_X:.2e}   static={err_S:.2e}   "
              f"{'OK exact' if max(err_X, err_S) < 1e-3 else 'WARNING'}")
        print(f"  recovered CGM mg/dL  min {cgm.min():.1f}  max {cgm.max():.1f}  mean {cgm.mean():.1f}")
        print(f"  recovered HbA1c %    min {hba1c.min():.2f} max {hba1c.max():.2f}")
        print(f"  recovered BMI        min {bmi.min():.1f}  max {bmi.max():.1f}")
        print(f"  recovered Age        min {age.min():.0f}   max {age.max():.0f}")
        print(f"  fold sizes           {np.bincount(fold).tolist()}  (n={len(pid)}, subj={len(np.unique(pid))})")
        print(f"  saved -> {out}")

    print("\nDone. Raw, level-preserving .npz written to:", OUT_DIR)
    print("Apply normalization per fold at train time via foldwise_normalizer.py "
          "(fit on TRAIN participants only).")


if __name__ == "__main__":
    main()
