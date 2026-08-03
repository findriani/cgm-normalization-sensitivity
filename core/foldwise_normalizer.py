"""
foldwise_normalizer.py
======================================================================
Leakage-free, level-preserving normalization for the leakage-free pipeline.

Use this INSIDE the cross-validation loop: fit on the TRAINING participants
only, then transform train and test. This is the fix for the two review
defects:

  * NO LEAKAGE  -- statistics come only from training-fold participants.
  * LEVEL PRESERVED -- time series are standardized GLOBALLY (one mean/std per
    channel across all training samples), NOT per subject. Global scaling keeps
    between-subject differences, so the absolute glucose baseline survives in
    the CGM branch (the information the old per-subject z-score destroyed).

Design choices, matched to the original intent where harmless:
  * CGM, HR, Calories(Activity), METs: global z-score (fit on train).
  * Static age/BMI/HbA1c: global z-score.
  * Static macronutrients (calories/carbs/protein/fat/fiber): log1p if the
    TRAIN split is skewed (|skew|>1), then z-score -- same rule as before, but
    the skew test and stats are computed on TRAIN only.
  * Static gender + time/meal indicators (indices 1, 9..16): left untouched.
  * Targets y: never normalized (kept in mg/dL).

Example
-------
    import numpy as np
    from foldwise_normalizer import FoldNormalizer

    d = np.load("dexcom_binned_prediction_raw.npz", allow_pickle=True)
    X, static, y, fold = d["X"], d["static"], d["y"], d["fold"]

    for k in range(fold.max() + 1):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        nrm = FoldNormalizer().fit(X[tr], static[tr])
        Xtr, Str = nrm.transform(X[tr], static[tr])
        Xte, Ste = nrm.transform(X[te], static[te])
        # ... train model on (Xtr, Str, y[tr]); evaluate on (Xte, Ste, y[te]) ...
======================================================================
"""

import numpy as np
from scipy.stats import skew

# static layout (must match build_data.py)
STATIC_ZSCORE_IDX = {"age": 0, "bmi": 2, "hba1c": 3}
STATIC_MACRO_IDX = {"calories": 4, "carbs": 5, "protein": 6, "fat": 7, "fiber": 8}
STATIC_PASSTHROUGH_IDX = [1, 9, 10, 11, 12, 13, 14, 15, 16]  # gender + time/meal
SKEW_THRESHOLD = 1.0
_EPS = 1e-8


class FoldNormalizer:
    """Fit on training data only; preserves absolute level via global scaling."""

    def __init__(self):
        self.ts_mean_ = None      # (n_channels,)
        self.ts_std_ = None       # (n_channels,)
        self.static_ = {}         # idx -> dict(mean, std, log)

    # ---- fit ----
    def fit(self, X_train, static_train):
        X_train = np.asarray(X_train, dtype=float)
        static_train = np.asarray(static_train, dtype=float)

        # time series: one mean/std per channel, pooled over samples*timesteps
        n_ch = X_train.shape[2]
        flat = X_train.reshape(-1, n_ch)
        self.ts_mean_ = np.nanmean(flat, axis=0)
        self.ts_std_ = np.nanstd(flat, axis=0)
        self.ts_std_[self.ts_std_ < _EPS] = 1.0   # guard constant channels

        # static: age/BMI/HbA1c z-score
        self.static_ = {}
        for _, idx in STATIC_ZSCORE_IDX.items():
            col = static_train[:, idx]
            m, s = np.nanmean(col), np.nanstd(col)
            self.static_[idx] = {"mean": float(m), "std": float(s if s > _EPS else 1.0), "log": False}

        # static macronutrients: log1p if train-skewed, then z-score
        for _, idx in STATIC_MACRO_IDX.items():
            col = static_train[:, idx]
            clean = col[~np.isnan(col)]
            do_log = len(clean) > 0 and abs(skew(clean)) > SKEW_THRESHOLD
            vals = np.log1p(col) if do_log else col
            m, s = np.nanmean(vals), np.nanstd(vals)
            self.static_[idx] = {"mean": float(m), "std": float(s if s > _EPS else 1.0), "log": do_log}

        return self

    # ---- transform ----
    def transform(self, X, static):
        if self.ts_mean_ is None:
            raise RuntimeError("FoldNormalizer must be fit() before transform().")
        X = np.asarray(X, dtype=float).copy()
        static = np.asarray(static, dtype=float).copy()

        X = (X - self.ts_mean_) / self.ts_std_

        for idx, p in self.static_.items():
            col = static[:, idx]
            if p["log"]:
                col = np.log1p(col)
            static[:, idx] = (col - p["mean"]) / p["std"]
        # passthrough indices (gender, time, meal-type) left unchanged
        return X, static

    def fit_transform(self, X_train, static_train):
        self.fit(X_train, static_train)
        return self.transform(X_train, static_train)


# quick self-test on the leakage-free data (optional): python foldwise_normalizer.py
if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    f = os.path.join(here, "dexcom_binned_prediction_raw.npz")
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        X, static, fold = d["X"], d["static"], d["fold"]
        for k in range(int(fold.max()) + 1):
            tr = np.where(fold != k)[0]
            te = np.where(fold == k)[0]
            nrm = FoldNormalizer().fit(X[tr], static[tr])
            Xtr, Str = nrm.transform(X[tr], static[tr])
            Xte, Ste = nrm.transform(X[te], static[te])
            print(f"fold {k}: train CGM mean={Xtr[:,:,0].mean():+.3f} std={Xtr[:,:,0].std():.3f} | "
                  f"test CGM mean={Xte[:,:,0].mean():+.3f} (nonzero test mean = level preserved, no leakage)")
    else:
        print("Run build_data.py first to create the .npz files.")
