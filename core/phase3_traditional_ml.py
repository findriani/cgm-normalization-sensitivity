"""
phase3_traditional_ml.py  -- Table 7 reconstruction (fair traditional-ML baselines)
======================================================================
No surviving code existed for the paper's Table 7, so this reconstructs it
to be FAIR (the reviewer's request): every regressor sees the SAME features,
the SAME participant folds, and the SAME per-fold normalization as
the deep models. Reports all three horizons.

Features per meal (matches what the DL models receive, flattened):
  CGM 12  +  dynamic 12x3=36  +  static 17   = 65 features
All normalized per fold (fit on train only) via common.get_fold.

Methods: Dummy(mean), Ridge, SVR(RBF), RandomForest, LightGBM, CatBoost.
Missing optional libs (lightgbm/catboost) are skipped with a note.

Run in Colab:  !pip install catboost lightgbm  then  !python phase3_traditional_ml.py
======================================================================
"""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

from common import get_fold, compute_metrics_per_horizon, METRIC_COLS, N_FOLDS, HORIZONS, RANDOM_STATE

warnings.filterwarnings("ignore")


def make_regressor(name):
    if name == "Dummy (mean)":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    if name == "SVR":
        return SVR(kernel="rbf", C=10.0, gamma="scale")
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=400, random_state=RANDOM_STATE, verbose=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=400, random_seed=RANDOM_STATE, verbose=0)
    raise ValueError(name)


def flatten_features(X, S):
    """X (n,12,4) normalized, S (n,17) normalized -> (n,65)."""
    cgm = X[:, :, 0]                    # (n,12)
    dyn = X[:, :, 1:].reshape(len(X), -1)  # (n,36)
    return np.concatenate([cgm, dyn, S], axis=1)


METHODS = ["Dummy (mean)", "Ridge", "SVR", "Random Forest", "LightGBM", "CatBoost"]


def run_method(name):
    rows = []
    for fold_idx in range(N_FOLDS):
        fd = get_fold(fold_idx)
        Xtr, Str, ytr = fd["train"]; Xte, Ste, yte = fd["test"]
        Ftr, Fte = flatten_features(Xtr, Str), flatten_features(Xte, Ste)
        preds = np.zeros_like(yte)
        for i, _h in enumerate(HORIZONS):
            reg = make_regressor(name)
            reg.fit(Ftr, ytr[:, i])
            preds[:, i] = reg.predict(Fte)
        m = compute_metrics_per_horizon(yte, preds)
        m["fold"] = fold_idx; m["method"] = name
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    summary = []
    for name in METHODS:
        try:
            df = run_method(name)
        except ImportError:
            print(f"[skip] {name}: library not installed (pip install lightgbm catboost)")
            continue
        df.to_csv(f"results_traditional_{name.split()[0].lower()}.csv", index=False)
        s = df[METRIC_COLS].describe().loc[["mean", "std"]]
        summary.append({"method": name,
                        **{f"{m}_mean": s.loc["mean", m] for m in METRIC_COLS},
                        **{f"{m}_std": s.loc["std", m] for m in METRIC_COLS}})
        print(f"{name:16s} NRMSE60={s.loc['mean','NRMSE_60']:.3f}  R2_60={s.loc['mean','R2_60']:+.3f}")

    out = pd.DataFrame(summary).sort_values("NRMSE_60_mean").set_index("method")
    out.to_csv("table7_traditional_ml.csv")
    print("\nSaved table7_traditional_ml.csv (sorted by NRMSE_60)")
    print(out[["NRMSE_60_mean", "NRMSE_60_std", "R2_60_mean"]].to_string())


if __name__ == "__main__":
    main()
