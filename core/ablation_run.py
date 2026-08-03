"""
ablation_run.py  -- incremental-value feature ablation under a pre-specified estimator
======================================================================================
Estimates the INCREMENTAL PREDICTIVE VALUE of each feature group *under a fixed
Random Forest* (primary) -- NOT a model-agnostic "what drives glucose" claim. A
second estimator (HistGradientBoosting) is available as a sensitivity analysis.

Protocol (shared with the light-DL benchmark):
  * participant-level, diagnosis-STRATIFIED repeated CV (make_repeated_splits),
  * per-fold leakage-free normalization (FoldNormalizer) fit on all NON-TEST
    participants (RF uses no validation set, so train = train+val),
  * targets mg/dL; per-sample OOF predictions saved for participant-cluster inference.

Fixes from review:
  * RF randomness is keyed on (CV seed, fold, horizon) ONLY -- identical across
    configs, so paired contrasts differ by features, not by tree bootstrap noise.
  * Meal macro OUTLIERS are capped (pre-declared plausibility caps); run CLEAN=raw
    for the raw sensitivity (log1p alone does not help trees).
  * Two persistence floors: persistence_1min (exact last reading) and
    persistence_5min (last 5-min bin, the resolution the RF models actually see).
  * Robust checkpoint: OOF rows written BEFORE the metric row (metric = completion
    marker); orphan OOF cells reconciled on resume; a MANIFEST guards against mixing
    results after any data/seed/tree/config/clean change.

Run (local, no GPU, minutes):
    python ablation_run.py                    # primary: rf + capped
    ESTIMATOR=hgb python ablation_run.py      # sensitivity estimator
    CLEAN=raw   python ablation_run.py        # raw-meal sensitivity
    python ablation_analyze.py                # inference + decision rule
Needs: dexcom_binned_prediction_raw.npz, dexcom_raw_prediction_raw.npz,
       foldwise_normalizer.py, common.py, lightdl_data.py
======================================================================================
"""
import os
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from lightdl_data import DataBundle, assert_aligned, highres_baseline, make_repeated_splits
from foldwise_normalizer import FoldNormalizer
from common import compute_metrics_per_horizon, HORIZONS

# ------------------------- knobs -------------------------
N_SEEDS = int(os.environ.get("N_SEEDS", "8"))
N_FOLDS = 5
RF_TREES = 300
BASE_SEED = 42
ESTIMATOR = os.environ.get("ESTIMATOR", "rf")     # rf (primary) | hgb (sensitivity)
CLEAN = os.environ.get("CLEAN", "capped")         # capped (primary) | raw (sensitivity)
TAG = f"{ESTIMATOR}_{CLEAN}"
RAW_CSV = f"ablation_raw_{TAG}.csv"
OOF_CSV = f"ablation_oof_{TAG}.csv"
MANIFEST = f"ablation_manifest_{TAG}.json"

# ------------------------- static-feature layout (verified vs static_names) -------------------------
IDX_PERSON = [0, 1, 2, 3]                       # age, gender, bmi, hba1c
IDX_MEAL   = [4, 5, 6, 7, 8]                    # calories, carbs, protein, fat, fiber
IDX_TIME   = [9, 10, 11, 12, 13, 14, 15, 16]   # hour_sin/cos, is_morning/evening/weekend/breakfast/lunch/dinner

# pre-declared per-meal plausibility caps (winsorize); RAW mode disables them
MEAL_COL = {"calories": 4, "carbs": 5, "protein": 6, "fat": 7, "fiber": 8}
MEAL_CAPS = {"calories": 2000.0, "carbs": 300.0, "protein": 200.0, "fat": 150.0, "fiber": 80.0}

# ------------------------- pre-declared feature sets -------------------------
CONFIGS = {
    "persistence_1min": None,   # predict exact last 1-min pre-meal CGM
    "persistence_5min": None,   # predict last 5-min bin (models' own resolution)
    "cgm":             ["cgm"],
    "dyn":             ["dyn"],
    "person":          ["person"],
    "meal":            ["meal"],
    "static_all":      ["person", "meal", "time"],
    "cgm_dyn":         ["cgm", "dyn"],
    "cgm_person":      ["cgm", "person"],
    "cgm_meal":        ["cgm", "meal"],
    "cgm_person_meal": ["cgm", "person", "meal"],
    "cgm_static":      ["cgm", "person", "meal", "time"],
    "full":            ["cgm", "dyn", "person", "meal", "time"],
}


def clean_static(S):
    if CLEAN == "raw":
        return S
    Sc = S.copy()
    for name, cap in MEAL_CAPS.items():
        Sc[:, MEAL_COL[name]] = np.minimum(Sc[:, MEAL_COL[name]], cap)
    return Sc


def make_estimator(rs):
    if ESTIMATOR == "rf":
        return RandomForestRegressor(n_estimators=RF_TREES, random_state=rs, n_jobs=-1)
    if ESTIMATOR == "hgb":
        return HistGradientBoostingRegressor(random_state=rs)
    raise ValueError(f"unknown ESTIMATOR {ESTIMATOR}")


def _rs(seed, fold):
    """RF/HGB seed keyed on (CV seed, fold) ONLY -- same across configs (paired)."""
    return int(hashlib.md5(f"{BASE_SEED}|{seed}|{fold}".encode()).hexdigest()[:8], 16)


def _feats(X, S, groups):
    parts = []
    if "cgm" in groups:    parts.append(X[:, :, 0])
    if "dyn" in groups:    parts.append(X[:, :, 1:].reshape(len(X), -1))
    if "person" in groups: parts.append(S[:, IDX_PERSON])
    if "meal" in groups:   parts.append(S[:, IDX_MEAL])
    if "time" in groups:   parts.append(S[:, IDX_TIME])
    return np.concatenate(parts, axis=1)


def _prep(bundle, Sclean, sp, base1):
    """Fit-normalize on ALL non-test participants (RF needs no val); return
    train/test feature substrate + both persistence baselines for the test rows."""
    tr_idx = np.concatenate([sp["train"], sp["val"]])
    te_idx = sp["test"]
    nrm = FoldNormalizer().fit(bundle.X[tr_idx], Sclean[tr_idx])
    Xtr, Str = nrm.transform(bundle.X[tr_idx], Sclean[tr_idx])
    Xte, Ste = nrm.transform(bundle.X[te_idx], Sclean[te_idx])
    return {"Xtr": Xtr, "Str": Str, "ytr": bundle.y[tr_idx],
            "Xte": Xte, "Ste": Ste, "yte": bundle.y[te_idx],
            "idx": te_idx, "pid": bundle.pid[te_idx], "diag": bundle.diag[te_idx],
            "base1": base1[te_idx], "base5": bundle.X[te_idx, -1, 0]}


def run_config(name, groups, fp, sp):
    if groups is None:                                   # persistence floors
        p = fp["base1"] if name == "persistence_1min" else fp["base5"]
        return np.repeat(p.astype("float32")[:, None], 3, axis=1)
    Ftr, Fte = _feats(fp["Xtr"], fp["Str"], groups), _feats(fp["Xte"], fp["Ste"], groups)
    preds = np.zeros((len(fp["yte"]), 3), "float32")
    base = _rs(sp["seed"], sp["fold"])
    for i in range(3):
        est = make_estimator((base + i) % (2**31))       # same base across configs; per-horizon offset
        est.fit(Ftr, fp["ytr"][:, i]); preds[:, i] = est.predict(Fte)
    return preds


# ------------------------- checkpoint / manifest -------------------------
def _append_csv(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _data_hash(bundle, Sclean):
    h = hashlib.md5()
    for a in (bundle.X, bundle.y, Sclean, bundle.diag):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(bundle.pid).encode())
    return h.hexdigest()


def _manifest(dh):
    return {"data_hash": dh, "configs": list(CONFIGS), "n_seeds": N_SEEDS, "n_folds": N_FOLDS,
            "estimator": ESTIMATOR, "rf_trees": RF_TREES, "clean": CLEAN,
            "meal_caps": MEAL_CAPS, "base_seed": BASE_SEED}


def _guard_and_resume(man):
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST))
        if old != man:
            raise SystemExit(f"[abort] {MANIFEST} differs from current settings "
                             f"(data/seeds/trees/config/clean changed). Delete "
                             f"ablation_*_{TAG}.* to start a clean run.")
    else:
        json.dump(man, open(MANIFEST, "w"), indent=2)
    done = set()
    if os.path.exists(RAW_CSV):
        done = set(map(tuple, pd.read_csv(RAW_CSV)[["model", "seed", "fold"]].values.tolist()))
    if os.path.exists(OOF_CSV):
        oof = pd.read_csv(OOF_CSV, dtype={"pid": str})
        cellkey = oof[["model", "seed", "fold"]].apply(tuple, axis=1)
        orphans = set(cellkey) - done
        if orphans:                                      # OOF written but metric never marked -> drop
            oof[~cellkey.isin(orphans)].to_csv(OOF_CSV, index=False)
            print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s) from an interrupted write")
    return done


def main():
    b, h = DataBundle("binned"), DataBundle("highres")
    assert_aligned(b, h)
    base1 = highres_baseline()
    Sclean = clean_static(b.S)
    done = _guard_and_resume(_manifest(_data_hash(b, Sclean)))
    splits = make_repeated_splits(b.pid, b.diag, n_seeds=N_SEEDS, n_folds=N_FOLDS, base_seed=BASE_SEED)
    print(f"TAG={TAG} | seeds={N_SEEDS} folds={N_FOLDS} -> {len(splits)} splits x {len(CONFIGS)} configs "
          f"| already done: {len(done)}")

    for si, sp in enumerate(splits):
        fp = _prep(b, Sclean, sp, base1)                 # shared feature substrate for this split
        for name, groups in CONFIGS.items():
            key = (name, sp["seed"], sp["fold"])
            if key in done:
                continue
            preds = run_config(name, groups, fp, sp)
            oof = pd.DataFrame({
                "model": name, "seed": sp["seed"], "fold": sp["fold"],
                "sample_idx": fp["idx"], "pid": fp["pid"], "diag": fp["diag"],
                **{f"y_true_{hh}": fp["yte"][:, i] for i, hh in enumerate(HORIZONS)},
                **{f"y_pred_{hh}": preds[:, i] for i, hh in enumerate(HORIZONS)},
            })
            _append_csv(OOF_CSV, oof)                     # OOF first ...
            m = compute_metrics_per_horizon(fp["yte"], preds)
            m.update({"model": name, "seed": sp["seed"], "fold": sp["fold"]})
            _append_csv(RAW_CSV, pd.DataFrame([m]))       # ... metric row = completion marker
            done.add(key)
            print(f"  [{si+1}/{len(splits)}] {name:16s} R2_60={m['R2_60']:+.3f} NRMSE60={m['NRMSE_60']:.3f}")

    print(f"\nDone TAG={TAG}. Wrote {RAW_CSV}, {OOF_CSV}, {MANIFEST}."
          f"\nNext: python ablation_analyze.py")


if __name__ == "__main__":
    main()
