"""
traditional_run.py -- FAIRLY TRAINED traditional-ML baselines (replaces paper Table 7)
======================================================================================
Runs LOCALLY on CPU. No TensorFlow import: the deep-learning side of the comparison is
read from the attention study's existing OOF (`attn_oof_raw.csv`), which was fitted on
THESE EXACT SPLITS -- same base_seed, same 15 seeds, same 5 folds, same participant
partitions -- so the comparison is PAIRED per meal, not two independent columns.

WHY THE PAPER'S TABLE 7 CANNOT STAND
------------------------------------
It reports CatBoost NRMSE=0.264 against a mean-predictor dummy at 0.340. A gradient
booster with 257 informative features landing that close to the mean is not a weak
method; it is a broken fit. The code that produced it no longer exists, so the table
cannot be audited or defended in a response letter. This file rebuilds it from scratch.

WHAT "FAIRLY TRAINED" MEANS HERE -- every baseline gets the same deal the DL model got:
  1. SAME SPLITS.        make_repeated_splits(..., base_seed=42), participant-level and
                         diagnosis-stratified. Identical objects to lightdl/attn runs.
  2. SAME NORMALIZATION. FoldNormalizer fit on TRAINING participants only. No leakage,
                         absolute glucose level preserved.
  3. SAME FEATURES.      flatten(X, S) -- the identical normalized arrays the DL model
                         consumes, just laid flat: CGM(T) + dyn(3T) + static(17).
  4. SAME TARGET HANDLING. Targets standardized on TRAIN stats and inverted after
                         prediction, exactly as attn_run._standardize_targets does.
                         (Critical for SVR; a no-op for trees.)
  5. A REAL VALIDATION BUDGET. Each family gets a per-fold hyperparameter search scored
                         on the SAME validation participants the DL model uses for early
                         stopping, and the boosters get early stopping on that set. The
                         original table almost certainly used library defaults. Handing
                         the deep model early stopping while the baselines run at
                         defaults is precisely the asymmetry Reviewer 1 objected to.
  6. FIT ON TRAIN ONLY.  Not train+val, so the DL model holds no data disadvantage.

Predicting each horizon with its own regressor is the standard tabular treatment and
matches the existing `rf_highres` baseline.

Outputs (resumable; a cell is (model, seed, fold)):
  trad_results_raw.csv   per model/seed/fold NRMSE, R2, MARD at 30/60/120
  trad_oof.csv           per-sample OOF predictions, schema identical to attn_oof_raw.csv
  trad_best_params.csv   the hyperparameters selected per fold (auditable)
  trad_manifest.json     data hash + settings; guards a resume against edited code
======================================================================================
"""
import os
import json
import time
import hashlib
import platform
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from lightdl_data import (DataBundle, assert_aligned, make_repeated_splits,
                          normalized_fold, flatten)
from common import compute_metrics_per_horizon, HORIZONS

HERE = os.path.dirname(os.path.abspath(__file__))

SMOKE = os.environ.get("SMOKE", "0") == "1"
# 8 seeds, matching the modality ablation's seed count. The DL side of the comparison
# (attn_oof_raw.csv) has 15; pairing uses the common seeds 0-7, which is exact because
# make_repeated_splits builds seed s from base_seed + s independently of n_seeds.
N_SEEDS = 1 if SMOKE else int(os.environ.get("N_SEEDS", "8"))
N_FOLDS = 5
BASE_SEED = 42                       # MUST match attn_run.BASE_SEED for paired comparison
RF_TREES = 300                       # matches the existing rf_highres baseline
RF_SEARCH_TREES = 150                # grid is scored at half size, winner refit at RF_TREES
BOOST_CAP = 800                      # early stopping fires well before this on n~730
BOOST_PATIENCE = 40


# _seed_for and _append_csv are COPIED VERBATIM from lightdl_run.py rather than imported,
# because that module imports TensorFlow and this run is CPU/sklearn only. verify_helpers()
# below re-imports the originals when TF happens to be available and asserts they agree,
# so a future edit there cannot silently desynchronize the seeding.
def _seed_for(seed, fold, name):
    """Stable, order-independent seed for one fit (recorded in outputs)."""
    h = hashlib.md5(f"{BASE_SEED}|{seed}|{fold}|{name}".encode()).hexdigest()
    return int(h[:8], 16)


def _append_csv(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def verify_helpers():
    """No-op unless TensorFlow is importable. Guards the verbatim copies above."""
    try:
        import lightdl_run as LR
    except Exception:
        return "skipped (no tensorflow)"
    assert LR.BASE_SEED == BASE_SEED, f"BASE_SEED drift: {LR.BASE_SEED} vs {BASE_SEED}"
    for s, f, n in ((0, 0, "ridge@highres_60"), (7, 3, "catboost@highres_30")):
        assert LR._seed_for(s, f, n) == _seed_for(s, f, n), "_seed_for drift"
    return "ok"


RAW_CSV = "trad_results_raw.csv"
OOF_CSV = "trad_oof.csv"
PARAM_CSV = "trad_best_params.csv"
MANIFEST = "trad_manifest.json"
OUT_FILES = (RAW_CSV, OOF_CSV, PARAM_CSV)
FIT_CODE = ("traditional_run.py", "lightdl_data.py", "foldwise_normalizer.py",
            "common.py")

# Optional boosters. Absent libraries are skipped with a loud note rather than crashing,
# but CatBoost and LightGBM are named in the paper's own table so their absence would
# leave the replacement incomplete.
HAVE = {}
try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    HAVE["lightgbm"] = True
except Exception:
    HAVE["lightgbm"] = False
try:
    from catboost import CatBoostRegressor
    HAVE["catboost"] = True
except Exception:
    HAVE["catboost"] = False
try:
    from xgboost import XGBRegressor
    HAVE["xgboost"] = True
except Exception:
    HAVE["xgboost"] = False


# --------------------------------------------------------------- hyperparameter grids
# Deliberately small but covering the axis each family is actually sensitive to.
GRIDS = {
    "ridge":         [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)],
    "svr_rbf":       [{"C": c, "gamma": g, "epsilon": e}
                      for c in (1.0, 10.0, 100.0) for g in ("scale", 0.01) for e in (0.05, 0.2)],
    "random_forest": [{"max_features": mf, "min_samples_leaf": ml}
                      for mf in ("sqrt", 0.3, 1.0) for ml in (1, 3)],
    "hist_gbm":      [{"learning_rate": lr, "max_leaf_nodes": nl}
                      for lr in (0.05, 0.1) for nl in (15, 31)],
    "lightgbm":      [{"learning_rate": lr, "num_leaves": nl}
                      for lr in (0.05, 0.1) for nl in (15, 31)],
    "xgboost":       [{"learning_rate": lr, "max_depth": d}
                      for lr in (0.05, 0.1) for d in (3, 6)],
    "catboost":      [{"learning_rate": lr, "depth": d}
                      for lr in (0.05, 0.1) for d in (4, 6)],
}

TRAD = ["ridge", "svr_rbf", "random_forest", "hist_gbm", "lightgbm", "xgboost", "catboost"]
MODELS = (["dummy_mean", "persistence_5min"]
          + [f"{m}@highres" for m in TRAD]
          # binned variants: check that input resolution is not what decides the table
          + ["ridge@binned", "random_forest@binned", "hist_gbm@binned"])
MODELS = [m for m in MODELS
          if m.split("@")[0] not in HAVE or HAVE[m.split("@")[0]]]


def _make(name, params, seed, final=False):
    """Instantiate one family. `seed` is the deterministic per-cell seed. `final=True`
    means this is the refit of the winning configuration, not a grid probe."""
    rs = seed % (2 ** 31)
    if name == "ridge":
        return Ridge(**params)
    if name == "svr_rbf":
        return SVR(kernel="rbf", **params)
    if name == "random_forest":
        # Scored at half size, winner refit at full size. Ranking RF configurations is
        # stable in the number of trees; only the final variance reduction needs 300.
        return RandomForestRegressor(n_estimators=RF_TREES if final else RF_SEARCH_TREES,
                                     random_state=rs, n_jobs=-1, **params)
    if name == "hist_gbm":
        return HistGradientBoostingRegressor(max_iter=BOOST_CAP, early_stopping=False,
                                             random_state=rs, **params)
    if name == "lightgbm":
        return LGBMRegressor(n_estimators=BOOST_CAP, random_state=rs, n_jobs=-1,
                             verbose=-1, **params)
    if name == "xgboost":
        return XGBRegressor(n_estimators=BOOST_CAP, random_state=rs, n_jobs=-1,
                            early_stopping_rounds=BOOST_PATIENCE, verbosity=0, **params)
    if name == "catboost":
        return CatBoostRegressor(iterations=BOOST_CAP, random_seed=rs, verbose=0,
                                 allow_writing_files=False, **params)
    raise ValueError(name)


def _fit_one(name, params, Xtr, ztr, Xva, zva, seed, final=False):
    """Fit with early stopping on the validation participants where the family supports
    it -- the direct analogue of the DL model's MainTrajectoryEarlyStopping."""
    mdl = _make(name, params, seed, final=final)
    if name == "lightgbm":
        mdl.fit(Xtr, ztr, eval_X=Xva, eval_y=zva,
                callbacks=[early_stopping(BOOST_PATIENCE, verbose=False), log_evaluation(0)])
    elif name == "xgboost":
        mdl.fit(Xtr, ztr, eval_set=[(Xva, zva)], verbose=False)
    elif name == "catboost":
        mdl.fit(Xtr, ztr, eval_set=(Xva, zva),
                early_stopping_rounds=BOOST_PATIENCE, verbose=0)
    else:
        mdl.fit(Xtr, ztr)
    return mdl


def _select_and_fit(name, Xtr, ytr, Xva, yva, seed):
    """Per-fold search on the validation fold. Returns (fitted model, params, mu, sd).

    Target standardization uses TRAIN statistics only and is inverted after prediction,
    mirroring attn_run._standardize_targets exactly."""
    mu, sd = float(ytr.mean()), float(ytr.std())
    sd = sd if sd > 1e-6 else 1.0
    ztr, zva = (ytr - mu) / sd, (yva - mu) / sd

    best, best_rmse, best_p = None, np.inf, None
    for p in GRIDS[name]:
        try:
            mdl = _fit_one(name, p, Xtr, ztr, Xva, zva, seed)
            r = float(np.sqrt(np.mean((mdl.predict(Xva) - zva) ** 2)))
        except Exception as exc:                      # a grid point failing must not kill the cell
            print(f"      ! {name} {p} failed: {type(exc).__name__}: {exc}")
            continue
        if np.isfinite(r) and r < best_rmse:
            best, best_rmse, best_p = mdl, r, p
    if best is None:
        raise RuntimeError(f"every grid point failed for {name}")
    if name == "random_forest":                       # refit the winner at full size
        best = _fit_one(name, best_p, Xtr, ztr, Xva, zva, seed, final=True)
    return best, best_p, mu, sd


def run_model(model, fh, fb, seed, fold, base5):
    """Returns (preds (n_test,3) in mg/dL, params_record). Test rows are always the
    HIGH-RES fold's test indices; the bundles are row-aligned so every model's OOF
    lines up with the DL runs sample-for-sample."""
    te_h = fh["test"]
    n = len(te_h["y"])

    if model == "persistence_5min":
        p = base5[te_h["idx"]].astype("float32")
        return np.repeat(p[:, None], 3, axis=1), {}

    if model == "dummy_mean":
        preds = np.zeros((n, 3), "float32")
        for i in range(3):
            d = DummyRegressor(strategy="mean").fit(
                np.zeros((len(fh["train"]["y"]), 1)), fh["train"]["y"][:, i])
            preds[:, i] = d.predict(np.zeros((n, 1)))
        return preds, {}

    name, feat = model.split("@")
    fd = fh if feat == "highres" else fb
    tr, va, te = fd["train"], fd["val"], fd["test"]
    Ftr, Fva, Fte = (flatten(tr["X"], tr["S"]), flatten(va["X"], va["S"]),
                     flatten(te["X"], te["S"]))

    preds = np.zeros((n, 3), "float32")
    rec = {}
    for i, h in enumerate(HORIZONS):
        s = _seed_for(seed, fold, f"{model}_{h}")
        mdl, p, mu, sd = _select_and_fit(name, Ftr, tr["y"][:, i], Fva, va["y"][:, i], s)
        preds[:, i] = mdl.predict(Fte) * sd + mu
        rec[f"params_{h}"] = json.dumps(p)
    return preds, rec


# --------------------------------------------------------------- manifest / resume
def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _data_hash(b):
    return hashlib.sha256(np.ascontiguousarray(b.X).tobytes()
                          + np.ascontiguousarray(b.y).tobytes()).hexdigest()[:16]


def _manifest(dh_hi, dh_bi, n, n_part):
    return {"data_hash_highres": dh_hi, "data_hash_binned": dh_bi, "n_samples": int(n),
            "n_participants": int(n_part), "n_seeds": N_SEEDS, "n_folds": N_FOLDS,
            "base_seed": BASE_SEED, "rf_trees": RF_TREES, "models": MODELS,
            "grids": {k: len(v) for k, v in GRIDS.items()},
            "code_hash": {f: _sha(os.path.join(HERE, f)) for f in FIT_CODE},
            "python": platform.python_version(),
            "libs": {m: (HAVE.get(m) and __import__(m).__version__) for m in
                     ("lightgbm", "catboost", "xgboost")},
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def _guard_and_resume(man):
    """Resume is per (model, seed, fold). Refuse to append to results produced by
    different code or different data -- the failure mode that makes a table unauditable
    is exactly what happened to the original Table 7."""
    existing = [f for f in OUT_FILES if os.path.exists(os.path.join(HERE, f))]
    if existing and not os.path.exists(os.path.join(HERE, MANIFEST)):
        raise SystemExit(f"[abort] {existing} exist with no {MANIFEST}. Move them aside.")
    if os.path.exists(os.path.join(HERE, MANIFEST)):
        old = json.load(open(os.path.join(HERE, MANIFEST)))
        for k in ("data_hash_highres", "data_hash_binned", "n_samples", "base_seed",
                  "n_folds", "rf_trees", "code_hash"):
            if old.get(k) != man.get(k):
                raise SystemExit(f"[abort] manifest mismatch on {k!r}:\n"
                                 f"  stored:  {old.get(k)}\n  current: {man.get(k)}\n"
                                 f"Delete the result CSVs to start clean.")
    else:
        json.dump(man, open(os.path.join(HERE, MANIFEST), "w"), indent=2)
    done = set()
    if os.path.exists(os.path.join(HERE, RAW_CSV)):
        r = pd.read_csv(os.path.join(HERE, RAW_CSV))
        done = set(zip(r.model, r.seed, r.fold))
    return done


def main():
    t0 = time.time()
    highres = DataBundle("highres"); binned = DataBundle("binned")
    assert_aligned(binned, highres)
    base5 = highres.X[:, -5:, 0].mean(axis=1)
    n_part = len(np.unique(highres.pid))
    done = _guard_and_resume(_manifest(_data_hash(highres), _data_hash(binned),
                                       highres.n, n_part))
    splits = make_repeated_splits(highres.pid, highres.diag, n_seeds=N_SEEDS,
                                  n_folds=N_FOLDS, base_seed=BASE_SEED)
    total = len(splits) * len(MODELS)
    missing = [m for m in ("lightgbm", "catboost", "xgboost") if not HAVE.get(m)]
    print(f"SMOKE={SMOKE} seeds={N_SEEDS} folds={N_FOLDS} -> {len(splits)} splits x "
          f"{len(MODELS)} models = {total} cells | done: {len(done)}")
    print(f"  {highres.n} samples / {n_part} participants | features: "
          f"highres={highres.T * 4 + highres.n_static}, binned={binned.T * 4 + binned.n_static}")
    if missing:
        print(f"  ** NOT INSTALLED, skipped: {missing} **")
    print(f"  helper parity vs lightdl_run: {verify_helpers()}")

    for si, sp in enumerate(splits):
        if all((m, sp["seed"], sp["fold"]) in done for m in MODELS):
            continue
        fh = normalized_fold(highres, sp, with_aux=False)
        fb = normalized_fold(binned, sp, with_aux=False)
        te = fh["test"]
        for model in MODELS:
            if (model, sp["seed"], sp["fold"]) in done:
                continue
            preds, rec = run_model(model, fh, fb, sp["seed"], sp["fold"], base5)

            oof = pd.DataFrame({                              # OOF first, marker row last
                "model": model, "seed": sp["seed"], "fold": sp["fold"],
                "sample_idx": te["idx"], "pid": te["pid"], "diag": te["diag"],
                **{f"y_true_{h}": te["y"][:, i] for i, h in enumerate(HORIZONS)},
                **{f"y_pred_{h}": preds[:, i] for i, h in enumerate(HORIZONS)}})
            _append_csv(OOF_CSV, oof)
            if rec:
                _append_csv(PARAM_CSV, pd.DataFrame([{"model": model, "seed": sp["seed"],
                                                      "fold": sp["fold"], **rec}]))
            m = compute_metrics_per_horizon(te["y"], preds)
            m.update({"model": model, "seed": sp["seed"], "fold": sp["fold"]})
            _append_csv(RAW_CSV, pd.DataFrame([m]))
            done.add((model, sp["seed"], sp["fold"]))
            print(f"  [{si+1}/{len(splits)}] {model:24s} R2_60={m['R2_60']:+.3f} "
                  f"NRMSE60={m['NRMSE_60']:.3f}")

    print(f"\nDone. {len(done)} cells in {(time.time() - t0)/60:.1f} min. "
          f"Next: python traditional_analyze.py")


if __name__ == "__main__":
    main()
