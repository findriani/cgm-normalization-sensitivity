"""
attr_run.py -- E4: incremental value of CGM SHAPE once the pre-meal LEVEL is known
=================================================================================
E1 showed that destroying between-participant CGM level collapses CGM's measured
contribution. E4 asks the obvious follow-up: is the level ALL that matters?

    Once the model knows where a participant's glucose was sitting before the meal,
    does the SHAPE of the preceding hour add anything?

SCOPE -- WRITE THIS SENTENCE BEFORE ANY OTHER
---------------------------------------------
E4 estimates the incremental value of CGM shape UNDER THE PRESPECIFIED RANDOM FOREST,
ON CGMacros, WITH AN ABSOLUTE POSTPRANDIAL TARGET. That is the entire claim. It cannot
establish that sequence modelling is unnecessary in general, that one reading suffices,
or that the CGM trace carries no morphological information -- those need a sequence
model, other cohorts, and other targets. For novelty, say "we found no directly matched
decomposition under participant-independent evaluation and an absolute postprandial
target", never "nobody has published this".

ANCHOR CHANGE -- READ BEFORE REPORTING (2 August 2026)
------------------------------------------------------
The plan pre-registered the 12-bin WINDOW MEAN as the level statistic. The confirmatory
run under that choice (archived in `results_mean_anchor/`) found the mean to be a
significantly WORSE level summary than the final bin alone: dR2 = -0.033
[-0.060, -0.015], Holm p = 0.0048, 0 of 8 seeds favouring the mean. The level statistic
was therefore switched to the LAST BIN.

This is a data-driven change to a pre-registered choice and must be reported as one.
What makes it defensible is that the comparison itself was pre-registered -- the `L0_C`
arm existed in the plan precisely to test the level statistic -- so no new contrast was
mined. What changed is which arm the narrative calls "the level". Report the switch, its
direction and its reason. NEVER present the last-bin anchor as the original design.

THE DECOMPOSITION
-----------------
Normalization is held FIXED at the reference fold-wise global protocol in every
configuration, so nothing here is confounded with normalization -- only feature content
varies. Writing Z = (x - mu_g)/sigma_g for the globally normalized CGM window:

    L  = Z[:, -1]          the pre-meal LEVEL, the final bin (meal time)
    S  = Z - L             the pre-meal SHAPE, anchored on that bin
    Lm = mean_t Z          the 12-bin WINDOW MEAN -- retained as the comparator

S is the trajectory expressed as a DELTA against the meal-time reading:
Z - Z[:, -1] = (x - x_last)/sigma_g. Its final column is identically zero by
construction and is dropped, leaving 11 columns; the zero is asserted, not assumed.
This is the baseline-anchored delta representation of the prior-art report (GluNet's
`G_t + dG`), which is why the paper cites it as an established remedy and claims no
novelty for the representation itself.

L is an affine function of the raw final reading, and a Random Forest is invariant to
affine transforms of a single feature, so "level in raw mg/dL" and "level in normalized
units" are the same experiment here.

WHY Lm IS RETAINED
------------------
Keeping the window mean as its own arm preserves the pre-registered level-statistic
comparison, now stated in the new direction (last bin vs mean). Dropping it would erase
the evidence that motivated the switch.

NOTE -- THE FAMILY IS NOW FOUR, NOT FIVE. Once the level IS the last bin, "shape beyond
level" and "single-reading sufficiency" are the SAME contrast: L_S_C - L_C compares the
full window against that one reading. Holm therefore corrects over four contrasts.

CONFIGURATIONS
--------------
    C      context only                      (= the ablation's `static_all`)
    L_C    last-bin level + context
    S_C    shape + context
    L_S_C  level + shape + context
    Lm_C   12-bin window mean + context      (comparator: the pre-registered statistic)

BUILT-IN CHECKS
---------------
  * RECONSTRUCTION (not performance parity). Random Forests split on axis-aligned
    features and are NOT invariant to the change of coordinates from raw bins to
    level-plus-shape, so a performance difference between L_S_C and E1's `cgm_static` is
    a representation effect, not a bug. The correct check is numerical:
        x  ==  m + sigma_g * S      to floating-point tolerance
    asserted on every fold, for train and test.
  * PARITY. `C` uses exactly the ablation's `static_all` features, normalization and RF
    seed, so it must reproduce `ablation_oof_rf_capped.csv`. Checked in attr_analyze.py.

Run (local, CPU, ~20 min):
    python attr_run.py --smoke     # 1 seed, separate output files
    python attr_run.py             # 8 seeds, confirmatory
    python attr_analyze.py
=================================================================================
"""
import os
import sys
import json
import time
import hashlib
import argparse
import platform
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
for _p in (CORE, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Imported, not reimplemented: identical estimator, identical RF seeding, identical
# meal-outlier capping and static-feature layout as the primary ablation and E1.
from ablation_run import (clean_static, make_estimator, _rs, _feats,
                          IDX_PERSON, IDX_MEAL, IDX_TIME,
                          MEAL_CAPS, RF_TREES, BASE_SEED, N_FOLDS, CLEAN)
from lightdl_data import DataBundle, assert_aligned, make_repeated_splits
from foldwise_normalizer import FoldNormalizer
from common import compute_metrics_per_horizon, HORIZONS

N_SEEDS_DEFAULT = 8            # matches E1 and the other RF studies
RECON_TOL = 1e-3               # mg/dL; float32 bundle round-tripped through float64
LEVEL_TOL = 1e-9               # dimensionless; normalized-level identity, see below
CGM_CHANNEL = 0

# level: None | "last" (final bin, the level statistic) | "mean" (12-bin comparator)
# shape is anchored on the level statistic: S = Z - Z[:, -1], zero column dropped.
E4_CONFIGS = {
    "C":     {"level": None,   "shape": False, "label": "context only"},
    "L_C":   {"level": "last", "shape": False, "label": "level (last bin) + context"},
    "S_C":   {"level": None,   "shape": True,  "label": "shape + context"},
    "L_S_C": {"level": "last", "shape": True,  "label": "level + shape + context"},
    "Lm_C":  {"level": "mean", "shape": False, "label": "12-bin window mean + context"},
}

FIT_CODE = ("attr_run.py", "ablation_run.py", "lightdl_data.py",
            "foldwise_normalizer.py", "common.py")


# ----------------------------------------------------------------- features
def e4_feats(Z, S, spec):
    """Feature matrix for one configuration.

    `Z` is the globally normalized window (n, T, C); `S` the normalized statics.
    Context is built with the ablation's own `_feats`, so configuration `C` is
    byte-identical to `static_all` and the parity check in attr_analyze is meaningful."""
    z = Z[:, :, CGM_CHANNEL]                                  # (n, T)
    parts = []
    if spec["level"] == "last":
        parts.append(z[:, -1:])
    elif spec["level"] == "mean":
        parts.append(z.mean(axis=1, keepdims=True))
    if spec["shape"]:
        # Anchored on the final bin, so column -1 is identically zero. A constant
        # column carries no information and cannot be split on; drop it rather than
        # feed the forest a degenerate feature. Asserted, not assumed.
        d = z - z[:, -1:]
        if np.abs(d[:, -1]).max() != 0.0:
            raise SystemExit("[abort] last-bin shape anchor is not exactly zero -- "
                             "the decomposition is wrong.")
        parts.append(d[:, :-1])
    parts.append(_feats(Z, S, ["person", "meal", "time"]))     # context, ablation layout
    return np.concatenate(parts, axis=1)


def assert_reconstruction(X_raw, Z, gmu, gsd, where):
    """x == x_last + sigma_g * S, to tolerance, in RAW mg/dL.

    This is the integrity check for the decomposition -- NOT a comparison of model
    performance against E1. It catches axis errors, channel mix-ups and a stale
    normalizer. If it fails, nothing downstream is interpretable."""
    x = np.asarray(X_raw[:, :, CGM_CHANNEL], dtype=np.float64)
    z = np.asarray(Z[:, :, CGM_CHANNEL], dtype=np.float64)
    shape = z - z[:, -1:]                       # anchored on the final bin
    x_last = x[:, -1:]
    recon = x_last + gsd[CGM_CHANNEL] * shape
    err = float(np.abs(recon - x).max())
    if not np.isfinite(err) or err > RECON_TOL:
        raise SystemExit(f"[abort] reconstruction failed on {where}: "
                         f"max|x_last + sigma*S - x| = {err:.3e} mg/dL "
                         f"(tol {RECON_TOL:.0e}). The level/shape decomposition is wrong "
                         f"-- do not interpret any E4 result until this passes.")

    # The reconstruction above never touches gmu -- the global mean cancels when the
    # anchor bin is subtracted -- so it would pass even with an inconsistent or stale
    # ts_mean_. The level identity is what tests that: the LEVEL feature actually fed to
    # the forest is Z[:, -1], and it must equal (x_last - mu_g)/sigma_g. Enforced, not
    # merely measured; an unasserted diagnostic is not a check.
    lvl_err = float(np.abs(z[:, -1] - (x[:, -1] - gmu[CGM_CHANNEL])
                           / gsd[CGM_CHANNEL]).max())
    if not np.isfinite(lvl_err) or lvl_err > LEVEL_TOL:
        raise SystemExit(f"[abort] normalized-level identity failed on {where}: "
                         f"max|Z[:,-1] - (x_last - mu)/sigma| = {lvl_err:.3e} "
                         f"(tol {LEVEL_TOL:.0e}). The fold statistics used to build the "
                         f"LEVEL feature disagree with the normalizer -- do not interpret "
                         f"any E4 result until this passes.")

    # The window mean survives as the comparator arm `Lm_C`, so its own identity still
    # has to hold; without this the comparator could drift unchecked.
    mean_err = float(np.abs(z.mean(axis=1) - (x.mean(axis=1) - gmu[CGM_CHANNEL])
                            / gsd[CGM_CHANNEL]).max())
    if not np.isfinite(mean_err) or mean_err > LEVEL_TOL:
        raise SystemExit(f"[abort] window-mean identity failed on {where}: "
                         f"max|mean(Z) - (m - mu)/sigma| = {mean_err:.3e} "
                         f"(tol {LEVEL_TOL:.0e}).")
    return err, max(lvl_err, mean_err)


# ----------------------------------------------------------------- bookkeeping
def _sha(p):
    f = os.path.join(HERE, p)
    if not os.path.exists(f):
        f = os.path.join(CORE, p)
    with open(f, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def _data_hash(X, y, S, pid):
    h = hashlib.md5()
    for a in (X, y, S):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(pid).encode())
    return h.hexdigest()[:16]


def _manifest(dh, n_seeds):
    return {"experiment": "E4_level_vs_shape", "data_hash": dh,
            "configs": {k: {kk: vv for kk, vv in v.items()} for k, v in E4_CONFIGS.items()},
            "normalization": "fold-wise global (FoldNormalizer), fit on train+val "
                             "participants only -- identical in every configuration",
            "n_seeds": n_seeds, "n_folds": N_FOLDS, "base_seed": BASE_SEED,
            "rf_trees": RF_TREES, "clean": CLEAN, "meal_caps": MEAL_CAPS,
            "recon_tol": RECON_TOL, "level_tol": LEVEL_TOL,
            "code_hash": {f: _sha(f) for f in FIT_CODE},
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__}


def _guard_and_resume(man, raw_csv, oof_csv, manifest_path):
    if os.path.exists(raw_csv) and not os.path.exists(manifest_path):
        raise SystemExit(f"[abort] {raw_csv} exists with no {manifest_path}. Move it aside.")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        for k in ("data_hash", "configs", "n_folds", "base_seed", "rf_trees",
                  "clean", "code_hash"):
            if old.get(k) != man.get(k):
                raise SystemExit(f"[abort] manifest mismatch on {k!r}:\n  stored:  "
                                 f"{old.get(k)}\n  current: {man.get(k)}\n"
                                 f"Delete the attr_* result files to start clean.")
    else:
        json.dump(man, open(manifest_path, "w"), indent=2)
    # The RAW metric row is written AFTER the OOF rows and marks completion. A crash
    # between the two writes leaves orphan OOF predictions for a cell that will be
    # re-run on resume, silently duplicating rows. Reconcile exactly as ablation_run does.
    done = set()
    if os.path.exists(raw_csv):
        d = pd.read_csv(raw_csv)
        done = set(zip(d.model, d.seed, d.fold))
    if os.path.exists(oof_csv):
        oof = pd.read_csv(oof_csv, dtype={"pid": str})
        cellkey = list(zip(oof.model, oof.seed, oof.fold))
        orphans = set(cellkey) - done
        if orphans:
            keep = [k not in orphans for k in cellkey]
            oof[keep].to_csv(oof_csv, index=False)
            print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s) "
                  f"from an interrupted write")
    return done


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def fit_config(F_tr, F_te, ytr, n_te, sp):
    """Same estimator and the SAME per-fold RF seed as the ablation and E1, so every
    contrast differs by features rather than by bootstrap noise."""
    preds = np.zeros((n_te, 3), "float32")
    base = _rs(sp["seed"], sp["fold"])
    for i in range(3):
        est = make_estimator((base + i) % (2 ** 31))
        est.fit(F_tr, ytr[:, i])
        preds[:, i] = est.predict(F_te)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed, separate output files")
    ap.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    args = ap.parse_args()
    n_seeds = 1 if args.smoke else args.seeds
    sfx = "_smoke" if args.smoke else ""
    raw_csv = os.path.join(HERE, f"attr_results_raw{sfx}.csv")
    oof_csv = os.path.join(HERE, f"attr_oof{sfx}.csv")
    manifest = os.path.join(HERE, f"attr_manifest{sfx}.json")

    t0 = time.time()
    b, h = DataBundle("binned"), DataBundle("highres")
    assert_aligned(b, h)
    Sclean = clean_static(b.S)

    man = _manifest(_data_hash(b.X, b.y, Sclean, b.pid), n_seeds)
    done = _guard_and_resume(man, raw_csv, oof_csv, manifest)
    splits = make_repeated_splits(b.pid, b.diag, n_seeds=n_seeds, n_folds=N_FOLDS,
                                  base_seed=BASE_SEED)
    total = len(splits) * len(E4_CONFIGS)
    print(f"E4 | CLEAN={CLEAN} seeds={n_seeds} folds={N_FOLDS} -> {len(splits)} splits "
          f"x {len(E4_CONFIGS)} configs = {total} cells | done: {len(done)}")
    print(f"    window = {b.T} bins ({b.T * 5} min at 5-min resolution); "
          f"level = FINAL bin (meal time); shape anchored on it ({b.T - 1} cols); "
          f"Lm = {b.T}-bin mean, retained as comparator")
    for k, v in E4_CONFIGS.items():
        print(f"    {k:7s} {v['label']}")

    worst_recon = 0.0
    worst_level = 0.0
    for si, sp in enumerate(splits):
        tr_idx = np.concatenate([sp["train"], sp["val"]])   # RF needs no validation split
        te_idx = sp["test"]

        # Exactly ablation_run._prep: fold-wise global, fit on training participants only.
        nrm = FoldNormalizer().fit(b.X[tr_idx], Sclean[tr_idx])
        Ztr, Str = nrm.transform(b.X[tr_idx], Sclean[tr_idx])
        Zte, Ste = nrm.transform(b.X[te_idx], Sclean[te_idx])
        gmu, gsd = nrm.ts_mean_, nrm.ts_std_

        e_tr, l_tr = assert_reconstruction(b.X[tr_idx], Ztr, gmu, gsd, f"split {si} train")
        e_te, l_te = assert_reconstruction(b.X[te_idx], Zte, gmu, gsd, f"split {si} test")
        worst_recon = max(worst_recon, e_tr, e_te)
        worst_level = max(worst_level, l_tr, l_te)

        shown = {}
        for name, spec in E4_CONFIGS.items():
            key = (name, sp["seed"], sp["fold"])
            if key in done:
                continue
            F_tr = e4_feats(Ztr, Str, spec)
            F_te = e4_feats(Zte, Ste, spec)
            preds = fit_config(F_tr, F_te, b.y[tr_idx], len(te_idx), sp)

            _append(oof_csv, pd.DataFrame({
                "model": name, "seed": sp["seed"], "fold": sp["fold"],
                "sample_idx": te_idx, "pid": b.pid[te_idx], "diag": b.diag[te_idx],
                **{f"y_true_{hh}": b.y[te_idx][:, i] for i, hh in enumerate(HORIZONS)},
                **{f"y_pred_{hh}": preds[:, i] for i, hh in enumerate(HORIZONS)}}))
            m = compute_metrics_per_horizon(b.y[te_idx], preds)
            m.update({"model": name, "seed": sp["seed"], "fold": sp["fold"],
                      "n_features": F_tr.shape[1],
                      "recon_max_err": float(max(e_tr, e_te)),
                      "level_max_err": float(max(l_tr, l_te))})
            _append(raw_csv, pd.DataFrame([m]))
            shown[name] = m["R2_60"]
            done.add(key)

        if shown:
            trace = "  ".join(f"{c}={shown[c]:+.3f}" for c in E4_CONFIGS if c in shown)
            print(f"  [{si+1}/{len(splits)}] recon={max(e_tr, e_te):.2e}  {trace}")

    print(f"\nDone. {len(done)} cells in {(time.time() - t0) / 60:.1f} min.")
    print(f"Decomposition checks passed on every fold (train and test):")
    print(f"  worst |m + sigma*S - x|             = {worst_recon:.3e} mg/dL (tol {RECON_TOL:.0e})")
    print(f"  worst |mean(Z) - (m - mu)/sigma|    = {worst_level:.3e} (tol {LEVEL_TOL:.0e})")
    print(f"Wrote {raw_csv}, {oof_csv}, {manifest}")
    print("Next: python attr_analyze.py" + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()
