"""
sens_meal_exclusion_run.py -- excluding implausible meals instead of capping them
=================================================================================
Reviewer question: what happens if meals with implausible macronutrient values
are excluded entirely, rather than having their values capped?

19 of 913 meals (2.1%) exceed at least one cap threshold (6 participants).
This script removes those meals and re-runs the E1 primary comparison
(global vs subject_z, CGM increment at 60 min) with 8 seeds and 5 folds.

The result should be compared with the capped run (e1_interaction.csv) and
the uncapped run (Table S13) to show how the handling of implausible values
affects the primary conclusion.

    python sens_meal_exclusion_run.py             # full 8-seed run
    python sens_meal_exclusion_run.py --smoke     # 1 seed

Output: sens_meal_exclusion_oof{sfx}.csv   per-meal OOF predictions
        sens_meal_exclusion_raw{sfx}.csv   per-cell metrics (completion marker)
        sens_meal_exclusion_manifest{sfx}.json

Analysis: python sens_meal_exclusion_analyze.py
=================================================================================
"""
import os
import sys
import json
import time
import hashlib
import platform
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ablation_run import (CONFIGS as ABL_CONFIGS, MEAL_CAPS, MEAL_COL, _rs, _feats,
                          BASE_SEED, N_FOLDS, RF_TREES, CLEAN)
from sklearn.ensemble import RandomForestRegressor
from lightdl_data import DataBundle, assert_aligned, highres_baseline, make_repeated_splits
from common import compute_metrics_per_horizon, HORIZONS
from e1_normalizers import (normalize_split, between_subject_level_retained,
                            PROPERTIES)

PROTOCOLS = ["global", "subject_z"]
CONFIGS = ["persistence_5min", "cgm", "static_all", "cgm_static"]


def find_implausible(S_raw):
    """Return boolean mask: True for meals where ANY macronutrient exceeds its cap."""
    exceed = np.zeros(len(S_raw), dtype=bool)
    for name, cap in MEAL_CAPS.items():
        idx = MEAL_COL[name]
        exceed |= (S_raw[:, idx] > cap)
    return exceed


def _data_hash(X, y, S, pid):
    h = hashlib.md5()
    for a in (X, y, S):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(pid).encode())
    return h.hexdigest()[:16]


def _manifest(dh, n_seeds, n_excluded, n_kept):
    return {"data_hash": dh, "protocols": list(PROTOCOLS), "configs": CONFIGS,
            "n_seeds": n_seeds, "n_folds": N_FOLDS,
            "base_seed": BASE_SEED, "rf_trees": RF_TREES, "clean": "excluded",
            "meal_caps": MEAL_CAPS, "n_excluded": n_excluded, "n_kept": n_kept,
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__}


def _guard_and_resume(man, raw_csv, oof_csv, manifest_path):
    """Manifest check + OOF reconciliation, same pattern as e1_run.py."""
    if os.path.exists(raw_csv) and not os.path.exists(manifest_path):
        raise SystemExit(f"[abort] {raw_csv} exists with no {manifest_path}. "
                         f"Move it aside.")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        for k in ("data_hash", "protocols", "configs", "n_folds", "base_seed",
                  "rf_trees", "clean", "n_excluded", "n_kept"):
            if old.get(k) != man.get(k):
                raise SystemExit(
                    f"[abort] manifest mismatch on {k!r}:\n"
                    f"  stored:  {old.get(k)}\n  current: {man.get(k)}\n"
                    f"Delete the sens_meal_exclusion_* result files to start clean.")
    else:
        json.dump(man, open(manifest_path, "w"), indent=2)
    # Reconcile: drop OOF rows whose completion marker (raw_csv row) is missing
    done = set()
    if os.path.exists(raw_csv):
        d = pd.read_csv(raw_csv)
        done = set(zip(d.protocol, d.model, d.seed, d.fold))
    if os.path.exists(oof_csv):
        oof = pd.read_csv(oof_csv, dtype={"pid": str})
        cellkey = list(zip(oof.protocol, oof.model, oof.seed, oof.fold))
        orphans = set(cellkey) - done
        if orphans:
            keep = [k not in orphans for k in cellkey]
            oof[keep].to_csv(oof_csv, index=False)
            print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s)")
    return done


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def make_estimator(rs):
    return RandomForestRegressor(n_estimators=RF_TREES, random_state=rs, n_jobs=-1)


def run_config(name, groups, prep, sp):
    if groups is None:
        return np.repeat(prep["base5"].astype("float32")[:, None], 3, axis=1)
    Ftr = _feats(prep["Xtr"], prep["Str"], groups)
    Fte = _feats(prep["Xte"], prep["Ste"], groups)
    preds = np.zeros((len(prep["yte"]), 3), "float32")
    base = _rs(sp["seed"], sp["fold"])
    for i in range(3):
        est = make_estimator((base + i) % (2 ** 31))
        est.fit(Ftr, prep["ytr"][:, i])
        preds[:, i] = est.predict(Fte)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed only")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    n_seeds = 1 if args.smoke else args.seeds
    sfx = "_smoke" if args.smoke else ""
    raw_csv = os.path.join(HERE, f"sens_meal_exclusion_raw{sfx}.csv")
    oof_csv = os.path.join(HERE, f"sens_meal_exclusion_oof{sfx}.csv")
    manifest_path = os.path.join(HERE, f"sens_meal_exclusion_manifest{sfx}.json")

    t0 = time.time()
    b, h = DataBundle("binned"), DataBundle("highres")
    assert_aligned(b, h)

    # Identify and exclude implausible meals (using RAW static values, before capping)
    implausible = find_implausible(b.S)
    n_excluded = int(implausible.sum())
    n_kept = int((~implausible).sum())
    excluded_pids = np.unique(b.pid[implausible])

    print(f"Meal exclusion sensitivity")
    print(f"  Total meals: {len(b.S)}")
    print(f"  Excluded (exceed any cap): {n_excluded} ({n_excluded/len(b.S)*100:.1f}%)")
    print(f"  Kept: {n_kept}")
    print(f"  Participants with excluded meals: {len(excluded_pids)} of "
          f"{len(np.unique(b.pid))}")
    print(f"  Cap thresholds: {MEAL_CAPS}")
    print()

    # Build filtered arrays -- NO capping, just exclusion
    keep = ~implausible
    X_filt = b.X[keep]
    S_filt = b.S[keep]   # raw values, no capping
    y_filt = b.y[keep]
    pid_filt = b.pid[keep]
    diag_filt = b.diag[keep]

    # Map filtered positions back to original bundle indices (for OOF tracking)
    orig_idx = np.where(keep)[0]

    man = _manifest(_data_hash(X_filt, y_filt, S_filt, pid_filt),
                    n_seeds, n_excluded, n_kept)
    done = _guard_and_resume(man, raw_csv, oof_csv, manifest_path)

    splits = make_repeated_splits(pid_filt, diag_filt, n_seeds=n_seeds,
                                  n_folds=N_FOLDS, base_seed=BASE_SEED)

    total = len(splits) * len(PROTOCOLS) * len(CONFIGS)
    print(f"  seeds={n_seeds} folds={N_FOLDS} -> {len(splits)} splits "
          f"x {len(PROTOCOLS)} protocols x {len(CONFIGS)} configs = {total} cells "
          f"| done: {len(done)}")

    for si, sp in enumerate(splits):
        tr_idx = np.concatenate([sp["train"], sp["val"]])
        te_idx = sp["test"]
        for proto in PROTOCOLS:
            if all((proto, c, sp["seed"], sp["fold"]) in done for c in CONFIGS):
                continue
            Xtr, Str, Xte, Ste = normalize_split(
                X_filt, S_filt, pid_filt, tr_idx, te_idx, proto)
            lvl_te = between_subject_level_retained(Xte, pid_filt[te_idx])
            prep = {"Xtr": Xtr, "Str": Str, "ytr": y_filt[tr_idx],
                    "Xte": Xte, "Ste": Ste, "yte": y_filt[te_idx],
                    "idx": orig_idx[te_idx],
                    "pid": pid_filt[te_idx], "diag": diag_filt[te_idx],
                    "base5": X_filt[te_idx, -1, 0]}
            for name in CONFIGS:
                key = (proto, name, sp["seed"], sp["fold"])
                if key in done:
                    continue
                preds = run_config(name, ABL_CONFIGS[name], prep, sp)
                # OOF predictions first
                _append(oof_csv, pd.DataFrame({
                    "protocol": proto, "model": name, "seed": sp["seed"],
                    "fold": sp["fold"],
                    "sample_idx": prep["idx"], "pid": prep["pid"],
                    "diag": prep["diag"],
                    **{f"y_true_{hh}": prep["yte"][:, i]
                       for i, hh in enumerate(HORIZONS)},
                    **{f"y_pred_{hh}": preds[:, i]
                       for i, hh in enumerate(HORIZONS)}}))
                # Completion marker
                m = compute_metrics_per_horizon(prep["yte"], preds)
                m.update({"protocol": proto, "model": name, "seed": sp["seed"],
                          "fold": sp["fold"], "level_retained": lvl_te,
                          "n_meals": n_kept, "n_excluded": n_excluded})
                _append(raw_csv, pd.DataFrame([m]))
                done.add(key)
            print(f"  [{si+1}/{len(splits)}] {proto:16s} level={lvl_te:.3f}")

    elapsed = (time.time() - t0) / 60
    print(f"\nDone. {len(done)} cells in {elapsed:.1f} min.")
    print(f"Wrote {raw_csv}, {oof_csv}, {manifest_path}")
    print("Next: python sens_meal_exclusion_analyze.py"
          + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()
