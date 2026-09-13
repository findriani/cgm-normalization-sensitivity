"""
sens_trees_run.py -- sensitivity of the primary result to Random Forest tree count
==================================================================================
Reviewer question: does the number of trees (currently 300) affect the conclusion?

Runs the E1 primary comparison (global vs subject_z, CGM increment at 60 min) for
tree counts of 100, 300, 500, and 1000. Everything else is identical to e1_run.py:
same 8 seeds, 5 folds, same splits, same normalization, same feature slicing.

Only the two protocols and two configurations needed for the CGM increment are run,
so each tree-count setting adds 2 protocols x 2 configs x 8 seeds x 5 folds = 160
cells. Each cell trains 3 horizons, giving 480 RF fits per tree count and 1,920
fits total.

    python sens_trees_run.py             # full 8-seed run
    python sens_trees_run.py --smoke     # 1 seed for a quick check

Output: sens_trees_oof{sfx}.csv    per-meal OOF predictions
        sens_trees_raw{sfx}.csv    per-cell aggregate metrics
        sens_trees_manifest{sfx}.json

Analysis: python sens_trees_analyze.py
==================================================================================
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
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ablation_run import (CONFIGS as ABL_CONFIGS, clean_static, _rs, _feats,
                          BASE_SEED, N_FOLDS, MEAL_CAPS, CLEAN)
from lightdl_data import DataBundle, assert_aligned, highres_baseline, make_repeated_splits
from common import compute_metrics_per_horizon, HORIZONS
from e1_normalizers import normalize_split, between_subject_level_retained, PROPERTIES

TREE_COUNTS = [100, 300, 500, 1000]
PROTOCOLS = ["global", "subject_z"]
CONFIGS = ["cgm_static", "static_all"]


def _data_hash(X, y, S, pid):
    h = hashlib.md5()
    for a in (X, y, S):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(pid).encode())
    return h.hexdigest()[:16]


def _manifest(dh, n_seeds):
    return {"data_hash": dh, "tree_counts": TREE_COUNTS,
            "protocols": list(PROTOCOLS), "configs": CONFIGS,
            "n_seeds": n_seeds, "n_folds": N_FOLDS,
            "base_seed": BASE_SEED, "clean": CLEAN, "meal_caps": MEAL_CAPS,
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__}


def _guard_and_resume(man, raw_csv, oof_csv, manifest_path):
    """Manifest check + OOF reconciliation, same pattern as e1_run.py."""
    if os.path.exists(raw_csv) and not os.path.exists(manifest_path):
        raise SystemExit(f"[abort] {raw_csv} exists with no {manifest_path}. "
                         f"Move it aside.")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        for k in ("data_hash", "tree_counts", "protocols", "configs",
                  "n_folds", "base_seed", "clean"):
            if old.get(k) != man.get(k):
                raise SystemExit(
                    f"[abort] manifest mismatch on {k!r}:\n"
                    f"  stored:  {old.get(k)}\n  current: {man.get(k)}\n"
                    f"Delete the sens_trees_* result files to start clean.")
    else:
        json.dump(man, open(manifest_path, "w"), indent=2)
    # Reconcile: drop OOF rows whose completion marker (raw_csv row) is missing
    done = set()
    if os.path.exists(raw_csv):
        d = pd.read_csv(raw_csv)
        done = set(zip(d.trees, d.protocol, d.model, d.seed, d.fold))
    if os.path.exists(oof_csv):
        oof = pd.read_csv(oof_csv, dtype={"pid": str})
        cellkey = list(zip(oof.trees, oof.protocol, oof.model, oof.seed, oof.fold))
        orphans = set(cellkey) - done
        if orphans:
            keep = [k not in orphans for k in cellkey]
            oof[keep].to_csv(oof_csv, index=False)
            print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s)")
    return done


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_config(name, groups, prep, sp, n_trees):
    Ftr = _feats(prep["Xtr"], prep["Str"], groups)
    Fte = _feats(prep["Xte"], prep["Ste"], groups)
    preds = np.zeros((len(prep["yte"]), 3), "float32")
    base = _rs(sp["seed"], sp["fold"])
    for i in range(3):
        est = RandomForestRegressor(n_estimators=n_trees,
                                    random_state=(base + i) % (2 ** 31),
                                    n_jobs=-1)
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
    raw_csv = os.path.join(HERE, f"sens_trees_raw{sfx}.csv")
    oof_csv = os.path.join(HERE, f"sens_trees_oof{sfx}.csv")
    manifest_path = os.path.join(HERE, f"sens_trees_manifest{sfx}.json")

    t0 = time.time()
    b, h = DataBundle("binned"), DataBundle("highres")
    assert_aligned(b, h)
    Sclean = clean_static(b.S)

    man = _manifest(_data_hash(b.X, b.y, Sclean, b.pid), n_seeds)
    done = _guard_and_resume(man, raw_csv, oof_csv, manifest_path)

    splits = make_repeated_splits(b.pid, b.diag, n_seeds=n_seeds, n_folds=N_FOLDS,
                                  base_seed=BASE_SEED)

    total = len(TREE_COUNTS) * len(splits) * len(PROTOCOLS) * len(CONFIGS)
    print(f"Tree-count sensitivity | trees={TREE_COUNTS} seeds={n_seeds} "
          f"folds={N_FOLDS} -> {total} cells | done: {len(done)}")

    for n_trees in TREE_COUNTS:
        for si, sp in enumerate(splits):
            tr_idx = np.concatenate([sp["train"], sp["val"]])
            te_idx = sp["test"]
            for proto in PROTOCOLS:
                if all((n_trees, proto, c, sp["seed"], sp["fold"]) in done
                       for c in CONFIGS):
                    continue
                Xtr, Str, Xte, Ste = normalize_split(
                    b.X, Sclean, b.pid, tr_idx, te_idx, proto)
                prep = {"Xtr": Xtr, "Str": Str, "ytr": b.y[tr_idx],
                        "Xte": Xte, "Ste": Ste, "yte": b.y[te_idx],
                        "idx": te_idx, "pid": b.pid[te_idx],
                        "diag": b.diag[te_idx]}
                for name in CONFIGS:
                    key = (n_trees, proto, name, sp["seed"], sp["fold"])
                    if key in done:
                        continue
                    preds = run_config(name, ABL_CONFIGS[name], prep, sp, n_trees)
                    # OOF predictions first (written before the completion marker)
                    _append(oof_csv, pd.DataFrame({
                        "trees": n_trees,
                        "protocol": proto, "model": name,
                        "seed": sp["seed"], "fold": sp["fold"],
                        "sample_idx": prep["idx"], "pid": prep["pid"],
                        "diag": prep["diag"],
                        **{f"y_true_{hh}": prep["yte"][:, i]
                           for i, hh in enumerate(HORIZONS)},
                        **{f"y_pred_{hh}": preds[:, i]
                           for i, hh in enumerate(HORIZONS)}}))
                    # Completion marker
                    m = compute_metrics_per_horizon(prep["yte"], preds)
                    m.update({"trees": n_trees, "protocol": proto, "model": name,
                              "seed": sp["seed"], "fold": sp["fold"]})
                    _append(raw_csv, pd.DataFrame([m]))
                    done.add(key)
                print(f"  trees={n_trees} [{si+1}/{len(splits)}] {proto}")

    elapsed = (time.time() - t0) / 60
    print(f"\nDone. {len(done)} cells in {elapsed:.1f} min.")
    print(f"Wrote {raw_csv}, {oof_csv}, {manifest_path}")
    print("Next: python sens_trees_analyze.py" + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()
