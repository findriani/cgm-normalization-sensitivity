"""
shanghai_run.py -- E1 re-run on ShanghaiT2DM: does C1 replicate across cohorts?
===============================================================================
The modality ablation, five times, identical in every respect except the CGM normalization
protocol. Same design as `e1_run.py`; different cohort.

    python shanghai_run.py --smoke     # 1 seed, writes to *_smoke.csv
    python shanghai_run.py             # 8 seeds, confirmatory

CPU only, no TensorFlow. Shanghai's job is C1 and nothing else (REVISION_PLAN §6).

WHAT IS IMPORTED RATHER THAN REBUILT
-------------------------------------
The estimator, its seeding, the split rule, the metrics and every normalization arm come
from the CGMacros pipeline unchanged:

    ablation_run.make_estimator, _rs, RF_TREES, BASE_SEED, N_FOLDS
    lightdl_data.make_repeated_splits
    common.compute_metrics_per_horizon
    e1_normalizers._apply / PROTOCOLS / PROPERTIES  (via shanghai_normalizers)

So a difference between cohorts cannot be an artifact of a differently-implemented
estimator, seed, split or metric. Only the DATA and the STATIC LAYOUT differ.

TWO DELIBERATE DEPARTURES FROM e1_run.py
-----------------------------------------
1. `clean_static` IS NOT CALLED. It winsorizes static columns 4-8, which are meal
   macronutrients in CGMacros and CLOCK FEATURES in Shanghai. There is no meal content in
   this cohort, so there is nothing to winsorize and calling it would be meaningless at
   best. `CLEAN` is recorded as "n/a" in the manifest so the absence is explicit.

2. `static_all` = person + time, NOT person + meal + time. Shanghai has no meal-content
   modality. The context arm is therefore WEAKER than CGMacros', which matters for reading
   the result: see the interpretation note at the bottom of this docstring.

READING THE RESULT -- WHAT REPLICATION WOULD AND WOULD NOT MEAN
----------------------------------------------------------------
The primary estimand is E1's, unchanged:

    [cgm_static - static_all]_global - [cgm_static - static_all]_subject_z

A positive interaction replicates C1 across cohorts. But Shanghai's context branch lacks
meal content, so its context increment is smaller by construction and the CGM increment
correspondingly larger. The INTERACTION is still a fair test -- context is held identical
across arms within this cohort -- but the per-arm increments are NOT comparable to
CGMacros' numbers. Compare interactions to interactions; never put the two cohorts' CGM
increments side by side as though they measured the same thing.
===============================================================================
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

from ablation_run import make_estimator, _rs, RF_TREES, BASE_SEED, N_FOLDS
from common import compute_metrics_per_horizon, HORIZONS
from shanghai_data import ShanghaiBundle, CONFIGS, feats, make_repeated_splits, STATIC_NAMES
from shanghai_normalizers import (PROTOCOLS, PROPERTIES, normalize_split,
                                  assert_statics_invariant,
                                  between_subject_level_retained)

RUN_CONFIGS = ["persistence", "cgm", "static_all", "cgm_static"]
FIT_CODE = ("shanghai_run.py", "shanghai_data.py", "shanghai_normalizers.py",
            "shanghai_extract.py")
CORE_CODE = ("e1_normalizers.py", "ablation_run.py", "lightdl_data.py",
             "common.py")


def _sha(p):
    for d in (HERE, CORE):
        f = os.path.join(d, p)
        if os.path.exists(f):
            with open(f, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()[:16]
    return "missing"


def _data_hash(X, y, S, pid):
    h = hashlib.md5()
    for a in (X, y, S):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(pid).encode())
    return h.hexdigest()[:16]


def _manifest(dh, n_seeds, b):
    return {"experiment": "shanghai_external_C1", "cohort": "ShanghaiT2DM",
            "data_hash": dh, "protocols": list(PROTOCOLS), "properties": PROPERTIES,
            "configs": RUN_CONFIGS, "static_layout": STATIC_NAMES,
            "n_meals": int(b.n), "n_participants": int(len(np.unique(b.pid))),
            "n_channels": int(b.C), "window_offsets_min": b.pre_offsets,
            "n_seeds": n_seeds, "n_folds": N_FOLDS, "base_seed": BASE_SEED,
            "rf_trees": RF_TREES, "clean": "n/a -- no meal content in this cohort",
            "context_definition": "person + time (NO meal content; not CGMacros' context)",
            "subject_z_note": "one channel, so no per-subject min-max fires; subject_z is "
                              "the CGM component of the published transform",
            "diagnosis_field": "HbA1c tertile, NOT a clinical diagnosis",
            "code_hash": {f: _sha(f) for f in FIT_CODE + CORE_CODE},
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__}


def _guard_and_resume(man, raw_csv, oof_csv, manifest_path):
    if os.path.exists(raw_csv) and not os.path.exists(manifest_path):
        raise SystemExit(f"[abort] {raw_csv} exists with no {manifest_path}. Move it aside.")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        for k in ("data_hash", "protocols", "configs", "static_layout", "n_folds",
                  "base_seed", "rf_trees", "code_hash"):
            if old.get(k) != man.get(k):
                raise SystemExit(f"[abort] manifest mismatch on {k!r}:\n  stored:  "
                                 f"{old.get(k)}\n  current: {man.get(k)}\n"
                                 f"Delete the shanghai_results_* files to start clean.")
    else:
        json.dump(man, open(manifest_path, "w"), indent=2)
    # The RAW metric row is written AFTER the OOF rows, so it is the completion marker. A
    # crash between the two leaves orphan OOF predictions for a cell that will be re-run,
    # silently duplicating rows. Same reconciliation as e1_run.py.
    done = set()
    if os.path.exists(raw_csv):
        d = pd.read_csv(raw_csv)
        done = set(zip(d.protocol, d.model, d.seed, d.fold))
    if os.path.exists(oof_csv):
        oof = pd.read_csv(oof_csv, dtype={"pid": str})
        ck = list(zip(oof.protocol, oof.model, oof.seed, oof.fold))
        orphans = set(ck) - done
        if orphans:
            oof[[k not in orphans for k in ck]].to_csv(oof_csv, index=False)
            print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s)")
    return done


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_config(name, groups, prep, sp):
    """One configuration on one fold. Seeding is `_rs(seed, fold)` from ablation_run --
    identical across arms and configs, so the arms are paired fit-for-fit."""
    if groups is None:
        return np.repeat(prep["base"].astype("float32")[:, None], 3, axis=1)
    Ftr = feats(prep["Xtr"], prep["Str"], groups)
    Fte = feats(prep["Xte"], prep["Ste"], groups)
    preds = np.zeros((len(prep["yte"]), 3), "float32")
    base = _rs(sp["seed"], sp["fold"])
    for i in range(3):
        est = make_estimator((base + i) % (2 ** 31))
        est.fit(Ftr, prep["ytr"][:, i])
        preds[:, i] = est.predict(Fte)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed, separate output files")
    ap.add_argument("--seeds", type=int, default=8, help="matches E1's 8 seeds")
    args = ap.parse_args()
    n_seeds = 1 if args.smoke else args.seeds
    sfx = "_smoke" if args.smoke else ""
    raw_csv = os.path.join(HERE, f"shanghai_results_raw{sfx}.csv")
    oof_csv = os.path.join(HERE, f"shanghai_oof{sfx}.csv")
    manifest = os.path.join(HERE, f"shanghai_run_manifest{sfx}.json")

    t0 = time.time()
    b = ShanghaiBundle()
    unknown = [c for c in RUN_CONFIGS if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"[abort] configs not in shanghai_data.CONFIGS: {unknown}")

    man = _manifest(_data_hash(b.X, b.y, b.S, b.pid), n_seeds, b)
    done = _guard_and_resume(man, raw_csv, oof_csv, manifest)
    splits = make_repeated_splits(b.pid, b.diag, n_seeds=n_seeds, n_folds=N_FOLDS,
                                  base_seed=BASE_SEED)
    total = len(splits) * len(PROTOCOLS) * len(RUN_CONFIGS)
    print(f"SHANGHAI | seeds={n_seeds} folds={N_FOLDS} -> {len(splits)} splits x "
          f"{len(PROTOCOLS)} protocols x {len(RUN_CONFIGS)} configs = {total} cells "
          f"| done: {len(done)}")
    print(f"  {b}")
    print(f"  context = person + time (NO meal content -- not CGMacros' context)")
    for p in PROTOCOLS:
        q = PROPERTIES[p]
        print(f"    {p:16s} level_preserved={str(q['level_preserved']):5s} "
              f"leakage={str(q['leakage']):5s}  {q['label']}")

    for si, sp in enumerate(splits):
        tr_idx = np.concatenate([sp["train"], sp["val"]])   # RF needs no validation split
        te_idx = sp["test"]
        if si == 0:
            # The real control, checked once on the first split: the arms must differ ONLY
            # in the CGM transform. Cheap, and it aborts before any fitting is wasted.
            assert_statics_invariant(b.X, b.S, b.pid, tr_idx, te_idx)
            print("  statics invariant across all five arms: OK")
        for proto in PROTOCOLS:
            if all((proto, c, sp["seed"], sp["fold"]) in done for c in RUN_CONFIGS):
                continue
            Xtr, Str, Xte, Ste = normalize_split(b.X, b.S, b.pid, tr_idx, te_idx, proto)
            lvl_te = between_subject_level_retained(Xte, b.pid[te_idx])
            prep = {"Xtr": Xtr, "Str": Str, "ytr": b.y[tr_idx],
                    "Xte": Xte, "Ste": Ste, "yte": b.y[te_idx],
                    "idx": te_idx, "pid": b.pid[te_idx], "diag": b.diag[te_idx],
                    "base": b.X[te_idx, -1, 0]}          # t=0 raw reading, mg/dL
            shown = {}
            for name in RUN_CONFIGS:
                key = (proto, name, sp["seed"], sp["fold"])
                if key in done:
                    continue
                preds = run_config(name, CONFIGS[name], prep, sp)
                _append(oof_csv, pd.DataFrame({
                    "protocol": proto, "model": name, "seed": sp["seed"],
                    "fold": sp["fold"], "sample_idx": prep["idx"], "pid": prep["pid"],
                    "diag": prep["diag"],
                    **{f"y_true_{h}": prep["yte"][:, i] for i, h in enumerate(HORIZONS)},
                    **{f"y_pred_{h}": preds[:, i] for i, h in enumerate(HORIZONS)}}))
                m = compute_metrics_per_horizon(prep["yte"], preds)
                m.update({"protocol": proto, "model": name, "seed": sp["seed"],
                          "fold": sp["fold"], "level_retained": lvl_te,
                          "level_preserved": PROPERTIES[proto]["level_preserved"],
                          "leakage": PROPERTIES[proto]["leakage"]})
                _append(raw_csv, pd.DataFrame([m]))
                shown[name] = m["R2_60"]
                done.add(key)
            if shown:
                trace = "  ".join(f"{c}={shown[c]:+.3f}" for c in
                                  ("cgm", "static_all", "cgm_static") if c in shown)
                print(f"  [{si+1}/{len(splits)}] {proto:16s} level={lvl_te:.3f}  {trace}",
                      flush=True)

    print(f"\nDone. {len(done)} cells in {(time.time() - t0) / 60:.1f} min.")
    print(f"Wrote {raw_csv}, {oof_csv}, {manifest}")
    print("Next: analysis -- reuse e1_analyze.did_stats on shanghai_oof.csv")


if __name__ == "__main__":
    main()
