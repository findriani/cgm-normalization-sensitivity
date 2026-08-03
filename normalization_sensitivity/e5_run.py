"""
e5_run.py -- E5: does the normalization effect transfer to a DIFFERENT modality?
================================================================================
E1 varied the CGM transform. E5 runs the identical protocol against the ACTIVITY
channels, with CGM pinned to the reference global transform in every arm.

    Does per-subject normalization of activity distort ACTIVITY's measured
    attribution, the way per-subject CGM normalization distorts CGM's?

THE GATE -- PRE-REGISTERED, READ BEFORE INTERPRETING ANYTHING
-------------------------------------------------------------
The primary ablation already found that activity/HR adds little or nothing beyond
CGM + context. If the reference arm shows no activity contribution to begin with, then
there is NO attribution for normalization to distort, and a null interaction is
uninformative about distortion. In that case E5 must be reported as a SPECIFICITY check
-- the protocol correctly returns null where nothing is there -- and NOT as evidence that
normalization is harmless for activity. e5_analyze applies this gate automatically and
refuses to print the distortion reading when it fires.

Stated in advance so the conclusion cannot be chosen after seeing the numbers.

ARMS (activity channels only; CGM identical in all three)
---------------------------------------------------------
    global          fold-wise global                       reference
    subject_z       per-subject z / min-max, leaky         published style
    premeal_center  causal centering on own window, no leakage

CONFIGURATIONS
--------------
    static_all   context only                    -- uses NO temporal channels
    cgm_static   CGM + context                   -- uses NO activity channels
    dyn_static   activity + context              -- activity's own contribution
    full         CGM + activity + context        -- the primary increment's numerator

The first two use no activity data, so their predictions MUST be bit-identical across
arms. They are refitted independently in every arm rather than shared, which makes that a
real control instead of a construction -- CPU is cheap here, and E2's shared-fit design
needed extra machinery to stay safe under resume.

PRIMARY ESTIMAND
----------------
    DDR2 = [full - cgm_static]_global  -  [full - cgm_static]_subject_z

the same difference-in-differences, paired at the individual meal, with the same
participant-cluster bootstrap and sign-flip permutation as E1.

Run (local, CPU, ~25 min):
    python e5_run.py --smoke      # 1 seed, separate output files
    python e5_run.py              # 8 seeds, confirmatory
    python e5_analyze.py
================================================================================
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

from ablation_run import (clean_static, make_estimator, _rs, _feats,
                          CONFIGS, MEAL_CAPS, RF_TREES, BASE_SEED, N_FOLDS, CLEAN)
from lightdl_data import DataBundle, assert_aligned, make_repeated_splits
from common import compute_metrics_per_horizon, HORIZONS
from e5_normalizers import (ARMS, PROPERTIES, DYN_CHANNELS, normalize_split_dyn,
                            assert_cgm_untouched, between_subject_level_retained_dyn)

N_SEEDS_DEFAULT = 8
E5_CONFIGS = ["static_all", "cgm_static", "dyn_static", "full"]
# Configurations that touch no activity channel -- invariant across arms by design.
DYN_FREE = ["static_all", "cgm_static"]

FIT_CODE = ("e5_run.py", "e5_normalizers.py", "e1_normalizers.py", "ablation_run.py",
            "lightdl_data.py", "foldwise_normalizer.py", "common.py")

# `dyn_static` is not in the ablation's CONFIGS; define it with the same group vocabulary.
GROUPS = {
    "static_all": ["person", "meal", "time"],
    "cgm_static": ["cgm", "person", "meal", "time"],
    "dyn_static": ["dyn", "person", "meal", "time"],
    "full":       ["cgm", "dyn", "person", "meal", "time"],
}


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
    return {"experiment": "E5_cross_modality_activity", "data_hash": dh,
            "arms": {a: PROPERTIES[a] for a in ARMS},
            "configs": {k: GROUPS[k] for k in E5_CONFIGS},
            "dyn_channels": list(DYN_CHANNELS),
            "held_fixed": "CGM channel 0 = FoldNormalizer output in every arm",
            "normalization": "statics fold-wise global, training participants only, "
                             "identical in every arm",
            "n_seeds": n_seeds, "n_folds": N_FOLDS, "base_seed": BASE_SEED,
            "rf_trees": RF_TREES, "clean": CLEAN, "meal_caps": MEAL_CAPS,
            "code_hash": {f: _sha(f) for f in FIT_CODE},
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__}


def _guard_and_resume(man, raw_csv, oof_csv, manifest_path):
    if os.path.exists(raw_csv) and not os.path.exists(manifest_path):
        raise SystemExit(f"[abort] {raw_csv} exists with no {manifest_path}. Move it aside.")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        for k in ("data_hash", "arms", "configs", "n_folds", "base_seed", "rf_trees",
                  "clean", "code_hash"):
            if old.get(k) != man.get(k):
                raise SystemExit(f"[abort] manifest mismatch on {k!r}:\n  stored:  "
                                 f"{old.get(k)}\n  current: {man.get(k)}\n"
                                 f"Delete the e5_* result files to start clean.")
    else:
        json.dump(man, open(manifest_path, "w"), indent=2)
    # RAW row is written AFTER the OOF rows and marks the cell complete; a crash between
    # the two leaves orphan OOF predictions that would silently duplicate on resume.
    done = set()
    if os.path.exists(raw_csv):
        d = pd.read_csv(raw_csv)
        done = set(zip(d.arm, d.model, d.seed, d.fold))
    if os.path.exists(oof_csv):
        oof = pd.read_csv(oof_csv, dtype={"pid": str})
        key = list(zip(oof.arm, oof.model, oof.seed, oof.fold))
        orphans = set(key) - done
        if orphans:
            oof[[k not in orphans for k in key]].to_csv(oof_csv, index=False)
            print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s)")
    return done


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def fit_config(F_tr, F_te, ytr, n_te, seed, fold):
    """Identical estimator and per-fold RF seed as the ablation, E1 and E4, so contrasts
    differ by features and normalization rather than by estimator noise."""
    preds = np.zeros((n_te, 3), "float32")
    base = _rs(seed, fold)
    for i in range(3):
        est = make_estimator((base + i) % (2 ** 31))
        est.fit(F_tr, ytr[:, i])
        preds[:, i] = est.predict(F_te)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    args = ap.parse_args()
    n_seeds = 1 if args.smoke else args.seeds
    sfx = "_smoke" if args.smoke else ""
    raw_csv = os.path.join(HERE, f"e5_results_raw{sfx}.csv")
    oof_csv = os.path.join(HERE, f"e5_oof{sfx}.csv")
    manifest = os.path.join(HERE, f"e5_manifest{sfx}.json")

    t0 = time.time()
    b, h = DataBundle("binned"), DataBundle("highres")
    assert_aligned(b, h)
    Sclean = clean_static(b.S)

    man = _manifest(_data_hash(b.X, b.y, Sclean, b.pid), n_seeds)
    done = _guard_and_resume(man, raw_csv, oof_csv, manifest)
    splits = make_repeated_splits(b.pid, b.diag, n_seeds=n_seeds, n_folds=N_FOLDS,
                                  base_seed=BASE_SEED)
    total = len(splits) * len(ARMS) * len(E5_CONFIGS)
    print(f"E5 | CLEAN={CLEAN} seeds={n_seeds} folds={N_FOLDS} -> {len(splits)} splits "
          f"x {len(ARMS)} arms x {len(E5_CONFIGS)} configs = {total} cells | done: {len(done)}")
    print(f"    activity channels {DYN_CHANNELS} vary; CGM (channel 0) identical in all arms")
    for a in ARMS:
        p = PROPERTIES[a]
        print(f"    {a:15s} level_preserved={str(p['level_preserved']):5s} "
              f"leakage={str(p['leakage']):5s}  {p['label']}")

    worst_cgm = 0.0
    lvl_rows = []
    for si, sp in enumerate(splits):
        tr_idx = np.concatenate([sp["train"], sp["val"]])
        te_idx = sp["test"]

        prepped = {}
        for arm in ARMS:
            Xtr, Str, Xte, Ste = normalize_split_dyn(b.X, Sclean, b.pid,
                                                     tr_idx, te_idx, arm)
            prepped[arm] = (Xtr, Str, Xte, Ste)

        # The control that makes E5 an experiment: CGM must be untouched in every arm.
        for arm in ARMS:
            if arm == "global":
                continue
            worst_cgm = max(worst_cgm,
                            assert_cgm_untouched(prepped[arm][0], prepped["global"][0],
                                                 f"split {si} train / {arm}"),
                            assert_cgm_untouched(prepped[arm][2], prepped["global"][2],
                                                 f"split {si} test / {arm}"))

        for arm in ARMS:                      # activity level diagnostic, model-free
            for c in DYN_CHANNELS:
                lvl_rows.append({"seed": sp["seed"], "fold": sp["fold"], "arm": arm,
                                 "channel": c,
                                 "level_retained": between_subject_level_retained_dyn(
                                     prepped[arm][2], b.pid[te_idx], c)})

        shown = {}
        for arm in ARMS:
            Xtr, Str, Xte, Ste = prepped[arm]
            for name in E5_CONFIGS:
                key = (arm, name, sp["seed"], sp["fold"])
                if key in done:
                    continue
                F_tr = _feats(Xtr, Str, GROUPS[name])
                F_te = _feats(Xte, Ste, GROUPS[name])
                preds = fit_config(F_tr, F_te, b.y[tr_idx], len(te_idx),
                                   sp["seed"], sp["fold"])

                _append(oof_csv, pd.DataFrame({
                    "arm": arm, "model": name, "seed": sp["seed"], "fold": sp["fold"],
                    "sample_idx": te_idx, "pid": b.pid[te_idx], "diag": b.diag[te_idx],
                    **{f"y_true_{hh}": b.y[te_idx][:, i] for i, hh in enumerate(HORIZONS)},
                    **{f"y_pred_{hh}": preds[:, i] for i, hh in enumerate(HORIZONS)}}))
                m = compute_metrics_per_horizon(b.y[te_idx], preds)
                m.update({"arm": arm, "model": name, "seed": sp["seed"],
                          "fold": sp["fold"], "n_features": F_tr.shape[1]})
                _append(raw_csv, pd.DataFrame([m]))
                shown[f"{name}@{arm}"] = m["R2_60"]
                done.add(key)

        if shown:
            keys = [f"{c}@{a}" for a in ARMS for c in ("dyn_static", "full")]
            trace = "  ".join(f"{k}={shown[k]:+.3f}" for k in keys if k in shown)
            print(f"  [{si+1}/{len(splits)}] cgm_delta={worst_cgm:.1e}  {trace}")

    pd.DataFrame(lvl_rows).to_csv(os.path.join(HERE, f"e5_level_diag{sfx}.csv"), index=False)
    print(f"\nDone. {len(done)} cells in {(time.time() - t0) / 60:.1f} min.")
    print(f"CGM channel identical across arms on every fold: max|diff| = {worst_cgm:.3e}")
    print(f"Wrote {raw_csv}, {oof_csv}, {manifest}, e5_level_diag{sfx}.csv")
    print("Next: python e5_analyze.py" + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()
