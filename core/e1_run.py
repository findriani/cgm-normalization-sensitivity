"""
e1_run.py -- the controlled normalization experiment (the paper's principal result)
==================================================================================
Runs the modality ablation FIVE times, identical in every respect except the temporal
normalization protocol (see e1_normalizers.py for the design and what each arm isolates).

    python e1_run.py --smoke     # 1 seed, prints and writes to *_smoke.csv
    python e1_run.py             # 8 seeds, the confirmatory run

CPU only, no TensorFlow. The modality ablation is Random-Forest based, so the whole thing
is a local job of roughly half an hour rather than a GPU booking.

SCOPE -- WHAT `subject_z` IS AND IS NOT
---------------------------------------
`subject_z` reproduces the PUBLISHED CGM TRANSFORM. It is NOT a reproduction of the
published experiment, and the distinction must survive into the manuscript:

  * the published modality result used the deep simple-mid-fusion model (BiLSTM-32 CGM
    branch, GRU-16 dynamic branch, MLP static branch); E1 uses the Random Forest that the
    modality ablation uses;
  * E1 deliberately keeps STATIC preprocessing leakage-free in every arm, whereas the
    published pipeline normalized statics globally over the whole dataset.

Both departures are intentional -- holding everything except the CGM transform fixed is what
makes this a controlled experiment rather than a before/after. But it means C1 is
established *with the RF estimator*. The mechanism (removal of between-participant level)
is estimator-independent in principle, and the variance diagnostic below shows it directly,
but a deep-model sensitivity arm (`global` vs `subject_z` under the original simple
mid-fusion) is the clean way to close that gap and is NOT built here -- it needs TensorFlow.
Either add it, or state the scope explicitly in the text. Do not write "reproduces the
original experiment".

WHAT IS HELD FIXED ACROSS ARMS
------------------------------
Everything that could otherwise explain a difference:
  * the same participant-level, diagnosis-stratified splits (base_seed=42, 8 seeds, 5 folds)
  * the same Random Forest, the same tree count, and the SAME per-fold RF seed --
    `_rs(seed, fold)` is imported from ablation_run, so arm `global` is bit-comparable to
    the existing `ablation_*_rf_capped` results
  * the same static-feature normalization (fold-wise global, train-only) in every arm
  * the same meal-value capping (CLEAN=capped, the ablation's primary setting)
  * the same four configurations

CONFIGURATIONS -- deliberately only four
----------------------------------------
The claim needs exactly two increments and a floor:
    cgm_static - static_all   =  what CGM adds beyond context
    cgm_static - cgm          =  what context adds beyond CGM
Running the full 13-configuration ablation in every arm would quintuple the runtime to
answer questions this experiment is not asking.

WHAT IS RECORDED BEYOND PREDICTIONS
-----------------------------------
Each cell also records `level_retained`: the share of pre-meal CGM variance that is
between-participant after normalization. This is the mechanism the argument rests on -- it
should be ~0 exactly in the arms that centre per subject. Recording it turns "we believe
centering removes the baseline" into something a reader can check in the results file.
==================================================================================
"""
import os
import json
import time
import hashlib
import argparse
import platform
import numpy as np
import pandas as pd

# ablation_run imports only sklearn/numpy/pandas -- no TensorFlow -- so reusing its exact
# estimator, seeding and feature-slicing logic costs nothing and guarantees that arm
# `global` reproduces the published reference ablation rather than merely resembling it.
from ablation_run import (CONFIGS as ABL_CONFIGS, clean_static, make_estimator, _rs,
                          _feats, MEAL_CAPS, RF_TREES, BASE_SEED, N_FOLDS, CLEAN)
from lightdl_data import DataBundle, assert_aligned, highres_baseline, make_repeated_splits
from common import compute_metrics_per_horizon, HORIZONS
from e1_normalizers import (PROTOCOLS, PROPERTIES, normalize_split,
                            between_subject_level_retained)

HERE = os.path.dirname(os.path.abspath(__file__))

# Only the configurations the claim needs. Names and feature groups are taken from
# ablation_run.CONFIGS so they cannot drift from the primary study.
E1_CONFIGS = ["persistence_5min", "cgm", "static_all", "cgm_static"]

FIT_CODE = ("e1_run.py", "e1_normalizers.py", "ablation_run.py", "lightdl_data.py",
            "foldwise_normalizer.py", "common.py")


def _sha(p):
    with open(os.path.join(HERE, p), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _data_hash(X, y, S, pid):
    h = hashlib.md5()
    for a in (X, y, S):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(pid).encode())
    return h.hexdigest()[:16]


def _manifest(dh, n_seeds):
    return {"data_hash": dh, "protocols": list(PROTOCOLS), "properties": PROPERTIES,
            "configs": E1_CONFIGS, "n_seeds": n_seeds, "n_folds": N_FOLDS,
            "base_seed": BASE_SEED, "rf_trees": RF_TREES, "clean": CLEAN,
            "meal_caps": MEAL_CAPS,
            "code_hash": {f: _sha(f) for f in FIT_CODE},
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": __import__("sklearn").__version__}


def _guard_and_resume(man, raw_csv, oof_csv, manifest_path):
    if os.path.exists(raw_csv) and not os.path.exists(manifest_path):
        raise SystemExit(f"[abort] {raw_csv} exists with no {manifest_path}. Move it aside.")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        for k in ("data_hash", "protocols", "configs", "n_folds", "base_seed",
                  "rf_trees", "clean", "code_hash"):
            if old.get(k) != man.get(k):
                raise SystemExit(f"[abort] manifest mismatch on {k!r}:\n  stored:  "
                                 f"{old.get(k)}\n  current: {man.get(k)}\n"
                                 f"Delete the e1_* result files to start clean.")
    else:
        json.dump(man, open(manifest_path, "w"), indent=2)
    # Completion is marked by the RAW metric row, which is written AFTER the OOF rows. A
    # crash between the two writes leaves orphan OOF predictions for a cell that will be
    # re-run on resume, silently duplicating rows and corrupting build_ref. Reconcile the
    # same way ablation_run.py does: drop any OOF cell with no completion marker.
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
            print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s) "
                  f"from an interrupted write")
    return done


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def run_config(name, groups, prep, sp):
    """Identical to ablation_run.run_config, but reading this arm's feature substrate."""
    if groups is None:
        return np.repeat(prep["base5"].astype("float32")[:, None], 3, axis=1)
    Ftr = _feats(prep["Xtr"], prep["Str"], groups)
    Fte = _feats(prep["Xte"], prep["Ste"], groups)
    preds = np.zeros((len(prep["yte"]), 3), "float32")
    base = _rs(sp["seed"], sp["fold"])            # SAME seed as the primary ablation
    for i in range(3):
        est = make_estimator((base + i) % (2 ** 31))
        est.fit(Ftr, prep["ytr"][:, i])
        preds[:, i] = est.predict(Fte)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 seed, separate output files")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    n_seeds = 1 if args.smoke else args.seeds
    sfx = "_smoke" if args.smoke else ""
    raw_csv = os.path.join(HERE, f"e1_results_raw{sfx}.csv")
    oof_csv = os.path.join(HERE, f"e1_oof{sfx}.csv")
    manifest = os.path.join(HERE, f"e1_manifest{sfx}.json")

    t0 = time.time()
    b, h = DataBundle("binned"), DataBundle("highres")
    assert_aligned(b, h)
    base1 = highres_baseline()
    Sclean = clean_static(b.S)

    unknown = [c for c in E1_CONFIGS if c not in ABL_CONFIGS]
    if unknown:
        raise SystemExit(f"[abort] configs not in ablation_run.CONFIGS: {unknown}")

    man = _manifest(_data_hash(b.X, b.y, Sclean, b.pid), n_seeds)
    done = _guard_and_resume(man, raw_csv, oof_csv, manifest)
    splits = make_repeated_splits(b.pid, b.diag, n_seeds=n_seeds, n_folds=N_FOLDS,
                                  base_seed=BASE_SEED)
    total = len(splits) * len(PROTOCOLS) * len(E1_CONFIGS)
    print(f"E1 | CLEAN={CLEAN} seeds={n_seeds} folds={N_FOLDS} -> {len(splits)} splits "
          f"x {len(PROTOCOLS)} protocols x {len(E1_CONFIGS)} configs = {total} cells "
          f"| done: {len(done)}")
    for p in PROTOCOLS:
        q = PROPERTIES[p]
        print(f"    {p:16s} level_preserved={str(q['level_preserved']):5s} "
              f"leakage={str(q['leakage']):5s}  {q['label']}")

    for si, sp in enumerate(splits):
        tr_idx = np.concatenate([sp["train"], sp["val"]])   # RF needs no validation split
        te_idx = sp["test"]
        for proto in PROTOCOLS:
            if all((proto, c, sp["seed"], sp["fold"]) in done for c in E1_CONFIGS):
                continue
            Xtr, Str, Xte, Ste = normalize_split(b.X, Sclean, b.pid, tr_idx, te_idx, proto)
            lvl_te = between_subject_level_retained(Xte, b.pid[te_idx])
            prep = {"Xtr": Xtr, "Str": Str, "ytr": b.y[tr_idx],
                    "Xte": Xte, "Ste": Ste, "yte": b.y[te_idx],
                    "idx": te_idx, "pid": b.pid[te_idx], "diag": b.diag[te_idx],
                    "base5": b.X[te_idx, -1, 0]}
            shown = {}
            for name in E1_CONFIGS:
                key = (proto, name, sp["seed"], sp["fold"])
                if key in done:
                    continue
                preds = run_config(name, ABL_CONFIGS[name], prep, sp)
                _append(oof_csv, pd.DataFrame({
                    "protocol": proto, "model": name, "seed": sp["seed"], "fold": sp["fold"],
                    "sample_idx": prep["idx"], "pid": prep["pid"], "diag": prep["diag"],
                    **{f"y_true_{hh}": prep["yte"][:, i] for i, hh in enumerate(HORIZONS)},
                    **{f"y_pred_{hh}": preds[:, i] for i, hh in enumerate(HORIZONS)}}))
                m = compute_metrics_per_horizon(prep["yte"], preds)
                m.update({"protocol": proto, "model": name, "seed": sp["seed"],
                          "fold": sp["fold"], "level_retained": lvl_te,
                          "level_preserved": PROPERTIES[proto]["level_preserved"],
                          "leakage": PROPERTIES[proto]["leakage"]})
                _append(raw_csv, pd.DataFrame([m]))
                shown[name] = m["R2_60"]
                done.add(key)
            trace = "  ".join(f"{c}={shown[c]:+.3f}" for c in
                              ("cgm", "static_all", "cgm_static") if c in shown)
            print(f"  [{si+1}/{len(splits)}] {proto:16s} level={lvl_te:.3f}  {trace}")

    print(f"\nDone. {len(done)} cells in {(time.time() - t0) / 60:.1f} min.")
    print(f"Wrote {raw_csv}, {oof_csv}, {manifest}")
    print("Next: python e1_analyze.py" + (" --smoke" if args.smoke else ""))


if __name__ == "__main__":
    main()
