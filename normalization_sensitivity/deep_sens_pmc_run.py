"""
deep_sens_pmc_run.py -- E2b: the deep `premeal_center` arm, added WITHOUT touching E2
=====================================================================================
Written 2 August 2026, after an external review (REVISION_PLAN §11 item 1) found that E2
ran only two arms -- `ARMS = {"global", "subject_z"}` -- so the leakage-free MECHANISM arm
was never fitted in the deep model. C1's attenuation half transfers across estimators;
C1's mechanism half ("leakage is not necessary") is still Random-Forest-only. This run
closes exactly that gap and nothing else.

    contrast:  [cgm_static - static_all]_global  -  [cgm_static - static_all]_premeal_center

DO NOT ADD THE ARM TO deep_sens_run.py -- IT WOULD DESTROY E2
--------------------------------------------------------------
Appending "premeal_center" to that file's ARMS dict looks like the obvious one-line change.
It is not, for two independent reasons:

 1. `_guard_and_resume` compares the stored manifest's `arms` key and aborts on mismatch.
    That guard is correct and is doing its job.

 2. If the guard were bypassed, `_reconcile_shared` computes
        partial |= {k for k, arms in cells.items() if arms != set(ARMS)}
    Every existing SHARED cell holds arms {global, subject_z}, which no longer equals
    set(ARMS). So all 150 shared cells (75 splits x 2 configs) would be dropped and
    REFITTED. Deep fits are not reproducible on GPU -- cuDNN kernels are nondeterministic
    -- so the refitted `static_all` would differ from the one behind the E2 numbers already
    written into REVISION_PLAN §2, T3 and the forest plot. The published-in-plan E2 result
    would silently change underneath us.

    That routine is not buggy. It is protecting a two-arm invariant, and a third arm is
    simply outside its contract.

So E2's files are READ-ONLY here. This script writes its own.

SHARED CELLS ARE ALIASED, NOT REFITTED
---------------------------------------
`static_all` and `persistence_5min` use no CGM, and `normalize_split` applies IDENTICAL
static handling in every arm (e1_normalizers: FoldNormalizer fitted on training
participants only). E2 already relies on this to write one `static_all` fit under both its
arm labels. The same argument extends to a third arm, so this run COPIES E2's shared
predictions under the `premeal_center` label instead of refitting them.

That is not a shortcut -- it is the stronger choice. Refitting would inject optimization
noise into the context baseline and make the interaction noisier than it needs to be. The
copy makes the CGM-free control exactly invariant across all three arms.

The claim is still checked rather than assumed: `assert_statics_identical` compares the
actual normalized static matrices of `global` and `premeal_center` on every split and
aborts the run if they differ by so much as one ULP.

WORKLOAD
--------
    per (seed, fold):  premeal_center.cgm, premeal_center.cgm_static  = 2 fits
                       (static_all and persistence_5min are aliased, 0 fits)
    x 15 seeds x 5 folds = 150 fits

E2 did 375 fits in 241.6 min, so expect ~1.6 h on the same 3090.

Run (pod, GPU):
    SMOKE=1 python deep_sens_pmc_run.py      # 1 seed, 5 epochs, separate _smoke files
    python deep_sens_pmc_run.py              # 15 seeds, confirmatory
    python deep_sens_pmc_analyze.py
=====================================================================================
"""
import os
import sys
import json
import time
import argparse
import pandas as pd
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
for _p in (CORE, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Everything that defines the experiment is imported from the E2 runner, not restated:
# the normalization, the statics control, the fit, the row writer, the seeds, the budget.
# If E2 changes, this run changes with it -- or the code-hash guard below refuses to run.
from deep_sens_run import (prepare, assert_statics_identical, fit_one, write_cell,
                           _append, _sha, _data_hash, _env,
                           REFERENCE_ARM, FITTED, SHARED,
                           N_SEEDS_DEFAULT, FIT_CODE)
from e1_normalizers import PROPERTIES
from lightdl_run import EPOCHS, PATIENCE, BATCH, SMOKE
from ablation_run import clean_static, BASE_SEED, N_FOLDS, CLEAN
from lightdl_data import DataBundle, assert_aligned, make_repeated_splits
from deep_sens_models import build_midfusion, BRANCHES

NEW_ARM = "premeal_center"
CGM_CONFIGS = [c for c in FITTED if c not in SHARED]        # ["cgm", "cgm_static"]

# arm label -> e1_normalizers protocol. Both happen to be identity here, but E2 keeps the
# mapping explicit and so does this: an arm label is a column value, a protocol is a
# transform, and silently conflating them is how the wrong transform gets fitted.
ARM_PROTOCOL = {REFERENCE_ARM: "global", NEW_ARM: "premeal_center"}


def _load_parent(sfx):
    """E2's outputs, read-only. Returns (raw, oof, manifest)."""
    paths = {k: os.path.join(HERE, f"deep_sens_{k}{sfx}.{'json' if k == 'manifest' else 'csv'}")
             for k in ("raw", "oof", "manifest")}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f"[abort] parent E2 file missing: {p}\n"
                             f"This run extends E2; it cannot be the first thing you run.")
    return (pd.read_csv(paths["raw"]),
            pd.read_csv(paths["oof"], dtype={"pid": str}),
            json.load(open(paths["manifest"])),
            paths)


def _assert_comparable(parent_man, data_hash):
    """The new arm is only comparable to E2 if the DATA and the FITTING CODE are the ones
    E2 used. A changed transform, architecture or split rule would make the contrast
    between arms a comparison of two different experiments."""
    if parent_man.get("data_hash") != data_hash:
        raise SystemExit(f"[abort] data hash differs from E2's:\n"
                         f"  E2:      {parent_man.get('data_hash')}\n"
                         f"  current: {data_hash}\n"
                         f"The new arm would be fitted on different data.")
    now = {f: _sha(f) for f in FIT_CODE}
    old = parent_man.get("code_hash", {})
    drift = {f: (old.get(f), now[f]) for f in FIT_CODE if old.get(f) != now[f]}
    if drift:
        lines = "\n".join(f"    {f}: E2={a} current={b}" for f, (a, b) in drift.items())
        raise SystemExit(f"[abort] fitting code changed since E2:\n{lines}\n"
                         f"Restore the E2 versions, or version all three arms together.")
    if set(parent_man.get("arms", {})) != {"global", "subject_z"}:
        raise SystemExit(f"[abort] parent manifest arms are "
                         f"{sorted(parent_man.get('arms', {}))}, expected "
                         f"['global', 'subject_z']. This script extends the two-arm E2.")
    return True


def _alias_shared(parent_raw, parent_oof, cfg, seed, fold, raw_csv, oof_csv, done):
    """Copy E2's shared CGM-free cell under the new arm label. No fit."""
    key = (parent_oof.arm == REFERENCE_ARM) & (parent_oof.model == cfg) \
        & (parent_oof.seed == seed) & (parent_oof.fold == fold)
    o = parent_oof[key]
    if o.empty:
        raise SystemExit(f"[abort] E2 has no {cfg} cell for seed={seed} fold={fold} under "
                         f"arm {REFERENCE_ARM!r}. Cannot alias a cell that does not exist.")
    rkey = (parent_raw.arm == REFERENCE_ARM) & (parent_raw.model == cfg) \
        & (parent_raw.seed == seed) & (parent_raw.fold == fold)
    r = parent_raw[rkey]
    if len(r) != 1:
        raise SystemExit(f"[abort] expected exactly 1 E2 raw row for "
                         f"{cfg}/seed={seed}/fold={fold}, found {len(r)}.")
    o = o.copy(); o["arm"] = NEW_ARM
    r = r.copy(); r["arm"] = NEW_ARM; r["shared_fit"] = True
    _append(oof_csv, o)
    _append(raw_csv, r)
    done.add((NEW_ARM, cfg, seed, fold))
    return float(r.iloc[0]["R2_60"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    args = ap.parse_args()
    n_seeds = 1 if SMOKE else args.seeds
    sfx = "_smoke" if SMOKE else ""
    raw_csv = os.path.join(HERE, f"deep_sens_pmc_raw{sfx}.csv")
    oof_csv = os.path.join(HERE, f"deep_sens_pmc_oof{sfx}.csv")
    man_path = os.path.join(HERE, f"deep_sens_pmc_manifest{sfx}.json")

    t0 = time.time()
    b, hb = DataBundle("binned"), DataBundle("highres")
    assert_aligned(b, hb)
    Sclean = clean_static(b.S)
    data_hash = _data_hash(b.X, b.y, Sclean, b.pid)

    parent_raw, parent_oof, parent_man, parent_paths = _load_parent(sfx)
    _assert_comparable(parent_man, data_hash)

    params = {c: int(build_midfusion(b.T, b.n_static, **BRANCHES[c]).count_params())
              for c in FITTED}
    tf.keras.backend.clear_session()

    man = {"experiment": "E2b_deep_premeal_center_arm",
           "extends": {"experiment": parent_man.get("experiment"),
                       "files": {k: os.path.basename(v) for k, v in parent_paths.items()},
                       "data_hash": parent_man.get("data_hash"),
                       "code_hash": parent_man.get("code_hash")},
           "data_hash": data_hash, "new_arm": NEW_ARM,
           "arm_properties": {NEW_ARM: PROPERTIES[NEW_ARM]},
           "reference_arm": REFERENCE_ARM,
           "fitted_configs": CGM_CONFIGS,
           "aliased_configs": SHARED,
           "aliasing_note": "static_all and persistence_5min are COPIED from E2's "
                            "reference-arm cells, not refitted -- statics are identical "
                            "across arms by construction and the copy keeps the CGM-free "
                            "control exactly invariant across all three arms.",
           "n_seeds": n_seeds, "n_folds": N_FOLDS, "base_seed": BASE_SEED,
           "epochs": EPOCHS, "patience": PATIENCE, "batch": BATCH,
           "clean": CLEAN, "n_params": params,
           "code_hash": {f: _sha(f) for f in FIT_CODE}, "env": _env()}

    if os.path.exists(man_path):
        old = json.load(open(man_path))
        for k in ("data_hash", "new_arm", "fitted_configs", "aliased_configs",
                  "n_folds", "base_seed", "epochs", "patience", "batch", "code_hash"):
            if old.get(k) != man.get(k):
                raise SystemExit(f"[abort] manifest mismatch on {k!r}:\n  stored:  "
                                 f"{old.get(k)}\n  current: {man.get(k)}\n"
                                 f"Delete deep_sens_pmc_* to start clean.")
    else:
        json.dump(man, open(man_path, "w"), indent=2)

    done = set()
    if os.path.exists(raw_csv):
        d = pd.read_csv(raw_csv)
        done = set(zip(d.arm, d.model, d.seed, d.fold))
        if os.path.exists(oof_csv):
            oof = pd.read_csv(oof_csv, dtype={"pid": str})
            ck = list(zip(oof.arm, oof.model, oof.seed, oof.fold))
            orphans = set(ck) - done
            if orphans:
                oof[[k not in orphans for k in ck]].to_csv(oof_csv, index=False)
                print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s)")

    splits = make_repeated_splits(b.pid, b.diag, n_seeds=n_seeds, n_folds=N_FOLDS,
                                  base_seed=BASE_SEED)
    q = PROPERTIES[NEW_ARM]
    print(f"E2b | SMOKE={SMOKE} seeds={n_seeds} folds={N_FOLDS} epochs={EPOCHS} "
          f"patience={PATIENCE} batch={BATCH}")
    print(f"   new arm: {NEW_ARM}  level_preserved={q['level_preserved']} "
          f"leakage={q['leakage']}  {q['label']}")
    print(f"   {len(splits)} splits x {len(CGM_CONFIGS)} fits = "
          f"{len(splits) * len(CGM_CONFIGS)} fits | done cells: {len(done)}")
    print(f"   aliased from E2 (0 fits): {SHARED}")

    for si, sp in enumerate(splits):
        te_idx, pid, diag = sp["test"], b.pid[sp["test"]], b.diag[sp["test"]]
        yte, ytr, yva = b.y[sp["test"]], b.y[sp["train"]], b.y[sp["val"]]

        preps = {a: prepare(b.X, Sclean, b.pid, sp["train"], sp["val"], sp["test"], p)
                 for a, p in ARM_PROTOCOL.items()}
        # The real control: the arms must differ ONLY in the CGM transform.
        assert_statics_identical(preps[REFERENCE_ARM], preps[NEW_ARM])

        shown = {}
        for cfg in SHARED:
            if (NEW_ARM, cfg, sp["seed"], sp["fold"]) in done:
                continue
            shown[f"{cfg}(alias)"] = _alias_shared(
                parent_raw, parent_oof, cfg, sp["seed"], sp["fold"],
                raw_csv, oof_csv, done)

        for cfg in CGM_CONFIGS:
            if (NEW_ARM, cfg, sp["seed"], sp["fold"]) in done:
                continue
            preds, npar = fit_one(cfg, preps[NEW_ARM], ytr, yva, sp)
            m = write_cell(oof_csv, raw_csv, NEW_ARM, cfg, sp, te_idx, pid, diag,
                           yte, preds, npar, False, done)
            shown[f"{cfg}@{NEW_ARM}"] = m["R2_60"]

        if shown:
            trace = "  ".join(f"{k}={v:+.3f}" for k, v in shown.items())
            print(f"  [{si+1}/{len(splits)}] {trace}", flush=True)

    print(f"\nDone. {len(done)} cells in {(time.time() - t0) / 60:.1f} min.")
    print(f"Wrote {raw_csv}, {oof_csv}, {man_path}")
    print("E2's files were not modified.")
    print("Next: python deep_sens_pmc_analyze.py" + (" --smoke" if SMOKE else ""))


if __name__ == "__main__":
    main()
