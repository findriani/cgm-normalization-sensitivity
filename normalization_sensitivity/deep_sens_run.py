"""
deep_sens_run.py -- E2: does the normalization effect hold with the DEEP model?
==============================================================================
E1 established that per-subject CGM z-scoring attenuates modality attribution. It did so
with a RANDOM FOREST. The published result being disputed came from the DEEP simple
mid-fusion model. Until the same contrast is run with the deep model, a reviewer can
reasonably ask whether the problem is real or an artifact of the simpler estimator.

E2 answers exactly that and nothing else.

TWO ARMS, ONE DIFFERENCE
------------------------
    A1  `global`      fold-wise global CGM normalization, fit on training participants
    A0  `subject_z`   per-subject z-score, i.e. the PUBLISHED CGM TRANSFORM

Same splits, same seeds, same architecture, same leakage-free static handling, same
targets in mg/dL. The CGM transform is the only thing that varies -- which is what makes
this a clean single-factor test rather than the confounded comparison an earlier draft
proposed (that one changed normalization AND architecture at once).

The transforms are imported from `e1_normalizers`, so the CGM inputs here are bit-for-bit
the same transform E1 used.

A0 IS AN INTENTIONALLY LEAKY DIAGNOSTIC COMPARATOR
---------------------------------------------------
Its per-subject statistics are computed over each participant's COMPLETE record, so for a
held-out participant they use that participant's own data. This is deliberate: A0 exists
to reproduce the published transform, not to be a legitimate pipeline. Say so plainly in
the manuscript -- a reader who notices the leakage without being told will assume it was
an oversight.

A0 IS ALSO NOT A REPRODUCTION OF THE PUBLISHED EXPERIMENT. It is the published CGM
transform run inside the leakage-free pipeline: leakage-free statics, reference splits,
reference targets, and (see deep_sens_models) no dynamic branch.

SHARED CGM-FREE FITS -- READ BEFORE INTERPRETING THE INVARIANCE CHECK
----------------------------------------------------------------------
`static_all` and `persistence_5min` use no CGM at all, and static preprocessing is
identical in both arms. Fitting `static_all` separately per arm would inject optimization
noise (initialization, dropout, batch order, nondeterministic cuDNN kernels) into a
configuration that is supposed to be identical.

So `static_all` is FIT ONCE per (seed, fold) and its out-of-fold predictions are written
under BOTH arm labels; `persistence_5min` is a lookup on raw pre-meal CGM and is shared
the same way. This makes the control exactly invariant instead of
invariant-within-tolerance, and cuts the workload from 450 fits to 375.

CONSEQUENCE, WHICH MUST BE STATED: the CGM-free invariance check downstream is then an
ASSERTION THAT THE SHARING WORKED, not independent evidence that the arms are otherwise
identical. Every shared row carries `shared_fit=True` so this cannot be forgotten. The
check that the arms are otherwise identical is the static-matrix assertion below, which
compares the actual normalized static features between protocols.

WORKLOAD
--------
    per (seed, fold):  A0.cgm, A0.cgm_static, A1.cgm, A1.cgm_static, shared static_all
                       = 5 fits   (persistence is a lookup)
    x 15 seeds x 5 folds = 375 fits

Run (pod, GPU):
    SMOKE=1 python deep_sens_run.py          # 1 seed, 5 epochs, separate output files
    python deep_sens_run.py                  # 15 seeds, confirmatory
    python deep_sens_analyze.py

Commit to 15 seeds up front. Do NOT stop at 8 because the interim looks good -- optional
stopping on a favourable result would undercut a paper whose contribution is
methodological rigour.
==============================================================================
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
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
for _p in (CORE, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Imported, not reimplemented, so nothing can drift from the published deep pipeline or
# from E1: the CGM transforms, the early-stopping rule, the epoch/patience/batch budget,
# the RF-paired per-fold seed, and the meal-outlier capping.
from e1_normalizers import normalize_split, PROPERTIES
from lightdl_run import MainTrajectoryEarlyStopping, EPOCHS, PATIENCE, BATCH, SMOKE
from ablation_run import clean_static, _rs, MEAL_CAPS, BASE_SEED, N_FOLDS, CLEAN
from lightdl_data import DataBundle, assert_aligned, make_repeated_splits
from common import compute_metrics_per_horizon, HORIZONS
from deep_sens_models import build_midfusion, BRANCHES, model_inputs

# arm label -> e1_normalizers protocol
ARMS = {"global": "global", "subject_z": "subject_z"}
REFERENCE_ARM, PUBLISHED_ARM = "global", "subject_z"

FITTED = ["cgm", "static_all", "cgm_static"]
SHARED = ["static_all", "persistence_5min"]     # no CGM -> fit once, write for both arms
ALL_CONFIGS = ["persistence_5min"] + FITTED

N_SEEDS_DEFAULT = 15                            # matches the attention and window studies
STATIC_MATCH_TOL = 0.0                          # statics must be EXACTLY equal across arms

FIT_CODE = ("deep_sens_run.py", "deep_sens_models.py", "e1_normalizers.py",
            "lightdl_models.py", "lightdl_data.py", "foldwise_normalizer.py",
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


def _env():
    try:
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        gpus = []
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "tensorflow": tf.__version__,
            "gpus": gpus, "host": platform.node()}


def _manifest(dh, n_seeds, params):
    return {"experiment": "E2_deep_normalization_sensitivity", "data_hash": dh,
            "arms": ARMS, "arm_properties": {a: PROPERTIES[p] for a, p in ARMS.items()},
            "configs": ALL_CONFIGS, "shared_configs": SHARED,
            "backbone": "published simple mid-fusion, dynamic branch omitted "
                        "(no E2 configuration uses the HR/activity channels)",
            "normalizer_fit_on": "train participants only (deep convention; E1 used "
                                 "train+val because the RF has no validation split)",
            "n_seeds": n_seeds, "n_folds": N_FOLDS, "base_seed": BASE_SEED,
            "epochs": EPOCHS, "patience": PATIENCE, "batch": BATCH,
            "clean": CLEAN, "meal_caps": MEAL_CAPS, "n_params": params,
            "code_hash": {f: _sha(f) for f in FIT_CODE}, "env": _env()}


def _guard_and_resume(man, raw_csv, oof_csv, manifest_path):
    if os.path.exists(raw_csv) and not os.path.exists(manifest_path):
        raise SystemExit(f"[abort] {raw_csv} exists with no {manifest_path}. Move it aside.")
    if os.path.exists(manifest_path):
        old = json.load(open(manifest_path))
        # `env` is excluded deliberately: a pod restart may change driver/host without
        # invalidating results. Everything that defines the EXPERIMENT is compared.
        for k in ("data_hash", "arms", "configs", "shared_configs", "n_folds",
                  "base_seed", "epochs", "patience", "batch", "clean", "code_hash"):
            if old.get(k) != man.get(k):
                raise SystemExit(f"[abort] manifest mismatch on {k!r}:\n  stored:  "
                                 f"{old.get(k)}\n  current: {man.get(k)}\n"
                                 f"Delete the deep_sens_* result files to start clean.")
    else:
        json.dump(man, open(manifest_path, "w"), indent=2)
    done = set()
    if os.path.exists(raw_csv):
        d = pd.read_csv(raw_csv)
        done = set(zip(d.arm, d.model, d.seed, d.fold))
    if os.path.exists(oof_csv):
        oof = pd.read_csv(oof_csv, dtype={"pid": str})
        cellkey = list(zip(oof.arm, oof.model, oof.seed, oof.fold))
        orphans = set(cellkey) - done
        if orphans:
            keep = [k not in orphans for k in cellkey]
            oof[keep].to_csv(oof_csv, index=False)
            print(f"reconciled: dropped {len(orphans)} orphan OOF cell(s) "
                  f"from an interrupted write")
    return _reconcile_shared(raw_csv, oof_csv, done)


def _reconcile_shared(raw_csv, oof_csv, done):
    """Shared CGM-free cells are TWO alias rows written from ONE fit. Remove any that is
    not complete for every arm.

    Without this, an interruption between the two alias writes is silently corrupting: on
    resume the cell looks unfinished for the missing arm only, so the model is REFITTED
    and a DIFFERENT prediction is stored under that arm -- deep fits are not reproducible
    on GPU (cuDNN kernels are nondeterministic). The two aliases would then disagree, the
    invariance check would fail, and the CGM-free control -- the thing that makes E2 a
    controlled comparison -- would be broken by a mere restart.

    Dropping and refitting the pair costs at most one fit per interruption. Keeping a
    mismatched pair would cost the experiment."""
    partial = set()
    for cfg in SHARED:
        cells = {}
        for (arm, model, seed, fold) in done:
            if model == cfg:
                cells.setdefault((model, seed, fold), set()).add(arm)
        partial |= {k for k, arms in cells.items() if arms != set(ARMS)}
    if not partial:
        return done

    def _drop(path, is_oof):
        if not os.path.exists(path):
            return
        d = pd.read_csv(path, dtype={"pid": str}) if is_oof else pd.read_csv(path)
        keep = [(m, s, f) not in partial
                for m, s, f in zip(d.model, d.seed, d.fold)]
        d[keep].to_csv(path, index=False)

    _drop(raw_csv, False)
    _drop(oof_csv, True)
    print(f"reconciled: dropped {len(partial)} partially-written SHARED cell(s) "
          f"(one arm alias present, the other missing). They will be refitted as a unit "
          f"so both arms keep identical predictions.")
    return {k for k in done if (k[1], k[2], k[3]) not in partial}


def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ------------------------------------------------------------------ normalization
def prepare(X, S, pid, tr, va, te, protocol):
    """Normalized train/val/test under one CGM protocol.

    `normalize_split` fits on the first index set and transforms both, so it is called
    twice against the SAME training indices -- once to obtain the test split, once for
    the validation split. The train outputs must therefore be identical; that is asserted
    rather than assumed, because a silent disagreement would mean the two calls fitted
    different statistics.

    The normalizer is fitted on TRAIN ONLY, matching the deep pipeline's convention
    (lightdl_data.normalized_fold). E1 fitted on train+val because a Random Forest has no
    validation split. The difference is recorded in the manifest."""
    Xtr, Str, Xte, Ste = normalize_split(X, S, pid, tr, te, protocol)
    Xtr2, Str2, Xva, Sva = normalize_split(X, S, pid, tr, va, protocol)
    if not (np.allclose(Xtr, Xtr2, atol=0, rtol=0)
            and np.allclose(Str, Str2, atol=0, rtol=0)):
        raise SystemExit("[abort] the two normalize_split calls disagree on the training "
                         "split -- the fold statistics are not deterministic.")
    return {"Xtr": Xtr, "Str": Str, "Xva": Xva, "Sva": Sva, "Xte": Xte, "Ste": Ste}


def assert_statics_identical(pa, pb):
    """The arms must differ ONLY in the CGM transform.

    This is the real control. Once `static_all` predictions are shared between arms their
    equality is guaranteed by construction and proves nothing; what still needs checking
    is that the STATIC FEATURE MATRICES the two arms would have used are identical."""
    for k in ("Str", "Sva", "Ste"):
        d = float(np.abs(np.asarray(pa[k]) - np.asarray(pb[k])).max())
        if d > STATIC_MATCH_TOL:
            raise SystemExit(f"[abort] static features differ between arms on {k}: "
                             f"max|diff| = {d:.3e}. Something other than the CGM "
                             f"transform varies -- E2 is not controlled.")
    return True


# ------------------------------------------------------------------ fitting
def fit_one(cfg, prep, ytr, yva, sp):
    """Fit one configuration on one fold; returns (preds mg/dL, n_params).

    Seeding uses `_rs(seed, fold)` from ablation_run -- identical across arms AND
    configurations, so A0 and A1 start from the same initialization and see the same
    batch order for a given cell. Deep fits are still not bitwise reproducible on GPU
    (cuDNN kernels are nondeterministic), but the pairing removes the initialization
    component of the difference between arms."""
    tf.keras.utils.set_random_seed(_rs(sp["seed"], sp["fold"]) % (2 ** 31))
    tf.keras.backend.clear_session()

    Xtr = model_inputs(cfg, prep["Xtr"].astype("float32"), prep["Str"].astype("float32"))
    Xva = model_inputs(cfg, prep["Xva"].astype("float32"), prep["Sva"].astype("float32"))
    Xte = model_inputs(cfg, prep["Xte"].astype("float32"), prep["Ste"].astype("float32"))

    # train-fold target standardization; metrics are inverted back to mg/dL
    mu, sd = ytr.mean(axis=0), ytr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    ytr_s = [((ytr[:, i] - mu[i]) / sd[i]) for i in range(3)]
    yva_s = [((yva[:, i] - mu[i]) / sd[i]) for i in range(3)]

    model = build_midfusion(prep["Xtr"].shape[1], prep["Str"].shape[1], **BRANCHES[cfg])
    n_params = int(model.count_params())
    model.fit(Xtr, ytr_s, validation_data=(Xva, yva_s), epochs=EPOCHS, batch_size=BATCH,
              callbacks=[MainTrajectoryEarlyStopping(PATIENCE)], verbose=0)
    out = model.predict(Xte, verbose=0)
    if not isinstance(out, list):
        out = [out]
    preds = np.stack([np.asarray(out[i]).squeeze(-1) * sd[i] + mu[i] for i in range(3)],
                     axis=-1).astype("float32")
    tf.keras.backend.clear_session()
    return preds, n_params


def write_cell(oof_csv, raw_csv, arm, cfg, sp, te_idx, pid, diag, yte, preds,
               n_params, shared, done):
    _append(oof_csv, pd.DataFrame({
        "arm": arm, "model": cfg, "seed": sp["seed"], "fold": sp["fold"],
        "sample_idx": te_idx, "pid": pid, "diag": diag,
        **{f"y_true_{h}": yte[:, i] for i, h in enumerate(HORIZONS)},
        **{f"y_pred_{h}": preds[:, i] for i, h in enumerate(HORIZONS)}}))
    m = compute_metrics_per_horizon(yte, preds)
    m.update({"arm": arm, "model": cfg, "seed": sp["seed"], "fold": sp["fold"],
              "n_params": n_params, "shared_fit": bool(shared)})
    _append(raw_csv, pd.DataFrame([m]))
    done.add((arm, cfg, sp["seed"], sp["fold"]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    args = ap.parse_args()
    n_seeds = 1 if SMOKE else args.seeds
    sfx = "_smoke" if SMOKE else ""
    raw_csv = os.path.join(HERE, f"deep_sens_raw{sfx}.csv")
    oof_csv = os.path.join(HERE, f"deep_sens_oof{sfx}.csv")
    manifest = os.path.join(HERE, f"deep_sens_manifest{sfx}.json")

    t0 = time.time()
    b, hb = DataBundle("binned"), DataBundle("highres")
    assert_aligned(b, hb)
    Sclean = clean_static(b.S)

    params = {c: int(build_midfusion(b.T, b.n_static, **BRANCHES[c]).count_params())
              for c in FITTED}
    tf.keras.backend.clear_session()

    man = _manifest(_data_hash(b.X, b.y, Sclean, b.pid), n_seeds, params)
    done = _guard_and_resume(man, raw_csv, oof_csv, manifest)
    splits = make_repeated_splits(b.pid, b.diag, n_seeds=n_seeds, n_folds=N_FOLDS,
                                  base_seed=BASE_SEED)
    per_split = len(ARMS) * (len(FITTED) - len(set(SHARED) & set(FITTED))) \
        + len(set(SHARED) & set(FITTED))
    print(f"E2 | SMOKE={SMOKE} seeds={n_seeds} folds={N_FOLDS} epochs={EPOCHS} "
          f"patience={PATIENCE} batch={BATCH}")
    print(f"   {len(splits)} splits x {per_split} fits = {len(splits) * per_split} fits "
          f"| done cells: {len(done)}")
    for a, p in ARMS.items():
        q = PROPERTIES[p]
        print(f"    {a:12s} level_preserved={str(q['level_preserved']):5s} "
              f"leakage={str(q['leakage']):5s}  {q['label']}")
    print(f"    params: " + "  ".join(f"{c}={params[c]}" for c in FITTED))
    print(f"    shared (fit once, written for both arms): {SHARED}")

    for si, sp in enumerate(splits):
        te_idx, pid, diag = sp["test"], b.pid[sp["test"]], b.diag[sp["test"]]
        yte, ytr, yva = b.y[sp["test"]], b.y[sp["train"]], b.y[sp["val"]]

        preps = {a: prepare(b.X, Sclean, b.pid, sp["train"], sp["val"], sp["test"], p)
                 for a, p in ARMS.items()}
        assert_statics_identical(preps[REFERENCE_ARM], preps[PUBLISHED_ARM])

        shown = {}

        # ---- shared, CGM-free: one fit (or lookup), written under both arm labels ----
        for cfg in SHARED:
            present = [a for a in ARMS if (a, cfg, sp["seed"], sp["fold"]) in done]
            if len(present) == len(ARMS):
                continue
            if present:
                # _reconcile_shared removes partial cells at startup, so reaching here
                # means the invariant was violated within this process.
                raise SystemExit(f"[abort] shared cell ({cfg}, seed={sp['seed']}, "
                                 f"fold={sp['fold']}) is present for {present} but not "
                                 f"all arms. Delete the deep_sens_* files and version.")
            if cfg == "persistence_5min":
                base5 = b.X[te_idx, -1, 0]          # raw last pre-meal bin, mg/dL
                preds = np.repeat(base5.astype("float32")[:, None], 3, axis=1)
                npar = 0
            else:
                # Either arm's statics would do -- assert_statics_identical has already
                # established they are equal -- but always fit from the reference arm so
                # the shared cell is reproducible from the manifest alone.
                preds, npar = fit_one(cfg, preps[REFERENCE_ARM], ytr, yva, sp)
            m = None
            for a in ARMS:
                if (a, cfg, sp["seed"], sp["fold"]) in done:
                    continue
                m = write_cell(oof_csv, raw_csv, a, cfg, sp, te_idx, pid, diag,
                               yte, preds, npar, True, done)
            if m is not None:
                shown[cfg] = m["R2_60"]

        # ---- per-arm, CGM-dependent ----
        for a in ARMS:
            for cfg in FITTED:
                if cfg in SHARED:
                    continue
                if (a, cfg, sp["seed"], sp["fold"]) in done:
                    continue
                preds, npar = fit_one(cfg, preps[a], ytr, yva, sp)
                m = write_cell(oof_csv, raw_csv, a, cfg, sp, te_idx, pid, diag,
                               yte, preds, npar, False, done)
                shown[f"{cfg}@{a}"] = m["R2_60"]

        if shown:
            trace = "  ".join(f"{k}={v:+.3f}" for k, v in shown.items())
            print(f"  [{si+1}/{len(splits)}] {trace}", flush=True)

    print(f"\nDone. {len(done)} cells in {(time.time() - t0) / 60:.1f} min.")
    print(f"Wrote {raw_csv}, {oof_csv}, {manifest}")
    print("Next: python deep_sens_analyze.py" + (" --smoke" if SMOKE else ""))


if __name__ == "__main__":
    main()
