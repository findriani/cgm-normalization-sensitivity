"""
attn_run.py  -- repeated-CV FITTING of the attention grid (three loci, matched controls)
========================================================================================
Same protocol as fusion_run.py (participant-level diagnosis-stratified repeated CV,
leakage-free per-fold normalization, mg/dL targets, per-sample OOF, high-res 60x1-min
Dexcom, per-(seed,fold,config) seeding, supervised seed reset immediately before fit).
Statistics live in attn_analyze.py.

Reproducibility scope: runs are SEEDED, not bitwise-reproducible. cuDNN GRU/conv kernels
are non-deterministic on GPU and TF_DETERMINISTIC_OPS is deliberately NOT set (several
ops used here have no deterministic GPU kernel, and it would abort a 13 h job mid-run).
Seeds fix the data partition, the init and the shuffle order; they do not fix float
reduction order. The manifest records TF version + device so a re-run is auditable.

Differences from fusion_run.py, all deliberate (see PLAN_attention.md):
  * N_SEEDS = 15, not 5, and ENFORCED (PLANNED_N_SEEDS). The completed run's variance
    decomposition shows 58-65% of the cluster-bootstrap variance of dR2_60 is seed/CV-
    partition noise, and that at S=5 a refit contrast has CI half-width 0.024-0.038 >
    delta=0.02 -- i.e. NEGLIGIBLE was unreachable regardless of the data. S=15 buys
    superiority power (~25% narrower CI), NOT certified equivalence. This is a PROSPECTIVE
    power analysis on a DIFFERENT study's variance components, not the post-hoc seed
    expansion rejected in PLAN_fusion.md. S is frozen once this script starts.
  * NO SSL, NO retrieval, NO dynablate -- all null in the completed study; that GPU
    budget buys seeds instead. Hence no derived variants: every config is fitted.
  * TWO incumbents: `dl_current_binned` is the PAPER's actual model (12x5-min input);
    `dl_current_highres` is the high-resolution adaptation the fusion study called
    `dl_current`. They are NOT the same model and are no longer given the same name.
  * ATTENTION PROFILES are exported per (model, seed, fold, PARTICIPANT, t) so the
    uncertainty band can be a participant-cluster bootstrap rather than a spread over
    non-independent fold fits.
  * OCCLUSION SWEEP (attn_occlusion_*.csv): a fixed-model CGM perturbation sensitivity
    analysis. Occluded inputs are off-distribution, so the loss reflects model sensitivity
    rather than information content. Attention weights are over ENCODER STATES and for a
    GRU those are cumulative summaries, so they cannot carry that claim by themselves.
    The refit study (window_run.py) provides the direct window-length evidence.
    Costs 5 extra forward passes per (config, seed, fold) -- negligible.

    SMOKE=1 python attn_run.py     # 1 seed, 3 epochs, end-to-end check
    python attn_run.py             # CLEAN=raw (primary), N_SEEDS=15
    python attn_analyze.py
========================================================================================
"""
import os
import json
import hashlib
import platform
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

from lightdl_data import DataBundle, assert_aligned, make_repeated_splits, normalized_fold, flatten
from lightdl_models import build_current_multimodal
from lightdl_run import _seed_for, MainTrajectoryEarlyStopping, _append_csv
from common import compute_metrics_per_horizon, HORIZONS
import attn_models as AM

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------- knobs -------------------------
SMOKE = os.environ.get("SMOKE", "0") == "1"
PLANNED_N_SEEDS = 15                      # PRE-REGISTERED, PLAN_attention.md §1
N_SEEDS = 1 if SMOKE else int(os.environ.get("N_SEEDS", str(PLANNED_N_SEEDS)))
N_FOLDS = 5
EPOCHS = 3 if SMOKE else 100
PATIENCE = 2 if SMOKE else 10
BATCH = 32
LR = 1e-3
RF_TREES = 300
BASE_SEED = 42
D_MODEL = 24
N_HEADS = 4
CLEAN = os.environ.get("CLEAN", "raw")            # raw (primary) | capped (CONDITIONAL, §8)

if CLEAN not in ("raw", "capped"):
    raise SystemExit(f"[abort] CLEAN={CLEAN!r} is not one of 'raw' | 'capped'. "
                     f"Any other value used to silently fall through to capping.")
if not SMOKE and N_SEEDS != PLANNED_N_SEEDS:
    raise SystemExit(f"[abort] N_SEEDS={N_SEEDS} but the pre-registered value is "
                     f"{PLANNED_N_SEEDS} (PLAN_attention.md §1). Changing S after the plan "
                     f"is locked is exactly the post-hoc move this design rules out. "
                     f"Edit PLANNED_N_SEEDS only as a deliberate, documented re-registration.")

MEAL_COL = {"calories": 4, "carbs": 5, "protein": 6, "fat": 7, "fiber": 8}
MEAL_CAPS = {"calories": 2000.0, "carbs": 300.0, "protein": 200.0, "fat": 150.0, "fiber": 80.0}

TAG = CLEAN
RAW_CSV = f"attn_results_raw_{TAG}.csv"
OOF_CSV = f"attn_oof_{TAG}.csv"
CARD_CSV = f"attn_model_cards_{TAG}.csv"
PROF_CSV = f"attn_profiles_{TAG}.csv"
OCC_CSV = f"attn_occlusion_{TAG}.csv"
MANIFEST = f"attn_manifest_{TAG}.json"
OUT_FILES = (RAW_CSV, OOF_CSV, CARD_CSV, PROF_CSV, OCC_CSV)

# Files whose contents determine the FITTED NUMBERS. Hashed and compared on resume.
FIT_CODE = ("attn_models.py", "attn_run.py", "fusion_models.py", "lightdl_models.py",
            "lightdl_run.py", "lightdl_data.py", "foldwise_normalizer.py",
            "common.py")
# Files that only determine the INFERENCE. Hashed and recorded, but NOT compared -- fixing
# a bug in the analyzer must not invalidate a 13 h fit.
ANALYSIS_CODE = ("attn_analyze.py", "ablation_analyze.py")

# Ordered so a partially-completed run still yields WHOLE primary pairs first.
BASES = [
    "cnn_gru__concat",              # primary-1 control (GAP readout)
    "cnn_gru__concat+attnpool",     # primary-1 arm     (attention-pooling readout)
    "cnn__concat",                  # primary-2 control (2nd conv block)
    "cnn_sa__concat",               # primary-2 arm     (transformer block)
    "cnn_gru__coattn_ctl",          # primary-3 control (meal self-attn branch)
    "cnn_gru__coattn",              # primary-3 arm     (meal attends to CGM)
    "cnn__concat+attnpool",         # secondary: readout generalization to a 2nd encoder
    "cnn_gru__xattn",               # secondary: one-way cross-attn (+ replication)
    "sa__concat",                   # secondary: attention replacing the locality prior
    "cnn_sa__coattn",               # pre-declared attention champion (external contrast)
    "dl_current_highres",           # fusion study's `dl_current`; replication anchor
    "dl_current_binned",            # the PAPER's actual incumbent (12x5-min input)
    "rf_highres",                   # external non-DL bar
    "persistence_5min",             # sanity floor (free: no fit)
]
DL_BASES = [b for b in BASES if b not in ("rf_highres", "persistence_5min")]

# Input-level occlusion sweep, defined in MINUTES on the 60x1-min window (skipped for the
# 12x5-min binned model). "keep_last_k": everything before the last k minutes is back-filled
# with the value at minute T-k, so only the last k minutes carry any variation.
# "drop_last_10": the mirror image -- the last 10 minutes are frozen at the value before them.
OCC = [("keep_last_5", "keep", 5), ("keep_last_10", "keep", 10),
       ("keep_last_20", "keep", 20), ("keep_last_30", "keep", 30),
       ("drop_last_10", "drop", 10)]
OCC_T = 60


def clean_static(S):
    if CLEAN == "raw":
        return S
    Sc = S.copy()
    for name, cap in MEAL_CAPS.items():
        Sc[:, MEAL_COL[name]] = np.minimum(Sc[:, MEAL_COL[name]], cap)
    return Sc


def _seq_inputs(d):
    return [d["X"][:, :, 0:1], d["X"][:, :, 1:], d["S"]]


def _standardize_targets(ytr, yva):
    mu = ytr.mean(axis=0); sd = ytr.std(axis=0); sd[sd < 1e-6] = 1.0
    ztr = [((ytr[:, i] - mu[i]) / sd[i]) for i in range(3)]
    zva = [((yva[:, i] - mu[i]) / sd[i]) for i in range(3)]
    return ztr, zva, mu, sd


def _fit_predictor(model, tr, va, seed, fold, name):
    """Fit, return invert(inputs)->mg/dL. Seed RESET here, immediately before fit."""
    ztr, zva, mu, sd = _standardize_targets(tr["y"], va["y"])
    tf.keras.utils.set_random_seed(_seed_for(seed, fold, name))
    cb = MainTrajectoryEarlyStopping(PATIENCE)
    model.fit(_seq_inputs(tr), ztr, validation_data=(_seq_inputs(va), zva),
              epochs=EPOCHS, batch_size=BATCH, callbacks=[cb], verbose=0)

    def invert(inputs):
        out = model.predict(inputs, verbose=0)
        if not isinstance(out, list):
            out = [out]
        return np.stack([np.asarray(out[i]).squeeze(-1) * sd[i] + mu[i] for i in range(3)],
                        axis=-1).astype("float32")
    return invert


def _act_gate(model):
    try:
        return float(model.get_layer("act_gate").gate_value())
    except Exception:
        return np.nan


# ------------------------- attention profiles (per PARTICIPANT) -------------------------
def _profile_rows(model, te, name, seed, fold):
    """Mean attention-pooling weight per encoder-state position, aggregated PER PARTICIPANT
    on the TEST fold. Per-participant (not per-fold) so attn_analyze.py can put a
    participant-cluster bootstrap band on the curve -- the 75 fold fits are not independent
    samples (the same 40 people recur in every seed). Empty for GAP architectures.

    NOTE the interpretive limit documented in attn_models.attention_profile_model: these are
    weights over ENCODER STATES, which for a GRU are cumulative summaries of minutes 0..t.
    The input-level claim comes from the occlusion sweep below, not from this file."""
    pm = AM.attention_profile_model(model)
    if pm is None:
        return None
    w = pm.predict(_seq_inputs(te), verbose=0)[:, :, 0]        # (n_test, T)
    T = w.shape[1]
    df = pd.DataFrame({"pid": np.repeat(te["pid"], T),
                       "t": np.tile(np.arange(T), len(w)),
                       "w": w.ravel()})
    g = df.groupby(["pid", "t"], as_index=False).agg(w=("w", "mean"), n=("w", "size"))
    g.insert(0, "fold", fold); g.insert(0, "seed", seed); g.insert(0, "model", name)
    return g


# ------------------------- input-level occlusion -------------------------
def _occlude(X, kind, k):
    Z = X.copy()
    T = Z.shape[1]
    if kind == "keep":                       # only the final k minutes vary
        Z[:, :T - k, 0] = Z[:, T - k:T - k + 1, 0]
    else:                                    # the final k minutes are frozen
        Z[:, T - k:, 0] = Z[:, T - k - 1:T - k, 0]
    return Z


def _sse_by_pid(te, preds, model, seed, fold, variant):
    """Per-participant SUMMED squared error -- the exact sufficient statistic the
    participant-cluster bootstrap and the event-weighted dR2 both consume."""
    se = (te["y"] - preds) ** 2
    df = pd.DataFrame({"pid": te["pid"], "n": 1,
                       **{f"sse_{h}": se[:, i] for i, h in enumerate(HORIZONS)}})
    g = df.groupby("pid", as_index=False).sum()
    g.insert(0, "variant", variant); g.insert(0, "fold", fold)
    g.insert(0, "seed", seed); g.insert(0, "model", model)
    return g


def _occ_rows(invert, te, preds, name, seed, fold):
    """R2 loss under raw-window perturbation. Defined in minutes, so only for T=60."""
    if te["X"].shape[1] != OCC_T:
        return None
    rows = [_sse_by_pid(te, preds, name, seed, fold, "full")]
    for vname, kind, k in OCC:
        Xo = _occlude(te["X"], kind, k)
        p = invert([Xo[:, :, 0:1], Xo[:, :, 1:], te["S"]])
        rows.append(_sse_by_pid(te, p, name, seed, fold, vname))
    return pd.concat(rows, ignore_index=True)


# ------------------------- one config -------------------------
def run_base(base, fh, fb, seed, fold, base5):
    """Returns (preds (n_test,3) mg/dL, meta, profile_df|None, occlusion_df|None).
    Predictions are always indexed by the HIGH-RES fold's test rows; the two bundles are
    row-aligned (assert_aligned), so the binned model's OOF lines up with everything else."""
    tr, va, te = fh["train"], fh["val"], fh["test"]
    nan_meta = {"n_params": 0, "act_gate": np.nan}

    if base == "rf_highres":
        Ftr, Fte = flatten(tr["X"], tr["S"]), flatten(te["X"], te["S"])
        preds = np.zeros((len(te["y"]), 3), "float32")
        for i in range(3):
            rf = RandomForestRegressor(n_estimators=RF_TREES,
                                       random_state=_seed_for(seed, fold, base) % (2**31), n_jobs=-1)
            rf.fit(Ftr, tr["y"][:, i]); preds[:, i] = rf.predict(Fte)
        return preds, dict(nan_meta), None, None

    if base == "persistence_5min":
        p = base5[te["idx"]].astype("float32")
        return np.repeat(p[:, None], 3, axis=1), dict(nan_meta), None, None

    tf.keras.backend.clear_session()

    if base.startswith("dl_current"):
        fd = fb if base.endswith("binned") else fh
        dtr, dva, dte = fd["train"], fd["val"], fd["test"]
        tf.keras.utils.set_random_seed(_seed_for(seed, fold, base))
        model = build_current_multimodal(dtr["X"].shape[1], dtr["X"].shape[2] - 1, dtr["S"].shape[1])
        invert = _fit_predictor(model, dtr, dva, seed, fold, base)
        preds = invert(_seq_inputs(dte))
        meta = {"n_params": int(model.count_params()), "act_gate": np.nan}
        return preds, meta, None, _occ_rows(invert, dte, preds, base, seed, fold)

    core = base.split("+")[0]
    flags = set(base.split("+")[1:])
    readout = "attn" if "attnpool" in flags else "gap"
    dyn_mode = "off" if "nodyn" in flags else "gated"
    T, n_dyn, n_static = tr["X"].shape[1], tr["X"].shape[2] - 1, tr["S"].shape[1]

    tf.keras.utils.set_random_seed(_seed_for(seed, fold, base))
    model = AM.build(core, T, n_dyn, n_static, d=D_MODEL, n_heads=N_HEADS, lr=LR,
                     dyn_mode=dyn_mode, readout=readout)
    invert = _fit_predictor(model, tr, va, seed, fold, base)   # resets seed before fit
    preds = invert(_seq_inputs(te))
    meta = {"n_params": int(model.count_params()), "act_gate": _act_gate(model)}
    return (preds, meta, _profile_rows(model, te, base, seed, fold),
            _occ_rows(invert, te, preds, base, seed, fold))


# ------------------------- checkpoint / manifest -------------------------
def _data_hash(b):
    h = hashlib.md5()
    for a in (b.X, b.y, b.S, b.diag):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(b.pid).encode())
    return h.hexdigest()


def _hash_files(names, required=True):
    h = hashlib.md5()
    for f in names:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            if required:
                raise SystemExit(f"[abort] {f} is missing. Every file that determines the "
                                 f"result must be present and hashed into the manifest.")
            continue
        h.update(f.encode())
        h.update(open(p, "rb").read())
    return h.hexdigest()


def _manifest(dh_hi, dh_bi, n_samples, n_participants):
    return {"data_hash": dh_hi, "data_hash_binned": dh_bi,
            "code_hash": _hash_files(FIT_CODE),
            "code_hash_analysis": _hash_files(ANALYSIS_CODE, required=False),
            "tf": tf.__version__, "python": platform.python_version(),
            "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
            "deterministic": "seeded (not bitwise: cuDNN kernels are nondeterministic)",
            "emitted_configs": list(BASES), "dl_configs": list(DL_BASES),
            "n_seeds": N_SEEDS, "planned_n_seeds": PLANNED_N_SEEDS, "n_folds": N_FOLDS,
            "n_samples": int(n_samples), "n_participants": int(n_participants),
            "smoke": SMOKE, "epochs": EPOCHS, "patience": PATIENCE, "batch": BATCH, "lr": LR,
            "rf_trees": RF_TREES, "d_model": D_MODEL, "n_heads": N_HEADS,
            "base_seed": BASE_SEED, "clean": CLEAN, "meal_caps": MEAL_CAPS,
            "occ_variants": ["full"] + [v[0] for v in OCC], "occ_timesteps": OCC_T,
            "pos_mode": AM.POS_MODE, "ff_mult": AM.FF_MULT, "d_att": AM.D_ATT}


# manifest keys that may drift without invalidating the fitted numbers
SOFT_KEYS = {"code_hash_analysis"}


def _guard_and_resume(man):
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST))
        diff = sorted({k for k in set(old) | set(man)
                       if k not in SOFT_KEYS and old.get(k) != man.get(k)})
        if diff:
            raise SystemExit(f"[abort] {MANIFEST} differs from current settings on {diff}. "
                             f"N_SEEDS is FROZEN once a run starts (PLAN_attention.md §1). "
                             f"Delete attn_*_{TAG}.* only if you intend a full restart.")
        if old.get("code_hash_analysis") != man.get("code_hash_analysis"):
            print("  note: analysis code changed since this run started (does not affect fits)")
            old["code_hash_analysis"] = man["code_hash_analysis"]
            json.dump(old, open(MANIFEST, "w"), indent=2)
    else:
        stale = [p for p in OUT_FILES if os.path.exists(p)]
        if stale:
            raise SystemExit(f"[abort] {MANIFEST} is missing but result files already exist: "
                             f"{stale}. Refusing to write a fresh manifest over rows whose "
                             f"provenance cannot be verified -- that would silently certify "
                             f"stale results. Delete them for a clean restart.")
        json.dump(man, open(MANIFEST, "w"), indent=2)

    done = set()
    if os.path.exists(RAW_CSV):
        r = pd.read_csv(RAW_CSV)
        rd = r.drop_duplicates(["model", "seed", "fold"], keep="last")
        if len(rd) != len(r):
            rd.to_csv(RAW_CSV, index=False)
            print(f"deduped {len(r) - len(rd)} duplicate metric row(s)")
        done = set(map(tuple, rd[["model", "seed", "fold"]].values.tolist()))
    if os.path.exists(OOF_CSV):
        oof = pd.read_csv(OOF_CSV, dtype={"pid": str})
        n0 = len(oof)
        oof = oof.drop_duplicates(["model", "seed", "fold", "sample_idx"], keep="last")
        ck = oof[["model", "seed", "fold"]].apply(tuple, axis=1)
        oof = oof[~ck.isin(set(ck) - done)]                  # drop OOF without a metric marker
        if len(oof) != n0:
            oof.to_csv(OOF_CSV, index=False)
            print(f"reconciled OOF: dropped {n0 - len(oof)} orphan/duplicate row(s)")
    for path, keys in ((PROF_CSV, ["model", "seed", "fold", "pid", "t"]),
                       (OCC_CSV, ["model", "seed", "fold", "pid", "variant"])):
        if os.path.exists(path):
            d = pd.read_csv(path, dtype={"pid": str})
            k = d[["model", "seed", "fold"]].apply(tuple, axis=1)
            d2 = d[k.isin(done)].drop_duplicates(keys, keep="last")
            if len(d2) != len(d):
                d2.to_csv(path, index=False)
                print(f"reconciled {path}: dropped {len(d) - len(d2)} row(s)")
    return done


def main():
    highres = DataBundle("highres"); binned = DataBundle("binned")
    assert_aligned(binned, highres)                    # BEFORE any capping touches S
    dh_hi, dh_bi = _data_hash(highres), _data_hash(binned)
    highres.S = clean_static(highres.S)
    binned.S = clean_static(binned.S)                  # keep the two incumbents comparable
    base5 = highres.X[:, -5:, 0].mean(axis=1)          # last-5-min persistence (mg/dL)
    n_part = len(np.unique(highres.pid))
    done = _guard_and_resume(_manifest(dh_hi, dh_bi, highres.n, n_part))
    splits = make_repeated_splits(highres.pid, highres.diag, n_seeds=N_SEEDS,
                                  n_folds=N_FOLDS, base_seed=BASE_SEED)
    cards_seen = set(pd.read_csv(CARD_CSV)["model"]) if os.path.exists(CARD_CSV) else set()
    total = len(splits) * len(BASES)
    print(f"SMOKE={SMOKE} CLEAN={CLEAN} seeds={N_SEEDS} folds={N_FOLDS} -> {len(splits)} splits "
          f"x {len(BASES)} configs = {total} cells | done: {len(done)}")
    print(f"  {highres.n} samples / {n_part} participants; occlusion variants: "
          f"{['full'] + [v[0] for v in OCC]}")

    for si, sp in enumerate(splits):
        if all((b, sp["seed"], sp["fold"]) in done for b in BASES):
            continue
        fh = normalized_fold(highres, sp, with_aux=False)
        fb = normalized_fold(binned, sp, with_aux=False)     # for dl_current_binned only
        te = fh["test"]
        for base in BASES:
            if (base, sp["seed"], sp["fold"]) in done:
                continue
            preds, meta, prof, occ = run_base(base, fh, fb, sp["seed"], sp["fold"], base5)

            oof = pd.DataFrame({                               # OOF FIRST, marker last
                "model": base, "seed": sp["seed"], "fold": sp["fold"],
                "sample_idx": te["idx"], "pid": te["pid"], "diag": te["diag"],
                **{f"y_true_{h}": te["y"][:, i] for i, h in enumerate(HORIZONS)},
                **{f"y_pred_{h}": preds[:, i] for i, h in enumerate(HORIZONS)}})
            _append_csv(OOF_CSV, oof)
            if prof is not None:
                _append_csv(PROF_CSV, prof)
            if occ is not None:
                _append_csv(OCC_CSV, occ)

            m = compute_metrics_per_horizon(te["y"], preds)
            m.update({"model": base, "seed": sp["seed"], "fold": sp["fold"], **meta})
            _append_csv(RAW_CSV, pd.DataFrame([m]))
            if base not in cards_seen:
                _append_csv(CARD_CSV, pd.DataFrame([{"model": base, "n_params": meta["n_params"]}]))
                cards_seen.add(base)
            done.add((base, sp["seed"], sp["fold"]))
            print(f"  [{si+1}/{len(splits)}] {base:26s} R2_60={m['R2_60']:+.3f} "
                  f"NRMSE60={m['NRMSE_60']:.3f} params={meta['n_params']}")

    print(f"\nDone TAG={TAG}. Wrote {', '.join(OUT_FILES)}, {MANIFEST}."
          f"\nNext: python attn_analyze.py")


if __name__ == "__main__":
    main()
