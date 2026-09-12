"""
window_run.py  -- REFIT on shortened pre-meal windows
=====================================================
Follow-up to the attention run. The occlusion sweep there showed that blanking minutes
1-55 of the 60-min pre-meal window SIGNIFICANTLY IMPROVES every conv/GRU model
(+0.013..+0.016 R2_60, CI excludes 0). But occlusion feeds an off-distribution input to a
model fitted on the full window, so it measures fixed-model perturbation sensitivity
rather than information content. This study asks the question properly: REFIT from
scratch on a short window and compare.

Protocol is byte-identical to attn_run.py (same splits from make_repeated_splits with
base_seed=42, same per-fold FoldNormalizer, mg/dL targets, per-sample OOF, same early
stopping, same 15 seeds), so the numbers are directly comparable. The ONLY thing that
changes is which timesteps enter the model.

The window is applied to the RAW bundle BEFORE per-fold normalization, so the normalizer
sees exactly the input the model sees (still fit on training participants only).

Windows (the 60 slots are 1 minute apart; slot 59 is the last reading before the meal):
  w60        all 60 slots                     -- the current paper input, the control
  w10_1min   slots 50..59                     -- last 10 minutes at 1-min resolution
  w10_5min   slots 49, 54, 59                 -- last 10 minutes as 3 readings at 5-min
                                                 spacing, i.e. t=0, -5, -10 minutes
  w5_1min    slots 55..59                     -- last 5 minutes (what occlusion favoured)

Both 10-minute parameterizations are fitted because it is genuinely unclear which is
right: the series carries some sub-5-minute structure (~38 distinct values per 60 slots),
but not a full 60, so 10 one-minute samples may be partly redundant.

    SMOKE=1 python window_run.py     # 1 seed, 3 epochs
    python window_run.py             # N_SEEDS=15
    python window_analyze.py
=====================================================
"""
import os
import copy
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

SMOKE = os.environ.get("SMOKE", "0") == "1"
PLANNED_N_SEEDS = 15
N_SEEDS = 1 if SMOKE else int(os.environ.get("N_SEEDS", str(PLANNED_N_SEEDS)))
N_FOLDS = 5
EPOCHS = 3 if SMOKE else 100
PATIENCE = 2 if SMOKE else 10
BATCH, LR, RF_TREES, BASE_SEED, D_MODEL, N_HEADS = 32, 1e-3, 300, 42, 24, 4
CLEAN = os.environ.get("CLEAN", "raw")

if CLEAN not in ("raw", "capped"):
    raise SystemExit(f"[abort] CLEAN={CLEAN!r} not in 'raw'|'capped'")
if not SMOKE and N_SEEDS != PLANNED_N_SEEDS:
    raise SystemExit(f"[abort] N_SEEDS={N_SEEDS}, pre-registered {PLANNED_N_SEEDS}.")

MEAL_COL = {"calories": 4, "carbs": 5, "protein": 6, "fat": 7, "fiber": 8}
MEAL_CAPS = {"calories": 2000.0, "carbs": 300.0, "protein": 200.0, "fat": 150.0, "fiber": 80.0}

TAG = CLEAN
RAW_CSV, OOF_CSV = f"window_results_raw_{TAG}.csv", f"window_oof_{TAG}.csv"
CARD_CSV, MANIFEST = f"window_model_cards_{TAG}.csv", f"window_manifest_{TAG}.json"
OUT_FILES = (RAW_CSV, OOF_CSV, CARD_CSV)

# ---- windows: index arrays into the 60 one-minute slots (59 = last before the meal) ----
# NOTE: w10_1min covers indices 50-59 (t = -10 to -1, 10 readings, 9 intervals).
#       w10_5min covers indices 49, 54, 59 (t = -11, -6, -1, 3 readings, 10 intervals).
#       Their endpoints differ (50 vs 49), so they are NOT two samplings of the
#       same interval.  A direct resolution comparison is therefore invalid.
WINDOWS = {
    "w60":      np.arange(60),
    "w10_1min": np.arange(50, 60),
    "w10_5min": np.array([49, 54, 59]),        # t = -10, -5, 0 minutes
    "w5_1min":  np.arange(55, 60),
}
WINDOW_ORDER = ["w60", "w10_1min", "w10_5min", "w5_1min"]
ARCHS = ["cnn_gru__concat",      # representative recurrent encoder
         "sa__concat",           # the only arm that won outright in the attention run
         "dl_current"]           # the paper's own architecture

BASES = ([f"{a}@{w}" for w in WINDOW_ORDER for a in ARCHS]
         + [f"rf_highres@{w}" for w in ("w60", "w10_1min")]
         + ["persistence_5min"])

FIT_CODE = ("window_run.py", "attn_models.py", "fusion_models.py", "lightdl_models.py",
            "lightdl_run.py", "lightdl_data.py", "foldwise_normalizer.py",
            "common.py")
ANALYSIS_CODE = ("window_analyze.py", "attn_analyze.py", "ablation_analyze.py")
SOFT_KEYS = {"code_hash_analysis"}


def clean_static(S):
    if CLEAN == "raw":
        return S
    Sc = S.copy()
    for n, c in MEAL_CAPS.items():
        Sc[:, MEAL_COL[n]] = np.minimum(Sc[:, MEAL_COL[n]], c)
    return Sc


def sliced_bundle(b, idx):
    """Shallow copy with only the selected timesteps. Slicing BEFORE normalized_fold means
    the FoldNormalizer is fit on exactly the input the model receives."""
    nb = copy.copy(b)
    nb.X = np.ascontiguousarray(b.X[:, idx, :])
    nb.T = len(idx)
    return nb


def _seq_inputs(d):
    return [d["X"][:, :, 0:1], d["X"][:, :, 1:], d["S"]]


def _standardize_targets(ytr, yva):
    mu, sd = ytr.mean(axis=0), ytr.std(axis=0); sd[sd < 1e-6] = 1.0
    return ([(ytr[:, i] - mu[i]) / sd[i] for i in range(3)],
            [(yva[:, i] - mu[i]) / sd[i] for i in range(3)], mu, sd)


def _fit_predictor(model, tr, va, seed, fold, name):
    ztr, zva, mu, sd = _standardize_targets(tr["y"], va["y"])
    tf.keras.utils.set_random_seed(_seed_for(seed, fold, name))
    model.fit(_seq_inputs(tr), ztr, validation_data=(_seq_inputs(va), zva), epochs=EPOCHS,
              batch_size=BATCH, callbacks=[MainTrajectoryEarlyStopping(PATIENCE)], verbose=0)

    def invert(inputs):
        out = model.predict(inputs, verbose=0)
        if not isinstance(out, list):
            out = [out]
        return np.stack([np.asarray(out[i]).squeeze(-1) * sd[i] + mu[i] for i in range(3)],
                        axis=-1).astype("float32")
    return invert


def run_base(base, folds, seed, fold, base5, te_ref):
    nan_meta = {"n_params": 0, "act_gate": np.nan}
    if base == "persistence_5min":
        p = base5[te_ref["idx"]].astype("float32")
        return np.repeat(p[:, None], 3, axis=1), dict(nan_meta)

    arch, win = base.split("@")
    fh = folds[win]
    tr, va, te = fh["train"], fh["val"], fh["test"]

    if arch == "rf_highres":
        Ftr, Fte = flatten(tr["X"], tr["S"]), flatten(te["X"], te["S"])
        preds = np.zeros((len(te["y"]), 3), "float32")
        for i in range(3):
            rf = RandomForestRegressor(n_estimators=RF_TREES,
                                       random_state=_seed_for(seed, fold, base) % (2**31), n_jobs=-1)
            rf.fit(Ftr, tr["y"][:, i]); preds[:, i] = rf.predict(Fte)
        return preds, dict(nan_meta)

    tf.keras.backend.clear_session()
    T, n_dyn, n_static = tr["X"].shape[1], tr["X"].shape[2] - 1, tr["S"].shape[1]
    tf.keras.utils.set_random_seed(_seed_for(seed, fold, base))
    if arch == "dl_current":
        model = build_current_multimodal(T, n_dyn, n_static)
    else:
        model = AM.build(arch, T, n_dyn, n_static, d=D_MODEL, n_heads=N_HEADS, lr=LR)
    invert = _fit_predictor(model, tr, va, seed, fold, base)
    return invert(_seq_inputs(te)), {"n_params": int(model.count_params()), "act_gate": np.nan}


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
                raise SystemExit(f"[abort] {f} is missing.")
            continue
        h.update(f.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


def _manifest(dh, n_samples, n_part):
    return {"data_hash": dh, "code_hash": _hash_files(FIT_CODE),
            "code_hash_analysis": _hash_files(ANALYSIS_CODE, required=False),
            "tf": tf.__version__, "python": platform.python_version(),
            "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
            "deterministic": "seeded (not bitwise: cuDNN kernels are nondeterministic)",
            "emitted_configs": list(BASES),
            "windows": {k: [int(i) for i in v] for k, v in WINDOWS.items()},
            "archs": list(ARCHS), "n_seeds": N_SEEDS, "planned_n_seeds": PLANNED_N_SEEDS,
            "n_folds": N_FOLDS, "n_samples": int(n_samples), "n_participants": int(n_part),
            "smoke": SMOKE, "epochs": EPOCHS, "patience": PATIENCE, "batch": BATCH, "lr": LR,
            "rf_trees": RF_TREES, "d_model": D_MODEL, "n_heads": N_HEADS,
            "base_seed": BASE_SEED, "clean": CLEAN, "meal_caps": MEAL_CAPS}


def _guard_and_resume(man):
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST))
        diff = sorted({k for k in set(old) | set(man)
                       if k not in SOFT_KEYS and old.get(k) != man.get(k)})
        if diff:
            raise SystemExit(f"[abort] {MANIFEST} differs on {diff}. Delete window_*_{TAG}.* "
                             f"only if you intend a full restart.")
    else:
        stale = [p for p in OUT_FILES if os.path.exists(p)]
        if stale:
            raise SystemExit(f"[abort] {MANIFEST} missing but results exist: {stale}. "
                             f"Refusing to certify rows of unverifiable provenance.")
        json.dump(man, open(MANIFEST, "w"), indent=2)
    done = set()
    if os.path.exists(RAW_CSV):
        r = pd.read_csv(RAW_CSV).drop_duplicates(["model", "seed", "fold"], keep="last")
        r.to_csv(RAW_CSV, index=False)
        done = set(map(tuple, r[["model", "seed", "fold"]].values.tolist()))
    if os.path.exists(OOF_CSV):
        o = pd.read_csv(OOF_CSV, dtype={"pid": str})
        n0 = len(o)
        o = o.drop_duplicates(["model", "seed", "fold", "sample_idx"], keep="last")
        ck = o[["model", "seed", "fold"]].apply(tuple, axis=1)
        o = o[~ck.isin(set(ck) - done)]
        if len(o) != n0:
            o.to_csv(OOF_CSV, index=False)
            print(f"reconciled OOF: dropped {n0 - len(o)} row(s)")
    return done


def main():
    hi = DataBundle("highres"); bi = DataBundle("binned")
    assert_aligned(bi, hi)
    dh = _data_hash(hi)
    hi.S = clean_static(hi.S)
    base5 = hi.X[:, -5:, 0].mean(axis=1)
    n_part = len(np.unique(hi.pid))
    done = _guard_and_resume(_manifest(dh, hi.n, n_part))
    bundles = {w: sliced_bundle(hi, idx) for w, idx in WINDOWS.items()}
    splits = make_repeated_splits(hi.pid, hi.diag, n_seeds=N_SEEDS, n_folds=N_FOLDS,
                                  base_seed=BASE_SEED)
    cards = set(pd.read_csv(CARD_CSV)["model"]) if os.path.exists(CARD_CSV) else set()
    print(f"SMOKE={SMOKE} seeds={N_SEEDS} -> {len(splits)} splits x {len(BASES)} configs "
          f"= {len(splits)*len(BASES)} cells | done: {len(done)}")
    print("  windows: " + ", ".join(f"{w}(T={len(WINDOWS[w])})" for w in WINDOW_ORDER))

    for si, sp in enumerate(splits):
        if all((b, sp["seed"], sp["fold"]) in done for b in BASES):
            continue
        folds = {w: normalized_fold(bundles[w], sp, with_aux=False) for w in WINDOWS}
        te_ref = folds["w60"]["test"]
        for base in BASES:
            if (base, sp["seed"], sp["fold"]) in done:
                continue
            preds, meta = run_base(base, folds, sp["seed"], sp["fold"], base5, te_ref)
            oof = pd.DataFrame({
                "model": base, "seed": sp["seed"], "fold": sp["fold"],
                "sample_idx": te_ref["idx"], "pid": te_ref["pid"], "diag": te_ref["diag"],
                **{f"y_true_{h}": te_ref["y"][:, i] for i, h in enumerate(HORIZONS)},
                **{f"y_pred_{h}": preds[:, i] for i, h in enumerate(HORIZONS)}})
            _append_csv(OOF_CSV, oof)
            m = compute_metrics_per_horizon(te_ref["y"], preds)
            m.update({"model": base, "seed": sp["seed"], "fold": sp["fold"], **meta})
            _append_csv(RAW_CSV, pd.DataFrame([m]))
            if base not in cards:
                _append_csv(CARD_CSV, pd.DataFrame([{"model": base, "n_params": meta["n_params"]}]))
                cards.add(base)
            done.add((base, sp["seed"], sp["fold"]))
            print(f"  [{si+1}/{len(splits)}] {base:28s} R2_60={m['R2_60']:+.3f} "
                  f"params={meta['n_params']}")

    print(f"\nDone TAG={TAG}. Next: python window_analyze.py")


if __name__ == "__main__":
    main()
