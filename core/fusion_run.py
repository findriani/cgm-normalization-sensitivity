"""
fusion_run.py  -- repeated-CV FITTING of the modular fusion grid (+ SSL / retrieval)
====================================================================================
Same protocol as lightdl_run.py (participant-level stratified repeated CV, leakage-free
per-fold normalization, mg/dL targets, per-sample OOF). DL configs act on HIGH-RES
60x1-min Dexcom windows. Statistics live in fusion_analyze.py.

Review fixes implemented here:
  * PAIRED ADD-ONS: each BASE model is fit ONCE per (seed,fold); its +retrieval and
    +dynablate variants are DERIVED from that exact fitted model -- so a contrast
    isolates the mechanism, not re-initialisation / minibatch / dropout noise.
  * RNG: the supervised seed is reset immediately BEFORE model.fit(), so an SSL cache
    hit vs miss (after a resume) cannot change the supervised fit.
  * CROSS-DEVICE SSL: per-fold pretraining pools TRAIN-participant Dexcom + Libre CGM
    windows (leakage-free). Pooled-corpus weights (fusion_ssl.py CLI) are loaded by the
    explicit transductive `+sslpool` config.
  * RETRIEVAL SCALE is estimated from TRAIN memory only and fixed for val/test (no
    test-batch statistics); mixing weight tuned on validation.
  * CONTROLS added: dl_current, persistence_5min, no_dyn (hard gate), dynablate
    (frozen test-time dyn zero), film+ssl+retrieval, capped-vs-raw meal (CLEAN).
  * MANIFEST records configs, seeds, smoke, RF trees, batch, lr, mask params, retrieval
    grid, meal caps, code hash and TF version; a fixed seed count (no conditional
    expansion -- that would be a sequential-analysis decision).

Colab (TF/Keras 3):
    SMOKE=1 python fusion_run.py            # 1 seed, few epochs, end-to-end check
    python fusion_run.py                    # CLEAN=raw (primary) fixed N_SEEDS
    CLEAN=capped python fusion_run.py       # capped-meal fusion sensitivity
    python fusion_ssl.py                    # optional: pooled encoder (enables +sslpool)
    python fusion_analyze.py
Upload: dexcom_*_prediction_raw.npz, libre_raw_prediction_raw.npz, foldwise_normalizer.py,
common.py, lightdl_data.py, lightdl_models.py, fusion_models.py, fusion_ssl.py.
====================================================================================
"""
import os
import json
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

from lightdl_data import DataBundle, assert_aligned, make_repeated_splits, normalized_fold, flatten
from lightdl_models import build_current_multimodal
from lightdl_run import _seed_for, MainTrajectoryEarlyStopping, _append_csv
from common import compute_metrics_per_horizon, HORIZONS
import fusion_models as FM
from fusion_ssl import pretrain_encoder, load_libre_cgm

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------- knobs -------------------------
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_SEEDS = 1 if SMOKE else int(os.environ.get("N_SEEDS", "5"))   # FIXED; no conditional expansion
N_FOLDS = 5
EPOCHS = 3 if SMOKE else 100
PATIENCE = 2 if SMOKE else 10
SSL_EPOCHS = 3 if SMOKE else 40
BATCH = 32
LR = 1e-3
RF_TREES = 300
BASE_SEED = 42
D_MODEL = 24
MASK_FRAC, MASK_SPAN = 0.15, 5
CLEAN = os.environ.get("CLEAN", "raw")            # raw (primary) | capped (sensitivity)

RETR_K = 25
RETR_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RETR_TAUS = (0.5, 1.0, 2.0)

MEAL_COL = {"calories": 4, "carbs": 5, "protein": 6, "fat": 7, "fiber": 8}
MEAL_CAPS = {"calories": 2000.0, "carbs": 300.0, "protein": 200.0, "fat": 150.0, "fiber": 80.0}

TAG = CLEAN
RAW_CSV = f"fusion_results_raw_{TAG}.csv"
OOF_CSV = f"fusion_oof_{TAG}.csv"
CARD_CSV = f"fusion_model_cards_{TAG}.csv"
MANIFEST = f"fusion_manifest_{TAG}.json"


def clean_static(S):
    if CLEAN == "raw":
        return S
    Sc = S.copy()
    for name, cap in MEAL_CAPS.items():
        Sc[:, MEAL_COL[name]] = np.minimum(Sc[:, MEAL_COL[name]], cap)
    return Sc


def _pooled_weights_path(enc):
    return os.path.join(HERE, f"fusion_ssl_{enc}_pooled.weights.h5")


def _base_configs():
    bases = [f"{e}__concat" for e in FM.ENCODERS] + \
            ["cnn_gru__gate", "cnn_gru__film", "cnn_gru__xattn",
             "cnn_gru__film+ssl", "cnn_gru__film+nodyn",
             "dl_current", "rf_highres", "persistence_5min"]
    if os.path.exists(_pooled_weights_path("cnn_gru")):
        bases.append("cnn_gru__film+sslpool")
    return bases


def _emitted(bases):
    em = list(bases)
    if "cnn_gru__film" in bases:
        em += ["cnn_gru__film+retrieval", "cnn_gru__film+dynablate"]
    if "cnn_gru__film+ssl" in bases:
        em += ["cnn_gru__film+ssl+retrieval"]
    return em


# ------------------------- helpers -------------------------
def _seq_inputs(d):
    return [d["X"][:, :, 0:1], d["X"][:, :, 1:], d["S"]]


def _standardize_targets(ytr, yva):
    mu = ytr.mean(axis=0); sd = ytr.std(axis=0); sd[sd < 1e-6] = 1.0
    ztr = [((ytr[:, i] - mu[i]) / sd[i]) for i in range(3)]
    zva = [((yva[:, i] - mu[i]) / sd[i]) for i in range(3)]
    return ztr, zva, mu, sd


def _fit_predictor(model, tr, va, seed, fold, name):
    """Fit, return invert(inputs)->mg/dL. Seed RESET here (after any SSL pretrain)."""
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


# ------------------------- retrieval (derived, leakage-safe, fixed scale) -------------------------
def _emb_model(model):
    return tf.keras.Model(model.inputs, model.get_layer("head_dense").input)


def _train_scale(Zmem, rng):
    m = Zmem if len(Zmem) <= 500 else Zmem[rng.choice(len(Zmem), 500, replace=False)]
    d2 = ((m[:, None, :] - m[None, :, :]) ** 2).sum(-1)
    d2 = d2[d2 > 0]
    return float(np.median(d2) + 1e-9) if len(d2) else 1.0


def _knn_pred(Zmem, ymem, Zq, k, tau, scale):
    d2 = (Zq ** 2).sum(1)[:, None] + (Zmem ** 2).sum(1)[None, :] - 2 * Zq @ Zmem.T
    d2 = np.maximum(d2, 0.0)
    idx = np.argsort(d2, axis=1)[:, :k]
    preds = np.zeros((len(Zq), 3), "float32")
    for i in range(len(Zq)):
        dk = d2[i, idx[i]] / (tau * scale)
        w = np.exp(-(dk - dk.min())); w /= w.sum() + 1e-9
        preds[i] = w @ ymem[idx[i]]
    return preds


def _retrieval_from(model, tr, va, te, invert):
    """Blend the SAME model's base predictions with a train-memory kNN; scale fixed from
    train, (alpha,tau) tuned on val RMSE60."""
    emb = _emb_model(model)
    Ztr = emb.predict(_seq_inputs(tr), verbose=0)
    Zva = emb.predict(_seq_inputs(va), verbose=0)
    Zte = emb.predict(_seq_inputs(te), verbose=0)
    m, s = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr, Zva, Zte = (Ztr - m) / s, (Zva - m) / s, (Zte - m) / s
    ymem = tr["y"].astype("float32")
    scale = _train_scale(Ztr, np.random.default_rng(0))
    base_va, base_te = invert(_seq_inputs(va)), invert(_seq_inputs(te))
    best = (np.inf, 0.0, 1.0)
    for tau in RETR_TAUS:
        rva = _knn_pred(Ztr, ymem, Zva, RETR_K, tau, scale)
        for a in RETR_ALPHAS:
            blend = (1 - a) * base_va + a * rva
            rmse60 = float(np.sqrt(np.mean((va["y"][:, 1] - blend[:, 1]) ** 2)))
            if rmse60 < best[0]:
                best = (rmse60, a, tau)
    _, alpha, tau = best
    rte = _knn_pred(Ztr, ymem, Zte, RETR_K, tau, scale)
    return ((1 - alpha) * base_te + alpha * rte).astype("float32"), {"retr_alpha": alpha, "retr_tau": tau}


# ------------------------- SSL (cross-device, per fold, leakage-free) -------------------------
def _ssl_encoder(base, tr, seed, fold, libre, ssl_cache):
    enc_cfg = base.split("__")[0]
    key = (seed, fold, enc_cfg)
    if key in ssl_cache:
        return ssl_cache[key]
    T = tr["X"].shape[1]
    corpus = [tr["X"][:, :, 0:1].astype("float32")]            # fold-normalized Dexcom train CGM
    libX, libpid = libre
    if libX is not None:
        train_pids = set(map(str, tr["pid"]))
        mask = np.isin(libpid, list(train_pids))
        if mask.any():
            lib = libX[mask]
            lib = (lib - lib.mean()) / (lib.std() + 1e-6)      # device-standardized, train only
            corpus.append(lib.astype("float32"))
    X = np.concatenate(corpus, axis=0)
    enc = pretrain_encoder(X, base, T, d=D_MODEL, seed=_seed_for(seed, fold, "ssl_" + enc_cfg),
                           epochs=SSL_EPOCHS, mask_frac=MASK_FRAC, span=MASK_SPAN)
    ssl_cache[key] = enc
    return enc


# ------------------------- one base fit (+ derived) -------------------------
def run_base(base, fh, seed, fold, ssl_cache, libre, base5):
    tr, va, te = fh["train"], fh["val"], fh["test"]
    nan_meta = {"n_params": 0, "act_gate": np.nan, "retr_alpha": np.nan, "retr_tau": np.nan}

    if base == "rf_highres":
        Ftr, Fte = flatten(tr["X"], tr["S"]), flatten(te["X"], te["S"])
        preds = np.zeros((len(te["y"]), 3), "float32")
        for i in range(3):
            rf = RandomForestRegressor(n_estimators=RF_TREES,
                                       random_state=_seed_for(seed, fold, base) % (2**31), n_jobs=-1)
            rf.fit(Ftr, tr["y"][:, i]); preds[:, i] = rf.predict(Fte)
        return [(base, preds, dict(nan_meta))]

    if base == "persistence_5min":
        p = base5[te["idx"]].astype("float32")
        return [(base, np.repeat(p[:, None], 3, axis=1), dict(nan_meta))]

    tf.keras.backend.clear_session()
    if base == "dl_current":
        tf.keras.utils.set_random_seed(_seed_for(seed, fold, base))
        model = build_current_multimodal(tr["X"].shape[1], tr["X"].shape[2] - 1, tr["S"].shape[1])
        invert = _fit_predictor(model, tr, va, seed, fold, base)
        preds = invert(_seq_inputs(te))
        return [(base, preds, {"n_params": int(model.count_params()), "act_gate": np.nan,
                               "retr_alpha": np.nan, "retr_tau": np.nan})]

    # ---- fusion configs ----
    name = base
    core = name.split("+")[0]
    flags = set(name.split("+")[1:])
    dyn_mode = "off" if "nodyn" in flags else "gated"
    T, n_dyn, n_static = tr["X"].shape[1], tr["X"].shape[2] - 1, tr["S"].shape[1]

    tf.keras.utils.set_random_seed(_seed_for(seed, fold, name))
    model = FM.build(core, T, n_dyn, n_static, d=D_MODEL, lr=LR, dyn_mode=dyn_mode)
    if "ssl" in flags:
        FM.transfer_encoder(model, _ssl_encoder(core, tr, seed, fold, libre, ssl_cache))
    if "sslpool" in flags:
        enc = FM.build_cgm_encoder(core, T, d=D_MODEL)
        enc.load_weights(_pooled_weights_path(core.split("__")[0]))
        FM.transfer_encoder(model, enc)

    invert = _fit_predictor(model, tr, va, seed, fold, name)   # resets seed before fit
    preds = invert(_seq_inputs(te))
    meta = {"n_params": int(model.count_params()), "act_gate": _act_gate(model),
            "retr_alpha": np.nan, "retr_tau": np.nan}
    emitted = [(name, preds, meta)]

    # derived variants from THIS fitted model (paired, no re-fit)
    if name in ("cnn_gru__film", "cnn_gru__film+ssl"):
        rpreds, rmeta = _retrieval_from(model, tr, va, te, invert)
        emitted.append((name + "+retrieval", rpreds, {**meta, **rmeta}))
    if name == "cnn_gru__film":
        zero = np.zeros_like(te["X"][:, :, 1:])
        dpreds = invert([te["X"][:, :, 0:1], zero, te["S"]])   # frozen test-time dyn ablation
        emitted.append((name + "+dynablate", dpreds, dict(meta)))
    return emitted


# ------------------------- checkpoint / manifest -------------------------
def _data_hash(b):
    h = hashlib.md5()
    for a in (b.X, b.y, b.S, b.diag):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update("|".join(b.pid).encode())
    return h.hexdigest()


def _code_hash():
    h = hashlib.md5()
    for f in ("fusion_models.py", "fusion_run.py", "fusion_ssl.py"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()


def _manifest(dh, emitted):
    return {"data_hash": dh, "code_hash": _code_hash(), "tf": tf.__version__,
            "emitted_configs": emitted, "n_seeds": N_SEEDS, "n_folds": N_FOLDS, "smoke": SMOKE,
            "epochs": EPOCHS, "patience": PATIENCE, "ssl_epochs": SSL_EPOCHS, "batch": BATCH,
            "lr": LR, "rf_trees": RF_TREES, "d_model": D_MODEL, "base_seed": BASE_SEED,
            "clean": CLEAN, "meal_caps": MEAL_CAPS, "mask_frac": MASK_FRAC, "mask_span": MASK_SPAN,
            "retr_k": RETR_K, "retr_alphas": list(RETR_ALPHAS), "retr_taus": list(RETR_TAUS)}


def _guard_and_resume(man):
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST))
        if old != man:
            raise SystemExit(f"[abort] {MANIFEST} differs from current settings "
                             f"(data/code/seeds/clean/config changed). Delete fusion_*_{TAG}.* to restart.")
    else:
        json.dump(man, open(MANIFEST, "w"), indent=2)
    done = set()
    if os.path.exists(RAW_CSV):
        r = pd.read_csv(RAW_CSV)
        rd = r.drop_duplicates(["model", "seed", "fold"], keep="last")   # a partial-group
        if len(rd) != len(r):                                           # resume can re-emit
            rd.to_csv(RAW_CSV, index=False)                             # deterministic dup rows
            print(f"deduped {len(r) - len(rd)} duplicate metric row(s)")
        done = set(map(tuple, rd[["model", "seed", "fold"]].values.tolist()))
    if os.path.exists(OOF_CSV):
        oof = pd.read_csv(OOF_CSV, dtype={"pid": str})
        n0 = len(oof)
        oof = oof.drop_duplicates(["model", "seed", "fold", "sample_idx"], keep="last")
        ck = oof[["model", "seed", "fold"]].apply(tuple, axis=1)
        orphans = set(ck) - done                                        # OOF without a metric marker
        oof = oof[~ck.isin(orphans)]
        if len(oof) != n0:
            oof.to_csv(OOF_CSV, index=False)
            print(f"reconciled OOF: dropped {n0 - len(oof)} orphan/duplicate row(s)")
    return done


def main():
    highres = DataBundle("highres"); binned = DataBundle("binned")
    assert_aligned(binned, highres)
    highres.S = clean_static(highres.S)                        # meal caps (CLEAN=capped) or raw
    base5 = highres.X[:, -5:, 0].mean(axis=1)                  # last-5-min persistence (mg/dL)
    libre = load_libre_cgm()
    bases = _base_configs()
    emitted_names = _emitted(bases)
    done = _guard_and_resume(_manifest(_data_hash(highres), emitted_names))
    splits = make_repeated_splits(highres.pid, highres.diag, n_seeds=N_SEEDS,
                                  n_folds=N_FOLDS, base_seed=BASE_SEED)
    cards_seen = set(pd.read_csv(CARD_CSV)["model"]) if os.path.exists(CARD_CSV) else set()
    print(f"SMOKE={SMOKE} CLEAN={CLEAN} seeds={N_SEEDS} folds={N_FOLDS} -> {len(splits)} splits "
          f"x {len(bases)} bases ({len(emitted_names)} emitted) | done: {len(done)} | "
          f"libre={'yes' if libre[0] is not None else 'no'}")

    for si, sp in enumerate(splits):
        fh = normalized_fold(highres, sp, with_aux=False)
        ssl_cache = {}
        for base in bases:
            if (base, sp["seed"], sp["fold"]) in done:         # primary marker => base atomic
                continue
            emitted = run_base(base, fh, sp["seed"], sp["fold"], ssl_cache, libre, base5)
            te = fh["test"]
            for name, preds, meta in emitted:                  # OOF first (all)
                oof = pd.DataFrame({
                    "model": name, "seed": sp["seed"], "fold": sp["fold"],
                    "sample_idx": te["idx"], "pid": te["pid"], "diag": te["diag"],
                    **{f"y_true_{h}": te["y"][:, i] for i, h in enumerate(HORIZONS)},
                    **{f"y_pred_{h}": preds[:, i] for i, h in enumerate(HORIZONS)}})
                _append_csv(OOF_CSV, oof)
            for name, preds, meta in emitted[1:] + emitted[:1]:  # metrics; base primary LAST
                m = compute_metrics_per_horizon(te["y"], preds)
                m.update({"model": name, "seed": sp["seed"], "fold": sp["fold"], **meta})
                _append_csv(RAW_CSV, pd.DataFrame([m]))
                if name not in cards_seen:
                    _append_csv(CARD_CSV, pd.DataFrame([{"model": name, "n_params": meta["n_params"]}]))
                    cards_seen.add(name)
                done.add((name, sp["seed"], sp["fold"]))
                print(f"  [{si+1}/{len(splits)}] {name:26s} R2_60={m['R2_60']:+.3f} "
                      f"NRMSE60={m['NRMSE_60']:.3f} params={meta['n_params']}")

    print(f"\nDone TAG={TAG}. Wrote {RAW_CSV}, {OOF_CSV}, {CARD_CSV}, {MANIFEST}."
          f"\nNext: python fusion_analyze.py")


if __name__ == "__main__":
    main()
