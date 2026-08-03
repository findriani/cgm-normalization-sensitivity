"""
lightdl_run.py  -- entry point: repeated-CV FITTING of RF vs lightweight DL
======================================================================
This script ONLY fits models and records raw outputs. It does NOT run the
hypothesis tests -- that is lightdl_analyze.py, so that the statistics operate
on saved out-of-fold (OOF) predictions with participant-level clustering
(the fix for the pseudoreplication problem the review flagged).

For each (model, seed, fold) it writes, incrementally (checkpoint + resume):
  lightdl_results_raw.csv  - per-(model,seed,fold,horizon) metrics (descriptive)
  lightdl_oof.csv          - per-SAMPLE OOF predictions (model,seed,fold,pid,diag,
                             sample_idx, y_true_h, y_pred_h)  <-- used for inference
  lightdl_model_cards.csv  - exact trainable params + CPU predict latency per model

Design points from the review, implemented here:
  * DETERMINISTIC per-(seed,fold,model) seeding (order-independent, recorded).
  * TRAIN-FOLD TARGET STANDARDIZATION (fit on train y; inverted before metrics),
    so DL optimises on unit-scale targets while metrics stay in mg/dL.
  * EARLY STOPPING on the MAIN-TRAJECTORY val loss (sum of the 3 horizon losses;
    aux heads excluded from model selection).
  * OOF predictions saved for participant-cluster inference downstream.
  * CHECKPOINT + RESUME: re-running skips completed (model,seed,fold) cells.

NOTHING RUNS ON IMPORT. In Colab:
    !pip -q install scipy scikit-learn
    SMOKE=1 python lightdl_run.py     # quick end-to-end check (1 seed, 5 epochs)
    python lightdl_run.py             # full run
    python lightdl_analyze.py         # THEN the statistics / decision rule

Upload alongside: dexcom_binned_prediction_raw.npz, dexcom_raw_prediction_raw.npz,
foldwise_normalizer.py, common.py, lightdl_data.py, lightdl_models.py.
======================================================================
"""
import os
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.ensemble import RandomForestRegressor

from lightdl_data import (DataBundle, assert_aligned, highres_baseline,
                          make_repeated_splits, normalized_fold, flatten)
from lightdl_models import (build_tabular_attention, build_highres_tcn,
                            build_current_multimodal)
from common import compute_metrics_per_horizon, HORIZONS

# ------------------------- knobs -------------------------
SMOKE = os.environ.get("SMOKE", "0") == "1"
N_SEEDS = 1 if SMOKE else int(os.environ.get("N_SEEDS", "5"))   # final run: 10
N_FOLDS = 5
EPOCHS = 5 if SMOKE else 100
PATIENCE = 2 if SMOKE else 10
BATCH = 32
RF_TREES = 300
BASE_SEED = 42

RAW_CSV = "lightdl_results_raw.csv"
OOF_CSV = "lightdl_oof.csv"
CARD_CSV = "lightdl_model_cards.csv"


# ------------------------- experiment matrix (locked) -------------------------
def _experiments(dims):
    return [
        dict(name="rf_binned",  type="rf", feature="binned"),
        dict(name="rf_highres", type="rf", feature="highres"),
        dict(name="dl_current", type="dl", feature="binned", mode="seq", aux=False,
             build=lambda aux: build_current_multimodal(dims["Tb"], dims["n_dyn"], dims["n_static"], aux=aux)),
        dict(name="tabattn",     type="dl", feature="binned", mode="flat", aux=False,
             build=lambda aux: build_tabular_attention(dims["n_feat_binned"], aux=aux)),
        dict(name="tabattn_aux", type="dl", feature="binned", mode="flat", aux=True,
             build=lambda aux: build_tabular_attention(dims["n_feat_binned"], aux=aux)),
        dict(name="tcn",          type="dl", feature="highres", mode="seq", aux=False,
             build=lambda aux: build_highres_tcn(dims["Th"], dims["n_dyn"], dims["n_static"], aux=aux, gate=None)),
        dict(name="tcn_aux",      type="dl", feature="highres", mode="seq", aux=True,
             build=lambda aux: build_highres_tcn(dims["Th"], dims["n_dyn"], dims["n_static"], aux=aux, gate=None)),
        dict(name="tcn_gate",     type="dl", feature="highres", mode="seq", aux=False,
             build=lambda aux: build_highres_tcn(dims["Th"], dims["n_dyn"], dims["n_static"], aux=aux, gate="glu")),
        dict(name="tcn_aux_gate", type="dl", feature="highres", mode="seq", aux=True,
             build=lambda aux: build_highres_tcn(dims["Th"], dims["n_dyn"], dims["n_static"], aux=aux, gate="glu")),
    ]


def _seed_for(seed, fold, name):
    """Stable, order-independent seed for one fit (recorded in outputs)."""
    h = hashlib.md5(f"{BASE_SEED}|{seed}|{fold}|{name}".encode()).hexdigest()
    return int(h[:8], 16)


def _seq_inputs(d):
    return [d["X"][:, :, 0:1], d["X"][:, :, 1:], d["S"]]


class MainTrajectoryEarlyStopping(callbacks.Callback):
    """Early stopping on the SUM of the three main-horizon val losses (aux heads
    excluded), with best-weight restore. Falls back to val_loss if per-head keys
    are absent (e.g. single-output edge cases)."""
    def __init__(self, patience):
        super().__init__(); self.patience = patience
        self._keys = [f"val_out_{h}_loss" for h in HORIZONS]

    def on_train_begin(self, logs=None):
        self.best = np.inf; self.wait = 0; self.best_w = None; self.best_epoch = 0

    def _score(self, logs):
        if all(k in logs for k in self._keys):
            return float(sum(logs[k] for k in self._keys))
        return float(logs.get("val_loss", np.inf))

    def on_epoch_end(self, epoch, logs=None):
        s = self._score(logs or {})
        if s < self.best - 1e-9:
            self.best = s; self.wait = 0
            self.best_w = self.model.get_weights(); self.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_w is not None:
            self.model.set_weights(self.best_w)


def run_one(spec, fold, seed, fold_id):
    """Fit one model on one fold. Returns (preds (n_test,3) mg/dL, meta dict)."""
    tr, va, te = fold["train"], fold["val"], fold["test"]
    meta = {}

    if spec["type"] == "rf":
        Ftr, Fte = flatten(tr["X"], tr["S"]), flatten(te["X"], te["S"])
        preds = np.zeros((len(te["y"]), 3), dtype="float32")
        n_params = 0
        for i in range(3):
            rf = RandomForestRegressor(n_estimators=RF_TREES,
                                       random_state=_seed_for(seed, fold_id, spec["name"]) % (2**31),
                                       n_jobs=-1)
            rf.fit(Ftr, tr["y"][:, i]); preds[:, i] = rf.predict(Fte)
            n_params += sum(t.tree_.node_count for t in rf.estimators_)
        return preds, {"n_params": int(n_params)}

    # ---- deterministic, order-independent seeding ----
    tf.keras.utils.set_random_seed(_seed_for(seed, fold_id, spec["name"]))
    tf.keras.backend.clear_session()

    aux = spec["aux"]
    if spec["mode"] == "flat":
        Xtr, Xva, Xte = flatten(tr["X"], tr["S"]), flatten(va["X"], va["S"]), flatten(te["X"], te["S"])
    else:
        Xtr, Xva, Xte = _seq_inputs(tr), _seq_inputs(va), _seq_inputs(te)

    # ---- train-fold target standardization (metrics still in mg/dL) ----
    mu = tr["y"].mean(axis=0); sd = tr["y"].std(axis=0); sd[sd < 1e-6] = 1.0
    ytr = [((tr["y"][:, i] - mu[i]) / sd[i]) for i in range(3)]
    yva = [((va["y"][:, i] - mu[i]) / sd[i]) for i in range(3)]
    if aux:
        mua = tr["yaux"].mean(axis=0); sda = tr["yaux"].std(axis=0); sda[sda < 1e-6] = 1.0
        ytr += [((tr["yaux"][:, i] - mua[i]) / sda[i]) for i in range(3)]
        yva += [((va["yaux"][:, i] - mua[i]) / sda[i]) for i in range(3)]

    model = spec["build"](aux)
    meta["n_params"] = int(model.count_params())
    cb = MainTrajectoryEarlyStopping(PATIENCE)
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
              batch_size=BATCH, callbacks=[cb], verbose=0)
    out = model.predict(Xte, verbose=0)
    if not isinstance(out, list):
        out = [out]
    # first 3 heads = standardized main horizons -> invert to mg/dL
    preds = np.stack([np.asarray(out[i]).squeeze(-1) * sd[i] + mu[i] for i in range(3)], axis=-1)
    tf.keras.backend.clear_session()
    return preds.astype("float32"), meta


# NOTE: CPU predict latency is measured separately by lightdl_latency.py (CPU-only),
# because forcing a CPU predict here while the trained weights live on the Colab GPU
# raises an XLA cross-device error. Latency depends only on architecture + input
# shape, not on trained weights, so a clean standalone benchmark is the right place.


def _append_csv(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _done_set():
    if not os.path.exists(RAW_CSV):
        return set()
    d = pd.read_csv(RAW_CSV)
    return set(zip(d["model"], d["seed"], d["fold"]))


def main():
    binned, highres = DataBundle("binned"), DataBundle("highres")
    assert_aligned(binned, highres)
    baseline = highres_baseline()                       # true last pre-meal obs (mg/dL)
    dims = {"Tb": binned.T, "Th": highres.T, "n_dyn": binned.n_dyn, "n_static": binned.n_static,
            "n_feat_binned": binned.T + binned.T * binned.n_dyn + binned.n_static}
    splits = make_repeated_splits(binned.pid, binned.diag, n_seeds=N_SEEDS,
                                  n_folds=N_FOLDS, base_seed=BASE_SEED)
    experiments = _experiments(dims)
    done = _done_set()
    cards_seen = set(pd.read_csv(CARD_CSV)["model"]) if os.path.exists(CARD_CSV) else set()
    print(f"SMOKE={SMOKE} seeds={N_SEEDS} folds={N_FOLDS} -> {len(splits)} splits x "
          f"{len(experiments)} models | already done: {len(done)} cells")

    for si, sp in enumerate(splits):
        fb = normalized_fold(binned, sp, baseline=baseline, with_aux=True)
        fh = normalized_fold(highres, sp, baseline=baseline, with_aux=True)
        for spec in experiments:
            key = (spec["name"], sp["seed"], sp["fold"])
            if key in done:
                continue
            fold = fb if spec["feature"] == "binned" else fh
            preds, meta = run_one(spec, fold, sp["seed"], sp["fold"])
            te = fold["test"]
            m = compute_metrics_per_horizon(te["y"], preds)
            m.update({"model": spec["name"], "seed": sp["seed"], "fold": sp["fold"],
                      "n_params": meta["n_params"], "seed_used": _seed_for(sp["seed"], sp["fold"], spec["name"])})
            _append_csv(RAW_CSV, pd.DataFrame([m]))

            oof = pd.DataFrame({
                "model": spec["name"], "seed": sp["seed"], "fold": sp["fold"],
                "sample_idx": te["idx"], "pid": te["pid"], "diag": te["diag"],
                **{f"y_true_{h}": te["y"][:, i] for i, h in enumerate(HORIZONS)},
                **{f"y_pred_{h}": preds[:, i] for i, h in enumerate(HORIZONS)},
            })
            _append_csv(OOF_CSV, oof)

            if spec["name"] not in cards_seen:
                _append_csv(CARD_CSV, pd.DataFrame([{"model": spec["name"], "n_params": meta["n_params"]}]))
                cards_seen.add(spec["name"])

            done.add(key)
            print(f"  [{si+1}/{len(splits)}] {spec['name']:14s} R2_60={m['R2_60']:+.3f} "
                  f"NRMSE60={m['NRMSE_60']:.3f} params={meta['n_params']}")

    print(f"\nDone. Wrote {RAW_CSV}, {OOF_CSV}, {CARD_CSV}."
          f"\nNext: python lightdl_analyze.py  (participant-cluster inference + decision rule)")


if __name__ == "__main__":
    main()
