"""
fusion_ssl.py  -- self-supervised CGM-encoder pretraining (denoising autoencoder)
=================================================================================
Data-efficiency lever: pretrain the CGM encoder to RECONSTRUCT a masked/corrupted
CGM trajectory (self-supervised), then transfer its weights into the supervised
fusion model. Uses structure RF cannot exploit -- raw sequence shape.

Two entry points:
  * pretrain_encoder(cgm_seqs, encoder_cfg, T, ...)   [library, called by fusion_run.py]
      per-FOLD primary: TRAIN-participant Dexcom CGM windows only (no test leakage),
      normalized in the fold's own space. Returns a standalone encoder Model whose
      cgmenc_* weights transfer by name (fusion_models.transfer_encoder).
  * __main__  [pooled-corpus SENSITIVITY]
      pools Dexcom (913) + Libre (933) high-res windows -- a different device on the
      SAME 40 participants -- each device standardized separately, and pretrains the
      base encoder once. This is a LABELLED-pool sensitivity (not leakage-free per
      fold); reported as such. A hook is left for the full continuous CGMacros stream.

Masking = span corruption to the series mean (post-normalization ~0); the decoder
reconstructs the clean series (denoising AE). Simple, robust, no per-position loss
weighting. Keras-3-safe (corruption done in numpy, before fit).
=================================================================================
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import fusion_models as FM


def _corrupt(x, mask_frac, span, rng):
    """Zero out (~mean, post-norm) random spans covering ~mask_frac of each series.
    Start upper bound is T-span INCLUSIVE so the final timestep can be masked."""
    xc = x.copy()
    n, T, _ = x.shape
    span = min(span, T)
    n_span = max(1, int(round(mask_frac * T / span)))
    hi = T - span + 1                                  # inclusive of the last span
    for i in range(n):
        for _ in range(n_span):
            s = rng.integers(0, hi)
            xc[i, s:s + span, 0] = 0.0
    return xc


def load_libre_cgm():
    """Raw Libre CGM windows (n,T,1) + participant ids, for cross-device SSL. The runner
    filters by TRAIN participant ids and standardizes on that subset (leakage-free)."""
    import os
    HERE = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(HERE, "libre_raw_prediction_raw.npz"),
              os.path.join(HERE, "libre_raw_prediction_raw.npz")):
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            pid = np.asarray([str(q) for q in d["participant_id"]])
            return d["X"][:, :, 0:1].astype("float32"), pid
    return None, None


def _decoder(seq, T):
    """Map encoder sequence (T,d) back to (T,1) CGM reconstruction."""
    x = layers.Conv1D(16, 3, padding="same", activation="relu", name="ssl_dec1")(seq)
    return layers.Conv1D(1, 1, padding="same", name="ssl_dec_out")(x)


def pretrain_encoder(cgm_seqs, encoder_cfg, T, d=24, seed=0, epochs=40, batch=64,
                     mask_frac=0.15, span=5, verbose=0):
    """cgm_seqs: (n,T,1) NORMALIZED CGM (train participants only, fold space).
    Returns a standalone encoder Model (weights trained); transfer with
    fusion_models.transfer_encoder(full_model, this)."""
    tf.keras.utils.set_random_seed(int(seed) % (2**31))
    ci = layers.Input(shape=(T, 1), name="cgm_input")
    seq = FM._encode_cgm(ci, encoder_cfg.split("__")[0], d)     # cgmenc_* layers
    rec = _decoder(seq, T)
    ae = models.Model(ci, rec, name="ssl_ae")
    ae.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
    rng = np.random.default_rng(int(seed) % (2**31))
    Xc = _corrupt(cgm_seqs, mask_frac, span, rng)
    ae.fit(Xc, cgm_seqs, epochs=epochs, batch_size=batch, verbose=verbose)
    # rebuild a clean encoder-only model and copy the trained cgmenc_* weights across
    enc = FM.build_cgm_encoder(encoder_cfg, T, d)
    src = {l.name: l for l in ae.layers if l.name.startswith("cgmenc_")}
    for l in enc.layers:
        if l.name in src:
            l.set_weights(src[l.name].get_weights())
    return enc


# ------------------------- pooled-corpus sensitivity (CLI) -------------------------
def _load_cgm_windows(npz):
    d = np.load(npz, allow_pickle=True)
    x = d["X"][:, :, 0:1].astype("float32")                    # CGM channel
    mu, sd = float(x.mean()), float(x.std() + 1e-6)
    return (x - mu) / sd                                       # device-standardized


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    dex = os.path.join(HERE, "dexcom_raw_prediction_raw.npz")
    lib = os.path.join(HERE, "libre_raw_prediction_raw.npz")
    if not os.path.exists(lib):
        lib = os.path.join(HERE, "libre_raw_prediction_raw.npz")
    corpora = [_load_cgm_windows(dex)]
    if os.path.exists(lib):
        corpora.append(_load_cgm_windows(lib))
        print(f"pooled corpus: Dexcom + Libre = {sum(len(c) for c in corpora)} windows")
    else:
        print(f"pooled corpus: Dexcom only = {len(corpora[0])} windows (Libre npz not found)")
    X = np.concatenate(corpora, axis=0)
    T = X.shape[1]
    cfg = os.environ.get("SSL_CFG", "cnn_gru__film")
    enc = pretrain_encoder(X, cfg, T, seed=0, epochs=int(os.environ.get("SSL_EPOCHS", "40")), verbose=1)
    out = os.path.join(HERE, f"fusion_ssl_{cfg.split('__')[0]}_pooled.weights.h5")
    enc.save_weights(out)
    print(f"saved pooled-pretrained encoder weights -> {out}")
    print("NOTE: this is the labelled-POOL sensitivity; fusion_run.py does leakage-free "
          "per-fold pretraining on train participants only.")


if __name__ == "__main__":
    main()
