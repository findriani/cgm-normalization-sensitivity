"""
fusion_latency.py  -- CPU batch-1 forward-pass latency + param counts for the fusion grid
=========================================================================================
Standalone and CPU-ONLY (CUDA hidden before importing TF), like lightdl_latency.py:
latency depends only on architecture + input shape, not trained weights, so an untrained
model is the correct benchmark and there is no GPU/CPU cross-device issue.

Covers every emitted architecture (encoder sweep, fusion sweep, no_dyn, dl_current) so
the paper can report lightweight fusion cost alongside the light-DL models.

    python fusion_latency.py     # writes fusion_latency.csv
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # force CPU BEFORE importing TF

import time
import numpy as np
import pandas as pd
import tensorflow as tf

from lightdl_data import DataBundle
from lightdl_models import build_current_multimodal
import fusion_models as FM

N_WARM, N_TIME = 10, 100


def _time(model, inputs):
    for _ in range(N_WARM):
        model.predict(inputs, verbose=0)
    t0 = time.perf_counter()
    for _ in range(N_TIME):
        model.predict(inputs, verbose=0)
    return (time.perf_counter() - t0) / N_TIME * 1000.0     # ms/sample


def main():
    h = DataBundle("highres")
    T, n_dyn, n_static = h.T, h.n_dyn, h.n_static
    x1 = [np.zeros((1, T, 1), "float32"), np.zeros((1, T, n_dyn), "float32"),
          np.zeros((1, n_static), "float32")]

    rows = []
    configs = [f"{e}__concat" for e in FM.ENCODERS] + [f"cnn_gru__{f}" for f in FM.FUSIONS]
    for cfg in dict.fromkeys(configs):                       # dedup, keep order
        tf.keras.backend.clear_session()
        m = FM.build(cfg, T, n_dyn, n_static)
        rows.append({"model": cfg, "n_params": int(m.count_params()), "cpu_ms": round(_time(m, x1), 2)})
        print(f"{cfg:22s} params={m.count_params():>7,} cpu={rows[-1]['cpu_ms']:.2f} ms")

    tf.keras.backend.clear_session()
    m = FM.build("cnn_gru__film", T, n_dyn, n_static, dyn_mode="off")
    rows.append({"model": "cnn_gru__film+nodyn", "n_params": int(m.count_params()),
                 "cpu_ms": round(_time(m, x1), 2)})

    tf.keras.backend.clear_session()
    m = build_current_multimodal(T, n_dyn, n_static)
    rows.append({"model": "dl_current", "n_params": int(m.count_params()),
                 "cpu_ms": round(_time(m, x1), 2)})

    pd.DataFrame(rows).to_csv("fusion_latency.csv", index=False)
    print("\nSaved fusion_latency.csv")


if __name__ == "__main__":
    main()
