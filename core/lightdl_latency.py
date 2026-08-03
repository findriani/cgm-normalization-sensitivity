"""
lightdl_latency.py  -- exact params + CPU single-sample latency per architecture
======================================================================
Standalone, CPU-only. Latency depends only on architecture + input shape (not on
trained weights), so we build each of the 7 DL architectures untrained and time a
batch-1 forward pass. Kept OUT of lightdl_run.py because forcing a CPU predict
there (weights on the Colab GPU) raises an XLA cross-device error.

Forces CPU by hiding the GPU BEFORE importing TensorFlow, so it gives a genuine
CPU latency figure even on a GPU Colab runtime. Writes lightdl_latency.csv
(model, n_params, cpu_latency_ms); lightdl_analyze.py merges it into the summary.

Run:  python lightdl_latency.py      (a few seconds)
======================================================================
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # must precede the tensorflow import
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from lightdl_data import DataBundle
from lightdl_models import (build_tabular_attention, build_highres_tcn,
                            build_current_multimodal)

REPS = 200
WARMUP = 20


def _time(model, inputs):
    x = [a for a in inputs] if isinstance(inputs, list) else inputs
    for _ in range(WARMUP):
        model(x, training=False)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        model(x, training=False)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def main():
    b, h = DataBundle("binned"), DataBundle("highres")
    Tb, Th, n_dyn, n_static = b.T, h.T, b.n_dyn, b.n_static
    n_feat = Tb + Tb * n_dyn + n_static

    def flat(n):
        return np.zeros((1, n), np.float32)

    def seq(T):
        return [np.zeros((1, T, 1), np.float32), np.zeros((1, T, n_dyn), np.float32),
                np.zeros((1, n_static), np.float32)]

    specs = [
        ("dl_current",   build_current_multimodal(Tb, n_dyn, n_static, aux=False), seq(Tb)),
        ("tabattn",      build_tabular_attention(n_feat, aux=False),               flat(n_feat)),
        ("tabattn_aux",  build_tabular_attention(n_feat, aux=True),                flat(n_feat)),
        ("tcn",          build_highres_tcn(Th, n_dyn, n_static, aux=False, gate=None),  seq(Th)),
        ("tcn_aux",      build_highres_tcn(Th, n_dyn, n_static, aux=True,  gate=None),  seq(Th)),
        ("tcn_gate",     build_highres_tcn(Th, n_dyn, n_static, aux=False, gate="glu"), seq(Th)),
        ("tcn_aux_gate", build_highres_tcn(Th, n_dyn, n_static, aux=True,  gate="glu"), seq(Th)),
    ]

    rows = []
    for name, model, x in specs:
        med, p90 = _time(model, x)
        rows.append({"model": name, "n_params": int(model.count_params()),
                     "cpu_latency_ms": round(med, 3), "cpu_latency_p90_ms": round(p90, 3)})
        print(f"  {name:14s} params={model.count_params():6d}  CPU latency={med:6.3f} ms (p90 {p90:.3f})")
        tf.keras.backend.clear_session()

    df = pd.DataFrame(rows)
    df.to_csv("lightdl_latency.csv", index=False)
    print("\nSaved lightdl_latency.csv  (device: CPU, batch=1, median of "
          f"{REPS} reps).  RF has no comparable per-sample DL latency; report tree count instead.")


if __name__ == "__main__":
    main()
