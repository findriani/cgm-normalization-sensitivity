"""
attn_latency.py  -- CPU batch-1 latency + param counts for the attention grid
=============================================================================
Standalone and CPU-ONLY (CUDA hidden before importing TF), like fusion_latency.py: latency
depends only on architecture + input shape, not trained weights, so an untrained model is
the correct benchmark and there is no GPU/CPU cross-device issue.

TWO latency columns, because they measure different things:
  * `predict_ms`  -- end-to-end model.predict(), which is what an application pays, but is
                     DOMINATED by Keras data-adapter / retracing overhead at batch 1. The
                     completed fusion run measured this flat at ~113-130 ms across every
                     architecture: that number says almost nothing about the models.
  * `forward_ms`  -- a tf.function-compiled forward pass on pre-built tensors. This is the
                     one that actually separates architectures.
Both as MEDIAN and p90 over N_TIME calls (means are dominated by outliers here).

Because latency barely separates these models, the "lightweight" claim rests on PARAMETER
COUNT, and `delta_vs_matched_control` reports the parameter cost of each attention locus
against its own control:
  +attnpool   an additive-attention scorer (d*D_ATT + 2*D_ATT params)
  cnn_sa/sa   a pre-LN transformer block (MHSA + FFN); positional encoding is sinusoidal
              and adds ZERO parameters by design
  coattn      a 2nd MultiHeadAttention + LayerNorm over the meal tokens -- and note its
              control `coattn_ctl` has an IDENTICAL count, which is the point of that pair

Environment (TF version, CPU, thread counts) is written to attn_latency_env.json so the
numbers are quotable.

    python attn_latency.py     # writes attn_latency.csv + attn_latency_env.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # force CPU BEFORE importing TF

import json
import time
import platform
import numpy as np
import pandas as pd
import tensorflow as tf

from lightdl_data import DataBundle
from lightdl_models import build_current_multimodal
import attn_models as AM

N_WARM, N_TIME = 10, 100

# (config, readout, reported name) -- every high-res architecture attn_run.py fits
GRID = [
    ("cnn_gru__concat",     "gap",  "cnn_gru__concat"),
    ("cnn_gru__concat",     "attn", "cnn_gru__concat+attnpool"),
    ("cnn__concat",         "gap",  "cnn__concat"),
    ("cnn__concat",         "attn", "cnn__concat+attnpool"),
    ("cnn_sa__concat",      "gap",  "cnn_sa__concat"),
    ("sa__concat",          "gap",  "sa__concat"),
    ("cnn_gru__xattn",      "gap",  "cnn_gru__xattn"),
    ("cnn_gru__coattn_ctl", "gap",  "cnn_gru__coattn_ctl"),
    ("cnn_gru__coattn",     "gap",  "cnn_gru__coattn"),
    ("cnn_sa__coattn",      "gap",  "cnn_sa__coattn"),
]

CONTROL_OF = {"cnn_gru__concat+attnpool": "cnn_gru__concat",
              "cnn__concat+attnpool": "cnn__concat",
              "cnn_sa__concat": "cnn__concat",
              "sa__concat": "cnn__concat",
              "cnn_gru__coattn": "cnn_gru__coattn_ctl",
              "cnn_gru__coattn_ctl": "cnn_gru__xattn",
              "cnn_sa__coattn": "cnn_sa__concat"}


def _stats(fn, inputs):
    for _ in range(N_WARM):
        fn(inputs)
    ts = []
    for _ in range(N_TIME):
        t0 = time.perf_counter(); fn(inputs); ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def _measure(model, x_np):
    x_tf = [tf.constant(a) for a in x_np]
    fwd = tf.function(lambda z: model(z, training=False))
    p_med, p_p90 = _stats(lambda z: model.predict(z, verbose=0), x_np)
    f_med, f_p90 = _stats(fwd, x_tf)
    return {"n_params": int(model.count_params()), "predict_ms": round(p_med, 2),
            "predict_p90_ms": round(p_p90, 2), "forward_ms": round(f_med, 3),
            "forward_p90_ms": round(f_p90, 3)}


def main():
    h, b = DataBundle("highres"), DataBundle("binned")
    x_hi = [np.zeros((1, h.T, 1), "float32"), np.zeros((1, h.T, h.n_dyn), "float32"),
            np.zeros((1, h.n_static), "float32")]
    x_bi = [np.zeros((1, b.T, 1), "float32"), np.zeros((1, b.T, b.n_dyn), "float32"),
            np.zeros((1, b.n_static), "float32")]

    rows = []
    for cfg, ro, name in GRID:
        tf.keras.backend.clear_session()
        m = AM.build(cfg, h.T, h.n_dyn, h.n_static, readout=ro)
        rows.append({"model": name, "input": f"{h.T}x1min", **_measure(m, x_hi)})
        print(f"{name:28s} params={rows[-1]['n_params']:>7,} "
              f"predict={rows[-1]['predict_ms']:>7.2f} ms  forward={rows[-1]['forward_ms']:>6.3f} ms")

    for name, bundle, xs in (("dl_current_highres", h, x_hi), ("dl_current_binned", b, x_bi)):
        tf.keras.backend.clear_session()
        m = build_current_multimodal(bundle.T, bundle.n_dyn, bundle.n_static)
        rows.append({"model": name, "input": f"{bundle.T}x{'1' if bundle.T == 60 else '5'}min",
                     **_measure(m, xs)})
        print(f"{name:28s} params={rows[-1]['n_params']:>7,} "
              f"predict={rows[-1]['predict_ms']:>7.2f} ms  forward={rows[-1]['forward_ms']:>6.3f} ms")

    df = pd.DataFrame(rows)
    base = dict(zip(df.model, df.n_params))
    df["matched_control"] = [CONTROL_OF.get(n, "") for n in df.model]
    df["delta_vs_matched_control"] = [base[n] - base[CONTROL_OF[n]] if n in CONTROL_OF else 0
                                      for n in df.model]
    df.to_csv("attn_latency.csv", index=False)

    env = {"tf": tf.__version__, "python": platform.python_version(),
           "platform": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(),
           "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
           "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
           "device": "CPU (CUDA_VISIBLE_DEVICES='')", "n_warm": N_WARM, "n_time": N_TIME,
           "batch_size": 1}
    json.dump(env, open("attn_latency_env.json", "w"), indent=2)
    print(f"\nSaved attn_latency.csv + attn_latency_env.json"
          f"\n  delta_vs_matched_control = parameter cost of that attention component alone."
          f"\n  Quote forward_ms, not predict_ms, when comparing architectures."
          f"\n  Env: TF {env['tf']}, {env['cpu_count']} CPUs, intra/inter threads "
          f"{env['intra_op_threads']}/{env['inter_op_threads']}.")


if __name__ == "__main__":
    main()
