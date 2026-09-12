"""
window_analyze.py  -- did shortening the pre-meal window cost anything?
======================================================================
Runs AFTER window_run.py. Statistics are IMPORTED, not re-implemented: the same
event-weighted mean-single-run dR2_60, participant-cluster bootstrap, sign-flip
permutation, Holm, delta=0.02, symmetric win-fraction rule and precision decomposition
used by the attention study and the rest of the paper.

The question here is mostly an EQUIVALENCE question -- "does a 10-minute window lose
anything?" -- not a superiority one. That matters for reading the output:
  * NEGLIGIBLE  = the short window is as good as the full one, to within delta=0.02 R2_60.
                  THIS IS THE INFORMATIVE, PUBLISHABLE OUTCOME.
  * ADDS        = the short window is actually BETTER (what the occlusion sweep predicted).
  * INCONCLUSIVE= we cannot tell at this precision. attn_precision-style columns say
                  whether equivalence was reachable in principle.
Sign convention: dR2_60 > 0 means the SHORT window is better than w60.

Outputs: window_model_summary_<tag>.csv, window_verdict_<tag>.csv,
         window_precision_<tag>.csv, window_per_seed_r2_<tag>.csv
======================================================================
"""
import os
import json
import numpy as np
import pandas as pd

from ablation_analyze import (per_seed_pooled, build_ref, integrity, MARGIN_R2,
                              PRIMARY_H, HORIZONS, N_FOLDS)
from attn_analyze import evaluate, decide_paired          # identical rules, one source

TAG = os.environ.get("TAG", "raw")
PLANNED_N_SEEDS = 15
ARCHS = ["cnn_gru__concat", "sa__concat", "dl_current"]
CTRL = "w60"

# PRIMARY: both 10-minute parameterizations against the full window, per architecture.
# Both are fitted because it is genuinely open which resolution the signal lives at.
PRIMARY = [(f"{a}@{w}", f"{a}@{CTRL}", f"{a}: {w} vs full 60-min window")
           for w in ("w10_1min", "w10_5min") for a in ARCHS]
SECONDARY = (
    [(f"{a}@w5_1min", f"{a}@{CTRL}", f"{a}: last 5 min vs full window") for a in ARCHS]
    # NOTE: w10_1min (indices 50-59) and w10_5min (indices 49,54,59) have different
    # endpoints, so a direct resolution comparison is invalid. Removed.
    + [("rf_highres@w10_1min", "rf_highres@w60", "RF: does the non-DL model agree?"),
       ("cnn_gru__concat@w60", "persistence_5min", "sanity: DL beats the persistence floor")]
)


def validate_manifest(configs, seeds, nSamp, raw, oof):
    p = f"window_manifest_{TAG}.json"
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing -- cannot verify the run.")
    man = json.load(open(p))
    if man.get("smoke", True):
        raise SystemExit("[abort] manifest smoke=True -- not confirmatory.")
    if set(configs) != set(man["emitted_configs"]):
        raise SystemExit(f"[abort] configs differ from manifest: "
                         f"{sorted(set(configs) ^ set(man['emitted_configs']))}")
    if nSamp != man.get("n_samples"):
        raise SystemExit(f"[abort] {nSamp} samples but manifest says {man.get('n_samples')}.")
    if TAG == "raw":
        if man["n_seeds"] != PLANNED_N_SEEDS or sorted(seeds) != list(range(PLANNED_N_SEEDS)):
            raise SystemExit(f"[abort] seeds {sorted(seeds)}, expected 0..{PLANNED_N_SEEDS-1}.")
        if man.get("clean") != "raw":
            raise SystemExit("[abort] confirmatory analysis is CLEAN=raw only.")
    n, e = man["n_seeds"], len(man["emitted_configs"])
    assert len(raw) == e * n * man["n_folds"], f"raw rows {len(raw)} != {e*n*man['n_folds']}"
    assert len(oof) == e * n * man["n_samples"], f"oof rows {len(oof)} != {e*n*man['n_samples']}"
    return man


def main():
    oof_p = f"window_oof_{TAG}.csv"
    if not os.path.exists(oof_p):
        raise SystemExit(f"{oof_p} not found -- run window_run.py first.")
    oof = pd.read_csv(oof_p, dtype={"pid": str})
    raw = pd.read_csv(f"window_results_raw_{TAG}.csv")
    configs, seeds, nSamp = integrity(oof, raw, TAG)
    man = validate_manifest(configs, seeds, nSamp, raw, oof)
    print(f"[window/{TAG}] integrity+manifest OK: {len(configs)} configs x {len(seeds)} seeds "
          f"x {N_FOLDS} folds, {nSamp} samples / {man['n_participants']} participants")
    print("  windows: " + ", ".join(f"{k}(T={len(v)})" for k, v in man["windows"].items()))

    ps = pd.concat([per_seed_pooled(oof, h) for h in HORIZONS], ignore_index=True)
    ps.to_csv(f"window_per_seed_r2_{TAG}.csv", index=False)
    summ = (ps.groupby(["model", "horizon"]).agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                                                 RMSE_mean=("RMSE", "mean")).reset_index())
    summ.to_csv(f"window_model_summary_{TAG}.csv", index=False)

    ref = build_ref(oof, PRIMARY_H)
    piv = ps[ps.horizon == PRIMARY_H].pivot(index="seed", columns="model", values="R2")
    prim = evaluate(ref, piv, PRIMARY, "primary")
    sec = evaluate(ref, piv, SECONDARY, "secondary")
    verdict = pd.DataFrame(prim + sec)
    verdict.to_csv(f"window_verdict_{TAG}.csv", index=False)
    pcols = ["family", "increment", "with", "without", "dR2_60", "half_obs", "half_floor",
             "equiv_attainable", "seeds_needed_for_equiv", "verdict"]
    verdict[pcols].to_csv(f"window_precision_{TAG}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print(f"\n=== R2 by horizon (mean over {len(seeds)} seeds) ===")
    print(summ.pivot(index="model", columns="horizon", values="R2_mean")
              .sort_values(PRIMARY_H, ascending=False).round(3).to_string())

    # architecture x window grid at h=60 -- the table the paper would print
    r60 = summ[summ.horizon == PRIMARY_H].copy()
    r60 = r60[r60.model.str.contains("@")]
    r60["arch"] = r60.model.str.split("@").str[0]
    r60["window"] = r60.model.str.split("@").str[1]
    order = [w for w in ["w60", "w10_1min", "w10_5min", "w5_1min"] if w in set(r60.window)]
    print(f"\n=== R2_60 by architecture x window ===")
    print(r60.pivot(index="arch", columns="window", values="R2_mean")[order].round(3).to_string())

    cols = ["increment", "with", "without", "dR2_60", "R2_lo", "R2_hi", "dRMSE_60",
            "perm_p", "holm_p", "seed_pos_frac", "verdict"]
    for title, recs in [("PRIMARY -- 10-minute windows vs the full 60 (Holm within family)", prim),
                        ("SECONDARY / exploratory (own Holm)", sec)]:
        print(f"\n=== {title} ===")
        if recs:
            print(pd.DataFrame(recs)[cols].round(4).to_string(index=False))

    print("\n=== PRECISION ===")
    print(verdict[pcols].round(4).drop(columns=["family", "with", "without"]).to_string(index=False))
    print(f"\n  dR2_60 > 0 means the SHORT window is BETTER than the full 60-min one.")
    print(f"  NEGLIGIBLE (|dR2| + CI half-width < {MARGIN_R2}) is the informative outcome here:")
    print(f"  it means the short window loses nothing. INCONCLUSIVE means we cannot tell at")
    print(f"  this precision -- check equiv_attainable before reading anything into it.")
    print(f"\n  MULTIPLICITY NOTE: Holm correction controls family-wise error for the")
    print(f"  superiority (ADDS/HURTS) tests. It does NOT cover the multiple pointwise")
    print(f"  NEGLIGIBLE conclusions. Each equivalence interval is reported separately;")
    print(f"  do not claim family-wise equivalence across all window contrasts.")
    print(f"\nSaved: window_model_summary_{TAG}.csv, window_verdict_{TAG}.csv, "
          f"window_precision_{TAG}.csv, window_per_seed_r2_{TAG}.csv")


if __name__ == "__main__":
    main()
