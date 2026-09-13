"""
sens_meal_exclusion_analyze.py -- compare exclusion with capping and uncapped runs
==================================================================================
Reads sens_meal_exclusion_oof.csv and computes DDR2 (global vs subject_z) using
the same statistical procedure as e1_analyze.py: per-seed squared errors,
participant-cluster bootstrap, participant sign-flip permutation.

Prints the result alongside the primary capped result from e1_interaction.csv
for direct comparison.

    python sens_meal_exclusion_analyze.py
    python sens_meal_exclusion_analyze.py --smoke

Output: sens_meal_exclusion_summary.csv
==================================================================================
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ablation_analyze import build_ref
from e1_analyze import did_stats, REFERENCE_ARM, PUBLISHED_ARM

HORIZON = 60
INCREMENTS = [
    ("cgm_static", "static_all", "CGM adds beyond context"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    path = os.path.join(HERE, f"sens_meal_exclusion_oof{sfx}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[abort] {path} not found -- run "
                         f"sens_meal_exclusion_run.py first")

    oof = pd.read_csv(path, dtype={"pid": str})
    n_meals = len(oof[(oof.model == "cgm_static") & (oof.protocol == "global")
                      & (oof.seed == oof.seed.min())].sample_idx.unique())
    print(f"Meal-exclusion sensitivity at {HORIZON} min")
    print(f"  meals in exclusion run: {n_meals}")
    print(f"  seeds: {sorted(oof.seed.unique())}")
    print()

    # Tag models with protocol for build_ref
    tagged = oof.copy()
    tagged["model"] = tagged.model + "@" + tagged.protocol
    ref = build_ref(tagged, HORIZON)

    rows = []
    for a, b, label in INCREMENTS:
        st = did_stats(ref, a, b, REFERENCE_ARM, PUBLISHED_ARM)
        rows.append({"setting": "excluded", "increment": label,
                     "DDR2": st["ddR2"], "lo": st["lo"], "hi": st["hi"],
                     "perm_p": st["perm_p"],
                     "seed_pos_frac": st["seed_pos_frac"],
                     "n_meals": n_meals})
        print(f"  EXCLUDED  DDR2={st['ddR2']:+.4f} "
              f"[{st['lo']:+.4f}, {st['hi']:+.4f}]  "
              f"p={st['perm_p']:.4f}  seeds={st['seed_pos_frac']:.0%}")

    # Compare with capped result if available
    e1_path = os.path.join(HERE, f"e1_interaction{sfx}.csv")
    if os.path.exists(e1_path):
        e1 = pd.read_csv(e1_path)
        capped = e1[(e1.horizon == HORIZON)
                     & (e1.increment == "CGM adds beyond context")
                     & (e1.protocol_1 == REFERENCE_ARM)
                     & (e1.protocol_2 == PUBLISHED_ARM)]
        if not capped.empty:
            c = capped.iloc[0]
            rows.append({"setting": "capped", "increment": "CGM adds beyond context",
                         "DDR2": c.ddR2, "lo": c.lo, "hi": c.hi,
                         "perm_p": c.perm_p,
                         "seed_pos_frac": c.seed_pos_frac,
                         "n_meals": 913})
            print(f"  CAPPED    DDR2={c.ddR2:+.4f} [{c.lo:+.4f}, {c.hi:+.4f}]  "
                  f"p={c.perm_p:.4f}  seeds={c.seed_pos_frac:.0%}  (from e1)")
            diff = abs(rows[0]["DDR2"] - c.ddR2)
            print(f"\n  |point difference| = {diff:.4f}")
            print("  Compare the CIs to assess whether this difference is "
                  "meaningful relative to the sampling uncertainty.")

    out = os.path.join(HERE, f"sens_meal_exclusion_summary{sfx}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
