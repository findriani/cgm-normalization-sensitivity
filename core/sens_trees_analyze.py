"""
sens_trees_analyze.py -- does tree count change the normalization conclusion?
=============================================================================
Reads sens_trees_oof.csv and computes DDR2 (global vs subject_z, CGM increment)
at each tree count using the same statistical procedure as e1_analyze.py:
per-seed squared errors, participant-cluster bootstrap, participant sign-flip
permutation.

    python sens_trees_analyze.py
    python sens_trees_analyze.py --smoke

Output: sens_trees_summary.csv
=============================================================================
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
WITH_CGM = "cgm_static"
WITHOUT_CGM = "static_all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    path = os.path.join(HERE, f"sens_trees_oof{sfx}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[abort] {path} not found -- run sens_trees_run.py first")

    oof = pd.read_csv(path, dtype={"pid": str})
    tree_counts = sorted(oof.trees.unique())
    print(f"Tree-count sensitivity at {HORIZON} min")
    print(f"  trees: {tree_counts}")
    print(f"  seeds: {sorted(oof.seed.unique())}")
    print(f"  comparison: {REFERENCE_ARM} vs {PUBLISHED_ARM}, "
          f"increment = {WITH_CGM} - {WITHOUT_CGM}")
    print()

    rows = []
    for n_trees in tree_counts:
        sub = oof[oof.trees == n_trees].copy()
        # Tag models with protocol so build_ref sees distinct keys
        sub["model"] = sub["model"].astype(str) + "@" + sub["protocol"].astype(str)
        ref = build_ref(sub, HORIZON)

        st = did_stats(ref, WITH_CGM, WITHOUT_CGM,
                       REFERENCE_ARM, PUBLISHED_ARM)
        rows.append({"trees": n_trees, "DDR2": st["ddR2"],
                     "lo": st["lo"], "hi": st["hi"],
                     "perm_p": st["perm_p"],
                     "seed_pos_frac": st["seed_pos_frac"]})
        print(f"  trees={n_trees:5d}  DDR2={st['ddR2']:+.4f} "
              f"[{st['lo']:+.4f}, {st['hi']:+.4f}]  "
              f"p={st['perm_p']:.4f}  seeds={st['seed_pos_frac']:.0%}")

    out = os.path.join(HERE, f"sens_trees_summary{sfx}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Compare with the primary E1 result if available
    e1_path = os.path.join(HERE, f"e1_interaction{sfx}.csv")
    if os.path.exists(e1_path):
        e1 = pd.read_csv(e1_path)
        ref_row = e1[(e1.horizon == HORIZON)
                      & (e1.increment == "CGM adds beyond context")
                      & (e1.protocol_1 == REFERENCE_ARM)
                      & (e1.protocol_2 == PUBLISHED_ARM)]
        if not ref_row.empty:
            c = ref_row.iloc[0]
            print(f"\n  E1 reference (300 trees, capped): "
                  f"DDR2={c.ddR2:+.4f} [{c.lo:+.4f}, {c.hi:+.4f}]")

    # Descriptive summary -- no equivalence claim
    ddr2_vals = [r["DDR2"] for r in rows]
    spread = max(ddr2_vals) - min(ddr2_vals)
    print(f"\n  DDR2 range across tree counts: "
          f"[{min(ddr2_vals):+.4f}, {max(ddr2_vals):+.4f}], spread = {spread:.4f}")
    print("  Inspect the CIs above to assess whether the spread is "
          "meaningful relative to the sampling uncertainty.")


if __name__ == "__main__":
    main()
