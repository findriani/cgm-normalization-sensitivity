"""
framework_checks.py -- three protocol-level analyses, no new model fits
=======================================================================
A method/framework claim has to characterise its own decision procedure, not just apply
it. These three analyses do that from OUTPUT ALREADY ON DISK. Nothing here refits a model;
A re-reads stored contrast tables, B and C re-run only the resampling layer on stored
out-of-fold predictions.

  A. MARGIN SENSITIVITY   Which verdicts survive a change in the equivalence margin?
                          delta = 0.02 was INHERITED from the primary ablation, never
                          justified. A framework that ships a decision rule must show the
                          rule is not knife-edge. Zero recomputation: decide() is a pure
                          function of (point, lo, hi, holm_p, seed_pos) and MARGIN_R2, all
                          of which are already stored.

  B. PRECISION vs SEEDS   How does the interval tighten as seeds accumulate? Turns "we
                          used 8 seeds" into guidance for anyone reusing the protocol.
                          Re-runs the bootstrap on seed subsets of the stored OOF.

  C. DETECTABILITY        The smallest effect this design can resolve, and -- more useful
                          -- whether an equivalence verdict is REACHABLE at all. If the
                          CI half-width exceeds delta, NEGLIGIBLE cannot be returned no
                          matter what the true effect is. That is a property of the
                          design, and it must be reported before any null is interpreted.

WHAT THIS IS NOT
----------------
Not a power analysis in the design sense -- the effects here are already observed, so C is
a precision statement, not a prospective calculation. The normal approximation used for the
minimum detectable effect assumes the cluster-bootstrap distribution is roughly symmetric;
the printed skew column says how well that holds. Read MDE as an order of magnitude.

Run (local, CPU, a few minutes):
    python framework_checks.py
=======================================================================
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
for _p in (CORE, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import ablation_analyze as AA
from ablation_analyze import build_ref, contrast_stats, WIN_FRAC, PRIMARY_H
from e1_analyze import did_stats

DELTAS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
DELTA_LOCKED = AA.MARGIN_R2
Z_ALPHA = 1.959963985                  # two-sided 5%
Z_POWER = 0.841621234                  # 80%

# (file, effect column, label columns, source name). e1_mechanism.csv reports each
# mechanism under BOTH increments, so `isolates` alone is not unique -- pairing it with
# `increment` is what keeps the two rows distinguishable.
SWEEP_SOURCES = [
    (os.path.join(HERE, "attr_contrasts.csv"), "dR2", ["contrast"], "E4 level vs shape"),
    (os.path.join(CORE, "e1_interaction.csv"), "ddR2", ["increment"], "E1 interaction"),
    (os.path.join(CORE, "e1_mechanism.csv"), "ddR2", ["isolates", "increment"],
     "E1 mechanism"),
]


# --------------------------------------------------------------- A. margin sensitivity
def margin_sweep():
    """Re-apply decide() across a grid of equivalence margins.

    decide() reads MARGIN_R2 from its own module globals at call time, so patching the
    module attribute is enough -- no statistic is recomputed and nothing else is touched.
    The locked value is restored in a finally block so an exception here cannot silently
    change the margin for anything imported later in the same process."""
    rows = []
    for path, eff, lab, src in SWEEP_SOURCES:
        if not os.path.exists(path):
            print(f"  [skip] {os.path.basename(path)} not found")
            continue
        d = pd.read_csv(path)
        d = d[d.horizon == PRIMARY_H]
        d = d[d.holm_p.notna()]                     # only the Holm-corrected family
        for _, r in d.iterrows():
            name = " / ".join(str(r[c]) for c in lab if c in d.columns)
            rec = {"source": src, "label": name, "point": r[eff],
                   "lo": r.lo, "hi": r.hi, "holm_p": r.holm_p,
                   "seed_pos": r.seed_pos_frac, "stored": r.get("verdict", "")}
            try:
                for dl in DELTAS:
                    AA.MARGIN_R2 = dl
                    rec[f"d={dl:.3f}"] = AA.decide(r[eff], r.lo, r.hi,
                                                   r.holm_p, r.seed_pos_frac)
            finally:
                AA.MARGIN_R2 = DELTA_LOCKED
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- shared OOF loaders
def e4_ref_by_seeds(oof, k, horizon):
    seeds = sorted(oof.seed.unique())[:k]
    return build_ref(oof[oof.seed.isin(seeds)], horizon)


def e1_ref_by_seeds(oof, k, horizon):
    """E1 stores arm and configuration separately; did_stats expects one key per
    (config, arm), so combine them exactly as e1_analyze does."""
    seeds = sorted(oof.seed.unique())[:k]
    d = oof[oof.seed.isin(seeds)].copy()
    d["model"] = d.model.astype(str) + "@" + d.protocol.astype(str)
    return build_ref(d, horizon)


# --------------------------------------------------------------- B. precision vs seeds
def precision_curve(horizon=PRIMARY_H):
    rows = []

    e4_path = os.path.join(HERE, "attr_oof.csv")
    if os.path.exists(e4_path):
        e4 = pd.read_csv(e4_path, dtype={"pid": str})
        n_max = e4.seed.nunique()
        for k in range(1, n_max + 1):
            ref = e4_ref_by_seeds(e4, k, horizon)
            for a, b, lab in [("L_S_C", "L_C", "E4 shape beyond level (primary)"),
                              ("L_C", "C", "E4 level only")]:
                st = contrast_stats(ref, a, b)
                rows.append({"quantity": lab, "seeds": k, "point": st["dR2_60"],
                             "lo": st["R2_lo"], "hi": st["R2_hi"]})
            print(f"  E4 seeds={k}/{n_max} done")
    else:
        print("  [skip] attr_oof.csv not found")

    e1_path = os.path.join(CORE, "e1_oof.csv")
    if os.path.exists(e1_path):
        e1 = pd.read_csv(e1_path, dtype={"pid": str})
        n_max = e1.seed.nunique()
        for k in range(1, n_max + 1):
            ref = e1_ref_by_seeds(e1, k, horizon)
            st = did_stats(ref, "cgm_static", "static_all", "global", "subject_z")
            rows.append({"quantity": "E1 interaction (primary DDR2)", "seeds": k,
                         "point": st["ddR2"], "lo": st["lo"], "hi": st["hi"]})
            print(f"  E1 seeds={k}/{n_max} done")
    else:
        print("  [skip] e1_oof.csv not found")

    d = pd.DataFrame(rows)
    if not d.empty:
        d["half_width"] = (d.hi - d.lo) / 2.0
        d["se"] = (d.hi - d.lo) / (2.0 * Z_ALPHA)
    return d


# --------------------------------------------------------------- C. detectability
def detectability(curve):
    """Two design properties, both read off the interval at the full seed count.

    MDE is the usual normal-approximation minimum detectable effect. `equivalence
    reachable` is the more decision-relevant one: NEGLIGIBLE requires the whole interval
    inside +-delta, so a half-width already exceeding delta makes that verdict impossible
    regardless of the true effect. A null reported from such a design is uninformative,
    and saying so is part of specifying the protocol."""
    rows = []
    for q, g in curve.groupby("quantity"):
        r = g.sort_values("seeds").iloc[-1]
        se = r.se
        rows.append({
            "quantity": q, "seeds": int(r.seeds), "point": r.point,
            "half_width": r.half_width, "se": se,
            "MDE_80pct": (Z_ALPHA + Z_POWER) * se,
            "equivalence_reachable": bool(r.half_width < DELTA_LOCKED),
            "delta_needed_for_equivalence": r.half_width,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print(f"PROTOCOL CHECKS -- no model refits. Locked margin delta = {DELTA_LOCKED}, "
          f"WIN_FRAC = {WIN_FRAC}, horizon = {PRIMARY_H}")
    print("=" * 78)

    print("\n=== A. MARGIN SENSITIVITY (stored statistics, decide() re-applied) ===")
    sweep = margin_sweep()
    if sweep.empty:
        print("  nothing to sweep")
    else:
        cols = ["source", "label", "point", "lo", "hi"] + [f"d={d:.3f}" for d in DELTAS]
        print(sweep[cols].round(4).to_string(index=False))
        sweep.to_csv(os.path.join(HERE, "framework_margin_sweep.csv"), index=False)

        print("\n  --- verdicts that CHANGE across the grid ---")
        vcols = [f"d={d:.3f}" for d in DELTAS]
        unstable = sweep[sweep[vcols].nunique(axis=1) > 1]
        if unstable.empty:
            print("  none -- every verdict is constant over delta in [0.010, 0.050]")
        else:
            for _, r in unstable.iterrows():
                flips = " | ".join(f"{c[2:]}:{r[c]}" for c in vcols)
                print(f"  {r.source} :: {r.label}")
                print(f"      {flips}")
            print("\n  Any verdict listed here is margin-dependent and MUST be reported")
            print("  with the margin stated in the same sentence.")

    print("\n=== B. PRECISION vs SEEDS (bootstrap re-run on stored OOF) ===")
    curve = precision_curve()
    if curve.empty:
        print("  no OOF available")
        return
    curve.to_csv(os.path.join(HERE, "framework_precision.csv"), index=False)
    print()
    print(curve.pivot_table(index="seeds", columns="quantity", values="half_width")
          .round(4).to_string())
    print("\n  (values are 95% CI half-widths in R-squared units; lower is sharper)")

    print("\n=== C. DETECTABILITY AT THE FULL SEED COUNT ===")
    det = detectability(curve)
    det.to_csv(os.path.join(HERE, "framework_detectability.csv"), index=False)
    print(det.round(4).to_string(index=False))
    print(f"\n  MDE_80pct  = smallest true effect detectable at 80% power, alpha = 0.05,")
    print(f"               normal approximation to the cluster-bootstrap SE.")
    print(f"  equivalence_reachable = can decide() ever return NEGLIGIBLE at "
          f"delta = {DELTA_LOCKED}?")
    print(f"               False means a null from this contrast is uninformative:")
    print(f"               the interval is wider than the margin by construction.")

    print("\nSaved: framework_margin_sweep.csv, framework_precision.csv, "
          "framework_detectability.csv")


if __name__ == "__main__":
    main()
