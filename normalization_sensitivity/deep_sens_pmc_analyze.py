"""
deep_sens_pmc_analyze.py -- E2b: is the MECHANISM estimator-specific?
=====================================================================
E2 asked whether the ATTENUATION reproduces in the deep model. It does. E2b asks the
question E2 could not, because it never ran the arm:

    does removing absolute CGM level attenuate the CGM increment in the deep model
    even WITHOUT participant-record leakage?

    primary:  [cgm_static - static_all]_global - [cgm_static - static_all]_premeal_center

`premeal_center` centres each window on its own pre-meal mean, so it removes level while
using nothing but the observation window itself -- no held-out participant's record is
touched. A positive interaction here means leakage is NOT the cause, in the deep model,
and C1's mechanism half loses its Random-Forest scope limit.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
It does not re-derive statistics. The entire inferential core -- participant-cluster
bootstrap, sign-flip permutation, Holm, the margin, `decide()` -- is imported from
`deep_sens_analyze`, which imports it in turn from `ablation_analyze`. Only the COMPARATOR
ARM changes. Patching is done inside a try/finally that restores the module, following
`framework_checks.py`, so importing this file cannot leave `deep_sens_analyze` altered for
anything else in the same process.

VERDICT VOCABULARY IS NOT E2's
-------------------------------
E2's outcome strings are about the published transform. Here the comparator is a
leakage-free arm, so the same arithmetic carries a different meaning and gets its own
wording. In particular a NEGLIGIBLE result here would NOT mean "no attenuation" -- it would
mean level removal alone does not reproduce it in the deep model, which would make leakage
look necessary after all and would be a genuinely surprising, reportable result.

Run:  python deep_sens_pmc_analyze.py [--smoke] [--verify]

  --verify  re-runs the E2 contrast (global vs subject_z) through this same wrapper and
            checks it reproduces deep_sens_interaction.csv exactly. Costs no fits and
            proves the wrapper did not perturb the locked statistical core.
=====================================================================
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
for _p in (CORE, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import deep_sens_analyze as DA
from ablation_analyze import build_ref, holm, decide, PRIMARY_H, HORIZONS, WIN_FRAC

NEW_ARM = "premeal_center"
REFERENCE_ARM = "global"

VERDICT_TEXT = {
    "CONFIRMED POSITIVE": (
        "Removing absolute CGM level attenuates the CGM increment in the DEEP model "
        "WITHOUT any participant-record leakage. C1's mechanism half loses its "
        "Random-Forest scope limit: 'leakage is not necessary' now holds in both "
        "estimators, and the limitations sentence added on 2 August can be deleted."),
    "EQUIVALENT": (
        "POSITIVE EVIDENCE that level removal ALONE does not reproduce the attenuation in "
        "the deep model -- the interval lies wholly inside the margin. This does NOT "
        "confirm the mechanism; it points at leakage doing the work in this estimator, "
        "which would be a surprising and reportable divergence from the Random Forest. "
        "Do not bury it: C1's mechanism half would need to stay RF-scoped and the "
        "discrepancy would need its own paragraph."),
    "INCONCLUSIVE": (
        "NOTHING is established. This is a statement about precision, not about the "
        "world. C1's mechanism half KEEPS the Random-Forest scope limit in every claim, "
        "and the limitations sentence stays exactly as written."),
    "SUPPORTED OPPOSITE": (
        "CGM contributes MORE under pre-meal centering than under global normalization in "
        "the deep model. Check the integrity block above before reporting. This would "
        "contradict E1 and must not be written up as a minor discrepancy."),
}


def load(sfx, arm_b):
    """Combined OOF for the reference arm and one comparator arm.

    E2's file supplies `global` (and `subject_z`); E2b's supplies `premeal_center`. The
    reference-arm rows always come from E2 -- never duplicated -- so the baseline of both
    contrasts is literally the same predictions."""
    p_par = os.path.join(HERE, f"deep_sens_oof{sfx}.csv")
    if not os.path.exists(p_par):
        raise SystemExit(f"[abort] {p_par} not found -- E2 must exist first")
    d = pd.read_csv(p_par, dtype={"pid": str})

    if arm_b not in set(d.arm.unique()):
        p_new = os.path.join(HERE, f"deep_sens_pmc_oof{sfx}.csv")
        if not os.path.exists(p_new):
            raise SystemExit(f"[abort] {p_new} not found -- run deep_sens_pmc_run.py first")
        n = pd.read_csv(p_new, dtype={"pid": str})
        n = n[n.arm == arm_b]
        if n.empty:
            raise SystemExit(f"[abort] no {arm_b!r} rows in {p_new}")
        d = pd.concat([d[d.arm == REFERENCE_ARM], n], ignore_index=True)
    else:
        d = d[d.arm.isin([REFERENCE_ARM, arm_b])].copy()
    return d


def check_alias_exact(sfx):
    """The CGM-free cells were COPIED from E2, so they must be bit-identical. If they are
    not, someone refitted them and the context baseline is no longer shared."""
    p_new = os.path.join(HERE, f"deep_sens_pmc_oof{sfx}.csv")
    if not os.path.exists(p_new):
        return "skipped (no E2b file)"
    par = pd.read_csv(os.path.join(HERE, f"deep_sens_oof{sfx}.csv"), dtype={"pid": str})
    new = pd.read_csv(p_new, dtype={"pid": str})
    key = ["model", "seed", "fold", "sample_idx"]
    cols = [f"y_pred_{h}" for h in HORIZONS]
    worst = 0.0
    for cfg in DA.CGM_FREE:
        a = par[(par.arm == REFERENCE_ARM) & (par.model == cfg)].set_index(key)[cols]
        bset = new[(new.arm == NEW_ARM) & (new.model == cfg)].set_index(key)[cols]
        if bset.empty:
            continue
        j = a.join(bset, how="inner", lsuffix="_e2", rsuffix="_e2b")
        if len(j) != len(bset):
            raise SystemExit(f"[abort] {cfg}: {len(bset)} E2b rows but {len(j)} matched "
                             f"E2 rows -- the alias is not a copy of an existing cell.")
        for c in cols:
            worst = max(worst, float(np.abs(j[f"{c}_e2"] - j[f"{c}_e2b"]).max()))
    if worst != 0.0:
        raise SystemExit(f"[abort] aliased CGM-free predictions differ from E2 by "
                         f"{worst:.3e}. They must be EXACTLY equal -- they were copied, "
                         f"not refitted. Something rewrote them.")
    return f"CGM-free cells identical to E2 (max|diff| = {worst:.1e}) -- alias intact"


def run_contrast(d, arm_b):
    """E2's own estimators, with the comparator arm swapped. Restores the module after."""
    saved = (DA.PUBLISHED_ARM, DA.ARMS)
    try:
        DA.PUBLISHED_ARM = arm_b
        DA.ARMS = [REFERENCE_ARM, arm_b]
        tagged = d.copy()
        tagged["model"] = tagged.model + "@" + tagged.arm
        refs = {h: build_ref(tagged, h) for h in HORIZONS}
        inter = DA.interaction(refs)
        inc = DA.within_arm_increments(refs)
        mg = DA.paired_mgdl(refs)
        models = DA.per_arm_models(d)
    finally:
        DA.PUBLISHED_ARM, DA.ARMS = saved
    return inter, inc, mg, models


def verify(sfx):
    """Reproduce E2 through this wrapper. Any drift means the wrapper is not neutral."""
    ref = os.path.join(HERE, f"deep_sens_interaction{sfx}.csv")
    if not os.path.exists(ref):
        return "skipped (no stored E2 interaction)"
    d = load(sfx, "subject_z")
    inter, _, _, _ = run_contrast(d, "subject_z")
    old = pd.read_csv(ref)
    k = ["horizon", "increment"]
    j = old.merge(inter, on=k, suffixes=("_old", "_new"))
    if len(j) != len(old):
        return f"FAILED: {len(old)} stored rows, {len(j)} matched"
    worst = max(float(np.abs(j[f"{c}_old"] - j[f"{c}_new"]).max())
                for c in ("ddR2", "lo", "hi", "holm_p", "seed_pos_frac"))
    if worst > 1e-12:
        raise SystemExit(f"[abort] wrapper does not reproduce E2: max|diff| = {worst:.3e}. "
                         f"The statistical core was perturbed; do not trust E2b.")
    return f"reproduces deep_sens_interaction{sfx}.csv exactly (max|diff| = {worst:.1e})"


def classify(x):
    if not np.isfinite(x.holm_p):
        return "INCONCLUSIVE"
    return DA.classify(x.ddR2, x.lo, x.hi, x.holm_p, x.seed_pos_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="also reproduce the stored E2 contrast through this wrapper")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""

    print("=== E2b: deep premeal_center arm ===")
    if args.verify:
        print(f"  wrapper check: {verify(sfx)}")
    print(f"  alias check:   {check_alias_exact(sfx)}")

    d = load(sfx, NEW_ARM)
    print(f"  loaded {len(d)} OOF rows | arms {sorted(d.arm.unique())} "
          f"| configs {sorted(d.model.unique())} | seeds {len(d.seed.unique())}")
    print(f"  {DA.check_unique(d)}")

    inter, inc, mg, models = run_contrast(d, NEW_ARM)
    inter["outcome"] = [classify(x) for x in inter.itertuples()]

    for name, df in (("pmc_interaction", inter), ("pmc_increments", inc),
                     ("pmc_paired_mgdl", mg), ("pmc_models", models)):
        p = os.path.join(HERE, f"deep_sens_{name}{sfx}.csv")
        df.to_csv(p, index=False)
        print(f"  wrote {os.path.basename(p)}")

    print("\n=== PRIMARY (h = 60) ===")
    p = inter[(inter.horizon == PRIMARY_H)
              & (inter.increment == DA.PRIMARY_INCREMENT)]
    if p.empty:
        raise SystemExit("[abort] primary interaction missing")
    x = p.iloc[0]
    g = inc[(inc.horizon == PRIMARY_H)
            & (inc.increment == DA.PRIMARY_INCREMENT)].set_index("arm")
    for arm, tag in ((REFERENCE_ARM, "reference"), (NEW_ARM, "leakage-free")):
        if arm in g.index:
            r = g.loc[arm]
            print(f"  {tag:12s} ({arm:14s}) dR2 = {r.dR2:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]")
    print(f"  INTERACTION (paired)           DDR2 = {x.ddR2:+.4f} "
          f"[{x.lo:+.4f}, {x.hi:+.4f}]")
    print(f"                                 Holm p = {x.holm_p:.4g}   "
          f"seeds {x.seed_pos_frac:.0%} (WIN_FRAC {WIN_FRAC:.0%})   decide = {x.decide}")

    e1 = os.path.join(CORE, "e1_interaction.csv")
    if os.path.exists(e1):
        e = pd.read_csv(e1)
        e = e[(e.horizon == PRIMARY_H) & (e.increment == DA.PRIMARY_INCREMENT)
              & (e.protocol_1 == REFERENCE_ARM) & (e.protocol_2 == NEW_ARM)]
        if not e.empty:
            r = e.iloc[0]
            print(f"  E1 (Random Forest, same arms)  DDR2 = {r.ddR2:+.4f} "
                  f"[{r.lo:+.4f}, {r.hi:+.4f}]  -- the claim being transferred")

    print(f"\n  => {x.outcome}. {VERDICT_TEXT.get(x.outcome, '(no verdict)')}")
    print("\n  Scope: this transfers the MECHANISM across estimators, on one cohort. "
          "It is not cross-cohort evidence -- Shanghai remains the only thing that is.")


if __name__ == "__main__":
    main()
