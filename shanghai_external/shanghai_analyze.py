"""
shanghai_analyze.py -- does C1 replicate on a second cohort?
=============================================================
    primary:  [cgm_static - static_all]_global - [cgm_static - static_all]_subject_z

The same estimand E1 tested on CGMacros, on ShanghaiT2DM.

NOTHING STATISTICAL IS RE-DERIVED HERE
---------------------------------------
`did_stats`, `interaction`, `within_arm_increments`, `per_arm_models` and the integrity
checks are imported from `e1_analyze`, which imports the participant-cluster bootstrap,
sign-flip permutation, Holm and `decide()` from `ablation_analyze`. Only two module
constants are patched, inside a try/finally that restores them (the idiom from
`framework_checks.py` and `deep_sens_pmc_analyze.py`):

    CGM_FREE   "persistence_5min" -> "persistence"   (Shanghai's 15-min grid)
    ABL_OOF    disabled                              (no CGMacros ablation to compare to)

`--verify` re-runs E1's own CGMacros contrast through this wrapper and checks it reproduces
`e1_interaction.csv` exactly. If that passes, any difference in the Shanghai numbers is the
cohort, not the code.

WHAT THIS COHORT CAN AND CANNOT REPLICATE -- READ BEFORE WRITING ANY OF IT UP
------------------------------------------------------------------------------
C1 has two halves. Shanghai carries one of them.

  CARRIES:  CGM's measured contribution collapses when absolute level is removed.
  CANNOT:   context inflates to compensate.

Shanghai has no meal-content modality (the dietary entries are un-coded free text), so its
context branch is person + time only and predicts postprandial glucose barely at all --
`static_all` R2_60 ran NEGATIVE in most smoke folds against +0.353 on CGMacros. There is
almost no context increment for normalization to inflate, so the mirror effect is not
testable here and its absence is NOT evidence against it.

The interaction remains a fair test: context is held identical across arms within this
cohort. But the per-arm increments are not comparable to CGMacros'. Compare interactions to
interactions; never place the two cohorts' CGM increments side by side.

Run:  python shanghai_analyze.py [--smoke] [--verify]
=============================================================
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
for _p in (CORE, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import e1_analyze as E1A
from ablation_analyze import build_ref, decide, PRIMARY_H, HORIZONS, WIN_FRAC, MARGIN_R2

REFERENCE_ARM, PUBLISHED_ARM = "global", "subject_z"
CGM_FREE_SHANGHAI = ["static_all", "persistence"]

VERDICT_TEXT = {
    "ADDS": (
        "C1 REPLICATES ACROSS COHORTS. Per-subject CGM z-scoring attenuates the measured "
        "CGM contribution on an independent cohort, with a different device, a different "
        "population and a different context panel. This is the cross-cohort evidence the "
        "plan says only Shanghai can supply -- but only for the LEVEL-COLLAPSE half of C1 "
        "(see the docstring). State the scope explicitly."),
    "NEGLIGIBLE": (
        "POSITIVE EVIDENCE OF NON-REPLICATION: the whole interval lies inside the margin. "
        "The attenuation is CGMacros-specific. This is a real and publishable finding, not "
        "a failure -- it would mean the effect depends on cohort characteristics, and the "
        "paper must say so rather than quietly dropping the external validation."),
    "INCONCLUSIVE": (
        "NOTHING is established either way. A statement about precision, not the world. "
        "C1 stays CGMacros-only and Shanghai is reported as an underpowered attempt, or "
        "not reported at all -- but it must NOT be described as a failed replication."),
    "HURTS": (
        "The increment is LARGER under the published transform on this cohort -- the "
        "OPPOSITE direction to CGMacros. Check the integrity block above before writing "
        "anything. If it holds, it is a serious result that needs its own section."),
}


def check_invariance_storage_aware(ref):
    """CGM-free configurations must be identical across arms -- to STORAGE precision.

    E1's version tests against a fixed 1e-6. That is tighter than the precision the
    predictions are stored at, and Shanghai trips it: `static_all` came back differing on
    2 of 3050 rows by 1.0e-05, in two arms only.

    Diagnosed rather than waved away. The two values are 149.7203 and 149.72029 -- ADJACENT
    float32 numbers; the ULP at that magnitude is 1.526e-05. `run_config` accumulates
    predictions in a float32 array, and sklearn's RandomForest `predict` sums over trees
    with threaded accumulation (`n_jobs=-1`), so float64 addition order varies by ~5.7e-14
    between identical refits (measured, 3 refits, same seed, same features). Almost always
    that rounds to the same float32; twice in 3050 it straddled a rounding boundary and
    flipped one bit.

    So the tolerance is derived from the storage type at the observed magnitude instead of
    being a constant. Anything larger than a couple of float32 ULPs is a REAL difference and
    still fails loudly -- a genuine static-feature difference between arms would show up
    orders of magnitude above this.

    CGMacros passes at 1e-6 only because it has 913 meals against Shanghai's 3050; at that
    rate it expects well under one flip. The threshold there is lucky, not sound."""
    out = []
    for m in CGM_FREE_SHANGHAI:
        keys = [f"{m}@{p}" for p in E1A.PROTOCOLS if f"{m}@{p}" in ref["preds"]]
        if len(keys) < 2:
            continue
        base, worst, mag = keys[0], 0.0, 0.0
        for k in keys[1:]:
            for s in ref["seeds"]:
                a, b = ref["preds"][k][s], ref["preds"][base][s]
                worst = max(worst, float(np.abs(a - b).max()))
                mag = max(mag, float(np.abs(a).max()))
        tol = 2.0 * float(np.spacing(np.float32(mag)))
        ok = worst <= tol
        note = ("bit-identical" if worst == 0.0 else
                f"within {worst / float(np.spacing(np.float32(mag))):.1f} float32 ULP "
                f"-- storage rounding, not a model difference")
        out.append(f"    {m:16s} max|diff| = {worst:.3e}  (tol {tol:.3e} = 2 ULP at "
                   f"|{mag:.1f}|)  " + ("OK, " + note if ok else "*** NOT INVARIANT ***"))
        if not ok:
            raise SystemExit(f"[abort] {m} differs across arms by {worst:.3e}, above "
                             f"float32 storage noise ({tol:.3e}). Something other than "
                             f"the CGM transform varies -- the experiment is not "
                             f"controlled and the numbers must not be reported.")
    return "\n".join(out) if out else "    (no CGM-free configs found)"


def _patched():
    """Swap the two CGMacros-specific constants; restore whatever happens."""
    saved = (E1A.CGM_FREE, E1A.ABL_OOF)
    E1A.CGM_FREE = CGM_FREE_SHANGHAI
    E1A.ABL_OOF = "__shanghai_has_no_reference_ablation__"
    return saved


def _restore(saved):
    E1A.CGM_FREE, E1A.ABL_OOF = saved


def verify():
    """Reproduce E1's stored CGMacros interaction through this wrapper. Zero fits."""
    oof = os.path.join(CORE, "e1_oof.csv")
    ref = os.path.join(CORE, "e1_interaction.csv")
    if not (os.path.exists(oof) and os.path.exists(ref)):
        return "skipped (E1 files not found)"
    e1 = pd.read_csv(oof, dtype={"pid": str})
    tagged = e1.copy()
    tagged["model"] = tagged.model + "@" + tagged.protocol
    refs = {h: build_ref(tagged, h) for h in HORIZONS}
    got = E1A.interaction(refs, [E1A.PRIMARY_PAIR], "primary")
    old = pd.read_csv(ref)
    k = ["horizon", "increment"]
    j = old.merge(got, on=k, suffixes=("_old", "_new"))
    if len(j) != len(old):
        return f"FAILED: {len(old)} stored rows, {len(j)} matched"
    worst = max(float(np.abs(j[f"{c}_old"] - j[f"{c}_new"]).max())
                for c in ("ddR2", "lo", "hi", "holm_p", "seed_pos_frac"))
    if worst > 1e-12:
        raise SystemExit(f"[abort] wrapper does not reproduce E1: max|diff| = {worst:.3e}. "
                         f"The statistical core was perturbed; do not trust the Shanghai "
                         f"numbers produced by this run.")
    return f"reproduces e1_interaction.csv exactly (max|diff| = {worst:.1e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="also reproduce E1's stored CGMacros contrast through this wrapper")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""

    path = os.path.join(HERE, f"shanghai_oof{sfx}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[abort] {path} not found -- run shanghai_run.py first")

    print("=== SHANGHAI external validation of C1 ===")
    if args.verify:
        print(f"  wrapper check: {verify()}")

    d = pd.read_csv(path, dtype={"pid": str})
    print(f"  loaded {len(d)} OOF rows | arms {sorted(d.protocol.unique())} "
          f"| configs {sorted(d.model.unique())} | seeds {sorted(d.seed.unique())}")
    print(f"  participants: {d.pid.nunique()}   meals: {d.sample_idx.nunique()}")

    saved = _patched()
    try:
        print("\n=== INTEGRITY ===")
        print(f"  {E1A.check_unique(d)}")
        tagged = d.copy()
        tagged["model"] = tagged.model + "@" + tagged.protocol
        refs = {h: build_ref(tagged, h) for h in HORIZONS}
        print("  CGM-free configurations must be identical across arms:")
        print(check_invariance_storage_aware(refs[PRIMARY_H]))

        inter = E1A.interaction(refs, [E1A.PRIMARY_PAIR], "primary")
        mech = E1A.interaction(refs, E1A.MECHANISM_PAIRS, "mechanism")
        inc = E1A.within_arm_increments(refs)
        mods = E1A.per_arm_models(d)
    finally:
        _restore(saved)

    for name, df in (("interaction", inter), ("mechanism", mech),
                     ("increments", inc), ("models", mods)):
        p = os.path.join(HERE, f"shanghai_{name}{sfx}.csv")
        df.to_csv(p, index=False)
        print(f"  wrote {os.path.basename(p)}")

    raw = os.path.join(HERE, f"shanghai_results_raw{sfx}.csv")
    if os.path.exists(raw):
        lv = pd.read_csv(raw).groupby("protocol").level_retained.mean()
        print("\n=== LEVEL DIAGNOSTIC (model-free) ===")
        print("  between-participant share of variance in the MEAL-LEVEL MEAN pre-meal CGM")
        for p in E1A.PROTOCOLS:
            if p in lv.index:
                print(f"    {p:16s} {lv[p]:.4f}")

    print(f"\n=== R2 at {PRIMARY_H} min, by arm ===")
    piv = mods[mods.horizon == PRIMARY_H].pivot(index="protocol", columns="model",
                                                values="R2")
    print(piv.reindex([p for p in E1A.PROTOCOLS if p in piv.index]).round(3).to_string())

    print(f"\n=== PRIMARY INTERACTION (h = {PRIMARY_H}) ===")
    p = inter[(inter.horizon == PRIMARY_H)
              & (inter.increment == "CGM adds beyond context")]
    if p.empty:
        raise SystemExit("[abort] primary interaction missing")
    x = p.iloc[0]
    g = inc[(inc.horizon == PRIMARY_H)
            & (inc.increment == "CGM adds beyond context")].set_index("protocol")
    for arm, tag in ((REFERENCE_ARM, "reference "), (PUBLISHED_ARM, "published ")):
        if arm in g.index:
            r = g.loc[arm]
            print(f"  {tag} ({arm:14s}) dR2 = {r.dR2:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]")
    print(f"  INTERACTION (paired)            DDR2 = {x.ddR2:+.4f} "
          f"[{x.lo:+.4f}, {x.hi:+.4f}]")
    print(f"                                  Holm p = {x.holm_p:.4g}   "
          f"seeds {x.seed_pos_frac:.0%} (WIN_FRAC {WIN_FRAC:.0%})   "
          f"verdict = {x.verdict}  (margin {MARGIN_R2})")

    e1i = os.path.join(CORE, "e1_interaction.csv")
    if os.path.exists(e1i):
        e = pd.read_csv(e1i)
        e = e[(e.horizon == PRIMARY_H) & (e.increment == "CGM adds beyond context")]
        if not e.empty:
            r = e.iloc[0]
            print(f"  CGMacros (E1, same estimand)    DDR2 = {r.ddR2:+.4f} "
                  f"[{r.lo:+.4f}, {r.hi:+.4f}]")

    head = str(x.verdict).split("(")[0].strip()
    print(f"\n  => {x.verdict}. {VERDICT_TEXT.get(head, '(no verdict text)')}")
    print("\n  SCOPE: this is the LEVEL-COLLAPSE half of C1 only. Shanghai has no "
          "meal-content\n  modality, so the context-inflation half is not testable here "
          "and its absence is\n  not evidence against it. See the module docstring.")


if __name__ == "__main__":
    main()
