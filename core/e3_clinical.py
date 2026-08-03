"""
e3_clinical.py -- E3, clinical error bands for the primary increment
=====================================================================
NO REFITTING. Everything is re-measured from the stored per-meal out-of-fold predictions
in `e1_oof.csv`, exactly as `e1_mgdl.py` does. This is a third set of units on runs that
already happened, not a new experiment.

WHAT E3 IS ACTUALLY FOR, WHICH IS NOT WHAT THE PLAN ORIGINALLY ASSUMED
----------------------------------------------------------------------
REVISION_PLAN 11 says E3 exists because "the clinical-significance claims in the current
manuscript have no supporting analysis". There is exactly ONE such claim -- sn-article.tex
line 291, that modest RMSE reductions between FUSION DEPTHS "translate to clinically
relevant accuracy improvements" -- and it is attached to the fusion-depth claim that is
already being retracted. It disappears with its parent; E3 cannot rescue it and should not
try.

So E3 is run for the question the revised paper does raise. Table 3 now reports the
normalization effect in mg/dL: CGM is worth 7.5 mg/dL of RMSE under the reference
and 0.8 mg/dL under the published transform, a difference of 6.7. A reader will ask whether
6.7 mg/dL of RMSE is clinically legible at all. That is a fair question about C1's headline,
and it is answerable from data already on disk.

THE ESTIMAND IS THE SAME DIFFERENCE OF DIFFERENCES AS EVERYWHERE ELSE
---------------------------------------------------------------------
    within an arm   increment = metric(CGM + context) - metric(context only)
    interaction     (increment under `global`) - (increment under `subject_z`)

measured in PERCENTAGE POINTS of meals, with participant-cluster bootstrap intervals and a
participant sign-flip permutation. Unlike RMSE, every metric here is a mean of a per-meal
INDICATOR, so it decomposes additively over participants and the permutation is available
(see e1_mgdl.py's docstring for why RMSE is not).

TWO FAMILIES OF BAND, BECAUSE "CLINICAL ERROR BAND" MEANS BOTH IN THIS LITERATURE
---------------------------------------------------------------------------------
  * ABSOLUTE BANDS  -- share of meals predicted within +/-10, +/-20, +/-30 mg/dL. No grid
    geometry, no assumptions, and in the target's own units, which is what Reviewer 1 asked
    for. This is the one to lead with.
  * CLARKE ERROR GRID -- share in zone A and in zones A+B, the metric this literature quotes,
    included for comparability rather than because it is better.

    A CAVEAT THAT MUST TRAVEL WITH THE CLARKE NUMBERS. The grid was designed for BLOOD
    GLUCOSE METERS -- reference versus measurement at the SAME instant -- and its zones
    encode the clinical consequence of acting on a wrong reading now. Applying it to a
    60-minute-ahead FORECAST is an extension the original does not license: a forecast that
    lands in zone D was not a dangerous measurement, it was a prediction of a value that had
    not happened yet. The extension is common in the CGM-prediction literature and is
    reported here for that reason. Do not present zone percentages as safety evidence.

NO EQUIVALENCE VERDICT IS EMITTED, AND THAT IS DELIBERATE
----------------------------------------------------------
`decide()` needs a margin. MARGIN_R2 = 0.02 is a threshold on Delta-R-squared and is
explicitly NOT a clinical criterion (REVISION_PLAN 2). No clinical margin was declared
before these results existed, and declaring one now would be exactly the after-the-fact
move 10.6 refused for Shanghai. So this script reports intervals and a two-state label based
only on whether the interval excludes zero. There is no NEGLIGIBLE here: a percentage-point
difference cannot be called negligible without a threshold somebody committed to in advance.

Run:  python e3_clinical.py --verify
Out:  e3_clinical.csv
=====================================================================
"""
import os
import argparse
import numpy as np
import pandas as pd

from ablation_analyze import (build_ref, holm, PRIMARY_H, HORIZONS,
                              N_BOOT, N_PERM, BOOT_SEED, PERM_SEED)
from e1_analyze import REFERENCE_ARM, PUBLISHED_ARM

WITH, WITHOUT = "cgm_static", "static_all"
INCREMENT = "CGM adds beyond context"
ARMS = [REFERENCE_ARM, PUBLISHED_ARM]
BANDS = (10.0, 20.0, 30.0)
# Order matters for Holm only through the family, not through the correction itself.
METRICS = [f"within_{int(b)}" for b in BANDS] + ["clarke_A", "clarke_AB"]


def clarke_zones(ref, pred):
    """Clarke error-grid zone per (reference, prediction) pair, as a character array.

    The canonical zone geometry from Clarke et al. (1987), in the branch order the reference
    implementations use. ORDER IS PART OF THE DEFINITION: the zones are not disjoint as
    written, and testing them in a different sequence silently reclassifies points -- the A
    test must come first and the E test before C and D.

    Both arguments are mg/dL. Vectorised because the bootstrap calls the counting step
    thousands of times; the zones themselves depend only on (ref, pred) and so are computed
    once, up front, and merely re-selected inside the loop."""
    ref = np.asarray(ref, dtype=float)
    pred = np.asarray(pred, dtype=float)
    z = np.full(ref.shape, "B", dtype="<U1")

    a = ((pred <= 70) & (ref <= 70)) | ((pred <= 1.2 * ref) & (pred >= 0.8 * ref))
    e = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    c = (((ref >= 70) & (ref <= 290) & (pred >= ref + 110))
         | ((ref >= 130) & (ref <= 180) & (pred <= (7.0 / 5.0) * ref - 182)))
    d = (((ref >= 240) & (pred >= 70) & (pred <= 180))
         | ((ref <= 175.0 / 3.0) & (pred <= 180) & (pred >= 70))
         | ((ref >= 175.0 / 3.0) & (ref <= 70) & (pred >= (6.0 / 5.0) * ref)))

    z[d] = "D"
    z[c] = "C"
    z[e] = "E"
    z[a] = "A"                      # applied last so zone A wins, per the branch order
    return z


def indicators(ref):
    """Per-configuration boolean matrices, one row per seed, one column per meal.

    Every metric in E3 is the mean of one of these, so precomputing them turns the bootstrap
    into row-selection plus a mean."""
    y = ref["y"]
    out = {}
    for cfg in (WITH, WITHOUT):
        for arm in ARMS:
            key = f"{cfg}@{arm}"
            P = np.array([ref["preds"][key][s] for s in ref["seeds"]])   # (n_seeds, n_meals)
            err = np.abs(y[None, :] - P)
            ind = {f"within_{int(b)}": err <= b for b in BANDS}
            Z = np.array([clarke_zones(y, P[k]) for k in range(P.shape[0])])
            ind["clarke_A"] = Z == "A"
            ind["clarke_AB"] = np.isin(Z, ("A", "B"))
            ind["_zones"] = Z
            out[key] = ind
    return out


def _pct(ind, sel):
    """Percentage of meals satisfying the indicator: mean over meals, then over seeds."""
    return 100.0 * float(ind[:, sel].mean(axis=1).mean())


def analyse_horizon(ref):
    """Per-arm levels, per-arm increments and the interaction, from one shared bootstrap."""
    pid = ref["pid"]
    units = np.unique(pid)
    idx_by = {u: np.where(pid == u)[0] for u in units}
    allrows = np.arange(len(ref["y"]))
    IND = indicators(ref)

    def levels_on(sel):
        """{(arm, cfg, metric): pct}, {(arm, metric): increment}, {metric: interaction}."""
        lev, inc = {}, {}
        for arm in ARMS:
            for m in METRICS:
                w = _pct(IND[f"{WITH}@{arm}"][m], sel)
                o = _pct(IND[f"{WITHOUT}@{arm}"][m], sel)
                lev[(arm, WITH, m)] = w
                lev[(arm, WITHOUT, m)] = o
                inc[(arm, m)] = w - o
        inter = {m: inc[(ARMS[0], m)] - inc[(ARMS[1], m)] for m in METRICS}
        return lev, inc, inter

    p_lev, p_inc, p_inter = levels_on(allrows)

    # Same draw sequence as the locked core: default_rng(BOOT_SEED), then one
    # rng.choice(units, len(units), replace=True) per iteration. Keeping it identical means
    # E3's intervals are drawn on the same resamples as E1's, so the two are comparable
    # rather than merely similar.
    rng = np.random.default_rng(BOOT_SEED)
    b_lev = {k: np.empty(N_BOOT) for k in p_lev}
    b_inc = {k: np.empty(N_BOOT) for k in p_inc}
    b_inter = {m: np.empty(N_BOOT) for m in METRICS}
    for k in range(N_BOOT):
        sel = np.concatenate([idx_by[u] for u in rng.choice(units, len(units), replace=True)])
        lev, inc, inter = levels_on(sel)
        for key, v in lev.items():
            b_lev[key][k] = v
        for key, v in inc.items():
            b_inc[key][k] = v
        for m, v in inter.items():
            b_inter[m][k] = v

    # Participant sign-flip permutation on the interaction, plus per-seed sign agreement.
    # The permutation is legitimate here and not for RMSE: each metric is a mean of a
    # per-meal indicator, so the participant contributions to the numerator sum to the total
    # exactly.
    perm, seed_pos = {}, {}
    for m in METRICS:
        A_w, A_o = IND[f"{WITH}@{ARMS[0]}"][m], IND[f"{WITHOUT}@{ARMS[0]}"][m]
        B_w, B_o = IND[f"{WITH}@{ARMS[1]}"][m], IND[f"{WITHOUT}@{ARMS[1]}"][m]
        dd = (A_w.mean(axis=0) - A_o.mean(axis=0)) - (B_w.mean(axis=0) - B_o.mean(axis=0))
        dj = np.array([dd[idx_by[u]].sum() for u in units])
        rng2 = np.random.default_rng(PERM_SEED)
        T = (rng2.choice([-1.0, 1.0], size=(N_PERM, len(units))) * dj[None, :]).sum(axis=1)
        perm[m] = float((1 + np.sum(np.abs(T) >= abs(dj.sum()) - 1e-12)) / (N_PERM + 1))
        per_seed = ((A_w.mean(axis=1) - A_o.mean(axis=1))
                    - (B_w.mean(axis=1) - B_o.mean(axis=1)))
        seed_pos[m] = float(np.mean(per_seed > 0))

    def ci(b):
        return float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))

    rows = []
    for m in METRICS:
        for arm in ARMS:
            for cfg in (WITH, WITHOUT):
                lo, hi = ci(b_lev[(arm, cfg, m)])
                rows.append({"kind": "level", "metric": m, "arm": arm, "config": cfg,
                             "estimate": p_lev[(arm, cfg, m)], "lo": lo, "hi": hi})
            lo, hi = ci(b_inc[(arm, m)])
            rows.append({"kind": "increment", "metric": m, "arm": arm, "config": "",
                         "estimate": p_inc[(arm, m)], "lo": lo, "hi": hi})
        lo, hi = ci(b_inter[m])
        rows.append({"kind": "interaction", "metric": m,
                     "arm": f"{ARMS[0]} vs {ARMS[1]}", "config": "",
                     "estimate": p_inter[m], "lo": lo, "hi": hi, "perm_p": perm[m],
                     "seed_pos_frac": seed_pos[m]})
    return rows


def verify(ref_by_h):
    """Cross-check against results that already exist, so a drift shows up as a failure.

    Two checks. The first is arithmetic: the share of meals inside +/-30 mg/dL cannot be
    smaller than the share inside +/-20, and Clarke A cannot exceed Clarke A+B. The second
    ties E3 to the table it sits beside -- the reference arm's RMSE recomputed here must
    equal the one e1_mgdl.py wrote, since both read the same predictions."""
    msgs = []
    out = pd.read_csv("e3_clinical.csv")
    lev = out[out.kind == "level"]
    bad = []
    for (h, arm, cfg), g in lev.groupby(["horizon", "arm", "config"]):
        v = g.set_index("metric").estimate
        if not (v["within_10"] <= v["within_20"] + 1e-9 <= v["within_30"] + 1e-9):
            bad.append(f"band monotonicity h={h} {arm}/{cfg}")
        if v["clarke_A"] > v["clarke_AB"] + 1e-9:
            bad.append(f"A > A+B h={h} {arm}/{cfg}")
    msgs.append("monotonicity of bands and zones      : "
                + ("OK" if not bad else "*** " + "; ".join(bad) + " ***"))

    if os.path.exists("e1_mgdl_horizons.csv"):
        stored = pd.read_csv("e1_mgdl_horizons.csv")
        worst = 0.0
        for h, ref in ref_by_h.items():
            y = ref["y"]
            for arm in ARMS:
                r = np.sqrt(np.mean([np.mean((y - ref["preds"][f"{WITH}@{arm}"][s]) ** 2)
                                     for s in ref["seeds"]]))
                del r          # per-seed-then-mean is what the table uses; compute that way
                per = [np.sqrt(np.mean((y - ref["preds"][f"{WITH}@{arm}"][s]) ** 2))
                       for s in ref["seeds"]]
                po = [np.sqrt(np.mean((y - ref["preds"][f"{WITHOUT}@{arm}"][s]) ** 2))
                      for s in ref["seeds"]]
                mine = float(np.mean(po) - np.mean(per))
                g = stored[(stored.horizon == h) & (stored.kind == "within_arm")
                           & (stored.arm == arm)]
                if not g.empty:
                    worst = max(worst, abs(mine - float(g.iloc[0].dRMSE_mgdl)))
        msgs.append(f"dRMSE vs e1_mgdl_horizons.csv        : max|diff| = {worst:.1e}"
                    + ("  MATCH" if worst < 1e-9 else "  *** MISMATCH ***"))
    else:
        msgs.append("dRMSE vs e1_mgdl_horizons.csv        : SKIPPED (file absent)")
    return "\n  ".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if not os.path.exists("e1_oof.csv"):
        raise SystemExit("[abort] e1_oof.csv not found -- run e1_run.py first")
    oof = pd.read_csv("e1_oof.csv", dtype={"pid": str})
    oof = oof[oof.protocol.isin(ARMS) & oof.model.isin([WITH, WITHOUT])]
    print(f"loaded e1_oof.csv: {len(oof)} rows | arms {sorted(oof.protocol.unique())} "
          f"| seeds {sorted(oof.seed.unique())}")

    tagged = oof.copy()
    tagged["model"] = tagged.model + "@" + tagged.protocol

    recs, ref_by_h = [], {}
    for h in HORIZONS:
        ref = build_ref(tagged, h)
        ref_by_h[h] = ref
        for r in analyse_horizon(ref):
            recs.append({"horizon": h, "increment": INCREMENT, **r})
        print(f"  h = {h:3d} done ({len(ref['seeds'])} seeds, {len(ref['y'])} meals, "
              f"{len(np.unique(ref['pid']))} participants)")
    out = pd.DataFrame(recs)

    # Holm within the family of metrics tested at the confirmatory horizon. Five metrics,
    # one interaction each. The other horizons are secondary and unadjusted, exactly as in
    # E1 -- printing an adjusted p there would imply a family that was never declared.
    inter = out[(out.kind == "interaction") & (out.horizon == PRIMARY_H)]
    adj = dict(zip(inter.metric, holm(inter.perm_p.tolist())))
    out["holm_p"] = [adj.get(x.metric, np.nan)
                     if (x.kind == "interaction" and x.horizon == PRIMARY_H) else np.nan
                     for x in out.itertuples()]

    # Two states only. NEGLIGIBLE is unavailable: it requires a margin, and no clinical
    # margin was declared before these numbers existed (see the module docstring).
    #
    # CHANGES requires BOTH the bootstrap interval to exclude zero AND the Holm-adjusted
    # permutation p to clear 0.05. They are different procedures and they do disagree here --
    # the +/-30 mg/dL and Clarke A+B rows have intervals excluding zero at Holm p = 0.073.
    # Labelling those CHANGES on the interval alone would print a positive verdict beside a
    # non-significant p in the same row, which is precisely the kind of thing Reviewer 1 went
    # looking for.
    def _verdict(x):
        if not (x.kind == "interaction" and x.horizon == PRIMARY_H):
            return ""
        return "CHANGES" if ((x.lo > 0 or x.hi < 0) and x.holm_p <= 0.05) else "INCONCLUSIVE"

    out["verdict"] = [_verdict(x) for x in out.itertuples()]

    out.to_csv("e3_clinical.csv", index=False)

    lab = {"within_10": "within +/-10 mg/dL", "within_20": "within +/-20 mg/dL",
           "within_30": "within +/-30 mg/dL", "clarke_A": "Clarke zone A",
           "clarke_AB": "Clarke zones A+B"}
    print(f"\n=== E3, clinical error bands | {INCREMENT} | h = {PRIMARY_H} min "
          f"(confirmatory) ===")
    print("  levels are % of held-out meals; increments and the interaction are "
          "PERCENTAGE POINTS\n")
    g = out[out.horizon == PRIMARY_H]
    for m in METRICS:
        print(f"  {lab[m]}")
        for arm in ARMS:
            lv = g[(g.kind == "level") & (g.metric == m) & (g.arm == arm)]
            w = lv[lv.config == WITH].iloc[0]
            o = lv[lv.config == WITHOUT].iloc[0]
            inc = g[(g.kind == "increment") & (g.metric == m) & (g.arm == arm)].iloc[0]
            print(f"    {arm:14s} context {o.estimate:5.1f}%  -> +CGM {w.estimate:5.1f}%"
                  f"   increment {inc.estimate:+5.2f} pp "
                  f"[{inc.lo:+5.2f}, {inc.hi:+5.2f}]")
        it = g[(g.kind == "interaction") & (g.metric == m)].iloc[0]
        print(f"    {'INTERACTION':14s} {it.estimate:+5.2f} pp "
              f"[{it.lo:+5.2f}, {it.hi:+5.2f}]  perm {it.perm_p:.4f}  "
              f"Holm {it.holm_p:.4f}  seeds {it.seed_pos_frac:.0%}  {it.verdict}\n")

    print("=== other horizons (secondary, unadjusted): interaction only ===")
    for h in HORIZONS:
        if h == PRIMARY_H:
            continue
        gg = out[(out.horizon == h) & (out.kind == "interaction")]
        cells = "  ".join(f"{lab[r.metric].split()[-2] if 'within' in r.metric else r.metric}"
                          f" {r.estimate:+.2f}" for _, r in gg.iterrows())
        print(f"  h = {h:3d}  {cells}")

    if args.verify:
        print("\n=== VERIFY ===\n  " + verify(ref_by_h))
    print("\nwrote e3_clinical.csv")


if __name__ == "__main__":
    main()
