"""
e1_mgdl.py -- the primary estimand in mg/dL, at all three horizons
===================================================================
Reviewer 1 asked for errors in mg/dL and for complete results at every horizon; a later
review added that MARD does not discharge that request, because it is a percentage rather
than an error in the target's units. `e1_interaction.csv` carries the interaction in
Delta-R-squared only. This script closes that gap.

NO REFITTING. Everything is derived from the stored per-meal out-of-fold predictions in
`e1_oof.csv`. This is a re-measurement of runs that already happened, not a new experiment.

WHAT IS COMPUTED
-----------------
At each horizon, for the primary increment (CGM beyond context):

    within `global`      dR2, dRMSE, dMAE   -- the increment under the reference
    within `subject_z`   dR2, dRMSE, dMAE   -- the increment under the published transform
    interaction          ddR2, ddRMSE, ddMAE -- how much the increment itself moves

Sign convention throughout matches `ablation_analyze.contrast_stats`: POSITIVE means the
first-named configuration is better, i.e. lower error. So a positive dRMSE means adding CGM
REDUCES error by that many mg/dL, and a positive ddRMSE means that reduction is larger under
the reference arm than under the published transform.

WHY THE p-VALUE COLUMN IS THE R-SQUARED ONE
--------------------------------------------
The sign-flip permutation in the locked core tests a statistic that decomposes additively
over participants. Delta-R-squared does (it is a sum of per-meal error reductions over a
fixed total sum of squares) and so does MAE. RMSE does NOT -- it is a square root of a mean,
so per-participant contributions do not sum to the total. Rather than run a permutation for
two of the three metrics and leave a hole in the third, or invent a different test per
column, this script does what both reviews asked for explicitly: R-squared stays primary and
carries the inference, the mg/dL columns are confirmatory and carry participant-cluster
bootstrap intervals. The table notes say so.

MATCHING THE STORED RESULT EXACTLY
-----------------------------------
One bootstrap loop per horizon computes every metric on the SAME resample. Because the loop
recreates the core's draw sequence exactly -- `default_rng(BOOT_SEED)`, one
`rng.choice(units, len(units), replace=True)` per iteration, `units = np.unique(pid)` -- the
k-th resample here is the k-th resample there. `--verify` checks that the Delta-R-squared
columns reproduce `e1_interaction.csv` and `e1_increments.csv`, so a drift in the core
surfaces as a failed check rather than as two tables that quietly disagree.

Run:  python e1_mgdl.py --verify
Out:  e1_mgdl_horizons.csv
===================================================================
"""
import os
import argparse
import numpy as np
import pandas as pd

from ablation_analyze import (build_ref, decide, holm, MARGIN_R2, WIN_FRAC,
                              PRIMARY_H, HORIZONS, N_BOOT, N_PERM, BOOT_SEED, PERM_SEED)
from e1_analyze import REFERENCE_ARM, PUBLISHED_ARM

WITH, WITHOUT = "cgm_static", "static_all"
INCREMENT = "CGM adds beyond context"
ARMS = [REFERENCE_ARM, PUBLISHED_ARM]


def _errs(ref, cfg, arm):
    """Signed errors for one configuration under one arm: (n_seeds, n_meals)."""
    key = f"{cfg}@{arm}"
    P = np.array([ref["preds"][key][s] for s in ref["seeds"]])
    return ref["y"][None, :] - P


def _metrics(sel, E_with, E_without, ss_getter):
    """The three increments on one row-selection, given signed-error matrices.

    dR2   event-weighted mean-single-run, exactly the core's estimand
    dRMSE mean over seeds of RMSE(without) - RMSE(with)   [mg/dL, positive = with is better]
    dMAE  mean over seeds of MAE(without)  - MAE(with)    [mg/dL, same convention]
    """
    Ew, Eo = E_with[:, sel], E_without[:, sel]
    ss = ss_getter(sel)
    dr2 = ((Eo ** 2).mean(axis=0) - (Ew ** 2).mean(axis=0)).sum() / ss if ss > 0 else np.nan
    drmse = float(np.mean(np.sqrt((Eo ** 2).mean(axis=1)) - np.sqrt((Ew ** 2).mean(axis=1))))
    dmae = float(np.mean(np.abs(Eo).mean(axis=1) - np.abs(Ew).mean(axis=1)))
    return dr2, drmse, dmae


def analyse_horizon(ref):
    """All within-arm increments and the interaction, from one shared bootstrap."""
    y, pid = ref["y"], ref["pid"]
    units = np.unique(pid)
    idx_by = {u: np.where(pid == u)[0] for u in units}
    allrows = np.arange(len(y))

    def ss_of(sel):
        yt = y[sel]
        return float(((yt - yt.mean()) ** 2).sum())

    E = {a: (_errs(ref, WITH, a), _errs(ref, WITHOUT, a)) for a in ARMS}

    def stats_on(sel):
        """(per-arm metrics, interaction metrics) for one row-selection."""
        per = {a: _metrics(sel, *E[a], ss_of) for a in ARMS}
        inter = tuple(per[ARMS[0]][i] - per[ARMS[1]][i] for i in range(3))
        return per, inter

    point_per, point_inter = stats_on(allrows)

    rng = np.random.default_rng(BOOT_SEED)
    b_per = {a: np.empty((N_BOOT, 3)) for a in ARMS}
    b_inter = np.empty((N_BOOT, 3))
    for k in range(N_BOOT):
        sel = np.concatenate([idx_by[u] for u in rng.choice(units, len(units), replace=True)])
        per, inter = stats_on(sel)
        for a in ARMS:
            b_per[a][k] = per[a]
        b_inter[k] = inter

    # Inference on the R-squared interaction only -- see the module docstring.
    ss_tot = ss_of(allrows)
    Ew1, Eo1 = E[ARMS[0]]
    Ew2, Eo2 = E[ARMS[1]]
    DD = (((Eo1 ** 2).mean(axis=0) - (Ew1 ** 2).mean(axis=0))
          - ((Eo2 ** 2).mean(axis=0) - (Ew2 ** 2).mean(axis=0)))
    dj = np.array([DD[idx_by[u]].sum() for u in units])
    rng2 = np.random.default_rng(PERM_SEED)
    Tstar = (rng2.choice([-1.0, 1.0], size=(N_PERM, len(units))) * dj[None, :]).sum(axis=1)
    perm_p = float((1 + np.sum(np.abs(Tstar) >= abs(dj.sum()) - 1e-12)) / (N_PERM + 1))
    per_seed = np.array([(((Eo1[k] ** 2) - (Ew1[k] ** 2))
                          - ((Eo2[k] ** 2) - (Ew2[k] ** 2))).sum() / ss_tot
                         for k in range(len(ref["seeds"]))])

    def row(kind, arm, point, boot):
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
        return {"kind": kind, "arm": arm,
                "dR2": point[0], "dR2_lo": lo[0], "dR2_hi": hi[0],
                "dRMSE_mgdl": point[1], "dRMSE_lo": lo[1], "dRMSE_hi": hi[1],
                "dMAE_mgdl": point[2], "dMAE_lo": lo[2], "dMAE_hi": hi[2]}

    rows = [row("within_arm", a, point_per[a], b_per[a]) for a in ARMS]
    r = row("interaction", f"{ARMS[0]} vs {ARMS[1]}", point_inter, b_inter)
    r.update({"perm_p": perm_p, "seed_pos_frac": float(np.mean(per_seed > 0))})
    rows.append(r)
    return rows


def verify(out):
    """Reproduce the stored Delta-R-squared results through this script's own bootstrap.

    If the locked core is ever perturbed, or this script's resample sequence drifts from it,
    the mg/dL table would silently disagree with the R-squared table it sits beside. This is
    the check that turns that into a visible failure."""
    msgs = []
    inter = pd.read_csv("e1_interaction.csv")
    inter = inter[inter.increment == INCREMENT].set_index("horizon")
    mine = out[out.kind == "interaction"].set_index("horizon")
    worst = 0.0
    for h in HORIZONS:
        for col_a, col_b in (("ddR2", "dR2"), ("lo", "dR2_lo"), ("hi", "dR2_hi")):
            worst = max(worst, abs(float(inter.loc[h, col_a]) - float(mine.loc[h, col_b])))
    msgs.append(f"interaction vs e1_interaction.csv : max|diff| = {worst:.1e}"
                + ("  MATCH" if worst < 1e-12 else "  *** MISMATCH ***"))

    # The permutation is recomputed here, so it is worth checking too: if it drifts, the
    # inherited Holm value would be attached to a different test than the one measured.
    wp = max(abs(float(inter.loc[h, "perm_p"]) - float(mine.loc[h, "perm_p"]))
             for h in HORIZONS)
    msgs.append(f"perm p vs e1_interaction.csv      : max|diff| = {wp:.1e}"
                + ("  MATCH" if wp < 1e-12 else "  *** MISMATCH ***"))

    inc = pd.read_csv("e1_increments.csv")
    inc = inc[inc.increment == INCREMENT]
    w = out[out.kind == "within_arm"]
    worst2 = 0.0
    for _, r in w.iterrows():
        g = inc[(inc.horizon == r.horizon) & (inc.protocol == r.arm)]
        if g.empty:
            continue
        g = g.iloc[0]
        for col_a, col_b in (("dR2", "dR2"), ("lo", "dR2_lo"), ("hi", "dR2_hi")):
            worst2 = max(worst2, abs(float(g[col_a]) - float(r[col_b])))
    msgs.append(f"within-arm vs e1_increments.csv   : max|diff| = {worst2:.1e}"
                + ("  MATCH" if worst2 < 1e-12 else "  *** MISMATCH ***"))
    return "\n  ".join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check the R^2 columns against the stored E1 results")
    args = ap.parse_args()

    if not os.path.exists("e1_oof.csv"):
        raise SystemExit("[abort] e1_oof.csv not found -- run e1_run.py first")
    oof = pd.read_csv("e1_oof.csv", dtype={"pid": str})
    oof = oof[oof.protocol.isin(ARMS) & oof.model.isin([WITH, WITHOUT])]
    print(f"loaded e1_oof.csv: {len(oof)} rows | arms {sorted(oof.protocol.unique())} "
          f"| seeds {sorted(oof.seed.unique())}")

    tagged = oof.copy()
    tagged["model"] = tagged.model + "@" + tagged.protocol

    recs = []
    for h in HORIZONS:
        ref = build_ref(tagged, h)
        for r in analyse_horizon(ref):
            recs.append({"horizon": h, "increment": INCREMENT, **r})
        print(f"  h = {h:3d} done ({len(ref['seeds'])} seeds, {len(ref['y'])} meals, "
              f"{len(np.unique(ref['pid']))} participants)")
    out = pd.DataFrame(recs)

    # Holm is TAKEN FROM e1_interaction.csv, not recomputed. It must not be recomputed here:
    # e1_analyze's primary family at the confirmatory horizon contains BOTH increments (CGM
    # beyond context and context beyond CGM), so adjusting the one increment this script
    # carries would return 0.0001 where the primary table reports 0.0002 -- two tables in one
    # paper disagreeing on the adjusted p of the same test. The p belongs to the R-squared
    # estimand, which is already computed and stored; this script re-measures the same runs in
    # different units and inherits it.
    stored = pd.read_csv("e1_interaction.csv")
    stored = stored[stored.increment == INCREMENT].set_index("horizon")
    out["holm_p"] = [
        (float(stored.loc[x.horizon, "holm_p"]) if x.kind == "interaction" else np.nan)
        for x in out.itertuples()]
    out["verdict"] = [
        (decide(x.dR2, x.dR2_lo, x.dR2_hi, x.holm_p, x.seed_pos_frac)
         if (x.kind == "interaction" and np.isfinite(x.holm_p)) else "")
        for x in out.itertuples()]

    out.to_csv("e1_mgdl_horizons.csv", index=False)

    if args.verify:
        print("\n=== VERIFY ===\n  " + verify(out))

    print("\n=== PRIMARY INCREMENT (CGM beyond context), all horizons ===")
    print("  positive dRMSE / dMAE = adding CGM REDUCES error by that many mg/dL")
    for h in HORIZONS:
        g = out[out.horizon == h]
        print(f"\n-- {h} min --")
        for _, r in g.iterrows():
            tag = "INTERACTION" if r.kind == "interaction" else r.arm
            extra = ""
            if r.kind == "interaction":
                hp = f"{r.holm_p:.4f}" if np.isfinite(r.holm_p) else "--(secondary)"
                extra = f"  Holm {hp}  seeds {r.seed_pos_frac:.0%}  {r.verdict}"
            print(f"  {tag:14s} dR2 {r.dR2:+.4f} [{r.dR2_lo:+.4f}, {r.dR2_hi:+.4f}]"
                  f"   dRMSE {r.dRMSE_mgdl:+6.2f} [{r.dRMSE_lo:+6.2f}, {r.dRMSE_hi:+6.2f}]"
                  f"   dMAE {r.dMAE_mgdl:+6.2f} [{r.dMAE_lo:+6.2f}, {r.dMAE_hi:+6.2f}]{extra}")

    print("\nwrote e1_mgdl_horizons.csv")


if __name__ == "__main__":
    main()
