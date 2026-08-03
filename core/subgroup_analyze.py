"""
subgroup_analyze.py -- performance by diabetes diagnosis (Reviewer 1's subgroup request)
========================================================================================
Post-hoc, CPU-only. Reads the OOF predictions already produced by ablation_run.py (and
optionally window_run.py). Fits nothing, so it changes no confirmatory result.

STATISTICAL CORE is the one used everywhere else in this paper: mean-single-run metrics
(compute per seed, then average over seeds) with a PARTICIPANT-cluster bootstrap. The only
change is that resampling happens WITHIN a diagnosis group, because that is the population
each subgroup estimate refers to.

THE TRAP THIS FILE IS BUILT AROUND
-----------------------------------
R2 is scored against the variance of whatever rows it is computed on. In CGMacros the
60-min target SD is 39.8 / 42.0 / 63.9 mg/dL for healthy / prediabetes / T2D. So a model
with IDENTICAL absolute error in every group still posts a much higher R2 in T2D, purely
because the denominator is larger. Reporting subgroup R2 alone would manufacture the
conclusion "the model works best in T2D".

Therefore:
  * RMSE / MAE (mg/dL) are the CROSS-GROUP comparison. They are on a fixed scale.
  * R2 is reported per group but is a WITHIN-group statistic only. Never rank groups by it.
  * The between-group table (subgroup_gaps) is RMSE/MAE only, by construction.
  * skill_vs_persistence is included because it is the scale-free quantity that R2 was
    meant to be: how much the model beats the same-group persistence floor.

Bootstrap is exact and cheap because every metric here is a ratio of sums over rows, so
each participant is reduced to sufficient statistics ONCE and a resample is a sum over
drawn participants (no re-indexing of the raw rows).

Outputs
  subgroup_descriptives.csv   group sizes and target distribution
  subgroup_performance.csv    model x group x horizon: R2, RMSE, MAE, bias + CIs
  subgroup_gaps.csv           between-group RMSE/MAE differences + CIs
  subgroup_increments.csv     do the headline ablation increments hold inside each group
========================================================================================
"""
import numpy as np
import pandas as pd

from ablation_analyze import (MARGIN_R2, WIN_FRAC, HORIZONS, N_BOOT, N_PERM,
                              BOOT_SEED, PERM_SEED, holm, decide)

TAG = "rf_capped"                 # ablation_analyze.PRIMARY_TAG -- the confirmatory tag
DIAG_NAME = {0: "healthy", 1: "prediabetes", 2: "t2d"}
PERSIST = "persistence_5min"

# Models carried into the subgroup report. Kept small on purpose: these are the ones the
# revision's argument actually rests on.
MODELS = ["full", "cgm_static", "cgm_person_meal", "cgm", "static_all", PERSIST]

# The increments whose direction the revision claims. Re-tested inside each group.
INCREMENTS = [
    ("cgm_static", "static_all", "CGM adds beyond all static features"),
    ("cgm_static", "cgm", "static features add beyond CGM"),
    ("full", "cgm_static", "activity/HR add beyond CGM+static"),
    ("cgm_person_meal", "cgm_person", "meal content adds beyond person+CGM"),
]


# ---------------------------------------------------------------- sufficient statistics
def suff_stats(oof, model, horizon):
    """Per-participant sufficient statistics for `model` at `horizon`.

    Returns pid array plus arrays shaped (n_seeds, n_participants) for the error sums, and
    (n_participants,) for the target sums. Every metric below is a ratio of sums of these,
    which is what makes an exact 5000-draw cluster bootstrap essentially free."""
    yt, yp = f"y_true_{horizon}", f"y_pred_{horizon}"
    g = oof[oof.model == model]
    seeds = np.sort(g.seed.unique())
    pids = np.sort(g.pid.unique())
    pidx = {p: i for i, p in enumerate(pids)}

    n = np.zeros(len(pids))
    sum_y = np.zeros(len(pids))
    sum_y2 = np.zeros(len(pids))
    sse = np.zeros((len(seeds), len(pids)))
    sae = np.zeros((len(seeds), len(pids)))
    sbias = np.zeros((len(seeds), len(pids)))

    for k, s in enumerate(seeds):
        gs = g[g.seed == s]
        j = gs.pid.map(pidx).values
        e = gs[yp].values - gs[yt].values
        np.add.at(sse, (k, j), e ** 2)
        np.add.at(sae, (k, j), np.abs(e))
        np.add.at(sbias, (k, j), e)
        if k == 0:
            np.add.at(n, j, 1.0)
            np.add.at(sum_y, j, gs[yt].values)
            np.add.at(sum_y2, j, gs[yt].values ** 2)
    return {"pids": pids, "n": n, "sum_y": sum_y, "sum_y2": sum_y2,
            "sse": sse, "sae": sae, "sbias": sbias, "seeds": seeds}


def metrics_from(st, sel):
    """Mean-single-run metrics over the participants indexed by `sel` (with repeats)."""
    n = st["n"][sel].sum()
    if n <= 1:
        return dict(R2=np.nan, RMSE=np.nan, MAE=np.nan, bias=np.nan)
    sy, sy2 = st["sum_y"][sel].sum(), st["sum_y2"][sel].sum()
    ss_tot = sy2 - sy * sy / n
    sse = st["sse"][:, sel].sum(axis=1)          # per seed
    rmse = np.sqrt(sse / n)
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else np.full_like(sse, np.nan)
    return dict(R2=float(np.mean(r2)), RMSE=float(np.mean(rmse)),
                MAE=float(np.mean(st["sae"][:, sel].sum(axis=1) / n)),
                bias=float(np.mean(st["sbias"][:, sel].sum(axis=1) / n)))


def boot_draws(n_units, rng):
    return rng.integers(0, n_units, size=(N_BOOT, n_units))


# ---------------------------------------------------------------- main tables
def descriptives(oof):
    rows = []
    one = oof[(oof.model == MODELS[0]) & (oof.seed == oof.seed.min())]
    for d, g in one.groupby("diag"):
        r = {"diag": d, "group": DIAG_NAME[d],
             "n_participants": g.pid.nunique(), "n_meals": len(g),
             "meals_per_participant": round(len(g) / g.pid.nunique(), 1)}
        for h in HORIZONS:
            r[f"y{h}_mean"] = round(g[f"y_true_{h}"].mean(), 1)
            r[f"y{h}_sd"] = round(g[f"y_true_{h}"].std(), 1)
        rows.append(r)
    return pd.DataFrame(rows)


def performance(oof):
    rows = []
    for h in HORIZONS:
        for d in sorted(oof.diag.unique()):
            sub = oof[oof.diag == d]
            floor = suff_stats(sub, PERSIST, h)
            rng = np.random.default_rng(BOOT_SEED)
            draws = boot_draws(len(floor["pids"]), rng)   # SAME draws for every model
            for m in MODELS:
                if m not in set(sub.model.unique()):
                    continue
                st = suff_stats(sub, m, h)
                assert np.array_equal(st["pids"], floor["pids"]), "participant mismatch"
                obs = metrics_from(st, np.arange(len(st["pids"])))
                obs_f = metrics_from(floor, np.arange(len(floor["pids"])))
                bR2 = np.empty(N_BOOT); bRM = np.empty(N_BOOT)
                bMA = np.empty(N_BOOT); bSK = np.empty(N_BOOT)
                for k in range(N_BOOT):
                    sel = draws[k]
                    mm = metrics_from(st, sel); ff = metrics_from(floor, sel)
                    bR2[k] = mm["R2"]; bRM[k] = mm["RMSE"]; bMA[k] = mm["MAE"]
                    bSK[k] = 1.0 - mm["RMSE"] / ff["RMSE"] if ff["RMSE"] > 0 else np.nan
                skill = 1.0 - obs["RMSE"] / obs_f["RMSE"] if obs_f["RMSE"] > 0 else np.nan
                rows.append({
                    "horizon": h, "diag": d, "group": DIAG_NAME[d], "model": m,
                    "n_participants": len(st["pids"]), "n_meals": int(st["n"].sum()),
                    "R2": obs["R2"], "R2_lo": np.nanpercentile(bR2, 2.5),
                    "R2_hi": np.nanpercentile(bR2, 97.5),
                    "RMSE": obs["RMSE"], "RMSE_lo": np.nanpercentile(bRM, 2.5),
                    "RMSE_hi": np.nanpercentile(bRM, 97.5),
                    "MAE": obs["MAE"], "MAE_lo": np.nanpercentile(bMA, 2.5),
                    "MAE_hi": np.nanpercentile(bMA, 97.5),
                    "bias": obs["bias"],
                    "skill_vs_persistence": skill,
                    "skill_lo": np.nanpercentile(bSK, 2.5),
                    "skill_hi": np.nanpercentile(bSK, 97.5)})
    return pd.DataFrame(rows)


def gaps(oof):
    """Between-group differences. RMSE/MAE ONLY -- R2 is not comparable across groups."""
    rows = []
    pairs = [(2, 0), (2, 1), (1, 0)]
    for h in HORIZONS:
        for m in MODELS:
            for da, db in pairs:
                sa = suff_stats(oof[oof.diag == da], m, h)
                sb = suff_stats(oof[oof.diag == db], m, h)
                oa = metrics_from(sa, np.arange(len(sa["pids"])))
                ob = metrics_from(sb, np.arange(len(sb["pids"])))
                rng = np.random.default_rng(BOOT_SEED)
                dA = boot_draws(len(sa["pids"]), rng)
                dB = boot_draws(len(sb["pids"]), rng)
                bR = np.empty(N_BOOT); bM = np.empty(N_BOOT)
                for k in range(N_BOOT):
                    ma = metrics_from(sa, dA[k]); mb = metrics_from(sb, dB[k])
                    bR[k] = ma["RMSE"] - mb["RMSE"]; bM[k] = ma["MAE"] - mb["MAE"]
                rows.append({
                    "horizon": h, "model": m,
                    "comparison": f"{DIAG_NAME[da]} - {DIAG_NAME[db]}",
                    "dRMSE": oa["RMSE"] - ob["RMSE"],
                    "dRMSE_lo": np.percentile(bR, 2.5), "dRMSE_hi": np.percentile(bR, 97.5),
                    "dMAE": oa["MAE"] - ob["MAE"],
                    "dMAE_lo": np.percentile(bM, 2.5), "dMAE_hi": np.percentile(bM, 97.5),
                    "worse_in_first": bool((oa["RMSE"] - ob["RMSE"]) > 0)})
    return pd.DataFrame(rows)


def increments(oof, horizon=60):
    """Re-test the revision's headline increments WITHIN each diagnosis group.

    Same estimand and same machinery as ablation_analyze.contrast_stats (event-weighted
    mean-single-run dR2, cluster bootstrap, participant sign-flip permutation), just
    restricted to one group. Holm is applied within group across the four increments.
    Power is low by construction -- 11 to 16 participants -- so INCONCLUSIVE is expected
    and is NOT evidence that an increment fails in a subgroup."""
    yt, yp = f"y_true_{horizon}", f"y_pred_{horizon}"
    rows = []
    for d in sorted(oof.diag.unique()):
        sub = oof[oof.diag == d]
        seeds = np.sort(sub.seed.unique())
        base = sub[(sub.model == MODELS[0]) & (sub.seed == seeds[0])].sort_values("sample_idx")
        order, y, pid = base.sample_idx.values, base[yt].values.astype(float), base.pid.values
        P = {}
        for (m, s), g in sub.groupby(["model", "seed"]):
            g = g.sort_values("sample_idx")
            assert np.array_equal(g.sample_idx.values, order), f"order mismatch {m}/{s}"
            P.setdefault(m, {})[s] = g[yp].values.astype(float)
        units = np.unique(pid); idx_by = {u: np.where(pid == u)[0] for u in units}
        recs = []
        for a, b, label in INCREMENTS:
            if a not in P or b not in P:
                continue
            Ea = (y[None, :] - np.array([P[a][s] for s in seeds])) ** 2
            Eb = (y[None, :] - np.array([P[b][s] for s in seeds])) ** 2
            Dbar = (Eb - Ea).mean(axis=0)

            def dR2(sel):
                yy = y[sel]; ss = ((yy - yy.mean()) ** 2).sum()
                return Dbar[sel].sum() / ss if ss > 0 else np.nan

            point = dR2(np.arange(len(y)))
            rng = np.random.default_rng(BOOT_SEED)
            bb = np.array([dR2(np.concatenate([idx_by[u] for u in
                                               rng.choice(units, len(units), replace=True)]))
                           for _ in range(N_BOOT)])
            lo, hi = np.nanpercentile(bb, [2.5, 97.5])
            dj = np.array([Dbar[idx_by[u]].sum() for u in units])
            rng2 = np.random.default_rng(PERM_SEED)
            Tstar = (rng2.choice([-1.0, 1.0], size=(N_PERM, len(units))) * dj[None, :]).sum(axis=1)
            p = float((1 + np.sum(np.abs(Tstar) >= abs(dj.sum()) - 1e-12)) / (N_PERM + 1))
            ss_tot = ((y - y.mean()) ** 2).sum()
            per_seed = np.array([(Eb[k] - Ea[k]).sum() / ss_tot for k in range(len(seeds))])
            spos = float(np.mean(per_seed > 0))
            recs.append({"diag": d, "group": DIAG_NAME[d], "increment": label,
                         "with": a, "without": b, "n_participants": len(units),
                         "dR2_60": point, "lo": float(lo), "hi": float(hi),
                         "perm_p": p, "seed_pos_frac": spos})
        if recs:
            adj = holm(np.array([r["perm_p"] for r in recs]))
            for r, pa in zip(recs, adj):
                r["holm_p"] = float(pa)
                r["verdict"] = decide(r["dR2_60"], r["lo"], r["hi"], pa, r["seed_pos_frac"])
            rows += recs
    return pd.DataFrame(rows)


def main():
    oof = pd.read_csv(f"ablation_oof_{TAG}.csv", dtype={"pid": str})
    assert set(MODELS) - {PERSIST} <= set(oof.model.unique()), "missing a model"
    print(f"loaded ablation_oof_{TAG}.csv: {len(oof)} rows, "
          f"{oof.model.nunique()} models, {oof.seed.nunique()} seeds")

    d = descriptives(oof)
    d.to_csv("subgroup_descriptives.csv", index=False)
    print("\n=== DESCRIPTIVES ===\n" + d.to_string(index=False))

    p = performance(oof)
    p.to_csv("subgroup_performance.csv", index=False)
    show = p[p.horizon == 60][["group", "model", "n_meals", "R2", "RMSE", "RMSE_lo",
                               "RMSE_hi", "MAE", "bias", "skill_vs_persistence"]]
    print("\n=== PERFORMANCE @60 min ===\n" + show.round(3).to_string(index=False))

    g = gaps(oof)
    g.to_csv("subgroup_gaps.csv", index=False)
    gg = g[(g.horizon == 60) & (g.model.isin(["cgm_static", "full"]))]
    print("\n=== BETWEEN-GROUP GAPS @60 min (mg/dL) ===\n"
          + gg[["model", "comparison", "dRMSE", "dRMSE_lo", "dRMSE_hi",
                "dMAE"]].round(2).to_string(index=False))

    i = increments(oof)
    i.to_csv("subgroup_increments.csv", index=False)
    print("\n=== HEADLINE INCREMENTS WITHIN EACH GROUP @60 min ===\n"
          + i[["group", "increment", "n_participants", "dR2_60", "lo", "hi",
               "holm_p", "verdict"]].round(4).to_string(index=False))

    print("\nR2 is a WITHIN-group statistic: the target SD differs by group "
          "(39.8 / 42.0 / 63.9 mg/dL at 60 min), so a higher subgroup R2 does NOT mean\n"
          "lower error. Use RMSE/MAE to compare groups, and skill_vs_persistence for a\n"
          "scale-free read. Subgroup increments have 11-16 participants: INCONCLUSIVE is\n"
          "the expected result and is not evidence of absence.")
    print("\nSaved: subgroup_descriptives.csv, subgroup_performance.csv, "
          "subgroup_gaps.csv, subgroup_increments.csv")


if __name__ == "__main__":
    main()
