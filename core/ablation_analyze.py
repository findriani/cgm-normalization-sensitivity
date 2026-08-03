"""
ablation_analyze.py  -- participant-clustered inference for the feature ablation
================================================================================
Runs AFTER ablation_run.py. Self-contained statistics (does not reuse the light-DL
seed-averaging, which estimated a cross-seed ENSEMBLE). Here every quantity targets
the SAME estimand the ranking reports: the MEAN SINGLE-RUN procedure.

For each pre-declared contrast (with_group vs without_group):
  * point ΔR²60 = mean over seeds of (pooled-R²_with,s - pooled-R²_without,s),
  * 95% CI: participant-cluster bootstrap of that mean-single-run ΔR² (and ΔRMSE mg/dL),
  * p-value: participant-level SIGN-FLIP PERMUTATION on per-participant squared-error
    differences (a proper paired test; the percentile-bootstrap tail is NOT used as p),
  * verdict: equivalence-first, with a distinction between statistically positive and
    MEANINGFULLY positive (whole CI beyond the margin).

Multiplicity: PRIMARY family Holm-corrected; SECONDARY family Holm-corrected
separately and labeled exploratory.

Scope note: this is incremental value UNDER A FIXED ESTIMATOR (RF primary). Run
ESTIMATOR=hgb / CLEAN=raw for sensitivity; their agreement is reported here.

Outputs: ablation_model_summary.csv (all horizons), ablation_per_seed_r2.csv,
         ablation_verdict.csv
================================================================================
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ------------------------- locked constants -------------------------
PRIMARY_TAG = "rf_capped"
PRIMARY_H = 60
HORIZONS = [30, 60, 120]
N_FOLDS = 5
MARGIN_R2 = 0.02          # equivalence margin; ~<1 mg/dL RMSE at this operating point (see PLAN)
WIN_FRAC = 0.80           # a win must hold in >= this fraction of seeds
N_BOOT = 5000
N_PERM = 10000
BOOT_SEED = 12345
PERM_SEED = 999

PRIMARY = [
    ("cgm_static",      "static_all",       "CGM adds beyond ALL static"),
    ("cgm_static",      "cgm",              "static adds beyond CGM"),
    ("full",            "cgm_static",       "activity/HR adds beyond CGM+static"),
    ("cgm_person_meal", "cgm_person",       "meal content adds beyond person+CGM"),
]
SECONDARY = [
    ("cgm_person",      "cgm",              "person adds beyond CGM (who-you-are)"),
    ("cgm_static",      "cgm_person_meal",  "time/context adds beyond CGM+person+meal"),
    ("cgm_person_meal", "cgm_meal",         "person adds beyond meal+CGM"),
    ("cgm_dyn",         "cgm",              "activity adds beyond CGM alone"),
    ("cgm",             "persistence_5min", "CGM model beats 5-min persistence floor"),
    ("cgm_static",      "persistence_5min", "CGM+static beats persistence floor"),
    ("full",            "persistence_5min", "full beats persistence floor"),
    ("persistence_1min","persistence_5min", "1-min vs 5-min baseline resolution"),
]


def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------- load + integrity -------------------------
def load_tag(tag):
    oof = pd.read_csv(f"ablation_oof_{tag}.csv", dtype={"pid": str})
    raw = pd.read_csv(f"ablation_raw_{tag}.csv")
    return oof, raw


def integrity(oof, raw, tag):
    configs = sorted(raw.model.unique()); seeds = sorted(raw.seed.unique())
    nC, nS, nSamp = len(configs), len(seeds), oof.sample_idx.nunique()
    assert (raw.groupby(["model", "seed"]).fold.nunique() == N_FOLDS).all(), f"[{tag}] a (model,seed) is missing folds"
    assert len(raw) == nC * nS * N_FOLDS, f"[{tag}] metric rows {len(raw)} != {nC*nS*N_FOLDS}"
    assert (oof.groupby(["model", "seed"]).sample_idx.nunique() == nSamp).all(), f"[{tag}] incomplete OOF coverage"
    assert oof.duplicated(["model", "seed", "sample_idx"]).sum() == 0, f"[{tag}] duplicate OOF predictions"
    assert len(oof) == nC * nS * nSamp, f"[{tag}] OOF rows {len(oof)} != {nC*nS*nSamp}"
    return configs, seeds, nSamp


# ------------------------- per-seed pooled R2 -------------------------
def per_seed_pooled(oof, horizon):
    yt, yp = f"y_true_{horizon}", f"y_pred_{horizon}"
    rows = []
    for (m, s), g in oof.groupby(["model", "seed"]):
        rows.append({"model": m, "seed": s, "horizon": horizon,
                     "R2": r2_score(g[yt], g[yp]), "RMSE": _rmse(g[yt].values, g[yp].values)})
    return pd.DataFrame(rows)


# ------------------------- reference tables for fast bootstrap -------------------------
def build_ref(oof, horizon):
    seeds = sorted(oof.seed.unique())
    yp = f"y_pred_{horizon}"
    m0 = oof.model.iloc[0]
    r0 = oof[(oof.model == m0) & (oof.seed == seeds[0])].sort_values("sample_idx")
    order = r0.sample_idx.values
    y = r0[f"y_true_{horizon}"].values.astype(float)
    pid = r0.pid.values
    preds = {}
    for (m, s), g in oof.groupby(["model", "seed"]):
        g = g.sort_values("sample_idx")
        assert np.array_equal(g.sample_idx.values, order), f"sample order mismatch: {m}/{s}"
        preds.setdefault(m, {})[s] = g[yp].values.astype(float)
    return {"order": order, "y": y, "pid": pid, "seeds": seeds, "preds": preds}


def contrast_stats(ref, a, b):
    """Mean-single-run ΔR²60 (a - b): point + cluster-bootstrap CI + ΔRMSE CI +
    participant sign-flip permutation p. Positive ΔR² => `a` (the with-group) better."""
    y, seeds, pid = ref["y"], ref["seeds"], ref["pid"]
    Pa = np.array([ref["preds"][a][s] for s in seeds])
    Pb = np.array([ref["preds"][b][s] for s in seeds])
    Ea = (y[None, :] - Pa) ** 2
    Eb = (y[None, :] - Pb) ** 2
    Dbar = (Eb - Ea).mean(axis=0)                       # per-sample mean error reduction (a vs b)
    units = np.unique(pid); idx_by = {u: np.where(pid == u)[0] for u in units}

    def dR2(sel):
        yt = y[sel]; ss = ((yt - yt.mean()) ** 2).sum()
        return Dbar[sel].sum() / ss if ss > 0 else np.nan

    def dRMSE(sel):
        return float(np.mean([np.sqrt(Ea[k, sel].mean()) - np.sqrt(Eb[k, sel].mean())
                              for k in range(len(seeds))]))

    allrows = np.arange(len(y))
    point_r2, point_rmse = dR2(allrows), dRMSE(allrows)

    rng = np.random.default_rng(BOOT_SEED)
    bR2 = np.empty(N_BOOT); bRM = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sel = np.concatenate([idx_by[u] for u in rng.choice(units, len(units), replace=True)])
        bR2[k] = dR2(sel); bRM[k] = dRMSE(sel)
    lo, hi = np.percentile(bR2, [2.5, 97.5]); rlo, rhi = np.percentile(bRM, [2.5, 97.5])

    # Per-participant SUMMED error reduction so the sign-flip test targets the SAME
    # event-weighted estimand as the point/CI: point dR2 = Dbar[all].sum()/ss_tot =
    # (sum_j dj)/ss_tot, so flipping a participant's sign flips its full contribution to
    # the numerator. (Using per-participant MEANs would test a participant-balanced
    # estimand and mismatch the event-weighted CI -- rows/participant range 1..54.)
    dj = np.array([Dbar[idx_by[u]].sum() for u in units])    # per-participant summed reduction
    T = dj.sum()
    rng2 = np.random.default_rng(PERM_SEED)
    Tstar = (rng2.choice([-1.0, 1.0], size=(N_PERM, len(units))) * dj[None, :]).sum(axis=1)
    perm_p = float((1 + np.sum(np.abs(Tstar) >= abs(T) - 1e-12)) / (N_PERM + 1))

    return {"dR2_60": point_r2, "R2_lo": float(lo), "R2_hi": float(hi),
            "dRMSE_60": point_rmse, "RMSE_lo": float(rlo), "RMSE_hi": float(rhi), "perm_p": perm_p}


def decide(point, lo, hi, p_adj, seed_pos):
    """Equivalence-first; distinguish meaningfully-positive (CI beyond margin) from
    merely statistically-positive."""
    if lo >= -MARGIN_R2 and hi <= MARGIN_R2:
        return "NEGLIGIBLE"
    if point > 0 and lo > 0 and p_adj < 0.05 and seed_pos >= WIN_FRAC:
        return "ADDS" if lo > MARGIN_R2 else "ADDS(small)"
    if point < 0 and hi < 0 and p_adj < 0.05:
        return "HURTS" if hi < -MARGIN_R2 else "HURTS(small)"
    return "INCONCLUSIVE"


def evaluate(ref, ps_pivot, pairs, family):
    recs = []
    for a, b, label in pairs:
        if a not in ref["preds"] or b not in ref["preds"]:
            continue
        st = contrast_stats(ref, a, b)
        seed_pos = float((ps_pivot[a] > ps_pivot[b]).mean())
        recs.append({"family": family, "increment": label, "with": a, "without": b,
                     "seed_pos_frac": seed_pos, **st})
    if recs:                                             # Holm within this family
        adj = holm(np.array([r["perm_p"] for r in recs]))
        for r, pa in zip(recs, adj):
            r["holm_p"] = float(pa)
            r["verdict"] = decide(r["dR2_60"], r["R2_lo"], r["R2_hi"], pa, r["seed_pos_frac"])
    return recs


def main():
    if not os.path.exists(f"ablation_oof_{PRIMARY_TAG}.csv"):
        raise SystemExit(f"ablation_oof_{PRIMARY_TAG}.csv not found -- run ablation_run.py first.")
    oof, raw = load_tag(PRIMARY_TAG)
    configs, seeds, nSamp = integrity(oof, raw, PRIMARY_TAG)
    print(f"[{PRIMARY_TAG}] integrity OK: {len(configs)} configs x {len(seeds)} seeds x {N_FOLDS} folds, "
          f"{nSamp} samples/seed")

    # all-horizon per-seed pooled + summary
    ps_all = pd.concat([per_seed_pooled(oof, h) for h in HORIZONS], ignore_index=True)
    ps_all.to_csv("ablation_per_seed_r2.csv", index=False)
    summ = (ps_all.groupby(["model", "horizon"]).agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                                                     RMSE_mean=("RMSE", "mean")).reset_index())
    summ.to_csv("ablation_model_summary.csv", index=False)

    ref = build_ref(oof, PRIMARY_H)
    ps60 = ps_all[ps_all.horizon == PRIMARY_H]
    ps_pivot = ps60.pivot(index="seed", columns="model", values="R2")

    prim = evaluate(ref, ps_pivot, PRIMARY, "primary")
    sec = evaluate(ref, ps_pivot, SECONDARY, "secondary")
    verdict = pd.DataFrame(prim + sec)
    verdict.to_csv("ablation_verdict.csv", index=False)

    # ---------------- sensitivity across estimator/clean tags ----------------
    other_tags = sorted({os.path.basename(p)[len("ablation_oof_"):-4]
                         for p in glob.glob("ablation_oof_*.csv")} - {PRIMARY_TAG})
    sens = []
    for tag in other_tags:
        try:
            o2, r2raw = load_tag(tag); integrity(o2, r2raw, tag)
            ref2 = build_ref(o2, PRIMARY_H)
            ps2 = per_seed_pooled(o2, PRIMARY_H).pivot(index="seed", columns="model", values="R2")
            for a, b, label in PRIMARY:
                if a in ref2["preds"] and b in ref2["preds"]:
                    st = contrast_stats(ref2, a, b)
                    v = decide(st["dR2_60"], st["R2_lo"], st["R2_hi"], st["perm_p"],
                               float((ps2[a] > ps2[b]).mean()))
                    sens.append({"tag": tag, "increment": label, "dR2_60": st["dR2_60"], "verdict": v})
        except Exception as e:
            print(f"  [sensitivity {tag}] skipped: {e}")
    if sens:
        pd.DataFrame(sens).to_csv("ablation_sensitivity.csv", index=False)

    # ---------------- console ----------------
    pd.set_option("display.width", 175, "display.max_columns", 40)
    piv = summ.pivot(index="model", columns="horizon", values="R2_mean").sort_values(PRIMARY_H, ascending=False)
    print(f"\n=== feature-set ranking: pooled-OOF R2 by horizon (mean over {len(seeds)} seeds) ===")
    print(piv.round(3).to_string())
    cols = ["increment", "with", "without", "dR2_60", "R2_lo", "R2_hi", "dRMSE_60",
            "perm_p", "holm_p", "seed_pos_frac", "verdict"]
    print("\n=== PRIMARY incremental tests (Holm within family; sign-flip permutation p) ===")
    if prim:
        print(pd.DataFrame(prim)[cols].round(4).to_string(index=False))
    print("\n=== SECONDARY / exploratory (own Holm family) ===")
    if sec:
        print(pd.DataFrame(sec)[cols].round(4).to_string(index=False))
    if sens:
        print("\n=== SENSITIVITY (do other estimator/clean settings agree on PRIMARY?) ===")
        print(pd.DataFrame(sens).round(4).to_string(index=False))
    print(f"\ndR2_60>0 => the 'with' group ADDS value.  ADDS needs dR2>0 & CI_lo>0 & Holm-p<0.05 "
          f"& >={int(WIN_FRAC*100)}% seeds; 'ADDS' = CI beyond +/-{MARGIN_R2}, 'ADDS(small)' = "
          f"significant but within margin. NEGLIGIBLE = whole CI within +/-{MARGIN_R2}.")
    print("Saved: ablation_model_summary.csv, ablation_per_seed_r2.csv, ablation_verdict.csv"
          + (", ablation_sensitivity.csv" if sens else ""))


if __name__ == "__main__":
    main()
