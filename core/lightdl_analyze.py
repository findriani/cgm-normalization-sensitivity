"""
lightdl_analyze.py  -- participant-clustered inference + the locked decision rule
======================================================================
Runs AFTER lightdl_run.py. Operates on the saved out-of-fold (OOF) predictions,
not on per-fold summary numbers, so the statistics respect the fact that the
independent unit is the PARTICIPANT (n=40), not the seed-fold cell (the
pseudoreplication the review flagged).

What it does:
  1. Per-seed POOLED-OOF R2 (each seed tests every participant exactly once, so
     one honest R2 per seed -- not a mean of per-fold R2s).
  2. SEED-AVERAGED OOF predictions (average each sample's prediction across seeds).
  3. PARTICIPANT-CLUSTER BOOTSTRAP of the paired difference (model - RF baseline)
     for R2_60 and RMSE_60: resample the 40 participants with replacement ->
     95% percentile CI + bootstrap p-value. Clusters = participants.
  4. Secondary sanity: NADEAU-BENGIO corrected resampled t-test on per-(seed,fold)
     R2_60 differences (accounts for train-set overlap across folds).
  5. LOCKED DECISION RULE with a pre-set EQUIVALENCE MARGIN, HOLM correction over
     the pre-declared PRIMARY family, and a per-seed sign-consistency check.

Primary family (pre-declared, Holm-corrected):
     tabattn  vs rf_binned      (does lightweight tabular attention beat trees on
                                 the tabular feature set where trees currently win?)
     tcn      vs rf_highres     (does a high-res temporal model beat a flattened-
                                 sequence RF given the same 60x1-min information?)
Secondary / exploratory (reported, NOT family-controlled): aux/gate variants and
dl_current.

Outputs:
  lightdl_per_seed_r2.csv, lightdl_model_summary.csv, lightdl_verdict.csv
======================================================================
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import t as tdist

OOF_CSV = "lightdl_oof.csv"
RAW_CSV = "lightdl_results_raw.csv"
CARD_CSV = "lightdl_model_cards.csv"

PRIMARY_H = 60                 # primary metric horizon
MARGIN_R2 = 0.02              # equivalence margin on R2_60 (~ observed run-to-run noise)
N_BOOT = 5000
BOOT_SEED = 12345
WIN_SEED_FRAC = 0.80          # a win must hold in >= this fraction of seeds

PRIMARY = [("tabattn", "rf_binned"), ("tcn", "rf_highres")]
SECONDARY = [("tabattn_aux", "rf_binned"), ("dl_current", "rf_binned"),
             ("tcn_aux", "rf_highres"), ("tcn_gate", "rf_highres"),
             ("tcn_aux_gate", "rf_highres")]


# ------------------------- helpers -------------------------
def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def per_seed_pooled(oof, horizon):
    """One pooled R2/RMSE per (model, seed): each seed covers all samples once."""
    yt, yp = f"y_true_{horizon}", f"y_pred_{horizon}"
    rows = []
    for (mdl, sd), g in oof.groupby(["model", "seed"]):
        rows.append({"model": mdl, "seed": sd, "n": len(g),
                     "R2": r2_score(g[yt], g[yp]), "RMSE": _rmse(g[yt].values, g[yp].values)})
    return pd.DataFrame(rows)


def seed_averaged(oof, horizon):
    """Average each sample's prediction across seeds -> one row per (model, sample)."""
    yt, yp = f"y_true_{horizon}", f"y_pred_{horizon}"
    g = oof.groupby(["model", "sample_idx"]).agg(pid=("pid", "first"),
                                                 y_true=(yt, "first"),
                                                 y_pred=(yp, "mean")).reset_index()
    return g


def cluster_bootstrap(dfa, dfb, metric="R2", n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired participant-cluster bootstrap of (model_a - model_b) for one metric.
    dfa/dfb: seed-averaged frames (columns pid, y_true, y_pred) on the SAME samples.
    Returns dict(point, lo, hi, p). For R2 higher is better; for RMSE lower is
    better (we report a - b, so RMSE point<0 means a better)."""
    a = dfa.sort_values("sample_idx"); b = dfb.sort_values("sample_idx")
    assert np.array_equal(a["sample_idx"].values, b["sample_idx"].values), "sample misalignment"
    pids = a["pid"].values
    yt = a["y_true"].values; pa = a["y_pred"].values; pb = b["y_pred"].values
    units = np.unique(pids)
    idx_by_unit = {u: np.where(pids == u)[0] for u in units}

    def stat(sel):
        if metric == "R2":
            return r2_score(yt[sel], pa[sel]) - r2_score(yt[sel], pb[sel])
        return _rmse(yt[sel], pa[sel]) - _rmse(yt[sel], pb[sel])

    full = np.arange(len(yt))
    point = stat(full)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        chosen = rng.choice(units, size=len(units), replace=True)
        sel = np.concatenate([idx_by_unit[u] for u in chosen])
        boots[k] = stat(sel)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p vs 0
    p = 2.0 * min(np.mean(boots >= 0), np.mean(boots <= 0))
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p))}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (keeps input order)."""
    order = np.argsort(pvals)
    m = len(pvals); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def nadeau_bengio(raw, a, b, horizon, n_test, n_train):
    """Corrected resampled t-test on per-(seed,fold) R2 differences."""
    col = f"R2_{horizon}"
    da = raw[raw.model == a].set_index(["seed", "fold"])[col]
    db = raw[raw.model == b].set_index(["seed", "fold"])[col]
    common = da.index.intersection(db.index)
    d = (da.loc[common] - db.loc[common]).values
    J = len(d)
    if J < 2 or np.var(d, ddof=1) == 0:
        return {"nb_mean": float(np.mean(d)) if J else float("nan"), "nb_t": float("nan"), "nb_p": float("nan")}
    corr = (1.0 / J + n_test / n_train)
    tstat = np.mean(d) / np.sqrt(np.var(d, ddof=1) * corr)
    p = 2 * tdist.sf(abs(tstat), df=J - 1)
    return {"nb_mean": float(np.mean(d)), "nb_t": float(tstat), "nb_p": float(p)}


def decide(point, lo, hi, p_adj, seed_pos_frac):
    """Locked decision, primary metric R2_60 (higher is better). Order matters:
    equivalence (TOST) is checked FIRST, so a statistically detectable but
    practically negligible gap (whole CI inside the margin) reads as EQUIVALENT,
    not as a 'win'. Only a gap that leaves the margin AND excludes 0 counts as a
    win for either side."""
    if lo >= -MARGIN_R2 and hi <= MARGIN_R2:              # CI entirely within margin
        return "EQUIVALENT"
    if point > 0 and lo > 0 and p_adj < 0.05 and seed_pos_frac >= WIN_SEED_FRAC:
        return "DL_WINS"
    if point < 0 and hi < 0 and p_adj < 0.05:
        return "RF_WINS"
    return "INCONCLUSIVE"


def _fold_sizes(oof):
    """Mean test/train SAMPLE sizes per fold (for the NB correction)."""
    n_total = oof["sample_idx"].nunique()
    # per (seed,fold) test size from any one model
    m0 = oof.model.iloc[0]
    g = oof[oof.model == m0].groupby(["seed", "fold"])["sample_idx"].nunique()
    n_test = float(g.mean()); n_train = float(n_total - n_test)
    return n_test, n_train


# ------------------------- main -------------------------
def main():
    if not os.path.exists(OOF_CSV):
        raise SystemExit(f"{OOF_CSV} not found -- run lightdl_run.py first.")
    oof = pd.read_csv(OOF_CSV, dtype={"pid": str})
    raw = pd.read_csv(RAW_CSV) if os.path.exists(RAW_CSV) else None
    cards = pd.read_csv(CARD_CSV) if os.path.exists(CARD_CSV) else None
    n_seeds = oof["seed"].nunique()

    # 1) per-seed pooled R2 (primary horizon) + across-seed summary
    ps = per_seed_pooled(oof, PRIMARY_H)
    ps.to_csv("lightdl_per_seed_r2.csv", index=False)
    summ = ps.groupby("model").agg(R2_60_mean=("R2", "mean"), R2_60_std=("R2", "std"),
                                   RMSE_60_mean=("RMSE", "mean"), n_seeds=("seed", "nunique")).reset_index()
    if cards is not None:
        summ = summ.merge(cards, on="model", how="left")
    if os.path.exists("lightdl_latency.csv"):        # optional CPU-latency table
        lat = pd.read_csv("lightdl_latency.csv")[["model", "cpu_latency_ms"]]
        summ = summ.merge(lat, on="model", how="left")
    summ = summ.sort_values("R2_60_mean", ascending=False)
    summ.to_csv("lightdl_model_summary.csv", index=False)

    # 2) seed-averaged OOF per model
    sa = seed_averaged(oof, PRIMARY_H)
    frames = {m: g.reset_index(drop=True) for m, g in sa.groupby("model")}

    # per-seed pooled R2 lookup for the sign-consistency check
    ps_pivot = ps.pivot(index="seed", columns="model", values="R2")
    n_test, n_train = _fold_sizes(oof)

    # 3-5) contrasts
    def evaluate(pairs, family):
        recs = []
        for a, b in pairs:
            if a not in frames or b not in frames:
                continue
            r2 = cluster_bootstrap(frames[a], frames[b], "R2")
            rmse = cluster_bootstrap(frames[a], frames[b], "RMSE")
            seed_pos = float((ps_pivot[a] > ps_pivot[b]).mean()) if a in ps_pivot and b in ps_pivot else float("nan")
            rec = {"family": family, "model": a, "vs": b,
                   "dR2_60": r2["point"], "R2_lo": r2["lo"], "R2_hi": r2["hi"], "R2_boot_p": r2["p"],
                   "dRMSE_60": rmse["point"], "RMSE_lo": rmse["lo"], "RMSE_hi": rmse["hi"],
                   "seed_pos_frac": seed_pos}
            if raw is not None:
                rec.update(nadeau_bengio(raw, a, b, PRIMARY_H, n_test, n_train))
            recs.append(rec)
        return recs

    prim = evaluate(PRIMARY, "primary")
    if prim:
        adj = holm(np.array([r["R2_boot_p"] for r in prim]))
        for r, pa in zip(prim, adj):
            r["holm_p"] = float(pa)
            r["decision"] = decide(r["dR2_60"], r["R2_lo"], r["R2_hi"], pa, r["seed_pos_frac"])
    sec = evaluate(SECONDARY, "secondary")
    for r in sec:                                        # secondary: raw p, no family control
        r["holm_p"] = float("nan")
        r["decision"] = decide(r["dR2_60"], r["R2_lo"], r["R2_hi"], r["R2_boot_p"], r["seed_pos_frac"])

    verdict = pd.DataFrame(prim + sec)
    verdict.to_csv("lightdl_verdict.csv", index=False)

    # ------------- console report -------------
    pd.set_option("display.width", 160, "display.max_columns", 30)
    print(f"\n=== model summary (pooled-OOF R2_60, mean over {n_seeds} seeds) ===")
    cols = [c for c in ["model", "R2_60_mean", "R2_60_std", "RMSE_60_mean", "n_params", "cpu_latency_ms"] if c in summ]
    print(summ[cols].to_string(index=False))
    print("\n=== PRIMARY family (Holm-corrected; participant-cluster bootstrap) ===")
    pcols = ["model", "vs", "dR2_60", "R2_lo", "R2_hi", "R2_boot_p", "holm_p",
             "seed_pos_frac", "dRMSE_60", "nb_p", "decision"]
    if prim:
        print(pd.DataFrame(prim)[[c for c in pcols if c in prim[0]]].to_string(index=False))
    print("\n=== SECONDARY / exploratory (not family-controlled) ===")
    if sec:
        print(pd.DataFrame(sec)[[c for c in pcols if c in sec[0]]].to_string(index=False))
    print(f"\nEquivalence margin: |dR2_60| <= {MARGIN_R2}.  n_seeds={n_seeds} (>=8 recommended for the final run).")
    print("DL_WINS needs: dR2>0 & CI_lo>0 & adj-p<0.05 & win in >=80% of seeds.")
    print("Saved: lightdl_per_seed_r2.csv, lightdl_model_summary.csv, lightdl_verdict.csv")


if __name__ == "__main__":
    main()
