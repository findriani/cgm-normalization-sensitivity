"""
traditional_analyze.py -- builds the replacement for the paper's Table 7
========================================================================
Reads trad_oof.csv (this run) and attn_oof_raw.csv (the DL side, already fitted on the
IDENTICAL splits) and produces:

  trad_table7.csv     the replacement table: NRMSE/R2/RMSE/MARD at 60 min per method
  trad_table7.tex     the same, LaTeX, drop-in for sn-article.tex
  trad_paired.csv     DL vs each baseline, PAIRED, participant-cluster bootstrap + Holm
  trad_horizons.csv   the full 30/60/120 grid for the appendix

WHY PAIRED. The paper's Table 7 puts two independent columns side by side and reads a
130% gap off them. Because every model here saw the same participants in the same folds
under the same normalizer, the difference can be computed PER MEAL and bootstrapped over
participants. That is both far more precise and the only version that supports a claim
about one method beating another. The statistical core is imported from
ablation_analyze, unchanged, so the decision rule is the one used throughout the paper.

READ THE SIGN. dR2_60 > 0 means the DEEP model is better.
========================================================================
"""
import os
import numpy as np
import pandas as pd

from ablation_analyze import (build_ref, contrast_stats, decide, holm,
                              MARGIN_R2, WIN_FRAC, HORIZONS)
from common import Y_RANGE

TRAD_OOF = "trad_oof.csv"
TRAD_RAW = "trad_results_raw.csv"
# The attention study's OOF was downloaded into attn_outputs/; fall back to the cwd so
# this also works if the file is ever placed alongside the scripts.
ATTN_OOF = next((p for p in (os.path.join("attn_outputs", "attn_oof_raw.csv"),
                             "attn_oof_raw.csv") if os.path.exists(p)),
                "attn_oof_raw.csv")

# The DL rows to pull from the attention study. dl_current_binned is the model the PAPER
# actually reports in Table 7; the other two contextualize it.
DL_MODELS = {
    "dl_current_binned": "Mid-deep Fusion (this work)",
    "dl_current_highres": "Mid-deep Fusion (1-min input)",
    "cnn_sa__coattn": "Self-attention + co-attention",
}
PRIMARY_DL = "dl_current_binned"

LABEL = {
    "catboost@highres": ("CatBoost", "Ensemble"),
    "lightgbm@highres": ("LightGBM", "Ensemble"),
    "xgboost@highres": ("XGBoost", "Ensemble"),
    "hist_gbm@highres": ("Hist. Gradient Boosting", "Ensemble"),
    "random_forest@highres": ("Random Forest", "Ensemble"),
    "ridge@highres": ("Ridge", "Linear"),
    "svr_rbf@highres": ("SVR (RBF)", "Kernel"),
    "ridge@binned": ("Ridge (5-min input)", "Linear"),
    "random_forest@binned": ("Random Forest (5-min input)", "Ensemble"),
    "hist_gbm@binned": ("Hist. Gradient Boosting (5-min input)", "Ensemble"),
    "persistence_5min": ("Persistence (last reading)", "Baseline"),
    "dummy_mean": ("Dummy (train mean)", "Baseline"),
}
# Order of the printed table: DL first, then descending capability.
ORDER = ([PRIMARY_DL, "dl_current_highres", "cnn_sa__coattn"]
         + ["hist_gbm@highres", "catboost@highres", "lightgbm@highres", "xgboost@highres",
            "random_forest@highres", "ridge@highres", "svr_rbf@highres",
            "hist_gbm@binned", "random_forest@binned", "ridge@binned",
            "persistence_5min", "dummy_mean"])


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_pooled():
    """trad + DL OOF on the COMMON seeds, with the row alignment actually verified."""
    t = pd.read_csv(TRAD_OOF, dtype={"pid": str})
    if not os.path.exists(ATTN_OOF):
        raise SystemExit(f"[abort] {ATTN_OOF} not found -- the DL side of the comparison "
                         f"comes from the attention study's OOF.")
    a = pd.read_csv(ATTN_OOF, dtype={"pid": str})
    a = a[a.model.isin(DL_MODELS)]

    common = sorted(set(t.seed.unique()) & set(a.seed.unique()))
    if not common:
        raise SystemExit("[abort] no seeds shared between trad and attn OOF")
    t, a = t[t.seed.isin(common)], a[a.seed.isin(common)]

    # Hard check: for every shared (seed, fold) the two runs must hold the SAME test rows.
    # If make_repeated_splits or base_seed had drifted, this is where it shows up, and a
    # silent mismatch here would invalidate every paired number below.
    tk = t.groupby(["seed", "fold"]).sample_idx.apply(lambda s: tuple(sorted(set(s))))
    ak = a.groupby(["seed", "fold"]).sample_idx.apply(lambda s: tuple(sorted(set(s))))
    shared = tk.index.intersection(ak.index)
    assert len(shared) == len(tk), "trad has (seed,fold) cells the DL run lacks"
    bad = [k for k in shared if tk[k] != ak[k]]
    assert not bad, f"split mismatch on {len(bad)} cells, e.g. {bad[:3]} -- splits diverged"

    pooled = pd.concat([t, a], ignore_index=True)
    print(f"pooled OOF: {len(pooled)} rows | seeds {common} | "
          f"{pooled.model.nunique()} models | splits verified identical")
    return pooled, common


def summary_table(pooled, horizon):
    """Mean-single-run metrics: compute per seed, then average. Reported alongside the
    across-cell SD so the table is comparable in shape to the one it replaces."""
    yt, yp = f"y_true_{horizon}", f"y_pred_{horizon}"
    rng = float(Y_RANGE[HORIZONS.index(horizon)])
    rows = []
    for m, g in pooled.groupby("model"):
        per_seed = []
        for s, gs in g.groupby("seed"):
            e = gs[yp].values - gs[yt].values
            ss_res = float(np.sum(e ** 2))
            ss_tot = float(np.sum((gs[yt].values - gs[yt].values.mean()) ** 2))
            per_seed.append({
                "RMSE": float(np.sqrt(np.mean(e ** 2))),
                "R2": 1.0 - ss_res / ss_tot,
                "MAE": float(np.mean(np.abs(e))),
                "MARD": float(np.mean(np.abs(e) / np.maximum(np.abs(gs[yt].values), 1e-6)) * 100)})
        d = pd.DataFrame(per_seed)
        rows.append({"model": m, "horizon": horizon,
                     "NRMSE": d.RMSE.mean() / rng, "NRMSE_sd": d.RMSE.std() / rng,
                     "RMSE": d.RMSE.mean(), "RMSE_sd": d.RMSE.std(),
                     "R2": d.R2.mean(), "R2_sd": d.R2.std(),
                     "MAE": d.MAE.mean(), "MARD": d.MARD.mean()})
    return pd.DataFrame(rows)


def paired(pooled, horizon=60):
    """DL vs every baseline on the same meals. Holm across the baselines compared."""
    sub = pooled[pooled.horizon_ok] if "horizon_ok" in pooled else pooled
    ref = build_ref(sub, horizon)
    others = [m for m in ref["preds"] if m not in DL_MODELS]
    ps = {m: {} for m in ref["preds"]}
    for m in ref["preds"]:
        for s in ref["seeds"]:
            p = ref["preds"][m][s]
            ss_tot = float(np.sum((ref["y"] - ref["y"].mean()) ** 2))
            ps[m][s] = 1.0 - float(np.sum((ref["y"] - p) ** 2)) / ss_tot
    recs = []
    for b in others:
        st = contrast_stats(ref, PRIMARY_DL, b)
        seed_pos = float(np.mean([ps[PRIMARY_DL][s] > ps[b][s] for s in ref["seeds"]]))
        recs.append({"deep": PRIMARY_DL, "baseline": b,
                     "baseline_label": LABEL.get(b, (b, ""))[0],
                     "seed_pos_frac": seed_pos, **st})
    adj = holm(np.array([r["perm_p"] for r in recs]))
    for r, pa in zip(recs, adj):
        r["holm_p"] = float(pa)
        r["verdict"] = decide(r["dR2_60"], r["R2_lo"], r["R2_hi"], pa, r["seed_pos_frac"])
    return pd.DataFrame(recs).sort_values("dR2_60")


def to_latex(tab, path):
    rows = [m for m in ORDER if m in set(tab.model)]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Performance comparison at the 60-minute horizon. All methods use "
             r"identical participant-level splits, identical fold-wise normalization fitted "
             r"on training participants only, and an equal per-fold hyperparameter search "
             r"budget. NRMSE is RMSE divided by the target range; RMSE is in mg/dL. "
             r"Mean $\pm$ SD across 8 seeds.}",
             r"\label{tab:traditional_ml}",
             r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"\textbf{Method} & \textbf{Type} & \textbf{NRMSE} & \textbf{RMSE (mg/dL)} \\",
             r"\midrule"]
    for i, m in enumerate(rows):
        r = tab[tab.model == m].iloc[0]
        name, kind = LABEL.get(m, (DL_MODELS.get(m, m), "Deep learning"))
        if m in DL_MODELS:
            name, kind = DL_MODELS[m], "Deep learning"
        if i == len(DL_MODELS):
            lines.append(r"\midrule")
        lines.append(f"{name} & {kind} & {r.NRMSE:.3f}$\\pm${r.NRMSE_sd:.3f} & "
                     f"{r.RMSE:.1f}$\\pm${r.RMSE_sd:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    pooled, seeds = load_pooled()

    allh = pd.concat([summary_table(pooled, h) for h in HORIZONS], ignore_index=True)
    allh.to_csv("trad_horizons.csv", index=False)

    t60 = allh[allh.horizon == 60].copy()
    t60["order"] = t60.model.map({m: i for i, m in enumerate(ORDER)})
    t60 = t60.sort_values("order").drop(columns="order")
    t60.to_csv("trad_table7.csv", index=False)
    to_latex(t60, "trad_table7.tex")

    show = t60.copy()
    show["method"] = show.model.map(lambda m: DL_MODELS.get(m) or LABEL.get(m, (m,))[0])
    print("\n=== REPLACEMENT TABLE 7 (60 min) ===")
    print(show[["method", "NRMSE", "NRMSE_sd", "RMSE", "R2", "MAE", "MARD"]]
          .round(3).to_string(index=False))

    pr = paired(pooled)
    pr.to_csv("trad_paired.csv", index=False)
    print("\n=== PAIRED: deep model vs each baseline (positive = DEEP better) ===")
    print(pr[["baseline_label", "dR2_60", "R2_lo", "R2_hi", "dRMSE_60",
              "holm_p", "seed_pos_frac", "verdict"]].round(4).to_string(index=False))

    print(f"\nseeds used: {seeds}")
    print("Saved: trad_table7.csv, trad_table7.tex, trad_paired.csv, trad_horizons.csv")


if __name__ == "__main__":
    main()
