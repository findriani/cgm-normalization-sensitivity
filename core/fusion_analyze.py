"""
fusion_analyze.py  -- participant-clustered inference for the fusion grid
=========================================================================
Runs AFTER fusion_run.py. Reuses the SAME statistical core as ablation_analyze.py
(event-weighted mean-single-run ΔR²60, participant-cluster bootstrap CIs, participant
sign-flip permutation p on SUMMED per-participant error diffs -- aligned to the point
estimand -- equivalence-first decision, δ=0.02 R²60). Methodology is identical across
the whole paper.

Review fixes reflected here:
  * MANIFEST is validated against the plan: smoke=False, exact emitted configs, exact
    seed/row totals -- a completed one-seed smoke can no longer pass as "integrity OK".
  * Multiplicity: PRIMARY family Holm-corrected. Equivalence/NEGLIGIBLE claims are
    DESCRIPTIVE except the single pre-declared CONFIRMATORY external equivalence
    (cnn_gru__film vs rf_highres). A pre-declared COMPOSITE (film+ssl+retrieval) vs RF
    is also reported (the system most likely to beat RF).
  * Activity is IDENTIFIED, not read off the raw gate: film vs film+nodyn (hard gate)
    and film vs film+dynablate (frozen test-time dyn zero).
  * CLEAN=capped is a fusion sensitivity for the RF-ablation-conditional design brief.

Outputs: fusion_model_summary.csv, fusion_per_seed_r2.csv, fusion_verdict.csv,
         fusion_sensitivity.csv (across CLEAN tags)
=========================================================================
"""
import os
import glob
import json
import numpy as np
import pandas as pd

from ablation_analyze import (holm, per_seed_pooled, build_ref, contrast_stats, decide,
                              integrity, MARGIN_R2, WIN_FRAC, PRIMARY_H, HORIZONS, N_FOLDS)

PRIMARY_TAG = "raw"                 # primary fusion run (honest raw meal); capped = sensitivity
RF_REF = "rf_highres"
CHAMPION = "cnn_gru__film"          # pre-declared DL champion for the confirmatory external test
COMPOSITE = "cnn_gru__film+ssl+retrieval"

PRIMARY = [
    ("cnn_gru__film",           "cnn_gru__concat", "FiLM beats concat"),
    ("cnn_gru__xattn",          "cnn_gru__concat", "cross-attention beats concat"),
    ("cnn_gru__film+ssl",       "cnn_gru__film",   "SSL pretraining adds"),
    ("cnn_gru__film+retrieval", "cnn_gru__film",   "retrieval adds"),
]
SECONDARY = [
    ("cnn_gru__gate",    "cnn_gru__concat",       "GLU gate beats concat"),
    ("bilstm__concat",   "cnn_gru__concat",       "encoder: bilstm vs cnn_gru"),
    ("gru__concat",      "cnn_gru__concat",       "encoder: gru vs cnn_gru"),
    ("cnn__concat",      "cnn_gru__concat",       "encoder: cnn vs cnn_gru"),
    ("tcn__concat",      "cnn_gru__concat",       "encoder: tcn vs cnn_gru"),
    ("cnn_gru__film",    "cnn_gru__film+nodyn",   "activity adds (film vs no_dyn)"),
    ("cnn_gru__film",    "cnn_gru__film+dynablate","dyn used at test (frozen ablation)"),
    ("cnn_gru__film",    "persistence_5min",      "FiLM beats 5-min persistence"),
    ("dl_current",       RF_REF,                  "paper's current DL vs RF"),
    ("cnn_gru__film+sslpool", "cnn_gru__film",    "pooled(transductive) SSL vs film"),
]


def decide_external(point, lo, hi, p, seed_pos):
    if lo >= -MARGIN_R2 and hi <= MARGIN_R2:
        return "EQUIVALENT"
    if point > 0 and lo > 0 and p < 0.05 and seed_pos >= WIN_FRAC:
        return "DL_WINS" if lo > MARGIN_R2 else "DL_WINS(small)"
    if point < 0 and hi < 0 and p < 0.05:
        return "RF_WINS" if hi < -MARGIN_R2 else "RF_WINS(small)"
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
    if recs:
        adj = holm(np.array([r["perm_p"] for r in recs]))
        for r, pa in zip(recs, adj):
            r["holm_p"] = float(pa)
            r["verdict"] = decide(r["dR2_60"], r["R2_lo"], r["R2_hi"], pa, r["seed_pos_frac"])
    return recs


def validate_manifest(configs, seeds, nSamp, raw, oof):
    man_path = f"fusion_manifest_{PRIMARY_TAG}.json"
    if not os.path.exists(man_path):
        raise SystemExit(f"{man_path} missing -- cannot verify the run matches the plan.")
    man = json.load(open(man_path))
    if man.get("smoke", True):
        raise SystemExit("[abort] manifest smoke=True -- this is a smoke run, not confirmatory.")
    exp = set(man["emitted_configs"])
    if set(configs) != exp:
        raise SystemExit(f"[abort] configs {sorted(set(configs) ^ exp)} differ from manifest.")
    nS = man["n_seeds"]
    if len(seeds) != nS:
        raise SystemExit(f"[abort] {len(seeds)} seeds present, manifest declares {nS}.")
    assert len(raw) == len(exp) * nS * man["n_folds"], f"raw rows {len(raw)} != {len(exp)*nS*man['n_folds']}"
    assert len(oof) == len(exp) * nS * nSamp, f"oof rows {len(oof)} != {len(exp)*nS*nSamp}"
    return man


def main():
    oof_path = f"fusion_oof_{PRIMARY_TAG}.csv"
    if not os.path.exists(oof_path):
        raise SystemExit(f"{oof_path} not found -- run fusion_run.py first.")
    oof = pd.read_csv(oof_path, dtype={"pid": str})
    raw = pd.read_csv(f"fusion_results_raw_{PRIMARY_TAG}.csv")
    configs, seeds, nSamp = integrity(oof, raw, PRIMARY_TAG)
    man = validate_manifest(configs, seeds, nSamp, raw, oof)
    print(f"[fusion/{PRIMARY_TAG}] integrity+manifest OK: {len(configs)} configs x {len(seeds)} seeds "
          f"x {N_FOLDS} folds, {nSamp} samples/seed (tf {man['tf']}, code {man['code_hash'][:8]})")

    ps_all = pd.concat([per_seed_pooled(oof, h) for h in HORIZONS], ignore_index=True)
    ps_all.to_csv("fusion_per_seed_r2.csv", index=False)
    summ = (ps_all.groupby(["model", "horizon"]).agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                                                     RMSE_mean=("RMSE", "mean")).reset_index())
    summ.to_csv("fusion_model_summary.csv", index=False)

    ref = build_ref(oof, PRIMARY_H)
    ps60 = ps_all[ps_all.horizon == PRIMARY_H]
    ps_pivot = ps60.pivot(index="seed", columns="model", values="R2")

    prim = evaluate(ref, ps_pivot, PRIMARY, "primary")
    sec = evaluate(ref, ps_pivot, SECONDARY, "secondary")

    ext = []
    for champ, tagname, conf in [(CHAMPION, "champion (confirmatory)", True),
                                 (COMPOSITE, "composite (pre-declared)", False)]:
        if champ in ref["preds"] and RF_REF in ref["preds"]:
            st = contrast_stats(ref, champ, RF_REF)
            sp = float((ps_pivot[champ] > ps_pivot[RF_REF]).mean())
            v = decide_external(st["dR2_60"], st["R2_lo"], st["R2_hi"], st["perm_p"], sp)
            ext.append({"family": "external", "increment": f"{tagname}: {champ} vs {RF_REF}",
                        "with": champ, "without": RF_REF, "seed_pos_frac": sp,
                        "holm_p": st["perm_p"], "confirmatory": conf, "verdict": v, **st})

    verdict = pd.DataFrame(prim + sec + ext)
    verdict.to_csv("fusion_verdict.csv", index=False)

    # ---------------- sensitivity across CLEAN tags ----------------
    other = sorted({os.path.basename(p)[len("fusion_oof_"):-4]
                    for p in glob.glob("fusion_oof_*.csv")} - {PRIMARY_TAG})
    sens = []
    for tag in other:
        try:
            o2 = pd.read_csv(f"fusion_oof_{tag}.csv", dtype={"pid": str})
            r2 = pd.read_csv(f"fusion_results_raw_{tag}.csv")
            integrity(o2, r2, tag)
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
        pd.DataFrame(sens).to_csv("fusion_sensitivity.csv", index=False)

    # ---------------- console ----------------
    pd.set_option("display.width", 190, "display.max_columns", 40)
    piv = summ.pivot(index="model", columns="horizon", values="R2_mean").sort_values(PRIMARY_H, ascending=False)
    print(f"\n=== fusion-grid ranking: pooled-OOF R2 by horizon (mean over {len(seeds)} seeds) ===")
    print(piv.round(3).to_string())
    if "act_gate" in raw.columns:
        g = raw[raw.act_gate.notna()].groupby("model").act_gate.mean().sort_values()
        if len(g):
            print("\n=== learned activity gate (DESCRIPTIVE only; use film-vs-no_dyn for the claim) ===")
            print(g.round(3).to_string())
    cols = ["increment", "with", "without", "dR2_60", "R2_lo", "R2_hi", "dRMSE_60",
            "perm_p", "holm_p", "seed_pos_frac", "verdict"]
    print("\n=== PRIMARY contrasts (Holm within family; sign-flip permutation p) ===")
    if prim:
        print(pd.DataFrame(prim)[cols].round(4).to_string(index=False))
    print("\n=== EXTERNAL vs RF (champion = CONFIRMATORY equivalence; composite pre-declared) ===")
    if ext:
        print(pd.DataFrame(ext)[cols + ["confirmatory"]].round(4).to_string(index=False))
    print("\n=== SECONDARY / exploratory (own Holm; equivalence here is DESCRIPTIVE) ===")
    if sec:
        print(pd.DataFrame(sec)[cols].round(4).to_string(index=False))
    if sens:
        print("\n=== SENSITIVITY: capped-vs-raw meal agree on PRIMARY? ===")
        print(pd.DataFrame(sens).round(4).to_string(index=False))
    print(f"\ndR2_60>0 => 'with' better. ADDS/DL_WINS needs dR2>0 & CI_lo>0 & Holm-p<0.05 & "
          f">={int(WIN_FRAC*100)}% seeds; '(small)' = significant but within +/-{MARGIN_R2}. "
          f"Only the champion external equivalence is confirmatory; other NEGLIGIBLE/EQUIVALENT "
          f"are descriptive.")
    print("Saved: fusion_model_summary.csv, fusion_per_seed_r2.csv, fusion_verdict.csv"
          + (", fusion_sensitivity.csv" if sens else ""))


if __name__ == "__main__":
    main()
