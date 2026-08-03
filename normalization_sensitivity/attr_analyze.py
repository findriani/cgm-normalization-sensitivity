"""
attr_analyze.py -- E4: does CGM SHAPE add anything once the LEVEL is known?
==========================================================================
Consumes attr_oof.csv. The statistical core (participant-cluster bootstrap, participant
sign-flip permutation, Holm, equivalence-first verdicts) is imported unchanged from
ablation_analyze, so nobody can claim the statistics were retuned for this experiment.

PRIMARY TEST
------------
    shape beyond level  =  (L_S_C) - (L_C)     at 60 min

with participant-clustered CI, the pre-registered equivalence margin (MARGIN_R2 = 0.02,
inherited from E1), and Holm correction within the E4 family at the confirmatory horizon.
30 and 120 min are secondary and uncorrected. Paired dRMSE in mg/dL, with its own
bootstrap CI, is reported for every contrast alongside dR2.

SECONDARY CONTRASTS
-------------------
    level only      (L_C)  - C       how much is just the baseline
    shape only      (S_C)  - C       E1 predicts ~ 0
    level statistic (L_C)  - (Lm_C)  does the last bin beat the 12-bin mean?

THE FAMILY IS FOUR, NOT FIVE. The level statistic is now the FINAL BIN (see the anchor
change documented in attr_run.py). Consequently the primary contrast IS the single-
reading test: L_S_C - L_C compares the full 12-bin window against that one reading.
There is no separate single-reading arm to report, and Holm corrects over four.

`level statistic` is retained, and stated in the new direction, because it is the
pre-registered comparison that motivated the switch. Do not drop it -- it is the
evidence for the change.

ONE DECISION RULE, USED ONCE
-----------------------------
Verdicts come from `ablation_analyze.decide()`, which applies equivalence first and
requires Holm-adjusted p < 0.05 AND seed consistency (WIN_FRAC) before calling anything a
win. The narrative block at the end READS THAT VERDICT rather than re-deriving a decision
from the confidence interval -- an earlier version re-derived it and could therefore print
a stronger conclusion than the CSV recorded.

SCOPE -- do not exceed it
-------------------------
Every statement here is "under the prespecified Random Forest, on CGMacros, with an
absolute postprandial target". A null shape effect does not establish that sequence
modelling is unnecessary in general, that one reading suffices universally, or that CGM
morphology carries no information. Those need a sequence model, other cohorts, other
targets.

Outputs
  attr_contrasts.csv   all contrasts, all horizons, dR2 and dRMSE with CIs, verdicts
  attr_models.csv      per-configuration R2/RMSE, all horizons
==========================================================================
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

from ablation_analyze import (build_ref, contrast_stats, decide, holm, MARGIN_R2,
                              WIN_FRAC, PRIMARY_H, HORIZONS, N_BOOT, N_PERM,
                              BOOT_SEED, PERM_SEED)
from attr_run import E4_CONFIGS, N_SEEDS_DEFAULT, N_FOLDS

# (better, worse, label, is_primary)
CONTRASTS = [
    ("L_S_C", "L_C",   "shape beyond level (= full window vs one reading)", True),
    ("L_C",   "C",     "level only", False),
    ("S_C",   "C",     "shape only", False),
    ("L_C",   "Lm_C",  "level statistic: last bin vs 12-bin mean", False),
]

PRIMARY = "shape beyond level (= full window vs one reading)"
# The level statistic is the final bin, so the primary contrast already IS the
# single-reading test. There is no separate arm; SINGLE_READING is deliberately unset.
SINGLE_READING = None
ABL_OOF = "ablation_oof_rf_capped.csv"
SCOPE = ("under the prespecified Random Forest, on CGMacros, with an absolute "
         "postprandial target")


# ------------------------------------------------------------------ integrity
def check_unique(d):
    """Duplicate key deliberately EXCLUDES `fold`.

    Within one seed a sample belongs to exactly one test fold, so the same
    (model, seed, sample_idx) appearing twice is a defect even when the fold numbers
    differ -- which a key including `fold` would not catch."""
    dup = d.duplicated(["model", "seed", "sample_idx"]).sum()
    if dup:
        raise SystemExit(f"[abort] {dup} duplicate (model, seed, sample_idx) rows. Either "
                         f"an interrupted run was resumed without reconciliation, or a "
                         f"sample was assigned to two folds within one seed. Delete "
                         f"attr_oof.csv and version.")
    return "no duplicate (model, seed, sample_idx) rows -- fold excluded from the key"


def completeness(d, expect_seeds):
    """Return a list of problems; empty means the run is a complete confirmatory set."""
    problems = []
    configs = list(E4_CONFIGS)

    seeds = sorted(int(s) for s in d.seed.unique())
    if seeds != list(range(expect_seeds)):
        problems.append(f"seeds present {seeds}, expected 0..{expect_seeds - 1}")

    cover = None
    for cfg in configs:
        g = d[d.model == cfg]
        if g.empty:
            problems.append(f"configuration {cfg!r} entirely missing")
            continue
        for s in seeds:
            gs = g[g.seed == s]
            folds = sorted(int(f) for f in gs.fold.unique())
            if folds != list(range(N_FOLDS)):
                problems.append(f"{cfg} seed {s}: folds {folds}, expected "
                                f"0..{N_FOLDS - 1}")
            idx = np.sort(gs.sample_idx.values)
            if cover is None:
                cover = idx
            elif not np.array_equal(idx, cover):
                problems.append(f"{cfg} seed {s}: sample coverage differs from the "
                                f"first cell ({len(idx)} vs {len(cover)} rows)")

    if cover is not None:
        expected = len(configs) * expect_seeds * len(cover)
        if len(d) != expected:
            problems.append(f"{len(d)} OOF rows, expected {expected} "
                            f"({len(configs)} configs x {expect_seeds} seeds x "
                            f"{len(cover)} samples)")
    return problems


def check_manifest(expect_seeds, sfx):
    """The manifest records n_seeds at FIRST write. Extending an 8-seed run to 15 leaves
    it stale, so it is reported rather than trusted."""
    p = os.path.join(HERE, f"attr_manifest{sfx}.json")
    if not os.path.exists(p):
        return "no manifest found"
    import json
    n = json.load(open(p)).get("n_seeds")
    if n == expect_seeds:
        return f"manifest n_seeds={n}, matches"
    return (f"manifest n_seeds={n} but {expect_seeds} expected -- stale manifest (written "
            f"at the first run and not updated when seeds were added). Completeness is "
            f"judged from the OOF data, not this field.")


def verify_context_parity(d):
    """Configuration `C` uses the ablation's own context features, normalization and RF
    seed, so it must reproduce `static_all` from the reference ablation AT EVERY HORIZON.
    A mismatch means E4 is not sitting on the same pipeline as E1, which invalidates every
    comparison drawn between them -- so this ABORTS rather than warning."""
    path = None
    for cand in (os.path.join(CORE, ABL_OOF), os.path.join(HERE, ABL_OOF)):
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        return f"SKIPPED ({ABL_OOF} not found -- parity unverified)"

    a = pd.read_csv(path, dtype={"pid": str})
    a = a[a.model == "static_all"]
    c = d[d.model == "C"]
    seeds = sorted(set(c.seed) & set(a.seed))
    if not seeds or a.empty:
        return "SKIPPED (no overlapping seeds)"
    k = ["seed", "fold", "sample_idx"]
    cols = [f"y_pred_{h}" for h in HORIZONS]
    m = (c[c.seed.isin(seeds)][k + cols]
         .merge(a[a.seed.isin(seeds)][k + cols], on=k, suffixes=("_e4", "_abl")))
    if m.empty:
        return "SKIPPED (no overlapping cells)"
    worst, worst_h = 0.0, None
    for h in HORIZONS:
        diff = float(np.abs(m[f"y_pred_{h}_e4"] - m[f"y_pred_{h}_abl"]).max())
        if diff > worst:
            worst, worst_h = diff, h
    if worst >= 1e-4:
        raise SystemExit(
            f"[abort] context parity FAILED: `C` does not reproduce the ablation's "
            f"`static_all` (worst max|diff| = {worst:.3e} at h={worst_h} over {len(m)} "
            f"cells). E4 is not on the same pipeline as E1 and no comparison between them "
            f"is valid. Investigate before interpreting anything.")
    return (f"{len(m)} cells x {len(HORIZONS)} horizons | worst max|diff| = {worst:.3e} "
            f"(h={worst_h}) | MATCH")


def report_decomposition(sfx):
    """The decomposition checks run inside attr_run and abort on failure; this surfaces
    the recorded worst cases so the analysis output is self-contained."""
    raw = os.path.join(HERE, f"attr_results_raw{sfx}.csv")
    if not os.path.exists(raw):
        return "SKIPPED (raw results not found)"
    r = pd.read_csv(raw)
    if "recon_max_err" not in r:
        return "SKIPPED (no recon_max_err column)"
    n = r[["seed", "fold"]].drop_duplicates().shape[0]
    lvl = (f", worst |mean(Z) - (m-mu)/sigma| = {r.level_max_err.max():.3e}"
           if "level_max_err" in r else "")
    return (f"worst |m + sigma*S - x| = {r.recon_max_err.max():.3e} mg/dL{lvl}, "
            f"over {n} folds -- decomposition exact")


# ------------------------------------------------------------------ estimands
def contrasts(refs):
    recs = []
    for h, ref in refs.items():
        for a, b, label, is_primary in CONTRASTS:
            if a not in ref["preds"] or b not in ref["preds"]:
                continue
            st = contrast_stats(ref, a, b)
            pos = np.mean([
                np.sum((ref["y"] - ref["preds"][b][s]) ** 2)
                > np.sum((ref["y"] - ref["preds"][a][s]) ** 2) for s in ref["seeds"]])
            recs.append({"horizon": h, "contrast": label, "better": a, "worse": b,
                         "primary": is_primary,
                         "dR2": st["dR2_60"], "lo": st["R2_lo"], "hi": st["R2_hi"],
                         "dRMSE_mgdl": st["dRMSE_60"],
                         "dRMSE_lo": st["RMSE_lo"], "dRMSE_hi": st["RMSE_hi"],
                         "perm_p": st["perm_p"], "seed_pos_frac": float(pos)})
    r = pd.DataFrame(recs)
    if r.empty:
        return r
    # Holm within the E4 family, at the confirmatory horizon only.
    r["holm_p"] = np.nan
    m = r.horizon == PRIMARY_H
    r.loc[m, "holm_p"] = holm(r.loc[m, "perm_p"].values)
    r["verdict"] = [(decide(x.dR2, x.lo, x.hi, x.holm_p, x.seed_pos_frac)
                     if np.isfinite(x.holm_p) else "") for x in r.itertuples()]
    return r


def per_config_models(d):
    rows = []
    for model, g in d.groupby("model"):
        for h in HORIZONS:
            yt, yp = f"y_true_{h}", f"y_pred_{h}"
            per_seed = []
            for _, gs in g.groupby("seed"):
                e = gs[yp].values - gs[yt].values
                ss = float(np.sum((gs[yt].values - gs[yt].values.mean()) ** 2))
                per_seed.append({"R2": 1.0 - float(np.sum(e ** 2)) / ss,
                                 "RMSE": float(np.sqrt(np.mean(e ** 2)))})
            f = pd.DataFrame(per_seed)
            rows.append({"model": model, "label": E4_CONFIGS.get(model, {}).get("label", ""),
                         "horizon": h, "R2": f.R2.mean(), "R2_sd": f.R2.std(),
                         "RMSE": f.RMSE.mean(), "RMSE_sd": f.RMSE.std()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ interpretation
# Read STRAIGHT from decide()'s verdict. Do not re-derive a decision from the interval:
# decide() also requires Holm-adjusted p < 0.05 and seed consistency >= WIN_FRAC, and a
# narrative that looked only at the CI could announce a win the CSV does not record.
PRIMARY_TEXT = {
    "NEGLIGIBLE": (
        "EQUIVALENT within the pre-registered margin. The full 12-bin window buys nothing "
        "over the single meal-time reading, {scope}. Because the level IS that one "
        "reading, this contrast licenses the single-reading statement directly -- but "
        "only for this estimator, this cohort and this target."),
    "ADDS": (
        "SHAPE ADDS, by more than the margin. The window beyond the meal-time reading "
        "carries real signal; do NOT write that one reading suffices."),
    "ADDS(small)": (
        "SHAPE ADDS, but by less than the margin -- statistically detectable and "
        "practically small. Say both halves of that sentence, and do not write that one "
        "reading suffices."),
    "HURTS": (
        "The level-plus-shape representation predicts WORSE than level alone. Note that "
        "L_S_C strictly CONTAINS L_C's information plus the shape columns, so this cannot "
        "be an information effect: it is the extra columns degrading the forest's splits "
        "(more candidate features, same signal). Report it as a representation effect of "
        "the estimator, not as evidence that shape is harmful."),
    "HURTS(small)": (
        "Level-plus-shape is slightly worse than level alone, within the margin. As "
        "above, L_S_C contains strictly more information than L_C, so this is a "
        "representation effect on the forest rather than an information effect."),
    "INCONCLUSIVE": (
        "INCONCLUSIVE at this precision. Report the bound and claim neither an effect nor "
        "its absence."),
}

STAT_TEXT = {
    "NEGLIGIBLE": ("the two level statistics are EQUIVALENT within the margin. The switch "
                   "to the last bin is then a simplification, not an improvement -- say so."),
    "ADDS": "the last bin beats the 12-bin mean. This is the evidence for the anchor switch.",
    "ADDS(small)": ("the last bin beats the 12-bin mean by less than the margin. Report the "
                    "switch as small but consistent, not as a large gain."),
    "HURTS": ("the 12-bin MEAN beats the last bin -- the opposite of the archived run that "
              "motivated the switch. STOP and reconcile the two runs before reporting."),
    "HURTS(small)": ("the 12-bin mean is slightly better than the last bin, contradicting the "
                     "archived run. Reconcile before reporting."),
    "INCONCLUSIVE": ("inconclusive here. The archived mean-anchored run found the mean worse; "
                     "report the switch as motivated by that run and not re-confirmed."),
}


def interpret(con):
    p = con[(con.contrast == PRIMARY) & (con.horizon == PRIMARY_H)]
    if p.empty:
        return "cannot interpret: the primary contrast is missing"
    x = p.iloc[0]
    lines = [f"  PRIMARY  shape beyond level (L_S_C - L_C) at {PRIMARY_H} min",
             "    the level IS the final bin, so this is also the single-reading test",
             f"    dR2   = {x.dR2:+.4f} [{x.lo:+.4f}, {x.hi:+.4f}]",
             f"    dRMSE = {x.dRMSE_mgdl:+.3f} mg/dL "
             f"[{x.dRMSE_lo:+.3f}, {x.dRMSE_hi:+.3f}]",
             f"    Holm p = {x.holm_p:.4g}   seeds {x.seed_pos_frac:.0%} "
             f"(WIN_FRAC = {WIN_FRAC:.0%})   verdict = {x.verdict}"]
    lines.append("    => " + PRIMARY_TEXT.get(
        x.verdict, f"unrecognised verdict {x.verdict!r}").format(scope=SCOPE))

    s = con[(con.contrast == "level statistic: last bin vs 12-bin mean")
            & (con.horizon == PRIMARY_H)]
    if not s.empty:
        y = s.iloc[0]
        lines += ["", f"  Level statistic (L_C - Lm_C) at {PRIMARY_H} min "
                      f"-- the pre-registered comparison behind the anchor switch",
                  f"    dR2   = {y.dR2:+.4f} [{y.lo:+.4f}, {y.hi:+.4f}]   "
                  f"dRMSE = {y.dRMSE_mgdl:+.3f} mg/dL "
                  f"[{y.dRMSE_lo:+.3f}, {y.dRMSE_hi:+.3f}]",
                  f"    Holm p = {y.holm_p:.4g}   verdict = {y.verdict}",
                  "    => " + STAT_TEXT.get(
                      y.verdict, f"unrecognised verdict {y.verdict!r}").format(scope=SCOPE)]
    lines += ["", "  ANCHOR SWITCH: the level statistic was changed from the pre-registered",
              "  12-bin window mean to the final bin AFTER seeing the archived run in",
              "  results_mean_anchor/. Report it as a documented post-hoc change; the",
              "  comparison itself was pre-registered, the choice of primary was not."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--expect-seeds", type=int, default=N_SEEDS_DEFAULT,
                    help="seeds required for a complete confirmatory run")
    ap.add_argument("--allow-partial", action="store_true",
                    help="analyze an incomplete run; suppresses the pre-committed "
                         "interpretation")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    expect = 1 if args.smoke else args.expect_seeds

    path = os.path.join(HERE, f"attr_oof{sfx}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[abort] {path} not found -- run attr_run.py first")
    d = pd.read_csv(path, dtype={"pid": str})
    print(f"loaded {os.path.basename(path)}: {len(d)} rows | configs "
          f"{sorted(d.model.unique())} | seeds {sorted(d.seed.unique())}")

    print("\n=== INTEGRITY ===")
    print(f"  {check_unique(d)}")
    print(f"  decomposition: {report_decomposition(sfx)}")
    print(f"  context parity (`C` vs ablation `static_all`): {verify_context_parity(d)}")
    print(f"  manifest: {check_manifest(expect, sfx)}")

    problems = completeness(d, expect)
    if problems:
        msg = (f"  INCOMPLETE -- {len(problems)} problem(s) against a {expect}-seed "
               f"confirmatory run:\n" + "\n".join(f"    - {p}" for p in problems))
        if not args.allow_partial:
            raise SystemExit("\n[abort] this is not a complete confirmatory run.\n" + msg
                             + "\n\n  Finish the run, or pass --allow-partial to inspect "
                               "it (the pre-committed\n  interpretation is suppressed in "
                               "that mode).")
        print(msg)
    else:
        print(f"  COMPLETE: {expect} seeds x {N_FOLDS} folds x {len(E4_CONFIGS)} configs, "
              f"identical sample coverage in every cell")

    mods = per_config_models(d)
    mods.to_csv(os.path.join(HERE, f"attr_models{sfx}.csv"), index=False)
    print(f"\n=== R2 / RMSE at {PRIMARY_H} min, by configuration ===")
    print(mods[mods.horizon == PRIMARY_H]
          .set_index("model").reindex(list(E4_CONFIGS))
          [["label", "R2", "R2_sd", "RMSE"]].round(4).to_string())

    refs = {h: build_ref(d, h) for h in HORIZONS}
    con = contrasts(refs)
    con.to_csv(os.path.join(HERE, f"attr_contrasts{sfx}.csv"), index=False)

    print(f"\n=== CONTRASTS at {PRIMARY_H} min (Holm within the E4 family) ===")
    print(con[con.horizon == PRIMARY_H][["contrast", "dR2", "lo", "hi", "dRMSE_mgdl",
                                         "dRMSE_lo", "dRMSE_hi", "holm_p",
                                         "seed_pos_frac", "verdict"]]
          .round(4).to_string(index=False))
    print(f"\n=== ALL HORIZONS (30 and 120 secondary, uncorrected) ===")
    print(con[["horizon", "contrast", "dR2", "lo", "hi", "dRMSE_mgdl", "perm_p",
               "seed_pos_frac"]].round(4).to_string(index=False))

    if problems:
        print("\n=== PRE-COMMITTED INTERPRETATION: SUPPRESSED ===")
        print("  The run is incomplete. Numbers above are provisional and must not be "
              "quoted.")
    else:
        print("\n=== PRE-COMMITTED INTERPRETATION ===")
        print(interpret(con))

    print(f"\n  SCOPE: every statement above holds {SCOPE}. For novelty, write \"we found "
          f"no directly\n  matched decomposition under participant-independent evaluation "
          f"and an absolute\n  postprandial target\" -- not \"nobody has published this\". "
          f"Closely related\n  level-versus-trend comparisons exist (Pustozerov 2020, "
          f"Shen 2025).")
    print(f"\nSaved: attr_contrasts{sfx}.csv, attr_models{sfx}.csv")


if __name__ == "__main__":
    main()
