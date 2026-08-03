"""
deep_sens_analyze.py -- E2: the deep-model replication of E1
============================================================
Consumes deep_sens_oof.csv. The estimator IS E1's: `did_stats` is imported unchanged from
e1_analyze, and the statistical core (participant-cluster bootstrap, participant sign-flip
permutation, Holm, equivalence-first decision rule) comes from ablation_analyze. Nothing
is retuned for this experiment.

PRIMARY ESTIMAND
----------------
    DDR2 = [cgm_static - static_all]_global  -  [cgm_static - static_all]_subject_z

at 60 min, on matched out-of-fold rows. Positive means CGM contributes MORE under correct
normalization -- the direction E1 found with the Random Forest (+0.173 [0.117, 0.260]).
30 and 120 min are secondary and uncorrected.

A companion quantity is reported in mg/dL: the paired difference in FULL-MODEL error
between arms, `cgm_static@global` vs `cgm_static@subject_z`, with its own bootstrap CI.
That is a different estimand from the interaction -- it compares models, not increments --
and is labelled as such wherever it appears.

FOUR OUTCOMES, DERIVED FROM decide() -- NOT FROM THE INTERVAL ALONE
-------------------------------------------------------------------
`classify()` maps `ablation_analyze.decide()` onto E2's vocabulary, so the E2 verdict
inherits the paper's standard criteria and cannot be more permissive than they are:
equivalence is tested first, and a win additionally requires Holm-adjusted p < 0.05 and
seed consistency >= WIN_FRAC. An earlier version examined only the bootstrap CI and could
therefore print "CONFIRMED POSITIVE" for a result that failed Holm or the seed criterion.

  * EQUIVALENT   (decide: NEGLIGIBLE) the CI lies entirely within +/- MARGIN_R2. Positive
                 evidence that the deep-model effect is negligible. A finding.
  * INCONCLUSIVE nothing is established either way -- a statement about precision, not
                 about the world. Writing "no effect" here would be wrong.
  * CONFIRMED POSITIVE (decide: ADDS) the effect is not estimator-specific.
  * SUPPORTED OPPOSITE (decide: HURTS) not automatically a bug; check integrity, then
                 report.

Note one asymmetry inherited deliberately from `decide()`: the HURTS branch does not
require seed consistency, while ADDS does. That is the paper's existing rule and is left
untouched rather than quietly redefined for this experiment.

THE INVARIANCE CHECK MEANS SOMETHING DIFFERENT HERE
-----------------------------------------------------
In E1, `static_all` and `persistence_5min` were fitted independently per arm and their
bit-identity was real evidence that nothing but the CGM transform varied. In E2 those
configurations are FIT ONCE and written under both arm labels (see deep_sens_run), so
their equality is guaranteed by construction. The check below verifies THAT THE SHARING
WORKED -- it is not independent evidence. The real control is
`deep_sens_run.assert_statics_identical`, which compares the normalized static matrices
between arms and aborts the fit if they differ at all.

Outputs
  deep_sens_interaction.csv   the primary DDR2 tests, all horizons
  deep_sens_increments.csv    within-arm increments (descriptive context)
  deep_sens_paired_mgdl.csv   paired between-arm model comparisons in mg/dL
  deep_sens_models.csv        per-arm R2/RMSE per configuration, all horizons
============================================================
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

from ablation_analyze import (build_ref, contrast_stats, decide, holm, MARGIN_R2,
                              WIN_FRAC, PRIMARY_H, HORIZONS, N_BOOT, N_PERM,
                              BOOT_SEED, PERM_SEED)
from e1_analyze import did_stats            # the E1 estimator, imported not copied

REFERENCE_ARM, PUBLISHED_ARM = "global", "subject_z"
ARMS = [REFERENCE_ARM, PUBLISHED_ARM]
INCREMENTS = [
    ("cgm_static", "static_all", "CGM adds beyond context"),
    ("cgm_static", "cgm", "context adds beyond CGM"),
]
PRIMARY_INCREMENT = "CGM adds beyond context"
CGM_FREE = ["static_all", "persistence_5min"]
ALL_CONFIGS = ["persistence_5min", "cgm", "static_all", "cgm_static"]
N_SEEDS_DEFAULT = 15
N_FOLDS = 5
E1_INTERACTION = os.path.join(CORE, "e1_interaction.csv")


# ------------------------------------------------------------------ integrity
def check_unique(d):
    """Duplicate key deliberately EXCLUDES `fold` -- within one seed a sample belongs to
    exactly one test fold, so the same (arm, model, seed, sample_idx) appearing twice is a
    defect even when the fold numbers differ."""
    dup = d.duplicated(["arm", "model", "seed", "sample_idx"]).sum()
    if dup:
        raise SystemExit(f"[abort] {dup} duplicate (arm, model, seed, sample_idx) rows. "
                         f"Either an interrupted run was resumed without reconciliation, "
                         f"or a sample was assigned to two folds within one seed. Delete "
                         f"deep_sens_oof.csv and version.")
    return "no duplicate (arm, model, seed, sample_idx) rows -- fold excluded from the key"


def completeness(d, expect_seeds):
    """Return a list of problems; empty means the run is a complete confirmatory set."""
    problems = []
    seeds = sorted(int(s) for s in d.seed.unique())
    if seeds != list(range(expect_seeds)):
        problems.append(f"seeds present {seeds}, expected 0..{expect_seeds - 1}")

    cover = None
    for arm in ARMS:
        for cfg in ALL_CONFIGS:
            g = d[(d.arm == arm) & (d.model == cfg)]
            if g.empty:
                problems.append(f"cell {cfg}@{arm} entirely missing")
                continue
            for s in seeds:
                gs = g[g.seed == s]
                folds = sorted(int(f) for f in gs.fold.unique())
                if folds != list(range(N_FOLDS)):
                    problems.append(f"{cfg}@{arm} seed {s}: folds {folds}, expected "
                                    f"0..{N_FOLDS - 1}")
                idx = np.sort(gs.sample_idx.values)
                if cover is None:
                    cover = idx
                elif not np.array_equal(idx, cover):
                    problems.append(f"{cfg}@{arm} seed {s}: sample coverage differs from "
                                    f"the first cell ({len(idx)} vs {len(cover)} rows)")

    if cover is not None:
        expected = len(ARMS) * len(ALL_CONFIGS) * expect_seeds * len(cover)
        if len(d) != expected:
            problems.append(f"{len(d)} OOF rows, expected {expected} ({len(ARMS)} arms x "
                            f"{len(ALL_CONFIGS)} configs x {expect_seeds} seeds x "
                            f"{len(cover)} samples)")
    return problems


def check_manifest(expect_seeds, sfx):
    """The manifest records n_seeds at FIRST write, so extending an 8-seed run to 15
    leaves it stale. Reported, not trusted -- completeness is judged from the OOF data."""
    p = os.path.join(HERE, f"deep_sens_manifest{sfx}.json")
    if not os.path.exists(p):
        return "no manifest found"
    n = json.load(open(p)).get("n_seeds")
    if n == expect_seeds:
        return f"manifest n_seeds={n}, matches"
    return (f"manifest n_seeds={n} but {expect_seeds} expected -- stale manifest (written "
            f"at the first run, not updated when seeds were added)")


def check_sharing(ref):
    """CGM-free configurations were fit once and written for both arms, so they must be
    EXACTLY equal. This verifies the sharing, not the experiment (see module docstring)."""
    out = []
    for m in CGM_FREE:
        keys = [f"{m}@{a}" for a in ARMS if f"{m}@{a}" in ref["preds"]]
        if len(keys) < 2:
            continue
        worst = 0.0
        for s in ref["seeds"]:
            worst = max(worst, float(np.abs(ref["preds"][keys[0]][s]
                                            - ref["preds"][keys[1]][s]).max()))
        flag = "OK (shared fit)" if worst == 0.0 else "*** SHARING BROKEN ***"
        out.append(f"    {m:16s} max|diff| between arms = {worst:.3e}  {flag}")
    return "\n".join(out) if out else "    (no CGM-free configs found)"


def check_shared_flag(sfx):
    raw = os.path.join(HERE, f"deep_sens_raw{sfx}.csv")
    if not os.path.exists(raw):
        return "SKIPPED (raw results not found)"
    r = pd.read_csv(raw)
    if "shared_fit" not in r:
        return "SKIPPED (no shared_fit column)"
    bad = r[(r.model.isin(CGM_FREE)) & (~r.shared_fit.astype(bool))]
    if len(bad):
        return f"*** {len(bad)} CGM-free rows NOT marked shared_fit -- investigate ***"
    n = int(r.shared_fit.astype(bool).sum())
    return f"{n} rows carry shared_fit=True; all CGM-free rows are marked"


def compare_to_e1():
    """E1's Random-Forest estimate, so the deep number is read against the one it is meant
    to replicate rather than in isolation."""
    if not os.path.exists(E1_INTERACTION):
        return None
    e = pd.read_csv(E1_INTERACTION)
    e = e[(e.horizon == PRIMARY_H) & (e.increment == PRIMARY_INCREMENT)
          & (e.protocol_1 == REFERENCE_ARM) & (e.protocol_2 == PUBLISHED_ARM)]
    return None if e.empty else e.iloc[0]


# ------------------------------------------------------------------ estimands
def classify(point, lo, hi, holm_p, seed_pos):
    """E2's four outcomes, derived from ablation_analyze.decide().

    Built on `decide()` rather than on the interval so the criteria cannot diverge from
    the rest of the paper: equivalence first, then a win requires the CI to exclude zero
    AND Holm-adjusted p < 0.05 AND seed consistency >= WIN_FRAC."""
    if not np.isfinite(holm_p):
        return ""
    d = decide(point, lo, hi, holm_p, seed_pos)
    if d == "NEGLIGIBLE":
        return "EQUIVALENT"
    if d.startswith("ADDS"):
        return "CONFIRMED POSITIVE"
    if d.startswith("HURTS"):
        return "SUPPORTED OPPOSITE"
    return "INCONCLUSIVE"


def interaction(refs):
    recs = []
    for h, ref in refs.items():
        for a, b, label in INCREMENTS:
            if any(f"{m}@{p}" not in ref["preds"] for m in (a, b) for p in ARMS):
                continue
            st = did_stats(ref, a, b, REFERENCE_ARM, PUBLISHED_ARM)
            recs.append({"horizon": h, "increment": label,
                         "protocol_1": REFERENCE_ARM, "protocol_2": PUBLISHED_ARM, **st})
    r = pd.DataFrame(recs)
    if r.empty:
        return r
    r["holm_p"] = np.nan
    m = r.horizon == PRIMARY_H
    r.loc[m, "holm_p"] = holm(r.loc[m, "perm_p"].values)
    r["decide"] = [(decide(x.ddR2, x.lo, x.hi, x.holm_p, x.seed_pos_frac)
                    if np.isfinite(x.holm_p) else "") for x in r.itertuples()]
    r["outcome"] = [classify(x.ddR2, x.lo, x.hi, x.holm_p, x.seed_pos_frac)
                    for x in r.itertuples()]
    return r


def paired_mgdl(refs):
    """Between-arm model comparisons in mg/dL, with participant-cluster bootstrap CIs.

    NOT the interaction: this compares MODELS between arms, whereas the primary estimand
    compares INCREMENTS. Reported because a difference in R-squared units is hard to read
    clinically, and because `cgm` and `cgm_static` are the two configurations whose inputs
    actually change between arms."""
    recs = []
    for h, ref in refs.items():
        for cfg in ("cgm_static", "cgm"):
            ka, kb = f"{cfg}@{REFERENCE_ARM}", f"{cfg}@{PUBLISHED_ARM}"
            if ka not in ref["preds"] or kb not in ref["preds"]:
                continue
            st = contrast_stats(ref, ka, kb)
            recs.append({"horizon": h, "model": cfg,
                         "comparison": f"{REFERENCE_ARM} - {PUBLISHED_ARM}",
                         "dR2": st["dR2_60"], "lo": st["R2_lo"], "hi": st["R2_hi"],
                         "dRMSE_mgdl": st["dRMSE_60"],
                         "dRMSE_lo": st["RMSE_lo"], "dRMSE_hi": st["RMSE_hi"],
                         "perm_p": st["perm_p"]})
    return pd.DataFrame(recs)


def within_arm_increments(refs):
    recs = []
    for h, ref in refs.items():
        for arm in ARMS:
            for a, b, label in INCREMENTS:
                ka, kb = f"{a}@{arm}", f"{b}@{arm}"
                if ka not in ref["preds"] or kb not in ref["preds"]:
                    continue
                st = contrast_stats(ref, ka, kb)
                pos = np.mean([
                    np.sum((ref["y"] - ref["preds"][kb][s]) ** 2)
                    > np.sum((ref["y"] - ref["preds"][ka][s]) ** 2) for s in ref["seeds"]])
                recs.append({"horizon": h, "arm": arm, "increment": label,
                             "dR2": st["dR2_60"], "lo": st["R2_lo"], "hi": st["R2_hi"],
                             "dRMSE_mgdl": st["dRMSE_60"],
                             "dRMSE_lo": st["RMSE_lo"], "dRMSE_hi": st["RMSE_hi"],
                             "perm_p": st["perm_p"], "seed_pos_frac": float(pos)})
    return pd.DataFrame(recs)


def per_arm_models(d):
    rows = []
    for (arm, model), g in d.groupby(["arm", "model"]):
        for h in HORIZONS:
            yt, yp = f"y_true_{h}", f"y_pred_{h}"
            per_seed = []
            for _, gs in g.groupby("seed"):
                e = gs[yp].values - gs[yt].values
                ss = float(np.sum((gs[yt].values - gs[yt].values.mean()) ** 2))
                per_seed.append({"R2": 1.0 - float(np.sum(e ** 2)) / ss,
                                 "RMSE": float(np.sqrt(np.mean(e ** 2)))})
            f = pd.DataFrame(per_seed)
            rows.append({"arm": arm, "model": model, "horizon": h,
                         "R2": f.R2.mean(), "R2_sd": f.R2.std(),
                         "RMSE": f.RMSE.mean(), "RMSE_sd": f.RMSE.std()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ interpretation
VERDICT_TEXT = {
    "CONFIRMED POSITIVE": (
        "The attenuation is NOT estimator-specific: it reproduces with the deep model "
        "that produced the disputed published result. C1 may be stated without the "
        "Random-Forest caveat, and the mechanistic title becomes available."),
    "EQUIVALENT": (
        "POSITIVE EVIDENCE that the deep-model effect is negligible (the whole interval "
        "lies inside the pre-registered margin). This is a finding, not a failure: the "
        "attenuation is estimator-dependent -- substantial under the Random Forest, "
        "negligible under simple mid-fusion. C1 narrows to the RF *with evidence*, and "
        "the paper should say why that difference is interesting."),
    "INCONCLUSIVE": (
        "NOTHING is established either way -- this is a statement about precision, not "
        "about the world. C1 keeps the Random-Forest scope limit in EVERY claim "
        "(abstract, contributions, results, conclusion), and the text must say this is a "
        "power limitation, not evidence of absence. Do not write 'no effect'."),
    "SUPPORTED OPPOSITE": (
        "CGM contributes MORE under the published transform in the deep model. This is "
        "not automatically an error. Confirm the integrity checks above pass, then "
        "report it -- it would be a serious and publishable result, and it would mean "
        "the mechanism interacts with the estimator in a way E1 alone could not see."),
}


def interpret(inter, inc, mg):
    p = inter[(inter.horizon == PRIMARY_H) & (inter.increment == PRIMARY_INCREMENT)]
    if p.empty:
        return "cannot interpret: the primary interaction is missing"
    x = p.iloc[0]
    g = inc[(inc.horizon == PRIMARY_H)
            & (inc.increment == PRIMARY_INCREMENT)].set_index("arm")
    lines = []
    for arm, tag in ((REFERENCE_ARM, "reference "), (PUBLISHED_ARM, "published ")):
        if arm in g.index:
            r = g.loc[arm]
            lines.append(f"  {tag} ({arm:11s}) dR2 = {r.dR2:+.4f} "
                         f"[{r.lo:+.4f}, {r.hi:+.4f}]")
    lines.append(f"  INTERACTION (paired)        DDR2 = {x.ddR2:+.4f} "
                 f"[{x.lo:+.4f}, {x.hi:+.4f}]")
    lines.append(f"                              Holm p = {x.holm_p:.4g}   "
                 f"seeds {x.seed_pos_frac:.0%} (WIN_FRAC = {WIN_FRAC:.0%})   "
                 f"decide = {x.decide}")

    e1 = compare_to_e1()
    if e1 is not None:
        lines.append(f"  E1 (Random Forest, same estimand)  DDR2 = {e1.ddR2:+.4f} "
                     f"[{e1.lo:+.4f}, {e1.hi:+.4f}]")

    f = mg[(mg.horizon == PRIMARY_H) & (mg.model == "cgm_static")]
    if not f.empty:
        r = f.iloc[0]
        lines += ["", f"  Companion (NOT the interaction): full model {REFERENCE_ARM} "
                      f"vs {PUBLISHED_ARM}",
                  f"    dRMSE = {r.dRMSE_mgdl:+.3f} mg/dL "
                  f"[{r.dRMSE_lo:+.3f}, {r.dRMSE_hi:+.3f}]  "
                  f"(positive = {REFERENCE_ARM} has the LOWER error)"]

    lines.append("")
    lines.append(f"  => {x.outcome}. {VERDICT_TEXT.get(x.outcome, '(no verdict)')}")
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

    path = os.path.join(HERE, f"deep_sens_oof{sfx}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"[abort] {path} not found -- run deep_sens_run.py first")
    d = pd.read_csv(path, dtype={"pid": str})
    print(f"loaded {os.path.basename(path)}: {len(d)} rows | arms "
          f"{sorted(d.arm.unique())} | configs {sorted(d.model.unique())} "
          f"| seeds {sorted(d.seed.unique())}")

    print("\n=== INTEGRITY ===")
    print(f"  {check_unique(d)}")
    print(f"  shared-fit bookkeeping: {check_shared_flag(sfx)}")
    print(f"  manifest: {check_manifest(expect, sfx)}")

    tagged = d.copy()
    tagged["model"] = tagged.model + "@" + tagged.arm
    refs = {h: build_ref(tagged, h) for h in HORIZONS}
    print("  CGM-free configurations (fit ONCE, written for both arms):")
    print(check_sharing(refs[PRIMARY_H]))
    print("    NOTE: this verifies the SHARING, not the experiment. The control that the")
    print("    arms differ only in the CGM transform is the static-matrix assertion in")
    print("    deep_sens_run.assert_statics_identical, which aborts the fit if it fails.")

    problems = completeness(d, expect)
    if problems:
        msg = (f"  INCOMPLETE -- {len(problems)} problem(s) against a {expect}-seed "
               f"confirmatory run:\n" + "\n".join(f"    - {p}" for p in problems[:20]))
        if len(problems) > 20:
            msg += f"\n    ... and {len(problems) - 20} more"
        if not args.allow_partial:
            raise SystemExit("\n[abort] this is not a complete confirmatory run.\n" + msg
                             + "\n\n  Finish the run, or pass --allow-partial to inspect "
                               "it (the pre-committed\n  interpretation is suppressed in "
                               "that mode).")
        print(msg)
    else:
        print(f"  COMPLETE: {expect} seeds x {N_FOLDS} folds x {len(ARMS)} arms x "
              f"{len(ALL_CONFIGS)} configs, identical sample coverage in every cell")

    mods = per_arm_models(d)
    mods.to_csv(os.path.join(HERE, f"deep_sens_models{sfx}.csv"), index=False)
    print(f"\n=== R2 at {PRIMARY_H} min, by arm ===")
    print(mods[mods.horizon == PRIMARY_H].pivot(index="arm", columns="model", values="R2")
          .reindex(ARMS).round(3).to_string())

    inc = within_arm_increments(refs)
    inc.to_csv(os.path.join(HERE, f"deep_sens_increments{sfx}.csv"), index=False)
    print(f"\n=== WITHIN-ARM INCREMENTS at {PRIMARY_H} min (descriptive) ===")
    print(inc[inc.horizon == PRIMARY_H][["arm", "increment", "dR2", "lo", "hi",
                                         "dRMSE_mgdl", "perm_p", "seed_pos_frac"]]
          .round(4).to_string(index=False))

    mg = paired_mgdl(refs)
    mg.to_csv(os.path.join(HERE, f"deep_sens_paired_mgdl{sfx}.csv"), index=False)
    print(f"\n=== PAIRED BETWEEN-ARM MODEL COMPARISON in mg/dL "
          f"(companion, NOT the interaction) ===")
    print(mg[mg.horizon == PRIMARY_H][["model", "comparison", "dRMSE_mgdl", "dRMSE_lo",
                                       "dRMSE_hi", "dR2", "perm_p"]]
          .round(4).to_string(index=False))

    inter = interaction(refs)
    inter.to_csv(os.path.join(HERE, f"deep_sens_interaction{sfx}.csv"), index=False)
    print(f"\n=== PRIMARY: cross-arm interaction ({REFERENCE_ARM} vs {PUBLISHED_ARM}) ===")
    print("  positive DDR2 = CGM contributes MORE under correct normalization")
    print(inter[["horizon", "increment", "ddR2", "lo", "hi", "perm_p", "holm_p",
                 "seed_pos_frac", "decide", "outcome"]].round(4).to_string(index=False))

    if problems:
        print("\n=== PRE-COMMITTED INTERPRETATION: SUPPRESSED ===")
        print("  The run is incomplete. Numbers above are provisional and must not be "
              "quoted.")
    else:
        print("\n=== PRE-COMMITTED INTERPRETATION ===")
        print(interpret(inter, inc, mg))

    print(f"\n  Margin: MARGIN_R2 = {MARGIN_R2}, WIN_FRAC = {WIN_FRAC}, both inherited "
          f"from the primary\n  ablation and not chosen here. To express the margin in "
          f"mg/dL, derive it from the\n  paired dRMSE bounds above rather than from a "
          f"fixed conversion -- the R-squared-to-\n  RMSE mapping depends on the "
          f"operating error level and is not a constant.")
    print(f"\nSaved: deep_sens_interaction{sfx}.csv, deep_sens_increments{sfx}.csv, "
          f"deep_sens_paired_mgdl{sfx}.csv, deep_sens_models{sfx}.csv")


if __name__ == "__main__":
    main()
