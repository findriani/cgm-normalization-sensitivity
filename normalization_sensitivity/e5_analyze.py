"""
e5_analyze.py -- E5: cross-modality transfer of the normalization protocol
==========================================================================
Same estimand, same inference, same decision rule as E1. Everything statistical is
imported from `ablation_analyze` and `e1_analyze`; nothing is reimplemented here, so the
two experiments cannot drift apart.

THE GATE
--------
E5 only measures DISTORTION if there is an attribution to distort. The primary ablation
already suggested activity/HR adds little beyond CGM + context. So the reference arm's
activity increment is evaluated FIRST:

    if the reference increment is NEGLIGIBLE or INCONCLUSIVE
        -> E5 is a SPECIFICITY check. Report that the protocol returns null where no
           contribution exists. Do NOT report the interaction as evidence that
           normalization is harmless for activity -- nothing was there to harm.
    else
        -> the interaction is interpretable as distortion, exactly as in E1.

This is pre-registered in e5_run.py's docstring and enforced below, so the reading cannot
be selected after seeing the numbers.

CONTROLS
--------
  * `static_all` and `cgm_static` use no activity channel and are refitted independently
    in every arm, so their predictions must be bit-identical across arms. Any difference
    means something other than the activity transform varies, and E5 is not controlled.
  * The `global` arm must reproduce the published reference ablation for the
    configurations they share.
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
                              WIN_FRAC, PRIMARY_H, HORIZONS)
from e1_analyze import did_stats
from e5_normalizers import ARMS, PROPERTIES, DYN_CHANNELS
from e5_run import E5_CONFIGS, DYN_FREE, N_SEEDS_DEFAULT, N_FOLDS

REFERENCE_ARM = "global"
INCREMENTS = [
    ("full", "cgm_static", "activity adds beyond CGM+context"),
    ("dyn_static", "static_all", "activity adds beyond context alone"),
]
PRIMARY_INCREMENT = "activity adds beyond CGM+context"
ABL_OOF = "ablation_oof_rf_capped.csv"
SCOPE = ("under the prespecified Random Forest, on CGMacros, with an absolute "
         "postprandial target")


# ------------------------------------------------------------------ integrity
def completeness(d, n_seeds):
    problems = []
    seeds = sorted(d.seed.unique())
    if len(seeds) != n_seeds:
        problems.append(f"expected {n_seeds} seeds, found {len(seeds)}: {seeds}")
    n_samp = d.sample_idx.nunique()
    for (arm, model), g in d.groupby(["arm", "model"]):
        if sorted(g.seed.unique()) != seeds:
            problems.append(f"{arm}/{model}: seed set differs")
        if g.fold.nunique() != N_FOLDS:
            problems.append(f"{arm}/{model}: {g.fold.nunique()} folds, expected {N_FOLDS}")
        for s, gs in g.groupby("seed"):
            if gs.sample_idx.nunique() != n_samp:
                problems.append(f"{arm}/{model} seed {s}: partial sample coverage")
    missing = ({(a, c) for a in ARMS for c in E5_CONFIGS}
               - set(map(tuple, d[["arm", "model"]].drop_duplicates().values)))
    if missing:
        problems.append(f"missing arm/config cells: {sorted(missing)}")
    if d.duplicated(["arm", "model", "seed", "sample_idx"]).sum():
        problems.append("duplicate (arm, model, seed, sample_idx) rows")
    return problems


def check_invariance(d):
    """Configurations with no activity input must be identical across arms."""
    print("  activity-free configurations (independently refitted in every arm):")
    worst = 0.0
    for cfg in DYN_FREE:
        g = d[d.model == cfg]
        piv = g.pivot_table(index=["seed", "fold", "sample_idx"], columns="arm",
                            values=f"y_pred_{PRIMARY_H}")
        base = piv[REFERENCE_ARM]
        m = max(float(np.abs(piv[a] - base).max()) for a in ARMS if a != REFERENCE_ARM)
        worst = max(worst, m)
        flag = "OK" if m == 0.0 else "*** MISMATCH ***"
        print(f"    {cfg:12s} max|diff| between arms = {m:.3e}  {flag}")
    if worst != 0.0:
        raise SystemExit("[abort] an activity-free configuration differs between arms. "
                         "Something other than the activity transform varies -- E5 is not "
                         "controlled and nothing below is interpretable.")
    return worst


def verify_reference(d):
    """The `global` arm must reproduce the published reference ablation."""
    path = os.path.join(CORE, ABL_OOF)
    if not os.path.exists(path):
        print(f"  reference parity: {ABL_OOF} not found -- SKIPPED")
        return
    abl = pd.read_csv(path, dtype={"pid": str})
    shared = sorted(set(E5_CONFIGS) & set(abl.model.unique()))
    worst, n = 0.0, 0
    for h in HORIZONS:
        for cfg in shared:
            a = (d[(d.arm == REFERENCE_ARM) & (d.model == cfg)]
                 .set_index(["seed", "fold", "sample_idx"])[f"y_pred_{h}"])
            bb = (abl[abl.model == cfg]
                  .set_index(["seed", "fold", "sample_idx"])[f"y_pred_{h}"])
            common = a.index.intersection(bb.index)
            if len(common) == 0:
                continue
            worst = max(worst, float(np.abs(a.loc[common] - bb.loc[common]).max()))
            n += len(common)
    print(f"  reference parity vs {ABL_OOF}: configs {shared} | {n} cells | "
          f"max|diff| = {worst:.3e} | {'MATCH' if worst == 0.0 else '*** MISMATCH ***'}")
    if worst != 0.0:
        raise SystemExit("[abort] the reference arm does not reproduce the published "
                         "ablation. E5 does not sit on the same pipeline as E1.")


# ------------------------------------------------------------------ estimands
def within_arm(d):
    rows = []
    for arm in ARMS:
        da = d[d.arm == arm]
        for h in HORIZONS:
            ref = build_ref(da, h)
            recs = []
            for a, b, lab in INCREMENTS:
                if a not in ref["preds"] or b not in ref["preds"]:
                    continue
                st = contrast_stats(ref, a, b)
                pos = np.mean([np.sum((ref["y"] - ref["preds"][b][s]) ** 2)
                               > np.sum((ref["y"] - ref["preds"][a][s]) ** 2)
                               for s in ref["seeds"]])
                recs.append({"arm": arm, "horizon": h, "increment": lab,
                             "dR2": st["dR2_60"], "lo": st["R2_lo"], "hi": st["R2_hi"],
                             "dRMSE_mgdl": st["dRMSE_60"], "perm_p": st["perm_p"],
                             "seed_pos_frac": float(pos)})
            if recs and h == PRIMARY_H:
                adj = holm(np.array([r["perm_p"] for r in recs]))
                for r, pa in zip(recs, adj):
                    r["holm_p"] = float(pa)
                    r["verdict"] = decide(r["dR2"], r["lo"], r["hi"], pa,
                                          r["seed_pos_frac"])
            rows += recs
    return pd.DataFrame(rows)


def interactions(d):
    rows = []
    for h in HORIZONS:
        dd = d.copy()
        dd["model"] = dd.model.astype(str) + "@" + dd.arm.astype(str)
        ref = build_ref(dd, h)
        recs = []
        for a, b, lab in INCREMENTS:
            for other in [x for x in ARMS if x != REFERENCE_ARM]:
                keys = [f"{m}@{p}" for m in (a, b) for p in (REFERENCE_ARM, other)]
                if any(k not in ref["preds"] for k in keys):
                    continue
                st = did_stats(ref, a, b, REFERENCE_ARM, other)
                recs.append({"horizon": h, "increment": lab,
                             "arm_1": REFERENCE_ARM, "arm_2": other, **st})
        if recs and h == PRIMARY_H:
            adj = holm(np.array([r["perm_p"] for r in recs]))
            for r, pa in zip(recs, adj):
                r["holm_p"] = float(pa)
                r["verdict"] = decide(r["ddR2"], r["lo"], r["hi"], pa,
                                      r["seed_pos_frac"])
        rows += recs
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the gate
def interpret(within, inter):
    w = within[(within.arm == REFERENCE_ARM) & (within.horizon == PRIMARY_H)
               & (within.increment == PRIMARY_INCREMENT)]
    if w.empty:
        return "cannot interpret: reference-arm increment missing"
    r = w.iloc[0]
    lines = [f"  GATE -- activity's contribution under the reference arm, h = {PRIMARY_H}",
             f"    dR2 = {r.dR2:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]   "
             f"Holm p = {r.holm_p:.4g}   seeds {r.seed_pos_frac:.0%}   "
             f"verdict = {r.verdict}"]

    # THREE branches, not two. A NEGATIVE increment is not a contribution: it means
    # including activity makes the forest worse. There is no attribution to attenuate in
    # that case either, so HURTS closes the gate just as a null does -- it simply closes
    # it for a different reason, and the text has to say which.
    verdict = str(r.verdict)
    gate_open = verdict.startswith("ADDS")
    if gate_open:
        lines += ["", "    GATE OPEN. Activity contributes positively under the reference",
                  "    arm, so the interaction below is interpretable as distortion,",
                  "    exactly as in E1."]
    elif verdict.startswith("HURTS"):
        lines += [
            "",
            "    GATE CLOSED (negative increment). Activity does not contribute under",
            "    correct normalization -- it measurably DEGRADES the model. Since L_full",
            "    strictly contains cgm_static's inputs plus the activity columns, this",
            "    cannot be an information effect; it is the extra columns diluting the",
            "    forest's splits, the same representation effect seen in E4.",
            "",
            "    There is no positive attribution for the transform to attenuate, so the",
            "    interaction is NOT a distortion measurement. Report E5 as a SPECIFICITY",
            "    result and state the negative increment explicitly rather than rounding",
            "    it to 'activity contributes nothing'.",
        ]
    else:
        lines += [
            "",
            "    GATE CLOSED (null increment). Activity does not measurably contribute",
            "    under correct normalization, so there is NO attribution for the transform",
            "    to distort. Report E5 as a SPECIFICITY result: the protocol, applied to a",
            f"    modality carrying no incremental signal {SCOPE}, returns null -- it does",
            "    not manufacture an effect. Evidence the procedure is not tuned to CGM.",
        ]
    if not gate_open:
        lines += ["",
                  "    DO NOT write that per-subject normalization is harmless for activity.",
                  "    Nothing positive was there to harm, and this design cannot separate",
                  "    'no distortion' from 'nothing to distort'."]

    p = inter[(inter.horizon == PRIMARY_H) & (inter.increment == PRIMARY_INCREMENT)]
    for _, x in p.iterrows():
        lines += ["", f"  INTERACTION  {x.arm_1} vs {x.arm_2}  ({x.increment})",
                  f"    DDR2 = {x.ddR2:+.4f} [{x.lo:+.4f}, {x.hi:+.4f}]   "
                  f"Holm p = {x.holm_p:.4g}   seeds {x.seed_pos_frac:.0%}   "
                  f"verdict = {x.verdict}"]
        if not gate_open:
            lines.append("    (uninterpretable as distortion -- see GATE CLOSED above)")

    lines += ["", f"  SCOPE: {SCOPE}. E5 is transfer across MODALITIES on one cohort;",
              "  it is not transfer across cohorts. Only an external dataset gives that."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    n_seeds = 1 if args.smoke else N_SEEDS_DEFAULT

    d = pd.read_csv(os.path.join(HERE, f"e5_oof{sfx}.csv"), dtype={"pid": str})
    print(f"loaded e5_oof{sfx}.csv: {len(d)} rows | arms {sorted(d.arm.unique())} | "
          f"configs {sorted(d.model.unique())} | seeds {sorted(d.seed.unique())}")

    print("\n=== INTEGRITY ===")
    check_invariance(d)
    verify_reference(d)
    probs = completeness(d, n_seeds)
    if probs:
        for p in probs:
            print(f"  INCOMPLETE: {p}")
        if not args.allow_partial:
            raise SystemExit("[abort] run is incomplete; finish it or pass --allow-partial "
                             "(which suppresses the interpretation).")
    else:
        print(f"  COMPLETE: {n_seeds} seeds x {N_FOLDS} folds x {len(ARMS)} arms "
              f"x {len(E5_CONFIGS)} configs")

    lvl_path = os.path.join(HERE, f"e5_level_diag{sfx}.csv")
    if os.path.exists(lvl_path):
        lv = pd.read_csv(lvl_path)
        print("\n=== ACTIVITY LEVEL RETAINED (model-free diagnostic, mean over folds) ===")
        print(lv.pivot_table(index="arm", columns="channel", values="level_retained")
              .round(4).to_string())
        print("  channels: 1 = HR, 2 = Calories, 3 = METs. Centering arms should be ~0.")

    print(f"\n=== R2 at {PRIMARY_H} min, by arm ===")
    rows = []
    for (arm, model), g in d.groupby(["arm", "model"]):
        rows.append({"arm": arm, "model": model,
                     "R2": 1 - ((g[f"y_true_{PRIMARY_H}"] - g[f"y_pred_{PRIMARY_H}"]) ** 2).sum()
                           / ((g[f"y_true_{PRIMARY_H}"] - g[f"y_true_{PRIMARY_H}"].mean()) ** 2).sum()})
    print(pd.DataFrame(rows).pivot_table(index="arm", columns="model", values="R2")
          .round(3).to_string())

    within = within_arm(d)
    within.to_csv(os.path.join(HERE, f"e5_increments{sfx}.csv"), index=False)
    print(f"\n=== WITHIN-ARM INCREMENTS at {PRIMARY_H} min ===")
    print(within[within.horizon == PRIMARY_H]
          [["arm", "increment", "dR2", "lo", "hi", "dRMSE_mgdl", "holm_p",
            "seed_pos_frac", "verdict"]].round(4).to_string(index=False))

    inter = interactions(d)
    inter.to_csv(os.path.join(HERE, f"e5_interaction{sfx}.csv"), index=False)
    print(f"\n=== INTERACTIONS (positive = increment LARGER under the reference arm) ===")
    print(inter[inter.horizon == PRIMARY_H]
          [["increment", "arm_1", "arm_2", "ddR2", "lo", "hi", "perm_p", "holm_p",
            "seed_pos_frac", "verdict"]].round(4).to_string(index=False))

    print("\n=== PRE-COMMITTED INTERPRETATION ===")
    print(interpret(within, inter))
    print(f"\nSaved: e5_increments{sfx}.csv, e5_interaction{sfx}.csv")


if __name__ == "__main__":
    main()
