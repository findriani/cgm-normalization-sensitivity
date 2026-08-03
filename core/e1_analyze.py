"""
e1_analyze.py -- does normalization change modality attribution?
================================================================
Consumes e1_oof.csv. The statistical core (participant-cluster bootstrap, participant
sign-flip permutation, Holm, equivalence-first verdicts) is imported unchanged from
ablation_analyze.

THE PRIMARY ESTIMAND IS AN INTERACTION, NOT TWO SEPARATE INCREMENTS
--------------------------------------------------------------------
The claim is not "CGM adds a lot under one protocol and little under another" measured
separately. It is that the increment ITSELF changes with normalization. So the primary
quantity is a paired difference-in-differences on matched OOF rows:

    DDR2 = [cgm_static - static_all]_global  -  [cgm_static - static_all]_subject_z

Every arm sees the same participants, folds, seeds and RF seeds, so this differences at the
level of the individual meal before any aggregation. Positive DDR2 means the increment is
LARGER under the reference protocol. Uncertainty is a participant-cluster bootstrap and
inference is a participant sign-flip permutation on summed per-participant differences --
the same estimand structure used everywhere else in the paper.

MECHANISM: PAIRED CONTRASTS, NOT FACTOR AVERAGES
-------------------------------------------------
Averaging arms as if they were factorial replicates would be wrong: the arm set is
unbalanced, leakage is confounded with the type of transform, and an average of point
estimates carries no uncertainty. Instead each mechanism gets one paired contrast that
holds everything else fixed:

    subject_scale vs subject_z   -> CENTERING, with subject scaling and leakage held fixed
    subject_center vs subject_z  -> SCALING, with subject centering held fixed
    global vs premeal_center     -> causal LEVEL REMOVAL, with no leakage in either arm

WHAT THE `premeal_center` ARM DOES AND DOES NOT SHOW
-----------------------------------------------------
It shows that removing between-participant level is SUFFICIENT to degrade attribution
without any leakage. It does NOT show that leakage is harmless, and it does not give an
independent estimate of a leakage effect. No arm here does; per-subject statistics for a
held-out participant necessarily use that participant's data, so the two cannot be fully
crossed. Say that in the text rather than implying a clean 2x2.

Outputs
  e1_interaction.csv   the primary DDR2 tests, all horizons
  e1_mechanism.csv     the three paired mechanism contrasts
  e1_increments.csv    within-arm increments (descriptive context, not the primary test)
  e1_models.csv        per-arm R2/RMSE per configuration, all horizons
  e1_figure_data.csv   tidy long form for the principal figure
================================================================
"""
import os
import argparse
import numpy as np
import pandas as pd

from ablation_analyze import (build_ref, contrast_stats, decide, holm, MARGIN_R2,
                              PRIMARY_H, HORIZONS, N_BOOT, N_PERM, BOOT_SEED, PERM_SEED)
from e1_normalizers import PROTOCOLS, PROPERTIES

INCREMENTS = [
    ("cgm_static", "static_all", "CGM adds beyond context"),
    ("cgm_static", "cgm", "context adds beyond CGM"),
]
REFERENCE_ARM = "global"
PUBLISHED_ARM = "subject_z"
PRIMARY_PAIR = (REFERENCE_ARM, PUBLISHED_ARM)

MECHANISM_PAIRS = [
    ("subject_scale", "subject_z",
     "CENTERING (subject scaling + leakage held fixed)"),
    ("subject_center", "subject_z",
     "SCALING (subject centering held fixed)"),
    ("global", "premeal_center",
     "causal LEVEL REMOVAL (no leakage in either arm)"),
]

# Configurations that use no CGM at all. Their predictions MUST be bit-identical across
# every arm -- same static normalization, same RF seed, no temporal input. If they are not,
# something other than the CGM transform differs between arms and E1 is not controlled.
CGM_FREE = ["static_all", "persistence_5min"]
ABL_OOF = "ablation_oof_rf_capped.csv"


# ------------------------------------------------------------------ integrity
def check_unique(e1):
    dup = e1.duplicated(["protocol", "model", "seed", "fold", "sample_idx"]).sum()
    if dup:
        raise SystemExit(f"[abort] {dup} duplicate OOF rows. An interrupted run was resumed "
                         f"without reconciliation -- delete e1_oof.csv and re-run.")
    return "no duplicate (protocol, model, seed, fold, sample_idx) rows"


def check_invariance(ref):
    """CGM-free configurations must be identical across arms."""
    out = []
    for m in CGM_FREE:
        keys = [f"{m}@{p}" for p in PROTOCOLS if f"{m}@{p}" in ref["preds"]]
        if len(keys) < 2:
            continue
        base = keys[0]
        worst = 0.0
        for k in keys[1:]:
            for s in ref["seeds"]:
                worst = max(worst, float(np.abs(ref["preds"][k][s]
                                                - ref["preds"][base][s]).max()))
        out.append(f"    {m:16s} max|diff| across {len(keys)} arms = {worst:.3e}"
                   + ("  OK" if worst < 1e-6 else "  *** NOT INVARIANT ***"))
    return "\n".join(out) if out else "    (no CGM-free configs found)"


def verify_reference(e1, path=ABL_OOF):
    """Arm `global` re-runs the reference ablation with identical splits, estimator and RF
    seeds, so it must reproduce it. Checked, not assumed."""
    if not os.path.exists(path):
        return f"SKIPPED ({path} not found)"
    a = pd.read_csv(path, dtype={"pid": str})
    g = e1[e1.protocol == REFERENCE_ARM]
    common = sorted(set(g.model) & set(a.model))
    seeds = sorted(set(g.seed) & set(a.seed))
    if not common or not seeds:
        return "SKIPPED (no overlapping models/seeds)"
    k = ["model", "seed", "fold", "sample_idx"]
    m = (g[g.model.isin(common) & g.seed.isin(seeds)][k + ["y_pred_60"]]
         .merge(a[a.model.isin(common) & a.seed.isin(seeds)][k + ["y_pred_60"]],
                on=k, suffixes=("_e1", "_abl")))
    if m.empty:
        return "SKIPPED (no overlapping cells)"
    d = np.abs(m.y_pred_60_e1 - m.y_pred_60_abl)
    return (f"{len(m)} cells | max|diff| = {d.max():.3e} | "
            + ("MATCH" if d.max() < 1e-4 else "*** MISMATCH -- investigate ***"))


# ------------------------------------------------------------------ estimands
def _sq_err(ref, key):
    P = np.array([ref["preds"][key][s] for s in ref["seeds"]])
    return (ref["y"][None, :] - P) ** 2


def did_stats(ref, a, b, p1, p2):
    """Paired difference-in-differences of the (a - b) increment between protocols p1, p2.

    Positive => the increment is LARGER under p1. Structure mirrors
    ablation_analyze.contrast_stats: event-weighted point estimate, participant-cluster
    bootstrap CI, participant sign-flip permutation on summed per-participant differences."""
    y, seeds, pid = ref["y"], ref["seeds"], ref["pid"]
    Ea1, Eb1 = _sq_err(ref, f"{a}@{p1}"), _sq_err(ref, f"{b}@{p1}")
    Ea2, Eb2 = _sq_err(ref, f"{a}@{p2}"), _sq_err(ref, f"{b}@{p2}")
    D1 = (Eb1 - Ea1).mean(axis=0)          # per-meal mean error reduction under p1
    D2 = (Eb2 - Ea2).mean(axis=0)
    DD = D1 - D2

    units = np.unique(pid)
    idx_by = {u: np.where(pid == u)[0] for u in units}

    def ddr2(sel):
        yt = y[sel]
        ss = ((yt - yt.mean()) ** 2).sum()
        return DD[sel].sum() / ss if ss > 0 else np.nan

    point = ddr2(np.arange(len(y)))
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        sel = np.concatenate([idx_by[u] for u in rng.choice(units, len(units), replace=True)])
        boot[k] = ddr2(sel)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    dj = np.array([DD[idx_by[u]].sum() for u in units])
    rng2 = np.random.default_rng(PERM_SEED)
    Tstar = (rng2.choice([-1.0, 1.0], size=(N_PERM, len(units))) * dj[None, :]).sum(axis=1)
    perm_p = float((1 + np.sum(np.abs(Tstar) >= abs(dj.sum()) - 1e-12)) / (N_PERM + 1))

    ss_tot = ((y - y.mean()) ** 2).sum()
    per_seed = np.array([((Eb1[k] - Ea1[k]) - (Eb2[k] - Ea2[k])).sum() / ss_tot
                         for k in range(len(seeds))])
    return {"ddR2": point, "lo": float(lo), "hi": float(hi), "perm_p": perm_p,
            "seed_pos_frac": float(np.mean(per_seed > 0))}


def interaction(refs, pairs, family):
    recs = []
    for h, ref in refs.items():
        for a, b, label in INCREMENTS:
            for p1, p2, *why in pairs:
                if any(f"{m}@{p}" not in ref["preds"] for m in (a, b) for p in (p1, p2)):
                    continue
                st = did_stats(ref, a, b, p1, p2)
                recs.append({"family": family, "horizon": h, "increment": label,
                             "protocol_1": p1, "protocol_2": p2,
                             "isolates": (why[0] if why else
                                          "normalization effect on the increment"),
                             **st})
    r = pd.DataFrame(recs)
    if r.empty:
        return r
    # Holm within family, at the confirmatory horizon only; other horizons are secondary.
    r["holm_p"] = np.nan
    m = r.horizon == PRIMARY_H
    r.loc[m, "holm_p"] = holm(r.loc[m, "perm_p"].values)
    r["verdict"] = [
        (decide(x.ddR2, x.lo, x.hi, x.holm_p, x.seed_pos_frac)
         if np.isfinite(x.holm_p) else "")
        for x in r.itertuples()]
    return r


def within_arm_increments(refs):
    """Descriptive context: the increment inside each arm. NOT the primary test."""
    recs = []
    for h, ref in refs.items():
        for proto in PROTOCOLS:
            for a, b, label in INCREMENTS:
                ka, kb = f"{a}@{proto}", f"{b}@{proto}"
                if ka not in ref["preds"] or kb not in ref["preds"]:
                    continue
                st = contrast_stats(ref, ka, kb)
                ss_tot = float(np.sum((ref["y"] - ref["y"].mean()) ** 2))
                pos = np.mean([
                    np.sum((ref["y"] - ref["preds"][kb][s]) ** 2)
                    > np.sum((ref["y"] - ref["preds"][ka][s]) ** 2) for s in ref["seeds"]])
                recs.append({"horizon": h, "protocol": proto, "increment": label,
                             "dR2": st["dR2_60"], "lo": st["R2_lo"], "hi": st["R2_hi"],
                             "perm_p": st["perm_p"], "seed_pos_frac": float(pos),
                             **PROPERTIES[proto]})
    return pd.DataFrame(recs)


def per_arm_models(e1):
    rows = []
    for (proto, model), g in e1.groupby(["protocol", "model"]):
        for h in HORIZONS:
            yt, yp = f"y_true_{h}", f"y_pred_{h}"
            per_seed = []
            for _, gs in g.groupby("seed"):
                e = gs[yp].values - gs[yt].values
                ss_tot = float(np.sum((gs[yt].values - gs[yt].values.mean()) ** 2))
                per_seed.append({"R2": 1.0 - float(np.sum(e ** 2)) / ss_tot,
                                 "RMSE": float(np.sqrt(np.mean(e ** 2)))})
            d = pd.DataFrame(per_seed)
            rows.append({"protocol": proto, "model": model, "horizon": h,
                         "R2": d.R2.mean(), "R2_sd": d.R2.std(),
                         "RMSE": d.RMSE.mean(), "RMSE_sd": d.RMSE.std(),
                         **PROPERTIES[proto]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ interpretation
def interpret(inc, inter):
    """Pre-committed reading. A REVERSAL requires all three conditions:
         (a) the increment is positive under the reference arm, CI excluding 0;
         (b) it is negative under the published transform, CI excluding 0;
         (c) the paired cross-protocol change itself excludes 0.
    Anything less is 'changes', not 'reverses'."""
    key = "CGM adds beyond context"
    g = inc[(inc.increment == key) & (inc.horizon == PRIMARY_H)].set_index("protocol")
    d = inter[(inter.increment == key) & (inter.horizon == PRIMARY_H)
              & (inter.protocol_1 == PRIMARY_PAIR[0])
              & (inter.protocol_2 == PRIMARY_PAIR[1])]
    if REFERENCE_ARM not in g.index or PUBLISHED_ARM not in g.index or d.empty:
        return "cannot interpret: required arms or the interaction test are missing"
    r, p, dd = g.loc[REFERENCE_ARM], g.loc[PUBLISHED_ARM], d.iloc[0]
    lines = [
        f"  reference  ({REFERENCE_ARM:14s}) dR2 = {r.dR2:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]",
        f"  published  ({PUBLISHED_ARM:14s}) dR2 = {p.dR2:+.4f} [{p.lo:+.4f}, {p.hi:+.4f}]",
        f"  INTERACTION (paired)          DDR2 = {dd.ddR2:+.4f} [{dd.lo:+.4f}, {dd.hi:+.4f}]"
        f"  Holm p = {dd.holm_p:.4g}  seeds {dd.seed_pos_frac:.0%}",
    ]
    ref_pos = r.dR2 > 0 and r.lo > 0
    pub_neg = p.dR2 < 0 and p.hi < 0
    inter_ok = (dd.lo > 0) or (dd.hi < 0)
    if ref_pos and pub_neg and inter_ok:
        v = ("REVERSES. All three conditions met -- the stronger title is supported: "
             "per-subject normalization CAN REVERSE modality-importance conclusions.")
    elif inter_ok:
        v = ("CHANGES but does not reverse. The interaction is real (CI excludes 0) but "
             "the published-arm increment does not cross below zero with confidence. "
             "Title must say CHANGES, not reverses.")
    elif abs(dd.ddR2) < MARGIN_R2 and dd.lo > -MARGIN_R2 and dd.hi < MARGIN_R2:
        v = ("MATERIALLY UNCHANGED. The CGM transform does not explain the published "
             "conclusion. Revise the EXPLANATION, not the conclusion, and report this "
             "plainly -- it is still a useful negative result.")
    else:
        v = ("INCONCLUSIVE at this precision. Report the interval and do not claim either "
             "a change or an absence of one.")
    lines.append(f"  => {v}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""
    path = f"e1_oof{sfx}.csv"
    if not os.path.exists(path):
        raise SystemExit(f"[abort] {path} not found -- run e1_run.py first")
    e1 = pd.read_csv(path, dtype={"pid": str})
    print(f"loaded {path}: {len(e1)} rows | protocols {sorted(e1.protocol.unique())} "
          f"| seeds {sorted(e1.seed.unique())}")

    print("\n=== INTEGRITY ===")
    print(f"  {check_unique(e1)}")
    print(f"  reference-arm parity vs {ABL_OOF}: {verify_reference(e1)}")

    tagged = e1.copy()
    tagged["model"] = tagged.model + "@" + tagged.protocol
    refs = {h: build_ref(tagged, h) for h in HORIZONS}
    print("  CGM-free configurations must be identical across arms:")
    print(check_invariance(refs[PRIMARY_H]))

    mods = per_arm_models(e1)
    mods.to_csv(f"e1_models{sfx}.csv", index=False)
    print(f"\n=== R2 at {PRIMARY_H} min, by arm ===")
    print(mods[mods.horizon == PRIMARY_H].pivot(index="protocol", columns="model",
                                                values="R2")
          .reindex(list(PROTOCOLS)).round(3).to_string())

    raw = f"e1_results_raw{sfx}.csv"
    if os.path.exists(raw):
        lv = pd.read_csv(raw).groupby("protocol").level_retained.mean().reindex(list(PROTOCOLS))
        print("\n  between-participant share of variance in MEAL-LEVEL MEAN pre-meal CGM")
        print("  (the mechanism; ~0 == between-participant level destroyed):")
        print(lv.round(4).to_string())

    inc = within_arm_increments(refs)
    inc.to_csv(f"e1_increments{sfx}.csv", index=False)
    print(f"\n=== WITHIN-ARM INCREMENTS at {PRIMARY_H} min (descriptive) ===")
    for label, g in inc[inc.horizon == PRIMARY_H].groupby("increment"):
        g = g.set_index("protocol").reindex(list(PROTOCOLS)).reset_index()
        print(f"\n-- {label} --")
        print(g[["protocol", "level_preserved", "leakage", "dR2", "lo", "hi",
                 "seed_pos_frac"]].round(4).to_string(index=False))

    inter = interaction(refs, [PRIMARY_PAIR], "primary")
    inter.to_csv(f"e1_interaction{sfx}.csv", index=False)
    print(f"\n=== PRIMARY: cross-protocol interaction ({PRIMARY_PAIR[0]} vs "
          f"{PRIMARY_PAIR[1]}) ===")
    print("  positive DDR2 = the increment is LARGER under the reference protocol")
    print(inter[["horizon", "increment", "ddR2", "lo", "hi", "perm_p", "holm_p",
                 "seed_pos_frac", "verdict"]].round(4).to_string(index=False))

    mech = interaction(refs, MECHANISM_PAIRS, "mechanism")
    mech.to_csv(f"e1_mechanism{sfx}.csv", index=False)
    print("\n=== MECHANISM: paired contrasts (confirmatory horizon) ===")
    print(mech[mech.horizon == PRIMARY_H][["increment", "protocol_1", "protocol_2",
                                           "isolates", "ddR2", "lo", "hi", "holm_p",
                                           "verdict"]].round(4).to_string(index=False))

    fig = pd.concat([
        inc.assign(kind="within_arm").rename(columns={"dR2": "estimate"})
           [["kind", "horizon", "protocol", "increment", "estimate", "lo", "hi",
             "level_preserved", "leakage"]],
        pd.concat([inter, mech]).assign(kind="interaction")
           .rename(columns={"ddR2": "estimate", "protocol_1": "protocol"})
           [["kind", "horizon", "protocol", "increment", "estimate", "lo", "hi"]],
    ], ignore_index=True)
    fig.to_csv(f"e1_figure_data{sfx}.csv", index=False)

    print("\n=== PRE-COMMITTED INTERPRETATION ===")
    print(interpret(inc, inter))
    print("\n  NOTE: `premeal_center` shows level removal is SUFFICIENT to degrade "
          "attribution\n  without leakage. It does NOT show leakage is harmless, and no arm "
          "here estimates\n  a leakage effect independently -- per-subject statistics for a "
          "held-out participant\n  necessarily use that participant's data.")
    print(f"\nSaved: e1_interaction{sfx}.csv, e1_mechanism{sfx}.csv, "
          f"e1_increments{sfx}.csv, e1_models{sfx}.csv, e1_figure_data{sfx}.csv")


if __name__ == "__main__":
    main()
