"""
attn_analyze.py  -- participant-clustered inference for the attention grid
==========================================================================
Runs AFTER attn_run.py. Reuses the SAME statistical core as ablation_analyze.py /
fusion_analyze.py (event-weighted mean-single-run dR2_60, participant-cluster bootstrap
CIs, participant sign-flip permutation p on SUMMED per-participant error diffs,
equivalence-first decision, delta=0.02 R2_60). Nothing is re-tuned after seeing results.

FOUR additions over fusion_analyze.py (PLAN_attention.md §1/§6/§7):

  1. PRECISION REPORT (attn_precision_<tag>.csv). The cluster-bootstrap variance of dR2_60
     is decomposed into a PARTICIPANT component (irreducible at n=40) and a SEED component
     (shrinks as 1/S). Reported per contrast: the obtained CI half-width, the projected
     half-width, the n=40 FLOOR (S->inf), the seed share, whether equivalence at delta was
     ATTAINABLE IN PRINCIPLE, and the seeds that would be needed if it is. Every
     INCONCLUSIVE is annotated with that flag, so a null is never reported as evidence of
     no effect when it is only insufficient precision. This is a HEURISTIC PROJECTION, not
     a full power analysis -- see precision_decomp.__doc__ for what it assumes. The
     bootstrap CI stays authoritative.

  2. EXTERNAL contrasts are SUPERIORITY-ONLY. The measured floor for an unpaired contrast
     at n=40 is ~0.031 > delta=0.02, so "EQUIVALENT" is unattainable at any seed count.
     decide_external_attn() cannot emit it; the null verdict distinguishes
     NO_DIFFERENCE_DETECTED from POSITIVE_BUT_UNSTABLE (CI excludes zero and p < 0.05
     but fewer than 80% of seeds agree).

  3. ATTENTION PROFILES with a participant-cluster bootstrap band, built from per-
     participant rows. The band is over PARTICIPANTS (the resampling unit used everywhere
     else in this paper), not over the 75 non-independent fold fits.

  4. OCCLUSION SWEEP -- a fixed-model CGM perturbation sensitivity analysis. Occluded
     inputs are off-distribution, so the loss reflects model sensitivity rather than
     information content. Attention weights are over encoder states (cumulative for a
     GRU), so they cannot support a raw-minute claim on their own. The refit study
     (window_run.py) provides the direct window-length evidence.

Outputs (all suffixed by tag; TAG!=raw is SENSITIVITY-ONLY and never overwrites the
confirmatory files): attn_model_summary_<tag>.csv, attn_per_seed_r2_<tag>.csv,
attn_verdict_<tag>.csv, attn_precision_<tag>.csv, attn_profile_curve_<tag>.csv,
attn_profile_summary_<tag>.csv, attn_occlusion_summary_<tag>.csv
==========================================================================
"""
import os
import json
import numpy as np
import pandas as pd

from ablation_analyze import (holm, per_seed_pooled, build_ref, contrast_stats, decide,
                              integrity, MARGIN_R2, WIN_FRAC, PRIMARY_H, HORIZONS, N_FOLDS,
                              N_BOOT, BOOT_SEED)

try:                       # only for the encoder receptive-field labels; needs TensorFlow
    from attn_models import RECEPTIVE_FIELD as AM_RF
except Exception:
    AM_RF = {}

PRIMARY_TAG = os.environ.get("TAG", "raw")
PLANNED_N_SEEDS = 15               # must match attn_run.PLANNED_N_SEEDS
RF_REF = "rf_highres"
INCUMBENT_PAPER = "dl_current_binned"      # the model the paper actually reports (12x5-min)
INCUMBENT_HR = "dl_current_highres"        # the high-res adaptation (fusion study's dl_current)
CHAMPION = "cnn_sa__coattn"                # pre-declared attention champion

# PRIMARY: one per attention locus, each matched on ONE component. Holm within family.
PRIMARY = [
    ("cnn_gru__concat+attnpool", "cnn_gru__concat",
     "readout: attention pooling vs GAP (strict generalization of it)"),
    ("cnn_sa__concat", "cnn__concat",
     "encoder: transformer block vs 2nd conv block, same Conv1D(d,5) front-end"),
    ("cnn_gru__coattn", "cnn_gru__coattn_ctl",
     "cross-modal: meal branch reads CGM vs meal self-attn (param-matched)"),
]
SECONDARY = [
    ("cnn__concat+attnpool", "cnn__concat",
     "does the readout effect generalize to a 2nd encoder (pre-declared guard)"),
    ("cnn_gru__coattn_ctl", "cnn_gru__xattn",
     "decomposition: the extra meal branch + wider head, WITHOUT the direction"),
    ("cnn_gru__coattn", "cnn_gru__xattn",
     "UNMATCHED: branch + direction together (the naive bidirectionality contrast)"),
    ("sa__concat", "cnn__concat",
     "UNMATCHED: attention replacing the locality prior (architectural alternative)"),
    ("cnn_sa__coattn", "cnn_sa__concat", "co-attention on top of a self-attention encoder"),
    ("cnn_sa__concat", "cnn_gru__concat", "bridge: self-attn encoder vs the old reference"),
    ("cnn__concat", "cnn_gru__concat", "REPLICATION: cnn vs cnn_gru (was +0.030 INCONCL at S=5)"),
    ("cnn_gru__xattn", "cnn_gru__concat", "REPLICATION: xattn vs concat (was -0.001 INCONCL at S=5)"),
    (INCUMBENT_HR, RF_REF, "ANCHOR: reproduces the fusion study's dl_current vs RF (+0.069 ADDS)"),
    (INCUMBENT_HR, INCUMBENT_PAPER, "does the high-res adaptation beat the paper's own model"),
    ("cnn_gru__concat", "persistence_5min", "sanity: DL beats the 5-min persistence floor"),
]
EXTERNAL = [
    (CHAMPION, INCUMBENT_PAPER, "champion vs the PAPER'S incumbent (the model it would change FROM)"),
    (CHAMPION, INCUMBENT_HR, "champion vs the protocol-matched high-res incumbent"),
    (CHAMPION, RF_REF, "champion vs RF"),
]


# ------------------------- decision rules -------------------------
def decide_paired(point, lo, hi, p_adj, seed_pos):
    """Identical to ablation_analyze.decide EXCEPT that the >=80%-of-seeds consistency
    requirement is applied SYMMETRICALLY: a HURTS verdict now needs the loss to hold in
    >=80% of seeds, just as ADDS needs the gain to. The shared rule only guarded positives.
    This is strictly more conservative and cannot manufacture a positive finding; it is
    pre-declared here so the deviation from the shared core is explicit. Both verdicts are
    written out (verdict, verdict_shared_rule) so any disagreement is visible."""
    if lo >= -MARGIN_R2 and hi <= MARGIN_R2:
        return "NEGLIGIBLE"
    if point > 0 and lo > 0 and p_adj < 0.05 and seed_pos >= WIN_FRAC:
        return "ADDS" if lo > MARGIN_R2 else "ADDS(small)"
    if point < 0 and hi < 0 and p_adj < 0.05 and (1.0 - seed_pos) >= WIN_FRAC:
        return "HURTS" if hi < -MARGIN_R2 else "HURTS(small)"
    return "INCONCLUSIVE"


def decide_external_attn(point, lo, hi, p, seed_pos):
    """SUPERIORITY ONLY. 'EQUIVALENT' is deliberately NOT reachable: the measured n=40
    floor for an unpaired contrast (~0.031) exceeds delta=0.02, so no seed count could
    justify that claim (PLAN_attention.md §1). The losing side is named REFERENCE_WINS,
    not INCUMBENT_WINS -- one of these comparators is a random forest.

    POSITIVE_BUT_UNSTABLE / NEGATIVE_BUT_UNSTABLE: CI excludes zero and p < 0.05 but
    the verdict is POSITIVE_BUT_UNSTABLE / NEGATIVE_BUT_UNSTABLE rather than
    ATTN_WINS / REFERENCE_WINS because fewer than 80% of seeds agree in direction."""
    if point > 0 and lo > 0 and p < 0.05 and seed_pos >= WIN_FRAC:
        return "ATTN_WINS" if lo > MARGIN_R2 else "ATTN_WINS(small)"
    if point < 0 and hi < 0 and p < 0.05 and (1.0 - seed_pos) >= WIN_FRAC:
        return "REFERENCE_WINS" if hi < -MARGIN_R2 else "REFERENCE_WINS(small)"
    if point > 0 and lo > 0 and p < 0.05:
        return "POSITIVE_BUT_UNSTABLE"
    if point < 0 and hi < 0 and p < 0.05:
        return "NEGATIVE_BUT_UNSTABLE"
    return "NO_DIFFERENCE_DETECTED"


# ------------------------- precision decomposition -------------------------
def precision_decomp(ref, a, b, point):
    """Split Var(dR2_60) into a participant part (irreducible at n=40) and a seed part
    (1/S). point dR2 = sum_j D_j / SS_tot with D_j = mean_s d_sj, d_sj = participant j's
    SUMMED error reduction at seed s. Cluster bootstrap over J participants =>
    SE = sqrt(J * Var_j(D_j)) / SS_tot, and Var_j(D_j; S) = var_mu + sig2_e / S.

    THIS IS A HEURISTIC PROJECTION, NOT A POWER ANALYSIS. It assumes (i) participants are
    independent clusters -- the same assumption the pre-registered cluster bootstrap already
    makes, so it adds nothing there; but also (ii) the seed component is homoscedastic
    across participants, and (iii) a FUTURE contrast has variance components resembling the
    one measured. It ignores the covariance across participants induced by their sharing a
    fitted fold model. Use it to choose S and to label a null as informative-or-imprecise;
    do NOT quote seeds_needed_for_equiv as a guaranteed sample size. The percentile
    bootstrap CI (half_obs) remains the authoritative interval."""
    y, seeds, pid = ref["y"], ref["seeds"], ref["pid"]
    units = np.unique(pid); idx_by = {u: np.where(pid == u)[0] for u in units}
    J, S = len(units), len(seeds)
    SS = ((y - y.mean()) ** 2).sum()
    Ea = np.array([(y - ref["preds"][a][s]) ** 2 for s in seeds])
    Eb = np.array([(y - ref["preds"][b][s]) ** 2 for s in seeds])
    Dsj = np.array([[(Eb[k] - Ea[k])[idx_by[u]].sum() for u in units] for k in range(S)])
    var_tot = float(Dsj.mean(axis=0).var(ddof=1))
    sig2_e = float(np.mean(Dsj.var(axis=0, ddof=1))) if S > 1 else 0.0
    var_mu_raw = var_tot - sig2_e / S
    var_mu = max(var_mu_raw, 0.0)
    # When var_mu_raw < 0 the participant component was clipped to zero, making
    # seed_share > 1 and floor/seeds_needed unreliable.  Mark them unavailable.
    boundary_estimate = var_mu_raw < 0

    def half(n):
        return 1.96 * np.sqrt(J * (var_mu + sig2_e / max(n, 1e-9))) / SS

    slack = MARGIN_R2 - abs(point)                     # room left for the CI half-width
    if boundary_estimate:
        floor = np.nan
        attainable = False
        need = np.nan
    else:
        floor = half(np.inf)
        attainable = bool(slack > floor)
        need = np.nan
        if attainable and slack > 0:
            C = ((slack * SS) / 1.96) ** 2 / J         # required Var_j(D_j)
            need = float(np.ceil(sig2_e / (C - var_mu))) if C > var_mu and sig2_e > 0 else float(S)
    return {"half_proj": float(half(S)), "half_floor": float(floor) if not boundary_estimate else np.nan,
            "seed_share": float((sig2_e / S) / var_tot) if var_tot > 0 and not boundary_estimate else np.nan,
            "equiv_attainable": attainable, "seeds_needed_for_equiv": need}


# ------------------------- evaluation -------------------------
def evaluate(ref, ps_pivot, pairs, family, decider=None):
    recs = []
    for a, b, label in pairs:
        if a not in ref["preds"] or b not in ref["preds"]:
            miss = a if a not in ref["preds"] else b
            print(f"  [skip {family}] {label}: missing {miss}")
            continue
        st = contrast_stats(ref, a, b)
        seed_pos = float((ps_pivot[a] > ps_pivot[b]).mean())
        recs.append({"family": family, "increment": label, "with": a, "without": b,
                     "seed_pos_frac": seed_pos, **st,
                     **precision_decomp(ref, a, b, st["dR2_60"])})
    if recs:
        adj = holm(np.array([r["perm_p"] for r in recs]))
        for r, pa in zip(recs, adj):
            r["holm_p"] = float(pa)
            r["verdict"] = (decider or decide_paired)(r["dR2_60"], r["R2_lo"], r["R2_hi"], pa,
                                                      r["seed_pos_frac"])
            r["verdict_shared_rule"] = decide(r["dR2_60"], r["R2_lo"], r["R2_hi"], pa,
                                              r["seed_pos_frac"])
            r["half_obs"] = (r["R2_hi"] - r["R2_lo"]) / 2.0
    return recs


# ------------------------- integrity -------------------------
def validate_manifest(configs, seeds, nSamp, raw, oof):
    man_path = f"attn_manifest_{PRIMARY_TAG}.json"
    if not os.path.exists(man_path):
        raise SystemExit(f"{man_path} missing -- cannot verify the run matches the plan.")
    man = json.load(open(man_path))
    if man.get("smoke", True):
        raise SystemExit("[abort] manifest smoke=True -- this is a smoke run, not confirmatory.")
    exp = set(man["emitted_configs"])
    if set(configs) != exp:
        raise SystemExit(f"[abort] configs {sorted(set(configs) ^ exp)} differ from manifest.")
    if len(seeds) != man["n_seeds"]:
        raise SystemExit(f"[abort] {len(seeds)} seeds present, manifest declares "
                         f"{man['n_seeds']}. N_SEEDS is frozen (PLAN_attention.md §1).")
    if nSamp != man.get("n_samples"):
        raise SystemExit(f"[abort] {nSamp} samples/seed but the manifest recorded "
                         f"{man.get('n_samples')}. A uniformly TRUNCATED OOF would otherwise "
                         f"pass, because the row counts are inferred from the file itself.")
    if PRIMARY_TAG == "raw":                      # the confirmatory analysis, fully pinned
        if man["n_seeds"] != PLANNED_N_SEEDS:
            raise SystemExit(f"[abort] manifest n_seeds={man['n_seeds']}, pre-registered "
                             f"{PLANNED_N_SEEDS}. A fresh N_SEEDS=10 run writes a consistent "
                             f"10-seed manifest -- which is exactly what this check catches.")
        if sorted(seeds) != list(range(PLANNED_N_SEEDS)):
            raise SystemExit(f"[abort] seeds present are {sorted(seeds)}, expected 0..{PLANNED_N_SEEDS-1}.")
        if man.get("clean") != "raw":
            raise SystemExit(f"[abort] manifest clean={man.get('clean')!r}; the confirmatory "
                             f"analysis is defined on CLEAN=raw only.")
    n = man["n_seeds"]
    assert len(raw) == len(exp) * n * man["n_folds"], \
        f"raw rows {len(raw)} != {len(exp) * n * man['n_folds']}"
    assert len(oof) == len(exp) * n * man["n_samples"], \
        f"oof rows {len(oof)} != {len(exp) * n * man['n_samples']}"
    return man


def validate_side_table(path, man, keys, label, expected_models=None,
                        expected_seeds=None, expected_rows=None, required=False):
    """Exact completeness check for the profile / occlusion tables: each participant appears
    in exactly one test fold per seed, so every (model, seed[, variant]) group must cover all
    J participants. Silent partial coverage would bias the curve toward whoever finished.

    When expected_models, expected_seeds, or expected_rows are given, the table is checked
    against those exact sets/counts.  When required=True, an absent file is an error."""
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"[abort] {path} required but absent.")
        print(f"  [{label}] {path} absent -- skipped")
        return None
    d = pd.read_csv(path, dtype={"pid": str})
    J = man["n_participants"]
    # Check expected models
    if expected_models is not None:
        found = set(d.model.unique())
        if found != set(expected_models):
            missing = set(expected_models) - found
            extra = found - set(expected_models)
            raise SystemExit(f"[abort] {path}: model mismatch. "
                             f"missing={sorted(missing)}, extra={sorted(extra)}")
    # Check expected seeds
    if expected_seeds is not None:
        found_seeds = set(d.seed.unique())
        if found_seeds != set(expected_seeds):
            raise SystemExit(f"[abort] {path}: seed mismatch. "
                             f"expected={sorted(expected_seeds)}, found={sorted(found_seeds)}")
    # Check expected row count
    if expected_rows is not None and len(d) != expected_rows:
        raise SystemExit(f"[abort] {path}: expected {expected_rows} rows, found {len(d)}.")
    # Per-group participant coverage
    bad = []
    for k, g in d.groupby(keys):
        if g.pid.nunique() != J:
            bad.append((k, g.pid.nunique()))
    if bad:
        raise SystemExit(f"[abort] {path}: {len(bad)} group(s) do not cover all {J} "
                         f"participants, e.g. {bad[:3]}. The aggregate would be biased.")
    print(f"  [{label}] {path}: {len(d):,} rows, {d.model.nunique()} models, "
          f"all groups cover {J}/{J} participants")
    return d


def _y_stats(oof, horizon):
    """Per-participant sufficient statistics of the target, so the occlusion bootstrap can
    recompute SS_tot on each resample exactly as contrast_stats does (rather than holding it
    fixed, which would understate the interval)."""
    m0, s0 = oof.model.iloc[0], sorted(oof.seed.unique())[0]
    r = oof[(oof.model == m0) & (oof.seed == s0)]
    y = r[f"y_true_{horizon}"].values.astype(float)
    df = pd.DataFrame({"pid": r.pid.values, "n": 1.0, "sum_y": y, "sumsq_y": y ** 2})
    return df.groupby("pid").sum()


# ------------------------- attention profiles -------------------------
def profile_tables(pr, n_boot=N_BOOT):
    """Participant-weighted mean attention profile per model, with a participant-cluster
    bootstrap band. t=0 is the OLDEST encoder state, t=T-1 the last.

    These are weights over ENCODER STATES, not raw minutes (see
    attn_models.attention_profile_model). For `cnn` the state receptive field is 7 min, so
    the mapping is near-positional; for `cnn_gru` states are CUMULATIVE and the profile
    cannot be read as a distribution over minutes. The occlusion table is what carries the
    input-level claim."""
    curves, summ = [], []
    for m, g in pr.groupby("model"):
        seeds = np.array(sorted(g.seed.unique()))
        pids = np.array(sorted(g.pid.unique()))
        ts = np.array(sorted(g.t.unique()))
        W = (g.pivot_table(index=["seed", "pid"], columns="t", values="w")
              .reindex(index=pd.MultiIndex.from_product([seeds, pids]), columns=ts)
              .values.reshape(len(seeds), len(pids), len(ts)))
        NW = (g.groupby(["seed", "pid"]).n.first()
               .reindex(pd.MultiIndex.from_product([seeds, pids]))
               .values.reshape(len(seeds), len(pids)))
        if np.isnan(W).any() or np.isnan(NW).any():
            raise SystemExit(f"[abort] profile table for {m} has missing (seed,pid,t) cells.")

        def curve_of(sel):                         # event-weighted, then mean over seeds
            num = (NW[:, sel, None] * W[:, sel, :]).sum(axis=1)
            return (num / NW[:, sel].sum(axis=1)[:, None]).mean(axis=0)

        pt = curve_of(np.arange(len(pids)))
        rng = np.random.default_rng(BOOT_SEED)
        B = np.stack([curve_of(rng.integers(0, len(pids), len(pids))) for _ in range(n_boot)])
        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        curves.append(pd.DataFrame({"model": m, "t": ts, "w": pt, "w_lo": lo, "w_hi": hi}))

        T = len(ts)
        enc = m.split("__")[0]
        rf_kind, rf_span = AM_RF.get(enc, ("unknown", None))

        def scalars(c):
            c = c / c.sum()
            return (c[-10:].sum(), c[-20:].sum(), float(c.max() * T),
                    float(-(c * np.log(c + 1e-12)).sum() / np.log(T)))
        s_pt = scalars(pt)
        s_b = np.array([scalars(b) for b in B])
        lo_b, hi_b = np.percentile(s_b, [2.5, 97.5], axis=0)
        summ.append({"model": m, "T": T, "encoder_state_span": rf_kind,
                     "state_receptive_field_min": rf_span,
                     "mass_last_10": s_pt[0], "mass_last_10_lo": lo_b[0], "mass_last_10_hi": hi_b[0],
                     "mass_last_20": s_pt[1], "mass_last_20_lo": lo_b[1], "mass_last_20_hi": hi_b[1],
                     "uniform_last_10": 10.0 / T, "peak_t": int(np.argmax(pt)),
                     "max_over_uniform": s_pt[2],
                     "entropy_ratio_vs_uniform": s_pt[3],
                     "entropy_ratio_lo": lo_b[3], "entropy_ratio_hi": hi_b[3]})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summ)


# ------------------------- occlusion -------------------------
def occlusion_table(oc, ystat, n_boot=N_BOOT):
    """R2_60 under raw-window perturbation, with participant-cluster bootstrap CIs on the
    LOSS relative to the intact window. This is a fixed-model perturbation sensitivity
    analysis -- architecture-agnostic and applicable to models with no attention at all,
    including the paper's own Random Forest.

    Caveat to report with it: an occluded window is off-distribution for the fitted model,
    so the loss reflects model sensitivity rather than a guaranteed upper-bound on the
    information content. A refit study (window_run.py) answers a different question: how
    much signal is available to a model that only sees the shorter window."""
    pids = np.array(sorted(ystat.index))
    n_j = ystat["n"].reindex(pids).values
    sy = ystat["sum_y"].reindex(pids).values
    syy = ystat["sumsq_y"].reindex(pids).values
    rows = []
    for m, g in oc.groupby("model"):
        seeds = np.array(sorted(g.seed.unique()))
        variants = [v for v in ["full"] + sorted(set(g.variant) - {"full"})]
        E = {}
        for v in variants:
            E[v] = (g[g.variant == v].pivot_table(index="seed", columns="pid",
                                                  values="sse_60", aggfunc="sum")
                    .reindex(index=seeds, columns=pids).values)
            if np.isnan(E[v]).any():
                raise SystemExit(f"[abort] occlusion table for {m}/{v} has missing cells.")
        Ev = np.stack([E[v] for v in variants])                     # (V,S,J)
        D = E["full"][None] - Ev                                    # (V,S,J); negative = loss
        rng = np.random.default_rng(BOOT_SEED)
        draws = [np.arange(len(pids))] + [rng.integers(0, len(pids), len(pids))
                                          for _ in range(n_boot)]
        out = np.empty((len(draws), len(variants)))
        r2v = np.empty((len(draws), len(variants)))
        for k, sel in enumerate(draws):
            n = n_j[sel].sum()
            SS = syy[sel].sum() - sy[sel].sum() ** 2 / n            # exact, per resample
            out[k] = (D[:, :, sel].sum(axis=2) / SS).mean(axis=1)
            r2v[k] = (1.0 - Ev[:, :, sel].sum(axis=2) / SS).mean(axis=1)
        lo, hi = np.percentile(out[1:], [2.5, 97.5], axis=0)
        for vi, v in enumerate(variants):
            rows.append({"model": m, "variant": v, "R2_60": r2v[0, vi],
                         "dR2_60_vs_full": out[0, vi], "lo": lo[vi], "hi": hi[vi]})
    return pd.DataFrame(rows)


# ------------------------- main -------------------------
def main():
    if PRIMARY_TAG != "raw":
        print("=" * 78)
        print(f"  TAG={PRIMARY_TAG}: SENSITIVITY ANALYSIS ONLY. The confirmatory result is\n"
              f"  defined on CLEAN=raw (PLAN_attention.md §8). Outputs are suffixed with the\n"
              f"  tag and do NOT overwrite the confirmatory files.")
        print("=" * 78)

    oof_path = f"attn_oof_{PRIMARY_TAG}.csv"
    if not os.path.exists(oof_path):
        raise SystemExit(f"{oof_path} not found -- run attn_run.py first.")
    oof = pd.read_csv(oof_path, dtype={"pid": str})
    raw = pd.read_csv(f"attn_results_raw_{PRIMARY_TAG}.csv")
    configs, seeds, nSamp = integrity(oof, raw, PRIMARY_TAG)
    man = validate_manifest(configs, seeds, nSamp, raw, oof)
    print(f"[attn/{PRIMARY_TAG}] integrity+manifest OK: {len(configs)} configs x {len(seeds)} "
          f"seeds x {N_FOLDS} folds, {nSamp} samples / {man['n_participants']} participants "
          f"(tf {man['tf']}, fit-code {man['code_hash'][:8]}, pos={man['pos_mode']})")
    print(f"  reproducibility: {man.get('deterministic')}")
    expected_seeds = list(range(man["n_seeds"]))
    profile_models = man.get("profile_models")
    occlusion_models = man.get("occlusion_models")
    pr = validate_side_table(f"attn_profiles_{PRIMARY_TAG}.csv", man,
                             ["model", "seed"], "profiles",
                             expected_models=profile_models,
                             expected_seeds=expected_seeds)
    oc = validate_side_table(f"attn_occlusion_{PRIMARY_TAG}.csv", man,
                             ["model", "seed", "variant"], "occlusion",
                             expected_models=occlusion_models,
                             expected_seeds=expected_seeds)

    ps_all = pd.concat([per_seed_pooled(oof, h) for h in HORIZONS], ignore_index=True)
    ps_all.to_csv(f"attn_per_seed_r2_{PRIMARY_TAG}.csv", index=False)
    summ = (ps_all.groupby(["model", "horizon"]).agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                                                     RMSE_mean=("RMSE", "mean")).reset_index())
    summ.to_csv(f"attn_model_summary_{PRIMARY_TAG}.csv", index=False)

    ref = build_ref(oof, PRIMARY_H)
    ps_pivot = ps_all[ps_all.horizon == PRIMARY_H].pivot(index="seed", columns="model", values="R2")

    prim = evaluate(ref, ps_pivot, PRIMARY, "primary")
    sec = evaluate(ref, ps_pivot, SECONDARY, "secondary")
    ext = evaluate(ref, ps_pivot, EXTERNAL, "external", decider=decide_external_attn)

    verdict = pd.DataFrame(prim + sec + ext)
    verdict.to_csv(f"attn_verdict_{PRIMARY_TAG}.csv", index=False)
    pcols = ["family", "increment", "with", "without", "dR2_60", "half_obs", "half_proj",
             "half_floor", "seed_share", "equiv_attainable", "seeds_needed_for_equiv"]
    verdict[pcols].to_csv(f"attn_precision_{PRIMARY_TAG}.csv", index=False)

    curve = prof = occs = None
    if pr is not None:
        curve, prof = profile_tables(pr)
        curve.to_csv(f"attn_profile_curve_{PRIMARY_TAG}.csv", index=False)
        prof.to_csv(f"attn_profile_summary_{PRIMARY_TAG}.csv", index=False)
    if oc is not None:
        occs = occlusion_table(oc, _y_stats(oof, PRIMARY_H))
        occs.to_csv(f"attn_occlusion_summary_{PRIMARY_TAG}.csv", index=False)

    # ---------------- console ----------------
    pd.set_option("display.width", 200, "display.max_columns", 40)
    piv = summ.pivot(index="model", columns="horizon", values="R2_mean").sort_values(PRIMARY_H, ascending=False)
    print(f"\n=== attention-grid ranking: pooled-OOF R2 by horizon (mean over {len(seeds)} seeds) ===")
    print(piv.round(3).to_string())

    cols = ["increment", "with", "without", "dR2_60", "R2_lo", "R2_hi", "dRMSE_60",
            "perm_p", "holm_p", "seed_pos_frac", "verdict", "verdict_shared_rule"]
    for title, recs in [("PRIMARY -- one per attention locus, matched controls (Holm within family)", prim),
                        ("SECONDARY / exploratory (own Holm; 2 replications + 1 anchor)", sec),
                        ("EXTERNAL -- SUPERIORITY ONLY ('EQUIVALENT' is not reachable at n=40)", ext)]:
        print(f"\n=== {title} ===")
        if recs:
            print(pd.DataFrame(recs)[cols].round(4).to_string(index=False))

    print("\n=== PRECISION: is each null INFORMATIVE, or just imprecise? ===")
    pp = verdict[pcols + ["verdict"]].copy()
    pp["note"] = np.where(pp.equiv_attainable, "equivalence reachable",
                          "equivalence UNREACHABLE at n=40 -- floor > delta")
    print(pp.drop(columns=["family", "with", "without"]).round(4).to_string(index=False))
    print(f"  half_obs   = bootstrap CI half-width obtained (AUTHORITATIVE)\n"
          f"  half_proj / half_floor / seeds_needed = heuristic projection only; see\n"
          f"               precision_decomp.__doc__ for what it assumes.\n"
          f"  NEGLIGIBLE needs |dR2| + half < delta={MARGIN_R2}.\n"
          f"  An INCONCLUSIVE verdict means 'NO SUPERIORITY DETECTED AT THE ACHIEVED\n"
          f"  PRECISION'. It does NOT license 'attention provides no benefit', and it does\n"
          f"  not support the incumbent architecture. Only NEGLIGIBLE rules out an effect\n"
          f"  inside the margin -- and NEGLIGIBLE verdicts are NOT multiplicity-adjusted\n"
          f"  (Holm corrects the superiority p-values only), so across "
          f"{len(PRIMARY)} primaries the\n"
          f"  family-wise confidence for a set of equivalence claims is below 95%.")

    if prof is not None:
        print("\n=== ATTENTION-POOLING PROFILE (reported regardless of any dR2 verdict) ===")
        print(prof.round(4).to_string(index=False))
        print("  Weights are over ENCODER STATES, not raw minutes. encoder_state_span tells\n"
              "  you how far each can be read back: 'local' (cnn, RF 7 min) is near-positional;\n"
              "  'cumulative' (cnn_gru) is NOT -- state t already summarizes minutes 0..t, so\n"
              "  mass on late states does not mean late minutes drove the prediction.\n"
              "  entropy_ratio_vs_uniform: 1.000 = uniform = exactly GAP; lower = concentrated.\n"
              "  Bands are participant-cluster bootstrap (the unit used everywhere in this paper).")

    if occs is not None:
        print("\n=== OCCLUSION: fixed-model perturbation sensitivity (input-level) ===")
        print(occs.pivot(index="model", columns="variant", values="R2_60").round(3).to_string())
        print("\n  loss vs the intact window (dR2_60, participant-cluster 95% CI):")
        d = occs[occs.variant != "full"].copy()
        d["ci"] = d.apply(lambda r: f"[{r.lo:+.3f},{r.hi:+.3f}]", axis=1)
        print(d.pivot(index="model", columns="variant", values="dR2_60_vs_full").round(3).to_string())
        print("  keep_last_k: everything before the final k minutes is back-filled flat, so\n"
              "               only the last k minutes vary. drop_last_10: the mirror image.\n"
              "  Occluded inputs are off-distribution for a fitted model, so these losses\n"
              "  reflect model sensitivity, not a guaranteed bound on information content.\n"
              "  The refit study (window_run.py) provides the direct window-length evidence.")

    written = [f"attn_model_summary_{PRIMARY_TAG}.csv", f"attn_per_seed_r2_{PRIMARY_TAG}.csv",
               f"attn_verdict_{PRIMARY_TAG}.csv", f"attn_precision_{PRIMARY_TAG}.csv"]
    if prof is not None:
        written += [f"attn_profile_curve_{PRIMARY_TAG}.csv", f"attn_profile_summary_{PRIMARY_TAG}.csv"]
    if occs is not None:
        written += [f"attn_occlusion_summary_{PRIMARY_TAG}.csv"]
    print("\nSaved: " + ", ".join(written))


if __name__ == "__main__":
    main()
