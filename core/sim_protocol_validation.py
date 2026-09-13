"""
sim_protocol_validation.py -- does the protocol reach the right verdict?
=========================================================================
Generates synthetic CGM + context data where the true contributions of glucose
level and trajectory shape are controlled.  Applies the full five-condition
normalization protocol and classifies each result using the manuscript's
equivalence-first decision rule (NEGLIGIBLE / CHANGES / INCONCLUSIVE).

The primary metric is **verdict accuracy**: how often does the protocol reach
the expected conclusion at the empirical sample size?

Phase 1 (reference)
    A single large-sample run (400 participants) estimates the expected DDR2
    for each scenario-comparison pair.  This is the "known effect" for the
    estimator on this data-generating process.

Phase 2 (evaluation)
    Many replications at the empirical sample size (40 participants).  Each
    replication records the DDR2, 95 % CI, permutation p, and verdict.

Scenarios
---------
  level_dominant : level carries most CGM value (matches the empirical finding)
  shape_only     : trajectory shape carries CGM value, level does not
  no_cgm         : CGM contributes nothing (false-positive control)
  both_contribute: level and shape both contribute (unequal signal variance)

Estimator
---------
Ridge regression with fixed alpha = 1.0 (no LOO tuning, no inner-fold leakage).
Holm correction is applied across the three comparisons within each replication.

Usage
-----
    python sim_protocol_validation.py              # full run (200 sims)
    python sim_protocol_validation.py --smoke      # quick check (20 sims)
    python sim_protocol_validation.py --sims 500   # custom count

Output
------
    sim_protocol_manifest.json
    sim_protocol_results{_smoke}.csv    per-simulation DDR2, CI, p, verdict
    sim_protocol_summary{_smoke}.csv    verdict frequencies and reference DDR2
=========================================================================
"""
import os
import sys
import json
import time
import hashlib
import platform
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import e1_normalizers
from e1_normalizers import normalize_split, PROTOCOLS as ALL_PROTOCOLS
from e1_analyze import did_stats
from ablation_analyze import build_ref, MARGIN_R2

# ── Simulation parameters ─────────────────────────────────────────

SCENARIOS = {
    "level_dominant":  {"beta_level": 0.8, "beta_shape": 0.05, "beta_context": 0.4},
    "shape_only":      {"beta_level": 0.0, "beta_shape": 0.8,  "beta_context": 0.4},
    "no_cgm":          {"beta_level": 0.0, "beta_shape": 0.0,  "beta_context": 0.6},
    "both_contribute": {"beta_level": 0.4, "beta_shape": 0.4,  "beta_context": 0.4},
}

N_PARTICIPANTS = 40
MEALS_PER_PERSON = 23
N_TIMESTEPS = 12
N_SEEDS = 4
N_FOLDS = 5
BASE_SEED = 42
RHO_CGM = 0.95
NOISE_SD = 20.0
RIDGE_ALPHA = 1.0               # fixed; no LOO tuning

# All five protocol conditions from the manuscript
PROTOCOLS = list(ALL_PROTOCOLS)  # global, subject_scale, premeal_center, subject_center, subject_z
CONFIGS = ["static_all", "cgm_static"]

# Protocol comparisons to evaluate
COMPARISONS = [
    ("global_vs_premeal",  "global", "premeal_center"),
    ("global_vs_subz",     "global", "subject_z"),
    ("global_vs_subcenter","global", "subject_center"),
]

# Reference-run parameters
REF_PARTICIPANTS = 5000
REF_SEEDS = 8


# ── Decision rule (matches the manuscript) ─────────────────────────

def holm(pvals):
    """Holm-Bonferroni step-down adjustment."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def verdict(ddR2, lo, hi, perm_p, seed_pos_frac):
    """Equivalence-first decision rule from ablation_analyze.decide().

    DDR2 is the difference-in-differences: positive means the increment is
    LARGER under the reference (first) protocol.  perm_p should be
    Holm-adjusted when multiple comparisons are evaluated jointly.
    """
    if lo >= -MARGIN_R2 and hi <= MARGIN_R2:
        return "NEGLIGIBLE"
    if ddR2 > 0 and lo > 0 and perm_p < 0.05 and seed_pos_frac >= 0.80:
        return "CHANGES" if lo > MARGIN_R2 else "CHANGES(small)"
    if ddR2 < 0 and hi < 0 and perm_p < 0.05:
        return "CHANGES_NEG" if hi < -MARGIN_R2 else "CHANGES_NEG(small)"
    return "INCONCLUSIVE"


# ── Data-generating process ───────────────────────────────────────

def generate_dataset(rng, n_participants, beta_level, beta_shape, beta_context):
    """Generate one synthetic dataset.

    The target depends on:
      level_signal  : standardized final CGM bin (removed by centering)
      shape_signal  : standardized trajectory slope (preserved by centering,
                      but scaled differently by subject z-scoring)
      ctx_signal    : standardized context composite

    Static features follow the 17-column layout expected by FoldNormalizer.
    """
    n = n_participants * MEALS_PER_PERSON
    mu = rng.normal(130, 30, n_participants)
    sigma = rng.uniform(8, 25, n_participants)
    hba1c = 4.5 + 0.012 * mu + rng.normal(0, 0.3, n_participants)
    bmi = 22 + 0.03 * mu + rng.normal(0, 3, n_participants)
    age = rng.normal(50, 12, n_participants)

    n_per = n_participants // 3
    diag_labels = np.array(
        ["ND"] * n_per + ["PD"] * n_per
        + ["T2D"] * (n_participants - 2 * n_per))

    X = np.zeros((n, N_TIMESTEPS, 1))
    S = np.zeros((n, 17))
    y = np.zeros(n)
    pid = np.empty(n, dtype="U10")
    diag = np.empty(n, dtype="U4")

    idx = 0
    for i in range(n_participants):
        innov_sd = sigma[i] * np.sqrt(max(1e-12, 1 - RHO_CGM ** 2))
        gender_i = rng.choice([0.0, 1.0])
        for j in range(MEALS_PER_PERSON):
            cgm = np.zeros(N_TIMESTEPS)
            cgm[0] = mu[i] + rng.normal(0, sigma[i])
            for t in range(1, N_TIMESTEPS):
                cgm[t] = mu[i] + RHO_CGM * (cgm[t - 1] - mu[i]) + rng.normal(0, innov_sd)
            X[idx, :, 0] = cgm

            carbs = max(5.0, rng.normal(50, 20))
            cals = carbs * 4 + max(0, rng.normal(100, 50))
            protein = max(1.0, rng.normal(25, 10))
            fat = max(1.0, rng.normal(20, 10))
            fiber = max(0.5, rng.normal(5, 3))
            hour = rng.choice([8, 12, 18])
            weekend = rng.choice([0.0, 1.0], p=[5 / 7, 2 / 7])
            meal_type = rng.choice([0, 1, 2])

            S[idx, 0] = age[i]
            S[idx, 1] = gender_i
            S[idx, 2] = bmi[i]
            S[idx, 3] = hba1c[i]
            S[idx, 4] = cals
            S[idx, 5] = carbs
            S[idx, 6] = protein
            S[idx, 7] = fat
            S[idx, 8] = fiber
            S[idx, 9] = hour / 24.0
            S[idx, 10] = weekend
            S[idx, 11 + meal_type] = 1.0

            level_signal = (cgm[-1] - 130) / 30
            shape_signal = ((cgm[-1] - cgm[0]) / (N_TIMESTEPS - 1)) / 3

            ctx_signal = ((carbs - 50) / 20 + (hba1c[i] - 6) / 0.5) / 2

            y[idx] = (160
                      + beta_level * level_signal * 30
                      + beta_shape * shape_signal * 30
                      + beta_context * ctx_signal * 30
                      + rng.normal(0, NOISE_SD))

            pid[idx] = f"P{i:03d}"
            diag[idx] = diag_labels[i]
            idx += 1

    return X, S, y, pid, diag


# ── Cross-validation ──────────────────────────────────────────────

def make_splits(pid, diag, n_seeds=N_SEEDS):
    seen = {}
    for p, d in zip(pid, diag):
        if p not in seen:
            seen[p] = d
    unique_pid = np.array(list(seen.keys()))
    pid_diag = np.array(list(seen.values()))

    splits = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(N_FOLDS, shuffle=True,
                              random_state=BASE_SEED + seed)
        for fold, (tr_pi, te_pi) in enumerate(skf.split(unique_pid, pid_diag)):
            tr_set = set(unique_pid[tr_pi])
            te_set = set(unique_pid[te_pi])
            tr_idx = np.array([i for i, p in enumerate(pid) if p in tr_set])
            te_idx = np.array([i for i, p in enumerate(pid) if p in te_set])
            splits.append({"seed": seed, "fold": fold,
                           "train": tr_idx, "test": te_idx})
    return splits


# ── One simulation replicate ──────────────────────────────────────

def run_protocol(X, S, y, pid, diag, n_seeds=N_SEEDS):
    """Run the full protocol and return DDR2 stats for each comparison."""
    splits = make_splits(pid, diag, n_seeds=n_seeds)

    # Cache per-subject stats: _subject_stats iterates over all participants, which
    # is expensive at large N. The stats depend only on (X, pid), not on the split,
    # so computing them once and patching the module avoids 120× redundant work per
    # scenario in the reference phase.
    cached = e1_normalizers._subject_stats(X, pid)
    orig_fn = e1_normalizers._subject_stats
    e1_normalizers._subject_stats = lambda X_all, pid_all: cached

    oof_rows = []
    for sp in splits:
        tr, te = sp["train"], sp["test"]
        for proto in PROTOCOLS:
            Xtr, Str, Xte, Ste = normalize_split(X, S, pid, tr, te, proto)
            for config in CONFIGS:
                if config == "static_all":
                    F_tr, F_te = Str.copy(), Ste.copy()
                else:
                    F_tr = np.hstack([Xtr.reshape(len(tr), -1), Str])
                    F_te = np.hstack([Xte.reshape(len(te), -1), Ste])

                mdl = Ridge(alpha=RIDGE_ALPHA)
                mdl.fit(F_tr, y[tr])
                preds = mdl.predict(F_te)

                oof_rows.append(pd.DataFrame({
                    "model": f"{config}@{proto}",
                    "seed": sp["seed"], "fold": sp["fold"],
                    "sample_idx": te, "pid": pid[te],
                    "y_true_60": y[te],
                    "y_pred_60": preds,
                }))

    e1_normalizers._subject_stats = orig_fn  # restore

    oof = pd.concat(oof_rows, ignore_index=True)
    ref = build_ref(oof, 60)

    results = {}
    for label, p1, p2 in COMPARISONS:
        key_a1 = f"cgm_static@{p1}"
        key_b1 = f"static_all@{p1}"
        key_a2 = f"cgm_static@{p2}"
        key_b2 = f"static_all@{p2}"
        if all(k in ref["preds"] for k in [key_a1, key_b1, key_a2, key_b2]):
            results[label] = did_stats(ref, "cgm_static", "static_all", p1, p2)

    # Holm-adjust across comparisons within this replication
    if results:
        labels = list(results.keys())
        raw_p = np.array([results[l]["perm_p"] for l in labels])
        adj_p = holm(raw_p)
        for l, hp in zip(labels, adj_p):
            results[l]["holm_p"] = float(hp)
            results[l]["verdict"] = verdict(
                results[l]["ddR2"], results[l]["lo"], results[l]["hi"],
                hp, results[l]["seed_pos_frac"])

    return results


# ── Reference DDR2 from a large sample ────────────────────────────

def compute_reference(scenario_params):
    """Run one large-sample simulation to estimate the expected DDR2."""
    rng = np.random.default_rng(77777)
    X, S, y, pid, diag = generate_dataset(
        rng, REF_PARTICIPANTS, **scenario_params)
    return run_protocol(X, S, y, pid, diag, n_seeds=REF_SEEDS)


# ── Checkpointing ────────────────────────────────────────────────

def _append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ── Main ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Simulation study for the normalization-sensitivity protocol")
    ap.add_argument("--smoke", action="store_true",
                    help="20 simulations per scenario")
    ap.add_argument("--sims", type=int, default=200,
                    help="simulations per scenario (default 200)")
    args = ap.parse_args()
    n_sims = 20 if args.smoke else args.sims
    sfx = "_smoke" if args.smoke else ""
    results_csv = os.path.join(HERE, f"sim_protocol_results{sfx}.csv")
    summary_csv = os.path.join(HERE, f"sim_protocol_summary{sfx}.csv")
    manifest_path = os.path.join(HERE, f"sim_protocol_manifest{sfx}.json")

    man = {
        "scenarios": {k: v for k, v in SCENARIOS.items()},
        "n_sims": n_sims, "n_participants": N_PARTICIPANTS,
        "meals_per_person": MEALS_PER_PERSON, "n_timesteps": N_TIMESTEPS,
        "n_seeds": N_SEEDS, "n_folds": N_FOLDS, "base_seed": BASE_SEED,
        "rho_cgm": RHO_CGM, "noise_sd": NOISE_SD, "ridge_alpha": RIDGE_ALPHA,
        "margin_r2": MARGIN_R2, "protocols": PROTOCOLS,
        "ref_participants": REF_PARTICIPANTS, "ref_seeds": REF_SEEDS,
        "python": platform.python_version(), "numpy": np.__version__,
    }
    json.dump(man, open(manifest_path, "w"), indent=2)

    print("Simulation study: normalization-sensitivity protocol validation")
    print(f"  {n_sims} sims/scenario, {N_PARTICIPANTS} participants, "
          f"Ridge(alpha={RIDGE_ALPHA})")
    print(f"  Protocols: {PROTOCOLS}")
    print(f"  Margin: +/-{MARGIN_R2}")
    print(f"  Decision rule: NEGLIGIBLE if CI in [-{MARGIN_R2}, +{MARGIN_R2}]; "
          f"CHANGES if CI > 0 and Holm-p < 0.05; else INCONCLUSIVE")
    print()

    # ── Phase 1: reference DDR2 ────────────────────────────────────
    print("Phase 1: computing reference DDR2 "
          f"({REF_PARTICIPANTS} participants, {REF_SEEDS} seeds)")
    ref_values = {}
    for scenario, params in SCENARIOS.items():
        t0 = time.time()
        ref_res = compute_reference(params)
        elapsed = time.time() - t0
        print(f"  {scenario} ({elapsed:.0f}s):")
        for label, st in ref_res.items():
            ref_values[(scenario, label)] = st["ddR2"]
            print(f"    {label}: DDR2 = {st['ddR2']:+.4f} "
                  f"[{st['lo']:+.4f}, {st['hi']:+.4f}]")
    print()

    # ── Phase 2: evaluation ────────────────────────────────────────
    # Remove stale results from a previous run
    for f in [results_csv]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Phase 2: {n_sims} replications per scenario")
    t_total = time.time()

    for scenario, params in SCENARIOS.items():
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario}  "
              f"(beta_level={params['beta_level']}, "
              f"beta_shape={params['beta_shape']}, "
              f"beta_context={params['beta_context']})")

        t0 = time.time()
        for sim in range(n_sims):
            rng = np.random.default_rng(
                10_000 * (list(SCENARIOS).index(scenario) + 1) + sim)
            X, S, y, pid, diag = generate_dataset(
                rng, N_PARTICIPANTS, **params)
            res = run_protocol(X, S, y, pid, diag)

            rows = []
            for label, st in res.items():
                ref_ddr2 = ref_values.get((scenario, label), np.nan)
                rows.append({
                    "scenario": scenario, "sim": sim,
                    "comparison": label,
                    "ddR2": st["ddR2"], "lo": st["lo"], "hi": st["hi"],
                    "perm_p": st["perm_p"],
                    "holm_p": st["holm_p"],
                    "seed_pos_frac": st["seed_pos_frac"],
                    "verdict": st["verdict"],
                    "ref_ddR2": ref_ddr2,
                    "ci_covers_ref": (st["lo"] <= ref_ddr2 <= st["hi"]),
                })
            _append(results_csv, pd.DataFrame(rows))

            if (sim + 1) % max(1, n_sims // 5) == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  [{sim + 1}/{n_sims}] {elapsed:.1f} min")

        elapsed = (time.time() - t0) / 60
        print(f"  Completed in {elapsed:.1f} min")

    # ── Summary ────────────────────────────────────────────────────
    df = pd.read_csv(results_csv)

    print(f"\n{'=' * 60}")
    print("OPERATING CHARACTERISTICS")
    print(f"{'=' * 60}")

    summary_rows = []
    for scenario in SCENARIOS:
        print(f"\n--- {scenario} ---")
        for label, _, _ in COMPARISONS:
            sub = df[(df.scenario == scenario) & (df.comparison == label)]
            if sub.empty:
                continue
            n = len(sub)
            se_frac = lambda f: 1.96 * np.sqrt(f * (1 - f) / max(n, 1))

            negligible = (sub.verdict == "NEGLIGIBLE").mean()
            changes = sub.verdict.str.startswith("CHANGES").mean()
            inconclusive = (sub.verdict == "INCONCLUSIVE").mean()
            raw_reject = (sub.perm_p < 0.05).mean()
            mean_ddr2 = sub.ddR2.mean()
            ref_ddr2 = sub.ref_ddR2.iloc[0] if "ref_ddR2" in sub else np.nan
            bias = mean_ddr2 - ref_ddr2
            coverage = sub.ci_covers_ref.mean() if "ci_covers_ref" in sub else np.nan

            summary_rows.append({
                "scenario": scenario, "comparison": label,
                "n_sims": n, "ref_ddR2": ref_ddr2,
                "mean_ddR2": mean_ddr2, "bias": bias,
                "negligible": negligible, "changes": changes,
                "inconclusive": inconclusive, "raw_reject": raw_reject,
                "coverage": coverage,
            })

            print(f"  {label}:")
            print(f"    Ref DDR2    = {ref_ddr2:+.4f}  |  "
                  f"Mean DDR2 = {mean_ddr2:+.4f}  (bias {bias:+.4f})")
            print(f"    NEGLIGIBLE  = {negligible:.2f} (+/-{se_frac(negligible):.2f})")
            print(f"    CHANGES     = {changes:.2f} (+/-{se_frac(changes):.2f})")
            print(f"    INCONCLUSIVE= {inconclusive:.2f}")
            print(f"    Raw reject  = {raw_reject:.2f}  |  "
                  f"Coverage(ref) = {coverage:.2f}")

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\nResults: {results_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Manifest: {manifest_path}")
    print(f"Total: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
