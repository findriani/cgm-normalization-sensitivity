"""
e1_normalizers.py -- the five temporal-normalization protocols compared in E1
=============================================================================
E1 asks ONE question: does participant-specific CGM normalization distort modality
attribution when the target stays in absolute mg/dL?

Answering it properly requires separating two things the published procedure conflates.
Per-subject z-scoring, (x - mu_s) / sigma_s, does BOTH of the following at once:

  * CENTERING  -- subtracting a per-subject mean removes each participant's absolute
                  glucose LEVEL, which is the quantity the mg/dL target depends on;
  * SCALING    -- dividing by a per-subject SD removes each participant's amplitude.

and, because those statistics come from the participant's complete record, it also

  * LEAKS      -- a held-out participant's own future data informs their normalization.

Reviewer 1 wrote that the procedure "either removes the absolute glucose baseline needed
for prediction or uses information from the complete record of a held-out participant."
The arms below resolve that either/or empirically instead of arguing it, by crossing the
two factors:

    protocol          level preserved?   leakage?   what it is
    ----------------------------------------------------------------------------
    global                  YES            NO       the reference protocol (reference)
    subject_scale           YES            YES      per-subject amplitude only
    premeal_center          NO             NO       causal centering, available at test time
    subject_center          NO             YES      per-subject level removal only
    subject_z               NO             YES      AS PUBLISHED (centering + scaling)

If the damage tracks the LEVEL column and not the LEAKAGE column, the finding is sharp and
actionable: per-subject *scaling* is harmless, per-subject *centering* is not, and the
problem is not fixable by removing leakage alone.

`premeal_center` is the arm that makes this a controlled experiment rather than a
before/after. It removes the participant's level using ONLY the current meal's own pre-meal
window -- information genuinely available at prediction time -- so it destroys level with
no leakage whatsoever. Any degradation it shows cannot be blamed on leakage.

STATIC FEATURES ARE NORMALIZED IDENTICALLY IN EVERY ARM (fold-wise global, fit on training
participants only, via FoldNormalizer). Only the temporal transform varies. That is what
makes the comparison controlled: the static branch is held fixed, so any change in the
CGM-vs-static contrast is attributable to the CGM normalization alone.

Channel layout follows the data bundle: 0 = CGM, 1 = HR, 2 = Calories, 3 = METs.
=============================================================================
"""
import numpy as np

from foldwise_normalizer import FoldNormalizer

PROTOCOLS = ("global", "subject_scale", "premeal_center", "subject_center", "subject_z")

# Declared properties of each arm, used by e1_analyze.py to group results by factor and
# printed into the manifest so the design is self-documenting.
PROPERTIES = {
    "global":         {"level_preserved": True,  "leakage": False, "label": "Fold-wise global (reference)"},
    "subject_scale":  {"level_preserved": True,  "leakage": True,  "label": "Per-subject scaling only"},
    "premeal_center": {"level_preserved": False, "leakage": False, "label": "Causal pre-meal centering"},
    "subject_center": {"level_preserved": False, "leakage": True,  "label": "Per-subject centering only"},
    "subject_z":      {"level_preserved": False, "leakage": True,  "label": "Per-subject z-score (as published)"},
}

# The published description applied MIN-MAX (not z-score) to the activity channels:
# "activity-related features (caloric expenditure and METs) were scaled to [0,1] using
# subject-specific minimum and maximum values". Reproduced in `subject_z` for fidelity.
# NOTE: none of the four E1 configurations uses the dynamic channels, so this affects
# nothing in the reported results; it is here so `subject_z` is a faithful reproduction.
MINMAX_CHANNELS = (2, 3)
_EPS = 1e-8


def _global_stats(X_tr):
    """Per-channel mean/std over training rows and timesteps (what FoldNormalizer uses).

    The float64 cast is REQUIRED, not cosmetic. The bundle is float32, and np.nanmean over
    a float32 array accumulates in float32, which differs from FoldNormalizer's float64
    result by ~1.7e-4. Random Forest split thresholds are discontinuous in the features, so
    a difference that small moves predictions by whole mg/dL and breaks the `global` arm's
    parity with the published ablation. Caught by e1_analyze.verify_reference()."""
    flat = np.asarray(X_tr, dtype=np.float64).reshape(-1, X_tr.shape[2])
    mu = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    sd[sd < _EPS] = 1.0
    return mu, sd


def _subject_stats(X_all, pid_all):
    """Per-participant per-channel statistics over that participant's COMPLETE record.

    This is the leaky quantity: for a held-out participant it is computed from data the
    model is not supposed to have seen. Reproduced deliberately -- it is what the published
    pipeline did, and arm `premeal_center` exists to show whether the leakage matters."""
    stats = {}
    for s in np.unique(pid_all):
        Z = np.asarray(X_all[pid_all == s], dtype=np.float64)      # (n_s, T, C); see _global_stats
        flat = Z.reshape(-1, Z.shape[2])
        mu = np.nanmean(flat, axis=0)
        sd = np.nanstd(flat, axis=0)
        sd[sd < _EPS] = 1.0
        lo = np.nanmin(flat, axis=0)
        hi = np.nanmax(flat, axis=0)
        rng = hi - lo
        rng[rng < _EPS] = 1.0
        stats[s] = {"mu": mu, "sd": sd, "lo": lo, "rng": rng}
    return stats


def _apply(X, pid, protocol, gmu, gsd, sstats):
    """Transform one subset. `gmu`/`gsd` are TRAIN-fitted global statistics; `sstats` are
    per-participant statistics over the complete record."""
    Z = np.asarray(X, dtype=float).copy()

    if protocol == "global":
        return (Z - gmu) / gsd

    if protocol == "premeal_center":
        # Centre each row on ITS OWN pre-meal window mean, then apply the global scale.
        # Uses only the observation window itself, so nothing leaks.
        m = np.nanmean(Z, axis=1, keepdims=True)            # (n, 1, C)
        return (Z - m) / gsd

    mu = np.stack([sstats[p]["mu"] for p in pid])            # (n, C)
    sd = np.stack([sstats[p]["sd"] for p in pid])
    mu = mu[:, None, :]
    sd = sd[:, None, :]

    if protocol == "subject_center":
        return (Z - mu) / gsd
    if protocol == "subject_scale":
        return (Z - gmu) / sd
    if protocol == "subject_z":
        lo = np.stack([sstats[p]["lo"] for p in pid])[:, None, :]
        rng = np.stack([sstats[p]["rng"] for p in pid])[:, None, :]
        out = (Z - mu) / sd
        for c in MINMAX_CHANNELS:                            # faithful to the published text
            if c < out.shape[2]:
                out[:, :, c] = (Z[:, :, c] - lo[:, :, c]) / rng[:, :, c]
        return out

    raise ValueError(f"unknown protocol {protocol!r}")


def normalize_split(X_all, S_all, pid_all, tr_idx, te_idx, protocol):
    """Return (Xtr, Str, Xte, Ste) for one fold under `protocol`.

    Static handling is IDENTICAL in every arm: FoldNormalizer fitted on training
    participants only. The temporal transform is the only thing that varies."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")

    nrm = FoldNormalizer().fit(X_all[tr_idx], S_all[tr_idx])
    Xtr_g, Str = nrm.transform(X_all[tr_idx], S_all[tr_idx])
    Xte_g, Ste = nrm.transform(X_all[te_idx], S_all[te_idx])

    if protocol == "global":
        # Take FoldNormalizer's own output rather than recomputing an identical formula.
        # Parity with the published ablation is then exact BY CONSTRUCTION instead of
        # depending on two code paths agreeing to the last bit.
        return Xtr_g, Str, Xte_g, Ste

    gmu, gsd = _global_stats(X_all[tr_idx])
    sstats = _subject_stats(X_all, pid_all) if protocol.startswith("subject") else None

    Xtr = _apply(X_all[tr_idx], pid_all[tr_idx], protocol, gmu, gsd, sstats)
    Xte = _apply(X_all[te_idx], pid_all[te_idx], protocol, gmu, gsd, sstats)
    # float64 throughout, matching ablation_run._prep. Downcasting to float32 here would
    # feed the RF slightly different values than the published run did.
    return Xtr, Str, Xte, Ste


def between_subject_level_retained(X_norm, pid, channel=0):
    """Between-participant share of variance in the MEAL-LEVEL MEAN pre-meal CGM.

    Precisely: reduce each meal's pre-meal window to its mean, then report
    var_between_participants(those means) / var_total(those means). This is NOT the
    between-participant share of variance across all individual pre-meal readings -- it
    ignores within-window variation entirely. Describe it with that exact wording in the
    manuscript; the looser phrasing overstates what is computed.

    It is the quantity the mg/dL target depends on and the quantity per-subject centering
    destroys. Expect ~0 for the centering arms and clearly positive for the level-preserving
    ones. e1_run.py records it per fold so the mechanism is shown, not assumed."""
    v = X_norm[:, :, channel].mean(axis=1)
    tot = float(np.var(v))
    if tot < _EPS:
        return 0.0
    means = np.array([v[pid == s].mean() for s in np.unique(pid)])
    w = np.array([(pid == s).sum() for s in np.unique(pid)], dtype=float)
    grand = np.average(means, weights=w)
    between = float(np.average((means - grand) ** 2, weights=w))
    return between / tot


if __name__ == "__main__":
    # Self-test on the real bundle: confirms `global` matches FoldNormalizer exactly and
    # that the level diagnostic separates the arms the way the design claims.
    from lightdl_data import DataBundle
    b = DataBundle("binned")
    tr = np.where(np.isin(b.pid, np.unique(b.pid)[:30]))[0]
    te = np.where(np.isin(b.pid, np.unique(b.pid)[30:]))[0]
    ref = FoldNormalizer().fit(b.X[tr], b.S[tr])
    Xte_ref, _ = ref.transform(b.X[te], b.S[te])
    print(f"{'protocol':16s} {'level_retained':>15s}  {'max|diff vs FoldNorm|':>22s}")
    for p in PROTOCOLS:
        _, _, Xte, _ = normalize_split(b.X, b.S, b.pid, tr, te, p)
        lvl = between_subject_level_retained(Xte, b.pid[te])
        d = float(np.abs(Xte - Xte_ref).max())
        print(f"{p:16s} {lvl:15.4f}  {d:22.2e}")
    print("\n'global' must show ~0 difference from FoldNormalizer; the centering arms "
          "(premeal_center, subject_center, subject_z) must show level_retained ~ 0.")
