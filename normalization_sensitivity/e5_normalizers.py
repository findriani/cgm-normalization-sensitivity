"""
e5_normalizers.py -- the same normalization arms, applied to the ACTIVITY channels
==================================================================================
E1 varied the CGM transform and held everything else fixed. E5 does the mirror image:
it holds CGM fixed at the reference fold-wise global protocol in EVERY arm and varies the
transform applied to the activity channels (HR, calories, METs) instead.

WHY THIS EXISTS
---------------
E1 and E4 are one modality on one cohort. A protocol claim needs evidence that the
procedure is not a one-off tuned to CGM. E5 is that evidence: same arms, same estimand,
same decision rule, different modality. It costs one CPU run and cannot embarrass the
paper -- a null here is a specificity result, not a failure (see the GATE in e5_analyze).

WHAT IS HELD FIXED
------------------
Channel 0 (CGM) is taken from FoldNormalizer's own output in every arm, byte for byte.
Only channels 1-3 are overwritten with the arm's transform. Two consequences, both used
as controls in e5_analyze:

  * configurations that use no activity channels (`static_all`, `cgm_static`) MUST produce
    bit-identical predictions across all arms -- they are refitted independently per arm
    rather than shared, so this is a genuine check and not an artefact of construction;
  * the `global` arm must reproduce the published reference ablation exactly, for the same
    reason it does in E1.

Channel layout follows the data bundle: 0 = CGM, 1 = HR, 2 = Calories, 3 = METs.
The published text applied per-subject MIN-MAX to calories and METs and z-scoring
elsewhere; `_apply` already encodes that, so `subject_z` here is faithful to it.
==================================================================================
"""
import numpy as np

from foldwise_normalizer import FoldNormalizer
from e1_normalizers import _apply, _global_stats, _subject_stats

DYN_CHANNELS = (1, 2, 3)          # HR, calories, METs -- everything except CGM
CGM_CHANNEL = 0

ARMS = ("global", "subject_z", "premeal_center")

PROPERTIES = {
    "global": {"level_preserved": True, "leakage": False,
               "label": "Fold-wise global (reference) -- reference"},
    "subject_z": {"level_preserved": False, "leakage": True,
                  "label": "Per-subject z / min-max on activity (published style)"},
    "premeal_center": {"level_preserved": False, "leakage": False,
                       "label": "Causal pre-meal centering of activity"},
}


def normalize_split_dyn(X_all, S_all, pid_all, tr_idx, te_idx, arm):
    """Return (Xtr, Str, Xte, Ste) for one fold with the arm applied to ACTIVITY only.

    Statics are handled identically in every arm (FoldNormalizer, training participants
    only), exactly as in E1 -- so the static branch is held fixed and any change in the
    activity-versus-context contrast is attributable to the activity transform alone."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    nrm = FoldNormalizer().fit(X_all[tr_idx], S_all[tr_idx])
    Xtr, Str = nrm.transform(X_all[tr_idx], S_all[tr_idx])
    Xte, Ste = nrm.transform(X_all[te_idx], S_all[te_idx])

    if arm == "global":
        return Xtr, Str, Xte, Ste

    gmu, gsd = _global_stats(X_all[tr_idx])
    sstats = _subject_stats(X_all, pid_all) if arm.startswith("subject") else None

    Atr = _apply(X_all[tr_idx], pid_all[tr_idx], arm, gmu, gsd, sstats)
    Ate = _apply(X_all[te_idx], pid_all[te_idx], arm, gmu, gsd, sstats)

    # Overwrite ONLY the activity channels. CGM keeps FoldNormalizer's own values, so the
    # CGM-only configurations stay exactly comparable to the published ablation.
    Xtr = np.asarray(Xtr, dtype=float).copy()
    Xte = np.asarray(Xte, dtype=float).copy()
    for c in DYN_CHANNELS:
        Xtr[:, :, c] = Atr[:, :, c]
        Xte[:, :, c] = Ate[:, :, c]
    return Xtr, Str, Xte, Ste


def assert_cgm_untouched(X_arm, X_global, where, tol=0.0):
    """The whole design rests on CGM being identical across arms. Assert it, do not assume.

    tol is 0.0 deliberately: the CGM channel is copied, not recomputed, so any difference
    at all means a channel index is wrong somewhere."""
    d = float(np.abs(np.asarray(X_arm[:, :, CGM_CHANNEL], dtype=np.float64)
                     - np.asarray(X_global[:, :, CGM_CHANNEL], dtype=np.float64)).max())
    if not np.isfinite(d) or d > tol:
        raise SystemExit(f"[abort] CGM channel differs between arms on {where}: "
                         f"max|diff| = {d:.3e}. E5 is not controlled -- the arms must "
                         f"differ ONLY in the activity transform.")
    return d


def between_subject_level_retained_dyn(X_norm, pid, channel):
    """Between-participant share of variance in the meal-level mean of one channel.

    Same diagnostic as E1's, pointed at an activity channel. Expect ~0 for the centering
    arms and clearly positive for `global`; it is what shows the transform did what the
    arm name claims, without relying on any model fit."""
    v = X_norm[:, :, channel].mean(axis=1)
    tot = float(np.var(v))
    if tot < 1e-8:
        return 0.0
    units = np.unique(pid)
    means = np.array([v[pid == s].mean() for s in units])
    w = np.array([(pid == s).sum() for s in units], dtype=float)
    grand = np.average(means, weights=w)
    return float(np.average((means - grand) ** 2, weights=w)) / tot
