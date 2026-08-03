"""
shanghai_normalizers.py -- the five normalization arms, on a one-channel cohort
===============================================================================
The ARM LOGIC is imported from `e1_normalizers`, not restated: `_global_stats`,
`_subject_stats`, `_apply` and the PROTOCOLS/PROPERTIES tables are the same objects E1 and
E2 used. If a transform changes there, it changes here, and the CGMacros-vs-Shanghai
comparison cannot silently drift.

What must be replaced is the STATIC normalizer, for a specific and dangerous reason.

WHY `FoldNormalizer` CANNOT BE REUSED
--------------------------------------
It log1p-transforms static indices 4-8 when the training column is skewed. In CGMacros
those are meal macronutrients (non-negative, right-skewed -- log1p is right). In Shanghai
they are hour_sin, hour_cos, is_morning, is_evening, is_weekend. `hour_sin` reaches exactly
-1, and `log1p(-1) = -inf`. Reusing it would not mis-scale the features; it would inject
infinities into the static branch and the failure would surface far downstream, if at all.

`ShanghaiFoldNormalizer` therefore z-scores age/BMI/HbA1c and passes everything else
through. That is the whole difference.

WHAT `subject_z` MEANS IN A ONE-CHANNEL COHORT
-----------------------------------------------
The published transform applies per-subject z-scoring to CGM and per-subject MIN-MAX to
channels 2 and 3 (calories, METs). Shanghai has channel 0 only, so `_apply`'s
`if c < out.shape[2]` guard means no min-max fires and `subject_z` reduces to a pure
per-subject z-score of CGM.

That is the correct comparator -- the CGM half of the published transform is exactly the
half C1 is about -- but the manuscript must say so. Do not write that Shanghai reproduces
"the published normalization"; it reproduces its CGM component, which is all this cohort
can carry.

WHAT IS HELD FIXED ACROSS ARMS
-------------------------------
Static handling is identical in every arm -- fitted on training participants only -- so the
temporal transform is the only thing that varies. `assert_statics_invariant` checks this
rather than trusting it.
===============================================================================
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
for _p in (CORE, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# The arms themselves -- imported, never reimplemented.
from e1_normalizers import (PROTOCOLS, PROPERTIES, MINMAX_CHANNELS, _EPS,
                            _global_stats, _subject_stats, _apply,
                            between_subject_level_retained)             # noqa: F401
from shanghai_data import ZSCORE_IDX, PASSTHROUGH_IDX


class ShanghaiFoldNormalizer:
    """Train-only static normalizer for the 12-column Shanghai layout.

    Mirrors `FoldNormalizer`'s global CGM scaling (which preserves absolute level) but
    z-scores only the three continuous statics and touches nothing else. No log1p anywhere
    -- there are no macronutrients in this cohort to justify it."""

    def __init__(self):
        self.ts_mean_ = None
        self.ts_std_ = None
        self.static_ = {}

    def fit(self, X_train, S_train):
        X_train = np.asarray(X_train, dtype=float)
        S_train = np.asarray(S_train, dtype=float)
        n_ch = X_train.shape[2]
        flat = X_train.reshape(-1, n_ch)
        self.ts_mean_ = np.nanmean(flat, axis=0)
        self.ts_std_ = np.nanstd(flat, axis=0)
        self.ts_std_[self.ts_std_ < _EPS] = 1.0
        self.static_ = {}
        for _, idx in ZSCORE_IDX.items():
            col = S_train[:, idx]
            m, s = float(np.nanmean(col)), float(np.nanstd(col))
            self.static_[idx] = {"mean": m, "std": s if s > _EPS else 1.0}
        return self

    def transform(self, X, S):
        if self.ts_mean_ is None:
            raise RuntimeError("ShanghaiFoldNormalizer must be fit() before transform().")
        X = (np.asarray(X, dtype=float).copy() - self.ts_mean_) / self.ts_std_
        S = np.asarray(S, dtype=float).copy()
        for idx, p in self.static_.items():
            S[:, idx] = (S[:, idx] - p["mean"]) / p["std"]
        return X, S


def normalize_split(X_all, S_all, pid_all, tr_idx, te_idx, protocol):
    """Return (Xtr, Str, Xte, Ste) for one fold under `protocol`.

    Same structure as `e1_normalizers.normalize_split`, with Shanghai's static normalizer.
    Static handling is IDENTICAL in every arm; only the temporal transform varies."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")

    nrm = ShanghaiFoldNormalizer().fit(X_all[tr_idx], S_all[tr_idx])
    Xtr_g, Str = nrm.transform(X_all[tr_idx], S_all[tr_idx])
    Xte_g, Ste = nrm.transform(X_all[te_idx], S_all[te_idx])

    if protocol == "global":
        # Take the normalizer's own output rather than recomputing an identical formula,
        # so the reference arm is exactly the fold-wise global pipeline by construction.
        return Xtr_g, Str, Xte_g, Ste

    gmu, gsd = _global_stats(X_all[tr_idx])
    sstats = _subject_stats(X_all, pid_all) if protocol.startswith("subject") else None
    Xtr = _apply(X_all[tr_idx], pid_all[tr_idx], protocol, gmu, gsd, sstats)
    Xte = _apply(X_all[te_idx], pid_all[te_idx], protocol, gmu, gsd, sstats)
    return Xtr, Str, Xte, Ste


def assert_statics_invariant(X_all, S_all, pid_all, tr_idx, te_idx, tol=0.0):
    """The arms must differ ONLY in the CGM transform. Compare the actual normalized static
    matrices across all five arms and abort on any difference."""
    ref = None
    for p in PROTOCOLS:
        _, Str, _, Ste = normalize_split(X_all, S_all, pid_all, tr_idx, te_idx, p)
        if ref is None:
            ref, ref_p = (Str, Ste), p
            continue
        for k, (a, b) in enumerate(zip(ref, (Str, Ste))):
            d = float(np.abs(a - b).max())
            if d > tol:
                raise SystemExit(f"[abort] static features differ between arms {ref_p!r} "
                                 f"and {p!r} on {'train' if k == 0 else 'test'}: "
                                 f"max|diff| = {d:.3e}. Something other than the CGM "
                                 f"transform varies -- the experiment is not controlled.")
    return True


if __name__ == "__main__":
    # Self-test on the real bundle: the level diagnostic must separate the arms the way the
    # design claims, and the statics must be invariant across all five.
    from shanghai_data import ShanghaiBundle
    b = ShanghaiBundle()
    subs = np.unique(b.pid)
    tr = np.where(np.isin(b.pid, subs[: int(0.7 * len(subs))]))[0]
    te = np.where(~np.isin(b.pid, subs[: int(0.7 * len(subs))]))[0]
    print(f"minmax channels {MINMAX_CHANNELS} vs C={b.C} -> "
          f"{'no min-max fires (subject_z is pure per-subject z on CGM)' if min(MINMAX_CHANNELS) >= b.C else 'MIN-MAX ACTIVE'}")
    print(f"{'arm':16s} {'level_retained':>15s}   properties")
    for p in PROTOCOLS:
        _, _, Xte, _ = normalize_split(b.X, b.S, b.pid, tr, te, p)
        lvl = between_subject_level_retained(Xte, b.pid[te])
        q = PROPERTIES[p]
        print(f"{p:16s} {lvl:15.4f}   level_preserved={str(q['level_preserved']):5s} "
              f"leakage={str(q['leakage']):5s}")
    print(f"\nstatics invariant across all five arms: "
          f"{assert_statics_invariant(b.X, b.S, b.pid, tr, te)}")
