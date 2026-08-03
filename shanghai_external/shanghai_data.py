"""
shanghai_data.py -- loader for the ShanghaiT2DM bundle
=======================================================
`lightdl_data.DataBundle` assumes CGMacros' shape: 12 bins x 4 channels, 17 statics in a
fixed layout. Shanghai is 5 slots x 1 channel and 12 statics in a DIFFERENT layout, so it
needs its own loader. Everything downstream of the substrate -- splits, estimator, seeding,
metrics, inference -- is imported from the CGMacros pipeline unchanged.

TWO LAYOUT TRAPS, BOTH LIVE
----------------------------
1. `ablation_run.clean_static` winsorizes static columns 4-8, which are CGMacros' MEAL
   MACRONUTRIENTS. In Shanghai those same indices are hour_sin, hour_cos, is_morning,
   is_evening, is_weekend. Applying it is semantically wrong, so `clean_static` MUST NOT be
   called on this bundle. There is nothing to clean here -- Shanghai has no meal content --
   so the correct action is to skip it, which `shanghai_run.py` does explicitly.

2. `foldwise_normalizer.FoldNormalizer` log1p-transforms those same indices 4-8 when the
   training column is skewed. `hour_sin` lives in [-1, 1] and `log1p(-1) = -inf`, so reusing
   it would not merely mis-scale the features, it would poison them with infinities. See
   `shanghai_normalizers.ShanghaiFoldNormalizer`.

The z-score indices are the lucky case: CGMacros z-scores age=0, bmi=2, hba1c=3, and
Shanghai's first four statics are age, sex, bmi, hba1c_pct in that order. That coincidence
is ASSERTED below rather than relied upon silently.

WHAT "CONTEXT" MEANS HERE, AND WHY IT IS NOT CGMACROS' CONTEXT
---------------------------------------------------------------
CGMacros' `static_all` = person + MEAL CONTENT + time. Shanghai has no meal-content
modality (the dietary entries are free text; see the extraction README, decision 7), so
`static_all` here = person + time only. Any comparison against CGMacros' context increment
is therefore NOT like-for-like, and the manuscript must not present it as a replication of
the meal-content result. It replicates C1 -- normalization's effect on CGM-versus-context
attribution -- and nothing else.
=======================================================
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
for _p in (CORE, HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Splits are imported, not reimplemented: participant-level, stratified, same base seed.
from lightdl_data import make_repeated_splits                      # noqa: F401
from foldwise_normalizer import STATIC_ZSCORE_IDX as _CGM_ZSCORE

NPZ = os.path.join(HERE, "shanghai_postprandial_raw.npz")

# Shanghai static layout -- 12 columns, NOT CGMacros' 17.
STATIC_NAMES = ["age", "sex", "bmi", "hba1c_pct",
                "hour_sin", "hour_cos", "is_morning", "is_evening", "is_weekend",
                "meal_breakfast", "meal_lunch", "meal_dinner"]
IDX_PERSON = [0, 1, 2, 3]                       # age, sex, bmi, hba1c
IDX_TIME = [4, 5, 6, 7, 8, 9, 10, 11]           # clock + meal-type indicators
IDX_MEAL = []                                   # NO meal content in this cohort

# Continuous statics to z-score. Binary and cyclical columns pass through untouched.
ZSCORE_IDX = {"age": 0, "bmi": 2, "hba1c": 3}
PASSTHROUGH_IDX = [i for i in range(len(STATIC_NAMES)) if i not in ZSCORE_IDX.values()]


class ShanghaiBundle:
    """Mirrors the attribute names of `lightdl_data.DataBundle` so downstream code that
    reads `.X`, `.y`, `.S`, `.pid`, `.diag`, `.T`, `.n_static` works unchanged."""

    def __init__(self, path=NPZ):
        if not os.path.exists(path):
            raise SystemExit(f"[abort] {path} not found -- run shanghai_extract.py first")
        z = np.load(path, allow_pickle=True)
        self.X = z["X"].astype(np.float64)              # (n, 5, 1) raw mg/dL
        self.S = z["static"].astype(np.float64)         # (n, 12)
        self.y = z["y"].astype(np.float64)              # (n, 3) mg/dL at 30/60/120
        self.pid = np.array([str(p) for p in z["participant_id"]])
        self.diag = z["diagnosis"].astype(int)          # HbA1c tertile -- NOT a diagnosis
        self.static_names = [str(s) for s in z["static_names"]]
        self.pre_offsets = z["pre_offsets"].tolist()
        self.horizons = z["horizons"].tolist()
        self.n, self.T, self.C = self.X.shape
        self.n_static = self.S.shape[1]
        self._validate()

    def _validate(self):
        if self.static_names != STATIC_NAMES:
            raise SystemExit(f"[abort] static layout changed.\n  expected: {STATIC_NAMES}\n"
                             f"  found:    {self.static_names}\n"
                             f"Every index constant in this file is now wrong.")
        if self.C != 1:
            raise SystemExit(f"[abort] expected 1 CGM channel, found {self.C}. The "
                             f"single-channel assumptions in shanghai_normalizers are void.")
        for name, a in (("X", self.X), ("static", self.S), ("y", self.y)):
            if not np.isfinite(a).all():
                raise SystemExit(f"[abort] {name} contains non-finite values; the "
                                 f"extraction is supposed to drop rather than impute.")
        if not (len(self.X) == len(self.S) == len(self.y) == len(self.pid) == len(self.diag)):
            raise SystemExit("[abort] arrays disagree on length.")
        # The coincidence this module depends on, checked rather than assumed.
        if dict(_CGM_ZSCORE) != ZSCORE_IDX:
            raise SystemExit(f"[abort] CGMacros z-score indices {dict(_CGM_ZSCORE)} no "
                             f"longer match Shanghai's {ZSCORE_IDX}. They coincided by "
                             f"luck; that luck has run out. Set ZSCORE_IDX explicitly.")

    def __repr__(self):
        return (f"ShanghaiBundle(n={self.n}, T={self.T}, C={self.C}, "
                f"static={self.n_static}, participants={len(np.unique(self.pid))})")


def feats(X, S, groups):
    """Shanghai's `_feats`. Deliberately NOT imported from ablation_run: that version
    slices `S[:, IDX_MEAL]` with CGMacros' indices, which point at clock features here."""
    parts = []
    if "cgm" in groups:
        parts.append(X[:, :, 0])
    if "dyn" in groups:
        raise SystemExit("[abort] Shanghai has no wearable/activity channels; 'dyn' is not "
                         "an available modality in this cohort.")
    if "meal" in groups:
        raise SystemExit("[abort] Shanghai has no meal-content modality -- the dietary "
                         "entries are un-coded free text (extraction README, decision 7). "
                         "Do not silently substitute person+time for it.")
    if "person" in groups:
        parts.append(S[:, IDX_PERSON])
    if "time" in groups:
        parts.append(S[:, IDX_TIME])
    if not parts:
        raise SystemExit(f"[abort] no known feature groups in {groups}")
    return np.concatenate(parts, axis=1)


# Config -> feature groups. Names match ablation_run.CONFIGS where the config exists there,
# so downstream analysis code needs no translation table.
CONFIGS = {
    "persistence": None,                    # predict the t=0 pre-meal reading
    "cgm": ["cgm"],
    "static_all": ["person", "time"],       # person + time ONLY -- no meal content
    "cgm_static": ["cgm", "person", "time"],
}


if __name__ == "__main__":
    b = ShanghaiBundle()
    print(b)
    print(f"  window offsets : {b.pre_offsets} min")
    print(f"  horizons       : {b.horizons} min")
    print(f"  strata (meals) : {dict(zip(*np.unique(b.diag, return_counts=True)))}")
    for name, groups in CONFIGS.items():
        if groups is None:
            continue
        print(f"  {name:12s} -> {feats(b.X, b.S, groups).shape[1]:2d} features {groups}")
