"""
registry.py -- the single source of truth for every number that may appear in the manuscript
=============================================================================================
THE REGISTRY STORES QUERIES, NOT VALUES. That is the whole design.

A registry that stored the numbers would reintroduce exactly the transcription error it
exists to catch: someone types +0.1733 into this file, the analysis is re-run, and now
there are two wrong copies instead of one. Every entry below names a CSV and the row that
identifies it; `audit_numbers.py` derives the value at run time. If a version changes a
result, this file does not change and every check downstream moves with the data.

Consequence worth stating plainly: **a green audit proves the manuscript agrees with the
CSVs on disk.** It does not prove the CSVs are right. Integrity of the runs themselves is
§4.6 / §6.3 / §17.1's job, not this file's.

Three things the entries encode that prose cannot:

  sign   Some quantities are reported in the manuscript with the opposite sign to the
         stored column. The context-block INFLATION is the stored `context adds beyond CGM`
         mechanism row, which is -0.2799 on disk and +0.280 in the paper, because the paper
         frames it as how much the transform ADDS to the context block. This is the §3.4
         vocabulary defect in numeric form and it is the single most likely sign error in
         the whole draft -- so it is declared here, once, and applied mechanically.

  dp     How many decimals the manuscript uses. The audit matches a literal at the entry's
         own precision AND at every shorter rounding, so 0.198, 0.1978 and 0.20 all match
         the same underlying value; 0.197 does not.

  scale  A unit conversion applied before reporting. Currently unused -- E3 already stores
         percentage points -- but kept because the first draft of this file assumed it did
         not, and multiplied five clinical results by 100.

Add an entry whenever a number earns a place in the text. Entries cost nothing and an
unregistered number is one the audit cannot vouch for.

-------------------------------------------------------------------------------------------
THE mg/dL SIGN CONVENTION IS NOT CONSISTENT ACROSS RESULT FILES. READ THIS BEFORE WRITING
ANY mg/dL NUMBER.
-------------------------------------------------------------------------------------------
At h = 60, adding CGM to the context block lowers RMSE by about 7 mg/dL. Three files record
that same physical fact with two different signs:

    e1_mgdl_horizons.csv        dRMSE_mgdl = +7.49    positive = error FALLS
    deep_sens_pmc_increments    dRMSE_mgdl = -7.04    negative = error FALLS
    attr_contrasts.csv          dRMSE_mgdl = -7.35    negative = error FALLS

Nothing is wrong with the analyses; the analyzers were written at different times and
differ in whether the column means "the reduction achieved" or "the change in RMSE". The
hazard is entirely in transcription. A drafter working from both files and copying signs
literally reports the deep model as ADDING 7 mg/dL of error while the forest removes it --
a contradiction between two adjacent sentences, in the estimator-agreement paragraph that
exists specifically to show the two models agree.

This registry normalises it: every mg/dL entry reports **the reduction as a positive
number**, and each declares the `sign` its own source needs. Take mg/dL numbers from
`values.md` and nowhere else. Recorded as a defect in the plan (SS15).
=============================================================================================
"""
from dataclasses import dataclass, field
from typing import Optional

# Row-key constants. Spelled once: these strings are compared against CSV cells, and a typo
# here surfaces as "matched 0 rows" rather than as a silently skipped check.
CGM_INC = "CGM adds beyond context"
CTX_INC = "context adds beyond CGM"
ACT_INC = "activity adds beyond CGM+context"
LEVEL_REMOVAL = "causal LEVEL REMOVAL (no leakage in either arm)"
CENTERING = "CENTERING (subject scaling + leakage held fixed)"
SCALING = "SCALING (subject centering held fixed)"

H = 60          # the primary horizon; all-horizon entries are generated in expand()


@dataclass(frozen=True)
class Entry:
    id: str
    label: str
    plan_section: str
    source: str                     # relative to `data preprocessed/`
    where: dict                     # column -> value; must select exactly one row
    cols: tuple                     # (estimate,) or (estimate, lo, hi)
    dp: int = 4
    sign: int = 1                   # +1 as stored; -1 when the paper flips it
    scale: float = 1.0              # multiply before reporting (E3: fraction -> points)
    unit: str = ""
    note: str = ""


E1 = "core/e1_interaction.csv"
E1M = "core/e1_mechanism.csv"
E1F = "core/e1_figure_data.csv"
E1H = "core/e1_mgdl_horizons.csv"
E2 = "normalization_sensitivity/deep_sens_interaction.csv"
E2I = "normalization_sensitivity/deep_sens_increments.csv"
E2B = "normalization_sensitivity/deep_sens_pmc_interaction.csv"
E4 = "normalization_sensitivity/attr_contrasts.csv"
E5 = "normalization_sensitivity/e5_interaction.csv"
E5I = "normalization_sensitivity/e5_increments.csv"
E3 = "core/e3_clinical.csv"
SH = "shanghai_external/shanghai_interaction.csv"
SHM = "shanghai_external/shanghai_mechanism.csv"
SHI = "shanghai_external/shanghai_increments.csv"
TRAD = "core/trad_paired.csv"


ENTRIES = [
    # ---- C1's primary estimand, both estimators -------------------------------------
    Entry("primary_rf", "Primary interaction ddR2, RF", "4.5", E1,
          {"family": "primary", "horizon": H, "increment": CGM_INC,
           "protocol_1": "global", "protocol_2": "subject_z"},
          ("ddR2", "lo", "hi")),
    Entry("primary_deep", "Primary interaction ddR2, deep mid-fusion", "5.2", E2,
          {"horizon": H, "increment": CGM_INC,
           "protocol_1": "global", "protocol_2": "subject_z"},
          ("ddR2", "lo", "hi")),

    # ---- the mechanism: level removal, leakage-free in BOTH arms ---------------------
    Entry("mech_rf", "Level removal (leakage-free) ddR2, RF", "4.7", E1M,
          {"horizon": H, "increment": CGM_INC, "isolates": LEVEL_REMOVAL},
          ("ddR2", "lo", "hi")),
    Entry("mech_deep", "Level removal (leakage-free) ddR2, deep", "6.2", E2B,
          {"horizon": H, "increment": CGM_INC,
           "protocol_1": "global", "protocol_2": "premeal_center"},
          ("ddR2", "lo", "hi")),
    Entry("mech_centering_rf", "Centering component ddR2, RF", "4.7", E1M,
          {"horizon": H, "increment": CGM_INC, "isolates": CENTERING},
          ("ddR2", "lo", "hi")),
    Entry("mech_scaling_rf", "Scaling component ddR2, RF (null)", "4.7", E1M,
          {"horizon": H, "increment": CGM_INC, "isolates": SCALING},
          ("ddR2", "lo", "hi")),

    # ---- the mirror: context-block inflation. SIGN IS FLIPPED IN THE PAPER -----------
    Entry("inflation_rf", "Context-block inflation, leakage-free RF", "4.8", E1M,
          {"horizon": H, "increment": CTX_INC, "isolates": LEVEL_REMOVAL},
          ("ddR2", "lo", "hi"), sign=-1,
          note="stored as `context adds beyond CGM` global-vs-premeal_center = -0.2799; "
               "the paper reports the INFLATION, i.e. +0.280. Sign flip is deliberate "
               "(SS3.4). Reporting -0.280 in the text is a defect, not a convention."),
    Entry("inflation_deep", "Context-block inflation, deep, published transform", "4.8", E2,
          {"horizon": H, "increment": CTX_INC,
           "protocol_1": "global", "protocol_2": "subject_z"},
          ("ddR2", "lo", "hi"), sign=-1,
          note="the INTERACTION (-0.1703 stored), not the per-arm increment. The per-arm "
               "figure under subject_z is 0.3388, and quoting that as the inflation "
               "overstates it twofold -- it includes the increment the reference arm "
               "already had. The first draft of this registry made exactly that error."),

    # ---- per-arm CGM increments: the 'falls from X to Y' sentence --------------------
    Entry("cgm_global_r2", "CGM increment, reference arm, R2", "4.8", E1F,
          {"kind": "within_arm", "horizon": H, "protocol": "global", "increment": CGM_INC},
          ("estimate", "lo", "hi")),
    Entry("cgm_subjz_r2", "CGM increment, published transform, R2", "4.8", E1F,
          {"kind": "within_arm", "horizon": H, "protocol": "subject_z", "increment": CGM_INC},
          ("estimate", "lo", "hi")),
    Entry("cgm_pmc_r2", "CGM increment, premeal_center arm, R2", "4.8", E1F,
          {"kind": "within_arm", "horizon": H, "protocol": "premeal_center",
           "increment": CGM_INC},
          ("estimate", "lo", "hi")),
    Entry("cgm_global_rmse", "CGM increment, reference arm, RMSE reduction", "13.8", E1H,
          {"horizon": H, "increment": CGM_INC, "kind": "within_arm", "arm": "global"},
          ("dRMSE_mgdl", "dRMSE_lo", "dRMSE_hi"), dp=2, sign=+1, unit="mg/dL",
          note="e1_mgdl_horizons stores the REDUCTION (positive = error falls), the "
               "opposite of attr_contrasts and the deep files. See the header."),
    Entry("cgm_subjz_rmse", "CGM increment, published transform, RMSE reduction", "13.8",
          E1H, {"horizon": H, "increment": CGM_INC, "kind": "within_arm",
                "arm": "subject_z"},
          ("dRMSE_mgdl", "dRMSE_lo", "dRMSE_hi"), dp=2, sign=+1, unit="mg/dL",
          note="the 'falls from 7.5 to 0.8 mg/dL' sentence pairs this with "
               "`cgm_global_rmse`. Both are reductions; neither is negative in the text."),
    Entry("mech_deep_rmse", "Level removal, deep, RMSE reduction, reference arm", "6.2",
          "normalization_sensitivity/deep_sens_pmc_increments.csv",
          {"horizon": H, "increment": CGM_INC, "arm": "global"},
          ("dRMSE_mgdl", "dRMSE_lo", "dRMSE_hi"), dp=2, sign=-1, unit="mg/dL",
          note="OPPOSITE stored convention to e1_mgdl_horizons -- stored -7.04, reported "
               "as a 7.04 mg/dL reduction. This entry and `cgm_global_rmse` are the pair "
               "that would contradict each other if signs were copied literally."),


    # ---- Fair traditional-ML baseline -------------------------------------------------
    Entry("fair_deep_rf", "Matched deep model versus 5-minute Random Forest", "fair baseline",
          TRAD, {"deep": "dl_current_binned", "baseline": "random_forest@binned"},
          ("dR2_60", "R2_lo", "holm_p"), dp=3,
          note="The reported upper CI bound rounds to 0.060, which already coincides with a "
               "registered level-statistic bound; the estimate, lower bound, and adjusted p "
               "value uniquely trace this comparison."),
    # ---- E4: the level is one reading ------------------------------------------------
    Entry("e4_shape", "Shape beyond level (full window vs one reading)", "7.3", E4,
          {"horizon": H, "contrast": "shape beyond level (= full window vs one reading)"},
          ("dR2", "lo", "hi"), dp=3,
          note="EQUIVALENT against delta = 0.02. The only contrast in the paper where an "
               "equivalence verdict is reachable rather than a precision artifact (SS9.3)."),
    Entry("e4_level", "Level only, R2", "7.3", E4,
          {"horizon": H, "contrast": "level only"}, ("dR2", "lo", "hi"), dp=3),
    Entry("e4_level_rmse", "Level only, RMSE", "7.3", E4,
          {"horizon": H, "contrast": "level only"},
          ("dRMSE_mgdl", "dRMSE_lo", "dRMSE_hi"), dp=2, unit="mg/dL",
          note="stored negative (error falls). Paper wording follows SS7.3."),
    Entry("e4_level_stat", "Level statistic: last bin vs 12-bin mean", "7.4", E4,
          {"horizon": H, "contrast": "level statistic: last bin vs 12-bin mean"},
          ("dR2", "lo", "hi"), dp=3,
          note="the post-hoc deviation that SS7.4 obliges us to disclose."),

    # ---- E5: the negative control ----------------------------------------------------
    Entry("e5_interaction", "Activity interaction ddR2 (specificity)", "8.3", E5,
          {"horizon": H, "increment": ACT_INC, "arm_1": "global", "arm_2": "subject_z"},
          ("ddR2", "lo", "hi")),
    Entry("e5_activity_own", "Activity's own increment, reference arm", "8.2", E5I,
          {"horizon": H, "increment": ACT_INC, "arm": "global"}, ("dR2", "lo", "hi")),

    # ---- Shanghai --------------------------------------------------------------------
    Entry("shanghai_primary", "Cross-cohort primary interaction ddR2", "10.1", SH,
          {"family": "primary", "horizon": H, "increment": CGM_INC,
           "protocol_1": "global", "protocol_2": "subject_z"},
          ("ddR2", "lo", "hi")),
    Entry("shanghai_mech", "Cross-cohort level removal ddR2", "10.1", SHM,
          {"horizon": H, "increment": CGM_INC, "isolates": LEVEL_REMOVAL},
          ("ddR2", "lo", "hi")),
    Entry("shanghai_cgm_global", "Shanghai CGM increment, reference arm", "10.1b", SHI,
          {"horizon": H, "increment": CGM_INC, "protocol": "global"}, ("dR2", "lo", "hi"),
          note="carries the 52%-vs-88% proportional-cut comparison (SS10.1b). Do not quote "
               "the interaction without it."),
    Entry("shanghai_cgm_subjz", "Shanghai CGM increment, published transform", "10.1b", SHI,
          {"horizon": H, "increment": CGM_INC, "protocol": "subject_z"},
          ("dR2", "lo", "hi")),

    # ---- E3: clinical bands, stored as fractions, written as percentage points --------
    # E3 already stores percentage points; an earlier draft applied scale=100
    # here and printed "+628 points of meals", which is not a quantity that can
    # exist. Left as a comment because the error was silent -- every check passed.
    Entry("e3_clarkeA_interaction", "Clarke A interaction (points)", "11.3", E3,
          {"horizon": H, "increment": CGM_INC, "kind": "interaction", "metric": "clarke_A"},
          ("estimate", "lo", "hi"), dp=2, unit="points"),
    Entry("e3_w20_interaction", "Within 20 mg/dL interaction (points)", "11.3", E3,
          {"horizon": H, "increment": CGM_INC, "kind": "interaction", "metric": "within_20"},
          ("estimate", "lo", "hi"), dp=2, unit="points"),
    Entry("e3_w10_interaction", "Within 10 mg/dL interaction (points)", "11.3", E3,
          {"horizon": H, "increment": CGM_INC, "kind": "interaction", "metric": "within_10"},
          ("estimate", "lo", "hi"), dp=2, unit="points"),
]


def by_id(eid):
    for e in ENTRIES:
        if e.id == eid:
            return e
    raise KeyError(eid)
