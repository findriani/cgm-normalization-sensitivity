"""
shanghai_extract.py -- build a CGMacros-compatible postprandial bundle from Shanghai T2DM
=========================================================================================
EXTERNAL VALIDATION for the normalization claim. Produces an .npz with the SAME key names
and row semantics as `dexcom_binned_prediction_raw.npz`, so the existing training and
analysis code can consume it with only the documented differences below.

    python shanghai_extract.py --report      # attrition + audit only, writes NOTHING
    python shanghai_extract.py               # writes the bundle

WHAT THIS DATASET CAN AND CANNOT TEST
-------------------------------------
CAN:    N1 (per-subject normalization inverts modality attribution) -- the static panel
        carries HbA1c/age/BMI/sex, which is exactly the branch that retains the participant's
        glucose baseline when per-subject scaling strips it from the CGM branch.
        N4 (architecture does not matter).
PARTLY: N2 -- person/clinical context yes, MEAL CONTENT NO. Shanghai logs dietary intake as
        free text (Chinese), with no macronutrients. `meal` is absent, not zero.
        N3 -- 60 vs 30 vs 15 min is testable at 15-min granularity; 1-min vs 5-min is not.
CANNOT: anything requiring wearable activity or heart rate. Shanghai has neither.

DIFFERENCES FROM THE CGMacros BUNDLE -- read before using
----------------------------------------------------------
1. X is (n, 5, 1), not (n, 12, 4). Five 15-minute slots at t = -60, -45, -30, -15, 0 and
   ONE channel (CGM). n_dyn = 0. Any model that slices X[:, :, 1:] must be run with the
   dynamic branch disabled.
2. `static` is 12 columns, not 17, and the LAYOUT IS DIFFERENT (see STATIC_LAYOUT). The
   meal-macronutrient block has no counterpart. Do not reuse CGMacros column indices.
3. `diagnosis` is NOT a clinical diagnosis. Every patient here is Type 2. The field carries
   an HbA1c tertile purely so the existing diagnosis-stratified CV splitter has a balanced
   stratification variable. Named `diagnosis` only for schema compatibility.
4. HbA1c is converted from Shanghai's IFCC mmol/mol to NGSP % so it is on the CGMacros scale.
5. Grouping is by PATIENT, not by file. Nine patients have two visits; treating files as
   independent would reintroduce exactly the participant leakage this paper is about.
6. This is a hospitalized cohort on insulin and oral hypoglycemic agents. Postprandial
   response is pharmacologically modified in a way CGMacros is not. This is a population
   difference to disclose, not a defect.

EVERY INCLUSION RULE IS A KNOB BELOW, and --report prints the attrition at each stage so the
criteria can be audited before anything is written.
=========================================================================================
"""
import os
import re
import sys
import json
import glob
import time
import hashlib
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- source
SRC = os.environ.get("SHANGHAI_DIR", os.path.join(
    "E:", os.sep, "Ilkom ULM", "Penelitian", "Prediksi Glukosa", "Bima 2026",
    "UserEmbedding", "paper_personalization", "Paper1",
    "NFP for Few Shot Postprandial Glucose", "AIH", "dataset"))
CGM_DIR = os.path.join(SRC, "Shanghai_T2DM")
SUMMARY = os.path.join(SRC, "Shanghai_T2DM_Summary.xlsx")

OUT_NPZ = os.path.join(HERE, "shanghai_postprandial_raw.npz")
OUT_MEALS = os.path.join(HERE, "shanghai_meal_text.csv")       # free text, for later coding
OUT_ATTRITION = os.path.join(HERE, "shanghai_attrition.csv")
OUT_MANIFEST = os.path.join(HERE, "shanghai_manifest.json")

# ----------------------------------------------------------------- inclusion knobs
PRE_OFFSETS = [-60, -45, -30, -15, 0]   # minutes relative to meal; 0 = last pre-meal reading
HORIZONS = [30, 60, 120]                # minutes post-meal, matches CGMacros
SLOT_TOL = 7                            # +/- min when snapping to a grid slot (< half of 15)
TARGET_TOL = 8                          # +/- min when locating a target reading
REQUIRE_ALL_SLOTS = True                # every pre-meal slot must be present (no imputation)

# A meal is only usable if its pre-meal window is not contaminated by an earlier meal, and
# (optionally) if no other meal lands inside the prediction horizon. CGMacros required
# "60 consecutive minutes of pre-meal data"; this is the equivalent rule made explicit.
MIN_GAP_BEFORE = 60                     # min since the previous meal
EXCLUDE_MEAL_WITHIN_HORIZON = True      # drop if another meal occurs within 120 min after
HORIZON_GUARD = 120

# CGMacros kept only meals labelled breakfast / lunch / dinner. Shanghai has no labels, so
# the meal type is derived from clock time and meals outside these windows are dropped.
MEAL_WINDOWS = {"breakfast": (5, 10), "lunch": (11, 15), "dinner": (17, 22)}
REQUIRE_MEAL_TYPE = True

CGM_MIN, CGM_MAX = 20.0, 600.0          # physiological sanity bounds (mg/dL)
MIN_MEALS_PER_PATIENT = 5               # drop patients too sparse to contribute a fold

# ----------------------------------------------------------------- static layout
# NOTE: deliberately NOT the CGMacros layout. There is no macronutrient block.
STATIC_LAYOUT = [
    "age", "sex", "bmi", "hba1c_pct",                       # 0-3  person (CGMacros 0-3)
    "hour_sin", "hour_cos", "is_morning", "is_evening",     # 4-7   time
    "is_weekend", "meal_breakfast", "meal_lunch", "meal_dinner",  # 8-11 time / meal type
]
PERSON_IDX = [0, 1, 2, 3]
TIME_IDX = [4, 5, 6, 7, 8, 9, 10, 11]

# Summary-sheet column names, kept verbatim so a rename upstream fails loudly.
COL_PID = "Patient Number"
COL_SEX = "Gender (Female=1, Male=2)"
COL_AGE = "Age (years)"
COL_BMI = "BMI (kg/m2)"
COL_HBA1C = "HbA1c (mmol/mol)"


def hba1c_ifcc_to_ngsp(mmol_per_mol):
    """IFCC (mmol/mol) -> NGSP (%). Standard master equation, so the value is on the same
    scale as the CGMacros HbA1c feature."""
    return 0.09148 * np.asarray(mmol_per_mol, dtype=float) + 2.152


def meal_type_of(hour):
    for name, (lo, hi) in MEAL_WINDOWS.items():
        if lo <= hour <= hi:
            return name
    return None


# ----------------------------------------------------------------- loading
def patient_id_of(path):
    """'2001_1_20201117.xlsx' -> '2001'. The middle token is the VISIT index; two visits by
    the same patient must share a participant id or CV folds leak."""
    return os.path.basename(path).split("_")[0]


def load_summary():
    s = pd.read_excel(SUMMARY)
    missing = [c for c in (COL_PID, COL_SEX, COL_AGE, COL_BMI, COL_HBA1C) if c not in s.columns]
    if missing:
        raise SystemExit(f"[abort] summary sheet is missing {missing}. Columns present:\n"
                         f"{list(s.columns)}")
    s = s.copy()
    s["pid"] = s[COL_PID].astype(str).str.extract(r"(\d+)")[0]
    # One row per patient. Repeat visits carry near-identical demographics; take the first
    # and record how many collapsed so the choice is visible in the manifest.
    g = s.groupby("pid")
    static = g.first()
    static["_n_visits"] = g.size()
    # This sheet writes missing values as the STRING "/", which makes every static column
    # object-dtype and made `.astype(float)` raise. Coerce, then drop -- do NOT impute.
    # HbA1c is the feature that carries each patient's glucose baseline, and it is the
    # whole reason this cohort can test C1 at all (see README). Filling it with a cohort
    # median would blunt precisely the signal under test, so a patient missing it cannot
    # contribute. This matches the script's stance elsewhere: decision 6 drops meals with
    # an incomplete pre-meal window rather than interpolating them.
    num = {c: pd.to_numeric(static[c], errors="coerce")
           for c in (COL_AGE, COL_SEX, COL_BMI, COL_HBA1C)}
    out = pd.DataFrame({
        "age": num[COL_AGE],
        "sex": num[COL_SEX] - 1.0,                             # -> 0 = female, 1 = male
        "bmi": num[COL_BMI],
        "hba1c_pct": hba1c_ifcc_to_ngsp(num[COL_HBA1C]),
        "_n_visits": static["_n_visits"],
    })
    bad = out[["age", "sex", "bmi", "hba1c_pct"]].isna().any(axis=1)
    dropped = sorted(out.index[bad])
    if dropped:
        per_col = {c: int(out[c].isna().sum())
                   for c in ("age", "sex", "bmi", "hba1c_pct") if out[c].isna().any()}
        print(f"summary       : dropped {len(dropped)} patient(s) with a missing static "
              f"{per_col} -- coerced from non-numeric (e.g. '/'), not imputed")
        out = out[~bad]
    return out, set(dropped)


def read_cgm_file(path):
    d = pd.read_excel(path)
    dc = [c for c in d.columns if "Date" in str(c)]
    gc = [c for c in d.columns if "CGM" in str(c)]
    mc = [c for c in d.columns if "Dietary" in str(c)]
    if not (dc and gc and mc):
        return None, f"missing columns (has {list(d.columns)[:4]}...)"
    d = d[[dc[0], gc[0], mc[0]]].copy()
    d.columns = ["t", "cgm", "meal"]
    d["t"] = pd.to_datetime(d["t"], errors="coerce")
    d = d.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    return d, None


# ----------------------------------------------------------------- extraction
def extract_file(path, attrition):
    """Return a list of usable meal records from one patient-visit file."""
    d, err = read_cgm_file(path)
    if d is None:
        attrition["unreadable"] += 1
        return [], err

    cgm = d.dropna(subset=["cgm"]).copy()
    cgm = cgm[(cgm.cgm >= CGM_MIN) & (cgm.cgm <= CGM_MAX)]
    if len(cgm) < 10:
        attrition["too_few_cgm"] += 1
        return [], "fewer than 10 valid CGM readings"

    tt = cgm["t"].values.astype("datetime64[m]").astype(np.int64)
    vv = cgm["cgm"].values.astype(float)

    meals = d[d["meal"].notna()].copy()
    attrition["meal_events"] += len(meals)
    if meals.empty:
        return [], None
    mt = meals["t"].values.astype("datetime64[m]").astype(np.int64)

    pid = patient_id_of(path)
    recs = []
    for i, (_, m) in enumerate(meals.iterrows()):
        t0 = int(mt[i])
        ts = pd.Timestamp(m["t"])

        mtype = meal_type_of(ts.hour)
        if REQUIRE_MEAL_TYPE and mtype is None:
            attrition["no_meal_type"] += 1
            continue

        prev = mt[mt < t0]
        if len(prev) and (t0 - prev.max()) < MIN_GAP_BEFORE:
            attrition["meal_too_close_before"] += 1
            continue
        if EXCLUDE_MEAL_WITHIN_HORIZON:
            nxt = mt[(mt > t0) & (mt <= t0 + HORIZON_GUARD)]
            if len(nxt):
                attrition["meal_within_horizon"] += 1
                continue

        # pre-meal grid: snap each offset to the nearest reading within SLOT_TOL
        window, ok = [], True
        for off in PRE_OFFSETS:
            want = t0 + off
            k = int(np.abs(tt - want).argmin())
            if abs(int(tt[k]) - want) > SLOT_TOL:
                ok = False
                break
            window.append(vv[k])
        if not ok and REQUIRE_ALL_SLOTS:
            attrition["incomplete_pre_window"] += 1
            continue

        ys, ok = [], True
        for h in HORIZONS:
            want = t0 + h
            k = int(np.abs(tt - want).argmin())
            if abs(int(tt[k]) - want) > TARGET_TOL:
                ok = False
                break
            ys.append(vv[k])
        if not ok:
            attrition["missing_target"] += 1
            continue

        recs.append({
            "pid": pid, "file": os.path.basename(path), "t": ts,
            "meal_type": mtype, "meal_text": str(m["meal"]),
            **{f"cgm_{off}": w for off, w in zip(PRE_OFFSETS, window)},
            **{f"y{h}": y for h, y in zip(HORIZONS, ys)},
        })
        attrition["kept"] += 1
    return recs, None


def build_static(rows, person):
    """(n, 12) static matrix in STATIC_LAYOUT order. Person attributes are joined per
    patient; time attributes are derived per meal."""
    n = len(rows)
    S = np.zeros((n, len(STATIC_LAYOUT)), dtype="float32")
    for i, r in enumerate(rows.itertuples()):
        p = person.loc[r.pid]
        ts = r.t
        hour = ts.hour + ts.minute / 60.0
        S[i] = [
            p.age, p.sex, p.bmi, p.hba1c_pct,
            np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
            1.0 if 5 <= ts.hour < 12 else 0.0,
            1.0 if 17 <= ts.hour < 24 else 0.0,
            1.0 if ts.dayofweek >= 5 else 0.0,
            1.0 if r.meal_type == "breakfast" else 0.0,
            1.0 if r.meal_type == "lunch" else 0.0,
            1.0 if r.meal_type == "dinner" else 0.0,
        ]
    return S


def hba1c_tertiles(person, pids):
    """Stratification variable for the participant-level CV splitter. NOT a diagnosis --
    every patient in this cohort is Type 2. Tertiles are computed over the PATIENTS that
    survive extraction, so the three strata are balanced in the split that actually runs."""
    h = person.loc[pids, "hba1c_pct"].astype(float)
    q = h.quantile([1 / 3, 2 / 3]).values
    return np.digitize(h.values, q).astype(int), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="print the audit and write NOTHING")
    args = ap.parse_args()

    t0 = time.time()
    files = sorted(glob.glob(os.path.join(CGM_DIR, "*.xls"))) + \
        sorted(glob.glob(os.path.join(CGM_DIR, "*.xlsx")))
    if not files:
        raise SystemExit(f"[abort] no .xls/.xlsx under {CGM_DIR}")
    person, no_static = load_summary()
    print(f"source        : {CGM_DIR}")
    print(f"patient-visits: {len(files)} files")
    print(f"summary rows  : {len(person)} patients "
          f"({int((person._n_visits > 1).sum())} with repeat visits)")

    attrition = dict(meal_events=0, unreadable=0, too_few_cgm=0, no_meal_type=0,
                     meal_too_close_before=0, meal_within_horizon=0,
                     incomplete_pre_window=0, missing_target=0, kept=0)
    all_recs, problems = [], []
    for f in files:
        recs, err = extract_file(f, attrition)
        all_recs += recs
        if err:
            problems.append((os.path.basename(f), err))

    if not all_recs:
        raise SystemExit("[abort] no usable meals -- loosen the knobs or check the source")
    r = pd.DataFrame(all_recs)

    # patients with too few meals cannot support participant-level CV
    counts = r.groupby("pid").size()
    keep = set(counts[counts >= MIN_MEALS_PER_PATIENT].index)
    dropped_pat = sorted(set(counts.index) - keep)
    attrition["dropped_sparse_patient_meals"] = int((~r.pid.isin(keep)).sum())
    r = r[r.pid.isin(keep)].reset_index(drop=True)

    # Patients dropped for a missing static are EXPECTED and become an attrition row.
    # A patient with no summary row at all is a schema mismatch and still aborts -- the two
    # causes must not be conflated, or a broken join would look like ordinary attrition.
    drop_no_static = sorted(set(r.pid) & no_static)
    if drop_no_static:
        attrition["dropped_missing_static_meals"] = int(r.pid.isin(no_static).sum())
        print(f"              : dropped {len(drop_no_static)} patient(s) "
              f"({attrition['dropped_missing_static_meals']} meals) for a missing static")
        r = r[~r.pid.isin(no_static)].reset_index(drop=True)

    missing_static = sorted(set(r.pid) - set(person.index))
    if missing_static:
        raise SystemExit(f"[abort] {len(missing_static)} patients have CGM but no summary "
                         f"row at all: {missing_static[:8]}. This is a join failure, not "
                         f"attrition -- check the patient-id extraction.")

    pids = sorted(r.pid.unique())
    X = np.stack([r[[f"cgm_{o}" for o in PRE_OFFSETS]].values.astype("float32")], axis=-1)
    S = build_static(r, person)
    y = r[[f"y{h}" for h in HORIZONS]].values.astype("float32")
    strat_by_pid, cuts = hba1c_tertiles(person, pids)
    strat_map = dict(zip(pids, strat_by_pid))
    diagnosis = np.array([strat_map[p] for p in r.pid], dtype=int)

    assert X.shape == (len(r), len(PRE_OFFSETS), 1), X.shape
    assert S.shape == (len(r), len(STATIC_LAYOUT)), S.shape
    assert not np.isnan(X).any() and not np.isnan(S).any() and not np.isnan(y).any()

    # ---------------- audit ----------------
    order = ["meal_events", "no_meal_type", "meal_too_close_before", "meal_within_horizon",
             "incomplete_pre_window", "missing_target", "kept",
             "dropped_sparse_patient_meals"]
    att = pd.DataFrame({"stage": order, "n": [attrition[k] for k in order]})
    print("\n=== ATTRITION ===")
    print(att.to_string(index=False))
    if problems:
        print(f"\nfiles with issues ({len(problems)}): {problems[:5]}")
    print(f"\n=== FINAL BUNDLE ===")
    print(f"  meals        : {len(r)}")
    print(f"  patients     : {len(pids)}  (dropped {len(dropped_pat)} with "
          f"<{MIN_MEALS_PER_PATIENT} meals)")
    print(f"  X            : {X.shape}  (5 slots x 15 min, CGM only, n_dyn=0)")
    print(f"  static       : {S.shape}  {STATIC_LAYOUT}")
    print(f"  meals/patient: median {int(r.groupby('pid').size().median())}, "
          f"range {r.groupby('pid').size().min()}-{r.groupby('pid').size().max()}")
    print(f"  HbA1c strata : {np.bincount(strat_by_pid).tolist()} patients "
          f"(cuts at {np.round(cuts, 2).tolist()} %)")
    print(f"  meal types   : {r.meal_type.value_counts().to_dict()}")
    print("\n  CGM window (mg/dL):")
    print(pd.DataFrame(X[:, :, 0], columns=[f"t{o:+d}" for o in PRE_OFFSETS])
          .describe().loc[["mean", "std", "min", "max"]].round(1).to_string())
    print("\n  targets (mg/dL):")
    print(pd.DataFrame(y, columns=[f"y{h}" for h in HORIZONS])
          .describe().loc[["mean", "std", "min", "max"]].round(1).to_string())
    print("\n  static:")
    print(pd.DataFrame(S, columns=STATIC_LAYOUT).describe()
          .loc[["mean", "std", "min", "max"]].round(2).to_string())

    # A CGMacros-comparable sanity number: how well does the last pre-meal reading alone do?
    p = X[:, -1, 0]
    for i, h in enumerate(HORIZONS):
        ss_res = float(np.sum((y[:, i] - p) ** 2))
        ss_tot = float(np.sum((y[:, i] - y[:, i].mean()) ** 2))
        print(f"  persistence floor R2_{h:<3} = {1 - ss_res / ss_tot:+.3f}   "
              f"RMSE = {np.sqrt(ss_res / len(y)):.1f} mg/dL")

    if args.report:
        print(f"\n--report: nothing written. ({time.time() - t0:.0f}s)")
        return

    # ---------------- write ----------------
    np.savez_compressed(
        OUT_NPZ, X=X, static=S, y=y,
        participant_id=np.array(r.pid.values, dtype=object),
        diagnosis=diagnosis,
        static_names=np.array(STATIC_LAYOUT, dtype=object),
        pre_offsets=np.array(PRE_OFFSETS), horizons=np.array(HORIZONS))
    r[["pid", "file", "t", "meal_type", "meal_text"]].to_csv(OUT_MEALS, index=False,
                                                             encoding="utf-8")
    att.to_csv(OUT_ATTRITION, index=False)
    json.dump({
        "source_dir": CGM_DIR, "n_files": len(files),
        "n_meals": int(len(r)), "n_patients": int(len(pids)),
        "X_shape": list(X.shape), "static_layout": STATIC_LAYOUT,
        "pre_offsets_min": PRE_OFFSETS, "horizons_min": HORIZONS,
        "knobs": {"SLOT_TOL": SLOT_TOL, "TARGET_TOL": TARGET_TOL,
                  "REQUIRE_ALL_SLOTS": REQUIRE_ALL_SLOTS,
                  "MIN_GAP_BEFORE": MIN_GAP_BEFORE,
                  "EXCLUDE_MEAL_WITHIN_HORIZON": EXCLUDE_MEAL_WITHIN_HORIZON,
                  "HORIZON_GUARD": HORIZON_GUARD, "MEAL_WINDOWS": MEAL_WINDOWS,
                  "REQUIRE_MEAL_TYPE": REQUIRE_MEAL_TYPE,
                  "MIN_MEALS_PER_PATIENT": MIN_MEALS_PER_PATIENT,
                  "CGM_BOUNDS": [CGM_MIN, CGM_MAX]},
        "hba1c": {"converted": "IFCC mmol/mol -> NGSP % via 0.09148*x + 2.152",
                  "tertile_cuts_pct": np.round(cuts, 4).tolist()},
        "diagnosis_field": "HbA1c tertile, NOT a clinical diagnosis (cohort is all T2D)",
        "attrition": attrition,
        # Two distinct exclusions, kept apart on purpose. `dropped_patients` is the sparsity
        # rule (decision 8); `dropped_missing_static` is the "/" coercion added 2 August.
        # Collapsing them would hide a cohort-defining choice inside a housekeeping filter.
        "dropped_patients": dropped_pat,
        "dropped_patients_reason": f"fewer than {MIN_MEALS_PER_PATIENT} usable meals",
        "dropped_missing_static": sorted(no_static),
        "dropped_missing_static_reason": "a required static (age/sex/BMI/HbA1c) is absent "
                                         "in the summary sheet, written there as '/'. "
                                         "Coerced to NaN and DROPPED, never imputed: HbA1c "
                                         "carries the participant's glucose baseline and is "
                                         "the feature this cohort exists to test.",
        "code_sha": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
        "numpy": np.__version__, "pandas": pd.__version__,
        "python": sys.version.split()[0],
    }, open(OUT_MANIFEST, "w"), indent=2)
    print(f"\nWrote:\n  {OUT_NPZ}\n  {OUT_MEALS}\n  {OUT_ATTRITION}\n  {OUT_MANIFEST}")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
