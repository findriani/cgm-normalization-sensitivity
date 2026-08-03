# -*- coding: utf-8 -*-
"""CGMacros Data Preprocessing for Glucose Prediction

Modified to predict glucose levels at 30, 60, and 120 minutes after meals with two input versions:
- **Raw version**: 60 timesteps (1-minute resolution)
- **Binned version**: 12 timesteps (5-minute bins)

Usage:
    1. Update BASE_DIR below to point to your CGMacros dataset directory
    2. Ensure directory contains: bio.csv and participant folders (CGMacros-XXX/)
    3. Run: python glucose_prediction_preprocessing.py
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.stats import skew

# Configuration - UPDATE THESE PATHS
BASE_DIR = "./CGMacros"  # Change to your dataset path (contains bio.csv and CGMacros-XXX folders)
SAVE_DIR = os.path.join(BASE_DIR, "Prediction")
os.makedirs(SAVE_DIR, exist_ok=True)

# Parameters
WINDOW_SIZE = 60  # 60 minutes before meal
BIN_SIZE = 5  # 5 minutes per bin
N_BINS = WINDOW_SIZE // BIN_SIZE  # 12 bins
TARGET_HORIZONS = [30, 60, 120]  # minutes after meal
CGM_SOURCES = ["Libre GL", "Dexcom GL"]

print(f"Window size: {WINDOW_SIZE} minutes")
print(f"Number of bins: {N_BINS}")
print(f"Target horizons: {TARGET_HORIZONS} minutes")

"""## Load Participant Metadata"""

# Load participant info
bio_df = pd.read_csv(os.path.join(BASE_DIR, "bio.csv"))
bio_df["participant_id"] = bio_df["subject"].apply(lambda x: f"CGMacros-{int(x):03d}")

# Create diagnosis classification
def classify_hba1c(hba1c):
    if pd.isna(hba1c):
        return -1  # Unknown
    elif hba1c < 5.7:
        return 0  # Healthy
    elif hba1c <= 6.4:
        return 1  # Pre-diabetes
    else:
        return 2  # Type 2 Diabetes

bio_df["Diagnosis"] = bio_df["A1c PDL (Lab)"].apply(classify_hba1c)

# Create mappings for both static features and diagnosis
bio_static_map = bio_df.set_index("participant_id")[["Age", "Gender", "BMI", "A1c PDL (Lab)"]].to_dict("index")
diagnosis_map = dict(zip(bio_df["participant_id"], bio_df["Diagnosis"]))

print(f"Loaded {len(bio_df)} participants")
print(f"\nDiagnosis distribution:")
print(bio_df["Diagnosis"].value_counts().sort_index())
print("\n0=Healthy, 1=Pre-diabetes, 2=Type 2 Diabetes, -1=Unknown")

"""## Helper Functions"""

def load_participant_file(folder):
    """Load and preprocess participant data"""
    csv_file = glob(os.path.join(folder, "*.csv"))[0]
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    df["timestamp"] = pd.to_datetime(df["Timestamp"])

    # HR: ffill + bfill, then fill any remaining with median
    if "HR" in df.columns:
        df["HR"] = df["HR"].ffill().bfill()
        if df["HR"].isnull().any():
            df["HR"] = df["HR"].fillna(df["HR"].median())
    else:
        df["HR"] = np.nan

    # METs/Intensity handling
    if "METs" in df.columns:
        pass  # use as is
    elif "Intensity" in df.columns:
        df["METs"] = df["Intensity"].map({0: 10, 1: 30}).fillna(10)
    else:
        df["METs"] = 10  # default if neither exists

    # Handle Calories (Activity)
    if "Calories (Activity)" in df.columns:
        pass  # use as is
    elif "Steps" in df.columns:
        df["Calories (Activity)"] = df["Steps"] * 0.05  # approximate mapping
    else:
        df["Calories (Activity)"] = 0.0  # fallback default

    return df

def get_static_features(pid, row):
    """Extract static features for a meal"""
    meta = bio_static_map.get(pid, {"Age": np.nan, "Gender": "Unknown", "BMI": np.nan, "A1c PDL (Lab)": np.nan})

    # Time features
    timestamp = row["timestamp"]
    hour = timestamp.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)  # Cyclical encoding
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Binary time indicators
    is_morning = 1 if 6 <= hour < 12 else 0
    is_evening = 1 if 18 <= hour < 24 else 0
    is_weekend = 1 if timestamp.dayofweek >= 5 else 0  # Saturday=5, Sunday=6

    # Meal type one-hot encoding
    meal_type = row.get("Meal Type", "").lower()
    is_breakfast = 1 if meal_type == "breakfast" else 0
    is_lunch = 1 if meal_type == "lunch" else 0
    is_dinner = 1 if meal_type == "dinner" else 0

    return [
        # Demographic features
        meta["Age"],
        1 if str(meta["Gender"]).lower().startswith("m") else 0,
        meta["BMI"],
        meta["A1c PDL (Lab)"],  # HbA1c value
        # Meal macronutrients
        row.get("Calories", np.nan),
        row.get("Carbs", np.nan),
        row.get("Protein", np.nan),
        row.get("Fat", np.nan),
        row.get("Fiber", np.nan),
        # Time features
        hour_sin,
        hour_cos,
        is_morning,
        is_evening,
        is_weekend,
        # Meal type (one-hot)
        is_breakfast,
        is_lunch,
        is_dinner
    ]

def bin_sequence(sequence, n_bins):
    """Average sequence into n_bins of equal size"""
    bin_size = len(sequence) // n_bins
    binned = []
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size
        bin_data = sequence[start_idx:end_idx].mean(axis=0)
        binned.append(bin_data)
    return np.array(binned)

def extract_meal_windows(df, pid, cgm_col):
    """Extract meal windows with 60 min before meal and targets at 30, 60, 120 min after"""
    output_X_raw, output_X_binned = [], []
    output_y, output_static, output_pid, output_diagnosis = [], [], [], []
    required_cols = [cgm_col, "HR", "Calories (Activity)", "METs"]

    # Get diagnosis for this participant
    diagnosis_label = diagnosis_map.get(pid, -1)

    for _, row in df.iterrows():
        if pd.isnull(row["Meal Type"]) or row["Meal Type"] not in ["breakfast", "lunch", "dinner"]:
            continue

        meal_time = row["timestamp"]

        # Extract 60 minutes BEFORE meal
        start_before = meal_time - pd.Timedelta(minutes=WINDOW_SIZE)
        end_before = meal_time
        segment_before = df[(df["timestamp"] >= start_before) & (df["timestamp"] < end_before)]

        # Check if we have enough data before meal and no missing values
        if len(segment_before) < WINDOW_SIZE or segment_before[required_cols].isnull().any().any():
            continue

        # Extract target glucose values at 30, 60, 120 minutes AFTER meal
        targets = []
        valid_targets = True
        for horizon in TARGET_HORIZONS:
            target_time = meal_time + pd.Timedelta(minutes=horizon)
            # Find closest glucose reading within ±2 minutes
            target_window = df[
                (df["timestamp"] >= target_time - pd.Timedelta(minutes=2)) &
                (df["timestamp"] <= target_time + pd.Timedelta(minutes=2))
            ]
            # keep only non-NaN CGM rows
            tw_nonan = target_window[target_window[cgm_col].notna()]
            if tw_nonan.empty:
                valid_targets = False
                break

            closest_idx = (tw_nonan["timestamp"] - target_time).abs().idxmin()
            targets.append(tw_nonan.loc[closest_idx, cgm_col])

        if not valid_targets:
            continue

        # Create input sequences
        x_seq_raw = segment_before[required_cols].values
        x_seq_binned = bin_sequence(x_seq_raw, N_BINS)

        # Get static features
        s_feat = get_static_features(pid, row)

        output_X_raw.append(x_seq_raw)
        output_X_binned.append(x_seq_binned)
        output_y.append(targets)
        output_static.append(s_feat)
        output_pid.append(pid)
        output_diagnosis.append(diagnosis_label)

    return output_X_raw, output_X_binned, output_y, output_static, output_pid, output_diagnosis

"""## Process All Participants"""

# Process all CGM sources
for cgm_col in CGM_SOURCES:
    cgm_type = "Libre" if cgm_col == "Libre GL" else "Dexcom"
    print(f"\n{'='*60}")
    print(f"Processing {cgm_type} data...")
    print(f"{'='*60}")

    all_X_raw, all_X_binned = [], []
    all_y, all_static, all_pid, all_diagnosis = [], [], [], []

    for folder in tqdm(glob(os.path.join(BASE_DIR, "CGMacros-0*/"))):
        folder_name = os.path.basename(folder.rstrip("/"))
        df = load_participant_file(folder)

        X_raw, X_binned, y, static, pids, diagnosis = extract_meal_windows(df, folder_name, cgm_col)

        all_X_raw.extend(X_raw)
        all_X_binned.extend(X_binned)
        all_y.extend(y)
        all_static.extend(static)
        all_pid.extend(pids)
        all_diagnosis.extend(diagnosis)

    # Save raw version (60 timesteps)
    np.savez_compressed(
        os.path.join(SAVE_DIR, f"{cgm_type.lower()}_raw_prediction.npz"),
        X=np.array(all_X_raw),
        static=np.array(all_static),
        y=np.array(all_y),
        participant_id=np.array(all_pid),
        diagnosis=np.array(all_diagnosis)
    )
    print(f"\nSaved: {cgm_type}_raw — {len(all_X_raw)} samples")
    print(f"   X shape: {np.array(all_X_raw).shape}")
    print(f"   static shape: {np.array(all_static).shape}")
    print(f"   y shape: {np.array(all_y).shape}")
    print(f"   diagnosis distribution: {np.unique(all_diagnosis, return_counts=True)}")

    # Save binned version (12 timesteps)
    np.savez_compressed(
        os.path.join(SAVE_DIR, f"{cgm_type.lower()}_binned_prediction.npz"),
        X=np.array(all_X_binned),
        static=np.array(all_static),
        y=np.array(all_y),
        participant_id=np.array(all_pid),
        diagnosis=np.array(all_diagnosis)
    )
    print(f"\nSaved: {cgm_type}_binned — {len(all_X_binned)} samples")
    print(f"   X shape: {np.array(all_X_binned).shape}")
    print(f"   static shape: {np.array(all_static).shape}")
    print(f"   y shape: {np.array(all_y).shape}")
    print(f"   diagnosis distribution: {np.unique(all_diagnosis, return_counts=True)}")

"""## Summary"""

print("\n" + "="*60)
print("Output files (4 total):")
print("="*60)
print("\nEach .npz file contains:")
print("- X: (n_samples, seq_len, 4) — time series of CGM, HR, Calories, METs")
print("  * raw version: seq_len = 60 (1 minute resolution)")
print("  * binned version: seq_len = 12 (5 minute bins)")
print("\n- static: (n_samples, 17) — static features per meal:")
print("  Demographic: [Age, Gender (0/1), BMI, HbA1c]")
print("  Macronutrients: [Calories, Carbs, Protein, Fat, Fiber]")
print("  Time of meal: [hour_sin, hour_cos, is_morning, is_evening, is_weekend, is_breakfast, is_lunch, is_dinner]")
print("\n- y: (n_samples, 3) — glucose levels at [30min, 60min, 120min] after meal")
print("\n- participant_id: for grouped CV")
print("\n- diagnosis: (n_samples,) — diagnosis category for stratified analysis")
print("  * 0 = Healthy (HbA1c < 5.7)")
print("  * 1 = Pre-diabetes (5.7 ≤ HbA1c ≤ 6.4)")
print("  * 2 = Type 2 Diabetes (HbA1c > 6.4)")
print("  * -1 = Unknown")
print("\nFiles saved to:", SAVE_DIR)

"""## Verification"""

# Load and inspect one file to verify
sample_file = os.path.join(SAVE_DIR, "libre_raw_prediction.npz")
if os.path.exists(sample_file):
    data = np.load(sample_file)
    print("\nSample file inspection:")
    print(f"X shape: {data['X'].shape}")
    print(f"static shape: {data['static'].shape}")
    print(f"y shape: {data['y'].shape}")
    print(f"participant_id shape: {data['participant_id'].shape}")
    print(f"diagnosis shape: {data['diagnosis'].shape}")
    print(f"\nSample X (first 3 timesteps):\n{data['X'][0][:3]}")
    print(f"\nSample static features (17 features):")
    static_names = ['Age', 'Gender', 'BMI', 'HbA1c',
                    'Calories', 'Carbs', 'Protein', 'Fat', 'Fiber',
                    'hour_sin', 'hour_cos', 'is_morning', 'is_evening', 'is_weekend',
                    'is_breakfast', 'is_lunch', 'is_dinner']
    for i, (name, val) in enumerate(zip(static_names, data['static'][0])):
        print(f"  {i}. {name}: {val}")
    print(f"\nSample y (targets at 30, 60, 120 min):\n{data['y'][0]}")
    print(f"\nSample participant_id: {data['participant_id'][0]}")
    print(f"Sample diagnosis: {data['diagnosis'][0]} (0=Healthy, 1=Pre-diabetes, 2=T2D, -1=Unknown)")

"""## Normalization"""

# Normalization Cell - Run after creating the raw .npz files

# Feature indices in static array (17 features total)
INDICES = {
    'age': 0, 'gender': 1, 'bmi': 2, 'hba1c': 3,
    'calories': 4, 'carbs': 5, 'protein': 6, 'fat': 7, 'fiber': 8,
    'hour_sin': 9, 'hour_cos': 10,
    'is_morning': 11, 'is_evening': 12, 'is_weekend': 13,
    'is_breakfast': 14, 'is_lunch': 15, 'is_dinner': 16
}

def normalize_data(data_file):
    """Normalize data following the suggested approach"""
    print(f"\nProcessing: {os.path.basename(data_file)}")

    # Load data
    data = np.load(data_file)
    X = data['X'].copy()  # (n_samples, seq_len, 4) - [CGM, HR, Calories, METs]
    static = data['static'].copy()  # (n_samples, 17)
    y = data['y'].copy()  # (n_samples, 3) - keep unchanged (mg/dL)
    participant_ids = data['participant_id']
    diagnosis = data['diagnosis']

    print(f"  Samples: {X.shape[0]}")

    # Initialize normalized arrays
    X_norm = X.copy()
    static_norm = static.copy()

    # Storage for normalization parameters
    norm_params = {'per_subject': {}, 'global': {}}

    # ========================================
    # 1. TIME SERIES (X) - Per-subject normalization
    # ========================================
    print("  Normalizing time series per subject...")
    unique_subjects = np.unique(participant_ids)

    for subject in unique_subjects:
        subject_mask = participant_ids == subject
        subject_X = X[subject_mask]
        subject_X_flat = subject_X.reshape(-1, 4)

        # CGM (index 0): z-score per subject
        cgm_mean = np.nanmean(subject_X_flat[:, 0])
        cgm_std = np.nanstd(subject_X_flat[:, 0])
        if cgm_std > 0:
            X_norm[subject_mask, :, 0] = (X[subject_mask, :, 0] - cgm_mean) / cgm_std

        # HR (index 1): z-score per subject
        hr_mean = np.nanmean(subject_X_flat[:, 1])
        hr_std = np.nanstd(subject_X_flat[:, 1])
        if hr_std > 0:
            X_norm[subject_mask, :, 1] = (X[subject_mask, :, 1] - hr_mean) / hr_std

        # Calories/Activity (index 2): min-max per subject
        cal_min = np.nanmin(subject_X_flat[:, 2])
        cal_max = np.nanmax(subject_X_flat[:, 2])
        if cal_max > cal_min:
            X_norm[subject_mask, :, 2] = (X[subject_mask, :, 2] - cal_min) / (cal_max - cal_min)

        # METs (index 3): min-max per subject
        mets_min = np.nanmin(subject_X_flat[:, 3])
        mets_max = np.nanmax(subject_X_flat[:, 3])
        if mets_max > mets_min:
            X_norm[subject_mask, :, 3] = (X[subject_mask, :, 3] - mets_min) / (mets_max - mets_min)

        # Store per-subject parameters
        norm_params['per_subject'][subject] = {
            'cgm_mean': cgm_mean, 'cgm_std': cgm_std,
            'hr_mean': hr_mean, 'hr_std': hr_std,
            'cal_min': cal_min, 'cal_max': cal_max,
            'mets_min': mets_min, 'mets_max': mets_max
        }

    # ========================================
    # 2. STATIC FEATURES - Global normalization
    # ========================================
    print("  Normalizing static features globally...")

    # Age: global z-score
    age_mean = np.nanmean(static[:, INDICES['age']])
    age_std = np.nanstd(static[:, INDICES['age']])
    if age_std > 0:
        static_norm[:, INDICES['age']] = (static[:, INDICES['age']] - age_mean) / age_std
    norm_params['global']['age'] = {'mean': float(age_mean), 'std': float(age_std)}

    # BMI: global z-score
    bmi_mean = np.nanmean(static[:, INDICES['bmi']])
    bmi_std = np.nanstd(static[:, INDICES['bmi']])
    if bmi_std > 0:
        static_norm[:, INDICES['bmi']] = (static[:, INDICES['bmi']] - bmi_mean) / bmi_std
    norm_params['global']['bmi'] = {'mean': float(bmi_mean), 'std': float(bmi_std)}

    # HbA1c: global z-score
    hba1c_mean = np.nanmean(static[:, INDICES['hba1c']])
    hba1c_std = np.nanstd(static[:, INDICES['hba1c']])
    if hba1c_std > 0:
        static_norm[:, INDICES['hba1c']] = (static[:, INDICES['hba1c']] - hba1c_mean) / hba1c_std
    norm_params['global']['hba1c'] = {'mean': float(hba1c_mean), 'std': float(hba1c_std)}

    # Macronutrients: global z-score (with optional log transform)
    macro_features = ['calories', 'carbs', 'protein', 'fat', 'fiber']
    for feat in macro_features:
        idx = INDICES[feat]
        values = static[:, idx]
        values_clean = values[~np.isnan(values)]

        if len(values_clean) > 0:
            skewness = skew(values_clean)

            if abs(skewness) > 1.0:  # Highly skewed
                print(f"    {feat} is skewed ({skewness:.2f}), applying log(x+1) transform")
                values_transformed = np.log1p(values)
                mean_val = np.nanmean(values_transformed)
                std_val = np.nanstd(values_transformed)
                if std_val > 0:
                    static_norm[:, idx] = (values_transformed - mean_val) / std_val
                norm_params['global'][feat] = {
                    'mean': float(mean_val), 'std': float(std_val), 'log_transformed': True
                }
            else:
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val > 0:
                    static_norm[:, idx] = (values - mean_val) / std_val
                norm_params['global'][feat] = {
                    'mean': float(mean_val), 'std': float(std_val), 'log_transformed': False
                }

    print(f"  Normalization complete!")
    print(f"     X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print(f"     Static range: [{static_norm.min():.3f}, {static_norm.max():.3f}]")
    print(f"     y unchanged: [{y.min():.1f}, {y.max():.1f}] mg/dL")

    return X_norm, static_norm, y, participant_ids, diagnosis, norm_params

# ========================================
# Process all files
# ========================================
OUTPUT_DIR_NORM = os.path.join(SAVE_DIR, "..", "Prediction_Normalized")
os.makedirs(OUTPUT_DIR_NORM, exist_ok=True)

all_norm_params = {}
CGM_TYPES = ["libre", "dexcom"]
VERSIONS = ["raw", "binned"]

for cgm_type in CGM_TYPES:
    for version in VERSIONS:
        filename = f"{cgm_type}_{version}_prediction.npz"
        input_file = os.path.join(SAVE_DIR, filename)

        if not os.path.exists(input_file):
            print(f"  File not found: {filename}")
            continue

        # Normalize
        X_norm, static_norm, y, pids, diagnosis, norm_params = normalize_data(input_file)

        # Save normalized data
        output_file = os.path.join(OUTPUT_DIR_NORM, filename)
        np.savez_compressed(
            output_file,
            X=X_norm,
            static=static_norm,
            y=y,
            participant_id=pids,
            diagnosis=diagnosis
        )
        print(f"  Saved: {output_file}\n")

        # Store normalization parameters
        all_norm_params[f"{cgm_type}_{version}"] = norm_params

# Save normalization parameters
params_file = os.path.join(OUTPUT_DIR_NORM, "normalization_params.pkl")
with open(params_file, 'wb') as f:
    pickle.dump(all_norm_params, f)
print(f"Normalization parameters saved to: {params_file}")

print("\n" + "="*60)
print("NORMALIZATION COMPLETE!")
print("="*60)
print(f"\nNormalized files saved to: {OUTPUT_DIR_NORM}")
print("\nFiles created:")
for cgm_type in CGM_TYPES:
    for version in VERSIONS:
        print(f"  - {cgm_type}_{version}_prediction.npz")
print("  - normalization_params.pkl")
print("\nTarget glucose (y) kept in mg/dL for evaluation.")