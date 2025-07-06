#!/bin/bash

# Immediate Data Exploration Script
# Run this in your dataset directory

echo
"=============================================="
echo
"DIABETES ECG DATASET DETAILED EXPLORATION"
echo
"=============================================="

# 1. Check current directory
echo
"📍 Current location: $(pwd)"
echo
"📊 Total files: $(find . -type f | wc -l)"
echo
""

# 2. Analyze Excel files content
echo
"📋 ANALYZING EXCEL FILES..."
echo
"----------------------------------------------"

python3 << 'EOF'
import pandas as pd
import numpy as np

print("=== CLINICAL INDICATORS ===")
try:
    clinical = pd.read_excel('Clinical indicators.xlsx')
    print(f"Shape: {clinical.shape[0]} subjects × {clinical.shape[1]} indicators")
    print(f"Columns: {list(clinical.columns)}")
    print("\nFirst few rows:")
    print(clinical.head())

    # Key diabetes markers
    diabetes_cols = [col for col in clinical.columns if any(keyword in col.lower()
                                                            for keyword in ['fbg', 'hba1c', 'glucose', 'diabetes'])]
    if diabetes_cols:
        print(f"\n🩺 Diabetes markers found: {diabetes_cols}")
        for col in diabetes_cols:
            if clinical[col].dtype in ['float64', 'int64']:
                values = clinical[col].dropna()
                print(f"  {col}: mean={values.mean():.2f}, range=[{values.min():.2f}, {values.max():.2f}]")

    # Missing data analysis
    missing = clinical.isnull().sum()
    print(f"\n📊 Missing data per column:")
    for col, miss_count in missing.items():
        if miss_count > 0:
            print(f"  {col}: {miss_count}/{len(clinical)} missing ({miss_count / len(clinical) * 100:.1f}%)")

except Exception as e:
    print(f"❌ Error reading Clinical indicators: {e}")

print("\n" + "=" * 60 + "\n")

print("=== OBJECTIVE SLEEP QUALITY ===")
try:
    obj_sleep = pd.read_excel('Objective sleep quality.xlsx')
    print(f"Shape: {obj_sleep.shape[0]} subjects × {obj_sleep.shape[1]} metrics")
    print(f"Columns: {list(obj_sleep.columns)}")
    print("\nFirst few rows:")
    print(obj_sleep.head())

    # Sleep metrics summary
    numeric_cols = obj_sleep.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n😴 Sleep metrics summary:")
        for col in numeric_cols:
            values = obj_sleep[col].dropna()
            if len(values) > 0:
                print(f"  {col}: mean={values.mean():.2f}, range=[{values.min():.2f}, {values.max():.2f}]")

except Exception as e:
    print(f"❌ Error reading Objective sleep quality: {e}")

print("\n" + "=" * 60 + "\n")

print("=== SUBJECTIVE SLEEP QUALITY ===")
try:
    subj_sleep = pd.read_excel('Subjective sleep quality.xlsx')
    print(f"Shape: {subj_sleep.shape[0]} subjects × {subj_sleep.shape[1]} metrics")
    print(f"Columns: {list(subj_sleep.columns)}")
    print("\nFirst few rows:")
    print(subj_sleep.head())

    # PSQI components if available
    psqi_cols = [col for col in subj_sleep.columns if 'psqi' in col.lower()]
    if psqi_cols:
        print(f"\n🛏️ PSQI components found: {psqi_cols}")

except Exception as e:
    print(f"❌ Error reading Subjective sleep quality: {e}")

EOF

echo
""
echo
"🔬 ANALYZING MATLAB FILES..."
echo
"----------------------------------------------"

# 3. ECG file analysis
echo
"📈 ECG Files Analysis:"
echo
"Found $(ls ECG/*.mat | wc -l) ECG files"

python3 << 'EOF'
import scipy.io
import numpy as np
import os

# Analyze first 3 ECG files
ecg_files = sorted([f for f in os.listdir('ECG') if f.endswith('.mat')])[:3]

for i, file in enumerate(ecg_files):
    print(f"\n📊 ECG File {i + 1}: {file}")
    try:
        data = scipy.io.loadmat(f'Dataset_on_electrocardiograph/dataset_ecg/ECG/{file}')

        # Show variables
        variables = [k for k in data.keys() if not k.startswith('__')]
        print(f"   Variables: {variables}")

        for var in variables:
            arr = data[var]
            if isinstance(arr, np.ndarray):
                print(f"   {var}: shape {arr.shape}, dtype {arr.dtype}")

                if arr.size > 0:
                    # Calculate duration assuming 250 Hz
                    if len(arr.shape) <= 2:
                        total_samples = arr.size
                        duration_hours = total_samples / 250 / 3600
                        duration_minutes = total_samples / 250 / 60
                        print(f"     Duration: {duration_hours:.1f}h ({duration_minutes:.0f}min)")
                        print(f"     Samples: {total_samples:,}")
                        print(f"     Value range: [{np.min(arr):.3f}, {np.max(arr):.3f}]")

                        # Check for reasonable ECG values
                        if np.max(arr) > 10 or np.min(arr) < -10:
                            print(f"     ⚠️ Unusual ECG amplitude range")

    except Exception as e:
        print(f"   ❌ Error: {e}")

EOF

echo
""
echo
"💓 RR-interval Files Analysis:"
echo
"Found $(ls RR-interval/*.mat | wc -l) RR-interval files"

python3 << 'EOF'
import scipy.io
import numpy as np
import os

# Analyze first 3 RR-interval files
rr_files = sorted([f for f in os.listdir('RR-interval') if f.endswith('.mat')])[:3]

for i, file in enumerate(rr_files):
    print(f"\n💓 RR-interval File {i + 1}: {file}")
    try:
        data = scipy.io.loadmat(f'Dataset_on_electrocardiograph/dataset_ecg/RR_interval/{file}')

        # Show variables
        variables = [k for k in data.keys() if not k.startswith('__')]
        print(f"   Variables: {variables}")

        for var in variables:
            arr = data[var]
            if isinstance(arr, np.ndarray) and arr.size > 0:
                arr_flat = arr.flatten()
                print(f"   {var}: {len(arr_flat)} intervals")

                if len(arr_flat) > 0:
                    mean_rr = np.mean(arr_flat)
                    hr_estimate = 60 / mean_rr if mean_rr > 0 else 0
                    print(f"     Mean RR: {mean_rr:.3f}s")
                    print(f"     HR estimate: {hr_estimate:.1f} bpm")
                    print(f"     Range: [{np.min(arr_flat):.3f}, {np.max(arr_flat):.3f}]s")
                    print(f"     Total duration: ~{len(arr_flat) * mean_rr / 3600:.1f}h")

                    # Sleep stage analysis (based on variable names)
                    if var.upper() in ['DS', 'UNSTABLE']:
                        print(f"     🌙 Unstable sleep intervals")
                    elif var.upper() in ['RS', 'STABLE']:
                        print(f"     😴 Stable sleep intervals")
                    elif var.upper() == 'REM':
                        print(f"     🧠 REM sleep intervals")

    except Exception as e:
        print(f"   ❌ Error: {e}")

EOF

echo
""
echo
"🔍 DATA COMPLETENESS CHECK..."
echo
"----------------------------------------------"

# 4. Check data completeness
python3 << 'EOF'
import os
import re

# Get all ECG files
ecg_files = set([f.replace('.mat', '') for f in os.listdir('ECG') if f.endswith('.mat')])
rr_files = set([f.replace('.mat', '') for f in os.listdir('RR-interval') if f.endswith('.mat')])

print(f"📊 Data Completeness:")
print(f"   ECG files: {len(ecg_files)}")
print(f"   RR-interval files: {len(rr_files)}")

# Find subjects with missing RR-interval data
missing_rr = ecg_files - rr_files
if missing_rr:
    print(f"\n⚠️ Subjects with ECG but NO RR-interval data:")
    for subject in sorted(missing_rr):
        print(f"   - {subject}")

# Find subjects with RR-interval but no ECG (shouldn't happen)
missing_ecg = rr_files - ecg_files
if missing_ecg:
    print(f"\n⚠️ Subjects with RR-interval but NO ECG data:")
    for subject in sorted(missing_ecg):
        print(f"   - {subject}")

# Complete data subjects
complete_subjects = ecg_files.intersection(rr_files)
print(f"\n✅ Subjects with COMPLETE data (ECG + RR): {len(complete_subjects)}")

print(f"\n📋 Subject ID patterns:")
sample_ids = sorted(list(ecg_files))[:10]
for sid in sample_ids:
    print(f"   {sid}")
if len(ecg_files) > 10:
    print(f"   ... and {len(ecg_files) - 10} more")

EOF

echo
""
echo
"📋 DATASET SUMMARY FOR RESEARCH..."
echo
"----------------------------------------------"

python3 << 'EOF'
import pandas as pd
import os

print("🎯 RESEARCH READINESS ASSESSMENT:")
print()

# Count complete data
ecg_count = len([f for f in os.listdir('ECG') if f.endswith('.mat')])
rr_count = len([f for f in os.listdir('RR-interval') if f.endswith('.mat')])

print(f"✅ Available ECG recordings: {ecg_count}")
print(f"✅ Available RR-interval files: {rr_count}")
print(f"✅ Complete ECG+RR pairs: {min(ecg_count, rr_count)}")

# Check clinical data availability
try:
    clinical = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/clinical_indicators.xlsx')
    print(f"✅ Clinical data: {clinical.shape[0]} subjects, {clinical.shape[1]} indicators")

    # Check for key diabetes markers
    has_fbg = any('fbg' in col.lower() for col in clinical.columns)
    has_hba1c = any('hba1c' in col.lower() for col in clinical.columns)

    print(f"✅ Glucose data (FBG): {'Yes' if has_fbg else 'No'}")
    print(f"✅ HbA1c data: {'Yes' if has_hba1c else 'No'}")

except Exception as e:
    print(f"❌ Clinical data issue: {e}")

# Check sleep data
try:
    obj_sleep = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/objective_sleep_quality.xlsx')
    subj_sleep = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/subjective_sleep_quality.xlsx')
    print(f"✅ Objective sleep data: {obj_sleep.shape[0]} subjects")
    print(f"✅ Subjective sleep data: {subj_sleep.shape[0]} subjects")
except Exception as e:
    print(f"⚠️ Sleep data issue: {e}")

print()
print("🚀 RECOMMENDED NEXT STEPS:")
print("1. Create data preprocessing pipeline")
print("2. Handle missing RR-interval files (4 subjects)")
print("3. Design train/validation/test splits")
print("4. Implement ECG feature extraction")
print("5. Start with baseline models")
print()
print("📖 Dataset appears suitable for your research!")

EOF

echo
""
echo
"✅ EXPLORATION COMPLETE!"
echo
"=============================================="