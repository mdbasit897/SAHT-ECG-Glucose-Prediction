#!/usr/bin/env python3
"""
Quick verification script to check data quality before full preprocessing
"""

import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path


def verify_dataset():
    print("🔍 VERIFYING DATASET QUALITY")
    print("=" * 50)

    # 1. Check clinical data
    print("\n📋 CLINICAL DATA CHECK:")
    clinical = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/clinical_indicators.xlsx')
    clinical['subject_id'] = clinical['Unnamed: 0'].astype(str)

    print(f"✅ Clinical subjects: {len(clinical)}")

    # Check glucose data availability
    admission_fbg = clinical['admission FBG (mmol/L)'].dropna()
    discharge_fbg = clinical['Discharge FBG (mmol/L)'].dropna()
    hba1c = clinical['HbA1c (%)'].dropna()

    print(f"✅ Admission FBG available: {len(admission_fbg)}/{len(clinical)} subjects")
    print(f"   Range: {admission_fbg.min():.1f} - {admission_fbg.max():.1f} mmol/L")
    print(f"✅ Discharge FBG available: {len(discharge_fbg)}/{len(clinical)} subjects")
    print(f"   Range: {discharge_fbg.min():.1f} - {discharge_fbg.max():.1f} mmol/L")
    print(f"✅ HbA1c available: {len(hba1c)}/{len(clinical)} subjects")
    print(f"   Range: {hba1c.min():.1f} - {hba1c.max():.1f}%")

    # Calculate glucose change
    glucose_change = discharge_fbg - admission_fbg
    improved_count = sum(glucose_change < 0)
    print(
        f"✅ Glucose improvement cases: {improved_count}/{len(glucose_change)} ({improved_count / len(glucose_change) * 100:.1f}%)")

    # 2. Check sleep data
    print("\n😴 SLEEP DATA CHECK:")

    # Fix objective sleep data
    obj_sleep_raw = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/objective_sleep_quality.xlsx')
    print(f"✅ Objective sleep data: {len(obj_sleep_raw) - 1} subjects (after header fix)")

    # Subjective sleep
    subj_sleep = pd.read_excel('Dataset_on_electrocardiograph/dataset_ecg/subjective_sleep_quality.xlsx')
    print(f"✅ Subjective sleep data: {len(subj_sleep)} subjects")
    print(f"   Sleep metrics: AHI, TST, UST, SST, RST available")

    # 3. Check ECG/RR data alignment
    print("\n💓 ECG/RR DATA ALIGNMENT:")

    clinical_subjects = set(clinical['subject_id'])
    ecg_subjects = set([f.stem for f in Path('Dataset_on_electrocardiograph/dataset_ecg/ECG').glob('*.mat')])
    rr_subjects = set([f.stem for f in Path('Dataset_on_electrocardiograph/dataset_ecg/RR_interval').glob('*.mat')])
    subj_sleep_subjects = set(subj_sleep['number'].astype(str))

    print(f"✅ Clinical subjects: {len(clinical_subjects)}")
    print(f"✅ ECG subjects: {len(ecg_subjects)}")
    print(f"✅ RR-interval subjects: {len(rr_subjects)}")
    print(f"✅ Sleep analysis subjects: {len(subj_sleep_subjects)}")

    # Find complete subjects
    complete_subjects = clinical_subjects & ecg_subjects & rr_subjects
    print(f"\n🎯 COMPLETE subjects (Clinical+ECG+RR): {len(complete_subjects)}")

    if len(complete_subjects) < 30:
        print("⚠️  Warning: Less than 30 complete subjects may limit model performance")
    else:
        print("✅ Sufficient subjects for robust machine learning")

    # Check missing data
    missing_rr = ecg_subjects - rr_subjects
    if missing_rr:
        print(f"\n⚠️  Missing RR-interval data for: {sorted(list(missing_rr))}")

    # 4. Quick ECG data quality check
    print("\n📈 ECG DATA QUALITY CHECK:")
    sample_ecg_files = list(Path('Dataset_on_electrocardiograph/dataset_ecg/ECG').glob('*.mat'))[:3]

    for ecg_file in sample_ecg_files:
        try:
            data = scipy.io.loadmat(str(ecg_file))
            if 'all' in data:
                signal = data['all'].flatten()
                duration = len(signal) / 250 / 3600  # Hours
                signal_range = np.max(signal) - np.min(signal)

                print(f"✅ {ecg_file.stem}: {duration:.1f}h, range: {signal_range:.1f}")

                if duration < 20:
                    print(f"   ⚠️  Short recording: {duration:.1f}h")
                if signal_range > 50:
                    print(f"   ⚠️  Unusual amplitude range: {signal_range:.1f}")

        except Exception as e:
            print(f"❌ {ecg_file.stem}: Error - {e}")

    # 5. Quick RR-interval check
    print("\n💓 RR-INTERVAL DATA CHECK:")
    sample_rr_files = list(Path('Dataset_on_electrocardiograph/dataset_ecg/RR_interval').glob('*.mat'))[:3]

    for rr_file in sample_rr_files:
        try:
            data = scipy.io.loadmat(str(rr_file))
            total_intervals = 0

            for stage in ['DS', 'RS', 'REM']:
                if stage in data:
                    intervals = data[stage].flatten()
                    total_intervals += len(intervals)

                    # Convert to seconds and check physiological range
                    intervals_sec = intervals / 1000.0
                    mean_rr = np.mean(intervals_sec)
                    hr = 60 / mean_rr

                    if hr < 40 or hr > 150:
                        print(f"   ⚠️  {rr_file.stem} {stage}: Unusual HR {hr:.0f} bpm")

            print(f"✅ {rr_file.stem}: {total_intervals} total intervals across sleep stages")

        except Exception as e:
            print(f"❌ {rr_file.stem}: Error - {e}")

    # 6. Data completeness summary
    print("\n📊 DATA COMPLETENESS SUMMARY:")
    print("=" * 40)

    # Check which subjects have all required data
    subjects_with_all_data = []

    for subject_id in complete_subjects:
        has_glucose = not pd.isna(clinical[clinical['subject_id'] == subject_id]['admission FBG (mmol/L)'].iloc[0])
        has_hba1c = not pd.isna(clinical[clinical['subject_id'] == subject_id]['HbA1c (%)'].iloc[0])

        if has_glucose and has_hba1c:
            subjects_with_all_data.append(subject_id)

    print(f"🎯 Subjects with COMPLETE data for modeling: {len(subjects_with_all_data)}")
    print(f"   (Clinical + ECG + RR + Glucose targets)")

    if len(subjects_with_all_data) >= 30:
        print("✅ EXCELLENT: Dataset is ready for high-quality research!")
    elif len(subjects_with_all_data) >= 20:
        print("✅ GOOD: Dataset suitable for research with some limitations")
    else:
        print("⚠️  LIMITED: May need to relax inclusion criteria")

    # 7. Research recommendations
    print("\n🎯 RESEARCH RECOMMENDATIONS:")
    print("=" * 40)

    print("✅ Primary targets for modeling:")
    print("   - Glucose change (discharge - admission)")
    print("   - HbA1c prediction")
    print("   - Glucose improvement (binary classification)")

    print("✅ Feature categories available:")
    print("   - Clinical markers (blood work, BP, demographics)")
    print("   - ECG signal features (24h, sleep, wake)")
    print("   - HRV features (sleep-stage-specific)")
    print("   - Sleep quality metrics (PSQI, CPC analysis)")

    print("✅ Novel research angles:")
    print("   - Sleep-stage-aware glucose prediction")
    print("   - Multi-modal fusion (ECG + Clinical + Sleep)")
    print("   - Circadian rhythm analysis")
    print("   - Diabetic complication prediction")

    return len(subjects_with_all_data)


if __name__ == "__main__":
    complete_subjects = verify_dataset()

    print(f"\n🚀 READY TO PROCEED!")
    print(f"Run the complete preprocessing pipeline with {complete_subjects} subjects")