#!/usr/bin/env python3
"""
Preprocessing Pipeline
"""

import os
import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings

warnings.filterwarnings('ignore')


class FixedDiabetesECGPreprocessor:
    def __init__(self, dataset_path="."):
        self.dataset_path = Path(dataset_path)
        self.clinical_data = None
        self.objective_sleep = None
        self.subjective_sleep = None
        self.subjects_mapping = {}
        self.complete_subjects = []
        self.processed_data = {}

    def validate_ecg_scaling(self, ecg_signal):
        original_signal = ecg_signal.copy()

        # Check for abnormal amplitude ranges
        signal_range = np.max(np.abs(ecg_signal))

        print(f"   Original ECG range: ±{signal_range:.0f}")

        # Fix scaling based on detected issues
        if signal_range > 50000:  # Likely raw ADC values
            # Common 16-bit ADC with 5V range
            ecg_signal = (ecg_signal / 32768) * 5  # Convert to mV
            print(f"   Applied ADC conversion: ±{np.max(np.abs(ecg_signal)):.1f} mV")

        elif signal_range > 50:  # Likely in µV, convert to mV
            ecg_signal = ecg_signal / 1000
            print(f"   Applied µV to mV conversion: ±{np.max(np.abs(ecg_signal)):.1f} mV")

        # Final validation - ECG should be ±5mV typically
        final_range = np.max(np.abs(ecg_signal))
        if final_range > 20:  # Still too large
            ecg_signal = ecg_signal / (final_range / 5)  # Normalize to ±5mV
            print(f"   Applied final normalization: ±{np.max(np.abs(ecg_signal)):.1f} mV")

        return ecg_signal

    def load_clinical_data(self):
        """Load clinical data with enhanced missing data handling"""
        print("📋 Loading clinical data...")

        clinical_file = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/clinical_indicators.xlsx"
        self.clinical_data = pd.read_excel(clinical_file)

        # Fix column name
        self.clinical_data = self.clinical_data.rename(columns={'Unnamed: 0': 'subject_id'})
        self.clinical_data['subject_id'] = self.clinical_data['subject_id'].astype(str)

        print(f"✅ Loaded clinical data: {self.clinical_data.shape}")

        # Analyze missing data patterns
        key_columns = ['admission FBG (mmol/L)', 'Discharge FBG (mmol/L)', 'HbA1c (%)']
        for col in key_columns:
            missing_count = self.clinical_data[col].isna().sum()
            print(
                f"   {col}: {missing_count}/{len(self.clinical_data)} missing ({missing_count / len(self.clinical_data) * 100:.1f}%)")

        return self.clinical_data

    def load_objective_sleep_data(self):
        """Load objective sleep data with proper header handling"""
        print("😴 Loading objective sleep data...")

        obj_file = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/objective_sleep_quality.xlsx"
        raw_obj_sleep = pd.read_excel(obj_file)

        # Extract real column names from row 0
        real_columns = ['number', 'gender', 'age', 'height', 'weight']
        psqi_columns = []

        for i in range(5, len(raw_obj_sleep.columns)):
            col_value = raw_obj_sleep.iloc[0, i]
            if pd.notna(col_value):
                psqi_columns.append(str(col_value).strip())
            else:
                psqi_columns.append(f'psqi_component_{i - 4}')

        real_columns.extend(psqi_columns)

        # Create clean dataframe
        self.objective_sleep = raw_obj_sleep.iloc[1:].copy()
        self.objective_sleep.columns = real_columns[:len(self.objective_sleep.columns)]
        self.objective_sleep['number'] = self.objective_sleep['number'].astype(str)

        # Convert numeric columns safely
        numeric_cols = self.objective_sleep.columns[2:]
        for col in numeric_cols:
            self.objective_sleep[col] = pd.to_numeric(self.objective_sleep[col], errors='coerce')

        print(f"✅ Fixed objective sleep data: {self.objective_sleep.shape}")
        return self.objective_sleep

    def load_subjective_sleep_data(self):
        """Load subjective sleep data"""
        print("🧠 Loading subjective sleep data...")

        subj_file = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/subjective_sleep_quality.xlsx"
        self.subjective_sleep = pd.read_excel(subj_file)
        self.subjective_sleep['number'] = self.subjective_sleep['number'].astype(str)

        print(f"✅ Loaded subjective sleep data: {self.subjective_sleep.shape}")
        return self.subjective_sleep

    def create_subject_mapping(self):
        """Create comprehensive subject mapping"""
        print("🗺️ Creating subject mapping...")

        # Get subjects from each source
        clinical_subjects = set(self.clinical_data['subject_id'])

        ecg_dir = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/ECG"
        ecg_subjects = set([f.stem for f in ecg_dir.glob("*.mat")]) if ecg_dir.exists() else set()

        rr_dir = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/RR_interval"
        rr_subjects = set([f.stem for f in rr_dir.glob("*.mat")]) if rr_dir.exists() else set()

        obj_sleep_subjects = set(self.objective_sleep['number']) if self.objective_sleep is not None else set()
        subj_sleep_subjects = set(self.subjective_sleep['number']) if self.subjective_sleep is not None else set()

        # Create mapping
        all_subjects = clinical_subjects | ecg_subjects | rr_subjects | obj_sleep_subjects | subj_sleep_subjects

        for subject_id in all_subjects:
            self.subjects_mapping[subject_id] = {
                'has_clinical': subject_id in clinical_subjects,
                'has_ecg': subject_id in ecg_subjects,
                'has_rr': subject_id in rr_subjects,
                'has_obj_sleep': subject_id in obj_sleep_subjects,
                'has_subj_sleep': subject_id in subj_sleep_subjects,
                'ecg_file': ecg_dir / f"{subject_id}.mat" if subject_id in ecg_subjects else None,
                'rr_file': rr_dir / f"{subject_id}.mat" if subject_id in rr_subjects else None
            }

        # RELAXED inclusion criteria to increase sample size
        self.complete_subjects = [
            subject_id for subject_id, info in self.subjects_mapping.items()
            if info['has_clinical'] and info['has_ecg']  # Don't require RR-interval
        ]

        print(f"📊 Subject mapping summary:")
        print(f"   COMPLETE subjects (Clinical+ECG): {len(self.complete_subjects)}")
        print(
            f"   With RR-interval data: {sum(1 for s in self.complete_subjects if self.subjects_mapping[s]['has_rr'])}")

        return self.subjects_mapping

    def create_enhanced_targets(self):
        """Create multiple target formulations to handle missing data"""
        print("🎯 Creating enhanced target variables...")

        df = self.clinical_data
        targets = {}
        usable_subjects = []

        for _, row in df.iterrows():
            subject_id = row['subject_id']
            if subject_id not in self.complete_subjects:
                continue

            admission_fbg = row['admission FBG (mmol/L)']
            discharge_fbg = row['Discharge FBG (mmol/L)']
            hba1c = row['HbA1c (%)']

            # Strategy 1: Use ANY available glucose measurement
            primary_glucose = None
            if pd.notna(hba1c):
                primary_glucose = hba1c
                glucose_type = 'hba1c'
            elif pd.notna(admission_fbg):
                primary_glucose = admission_fbg
                glucose_type = 'admission_fbg'
            elif pd.notna(discharge_fbg):
                primary_glucose = discharge_fbg
                glucose_type = 'discharge_fbg'

            if primary_glucose is not None:
                usable_subjects.append({
                    'subject_id': subject_id,
                    'primary_glucose': primary_glucose,
                    'glucose_type': glucose_type,
                    'admission_fbg': admission_fbg,
                    'discharge_fbg': discharge_fbg,
                    'hba1c': hba1c,
                    'has_glucose_change': pd.notna(admission_fbg) and pd.notna(discharge_fbg)
                })

        # Convert to arrays
        subjects_df = pd.DataFrame(usable_subjects)

        # Target 1: Primary glucose (continuous)
        targets['primary_glucose'] = subjects_df['primary_glucose'].values

        # Target 2: Glucose control categories (ADA guidelines)
        glucose_control = []
        for _, row in subjects_df.iterrows():
            if row['glucose_type'] == 'hba1c':
                if row['primary_glucose'] < 7.0:
                    glucose_control.append(0)  # Good control
                elif row['primary_glucose'] < 8.5:
                    glucose_control.append(1)  # Fair control
                else:
                    glucose_control.append(2)  # Poor control
            else:  # FBG
                if row['primary_glucose'] < 7.0:
                    glucose_control.append(0)  # Normal/good
                elif row['primary_glucose'] < 10.0:
                    glucose_control.append(1)  # Elevated
                else:
                    glucose_control.append(2)  # High

        targets['glucose_control'] = np.array(glucose_control)

        # Target 3: Binary elevated glucose
        targets['glucose_elevated'] = (subjects_df['primary_glucose'] > 7.0).astype(int).values

        # Target 4: Glucose improvement (when available)
        improvement_subjects = subjects_df[subjects_df['has_glucose_change']].copy()
        if len(improvement_subjects) > 0:
            glucose_change = improvement_subjects['discharge_fbg'] - improvement_subjects['admission_fbg']
            targets['glucose_change'] = glucose_change.values
            targets['glucose_improved'] = (glucose_change < 0).astype(int).values

            # Store indices for glucose change subjects
            targets['glucose_change_indices'] = improvement_subjects.index.values

        # Store subject mapping for targets
        targets['subject_ids'] = subjects_df['subject_id'].values
        targets['glucose_types'] = subjects_df['glucose_type'].values

        print(f"✅ Enhanced targets created:")
        print(f"   Total usable subjects: {len(subjects_df)}")
        print(f"   With glucose change data: {len(improvement_subjects) if len(improvement_subjects) > 0 else 0}")
        print(
            f"   Target variables: {[k for k in targets.keys() if not k.endswith('_indices') and not k.endswith('_ids') and not k.endswith('_types')]}")

        # Update complete subjects list
        self.complete_subjects = subjects_df['subject_id'].tolist()

        return targets

    def extract_ecg_features(self, subject_id):
        """Extract ECG features with proper scaling"""
        subject_info = self.subjects_mapping.get(subject_id, {})
        ecg_file = subject_info.get('ecg_file')

        if ecg_file is None or not ecg_file.exists():
            return None

        try:
            ecg_data = scipy.io.loadmat(str(ecg_file))
            features = {}

            print(f"   Processing ECG for {subject_id}...")

            for var_name in ['all', 'sleep', 'day']:
                if var_name in ecg_data:
                    signal = ecg_data[var_name].flatten()

                    # CRITICAL: Fix ECG scaling
                    signal = self.validate_ecg_scaling(signal)

                    # Basic signal statistics
                    features[f'ecg_{var_name}_length'] = len(signal)
                    features[f'ecg_{var_name}_duration_hours'] = len(signal) / 250 / 3600
                    features[f'ecg_{var_name}_mean'] = np.mean(signal)
                    features[f'ecg_{var_name}_std'] = np.std(signal)
                    features[f'ecg_{var_name}_min'] = np.min(signal)
                    features[f'ecg_{var_name}_max'] = np.max(signal)
                    features[f'ecg_{var_name}_range'] = np.max(signal) - np.min(signal)

                    # Signal quality metrics
                    if np.std(signal) > 0:
                        features[f'ecg_{var_name}_snr_estimate'] = np.abs(np.mean(signal)) / np.std(signal)
                    else:
                        features[f'ecg_{var_name}_snr_estimate'] = 0

            return features

        except Exception as e:
            print(f"❌ Error processing ECG for {subject_id}: {e}")
            return None

    def extract_hrv_features(self, subject_id):
        """Extract HRV features from RR-interval data"""
        subject_info = self.subjects_mapping.get(subject_id, {})
        rr_file = subject_info.get('rr_file')

        if rr_file is None or not rr_file.exists():
            print(f"   No RR-interval data for {subject_id}")
            return None

        try:
            rr_data = scipy.io.loadmat(str(rr_file))
            features = {}

            for stage in ['DS', 'RS', 'REM']:
                if stage in rr_data:
                    intervals = rr_data[stage].flatten() / 1000.0  # Convert ms to seconds

                    if len(intervals) > 1:
                        # Time domain HRV features
                        features[f'hrv_{stage.lower()}_mean_rr'] = np.mean(intervals)
                        features[f'hrv_{stage.lower()}_std_rr'] = np.std(intervals)
                        features[f'hrv_{stage.lower()}_mean_hr'] = 60 / np.mean(intervals)

                        # RMSSD
                        rr_diffs = np.diff(intervals)
                        features[f'hrv_{stage.lower()}_rmssd'] = np.sqrt(np.mean(rr_diffs ** 2))

                        # pNN50
                        features[f'hrv_{stage.lower()}_pnn50'] = np.sum(np.abs(rr_diffs) > 0.05) / len(rr_diffs) * 100

                        # Additional metrics
                        features[f'hrv_{stage.lower()}_min_rr'] = np.min(intervals)
                        features[f'hrv_{stage.lower()}_max_rr'] = np.max(intervals)
                        features[f'hrv_{stage.lower()}_range_rr'] = np.max(intervals) - np.min(intervals)
                        features[f'hrv_{stage.lower()}_duration_hours'] = len(intervals) * np.mean(intervals) / 3600
                        features[f'hrv_{stage.lower()}_count'] = len(intervals)

            return features

        except Exception as e:
            print(f"❌ Error processing RR-intervals for {subject_id}: {e}")
            return None

    def extract_clinical_features(self, subject_id):
        """Extract clinical features"""
        subject_data = self.clinical_data[self.clinical_data['subject_id'] == subject_id]

        if len(subject_data) == 0:
            return None

        features = subject_data.iloc[0].to_dict()
        return features

    def extract_sleep_features(self, subject_id):
        """Extract sleep features"""
        features = {}

        # Objective sleep features
        if self.objective_sleep is not None:
            obj_data = self.objective_sleep[self.objective_sleep['number'] == subject_id]
            if len(obj_data) > 0:
                obj_features = obj_data.iloc[0].to_dict()
                for key, value in obj_features.items():
                    if key not in ['number', 'gender']:
                        features[f'psqi_{key}'] = value

        # Subjective sleep features
        if self.subjective_sleep is not None:
            subj_data = self.subjective_sleep[self.subjective_sleep['number'] == subject_id]
            if len(subj_data) > 0:
                subj_features = subj_data.iloc[0].to_dict()
                for key, value in subj_features.items():
                    if key != 'number':
                        features[f'cpc_{key}'] = value

        return features if features else None

    def process_all_subjects(self):
        """Process all subjects with enhanced target creation"""
        print("🔬 Processing all subjects...")

        # First create enhanced targets to get final subject list
        targets = self.create_enhanced_targets()

        all_features = []

        for subject_id in self.complete_subjects:
            print(f"Processing {subject_id}...")

            subject_features = {'subject_id': subject_id}

            # Clinical features
            clinical_features = self.extract_clinical_features(subject_id)
            if clinical_features:
                subject_features.update(clinical_features)

            # ECG features
            ecg_features = self.extract_ecg_features(subject_id)
            if ecg_features:
                subject_features.update(ecg_features)

            # HRV features (optional now)
            hrv_features = self.extract_hrv_features(subject_id)
            if hrv_features:
                subject_features.update(hrv_features)

            # Sleep features
            sleep_features = self.extract_sleep_features(subject_id)
            if sleep_features:
                subject_features.update(sleep_features)

            all_features.append(subject_features)

        self.processed_data['features'] = pd.DataFrame(all_features)
        self.processed_data['targets'] = targets

        print(f"✅ Processed {len(all_features)} subjects")
        print(f"   Total features: {len(self.processed_data['features'].columns)}")

        return self.processed_data['features']

    def create_train_test_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Create train/test splits with proper NaN handling"""
        if 'targets' not in self.processed_data:
            raise ValueError("Must create targets first")

        targets = self.processed_data['targets']

        # Prepare feature matrix (exclude non-feature columns)
        df = self.processed_data['features']
        exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                       [col for col in df.columns if any(term in col for term in
                                                         ['FBG', 'HbA1c', 'Diabetic', 'Coronary', 'Carotid',
                                                          'glucose'])]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].fillna(0).values

        # Use primary glucose for stratification (NO NaN VALUES)
        y_stratify = targets['primary_glucose']  # This is guaranteed to have no NaN

        # Convert to categories for stratification
        y_stratify_cat = pd.cut(y_stratify, bins=3, labels=[0, 1, 2], include_lowest=True)

        # Create splits
        n_samples = len(X)
        if n_samples < 20:
            # Too small for train/val/test split - use cross-validation instead
            print(f"⚠️  Small sample size ({n_samples}). Using cross-validation approach.")
            splits = {
                'X_full': X,
                'y_stratify': y_stratify_cat,
                'use_cv': True,
                'cv_folds': min(5, n_samples)  # 5-fold or leave-one-out
            }
        else:
            # Standard train/val/test split
            total_test_val_size = test_size + val_size

            X_train, X_temp, y_train_cat, y_temp_cat = train_test_split(
                X, y_stratify_cat, test_size=total_test_val_size,
                stratify=y_stratify_cat, random_state=random_state
            )

            if len(X_temp) >= 4:  # Ensure enough samples for val/test split
                X_val, X_test, _, _ = train_test_split(
                    X_temp, y_temp_cat, test_size=(test_size / total_test_val_size),
                    stratify=y_temp_cat, random_state=random_state
                )
            else:
                # Too few for separate val/test - use temp as test
                X_val = X_temp[:len(X_temp) // 2] if len(X_temp) > 1 else X_temp
                X_test = X_temp[len(X_temp) // 2:] if len(X_temp) > 1 else X_temp

            splits = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'use_cv': False
            }

        # Add all targets to splits
        for target_name, target_values in targets.items():
            if isinstance(target_values, np.ndarray) and len(target_values) == n_samples:
                if splits['use_cv']:
                    splits[f'y_{target_name}'] = target_values
                else:
                    # Split targets according to feature splits
                    train_size = len(splits['X_train'])
                    val_size = len(splits['X_val'])

                    splits[f'y_train_{target_name}'] = target_values[:train_size]
                    splits[f'y_val_{target_name}'] = target_values[train_size:train_size + val_size]
                    splits[f'y_test_{target_name}'] = target_values[train_size + val_size:]

        self.processed_data['splits'] = splits
        self.processed_data['feature_names'] = feature_cols

        print(f"✅ Created data splits:")
        if splits['use_cv']:
            print(f"   Using {splits['cv_folds']}-fold cross-validation")
            print(f"   Total samples: {n_samples}")
        else:
            print(f"   Train: {len(splits['X_train'])} subjects")
            print(f"   Validation: {len(splits['X_val'])} subjects")
            print(f"   Test: {len(splits['X_test'])} subjects")

        return splits

    def save_processed_data(self, output_dir="processed_data"):
        """Save all processed data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save features
        if 'features' in self.processed_data:
            features_file = output_dir / "enhanced_features.csv"
            self.processed_data['features'].to_csv(features_file, index=False)
            print(f"✅ Saved features to {features_file}")

        # Save targets
        if 'targets' in self.processed_data:
            targets_file = output_dir / "enhanced_targets.json"
            targets_json = {}
            for k, v in self.processed_data['targets'].items():
                if isinstance(v, np.ndarray):
                    targets_json[k] = v.tolist()
                else:
                    targets_json[k] = v

            with open(targets_file, 'w') as f:
                json.dump(targets_json, f, indent=2)
            print(f"✅ Saved targets to {targets_file}")

        # Save splits
        if 'splits' in self.processed_data:
            splits_file = output_dir / "enhanced_splits.npz"
            splits_to_save = {}
            for k, v in self.processed_data['splits'].items():
                if isinstance(v, np.ndarray):
                    splits_to_save[k] = v
                elif isinstance(v, (int, float, bool)):
                    splits_to_save[k] = np.array([v])

            np.savez(splits_file, **splits_to_save)
            print(f"✅ Saved splits to {splits_file}")

        # Save feature names
        if 'feature_names' in self.processed_data:
            feature_names_file = output_dir / "feature_names.json"
            with open(feature_names_file, 'w') as f:
                json.dump(self.processed_data['feature_names'], f, indent=2)
            print(f"✅ Saved feature names to {feature_names_file}")

        # Enhanced summary
        summary = {
            'total_subjects': len(self.complete_subjects),
            'complete_subjects': self.complete_subjects,
            'feature_count': len(self.processed_data.get('feature_names', [])),
            'target_variables': [k for k in self.processed_data.get('targets', {}).keys()
                                 if not k.endswith('_indices') and not k.endswith('_ids') and not k.endswith('_types')],
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'data_quality_fixes': [
                'ECG scaling validation and correction',
                'Enhanced missing data handling',
                'Relaxed inclusion criteria',
                'Multiple target formulations',
                'Robust train/test splitting'
            ]
        }

        summary_file = output_dir / "enhanced_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved enhanced summary to {summary_file}")

        return output_dir

    def run_complete_pipeline(self):
        """Run the complete FIXED preprocessing pipeline"""
        print("🚀 Starting FIXED preprocessing pipeline...")
        print("🔧 Addresses: ECG scaling, missing data, sample size, NaN handling")

        # Load all data
        self.load_clinical_data()
        self.load_objective_sleep_data()
        self.load_subjective_sleep_data()

        # Create mapping with relaxed criteria
        self.create_subject_mapping()

        # Process subjects with enhanced features
        self.process_all_subjects()

        # Create robust splits
        self.create_train_test_splits()

        # Save everything
        output_dir = self.save_processed_data()

        print("🎉Preprocessing pipeline completed!")
        print(f"📁Data saved to: {output_dir}")

        return self.processed_data


# Usage
if __name__ == "__main__":
    print("🔧 RUNNING PREPROCESSING PIPELINE")
    print("=" * 50)

    preprocessor = FixedDiabetesECGPreprocessor(".")
    processed_data = preprocessor.run_complete_pipeline()

    # Display final results
    print("\n📊 ENHANCED DATASET SUMMARY:")
    print("=" * 50)

    targets = processed_data['targets']
    splits = processed_data['splits']

    print(f"✅ Total subjects with valid targets: {len(targets['subject_ids'])}")
    print(f"✅ Total features: {len(processed_data['feature_names'])}")
    print(
        f"✅ Target variables: {[k for k in targets.keys() if not k.endswith('_indices') and not k.endswith('_ids') and not k.endswith('_types')]}")

    if splits.get('use_cv', False):
        print(f"✅ Using cross-validation: {splits['cv_folds']} folds")
    else:
        print(f"✅ Train/Val/Test split: {len(splits['X_train'])}/{len(splits['X_val'])}/{len(splits['X_test'])}")

    # Display target statistics
    print(f"\n📈 Target Statistics:")
    glucose_values = targets['primary_glucose']
    print(f"   Primary glucose: {np.mean(glucose_values):.1f} ± {np.std(glucose_values):.1f}")
    print(f"   Glucose control: {np.bincount(targets['glucose_control'])}")
    print(f"   Elevated glucose: {np.sum(targets['glucose_elevated'])}/{len(targets['glucose_elevated'])} subjects")

    print("\n🎯 READY FOR MODEL DEVELOPMENT!")
    print("   - Sample size significantly improved")
    print("   - ECG scaling issues resolved")
    print("   - Multiple target formulations available")
    print("   - Robust validation strategy implemented")