#!/usr/bin/env python3
"""
Complete Data Preprocessing Pipeline for Diabetes ECG Research
Handles all data loading, cleaning, and feature extraction
"""

import os
import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class DiabetesECGPreprocessor:
    def __init__(self, dataset_path="."):
        self.dataset_path = Path(dataset_path)
        self.clinical_data = None
        self.objective_sleep = None
        self.subjective_sleep = None
        self.subjects_mapping = {}
        self.complete_subjects = []
        self.processed_data = {}

    def load_clinical_data(self):
        """Load and clean clinical indicators"""
        print("📋 Loading clinical data...")

        clinical_file = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/clinical_indicators.xlsx"
        self.clinical_data = pd.read_excel(clinical_file)

        # Rename subject ID column
        self.clinical_data = self.clinical_data.rename(columns={'Unnamed: 0': 'subject_id'})

        # Convert subject_id to string for consistency
        self.clinical_data['subject_id'] = self.clinical_data['subject_id'].astype(str)

        print(f"✅ Loaded clinical data: {self.clinical_data.shape}")
        print(f"   Subjects: {len(self.clinical_data)}")
        print(f"   Features: {len(self.clinical_data.columns)}")

        return self.clinical_data

    def load_objective_sleep_data(self):
        """Load and fix objective sleep quality (PSQI) data"""
        print("😴 Loading objective sleep data...")

        obj_file = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/objective_sleep_quality.xlsx"
        raw_obj_sleep = pd.read_excel(obj_file)

        # Fix the header issue - real column names are in row 0
        # Extract the real column names from row 0, columns 5 onwards
        real_columns = ['number', 'gender', 'age', 'height', 'weight']

        # Get PSQI component names from row 0
        psqi_columns = []
        for i in range(5, len(raw_obj_sleep.columns)):
            col_value = raw_obj_sleep.iloc[0, i]
            if pd.notna(col_value):
                psqi_columns.append(str(col_value))
            else:
                psqi_columns.append(f'psqi_component_{i - 4}')

        real_columns.extend(psqi_columns)

        # Create properly formatted dataframe (skip row 0 which has headers)
        self.objective_sleep = raw_obj_sleep.iloc[1:].copy()
        self.objective_sleep.columns = real_columns[:len(self.objective_sleep.columns)]

        # Clean and convert data types
        self.objective_sleep['number'] = self.objective_sleep['number'].astype(str)

        # Convert numeric columns
        numeric_cols = self.objective_sleep.columns[2:]  # Skip number, gender
        for col in numeric_cols:
            self.objective_sleep[col] = pd.to_numeric(self.objective_sleep[col], errors='coerce')

        print(f"✅ Fixed objective sleep data: {self.objective_sleep.shape}")
        print(f"   Columns: {list(self.objective_sleep.columns)}")

        return self.objective_sleep

    def load_subjective_sleep_data(self):
        """Load subjective sleep quality (CPC analysis) data"""
        print("🧠 Loading subjective sleep data...")

        subj_file = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/subjective_sleep_quality.xlsx"
        self.subjective_sleep = pd.read_excel(subj_file)

        # Convert subject ID to string
        self.subjective_sleep['number'] = self.subjective_sleep['number'].astype(str)

        print(f"✅ Loaded subjective sleep data: {self.subjective_sleep.shape}")
        print(f"   Sleep metrics: {list(self.subjective_sleep.columns[1:])}")

        return self.subjective_sleep

    def create_subject_mapping(self):
        """Create comprehensive subject mapping across all data sources"""
        print("🗺️ Creating subject mapping...")

        # Get subjects from each data source
        clinical_subjects = set(self.clinical_data['subject_id'])

        # ECG subjects
        ecg_dir = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/ECG"
        ecg_subjects = set([f.stem for f in ecg_dir.glob("*.mat")]) if ecg_dir.exists() else set()

        # RR-interval subjects
        rr_dir = self.dataset_path / "Dataset_on_electrocardiograph/dataset_ecg/RR_interval"
        rr_subjects = set([f.stem for f in rr_dir.glob("*.mat")]) if rr_dir.exists() else set()

        # Sleep subjects
        obj_sleep_subjects = set(self.objective_sleep['number']) if self.objective_sleep is not None else set()
        subj_sleep_subjects = set(self.subjective_sleep['number']) if self.subjective_sleep is not None else set()

        # Create comprehensive mapping
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

        # Identify complete subjects (have all critical data)
        self.complete_subjects = [
            subject_id for subject_id, info in self.subjects_mapping.items()
            if info['has_clinical'] and info['has_ecg'] and info['has_rr']
        ]

        print(f"📊 Subject mapping summary:")
        print(f"   Total unique subjects: {len(all_subjects)}")
        print(f"   Clinical data: {len(clinical_subjects)}")
        print(f"   ECG data: {len(ecg_subjects)}")
        print(f"   RR-interval data: {len(rr_subjects)}")
        print(f"   Objective sleep: {len(obj_sleep_subjects)}")
        print(f"   Subjective sleep: {len(subj_sleep_subjects)}")
        print(f"   COMPLETE subjects (Clinical+ECG+RR): {len(self.complete_subjects)}")

        return self.subjects_mapping

    def extract_clinical_features(self, subject_id):
        """Extract clinical features for a subject"""
        subject_data = self.clinical_data[self.clinical_data['subject_id'] == subject_id]

        if len(subject_data) == 0:
            return None

        features = subject_data.iloc[0].to_dict()

        # Create target variables
        admission_fbg = features.get('admission FBG (mmol/L)', np.nan)
        discharge_fbg = features.get('Discharge FBG (mmol/L)', np.nan)

        if pd.notna(admission_fbg) and pd.notna(discharge_fbg):
            features['glucose_change'] = discharge_fbg - admission_fbg
            features['glucose_improved'] = 1 if (discharge_fbg < admission_fbg) else 0

        return features

    def extract_ecg_features(self, subject_id):
        """Extract features from ECG data"""
        subject_info = self.subjects_mapping.get(subject_id, {})
        ecg_file = subject_info.get('ecg_file')

        if ecg_file is None or not ecg_file.exists():
            return None

        try:
            ecg_data = scipy.io.loadmat(str(ecg_file))
            features = {}

            for var_name in ['all', 'sleep', 'day']:
                if var_name in ecg_data:
                    signal = ecg_data[var_name].flatten()

                    # Basic signal statistics
                    features[f'ecg_{var_name}_length'] = len(signal)
                    features[f'ecg_{var_name}_duration_hours'] = len(signal) / 250 / 3600
                    features[f'ecg_{var_name}_mean'] = np.mean(signal)
                    features[f'ecg_{var_name}_std'] = np.std(signal)
                    features[f'ecg_{var_name}_min'] = np.min(signal)
                    features[f'ecg_{var_name}_max'] = np.max(signal)
                    features[f'ecg_{var_name}_range'] = np.max(signal) - np.min(signal)

                    # Signal quality metrics
                    features[f'ecg_{var_name}_snr_estimate'] = np.mean(signal) / np.std(signal)

            return features

        except Exception as e:
            print(f"❌ Error processing ECG for {subject_id}: {e}")
            return None

    def extract_hrv_features(self, subject_id):
        """Extract HRV features from RR-interval data"""
        subject_info = self.subjects_mapping.get(subject_id, {})
        rr_file = subject_info.get('rr_file')

        if rr_file is None or not rr_file.exists():
            return None

        try:
            rr_data = scipy.io.loadmat(str(rr_file))
            features = {}

            for stage in ['DS', 'RS', 'REM']:  # Unstable, Stable, REM sleep
                if stage in rr_data:
                    intervals = rr_data[stage].flatten() / 1000.0  # Convert to seconds

                    if len(intervals) > 1:
                        # Time domain HRV features
                        features[f'hrv_{stage.lower()}_mean_rr'] = np.mean(intervals)
                        features[f'hrv_{stage.lower()}_std_rr'] = np.std(intervals)
                        features[f'hrv_{stage.lower()}_mean_hr'] = 60 / np.mean(intervals)

                        # RMSSD (root mean square of successive differences)
                        rr_diffs = np.diff(intervals)
                        features[f'hrv_{stage.lower()}_rmssd'] = np.sqrt(np.mean(rr_diffs ** 2))

                        # pNN50 (percentage of successive RR intervals that differ by more than 50ms)
                        features[f'hrv_{stage.lower()}_pnn50'] = np.sum(np.abs(rr_diffs) > 0.05) / len(rr_diffs) * 100

                        # Additional metrics
                        features[f'hrv_{stage.lower()}_min_rr'] = np.min(intervals)
                        features[f'hrv_{stage.lower()}_max_rr'] = np.max(intervals)
                        features[f'hrv_{stage.lower()}_range_rr'] = np.max(intervals) - np.min(intervals)

                        # Duration
                        features[f'hrv_{stage.lower()}_duration_hours'] = len(intervals) * np.mean(intervals) / 3600
                        features[f'hrv_{stage.lower()}_count'] = len(intervals)

            return features

        except Exception as e:
            print(f"❌ Error processing RR-intervals for {subject_id}: {e}")
            return None

    def extract_sleep_features(self, subject_id):
        """Extract sleep quality features"""
        features = {}

        # Objective sleep (PSQI) features
        if self.objective_sleep is not None:
            obj_data = self.objective_sleep[self.objective_sleep['number'] == subject_id]
            if len(obj_data) > 0:
                obj_features = obj_data.iloc[0].to_dict()
                for key, value in obj_features.items():
                    if key not in ['number', 'gender']:
                        features[f'psqi_{key}'] = value

        # Subjective sleep (CPC) features
        if self.subjective_sleep is not None:
            subj_data = self.subjective_sleep[self.subjective_sleep['number'] == subject_id]
            if len(subj_data) > 0:
                subj_features = subj_data.iloc[0].to_dict()
                for key, value in subj_features.items():
                    if key != 'number':
                        features[f'cpc_{key}'] = value

        return features if features else None

    def process_all_subjects(self):
        """Process all complete subjects and create feature dataset"""
        print("🔬 Processing all subjects...")

        all_features = []

        for subject_id in self.complete_subjects:
            print(f"Processing {subject_id}...")

            subject_features = {'subject_id': subject_id}

            # Clinical features (including targets)
            clinical_features = self.extract_clinical_features(subject_id)
            if clinical_features:
                subject_features.update(clinical_features)

            # ECG features
            ecg_features = self.extract_ecg_features(subject_id)
            if ecg_features:
                subject_features.update(ecg_features)

            # HRV features
            hrv_features = self.extract_hrv_features(subject_id)
            if hrv_features:
                subject_features.update(hrv_features)

            # Sleep features
            sleep_features = self.extract_sleep_features(subject_id)
            if sleep_features:
                subject_features.update(sleep_features)

            all_features.append(subject_features)

        # Create comprehensive dataset
        self.processed_data['features'] = pd.DataFrame(all_features)

        print(f"✅ Processed {len(all_features)} complete subjects")
        print(f"   Total features: {len(self.processed_data['features'].columns)}")

        return self.processed_data['features']

    def create_targets(self):
        """Create target variables for modeling"""
        if 'features' not in self.processed_data:
            raise ValueError("Must process subjects first")

        df = self.processed_data['features']
        targets = {}

        # Primary targets
        if 'glucose_change' in df.columns:
            targets['glucose_change'] = df['glucose_change'].values

        if 'admission FBG (mmol/L)' in df.columns:
            targets['admission_fbg'] = df['admission FBG (mmol/L)'].values

        if 'Discharge FBG (mmol/L)' in df.columns:
            targets['discharge_fbg'] = df['Discharge FBG (mmol/L)'].values

        if 'HbA1c (%)' in df.columns:
            targets['hba1c'] = df['HbA1c (%)'].values

        # Classification targets
        if 'glucose_improved' in df.columns:
            targets['glucose_improved'] = df['glucose_improved'].values

        # Complications
        complication_cols = [col for col in df.columns if 'Diabetic' in col or 'Coronary' in col or 'Carotid' in col]
        for col in complication_cols:
            clean_name = col.lower().replace(' ', '_').replace('diabetic_', '').replace(
                'coronary_artery_disease_and_cardiac_insufficiency', 'cardiac')
            targets[clean_name] = df[col].values

        self.processed_data['targets'] = targets
        print(f"✅ Created {len(targets)} target variables")

        return targets

    def prepare_model_data(self, feature_categories=None):
        """Prepare data for machine learning models"""
        if 'features' not in self.processed_data:
            raise ValueError("Must process subjects first")

        df = self.processed_data['features']

        # Define feature categories
        if feature_categories is None:
            feature_categories = {
                'demographic': ['age', 'height', 'weight'],
                'clinical_blood': [col for col in df.columns if
                                   any(marker in col for marker in ['WBC', 'Hb', 'PLT', 'CRP'])],
                'clinical_metabolic': [col for col in df.columns if any(
                    marker in col for marker in ['ALT', 'AST', 'BUN', 'UA', 'TG', 'HDL', 'LDL'])],
                'clinical_bp': [col for col in df.columns if any(marker in col for marker in ['SBP', 'DBP'])],
                'ecg': [col for col in df.columns if col.startswith('ecg_')],
                'hrv': [col for col in df.columns if col.startswith('hrv_')],
                'sleep': [col for col in df.columns if col.startswith('psqi_') or col.startswith('cpc_')]
            }

        # Prepare feature matrices for each category
        model_data = {}

        for category, columns in feature_categories.items():
            available_cols = [col for col in columns if col in df.columns]
            if available_cols:
                model_data[category] = df[available_cols].fillna(0).values
                print(f"✅ {category}: {len(available_cols)} features")

        # Combined feature matrix
        all_feature_cols = []
        for cols in feature_categories.values():
            all_feature_cols.extend([col for col in cols if col in df.columns])

        # Remove duplicates and non-feature columns
        exclude_cols = ['subject_id', 'gender', 'glucose_change', 'glucose_improved'] + \
                       [col for col in df.columns if
                        'Diabetic' in col or 'admission FBG' in col or 'Discharge FBG' in col or 'HbA1c' in col]

        feature_cols = [col for col in all_feature_cols if col not in exclude_cols]
        feature_cols = list(dict.fromkeys(feature_cols))  # Remove duplicates while preserving order

        model_data['all_features'] = df[feature_cols].fillna(0).values
        model_data['feature_names'] = feature_cols

        self.processed_data['model_data'] = model_data

        print(f"✅ Prepared model data with {len(feature_cols)} total features")

        return model_data

    def create_train_test_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Create stratified train/validation/test splits"""
        if 'targets' not in self.processed_data or 'model_data' not in self.processed_data:
            raise ValueError("Must create targets and model data first")

        X = self.processed_data['model_data']['all_features']

        # Use HbA1c for stratification (good proxy for diabetes severity)
        y_stratify = self.processed_data['targets'].get('hba1c')
        if y_stratify is None:
            y_stratify = self.processed_data['targets'].get('admission_fbg')

        # Create stratified splits
        splits = {}

        if y_stratify is not None:
            # Convert to categorical for stratification
            y_stratify_cat = pd.cut(y_stratify, bins=3, labels=['low', 'medium', 'high'])

            # Train/temp split
            X_train, X_temp, y_train_cat, y_temp_cat = train_test_split(
                X, y_stratify_cat, test_size=(test_size + val_size),
                stratify=y_stratify_cat, random_state=random_state
            )

            # Validation/test split
            X_val, X_test, _, _ = train_test_split(
                X_temp, y_temp_cat, test_size=(test_size / (test_size + val_size)),
                stratify=y_temp_cat, random_state=random_state
            )

            splits['X_train'] = X_train
            splits['X_val'] = X_val
            splits['X_test'] = X_test

            # Get indices for target splits
            train_indices = np.arange(len(X))[~np.isin(np.arange(len(X)),
                                                       np.concatenate([X_val, X_test]))]
            val_indices = np.arange(len(X_temp))[:len(X_val)]
            test_indices = np.arange(len(X_temp))[len(X_val):]

            # Split all targets
            for target_name, target_values in self.processed_data['targets'].items():
                splits[f'y_train_{target_name}'] = target_values[train_indices]
                splits[f'y_val_{target_name}'] = target_values[val_indices]
                splits[f'y_test_{target_name}'] = target_values[test_indices]

        self.processed_data['splits'] = splits

        print(f"✅ Created data splits:")
        print(f"   Train: {splits['X_train'].shape[0]} subjects")
        print(f"   Validation: {splits['X_val'].shape[0]} subjects")
        print(f"   Test: {splits['X_test'].shape[0]} subjects")

        return splits

    def save_processed_data(self, output_dir="processed_data"):
        """Save all processed data"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save feature dataset
        if 'features' in self.processed_data:
            features_file = output_dir / "complete_features.csv"
            self.processed_data['features'].to_csv(features_file, index=False)
            print(f"✅ Saved features to {features_file}")

        # Save targets
        if 'targets' in self.processed_data:
            targets_file = output_dir / "targets.json"
            # Convert numpy arrays to lists for JSON serialization
            targets_json = {k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in self.processed_data['targets'].items()}
            with open(targets_file, 'w') as f:
                json.dump(targets_json, f, indent=2)
            print(f"✅ Saved targets to {targets_file}")

        # Save model data
        if 'model_data' in self.processed_data:
            model_file = output_dir / "model_data.npz"
            np.savez(model_file, **{k: v for k, v in self.processed_data['model_data'].items()
                                    if isinstance(v, np.ndarray)})

            # Save feature names separately
            feature_names_file = output_dir / "feature_names.json"
            with open(feature_names_file, 'w') as f:
                json.dump(self.processed_data['model_data']['feature_names'], f, indent=2)
            print(f"✅ Saved model data to {model_file}")

        # Save splits
        if 'splits' in self.processed_data:
            splits_file = output_dir / "data_splits.npz"
            np.savez(splits_file, **{k: v for k, v in self.processed_data['splits'].items()
                                     if isinstance(v, np.ndarray)})
            print(f"✅ Saved data splits to {splits_file}")

        # Save summary
        summary = {
            'total_subjects': len(self.complete_subjects),
            'complete_subjects': self.complete_subjects,
            'feature_count': len(self.processed_data.get('model_data', {}).get('feature_names', [])),
            'target_variables': list(self.processed_data.get('targets', {}).keys()),
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }

        summary_file = output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved summary to {summary_file}")

        return output_dir

    def run_complete_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("🚀 Starting complete preprocessing pipeline...")

        # Step 1: Load all data
        self.load_clinical_data()
        self.load_objective_sleep_data()
        self.load_subjective_sleep_data()

        # Step 2: Create subject mapping
        self.create_subject_mapping()

        # Step 3: Process all subjects
        self.process_all_subjects()

        # Step 4: Create targets
        self.create_targets()

        # Step 5: Prepare model data
        self.prepare_model_data()

        # Step 6: Create splits
        self.create_train_test_splits()

        # Step 7: Save everything
        output_dir = self.save_processed_data()

        print("🎉 Complete preprocessing pipeline finished!")
        print(f"📁 All data saved to: {output_dir}")

        return self.processed_data


# Usage example
if __name__ == "__main__":
    # Initialize and run preprocessing
    preprocessor = DiabetesECGPreprocessor(".")
    processed_data = preprocessor.run_complete_pipeline()

    # Display summary
    print("\n📊 FINAL DATASET SUMMARY:")
    print("=" * 50)

    features_df = processed_data['features']
    targets = processed_data['targets']

    print(f"Complete subjects: {len(features_df)}")
    print(f"Total features: {len(processed_data['model_data']['feature_names'])}")
    print(f"Target variables: {list(targets.keys())}")

    # Show target statistics
    for target_name, target_values in targets.items():
        if target_name in ['glucose_change', 'admission_fbg', 'discharge_fbg', 'hba1c']:
            valid_values = target_values[~np.isnan(target_values)]
            if len(valid_values) > 0:
                print(f"{target_name}: mean={np.mean(valid_values):.2f}, std={np.std(valid_values):.2f}")

    print("\n✅ Ready for model development!")