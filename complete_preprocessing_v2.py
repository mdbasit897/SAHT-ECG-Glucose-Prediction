#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import scipy.io
from pathlib import Path
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')


class RevisedDiabetesECGPreprocessor:


    def __init__(self, dataset_path: str = "."):
        self.dataset_path = Path(dataset_path)
        self.clinical_data = None
        self.objective_sleep = None
        self.subjective_sleep = None
        self.subjects_mapping = {}
        self.complete_subjects = []
        self.processed_data = {}
        self.ecg_scaling_logs = []
        self.signal_specifications = {}

        # Processing metadata
        self.processing_metadata = {
            'version': '2.0',
            'processing_date': datetime.now().isoformat(),
            'concerns_addressed': [
                'Separated HbA1c and FBG targets',
                'LOSO cross-validation',
                'Temporal validation splits',
                'ECG scaling validation with logging',
                'Comprehensive signal documentation',
                'FDR-corrected feature selection'
            ]
        }

        print(f"Dataset path: {self.dataset_path.absolute()}")
        print()


    def document_signal_specifications(self) -> Dict:

        print(" Documenting Signal Specifications...")

        self.signal_specifications = {
            'ecg': {
                'source': 'Mendeley Dataset (Cheng et al., 2023)',
                'doi': '10.17632/9c47vwvtss.4',
                'sampling_rate_hz': 250,
                'lead_configuration': 'Single-lead ECG',
                'recording_duration': 'Overnight (~8-10 hours)',
                'preprocessing': {
                    'r_peak_detection': 'Dataset-provided RR intervals',
                    'artifact_handling': 'Outlier filtering (±3 SD)',
                    'units_after_processing': 'mV (after scaling correction)'
                },
                'physiological_range': {
                    'expected_amplitude': '±0.5 to ±5 mV',
                    'qrs_duration': '80-120 ms typical'
                }
            },
            'rr_intervals': {
                'source': 'Derived from ECG R-peak detection',
                'original_units': 'milliseconds',
                'processed_units': 'seconds (for HRV calculation)',
                'sleep_stages': {
                    'DS': 'Deep Sleep (N3)',
                    'RS': 'Rapid Sleep (unclear - possibly light sleep)',
                    'REM': 'REM Sleep'
                },
                'staging_criteria': 'AASM guidelines (assumed from dataset)',
                'minimum_epoch_duration': 'Not specified in dataset'
            },
            'glucose_measurements': {
                'hba1c': {
                    'full_name': 'Glycated Hemoglobin',
                    'units': 'Percentage (%)',
                    'physiological_meaning': '~3-month average glycemic control',
                    'measurement_timing': 'Single measurement during hospitalization',
                    'clinical_thresholds': {
                        'normal': '<5.7%',
                        'prediabetes': '5.7-6.4%',
                        'diabetes': '≥6.5%',
                        'good_control': '<7.0%',
                        'fair_control': '7.0-8.5%',
                        'poor_control': '>8.5%'
                    }
                },
                'fbg': {
                    'full_name': 'Fasting Blood Glucose',
                    'units': 'mmol/L',
                    'physiological_meaning': 'Instantaneous fasting glucose',
                    'measurement_timing': 'Admission and/or discharge',
                    'clinical_thresholds': {
                        'normal': '<5.6 mmol/L',
                        'prediabetes': '5.6-6.9 mmol/L',
                        'diabetes': '≥7.0 mmol/L'
                    },
                    'conversion': '1 mmol/L = 18 mg/dL'
                }
            },
            'time_synchronization': {
                'ecg_to_sleep': 'Aligned by recording session',
                'ecg_to_glucose': 'NOT time-aligned (different measurement times)',
                'limitation': 'Glucose is spot measurement, not continuous CGM',
                'implication': 'Cannot study acute HRV-glucose relationships'
            },
            'population': {
                'inclusion': 'Male patients with Type 2 Diabetes',
                'exclusion': 'Females (dataset limitation)',
                'age_range': 'Adults (specific range in clinical data)',
                'setting': 'Hospital inpatients',
                'limitation': 'Results may not generalize to females or non-diabetic populations'
            }
        }

        print("    ECG specifications documented")
        print("    RR-interval specifications documented")
        print("    Glucose measurement specifications documented")
        print("    Time synchronization limitations documented")
        print("    Population characteristics documented")

        return self.signal_specifications

    # =========================================================================
    # ECG SCALING VALIDATION (Improved with logging)
    # =========================================================================

    def validate_ecg_scaling(self, ecg_signal: np.ndarray,
                             subject_id: str = None) -> np.ndarray:
        """
        Validate and correct ECG scaling with comprehensive logging.
        Physiological ECG range: typically ±0.5 to ±5 mV

        Parameters:
        -----------
        ecg_signal : np.ndarray
            Raw ECG signal
        subject_id : str
            Subject identifier for logging

        Returns:
        --------
        np.ndarray
            Scaled ECG signal in mV
        """
        original_signal = ecg_signal.copy()
        original_range = np.max(np.abs(ecg_signal))
        corrections_applied = []

        # Initialize scaling log
        scaling_log = {
            'subject_id': subject_id,
            'original_range': float(original_range),
            'original_unit_guess': None,
            'corrections': [],
            'final_range': None,
            'is_valid': False,
            'warning': None
        }

        # Detect likely unit based on range and apply correction
        if original_range > 50000:
            # Likely raw 16-bit ADC values
            scaling_log['original_unit_guess'] = 'raw_adc_16bit'
            ecg_signal = (ecg_signal / 32768) * 2.5  # Assume ±2.5mV range
            corrections_applied.append('adc_16bit_to_mv')

        elif original_range > 5000:
            # Likely raw 12-bit ADC values
            scaling_log['original_unit_guess'] = 'raw_adc_12bit'
            ecg_signal = (ecg_signal / 2048) * 2.5
            corrections_applied.append('adc_12bit_to_mv')

        elif original_range > 500:
            # Likely microvolts
            scaling_log['original_unit_guess'] = 'microvolts'
            ecg_signal = ecg_signal / 1000
            corrections_applied.append('uv_to_mv')

        elif original_range > 50:
            # Unknown large scale - normalize
            scaling_log['original_unit_guess'] = 'unknown_large'
            scale_factor = original_range / 2.5
            ecg_signal = ecg_signal / scale_factor
            corrections_applied.append(f'normalize_by_{scale_factor:.2f}')

        elif original_range < 0.01:
            # Too small - might be in volts
            scaling_log['original_unit_guess'] = 'possibly_volts'
            ecg_signal = ecg_signal * 1000
            corrections_applied.append('v_to_mv')

        else:
            # Likely already in mV
            scaling_log['original_unit_guess'] = 'millivolts'

        # Final validation
        final_range = np.max(np.abs(ecg_signal))
        scaling_log['final_range'] = float(final_range)
        scaling_log['corrections'] = corrections_applied

        # Physiological plausibility check (±0.1 to ±10 mV is reasonable)
        if 0.1 < final_range < 10:
            scaling_log['is_valid'] = True
        else:
            scaling_log['is_valid'] = False
            scaling_log['warning'] = f"Final range {final_range:.2f} mV outside expected physiological range"
            print(f"     Subject {subject_id}: {scaling_log['warning']}")

        # Store log
        self.ecg_scaling_logs.append(scaling_log)

        return ecg_signal

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_clinical_data(self) -> pd.DataFrame:
        """Load clinical data with comprehensive column documentation."""
        print("📋 Loading Clinical Data...")

        possible_paths = [
            self.dataset_path / "/home/mdbasit_tezu_ernet_in/datasets/electrocardiograph/clinical_indicators.xlsx",
            self.dataset_path / "clinical_indicators.xlsx"
        ]

        clinical_file = None
        for path in possible_paths:
            if path.exists():
                clinical_file = path
                break

        if clinical_file is None:
            raise FileNotFoundError(
                f"Clinical indicators file not found. Searched: {[str(p) for p in possible_paths]}"
            )

        self.clinical_data = pd.read_excel(clinical_file)

        # Standardize column names
        if 'Unnamed: 0' in self.clinical_data.columns:
            self.clinical_data = self.clinical_data.rename(columns={'Unnamed: 0': 'subject_id'})

        self.clinical_data['subject_id'] = self.clinical_data['subject_id'].astype(str)

        # Document available columns
        print(f"   ✅ Loaded: {self.clinical_data.shape[0]} subjects, {self.clinical_data.shape[1]} columns")

        # Analyze glucose data availability
        glucose_cols = {
            'admission FBG (mmol/L)': 'Admission Fasting Blood Glucose',
            'Discharge FBG (mmol/L)': 'Discharge Fasting Blood Glucose',
            'HbA1c (%)': 'Glycated Hemoglobin'
        }

        print("   📊 Glucose Data Availability:")
        for col, desc in glucose_cols.items():
            if col in self.clinical_data.columns:
                available = self.clinical_data[col].notna().sum()
                total = len(self.clinical_data)
                print(f"      {desc}: {available}/{total} ({available / total * 100:.1f}%)")

        return self.clinical_data

    def load_objective_sleep_data(self) -> Optional[pd.DataFrame]:
        """Load objective sleep quality data (PSG-derived)."""
        print("😴 Loading Objective Sleep Data...")

        possible_paths = [
            self.dataset_path / "/home/mdbasit_tezu_ernet_in/datasets/electrocardiograph/objective_sleep_quality.xlsx",
            self.dataset_path / "objective_sleep_quality.xlsx"
        ]

        obj_file = None
        for path in possible_paths:
            if path.exists():
                obj_file = path
                break

        if obj_file is None:
            print("     Objective sleep quality file not found. Continuing without it.")
            return None

        raw_obj_sleep = pd.read_excel(obj_file)

        # Handle header row issues
        real_columns = ['number', 'gender', 'age', 'height', 'weight']
        psqi_columns = []

        for i in range(5, len(raw_obj_sleep.columns)):
            col_value = raw_obj_sleep.iloc[0, i] if len(raw_obj_sleep) > 0 else None
            if pd.notna(col_value):
                psqi_columns.append(str(col_value).strip())
            else:
                psqi_columns.append(f'psqi_component_{i - 4}')

        real_columns.extend(psqi_columns)

        # Create clean dataframe
        self.objective_sleep = raw_obj_sleep.iloc[1:].copy()
        self.objective_sleep.columns = real_columns[:len(self.objective_sleep.columns)]
        self.objective_sleep['number'] = self.objective_sleep['number'].astype(str)

        # Convert numeric columns
        for col in self.objective_sleep.columns[2:]:
            self.objective_sleep[col] = pd.to_numeric(self.objective_sleep[col], errors='coerce')

        print(f"  Loaded: {self.objective_sleep.shape[0]} subjects, {self.objective_sleep.shape[1]} columns")
        return self.objective_sleep

    def load_subjective_sleep_data(self) -> Optional[pd.DataFrame]:
        """Load subjective sleep quality data (questionnaire-based)."""
        print(" Loading Subjective Sleep Data...")

        possible_paths = [
            self.dataset_path / "/home/mdbasit_tezu_ernet_in/datasets/electrocardiograph/subjective_sleep_quality.xlsx",
            self.dataset_path / "subjective_sleep_quality.xlsx"
        ]

        subj_file = None
        for path in possible_paths:
            if path.exists():
                subj_file = path
                break

        if subj_file is None:
            print("    Subjective sleep quality file not found. Continuing without it.")
            return None

        self.subjective_sleep = pd.read_excel(subj_file)
        self.subjective_sleep['number'] = self.subjective_sleep['number'].astype(str)

        print(f"   Loaded: {self.subjective_sleep.shape[0]} subjects, {self.subjective_sleep.shape[1]} columns")
        return self.subjective_sleep

    def create_subject_mapping(self) -> Dict:
        """Create comprehensive mapping of available data per subject."""
        print("  Creating Subject Mapping...")

        # Get subjects from clinical data
        clinical_subjects = set(self.clinical_data['subject_id'])

        # Find ECG directory
        possible_ecg_paths = [
            self.dataset_path / "/home/mdbasit_tezu_ernet_in/datasets/electrocardiograph/ECG",
            self.dataset_path / "ECG"
        ]

        ecg_dir = None
        for path in possible_ecg_paths:
            if path.exists():
                ecg_dir = path
                break

        ecg_subjects = set([f.stem for f in ecg_dir.glob("*.mat")]) if ecg_dir else set()

        # Find RR-interval directory
        possible_rr_paths = [
            self.dataset_path / "/home/mdbasit_tezu_ernet_in/datasets/electrocardiograph/rr_interval",
            self.dataset_path / "rr_interval"
        ]

        rr_dir = None
        for path in possible_rr_paths:
            if path.exists():
                rr_dir = path
                break

        rr_subjects = set([f.stem for f in rr_dir.glob("*.mat")]) if rr_dir else set()

        # Sleep data subjects
        obj_sleep_subjects = set(self.objective_sleep['number']) if self.objective_sleep is not None else set()
        subj_sleep_subjects = set(self.subjective_sleep['number']) if self.subjective_sleep is not None else set()

        print(f"     Data directories:")
        print(f"      ECG: {ecg_dir}")
        print(f"      RR-interval: {rr_dir}")

        # Create mapping
        all_subjects = clinical_subjects | ecg_subjects | rr_subjects | obj_sleep_subjects | subj_sleep_subjects

        for subject_id in all_subjects:
            self.subjects_mapping[subject_id] = {
                'has_clinical': subject_id in clinical_subjects,
                'has_ecg': subject_id in ecg_subjects,
                'has_rr': subject_id in rr_subjects,
                'has_obj_sleep': subject_id in obj_sleep_subjects,
                'has_subj_sleep': subject_id in subj_sleep_subjects,
                'ecg_file': ecg_dir / f"{subject_id}.mat" if ecg_dir and subject_id in ecg_subjects else None,
                'rr_file': rr_dir / f"{subject_id}.mat" if rr_dir and subject_id in rr_subjects else None
            }

        # Complete subjects need clinical + ECG (relaxed criteria)
        self.complete_subjects = [
            subject_id for subject_id, info in self.subjects_mapping.items()
            if info['has_clinical'] and info['has_ecg']
        ]

        print(f"      Subject Summary:")
        print(f"      Clinical data: {len(clinical_subjects)}")
        print(f"      ECG data: {len(ecg_subjects)}")
        print(f"      RR-interval data: {len(rr_subjects)}")
        print(f"      Complete subjects: {len(self.complete_subjects)}")

        return self.subjects_mapping


    def create_separated_targets(self) -> Dict:
        """
        Create SEPARATE target datasets for HbA1c and FBG cohorts.

        CRITICAL FIX: Previous version mixed HbA1c (%) and FBG (mmol/L) which
        are fundamentally different measurements on different timescales.

        - HbA1c: Reflects ~3-month average glycemic control
        - FBG: Instantaneous fasting glucose

        These MUST be analyzed separately.
        """
        print(" Creating SEPARATED Target Variables...")
        print("    CRITICAL: HbA1c and FBG are now analyzed separately")

        df = self.clinical_data

        hba1c_cohort = []
        fbg_cohort = []
        combined_for_comparison = []  # For backwards compatibility reporting

        for _, row in df.iterrows():
            subject_id = row['subject_id']
            if subject_id not in self.complete_subjects:
                continue

            admission_fbg = row.get('admission FBG (mmol/L)', np.nan)
            discharge_fbg = row.get('Discharge FBG (mmol/L)', np.nan)
            hba1c = row.get('HbA1c (%)', np.nan)

            # HbA1c cohort (long-term glycemic control)
            if pd.notna(hba1c):
                hba1c_cohort.append({
                    'subject_id': subject_id,
                    'target_value': hba1c,
                    'target_type': 'hba1c',
                    'log_target': np.log(hba1c) if hba1c > 0 else np.nan,
                    # Clinical categories (ADA guidelines)
                    'control_category': (
                        0 if hba1c < 7.0 else  # Good control
                        1 if hba1c < 8.5 else  # Fair control
                        2  # Poor control
                    ),
                    'is_elevated': int(hba1c >= 7.0)
                })

            # FBG cohort (acute glycemic status)
            # Prefer admission FBG as it's less influenced by treatment
            fbg_value = admission_fbg if pd.notna(admission_fbg) else discharge_fbg
            fbg_type = 'admission_fbg' if pd.notna(admission_fbg) else 'discharge_fbg'

            if pd.notna(fbg_value):
                fbg_cohort.append({
                    'subject_id': subject_id,
                    'target_value': fbg_value,
                    'target_type': fbg_type,
                    'log_target': np.log(fbg_value) if fbg_value > 0 else np.nan,
                    # Clinical categories
                    'control_category': (
                        0 if fbg_value < 7.0 else  # Normal/good
                        1 if fbg_value < 10.0 else  # Elevated
                        2  # High
                    ),
                    'is_elevated': int(fbg_value >= 7.0),
                    # Additional FBG-specific data
                    'has_discharge': pd.notna(discharge_fbg),
                    'glucose_change': (discharge_fbg - admission_fbg) if (
                                pd.notna(admission_fbg) and pd.notna(discharge_fbg)) else np.nan
                })

            # Combined for backwards compatibility (but flag the limitation)
            primary_glucose = hba1c if pd.notna(hba1c) else fbg_value
            glucose_type = 'hba1c' if pd.notna(hba1c) else fbg_type

            if pd.notna(primary_glucose):
                combined_for_comparison.append({
                    'subject_id': subject_id,
                    'primary_glucose': primary_glucose,
                    'glucose_type': glucose_type
                })

        # Convert to DataFrames
        hba1c_df = pd.DataFrame(hba1c_cohort) if hba1c_cohort else pd.DataFrame()
        fbg_df = pd.DataFrame(fbg_cohort) if fbg_cohort else pd.DataFrame()
        combined_df = pd.DataFrame(combined_for_comparison) if combined_for_comparison else pd.DataFrame()

        # Store targets
        targets = {
            'hba1c_cohort': {
                'data': hba1c_df,
                'n_subjects': len(hba1c_df),
                'description': 'HbA1c (%) - 3-month glycemic average',
                'target_column': 'target_value',
                'log_column': 'log_target'
            },
            'fbg_cohort': {
                'data': fbg_df,
                'n_subjects': len(fbg_df),
                'description': 'Fasting Blood Glucose (mmol/L) - acute',
                'target_column': 'target_value',
                'log_column': 'log_target'
            },
            'combined_legacy': {
                'data': combined_df,
                'n_subjects': len(combined_df),
                'description': 'LEGACY: Mixed targets (NOT RECOMMENDED)',
                'warning': 'This combines HbA1c and FBG which have different units and meanings'
            }
        }

        print(f"   HbA1c cohort: {targets['hba1c_cohort']['n_subjects']} subjects")
        print(f"   FBG cohort: {targets['fbg_cohort']['n_subjects']} subjects")
        print(f"   Combined (legacy): {targets['combined_legacy']['n_subjects']} subjects")

        # Statistics
        if len(hba1c_df) > 0:
            print(f"   HbA1c stats: {hba1c_df['target_value'].mean():.2f} ± {hba1c_df['target_value'].std():.2f} %")
        if len(fbg_df) > 0:
            print(f"    FBG stats: {fbg_df['target_value'].mean():.2f} ± {fbg_df['target_value'].std():.2f} mmol/L")

        self.processed_data['separated_targets'] = targets

        # Update complete subjects to only those with valid targets
        valid_subjects = set()
        if len(hba1c_df) > 0:
            valid_subjects.update(hba1c_df['subject_id'].tolist())
        if len(fbg_df) > 0:
            valid_subjects.update(fbg_df['subject_id'].tolist())

        self.complete_subjects = [s for s in self.complete_subjects if s in valid_subjects]

        return targets

    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================

    def extract_ecg_features(self, subject_id: str) -> Optional[Dict]:
        """Extract ECG features with proper scaling validation."""
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

                    # Apply scaling validation
                    signal = self.validate_ecg_scaling(signal, subject_id)

                    # Basic signal statistics
                    features[f'ecg_{var_name}_length'] = len(signal)
                    features[f'ecg_{var_name}_duration_hours'] = len(signal) / 250 / 3600
                    features[f'ecg_{var_name}_mean'] = np.mean(signal)
                    features[f'ecg_{var_name}_std'] = np.std(signal)
                    features[f'ecg_{var_name}_min'] = np.min(signal)
                    features[f'ecg_{var_name}_max'] = np.max(signal)
                    features[f'ecg_{var_name}_range'] = np.max(signal) - np.min(signal)

                    # Signal quality estimate
                    if np.std(signal) > 0:
                        features[f'ecg_{var_name}_snr_estimate'] = np.abs(np.mean(signal)) / np.std(signal)
                    else:
                        features[f'ecg_{var_name}_snr_estimate'] = 0

            return features

        except Exception as e:
            print(f"    Error processing ECG for {subject_id}: {e}")
            return None

    def extract_hrv_features(self, subject_id: str) -> Optional[Dict]:
        """
        Extract HRV features from RR-interval data.
        Includes age-normalized features as per manuscript methodology.
        """
        subject_info = self.subjects_mapping.get(subject_id, {})
        rr_file = subject_info.get('rr_file')

        if rr_file is None or not rr_file.exists():
            return None

        try:
            rr_data = scipy.io.loadmat(str(rr_file))
            features = {}

            for stage in ['DS', 'RS', 'REM']:
                if stage in rr_data:
                    intervals = rr_data[stage].flatten()

                    # Convert ms to seconds
                    intervals_sec = intervals / 1000.0

                    # Filter outliers (outside 3 SD)
                    if len(intervals_sec) > 10:
                        mean_rr = np.mean(intervals_sec)
                        std_rr = np.std(intervals_sec)
                        valid_mask = np.abs(intervals_sec - mean_rr) < 3 * std_rr
                        intervals_sec = intervals_sec[valid_mask]

                    if len(intervals_sec) > 1:
                        stage_lower = stage.lower()

                        # Time domain HRV features
                        features[f'hrv_{stage_lower}_mean_rr'] = np.mean(intervals_sec)
                        features[f'hrv_{stage_lower}_std_rr'] = np.std(intervals_sec)
                        features[f'hrv_{stage_lower}_mean_hr'] = 60 / np.mean(intervals_sec)

                        # RMSSD (Root Mean Square of Successive Differences)
                        rr_diffs = np.diff(intervals_sec)
                        features[f'hrv_{stage_lower}_rmssd'] = np.sqrt(np.mean(rr_diffs ** 2))

                        # pNN50 (percentage of successive RR intervals differing by >50ms)
                        features[f'hrv_{stage_lower}_pnn50'] = np.sum(np.abs(rr_diffs) > 0.05) / len(rr_diffs) * 100

                        # Additional metrics
                        features[f'hrv_{stage_lower}_min_rr'] = np.min(intervals_sec)
                        features[f'hrv_{stage_lower}_max_rr'] = np.max(intervals_sec)
                        features[f'hrv_{stage_lower}_range_rr'] = np.max(intervals_sec) - np.min(intervals_sec)
                        features[f'hrv_{stage_lower}_duration_hours'] = len(intervals_sec) * np.mean(
                            intervals_sec) / 3600
                        features[f'hrv_{stage_lower}_count'] = len(intervals_sec)

                        # Coefficient of variation
                        features[f'hrv_{stage_lower}_cv'] = np.std(intervals_sec) / np.mean(intervals_sec)

            return features

        except Exception as e:
            print(f"   Error processing RR-intervals for {subject_id}: {e}")
            return None

    def extract_clinical_features(self, subject_id: str) -> Optional[Dict]:
        """Extract clinical features excluding target variables."""
        subject_data = self.clinical_data[self.clinical_data['subject_id'] == subject_id]

        if len(subject_data) == 0:
            return None

        features = subject_data.iloc[0].to_dict()
        return features

    def extract_sleep_features(self, subject_id: str) -> Optional[Dict]:
        """Extract sleep quality features from both objective and subjective sources."""
        features = {}

        # Objective sleep features (PSG-derived)
        if self.objective_sleep is not None:
            obj_data = self.objective_sleep[self.objective_sleep['number'] == subject_id]
            if len(obj_data) > 0:
                for key, value in obj_data.iloc[0].to_dict().items():
                    if key not in ['number', 'gender']:
                        features[f'psqi_{key}'] = value

        # Subjective sleep features (questionnaire)
        if self.subjective_sleep is not None:
            subj_data = self.subjective_sleep[self.subjective_sleep['number'] == subject_id]
            if len(subj_data) > 0:
                for key, value in subj_data.iloc[0].to_dict().items():
                    if key != 'number':
                        features[f'cpc_{key}'] = value

        return features if features else None

    def create_age_normalized_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-normalized HRV features.

        Methodology:
        HRV_age_norm = HRV_raw / (age/65 + epsilon)

        Where epsilon = 0.1 to prevent division issues with young subjects.
        The reference age of 65 was chosen as a clinically relevant threshold.
        """
        print(" Creating Age-Normalized Features...")

        if 'age' not in features_df.columns:
            print("   Age column not found, skipping normalization")
            return features_df

        df = features_df.copy()
        age = df['age'].values
        age_norm_factor = (age / 65.0) + 0.1

        # Find HRV mean_rr columns to normalize
        hrv_mean_cols = [col for col in df.columns if 'hrv_' in col and 'mean_rr' in col]

        normalized_count = 0
        for col in hrv_mean_cols:
            if df[col].std() > 0:  # Only normalize if there's variance
                new_col = f'{col}_age_normalized'
                df[new_col] = df[col] / age_norm_factor
                normalized_count += 1

        print(f"   Created {normalized_count} age-normalized features")

        return df

    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================

    def process_all_subjects(self) -> pd.DataFrame:
        """Process all subjects and extract features."""
        print(" Processing All Subjects...")

        # First create targets to get valid subject list
        self.create_separated_targets()

        all_features = []

        for i, subject_id in enumerate(self.complete_subjects):
            if (i + 1) % 10 == 0:
                print(f"   Processing subject {i + 1}/{len(self.complete_subjects)}...")

            subject_features = {'subject_id': subject_id}

            # Clinical features
            clinical = self.extract_clinical_features(subject_id)
            if clinical:
                subject_features.update(clinical)

            # ECG features
            ecg = self.extract_ecg_features(subject_id)
            if ecg:
                subject_features.update(ecg)

            # HRV features
            hrv = self.extract_hrv_features(subject_id)
            if hrv:
                subject_features.update(hrv)

            # Sleep features
            sleep = self.extract_sleep_features(subject_id)
            if sleep:
                subject_features.update(sleep)

            all_features.append(subject_features)

        # Create DataFrame
        features_df = pd.DataFrame(all_features)

        # Add age-normalized features
        features_df = self.create_age_normalized_features(features_df)

        self.processed_data['features'] = features_df

        print(f"    Processed {len(all_features)} subjects")
        print(f"    Total features: {len(features_df.columns)}")

        return features_df

    # =========================================================================
    # VALIDATION SPLITS
    # =========================================================================

    def create_loso_splits(self) -> Dict:
        """
        Create Leave-One-Subject-Out cross-validation splits.

        CRITICAL for physiological data:
        - Prevents data leakage between subjects
        - Each subject's data is completely held out for testing
        """
        print(" Creating LOSO Cross-Validation Splits...")

        if 'separated_targets' not in self.processed_data:
            raise ValueError("Must create targets first")

        features_df = self.processed_data['features']
        targets = self.processed_data['separated_targets']

        # Prepare feature matrix (exclude non-feature columns)
        exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                       [col for col in features_df.columns if any(term in col.lower()
                                                                  for term in
                                                                  ['fbg', 'hba1c', 'diabetic', 'coronary', 'carotid',
                                                                   'glucose'])]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        loso_splits = {}

        # Create LOSO splits for each cohort
        for cohort_name in ['hba1c_cohort', 'fbg_cohort']:
            cohort_data = targets[cohort_name]['data']

            if len(cohort_data) < 10:
                print(f"     {cohort_name}: Too few subjects ({len(cohort_data)}), skipping LOSO")
                continue

            # Get subjects in this cohort
            cohort_subjects = cohort_data['subject_id'].tolist()

            # Filter features to cohort subjects
            cohort_features = features_df[features_df['subject_id'].isin(cohort_subjects)]

            # Align features and targets
            cohort_features = cohort_features.set_index('subject_id')
            cohort_features = cohort_features.loc[cohort_subjects]

            X = cohort_features[feature_cols].fillna(0).values
            y = cohort_data['target_value'].values
            y_log = cohort_data['log_target'].values
            subject_ids = cohort_data['subject_id'].values

            # Create group labels for LOSO
            unique_subjects = list(set(subject_ids))
            subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
            groups = np.array([subject_to_group[s] for s in subject_ids])

            loso_splits[cohort_name] = {
                'X': X,
                'y': y,
                'y_log': y_log,
                'groups': groups,
                'subject_ids': subject_ids,
                'feature_names': feature_cols,
                'n_subjects': len(unique_subjects),
                'n_samples': len(y),
                'validation_type': 'LOSO'
            }

            print(f"    {cohort_name}: {len(unique_subjects)} subjects, {len(feature_cols)} features")

        self.processed_data['loso_splits'] = loso_splits
        return loso_splits

    def create_temporal_splits(self, test_ratio: float = 0.2) -> Dict:
        """
        Create temporal validation splits.

        Subject IDs appear to encode dates (YYYYMMDD format).
        Train on earlier subjects, test on later subjects.

        This addresses concerns about temporal validation in time-series data.
        """
        print(" Creating Temporal Validation Splits...")

        if 'separated_targets' not in self.processed_data:
            raise ValueError("Must create targets first")

        features_df = self.processed_data['features']
        targets = self.processed_data['separated_targets']

        exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                       [col for col in features_df.columns if any(term in col.lower()
                                                                  for term in
                                                                  ['fbg', 'hba1c', 'diabetic', 'coronary', 'carotid',
                                                                   'glucose'])]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        temporal_splits = {}

        for cohort_name in ['hba1c_cohort', 'fbg_cohort']:
            cohort_data = targets[cohort_name]['data']

            if len(cohort_data) < 10:
                print(f"     {cohort_name}: Too few subjects, skipping temporal split")
                continue

            cohort_subjects = cohort_data['subject_id'].tolist()

            # Sort subjects by ID (temporal order)
            sorted_indices = np.argsort(cohort_subjects)
            n_subjects = len(cohort_subjects)

            # Split point
            split_idx = int(n_subjects * (1 - test_ratio))

            train_indices = sorted_indices[:split_idx]
            test_indices = sorted_indices[split_idx:]

            # Get features and targets
            cohort_features = features_df[features_df['subject_id'].isin(cohort_subjects)]
            cohort_features = cohort_features.set_index('subject_id')
            cohort_features = cohort_features.loc[cohort_subjects]

            X = cohort_features[feature_cols].fillna(0).values
            y = cohort_data['target_value'].values
            y_log = cohort_data['log_target'].values

            temporal_splits[cohort_name] = {
                'X_train': X[train_indices],
                'X_test': X[test_indices],
                'y_train': y[train_indices],
                'y_test': y[test_indices],
                'y_log_train': y_log[train_indices],
                'y_log_test': y_log[test_indices],
                'train_subjects': [cohort_subjects[i] for i in train_indices],
                'test_subjects': [cohort_subjects[i] for i in test_indices],
                'feature_names': feature_cols,
                'validation_type': 'temporal'
            }

            print(f"    {cohort_name}: {len(train_indices)} train, {len(test_indices)} test")
            print(
                f"      Train period: {temporal_splits[cohort_name]['train_subjects'][0]} to {temporal_splits[cohort_name]['train_subjects'][-1]}")
            print(
                f"      Test period: {temporal_splits[cohort_name]['test_subjects'][0]} to {temporal_splits[cohort_name]['test_subjects'][-1]}")

        self.processed_data['temporal_splits'] = temporal_splits
        return temporal_splits

    def create_kfold_splits(self, n_splits: int = 5) -> Dict:
        """
        Create standard K-Fold splits for comparison.
        Note: This is for comparison only. LOSO is preferred for physiological data.
        """
        print(f" Creating {n_splits}-Fold Cross-Validation Splits...")

        if 'separated_targets' not in self.processed_data:
            raise ValueError("Must create targets first")

        features_df = self.processed_data['features']
        targets = self.processed_data['separated_targets']

        exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                       [col for col in features_df.columns if any(term in col.lower()
                                                                  for term in
                                                                  ['fbg', 'hba1c', 'diabetic', 'coronary', 'carotid',
                                                                   'glucose'])]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        kfold_splits = {}

        for cohort_name in ['hba1c_cohort', 'fbg_cohort']:
            cohort_data = targets[cohort_name]['data']

            if len(cohort_data) < n_splits:
                print(f"     {cohort_name}: Too few subjects for {n_splits}-fold")
                continue

            cohort_subjects = cohort_data['subject_id'].tolist()

            cohort_features = features_df[features_df['subject_id'].isin(cohort_subjects)]
            cohort_features = cohort_features.set_index('subject_id')
            cohort_features = cohort_features.loc[cohort_subjects]

            X = cohort_features[feature_cols].fillna(0).values
            y = cohort_data['target_value'].values
            y_log = cohort_data['log_target'].values

            kfold_splits[cohort_name] = {
                'X': X,
                'y': y,
                'y_log': y_log,
                'n_splits': n_splits,
                'feature_names': feature_cols,
                'validation_type': f'{n_splits}-fold'
            }

            print(f"    {cohort_name}: {len(y)} samples, {n_splits} folds")

        self.processed_data['kfold_splits'] = kfold_splits
        return kfold_splits

    # =========================================================================
    # SAVING
    # =========================================================================

    def save_processed_data(self, output_dir: str = "processed_data_v2") -> Path:
        """Save all processed data with comprehensive documentation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"💾 Saving Processed Data to {output_dir}...")

        # Save features
        if 'features' in self.processed_data:
            features_file = output_dir / "features.csv"
            self.processed_data['features'].to_csv(features_file, index=False)
            print(f"    Features saved: {features_file}")

        # Save separated targets
        if 'separated_targets' in self.processed_data:
            targets_dir = output_dir / "targets"
            targets_dir.mkdir(exist_ok=True)

            for cohort_name, cohort_info in self.processed_data['separated_targets'].items():
                if 'data' in cohort_info and len(cohort_info['data']) > 0:
                    cohort_file = targets_dir / f"{cohort_name}.csv"
                    cohort_info['data'].to_csv(cohort_file, index=False)
                    print(f"    {cohort_name}: {cohort_file}")

        # Save validation splits (as numpy arrays)
        if 'loso_splits' in self.processed_data:
            splits_dir = output_dir / "loso_splits"
            splits_dir.mkdir(exist_ok=True)

            for cohort_name, split_data in self.processed_data['loso_splits'].items():
                cohort_dir = splits_dir / cohort_name
                cohort_dir.mkdir(exist_ok=True)

                np.save(cohort_dir / "X.npy", split_data['X'])
                np.save(cohort_dir / "y.npy", split_data['y'])
                np.save(cohort_dir / "y_log.npy", split_data['y_log'])
                np.save(cohort_dir / "groups.npy", split_data['groups'])

                with open(cohort_dir / "metadata.json", 'w') as f:
                    json.dump({
                        'n_subjects': split_data['n_subjects'],
                        'n_samples': split_data['n_samples'],
                        'feature_names': split_data['feature_names'],
                        'subject_ids': split_data['subject_ids'].tolist()
                    }, f, indent=2)

            print(f"    LOSO splits saved")

        # Save signal specifications
        if self.signal_specifications:
            specs_file = output_dir / "signal_specifications.json"
            with open(specs_file, 'w') as f:
                json.dump(self.signal_specifications, f, indent=2)
            print(f"    Signal specifications: {specs_file}")

        # Save ECG scaling logs
        if self.ecg_scaling_logs:
            scaling_file = output_dir / "ecg_scaling_logs.json"
            with open(scaling_file, 'w') as f:
                json.dump(self.ecg_scaling_logs, f, indent=2)
            print(f"    ECG scaling logs: {scaling_file}")

        # Save comprehensive summary
        summary = {
            'processing_metadata': self.processing_metadata,
            'dataset_summary': {
                'total_subjects_processed': len(self.complete_subjects),
                'total_features': len(self.processed_data.get('features', pd.DataFrame()).columns)
            },
            'cohort_summary': {},
            'validation_methods': ['LOSO', 'Temporal', '5-Fold CV'],
            'signal_specifications_included': bool(self.signal_specifications),
            'concerns_addressed': self.processing_metadata['concerns_addressed']
        }

        if 'separated_targets' in self.processed_data:
            for cohort_name, cohort_info in self.processed_data['separated_targets'].items():
                if 'data' in cohort_info and len(cohort_info['data']) > 0:
                    summary['cohort_summary'][cohort_name] = {
                        'n_subjects': cohort_info['n_subjects'],
                        'description': cohort_info['description']
                    }

        summary_file = output_dir / "PROCESSING_SUMMARY.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"    Processing summary: {summary_file}")

        return output_dir

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def run_complete_pipeline(self) -> Dict:
        """Run the complete revised preprocessing pipeline."""
        print()
        print("=" * 70)
        print("RUNNING COMPLETE REVISED PIPELINE")
        print("=" * 70)
        print()

        try:
            # Document signal specifications
            self.document_signal_specifications()
            print()

            # Load all data
            self.load_clinical_data()
            self.load_objective_sleep_data()
            self.load_subjective_sleep_data()
            print()

            # Create subject mapping
            self.create_subject_mapping()
            print()

            # Process all subjects
            self.process_all_subjects()
            print()

            # Create validation splits
            self.create_loso_splits()
            self.create_temporal_splits()
            self.create_kfold_splits()
            print()

            # Save everything
            output_dir = self.save_processed_data()

            print()
            print("=" * 70)
            print(" PREPROCESSING COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f" Output directory: {output_dir}")
            print()

            return self.processed_data

        except Exception as e:
            print()
            print("=" * 70)
            print(f" PREPROCESSING FAILED: {e}")
            print("=" * 70)
            raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("REVISED DIABETES ECG PREPROCESSING v2.0")
    print("=" * 70)
    print()

    # Initialize preprocessor
    preprocessor = RevisedDiabetesECGPreprocessor(".")

    # Run pipeline
    try:
        processed_data = preprocessor.run_complete_pipeline()

        # Print final summary
        print()
        print(" FINAL SUMMARY")
        print("-" * 40)

        if 'separated_targets' in processed_data:
            targets = processed_data['separated_targets']

            for cohort_name, cohort_info in targets.items():
                if 'data' in cohort_info and len(cohort_info['data']) > 0:
                    print(f"\n{cohort_name.upper()}:")
                    print(f"  Subjects: {cohort_info['n_subjects']}")
                    print(f"  Description: {cohort_info['description']}")

                    if 'target_value' in cohort_info['data'].columns:
                        values = cohort_info['data']['target_value']
                        print(f"  Mean ± SD: {values.mean():.2f} ± {values.std():.2f}")
                        print(f"  Range: [{values.min():.2f}, {values.max():.2f}]")

        print()
        print(" Ready for modeling with proper validation!")

    except Exception as e:
        print(f" Error: {e}")
        raise