#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

from scipy.stats import pearsonr, ttest_rel, wilcoxon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Plot settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class RevisedAblationStudy:
    """
    Comprehensive ablation study with separated cohort analysis.

    Tests contribution of:
    1. Age normalization
    2. Sleep-stage-specific HRV features
    3. ECG features
    4. Clinical features
    5. Feature combinations
    """

    def __init__(self, data_dir: str = "processed_data_v2"):
        self.data_dir = Path(data_dir)
        self.results = {}

        print("=" * 70)
        print("REVISED ABLATION STUDY v2.0")
        print("Separated Cohort Analysis")
        print("=" * 70)

    def load_data(self, cohort: str = 'hba1c_cohort') -> Tuple:
        """Load data for a specific cohort."""
        print(f"\n📁 Loading {cohort} data...")

        # Load features
        features_df = pd.read_csv(self.data_dir / "features.csv")

        # Load targets
        targets_path = self.data_dir / "targets" / f"{cohort}.csv"
        if not targets_path.exists():
            raise FileNotFoundError(f"Target file not found: {targets_path}")

        targets_df = pd.read_csv(targets_path)

        # Get cohort subjects
        cohort_subjects = targets_df['subject_id'].tolist()
        cohort_features = features_df[features_df['subject_id'].isin(cohort_subjects)].copy()

        # Align data
        cohort_features = cohort_features.set_index('subject_id')
        cohort_features = cohort_features.loc[cohort_subjects].reset_index()

        print(f"   ✅ Loaded {len(cohort_subjects)} subjects")

        return cohort_features, targets_df

    def categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            'demographic': [],
            'clinical_vitals': [],
            'clinical_blood': [],
            'clinical_metabolic': [],
            'ecg_all': [],
            'ecg_sleep': [],
            'ecg_day': [],
            'hrv_deep': [],
            'hrv_rem': [],
            'hrv_rs': [],
            'hrv_age_normalized': [],
            'sleep_psqi': [],
            'sleep_cpc': [],
            'other': []
        }

        for feature in feature_names:
            f_lower = feature.lower()

            # Demographic
            if any(x in f_lower for x in ['age', 'height', 'weight']) and 'age_normalized' not in f_lower:
                categories['demographic'].append(feature)

            # Clinical - Vitals
            elif any(x in f_lower for x in ['sbp', 'dbp']):
                categories['clinical_vitals'].append(feature)

            # Clinical - Blood
            elif any(x in f_lower for x in ['wbc', 'hb', 'plt', 'crp', 'n%']):
                categories['clinical_blood'].append(feature)

            # Clinical - Metabolic
            elif any(
                    x in f_lower for x in ['alt', 'ast', 'ggt', 'bun', 'ua', 'tg', 'hdl', 'ldl', 'uma', 'ucr', 'uacr']):
                categories['clinical_metabolic'].append(feature)

            # ECG features
            elif 'ecg_all_' in f_lower:
                categories['ecg_all'].append(feature)
            elif 'ecg_sleep_' in f_lower:
                categories['ecg_sleep'].append(feature)
            elif 'ecg_day_' in f_lower:
                categories['ecg_day'].append(feature)

            # HRV features
            elif 'hrv_ds_' in f_lower and 'age_normalized' not in f_lower:
                categories['hrv_deep'].append(feature)
            elif 'hrv_rem_' in f_lower and 'age_normalized' not in f_lower:
                categories['hrv_rem'].append(feature)
            elif 'hrv_rs_' in f_lower and 'age_normalized' not in f_lower:
                categories['hrv_rs'].append(feature)

            # Age-normalized HRV
            elif 'age_normalized' in f_lower:
                categories['hrv_age_normalized'].append(feature)

            # Sleep questionnaires
            elif 'psqi_' in f_lower:
                categories['sleep_psqi'].append(feature)
            elif 'cpc_' in f_lower:
                categories['sleep_cpc'].append(feature)

            else:
                categories['other'].append(feature)

        return categories

    def define_ablation_configurations(self, feature_categories: Dict) -> Dict:
        """Define ablation study configurations."""
        configs = {}

        # All HRV features (all sleep stages)
        all_hrv = (feature_categories['hrv_deep'] +
                   feature_categories['hrv_rem'] +
                   feature_categories['hrv_rs'])

        # All ECG features
        all_ecg = (feature_categories['ecg_all'] +
                   feature_categories['ecg_sleep'] +
                   feature_categories['ecg_day'])

        # All clinical features
        all_clinical = (feature_categories['clinical_vitals'] +
                        feature_categories['clinical_blood'] +
                        feature_categories['clinical_metabolic'])

        # All sleep quality features
        all_sleep_quality = (feature_categories['sleep_psqi'] +
                             feature_categories['sleep_cpc'])

        # Configuration 1: Full model (all features)
        all_features = []
        for cat_features in feature_categories.values():
            all_features.extend(cat_features)
        configs['Full Model'] = {
            'features': all_features,
            'description': 'All available features',
            'is_baseline': True
        }

        # Configuration 2: Remove age-normalized features
        configs['No Age Normalization'] = {
            'features': [f for f in all_features if 'age_normalized' not in f.lower()],
            'description': 'Remove age-normalized HRV features',
            'tests': 'Age normalization contribution'
        }

        # Configuration 3: Only age-normalized features + demographics
        configs['Only Age-Normalized + Demographics'] = {
            'features': feature_categories['hrv_age_normalized'] + feature_categories['demographic'],
            'description': 'Age-normalized HRV + demographics only',
            'tests': 'Core contribution test'
        }

        # Configuration 4: Remove all HRV (sleep-stage specific)
        configs['No Sleep-Stage HRV'] = {
            'features': [f for f in all_features if
                         not any(x in f.lower() for x in ['hrv_ds_', 'hrv_rem_', 'hrv_rs_'])],
            'description': 'Remove sleep-stage-specific HRV',
            'tests': 'Sleep-stage HRV contribution'
        }

        # Configuration 5: HRV only (all HRV features including age-normalized)
        configs['HRV Only'] = {
            'features': all_hrv + feature_categories['hrv_age_normalized'] + feature_categories['demographic'],
            'description': 'Only HRV features + demographics',
            'tests': 'HRV sufficiency'
        }

        # Configuration 6: ECG only
        configs['ECG Only'] = {
            'features': all_ecg + feature_categories['demographic'],
            'description': 'Only ECG signal features + demographics',
            'tests': 'ECG feature contribution'
        }

        # Configuration 7: Clinical only
        configs['Clinical Only'] = {
            'features': all_clinical + feature_categories['demographic'],
            'description': 'Only clinical measurements + demographics',
            'tests': 'Clinical feature contribution'
        }

        # Configuration 8: Remove ECG
        configs['No ECG'] = {
            'features': [f for f in all_features if 'ecg_' not in f.lower()],
            'description': 'Remove all ECG features',
            'tests': 'ECG necessity'
        }

        # Configuration 9: Remove clinical
        configs['No Clinical'] = {
            'features': [f for f in all_features if not any(x in f.lower() for x in
                                                            ['sbp', 'dbp', 'wbc', 'hb', 'plt', 'crp', 'alt', 'ast',
                                                             'ggt',
                                                             'bun', 'ua', 'tg', 'hdl', 'ldl', 'uma', 'ucr', 'uacr'])],
            'description': 'Remove clinical blood/metabolic features',
            'tests': 'Clinical necessity'
        }

        # Configuration 10: Single sleep stage tests
        for stage, stage_name in [('ds', 'Deep Sleep'), ('rem', 'REM'), ('rs', 'Rapid Sleep')]:
            stage_features = [f for f in all_hrv if f'hrv_{stage}_' in f.lower()]
            age_norm_stage = [f for f in feature_categories['hrv_age_normalized'] if f'hrv_{stage}_' in f.lower()]
            configs[f'Only {stage_name} HRV'] = {
                'features': stage_features + age_norm_stage + feature_categories['demographic'],
                'description': f'Only {stage_name} HRV features + demographics',
                'tests': f'{stage_name} HRV contribution'
            }

        # Configuration 11: Demographics only (minimal baseline)
        configs['Demographics Only'] = {
            'features': feature_categories['demographic'],
            'description': 'Only demographic features (age, height, weight)',
            'tests': 'Minimal baseline'
        }

        return configs

    def run_single_configuration(self, X: np.ndarray, y: np.ndarray,
                                 groups: np.ndarray,
                                 feature_names: List[str],
                                 config_features: List[str],
                                 max_features: int = 15) -> Dict:
        """Run evaluation for a single ablation configuration."""
        # Get available features
        available_features = [f for f in config_features if f in feature_names]

        if len(available_features) < 3:
            return {
                'r2': np.nan,
                'mae': np.nan,
                'n_features_available': len(available_features),
                'n_features_selected': 0,
                'error': 'Too few features'
            }

        # Get feature indices
        feature_indices = [feature_names.index(f) for f in available_features]
        X_config = X[:, feature_indices]

        # Feature selection
        k = min(max_features, len(available_features))
        try:
            selector = SelectKBest(f_regression, k=k)
            X_selected = selector.fit_transform(X_config, y)
            selected_mask = selector.get_support()
            selected_features = [available_features[i] for i in range(len(available_features)) if selected_mask[i]]
        except:
            X_selected = X_config
            selected_features = available_features

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # Model and evaluation
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)

        # Use LOSO if groups available, else 5-fold CV
        if groups is not None and len(np.unique(groups)) >= 5:
            cv = LeaveOneGroupOut()
            predictions = cross_val_predict(model, X_scaled, y, cv=cv, groups=groups)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            predictions = cross_val_predict(model, X_scaled, y, cv=cv)

        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        corr, pval = pearsonr(y, predictions)

        return {
            'r2': r2,
            'mae': mae,
            'correlation': corr,
            'p_value': pval,
            'n_features_available': len(available_features),
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'predictions': predictions
        }

    def run_ablation_study(self, cohort: str = 'hba1c_cohort',
                           use_log_target: bool = True) -> pd.DataFrame:
        """Run complete ablation study for a cohort."""
        print(f"\n{'=' * 70}")
        print(f"ABLATION STUDY: {cohort.upper()}")
        print(f"Target: {'Log-transformed' if use_log_target else 'Raw'}")
        print(f"{'=' * 70}")

        # Load data
        features_df, targets_df = self.load_data(cohort)

        # Prepare feature matrix
        exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                       [col for col in features_df.columns if any(term in col.lower()
                                                                  for term in
                                                                  ['fbg', 'hba1c', 'diabetic', 'coronary', 'carotid',
                                                                   'glucose'])]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].fillna(0).values

        # Get target
        if use_log_target:
            y = targets_df['log_target'].values
        else:
            y = targets_df['target_value'].values

        # Create groups for LOSO
        subject_ids = targets_df['subject_id'].values
        unique_subjects = list(set(subject_ids))
        subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
        groups = np.array([subject_to_group[s] for s in subject_ids])

        print(f"\n   Subjects: {len(unique_subjects)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Validation: LOSO ({len(unique_subjects)} folds)")

        # Categorize features
        feature_categories = self.categorize_features(feature_cols)

        print("\n   Feature Categories:")
        for cat, feats in feature_categories.items():
            if len(feats) > 0:
                print(f"      {cat}: {len(feats)} features")

        # Define configurations
        configs = self.define_ablation_configurations(feature_categories)

        print(f"\n   Running {len(configs)} ablation configurations...")
        print("-" * 60)

        results = []
        baseline_predictions = None

        for config_name, config in configs.items():
            result = self.run_single_configuration(
                X, y, groups, feature_cols, config['features']
            )

            result['Configuration'] = config_name
            result['Description'] = config['description']
            results.append(result)

            # Store baseline predictions for statistical comparison
            if config.get('is_baseline'):
                baseline_predictions = result.get('predictions')

            # Print result
            if not np.isnan(result['r2']):
                print(
                    f"   {config_name:35} | R²: {result['r2']:7.3f} | MAE: {result['mae']:.3f} | Features: {result['n_features_selected']}")
            else:
                print(f"   {config_name:35} | SKIPPED ({result.get('error', 'Unknown error')})")

        results_df = pd.DataFrame(results)

        # Calculate improvements relative to baseline
        baseline_r2 = results_df[results_df['Configuration'] == 'Full Model']['r2'].values[0]
        results_df['Improvement_vs_Baseline'] = results_df['r2'] - baseline_r2
        results_df['Improvement_Pct'] = (results_df['Improvement_vs_Baseline'] / abs(baseline_r2)) * 100

        # Statistical comparison with baseline (if predictions available)
        if baseline_predictions is not None:
            significance = []
            for _, row in results_df.iterrows():
                if row['Configuration'] != 'Full Model' and 'predictions' in row and row['predictions'] is not None:
                    try:
                        # Paired t-test on absolute errors
                        baseline_errors = np.abs(baseline_predictions - y)
                        config_errors = np.abs(row['predictions'] - y)
                        _, p_val = ttest_rel(baseline_errors, config_errors)
                        significance.append(p_val)
                    except:
                        significance.append(np.nan)
                else:
                    significance.append(np.nan)
            results_df['p_value_vs_baseline'] = significance

        # Sort by R²
        results_df = results_df.sort_values('r2', ascending=False)

        # Store results
        self.results[cohort] = {
            'ablation_results': results_df,
            'feature_categories': feature_categories,
            'baseline_r2': baseline_r2,
            'n_subjects': len(unique_subjects)
        }

        return results_df

    def analyze_key_findings(self, cohort: str = 'hba1c_cohort') -> Dict:
        """Analyze key findings from ablation study."""
        if cohort not in self.results:
            raise ValueError(f"No results for {cohort}. Run ablation study first.")

        results_df = self.results[cohort]['ablation_results']
        baseline_r2 = self.results[cohort]['baseline_r2']

        print(f"\n{'=' * 70}")
        print(f"KEY FINDINGS: {cohort.upper()}")
        print(f"{'=' * 70}")

        findings = {}

        # 1. Age normalization contribution
        no_age_norm = results_df[results_df['Configuration'] == 'No Age Normalization']['r2'].values
        if len(no_age_norm) > 0:
            age_norm_contribution = baseline_r2 - no_age_norm[0]
            age_norm_pct = (age_norm_contribution / abs(no_age_norm[0])) * 100 if no_age_norm[0] != 0 else 0
            findings['age_normalization'] = {
                'contribution_r2': age_norm_contribution,
                'contribution_pct': age_norm_pct,
                'baseline_r2': baseline_r2,
                'without_r2': no_age_norm[0]
            }
            print(f"\n1. AGE NORMALIZATION CONTRIBUTION:")
            print(f"   With age normalization:    R² = {baseline_r2:.3f}")
            print(f"   Without age normalization: R² = {no_age_norm[0]:.3f}")
            print(f"   Improvement: {age_norm_contribution:+.3f} R² ({age_norm_pct:+.1f}%)")

        # 2. Sleep-stage HRV contribution
        no_sleep_hrv = results_df[results_df['Configuration'] == 'No Sleep-Stage HRV']['r2'].values
        if len(no_sleep_hrv) > 0:
            sleep_hrv_contribution = baseline_r2 - no_sleep_hrv[0]
            findings['sleep_stage_hrv'] = {
                'contribution_r2': sleep_hrv_contribution,
                'without_r2': no_sleep_hrv[0]
            }
            print(f"\n2. SLEEP-STAGE HRV CONTRIBUTION:")
            print(f"   Full model:              R² = {baseline_r2:.3f}")
            print(f"   Without sleep-stage HRV: R² = {no_sleep_hrv[0]:.3f}")
            print(f"   Contribution: {sleep_hrv_contribution:+.3f} R²")

        # 3. Best single-stage performance
        stage_configs = ['Only Deep Sleep HRV', 'Only REM HRV', 'Only Rapid Sleep HRV']
        stage_results = results_df[results_df['Configuration'].isin(stage_configs)]
        if len(stage_results) > 0:
            best_stage = stage_results.loc[stage_results['r2'].idxmax()]
            findings['best_single_stage'] = {
                'name': best_stage['Configuration'],
                'r2': best_stage['r2']
            }
            print(f"\n3. BEST SINGLE SLEEP STAGE:")
            print(f"   {best_stage['Configuration']}: R² = {best_stage['r2']:.3f}")

        # 4. ECG vs HRV comparison
        ecg_only = results_df[results_df['Configuration'] == 'ECG Only']['r2'].values
        hrv_only = results_df[results_df['Configuration'] == 'HRV Only']['r2'].values
        if len(ecg_only) > 0 and len(hrv_only) > 0:
            findings['ecg_vs_hrv'] = {
                'ecg_r2': ecg_only[0],
                'hrv_r2': hrv_only[0],
                'hrv_advantage': hrv_only[0] - ecg_only[0]
            }
            print(f"\n4. ECG vs HRV COMPARISON:")
            print(f"   ECG only: R² = {ecg_only[0]:.3f}")
            print(f"   HRV only: R² = {hrv_only[0]:.3f}")
            print(f"   HRV advantage: {hrv_only[0] - ecg_only[0]:+.3f} R²")

        # 5. Clinical features contribution
        no_clinical = results_df[results_df['Configuration'] == 'No Clinical']['r2'].values
        clinical_only = results_df[results_df['Configuration'] == 'Clinical Only']['r2'].values
        if len(no_clinical) > 0:
            clinical_contribution = baseline_r2 - no_clinical[0]
            findings['clinical_features'] = {
                'contribution_r2': clinical_contribution,
                'clinical_only_r2': clinical_only[0] if len(clinical_only) > 0 else np.nan
            }
            print(f"\n5. CLINICAL FEATURES:")
            print(f"   Clinical contribution: {clinical_contribution:+.3f} R²")
            if len(clinical_only) > 0:
                print(f"   Clinical only: R² = {clinical_only[0]:.3f}")

        # 6. Minimal baseline
        demo_only = results_df[results_df['Configuration'] == 'Demographics Only']['r2'].values
        if len(demo_only) > 0:
            findings['demographics_baseline'] = {
                'r2': demo_only[0],
                'improvement_from_baseline': baseline_r2 - demo_only[0]
            }
            print(f"\n6. DEMOGRAPHICS BASELINE:")
            print(f"   Demographics only: R² = {demo_only[0]:.3f}")
            print(f"   Full model improvement: {baseline_r2 - demo_only[0]:+.3f} R²")

        return findings

    def create_ablation_figure(self, cohort: str = 'hba1c_cohort',
                               output_path: str = None):
        """Create publication-quality ablation study figure."""
        if cohort not in self.results:
            raise ValueError(f"No results for {cohort}")

        results_df = self.results[cohort]['ablation_results']

        # Filter out NaN results and sort
        valid_results = results_df[results_df['r2'].notna()].copy()
        valid_results = valid_results.sort_values('r2', ascending=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Color coding
        colors = []
        for config in valid_results['Configuration']:
            if config == 'Full Model':
                colors.append('#27ae60')  # Green for baseline
            elif 'Age Norm' in config:
                colors.append('#e74c3c')  # Red for age normalization related
            elif 'HRV' in config:
                colors.append('#3498db')  # Blue for HRV
            elif 'ECG' in config:
                colors.append('#9b59b6')  # Purple for ECG
            elif 'Clinical' in config:
                colors.append('#f39c12')  # Orange for clinical
            else:
                colors.append('#95a5a6')  # Gray for others

        # R² bar plot
        bars1 = ax1.barh(range(len(valid_results)), valid_results['r2'],
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax1.set_yticks(range(len(valid_results)))
        ax1.set_yticklabels(valid_results['Configuration'], fontsize=9)
        ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Ablation Study Results\n{cohort.replace("_", " ").title()}',
                      fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, r2) in enumerate(zip(bars1, valid_results['r2'])):
            ax1.text(r2 + 0.005, i, f'{r2:.3f}', va='center', fontsize=8)

        # Feature count vs R² scatter
        ax2.scatter(valid_results['n_features_selected'], valid_results['r2'],
                    c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)

        # Add labels
        for i, (_, row) in enumerate(valid_results.iterrows()):
            # Shorten config names for readability
            short_name = row['Configuration'].replace('Only ', '').replace(' HRV', '')
            if len(short_name) > 15:
                short_name = short_name[:12] + '...'
            ax2.annotate(short_name,
                         (row['n_features_selected'], row['r2']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=7, alpha=0.8)

        ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax2.set_title('Feature Count vs Performance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Legend
        legend_elements = [
            mpatches.Patch(color='#27ae60', label='Full Model'),
            mpatches.Patch(color='#e74c3c', label='Age Normalization'),
            mpatches.Patch(color='#3498db', label='HRV Features'),
            mpatches.Patch(color='#9b59b6', label='ECG Features'),
            mpatches.Patch(color='#f39c12', label='Clinical Features'),
            mpatches.Patch(color='#95a5a6', label='Other')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=6,
                   bbox_to_anchor=(0.5, 0.02), fontsize=9)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: {output_path}")

        plt.close()

    def save_results(self, output_dir: str = "ablation_results"):
        """Save all ablation study results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"\n💾 Saving Results to {output_dir}...")

        for cohort, data in self.results.items():
            cohort_dir = output_dir / cohort
            cohort_dir.mkdir(exist_ok=True)

            # Save results table
            results_df = data['ablation_results'].drop(columns=['predictions', 'selected_features'], errors='ignore')
            results_df.to_csv(cohort_dir / "ablation_results.csv", index=False)

            # Create figure
            self.create_ablation_figure(cohort, str(cohort_dir / "ablation_figure.png"))

            # Save findings
            findings = self.analyze_key_findings(cohort)

            # Convert numpy types to Python types for JSON
            def convert_types(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                return obj

            findings_clean = convert_types(findings)

            with open(cohort_dir / "key_findings.json", 'w') as f:
                json.dump(findings_clean, f, indent=2)

            print(f"   ✅ Saved {cohort} results")

        print(f"\n✅ All results saved to {output_dir}")

    def run_complete_ablation(self):
        """Run ablation study for all available cohorts."""
        print("\n" + "=" * 70)
        print("RUNNING COMPLETE ABLATION STUDY")
        print("=" * 70)

        for cohort in ['hba1c_cohort', 'fbg_cohort']:
            try:
                self.run_ablation_study(cohort, use_log_target=True)
            except FileNotFoundError as e:
                print(f"\n⚠️  {cohort}: Data not found - {e}")
            except Exception as e:
                print(f"\n❌ {cohort}: Error - {e}")

        if self.results:
            self.save_results()

        return self.results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("REVISED ABLATION STUDY v2.0")
    print("=" * 70)
    print()

    # Initialize
    ablation = RevisedAblationStudy("processed_data_v2")

    # Run complete ablation study
    results = ablation.run_complete_ablation()

    if results:
        print()
        print("=" * 70)
        print("ABLATION STUDY COMPLETE")
        print("=" * 70)
    else:
        print("\n⚠️  No results generated. Run preprocessing first.")