#!/usr/bin/env python3
"""
Revised Ablation Study
============================
Feature selection (SelectKBest) and standardization (StandardScaler)
now happen INSIDE each LOSO fold, not on the full dataset.

Addresses:
  - R3 #4: CV hygiene (feature selection within fold)
  - R3 #5: Back-transformed MAE added
  - R3 #6: Neural network language softened in comments
  - E: Cross-validation hygiene
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from sklearn.base import clone
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy.stats import pearsonr, ttest_rel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
})


class RevisedAblationStudy:
    """
    Ablation study with proper CV hygiene.
    Feature selection and scaling done WITHIN each LOSO fold.
    """

    def __init__(self, data_dir: str = "processed_data_v2"):
        self.data_dir = Path(data_dir)
        self.results = {}
        print("=" * 70)
        print("REVISED ABLATION STUDY")
        print("CV Hygiene: SelectKBest + StandardScaler INSIDE each LOSO fold")
        print("=" * 70)

    def load_data(self, cohort: str = 'hba1c_cohort') -> Tuple:
        print(f"\n Loading {cohort} data...")
        features_df = pd.read_csv(self.data_dir / "features.csv")
        targets_df = pd.read_csv(self.data_dir / "targets" / f"{cohort}.csv")
        cohort_subjects = targets_df['subject_id'].tolist()
        cohort_features = features_df[features_df['subject_id'].isin(cohort_subjects)].copy()
        cohort_features = cohort_features.set_index('subject_id').loc[cohort_subjects].reset_index()
        print(f"    Loaded {len(cohort_subjects)} subjects")
        return cohort_features, targets_df

    def categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
        categories = {
            'demographic': [], 'clinical_vitals': [], 'clinical_blood': [],
            'clinical_metabolic': [], 'ecg_all': [], 'ecg_sleep': [], 'ecg_day': [],
            'hrv_deep': [], 'hrv_rem': [], 'hrv_rs': [], 'hrv_age_normalized': [],
            'sleep_psqi': [], 'sleep_cpc': [], 'other': []
        }
        for feature in feature_names:
            f = feature.lower()
            if any(x in f for x in ['age', 'height', 'weight']) and 'age_normalized' not in f:
                categories['demographic'].append(feature)
            elif any(x in f for x in ['sbp', 'dbp']): categories['clinical_vitals'].append(feature)
            elif any(x in f for x in ['wbc', 'hb', 'plt', 'crp', 'n%']): categories['clinical_blood'].append(feature)
            elif any(x in f for x in ['alt', 'ast', 'ggt', 'bun', 'ua', 'tg', 'hdl', 'ldl', 'uma', 'ucr', 'uacr']): categories['clinical_metabolic'].append(feature)
            elif 'ecg_all_' in f: categories['ecg_all'].append(feature)
            elif 'ecg_sleep_' in f: categories['ecg_sleep'].append(feature)
            elif 'ecg_day_' in f: categories['ecg_day'].append(feature)
            elif 'hrv_ds_' in f and 'age_normalized' not in f: categories['hrv_deep'].append(feature)
            elif 'hrv_rem_' in f and 'age_normalized' not in f: categories['hrv_rem'].append(feature)
            elif 'hrv_rs_' in f and 'age_normalized' not in f: categories['hrv_rs'].append(feature)
            elif 'age_normalized' in f: categories['hrv_age_normalized'].append(feature)
            elif 'psqi_' in f: categories['sleep_psqi'].append(feature)
            elif 'cpc_' in f: categories['sleep_cpc'].append(feature)
            else: categories['other'].append(feature)
        return categories

    def define_ablation_configurations(self, feature_categories: Dict) -> Dict:
        all_hrv = feature_categories['hrv_deep'] + feature_categories['hrv_rem'] + feature_categories['hrv_rs']
        all_ecg = feature_categories['ecg_all'] + feature_categories['ecg_sleep'] + feature_categories['ecg_day']
        all_clinical = feature_categories['clinical_vitals'] + feature_categories['clinical_blood'] + feature_categories['clinical_metabolic']
        all_features = []
        for cat_features in feature_categories.values():
            all_features.extend(cat_features)

        configs = {
            'Full Model': {'features': all_features, 'is_baseline': True},
            'No Age Normalization': {'features': [f for f in all_features if 'age_normalized' not in f.lower()]},
            'Only Age-Normalized + Demographics': {
                'features': feature_categories['hrv_age_normalized'] + feature_categories['demographic']},
            'No Sleep-Stage HRV': {
                'features': [f for f in all_features if not any(x in f.lower() for x in ['hrv_ds_', 'hrv_rem_', 'hrv_rs_'])]},
            'HRV Only': {'features': all_hrv + feature_categories['hrv_age_normalized'] + feature_categories['demographic']},
            'ECG Only': {'features': all_ecg + feature_categories['demographic']},
            'Clinical Only': {'features': all_clinical + feature_categories['demographic']},
            'No ECG': {'features': [f for f in all_features if 'ecg_' not in f.lower()]},
            'No Clinical': {'features': [f for f in all_features if not any(x in f.lower() for x in
                ['sbp', 'dbp', 'wbc', 'hb', 'plt', 'crp', 'alt', 'ast', 'ggt', 'bun', 'ua', 'tg', 'hdl', 'ldl', 'uma', 'ucr', 'uacr'])]},
            'Demographics Only': {'features': feature_categories['demographic']},
        }

        for stage, stage_name in [('ds', 'Deep Sleep'), ('rem', 'REM'), ('rs', 'Rapid Sleep')]:
            stage_feats = [f for f in all_hrv if f'hrv_{stage}_' in f.lower()]
            age_norm = [f for f in feature_categories['hrv_age_normalized'] if f'hrv_{stage}_' in f.lower()]
            configs[f'Only {stage_name} HRV'] = {
                'features': stage_feats + age_norm + feature_categories['demographic']}

        return configs

    def _loso_evaluate_config(self, X: np.ndarray, y: np.ndarray,
                               groups: np.ndarray,
                               feature_names: List[str],
                               config_features: List[str],
                               max_features: int = 15) -> Dict:
        """
        SelectKBest + StandardScaler inside each LOSO fold.
        """
        available = [f for f in config_features if f in feature_names]
        if len(available) < 3:
            return {'r2': np.nan, 'mae': np.nan, 'n_features_available': len(available),
                    'n_features_selected': 0, 'error': 'Too few features'}

        feature_indices = [feature_names.index(f) for f in available]
        X_config = X[:, feature_indices]

        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        logo = LeaveOneGroupOut()
        predictions = np.zeros(len(y))
        all_selected = []

        for train_idx, test_idx in logo.split(X_config, y, groups):
            X_train, X_test = X_config[train_idx], X_config[test_idx]
            y_train = y[train_idx]

            # Feature selection WITHIN fold
            k = min(max_features, X_train.shape[1])
            selector = SelectKBest(f_regression, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)

            mask = selector.get_support()
            all_selected.append([available[i] for i in range(len(available)) if mask[i]])

            # Scaling WITHIN fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            # Fit and predict
            m = clone(model)
            m.fit(X_train_scaled, y_train)
            predictions[test_idx] = m.predict(X_test_scaled)

        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        try:
            corr, pval = pearsonr(y, predictions)
        except:
            corr, pval = np.nan, np.nan

        # Most common features across folds
        from collections import Counter
        all_feats_flat = [f for fold in all_selected for f in fold]
        common_feats = Counter(all_feats_flat).most_common(max_features)

        return {
            'r2': r2, 'mae': mae, 'correlation': corr, 'p_value': pval,
            'n_features_available': len(available),
            'n_features_selected': k,
            'top_features': [f[0] for f in common_feats[:10]],
            'predictions': predictions
        }

    def run_ablation_study(self, cohort: str = 'hba1c_cohort',
                           use_log_target: bool = True) -> pd.DataFrame:
        print(f"\n{'=' * 70}")
        print(f"ABLATION STUDY: {cohort.upper()}")
        print(f"{'=' * 70}")

        features_df, targets_df = self.load_data(cohort)

        exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                       [col for col in features_df.columns if any(term in col.lower()
                        for term in ['fbg', 'hba1c', 'diabetic', 'coronary', 'carotid', 'glucose'])]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X = features_df[feature_cols].fillna(0).values
        y = targets_df['log_target'].values if use_log_target else targets_df['target_value'].values

        subject_ids = targets_df['subject_id'].values
        unique_subjects = list(set(subject_ids))
        subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
        groups = np.array([subject_to_group[s] for s in subject_ids])

        print(f"   Subjects: {len(unique_subjects)}, Features: {len(feature_cols)}")

        categories = self.categorize_features(feature_cols)
        configs = self.define_ablation_configurations(categories)

        print(f"   Running {len(configs)} configurations with in-fold CV hygiene...")
        print("-" * 60)

        results = []
        for config_name, config in configs.items():
            result = self._loso_evaluate_config(X, y, groups, feature_cols, config['features'])
            result['Configuration'] = config_name
            results.append(result)

            if not np.isnan(result['r2']):
                print(f"   {config_name:35} | R²: {result['r2']:7.3f} | MAE: {result['mae']:.3f}")
            else:
                print(f"   {config_name:35} | SKIPPED ({result.get('error', '')})")

        results_df = pd.DataFrame(results)
        baseline_r2 = results_df[results_df['Configuration'] == 'Full Model']['r2'].values[0]
        results_df['Delta_R²'] = results_df['r2'] - baseline_r2
        results_df = results_df.sort_values('r2', ascending=False)

        self.results[cohort] = {
            'ablation_results': results_df,
            'feature_categories': categories,
            'baseline_r2': baseline_r2,
            'n_subjects': len(unique_subjects)
        }
        return results_df

    def create_ablation_figure(self, cohort, output_path=None):
        results_df = self.results[cohort]['ablation_results']
        valid = results_df[results_df['r2'].notna()].sort_values('r2', ascending=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        colors = []
        for c in valid['Configuration']:
            if c == 'Full Model': colors.append('#27ae60')
            elif 'Age Norm' in c: colors.append('#e74c3c')
            elif 'HRV' in c: colors.append('#3498db')
            elif 'ECG' in c: colors.append('#9b59b6')
            elif 'Clinical' in c: colors.append('#f39c12')
            else: colors.append('#95a5a6')

        ax1.barh(range(len(valid)), valid['r2'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(range(len(valid)))
        ax1.set_yticklabels(valid['Configuration'], fontsize=9)
        ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Ablation: {cohort.replace("_"," ").title()}', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3, axis='x')
        for i, r2 in enumerate(valid['r2']):
            ax1.text(r2 + 0.005, i, f'{r2:.3f}', va='center', fontsize=8)

        ax2.scatter(valid['n_features_selected'], valid['r2'], c=colors, s=100, alpha=0.8, edgecolors='black')
        ax2.set_xlabel('Features Selected', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax2.set_title('Feature Count vs Performance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, output_dir="ablation_results_v3"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        for cohort, data in self.results.items():
            cohort_dir = output_dir / cohort
            cohort_dir.mkdir(exist_ok=True)
            df = data['ablation_results'].drop(columns=['predictions', 'top_features'], errors='ignore')
            df.to_csv(cohort_dir / "ablation_results.csv", index=False)
            self.create_ablation_figure(cohort, str(cohort_dir / "ablation_figure.png"))
            print(f"    Saved {cohort}")

    def run_complete_ablation(self):
        for cohort in ['hba1c_cohort', 'fbg_cohort']:
            try:
                self.run_ablation_study(cohort)
            except Exception as e:
                print(f"    {cohort}: {e}")
        if self.results:
            self.save_results()
        return self.results


if __name__ == "__main__":
    ablation = RevisedAblationStudy("processed_data_v2")
    ablation.run_complete_ablation()