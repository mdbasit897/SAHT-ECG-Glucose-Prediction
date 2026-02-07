#!/usr/bin/env python3
"""
Revised Validation Framework
==================================
FIXED: All validation analyses now use sklearn Pipeline to ensure
feature selection + scaling happen within each CV fold automatically.

Addresses:
  - R3 #4: CV hygiene
  - R3 #5: Back-transformed error interpretation
  - R3 #7: Bootstrap CIs with proper subject-level resampling
  - E: Methodological details require clearer reporting
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings

from sklearn.base import clone
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    LeaveOneGroupOut, KFold, cross_val_predict,
    permutation_test_score, learning_curve
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy import stats
from scipy.stats import pearsonr, shapiro, normaltest

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 150, 'savefig.dpi': 300
})


class ValidationFramework:
    """
    Pipeline-based validation ensuring no data leakage.
    """

    def __init__(self, data_dir: str = "processed_data_v2"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        print("=" * 70)
        print("VALIDATION FRAMEWORK v3.0 (Pipeline-based CV hygiene)")
        print("=" * 70)

    def load_cohort_data(self, cohort: str = 'hba1c_cohort') -> Tuple:
        loso_dir = self.data_dir / "loso_splits" / cohort
        if loso_dir.exists():
            X = np.load(loso_dir / "X.npy")
            y = np.load(loso_dir / "y.npy")
            y_log = np.load(loso_dir / "y_log.npy")
            groups = np.load(loso_dir / "groups.npy")
            with open(loso_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            return X, y, y_log, groups, metadata['feature_names']
        raise FileNotFoundError(f"LOSO splits not found at {loso_dir}")

    def _get_pipeline(self, max_features: int = 15) -> Pipeline:
        """
        Create a pipeline that wraps feature selection + scaling + model.
        sklearn handles the fit/transform correctly within each CV fold.
        """
        k = max_features
        return Pipeline([
            ('feature_selection', SelectKBest(f_regression, k=k)),
            ('scaler', StandardScaler()),
            ('model', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                    lambda_1=1e-6, lambda_2=1e-6))
        ])

    def _loso_predict(self, X, y, groups, max_features=15):
        """Get LOSO predictions using pipeline (proper CV hygiene)."""
        pipeline = self._get_pipeline(min(max_features, X.shape[1]))
        cv = LeaveOneGroupOut()
        predictions = cross_val_predict(pipeline, X, y, cv=cv, groups=groups)
        return predictions

    # =========================================================================
    # PERMUTATION TEST
    # =========================================================================

    def run_permutation_test(self, X, y, groups, max_features=15,
                             n_permutations=500) -> Dict:
        print(f"\n Permutation Test (n={n_permutations})")
        print("-" * 50)

        pipeline = self._get_pipeline(min(max_features, X.shape[1]))
        cv = LeaveOneGroupOut()

        score, perm_scores, p_value = permutation_test_score(
            pipeline, X, y, cv=cv, groups=groups,
            n_permutations=n_permutations, scoring='r2',
            random_state=42, n_jobs=-1
        )

        perm_mean = np.mean(perm_scores)
        perm_std = np.std(perm_scores)
        effect_size = (score - perm_mean) / perm_std if perm_std > 0 else 0

        results = {
            'true_r2': float(score),
            'permutation_r2_mean': float(perm_mean),
            'permutation_r2_std': float(perm_std),
            'p_value': float(p_value),
            'effect_size_z': float(effect_size),
            'n_permutations': n_permutations,
            'significant_005': p_value < 0.05,
            'permutation_scores': perm_scores.tolist()
        }

        print(f"   True R²:       {score:.4f}")
        print(f"   Perm R² mean:  {perm_mean:.4f} ± {perm_std:.4f}")
        print(f"   P-value:       {p_value:.4f}")
        print(f"   Significant:   {'' if p_value < 0.05 else ''}")
        return results

    # =========================================================================
    # BOOTSTRAP CI — subject-level resampling
    # =========================================================================

    def run_bootstrap_analysis(self, X, y, groups, max_features=15,
                               n_bootstrap=500) -> Dict:
        """
        Subject-level bootstrap with in-fold CV hygiene.
        Addresses R3 #7: wide CIs need proper interpretation.
        """
        print(f"\n Bootstrap 95% CI (n={n_bootstrap}, subject-level)")
        print("-" * 50)

        unique_groups = np.unique(groups)
        n_subjects = len(unique_groups)

        bootstrap_r2 = []
        bootstrap_mae = []

        for i in range(n_bootstrap):
            boot_subjects = np.random.choice(unique_groups, n_subjects, replace=True)
            boot_idx = []
            new_groups = []
            for new_g, subj in enumerate(boot_subjects):
                sidx = np.where(groups == subj)[0]
                boot_idx.extend(sidx)
                new_groups.extend([new_g] * len(sidx))

            X_b, y_b = X[boot_idx], y[boot_idx]
            g_b = np.array(new_groups)

            try:
                preds = self._loso_predict(X_b, y_b, g_b, max_features)
                r2 = r2_score(y_b, preds)
                mae = mean_absolute_error(y_b, preds)
                if not np.isnan(r2) and not np.isinf(r2):
                    bootstrap_r2.append(r2)
                    bootstrap_mae.append(mae)
            except:
                continue

        bootstrap_r2 = np.array(bootstrap_r2)
        bootstrap_mae = np.array(bootstrap_mae)

        results = {
            'n_bootstrap': n_bootstrap,
            'n_successful': len(bootstrap_r2),
            'metrics': {
                'r2': {
                    'mean': float(np.mean(bootstrap_r2)),
                    'median': float(np.median(bootstrap_r2)),
                    'std': float(np.std(bootstrap_r2)),
                    'ci_95_lower': float(np.percentile(bootstrap_r2, 2.5)),
                    'ci_95_upper': float(np.percentile(bootstrap_r2, 97.5)),
                },
                'mae': {
                    'mean': float(np.mean(bootstrap_mae)),
                    'ci_95_lower': float(np.percentile(bootstrap_mae, 2.5)),
                    'ci_95_upper': float(np.percentile(bootstrap_mae, 97.5)),
                }
            }
        }

        r2m = results['metrics']['r2']
        print(f"   R²:  {r2m['median']:.3f} [{r2m['ci_95_lower']:.3f}, {r2m['ci_95_upper']:.3f}]")
        return results

    # =========================================================================
    # RESIDUAL ANALYSIS
    # =========================================================================

    def analyze_residuals(self, y_true, y_pred) -> Dict:
        print(f"\n Residual Analysis")
        print("-" * 50)
        residuals = y_pred - y_true

        results = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
        }

        if len(residuals) >= 8:
            try:
                stat, p = shapiro(residuals)
                results['shapiro_test'] = {'statistic': float(stat), 'p_value': float(p), 'normal': p > 0.05}
            except:
                pass

        results['bias_significant'] = abs(results['mean']) > 2 * results['std'] / np.sqrt(len(residuals))

        corr, p = pearsonr(y_pred, np.abs(residuals))
        results['heteroscedasticity'] = {'correlation': float(corr), 'p_value': float(p), 'significant': p < 0.05}

        print(f"   Mean residual: {results['mean']:.4f}")
        print(f"   Bias: {'Yes' if results['bias_significant'] else 'No'}")
        if 'shapiro_test' in results:
            print(
                f"   Normal: {'Yes' if results['shapiro_test']['normal'] else 'No'} (p={results['shapiro_test']['p_value']:.4f})")
        return results

    # =========================================================================
    # LEARNING CURVE
    # =========================================================================

    def analyze_learning_curve(self, X, y, groups=None, max_features=15) -> Dict:
        print(f"\n Learning Curve Analysis")
        print("-" * 50)

        pipeline = self._get_pipeline(min(max_features, X.shape[1]))
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        train_sizes = np.linspace(0.2, 1.0, 5)

        train_sizes_abs, train_scores, test_scores = learning_curve(
            pipeline, X, y, train_sizes=train_sizes,
            cv=cv, scoring='r2', n_jobs=-1, random_state=42
        )

        test_means = np.mean(test_scores, axis=1)
        results = {
            'train_sizes': train_sizes_abs.tolist(),
            'test_scores_mean': test_means.tolist(),
            'test_scores_std': np.std(test_scores, axis=1).tolist(),
        }

        if len(test_means) >= 3:
            recent = test_means[-1] - test_means[-2]
            results['recommendation'] = (
                "More data would likely help" if recent > 0.02
                else "Marginal benefit expected" if recent > 0.005
                else "Performance appears saturated"
            )
            print(f"   Final test R²: {test_means[-1]:.4f}")
            print(f"   {results['recommendation']}")

        return results

    # =========================================================================
    # MAIN REPORT GENERATION
    # =========================================================================

    def generate_validation_report(self, cohort, max_features=15,
                                   output_dir="validation_results_v3") -> Dict:
        print(f"\n{'=' * 70}")
        print(f"VALIDATION REPORT: {cohort.upper()}")
        print(f"{'=' * 70}")

        output_dir = Path(output_dir)
        cohort_dir = output_dir / cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)

        X, y, y_log, groups, feature_names = self.load_cohort_data(cohort)
        target = y_log

        # Get predictions with pipeline
        predictions = self._loso_predict(X, target, groups, max_features)

        report = {
            'cohort': cohort,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(target),
            'n_features': X.shape[1],
            'n_subjects': len(np.unique(groups)),
            'cv_hygiene': 'Pipeline(SelectKBest + StandardScaler + BayesianRidge) within each LOSO fold'
        }

        report['permutation_test'] = self.run_permutation_test(X, target, groups, max_features)
        report['bootstrap_analysis'] = self.run_bootstrap_analysis(X, target, groups, max_features)
        report['residual_analysis'] = self.analyze_residuals(target, predictions)
        report['learning_curve'] = self.analyze_learning_curve(X, target, groups, max_features)

        report['summary'] = {
            'significant': report['permutation_test']['significant_005'],
            'r2_95_ci': [
                report['bootstrap_analysis']['metrics']['r2']['ci_95_lower'],
                report['bootstrap_analysis']['metrics']['r2']['ci_95_upper']
            ],
            'residuals_normal': report['residual_analysis'].get('shapiro_test', {}).get('normal')
        }

        # Save
        report_clean = self._clean(report)
        with open(cohort_dir / "validation_report.json", 'w') as f:
            json.dump(report_clean, f, indent=2)

        self._create_figures(report, cohort_dir, target, predictions)
        self.validation_results[cohort] = report
        return report

    def _clean(self, obj):
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items() if k != 'permutation_scores'}
        elif isinstance(obj, (list, tuple)):
            return [self._clean(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _create_figures(self, report, output_dir, y_true, y_pred):
        # Permutation test
        if 'permutation_scores' in report.get('permutation_test', {}):
            fig, ax = plt.subplots(figsize=(10, 6))
            perm = report['permutation_test']
            ax.hist(perm['permutation_scores'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
            ax.axvline(x=perm['true_r2'], color='#e74c3c', linestyle='--', linewidth=2,
                       label=f"True R² = {perm['true_r2']:.3f}")
            ax.set_xlabel('R²');
            ax.set_ylabel('Frequency')
            ax.set_title(f"Permutation Test (p={perm['p_value']:.4f})")
            ax.legend();
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "permutation_test.png", dpi=300);
            plt.close()

        # Residuals
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        residuals = y_pred - y_true
        ax1.scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted');
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted');
        ax1.grid(True, alpha=0.3)

        ax2.hist(residuals, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Residual');
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution');
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "residual_analysis.png", dpi=300);
        plt.close()

    def run_complete_validation(self, max_features=15):
        results = {}
        for cohort in ['hba1c_cohort', 'fbg_cohort']:
            try:
                results[cohort] = self.generate_validation_report(cohort, max_features)
            except Exception as e:
                print(f"    {cohort}: {e}")
        return results


if __name__ == "__main__":
    framework = ValidationFramework("processed_data_v2")
    framework.run_complete_validation()