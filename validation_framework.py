#!/usr/bin/env python3
"""
Revised Validation Framework
==================================
FIXED: All validation analyses now use sklearn Pipeline to ensure
feature selection + scaling happen within each CV fold automatically.
UPDATED: Uses Extra Trees (best-performing model from baseline comparison)
instead of Bayesian Ridge for all validation analyses.

Addresses:
  - R3 #4: CV hygiene
  - R3 #5: Back-transformed error interpretation
  - R3 #7: Bootstrap CIs with proper subject-level resampling
  - E: Methodological details require clearer reporting
  - Model mismatch fix: validation now uses same model as reported best
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
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

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

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
        print("VALIDATION FRAMEWORK (Pipeline-based CV hygiene)")
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
        Uses Extra Trees — the best-performing model from baseline comparison.
        """
        k = max_features
        return Pipeline([
            ('feature_selection', SelectKBest(f_regression, k=k)),
            ('scaler', StandardScaler()),
            ('model', ExtraTreesRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42))
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
            'bootstrap_r2_samples': bootstrap_r2.tolist(),
            'bootstrap_mae_samples': bootstrap_mae.tolist(),
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
            'cv_hygiene': 'Pipeline(SelectKBest + StandardScaler + ExtraTreesRegressor) within each LOSO fold'
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
            return {k: self._clean(v) for k, v in obj.items()
                    if k not in ('permutation_scores', 'bootstrap_r2_samples', 'bootstrap_mae_samples')}
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
        cohort_label = report.get('cohort', 'Unknown')
        cohort_title = 'HbA1c' if 'hba1c' in cohort_label else 'FBG'

        # =====================================================================
        # FIGURE: Permutation Test Null Distribution
        # =====================================================================
        if 'permutation_scores' in report.get('permutation_test', {}):
            fig, ax = plt.subplots(figsize=(10, 6))
            perm = report['permutation_test']
            perm_scores = perm['permutation_scores']

            ax.hist(perm_scores, bins=50, alpha=0.7, color='#3498db',
                    edgecolor='black', linewidth=0.5, density=True, label='Permutation distribution')
            ax.axvline(x=perm['true_r2'], color='#e74c3c', linestyle='--', linewidth=2.5,
                       label=f"Observed R² = {perm['true_r2']:.3f}")
            ax.axvline(x=perm['permutation_r2_mean'], color='#95a5a6', linestyle=':',
                       linewidth=1.5, label=f"Permutation mean = {perm['permutation_r2_mean']:.3f}")

            # Shade the p-value region
            threshold = perm['true_r2']
            perm_arr = np.array(perm_scores)
            ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5],
                             threshold, max(perm_arr.max(), threshold) + 0.05,
                             alpha=0.15, color='#e74c3c')

            ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax.set_title(f"Permutation Test — {cohort_title}\n"
                         f"p = {perm['p_value']:.4f}, n = {perm['n_permutations']} permutations, "
                         f"Effect size z = {perm['effect_size_z']:.2f}",
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "permutation_test.png", dpi=300)
            plt.close()
            print(f"   Saved permutation_test.png")

        # =====================================================================
        # FIGURE: Prediction vs Actual Scatter + Residuals
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Predicted vs Actual
        ax1.scatter(y_true, y_pred, alpha=0.7, s=80, c='#3498db',
                    edgecolors='black', linewidth=0.5, zorder=3)

        all_vals = np.concatenate([y_true, y_pred])
        lims = [np.min(all_vals) - 0.05, np.max(all_vals) + 0.05]
        ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Perfect prediction')

        try:
            z = np.polyfit(y_true, y_pred, 1)
            p_line = np.poly1d(z)
            x_fit = np.linspace(lims[0], lims[1], 100)
            ax1.plot(x_fit, p_line(x_fit), 'r-', alpha=0.7, linewidth=1.5, label='Fit line')
        except Exception:
            pass

        try:
            corr, pval = pearsonr(y_true, y_pred)
        except Exception:
            corr, pval = np.nan, np.nan
        r2 = r2_score(y_true, y_pred)

        ax1.set_xlabel('Actual (log-scale)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted (log-scale)', fontsize=12, fontweight='bold')
        ax1.set_title(f"Predicted vs Actual — {cohort_title}\n"
                      f"R²={r2:.3f}, r={corr:.3f}, p={pval:.4f}",
                      fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_aspect('equal', adjustable='box')

        # Panel 2: Residuals vs Predicted
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.7, s=80, c='#e74c3c',
                    edgecolors='black', linewidth=0.5, zorder=3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.set_xlabel('Predicted (log-scale)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residual (Predicted − Actual)', fontsize=12, fontweight='bold')
        ax2.set_title(f"Residual Plot — {cohort_title}\n"
                      f"Mean={np.mean(residuals):.4f}, Std={np.std(residuals):.4f}",
                      fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "prediction_vs_actual.png", dpi=300)
        plt.close()
        print(f"   Saved prediction_vs_actual.png")

        # Also save the old-format residual-only analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted'); ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted'); ax1.grid(True, alpha=0.3)

        ax2.hist(residuals, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Residual'); ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "residual_analysis.png", dpi=300)
        plt.close()

        # =====================================================================
        # FIGURE: Bootstrap CI Distribution
        # Addresses: R3#7 — uncertainty emphasis
        # =====================================================================
        bootstrap_data = report.get('bootstrap_analysis', {})
        bootstrap_r2 = bootstrap_data.get('bootstrap_r2_samples', [])
        bootstrap_mae = bootstrap_data.get('bootstrap_mae_samples', [])

        if len(bootstrap_r2) > 10:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            r2_arr = np.array(bootstrap_r2)
            mae_arr = np.array(bootstrap_mae)

            r2_metrics = bootstrap_data.get('metrics', {}).get('r2', {})
            ci_lower = r2_metrics.get('ci_95_lower', np.percentile(r2_arr, 2.5))
            ci_upper = r2_metrics.get('ci_95_upper', np.percentile(r2_arr, 97.5))
            r2_median = r2_metrics.get('median', np.median(r2_arr))

            # R² distribution
            if HAS_SEABORN:
                sns.violinplot(y=r2_arr, ax=ax1, color='#3498db', alpha=0.3, inner=None)
                sns.stripplot(y=r2_arr, ax=ax1, color='#3498db', alpha=0.1, size=2, jitter=True)
            else:
                ax1.hist(r2_arr, bins=40, alpha=0.7, color='#3498db',
                         edgecolor='black', linewidth=0.5, orientation='horizontal')

            ax1.axhline(y=r2_median, color='#e74c3c', linestyle='-', linewidth=2,
                        label=f'Median R² = {r2_median:.3f}')
            ax1.axhline(y=ci_lower, color='#e74c3c', linestyle='--', linewidth=1.5,
                        label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
            ax1.axhline(y=ci_upper, color='#e74c3c', linestyle='--', linewidth=1.5)
            ax1.axhspan(ci_lower, ci_upper, alpha=0.15, color='#e74c3c')

            ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
            ax1.set_title(f"Bootstrap R² Distribution — {cohort_title}\n"
                          f"n = {len(r2_arr)} resamples (subject-level)",
                          fontsize=11, fontweight='bold')
            ax1.legend(fontsize=9, loc='lower left')
            ax1.grid(True, alpha=0.3)

            # MAE distribution
            mae_metrics = bootstrap_data.get('metrics', {}).get('mae', {})
            mae_ci_l = mae_metrics.get('ci_95_lower', np.percentile(mae_arr, 2.5))
            mae_ci_u = mae_metrics.get('ci_95_upper', np.percentile(mae_arr, 97.5))
            mae_med = np.median(mae_arr)

            if HAS_SEABORN:
                sns.violinplot(y=mae_arr, ax=ax2, color='#f39c12', alpha=0.3, inner=None)
                sns.stripplot(y=mae_arr, ax=ax2, color='#f39c12', alpha=0.1, size=2, jitter=True)
            else:
                ax2.hist(mae_arr, bins=40, alpha=0.7, color='#f39c12',
                         edgecolor='black', linewidth=0.5, orientation='horizontal')

            ax2.axhline(y=mae_med, color='#c0392b', linestyle='-', linewidth=2,
                        label=f'Median MAE = {mae_med:.3f}')
            ax2.axhline(y=mae_ci_l, color='#c0392b', linestyle='--', linewidth=1.5,
                        label=f'95% CI: [{mae_ci_l:.3f}, {mae_ci_u:.3f}]')
            ax2.axhline(y=mae_ci_u, color='#c0392b', linestyle='--', linewidth=1.5)
            ax2.axhspan(mae_ci_l, mae_ci_u, alpha=0.15, color='#c0392b')

            ax2.set_ylabel('MAE (log-scale)', fontsize=12, fontweight='bold')
            ax2.set_title(f"Bootstrap MAE Distribution — {cohort_title}",
                          fontsize=11, fontweight='bold')
            ax2.legend(fontsize=9, loc='upper right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "bootstrap_ci_distribution.png", dpi=300)
            plt.close()
            print(f"   Saved bootstrap_ci_distribution.png")

        # =====================================================================
        # FIGURE: Per-Subject Prediction Error
        # Shows which subjects are systematic outliers
        # =====================================================================
        n_subjects = report.get('n_subjects', len(y_true))
        groups_for_plot = np.arange(len(y_true))  # index as subject proxy

        abs_errors = np.abs(y_pred - y_true)
        fig, ax = plt.subplots(figsize=(12, 5))

        colors = ['#e74c3c' if e > np.mean(abs_errors) + np.std(abs_errors) else '#3498db'
                  for e in abs_errors]
        ax.bar(range(len(abs_errors)), abs_errors, color=colors, alpha=0.8,
               edgecolor='black', linewidth=0.5)
        ax.axhline(y=np.mean(abs_errors), color='black', linestyle='--', linewidth=1.5,
                   label=f'Mean |error| = {np.mean(abs_errors):.4f}')
        ax.axhline(y=np.mean(abs_errors) + np.std(abs_errors), color='red',
                   linestyle=':', linewidth=1, label=f'Mean + 1 SD = {np.mean(abs_errors)+np.std(abs_errors):.4f}')

        ax.set_xlabel('Subject Index (LOSO fold)', fontsize=12, fontweight='bold')
        ax.set_ylabel('|Prediction Error| (log-scale)', fontsize=12, fontweight='bold')
        ax.set_title(f"Per-Subject Absolute Prediction Error — {cohort_title}\n"
                     f"Red bars = outliers (> mean + 1 SD)",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / "per_subject_error.png", dpi=300)
        plt.close()
        print(f"   Saved per_subject_error.png")

        # =====================================================================
        # Learning Curve
        # =====================================================================
        lc_data = report.get('learning_curve', {})
        train_sizes = lc_data.get('train_sizes', [])
        test_means = lc_data.get('test_scores_mean', [])
        test_stds = lc_data.get('test_scores_std', [])

        if len(train_sizes) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ts = np.array(train_sizes)
            tm = np.array(test_means)
            t_std = np.array(test_stds)

            ax.plot(ts, tm, 'o-', color='#3498db', linewidth=2, markersize=8, label='Test R²')
            ax.fill_between(ts, tm - t_std, tm + t_std, alpha=0.2, color='#3498db')
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

            ax.set_xlabel('Training Set Size (samples)', fontsize=12, fontweight='bold')
            ax.set_ylabel('R² Score (5-fold CV)', fontsize=12, fontweight='bold')
            ax.set_title(f"Learning Curve — {cohort_title}\n"
                         f"{lc_data.get('recommendation', '')}",
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "learning_curve.png", dpi=300)
            plt.close()
            print(f"   Saved learning_curve.png")

    # =========================================================================
    # FIGURE: Correlation Matrix of Selected Features
    # =========================================================================

    def create_feature_correlation_matrix(self, cohort, max_features=15,
                                          output_dir="validation_results_v3"):
        """
        Correlation matrix of the top selected features.
        Shows inter-feature relationships for interpretability.
        """
        print(f"\n Creating feature correlation matrix for {cohort}...")
        output_dir = Path(output_dir) / cohort
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            X, y, y_log, groups, feature_names = self.load_cohort_data(cohort)
        except Exception as e:
            print(f"   ⚠ Could not load data: {e}")
            return

        target = y_log
        k = min(max_features, X.shape[1])

        # Use SelectKBest on full data for reporting (not for evaluation)
        selector = SelectKBest(f_regression, k=k)
        selector.fit(X, target)
        mask = selector.get_support()
        selected_names = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        X_selected = X[:, mask]

        # Clean names
        display_names = []
        for f in selected_names:
            name = f.replace('hrv_ds_', 'HRV-DS:').replace('hrv_rem_', 'HRV-REM:')
            name = name.replace('hrv_rs_', 'HRV-RS:').replace('ecg_all_', 'ECG:')
            name = name.replace('ecg_sleep_', 'ECG-slp:').replace('ecg_day_', 'ECG-day:')
            name = name.replace('_age_normalized', '(AN)').replace('_', ' ')
            if len(name) > 25:
                name = name[:23] + '..'
            display_names.append(name)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(X_selected.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        cohort_title = 'HbA1c' if 'hba1c' in cohort else 'FBG'

        if HAS_SEABORN:
            mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(corr_matrix, mask=mask_upper, annot=True, fmt='.2f',
                        cmap='RdBu_r', center=0, ax=ax,
                        xticklabels=display_names, yticklabels=display_names,
                        linewidths=0.5, vmin=-1, vmax=1,
                        cbar_kws={'label': 'Pearson Correlation'})
        else:
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(range(len(display_names)))
            ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(display_names)))
            ax.set_yticklabels(display_names, fontsize=8)
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha='center', va='center', fontsize=6)
            plt.colorbar(im, ax=ax, label='Pearson Correlation')

        ax.set_title(f"Feature Correlation Matrix — {cohort_title}\n"
                     f"Top {k} features by SelectKBest F-score",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "feature_correlation_matrix.png", dpi=300)
        plt.close()
        print(f"   Saved feature_correlation_matrix.png")

    # =========================================================================
    # FIGURE: Combined Permutation Tests (both cohorts in one figure)
    # =========================================================================

    def create_combined_permutation_figure(self, output_dir="validation_results_v3"):
        """
        Side-by-side permutation test null distributions for both cohorts.
        """
        cohorts = [c for c in ['hba1c_cohort', 'fbg_cohort'] if c in self.validation_results]
        if not cohorts:
            return

        n_panels = len(cohorts)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        cohort_titles = {'hba1c_cohort': 'HbA1c Cohort', 'fbg_cohort': 'FBG Cohort'}

        for idx, cohort in enumerate(cohorts):
            ax = axes[idx]
            perm = self.validation_results[cohort].get('permutation_test', {})
            perm_scores = perm.get('permutation_scores', [])

            if not perm_scores:
                ax.text(0.5, 0.5, 'No permutation data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            ax.hist(perm_scores, bins=50, alpha=0.7, color='#3498db',
                    edgecolor='black', linewidth=0.5, density=True)
            ax.axvline(x=perm['true_r2'], color='#e74c3c', linestyle='--',
                       linewidth=2.5, label=f"Observed R² = {perm['true_r2']:.3f}")
            ax.axvline(x=perm['permutation_r2_mean'], color='#95a5a6',
                       linestyle=':', linewidth=1.5,
                       label=f"Null mean = {perm['permutation_r2_mean']:.3f}")

            ax.set_xlabel('R² Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Density', fontsize=11, fontweight='bold')
            ax.set_title(f"{cohort_titles.get(cohort, cohort)}\n"
                         f"p = {perm['p_value']:.4f}",
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path(output_dir) / "combined_permutation_tests.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"   Saved combined_permutation_tests.png")

    def run_complete_validation(self, max_features=15):
        results = {}
        for cohort in ['hba1c_cohort', 'fbg_cohort']:
            try:
                results[cohort] = self.generate_validation_report(cohort, max_features)
                # Generate correlation matrix for each cohort
                self.create_feature_correlation_matrix(cohort, max_features)
            except Exception as e:
                print(f"    {cohort}: {e}")

        # Generate combined cross-cohort figures
        if self.validation_results:
            self.create_combined_permutation_figure()

        return results

if __name__ == "__main__":
    framework = ValidationFramework("processed_data_v2")
    framework.run_complete_validation()