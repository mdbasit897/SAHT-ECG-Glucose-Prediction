#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    LeaveOneGroupOut, KFold, cross_val_predict,
    permutation_test_score, learning_curve
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, normaltest

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# Plot settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300
})


class ValidationFramework:
    """
    Comprehensive validation framework addressing all statistical concerns.

    Provides:
    - Permutation testing (addresses small sample validity)
    - Bootstrap confidence intervals (uncertainty quantification)
    - Cross-validation stability analysis
    - Effect size calculations
    - Learning curves for sample size analysis
    """

    def __init__(self, data_dir: str = "processed_data_v2"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}

        print("=" * 70)
        print("COMPREHENSIVE VALIDATION FRAMEWORK v2.0")
        print("=" * 70)

    def load_cohort_data(self, cohort: str = 'hba1c_cohort') -> Tuple:
        """Load data for validation."""
        print(f"\n Loading {cohort} data...")

        loso_dir = self.data_dir / "loso_splits" / cohort

        if loso_dir.exists():
            X = np.load(loso_dir / "X.npy")
            y = np.load(loso_dir / "y.npy")
            y_log = np.load(loso_dir / "y_log.npy")
            groups = np.load(loso_dir / "groups.npy")

            with open(loso_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)

            feature_names = metadata['feature_names']

            print(f"    Loaded: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"    Groups: {len(np.unique(groups))} subjects")

            return X, y, y_log, groups, feature_names
        else:
            raise FileNotFoundError(f"LOSO splits not found at {loso_dir}")

    # =========================================================================
    # PERMUTATION TESTING
    # =========================================================================

    def run_permutation_test(self, X: np.ndarray, y: np.ndarray,
                             groups: np.ndarray = None,
                             n_permutations: int = 1000,
                             n_jobs: int = -1) -> Dict:
        """
        Permutation test to verify results are not due to chance.

        Critical for small sample validation - demonstrates that the
        observed performance is statistically significant.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        groups : np.ndarray
            Subject groups for LOSO
        n_permutations : int
            Number of permutations (1000 recommended for publication)
        n_jobs : int
            Number of parallel jobs (-1 for all cores)

        Returns:
        --------
        Dict with permutation test results
        """
        print(f"\n Running Permutation Test (n={n_permutations})")
        print("-" * 50)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                              lambda_1=1e-6, lambda_2=1e-6)

        if groups is not None and len(np.unique(groups)) >= 5:
            cv = LeaveOneGroupOut()
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Run permutation test
        score, permutation_scores, p_value = permutation_test_score(
            model, X_scaled, y,
            cv=cv,
            groups=groups if groups is not None else None,
            n_permutations=n_permutations,
            scoring='r2',
            random_state=42,
            n_jobs=n_jobs
        )

        # Calculate effect size (Cohen's d equivalent for R²)
        perm_mean = np.mean(permutation_scores)
        perm_std = np.std(permutation_scores)
        effect_size = (score - perm_mean) / perm_std if perm_std > 0 else 0

        results = {
            'true_r2': float(score),
            'permutation_r2_mean': float(perm_mean),
            'permutation_r2_std': float(perm_std),
            'permutation_r2_min': float(np.min(permutation_scores)),
            'permutation_r2_max': float(np.max(permutation_scores)),
            'p_value': float(p_value),
            'effect_size_z': float(effect_size),
            'n_permutations': n_permutations,
            'significant_001': p_value < 0.001,
            'significant_005': p_value < 0.05,
            'significant_010': p_value < 0.10,
            'permutation_scores': permutation_scores.tolist()
        }

        print(f"   True R²:              {score:.4f}")
        print(f"   Permutation R² mean:  {perm_mean:.4f} ± {perm_std:.4f}")
        print(f"   P-value:              {p_value:.4f}")
        print(f"   Effect size (z):      {effect_size:.2f}")
        print(f"   Significant (p<0.05): {' Yes' if p_value < 0.05 else ' No'}")

        return results

    # =========================================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # =========================================================================

    def run_bootstrap_analysis(self, X: np.ndarray, y: np.ndarray,
                               groups: np.ndarray = None,
                               n_bootstrap: int = 1000,
                               ci_levels: List[float] = [90, 95, 99]) -> Dict:
        """
        Bootstrap analysis for confidence intervals.

        Provides uncertainty quantification for small samples.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        groups : np.ndarray
            Subject groups (not used in bootstrap, but kept for consistency)
        n_bootstrap : int
            Number of bootstrap iterations
        ci_levels : list
            Confidence interval levels to compute

        Returns:
        --------
        Dict with bootstrap results
        """
        print(f"\n Running Bootstrap Analysis (n={n_bootstrap})")
        print("-" * 50)

        scaler = StandardScaler()
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                              lambda_1=1e-6, lambda_2=1e-6)

        n_samples = len(y)
        bootstrap_r2 = []
        bootstrap_mae = []
        bootstrap_rmse = []
        bootstrap_corr = []

        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Scale and fit
            X_scaled = scaler.fit_transform(X_boot)

            # 5-fold CV on bootstrap sample
            try:
                cv_pred = cross_val_predict(model, X_scaled, y_boot, cv=5)

                r2 = r2_score(y_boot, cv_pred)
                mae = mean_absolute_error(y_boot, cv_pred)
                rmse = np.sqrt(mean_squared_error(y_boot, cv_pred))
                corr, _ = pearsonr(y_boot, cv_pred)

                if not np.isnan(r2) and not np.isinf(r2):
                    bootstrap_r2.append(r2)
                    bootstrap_mae.append(mae)
                    bootstrap_rmse.append(rmse)
                    bootstrap_corr.append(corr)
            except:
                continue

        bootstrap_r2 = np.array(bootstrap_r2)
        bootstrap_mae = np.array(bootstrap_mae)
        bootstrap_rmse = np.array(bootstrap_rmse)
        bootstrap_corr = np.array(bootstrap_corr)

        # Calculate confidence intervals
        results = {
            'n_bootstrap': n_bootstrap,
            'n_successful': len(bootstrap_r2),
            'metrics': {}
        }

        for metric_name, metric_values in [
            ('r2', bootstrap_r2),
            ('mae', bootstrap_mae),
            ('rmse', bootstrap_rmse),
            ('correlation', bootstrap_corr)
        ]:
            if len(metric_values) == 0:
                continue

            metric_results = {
                'mean': float(np.mean(metric_values)),
                'median': float(np.median(metric_values)),
                'std': float(np.std(metric_values)),
                'min': float(np.min(metric_values)),
                'max': float(np.max(metric_values))
            }

            # Compute CIs
            for ci_level in ci_levels:
                alpha = (100 - ci_level) / 2
                lower = np.percentile(metric_values, alpha)
                upper = np.percentile(metric_values, 100 - alpha)
                metric_results[f'ci_{ci_level}_lower'] = float(lower)
                metric_results[f'ci_{ci_level}_upper'] = float(upper)

            results['metrics'][metric_name] = metric_results

        # Print results
        if 'r2' in results['metrics']:
            r2_res = results['metrics']['r2']
            print(f"   R² Statistics:")
            print(f"      Mean ± SD:  {r2_res['mean']:.4f} ± {r2_res['std']:.4f}")
            print(f"      Median:     {r2_res['median']:.4f}")
            print(f"      95% CI:     [{r2_res['ci_95_lower']:.4f}, {r2_res['ci_95_upper']:.4f}]")

        if 'mae' in results['metrics']:
            mae_res = results['metrics']['mae']
            print(f"   MAE Statistics:")
            print(f"      Mean ± SD:  {mae_res['mean']:.4f} ± {mae_res['std']:.4f}")
            print(f"      95% CI:     [{mae_res['ci_95_lower']:.4f}, {mae_res['ci_95_upper']:.4f}]")

        return results

    # =========================================================================
    # CROSS-VALIDATION STABILITY ANALYSIS
    # =========================================================================

    def analyze_cv_stability(self, X: np.ndarray, y: np.ndarray,
                             groups: np.ndarray = None,
                             n_repeats: int = 100) -> Dict:
        """
        Analyze stability of cross-validation results.

        Tests how stable the results are across different random splits.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        groups : np.ndarray
            Subject groups for LOSO
        n_repeats : int
            Number of CV repetitions

        Returns:
        --------
        Dict with stability analysis results
        """
        print(f"\n Analyzing CV Stability (n={n_repeats} repeats)")
        print("-" * 50)

        scaler = StandardScaler()
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                              lambda_1=1e-6, lambda_2=1e-6)

        r2_scores = []
        mae_scores = []

        for i in range(n_repeats):
            X_scaled = scaler.fit_transform(X)

            # Use different random state each time
            cv = KFold(n_splits=5, shuffle=True, random_state=i)

            try:
                predictions = cross_val_predict(model, X_scaled, y, cv=cv)
                r2 = r2_score(y, predictions)
                mae = mean_absolute_error(y, predictions)

                if not np.isnan(r2):
                    r2_scores.append(r2)
                    mae_scores.append(mae)
            except:
                continue

        r2_scores = np.array(r2_scores)
        mae_scores = np.array(mae_scores)

        results = {
            'n_repeats': n_repeats,
            'n_successful': len(r2_scores),
            'r2': {
                'mean': float(np.mean(r2_scores)),
                'std': float(np.std(r2_scores)),
                'median': float(np.median(r2_scores)),
                'min': float(np.min(r2_scores)),
                'max': float(np.max(r2_scores)),
                'range': float(np.max(r2_scores) - np.min(r2_scores)),
                'cv_coefficient': float(np.std(r2_scores) / np.mean(r2_scores)) if np.mean(r2_scores) != 0 else 0
            },
            'mae': {
                'mean': float(np.mean(mae_scores)),
                'std': float(np.std(mae_scores)),
                'median': float(np.median(mae_scores)),
                'min': float(np.min(mae_scores)),
                'max': float(np.max(mae_scores))
            }
        }

        print(f"   R² Stability:")
        print(f"      Mean ± SD:  {results['r2']['mean']:.4f} ± {results['r2']['std']:.4f}")
        print(f"      Range:      [{results['r2']['min']:.4f}, {results['r2']['max']:.4f}]")
        print(f"      CV Coeff:   {results['r2']['cv_coefficient']:.3f}")

        # Stability interpretation
        cv_coeff = results['r2']['cv_coefficient']
        if cv_coeff < 0.1:
            stability = "Excellent (CV < 0.1)"
        elif cv_coeff < 0.2:
            stability = "Good (CV < 0.2)"
        elif cv_coeff < 0.3:
            stability = "Moderate (CV < 0.3)"
        else:
            stability = "Poor (CV ≥ 0.3)"

        results['stability_interpretation'] = stability
        print(f"   Stability:     {stability}")

        return results

    # =========================================================================
    # LEARNING CURVE ANALYSIS
    # =========================================================================

    def analyze_learning_curve(self, X: np.ndarray, y: np.ndarray,
                               groups: np.ndarray = None,
                               train_sizes: np.ndarray = None) -> Dict:
        """
        Learning curve analysis for sample size recommendations.

        Shows how performance changes with sample size and whether
        more data would likely improve results.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        groups : np.ndarray
            Subject groups
        train_sizes : np.ndarray
            Training set sizes to evaluate

        Returns:
        --------
        Dict with learning curve results
        """
        print(f"\n Analyzing Learning Curve")
        print("-" * 50)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                              lambda_1=1e-6, lambda_2=1e-6)

        if train_sizes is None:
            train_sizes = np.linspace(0.2, 1.0, 5)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X_scaled, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )

        results = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
            'test_scores_std': np.std(test_scores, axis=1).tolist()
        }

        # Analyze trend
        test_means = np.mean(test_scores, axis=1)

        # Is performance still improving?
        if len(test_means) >= 3:
            recent_improvement = test_means[-1] - test_means[-2]
            overall_improvement = test_means[-1] - test_means[0]

            results['analysis'] = {
                'final_test_r2': float(test_means[-1]),
                'recent_improvement': float(recent_improvement),
                'overall_improvement': float(overall_improvement),
                'saturating': recent_improvement < 0.01
            }

            if recent_improvement > 0.02:
                recommendation = "More data would likely improve performance significantly"
            elif recent_improvement > 0.005:
                recommendation = "More data might improve performance marginally"
            else:
                recommendation = "Performance appears saturated - more data may not help"

            results['recommendation'] = recommendation

            print(f"   Final test R²:      {test_means[-1]:.4f}")
            print(f"   Recent improvement: {recent_improvement:+.4f}")
            print(f"   Recommendation:     {recommendation}")

        return results

    # =========================================================================
    # RESIDUAL ANALYSIS
    # =========================================================================

    def analyze_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Comprehensive residual analysis.

        Checks model assumptions and identifies potential issues.

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values

        Returns:
        --------
        Dict with residual analysis results
        """
        print(f"\n Analyzing Residuals")
        print("-" * 50)

        residuals = y_pred - y_true

        # Basic statistics
        results = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'median': float(np.median(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals))
        }

        # Normality tests
        if len(residuals) >= 8:
            try:
                shapiro_stat, shapiro_p = shapiro(residuals)
                results['shapiro_test'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'normal': shapiro_p > 0.05
                }
            except:
                results['shapiro_test'] = None

        if len(residuals) >= 20:
            try:
                dagostino_stat, dagostino_p = normaltest(residuals)
                results['dagostino_test'] = {
                    'statistic': float(dagostino_stat),
                    'p_value': float(dagostino_p),
                    'normal': dagostino_p > 0.05
                }
            except:
                results['dagostino_test'] = None

        # Check for bias
        results['bias_significant'] = abs(results['mean']) > 2 * results['std'] / np.sqrt(len(residuals))

        # Heteroscedasticity check (correlation between predictions and absolute residuals)
        corr_pred_res, p_val = pearsonr(y_pred, np.abs(residuals))
        results['heteroscedasticity'] = {
            'correlation': float(corr_pred_res),
            'p_value': float(p_val),
            'significant': p_val < 0.05
        }

        print(f"   Residual mean:     {results['mean']:.4f}")
        print(f"   Residual std:      {results['std']:.4f}")
        print(f"   Skewness:          {results['skewness']:.3f}")
        print(f"   Bias significant:  {'Yes' if results['bias_significant'] else 'No'}")

        if results.get('shapiro_test'):
            print(
                f"   Normal (Shapiro):  {'Yes' if results['shapiro_test']['normal'] else 'No'} (p={results['shapiro_test']['p_value']:.4f})")

        return results

    # =========================================================================
    # GENERATE VALIDATION REPORT
    # =========================================================================

    def generate_validation_report(self, cohort: str,
                                   output_dir: str = "validation_results") -> Dict:
        """
        Generate comprehensive validation report for a cohort.

        Parameters:
        -----------
        cohort : str
            Cohort name ('hba1c_cohort' or 'fbg_cohort')
        output_dir : str
            Output directory for results

        Returns:
        --------
        Dict with complete validation results
        """
        print(f"\n{'=' * 70}")
        print(f"COMPREHENSIVE VALIDATION: {cohort.upper()}")
        print(f"{'=' * 70}")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        cohort_dir = output_dir / cohort
        cohort_dir.mkdir(exist_ok=True)

        # Load data
        X, y, y_log, groups, feature_names = self.load_cohort_data(cohort)

        # Use log target
        y_target = y_log

        # Feature selection (simple - use all for validation)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get predictions for residual analysis
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        if groups is not None and len(np.unique(groups)) >= 5:
            cv = LeaveOneGroupOut()
            predictions = cross_val_predict(model, X_scaled, y_target, cv=cv, groups=groups)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            predictions = cross_val_predict(model, X_scaled, y_target, cv=cv)

        # Compile results
        report = {
            'cohort': cohort,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_target),
            'n_features': X.shape[1],
            'n_subjects': len(np.unique(groups)) if groups is not None else len(y_target)
        }

        # Run all validation analyses
        print("\n" + "=" * 50)
        report['permutation_test'] = self.run_permutation_test(
            X_scaled, y_target, groups, n_permutations=500
        )

        print("\n" + "=" * 50)
        report['bootstrap_analysis'] = self.run_bootstrap_analysis(
            X_scaled, y_target, groups, n_bootstrap=500
        )

        print("\n" + "=" * 50)
        report['cv_stability'] = self.analyze_cv_stability(
            X_scaled, y_target, groups, n_repeats=50
        )

        print("\n" + "=" * 50)
        report['learning_curve'] = self.analyze_learning_curve(
            X_scaled, y_target, groups
        )

        print("\n" + "=" * 50)
        report['residual_analysis'] = self.analyze_residuals(y_target, predictions)

        # Summary
        report['summary'] = {
            'statistically_significant': report['permutation_test']['significant_005'],
            'r2_95_ci': [
                report['bootstrap_analysis']['metrics']['r2']['ci_95_lower'],
                report['bootstrap_analysis']['metrics']['r2']['ci_95_upper']
            ] if 'r2' in report['bootstrap_analysis']['metrics'] else None,
            'cv_stability': report['cv_stability']['stability_interpretation'],
            'residuals_normal': report['residual_analysis'].get('shapiro_test', {}).get('normal', None)
        }

        # Save results
        # Remove non-serializable data
        report_clean = self._clean_for_json(report)

        with open(cohort_dir / "validation_report.json", 'w') as f:
            json.dump(report_clean, f, indent=2)

        # Create visualizations
        self._create_validation_figures(report, cohort_dir, y_target, predictions)

        print(f"\n Validation report saved to {cohort_dir}")

        self.validation_results[cohort] = report

        return report

    def _clean_for_json(self, obj):
        """Clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()
                    if k != 'permutation_scores'}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def _create_validation_figures(self, report: Dict, output_dir: Path,
                                   y_true: np.ndarray, y_pred: np.ndarray):
        """Create validation visualization figures."""

        # Figure 1: Permutation test histogram
        if 'permutation_test' in report and 'permutation_scores' in report['permutation_test']:
            fig, ax = plt.subplots(figsize=(10, 6))

            perm_scores = report['permutation_test']['permutation_scores']
            true_score = report['permutation_test']['true_r2']

            ax.hist(perm_scores, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
            ax.axvline(x=true_score, color='#e74c3c', linestyle='--', linewidth=2,
                       label=f'True R² = {true_score:.3f}')

            ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'Permutation Test Results\np-value = {report["permutation_test"]["p_value"]:.4f}',
                         fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "permutation_test.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Figure 2: Bootstrap distribution
        if 'bootstrap_analysis' in report and 'r2' in report['bootstrap_analysis']['metrics']:
            fig, ax = plt.subplots(figsize=(10, 6))

            r2_metrics = report['bootstrap_analysis']['metrics']['r2']

            # We don't have the raw bootstrap samples in the report, so we'll create a representative plot
            ax.axvline(x=r2_metrics['mean'], color='#e74c3c', linestyle='-', linewidth=2,
                       label=f"Mean R² = {r2_metrics['mean']:.3f}")
            ax.axvline(x=r2_metrics['ci_95_lower'], color='#27ae60', linestyle='--', linewidth=2,
                       label=f"95% CI: [{r2_metrics['ci_95_lower']:.3f}, {r2_metrics['ci_95_upper']:.3f}]")
            ax.axvline(x=r2_metrics['ci_95_upper'], color='#27ae60', linestyle='--', linewidth=2)

            ax.axvspan(r2_metrics['ci_95_lower'], r2_metrics['ci_95_upper'],
                       alpha=0.3, color='#27ae60')

            ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
            ax.set_title('Bootstrap 95% Confidence Interval', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "bootstrap_ci.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Figure 3: Residual analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        residuals = y_pred - y_true

        # Residuals vs Predicted
        ax1 = axes[0]
        ax1.scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax1.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Residual histogram
        ax2 = axes[1]
        ax2.hist(residuals, bins=20, alpha=0.7, color='#3498db', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residual Value', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "residual_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_complete_validation(self) -> Dict:
        """Run validation for all available cohorts."""
        print("\n" + "=" * 70)
        print("RUNNING COMPLETE VALIDATION")
        print("=" * 70)

        all_results = {}

        for cohort in ['hba1c_cohort', 'fbg_cohort']:
            try:
                results = self.generate_validation_report(cohort)
                all_results[cohort] = results
            except FileNotFoundError as e:
                print(f"\n  {cohort}: Data not found - {e}")
            except Exception as e:
                print(f"\n {cohort}: Error - {e}")

        return all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION FRAMEWORK v2.0")
    print("=" * 70)
    print()

    # Initialize framework
    framework = ValidationFramework("processed_data_v2")

    # Run complete validation
    results = framework.run_complete_validation()

    if results:
        print()
        print("=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)

        for cohort, report in results.items():
            print(f"\n{cohort.upper()}:")
            summary = report.get('summary', {})
            print(f"   Statistically significant: {summary.get('statistically_significant', 'N/A')}")
            print(f"   95% CI: {summary.get('r2_95_ci', 'N/A')}")
            print(f"   CV Stability: {summary.get('cv_stability', 'N/A')}")
    else:
        print("\n  No validation results. Run preprocessing first.")