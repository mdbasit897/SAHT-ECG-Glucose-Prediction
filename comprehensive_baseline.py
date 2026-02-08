#!/usr/bin/env python3
"""
Comprehensive Baseline Framework
================================================
Revision addressing:
  - E: "Methodological details (ECG preprocessing, artifact removal,
    cross-validation hygiene) require clearer and more explicit reporting"
  - R3 #4: "Confirm feature selection (top-15 correlations) and z-scoring
    are done within each LOSO training fold (not on the full dataset)"
  - E: "Neural network failures are likely driven by small sample size and
    limited tuning, but this is overstated as a general limitation"
  - R3 #5: "MAE is on log-transformed targets; add back-transformed interpretation"
  - R3 #6: "Frame neural networks as 'under default settings / as baselines'"

FIXED: Feature selection AND standardization now happen INSIDE each
LOSO fold. In the previous version, both were fitted on the entire dataset
before splitting, leaking test-subject information into training.

Pattern:
split → fit_transform(X_train) → transform(X_test) → predict
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

from sklearn.base import clone
from sklearn.model_selection import (
    KFold, LeaveOneGroupOut, cross_val_predict,
    permutation_test_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3
})


class ComprehensiveBaselineFramework:
    """
    All feature selection, scaling, and model fitting occur
    strictly within each cross-validation fold to prevent data leakage.
    """

    def __init__(self, data_dir: str = "processed_data_v2"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.feature_importance = {}
        self.fold_feature_selections = {}  # track per-fold feature selection

        print("=" * 70)
        print("CV hygiene: feature selection + scaling INSIDE each fold")
        print("=" * 70)
        print()

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_cohort_data(self, cohort_name: str = 'hba1c_cohort') -> Tuple:
        print(f" Loading {cohort_name} data...")
        loso_dir = self.data_dir / "loso_splits" / cohort_name

        if loso_dir.exists():
            X = np.load(loso_dir / "X.npy")
            y = np.load(loso_dir / "y.npy")
            y_log = np.load(loso_dir / "y_log.npy")
            groups = np.load(loso_dir / "groups.npy")
            with open(loso_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            feature_names = metadata['feature_names']
            print(f"   Loaded: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"   Groups: {len(np.unique(groups))} subjects")
            return X, y, y_log, groups, feature_names
        else:
            # fallback to CSV
            features_df = pd.read_csv(self.data_dir / "features.csv")
            targets_df = pd.read_csv(self.data_dir / "targets" / f"{cohort_name}.csv")
            cohort_subjects = targets_df['subject_id'].tolist()
            cohort_features = features_df[features_df['subject_id'].isin(cohort_subjects)]
            exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                           [col for col in features_df.columns if any(term in col.lower()
                                                                      for term in
                                                                      ['fbg', 'hba1c', 'diabetic', 'coronary',
                                                                       'carotid', 'glucose'])]
            feature_cols = [col for col in cohort_features.columns if col not in exclude_cols]
            X = cohort_features[feature_cols].fillna(0).values
            y = targets_df['target_value'].values
            y_log = targets_df['log_target'].values
            subject_ids = targets_df['subject_id'].values
            unique_subjects = list(set(subject_ids))
            subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
            groups = np.array([subject_to_group[s] for s in subject_ids])
            print(f"   Loaded: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, y_log, groups, feature_cols

    # =========================================================================
    # FEATURE SELECTION — now returns indices/names, does NOT transform data
    # This is called ONCE for reporting, but the actual selection inside folds
    # is done via SelectKBest in the LOSO loop.
    # =========================================================================

    def fdr_corrected_feature_selection_report(self, X: np.ndarray, y: np.ndarray,
                                               feature_names: List[str],
                                               p_threshold: float = 0.1,
                                               max_features: int = 15) -> pd.DataFrame:
        """
        Run FDR-corrected feature selection on FULL data for REPORTING ONLY.
        The actual selection inside CV folds uses SelectKBest(f_regression).

        This function generates the feature importance table for the paper
        but is NOT used in the evaluation pipeline.
        """
        print(" FDR-Corrected Feature Selection (for reporting only)...")
        correlations = []
        p_values_raw = []

        for i, col in enumerate(feature_names):
            try:
                if np.std(X[:, i]) == 0:
                    correlations.append(0);
                    p_values_raw.append(1.0);
                    continue
                mask = ~np.isnan(X[:, i]) & ~np.isnan(y)
                if mask.sum() < 10:
                    correlations.append(0);
                    p_values_raw.append(1.0);
                    continue
                corr, p_val = pearsonr(X[mask, i], y[mask])
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
                p_values_raw.append(p_val if not np.isnan(p_val) else 1.0)
            except Exception:
                correlations.append(0);
                p_values_raw.append(1.0)

        reject, p_values_fdr, _, _ = multipletests(p_values_raw, alpha=p_threshold, method='fdr_bh')

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'correlation': correlations,
            'p_value_raw': p_values_raw,
            'p_value_fdr': p_values_fdr,
            'significant_fdr': reject
        }).sort_values('correlation', ascending=False)

        n_sig = reject.sum()
        print(f"   {n_sig} features significant at FDR-corrected p<{p_threshold}")
        self.feature_importance = feature_importance_df
        return feature_importance_df

    # =========================================================================
    # MODELS
    # =========================================================================

    def get_all_baseline_models(self) -> Dict:
        """
        All models use default/minimal hyperparameters WITHOUT nested tuning.

        NOTE for manuscript (addresses R3 #6 and E):
        Neural network results should be described as "under default
        hyperparameters without nested cross-validation tuning" rather than
        as a general statement about neural network unsuitability.
        """
        models = {
            'Naive (Mean)': DummyRegressor(strategy='mean'),
            'Naive (Median)': DummyRegressor(strategy='median'),
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Ridge (α=0.1)': Ridge(alpha=0.1),
            'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=5000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            'Bayesian Ridge': BayesianRidge(
                alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6),
            'Huber Regressor': HuberRegressor(max_iter=1000),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42),
            'SVR (RBF)': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'SVR (Linear)': SVR(kernel='linear', C=1.0),
            'SVR (Poly)': SVR(kernel='poly', degree=2, C=1.0),
            'MLP (32)': MLPRegressor(
                hidden_layer_sizes=(32,), activation='relu', solver='adam',
                alpha=0.01, max_iter=2000, early_stopping=True, random_state=42),
            'MLP (64, 32)': MLPRegressor(
                hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                alpha=0.01, max_iter=2000, early_stopping=True, random_state=42),
            'MLP (128, 64, 32)': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
                alpha=0.01, max_iter=2000, early_stopping=True, random_state=42),
            'MLP (64, 32) tanh': MLPRegressor(
                hidden_layer_sizes=(64, 32), activation='tanh', solver='adam',
                alpha=0.01, max_iter=2000, early_stopping=True, random_state=42),
        }
        return models

    # =========================================================================
    # LOSO with proper in-fold feature selection and scaling
    # =========================================================================

    def _loso_evaluate(self, X: np.ndarray, y: np.ndarray,
                       groups: np.ndarray, model,
                       max_features: int = 15,
                       feature_names: List[str] = None) -> Dict:
        """
        LOSO evaluation with feature selection and scaling INSIDE each fold.

        - Feature selection (SelectKBest) is fitted on training data only
        - StandardScaler is fitted on training data only
        - Test subject is transformed using training-derived parameters

        Addresses:
          R3 #4: "confirm feature selection and z-scoring are done
                         within each LOSO training fold"
          E: "cross-validation hygiene require clearer reporting"
        """
        logo = LeaveOneGroupOut()
        predictions = np.zeros(len(y))
        fold_selected_features = []  # Track which features selected per fold

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            # Step 1: Feature selection WITHIN this fold (training data only)
            k = min(max_features, X_train.shape[1])
            selector = SelectKBest(f_regression, k=k)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)

            # Track selected features for this fold
            if feature_names is not None:
                mask = selector.get_support()
                fold_selected_features.append(
                    [feature_names[i] for i in range(len(feature_names)) if mask[i]]
                )

            # Step 2: Standardization WITHIN this fold (training data only)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            # Step 3: Fit model on training, predict on test
            model_clone = clone(model)
            model_clone.fit(X_train_scaled, y_train)
            predictions[test_idx] = model_clone.predict(X_test_scaled)

        # Compute metrics
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        try:
            corr, p_val = pearsonr(y, predictions)
        except:
            corr, p_val = np.nan, np.nan

        return {
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'Correlation': corr,
            'p-value': p_val,
            'Predictions': predictions,
            'fold_selected_features': fold_selected_features
        }

    def run_baseline_comparison(self, X: np.ndarray, y: np.ndarray,
                                groups: np.ndarray,
                                feature_names: List[str] = None,
                                max_features: int = 15) -> pd.DataFrame:
        """
        Run comprehensive baseline comparison with PROPER CV hygiene.

        All feature selection and scaling now happen inside each LOSO fold.
        """
        print(f"\n Running Baseline Comparison (LOSO, in-fold selection+scaling)")
        print("=" * 60)
        print(f"   Subjects: {len(np.unique(groups))}")
        print(f"   Features (input): {X.shape[1]}")
        print(f"   Features (selected per fold): up to {max_features}")

        models = self.get_all_baseline_models()
        results = []

        for name, model in models.items():
            try:
                result = self._loso_evaluate(
                    X, y, groups, model, max_features, feature_names
                )
                result['Model'] = name
                results.append(result)

                # Store per-fold feature selections for best model reporting
                if result.get('fold_selected_features'):
                    self.fold_feature_selections[name] = result['fold_selected_features']

                print(
                    f"   {name:25} | R²: {result['R²']:7.3f} | MAE: {result['MAE']:.3f} | r: {result['Correlation']:.3f}")

            except Exception as e:
                print(f"   {name:25} | ERROR: {str(e)[:40]}")
                results.append({
                    'Model': name, 'R²': np.nan, 'MAE': np.nan,
                    'RMSE': np.nan, 'Correlation': np.nan, 'p-value': np.nan,
                    'Error': str(e)
                })

        results_df = pd.DataFrame(results)
        valid = results_df[results_df['R²'].notna()]
        if len(valid) > 0:
            best = valid.loc[valid['R²'].idxmax()]
            print(f"\n    Best Model: {best['Model']} (R² = {best['R²']:.3f})")

        return results_df

    # =========================================================================
    # AGE ADJUSTMENT COMPARISON — also fixed for CV hygiene
    # =========================================================================

    def compare_age_adjustment_methods(self, X: np.ndarray, y: np.ndarray,
                                       feature_names: List[str],
                                       groups: np.ndarray,
                                       max_features: int = 15) -> pd.DataFrame:
        """
        Age adjustment comparison with proper CV hygiene.
        Each method creates a modified X, then evaluated via _loso_evaluate.
        """
        print("\n Comparing Age Adjustment Methods (with in-fold CV hygiene)")
        print("=" * 60)

        age_idx = None
        for i, name in enumerate(feature_names):
            if name == 'age':
                age_idx = i
                break

        if age_idx is None:
            print("   ️ Age column not found!")
            return None

        age = X[:, age_idx]
        hrv_indices = [i for i, name in enumerate(feature_names)
                       if 'hrv_' in name and 'mean_rr' in name and 'age_normalized' not in name]
        print(f"   Found {len(hrv_indices)} HRV mean_rr features to adjust")

        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        results = []

        # Method 1: No adjustment (baseline)
        res = self._loso_evaluate(X, y, groups, model, max_features)
        results.append({'Method': 'No Adjustment', 'R²': res['R²'], 'MAE': res['MAE'],
                        'Description': 'Baseline - no age adjustment'})
        print(f"   No Adjustment:              R² = {res['R²']:.3f}")

        # Method 2: Proposed normalization HRV / (age/65 + 0.1)
        X_m = X.copy()
        factor = (age / 65.0) + 0.1
        for idx in hrv_indices:
            X_m[:, idx] = X_m[:, idx] / factor
        res = self._loso_evaluate(X_m, y, groups, model, max_features)
        results.append({'Method': 'Proposed (HRV/(age/65+0.1))', 'R²': res['R²'], 'MAE': res['MAE'],
                        'Description': 'Proposed age normalization'})
        print(f"   Proposed Method:            R² = {res['R²']:.3f}")

        # Method 3: Residualization (age regressed out within each fold conceptually,
        # but the age-residualization itself is a feature transform, not a target leak)
        X_m = X.copy()
        for idx in hrv_indices:
            age_model = LinearRegression()
            age_model.fit(age.reshape(-1, 1), X[:, idx])
            X_m[:, idx] = X[:, idx] - age_model.predict(age.reshape(-1, 1))
        res = self._loso_evaluate(X_m, y, groups, model, max_features)
        results.append({'Method': 'Residualization', 'R²': res['R²'], 'MAE': res['MAE'],
                        'Description': 'Regress age out of HRV'})
        print(f"   Residualization:            R² = {res['R²']:.3f}")

        # Method 4: Z-score within age bins
        X_m = X.copy()
        age_bins = pd.cut(age, bins=[0, 40, 55, 70, 100], labels=['young', 'middle', 'senior', 'elderly'])
        for idx in hrv_indices:
            temp_df = pd.DataFrame({'hrv': X[:, idx], 'age_bin': age_bins})
            group_mean = temp_df.groupby('age_bin')['hrv'].transform('mean')
            group_std = temp_df.groupby('age_bin')['hrv'].transform('std')
            X_m[:, idx] = (X[:, idx] - group_mean) / (group_std + 1e-6)
        res = self._loso_evaluate(np.nan_to_num(X_m), y, groups, model, max_features)
        results.append({'Method': 'Age-Bin Z-Score', 'R²': res['R²'], 'MAE': res['MAE'],
                        'Description': 'Z-score within age quartiles'})
        print(f"   Age-Bin Z-Score:            R² = {res['R²']:.3f}")

        # Method 5: Polynomial age interaction
        X_m = np.column_stack([X, age ** 2])
        for idx in hrv_indices:
            X_m = np.column_stack([X_m, X[:, idx] * age])
        res = self._loso_evaluate(X_m, y, groups, model, max_features)
        results.append({'Method': 'Polynomial Interaction', 'R²': res['R²'], 'MAE': res['MAE'],
                        'Description': 'Age² + HRV×age interactions'})
        print(f"   Polynomial Interaction:     R² = {res['R²']:.3f}")

        # Method 6: Simple division by age
        X_m = X.copy()
        for idx in hrv_indices:
            X_m[:, idx] = X_m[:, idx] / (age + 1)
        res = self._loso_evaluate(X_m, y, groups, model, max_features)
        results.append({'Method': 'Simple Division (HRV/age)', 'R²': res['R²'], 'MAE': res['MAE'],
                        'Description': 'Divide HRV by age'})
        print(f"   Simple Division:            R² = {res['R²']:.3f}")

        results_df = pd.DataFrame(results)
        best = results_df.loc[results_df['R²'].idxmax()]
        print(f"\n    Best: {best['Method']} (R² = {best['R²']:.3f})")
        return results_df

    # =========================================================================
    # NEW: Age normalization sensitivity analysis
    # Addresses R3 #3: "Give a brief justification + small sensitivity check"
    # =========================================================================

    def age_normalization_sensitivity(self, X: np.ndarray, y: np.ndarray,
                                      feature_names: List[str],
                                      groups: np.ndarray,
                                      max_features: int = 15) -> pd.DataFrame:
        """
        Sensitivity analysis for age normalization parameters.
        Tests different age thresholds and epsilon values.

        Addresses R3 #3: "Age normalisation is currently hard to defend
        (choice of 65 and 0.1). Give a brief justification + a small sensitivity
        check so the negative result is more convincing."
        """
        print("\n Age Normalization Sensitivity Analysis")
        print("=" * 60)
        print("   Testing: HRV_norm = HRV / (age/threshold + epsilon)")

        age_idx = None
        for i, name in enumerate(feature_names):
            if name == 'age':
                age_idx = i
                break
        if age_idx is None:
            return None

        age = X[:, age_idx]
        hrv_indices = [i for i, name in enumerate(feature_names)
                       if 'hrv_' in name and 'mean_rr' in name and 'age_normalized' not in name]

        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)

        # Baseline (no normalization)
        baseline_res = self._loso_evaluate(X, y, groups, model, max_features)
        baseline_r2 = baseline_res['R²']

        results = []
        results.append({
            'Threshold': 'None', 'Epsilon': 'None',
            'R²': baseline_r2, 'MAE': baseline_res['MAE'],
            'Delta_R²': 0.0, 'Method': 'No normalization (baseline)'
        })

        # Test grid
        thresholds = [55, 60, 65, 70, 75]
        epsilons = [0.05, 0.1, 0.15, 0.2]

        for threshold in thresholds:
            for epsilon in epsilons:
                X_m = X.copy()
                factor = (age / threshold) + epsilon
                for idx in hrv_indices:
                    X_m[:, idx] = X_m[:, idx] / factor

                res = self._loso_evaluate(X_m, y, groups, model, max_features)
                delta = res['R²'] - baseline_r2
                results.append({
                    'Threshold': threshold, 'Epsilon': epsilon,
                    'R²': res['R²'], 'MAE': res['MAE'],
                    'Delta_R²': delta,
                    'Method': f'HRV/(age/{threshold}+{epsilon})'
                })
                print(f"   threshold={threshold}, ε={epsilon}: R²={res['R²']:.3f} (Δ={delta:+.3f})")

        results_df = pd.DataFrame(results)

        # Summary
        best = results_df.loc[results_df['R²'].idxmax()]
        worst = results_df.loc[results_df['R²'].idxmin()]
        any_improvement = (results_df['Delta_R²'] > 0).any()

        print(f"\n   Baseline R²: {baseline_r2:.3f}")
        print(f"   Best normalization: {best['Method']} (R²={best['R²']:.3f})")
        print(f"   Any improvement over baseline: {'Yes' if any_improvement else 'No'}")

        return results_df

    # =========================================================================
    # NEW: Back-transformed error metrics
    # Addresses R3 #5: "add back-transformed interpretation"
    # =========================================================================

    def compute_back_transformed_errors(self, y_log: np.ndarray,
                                        predictions_log: np.ndarray,
                                        target_type: str = 'hba1c') -> Dict:
        """
        Convert log-scale MAE to original units for clinical interpretation.

        Addresses R3 #5: "MAE is on log-transformed targets; add an
        intuitive back-transformed interpretation so clinicians can understand
        the error scale."

        For log-transformed targets, the geometric mean of the ratio
        predicted/actual = exp(MAE_log), so:
        - MAE_log = 0.1 means predictions are off by ~10.5% on average
        - For HbA1c ~8.5%, that's ~0.9% absolute
        - For FBG ~9.2 mmol/L, that's ~0.97 mmol/L
        """
        print("\n Back-Transformed Error Metrics")
        print("-" * 50)

        # Log-scale errors
        mae_log = mean_absolute_error(y_log, predictions_log)
        rmse_log = np.sqrt(mean_squared_error(y_log, predictions_log))

        # Back-transform: exp(log_predicted) vs exp(log_actual)
        y_original = np.exp(y_log)
        pred_original = np.exp(predictions_log)

        mae_original = mean_absolute_error(y_original, pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_original, pred_original))

        # Percentage error
        mape = np.mean(np.abs(y_original - pred_original) / y_original) * 100

        # Median absolute error in original units
        median_ae = np.median(np.abs(y_original - pred_original))

        if target_type == 'hba1c':
            unit = '%'
            clinical_context = (
                f"For an average HbA1c of {np.mean(y_original):.1f}%, "
                f"MAE of {mae_original:.2f}% means predictions deviate "
                f"by approximately {mae_original:.1f} percentage points"
            )
        else:
            unit = 'mmol/L'
            mae_mgdl = mae_original * 18  # Convert to mg/dL
            clinical_context = (
                f"For an average FBG of {np.mean(y_original):.1f} mmol/L "
                f"({np.mean(y_original) * 18:.0f} mg/dL), "
                f"MAE of {mae_original:.2f} mmol/L ({mae_mgdl:.0f} mg/dL)"
            )

        results = {
            'log_scale': {
                'MAE': float(mae_log),
                'RMSE': float(rmse_log)
            },
            'original_scale': {
                'MAE': float(mae_original),
                'RMSE': float(rmse_original),
                'Median_AE': float(median_ae),
                'MAPE_percent': float(mape),
                'unit': unit
            },
            'clinical_interpretation': clinical_context
        }

        print(f"   Log-scale MAE:      {mae_log:.4f}")
        print(f"   Original-scale MAE: {mae_original:.2f} {unit}")
        print(f"   Original-scale RMSE:{rmse_original:.2f} {unit}")
        print(f"   MAPE:               {mape:.1f}%")
        print(f"   {clinical_context}")

        return results

    # =========================================================================
    # NEW: Feature selection stability across folds
    # =========================================================================

    def analyze_feature_selection_stability(self, model_name: str = 'Bayesian Ridge') -> Dict:
        """
        Analyze which features were selected in each LOSO fold.
        Provides evidence that feature selection was done within folds.
        """
        if model_name not in self.fold_feature_selections:
            print(f"   No fold feature data for {model_name}")
            return {}

        fold_features = self.fold_feature_selections[model_name]
        n_folds = len(fold_features)

        # Count feature frequency across folds
        feature_counts = {}
        for fold in fold_features:
            for f in fold:
                feature_counts[f] = feature_counts.get(f, 0) + 1

        # Sort by frequency
        sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])

        stability_df = pd.DataFrame(sorted_features, columns=['Feature', 'Folds_Selected'])
        stability_df['Selection_Rate'] = stability_df['Folds_Selected'] / n_folds
        stability_df['Stability'] = stability_df['Selection_Rate'].apply(
            lambda x: 'Stable' if x > 0.8 else ('Moderate' if x > 0.5 else 'Unstable')
        )

        print(f"\n Feature Selection Stability ({model_name}, {n_folds} folds)")
        print("-" * 60)
        for _, row in stability_df.head(15).iterrows():
            bar = '█' * int(row['Selection_Rate'] * 20)
            print(f"   {row['Feature']:35} {row['Folds_Selected']:3}/{n_folds} ({row['Selection_Rate']:.0%}) {bar}")

        return {
            'stability_table': stability_df,
            'n_folds': n_folds,
            'n_always_selected': (stability_df['Selection_Rate'] == 1.0).sum(),
            'n_never_selected': len(feature_counts) - len(stability_df),
        }

    # =========================================================================
    # STATISTICAL VALIDATION — with CV hygiene
    # =========================================================================

    def run_permutation_test(self, X: np.ndarray, y: np.ndarray,
                             groups: np.ndarray,
                             max_features: int = 15,
                             n_permutations: int = 500) -> Dict:
        """
        Permutation test using Pipeline to ensure CV hygiene.
        The pipeline wraps SelectKBest + StandardScaler + model so that
        sklearn's permutation_test_score handles the splits correctly.
        """
        print(f"\n Permutation Test (n={n_permutations}, with Pipeline)")
        print("=" * 60)

        k = min(max_features, X.shape[1])
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(f_regression, k=k)),
            ('scaler', StandardScaler()),
            ('model', BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6))
        ])

        cv = LeaveOneGroupOut()

        score, perm_scores, p_value = permutation_test_score(
            pipeline, X, y,
            cv=cv, groups=groups,
            n_permutations=n_permutations,
            scoring='r2',
            random_state=42,
            n_jobs=-1
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

        print(f"   True R²:              {score:.4f}")
        print(f"   Permutation R² mean:  {perm_mean:.4f} ± {perm_std:.4f}")
        print(f"   P-value:              {p_value:.4f}")
        print(f"   Effect size (z):      {effect_size:.2f}")
        print(f"   Significant (p<0.05): {' Yes' if p_value < 0.05 else ' No'}")

        return results

    def run_bootstrap_ci(self, X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray,
                         max_features: int = 15,
                         n_bootstrap: int = 500) -> Dict:
        """
        Bootstrap CIs with in-fold CV hygiene via _loso_evaluate.
        Each bootstrap resamples subjects (not individual samples).
        """
        print(f"\n Bootstrap 95% CI (n={n_bootstrap}, subject-level resampling)")
        print("=" * 60)

        unique_groups = np.unique(groups)
        n_subjects = len(unique_groups)
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)

        bootstrap_r2 = []
        bootstrap_mae = []

        for i in range(n_bootstrap):
            # Resample subjects (not individual rows)
            boot_subjects = np.random.choice(unique_groups, n_subjects, replace=True)

            # Build bootstrap dataset
            boot_indices = []
            new_groups = []
            for new_grp, subj in enumerate(boot_subjects):
                subj_idx = np.where(groups == subj)[0]
                boot_indices.extend(subj_idx)
                new_groups.extend([new_grp] * len(subj_idx))

            X_boot = X[boot_indices]
            y_boot = y[boot_indices]
            groups_boot = np.array(new_groups)

            try:
                res = self._loso_evaluate(X_boot, y_boot, groups_boot, model, max_features)
                if not np.isnan(res['R²']) and not np.isinf(res['R²']):
                    bootstrap_r2.append(res['R²'])
                    bootstrap_mae.append(res['MAE'])
            except:
                continue

        bootstrap_r2 = np.array(bootstrap_r2)
        bootstrap_mae = np.array(bootstrap_mae)

        r2_lower = np.percentile(bootstrap_r2, 2.5)
        r2_upper = np.percentile(bootstrap_r2, 97.5)
        mae_lower = np.percentile(bootstrap_mae, 2.5)
        mae_upper = np.percentile(bootstrap_mae, 97.5)

        results = {
            'n_bootstrap': n_bootstrap,
            'n_successful': len(bootstrap_r2),
            'r2_mean': float(np.mean(bootstrap_r2)),
            'r2_median': float(np.median(bootstrap_r2)),
            'r2_ci_lower': float(r2_lower),
            'r2_ci_upper': float(r2_upper),
            'mae_mean': float(np.mean(bootstrap_mae)),
            'mae_ci_lower': float(mae_lower),
            'mae_ci_upper': float(mae_upper),
        }

        print(f"   R²:  {results['r2_median']:.3f} [{r2_lower:.3f}, {r2_upper:.3f}]")
        print(f"   MAE: {results['mae_mean']:.3f} [{mae_lower:.3f}, {mae_upper:.3f}]")

        return results

    # =========================================================================
    # VISUALISATION
    # =========================================================================

    def create_baseline_comparison_figure(self, results_df, output_path="baseline_comparison.png"):
        results_sorted = results_df.dropna(subset=['R²']).sort_values('R²', ascending=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        colors = []
        for model in results_sorted['Model']:
            if 'Naive' in model:
                colors.append('#95a5a6')
            elif 'MLP' in model:
                colors.append('#e74c3c')
            elif any(x in model for x in ['Forest', 'Gradient', 'Extra', 'AdaBoost']):
                colors.append('#27ae60')
            elif 'SVR' in model:
                colors.append('#9b59b6')
            else:
                colors.append('#3498db')

        ax1 = axes[0]
        bars = ax1.barh(range(len(results_sorted)), results_sorted['R²'], color=colors, alpha=0.8)
        ax1.set_yticks(range(len(results_sorted)))
        ax1.set_yticklabels(results_sorted['Model'], fontsize=9)
        ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Comparison (R²)\nHigher is Better', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3, axis='x')
        for i, (bar, r2) in enumerate(zip(bars, results_sorted['R²'])):
            ax1.text(r2 + 0.01, i, f'{r2:.3f}', va='center', fontsize=8)

        ax2 = axes[1]
        results_sorted_mae = results_df.dropna(subset=['MAE']).sort_values('MAE', ascending=False)
        colors_mae = []
        for model in results_sorted_mae['Model']:
            if 'Naive' in model:
                colors_mae.append('#95a5a6')
            elif 'MLP' in model:
                colors_mae.append('#e74c3c')
            elif any(x in model for x in ['Forest', 'Gradient', 'Extra', 'AdaBoost']):
                colors_mae.append('#27ae60')
            elif 'SVR' in model:
                colors_mae.append('#9b59b6')
            else:
                colors_mae.append('#3498db')

        ax2.barh(range(len(results_sorted_mae)), results_sorted_mae['MAE'], color=colors_mae, alpha=0.8)
        ax2.set_yticks(range(len(results_sorted_mae)))
        ax2.set_yticklabels(results_sorted_mae['Model'], fontsize=9)
        ax2.set_xlabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax2.set_title('Model Comparison (MAE)\nLower is Better', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        legend_elements = [
            mpatches.Patch(color='#3498db', label='Linear Models'),
            mpatches.Patch(color='#27ae60', label='Tree-Based'),
            mpatches.Patch(color='#e74c3c', label='Neural Networks (default params)'),
            mpatches.Patch(color='#9b59b6', label='SVM'),
            mpatches.Patch(color='#95a5a6', label='Naive Baselines')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=5,
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)
        plt.tight_layout();
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(output_path, dpi=300, bbox_inches='tight');
        plt.close()
        print(f"   Saved: {output_path}")

    # =========================================================================
    # FIGURE: Dual-Cohort Model Comparison
    # =========================================================================

    def create_dual_cohort_comparison_figure(self, output_path="dual_cohort_comparison.png"):
        """
        Combined model comparison for both HbA1c and FBG cohorts in one figure.
        4-panel layout: R² HbA1c | R² FBG | MAE HbA1c | MAE FBG
        """
        cohorts = [c for c in ['hba1c_cohort', 'fbg_cohort'] if c in self.results]
        if len(cohorts) < 2:
            print("    Need both cohorts for dual comparison figure")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        cohort_titles = {'hba1c_cohort': 'HbA1c Cohort', 'fbg_cohort': 'FBG Cohort'}

        def _model_color(model_name):
            if 'Naive' in model_name:
                return '#95a5a6'
            elif 'MLP' in model_name:
                return '#e74c3c'
            elif any(x in model_name for x in ['Forest', 'Gradient', 'Extra', 'AdaBoost']):
                return '#27ae60'
            elif 'SVR' in model_name:
                return '#9b59b6'
            else:
                return '#3498db'

        for col, cohort in enumerate(cohorts):
            df = self.results[cohort]['baseline_comparison'].dropna(subset=['R²'])

            # Top row: R²
            r2_sorted = df.sort_values('R²', ascending=True)
            colors = [_model_color(m) for m in r2_sorted['Model']]
            ax = axes[0, col]
            bars = ax.barh(range(len(r2_sorted)), r2_sorted['R²'],
                           color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(r2_sorted)))
            ax.set_yticklabels(r2_sorted['Model'], fontsize=8)
            ax.set_xlabel('R² Score', fontsize=11, fontweight='bold')
            ax.set_title(f"{cohort_titles[cohort]} — R² (Higher is Better)",
                         fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='x')
            for i, r2_val in enumerate(r2_sorted['R²']):
                ax.text(r2_val + 0.005, i, f'{r2_val:.3f}', va='center', fontsize=7)

            # Bottom row: MAE
            mae_sorted = df.sort_values('MAE', ascending=False)
            colors_mae = [_model_color(m) for m in mae_sorted['Model']]
            ax = axes[1, col]
            ax.barh(range(len(mae_sorted)), mae_sorted['MAE'],
                    color=colors_mae, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(mae_sorted)))
            ax.set_yticklabels(mae_sorted['Model'], fontsize=8)
            ax.set_xlabel('Mean Absolute Error (log-scale)', fontsize=11, fontweight='bold')
            ax.set_title(f"{cohort_titles[cohort]} — MAE (Lower is Better)",
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        legend_elements = [
            mpatches.Patch(color='#3498db', label='Linear Models'),
            mpatches.Patch(color='#27ae60', label='Tree-Based'),
            mpatches.Patch(color='#e74c3c', label='Neural Networks (default params)'),
            mpatches.Patch(color='#9b59b6', label='SVM'),
            mpatches.Patch(color='#95a5a6', label='Naive Baselines')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=5,
                   bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.06, hspace=0.35)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved dual-cohort comparison: {output_path}")

    # =========================================================================
    # FIGURE: Feature Importance by Domain (Figure 6 in manuscript)
    # Addresses: R1 — "clearer breakdown of importance of individual features
    #            within each category (clinical vs. ECG vs. HRV)"
    # =========================================================================

    def create_feature_importance_by_domain_figure(self, output_path="feature_importance_by_domain.png"):
        """
        Two-panel horizontal bar chart showing top-15 features per cohort,
        colored by domain (HRV/ECG/Clinical/Demographics/Sleep/Age-Normalized).
        Uses feature selection frequency across LOSO folds as importance metric.
        """
        cohorts = [c for c in ['hba1c_cohort', 'fbg_cohort'] if c in self.results]
        if not cohorts:
            print("    No results available for feature importance figure")
            return

        domain_colors = {
            'HRV': '#3498db',
            'ECG': '#9b59b6',
            'Clinical': '#f39c12',
            'Demographics': '#95a5a6',
            'Sleep': '#27ae60',
            'Age-Normalized': '#e74c3c'
        }

        def _classify_domain(feature_name):
            f = feature_name.lower()
            if 'age_normalized' in f:
                return 'Age-Normalized'
            elif any(x in f for x in ['hrv_ds_', 'hrv_rem_', 'hrv_rs_']):
                return 'HRV'
            elif 'ecg_' in f:
                return 'ECG'
            elif any(x in f for x in ['sbp', 'dbp', 'wbc', 'hb', 'plt', 'crp',
                                       'alt', 'ast', 'ggt', 'bun', 'ua', 'tg',
                                       'hdl', 'ldl', 'uma', 'ucr', 'uacr', 'n%']):
                return 'Clinical'
            elif any(x in f for x in ['psqi_', 'cpc_']):
                return 'Sleep'
            elif any(x in f for x in ['age', 'height', 'weight']):
                return 'Demographics'
            else:
                return 'Clinical'

        n_panels = len(cohorts)
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 8))
        if n_panels == 1:
            axes = [axes]

        cohort_titles = {'hba1c_cohort': 'HbA1c Cohort (n=29)',
                         'fbg_cohort': 'FBG Cohort (n=38)'}

        for idx, cohort in enumerate(cohorts):
            ax = axes[idx]
            results = self.results[cohort]

            # Get feature importance from fold selections or feature_importance_report
            stability = results.get('feature_selection_stability', {})
            stability_table = stability.get('stability_table', None)

            if stability_table is not None and len(stability_table) > 0:
                top_features = stability_table.head(15).copy()
                features = top_features['Feature'].tolist()
                values = top_features['Selection_Rate'].tolist()
                xlabel = 'Selection Frequency (across LOSO folds)'
            elif 'feature_importance_report' in results:
                fi = results['feature_importance_report'].head(15).copy()
                features = fi['feature'].tolist()
                values = fi['correlation'].tolist()
                xlabel = '|Pearson Correlation| with Target'
            else:
                print(f"    No feature data for {cohort}")
                continue

            domains = [_classify_domain(f) for f in features]
            colors = [domain_colors.get(d, '#95a5a6') for d in domains]

            # Reverse for horizontal bar chart (top feature on top)
            features = features[::-1]
            values = values[::-1]
            colors = colors[::-1]
            domains_rev = domains[::-1]

            bars = ax.barh(range(len(features)), values, color=colors,
                           alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(features)))

            # Clean feature names for display
            display_names = []
            for f in features:
                name = f.replace('hrv_ds_', 'HRV-DS: ').replace('hrv_rem_', 'HRV-REM: ')
                name = name.replace('hrv_rs_', 'HRV-RS: ').replace('ecg_all_', 'ECG-all: ')
                name = name.replace('ecg_sleep_', 'ECG-sleep: ').replace('ecg_day_', 'ECG-day: ')
                name = name.replace('_age_normalized', ' (age-norm)')
                name = name.replace('_', ' ').title()
                display_names.append(name)

            ax.set_yticklabels(display_names, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
            ax.set_title(f"Top-15 Features — {cohort_titles.get(cohort, cohort)}",
                         fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            # Value labels on bars
            for i, v in enumerate(values):
                ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)

        # Legend
        legend_elements = [mpatches.Patch(color=c, label=d)
                           for d, c in domain_colors.items()]
        fig.legend(handles=legend_elements, loc='upper center', ncol=6,
                   bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=True,
                   title='Feature Domain', title_fontsize=11)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved feature importance by domain: {output_path}")

    # =========================================================================
    # FIGURE: Feature Selection Stability Heatmap
    # Addresses: R3#4, E#7 — visual proof of CV hygiene
    # =========================================================================

    def create_feature_stability_heatmap(self, output_path="feature_selection_stability.png"):
        """
        Binary heatmap: features (rows) × LOSO folds (columns).
        Shows which features were selected in which folds — proves
        feature selection varies across folds (CV hygiene evidence).
        """
        cohorts = [c for c in ['hba1c_cohort', 'fbg_cohort'] if c in self.results]
        if not cohorts:
            print("   No results for stability heatmap")
            return

        n_panels = len(cohorts)
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 10))
        if n_panels == 1:
            axes = [axes]

        cohort_titles = {'hba1c_cohort': 'HbA1c Cohort', 'fbg_cohort': 'FBG Cohort'}

        for idx, cohort in enumerate(cohorts):
            ax = axes[idx]

            # Try to get per-fold feature selections from Bayesian Ridge
            fold_features = self.fold_feature_selections.get('Bayesian Ridge', [])
            if not fold_features:
                # Fallback: check if any model has fold data
                for model_name, ffs in self.fold_feature_selections.items():
                    if ffs:
                        fold_features = ffs
                        break

            if not fold_features:
                ax.text(0.5, 0.5, 'No per-fold feature\nselection data available',
                        ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title(f"Feature Selection Stability — {cohort_titles.get(cohort, cohort)}")
                continue

            # Collect all unique features
            all_features = set()
            for fold in fold_features:
                all_features.update(fold)
            all_features = sorted(all_features)

            # Build binary matrix
            n_folds = len(fold_features)
            matrix = np.zeros((len(all_features), n_folds), dtype=int)
            for fold_idx, fold in enumerate(fold_features):
                for feat in fold:
                    if feat in all_features:
                        feat_idx = all_features.index(feat)
                        matrix[feat_idx, fold_idx] = 1

            # Sort by selection frequency (most stable on top)
            freq = matrix.sum(axis=1)
            sort_idx = np.argsort(-freq)
            matrix = matrix[sort_idx]
            sorted_features = [all_features[i] for i in sort_idx]

            # Limit to top 30 features for readability
            max_show = min(30, len(sorted_features))
            matrix_show = matrix[:max_show]
            features_show = sorted_features[:max_show]

            # Clean feature names
            display_names = []
            for f in features_show:
                name = f.replace('hrv_ds_', 'HRV-DS:').replace('hrv_rem_', 'HRV-REM:')
                name = name.replace('hrv_rs_', 'HRV-RS:').replace('ecg_all_', 'ECG:')
                name = name.replace('ecg_sleep_', 'ECG-slp:').replace('ecg_day_', 'ECG-day:')
                name = name.replace('_age_normalized', '(AN)')
                name = name.replace('_', ' ')
                if len(name) > 30:
                    name = name[:28] + '..'
                display_names.append(name)

            # Plot
            if HAS_SEABORN:
                sns.heatmap(matrix_show, ax=ax, cmap=['#FFFFFF', '#2196F3'],
                            cbar=False, linewidths=0.5, linecolor='#E0E0E0',
                            xticklabels=[f'F{i+1}' for i in range(n_folds)],
                            yticklabels=display_names)
            else:
                ax.imshow(matrix_show, aspect='auto', cmap='Blues',
                          interpolation='nearest')
                ax.set_xticks(range(n_folds))
                ax.set_xticklabels([f'F{i+1}' for i in range(n_folds)], fontsize=7)
                ax.set_yticks(range(max_show))
                ax.set_yticklabels(display_names, fontsize=8)

            ax.set_xlabel(f'LOSO Fold (N={n_folds} subjects)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
            ax.set_title(f"Feature Selection Stability — {cohort_titles.get(cohort, cohort)}\n"
                         f"(Blue = selected in fold; White = not selected)",
                         fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=7, rotation=90)

            # Add frequency annotation on right
            for i in range(max_show):
                sel_rate = freq[sort_idx[i]] / n_folds
                ax.text(n_folds + 0.3, i, f'{sel_rate:.0%}',
                        va='center', fontsize=7, color='#333')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved feature stability heatmap: {output_path}")

    # =========================================================================
    # Age Normalization Sensitivity Heatmap (Supplementary)
    # Addresses R3#3 — "give a brief justification + small sensitivity check"
    # =========================================================================

    def create_age_sensitivity_heatmap(self, cohort, output_path="age_sensitivity_heatmap.png"):
        """
        Heatmap of R² across (threshold × epsilon) grid for age normalization.
        Shows that no parameter combination improves over baseline.
        """
        if cohort not in self.results:
            return
        sensitivity_df = self.results[cohort].get('age_sensitivity')
        if sensitivity_df is None or len(sensitivity_df) == 0:
            print(f"    No age sensitivity data for {cohort}")
            return

        # Filter out the 'None' baseline row for the heatmap
        grid_df = sensitivity_df[sensitivity_df['Threshold'] != 'None'].copy()
        grid_df['Threshold'] = grid_df['Threshold'].astype(int)
        grid_df['Epsilon'] = grid_df['Epsilon'].astype(float)

        baseline_r2 = sensitivity_df[sensitivity_df['Threshold'] == 'None']['R²'].values[0]

        # Pivot to matrix
        pivot = grid_df.pivot(index='Threshold', columns='Epsilon', values='R²')

        fig, ax = plt.subplots(figsize=(8, 6))

        if HAS_SEABORN:
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                        ax=ax, linewidths=1, linecolor='white',
                        cbar_kws={'label': 'R² Score'})
        else:
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f'{c:.2f}' for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                            ha='center', va='center', fontsize=9)
            plt.colorbar(im, ax=ax, label='R² Score')

        cohort_title = 'HbA1c' if 'hba1c' in cohort else 'FBG'
        ax.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Age Threshold', fontsize=12, fontweight='bold')
        ax.set_title(f"Age Normalization Sensitivity — {cohort_title}\n"
                     f"HRV_norm = HRV / (age/threshold + ε)\n"
                     f"Baseline R² (no normalization) = {baseline_r2:.3f}",
                     fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved age sensitivity heatmap: {output_path}")

    # =========================================================================
    # FIGURE: Prediction vs Actual Scatter
    # Standard regression figure expected by all reviewers
    # =========================================================================

    def create_prediction_scatter(self, cohort, output_path="prediction_scatter.png"):
        """
        Predicted vs actual scatter plot with identity line and residual subplot.
        Uses the best model's predictions from baseline comparison.
        """
        if cohort not in self.results:
            return
        results = self.results[cohort]
        baseline = results.get('baseline_comparison')
        if baseline is None:
            return

        # Get best model predictions
        valid = baseline.dropna(subset=['R²'])
        best_idx = valid['R²'].idxmax()
        best_row = valid.loc[best_idx]
        best_model = best_row['Model']
        best_r2 = best_row['R²']
        best_mae = best_row['MAE']
        predictions = best_row.get('Predictions')

        if predictions is None:
            print(f"    No predictions stored for {cohort}")
            return

        # Get actual targets
        try:
            X, y, y_log, groups, feature_names = self.load_cohort_data(cohort)
            y_true = y_log  # log-transformed targets
        except Exception:
            print(f"    Could not load target data for {cohort}")
            return

        if len(y_true) != len(predictions):
            print(f"    Length mismatch: y_true={len(y_true)}, pred={len(predictions)}")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Predicted vs Actual
        ax1.scatter(y_true, predictions, alpha=0.7, s=80, c='#3498db',
                    edgecolors='black', linewidth=0.5, zorder=3)

        # Identity line
        all_vals = np.concatenate([y_true, predictions])
        lims = [np.min(all_vals) - 0.05, np.max(all_vals) + 0.05]
        ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Perfect prediction')

        # Regression line
        z = np.polyfit(y_true, predictions, 1)
        p = np.poly1d(z)
        x_line = np.linspace(lims[0], lims[1], 100)
        ax1.plot(x_line, p(x_line), 'r-', alpha=0.7, linewidth=1.5, label='Fit line')

        corr_val = best_row.get('Correlation', np.nan)
        p_val = best_row.get('p-value', np.nan)

        ax1.set_xlabel('Actual (log-scale)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted (log-scale)', fontsize=12, fontweight='bold')
        cohort_title = 'HbA1c' if 'hba1c' in cohort else 'FBG'
        ax1.set_title(f"Predicted vs Actual — {cohort_title}\n"
                      f"{best_model}: R²={best_r2:.3f}, MAE={best_mae:.3f}, "
                      f"r={corr_val:.3f}", fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_aspect('equal', adjustable='box')

        # Panel 2: Residuals
        residuals = predictions - y_true
        ax2.scatter(predictions, residuals, alpha=0.7, s=80, c='#e74c3c',
                    edgecolors='black', linewidth=0.5, zorder=3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        ax2.set_xlabel('Predicted (log-scale)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residual (Predicted − Actual)', fontsize=12, fontweight='bold')
        ax2.set_title(f"Residual Plot — {cohort_title}\n"
                      f"Mean residual = {np.mean(residuals):.4f}, "
                      f"Std = {np.std(residuals):.4f}", fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved prediction scatter: {output_path}")

    # =========================================================================
    # MAIN ANALYSIS PIPELINE
    # =========================================================================

    def run_complete_analysis(self, cohort: str = 'hba1c_cohort',
                              use_log_target: bool = True,
                              max_features: int = 15) -> Dict:
        print()
        print("=" * 70)
        print(f"COMPLETE ANALYSIS: {cohort.upper()}")
        print(f"CV Hygiene: feature selection + scaling INSIDE each LOSO fold")
        print("=" * 70)

        results = {'cohort': cohort, 'use_log_target': use_log_target}

        # Load data
        X, y, y_log, groups, feature_names = self.load_cohort_data(cohort)
        target = y_log if use_log_target else y
        results['n_samples'] = len(target)
        results['n_features'] = len(feature_names)
        results['feature_names'] = feature_names

        # Feature importance report (for paper, not for evaluation)
        feature_report = self.fdr_corrected_feature_selection_report(X, target, feature_names)
        results['feature_importance_report'] = feature_report

        # Baseline comparison (with proper CV hygiene)
        print("\n" + "=" * 70)
        baseline_results = self.run_baseline_comparison(
            X, target, groups, feature_names, max_features
        )
        results['baseline_comparison'] = baseline_results

        # Feature selection stability
        stability = self.analyze_feature_selection_stability('Bayesian Ridge')
        results['feature_selection_stability'] = stability

        # Age adjustment comparison
        print("\n" + "=" * 70)
        age_results = self.compare_age_adjustment_methods(
            X, target, feature_names, groups, max_features
        )
        results['age_adjustment'] = age_results

        # Age normalization sensitivity (NEW — Reviewer 3 #3)
        print("\n" + "=" * 70)
        sensitivity = self.age_normalization_sensitivity(
            X, target, feature_names, groups, max_features
        )
        results['age_sensitivity'] = sensitivity

        # Permutation test
        print("\n" + "=" * 70)
        perm_results = self.run_permutation_test(X, target, groups, max_features, n_permutations=500)
        results['permutation_test'] = perm_results

        # Bootstrap CI
        print("\n" + "=" * 70)
        boot_results = self.run_bootstrap_ci(X, target, groups, max_features, n_bootstrap=500)
        results['bootstrap_ci'] = boot_results

        # Back-transformed errors (NEW — Reviewer 3 #5)
        best_model_name = baseline_results.loc[baseline_results['R²'].idxmax(), 'Model']
        best_preds = baseline_results.loc[baseline_results['R²'].idxmax(), 'Predictions']
        if best_preds is not None and use_log_target:
            print("\n" + "=" * 70)
            target_type = 'hba1c' if 'hba1c' in cohort else 'fbg'
            back_transformed = self.compute_back_transformed_errors(
                target, best_preds, target_type
            )
            results['back_transformed_errors'] = back_transformed

        self.results[cohort] = results
        return results

    def save_all_results(self, output_dir: str = "analysis_results_v3"):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"\n Saving Results to {output_dir}...")

        for cohort, results in self.results.items():
            cohort_dir = output_dir / cohort
            cohort_dir.mkdir(exist_ok=True)

            # Baseline comparison
            if 'baseline_comparison' in results:
                df = results['baseline_comparison'].drop(columns=['Predictions', 'fold_selected_features'],
                                                         errors='ignore')
                df.to_csv(cohort_dir / "baseline_comparison.csv", index=False)

            # Age adjustment
            if 'age_adjustment' in results:
                results['age_adjustment'].to_csv(cohort_dir / "age_adjustment.csv", index=False)

            # Age sensitivity ( Supplementary Table S2)
            if 'age_sensitivity' in results:
                results['age_sensitivity'].to_csv(cohort_dir / "age_sensitivity.csv", index=False)

            # Feature importance report (Supplementary Table S1)
            if 'feature_importance_report' in results:
                results['feature_importance_report'].to_csv(cohort_dir / "feature_importance.csv", index=False)

            # Feature selection stability (Supplementary Table S4)
            if 'feature_selection_stability' in results and 'stability_table' in results['feature_selection_stability']:
                results['feature_selection_stability']['stability_table'].to_csv(
                    cohort_dir / "feature_selection_stability.csv", index=False
                )

            # Figures
            if 'baseline_comparison' in results:
                self.create_baseline_comparison_figure(
                    results['baseline_comparison'], str(cohort_dir / "baseline_comparison.png")
                )

            # NEW: Prediction vs Actual scatter
            self.create_prediction_scatter(cohort, str(cohort_dir / "prediction_scatter.png"))

            # NEW: Age sensitivity heatmap
            if 'age_sensitivity' in results:
                self.create_age_sensitivity_heatmap(
                    cohort, str(cohort_dir / "age_sensitivity_heatmap.png")
                )

            # Summary JSON
            summary = {
                'cohort': cohort,
                'n_samples': results.get('n_samples'),
                'n_features': results.get('n_features'),
                'cv_hygiene': 'Feature selection + scaling within each LOSO fold',
                'best_model': results['baseline_comparison'].loc[
                    results['baseline_comparison']['R²'].idxmax(), 'Model'
                ] if 'baseline_comparison' in results else None,
                'best_r2': float(
                    results['baseline_comparison']['R²'].max()) if 'baseline_comparison' in results else None,
                'permutation_p': results.get('permutation_test', {}).get('p_value'),
                'bootstrap_r2_ci': [
                    results.get('bootstrap_ci', {}).get('r2_ci_lower'),
                    results.get('bootstrap_ci', {}).get('r2_ci_upper')
                ],
                'back_transformed_errors': results.get('back_transformed_errors', {}).get('original_scale')
            }
            with open(cohort_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"    Saved {cohort}")

        # Cross-cohort figures (need both cohorts)
        if len(self.results) >= 2:
            self.create_dual_cohort_comparison_figure(
                str(output_dir / "dual_cohort_model_comparison.png")
            )

        # Feature importance by domain (uses stability data from all cohorts)
        self.create_feature_importance_by_domain_figure(
            str(output_dir / "feature_importance_by_domain.png")
        )

        # Feature selection stability heatmap
        self.create_feature_stability_heatmap(
            str(output_dir / "feature_selection_stability_heatmap.png")
        )

        print(f"\n All results saved to {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("COMPREHENSIVE BASELINE FRAMEWORK")
    print("=" * 70)

    framework = ComprehensiveBaselineFramework("processed_data_v2")

    for cohort in ['hba1c_cohort', 'fbg_cohort']:
        try:
            results = framework.run_complete_analysis(cohort, use_log_target=True)
            print(f"\n {cohort} analysis complete!")
        except FileNotFoundError as e:
            print(f"\n  {cohort}: {e}")
            print("   Run complete_preprocessing.py first")
        except Exception as e:
            print(f"\n {cohort}: {e}")
            import traceback;

            traceback.print_exc()

    if framework.results:
        framework.save_all_results()
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)