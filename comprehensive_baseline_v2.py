#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

# Sklearn imports
from sklearn.model_selection import (
    KFold, LeaveOneGroupOut, cross_val_predict,
    permutation_test_score, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
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

# Statistical imports
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# Publication-quality plot settings
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

    def __init__(self, data_dir: str = "processed_data_v2"):
        self.data_dir = Path(data_dir)
        self.results = {}
        self.feature_importance = {}

        print("=" * 70)
        print("COMPREHENSIVE BASELINE FRAMEWORK v2.0")
        print("=" * 70)
        print()

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def load_cohort_data(self, cohort_name: str = 'hba1c_cohort') -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load data for a specific cohort.

        Parameters:
        -----------
        cohort_name : str
            Either 'hba1c_cohort' or 'fbg_cohort'

        Returns:
        --------
        X, y, groups, feature_names
        """
        print(f" Loading {cohort_name} data...")

        # Try LOSO splits first (preferred)
        loso_dir = self.data_dir / "loso_splits" / cohort_name

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
            # Fallback to features.csv and targets
            print("     LOSO splits not found, loading from CSV...")

            features_df = pd.read_csv(self.data_dir / "features.csv")
            targets_df = pd.read_csv(self.data_dir / "targets" / f"{cohort_name}.csv")

            # Merge and prepare
            cohort_subjects = targets_df['subject_id'].tolist()
            cohort_features = features_df[features_df['subject_id'].isin(cohort_subjects)]

            # Exclude non-feature columns
            exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                           [col for col in features_df.columns if any(term in col.lower()
                                                                      for term in
                                                                      ['fbg', 'hba1c', 'diabetic', 'coronary',
                                                                       'carotid', 'glucose'])]

            feature_cols = [col for col in cohort_features.columns if col not in exclude_cols]

            X = cohort_features[feature_cols].fillna(0).values
            y = targets_df['target_value'].values
            y_log = targets_df['log_target'].values

            # Create groups
            subject_ids = targets_df['subject_id'].values
            unique_subjects = list(set(subject_ids))
            subject_to_group = {s: i for i, s in enumerate(unique_subjects)}
            groups = np.array([subject_to_group[s] for s in subject_ids])

            print(f"    Loaded: {X.shape[0]} samples, {X.shape[1]} features")

            return X, y, y_log, groups, feature_cols

    # =========================================================================
    # FEATURE SELECTION WITH FDR CORRECTION
    # =========================================================================

    def fdr_corrected_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str],
                                        p_threshold: float = 0.1,
                                        max_features: int = 15) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        """
        Feature selection with FDR (False Discovery Rate) correction.

        Addresses statistical rigor concerns by properly controlling for
        multiple comparisons.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        feature_names : list
            Names of features
        p_threshold : float
            FDR-corrected p-value threshold
        max_features : int
            Maximum number of features to select

        Returns:
        --------
        X_selected, feature_importance_df, selected_feature_names
        """
        print(" FDR-Corrected Feature Selection...")

        correlations = []
        p_values_raw = []

        for i, col in enumerate(feature_names):
            try:
                # Handle constant features
                if np.std(X[:, i]) == 0:
                    correlations.append(0)
                    p_values_raw.append(1.0)
                    continue

                # Remove NaN pairs
                mask = ~np.isnan(X[:, i]) & ~np.isnan(y)
                if mask.sum() < 10:
                    correlations.append(0)
                    p_values_raw.append(1.0)
                    continue

                corr, p_val = pearsonr(X[mask, i], y[mask])
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
                p_values_raw.append(p_val if not np.isnan(p_val) else 1.0)

            except Exception:
                correlations.append(0)
                p_values_raw.append(1.0)

        # Apply FDR correction (Benjamini-Hochberg)
        reject, p_values_fdr, _, _ = multipletests(
            p_values_raw,
            alpha=p_threshold,
            method='fdr_bh'
        )

        # Create importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'correlation': correlations,
            'p_value_raw': p_values_raw,
            'p_value_fdr': p_values_fdr,
            'significant_fdr': reject
        }).sort_values('correlation', ascending=False)

        # Select features
        significant_features = feature_importance_df[
            feature_importance_df['p_value_fdr'] < p_threshold
            ]

        if len(significant_features) >= 5:
            selected_df = significant_features.head(max_features)
        else:
            # Fallback: use top features by correlation
            print(f"     Only {len(significant_features)} significant features at FDR-corrected p<{p_threshold}")
            print(f"      Using top {max_features} by correlation")
            selected_df = feature_importance_df.head(max_features)

        selected_feature_names = selected_df['feature'].tolist()
        selected_indices = [feature_names.index(f) for f in selected_feature_names]
        X_selected = X[:, selected_indices]

        print(f"    Selected {len(selected_feature_names)} features")
        print(f"   FDR correction applied (Benjamini-Hochberg)")

        # Store for later use
        self.feature_importance = feature_importance_df

        return X_selected, feature_importance_df, selected_feature_names

    # =========================================================================
    # COMPREHENSIVE BASELINE MODELS
    # =========================================================================

    def get_all_baseline_models(self) -> Dict:

        models = {
            # ===== NAIVE BASELINES =====
            'Naive (Mean)': DummyRegressor(strategy='mean'),
            'Naive (Median)': DummyRegressor(strategy='median'),

            # ===== LINEAR MODELS =====
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Ridge (α=0.1)': Ridge(alpha=0.1),
            'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=5000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            'Bayesian Ridge': BayesianRidge(
                alpha_1=1e-6, alpha_2=1e-6,
                lambda_1=1e-6, lambda_2=1e-6
            ),
            'Huber Regressor': HuberRegressor(max_iter=1000),

            # ===== TREE-BASED MODELS =====
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=5,
                min_samples_leaf=3, random_state=42
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=100, max_depth=5,
                min_samples_leaf=3, random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=3,
                learning_rate=0.05, random_state=42
            ),
            'AdaBoost': AdaBoostRegressor(
                n_estimators=50, learning_rate=0.1, random_state=42
            ),

            # ===== SVM =====
            'SVR (RBF)': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'SVR (Linear)': SVR(kernel='linear', C=1.0),
            'SVR (Poly)': SVR(kernel='poly', degree=2, C=1.0),

            # ===== NEURAL NETWORKS  =====
            'MLP (32)': MLPRegressor(
                hidden_layer_sizes=(32,),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=2000,
                early_stopping=True,
                random_state=42
            ),
            'MLP (64, 32)': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=2000,
                early_stopping=True,
                random_state=42
            ),
            'MLP (128, 64, 32)': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=2000,
                early_stopping=True,
                random_state=42
            ),
            'MLP (64, 32) tanh': MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='tanh',
                solver='adam',
                alpha=0.01,
                max_iter=2000,
                early_stopping=True,
                random_state=42
            ),
        }

        return models

    def run_baseline_comparison(self, X: np.ndarray, y: np.ndarray,
                                groups: np.ndarray = None,
                                validation: str = 'loso') -> pd.DataFrame:
        """
        Run comprehensive baseline comparison.

        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (already selected)
        y : np.ndarray
            Target values
        groups : np.ndarray
            Subject groups for LOSO
        validation : str
            'loso', 'kfold', or 'temporal'

        Returns:
        --------
        DataFrame with results for all models
        """
        print(f"\n Running Baseline Comparison ({validation.upper()} validation)")
        print("=" * 60)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Set up cross-validation
        if validation == 'loso' and groups is not None:
            cv = LeaveOneGroupOut()
            cv_splits = list(cv.split(X_scaled, y, groups))
            cv_name = 'LOSO'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_splits = list(cv.split(X_scaled, y))
            cv_name = '5-Fold CV'

        print(f"   Validation: {cv_name} ({len(cv_splits)} splits)")

        models = self.get_all_baseline_models()
        results = []

        for name, model in models.items():
            try:
                # Get predictions
                if validation == 'loso' and groups is not None:
                    predictions = cross_val_predict(model, X_scaled, y, cv=cv, groups=groups)
                else:
                    predictions = cross_val_predict(model, X_scaled, y, cv=cv)

                # Calculate metrics
                r2 = r2_score(y, predictions)
                mae = mean_absolute_error(y, predictions)
                rmse = np.sqrt(mean_squared_error(y, predictions))

                # Correlation
                corr, p_val = pearsonr(y, predictions)

                results.append({
                    'Model': name,
                    'R²': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Correlation': corr,
                    'p-value': p_val,
                    'Predictions': predictions
                })

                print(f"   {name:25} | R²: {r2:7.3f} | MAE: {mae:.3f} | r: {corr:.3f}")

            except Exception as e:
                print(f"   {name:25} | ERROR: {str(e)[:30]}")
                results.append({
                    'Model': name,
                    'R²': np.nan,
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'Correlation': np.nan,
                    'p-value': np.nan,
                    'Error': str(e)
                })

        results_df = pd.DataFrame(results)

        # Summary
        valid_results = results_df[results_df['R²'].notna()]
        if len(valid_results) > 0:
            best_model = valid_results.loc[valid_results['R²'].idxmax()]
            print()
            print(f"    Best Model: {best_model['Model']} (R² = {best_model['R²']:.3f})")

        return results_df

    # =========================================================================
    # AGE ADJUSTMENT COMPARISON
    # =========================================================================

    def compare_age_adjustment_methods(self, X: np.ndarray, y: np.ndarray,
                                       feature_names: List[str],
                                       groups: np.ndarray = None) -> pd.DataFrame:

        print("\n Comparing Age Adjustment Methods")
        print("=" * 60)

        # Find age column index
        age_idx = None
        for i, name in enumerate(feature_names):
            if name == 'age':
                age_idx = i
                break

        if age_idx is None:
            print("    Age column not found!")
            return None

        age = X[:, age_idx]

        # Find HRV mean_rr columns
        hrv_indices = [i for i, name in enumerate(feature_names)
                       if 'hrv_' in name and 'mean_rr' in name and 'age_normalized' not in name]

        print(f"   Found {len(hrv_indices)} HRV mean_rr features to adjust")

        scaler = StandardScaler()
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)

        if groups is not None:
            cv = LeaveOneGroupOut()
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        results = []

        # Method 1: No adjustment (baseline)
        X_method = X.copy()
        X_scaled = scaler.fit_transform(X_method)
        if groups is not None:
            pred = cross_val_predict(model, X_scaled, y, cv=cv, groups=groups)
        else:
            pred = cross_val_predict(model, X_scaled, y, cv=cv)
        r2 = r2_score(y, pred)
        results.append({'Method': 'No Adjustment', 'R²': r2, 'Description': 'Baseline - no age adjustment'})
        print(f"   No Adjustment:              R² = {r2:.3f}")

        # Method 2: Your proposed method: HRV / (age/65 + 0.1)
        X_method = X.copy()
        age_norm_factor = (age / 65.0) + 0.1
        for idx in hrv_indices:
            X_method[:, idx] = X_method[:, idx] / age_norm_factor
        X_scaled = scaler.fit_transform(X_method)
        if groups is not None:
            pred = cross_val_predict(model, X_scaled, y, cv=cv, groups=groups)
        else:
            pred = cross_val_predict(model, X_scaled, y, cv=cv)
        r2_your = r2_score(y, pred)
        results.append({'Method': 'Your Method (HRV/(age/65+0.1))', 'R²': r2_your,
                        'Description': 'Proposed age normalization'})
        print(f"   Your Method:                R² = {r2_your:.3f}")

        # Method 3: Residualization
        X_method = X.copy()
        for idx in hrv_indices:
            # Fit: HRV = a*age + b
            age_model = LinearRegression()
            age_model.fit(age.reshape(-1, 1), X[:, idx])
            # Residual = HRV - predicted
            X_method[:, idx] = X[:, idx] - age_model.predict(age.reshape(-1, 1))
        X_scaled = scaler.fit_transform(X_method)
        if groups is not None:
            pred = cross_val_predict(model, X_scaled, y, cv=cv, groups=groups)
        else:
            pred = cross_val_predict(model, X_scaled, y, cv=cv)
        r2 = r2_score(y, pred)
        results.append({'Method': 'Residualization', 'R²': r2,
                        'Description': 'Regress age out of HRV'})
        print(f"   Residualization:            R² = {r2:.3f}")

        # Method 4: Z-score within age bins
        X_method = X.copy()
        age_bins = pd.cut(age, bins=[0, 40, 55, 70, 100], labels=['young', 'middle', 'senior', 'elderly'])
        for idx in hrv_indices:
            temp_df = pd.DataFrame({'hrv': X[:, idx], 'age_bin': age_bins})
            group_mean = temp_df.groupby('age_bin')['hrv'].transform('mean')
            group_std = temp_df.groupby('age_bin')['hrv'].transform('std')
            X_method[:, idx] = (X[:, idx] - group_mean) / (group_std + 1e-6)
        X_scaled = scaler.fit_transform(np.nan_to_num(X_method))
        if groups is not None:
            pred = cross_val_predict(model, X_scaled, y, cv=cv, groups=groups)
        else:
            pred = cross_val_predict(model, X_scaled, y, cv=cv)
        r2 = r2_score(y, pred)
        results.append({'Method': 'Age-Bin Z-Score', 'R²': r2,
                        'Description': 'Z-score within age quartiles'})
        print(f"   Age-Bin Z-Score:            R² = {r2:.3f}")

        # Method 5: Polynomial age interaction
        X_method = np.column_stack([X, age ** 2])  # Add age squared
        for idx in hrv_indices:
            X_method = np.column_stack([X_method, X[:, idx] * age])  # HRV × age
        X_scaled = scaler.fit_transform(X_method)
        if groups is not None:
            pred = cross_val_predict(model, X_scaled, y, cv=cv, groups=groups)
        else:
            pred = cross_val_predict(model, X_scaled, y, cv=cv)
        r2 = r2_score(y, pred)
        results.append({'Method': 'Polynomial Interaction', 'R²': r2,
                        'Description': 'Age² + HRV×age interactions'})
        print(f"   Polynomial Interaction:     R² = {r2:.3f}")

        # Method 6: Simple division by age
        X_method = X.copy()
        for idx in hrv_indices:
            X_method[:, idx] = X_method[:, idx] / (age + 1)
        X_scaled = scaler.fit_transform(X_method)
        if groups is not None:
            pred = cross_val_predict(model, X_scaled, y, cv=cv, groups=groups)
        else:
            pred = cross_val_predict(model, X_scaled, y, cv=cv)
        r2 = r2_score(y, pred)
        results.append({'Method': 'Simple Division (HRV/age)', 'R²': r2,
                        'Description': 'Divide HRV by age'})
        print(f"   Simple Division (HRV/age):  R² = {r2:.3f}")

        results_df = pd.DataFrame(results)

        # Calculate improvement from your method
        baseline_r2 = results_df[results_df['Method'] == 'No Adjustment']['R²'].values[0]
        improvement = r2_your - baseline_r2
        improvement_pct = (improvement / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0

        print()
        print(f"   Your method improvement: {improvement:+.3f} R² ({improvement_pct:+.1f}%)")

        best_method = results_df.loc[results_df['R²'].idxmax()]
        print(f"   Best method: {best_method['Method']} (R² = {best_method['R²']:.3f})")

        return results_df

    # =========================================================================
    # STATISTICAL VALIDATION
    # =========================================================================

    def run_permutation_test(self, X: np.ndarray, y: np.ndarray,
                             groups: np.ndarray = None,
                             n_permutations: int = 1000) -> Dict:
        """
        Permutation test to verify results are not due to chance.

        Addresses concerns about small sample size validity.
        """
        print(f"\n Running Permutation Test (n={n_permutations})")
        print("=" * 60)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)

        if groups is not None:
            cv = LeaveOneGroupOut()
        else:
            cv = 5

        score, permutation_scores, p_value = permutation_test_score(
            model, X_scaled, y,
            cv=cv,
            groups=groups,
            n_permutations=n_permutations,
            scoring='r2',
            random_state=42,
            n_jobs=-1
        )

        print(f"   True R²:              {score:.3f}")
        print(f"   Permutation R² mean:  {np.mean(permutation_scores):.3f} ± {np.std(permutation_scores):.3f}")
        print(f"   P-value:              {p_value:.4f}")
        print(f"   Significant (p<0.05): {'Yes' if p_value < 0.05 else '❌ No'}")

        return {
            'true_score': score,
            'permutation_scores': permutation_scores,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_permutations': n_permutations
        }

    def run_bootstrap_ci(self, X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray = None,
                         n_bootstrap: int = 1000,
                         ci: float = 95) -> Dict:
        """
        Bootstrap confidence intervals for R².

        Provides uncertainty quantification for small samples.
        """
        print(f"\n📊 Bootstrap {ci}% Confidence Intervals (n={n_bootstrap})")
        print("=" * 60)

        scaler = StandardScaler()
        model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)

        n_samples = len(y)
        bootstrap_r2s = []
        bootstrap_maes = []

        for i in range(n_bootstrap):
            # Bootstrap sample (sample with replacement)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            if groups is not None:
                groups_boot = groups[indices]

            # Scale
            X_scaled = scaler.fit_transform(X_boot)

            # 5-fold CV on bootstrap sample
            try:
                cv_pred = cross_val_predict(model, X_scaled, y_boot, cv=5)
                r2 = r2_score(y_boot, cv_pred)
                mae = mean_absolute_error(y_boot, cv_pred)

                if not np.isnan(r2) and not np.isinf(r2):
                    bootstrap_r2s.append(r2)
                    bootstrap_maes.append(mae)
            except:
                continue

        bootstrap_r2s = np.array(bootstrap_r2s)
        bootstrap_maes = np.array(bootstrap_maes)

        # Calculate confidence intervals
        alpha = (100 - ci) / 2

        r2_lower = np.percentile(bootstrap_r2s, alpha)
        r2_upper = np.percentile(bootstrap_r2s, 100 - alpha)
        r2_median = np.median(bootstrap_r2s)

        mae_lower = np.percentile(bootstrap_maes, alpha)
        mae_upper = np.percentile(bootstrap_maes, 100 - alpha)
        mae_median = np.median(bootstrap_maes)

        print(f"   R²:  {r2_median:.3f} [{r2_lower:.3f}, {r2_upper:.3f}]")
        print(f"   MAE: {mae_median:.3f} [{mae_lower:.3f}, {mae_upper:.3f}]")

        return {
            'r2_median': r2_median,
            'r2_ci_lower': r2_lower,
            'r2_ci_upper': r2_upper,
            'mae_median': mae_median,
            'mae_ci_lower': mae_lower,
            'mae_ci_upper': mae_upper,
            'bootstrap_r2s': bootstrap_r2s,
            'bootstrap_maes': bootstrap_maes,
            'n_bootstrap': n_bootstrap,
            'ci_level': ci
        }

    # =========================================================================
    # LOSO EVALUATION
    # =========================================================================

    def detailed_loso_evaluation(self, X: np.ndarray, y: np.ndarray,
                                 groups: np.ndarray,
                                 model_name: str = 'Bayesian Ridge') -> Dict:
        """
        Detailed Leave-One-Subject-Out evaluation with per-fold statistics.

        """
        print(f"\n🔬 Detailed LOSO Evaluation ({model_name})")
        print("=" * 60)

        scaler = StandardScaler()

        # Select model
        if model_name == 'Bayesian Ridge':
            model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        elif model_name == 'MLP':
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
        else:
            model = BayesianRidge()

        logo = LeaveOneGroupOut()

        all_predictions = np.zeros(len(y))
        all_actuals = np.zeros(len(y))
        fold_results = []

        unique_groups = np.unique(groups)

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit and predict
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)

            all_predictions[test_idx] = pred
            all_actuals[test_idx] = y_test

            # Per-fold metrics
            if len(y_test) > 1:
                fold_r2 = r2_score(y_test, pred)
            else:
                fold_r2 = np.nan

            fold_mae = mean_absolute_error(y_test, pred)

            fold_results.append({
                'fold': fold_idx,
                'group': unique_groups[groups[test_idx[0]]],
                'n_test': len(test_idx),
                'n_train': len(train_idx),
                'r2': fold_r2,
                'mae': fold_mae,
                'actual': y_test.tolist(),
                'predicted': pred.tolist()
            })

        # Overall metrics
        overall_r2 = r2_score(all_actuals, all_predictions)
        overall_mae = mean_absolute_error(all_actuals, all_predictions)
        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        overall_corr, overall_pval = pearsonr(all_actuals, all_predictions)

        print(f"   Overall R²:          {overall_r2:.3f}")
        print(f"   Overall MAE:         {overall_mae:.3f}")
        print(f"   Overall RMSE:        {overall_rmse:.3f}")
        print(f"   Overall Correlation: {overall_corr:.3f} (p={overall_pval:.4f})")
        print(f"   N folds:             {len(fold_results)}")

        # Fold-level statistics
        valid_r2s = [f['r2'] for f in fold_results if not np.isnan(f['r2'])]
        if valid_r2s:
            print(f"   Fold R² mean ± SD:   {np.mean(valid_r2s):.3f} ± {np.std(valid_r2s):.3f}")

        return {
            'overall': {
                'r2': overall_r2,
                'mae': overall_mae,
                'rmse': overall_rmse,
                'correlation': overall_corr,
                'p_value': overall_pval
            },
            'fold_results': fold_results,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'model_name': model_name
        }

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def create_baseline_comparison_figure(self, results_df: pd.DataFrame,
                                          output_path: str = "baseline_comparison.png"):
        """Create publication-quality baseline comparison figure."""
        print("\n Creating Baseline Comparison Figure...")

        # Sort by R²
        results_sorted = results_df.dropna(subset=['R²']).sort_values('R²', ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Color coding by model type
        colors = []
        for model in results_sorted['Model']:
            if 'Naive' in model:
                colors.append('#95a5a6')  # Gray for naive
            elif 'MLP' in model:
                colors.append('#e74c3c')  # Red for neural networks
            elif any(x in model for x in ['Forest', 'Gradient', 'Extra', 'AdaBoost']):
                colors.append('#27ae60')  # Green for tree-based
            elif 'SVR' in model:
                colors.append('#9b59b6')  # Purple for SVM
            else:
                colors.append('#3498db')  # Blue for linear

        # R² bar plot
        ax1 = axes[0]
        bars = ax1.barh(range(len(results_sorted)), results_sorted['R²'], color=colors, alpha=0.8)
        ax1.set_yticks(range(len(results_sorted)))
        ax1.set_yticklabels(results_sorted['Model'], fontsize=9)
        ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Comparison (R²)\nHigher is Better', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, r2) in enumerate(zip(bars, results_sorted['R²'])):
            ax1.text(r2 + 0.01, i, f'{r2:.3f}', va='center', fontsize=8)

        # MAE bar plot
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

        bars_mae = ax2.barh(range(len(results_sorted_mae)), results_sorted_mae['MAE'],
                            color=colors_mae, alpha=0.8)
        ax2.set_yticks(range(len(results_sorted_mae)))
        ax2.set_yticklabels(results_sorted_mae['Model'], fontsize=9)
        ax2.set_xlabel('Mean Absolute Error', fontsize=12, fontweight='bold')
        ax2.set_title('Model Comparison (MAE)\nLower is Better', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Legend
        legend_elements = [
            mpatches.Patch(color='#3498db', label='Linear Models'),
            mpatches.Patch(color='#27ae60', label='Tree-Based'),
            mpatches.Patch(color='#e74c3c', label='Neural Networks'),
            mpatches.Patch(color='#9b59b6', label='SVM'),
            mpatches.Patch(color='#95a5a6', label='Naive Baselines')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=5,
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   Saved: {output_path}")

    def create_age_adjustment_figure(self, results_df: pd.DataFrame,
                                     output_path: str = "age_adjustment_comparison.png"):
        """Create age adjustment comparison figure."""
        print("\n Creating Age Adjustment Comparison Figure...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by R²
        results_sorted = results_df.sort_values('R²', ascending=True)

        # Color based on whether it's your method
        colors = ['#e74c3c' if 'Your Method' in m else '#3498db' for m in results_sorted['Method']]

        bars = ax.barh(range(len(results_sorted)), results_sorted['R²'], color=colors, alpha=0.8)

        ax.set_yticks(range(len(results_sorted)))
        ax.set_yticklabels(results_sorted['Method'], fontsize=10)
        ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Age Adjustment Method Comparison', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, r2) in enumerate(zip(bars, results_sorted['R²'])):
            ax.text(r2 + 0.005, i, f'{r2:.3f}', va='center', fontsize=9, fontweight='bold')

        # Highlight your method
        legend_elements = [
            mpatches.Patch(color='#e74c3c', label='Your Proposed Method'),
            mpatches.Patch(color='#3498db', label='Alternative Methods')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {output_path}")

    def create_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  cohort_name: str = '',
                                  output_path: str = "prediction_scatter.png"):
        """Create prediction vs actual scatter plot."""
        print("\n Creating Prediction Scatter Plot...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # Metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        corr, pval = pearsonr(y_true, y_pred)

        ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax1.set_title(f'Prediction vs Actual\n{cohort_name}', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Add metrics text box
        textstr = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nr = {corr:.3f} (p={pval:.2e})'
        ax1.text(0.95, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Residual plot
        ax2 = axes[1]
        residuals = y_pred - y_true
        ax2.scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals (Predicted - Actual)', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Analysis', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    Saved: {output_path}")

    # =========================================================================
    # MAIN ANALYSIS
    # =========================================================================

    def run_complete_analysis(self, cohort: str = 'hba1c_cohort',
                              use_log_target: bool = True) -> Dict:
        """
        Run complete analysis for a cohort.

        Parameters:
        -----------
        cohort : str
            'hba1c_cohort' or 'fbg_cohort'
        use_log_target : bool
            Whether to use log-transformed target
        """
        print()
        print("=" * 70)
        print(f"COMPLETE ANALYSIS: {cohort.upper()}")
        print(f"Target: {'Log-transformed' if use_log_target else 'Raw'}")
        print("=" * 70)
        print()

        results = {'cohort': cohort, 'use_log_target': use_log_target}

        # Load data
        X, y, y_log, groups, feature_names = self.load_cohort_data(cohort)

        # Select target
        target = y_log if use_log_target else y
        results['n_samples'] = len(target)
        results['n_features_original'] = len(feature_names)

        # Feature selection with FDR correction
        X_selected, feature_importance, selected_features = self.fdr_corrected_feature_selection(
            X, target, feature_names
        )
        results['n_features_selected'] = len(selected_features)
        results['feature_importance'] = feature_importance

        # Baseline comparison
        print("\n" + "=" * 70)
        baseline_results = self.run_baseline_comparison(X_selected, target, groups, validation='loso')
        results['baseline_comparison'] = baseline_results

        # Age adjustment comparison
        print("\n" + "=" * 70)
        age_results = self.compare_age_adjustment_methods(X, target, feature_names, groups)
        results['age_adjustment'] = age_results

        # Permutation test
        print("\n" + "=" * 70)
        perm_results = self.run_permutation_test(X_selected, target, groups, n_permutations=500)
        results['permutation_test'] = perm_results

        # Bootstrap CI
        print("\n" + "=" * 70)
        boot_results = self.run_bootstrap_ci(X_selected, target, groups, n_bootstrap=500)
        results['bootstrap_ci'] = boot_results

        # Detailed LOSO evaluation
        print("\n" + "=" * 70)
        loso_results = self.detailed_loso_evaluation(X_selected, target, groups)
        results['loso_detailed'] = loso_results

        # Save results
        self.results[cohort] = results

        return results

    def save_all_results(self, output_dir: str = "analysis_results"):
        """Save all analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"\n💾 Saving Results to {output_dir}...")

        for cohort, results in self.results.items():
            cohort_dir = output_dir / cohort
            cohort_dir.mkdir(exist_ok=True)

            # Save baseline comparison
            if 'baseline_comparison' in results:
                results['baseline_comparison'].to_csv(
                    cohort_dir / "baseline_comparison.csv", index=False
                )

            # Save age adjustment
            if 'age_adjustment' in results:
                results['age_adjustment'].to_csv(
                    cohort_dir / "age_adjustment_comparison.csv", index=False
                )

            # Save feature importance
            if 'feature_importance' in results:
                results['feature_importance'].to_csv(
                    cohort_dir / "feature_importance.csv", index=False
                )

            # Create figures
            if 'baseline_comparison' in results:
                self.create_baseline_comparison_figure(
                    results['baseline_comparison'],
                    str(cohort_dir / "baseline_comparison.png")
                )

            if 'age_adjustment' in results:
                self.create_age_adjustment_figure(
                    results['age_adjustment'],
                    str(cohort_dir / "age_adjustment_comparison.png")
                )

            if 'loso_detailed' in results:
                self.create_prediction_scatter(
                    results['loso_detailed']['actuals'],
                    results['loso_detailed']['predictions'],
                    cohort,
                    str(cohort_dir / "prediction_scatter.png")
                )

            # Save summary
            summary = {
                'cohort': cohort,
                'n_samples': results.get('n_samples'),
                'n_features_original': results.get('n_features_original'),
                'n_features_selected': results.get('n_features_selected'),
                'best_model': results['baseline_comparison'].loc[
                    results['baseline_comparison']['R²'].idxmax(), 'Model'
                ] if 'baseline_comparison' in results else None,
                'best_r2': float(
                    results['baseline_comparison']['R²'].max()) if 'baseline_comparison' in results else None,
                'permutation_p_value': results['permutation_test'][
                    'p_value'] if 'permutation_test' in results else None,
                'bootstrap_r2_ci': [
                    float(results['bootstrap_ci']['r2_ci_lower']),
                    float(results['bootstrap_ci']['r2_ci_upper'])
                ] if 'bootstrap_ci' in results else None
            }

            with open(cohort_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"   Saved {cohort} results")

        print(f"\n All results saved to {output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 70)
    print("COMPREHENSIVE BASELINE FRAMEWORK v2.0")
    print("=" * 70)
    print()

    # Initialize framework
    framework = ComprehensiveBaselineFramework("processed_data_v2")

    # Run analysis for both cohorts
    for cohort in ['hba1c_cohort', 'fbg_cohort']:
        try:
            print(f"\n{'=' * 70}")
            print(f"ANALYZING: {cohort.upper()}")
            print(f"{'=' * 70}\n")

            results = framework.run_complete_analysis(cohort, use_log_target=True)

            print(f"\n {cohort} analysis complete!")

        except FileNotFoundError as e:
            print(f"\n  {cohort}: Data not found - {e}")
            print("   Run complete_preprocessing_v2.py first")
        except Exception as e:
            print(f"\n {cohort}: Error - {e}")

    # Save all results
    if framework.results:
        framework.save_all_results()

        print()
        print("=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)