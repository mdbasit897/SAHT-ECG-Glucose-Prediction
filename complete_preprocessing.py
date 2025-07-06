#!/usr/bin/env python3
"""
Advanced Baseline Improvement Strategy for ECG-Glucose Prediction
Target: Achieve R² > 0.6 for Q1 Publication
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AdvancedBaselineImprovement:
    def __init__(self, data_dir="processed_data"):
        self.data_dir = Path(data_dir)
        self.features = None
        self.targets = None
        self.X_processed = None
        self.y = None

        print("🚀 ADVANCED BASELINE IMPROVEMENT")
        print("=" * 50)
        print("🎯 Target: R² > 0.6 for Q1 Publication")
        print("=" * 50)

    def load_and_analyze_data(self):
        """Load data and perform detailed analysis"""
        print("📊 Loading and analyzing data...")

        # Load features and targets
        self.features = pd.read_csv(self.data_dir / "FINAL_features.csv")

        with open(self.data_dir / "FINAL_targets.json", 'r') as f:
            import json
            targets_dict = json.load(f)

        self.targets = {k: np.array(v) for k, v in targets_dict.items()
                        if isinstance(v, list)}

        # Focus on primary glucose prediction
        self.y = self.targets['primary_glucose']

        print(f"✅ Loaded {len(self.features)} subjects")
        print(f"✅ Target range: {self.y.min():.1f} - {self.y.max():.1f} mmol/L")
        print(f"✅ Target mean: {self.y.mean():.1f} ± {self.y.std():.1f} mmol/L")

        return True

    def advanced_feature_engineering(self):
        """Create high-impact composite features"""
        print("🔧 Advanced feature engineering...")

        # Start with cleaned feature set
        exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                       [col for col in self.features.columns if any(term in col for term in
                                                                    ['FBG', 'HbA1c', 'Diabetic', 'Coronary', 'Carotid',
                                                                     'glucose'])]

        feature_cols = [col for col in self.features.columns if col not in exclude_cols]
        X_base = self.features[feature_cols].fillna(0)

        # Convert to numeric and handle any remaining issues
        for col in X_base.columns:
            X_base[col] = pd.to_numeric(X_base[col], errors='coerce').fillna(0)

        print(f"   Base features: {len(feature_cols)}")

        # 1. Sleep-Metabolic Interaction Features
        print("   Creating sleep-metabolic interactions...")
        sleep_features = {}

        # HRV sleep stage ratios
        if 'hrv_ds_mean_rr' in X_base.columns and 'hrv_rem_mean_rr' in X_base.columns:
            sleep_features['hrv_deep_rem_ratio'] = (X_base['hrv_ds_mean_rr'] /
                                                    (X_base['hrv_rem_mean_rr'] + 1e-6))

        # Sleep efficiency indicators
        if 'cpc_SSP (%)' in X_base.columns and 'cpc_USP (%)' in X_base.columns:
            sleep_features['sleep_quality_index'] = (X_base['cpc_SSP (%)'] + X_base['cpc_RSP (%)']) / \
                                                    (X_base['cpc_USP (%)'] + 1e-6)

        # 2. Circadian-Metabolic Features
        print("   Creating circadian-metabolic features...")

        # ECG circadian variation
        if 'ecg_sleep_mean' in X_base.columns and 'ecg_day_mean' in X_base.columns:
            sleep_features['ecg_circadian_ratio'] = X_base['ecg_sleep_mean'] / \
                                                    (X_base['ecg_day_mean'] + 1e-6)

        # HRV circadian patterns
        if 'hrv_ds_std_rr' in X_base.columns and 'hrv_rem_std_rr' in X_base.columns:
            sleep_features['hrv_variability_index'] = (X_base['hrv_ds_std_rr'] +
                                                       X_base['hrv_rem_std_rr']) / 2

        # 3. Autonomic Balance Features
        print("   Creating autonomic balance features...")

        # Sympathetic/Parasympathetic balance indicators
        if 'hrv_ds_mean_hr' in X_base.columns and 'hrv_rem_mean_hr' in X_base.columns:
            sleep_features['autonomic_balance'] = (X_base['hrv_ds_mean_hr'] -
                                                   X_base['hrv_rem_mean_hr']) / \
                                                  (X_base['hrv_ds_mean_hr'] +
                                                   X_base['hrv_rem_mean_hr'] + 1e-6)

        # 4. Clinical-Physiological Interactions
        print("   Creating clinical-physiological interactions...")

        # Age-adjusted features
        if 'age' in X_base.columns:
            for hrv_col in [col for col in X_base.columns if 'hrv_' in col]:
                if X_base[hrv_col].std() > 0:
                    sleep_features[f'{hrv_col}_age_adjusted'] = X_base[hrv_col] / \
                                                                (X_base['age'] / 50 + 0.1)

        # BMI-adjusted features (if height/weight available)
        if 'height' in X_base.columns and 'weight' in X_base.columns:
            bmi = X_base['weight'] / ((X_base['height'] / 100) ** 2)
            sleep_features['bmi'] = bmi

            # BMI-adjusted HRV
            for hrv_col in [col for col in X_base.columns if 'hrv_' in col]:
                if X_base[hrv_col].std() > 0:
                    sleep_features[f'{hrv_col}_bmi_adjusted'] = X_base[hrv_col] / \
                                                                (bmi / 25 + 0.1)

        # 5. Signal Quality Features
        print("   Creating signal quality features...")

        # ECG quality indicators
        snr_cols = [col for col in X_base.columns if 'snr' in col]
        if snr_cols:
            sleep_features['avg_signal_quality'] = X_base[snr_cols].mean(axis=1)
            sleep_features['min_signal_quality'] = X_base[snr_cols].min(axis=1)

        # Combine all features
        engineered_df = pd.DataFrame(sleep_features, index=X_base.index)
        X_enhanced = pd.concat([X_base, engineered_df], axis=1)

        print(f"   ✅ Added {len(sleep_features)} engineered features")
        print(f"   ✅ Total features: {len(X_enhanced.columns)}")

        self.X_enhanced = X_enhanced
        return X_enhanced

    def intelligent_feature_selection(self):
        """Apply multiple feature selection strategies"""
        print("🎯 Intelligent feature selection...")

        X = self.X_enhanced.values
        y = self.y

        # 1. Remove low-variance features
        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=0.01)
        X_var = var_selector.fit_transform(X)
        selected_features = self.X_enhanced.columns[var_selector.get_support()]

        print(f"   After variance filter: {len(selected_features)} features")

        # 2. Statistical significance filter
        selector = SelectKBest(score_func=f_regression, k=min(50, len(selected_features)))
        X_stat = selector.fit_transform(X_var, y)
        stat_features = selected_features[selector.get_support()]

        print(f"   After statistical filter: {len(stat_features)} features")

        # 3. Correlation analysis
        selected_df = self.X_enhanced[stat_features]

        # Remove highly correlated features
        corr_matrix = selected_df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation > 0.95
        high_corr_features = [column for column in upper_triangle.columns
                              if any(upper_triangle[column] > 0.95)]

        final_features = [f for f in stat_features if f not in high_corr_features]

        print(f"   After correlation filter: {len(final_features)} features")

        # 4. Recursive Feature Elimination with Cross-Validation
        if len(final_features) > 30:
            rf_selector = RFE(RandomForestRegressor(n_estimators=50, random_state=42),
                              n_features_to_select=25)
            X_final = rf_selector.fit_transform(selected_df[final_features], y)
            final_features = np.array(final_features)[rf_selector.get_support()]

            print(f"   After RFE: {len(final_features)} features")

        self.selected_features = final_features
        self.X_selected = selected_df[final_features]

        return self.X_selected

    def advanced_modeling_ensemble(self):
        """Implement sophisticated ensemble approach"""
        print("🤖 Advanced ensemble modeling...")

        X = self.X_selected.values
        y = self.y

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 1. Base Models with Optimized Parameters
        base_models = {
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'BayesianRidge': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6,
                                           lambda_1=1e-6, lambda_2=1e-6),
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=5,
                                                  min_samples_split=3, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                          learning_rate=0.05, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, random_state=42, verbosity=0)
        }

        # 2. Cross-validation with Leave-One-Out
        loo = LeaveOneOut()
        cv_predictions = np.zeros((len(y), len(base_models)))
        cv_scores = {}

        print("   Performing Leave-One-Out cross-validation...")

        for i, (model_name, model) in enumerate(base_models.items()):
            print(f"     {model_name}...")

            fold_predictions = []
            fold_scores = []

            for train_idx, test_idx in loo.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train model
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)
                fold_predictions.append(y_pred[0])

                # Calculate score for this fold
                fold_scores.append(r2_score([y_test[0]], [y_pred[0]]))

            cv_predictions[:, i] = fold_predictions
            cv_scores[model_name] = {
                'CV_R2': np.mean(fold_scores),
                'CV_MAE': mean_absolute_error(y, fold_predictions),
                'CV_R2_full': r2_score(y, fold_predictions)
            }

            print(f"       R²: {cv_scores[model_name]['CV_R2_full']:.3f}, "
                  f"MAE: {cv_scores[model_name]['CV_MAE']:.3f}")

        # 3. Meta-Learning Ensemble
        print("   Training meta-learner...")

        # Use cross-validated predictions as features for meta-learner
        meta_model = ElasticNet(alpha=0.01, random_state=42)

        # Train meta-model using LOO on the base predictions
        meta_predictions = []

        for train_idx, test_idx in loo.split(cv_predictions):
            X_meta_train = cv_predictions[train_idx]
            y_meta_train = y[train_idx]
            X_meta_test = cv_predictions[test_idx]

            meta_model.fit(X_meta_train, y_meta_train)
            meta_pred = meta_model.predict(X_meta_test)
            meta_predictions.append(meta_pred[0])

        meta_predictions = np.array(meta_predictions)

        # Calculate ensemble performance
        ensemble_r2 = r2_score(y, meta_predictions)
        ensemble_mae = mean_absolute_error(y, meta_predictions)

        print(f"   🎯 Ensemble Performance:")
        print(f"     R²: {ensemble_r2:.3f}")
        print(f"     MAE: {ensemble_mae:.3f} mmol/L")

        # Store results
        self.base_models = base_models
        self.cv_scores = cv_scores
        self.meta_model = meta_model
        self.ensemble_predictions = meta_predictions
        self.ensemble_performance = {
            'R2': ensemble_r2,
            'MAE': ensemble_mae
        }

        return ensemble_r2, ensemble_mae

    def advanced_validation_analysis(self):
        """Perform comprehensive validation analysis"""
        print("📊 Advanced validation analysis...")

        y_true = self.y
        y_pred = self.ensemble_predictions

        # 1. Statistical Significance Tests
        from scipy.stats import pearsonr, spearmanr, ttest_1samp

        # Correlation analysis
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)

        # Error analysis
        errors = y_pred - y_true
        _, normality_p = ttest_1samp(errors, 0)

        print(f"   Statistical Analysis:")
        print(f"     Pearson correlation: r={pearson_r:.3f}, p={pearson_p:.3f}")
        print(f"     Spearman correlation: r={spearman_r:.3f}, p={spearman_p:.3f}")
        print(f"     Error bias test: p={normality_p:.3f}")

        # 2. Clinical Significance Analysis
        clinical_thresholds = [0.5, 1.0, 1.5, 2.0]
        abs_errors = np.abs(errors)

        print(f"   Clinical Acceptance Rates:")
        for threshold in clinical_thresholds:
            rate = np.mean(abs_errors <= threshold) * 100
            print(f"     Within ±{threshold} mmol/L: {rate:.1f}%")

        # 3. Residual Analysis
        plt.figure(figsize=(15, 10))

        # Prediction vs Actual
        plt.subplot(2, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual Glucose (mmol/L)')
        plt.ylabel('Predicted Glucose (mmol/L)')
        plt.title(f'Prediction vs Actual\nR² = {self.ensemble_performance["R2"]:.3f}')

        # Residuals
        plt.subplot(2, 3, 2)
        plt.scatter(y_pred, errors, alpha=0.7)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Glucose (mmol/L)')
        plt.ylabel('Residuals (mmol/L)')
        plt.title('Residuals vs Predicted')

        # Error distribution
        plt.subplot(2, 3, 3)
        plt.hist(errors, bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel('Prediction Error (mmol/L)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')

        # Model comparison
        plt.subplot(2, 3, 4)
        model_names = list(self.cv_scores.keys()) + ['Ensemble']
        r2_scores = [self.cv_scores[name]['CV_R2_full'] for name in self.cv_scores.keys()] + \
                    [self.ensemble_performance['R2']]

        bars = plt.bar(range(len(model_names)), r2_scores)
        bars[-1].set_color('red')  # Highlight ensemble
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.ylabel('R² Score')
        plt.title('Model Comparison')

        # Clinical zones
        plt.subplot(2, 3, 5)
        clinical_zones = ['<7.0', '7.0-8.5', '>8.5']
        actual_counts = [np.sum(y_true < 7.0),
                         np.sum((y_true >= 7.0) & (y_true < 8.5)),
                         np.sum(y_true >= 8.5)]
        predicted_counts = [np.sum(y_pred < 7.0),
                            np.sum((y_pred >= 7.0) & (y_pred < 8.5)),
                            np.sum(y_pred >= 8.5)]

        x = np.arange(len(clinical_zones))
        width = 0.35
        plt.bar(x - width / 2, actual_counts, width, label='Actual', alpha=0.7)
        plt.bar(x + width / 2, predicted_counts, width, label='Predicted', alpha=0.7)
        plt.xlabel('Glucose Control Zones (mmol/L)')
        plt.ylabel('Number of Subjects')
        plt.title('Clinical Zone Distribution')
        plt.xticks(x, clinical_zones)
        plt.legend()

        # Feature importance (if available)
        plt.subplot(2, 3, 6)
        if hasattr(self.base_models['RandomForest'], 'feature_importances_'):
            importance = self.base_models['RandomForest'].feature_importances_
            top_indices = np.argsort(importance)[-10:]
            top_features = [self.selected_features[i] for i in top_indices]
            top_importance = importance[top_indices]

            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)),
                       [f.replace('_', ' ')[:20] + '...' if len(f) > 20 else f.replace('_', ' ')
                        for f in top_features])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')

        plt.tight_layout()
        plt.savefig('advanced_baseline_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Bootstrap Confidence Intervals
        print("   Computing bootstrap confidence intervals...")

        n_bootstrap = 1000
        bootstrap_r2 = []
        bootstrap_mae = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            y_boot_true = y_true[indices]
            y_boot_pred = y_pred[indices]

            # Calculate metrics
            bootstrap_r2.append(r2_score(y_boot_true, y_boot_pred))
            bootstrap_mae.append(mean_absolute_error(y_boot_true, y_boot_pred))

        r2_ci = np.percentile(bootstrap_r2, [2.5, 97.5])
        mae_ci = np.percentile(bootstrap_mae, [2.5, 97.5])

        print(f"   Bootstrap Confidence Intervals (95%):")
        print(f"     R²: {r2_ci[0]:.3f} - {r2_ci[1]:.3f}")
        print(f"     MAE: {mae_ci[0]:.3f} - {mae_ci[1]:.3f} mmol/L")

        # Store validation results
        self.validation_results = {
            'pearson_correlation': (pearson_r, pearson_p),
            'spearman_correlation': (spearman_r, spearman_p),
            'clinical_acceptance': {f'within_{t}': np.mean(abs_errors <= t) * 100
                                    for t in clinical_thresholds},
            'bootstrap_ci': {'R2': r2_ci, 'MAE': mae_ci},
            'error_bias_p': normality_p
        }

        return self.validation_results

    def generate_improvement_report(self):
        """Generate comprehensive improvement report"""
        print("📝 Generating improvement report...")

        report = []
        report.append("# ADVANCED BASELINE IMPROVEMENT REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Performance Summary
        report.append("## PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Final Ensemble Performance:")
        report.append(f"  R² Score: {self.ensemble_performance['R2']:.3f}")
        report.append(f"  MAE: {self.ensemble_performance['MAE']:.3f} mmol/L")
        report.append("")

        # Q1 Publication Assessment
        r2_score = self.ensemble_performance['R2']
        mae_score = self.ensemble_performance['MAE']

        report.append("## Q1 PUBLICATION READINESS")
        report.append("-" * 30)

        if r2_score >= 0.6 and mae_score <= 0.8:
            status = "✅ EXCELLENT - Q1 Ready"
        elif r2_score >= 0.4 and mae_score <= 1.0:
            status = "⚠️ GOOD - Conference Ready"
        elif r2_score >= 0.2 and mae_score <= 1.5:
            status = "🔧 FAIR - Needs Improvement"
        else:
            status = "❌ POOR - Major Revision Needed"

        report.append(f"Publication Readiness: {status}")
        report.append("")

        # Individual Model Performance
        report.append("## INDIVIDUAL MODEL PERFORMANCE")
        report.append("-" * 30)
        for model_name, scores in self.cv_scores.items():
            report.append(f"{model_name}:")
            report.append(f"  R²: {scores['CV_R2_full']:.3f}")
            report.append(f"  MAE: {scores['CV_MAE']:.3f} mmol/L")
        report.append("")

        # Statistical Analysis
        if hasattr(self, 'validation_results'):
            report.append("## STATISTICAL VALIDATION")
            report.append("-" * 30)

            pearson_r, pearson_p = self.validation_results['pearson_correlation']
            report.append(f"Pearson Correlation: r={pearson_r:.3f}, p={pearson_p:.3f}")

            r2_ci = self.validation_results['bootstrap_ci']['R2']
            mae_ci = self.validation_results['bootstrap_ci']['MAE']
            report.append(f"Bootstrap 95% CI:")
            report.append(f"  R²: {r2_ci[0]:.3f} - {r2_ci[1]:.3f}")
            report.append(f"  MAE: {mae_ci[0]:.3f} - {mae_ci[1]:.3f} mmol/L")
            report.append("")

            # Clinical acceptance
            report.append("Clinical Acceptance Rates:")
            for threshold, rate in self.validation_results['clinical_acceptance'].items():
                threshold_val = threshold.split('_')[1]
                report.append(f"  Within ±{threshold_val} mmol/L: {rate:.1f}%")

        report.append("")

        # Feature Engineering Impact
        report.append("## FEATURE ENGINEERING IMPACT")
        report.append("-" * 30)
        report.append(f"Selected Features: {len(self.selected_features)}")
        report.append("Key Feature Categories:")

        # Categorize selected features
        categories = {
            'Sleep-HRV': [f for f in self.selected_features if 'hrv_' in f],
            'ECG': [f for f in self.selected_features if 'ecg_' in f],
            'Clinical': [f for f in self.selected_features if any(term in f for term in
                                                                  ['age', 'weight', 'height', 'SBP', 'DBP'])],
            'Sleep Quality': [f for f in self.selected_features if any(term in f for term in
                                                                       ['psqi_', 'cpc_'])],
            'Engineered': [f for f in self.selected_features if any(term in f for term in
                                                                    ['ratio', 'index', 'balance', 'adjusted'])]
        }

        for category, features in categories.items():
            if features:
                report.append(f"  {category}: {len(features)} features")

        report.append("")

        # Recommendations
        report.append("## RECOMMENDATIONS")
        report.append("-" * 30)

        if r2_score >= 0.6:
            report.append("🎯 Ready for Q1 Journal Submission:")
            report.append("  - Target: Nature Machine Intelligence, Neural Networks")
            report.append("  - Frame as methodology paper")
            report.append("  - Emphasize sleep-aware architecture")
        elif r2_score >= 0.4:
            report.append("🎯 Ready for Conference Submission:")
            report.append("  - Target: IEEE EMBC, Computing in Cardiology")
            report.append("  - Focus on proof-of-concept")
            report.append("  - Continue improving for journal")
        else:
            report.append("🔧 Further Improvements Needed:")
            report.append("  - Consider additional feature engineering")
            report.append("  - Explore deep learning approaches")
            report.append("  - Review data quality issues")

        # Save report
        output_dir = Path("baseline_results")
        output_dir.mkdir(exist_ok=True)

        report_text = "\n".join(report)
        with open(output_dir / "advanced_baseline_report.txt", "w") as f:
            f.write(report_text)

        print(f"✅ Report saved to: {output_dir / 'advanced_baseline_report.txt'}")

        return report_text

    def run_complete_improvement(self):
        """Run complete baseline improvement pipeline"""
        print("🚀 STARTING COMPLETE BASELINE IMPROVEMENT")
        print("=" * 60)

        # Step 1: Load and analyze data
        if not self.load_and_analyze_data():
            return False

        # Step 2: Advanced feature engineering
        self.advanced_feature_engineering()

        # Step 3: Intelligent feature selection
        self.intelligent_feature_selection()

        # Step 4: Advanced ensemble modeling
        final_r2, final_mae = self.advanced_modeling_ensemble()

        # Step 5: Comprehensive validation
        self.advanced_validation_analysis()

        # Step 6: Generate report
        self.generate_improvement_report()

        print("\n🎉 BASELINE IMPROVEMENT COMPLETED!")
        print("=" * 60)
        print(f"🏆 Final Performance:")
        print(f"   R² Score: {final_r2:.3f}")
        print(f"   MAE: {final_mae:.3f} mmol/L")
        print("")

        # Q1 readiness assessment
        if final_r2 >= 0.6 and final_mae <= 0.8:
            print("🎯 STATUS: Q1 JOURNAL READY! 🎉")
            print("   Recommended action: Proceed with Q1 submission")
        elif final_r2 >= 0.4 and final_mae <= 1.0:
            print("🎯 STATUS: CONFERENCE READY! 📝")
            print("   Recommended action: Submit to conference, continue improving")
        else:
            print("🎯 STATUS: NEEDS FURTHER IMPROVEMENT 🔧")
            print("   Recommended action: Explore deep learning approaches")

        return True


# Usage example
if __name__ == "__main__":
    # Initialize improvement system
    improver = AdvancedBaselineImprovement("processed_data")

    # Run complete improvement
    success = improver.run_complete_improvement()

    if success:
        print("\n📊 NEXT STEPS FOR Q1 PUBLICATION:")
        print("=" * 50)
        print("1. Review advanced_baseline_report.txt")
        print("2. Analyze feature importance patterns")
        print("3. If R² > 0.6: Proceed with transformer architecture")
        print("4. If R² < 0.6: Consider deep learning approaches")
        print("5. Prepare manuscript focusing on methodology")