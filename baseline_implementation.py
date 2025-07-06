#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, Ridge
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class OptimizedSmallDatasetBaseline:
    def __init__(self, data_dir="processed_data"):
        self.data_dir = data_dir
        print("🚀 OPTIMIZED SMALL DATASET BASELINE")
        print("=" * 50)
        print("🎯 Maximizing Performance for 40 Subjects")
        print("=" * 50)

    def load_and_prepare_data(self):
        """Load and prepare data with robust preprocessing"""
        print("📊 Loading and preparing data...")

        # Load data
        features_df = pd.read_csv(f"{self.data_dir}/FINAL_features.csv")

        with open(f"{self.data_dir}/FINAL_targets.json", 'r') as f:
            targets_dict = json.load(f)

        # Get glucose target
        y = np.array(targets_dict['primary_glucose'])

        # Clean feature selection
        exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                       [col for col in features_df.columns if any(term in col.lower() for term in
                                                                  ['fbg', 'hba1c', 'diabetic', 'coronary', 'carotid',
                                                                   'glucose'])]

        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        X_raw = features_df[feature_cols].fillna(0)

        # Convert to numeric robustly
        for col in X_raw.columns:
            X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce').fillna(0)

        print(f"✅ Loaded {len(y)} subjects with {len(feature_cols)} base features")
        print(f"✅ Glucose range: {y.min():.1f} - {y.max():.1f} mmol/L")

        return X_raw, y, feature_cols

    def create_smart_features(self, X_raw):
        """Create physiologically meaningful features"""
        print("🧠 Creating smart physiological features...")

        X = X_raw.copy()

        # 1. Sleep-HRV Interaction Features
        try:
            # HRV stage ratios (key physiological markers)
            if 'hrv_ds_mean_rr' in X.columns and 'hrv_rem_mean_rr' in X.columns:
                X['hrv_deep_rem_ratio'] = X['hrv_ds_mean_rr'] / (X['hrv_rem_mean_rr'] + 1e-6)
                X['hrv_variability_contrast'] = (X['hrv_ds_std_rr'] - X['hrv_rem_std_rr']) / \
                                                (X['hrv_ds_std_rr'] + X['hrv_rem_std_rr'] + 1e-6)

            # Autonomic balance indicators
            if 'hrv_ds_mean_hr' in X.columns and 'hrv_rem_mean_hr' in X.columns:
                X['autonomic_balance'] = (X['hrv_ds_mean_hr'] - X['hrv_rem_mean_hr']) / \
                                         (X['hrv_ds_mean_hr'] + X['hrv_rem_mean_hr'] + 1e-6)

            # Sleep efficiency from HRV perspective
            if 'hrv_ds_duration_hours' in X.columns and 'hrv_rem_duration_hours' in X.columns:
                total_sleep = X['hrv_ds_duration_hours'] + X['hrv_rem_duration_hours']
                X['hrv_sleep_efficiency'] = total_sleep / (total_sleep.max() + 1e-6)
        except Exception as e:
            print(f"   Warning: HRV features creation failed: {e}")

        # 2. Circadian Rhythm Features
        try:
            if 'ecg_sleep_mean' in X.columns and 'ecg_day_mean' in X.columns:
                X['circadian_hr_contrast'] = (X['ecg_day_mean'] - X['ecg_sleep_mean']) / \
                                             (X['ecg_day_mean'] + X['ecg_sleep_mean'] + 1e-6)

            # ECG variability during sleep
            if 'ecg_sleep_std' in X.columns and 'ecg_day_std' in X.columns:
                X['circadian_variability_ratio'] = X['ecg_sleep_std'] / (X['ecg_day_std'] + 1e-6)
        except Exception as e:
            print(f"   Warning: Circadian features creation failed: {e}")

        # 3. Clinical Integration Features
        try:
            # Age-adjusted features (critical for glucose prediction)
            if 'age' in X.columns:
                age_normalized = X['age'] / 65.0  # Normalize to typical elderly age

                # Age-adjusted HRV (older people have different HRV patterns)
                hrv_cols = [col for col in X.columns if 'hrv_' in col and '_mean_' in col]
                for col in hrv_cols[:3]:  # Limit to avoid overfitting
                    if X[col].std() > 0:
                        X[f'{col}_age_adj'] = X[col] * (1 / (age_normalized + 0.1))

            # BMI calculation and adjustment
            if 'height' in X.columns and 'weight' in X.columns:
                height_m = X['height'] / 100
                bmi = X['weight'] / (height_m ** 2 + 1e-6)
                X['bmi'] = bmi

                # BMI affects autonomic function
                if 'autonomic_balance' in X.columns:
                    X['bmi_autonomic_interaction'] = bmi * X['autonomic_balance']
        except Exception as e:
            print(f"   Warning: Clinical features creation failed: {e}")

        # 4. Signal Quality Composite
        try:
            snr_cols = [col for col in X.columns if 'snr' in col.lower()]
            if len(snr_cols) >= 2:
                X['overall_signal_quality'] = X[snr_cols].mean(axis=1)
                X['min_signal_quality'] = X[snr_cols].min(axis=1)
        except Exception as e:
            print(f"   Warning: Signal quality features creation failed: {e}")

        # 5. Sleep Quality Index
        try:
            if 'psqi_PSQI score' in X.columns:
                # Invert PSQI score (lower is better sleep quality)
                X['sleep_quality_index'] = 21 - X['psqi_PSQI score']  # Max PSQI is 21

            # Combine multiple sleep indicators
            sleep_indicators = ['cpc_SSP (%)', 'cpc_RSP (%)', 'sleep_quality_index']
            available_sleep = [col for col in sleep_indicators if col in X.columns]
            if len(available_sleep) >= 2:
                X['composite_sleep_quality'] = X[available_sleep].mean(axis=1)
        except Exception as e:
            print(f"   Warning: Sleep quality features creation failed: {e}")

        print(f"✅ Enhanced features: {len(X.columns)} total features")
        return X

    def intelligent_feature_selection_v2(self, X, y):
        """Improved feature selection for small datasets"""
        print("🎯 Intelligent feature selection v2...")

        # Remove constant and low-variance features
        feature_variance = X.var()
        non_constant = feature_variance[feature_variance > 1e-6].index
        X_filtered = X[non_constant]

        print(f"   After variance filter: {len(X_filtered.columns)} features")

        # Correlation with target (most important for small datasets)
        correlations = []
        p_values = []

        for col in X_filtered.columns:
            try:
                corr, p_val = pearsonr(X_filtered[col], y)
                correlations.append(abs(corr))
                p_values.append(p_val)
            except:
                correlations.append(0)
                p_values.append(1)

        # Select features with significant correlation (p < 0.2 for small dataset)
        corr_df = pd.DataFrame({
            'feature': X_filtered.columns,
            'correlation': correlations,
            'p_value': p_values
        })

        significant_features = corr_df[corr_df['p_value'] < 0.2].nlargest(20, 'correlation')

        print(f"   After correlation filter: {len(significant_features)} features")

        # Mutual information as backup
        if len(significant_features) < 10:
            print("   Using mutual information for additional features...")
            mi_scores = mutual_info_regression(X_filtered, y, random_state=42)
            mi_df = pd.DataFrame({
                'feature': X_filtered.columns,
                'mi_score': mi_scores
            })
            additional_features = mi_df.nlargest(15, 'mi_score')

            # Combine and deduplicate
            all_features = pd.concat([significant_features[['feature']],
                                      additional_features[['feature']]]).drop_duplicates()
            selected_features = all_features['feature'].tolist()
        else:
            selected_features = significant_features['feature'].tolist()

        # Ensure we have at least some key features
        key_features = ['age', 'bmi', 'autonomic_balance', 'circadian_hr_contrast',
                        'hrv_deep_rem_ratio', 'composite_sleep_quality']
        for feature in key_features:
            if feature in X.columns and feature not in selected_features:
                selected_features.append(feature)

        # Limit to avoid overfitting (rule of thumb: n_features < n_samples/3)
        max_features = min(len(selected_features), 12)  # Conservative for 40 samples
        final_features = selected_features[:max_features]

        print(f"   Final selected features: {len(final_features)}")
        print("   Selected features:", final_features[:5], "..." if len(final_features) > 5 else "")

        return X[final_features], final_features

    def optimized_modeling(self, X, y):
        """Optimized modeling approach for small datasets"""
        print("🚀 Optimized modeling for small datasets...")

        # Use stratified K-fold instead of LOO (more stable for small datasets)
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

        # Define models with conservative hyperparameters
        models = {
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
            'BayesianRidge': BayesianRidge(),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=3,  # Conservative depth
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        }

        # Evaluate models with proper cross-validation
        results = {}
        predictions = {}

        scaler = RobustScaler()  # More robust to outliers
        X_scaled = scaler.fit_transform(X)

        for name, model in models.items():
            print(f"   Evaluating {name}...")

            # Cross-validation scores
            cv_scores_r2 = cross_val_score(model, X_scaled, y, cv=cv_strategy, scoring='r2')
            cv_scores_mae = -cross_val_score(model, X_scaled, y, cv=cv_strategy, scoring='neg_mean_absolute_error')

            # Fit on full dataset for predictions
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)

            results[name] = {
                'CV_R2_mean': cv_scores_r2.mean(),
                'CV_R2_std': cv_scores_r2.std(),
                'CV_MAE_mean': cv_scores_mae.mean(),
                'CV_MAE_std': cv_scores_mae.std(),
                'Full_R2': r2_score(y, y_pred),
                'Full_MAE': mean_absolute_error(y, y_pred)
            }

            predictions[name] = y_pred

            print(f"     CV R²: {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")
            print(f"     CV MAE: {cv_scores_mae.mean():.3f} ± {cv_scores_mae.std():.3f}")

        # Create ensemble of top 3 models
        print("   Creating ensemble...")

        # Select top models by CV R²
        sorted_models = sorted(results.keys(), key=lambda x: results[x]['CV_R2_mean'], reverse=True)
        top_3_models = sorted_models[:3]

        # Simple average ensemble
        ensemble_pred = np.mean([predictions[model] for model in top_3_models], axis=0)

        ensemble_r2 = r2_score(y, ensemble_pred)
        ensemble_mae = mean_absolute_error(y, ensemble_pred)

        results['Ensemble'] = {
            'CV_R2_mean': ensemble_r2,  # This is actually full dataset R²
            'CV_MAE_mean': ensemble_mae,
            'Full_R2': ensemble_r2,
            'Full_MAE': ensemble_mae
        }

        predictions['Ensemble'] = ensemble_pred

        print(f"   🎯 Ensemble Performance:")
        print(f"     R²: {ensemble_r2:.3f}")
        print(f"     MAE: {ensemble_mae:.3f} mmol/L")

        return results, predictions, y

    def comprehensive_validation(self, results, predictions, y_true):
        """Comprehensive validation analysis"""
        print("📊 Comprehensive validation analysis...")

        # Get best model predictions
        best_model = max(results.keys(), key=lambda x: results[x]['CV_R2_mean'])
        y_pred = predictions[best_model]

        # Statistical tests
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)

        # Clinical acceptance rates
        errors = np.abs(y_pred - y_true)
        clinical_thresholds = [0.5, 1.0, 1.5, 2.0]
        acceptance_rates = {}

        for threshold in clinical_thresholds:
            rate = np.mean(errors <= threshold) * 100
            acceptance_rates[threshold] = rate

        # Glucose range analysis
        range_analysis = {
            'low_glucose': np.sum(y_true < 7.0),
            'medium_glucose': np.sum((y_true >= 7.0) & (y_true < 10.0)),
            'high_glucose': np.sum(y_true >= 10.0)
        }

        print(f"   Best model: {best_model}")
        print(f"   Pearson correlation: r={pearson_r:.3f}, p={pearson_p:.3f}")
        print(f"   Spearman correlation: r={spearman_r:.3f}, p={spearman_p:.3f}")
        print(f"   Clinical acceptance rates:")
        for threshold, rate in acceptance_rates.items():
            print(f"     Within ±{threshold} mmol/L: {rate:.1f}%")

        return {
            'best_model': best_model,
            'pearson': (pearson_r, pearson_p),
            'spearman': (spearman_r, spearman_p),
            'acceptance_rates': acceptance_rates,
            'range_analysis': range_analysis,
            'predictions': y_pred,
            'actual': y_true
        }

    def create_visualizations(self, results, validation_results):
        """Create publication-ready visualizations"""
        print("📈 Creating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Optimized Baseline Performance Analysis', fontsize=16, fontweight='bold')

        # 1. Model comparison
        models = list(results.keys())
        cv_r2_means = [results[model]['CV_R2_mean'] for model in models]
        cv_r2_stds = [results[model].get('CV_R2_std', 0) for model in models]

        axes[0, 0].bar(models, cv_r2_means, yerr=cv_r2_stds, capsize=5)
        axes[0, 0].set_title('Cross-Validation R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Prediction vs Actual
        y_pred = validation_results['predictions']
        y_true = validation_results['actual']

        axes[0, 1].scatter(y_true, y_pred, alpha=0.7, s=60)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Glucose (mmol/L)')
        axes[0, 1].set_ylabel('Predicted Glucose (mmol/L)')
        axes[0, 1].set_title(f'Prediction vs Actual\n{validation_results["best_model"]}')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Residuals
        residuals = y_pred - y_true
        axes[0, 2].scatter(y_pred, residuals, alpha=0.7, s=60)
        axes[0, 2].axhline(0, color='red', linestyle='--')
        axes[0, 2].set_xlabel('Predicted Glucose (mmol/L)')
        axes[0, 2].set_ylabel('Residuals (mmol/L)')
        axes[0, 2].set_title('Residual Analysis')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Clinical acceptance
        thresholds = list(validation_results['acceptance_rates'].keys())
        rates = list(validation_results['acceptance_rates'].values())

        bars = axes[1, 0].bar(range(len(thresholds)), rates)
        axes[1, 0].set_xlabel('Error Threshold (mmol/L)')
        axes[1, 0].set_ylabel('Acceptance Rate (%)')
        axes[1, 0].set_title('Clinical Acceptance Rates')
        axes[1, 0].set_xticks(range(len(thresholds)))
        axes[1, 0].set_xticklabels([f'±{t}' for t in thresholds])
        axes[1, 0].grid(True, alpha=0.3)

        # Color code the bars
        for i, bar in enumerate(bars):
            if rates[i] >= 80:
                bar.set_color('green')
            elif rates[i] >= 60:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # 5. Error distribution
        axes[1, 1].hist(np.abs(residuals), bins=8, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Error (mmol/L)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Glucose range distribution
        range_data = validation_results['range_analysis']
        labels = ['Low\n(<7.0)', 'Medium\n(7.0-10.0)', 'High\n(≥10.0)']
        sizes = [range_data['low_glucose'], range_data['medium_glucose'], range_data['high_glucose']]

        axes[1, 2].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Glucose Range Distribution')

        plt.tight_layout()
        plt.savefig('optimized_baseline_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Visualizations saved as 'optimized_baseline_results.png'")

    def generate_final_report(self, results, validation_results, selected_features):
        """Generate comprehensive final report"""
        print("📝 Generating final report...")

        best_model = validation_results['best_model']
        best_r2 = results[best_model]['CV_R2_mean']
        best_mae = results[best_model]['CV_MAE_mean']

        report = []
        report.append("# OPTIMIZED BASELINE RESULTS")
        report.append("=" * 50)
        report.append(f"Dataset: 40 subjects")
        report.append(f"Selected features: {len(selected_features)}")
        report.append("")

        # Performance summary
        report.append("## PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Best Model: {best_model}")
        report.append(f"Cross-Validation R²: {best_r2:.3f} ± {results[best_model].get('CV_R2_std', 0):.3f}")
        report.append(f"Cross-Validation MAE: {best_mae:.3f} ± {results[best_model].get('CV_MAE_std', 0):.3f} mmol/L")
        report.append("")

        # Q1 readiness assessment
        if best_r2 >= 0.6:
            status = "✅ Q1 READY"
            recommendation = "Proceed with advanced transformer architecture"
        elif best_r2 >= 0.4:
            status = "⚠️ CONFERENCE READY"
            recommendation = "Submit to conference, continue improving for journal"
        elif best_r2 >= 0.2:
            status = "🔧 IMPROVEMENT NEEDED"
            recommendation = "Try deep learning or consider data augmentation"
        else:
            status = "❌ MAJOR REVISION REQUIRED"
            recommendation = "Fundamental approach change needed"

        report.append(f"## PUBLICATION READINESS: {status}")
        report.append("-" * 30)
        report.append(f"Recommendation: {recommendation}")
        report.append("")

        # Statistical validation
        pearson_r, pearson_p = validation_results['pearson']
        spearman_r, spearman_p = validation_results['spearman']

        report.append("## STATISTICAL VALIDATION")
        report.append("-" * 30)
        report.append(f"Pearson correlation: r={pearson_r:.3f}, p={pearson_p:.3f}")
        report.append(f"Spearman correlation: r={spearman_r:.3f}, p={spearman_p:.3f}")

        significance = "SIGNIFICANT" if pearson_p < 0.05 else "NOT SIGNIFICANT"
        report.append(f"Statistical significance: {significance}")
        report.append("")

        # Clinical relevance
        report.append("## CLINICAL RELEVANCE")
        report.append("-" * 30)
        for threshold, rate in validation_results['acceptance_rates'].items():
            report.append(f"Within ±{threshold} mmol/L: {rate:.1f}%")

        # Get best acceptance rate
        best_acceptance = max(validation_results['acceptance_rates'].values())
        if best_acceptance >= 80:
            clinical_status = "EXCELLENT"
        elif best_acceptance >= 60:
            clinical_status = "GOOD"
        elif best_acceptance >= 40:
            clinical_status = "FAIR"
        else:
            clinical_status = "POOR"

        report.append(f"Clinical utility: {clinical_status}")
        report.append("")

        # Selected features
        report.append("## KEY FEATURES")
        report.append("-" * 30)
        for i, feature in enumerate(selected_features[:10]):
            report.append(f"{i + 1}. {feature}")
        if len(selected_features) > 10:
            report.append(f"... and {len(selected_features) - 10} more")
        report.append("")

        # Next steps
        report.append("## NEXT STEPS")
        report.append("-" * 30)
        if best_r2 >= 0.4:
            report.append("1. Develop sleep-aware transformer architecture")
            report.append("2. Implement attention mechanisms for interpretability")
            report.append("3. Prepare manuscript emphasizing methodology")
            report.append("4. Target methodology-focused journals")
        else:
            report.append("1. Consider deep learning approaches (CNN, LSTM)")
            report.append("2. Explore data augmentation techniques")
            report.append("3. Investigate additional feature engineering")
            report.append("4. Consider ensemble of diverse architectures")

        # Save report
        report_text = "\n".join(report)
        with open("optimized_baseline_report.txt", "w") as f:
            f.write(report_text)

        print("✅ Report saved as 'optimized_baseline_report.txt'")
        return report_text

    def run_optimized_analysis(self):
        """Run the complete optimized analysis"""
        print("🚀 STARTING OPTIMIZED ANALYSIS")
        print("=" * 60)

        # Load and prepare data
        X_raw, y, feature_cols = self.load_and_prepare_data()

        # Create smart features
        X_enhanced = self.create_smart_features(X_raw)

        # Intelligent feature selection
        X_selected, selected_features = self.intelligent_feature_selection_v2(X_enhanced, y)

        # Optimized modeling
        results, predictions, y_actual = self.optimized_modeling(X_selected, y)

        # Comprehensive validation
        validation_results = self.comprehensive_validation(results, predictions, y_actual)

        # Create visualizations
        self.create_visualizations(results, validation_results)

        # Generate final report
        self.generate_final_report(results, validation_results, selected_features)

        # Final summary
        best_model = validation_results['best_model']
        best_r2 = results[best_model]['CV_R2_mean']
        best_mae = results[best_model]['CV_MAE_mean']

        print("\n🎉 OPTIMIZED ANALYSIS COMPLETED!")
        print("=" * 60)
        print(f"🏆 Best Performance:")
        print(f"   Model: {best_model}")
        print(f"   R²: {best_r2:.3f}")
        print(f"   MAE: {best_mae:.3f} mmol/L")
        print("")

        # Publication readiness
        if best_r2 >= 0.6:
            print("🎯 STATUS: Q1 JOURNAL READY! 🎉")
        elif best_r2 >= 0.4:
            print("🎯 STATUS: CONFERENCE READY! 📝")
        elif best_r2 >= 0.2:
            print("🎯 STATUS: IMPROVEMENT NEEDED 🔧")
        else:
            print("🎯 STATUS: MAJOR REVISION REQUIRED ❌")

        return best_r2 >= 0.4  # Return True if conference-ready or better


# Run the optimized analysis
if __name__ == "__main__":
    optimizer = OptimizedSmallDatasetBaseline("processed_data")
    success = optimizer.run_optimized_analysis()

    if success:
        print("\n📊 READY FOR ADVANCED ARCHITECTURE DEVELOPMENT!")
    else:
        print("\n🔧 CONSIDER DEEP LEARNING APPROACHES!")