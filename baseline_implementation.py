#!/usr/bin/env python3
"""
Comprehensive Baseline Implementation for Diabetes ECG Research
Publication-Quality Analysis with Multiple Algorithms and Visualizations
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Evaluation and Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score)

# Statistical Analysis
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from scipy.stats import normaltest, shapiro
import scipy.stats as stats

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True


class ComprehensiveBaselineAnalysis:
    def __init__(self, data_dir="processed_data"):
        self.data_dir = Path(data_dir)
        self.features = None
        self.targets = None
        self.splits = None
        self.feature_names = None

        self.regression_results = {}
        self.classification_results = {}
        self.feature_importance = {}

        print("🔬 COMPREHENSIVE BASELINE ANALYSIS")
        print("=" * 60)
        print("📊 Publication-Quality Evaluation Framework")
        print("🎯 Multiple Algorithms | Statistical Analysis | Clinical Interpretation")
        print("=" * 60)

    def load_processed_data(self):
        """Load the preprocessed diabetes ECG dataset"""
        print("📁 Loading processed dataset...")

        try:
            # Check if data directory exists
            if not self.data_dir.exists():
                print(f"❌ Data directory does not exist: {self.data_dir}")
                return False

            # Check for required files
            required_files = [
                "FINAL_features.csv",
                "FINAL_targets.json",
                "FINAL_splits.npz",
                "FINAL_feature_names.json"
            ]

            missing_files = []
            for file in required_files:
                if not (self.data_dir / file).exists():
                    missing_files.append(file)

            if missing_files:
                print(f"❌ Missing required files: {missing_files}")
                print(f"📁 Available files in {self.data_dir}:")
                for file in self.data_dir.iterdir():
                    print(f"   - {file.name}")
                return False

            # Load features
            print("   Loading features...")
            self.features = pd.read_csv(self.data_dir / "FINAL_features.csv")

            # Load targets
            print("   Loading targets...")
            with open(self.data_dir / "FINAL_targets.json", 'r') as f:
                targets_dict = json.load(f)
            self.targets = {k: np.array(v) for k, v in targets_dict.items()
                            if isinstance(v, list)}

            # Load splits (with allow_pickle=True to handle object arrays)
            print("   Loading splits...")
            splits_data = np.load(self.data_dir / "FINAL_splits.npz", allow_pickle=True)
            self.splits = {k: v for k, v in splits_data.items()}

            # Load feature names
            print("   Loading feature names...")
            with open(self.data_dir / "FINAL_feature_names.json", 'r') as f:
                self.feature_names = json.load(f)

            print(f"✅ Dataset loaded successfully:")
            print(f"   Subjects: {len(self.features)}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Targets: {list(self.targets.keys())}")

            # Check split sizes
            if 'X_train' in self.splits and 'X_val' in self.splits and 'X_test' in self.splits:
                print(
                    f"   Splits: {len(self.splits['X_train'])}/{len(self.splits['X_val'])}/{len(self.splits['X_test'])}")
            else:
                print(f"   Available split keys: {list(self.splits.keys())}")

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print(f"📁 Checking data directory: {self.data_dir}")
            if self.data_dir.exists():
                print("   Available files:")
                for file in self.data_dir.iterdir():
                    print(f"     - {file.name}")
            return False

    def prepare_data_matrices(self):
        """Prepare standardized data matrices for modeling"""
        print("🔧 Preparing standardized data matrices...")

        try:
            # Get feature columns (exclude non-features)
            exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                           [col for col in self.features.columns if any(term in col for term in
                                                                        ['FBG', 'HbA1c', 'Diabetic', 'Coronary',
                                                                         'Carotid',
                                                                         'glucose'])]

            feature_cols = [col for col in self.features.columns if col not in exclude_cols]
            print(f"   Selected {len(feature_cols)} feature columns")

            # Extract feature matrix and handle missing values
            X = self.features[feature_cols].fillna(0).values
            print(f"   Feature matrix shape: {X.shape}")

            # Check for any remaining non-numeric data
            if not np.issubdtype(X.dtype, np.number):
                print("   Converting non-numeric data...")
                # Convert to numeric, forcing errors to NaN then to 0
                X_numeric = []
                for i in range(X.shape[1]):
                    col_data = pd.to_numeric(X[:, i], errors='coerce')
                    col_data = np.nan_to_num(col_data, nan=0.0)
                    X_numeric.append(col_data)
                X = np.column_stack(X_numeric)

            # Standardize features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Create train/val/test splits
            n_train = len(self.splits['X_train'])
            n_val = len(self.splits['X_val'])

            self.X_train = X_scaled[:n_train]
            self.X_val = X_scaled[n_train:n_train + n_val]
            self.X_test = X_scaled[n_train + n_val:]

            # Extract targets for each split
            self.y_splits = {}
            for target_name in ['primary_glucose', 'glucose_control', 'glucose_elevated']:
                if target_name in self.targets:
                    y = self.targets[target_name]
                    if len(y) >= (n_train + n_val):
                        self.y_splits[target_name] = {
                            'train': y[:n_train],
                            'val': y[n_train:n_train + n_val],
                            'test': y[n_train + n_val:]
                        }
                    else:
                        print(f"   ⚠️ Target {target_name} has insufficient data: {len(y)}")

            print(f"✅ Data matrices prepared:")
            print(f"   Feature matrix: {X_scaled.shape}")
            print(f"   Train: {self.X_train.shape}")
            print(f"   Validation: {self.X_val.shape}")
            print(f"   Test: {self.X_test.shape}")
            print(f"   Available targets: {list(self.y_splits.keys())}")

            return feature_cols

        except Exception as e:
            print(f"❌ Error preparing data matrices: {e}")
            return None

    def initialize_baseline_models(self):
        """Initialize comprehensive set of baseline models"""
        print("🤖 Initializing baseline model ensemble...")

        # Regression models for continuous glucose prediction
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        }

        # Classification models for categorical predictions
        self.classification_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
            'Support Vector Classifier': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        }

        print(f"✅ Initialized {len(self.regression_models)} regression models")
        print(f"✅ Initialized {len(self.classification_models)} classification models")

    def evaluate_regression_models(self):
        """Comprehensive evaluation of regression models for glucose prediction"""
        print("📈 Evaluating regression models for primary glucose prediction...")

        target_name = 'primary_glucose'
        if target_name not in self.y_splits:
            print(f"❌ Target {target_name} not available")
            return None

        y_train = self.y_splits[target_name]['train']
        y_val = self.y_splits[target_name]['val']
        y_test = self.y_splits[target_name]['test']

        results = []
        predictions = {}

        for model_name, model in self.regression_models.items():
            print(f"   Training {model_name}...")

            try:
                # Train model
                model.fit(self.X_train, y_train)

                # Make predictions
                y_train_pred = model.predict(self.X_train)
                y_val_pred = model.predict(self.X_val)
                y_test_pred = model.predict(self.X_test)

                # Calculate metrics
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                train_r2 = r2_score(y_train, y_train_pred)

                val_mae = mean_absolute_error(y_val, y_val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_r2 = r2_score(y_val, y_val_pred)

                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_r2 = r2_score(y_test, y_test_pred)

                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, y_train,
                                            cv=5, scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()

                # Store results
                result = {
                    'Model': model_name,
                    'Train_MAE': train_mae,
                    'Train_RMSE': train_rmse,
                    'Train_R2': train_r2,
                    'Val_MAE': val_mae,
                    'Val_RMSE': val_rmse,
                    'Val_R2': val_r2,
                    'Test_MAE': test_mae,
                    'Test_RMSE': test_rmse,
                    'Test_R2': test_r2,
                    'CV_MAE': cv_mae,
                    'CV_STD': cv_std
                }

                results.append(result)

                # Store predictions for analysis
                predictions[model_name] = {
                    'train_true': y_train,
                    'train_pred': y_train_pred,
                    'val_true': y_val,
                    'val_pred': y_val_pred,
                    'test_true': y_test,
                    'test_pred': y_test_pred
                }

                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importance[model_name] = np.abs(model.coef_)

                print(f"     R²: {test_r2:.3f}, MAE: {test_mae:.3f}")

            except Exception as e:
                print(f"   ⚠️ Error with {model_name}: {e}")
                continue

        if not results:
            print("❌ No models completed successfully")
            return None

        self.regression_results = {
            'results_df': pd.DataFrame(results),
            'predictions': predictions
        }

        print(f"✅ Completed regression evaluation for {len(results)} models")
        return self.regression_results

    def evaluate_classification_models(self):
        """Comprehensive evaluation of classification models"""
        print("📊 Evaluating classification models for glucose control prediction...")

        results = {}

        for target_name in ['glucose_control', 'glucose_elevated']:
            if target_name not in self.y_splits:
                print(f"   ⚠️ Target {target_name} not available, skipping...")
                continue

            print(f"   Evaluating {target_name}...")

            y_train = self.y_splits[target_name]['train']
            y_val = self.y_splits[target_name]['val']
            y_test = self.y_splits[target_name]['test']

            # Check if we have valid classification targets
            unique_train = np.unique(y_train)
            if len(unique_train) < 2:
                print(f"   ⚠️ Insufficient classes in {target_name} training data: {unique_train}")
                continue

            target_results = []
            target_predictions = {}

            for model_name, model in self.classification_models.items():
                try:
                    # Train model
                    model.fit(self.X_train, y_train)

                    # Make predictions
                    y_train_pred = model.predict(self.X_train)
                    y_val_pred = model.predict(self.X_val)
                    y_test_pred = model.predict(self.X_test)

                    # Get prediction probabilities if available
                    try:
                        y_train_proba = model.predict_proba(self.X_train)
                        y_val_proba = model.predict_proba(self.X_val)
                        y_test_proba = model.predict_proba(self.X_test)
                    except:
                        y_train_proba = None
                        y_val_proba = None
                        y_test_proba = None

                    # Calculate metrics
                    train_acc = accuracy_score(y_train, y_train_pred)
                    val_acc = accuracy_score(y_val, y_val_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)

                    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
                    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

                    # AUC calculation
                    try:
                        if target_name == 'glucose_elevated' and len(unique_train) == 2:
                            # Binary classification
                            if y_train_proba is not None:
                                train_auc = roc_auc_score(y_train, y_train_proba[:, 1])
                                val_auc = roc_auc_score(y_val, y_val_proba[:, 1])
                                test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
                            else:
                                train_auc = val_auc = test_auc = np.nan
                        else:
                            # Multi-class AUC
                            if y_train_proba is not None:
                                train_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr')
                                val_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr')
                                test_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
                            else:
                                train_auc = val_auc = test_auc = np.nan
                    except Exception as auc_error:
                        print(f"     ⚠️ AUC calculation failed for {model_name}: {auc_error}")
                        train_auc = val_auc = test_auc = np.nan

                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train, y_train,
                                                cv=5, scoring='accuracy')
                    cv_acc = cv_scores.mean()
                    cv_std = cv_scores.std()

                    # Store results
                    result = {
                        'Model': model_name,
                        'Train_Acc': train_acc,
                        'Train_F1': train_f1,
                        'Train_AUC': train_auc,
                        'Val_Acc': val_acc,
                        'Val_F1': val_f1,
                        'Val_AUC': val_auc,
                        'Test_Acc': test_acc,
                        'Test_F1': test_f1,
                        'Test_AUC': test_auc,
                        'CV_Acc': cv_acc,
                        'CV_STD': cv_std
                    }

                    target_results.append(result)

                    # Store predictions
                    target_predictions[model_name] = {
                        'train_true': y_train,
                        'train_pred': y_train_pred,
                        'val_true': y_val,
                        'val_pred': y_val_pred,
                        'test_true': y_test,
                        'test_pred': y_test_pred,
                        'train_proba': y_train_proba,
                        'val_proba': y_val_proba,
                        'test_proba': y_test_proba
                    }

                    print(f"     {model_name}: Acc={test_acc:.3f}, F1={test_f1:.3f}")

                except Exception as e:
                    print(f"   ⚠️ Error with {model_name} for {target_name}: {e}")
                    continue

            if target_results:
                results[target_name] = {
                    'results_df': pd.DataFrame(target_results),
                    'predictions': target_predictions
                }

        self.classification_results = results
        print(f"✅ Completed classification evaluation for {len(results)} targets")
        return self.classification_results

    def create_performance_visualizations(self):
        """Create comprehensive performance visualizations"""
        print("📊 Creating publication-quality visualizations...")

        # Create output directory
        output_dir = Path("baseline_results")
        output_dir.mkdir(exist_ok=True)

        # 1. Regression Performance Summary
        if self.regression_results:
            self._plot_regression_performance(output_dir)

        # 2. Classification Performance Summary
        if self.classification_results:
            self._plot_classification_performance(output_dir)

        # 3. Model Comparison Heatmaps
        self._plot_performance_heatmaps(output_dir)

        # 4. Prediction vs Actual Plots
        if self.regression_results:
            self._plot_prediction_scatter(output_dir)

        # 5. Feature Importance Analysis
        if self.feature_importance:
            self._plot_feature_importance(output_dir)

        # 6. Error Analysis
        if self.regression_results:
            self._plot_error_analysis(output_dir)

        # 7. Cross-Validation Analysis
        if self.regression_results:
            self._plot_cross_validation_analysis(output_dir)

        # 8. Clinical Interpretation Plots
        if self.regression_results:
            self._plot_clinical_interpretation(output_dir)

        print(f"✅ All visualizations saved to: {output_dir}")

    def _plot_regression_performance(self, output_dir):
        """Plot comprehensive regression performance analysis"""
        if not self.regression_results or 'results_df' not in self.regression_results:
            return

        df = self.regression_results['results_df'].copy()
        if df.empty:
            return

        # Sort by validation performance
        df = df.sort_values('Val_R2', ascending=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Regression Model Performance - Primary Glucose Prediction', fontsize=16, fontweight='bold')

        # MAE comparison
        axes[0, 0].barh(df['Model'], df['Test_MAE'], color='lightcoral', alpha=0.7)
        axes[0, 0].set_xlabel('Mean Absolute Error (mmol/L)')
        axes[0, 0].set_title('Test Set MAE')
        axes[0, 0].grid(True, alpha=0.3)

        # R² comparison
        axes[0, 1].barh(df['Model'], df['Test_R2'], color='lightblue', alpha=0.7)
        axes[0, 1].set_xlabel('R² Score')
        axes[0, 1].set_title('Test Set R²')
        axes[0, 1].grid(True, alpha=0.3)

        # Training vs Validation R²
        x = np.arange(len(df))
        width = 0.35
        axes[1, 0].bar(x - width / 2, df['Train_R2'], width, label='Training', alpha=0.7)
        axes[1, 0].bar(x + width / 2, df['Val_R2'], width, label='Validation', alpha=0.7)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('Training vs Validation R²')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(df['Model'], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Cross-validation with error bars
        axes[1, 1].errorbar(range(len(df)), df['CV_MAE'], yerr=df['CV_STD'],
                            fmt='o', capsize=5, capthick=2)
        axes[1, 1].set_xlabel('Models')
        axes[1, 1].set_ylabel('Cross-Validation MAE (mmol/L)')
        axes[1, 1].set_title('Cross-Validation Performance')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels(df['Model'], rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'regression_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_classification_performance(self, output_dir):
        """Plot comprehensive classification performance analysis"""
        for target_name, results in self.classification_results.items():
            if 'results_df' not in results or results['results_df'].empty:
                continue

            df = results['results_df'].copy()
            # Sort by AUC, handling NaN values
            df = df.sort_values('Test_AUC', ascending=True, na_position='first')

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Classification Performance - {target_name.replace("_", " ").title()}',
                         fontsize=16, fontweight='bold')

            # Accuracy comparison
            axes[0, 0].barh(df['Model'], df['Test_Acc'], color='lightgreen', alpha=0.7)
            axes[0, 0].set_xlabel('Accuracy')
            axes[0, 0].set_title('Test Set Accuracy')
            axes[0, 0].grid(True, alpha=0.3)

            # AUC comparison (skip if all NaN)
            if not df['Test_AUC'].isna().all():
                valid_auc = df.dropna(subset=['Test_AUC'])
                axes[0, 1].barh(valid_auc['Model'], valid_auc['Test_AUC'], color='orange', alpha=0.7)
                axes[0, 1].set_xlabel('AUC Score')
                axes[0, 1].set_title('Test Set AUC')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'AUC not available', ha='center', va='center',
                                transform=axes[0, 1].transAxes)

            # F1 Score comparison
            axes[1, 0].barh(df['Model'], df['Test_F1'], color='purple', alpha=0.7)
            axes[1, 0].set_xlabel('F1 Score')
            axes[1, 0].set_title('Test Set F1 Score')
            axes[1, 0].grid(True, alpha=0.3)

            # Cross-validation accuracy
            axes[1, 1].errorbar(range(len(df)), df['CV_Acc'], yerr=df['CV_STD'],
                                fmt='o', capsize=5, capthick=2)
            axes[1, 1].set_xlabel('Models')
            axes[1, 1].set_ylabel('Cross-Validation Accuracy')
            axes[1, 1].set_title('Cross-Validation Performance')
            axes[1, 1].set_xticks(range(len(df)))
            axes[1, 1].set_xticklabels(df['Model'], rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'classification_performance_{target_name}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_performance_heatmaps(self, output_dir):
        """Create performance comparison heatmaps"""
        # Regression heatmap
        if self.regression_results and 'results_df' in self.regression_results:
            df_reg = self.regression_results['results_df'].copy()
            if not df_reg.empty:
                # Select key metrics for heatmap
                metrics = ['Train_R2', 'Val_R2', 'Test_R2', 'Train_MAE', 'Val_MAE', 'Test_MAE']
                heatmap_data = df_reg.set_index('Model')[metrics]

                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=0,
                            fmt='.3f', cbar_kws={'label': 'Performance Score'})
                plt.title('Regression Model Performance Heatmap', fontsize=16, fontweight='bold')
                plt.xlabel('Metrics')
                plt.ylabel('Models')
                plt.tight_layout()
                plt.savefig(output_dir / 'regression_performance_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()

        # Classification heatmaps
        for target_name, results in self.classification_results.items():
            if 'results_df' not in results or results['results_df'].empty:
                continue

            df_clf = results['results_df'].copy()

            metrics = ['Train_Acc', 'Val_Acc', 'Test_Acc', 'Train_AUC', 'Val_AUC', 'Test_AUC']
            heatmap_data = df_clf.set_index('Model')[metrics]

            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=0.5,
                        fmt='.3f', cbar_kws={'label': 'Performance Score'})
            plt.title(f'Classification Performance Heatmap - {target_name.replace("_", " ").title()}',
                      fontsize=16, fontweight='bold')
            plt.xlabel('Metrics')
            plt.ylabel('Models')
            plt.tight_layout()
            plt.savefig(output_dir / f'classification_heatmap_{target_name}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_prediction_scatter(self, output_dir):
        """Create prediction vs actual scatter plots"""
        if not self.regression_results or 'predictions' not in self.regression_results:
            return

        predictions = self.regression_results['predictions']
        if not predictions:
            return

        # Get best performing models
        df = self.regression_results['results_df']
        best_models = df.nlargest(min(3, len(df)), 'Test_R2')['Model'].tolist()

        fig, axes = plt.subplots(1, len(best_models), figsize=(6 * len(best_models), 6))
        if len(best_models) == 1:
            axes = [axes]

        for i, model_name in enumerate(best_models):
            if model_name not in predictions:
                continue

            pred_data = predictions[model_name]

            # Plot test set predictions
            y_true = pred_data['test_true']
            y_pred = pred_data['test_pred']

            axes[i].scatter(y_true, y_pred, alpha=0.7, s=50)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

            # Calculate metrics
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            axes[i].set_xlabel('Actual Glucose (mmol/L)')
            axes[i].set_ylabel('Predicted Glucose (mmol/L)')
            axes[i].set_title(f'{model_name}\nR² = {r2:.3f}, MAE = {mae:.3f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.suptitle('Prediction vs Actual - Best Performing Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_feature_importance(self, output_dir):
        """Plot feature importance analysis"""
        if not self.feature_importance:
            return

        # Select models with feature importance
        models_with_importance = ['Random Forest', 'Gradient Boosting', 'XGBoost']
        available_models = [m for m in models_with_importance if m in self.feature_importance]

        if not available_models:
            return

        fig, axes = plt.subplots(len(available_models), 1, figsize=(14, 6 * len(available_models)))
        if len(available_models) == 1:
            axes = [axes]

        for i, model_name in enumerate(available_models):
            importance = self.feature_importance[model_name]

            # Get top 20 features
            top_indices = np.argsort(importance)[-20:]
            top_features = [self.feature_names[j] if j < len(self.feature_names) else f'Feature_{j}'
                            for j in top_indices]
            top_importance = importance[top_indices]

            axes[i].barh(range(len(top_features)), top_importance)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features)
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'Top 20 Features - {model_name}')
            axes[i].grid(True, alpha=0.3)

        plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_analysis(self, output_dir):
        """Create error analysis visualizations"""
        if not self.regression_results or 'predictions' not in self.regression_results:
            return

        predictions = self.regression_results['predictions']
        if not predictions:
            return

        # Get best model
        df = self.regression_results['results_df']
        best_model = df.loc[df['Test_R2'].idxmax(), 'Model']

        if best_model not in predictions:
            return

        pred_data = predictions[best_model]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Error Analysis - {best_model}', fontsize=16, fontweight='bold')

        # Error distribution
        errors = pred_data['test_pred'] - pred_data['test_true']
        axes[0, 0].hist(errors, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Prediction Error (mmol/L)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals vs Predicted
        axes[0, 1].scatter(pred_data['test_pred'], errors, alpha=0.7)
        axes[0, 1].set_xlabel('Predicted Glucose (mmol/L)')
        axes[0, 1].set_ylabel('Residuals (mmol/L)')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot for normality check
        try:
            from scipy.stats import probplot
            probplot(errors, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot - Error Normality')
            axes[1, 0].grid(True, alpha=0.3)
        except:
            axes[1, 0].text(0.5, 0.5, 'Q-Q plot unavailable', ha='center', va='center',
                            transform=axes[1, 0].transAxes)

        # Error by glucose range
        try:
            glucose_ranges = pd.cut(pred_data['test_true'], bins=3, labels=['Low', 'Medium', 'High'])
            error_by_range = pd.DataFrame({'Range': glucose_ranges, 'Error': np.abs(errors)})
            error_by_range.boxplot(column='Error', by='Range', ax=axes[1, 1])
            axes[1, 1].set_xlabel('Glucose Range')
            axes[1, 1].set_ylabel('Absolute Error (mmol/L)')
            axes[1, 1].set_title('Error by Glucose Range')
        except:
            axes[1, 1].text(0.5, 0.5, 'Range analysis unavailable', ha='center', va='center',
                            transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cross_validation_analysis(self, output_dir):
        """Create cross-validation analysis"""
        if not self.regression_results or 'results_df' not in self.regression_results:
            return

        df = self.regression_results['results_df'].copy()
        if df.empty:
            return

        # Sort by CV performance
        df = df.sort_values('CV_MAE')

        plt.figure(figsize=(14, 8))

        # Create error bar plot
        x_pos = np.arange(len(df))
        plt.errorbar(x_pos, df['CV_MAE'], yerr=df['CV_STD'],
                     fmt='o', capsize=5, capthick=2, markersize=8)

        plt.xlabel('Models')
        plt.ylabel('Cross-Validation MAE (mmol/L)')
        plt.title('Cross-Validation Performance with Standard Deviation', fontsize=16, fontweight='bold')
        plt.xticks(x_pos, df['Model'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)

        # Add performance thresholds
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Excellent (MAE < 1.0)')
        plt.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Good (MAE < 1.5)')
        plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Acceptable (MAE < 2.0)')

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'cross_validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_clinical_interpretation(self, output_dir):
        """Create clinical interpretation visualizations"""
        if not self.regression_results or 'predictions' not in self.regression_results:
            return

        # Clinical significance analysis
        predictions = self.regression_results['predictions']
        if not predictions:
            return

        df = self.regression_results['results_df']
        best_model = df.loc[df['Test_R2'].idxmax(), 'Model']

        if best_model not in predictions:
            return

        pred_data = predictions[best_model]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clinical Interpretation Analysis', fontsize=16, fontweight='bold')

        # Clinical significance (errors within acceptable range)
        errors = np.abs(pred_data['test_pred'] - pred_data['test_true'])
        clinical_thresholds = [0.5, 1.0, 1.5, 2.0]
        percentages = [np.mean(errors <= t) * 100 for t in clinical_thresholds]

        axes[0, 0].bar(range(len(clinical_thresholds)), percentages, alpha=0.7)
        axes[0, 0].set_xlabel('Error Threshold (mmol/L)')
        axes[0, 0].set_ylabel('Percentage of Predictions (%)')
        axes[0, 0].set_title('Clinical Acceptance Rate')
        axes[0, 0].set_xticks(range(len(clinical_thresholds)))
        axes[0, 0].set_xticklabels([f'≤{t}' for t in clinical_thresholds])
        axes[0, 0].grid(True, alpha=0.3)

        # Glucose range analysis
        y_true = pred_data['test_true']
        y_pred = pred_data['test_pred']

        # Categorize by ADA guidelines
        def categorize_glucose(values):
            categories = []
            for v in values:
                if v < 7.0:
                    categories.append('Good Control')
                elif v < 8.5:
                    categories.append('Fair Control')
                else:
                    categories.append('Poor Control')
            return categories

        true_categories = categorize_glucose(y_true)
        pred_categories = categorize_glucose(y_pred)

        # Confusion matrix for clinical categories
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_categories, pred_categories,
                                  labels=['Good Control', 'Fair Control', 'Poor Control'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                        xticklabels=['Good Control', 'Fair Control', 'Poor Control'],
                        yticklabels=['Good Control', 'Fair Control', 'Poor Control'])
            axes[0, 1].set_xlabel('Predicted Category')
            axes[0, 1].set_ylabel('Actual Category')
            axes[0, 1].set_title('Clinical Category Prediction')
        except:
            axes[0, 1].text(0.5, 0.5, 'Category analysis unavailable', ha='center', va='center',
                            transform=axes[0, 1].transAxes)

        # Error by patient characteristics (placeholder)
        axes[1, 0].text(0.5, 0.5, 'Patient Subgroup Analysis\n(Requires additional\nclinical metadata)',
                        ha='center', va='center', transform=axes[1, 0].transAxes,
                        fontsize=14, style='italic')
        axes[1, 0].set_title('Error by Patient Subgroups')

        # Clinical decision support
        decision_ranges = ['Hypoglycemia Risk\n(<4.0)', 'Target Range\n(4.0-7.0)',
                           'Mild Elevation\n(7.0-10.0)', 'Severe Elevation\n(>10.0)']

        pred_counts = [
            np.sum(y_pred < 4.0),
            np.sum((y_pred >= 4.0) & (y_pred < 7.0)),
            np.sum((y_pred >= 7.0) & (y_pred < 10.0)),
            np.sum(y_pred >= 10.0)
        ]

        # Only plot if we have non-zero counts
        if sum(pred_counts) > 0:
            axes[1, 1].pie(pred_counts, labels=decision_ranges, autopct='%1.1f%%')
            axes[1, 1].set_title('Predicted Glucose Distribution\n(Clinical Decision Support)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No valid predictions', ha='center', va='center',
                            transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(output_dir / 'clinical_interpretation.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_statistical_report(self):
        """Generate comprehensive statistical analysis report"""
        print("📈 Generating statistical analysis report...")

        output_dir = Path("baseline_results")
        output_dir.mkdir(exist_ok=True)

        report = []
        report.append("# COMPREHENSIVE BASELINE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Dataset summary
        report.append("## DATASET SUMMARY")
        report.append("-" * 30)
        report.append(f"Total subjects: {len(self.features)}")
        report.append(f"Total features: {len(self.feature_names)}")
        report.append(f"Train/Val/Test split: {len(self.X_train)}/{len(self.X_val)}/{len(self.X_test)}")
        report.append("")

        # Regression results
        if self.regression_results and 'results_df' in self.regression_results:
            report.append("## REGRESSION RESULTS (Primary Glucose Prediction)")
            report.append("-" * 30)
            df_reg = self.regression_results['results_df']

            if not df_reg.empty:
                # Best performing models
                best_r2 = df_reg.loc[df_reg['Test_R2'].idxmax()]
                best_mae = df_reg.loc[df_reg['Test_MAE'].idxmin()]

                report.append(f"Best R² Score: {best_r2['Model']} (R² = {best_r2['Test_R2']:.3f})")
                report.append(f"Best MAE: {best_mae['Model']} (MAE = {best_mae['Test_MAE']:.3f} mmol/L)")
                report.append("")

                # Top 5 models table
                top_models = df_reg.nlargest(min(5, len(df_reg)), 'Test_R2')[
                    ['Model', 'Test_R2', 'Test_MAE', 'Test_RMSE', 'CV_MAE']]
                report.append("### Top 5 Models by Test R²:")
                report.append(top_models.to_string(index=False))
                report.append("")

                # Statistical significance testing
                report.append("### Statistical Analysis:")

                # Get predictions from best two models for comparison
                if len(df_reg) >= 2 and self.regression_results.get('predictions'):
                    model1 = df_reg.iloc[0]['Model']
                    model2 = df_reg.iloc[1]['Model']

                    predictions = self.regression_results['predictions']
                    if model1 in predictions and model2 in predictions:
                        pred1 = predictions[model1]['test_pred']
                        pred2 = predictions[model2]['test_pred']
                        y_true = predictions[model1]['test_true']

                        err1 = np.abs(pred1 - y_true)
                        err2 = np.abs(pred2 - y_true)

                        # Paired t-test
                        try:
                            stat, p_value = ttest_rel(err1, err2)
                            report.append(f"Paired t-test between {model1} and {model2}:")
                            report.append(f"  t-statistic: {stat:.3f}, p-value: {p_value:.3f}")

                            if p_value < 0.05:
                                better_model = model1 if np.mean(err1) < np.mean(err2) else model2
                                report.append(f"  Significant difference (p < 0.05): {better_model} performs better")
                            else:
                                report.append(f"  No significant difference (p ≥ 0.05)")
                        except Exception as e:
                            report.append(f"  Statistical test failed: {e}")

                report.append("")

        # Classification results
        for target_name, results in self.classification_results.items():
            if 'results_df' not in results or results['results_df'].empty:
                continue

            report.append(f"## CLASSIFICATION RESULTS ({target_name.replace('_', ' ').title()})")
            report.append("-" * 30)

            df_clf = results['results_df']

            # Best performing models
            if not df_clf['Test_AUC'].isna().all():
                best_auc = df_clf.loc[df_clf['Test_AUC'].idxmax()]
                report.append(f"Best AUC: {best_auc['Model']} (AUC = {best_auc['Test_AUC']:.3f})")

            best_acc = df_clf.loc[df_clf['Test_Acc'].idxmax()]
            report.append(f"Best Accuracy: {best_acc['Model']} (Acc = {best_acc['Test_Acc']:.3f})")
            report.append("")

            # Top 5 models
            if not df_clf['Test_AUC'].isna().all():
                top_models = df_clf.nlargest(min(5, len(df_clf)), 'Test_AUC')[
                    ['Model', 'Test_Acc', 'Test_F1', 'Test_AUC']]
            else:
                top_models = df_clf.nlargest(min(5, len(df_clf)), 'Test_Acc')[['Model', 'Test_Acc', 'Test_F1']]

            report.append("### Top 5 Models:")
            report.append(top_models.to_string(index=False))
            report.append("")

        # Clinical interpretation
        if self.regression_results and 'predictions' in self.regression_results:
            predictions = self.regression_results['predictions']
            if predictions:
                report.append("## CLINICAL INTERPRETATION")
                report.append("-" * 30)

                # Best regression model for clinical analysis
                df_reg = self.regression_results['results_df']
                best_model = df_reg.loc[df_reg['Test_R2'].idxmax(), 'Model']

                if best_model in predictions:
                    pred_data = predictions[best_model]

                    errors = np.abs(pred_data['test_pred'] - pred_data['test_true'])

                    report.append(f"Analysis based on best performing model: {best_model}")
                    report.append("")
                    report.append("Clinical Acceptance Rates:")

                    thresholds = [0.5, 1.0, 1.5, 2.0]
                    for threshold in thresholds:
                        rate = np.mean(errors <= threshold) * 100
                        report.append(f"  Predictions within ±{threshold} mmol/L: {rate:.1f}%")

                    report.append("")

                    # Glucose range analysis
                    y_true = pred_data['test_true']
                    report.append("Glucose Range Analysis:")
                    report.append(f"  Mean actual glucose: {np.mean(y_true):.1f} ± {np.std(y_true):.1f} mmol/L")
                    report.append(f"  Range: {np.min(y_true):.1f} - {np.max(y_true):.1f} mmol/L")

                    # Clinical categories
                    good_control = np.sum(y_true < 7.0)
                    fair_control = np.sum((y_true >= 7.0) & (y_true < 8.5))
                    poor_control = np.sum(y_true >= 8.5)

                    report.append(
                        f"  Good control (<7.0 mmol/L): {good_control} subjects ({good_control / len(y_true) * 100:.1f}%)")
                    report.append(
                        f"  Fair control (7.0-8.5 mmol/L): {fair_control} subjects ({fair_control / len(y_true) * 100:.1f}%)")
                    report.append(
                        f"  Poor control (>8.5 mmol/L): {poor_control} subjects ({poor_control / len(y_true) * 100:.1f}%)")

                    report.append("")

        # Recommendations for Sleep-Aware Transformer
        report.append("## RECOMMENDATIONS FOR SLEEP-AWARE TRANSFORMER")
        report.append("-" * 30)

        if self.regression_results and 'results_df' in self.regression_results:
            df_reg = self.regression_results['results_df']
            if not df_reg.empty:
                best_mae = df_reg['Test_MAE'].min()
                best_r2 = df_reg['Test_R2'].max()

                report.append("Performance Targets to Exceed:")
                report.append(f"  Target MAE: < {best_mae:.3f} mmol/L")
                report.append(f"  Target R²: > {best_r2:.3f}")
                report.append("")

        report.append("Key Findings for Novel Architecture:")

        # Feature importance insights
        if self.feature_importance:
            report.append("1. Feature Importance Insights:")

            # Aggregate feature importance across models
            all_importance = np.zeros(len(self.feature_names))
            model_count = 0

            for model_name, importance in self.feature_importance.items():
                if len(importance) == len(self.feature_names):
                    all_importance += importance
                    model_count += 1

            if model_count > 0:
                avg_importance = all_importance / model_count
                top_features_idx = np.argsort(avg_importance)[-10:]

                report.append("   Top 10 most important features:")
                for idx in reversed(top_features_idx):
                    feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'Feature_{idx}'
                    importance_val = avg_importance[idx]

                    # Categorize feature
                    if 'ecg_' in feature_name.lower():
                        category = "ECG"
                    elif 'hrv_' in feature_name.lower():
                        category = "HRV"
                    elif 'sleep' in feature_name.lower():
                        category = "Sleep"
                    else:
                        category = "Clinical"

                    report.append(f"     {feature_name} ({category}): {importance_val:.3f}")

        report.append("")
        report.append("2. Architecture Recommendations:")
        report.append("   - Focus attention mechanisms on high-importance feature categories")
        report.append("   - Implement sleep-stage-specific processing for HRV features")
        report.append("   - Use hierarchical attention for multi-scale ECG analysis")
        report.append("   - Include clinical features as contextual information")
        report.append("")

        report.append("3. Expected Performance Improvements:")
        report.append("   - Sleep-aware processing should improve HRV feature utilization")
        report.append("   - Multi-modal attention should enhance clinical interpretation")
        report.append("   - Hierarchical architecture should capture temporal patterns better")
        report.append("")

        # Save report
        report_text = "\n".join(report)
        with open(output_dir / "comprehensive_baseline_report.txt", "w") as f:
            f.write(report_text)

        print(f"✅ Statistical report saved to: {output_dir / 'comprehensive_baseline_report.txt'}")

        return report_text

    def run_complete_analysis(self):
        """Run the complete baseline analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE BASELINE ANALYSIS")
        print("=" * 60)

        # Load data
        if not self.load_processed_data():
            return False

        # Prepare data matrices
        feature_cols = self.prepare_data_matrices()
        if feature_cols is None:
            return False

        # Initialize models
        self.initialize_baseline_models()

        # Evaluate models
        regression_success = self.evaluate_regression_models()
        classification_success = self.evaluate_classification_models()

        if regression_success is None and not classification_success:
            print("❌ All model evaluations failed")
            return False

        # Create visualizations
        self.create_performance_visualizations()

        # Generate statistical report
        self.generate_statistical_report()

        print("\n🎉 BASELINE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Results saved to: baseline_results/")
        print("📊 Visualizations: Multiple comprehensive plots created")
        print("📈 Statistical report: comprehensive_baseline_report.txt")
        print("")
        print("🎯 READY FOR SLEEP-AWARE TRANSFORMER DEVELOPMENT!")
        print("   Performance targets established")
        print("   Feature importance analyzed")
        print("   Clinical benchmarks set")

        return True


# Main execution
if __name__ == "__main__":
    # Initialize and run comprehensive baseline analysis
    analyzer = ComprehensiveBaselineAnalysis("processed_data")

    success = analyzer.run_complete_analysis()

    if success:
        # Print summary of best results
        print("\n📊 BEST BASELINE PERFORMANCE SUMMARY:")
        print("=" * 50)

        # Best regression results
        if analyzer.regression_results and 'results_df' in analyzer.regression_results:
            df_reg = analyzer.regression_results['results_df']
            if not df_reg.empty:
                best_model = df_reg.loc[df_reg['Test_R2'].idxmax()]

                print(f"🏆 Best Regression Model: {best_model['Model']}")
                print(f"   Test R²: {best_model['Test_R2']:.3f}")
                print(f"   Test MAE: {best_model['Test_MAE']:.3f} mmol/L")
                print(f"   Test RMSE: {best_model['Test_RMSE']:.3f} mmol/L")

        # Best classification results
        for target_name, results in analyzer.classification_results.items():
            if 'results_df' not in results or results['results_df'].empty:
                continue

            df_clf = results['results_df']

            if not df_clf['Test_AUC'].isna().all():
                best_clf = df_clf.loc[df_clf['Test_AUC'].idxmax()]
                print(f"\n🏆 Best {target_name.replace('_', ' ').title()} Model: {best_clf['Model']}")
                print(f"   Test AUC: {best_clf['Test_AUC']:.3f}")
                print(f"   Test Accuracy: {best_clf['Test_Acc']:.3f}")
                print(f"   Test F1: {best_clf['Test_F1']:.3f}")
            else:
                best_clf = df_clf.loc[df_clf['Test_Acc'].idxmax()]
                print(f"\n🏆 Best {target_name.replace('_', ' ').title()} Model: {best_clf['Model']}")
                print(f"   Test Accuracy: {best_clf['Test_Acc']:.3f}")
                print(f"   Test F1: {best_clf['Test_F1']:.3f}")

        if analyzer.regression_results and 'results_df' in analyzer.regression_results:
            df_reg = analyzer.regression_results['results_df']
            if not df_reg.empty:
                best_model = df_reg.loc[df_reg['Test_R2'].idxmax()]
                print(f"\n🎯 TARGETS FOR SLEEP-AWARE TRANSFORMER:")
                print(f"   Beat Regression R²: > {best_model['Test_R2']:.3f}")
                print(f"   Beat Regression MAE: < {best_model['Test_MAE']:.3f} mmol/L")
                print(f"   Demonstrate superiority through statistical testing")
                print(f"   Provide clinical interpretability through attention mechanisms")

    else:
        print("❌ Baseline analysis failed. Please check error messages above.")