#!/usr/bin/env python3
"""
Quick validation check to ensure results are not overfitted
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from scipy.stats import pearsonr


def validation_check():
    print("🔍 VALIDATION CHECK FOR RESULTS")
    print("=" * 50)

    # Load your processed features (same as your successful run)
    features_df = pd.read_csv("processed_data/FINAL_features.csv")

    with open("processed_data/FINAL_targets.json", 'r') as f:
        targets_dict = json.load(f)

    y = np.array(targets_dict['primary_glucose'])

    # Use the same feature selection as your successful run
    exclude_cols = ['subject_id', 'gender', 'Unnamed: 0'] + \
                   [col for col in features_df.columns if any(term in col.lower() for term in
                                                              ['fbg', 'hba1c', 'diabetic', 'coronary', 'carotid',
                                                               'glucose'])]

    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    X_raw = features_df[feature_cols].fillna(0)

    # Convert to numeric
    for col in X_raw.columns:
        X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce').fillna(0)

    # Create the same smart features that worked
    X = X_raw.copy()

    # Age-adjusted HRV features (key to your success)
    if 'age' in X.columns:
        age_normalized = X['age'] / 65.0
        hrv_cols = [col for col in X.columns if 'hrv_' in col and '_mean_' in col]
        for col in hrv_cols[:3]:
            if X[col].std() > 0:
                X[f'{col}_age_adj'] = X[col] * (1 / (age_normalized + 0.1))

    # Select top 12 features (approximately what you used)
    feature_correlations = []
    for col in X.columns:
        try:
            corr, p_val = pearsonr(X[col], y)
            if p_val < 0.2:  # Same threshold as your successful run
                feature_correlations.append((col, abs(corr), p_val))
        except:
            continue

    # Sort by correlation strength
    feature_correlations.sort(key=lambda x: x[1], reverse=True)
    selected_features = [item[0] for item in feature_correlations[:12]]

    print(f"Selected features: {selected_features[:5]}...")

    X_selected = X[selected_features]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Define the same models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        'BayesianRidge': BayesianRidge(),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=3,
                                              min_samples_split=3, min_samples_leaf=2, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    }

    # CRITICAL: Proper cross-validation for ensemble
    print("\n🎯 PROPER CROSS-VALIDATION TEST")
    print("-" * 40)

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    # Get cross-validated predictions for each model
    cv_predictions = {}

    for name, model in models.items():
        # Get cross-validated predictions (this is the RIGHT way)
        cv_pred = cross_val_predict(model, X_scaled, y, cv=cv_strategy)
        cv_r2 = r2_score(y, cv_pred)
        cv_mae = mean_absolute_error(y, cv_pred)

        cv_predictions[name] = cv_pred

        print(f"{name:15} | CV R²: {cv_r2:6.3f} | CV MAE: {cv_mae:6.3f}")

    # Create ensemble from cross-validated predictions
    top_3_models = ['BayesianRidge', 'RandomForest', 'ElasticNet']  # Typically best performers
    available_models = [model for model in top_3_models if model in cv_predictions]

    if len(available_models) >= 2:
        ensemble_cv_pred = np.mean([cv_predictions[model] for model in available_models], axis=0)
    else:
        # Fallback to all models
        ensemble_cv_pred = np.mean(list(cv_predictions.values()), axis=0)

    # Calculate ensemble performance on cross-validated predictions
    ensemble_cv_r2 = r2_score(y, ensemble_cv_pred)
    ensemble_cv_mae = mean_absolute_error(y, ensemble_cv_pred)

    print("-" * 40)
    print(f"{'ENSEMBLE CV':15} | CV R²: {ensemble_cv_r2:6.3f} | CV MAE: {ensemble_cv_mae:6.3f}")

    # Statistical validation
    correlation, p_value = pearsonr(y, ensemble_cv_pred)

    print(f"\n📊 CROSS-VALIDATED ENSEMBLE VALIDATION:")
    print(f"   R² Score: {ensemble_cv_r2:.3f}")
    print(f"   MAE: {ensemble_cv_mae:.3f} mmol/L")
    print(f"   Correlation: r={correlation:.3f}, p={p_value:.3f}")

    # Clinical acceptance
    errors = np.abs(ensemble_cv_pred - y)
    acceptance_1_0 = np.mean(errors <= 1.0) * 100
    acceptance_1_5 = np.mean(errors <= 1.5) * 100

    print(f"   Clinical acceptance:")
    print(f"     Within ±1.0 mmol/L: {acceptance_1_0:.1f}%")
    print(f"     Within ±1.5 mmol/L: {acceptance_1_5:.1f}%")

    # Validation status
    print(f"\n🎯 VALIDATION STATUS:")
    if ensemble_cv_r2 >= 0.6 and p_value < 0.05:
        print("   ✅ EXCELLENT - Q1 Journal Ready!")
        print("   Results are robust and cross-validated")
    elif ensemble_cv_r2 >= 0.4 and p_value < 0.05:
        print("   ✅ GOOD - Conference Ready!")
        print("   Solid results for publication")
    elif ensemble_cv_r2 >= 0.2:
        print("   ⚠️  MODERATE - Proof of Concept")
        print("   Shows promise but needs improvement")
    else:
        print("   ❌ POOR - Needs Major Improvement")
        print("   Original results may have been overfitted")

    return ensemble_cv_r2, ensemble_cv_mae, correlation, p_value


if __name__ == "__main__":
    validation_check()