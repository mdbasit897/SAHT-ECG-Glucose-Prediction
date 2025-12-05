# Age-Normalized HRV Features for Glucose Prediction

## Revised Analysis Pipeline v2.0

This repository contains the **completely revised** analysis pipeline for investigating age-normalized heart rate variability (HRV) features as correlates of glycemic status in Type 2 Diabetes patients.

### 🔄 Key Revisions (Addressing All Reviewer Concerns)

| Reviewer | Concern | Solution |
|----------|---------|----------|
| **Reviewer 1** | Sample size, mixed glucose targets | Separated HbA1c and FBG cohorts with clear documentation |
| **Reviewer 2** | Lack of train/test independence | LOSO cross-validation ensures complete subject separation |
| **Reviewer 3** | Inadequate baselines, minimal age adjustment novelty | Neural network baselines + 6 age adjustment methods compared |
| **Reviewer 4** | Methodology rigor, signal documentation | Comprehensive signal specs, temporal validation |
| **Reviewer 5** | Age normalization method details | Full documentation and comparison with alternatives |

---

## 📁 Repository Structure

```
glucose_prediction_revised/
├── complete_preprocessing_v2.py    # Data preprocessing with separated cohorts
├── comprehensive_baseline_v2.py    # All baseline models including neural networks
├── ablation_study_v2.py            # Component contribution analysis
├── validation_framework.py         # Statistical validation (permutation, bootstrap)
├── run_complete_analysis.py        # Main runner script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your dataset in the working directory with this structure:
```
./Dataset_on_electrocardiograph/dataset_ecg/
├── clinical_indicators.xlsx
├── objective_sleep_quality.xlsx
├── subjective_sleep_quality.xlsx
├── ECG/
│   ├── 20200101.mat
│   └── ...
└── RR_interval/
    ├── 20200101.mat
    └── ...
```

### 3. Run Complete Analysis

```bash
python run_complete_analysis.py
```

Or run individual components:

```bash
# Step 1: Preprocessing only
python run_complete_analysis.py --only-preprocessing

# Step 2: Baseline comparison only (requires preprocessing)
python run_complete_analysis.py --skip-preprocessing --only-baselines

# Step 3: Ablation study only
python run_complete_analysis.py --skip-preprocessing --only-ablation
```

---

## 📊 Output Structure

After running the complete analysis:

```
./
├── processed_data_v2/
│   ├── features.csv                 # All extracted features
│   ├── signal_specifications.json   # ECG/HRV documentation
│   ├── ecg_scaling_logs.json        # ECG processing logs
│   ├── targets/
│   │   ├── hba1c_cohort.csv         # HbA1c targets (SEPARATED)
│   │   └── fbg_cohort.csv           # FBG targets (SEPARATED)
│   └── loso_splits/
│       ├── hba1c_cohort/            # LOSO validation splits
│       └── fbg_cohort/
│
├── analysis_results/
│   ├── hba1c_cohort/
│   │   ├── baseline_comparison.csv
│   │   ├── baseline_comparison.png
│   │   ├── age_adjustment_comparison.csv
│   │   ├── age_adjustment_comparison.png
│   │   ├── prediction_scatter.png
│   │   └── summary.json
│   └── fbg_cohort/
│       └── ...
│
├── ablation_results/
│   ├── hba1c_cohort/
│   │   ├── ablation_results.csv
│   │   ├── ablation_figure.png
│   │   └── key_findings.json
│   └── fbg_cohort/
│       └── ...
│
├── validation_results/
│   ├── hba1c_cohort/
│   │   ├── validation_report.json
│   │   ├── permutation_test.png
│   │   ├── bootstrap_ci.png
│   │   └── residual_analysis.png
│   └── fbg_cohort/
│       └── ...
│
└── final_report/
    ├── final_report.json
    └── ANALYSIS_SUMMARY.md
```

---

## 🔬 Methodology Improvements

### 1. Separated Glucose Targets (CRITICAL FIX)

**Previous Issue:** Mixed HbA1c (%) and FBG (mmol/L) as single target variable.

**Solution:**
- HbA1c cohort: Long-term glycemic control (~3-month average)
- FBG cohort: Acute glycemic status (instantaneous measurement)
- Each analyzed independently with appropriate clinical interpretation

### 2. Proper Cross-Validation

**Previous Issue:** Random K-fold CV could leak information between subjects.

**Solution:**
- **LOSO (Leave-One-Subject-Out):** Complete subject separation
- **Temporal validation:** Train on earlier subjects, test on later ones
- **Standard K-fold:** For comparison only

### 3. Comprehensive Baselines (Addresses Reviewer 3)

| Model Category | Models Included |
|----------------|-----------------|
| Naive | Mean, Median |
| Linear | Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge |
| Tree-based | Random Forest, Gradient Boosting, Extra Trees, AdaBoost |
| SVM | RBF, Linear, Polynomial kernels |
| **Neural Networks** | MLP (32), MLP (64,32), MLP (128,64,32) |

### 4. Age Adjustment Comparison (Addresses Reviewer 3)

| Method | Description |
|--------|-------------|
| No adjustment | Baseline |
| Your method | HRV / (age/65 + 0.1) |
| Residualization | Regress age out of HRV features |
| Age-Bin Z-Score | Z-score within age quartiles |
| Polynomial Interaction | Age², HRV×age terms |
| Simple Division | HRV / age |

### 5. Statistical Validation

- **Permutation Testing:** Verifies results are not due to chance
- **Bootstrap CI:** 95% confidence intervals for all metrics
- **CV Stability Analysis:** Tests reproducibility across random splits
- **Learning Curves:** Sample size recommendations

---

## 📝 Key Files Explained

### `complete_preprocessing_v2.py`

Main preprocessing pipeline:
- Loads clinical, ECG, and sleep data
- Extracts HRV features from RR intervals
- Creates age-normalized features
- **Separates HbA1c and FBG cohorts**
- Creates LOSO and temporal validation splits
- Documents all signal specifications

### `comprehensive_baseline_v2.py`

Baseline model comparison:
- 20+ models including neural networks
- LOSO cross-validation
- Age adjustment method comparison
- Permutation testing
- Bootstrap confidence intervals
- Publication-quality figures

### `ablation_study_v2.py`

Component contribution analysis:
- Tests each feature category's contribution
- Compares sleep stages (Deep Sleep, REM, RS)
- Quantifies age normalization benefit
- Separate analysis per cohort

### `validation_framework.py`

Statistical validation:
- Permutation tests (n=1000)
- Bootstrap CIs (n=1000)
- CV stability analysis
- Residual diagnostics
- Learning curve analysis

---

## 🎯 Expected Results

Based on the original analysis, you should expect:

| Cohort | Best Model | R² | MAE | Age Norm Improvement |
|--------|-----------|-----|-----|---------------------|
| HbA1c | Bayesian Ridge | ~0.16-0.20 | ~0.15 | ~0.03-0.04 R² |
| FBG | Bayesian Ridge | ~0.10-0.15 | ~0.20 | ~0.02-0.03 R² |

**Important:** These are exploratory pilot study results. The R² values are lower than state-of-the-art neural network approaches (R² ~0.94 in Gusev et al.) but this is expected given:
- Small sample size (~40 subjects)
- Traditional ML features vs. deep learning
- HRV-only vs. multi-modal approaches

---

## 📄 Citation

If you use this code, please cite:

```
Azam, M.B. (2025). Age-Normalized HRV Features as Correlates of Glycemic 
Status: A Pilot Study. Tezpur University, India.
```

---

## 🐛 Troubleshooting

### Data Not Found
```
FileNotFoundError: Clinical indicators file not found
```
**Solution:** Ensure dataset is in correct location (see Quick Start section).

### Memory Issues
```
MemoryError during neural network training
```
**Solution:** Reduce `n_bootstrap` or `n_permutations` parameters.

### Missing Dependencies
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution:** Run `pip install -r requirements.txt`

---

## 📧 Contact

For questions or issues:
- **Author:** Md Basit Azam
- **Affiliation:** Tezpur University, Department of Biomedical Engineering
- **Email:** [Your email]

---

## License

This project is for academic research purposes. Please contact the author for commercial use.