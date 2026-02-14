# Re-evaluating Heart Rate Variability Biomarkers for Glucose Sensing: The Impact of Age Normalisation and Subject-Independent Validation
### Research prototype • Not for clinical use

**1<sup>st</sup> Md Basit Azam**<sup></sup>  
*Department of Computer Science & Engineering*  
Tezpur University  
Napaam - 784 028, Tezpur, Assam, INDIA  
📧 [mdbasit@tezu.ernet.in](mailto:mdbasit@tezu.ernet.in)

**2<sup>nd</sup> Sarangthem Ibotombi Singh**  
*Department of Computer Science & Engineering*  
Tezpur University  
Napaam - 784 028, Tezpur, Assam, INDIA  
📧 [sis@tezu.ernet.in](mailto:sis@tezu.ernet.in)


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
### Key Results

| Metric | HbA1c cohort (n = 29) | FBG cohort (n = 38) |
|--------|:----------------------:|:--------------------:|
| Best model | Extra Trees | Extra Trees |
| R² | 0.222 | 0.086 |
| MAE (original scale) | 1.18 percentage points | 2.27 mmol/L (41 mg/dL) |
| Pearson *r* (*p*) | 0.476 (0.009) | 0.344 (0.034) |
| Permutation test *p* | 0.002 | 0.002 |
| Bootstrap 95% CI for R² | [0.13, 0.82] | [0.10, 0.72] |
| Age normalisation benefit | None (all 20 combinations ≤ baseline) | None |

> Bootstrap CIs exclude zero for both cohorts but remain wide, reflecting small-sample uncertainty. These findings should be interpreted as hypothesis-generating.

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

### 3. Run Preprocessing

```bash
python complete_preprocessing.py
```

This creates `processed_data_v2/` with extracted features, separated cohort targets, and LOSO fold definitions.

### 4. Run Complete Analysis

```bash
python run_complete_analysis.py
```

This executes three steps sequentially:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `comprehensive_baseline_revised.py` | 20-model baseline comparison with LOSO + 6 age-adjustment methods + 20-parameter sensitivity grid |
| 2 | `ablation_study_revised.py` | 13-configuration feature domain ablation |
| 3 | `validation_framework_revised.py` | Permutation testing (n=500), bootstrap CIs (n=500), residual diagnostics, learning curves |

> **Note:** The runner checks that `processed_data_v2/` exists before proceeding. All scripts must be run in order as each depends on the previous step's outputs.

---

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

## Output Structure

```
./
├── processed_data_v2/                       # From complete_preprocessing.py
│   ├── features.csv                         # 105 extracted features
│   ├── signal_specifications.json           # ECG/HRV signal documentation
│   ├── ecg_scaling_logs.json                # ECG amplitude scaling audit
│   ├── targets/
│   │   ├── hba1c_cohort.csv                 # HbA1c targets (n=29)
│   │   └── fbg_cohort.csv                   # FBG targets (n=38)
│   └── loso_splits/
│       ├── hba1c_cohort/                    # LOSO fold definitions
│       └── fbg_cohort/
│
├── analysis_results_v3/                     # From comprehensive_baseline_revised.py
│   ├── hba1c_cohort/
│   │   ├── baseline_comparison.csv          # 20-model R², MAE, correlation
│   │   ├── baseline_comparison.png          # Model comparison bar charts
│   │   ├── age_adjustment_comparison.csv    # 6 methods × R² results
│   │   ├── prediction_scatter.png           # Predicted vs actual plot
│   │   └── summary.json                     # Cohort-level summary
│   ├── fbg_cohort/
│   │   └── ...
│   ├── dual_cohort_model_comparison.png     # Side-by-side cohort comparison
│   ├── feature_importance_by_domain.png     # Domain contribution analysis
│   ├── feature_selection_stability_heatmap.png  # Fold-by-feature binary heatmap
│   └── age_sensitivity_heatmap.png          # 5×4 parameter grid heatmap
│
├── ablation_results_v3/                     # From ablation_study_revised.py
│   ├── hba1c_cohort/
│   │   ├── ablation_results.csv             # 13 configurations × metrics
│   │   └── ablation_figure.png              # Domain ablation bar chart
│   └── fbg_cohort/
│       └── ...
│
└── validation_results_v3/                   # From validation_framework_revised.py
    ├── hba1c_cohort/
    │   ├── validation_report.json           # Full statistical report
    ├── fbg_cohort/
        └── ...

```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `complete_preprocessing.py` | Loads raw clinical, ECG, and sleep data from the Mendeley dataset; extracts 105 features across 6 domains; validates ECG signal amplitude and documents scaling; creates separated HbA1c/FBG cohort targets and LOSO fold splits |
| `comprehensive_baseline_revised.py` | Runs 20 models under LOSO with within-fold SelectKBest (k=15) and StandardScaler; compares 6 age-adjustment methods; performs 20-combination sensitivity analysis; generates publication figures |
| `ablation_study_revised.py` | Evaluates 13 feature-domain configurations using Bayesian Ridge under LOSO with within-fold preprocessing; quantifies contributions of clinical, ECG, HRV, sleep, and demographic feature groups |
| `validation_framework_revised.py` | Permutation testing (n=500), bootstrap 95% CIs (n=500, subject-level resampling), residual diagnostics (Shapiro-Wilk, bias, heteroscedasticity), learning curve analysis |
| `run_complete_analysis.py` | Sequential runner for the three analysis steps; checks `processed_data_v2/` exists before proceeding |

---

---

## Methodology

### Separated Glycemic Targets

HbA1c (reflecting 3-month average glycemic control) and fasting blood glucose (FBG; reflecting acute metabolic status) are analysed as strictly separate cohorts, preventing the common methodological error of combining fundamentally different glucose metrics.

### Cross-Validation Hygiene

All preprocessing occurs strictly within each LOSO fold:

```
For each held-out subject:
  1. SelectKBest(f_regression, k=15) fitted on training subjects only
  2. StandardScaler fitted on training subjects only
  3. Held-out subject transformed using training-derived parameters
  4. Model fitted and prediction recorded
```

This prevents information leakage from held-out test subjects into feature selection or scaling — the single most impactful methodological correction in this study.

### 20 Baseline Models

| Category | Models |
|----------|--------|
| Naïve (2) | Mean predictor, Median predictor |
| Linear (7) | OLS, Ridge (α=0.1, 1.0), Lasso (α=0.1), ElasticNet, Bayesian Ridge, Huber Regressor |
| Tree ensembles (4) | Random Forest, Extra Trees, Gradient Boosting, AdaBoost |
| SVM (3) | SVR with RBF, linear, and polynomial (degree 2) kernels |
| Neural networks (4) | MLP (32), MLP (64,32), MLP (128,64,32), MLP (64,32) tanh |

All models use minimally configured hyperparameters (scikit-learn defaults with no nested tuning), deliberately providing conservative baselines on small samples.

### 6 Age-Adjustment Methods

| Method | Description |
|--------|-------------|
| No adjustment | Baseline (raw features) |
| Proposed formula | HRV / (age/65 + 0.1), threshold from Umetani et al. (1998) |
| Residualisation | Regress age out of HRV features via linear regression |
| Age-bin z-score | Z-score within age quartiles (young/middle/senior/elderly) |
| Polynomial interaction | Age² + HRV × age interaction terms |
| Simple division | HRV / age |

Additionally, a sensitivity analysis tests 20 parameter combinations (5 age thresholds: 55, 60, 65, 70, 75 × 4 stability constants: 0.05, 0.1, 0.15, 0.2). No combination improves over baseline.

### 13-Configuration Ablation Study

Using Bayesian Ridge for transparent, interpretable domain attribution:

| Configuration | Features included |
|---------------|-------------------|
| Full Model | All 105 features (baseline) |
| No Age Normalisation | All except age-normalised HRV |
| Only Age-Normalised + Demographics | Age-normalised HRV + demographics only |
| No Sleep-Stage HRV | All except per-stage HRV features |
| HRV Only | Stage-specific HRV + age-normalised HRV + demographics |
| ECG Only | ECG morphology + demographics |
| Clinical Only | Clinical measurements + demographics |
| No ECG | All except ECG morphology features |
| No Clinical | All except clinical measurement features |
| Demographics Only | Age, height, weight only |
| Only Deep Sleep HRV | Deep sleep HRV + age-normalised + demographics |
| Only REM HRV | REM sleep HRV + age-normalised + demographics |
| Only Rapid Sleep HRV | Rapid sleep HRV + age-normalised + demographics |

### Statistical Validation

- **Permutation testing:** n = 500 permutations; both cohorts p = 0.002
- **Bootstrap CIs:** n = 500 subject-level resamples; 95% confidence intervals
- **Residual diagnostics:** Shapiro-Wilk normality, mean-bias test, heteroscedasticity analysis
- **Learning curves:** Sample-size adequacy assessment

---

## Feature Domains (105 features)

| Domain | Count | Examples |
|--------|:-----:|---------|
| Demographics | 3 | Age, height, weight |
| Clinical measurements | 20 | Blood pressure, lipid panel, renal/liver function, haematology |
| ECG morphology | 24 | Signal statistics (mean, SD, range, SNR) for 24h / sleep / daytime |
| HRV time-domain | 33 | Mean RR, SDNN, RMSSD, pNN50, CV per sleep stage (DS, REM, RS) |
| Age-normalised HRV | 3 | Mean RR normalised by age factor per sleep stage |
| Sleep quality | 22 | PSQI components (11), CPC-derived metrics (11) |

---

***📚 Citation***

**If you use this work, please cite this repository**



## 🤝 Contributing
We welcome contributions!

## 📄 License  
This project uses an MIT License. See the [LICENSE file](LICENSE) for details.  
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE). 

This project is for academic research purposes. Please contact the author for commercial use.

## 🙏 Acknowledgments  
The authors acknowledge support from the Google Cloud Research Credits program under 
Award GCP19980904 and partial computing resources from Google’s TPU Research Cloud (TRC), 
both of which provided critical infrastructure for this research.

### Funding:
The authors declare no funding was received for this research.

## References
<a id="1">[1]</a> 
Cheng, Wenquan; Chen, Hongsen; Tian, Leirong; Ma, Zhimin; Cui, Xingran (2023), “Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus ”, Mendeley Data, V4, doi: 10.17632/9c47vwvtss.4
