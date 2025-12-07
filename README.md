# Re-evaluating HRV Biomarkers for Glucose Sensing: The Impact of Age Normalisation and Subject-Independent Validation
### Research prototype • Public domain (CC0-1.0) • Not for clinical use

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

##  Output Structure

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
    ├── hba1c_cohort/
    │   ├── validation_report.json
    │   ├── permutation_test.png
    │   ├── bootstrap_ci.png
    │   └── residual_analysis.png
    └── fbg_cohort/
        └── ...

```

---

## 🔬 Methodology Improvements

### 1. Separated Glucose Targets 

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

### 3. Comprehensive Baselines 

| Model Category | Models Included |
|----------------|-----------------|
| Naive | Mean, Median |
| Linear | Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge |
| Tree-based | Random Forest, Gradient Boosting, Extra Trees, AdaBoost |
| SVM | RBF, Linear, Polynomial kernels |
| **Neural Networks** | MLP (32), MLP (64,32), MLP (128,64,32) |

### 4. Age Adjustment Comparison 

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

### `complete_preprocessing.py`

Main preprocessing pipeline:
- Loads clinical, ECG, and sleep data
- Extracts HRV features from RR intervals
- Creates age-normalized features
- **Separates HbA1c and FBG cohorts**
- Creates LOSO and temporal validation splits
- Documents all signal specifications

### `comprehensive_baseline.py`

Baseline model comparison:
- 20+ models including neural networks
- LOSO cross-validation
- Age adjustment method comparison
- Permutation testing
- Bootstrap confidence intervals
- Publication-quality figures

### `ablation_study.py`

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


##  Troubleshooting

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

***📚 Citation***

**If you use this work, please cite our paper:**




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
