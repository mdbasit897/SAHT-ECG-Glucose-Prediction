# Age-Normalized HRV for Sleep-Aware Glucose Prediction
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

## 🔬 Abstract

This repository contains code and resources for a pilot, sleep-aware machine learning study on non-invasive glucose prediction using age-normalized heart rate variability (HRV) features derived from single-lead ECG. We analyze sleep-stage–specific HRV (REM/Deep/“Rapid”) and show that simple age normalization of HRV improves regression performance under 5-fold cross-validation on a cohort of 43 subjects.

**Key Contributions:**
- Age-normalized HRV improves log-glucose prediction by ~25.6% R² over non-normalized HRV in this pilot.
- Sleep-stage–specific features matter—REM sleep features rank among the strongest predictors.
- Lightweight model: scikit-learn BayesianRidge with 5-fold CV; clear, reproducible baseline.
= Single-lead ECG compatibility; preliminary tolerance analysis suggests practical signal value (research only).

## Dataset
This repo expects overnight ECG + sleep staging + clinical glucose for 43 adult subjects. In the paper we used publicly available data "Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus"[[1]](#1) from Mendeley Data(https://data.mendeley.com/datasets/9c47vwvtss/4):

## Method (in brief)

- **ECG → RR intervals**: R-peak detection with artifact handling and outlier filtering.

- **Sleep stages**: Use provided labels (AASM criteria) to split RR intervals into REM / Deep Sleep (DS) / Rapid Sleep (RS) segments.

- **HRV features (per stage)**: Time-domain metrics — Mean RR, RMSSD, SDNN, pNN50, range.

- **Age normalization**:
  $$
  \text{HRV}_{\text{age-norm}} = \frac{\text{HRV}_{\text{raw}}}{\frac{\text{age}}{65} + \epsilon}, \quad \epsilon = 0.1
  $$
  Applied to Mean RR in REM, DS, and RS stages.

- **Target engineering**: Natural log of glucose in mmol/L (ensure units are mmol/L before log-transform).

- **Feature selection**: Pearson correlation to log-glucose; retain features with *p* < 0.2 and keep top-*k* (e.g., 15) by absolute correlation |r|.

- **Model**: scikit-learn `BayesianRidge`; 5-fold cross-validation (random CV on samples).

- **Metrics**: R², MAE, Pearson *r* (with *p*-values), and tolerance (% within ±1.0 / ±1.5 / ±2.0 mmol/L).

## Reproducing Key Results (Pilot)

Using 5-fold cross-validation with `BayesianRidge`:

- **R²** ≈ 0.161 (±0.010)  
- **MAE** ≈ 0.182 mmol/L  
- **Pearson r** ≈ 0.409 (*p* < 0.001)  

### Ablation Study
- Age normalization improves R² by **~25.6%** compared to non-normalized HRV features.

### Tolerance (Prediction Accuracy)
- **~68.2%** within ±1.0 mmol/L  
- **~84.1%** within ±1.5 mmol/L  
- **~95.3%** within ±2.0 mmol/L  

> ⚠️ **Note**: These are pilot, within-dataset cross-validation results intended for research purposes. They are **not clinical accuracy claims** and should not be interpreted as such.

***📚 Citation***

**If you use this work, please cite our paper:**




## 🤝 Contributing
We welcome contributions!

## 📄 License  
This project uses an MIT License. See the [LICENSE file](LICENSE) for details.  
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) 

## 🙏 Acknowledgments  
The authors acknowledge support from the Google Cloud Research Credits program under 
Award GCP19980904 and partial computing resources from Google’s TPU Research Cloud (TRC), 
both of which provided critical infrastructure for this research.

### Funding:
The authors declare no funding was received for this research.

## References
<a id="1">[1]</a> 
Cheng, Wenquan; Chen, Hongsen; Tian, Leirong; Ma, Zhimin; Cui, Xingran (2023), “Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus ”, Mendeley Data, V4, doi: 10.17632/9c47vwvtss.4
