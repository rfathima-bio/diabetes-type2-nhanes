# Type 2 Diabetes Prediction & Risk Analysis (CDC NHANES)

An end-to-end epidemiological modeling and non-invasive machine learning project analyzing Type 2 Diabetes risk using microdata from the Centers for Disease Control and Prevention (CDC) National Health and Nutrition Examination Survey (NHANES).

---

## 📌 Project Overview

Type 2 Diabetes affects over 37 million Americans, and approximately **1 in 5 individuals remain undiagnosed** due to the requirement for invasive venous blood testing (HbA1c, fasting plasma glucose).

This project addresses two core questions:
1. **Nature vs. Nurture**: How much does biological family history determine diabetes risk compared to modifiable lifestyle choices (dietary sugar, fiber, physical activity, and BMI)?
2. **Non-Invasive Machine Learning Screen**: Can we accurately identify individuals with clinical diabetes (HbA1c ≥ 6.5%) **without using any blood or laboratory tests**?

---

## 🔬 Key Research Findings

### 1. Family History vs. Modifiable Lifestyle (Nature vs. Nurture)
- In hierarchical linear regression on HbA1c ($N = 4,095$), **family history** adds **+3.6%** of explained variance, while **lifestyle factors** add **+3.5%** above demographic aging. They have virtually equal predictive power.
- **Genetic Risk Can Be Neutralized**: Individuals with a first-degree family history who maintain a **normal weight (BMI < 25)** have an average HbA1c of **5.71%**, which is essentially identical to individuals with **no family history who are overweight (5.69%)**.
- **Diet Acts Through Adiposity**: Single-day dietary sugar and fiber intake do not show strong direct acute effects; rather, long-term diet drives chronic insulin resistance primarily through **measured BMI and visceral adiposity (waist circumference)**.

### 2. Non-Invasive Machine Learning Classifier (No Blood Work Required)
- **Problem**: Binary classification to predict $\text{HbA1c} \ge 6.5\%$ on $N = 4,897$ adults aged 18–79.
- **Feature Set**: 16 non-blood features (Age, Sex, Race/Ethnicity, Poverty Ratio, Waist Circumference, BMI, Weight, Height, Systolic BP, Diastolic BP, Resting Pulse, Family History, MVPA, Sedentary Hours, Smoking History, Sleep Duration).
- **Test Set Performance (20% holdout, 980 unseen adults)**:
  - **Regularized Logistic Regression**: **ROC-AUC = 0.850** | PR-AUC = 0.404 | F1 = 0.475
  - **Random Forest**: **ROC-AUC = 0.830** | PR-AUC = 0.419 | F1 = 0.447
  - **HistGradientBoosting**: **ROC-AUC = 0.798** | PR-AUC = 0.365 | F1 = 0.378
- **Top Predictors (Odds Ratios)**:
  1. **Age**: OR = **3.10** per SD (~16 years)
  2. **Family History**: OR = **3.02** (first-degree relative)
  3. **Waist Circumference**: OR = **1.88** per SD (outperforms BMI alone)
  4. **Resting Heart Rate / Pulse**: OR = **1.64** per SD
  5. **Male Sex**: OR = **1.59** (+59% higher odds vs. women)

---

## 📊 Machine Learning Benchmark Results

| Model | 5-Fold CV ROC-AUC | 5-Fold CV PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Accuracy |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (L2)** | **0.827** | **0.413** | **0.850** | 0.404 | **0.475** | 75.0% |
| **Random Forest** | 0.824 | 0.408 | 0.830 | **0.419** | 0.447 | 76.2% |
| **HistGradientBoosting** | 0.801 | 0.367 | 0.798 | 0.365 | 0.378 | 81.2% |

> **Clinical Significance**: Achieving an **ROC-AUC of 0.850** using zero blood work demonstrates that standard community or outpatient triage (scale, blood pressure cuff, measuring tape, 5-question intake) can reliably prioritize who needs a definitive venous blood test.

---

## 💻 Interactive Simulation Dashboard

The project includes a standalone, zero-dependency web dashboard:
- **File**: `diabetes_risk_dashboard.html`
- **Features**:
  - Interactive sliders for all 16 non-blood features.
  - Real-time client-side probability scoring for $\text{HbA1c} \ge 6.5\%$.
  - Dynamic **Top 5 Risk Drivers** showing which factors raise or lower risk.
  - One-click archetype presets (Young & Active: ~2% risk; Typical US Adult: ~16% risk; High Risk Metabolic: ~73% risk).
- **How to Run**: Simply double-click `diabetes_risk_dashboard.html` or open it in any web browser (Chrome, Safari, Firefox).

---

## 🛠️ Step-by-Step Python Pipeline

### 1. Ingestion & Preprocessing
`load_nhanes.py` & `diabetes_family_vs_lifestyle.py`
- Downloads SAS XPORT files from the CDC website (`DEMO_J`, `GHB_J`, `BMX_J`, `BPX_J`, `MCQ_J`, `PAQ_J`, `DR1TOT_J`, `SMQ_J`, `SLQ_J`).
- Merges all modules on participant sequence number (`SEQN`).
- Filters for adults aged 18–79 with positive Mobile Examination Center survey weights (`WTMEC2YR > 0`).

### 2. Nature vs. Nurture Regression
`diabetes_family_vs_lifestyle.py`
- Fits hierarchical OLS models comparing demographics, family history (`MCQ300C`), and lifestyle factors.
- Calculates standardized beta coefficients and 4-group joint distributions.

### 3. Non-Invasive Machine Learning Benchmark
`ml_diabetes_classifier.py`
- Implements stratified 80/20 train/test splitting.
- Sets up scikit-learn preprocessing pipelines (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`).
- Runs 5-fold stratified cross-validation and evaluates on the unseen test set.
- Exports calibrated weights into `diabetes_model_payload.json` for the web dashboard.

---

## 📁 Repository Structure

```
├── README.md                              # This summary document
├── diabetes_risk_dashboard.html           # Interactive web risk simulator (drag & drop to browser)
├── diabetes_model_payload.json            # Trained model weights & scaling constants
│
├── Python Scripts
│   ├── ml_diabetes_classifier.py          # ML training, cross-validation & evaluation
│   ├── diabetes_family_vs_lifestyle.py    # Nature vs Nurture regression analysis
│   └── load_nhanes.py                     # Automated CDC data downloader & merger
│
├── Reports & Visualizations
│   ├── diabetes_project_summary.md        # Comprehensive in-depth project documentation
│   ├── nhanes_ml_diabetes_benchmark.png   # ROC, PR curves & feature importance plot
│   └── nhanes_diabetes_family_vs_lifestyle.png # Forest plot & 4-group distribution boxplot
```

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/your-username/nhanes-diabetes-ml.git
cd nhanes-diabetes-ml

# Install required packages
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn pyreadstat requests

# Run the machine learning benchmark
python ml_diabetes_classifier.py

# Run the Nature vs Nurture regression analysis
python diabetes_family_vs_lifestyle.py
```

---

## 📄 Data Citation & License
- **Data Source**: CDC National Center for Health Statistics (NCHS), National Health and Nutrition Examination Survey (Public Domain).
- **Code License**: MIT License.
