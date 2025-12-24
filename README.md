Here is the full, complete version of your Cyber Threats & Financial Loss Prediction README. This includes all sections from start to end, formatted exactly like your Asset Portal project using only standard Markdown symbols (#, *, ---, ```).

You can copy the raw text below:

code
Markdown
download
content_copy
expand_less
# 🔐 Cyber Threats & Financial Loss Prediction (2015–2024)

**I built an end-to-end machine learning system to predict financial losses caused by cybersecurity threats. This project implements a complete Step-by-Step ML pipeline (Step 1 → Step 10), starting from raw data cleaning and ending with automated best-model deployment to enhance proactive risk assessment and security investment decisions.**

---

## 📸 Visual Analysis & Reports

### 📊 Model Performance

* [Feature Importance Plot](https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI/blob/main/plots/feature_importance.png)
* [Actual vs Predicted Loss](https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI/blob/main/plots/prediction_analysis.png)
* [Residual Distribution](https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI/blob/main/plots/residuals.png)
* [ROC-AUC Curve (Classification)](https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI/blob/main/plots/roc_auc.png)

### 📈 Benchmarking

* [Model Comparison Table](https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI/blob/main/reports/benchmark_summary.csv)
* [Hyperparameter Tuning Logs](https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI/blob/main/reports/tuning_results.txt)

---

## 🎯 Purpose

The purpose of this project is to predict financial loss (in million USD) from cyber attacks using structured data and machine learning. This enables organizations to:

* **Proactive risk assessment:** Identify high-impact threats before they occur.
* **Better resource allocation:** Assign security budgets where they are needed most.
* **Industry profiling:** Understand risk levels specific to sectors like Finance or Healthcare.
* **Data-driven decisions:** Move away from guesswork toward evidence-based security investments.

---

## 🚨 Problem Statement

Organizations often face these challenges when managing cyber risks:

* **Financial Unpredictability:** Difficulty in quantifying the potential dollar loss of a breach.
* **Complexity of Factors:** Threats depend on attack type, resolution time, and vulnerability type.
* **Resource Misalignment:** Overspending on low-risk areas while leaving high-loss gaps open.

**Solution:** This ML system provides a unified platform to:

* Track the full lifecycle of a cyber incident from detection to resolution.
* Predict financial impact using advanced algorithms like LightGBM and XGBoost.
* Identify key drivers of loss such as `Loss_per_User` and `Resolution Time`.
* Provide a reproducible pipeline for real-world security data.

---

## 🧠 Key Features

### ⚙️ Data Engineering
* Automated cleaning of raw Kaggle dataset (3,000+ records).
* Feature Engineering: `Loss_per_User` and interaction variables.
* Pipeline integration for scaling and encoding.

### 🧪 Machine Learning
* Multi-model benchmarking (Random Forest, XGBoost, LightGBM, CatBoost).
* Hyperparameter tuning for R² optimization.
* High-risk classification framing (Threshold > $75M).
* Model serialization for production deployment.

---

## 🏗️ System Workflow

### 🔄 The 10-Step Pipeline

1. **Data Preparation:** Raw cleaning and feature selection.
2. **Preprocessing:** Building the pipeline (scaling and encoding).
3. **Baseline Training:** Establishing initial performance floors.
4. **Optimization:** Engineering features and tuning parameters.
5. **Evaluation:** Applying classification framing for high-risk detection.
6. **Comparison:** Performance analysis of Baseline vs. Tuned models.
7. **Visualization:** Generating performance and error distribution plots.
8. **Importance Analysis:** Identifying top features driving financial loss.
9. **Benchmarking:** Final model selection based on generalization.
10. **Deployment:** Saving the best model to `models/production_model.joblib`.

---

## 🗂️ Project Structure

```text
CyberThreats_FinancialLoss_Prediction_ML/
├── data/
│   ├── raw/          # Original dataset CSVs
│   ├── interim/      # Cleaned features CSVs
│   └── processed/    # Stepwise results
├── notebooks/        # Step 1 → Step 10 notebooks
├── models/           # Saved models (.joblib)
├── reports/          # Final PDFs and CSVs
├── outputs/          # Pipeline artifacts
├── plots/            # Feature importance plots
├── README.md
└── requirements.txt
```
## 📊 Results Summary
Model Benchmarking
Model	Train R²	Test R²	Diagnosis
LightGBM	0.993	0.985	✅ Good generalization
XGBoost	0.999	0.980	⚠ High variance
Gradient Boosting	0.996	0.977	✅ Good generalization
CatBoost	0.999	0.973	⚠ High variance
RandomForest	0.989	0.972	✅ Good generalization
🛠️ Tech Stack

Language: Python 3.x

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn, LightGBM, XGBoost, CatBoost

Visualization: Matplotlib, Seaborn

Tooling: Jupyter Notebook, Joblib

🚀 Run the Project

1️⃣ Clone the repository:

code
Bash
download
content_copy
expand_less
git clone https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI
cd CyberThreats_FinancialLoss_Prediction_ML

2️⃣ Install dependencies:

code
Bash
download
content_copy
expand_less
pip install -r requirements.txt

3️⃣ Run the pipeline:

Execute notebooks Step 1 through Step 10 in order within Jupyter.

📌 Highlights

Achieved R² = 0.985 and ROC–AUC = 0.995.

Full production-ready ML pipeline from cleaning to deployment.

Implemented specific strategies (Regularization/CV) to handle high variance.

Detailed feature importance analysis for business insight.

Fully reproducible structure and environment.

📄 License

This project is for educational and learning purposes only.

Author: Nithushan Uthayarasa

code
Code
download
content_copy
expand_less
