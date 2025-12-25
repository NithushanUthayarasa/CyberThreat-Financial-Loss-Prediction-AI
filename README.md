# 🔐 Cyber Threats & Financial Loss Prediction (2015–2024)

An **end-to-end machine learning system** to predict **financial losses caused by cybersecurity threats**.  
This project implements a **complete Step-by-Step ML pipeline (Step 1 → Step 10)**, starting from raw data cleaning and ending with **automated best-model deployment**.

---

## 📌 Project Overview

Cybersecurity incidents are increasing globally, causing **significant financial damage across industries**.  
Organizations struggle to quantify potential losses due to complex factors such as **attack type, vulnerabilities, and incident resolution time**.

This project addresses that challenge by building a **production-ready ML pipeline** that predicts financial loss exposure from cyber attacks.

---

## ❓ Problem Statement

Estimating financial losses from cyber attacks is difficult due to:
- Diverse attack types  
- Varying resolution times  
- Industry-specific vulnerabilities  

Without accurate prediction, organizations risk:
- Poor resource allocation  
- Ineffective incident response  
- Underinvestment or overinvestment in security measures  

---

## 🎯 Purpose & Objectives

The goal is to predict **financial loss (in million USD)** from cyber attacks using structured data and machine learning, enabling:

- 🔍 Proactive cyber-risk assessment  
- 📊 Better resource allocation  
- 🏭 Industry-specific cyber-risk profiling  
- 💰 Data-driven security investment decisions  

---

## 🧾 Dataset & Features

**Source:** Kaggle — *Global Cybersecurity Threats (2015–2024)*  
**Records:** ~3,000 cybersecurity incidents  
**Original Features:** 10 columns  

### Key Original Columns
- Country  
- Year  
- Attack Type  
- Target Industry  
- Financial Loss *(target variable)*  
- Number of Affected Users  
- Attack Source  
- Security Vulnerability Type  
- Defense Mechanism Used  
- Incident Resolution Time  

### 🔧 Engineered Features
- **Loss_per_User** = Financial Loss ÷ Number of Affected Users  
- **AttackType_TargetIndustry** (interaction feature)

> ⚡ After encoding categorical variables, the final model uses **~20–30 input features**.

---

## 🔄 End-to-End ML Pipeline (Step 1 → Step 10)

1. Feature Selection & Data Cleaning  
2. Preprocessing Pipeline (scaling, encoding, train-test split)  
3. Baseline Model Training  
4. Feature Engineering & Hyperparameter Tuning  
5. Classification Framing (High-Risk Loss Detection)  
6. Baseline vs Tuned Model Comparison  
7. Visual Performance Analysis  
8. Feature Importance Analysis  
9. Model Benchmarking & Variance Check  
10. Final Model Deployment  

---

## 🧠 Feature Engineering & Preprocessing

- Removed irrelevant features  
- Handled missing values:
  - Median for numeric features  
  - Mode for categorical features  
- Dropped duplicate records  
- Created interpretable interaction features  

---

## 🧪 Machine Learning Models Used

- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  

---

## 🔍 Evaluation Metrics

### Regression Metrics
- RMSE  
- MAE  
- R² Score  

### Classification Metrics (High-Risk Loss Detection)
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC–AUC  

---

## 📊 Results Summary

| Step | Description | Model / Metrics | Notes |
|----|------------|----------------|------|
| 3 | Baseline Training | Negative R² | Underfitting |
| 4 | Feature Engineering + Tuning | **LightGBM:** RMSE=3.54, MAE=1.64, R²=0.985 | Best performance |
| 5 | Classification Framing | Accuracy=0.97, F1=0.94, ROC–AUC=0.995 | High-risk detection |
| 6 | Baseline vs Tuned | Tuned R² ≈ 0.97–0.98 | Massive improvement |
| 8 | Feature Importance | Loss_per_User, Resolution Time | Key drivers |
| 9 | Model Benchmarking | See table below | Generalization check |

---

## 📈 Model Benchmarking

| Model | Train R² | Test R² | Diagnosis |
|-----|---------|--------|----------|
| LightGBM | 0.993 | 0.985 | ✅ Good generalization |
| XGBoost | 0.9998 | 0.980 | ⚠ High variance |
| Gradient Boosting | 0.996 | 0.977 | ✅ Good generalization |
| CatBoost | 0.999 | 0.973 | ⚠ High variance |
| Random Forest | 0.989 | 0.972 | ✅ Good generalization |

**Inference Time:** ~0.02 seconds per batch

---

## 🛠 Handling High Variance

- Regularization (`reg_lambda`, `reg_alpha`, `l2_leaf_reg`)  
- Reduced model complexity (`max_depth`, `n_estimators`)  
- k-Fold Cross-Validation  
- Feature selection to remove noisy features  
- Dataset expansion / augmentation  

---

## 🚀 Deployment (Step 10)

- ✅ Best model: **LightGBM**
- 💾 Saved as: `models/production_model.joblib`
- 🔁 Reusable prediction function
- ♻️ Fully reproducible pipeline for real-world use

---

## 📁 Project Structure

```text
CyberThreats_FinancialLoss_Prediction_ML/
│── data/
│   ├── raw/          # Original dataset CSVs
│   ├── interim/      # Cleaned & selected features
│   └── processed/    # Step-wise processed data
│
│── notebooks/        # Step 1 → Step 10 notebooks
│── models/           # Trained models (.joblib)
│── reports/          # Reports & analysis
│── outputs/          # Images & pipelines
│── plots/            # Feature importance plots
│── README.md
│── requirements.txt
```

##🔧 How to Run

git clone https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI
cd CyberThreats_FinancialLoss_Prediction_ML
pip install -r requirements.txt
jupyter notebook


➡️ Run notebooks Step 1 → Step 10 sequentially

##🌍 Business & Social Impact

Cybersecurity risk assessment
Financial loss forecasting
Incident response prioritization
Industry-specific cyber-risk profiling
Data-driven security investment decisions

##🌟 Highlights

✅ Full production-ready ML pipeline
📈 R² = 0.985, ROC–AUC = 0.995
🔍 Interpretable features improve insights
⚙️ High-variance models analyzed & mitigated
🔁 Fully reproducible and deployable

##🛠 Tech Stack
Python
Pandas, NumPy
Scikit-Learn
LightGBM, XGBoost, CatBoost
Matplotlib, Seaborn
Jupyter Notebook

##👤 Author

Nithushan Uthayarasa
Machine Learning | Cybersecurity Analytics
