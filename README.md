# 🔐 Cyber Threats & Financial Loss Prediction (2015–2024)

An **end-to-end machine learning project** that predicts **financial losses caused by cybersecurity threats** using structured global incident data.  
The project implements a **complete ML pipeline (Step 1 → Step 10)**, starting from raw data cleaning and ending with automated best-model deployment.

---

## 📌 Project Overview

Cybersecurity incidents are increasing globally, causing significant financial damage across industries.  
Organizations struggle to quantify potential losses due to complex factors such as attack type, vulnerabilities, and incident resolution time.

---

## 🚨 Problem Statement

Organizations struggle to accurately estimate the **financial losses caused by cybersecurity attacks** due to:

- Diverse and evolving cyber threat types  
- Variations in incident resolution time  
- Industry-specific vulnerabilities and defense mechanisms  

Traditional risk assessment methods fail to capture the **complex, non-linear relationships** between these factors, leading to:

- Poor resource allocation  
- Ineffective incident response planning  
- Underinvestment or misallocation of cybersecurity budgets  

This project formulates the problem as a **machine learning task** to **predict financial loss (in million USD)** from structured cyber incident data and to **identify high-risk loss events** using both **regression and classification approaches**.


---

## 🎯 Purpose

This project is designed to predict **financial losses (in million USD)** resulting from cyber attacks using advanced machine learning techniques. By transforming structured incident data into actionable insights, the system enables organizations to:

- **Proactively assess cyber‑risk** before incidents escalate  
- **Allocate resources more effectively** to minimize financial and operational impact  
- **Profile risks by industry**, recognizing that different sectors face unique vulnerabilities  
- **Support data‑driven security investment decisions**, ensuring budgets are directed where they deliver the greatest protection


---

## 🧾 Dataset & Features

- **Source:** [Kaggle - Global Cybersecurity Threats, 2015–2024](https://www.kaggle.com/datasets/atharvasoundankar/global-cybersecurity-threats-2015-2024)
- **Rows:** ~3,000 cybersecurity incidents  
- **Original Columns:** 10 (e.g., Country, Year, Attack Type, Target Industry, Financial Loss, Number of Affected Users, Attack Source, Security Vulnerability Type, Defense Mechanism Used, Incident Resolution Time)

### Engineered Features

- **Loss_per_User** = Financial Loss ÷ Number of Affected Users  
- **AttackType_TargetIndustry** (interaction feature)  

> After encoding categorical variables, the model uses ~20–30 input features.

---

## 🔄 End-to-End ML Pipeline (Step 1 → Step 10)

1. Feature Selection & Cleaning  
2. Preprocessing Pipeline (scaling, encoding, splitting)  
3. Baseline Model Training  
4. Feature Engineering + Hyperparameter Tuning  
5. Additional Metrics (Classification Framing)  
6. Baseline vs Tuned Comparison  
7. Visual Performance Analysis  
8. Feature Importance Analysis  
9. Model Benchmarking & Variance Check  
10. Final Model Deployment  

---

## 🧠 Feature Engineering & Preprocessing

- **Data Cleaning:** Removed irrelevant features, handled missing values (median for numeric, mode for categorical), dropped duplicates  
- **Feature Engineering:** `Loss_per_User`, `AttackType_TargetIndustry`  

---

## 🧪 Machine Learning Models Used

- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- CatBoost  

---

## 🔍 Evaluation Metrics

**Regression:** RMSE, MAE, R²  
**Classification (High-Risk Loss Detection):** Accuracy, Precision, Recall, F1-Score, ROC–AUC  

---

## 📊 Results Summary

| Step | Description | Model / Metrics | Notes |
|------|------------|----------------|-------|
| 3 | Baseline Training | Negative R² | Underfitting |
| 4 | Feature Engineering + Tuning | LightGBM: RMSE=3.54, MAE=1.64, R²=0.985 | XGBoost & CatBoost also strong (R² ≈ 0.97–0.98) |
| 5 | Classification Framing | Threshold ~75M USD → LightGBM: Accuracy=0.97, F1=0.94, ROC–AUC=0.995 | High-risk detection |
| 6 | Baseline vs Tuned | Tuned R² = 0.97–0.98 | Massive improvement |
| 8 | Feature Importance | Loss_per_User, Incident Resolution Time, AttackType_TargetIndustry | Key drivers |
| 9 | Model Benchmarking | See table below | LightGBM generalizes best; XGBoost/CatBoost show high variance |

**Model Benchmarking Table**

| Model | Train R² | Test R² | Diagnosis |
|-------|----------|---------|-----------|
| LightGBM | 0.993 | 0.985 | ✅ Good generalization |
| XGBoost | 0.9998 | 0.980 | ⚠ High variance |
| Gradient Boosting | 0.996 | 0.977 | ✅ Good generalization |
| CatBoost | 0.999 | 0.973 | ⚠ High variance |
| RandomForest | 0.989 | 0.972 | ✅ Good generalization |

**Inference Time:** ~0.02 seconds per batch  

---

## 🛠 Handling High Variance

- Regularization: `reg_lambda`, `reg_alpha` (XGBoost/LightGBM), `l2_leaf_reg` (CatBoost)  
- Reduce Complexity: Limit `max_depth`, reduce number of estimators  
- Cross-Validation Tuning: Use k-fold CV  
- Feature Selection: Drop noisy/redundant features  
- Data Augmentation / More Samples: Expand dataset or simulate variations  

---

## 🚀 Deployment (Step 10)

- **Best Model:** LightGBM saved as `models/production_model.joblib`  
- Reusable prediction function for new cyber-threat data  
- Fully reproducible pipeline  

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
│── README.md

```
---

## 🔧 How to Run the Project

```bash
git clone https://github.com/NithushanUthayarasa/CyberThreat-Financial-Loss-Prediction-AI
cd CyberThreats_FinancialLoss_Prediction_ML
pip install -r requirements.txt
jupyter notebook
```
* ➡️ Run notebooks Step 1 → Step 10 sequentially

## 🌍 Business & Social Impact

* Cybersecurity risk assessment
* Financial loss forecasting
* Incident response prioritization
* Industry-specific cyber-risk profiling
* Data-driven security investment decisions

## 🌟 Highlights

* ✅ Full production-ready ML pipeline
* 📈 R² = 0.985, ROC–AUC = 0.995
* 🔍 Interpretable features improve insights
* ⚙️ High-variance models analyzed & mitigated
* 🔁 Fully reproducible and deployable

## 🛠 Tech Stack
* Python
* Pandas, NumPy
* Scikit-Learn
* LightGBM, XGBoost, CatBoost
* Matplotlib, Seaborn
* Jupyter Notebook

## 👤 Author
* Nithushan Uthayarasa

