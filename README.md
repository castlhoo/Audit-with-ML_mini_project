# 🧠 Audit Risk Classification Project

## 📂 Dataset Overview

This dataset contains audit-related data from public sector institutions. The task is to classify the **"Risk"** column (0 = Low Risk, 1 = High Risk) based on various financial, procedural, and historical indicators.

### Feature Description
| Column | Description |
|--------|-------------|
| Sector_score | Risk score of the institution based on past audit results |
| PARA_A | Errors in planned expenditure (Crores) |
| PARA_B | Errors in unplanned expenditure (Crores) |
| TOTAL | Total amount of detected errors |
| numbers | Count of past audit-related incidents |
| Marks | Score for risk category |
| Money_Value | Total money involved in prior audit problems |
| MONEY_Marks | Risk score for money issues |
| LOSS_SCORE | Risk score for losses |
| History_score | Historical risk score |
| Risk | ✅ Target variable (0 = Low Risk, 1 = High Risk) |

We started by understanding the dataset and noting important columns that could leak information or dominate predictions, such as `Score`, `SCORE_A`, and `SCORE_B`. These were later removed.


---

## 🔍 Project Overview
This project aims to predict the **risk level (high or low)** of audit reports using various financial and procedural features from an audit dataset. The goal is to identify key features contributing to audit risks and to evaluate the performance of different machine learning models in a classification task.

## 📌 Why This Project?
- **Real-World Use Case**: Financial institutions and auditors need tools to **prioritize risky audits** efficiently.
- **Class Imbalance Challenge**: Audit risk labels (0 = Low Risk, 1 = High Risk) are **imbalanced**, requiring special handling like SMOTE.
- **Model Interpretability vs. Accuracy**: A balance between a model's performance and its explainability is crucial for real-world deployment.
- **Motivation**: Initial Random Forest model showed perfect accuracy (100%) which hinted at data leakage or feature dominance, prompting in-depth analysis and refinement.

---

## 🔍 Step-by-Step Process

### ✅ Step 1: Data Loading & Initial Cleanup
```python
import pandas as pd
import numpy as np

audit_df1 = pd.read_csv("./audit_train.csv")

# Drop features that leak label info or are too dominant
columns_to_drop = ["LOCATION_ID", "Score", "SCORE_A", "SCORE_B", "District", "Loss", "History"]
audit_df1.drop(columns=columns_to_drop, inplace=True)
```

### ✅ Step 2: Handle Missing & Duplicate Data
```python
# Fill nulls
audit_df1["Money_Value"].fillna(audit_df1["Money_Value"].mean(), inplace=True)

# Check duplicated rows
audit_df1[audit_df1.duplicated()].count()
```
- One null in `Money_Value` was imputed with the mean.
- Duplicate rows were retained as their influence seemed tolerable.

### ✅ Step 3: Skewness & Outlier Detection
```python
import matplotlib.pyplot as plt
for col in audit_df1.columns:
    plt.hist(audit_df1[col])
    plt.title(col)
    plt.show()
```
```python
for col in audit_df1.select_dtypes(include=['int64', 'float64']).columns:
    plt.boxplot(audit_df1[col].dropna())
    plt.title(col)
    plt.show()
```
- Many features showed skewed distributions. A log1p transformation was applied later.

---

## 🗓️ Data Preprocessing Pipeline
```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

X_features = audit_df1.iloc[:, :-1]
y_labels = audit_df1.iloc[:, -1]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.3, stratify=y_labels, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Log Transformation
for col in X_train.columns:
    if (X_train[col] >= 0).all():
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])
        X_val[col] = np.log1p(X_val[col])

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# SMOTE Oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
```

---

## 🤖 Model Experiments

### 1️⃣ Random Forest (Initial Benchmark)
```python
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
rf_clf.fit(X_train_resampled, y_train_resampled)
pred = rf_clf.predict(X_test_scaled)
```
- **Initial accuracy was 100%**, indicating overfitting.
- **🔎 Feature Importance (Random Forest)**
| Feature         | Importance |
|----------------|------------|
| PARA_A         | 38.8%      |
| TOTAL          | 25.2%      |
| PARA_B         | 15.4%      |
| Money_Value    | 7.6%       |

Interpretation: **Planned errors (PARA_A)** and **total discovered errors** are strong indicators of audit risk.

- Feature importance showed Score-related features were too dominant (>30%), so we dropped them.
- After fixing, accuracy: **92.7%**
- 

### 2️⃣ XGBoost
```python
from xgboost import XGBClassifier
xgb_wrapper = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, eval_metric="logloss")
xgb_wrapper.fit(X_train_resampled, y_train_resampled)
```
- Accuracy: **92.3%**, ROC-AUC: **0.957**
- Boosting method, well-suited for tabular data

### 3️⃣ LightGBM + HyperOpt Tuning
```python
from hyperopt import hp, fmin, tpe, Trials
from lightgbm import LGBMClassifier
# Hyperopt tuning process here...
final_lgbm = LGBMClassifier(...)  # Best parameters
final_lgbm.fit(X_train_resampled, y_train_resampled)
```
- Tuned using KFold and `roc_auc`
- Accuracy: **90.5%**, ROC-AUC: **0.9548**
- Lightweight, fast training

### 4️⃣ Logistic Regression + GridSearchCV
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

params = {'solver': ['liblinear', 'lbfgs'], 'penalty': ['l2', 'l1'], 'C': [0.01, 0.1, 1, 5, 10]}
grid_clf = GridSearchCV(LogisticRegression(), param_grid=params, scoring='accuracy', cv=3)
grid_clf.fit(X_train_resampled, y_train_resampled)
```
- Best parameters: `C=1, penalty='l1', solver='liblinear'`
- Accuracy: **93.6%**, ROC-AUC: **0.964**
- Despite simplicity, performed best due to clean data and class balance

---

## 🔢 Evaluation Metrics Summary
| Metric | Meaning |
|--------|--------|
| Accuracy | Overall correctness of model |
| Precision | Fraction of predicted positives that are true |
| Recall | Fraction of actual positives correctly predicted |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Model's ability to distinguish between classes |

---

### 🔬 Why These Models?
- **Random Forest**: Easy to use, baseline benchmark, interpretable
- **XGBoost**: Known for high performance in competitions, great with structured data
- **LightGBM**: Lightweight boosting with efficient memory use
- **Logistic Regression**: Strong baseline, especially after handling skew, scale, and balance

---

## ✅ Lessons Learned
- Always avoid **data leakage** by separating test data and using only training data for fit/transform.
- **SMOTE** helps with imbalanced datasets but should not be used on test data.
- Even a simple model like **Logistic Regression** can perform well with good preprocessing and tuning.
- Feature importance analysis helps detect potential data leakage and **dominant features**.

---

## ✅ Conclusion
We successfully built multiple models to classify audit risks with high performance. Among them, **Logistic Regression** with hyperparameter tuning delivered the best **balance of performance and interpretability**.

This project is suitable for:
- **Auditing system automation**
- **Education in tabular modeling**
- **Portfolio demonstrating preprocessing and modeling pipelines**

---
