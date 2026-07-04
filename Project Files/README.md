# Credit Card Approval Prediction System

> **An intelligent machine learning system** that predicts credit card application approval using multiple classification algorithms with a professional banking-style web interface.

---

## Overview

This project uses the Credit Card Fraud Detection dataset to train and evaluate multiple machine learning models for binary classification. The web application simulates a real-world credit card approval system with a modern, responsive banking interface.

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python, HTML, CSS, JavaScript |
| **ML/DL** | Scikit-learn, XGBoost, SMOTE |
| **Data** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Frontend** | Bootstrap 5, Font Awesome |
| **Model Serialization** | Joblib |
| **Notebooks** | Jupyter |

## Project Structure

```
CreditCardApprovalPrediction/
│
├── dataset/
│   └── creditcard.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── models/
│   └── best_model.pkl
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   ├── images/
│   │   └── graphs/
│   └── icons/
│
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── predict.html
│   ├── result.html
│   └── performance.html
│
├── reports/
│   ├── er_diagram.svg
│   ├── architecture_diagram.svg
│   └── project_report.md
│
├── app.py
├── train_model.py
├── run.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# 1. Clone or navigate to the project
cd CreditCardApprovalPrediction

# 2. Create virtual environment (recommended)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux:   source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    Place creditcard.csv in dataset/

# 5. Train the model
python train_model.py

# 6. Run the application
python run.py
```

Open your browser to: **http://127.0.0.1:5000**

## Features

- **Multiple ML Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **SMOTE Balancing:** Handles extreme class imbalance (0.17% fraud)
- **Auto Model Selection:** Best model based on F1 score
- **Banking Form UI:** 13 applicant fields with icons and validation
- **Result Dashboard:** Probability gauge, risk level, recommendations
- **Performance Page:** Comparison table, ROC curves, confusion matrices
- **REST API:** JSON prediction endpoint at `/api/predict`
- **Professional Theme:** Blue/white banking design, responsive

## Pages

| Route | Page |
|-------|------|
| `/` | Home - Hero section with stats |
| `/predict` | Application form |
| `/result` | Approval/rejection dashboard |
| `/about` | Project information |
| `/performance` | Model metrics and charts |

## Models Evaluated

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9724 | 0.0618 | 0.9184 | 0.1158 | 0.9765 |
| Decision Tree | 0.9992 | 0.7333 | 0.7551 | 0.7440 | 0.8775 |
| Random Forest | 0.9996 | 0.9390 | 0.7857 | 0.8556 | 0.9657 |
| **XGBoost** | **0.9996** | **0.9390** | **0.8061** | **0.8675** | **0.9792** |

*Best model auto-selected and saved during training.*

## Dataset

**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions
- 30 features (Time, Amount, V1-V28 PCA)
- Binary target (0 = Approved, 1 = Rejected)
- 99.83% approved, 0.17% rejected (handled via SMOTE)

---

*Built for SmartBridge SkillWallet academic submission. | 2026*
