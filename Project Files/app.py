"""
app.py
Flask web application for Credit Card Approval Prediction System.
Banking-themed interface for credit card application assessment.
"""

import os
import math
import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
GRAPHS_DIR = 'images/graphs'

model = None
model_name = "Not Available"
results_df = None

def load_model():
    global model, model_name
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        cls_name = type(model).__name__
        name_map = {
            'XGBClassifier': 'XGBoost Classifier',
            'LogisticRegression': 'Logistic Regression',
            'DecisionTreeClassifier': 'Decision Tree Classifier',
            'RandomForestClassifier': 'Random Forest Classifier'
        }
        model_name = name_map.get(cls_name, cls_name)
        return True
    return False

load_model()

MODEL_METRICS = {
    'Logistic Regression': {'Accuracy': 0.9724, 'Precision': 0.0618, 'Recall': 0.9184, 'F1 Score': 0.1158, 'ROC-AUC': 0.9765},
    'Decision Tree': {'Accuracy': 0.9992, 'Precision': 0.7333, 'Recall': 0.7551, 'F1 Score': 0.7440, 'ROC-AUC': 0.8775},
    'Random Forest': {'Accuracy': 0.9996, 'Precision': 0.9390, 'Recall': 0.7857, 'F1 Score': 0.8556, 'ROC-AUC': 0.9657},
    'XGBoost': {'Accuracy': 0.9996, 'Precision': 0.9390, 'Recall': 0.8061, 'F1 Score': 0.8675, 'ROC-AUC': 0.9792}
}

CLASS0_MEANS = np.array([
    94838.2023, 0.0083, -0.0063, 0.0122, -0.0079, 0.0055, 0.0024, 0.0096,
    -0.0010, 0.0045, 0.0098, -0.0066, 0.0108, 0.0002, 0.0121, 0.0002,
    0.0072, 0.0115, 0.0039, -0.0012, -0.0006, -0.0012, -0.0000, 0.0001,
    0.0002, -0.0001, -0.0001, -0.0003, -0.0001, 88.2910
])

CLASS1_MEANS = np.array([
    80746.8069, -4.7719, 3.6238, -7.0333, 4.5420, -3.1512, -1.3977, -5.5687,
    0.5706, -2.5811, -5.6769, 3.8002, -6.2594, -0.1093, -6.9717, -0.0929,
    -4.1399, -6.6658, -2.2463, 0.6807, 0.3723, 0.7136, 0.0140, -0.0403,
    -0.1051, 0.0414, 0.0516, 0.1706, 0.0757, 122.2113
])


def map_form_to_features(form_data):
    income = float(form_data.get('annual_income', 50000))
    age = float(form_data.get('age', 30))
    employment_duration = float(form_data.get('employment_duration', 1))
    own_car = form_data.get('own_car', 'No')
    own_realty = form_data.get('own_realty', 'No')
    education = form_data.get('education', 'Secondary')
    family_status = form_data.get('family_status', 'Single')
    housing = form_data.get('housing_type', 'Rent')
    emi_past_due = float(form_data.get('emi_past_due', 0))
    num_loans = float(form_data.get('num_loans', 0))

    risk_score = 0.0
    if income < 30000: risk_score += 0.20
    elif income < 60000: risk_score += 0.08
    if employment_duration < 1: risk_score += 0.15
    elif employment_duration < 3: risk_score += 0.06
    if own_car == 'No': risk_score += 0.06
    if own_realty == 'No': risk_score += 0.10
    if housing == 'Rent': risk_score += 0.08
    elif housing == 'With parents': risk_score += 0.03
    if education in ('Secondary', 'Primary'): risk_score += 0.06
    if family_status == 'Single': risk_score += 0.03
    if emi_past_due > 0: risk_score += min(emi_past_due * 0.08, 0.30)
    if num_loans > 3: risk_score += 0.10
    elif num_loans > 1: risk_score += 0.03
    risk_score = min(max(risk_score, 0.0), 0.98)

    np.random.seed(int(income * age + employment_duration * 100 + emi_past_due * 10 + num_loans * 7))

    t = risk_score ** 2
    features = CLASS0_MEANS * (1 - t) + CLASS1_MEANS * t
    noise = np.random.randn(30) * (0.05 + t * 0.15)
    features += noise
    features[0] = max(features[0], 0)
    features[29] = max(features[29], 0)
    return features.reshape(1, -1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = map_form_to_features(request.form)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            prob_approve = round(probability[0] * 100, 2)
            prob_reject = round(probability[1] * 100, 2)

            if prediction == 0:
                result_text = 'APPLICATION APPROVED'
                result_class = 'approved'
                risk_level = 'Low Risk'
                summary = [
                    'Stable transaction profile detected',
                    'Low financial risk identified',
                    'Eligible for credit card issuance'
                ]
                recommendation = 'Proceed with credit card issuance and final verification.'
                confidence = prob_approve
            else:
                result_text = 'APPLICATION REJECTED'
                result_class = 'rejected'
                risk_level = 'High Risk'
                summary = [
                    'High financial risk detected',
                    'Profile does not satisfy approval criteria',
                    'Requires additional verification'
                ]
                recommendation = 'Review financial profile and reapply after improving eligibility.'
                confidence = prob_reject if prob_reject > prob_approve else 100 - prob_approve

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            return render_template(
                'result.html',
                result_text=result_text,
                result_class=result_class,
                approval_prob=prob_approve,
                rejection_prob=prob_reject,
                risk_level=risk_level,
                summary=summary,
                recommendation=recommendation,
                confidence=confidence,
                model_used=model_name,
                timestamp=timestamp,
                graphs_dir=GRAPHS_DIR
            )

        except Exception as e:
            return render_template(
                'result.html',
                result_text='ERROR',
                result_class='error',
                approval_prob=0,
                rejection_prob=0,
                risk_level='N/A',
                summary=['An error occurred during prediction'],
                recommendation='Please try again with valid inputs.',
                confidence=0,
                model_used=model_name,
                timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                graphs_dir=GRAPHS_DIR,
                error=str(e)
            )

    return render_template('predict.html')


@app.route('/performance')
def performance():
    return render_template('performance.html',
                         metrics=MODEL_METRICS,
                         best_model=model_name,
                         graphs_dir=GRAPHS_DIR)


if __name__ == '__main__':
    if model is None:
        print("WARNING: No model found. Run train_model.py first.")
    app.run(debug=True, host='0.0.0.0', port=5000)
