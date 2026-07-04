"""
train_model.py
Complete ML pipeline for Credit Card Approval Prediction System.
Trains multiple classifiers, evaluates, generates graphs, saves best model.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'creditcard.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
GRAPHS_DIR = os.path.join(BASE_DIR, 'static', 'images', 'graphs')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)


def load_data():
    print("=" * 60)
    print("CREDIT CARD APPROVAL PREDICTION - MODEL TRAINING")
    print("=" * 60)
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def explore_and_visualize(df):
    print("\n--- Data Exploration & Visualization ---")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")

    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle('Credit Card Data Analysis', fontsize=16, fontweight='bold')

    sns.countplot(x='Class', data=df, ax=axes[0,0], palette=['#1565C0', '#e53935'])
    axes[0,0].set_title('Class Distribution')
    axes[0,0].set_xticklabels(['Approved (0)', 'Rejected (1)'])

    axes[0,1].hist([df[df['Class']==0]['Amount'], df[df['Class']==1]['Amount']],
                    bins=50, label=['Approved', 'Rejected'],
                    alpha=0.7, color=['#1565C0', '#e53935'])
    axes[0,1].set_title('Amount Distribution by Class')
    axes[0,1].set_xlabel('Amount')
    axes[0,1].legend()

    corr_full = df.corr()
    corr = corr_full.drop('Class', axis=1).drop('Class', axis=0)
    sns.heatmap(corr, ax=axes[0,2], cmap='RdBu', center=0, cbar_kws={'shrink': 0.8})
    axes[0,2].set_title('Correlation Heatmap')

    class_counts = df['Class'].value_counts()
    axes[1,0].pie(class_counts, labels=['Approved', 'Rejected'],
                  autopct='%1.2f%%', startangle=90,
                  colors=['#1565C0', '#e53935'], explode=(0, 0.05))
    axes[1,0].set_title('Class Imbalance')

    axes[1,1].boxplot([df[df['Class']==0]['Amount'], df[df['Class']==1]['Amount']],
                       labels=['Approved', 'Rejected'], patch_artist=True,
                       boxprops=dict(facecolor='#1565C0'))
    axes[1,1].set_title('Amount Boxplot')

    for col in ['V1','V2','V3','V4','V5']:
        sns.kdeplot(df[df['Class']==0][col], ax=axes[1,2], color='#1565C0', alpha=0.4)
        sns.kdeplot(df[df['Class']==1][col], ax=axes[1,2], color='#e53935', alpha=0.4)
    axes[1,2].set_title('Feature Distributions (V1-V5)')

    plt.tight_layout()
    fig1.savefig(os.path.join(GRAPHS_DIR, 'eda_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: eda_analysis.png")

    fig2, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap='RdBu', center=0)
    ax.set_title('Feature Correlation Heatmap')
    fig2.savefig(os.path.join(GRAPHS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: correlation_heatmap.png")

    fig3, ax = plt.subplots(figsize=(10, 6))
    corr_with_target = corr_full['Class'].drop('Class')
    top_corr = corr_with_target.sort_values(key=abs, ascending=False).head(10)
    sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax, palette='RdBu_r')
    ax.set_title('Top 10 Features Correlated with Class')
    ax.set_xlabel('Correlation')
    fig3.savefig(os.path.join(GRAPHS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("Saved: feature_importance.png")


def preprocess_data(df):
    print("\n--- Preprocessing ---")
    initial = df.shape[0]
    df = df.drop_duplicates()
    print(f"Removed {initial - df.shape[0]} duplicates")

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])
    print("Scaled Amount and Time")

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_train_res.shape}")
    print(f"Class balance: {pd.Series(y_train_res).value_counts().to_dict()}")

    return X_train_res, X_test, y_train_res, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n--- Model Training & Evaluation ---")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, random_state=42, eval_metric='logloss',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            use_label_encoder=False
        )
    }

    results = []
    best_model = None
    best_f1 = 0
    best_name = ""
    cm_dict = {}
    roc_curves = []

    fig_cm, axes_cm = plt.subplots(2, 2, figsize=(12, 10))
    fig_cm.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    axes_cm = axes_cm.flatten()

    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))

    for idx, (name, model) in enumerate(models.items()):
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'ROC-AUC': round(roc_auc, 4)
        })

        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes_cm[idx], cmap='Blues', colorbar=False)
        axes_cm[idx].set_title(name)

        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc, name=name)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    fig_cm.savefig(os.path.join(GRAPHS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_cm)
    print("\nSaved: confusion_matrices.png")

    ax_roc.plot([0,1],[0,1], 'k--', label='Random')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.legend(loc='lower right')
    fig_roc.savefig(os.path.join(GRAPHS_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_roc)
    print("Saved: roc_curves.png")

    results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\nBest Model: {best_name} (F1: {best_f1:.4f})")

    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Model saved: {model_path}")

    fig_bar = plt.figure(figsize=(12, 6))
    results_df.set_index('Model').plot(kind='bar', ax=plt.gca(), colormap='viridis')
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=15)
    plt.legend(loc='lower right')
    plt.tight_layout()
    fig_bar.savefig(os.path.join(GRAPHS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_bar)
    print("Saved: model_comparison.png")

    results_df.to_csv(os.path.join(BASE_DIR, 'reports', 'model_comparison.csv'), index=False)
    return best_model, best_name, results_df


def main():
    df = load_data()
    explore_and_visualize(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    best_model, best_name, results_df = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"Best Model: {best_name}")
    print(f"Saved: models/best_model.pkl")
    print(f"Graphs: static/images/graphs/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
