#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick MLP demo (classification + regression) with metrics table and visualizations.
Datasets: Breast Cancer (classification), Diabetes (regression)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.pipeline import make_pipeline
import os

def main():
    # Classification
    data_clf = load_breast_cancer()
    Xc, yc = data_clf.data, data_clf.target
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                                      learning_rate_init=1e-3, max_iter=400, random_state=42))
    clf.fit(Xc_train, yc_train)
    yc_pred = clf.predict(Xc_test)
    yc_proba = clf.predict_proba(Xc_test)[:, 1]
    acc = accuracy_score(yc_test, yc_pred)
    prec = precision_score(yc_test, yc_pred)
    rec = recall_score(yc_test, yc_pred)
    f1 = f1_score(yc_test, yc_pred)
    fpr, tpr, _ = roc_curve(yc_test, yc_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(yc_test, yc_pred)
    mlp_step = clf.named_steps['mlpclassifier']
    loss_curve_clf = getattr(mlp_step, "loss_curve_", [])

    # Regression
    data_reg = load_diabetes()
    Xr, yr = data_reg.data, data_reg.target
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    regr = make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                                      learning_rate_init=1e-3, max_iter=800, random_state=42))
    regr.fit(Xr_train, yr_train)
    yr_pred = regr.predict(Xr_test)
    mse = mean_squared_error(yr_test, yr_pred)
    mae = mean_absolute_error(yr_test, yr_pred)
    r2 = r2_score(yr_test, yr_pred)
    mlp_step_reg = regr.named_steps['mlpregressor']
    loss_curve_reg = getattr(mlp_step_reg, "loss_curve_", [])

    # Metrics
    rows = [
        {"Task": "Classification (Breast Cancer)", "Metric": "Accuracy", "Value": acc},
        {"Task": "Classification (Breast Cancer)", "Metric": "Precision", "Value": prec},
        {"Task": "Classification (Breast Cancer)", "Metric": "Recall", "Value": rec},
        {"Task": "Classification (Breast Cancer)", "Metric": "F1-score", "Value": f1},
        {"Task": "Classification (Breast Cancer)", "Metric": "ROC AUC", "Value": roc_auc},
        {"Task": "Regression (Diabetes)", "Metric": "MSE", "Value": mse},
        {"Task": "Regression (Diabetes)", "Metric": "MAE", "Value": mae},
        {"Task": "Regression (Diabetes)", "Metric": "R2", "Value": r2},
    ]
    df = pd.DataFrame(rows)
    print("\n==== Metrics Summary ====")
    print(df.to_string(index=False))

    # Output dir
    out_dir = "mlp_demo_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Figures
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - MLP Classifier (Breast Cancer)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)

    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix - MLP Classifier")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, data_clf.target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, data_clf.target_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)

    if len(loss_curve_clf) > 0:
        plt.figure(figsize=(5,4))
        plt.plot(loss_curve_clf)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Loss Curve - MLP Classifier")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve_classifier.png"), dpi=150)

    if len(loss_curve_reg) > 0:
        plt.figure(figsize=(5,4))
        plt.plot(loss_curve_reg)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Loss Curve - MLP Regressor")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_curve_regressor.png"), dpi=150)

    plt.figure(figsize=(5,4))
    plt.scatter(yr_test, yr_pred, alpha=0.7)
    mins = min(yr_test.min(), yr_pred.min())
    maxs = max(yr_test.max(), yr_pred.max())
    plt.plot([mins, maxs], [mins, maxs], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Regression: True vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reg_true_vs_pred.png"), dpi=150)

if __name__ == "__main__":
    main()
