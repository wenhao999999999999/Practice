# -*- coding: utf-8 -*-
# 线性模型可视化对比（分类与回归）

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from pathlib import Path

from sklearn.datasets import (
    make_classification, make_moons, load_iris, load_diabetes, make_regression
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from sklearn.linear_model import (
    LogisticRegression, Perceptron, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ===== 统一保存目录 =====
OUTDIR = Path("LineModel/fig")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ====== 同图对比多个模型边界（支持多分类）======
def plot_decision_boundary(models, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig = plt.figure()

    # 背景填充：第一个模型
    Z_bg = models[0].predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z_bg, alpha=0.20)

    # 数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k')

    # 叠加每个模型的轮廓线（不同颜色）
    legend_handles = []
    colors = cm.get_cmap("tab10", len(models))
    for idx, m in enumerate(models):
        name = type(m).__name__
        Z = m.predict(grid).reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=np.unique(Z),
                    alpha=0.9, linewidths=1.5, colors=[colors(idx)])
        legend_handles.append(Line2D([0], [0], color=colors(idx), lw=2, label=name))

    plt.title(title)
    if legend_handles:
        plt.legend(handles=legend_handles, loc='upper right')

    out = OUTDIR / filename
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(out)


def classification_demo():
    results = []

    # 1) 线性可分数据
    X, y = make_classification(
        n_samples=800, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=2.0, random_state=42
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_tes = scaler.transform(X_tr), scaler.transform(X_te)

    clf_list = [
        LogisticRegression(max_iter=1000),
        LinearSVC(),
        Perceptron(max_iter=1000),
        LinearDiscriminantAnalysis()
    ]
    for clf in clf_list:
        clf.fit(X_trs, y_tr)
    accs = [(type(clf).__name__, accuracy_score(y_te, clf.predict(X_tes))) for clf in clf_list]
    results.append(("Linearly separable 2D", accs))
    plot_decision_boundary(clf_list, X_trs, y_tr,
                           "Linearly Separable (Train, 2D standardized)",
                           "cls_linear_sep.png")

    # 2) 非线性 moons 数据
    X, y = make_moons(n_samples=800, noise=0.2, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_tes = scaler.transform(X_tr), scaler.transform(X_te)

    clf_list2 = [
        LogisticRegression(max_iter=1000),
        LinearSVC(),
        Perceptron(max_iter=1000),
        LinearDiscriminantAnalysis()
    ]
    for clf in clf_list2:
        clf.fit(X_trs, y_tr)
    accs2 = [(type(clf).__name__, accuracy_score(y_te, clf.predict(X_tes))) for clf in clf_list2]
    results.append(("Nonlinear (Moons) 2D", accs2))
    plot_decision_boundary(clf_list2, X_trs, y_tr,
                           "Nonlinear (Moons) (Train, 2D standardized)",
                           "cls_moons.png")

    # 3) Iris（3类，多分类对比）
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                              random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_tes = scaler.transform(X_tr), scaler.transform(X_te)

    clf_softmax = LogisticRegression(max_iter=2000)
    clf_svm = LinearSVC()
    clf_lda = LinearDiscriminantAnalysis()
    for clf in [clf_softmax, clf_svm, clf_lda]:
        clf.fit(X_trs, y_tr)
    accs3 = [(type(clf).__name__,
              accuracy_score(y_te, clf.predict(X_tes)))
             for clf in [clf_softmax, clf_svm, clf_lda]]
    results.append(("Iris (3-class, full-dim)", accs3))

    # PCA -> 2D 用于可视化
    pca = PCA(n_components=2).fit(X_trs)
    X_trs_2d = pca.transform(X_trs)
    vis_models = [
        LogisticRegression(max_iter=2000),
        LinearSVC(),
        LinearDiscriminantAnalysis()
    ]
    for m in vis_models:
        m.fit(X_trs_2d, y_tr)
    plot_decision_boundary(vis_models, X_trs_2d, y_tr,
                           "Iris (Train PCA 2D) Linear Boundaries - Multi-model",
                           "cls_iris_pca2d_multimodel.png")

    return results


def plot_regression_fit(models, X, y, title, filename):
    xs = np.linspace(X.min()-1.0, X.max()+1.0, 400).reshape(-1, 1)
    fig = plt.figure()
    plt.scatter(X, y, s=20, alpha=0.7)
    for m in models:
        yhat = m.predict(xs)
        plt.plot(xs, yhat, linewidth=2, label=type(m).__name__)
    plt.title(title)
    plt.legend()
    out = OUTDIR / filename
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(out)


def regression_demo():
    results = []

    # 1) 1D 合成回归
    X, y = make_regression(n_samples=400, n_features=1, noise=15.0, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_tes = scaler.transform(X_tr), scaler.transform(X_te)

    regs = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.05, max_iter=10000),
        ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
    ]
    for r in regs:
        r.fit(X_trs, y_tr)
    metrics = []
    for r in regs:
        y_pred = r.predict(X_tes)
        metrics.append((type(r).__name__,
                        mean_squared_error(y_te, y_pred),
                        r2_score(y_te, y_pred)))
    results.append(("1D Synthetic", metrics))
    plot_regression_fit(regs, X_trs, y_tr,
                        "1D Regression (Train, standardized)",
                        "reg_1d.png")

    # 2) Diabetes 数据
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_trs, X_tes = scaler.transform(X_tr), scaler.transform(X_te)

    regs2 = [
        LinearRegression(),
        Ridge(alpha=1.0),
        Lasso(alpha=0.01, max_iter=10000),
        ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    ]
    for r in regs2:
        r.fit(X_trs, y_tr)
    metrics2 = []
    for r in regs2:
        y_pred = r.predict(X_tes)
        metrics2.append((type(r).__name__,
                         mean_squared_error(y_te, y_pred),
                         r2_score(y_te, y_pred)))
    results.append(("Diabetes (full-dim)", metrics2))

    # PCA -> 1D 用于画图（仅直觉展示）
    pca = PCA(n_components=1).fit(X_trs)
    X_trs_1d = pca.transform(X_trs)
    regs_vis = [LinearRegression(), Ridge(alpha=1.0),
                Lasso(alpha=0.01, max_iter=10000),
                ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)]
    for r in regs_vis:
        r.fit(X_trs_1d, y_tr)
    plot_regression_fit(regs_vis, X_trs_1d, y_tr,
                        "Diabetes (Train PCA 1D) Linear Fits (for intuition)",
                        "reg_diabetes_pca1d.png")

    return results


# ===== 运行并打印指标 / 保存图路径 =====
cls_results = classification_demo()
reg_results = regression_demo()

print("\n=== Classification Metrics (Accuracy) ===")
for name, accs in cls_results:
    print(f"\n[{name}]")
    for m, acc in accs:
        print(f"{m:>28s}: {acc:.4f}")

print("\n=== Regression Metrics (MSE, R2) ===")
for name, mets in reg_results:
    print(f"\n[{name}]")
    for m, mse, r2 in mets:
        print(f"{m:>28s}: MSE={mse:.2f}, R2={r2:.3f}")

print("\nSaved figures:")
for f in ["cls_linear_sep.png", "cls_moons.png",
          "cls_iris_pca2d_multimodel.png",
          "reg_1d.png", "reg_diabetes_pca1d.png"]:
    print(str(OUTDIR / f))
