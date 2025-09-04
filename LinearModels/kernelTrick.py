# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ===== 1. 生成非线性可分数据 =====
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler().fit(X_train)
X_train_std, X_test_std = scaler.transform(X_train), scaler.transform(X_test)

# ===== 2. 定义不同 SVM 模型 =====
models = {
    "Linear SVM": SVC(kernel="linear"),
    "Polynomial SVM": SVC(kernel="poly", degree=3, gamma="scale"),
    "RBF SVM": SVC(kernel="rbf", gamma="scale")
}

# ===== 3. 训练 & 评估 =====
results = {}
for name, clf in models.items():
    clf.fit(X_train_std, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_std))
    results[name] = acc
    print(f"{name}: Accuracy = {acc:.4f}")

# ===== 4. 可视化决策边界 =====
x_min, x_max = X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1
y_min, y_max = X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, clf) in zip(axes, models.items()):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X_train_std[:, 0], X_train_std[:, 1], c=y_train, edgecolor="k")
    ax.set_title(f"{name}\nAccuracy={results[name]:.2f}")

plt.tight_layout()
plt.show()
