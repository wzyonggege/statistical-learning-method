"""A small educational SMO-style support-vector machine implementation.

The implementation follows the derivation and update steps in the original
support-vector-machine notebook.  It is intentionally kept separate from
scikit-learn's SVC so that the notebook continues to demonstrate the
hand-written algorithm.
"""

from __future__ import annotations

import numpy as np


class SVM:
    def __init__(self, max_iter: int = 100, kernel: str = "linear"):
        self.max_iter = max_iter
        self._kernel = kernel

    def init_args(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0

        # 将 Ei 保存在一个列表里
        self.alpha = np.ones(self.m)
        self.E = [self._E(i) for i in range(self.m)]
        # 松弛变量
        self.C = 1.0

    def _KKT(self, i: int) -> bool:
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # g(x) 预测值，输入 xi（X[i]）
    def _g(self, i: int) -> float:
        result = self.b
        for j in range(self.m):
            result += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])
        return result

    # 核函数
    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        inner_product = float(np.dot(x1, x2))
        if self._kernel == "linear":
            return inner_product
        if self._kernel == "poly":
            return (inner_product + 1) ** 2
        raise ValueError(f"unsupported kernel: {self._kernel}")

    # E（x）为 g(x) 对输入 x 的预测值和 y 的差
    def _E(self, i: int) -> float:
        return self._g(i) - self.Y[i]

    def _init_alpha(self) -> tuple[int, int] | None:
        # 外层循环首先遍历所有满足 0<a<C 的样本点，检验是否满足 KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self._KKT(i):
                continue

            E1 = self.E[i]
            # 如果 E2 是正，选择最小的；如果 E2 是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j
        return None

    @staticmethod
    def _compare(alpha: float, lower: float, upper: float) -> float:
        if alpha > upper:
            return upper
        if alpha < lower:
            return lower
        return alpha

    def fit(self, features: np.ndarray, labels: np.ndarray) -> str:
        self.init_args(features, labels)

        for _ in range(self.max_iter):
            selected = self._init_alpha()
            if selected is None:
                break
            i1, i2 = selected

            # 边界
            if self.Y[i1] == self.Y[i2]:
                lower = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                upper = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                lower = max(0, self.alpha[i2] - self.alpha[i1])
                upper = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = (
                self.kernel(self.X[i1], self.X[i1])
                + self.kernel(self.X[i2], self.X[i2])
                - 2 * self.kernel(self.X[i1], self.X[i2])
            )
            if eta <= 0:
                continue

            alpha2_unc = self.alpha[i2] + self.Y[i2] * (E2 - E1) / eta
            alpha2_new = self._compare(alpha2_unc, lower, upper)
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (
                self.alpha[i2] - alpha2_new
            )

            b1_new = (
                -E1
                - self.Y[i1]
                * self.kernel(self.X[i1], self.X[i1])
                * (alpha1_new - self.alpha[i1])
                - self.Y[i2]
                * self.kernel(self.X[i2], self.X[i1])
                * (alpha2_new - self.alpha[i2])
                + self.b
            )
            b2_new = (
                -E2
                - self.Y[i1]
                * self.kernel(self.X[i1], self.X[i2])
                * (alpha1_new - self.alpha[i1])
                - self.Y[i2]
                * self.kernel(self.X[i2], self.X[i2])
                * (alpha2_new - self.alpha[i2])
                + self.b
            )

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)

        return "train done!"

    def predict(self, data: np.ndarray) -> int:
        result = self.b
        for i in range(self.m):
            result += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if result > 0 else -1

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        right_count = 0
        for i in range(len(X_test)):
            if self.predict(X_test[i]) == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self) -> np.ndarray:
        # linear model
        yx = self.Y.reshape(-1, 1) * self.X
        self.w = np.dot(yx.T, self.alpha)
        return self.w
