"""A small educational perceptron trained with stochastic updates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Perceptron:
    """Binary perceptron using the notebook's stochastic-gradient update.

    Labels are expected to be -1 and 1.  When no initial weights are given,
    they are initialized to ones when ``fit`` first sees the feature matrix,
    matching the original notebook.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        initial_weights: Sequence[float] | None = None,
        initial_bias: float = 0.0,
    ) -> None:
        self.l_rate = learning_rate
        self.max_iter = max_iter
        self.w = (
            np.asarray(initial_weights, dtype=float).copy()
            if initial_weights is not None
            else None
        )
        self.b = float(initial_bias)

    def sign(
        self,
        x: np.ndarray,
        w: np.ndarray | None = None,
        b: float | None = None,
    ) -> float:
        """Return the linear score used by the original notebook."""

        weights = self.w if w is None else np.asarray(w, dtype=float)
        if weights is None:
            raise RuntimeError("fit the perceptron or provide weights first")
        bias = self.b if b is None else b
        return float(np.dot(x, weights) + bias)

    def decision_function(self, x: np.ndarray) -> float:
        return self.sign(x)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> str:
        features = np.asarray(X_train, dtype=float)
        labels = np.asarray(y_train)
        if features.ndim != 2:
            raise ValueError("X_train must be a two-dimensional array")
        if labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("X_train and y_train must contain matching samples")
        if not np.all(np.isin(labels, (-1, 1))):
            raise ValueError("y_train labels must be -1 or 1")

        if self.w is None:
            self.w = np.ones(features.shape[1], dtype=float)
        if self.w.shape != (features.shape[1],):
            raise ValueError("initial_weights must match the feature count")

        for _ in range(self.max_iter):
            wrong_count = 0
            for feature, label in zip(features, labels):
                if label * self.sign(feature) <= 0:
                    self.w = self.w + self.l_rate * label * feature
                    self.b = self.b + self.l_rate * label
                    wrong_count += 1
            if wrong_count == 0:
                return "Perceptron Model!"

        raise RuntimeError(
            f"perceptron did not converge within {self.max_iter} iterations"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return np.where(features @ self.w + self.b >= 0, 1, -1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        labels = np.asarray(y)
        if labels.size == 0:
            raise ValueError("y must contain at least one sample")
        return float(np.mean(self.predict(X) == labels))


# Keep the notebook's original class name available to readers who copied it.
Model = Perceptron

__all__ = ["Model", "Perceptron"]
