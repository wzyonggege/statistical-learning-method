"""Educational distance and k-nearest-neighbors routines."""

from __future__ import annotations

from collections import Counter

import numpy as np


def L(x: np.ndarray, y: np.ndarray, p: float = 2) -> float:
    """Return the p-norm distance used in the original notebook."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape")
    if p <= 0 and p != np.inf:
        raise ValueError("p must be positive or numpy.inf")

    difference = np.abs(x_array - y_array)
    if p == np.inf:
        return float(np.max(difference))
    return float(np.sum(difference**p) ** (1 / p))


class KNN:
    """A brute-force k-nearest-neighbors classifier."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_neighbors: int = 3,
        p: float = 2,
    ) -> None:
        features = np.asarray(X_train, dtype=float)
        labels = np.asarray(y_train)
        if features.ndim != 2:
            raise ValueError("X_train must be a two-dimensional array")
        if labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("X_train and y_train must contain matching samples")
        if not 1 <= n_neighbors <= len(features):
            raise ValueError("n_neighbors must be between 1 and the sample count")

        self.n = n_neighbors
        self.p = p
        self.X_train = features
        self.y_train = labels

    def predict(self, X: np.ndarray) -> object:
        point = np.asarray(X, dtype=float)
        if point.shape != (self.X_train.shape[1],):
            raise ValueError("X must be one feature vector")

        distances = [
            (L(point, feature, p=self.p), label)
            for feature, label in zip(self.X_train, self.y_train)
        ]
        nearest = sorted(distances, key=lambda item: item[0])[: self.n]
        counts = Counter(label for _, label in nearest)
        # Counter preserves the nearest-neighbour order on ties, making the
        # result deterministic without preferring the numerically largest label.
        return max(counts, key=counts.get)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        features = np.asarray(X_test, dtype=float)
        labels = np.asarray(y_test)
        if len(features) != len(labels) or len(labels) == 0:
            raise ValueError("X_test and y_test must contain matching samples")
        predictions = [self.predict(point) for point in features]
        return float(np.mean(np.asarray(predictions) == labels))


__all__ = ["KNN", "L"]
