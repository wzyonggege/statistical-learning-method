"""Educational logistic regression trained with stochastic gradient updates."""

from __future__ import annotations

import numpy as np


class LogisticRegressionClassifier:
    """Binary logistic regression following the original notebook's update."""

    def __init__(self, max_iter: int = 200, learning_rate: float = 0.01) -> None:
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights: np.ndarray | None = None

    @staticmethod
    def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
        """Return the sigmoid element-wise for scalars and NumPy arrays."""

        values = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def data_matrix(X: np.ndarray) -> np.ndarray:
        """Add the intercept column used by the notebook's derivation."""

        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        return np.column_stack((np.ones(len(features)), features))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit with the original sample-by-sample gradient update."""

        features = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=float)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("X and y must contain matching samples")
        if not np.all(np.isin(labels, (0.0, 1.0))):
            raise ValueError("y labels must be 0 or 1")

        data_mat = self.data_matrix(features)
        self.weights = np.zeros((data_mat.shape[1], 1), dtype=float)

        for _ in range(self.max_iter):
            for row, label in zip(data_mat, labels):
                result = self.sigmoid(np.dot(row, self.weights)).item()
                error = label - result
                self.weights += self.learning_rate * error * row[:, None]

        print(
            "LogisticRegression Model(learning_rate={},max_iter={})".format(
                self.learning_rate, self.max_iter
            )
        )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the linear score before applying the sigmoid."""

        if self.weights is None:
            raise RuntimeError("fit the classifier before predicting")
        features = np.asarray(X, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return (self.data_matrix(features) @ self.weights).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using a zero linear-score threshold."""

        return (self.decision_function(X) >= 0).astype(int)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        labels = np.asarray(y_test)
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("y_test must be a non-empty one-dimensional array")
        predictions = self.predict(X_test)
        if len(predictions) != len(labels):
            raise ValueError("X_test and y_test must contain matching samples")
        return float(np.mean(predictions == labels))


# Keep the notebook's original misspelling available to existing readers.
LogisticReressionClassifier = LogisticRegressionClassifier

__all__ = ["LogisticRegressionClassifier", "LogisticReressionClassifier"]
