"""Educational Gaussian naive-Bayes routines."""

from __future__ import annotations

import math

import numpy as np


class NaiveBayes:
    """Gaussian naive Bayes using the formulas from the original notebook."""

    def __init__(self) -> None:
        self.model: dict[object, list[tuple[float, float]]] | None = None

    @staticmethod
    def mean(X: np.ndarray) -> float:
        return float(sum(X) / len(X))

    def stdev(self, X: np.ndarray) -> float:
        avg = self.mean(X)
        return math.sqrt(sum((x - avg) ** 2 for x in X) / len(X))

    @staticmethod
    def gaussian_probability(x: float, mean: float, stdev: float) -> float:
        if stdev <= 0:
            raise ValueError("stdev must be positive")
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev**2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def summarize(self, train_data: np.ndarray) -> list[tuple[float, float]]:
        return [(self.mean(feature), self.stdev(feature)) for feature in zip(*train_data)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> str:
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if features.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("X and y must contain matching samples")

        data: dict[object, list[np.ndarray]] = {}
        for feature, label in zip(features, labels):
            data.setdefault(label, []).append(feature)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return "gaussianNB train done!"

    def calculate_probabilities(self, input_data: np.ndarray) -> dict[object, float]:
        if self.model is None:
            raise RuntimeError("fit the model before calculating probabilities")
        sample = np.asarray(input_data, dtype=float)
        if sample.ndim != 1:
            raise ValueError("input_data must be one feature vector")

        probabilities: dict[object, float] = {}
        for label, summaries in self.model.items():
            probability = 1.0
            for feature, (mean, stdev) in zip(sample, summaries):
                probability *= self.gaussian_probability(feature, mean, stdev)
            probabilities[label] = probability
        return probabilities

    def predict(self, X_test: np.ndarray) -> object:
        probabilities = self.calculate_probabilities(X_test)
        return max(probabilities, key=probabilities.get)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        features = np.asarray(X_test, dtype=float)
        labels = np.asarray(y_test)
        if len(features) != len(labels) or len(labels) == 0:
            raise ValueError("X_test and y_test must contain matching samples")
        predictions = [self.predict(feature) for feature in features]
        return float(np.mean(np.asarray(predictions) == labels))


__all__ = ["NaiveBayes"]
