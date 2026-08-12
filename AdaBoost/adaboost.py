"""Educational AdaBoost with one-dimensional threshold weak learners."""

from __future__ import annotations

import numpy as np


class AdaBoost:
    """AdaBoost using the threshold stumps from the original notebook.

    Labels are expected to be ``-1`` and ``1``.  Thresholds are evaluated at
    midpoints between sorted unique feature values, which is the same stump
    idea as the notebook. The original ``learning_rate`` threshold grid is
    retained, with midpoint candidates added so a coarse step cannot miss a
    valid split between two observed values.
    """

    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.clf_num = n_estimators
        self.learning_rate = learning_rate
        self.clf_sets: list[tuple[int, float, str]] = []
        self.alpha: list[float] = []
        self.weights: np.ndarray | None = None
        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None

    def init_args(self, datasets: np.ndarray, labels: np.ndarray) -> None:
        features = np.asarray(datasets, dtype=float)
        targets = np.asarray(labels)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("datasets must be a non-empty two-dimensional array")
        if targets.ndim != 1 or len(features) != len(targets):
            raise ValueError("datasets and labels must contain matching samples")
        if not np.all(np.isin(targets, (-1, 1))):
            raise ValueError("labels must be -1 or 1")

        self.X = features
        self.Y = targets.astype(float)
        self.M, self.N = features.shape
        self.clf_sets = []
        self.weights = np.full(self.M, 1.0 / self.M, dtype=float)
        self.alpha = []

    @staticmethod
    def _stump_predict(features: np.ndarray, threshold: float, direct: str) -> np.ndarray:
        if direct == "positive":
            return np.where(features > threshold, 1, -1)
        if direct in {"negative", "nagetive"}:
            return np.where(features > threshold, -1, 1)
        raise ValueError(f"unknown stump direction: {direct!r}")

    def _G(
        self, features: np.ndarray, labels: np.ndarray, weights: np.ndarray
    ) -> tuple[float, str, float, np.ndarray]:
        """Find the weighted-error threshold stump for one feature column."""

        values = np.asarray(features, dtype=float)
        targets = np.asarray(labels)
        sample_weights = np.asarray(weights, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("features must be a non-empty one-dimensional array")
        if len(values) != len(targets) or len(values) != len(sample_weights):
            raise ValueError("features, labels, and weights must have equal length")

        unique_values = np.unique(values)
        midpoint_thresholds = [
            float((left + right) / 2)
            for left, right in zip(unique_values[:-1], unique_values[1:])
        ]
        grid_stop = int((unique_values[-1] - unique_values[0] + self.learning_rate)
                        // self.learning_rate)
        grid_thresholds = [
            float(unique_values[0] + self.learning_rate * step)
            for step in range(1, grid_stop)
            if unique_values[0] + self.learning_rate * step not in unique_values
        ]
        thresholds = sorted(set(midpoint_thresholds + grid_thresholds))
        # The two infinite thresholds represent constant -1 and +1 stumps.
        candidates = [
            (threshold, direct)
            for threshold in thresholds
            for direct in ("positive", "nagetive")
        ]
        candidates.extend(((float("inf"), "positive"), (float("inf"), "nagetive")))

        best: tuple[float, str, float, np.ndarray] | None = None
        for threshold, direct in candidates:
            prediction = self._stump_predict(values, threshold, direct)
            error = float(np.sum(sample_weights[prediction != targets]))
            if best is None or error < best[2]:
                best = (threshold, direct, error, prediction)

        assert best is not None
        return best

    @staticmethod
    def _alpha(error: float) -> float:
        """Return a finite stump coefficient, including for zero error."""

        if not 0 <= error <= 1:
            raise ValueError("error must be between 0 and 1")
        clipped_error = np.clip(error, 1e-12, 1 - 1e-12)
        return float(0.5 * np.log((1 - clipped_error) / clipped_error))

    def _Z(self, weights: np.ndarray, alpha: float, clf: np.ndarray) -> float:
        if self.Y is None:
            raise RuntimeError("fit has not initialized the labels")
        return float(np.sum(weights * np.exp(-alpha * self.Y * clf)))

    def _w(self, alpha: float, clf: np.ndarray, Z: float) -> None:
        if self.weights is None:
            raise RuntimeError("fit has not initialized the weights")
        if Z <= 0 or not np.isfinite(Z):
            raise RuntimeError("invalid AdaBoost normalization factor")
        self.weights = self.weights * np.exp(-alpha * self.Y * clf) / Z

    def G(self, x: float, v: float, direct: str) -> int:
        """Apply one learned threshold stump to one feature value."""

        return int(self._stump_predict(np.asarray([x]), v, direct)[0])

    def _f(self, feature: np.ndarray) -> float:
        """Return the weighted sum of learned stump predictions."""

        if not self.clf_sets:
            raise RuntimeError("fit the classifier before predicting")
        values = np.asarray(feature, dtype=float)
        return float(
            sum(
                alpha * self.G(values[axis], threshold, direct)
                for alpha, (axis, threshold, direct) in zip(self.alpha, self.clf_sets)
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.init_args(X, y)
        assert self.X is not None and self.Y is not None and self.weights is not None

        for _ in range(self.clf_num):
            best_clf_error = float("inf")
            best_v: float | None = None
            best_direct: str | None = None
            best_result: np.ndarray | None = None
            best_axis: int | None = None

            for axis in range(self.N):
                stump = self._G(self.X[:, axis], self.Y, self.weights)
                v, direct, error, result = stump
                if error < best_clf_error:
                    best_v = v
                    best_direct = direct
                    best_clf_error = error
                    best_result = result
                    best_axis = axis

            assert best_v is not None
            assert best_direct is not None
            assert best_result is not None
            assert best_axis is not None

            alpha = self._alpha(best_clf_error)
            self.alpha.append(alpha)
            self.clf_sets.append((best_axis, best_v, best_direct))
            normalization = self._Z(self.weights, alpha, best_result)
            self._w(alpha, best_result, normalization)

            if best_clf_error <= 1e-12:
                break

    def predict(self, feature: np.ndarray) -> int:
        return 1 if self._f(feature) > 0 else -1

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        features = np.asarray(X_test, dtype=float)
        labels = np.asarray(y_test)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError("X_test must be a non-empty two-dimensional array")
        if labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("X_test and y_test must contain matching samples")
        predictions = np.asarray([self.predict(feature) for feature in features])
        return float(np.mean(predictions == labels))


__all__ = ["AdaBoost"]
