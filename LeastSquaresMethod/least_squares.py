"""Small, testable helpers used by the least-squares notebook.

The notebook uses SciPy's nonlinear least-squares solver, but the model,
residual, and regularization functions remain explicit so that the
mathematical correspondence is visible and testable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import leastsq


def real_func(x: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the teaching signal ``sin(2*pi*x)``."""

    values = np.asarray(x, dtype=float)
    return np.sin(2 * np.pi * values)


def fit_func(parameters: Sequence[float] | np.ndarray, x: Sequence[float] | np.ndarray):
    """Evaluate a polynomial using NumPy's descending-coefficient order."""

    polynomial = np.poly1d(np.asarray(parameters, dtype=float))
    return polynomial(np.asarray(x, dtype=float))


def residuals_func(
    parameters: Sequence[float] | np.ndarray,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return model residuals for the unregularized fit."""

    return fit_func(parameters, x) - np.asarray(y, dtype=float)


def residuals_func_regularization(
    parameters: Sequence[float] | np.ndarray,
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    regularization: float = 0.0001,
) -> np.ndarray:
    """Append the L2 penalty residuals used by the teaching example."""

    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    parameter_values = np.asarray(parameters, dtype=float)
    residuals = np.atleast_1d(residuals_func(parameter_values, x, y))
    penalty = np.sqrt(0.5 * regularization * np.square(parameter_values))
    return np.concatenate((residuals, penalty))


def _initial_parameters(
    degree: int, initial: Sequence[float] | np.ndarray | None
) -> np.ndarray:
    if degree < 0:
        raise ValueError("degree must be non-negative")

    if initial is None:
        return np.zeros(degree + 1, dtype=float)

    parameters = np.asarray(initial, dtype=float)
    if parameters.shape != (degree + 1,):
        raise ValueError(f"initial must contain exactly {degree + 1} parameters")
    return parameters.copy()


def fit_polynomial(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    degree: int,
    initial: Sequence[float] | np.ndarray | None = None,
):
    """Fit a polynomial and return SciPy's familiar ``leastsq`` result tuple."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape")

    parameters = _initial_parameters(degree, initial)
    return leastsq(residuals_func, parameters, args=(x_values, y_values))


def fit_polynomial_regularized(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    degree: int,
    regularization: float = 0.0001,
    initial: Sequence[float] | np.ndarray | None = None,
):
    """Fit a polynomial with the notebook's L2 residual penalty."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape")

    parameters = _initial_parameters(degree, initial)
    return leastsq(
        residuals_func_regularization,
        parameters,
        args=(x_values, y_values, regularization),
    )
