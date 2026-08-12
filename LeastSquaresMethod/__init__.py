"""Hand-written least-squares teaching helpers."""

from .least_squares import (
    fit_func,
    fit_polynomial,
    fit_polynomial_regularized,
    real_func,
    residuals_func,
    residuals_func_regularization,
)

__all__ = [
    "fit_func",
    "fit_polynomial",
    "fit_polynomial_regularized",
    "real_func",
    "residuals_func",
    "residuals_func_regularization",
]
