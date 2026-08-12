import numpy as np

from LeastSquaresMethod.least_squares import (
    fit_func,
    fit_polynomial,
    residuals_func_regularization,
)


def test_fit_func_evaluates_polynomial_coefficients_in_numpy_order():
    x = np.array([-1.0, 0.0, 2.0])

    np.testing.assert_allclose(fit_func([2.0, -1.0, 3.0], x), [6.0, 3.0, 9.0])


def test_fit_polynomial_recovers_an_exact_quadratic():
    x = np.linspace(-1.0, 1.0, 9)
    y = 2.0 * x**2 - x + 3.0

    result = fit_polynomial(x, y, degree=2, initial=np.zeros(3))

    np.testing.assert_allclose(result[0], [2.0, -1.0, 3.0], atol=1e-8)


def test_regularized_residuals_append_the_l2_penalty_term():
    parameters = np.array([3.0, 4.0])
    x = np.array([0.0])
    y = np.array([0.0])

    residuals = residuals_func_regularization(
        parameters, x, y, regularization=0.5
    )

    np.testing.assert_allclose(residuals, [4.0, 1.5, 2.0])
