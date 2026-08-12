import numpy as np

from SVM.svm import SVM


def test_linear_kernel_matches_the_inner_product_without_fitting_first():
    model = SVM(kernel="linear")

    assert model.kernel(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 11.0


def test_polynomial_kernel_preserves_the_notebook_formula():
    model = SVM(kernel="poly")

    assert model.kernel(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 144.0


def test_handwritten_svm_separates_a_linearly_separable_dataset():
    features = np.array(
        [
            [2.0, 2.0],
            [3.0, 1.0],
            [2.5, 3.0],
            [-2.0, -2.0],
            [-3.0, -1.0],
            [-2.5, -3.0],
        ]
    )
    labels = np.array([1, 1, 1, -1, -1, -1])

    model = SVM(max_iter=200)

    assert model.fit(features, labels) == "train done!"
    assert model.score(features, labels) == 1.0
