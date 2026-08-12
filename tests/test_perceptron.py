import numpy as np

from Perceptron.perceptron import Perceptron


def test_decision_function_is_the_linear_score():
    model = Perceptron(initial_weights=[2.0, -1.0], initial_bias=0.5)

    assert model.decision_function(np.array([3.0, 2.0])) == 4.5


def test_perceptron_learns_a_linearly_separable_dataset():
    features = np.array(
        [
            [2.0, 2.0],
            [3.0, 1.0],
            [-2.0, -2.0],
            [-3.0, -1.0],
        ]
    )
    labels = np.array([1, 1, -1, -1])
    model = Perceptron(max_iter=100)

    assert model.fit(features, labels) == "Perceptron Model!"
    np.testing.assert_array_equal(model.predict(features), labels)


def test_score_reports_the_fraction_of_correct_predictions():
    features = np.array([[1.0, 1.0], [-1.0, -1.0]])
    labels = np.array([1, -1])
    model = Perceptron(initial_weights=[1.0, 1.0])

    assert model.score(features, labels) == 1.0
