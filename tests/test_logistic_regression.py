import numpy as np

from LogisticRegression.logistic_regression import LogisticRegressionClassifier


def test_sigmoid_accepts_numpy_vector_and_returns_matching_shape():
    model = LogisticRegressionClassifier()

    result = model.sigmoid(np.array([0.0, 2.0]))

    np.testing.assert_allclose(result, [0.5, 1 / (1 + np.exp(-2.0))])
    assert result.shape == (2,)


def test_handwritten_logistic_regression_learns_separable_points():
    features = np.array([[0.0], [0.5], [2.0], [2.5]])
    labels = np.array([0, 0, 1, 1])
    model = LogisticRegressionClassifier(max_iter=200, learning_rate=0.1)

    model.fit(features, labels)

    assert model.score(features, labels) == 1.0
    np.testing.assert_array_equal(model.predict(features), labels)


def test_historical_misspelled_classifier_name_remains_available():
    from LogisticRegression.logistic_regression import LogisticReressionClassifier

    assert LogisticReressionClassifier is LogisticRegressionClassifier
