import numpy as np

from AdaBoost.adaboost import AdaBoost


def test_adaboost_handles_a_perfect_threshold_with_finite_alpha():
    features = np.array([[0.0], [1.0]])
    labels = np.array([-1, 1])
    model = AdaBoost(n_estimators=5, learning_rate=0.5)

    model.fit(features, labels)

    assert model.score(features, labels) == 1.0
    assert all(np.isfinite(model.alpha))


def test_adaboost_uses_a_midpoint_when_threshold_step_is_too_coarse():
    features = np.array([[0.0], [0.1]])
    labels = np.array([-1, 1])
    model = AdaBoost(n_estimators=3, learning_rate=1.0)

    model.fit(features, labels)

    assert model.score(features, labels) == 1.0


def test_adaboost_handles_constant_features_with_a_majority_stump():
    features = np.ones((3, 1))
    labels = np.array([-1, 1, 1])
    model = AdaBoost(n_estimators=3, learning_rate=0.2)

    model.fit(features, labels)

    assert model.predict([1.0]) == 1
    assert model.score(features, labels) == 2 / 3
    np.testing.assert_allclose(model.weights.sum(), 1.0)
