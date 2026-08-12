import numpy as np

from KNearestNeighbors.knn import KNN, L


def test_minkowski_distance_supports_one_dimensional_points():
    assert L([1.0], [4.0], p=2) == 3.0


def test_knn_predicts_the_majority_label_not_the_largest_label():
    features = np.array([[0.0], [1.0], [10.0]])
    labels = np.array([0, 0, 1])
    model = KNN(features, labels, n_neighbors=3)

    assert model.predict([0.25]) == 0


def test_knn_score_reports_accuracy_for_the_training_examples():
    features = np.array([[0.0], [1.0], [10.0]])
    labels = np.array([0, 0, 1])
    model = KNN(features, labels, n_neighbors=1)

    assert model.score(features, labels) == 1.0
