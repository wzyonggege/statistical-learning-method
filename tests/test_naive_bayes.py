import numpy as np

from NaiveBayes.naive_bayes import NaiveBayes


def test_gaussian_probability_is_highest_at_the_mean():
    model = NaiveBayes()

    assert model.gaussian_probability(0.0, mean=0.0, stdev=1.0) > model.gaussian_probability(
        2.0, mean=0.0, stdev=1.0
    )


def test_handwritten_gaussian_naive_bayes_predicts_two_simple_classes():
    features = np.array([[0.0], [0.2], [5.0], [5.2]])
    labels = np.array([0, 0, 1, 1])
    model = NaiveBayes()

    assert model.fit(features, labels) == "gaussianNB train done!"
    assert model.predict([0.1]) == 0
    assert model.score(features, labels) == 1.0
