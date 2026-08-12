import pytest

from EM.em import EM


def test_em_pmf_uses_the_observation_and_mixture_weight():
    model = EM(prob=[0.5, 0.8, 0.2])
    updates = model.fit([0, 1])
    next(updates)

    assert model.pmf(0) == pytest.approx(0.2)
    assert model.pmf(1) == pytest.approx(0.8)


def test_em_fit_uses_the_data_argument_for_the_e_step():
    model = EM(prob=[0.5, 0.8, 0.2])
    updates = model.fit([0, 0, 0, 0])

    next(updates)
    next(updates)

    assert model.pro_A == pytest.approx(0.2)
    assert model.pro_B == pytest.approx(0.0)
    assert model.pro_C == pytest.approx(0.0)


def test_em_generator_reaches_the_textbook_fixed_point():
    model = EM(prob=[0.5, 0.5, 0.5])
    observations = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    updates = model.fit(observations)

    next(updates)
    next(updates)
    next(updates)

    assert model.pro_A == pytest.approx(0.5)
    assert model.pro_B == pytest.approx(0.6)
    assert model.pro_C == pytest.approx(0.6)
