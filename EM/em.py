"""Generator-based EM for a two-component Bernoulli mixture."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator


class EM:
    """Expectation-maximization following the original notebook formulas.

    ``fit`` remains a generator so the notebook can demonstrate each E/M
    update with ``next`` and ``send``. Observations are captured by the
    generator, avoiding the notebook's former dependency on global ``data``.
    """

    def __init__(self, prob: Iterable[float]) -> None:
        probabilities = tuple(prob)
        if len(probabilities) != 3:
            raise ValueError("prob must contain pro_A, pro_B, and pro_C")
        if not all(0 < value < 1 for value in probabilities):
            raise ValueError("all initial probabilities must be between 0 and 1")
        self.pro_A, self.pro_B, self.pro_C = map(float, probabilities)
        self._data: tuple[int, ...] | None = None

    def pmf(self, i: int, data: Iterable[int] | None = None) -> float:
        """Return the E-step posterior for observation ``i``."""

        observations = self._data if data is None else tuple(data)
        if observations is None:
            raise RuntimeError("fit must initialize the observations first")
        if not 0 <= i < len(observations):
            raise IndexError("observation index out of range")
        value = observations[i]
        if value not in (0, 1):
            raise ValueError("observations must be binary 0/1 values")

        pro_1 = self.pro_A * math.pow(self.pro_B, value) * math.pow(
            1 - self.pro_B, 1 - value
        )
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, value) * math.pow(
            1 - self.pro_C, 1 - value
        )
        return pro_1 / (pro_1 + pro_2)

    def fit(self, data: Iterable[int]) -> Iterator[None]:
        """Yield before each update, preserving the notebook's generator API."""

        observations = tuple(data)
        if not observations:
            raise ValueError("data must contain at least one observation")
        if not all(value in (0, 1) for value in observations):
            raise ValueError("observations must be binary 0/1 values")
        self._data = observations
        count = len(observations)
        print("init prob:{}, {}, {}".format(self.pro_A, self.pro_B, self.pro_C))

        for iteration in range(count):
            yield None
            responsibilities = [self.pmf(index) for index in range(count)]
            responsibility_sum = sum(responsibilities)
            complement_sum = count - responsibility_sum
            pro_A = responsibility_sum / count
            pro_B = sum(
                responsibility * value
                for responsibility, value in zip(responsibilities, observations)
            ) / responsibility_sum
            pro_C = sum(
                (1 - responsibility) * value
                for responsibility, value in zip(responsibilities, observations)
            ) / complement_sum
            print(
                "{}/{}  pro_a:{:.3f}, pro_b:{:.3f}, pro_c:{:.3f}".format(
                    iteration + 1, count, pro_A, pro_B, pro_C
                )
            )
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C


__all__ = ["EM"]
