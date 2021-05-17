"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from math import ceil
from random import random
from typing import Iterable, List, Any, Tuple

from ..api import KeywordAttack, Dataset, LeakagePattern, RangeAttack


class DummyAttack(KeywordAttack):
    """A dummy attack for validation purposes. It produces a line of roughly y = x with some noise applied."""

    __rate: float

    def __init__(self, known: Dataset):
        super(DummyAttack, self).__init__(known)

        self.__rate = known.sample_rate()

    @classmethod
    def name(cls) -> str:
        return 'dummy'

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return []

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        qlist = list(queries)
        with_random = min([1, max([0, self.__rate + ((random() - 0.5) / 10) * self.__rate])])
        correct = ceil(len(qlist) * with_random)

        return qlist[:correct] + (['dummy'] * (len(qlist) - correct))


class RangeBaselineAttack(RangeAttack):
    """
    Implements a baseline range attack that just guesses the min value
    """

    @classmethod
    def name(cls) -> str:
        return "Baseline"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[int]]:
        return []

    def recover(self, queries: Iterable[Tuple[int, int]]) -> List[int]:

        res = [self.db().get_min() for _ in range(len(self.db()))]

        return res


class RangeCountBaselineAttack(RangeAttack):
    """
    Implements a baseline count range attack that just guesses the average counts for each value [min...max]
    """

    @classmethod
    def name(cls) -> str:
        return "CountBaseline"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[int]]:
        return []

    def recover(self, queries: Iterable[Tuple[int, int]]) -> List[int]:

        big_n = self.db().get_max()

        res = [len(self.db()) // big_n for _ in range(big_n)]

        return res
