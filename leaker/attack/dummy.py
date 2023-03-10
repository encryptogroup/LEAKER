"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from math import ceil
from random import random
from typing import Iterable, List, Any, Tuple, Set, Type, TypeVar

from ..extension import IdentityExtension, SelectivityExtension
from ..api import KeywordAttack, Dataset, LeakagePattern, RangeAttack, RelationalAttack, RelationalQuery, \
    RelationalKeyword, RelationalDatabase, Extension

E = TypeVar("E", bound=Extension, covariant=True)


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


class DummyRelationalAttack(RelationalAttack):
    """A dummy attack for validation purposes."""
    @classmethod
    def name(cls) -> str:
        return 'dummy'

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return []

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {IdentityExtension, SelectivityExtension}

    def recover(self, dataset: RelationalDatabase, queries: Iterable[RelationalQuery]) -> List[RelationalKeyword]:
        qlist = list(queries)
        correct = ceil(len(qlist) * random())

        return qlist[:correct] + ([RelationalKeyword(None, -1, -1, "incorrect")] * (len(qlist) - correct))


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
