"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from abc import ABC, abstractmethod
from typing import Iterable, List, Generic, TypeVar, Union, Tuple

from .range_database import RangeDatabase
from .dataset import Dataset

T = TypeVar("T", covariant=True)


class LeakagePattern(ABC, Generic[T]):
    """A leakage pattern, that is, a function from a sequence of queries or values to some specific leakage type."""

    @abstractmethod
    def leak(self, dataset: Union[Dataset, RangeDatabase], queries: Union[Iterable[int], Iterable[str],
                                                                          Iterable[Tuple[int, int]]]) -> List[T]:
        """
        Calculates the leakage on the given data set and queries.

        Parameters
        ----------
        dataset : Union[Dataset, RangeDatabase]
            the data set or range DB to calculate the leakage on
        queries : Union[Iterable[int], Iterable[str], Iterable[Tuple[int, int]]]
            the values or queries to leak on

        Returns
        -------
        leak : List[T]
            the leakage
        """
        raise NotImplementedError

    def __call__(self, dataset: Union[Dataset, RangeDatabase], queries: Union[Iterable[int], Iterable[str],
                                                                              Iterable[Tuple[int, int]]]) -> List[T]:
        return self.leak(dataset, queries)
