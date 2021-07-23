"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from abc import ABC, abstractmethod
from logging import getLogger
from random import sample
from typing import Collection, Set, List, Iterator, Tuple, Any, Dict

import numpy as np

from .constants import Selectivity
from .dataset import Dataset, KeywordQueryLog
from .range_database import RangeDatabase

log = getLogger(__name__)


class QuerySpace(ABC, Collection):
    """
    A query space can be used to select a query sequence of arbitrary length (up to the size of the query space).

    It can be used to select multiple instances of queries.
    """

    @abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def select(self, n: int) -> Iterator:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, entry) -> bool:
        raise NotImplementedError


class KeywordQuerySpace(QuerySpace):
    __space: List[Set[Tuple[str, int]]]
    __allow_repetition: bool

    def __init__(self, full: Dataset, known: Dataset, selectivity: Selectivity, size: int,
                 query_log: KeywordQueryLog = None,
                 allow_repetition: bool = False):
        """
        Creates and populates the query space.

        Parameters
        ----------
        full : Dataset
            the full data set
        known : Dataset
            the known data set
        selectivity : Selectivity
            the selectivity of the keywords to use
        size : int
            the desired size of the query space
        query_log : QeryLog
            the query log of users
        allow_repetition : bool
            whether repetitions are allowed when drawing query sequences
        """
        self.__space: List[Set[Tuple[str, int]]] = []

        for i, candidate_keywords in enumerate(self._candidates(full, known, query_log)):
            if len(candidate_keywords) < size:
                log.warning(f"Set of candidate keywords with length {len(candidate_keywords)} at position {i} smaller "
                            f"than configured query space size of {size}. Requested selectivity ignored.")
                self.__space.append(candidate_keywords)
                continue
            if selectivity == Selectivity.High:
                self.__space.append(set(sorted(candidate_keywords, key=lambda item: full.selectivity(item[0]),
                                               reverse=True)[:size]))
            elif selectivity == Selectivity.Low:
                self.__space.append(set(sorted(candidate_keywords, key=lambda item: full.selectivity(item[0]))[:size]))
            elif selectivity == Selectivity.PseudoLow:
                self.__space.append(set(sorted(filter(lambda item: 10 <= full.selectivity(item[0]), candidate_keywords),
                                               key=lambda item: full.selectivity(item[0]))[:size]))
            elif selectivity == Selectivity.Independent:
                self.__space.append(set(sample(population=candidate_keywords, k=size)))

        self.__allow_repetition = allow_repetition

    @classmethod
    def create(cls, full: Dataset, known: Dataset, selectivity: Selectivity, size: int, query_log: KeywordQueryLog,
               allow_repetition: bool = False) -> 'KeywordQuerySpace':
        """
        Creates a query space.

        Parameters
        ----------
        full : Dataset
            the full data set
        known : Dataset
            the known data set
        selectivity : Selectivity
            the selectivity of the keywords to use
        size : int
            the desired size of the query space
        query_log : QeryLog
            the query log of users
        allow_repetition : bool
            whether repetitions are allowed when drawing query sequences

        Returns
        -------
        create : QuerySpace
            the created query space
        """
        return cls(full, known, selectivity, size, query_log, allow_repetition)

    def _get_space(self) -> Iterator[Set[Tuple[str, int]]]:
        yield from self.__space

    def select(self, n: int) -> Iterator[List[str]]:
        """
        Selects a query sequence of the desired length.

        Parameters
        ----------
        n : int
            the length of the query sequence

        Returns
        -------
        select : Iterator[List[str]]
            the selected queries per query space
        """
        # choice is expecting sequences with the same ordering here, hence the list(...)
        length = n
        for i, space in enumerate(self.__space):
            if len(space) == 0:
                log.warning(f"Encountered empty space ar position {i + 1} of {len(self.__space)}. "
                            f"Less evaluations than anticipated are performed.")
                continue
            if len(space) < n and not self.__allow_repetition:
                log.warning(
                    f"Encountered insufficiently large query space  with size {len(space)} at position {i + 1} of"
                    f" {len(self.__space)}.")
                length = len(space)
            space = list(space)
            queries = list(map(lambda item: item[0], space))
            p = np.array(list(map(lambda item: float(item[1]), space)))
            p /= p.sum()
            """We can't sample directly for relational queries because np sees tuples as arrays"""
            idx = np.random.choice(len(queries), length, p=p, replace=self.__allow_repetition)
            yield [queries[i] for i in idx]

    def __len__(self) -> int:
        return len(self.__space)

    def __iter__(self) -> Iterator[Iterator[str]]:
        for space in self.__space:
            yield from map(lambda item: item[0], iter(space))

    def __contains__(self, keyword: object) -> bool:
        return any([keyword in dict(space) for space in self.__space])

    @classmethod
    def is_multi_user(cls) -> bool:
        """Return True if multiple users are considered, False if a single user or queries aggregated from all users
        are considered."""
        return False

    @classmethod
    @abstractmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        """
        Returns one or multiple sets of keyword candidates for populating the query space. Multiple sets can be used to,
        e.g., yield queries of individual users. Keyword candidates consist of the keyword and their frequency/weights
        for query selection.

        Parameters
        ----------
        full : Dataset
            the full data set
        known : Dataset
            the known data set
        query_log : QeryLog
            the query log of users
        """
        raise NotImplementedError


class RangeQuerySpace(QuerySpace):
    """
    A class to represent a QuerySpace for range queries.
    :param n: An upper bound on the number of queries to return. Returns all if = -1.
    The query space is re-created after each sampling if resample is set True
    """
    __queries: List[List[Tuple[int, int]]]
    __db: RangeDatabase
    __allow_repetition: bool
    __allow_empty: bool
    __kwargs: Dict[str, Any]
    __n: int
    __resample: bool

    def __init__(self, db: RangeDatabase, n: int = -1, allow_repetition: bool = True, allow_empty: bool = True,
                 resample: bool = True, **kwargs):
        self.__allow_repetition = allow_repetition
        self.__allow_empty = allow_empty
        self.__kwargs = kwargs
        self.__db = db
        self.__n = n
        self.__resample = resample

        self.__queries = self.gen_queries(self.__db, self.__n, self.__allow_repetition, self.__allow_empty,
                                          **self.__kwargs)

    @classmethod
    def create(cls, db: RangeDatabase, allow_repetition: bool = True, allow_empty: bool = True, **kwargs) \
            -> 'RangeQuerySpace':
        return cls(db, allow_repetition, allow_empty, **kwargs)

    def get_size(self) -> int:
        """
        Return the number of queries.
        :return: Number of queries
        """
        return self.__len__()

    def select(self, n: int = -1) -> Iterator[List[Tuple[int, int]]]:
        """
        Return n queries from the query space for each of its users.
        :param n: The number of queries to return. Returns all if = -1.
        :return: The queries
        """
        for queries in self.__queries:
            if n == -1 or n >= len(queries):
                res = queries
            else:
                res = sample(population=queries, k=n)
            yield res

        if self.__resample:
            self.__queries = self.gen_queries(self.__db, self.__n, self.__allow_repetition, self.__allow_empty,
                                              **self.__kwargs)

    @classmethod
    @abstractmethod
    def gen_queries(cls, db: RangeDatabase, n: int, allow_repetition: bool = False, allow_empty: bool = False,
                    **kwargs) -> List[List[Tuple[int, int]]]:
        """This implements the actual query space, creating a sequence of n queries according to a distribution,
        possibly sampled for multiple users"""
        raise NotImplementedError

    def __len__(self):
        return len(self.__queries)

    def __iter__(self):
        return iter(self.__queries)

    def __contains__(self, item):
        return item in self.__queries
