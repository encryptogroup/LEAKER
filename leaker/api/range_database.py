"""
For License information see the LICENSE file.

Authors: Amos Treiber, Michael Yonli

"""
import random
import math

from collections import Counter
from itertools import chain, repeat
from logging import getLogger
from typing import List, Union, Set, Dict, Tuple, Iterable, Callable
from abc import ABC, abstractmethod
from numpy.random import choice
from numba import njit, prange

import numpy as np

from .constants import MIN_USER_QUERYLOG_ACTIVITY
from ..util import beta_intervals

log = getLogger(__name__)


class RangeDatabase:
    """
    Class that represents a simple integer range database represented by an array of values.
    If no manual domain {min_val, ..., max_val} is supplied, the original array is
    mapped to a domain {1 ... N}, and the original domain values are kept.

    Parameters
        ----------
        name : str
            The name of the RangeDatabase
        values : List[Union[int, float]]
            The values of the database (float values will be scaled to int)
        min_val : int
            If set, the min value of the domain. If not, it will be 1
        max_val : int
            If set, the max value of the domain. If not, it will be the largest kept value
        allow_repetition : bool
            Whether to keep repetitions of values/search keys or discard them
            default: True
    """

    __name: str
    __min_val: int
    __original_min_val: int
    __max_val: int
    __original_max_val: int
    __values: np.ndarray

    def __init__(self, name: str, values: List[Union[int, float]], min_val: int = -1, max_val: int = -1,
                 allow_repetition: bool = True):
        log.info(f"Forming Range Database {name}")

        self.__name = name

        if any(isinstance(val, float) for val in values):
            # For now: just round it.
            values = [int(val) for val in np.around(np.array(values), decimals=0)]

        if not allow_repetition:
            values = list(set(values))
            random.shuffle(values)

        self.__original_min_val = min(values)
        self.__original_max_val = max(values)

        if min_val == -1:
            self.__values = np.array(values) - self.__original_min_val + 1
            self.__min_val = 1
        else:
            self.__min_val = min_val
            self.__values = np.array(values)

        if max_val == -1:
            self.__max_val = max(self.__values)
        else:
            self.__max_val = max_val

    def name(self) -> str:
        return self.__name

    def query(self, *query):
        """
        Queries the range database
        :param query: lower and upper bound (inclusive) to include.
        :return: a numpy array containing all records (indices) of values that are within the range
        """
        if len(query) == 1:
            lower_bound = query[0][0]
            upper_bound = query[0][1]
        else:
            lower_bound = query[0]
            upper_bound = query[1]

        return np.where((self.__values >= lower_bound) & (self.__values <= upper_bound))[0]

    def selectivity(self, *query) -> int:
        return len(self.query(*query))

    def get_ordering(self) -> List:
        return list(np.argsort(self.get_numerical_values()))

    def get_numerical_values(self) -> List[Union[int, float]]:
        return list(self.__values)

    def get_original_min(self) -> int:
        """Returns the original minimal value"""
        return self.__original_min_val

    def get_min(self) -> int:
        return self.__min_val

    def get_max(self) -> int:
        return self.__max_val

    def get_original_max(self) -> int:
        return self.__original_max_val

    def get_n(self) -> int:
        return len(self)

    def get_density(self) -> float:
        return len(set(self.get_numerical_values())) / (self.get_max() - self.get_min() + 1)

    def get_rank(self, value: int) -> int:
        """leaking the number of records with lesser or equal values to given record value."""
        return len([val for val in self.get_numerical_values() if val <= value])

    def get_weights(self) -> Dict[int, int]:
        """return the probability distribultion of values occuring among records in the DB."""
        counts = Counter(self.get_numerical_values())
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    def get_num_of_values(self) -> int:
        """Returns the number of unique values appearing in the dataset"""
        return len(set(self.get_numerical_values()))

    def restrict_rate(self, rate: float) -> 'RangeDatabase':
        """Outputs a copy of the db restricted to num elements chosen uniformly at random"""
        num = int(rate * len(self))
        if num >= len(self):
            return self
        else:
            return type(self)(f"{self.name()}_r{rate}",
                              values=[a for a in random.sample(self.get_numerical_values(), k=num)],
                              min_val=self.get_min(), max_val=self.get_max())

    def __len__(self) -> int:
        return self.__values.size

    def __getitem__(self, item: int) -> int:
        return self.__values[item]

    def append(self, value: Union[float, int]):
        """In-place appending of values only works if manual min and max values were supplied in the beginning,
        as the scaling to 1...N is not done here yet"""
        if isinstance(value, float):
            value = int(round(value))
        self.__values = np.append(self.__values, value)
        if value > self.get_max():
            self.__max_val = value
        if value < self.get_min():
            self.__min_val = value

    def __setitem__(self, key: int, value: Union[float, int]):
        """In-place appending of values only works if manual min and max values were supplied in the beginning,
                as the scaling to 1...N is not done here yet"""
        if isinstance(value, float):
            value = int(round(value))
        self.__values[key] = value
        if value >= self.get_max():
            self.__max_val = value
        if value < self.get_min():
            self.__min_val = value


class RandomRangeDatabase(RangeDatabase):
    """Class that randomly generates a RangeDatabase given min, max values and length and/or density. Mainly used for
    testing purposes and to reproduce results.

    If length and density are given, a DB with the given length and density will be created.

    If just the density is given, the length will be computed based on the density, and a DB of that length is created.
        *If you allow repeated values here, the resulting density might not actually be achieved,
        because no length information is given. Using allow_repetition=False is recommended here.

    If just the length is given, the density will be a result of the random sampling and the length.

    If neither density nor length is given, a density of 1 will be assumed and values cannot be repeated, i.e.,
    a dense DB [min_value ... max_value] will be created."""

    def __init__(self, name: str, min_val: int, max_val: int, density: float = -1, length: int = -1,
                 allow_repetition: bool = False):
        if density != -1 and length != -1:
            assert (0 <= density)
            num_vals = round((max_val - min_val + 1) * density)
            assert length >= num_vals
            assert allow_repetition or length == num_vals
            value_set = np.random.choice(np.arange(min_val, max_val + 1), num_vals, False)
            vals = np.concatenate((value_set, np.random.choice(
                value_set, length - num_vals, allow_repetition)), axis=None)
            np.random.shuffle(vals)
        else:
            if density != -1:
                assert (0 <= density)
                length = round((max_val - min_val + 1) * density)
            elif length == -1:
                assert (not allow_repetition)
                log.debug(f"No density information given to RandomRangeDatabase. Using 1.")
                length = max_val - min_val + 1

            vals = np.random.choice(np.arange(min_val, max_val + 1), length, allow_repetition)

        super().__init__(name, vals, min_val, max_val, allow_repetition)


class PermutedBetaRandomRangeDatabase(RangeDatabase):
    """A random database without multiplicities sampled according to a PermutedBeta distribution (see KPT21)"""

    def __init__(self, name: str, min_val: int, max_val: int, density: float, alpha: int = 1, beta: int = 5):
        big_n = max_val - min_val + 1
        weights = np.array(beta_intervals(alpha, beta, big_n), dtype='float')
        weights /= weights.sum()

        np.random.shuffle(weights)

        values = choice(big_n, round(density * big_n), p=weights, replace=False)

        super().__init__(name, [min_val + v for v in values], min_val, max_val, allow_repetition=False)


class RegularRangeDatabase(RangeDatabase, ABC):
    """
    Class that gives a RangeDatabase querying according to a specific regular STE scheme, as defined by KPT21.
    Has to implement loss functions for the KPT21 attack.
    Additional Parameters:
     ----------
    big_t: Callable[[int], int]
        big_t[s] denotes the steps of ranges of width s<=N.
    compute_canonical_queries: bool
        whether all canonical queries shall be pre-computed.
    """
    _big_t: Callable[[int], int]
    _canonical_queries: List[Tuple[int, int]]

    def __init__(self, name: str, values: List[Union[int, float]], big_t: Union[None, Callable[[int], int]],
                 compute_canonical_queries: bool = False, min_val: int = -1, max_val: int = -1,
                 allow_repetition: bool = True):

        super().__init__(name, values, min_val, max_val, allow_repetition)
        self._big_t = big_t
        big_n = self.get_max() - self.get_min() + 1

        if compute_canonical_queries:
            self._canonical_queries = [(k * self._big_t(s) + 1, k * self._big_t(s) + s)
                                       for s in range(1, big_n + 1) if self._big_t(s) > 0
                                       for k in range(int(math.floor((big_n - s) / self._big_t(s))) + 1)]
        else:
            self._canonical_queries = []

    def num_canonical_queries(self) -> int:
        """Q(N) of KPT21, returns number of total canonical ranges"""
        return len(self._canonical_queries)

    def counting_function(self, r: Set[int], s: int) -> int:
        """
        C(r,s) of KPT21, returns number of canonical queries of width s that return response r. May be overwritten
        by subclasses.
        Parameters:
        ----------
        r: Set[int]
            ids of  DB values returned
        s: int
            width of queries
        """
        pass

    def global_counting_function(self, r: Set[int]) -> int:
        """
        G(r) of KPT21, returns number of canonical queries of any width that return response r. May be overwritten
        by subclasses. Can be used to calculate the loss function.
        Parameters:
        ----------
        r: Set[int]
            ids of DB values returned
        """
        return sum(self.counting_function(r, s) for s in range(1, self.get_max() - self.get_min() + 1))

    @abstractmethod
    def loss(self, big_l: List[int], theta, weights) -> float:
        """
        Loss function.
        Parameters:
        ----------
        big_l: List[int]
            Estimated distances
        """
        assert len(big_l) == len(self) + 1
        raise NotImplementedError


class QDRangeDatabase(RegularRangeDatabase):
    """Implements the Quadratic Schemes of KPT21"""

    def __init__(self, name: str, values: List[Union[int, float]], min_val: int = -1, max_val: int = -1,
                 allow_repetition: bool = True):
        def big_t(s: int) -> int:
            return 1

        super().__init__(name, values, big_t, False, min_val, max_val, allow_repetition)

    def num_canonical_queries(self) -> int:
        big_n = self.get_max() - self.get_min() + 1
        return int(big_n * (big_n + 1) / 2)

    def global_counting_function(self, r: Set[int]) -> int:
        # This function is currently not used and may need rewriting
        ordering = self.get_ordering()
        big_n = self.get_max() - self.get_min() + 1
        r = list(r)
        db_vals = self.get_numerical_values()
        r_val = [db_vals[idx] for idx in r]

        v_i = r[np.argmin(r_val)[0]]
        i = self.get_rank(v_i - 1) + 1  # i is the ordered index in KPT21
        v_j = np.argmax(r)[0]
        j = self.get_rank(v_j) + 1  # j = i + k + 1

        assert 1 <= i < j <= big_n + 1

        """Return L_i * L_i+k+1"""
        if i == 1:
            big_l_i = abs(v_i - self.get_min() + 1)
        else:
            big_l_i = abs(v_i - self.get_numerical_values()[ordering[i - 2]])  # we want v_i-1, but ordering starts at 0

        if j == len(self) + 1:
            big_l_j = abs(big_n + 1 - v_j)
        else:
            big_l_j = abs(v_j - self.get_numerical_values()[ordering[j - 2]])

        return big_l_i * big_l_j

    def loss(self, big_l: List[int], theta, weights) -> float:
        """TODO - not yet implemented"""
        raise NotImplementedError


def big_t_base(s: int) -> int:
    """T_BASE from KPT21"""
    if (s & (s - 1) == 0) and s != 0:
        """If s=2^i for some i"""
        return 1
    else:
        return 0


class BaseRangeDatabase(RegularRangeDatabase):
    """Implements the Base Scheme of KPT21"""

    def __init__(self, name: str, values: List[Union[int, float]], min_val: int = -1, max_val: int = -1,
                 allow_repetition: bool = True):

        # This check ensures that the following calculations are correct
        if min_val != -1 and min_val != 1:
            raise ValueError("min_val must be set to -1 or 1")

        m_val = int(max(values))
        """N has to be power of 2"""
        if max_val == -1 and not big_t_base(m_val):
            max_val = 1 << m_val.bit_length()  # next power of 2
        elif not big_t_base(max_val):
            max_val = 1 << int(max_val).bit_length()

        super().__init__(name, values, big_t_base, True, min_val, max_val, allow_repetition)

    def query(self, *query):
        """Finds closest matching canonical query q' to query and executes it"""
        if len(query) == 1:
            lower_bound = query[0][0]
            upper_bound = query[0][1]
        else:
            lower_bound = query[0]
            upper_bound = query[1]

        candidates = [x for x in self._canonical_queries if lower_bound >= x[0] and upper_bound <= x[1]]
        assert len(candidates) >= 1

        lengths = np.array([c[1] - c[0] for c in candidates])

        idx = np.argmin(lengths)
        q_p = candidates[idx]

        min_len = lengths.min()
        num = np.count_nonzero(lengths == min_len)

        if num > 1:
            for res in np.nonzero(lengths == min_len):
                idx = res[0]
                if candidates[idx][0] == lower_bound:
                    q_p = candidates[idx]
                    break
                elif candidates[idx][1] == upper_bound:
                    q_p = candidates[idx]

        return super().query(q_p)

    @staticmethod
    @njit(parallel=True)
    def numba_loss(n: int, big_n: int, big_l: List[int], theta: List[int], weights: List[int]) -> float:
        res1 = 0
        for i in prange(n + 1):
            for l in range(int(np.floor(np.log2(big_n))) + 1):
                res1 += max(0, round(big_l[i] - (1 << l)))
        res1 -= theta[0]
        res1 = weights[0] * res1 ** 2

        """big_l indices start at 0"""
        res2 = 0
        for k in prange(n):
            res3 = 0
            for i in range(1, n - k + 1):
                sum_big_l = 0
                for t in range(k + 1):
                    sum_big_l += big_l[i + t - 1]
                for l in range(int(np.floor(np.log2(big_n))) + 1):
                    res3 += max(0, min([big_l[i - 1],
                                        big_l[i + k],
                                        sum_big_l + big_l[i + k] - (1 << l),
                                        (1 << l) - sum_big_l + big_l[i - 1]]))
            res3 -= theta[k + 1]
            res3 = weights[k + 1] * res3 ** 2
            res2 += res3

        return res1 + res2

    def loss(self, big_l: List[int], theta: List[int], weights: List[int]) -> float:
        big_n = self.get_max() - self.get_min() + 1
        n = len(self)
        return self.numba_loss(n, big_n, big_l, theta, weights)


def big_t_bt(s: int) -> int:
    """T_BT from KPT21"""
    if (s & (s - 1) == 0) and s != 0:
        """If s=2^i for some i"""
        return s
    else:
        return 0


class BTRangeDatabase(RegularRangeDatabase):
    """Implements the BT Scheme of KPT21/Faber et al., ESORICS'15"""

    def __init__(self, name: str, values: List[Union[int, float]], min_val: int = -1, max_val: int = -1,
                 allow_repetition: bool = True):

        # This check ensures that the following calculations are correct
        if min_val != -1 and min_val != 1:
            raise ValueError("min_val must be set to -1 or 1")

        m_val = int(max(values))
        """N has to be power of 2"""
        if max_val == -1 and not big_t_base(m_val):
            max_val = 1 << m_val.bit_length()  # next power of 2
        elif not big_t_base(max_val):
            max_val = 1 << int(max_val).bit_length()

        super().__init__(name, values, big_t_bt, True, min_val, max_val, allow_repetition)

    def query(self, *query):
        """Finds closest matching canonical query q' to query and executes it"""
        if len(query) == 1:
            lower_bound = query[0][0]
            upper_bound = query[0][1]
        else:
            lower_bound = query[0]
            upper_bound = query[1]

        candidates = [x for x in self._canonical_queries if lower_bound >= x[0] and upper_bound <= x[1]]
        assert len(candidates) >= 1

        lengths = np.array([c[1] - c[0] for c in candidates])
        idx = np.argmin(lengths)
        q_p = candidates[idx]

        return super().query(q_p)

    @staticmethod
    @njit(parallel=True)
    def numba_loss(n: int, big_n: int, big_l: List[int], theta: List[int], weights: List[int]) -> float:
        res1 = 0
        for i in prange(n + 1):
            for l in range(int(np.floor(np.log2(big_n))) + 1):
                res1 += max(0, round((big_l[i] - (1 << l)) / (1 << l)))
        res1 -= theta[0]
        res1 = weights[0] * res1 ** 2

        """big_l indices start at 0"""
        res2 = 0
        for k in prange(n):
            res3 = 0
            for i in range(1, n - k + 1):
                sum_big_l = 0
                for t in range(k + 1):
                    sum_big_l += big_l[i + t - 1]
                for l in range(int(np.floor(np.log2(big_n))) + 1):
                    res3 += max(0, round(min([big_l[i - 1],
                                              big_l[i + k],
                                              sum_big_l + big_l[i + k] - (1 << l),
                                              (1 << l) - sum_big_l + big_l[i - 1]]
                                             ) / (1 << l)))
            res3 -= theta[k + 1]
            res3 = weights[k + 1] * res3 ** 2
            res2 += res3

        return res1 + res2

    def loss(self, big_l: List[int], theta: List[int], weights: List[int]) -> float:
        big_n = self.get_max() - self.get_min() + 1
        n = len(self)
        return self.numba_loss(n, big_n, big_l, theta, weights)


def big_t_abt(s: int) -> int:
    """T_ABT from KPT21"""
    if s == 1:
        return 1
    elif (s & (s - 1) == 0) and s != 0:
        """If s=2^i for some i"""
        return int(s / 2)
    else:
        return 0


class ABTRangeDatabase(RegularRangeDatabase):
    """Implements the ABT Scheme of KPT21/Demertzis et al., SIGMOD'16"""

    def __init__(self, name: str, values: List[Union[int, float]], min_val: int = -1, max_val: int = -1,
                 allow_repetition: bool = True):

        # This check ensures that the following calculations are correct
        if min_val != -1 and min_val != 1:
            raise ValueError("min_val must be set to -1 or 1")

        m_val = int(max(values))
        """N has to be power of 2"""
        if max_val == -1 and not big_t_base(m_val):
            max_val = 1 << m_val.bit_length()  # next power of 2
        elif not big_t_base(max_val):
            max_val = 1 << int(max_val).bit_length()

        super().__init__(name, values, big_t_abt, True, min_val, max_val, allow_repetition)

    def query(self, *query):
        """Finds closest matching canonical query q' to query and executes it"""
        if len(query) == 1:
            lower_bound = query[0][0]
            upper_bound = query[0][1]
        else:
            lower_bound = query[0]
            upper_bound = query[1]

        candidates = [x for x in self._canonical_queries if lower_bound >= x[0] and upper_bound <= x[1]]
        assert len(candidates) >= 1

        lengths = np.array([c[1] - c[0] for c in candidates])
        idx = np.argmin(lengths)
        q_p = candidates[idx]

        return super().query(q_p)

    @staticmethod
    @njit(parallel=True)
    def numba_loss(n: int, big_n: int, big_l: List[int], theta: List[int], weights: List[int]) -> float:
        res1 = 0
        for i in prange(n + 1):
            for l in range(int(np.floor(np.log2(big_n))) + 1):
                res1 += max(0, round((big_l[i] - (1 << l)) / max(1, 2 ** (l - 1))))
        res1 -= theta[0]
        res1 = weights[0] * res1 ** 2

        """big_l indices start at 0"""
        res2 = 0
        for k in prange(n):
            res3 = 0
            for i in range(1, n - k + 1):
                sum_big_l = 0
                for t in range(k + 1):
                    sum_big_l += big_l[i + t - 1]
                for l in range(int(np.floor(np.log2(big_n))) + 1):
                    res3 += max(0, round(min([big_l[i - 1],
                                              big_l[i + k],
                                              sum_big_l + big_l[i + k] - (1 << l),
                                              (1 << l) - sum_big_l + big_l[i - 1]]
                                             ) / max(1, 2 ** (l - 1))))
            res3 -= theta[k + 1]
            res3 = weights[k + 1] * res3 ** 2
            res2 += res3

        return res1 + res2

    def loss(self, big_l: List[int], theta: List[int], weights: List[int]) -> float:
        big_n = self.get_max() - self.get_min() + 1
        n = len(self)
        return self.numba_loss(n, big_n, big_l, theta, weights)


class RangeQueryLog:
    """
    A log of real-world queries issued by users, used for statistics and query space generation. It can be restricted
    to certain user ids. The RangeQueryLog takes as input already pre-processed information about frequencies of
    queries.

    Parameters
    ----------
    name: str
        the name of the query log (index)
    queries: Dict[str, Dict[Tuple[int, int], int]]
        the queries in the form: {user_id: {Query: frequency}}
    min_user_count: int, max_user_count: int
        If given, only consider queries of most_freq_users[min_user_count:max_user_count]
    reverse: bool
        If True, consider queries of least_freq_users[min_user_count:max_user_count] with
        min activity MIN_USER_QUERYLOG_ACTIVITY
    """

    __name: str
    __is_reversed: bool

    __queries: Dict[str, Dict[Tuple[int, int], int]]
    __sorted_user_ids: List[str]

    def __init__(self, name: str, queries: Dict[str, Dict[Tuple[int, int], int]], min_user_count: int = 0,
                 max_user_count: int = None, reverse: bool = False):
        log.info(f"Loading Range Query Log '{name}'")
        self.__name = name
        self.__queries = queries
        users = Counter({items[0]: sum(items[1].values()) for items in self.__queries.items()})
        if reverse:
            user_list = list(dict.fromkeys(list(chain.from_iterable(repeat(i, c) for i, c in users.most_common()[::-1]
                                                                    if c > MIN_USER_QUERYLOG_ACTIVITY))).keys())
        else:
            user_list = list(dict.fromkeys(list(chain.from_iterable(repeat(i, c) for i, c in
                                                                    users.most_common()))).keys())
        if max_user_count is None:
            max_user_count = len(user_list)

        user_list = user_list[min_user_count:max_user_count]

        if reverse:
            user_list.reverse()

        self.__is_reversed = reverse

        self.__sorted_user_ids = user_list

        for user_id in self.__queries.keys():
            if user_id not in self.__sorted_user_ids:
                self.__queries.pop(user_id, None)

        log.info(f"Loaded Range Query Log '{name}'")

    def name(self) -> str:
        return self.__name

    def restrict_user_ids(self, user_ids: Iterable) -> 'RangeQueryLog':
        """Removes any queries from this query log instance issued by users not in user_ids"""
        for user_id in user_ids:
            self.__queries.pop(user_id, None)
        return RangeQueryLog(self.name(), self.__queries, reverse=self.__is_reversed)

    def user_ids(self) -> List[str]:
        """Returns the unique identifiers of all users in this query log, ordered according to activity (descending,
        or ascending if the query log is reversed)."""
        return self.__sorted_user_ids

    def queries(self, user_id: str = None) -> Set[Tuple[int, int]]:
        """Returns the set of all queries in this query log (optionally restricted to a user_id)."""
        return set(self.__call__(user_id))

    def queries_freq(self, user_id: str = None) -> Counter:
        """Returns a Counter of all queries including frequency information in this query log
        (optionally restricted to a user_id)."""
        if user_id is not None:
            return Counter(self.__queries[user_id])
        else:
            return Counter({item[0]: item[1] for value in self.__queries.values() for item in value.items()})

    def __call__(self, user_id: str = None) -> Iterable[Tuple[int, int]]:
        if user_id is not None:
            yield from self.__queries[user_id].keys()
        else:
            yield from [val for value in self.__queries.values() for val in value.keys()]
