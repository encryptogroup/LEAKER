"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber, Michael Yonli

Implementations of various query spaces.
"""
from logging import getLogger

import numpy as np

from collections import Counter
from itertools import repeat
from random import random
from typing import Set, Iterator, Tuple, List

from numpy.random import choice, default_rng

from ..api import KeywordQuerySpace, KeywordQueryLog, RangeQuerySpace, Dataset, RangeDatabase, RangeQueryLog
from ..util import beta_intervals

log = getLogger(__name__)


class PartialQuerySpace(KeywordQuerySpace):
    """A query space using only the keywords known to the attacker. All keywords are weighted equally."""

    @classmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        yield set(zip(known.keywords(), [1 for _ in range(len(known.keywords()))]))


class FullQuerySpace(KeywordQuerySpace):
    """A query space using the keywords from the full data set.. All keywords are weighted equally."""

    @classmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        yield set(zip(full.keywords(), [1 for _ in range(len(full.keywords()))]))


class PartialQueryLogSpace(KeywordQuerySpace):
    """A query space using the keywords from the query log aggregated over all users.
    Keywords are weighted according to their frequency in the query log. Queries that do not appear in the
    relevant dataset are discarded"""

    @classmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        yield set([item for item in Counter(query_log.keywords_list()).items() if known.selectivity(item[0]) > 0])


class FullQueryLogSpace(KeywordQuerySpace):
    """A query space using the keywords from the query log aggregated over all users.
    Keywords are weighted according to their frequency in the query log. Queries that do not appear in the
    relevant dataset are discarded"""

    @classmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        yield set([item for item in Counter(query_log.keywords_list()).items() if full.selectivity(item[0]) > 0])


class PartialUserQueryLogSpace(KeywordQuerySpace):
    """A query space using the keywords from the query log for each user in the query log.
    Keywords are weighted according to their frequency in the query log of the user. Queries that do not appear in the
    relevant dataset are discarded"""

    @classmethod
    def is_multi_user(cls) -> bool:
        """Return True because multiple users are considered individually"""
        return True


    @classmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        for user_id in query_log.user_ids():
            yield set([item for item in Counter(query_log.keywords_list(user_id)).items()
                       if known.selectivity(item[0]) > 0])


class FullUserQueryLogSpace(KeywordQuerySpace):
    """A query space using the keywords from the query log for each user in the query log.
    Keywords are weighted according to their frequency in the query log of the user. Queries that do not appear in the
    relevant dataset are discarded"""

    @classmethod
    def is_multi_user(cls) -> bool:
        """Return True because multiple users are considered individually"""
        return True


    @classmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        for user_id in query_log.user_ids():
            yield set([item for item in Counter(query_log.keywords_list(user_id)).items()
                       if full.selectivity(item[0]) > 0])


class AuxiliaryKnowledgeQuerySpace(KeywordQuerySpace):
    @classmethod
    def _candidates(cls, full: Dataset, known: Dataset, query_log: KeywordQueryLog) -> Iterator[Set[Tuple[str, int]]]:
        yield set([item for item in Counter(query_log.keywords_list()).items()])
    
    def select(self, n: int=-1) -> Iterator[List[str]]:
        query_log = super()._get_log()
        if n > 0:
            yield from [query_log.generate_queries()]#[query_log.keywords_list()[:n]]
        else:
            yield from [query_log.generate_queries()]
        

class UniformRangeQuerySpace(RangeQuerySpace):
    """A Range query space where queries are generated uniformly at random. If the amount parameter is set to -1,
    this simulates enough queries s.t. all possible queries are issued."""

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int = -1, allow_repetition: bool = True, allow_empty: bool = True,
                    **kwargs) -> List[List[Tuple[int, int]]]:
        """
        Generate queries uniformly at random.

        :param db: The underlying range database.
        :param amount: Number of queries to generate. -1 = all.
        :param allow_repetition: Allow same query multiple times
        :param allow_empty: Allow queries with no matches.
        :return: A list of queries.
        """
        a = db.get_min()
        b = db.get_max()

        queries: List[Tuple[int, int]] = []
        for lower in range(a, b + 1):
            for upper in range(lower, b + 1):
                if not allow_empty and len(db.query((lower, upper))) == 0:
                    continue
                else:
                    queries.append((lower, upper))

        if amount == -1:
            amount = len(queries)

        actual_query_idx = choice(len(queries), amount, replace=allow_repetition)  # default is uniform

        return [[queries[x] for x in actual_query_idx]]


class ShortRangeQuerySpace(RangeQuerySpace):
    """
    Represents queries over short ranges.
    """

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int, allow_repetition: bool = True,
                    allow_empty: bool = True, **kwargs) \
            -> List[List[Tuple[int, int]]]:
        """
        Generate short range queries as per KPT20. This generates queries in a manner which prefers queries over short
            ranges. Queries are sorted by their increasing distance and are then assigned probabilities in that order
            from the Beta distribution Beta(alpha, beta).

        :param db: The underlying range database.
        :param amount: The number of queries to generate.
        :param allow_repetition: The same query may be chosen multiple times.
        :param allow_empty: A query with no replies may be chosen.
        :param kwargs: alpha: Alpha parameter for the Beta distribution that describes the probability of the queries.
                        beta: Beta parameter of the Beta distribution that describes the probability of the queries.
        :return: The generated queries.
        """

        alpha = 1
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        beta = 3
        if "beta" in kwargs:
            beta = kwargs["beta"]

        a = db.get_min()
        b = db.get_max()
        n = b - a + 1  # Not the same as n since we may not care about the replies.
        r = int(n * (n + 1) / 2)  # all possible range responses

        queries = []

        low_bound = upper_bound = a
        delta = 0
        ignored_queries_cnt = 0

        while len(queries) + ignored_queries_cnt < r:
            results = db.query((low_bound, upper_bound))

            if allow_empty or len(results) > 0:
                queries.append((low_bound, upper_bound))
            else:
                ignored_queries_cnt += 1

            if low_bound + delta + 1 <= b:
                low_bound += 1
            else:
                delta += 1
                low_bound = a

            upper_bound = low_bound + delta

        l_queries = len(queries)
        assert l_queries == r - ignored_queries_cnt

        samples = beta_intervals(alpha, beta, l_queries)
        noisy_samples = [sample * random() / l_queries for sample in samples]
        tot_prob = sum(noisy_samples)
        normalised_samples = [elem / tot_prob for elem in noisy_samples]

        if amount == -1:
            amount = l_queries

        actual_queries = choice(l_queries, amount, p=normalised_samples, replace=allow_repetition)

        return [[queries[x] for x in actual_queries]]


class ValueCenteredRangeQuerySpace(RangeQuerySpace):
    """
    Class that represents value centered queries as in KPT20 close to Fig. 10.
    """

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int, allow_repetition: bool = True, allow_empty: bool = True,
                    **kwargs) -> List[List[Tuple[int, int]]]:
        """
        Generate value centered queries as per KPT20.

        :param db: The underlying range database.
        :param amount: The number of queries to generate.
        :param allow_repetition: The same query may be chosen multiple times.
        :param allow_empty: A query with no replies may be chosen.
        :param kwargs: alpha: Alpha parameter for the Beta distribution that describes the probability of the queries.
                        beta: Beta parameter of the Beta distribution that describes the probability of the queries.
        :return: The generated queries.
        """

        alpha = 1
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        beta = 3
        if "beta" in kwargs:
            beta = kwargs["beta"]

        a = db.get_min()
        b = db.get_max()
        n = b - a + 1  # Not the same as n since we may not care about the replies.
        r = int(n * (n + 1) / 2)  # all possible range responses

        queries = []

        ignored_queries_cnt = 0
        vals = list(set(db.get_numerical_values()))
        old_vals = [a - 1, b + 1]

        while len(queries) + ignored_queries_cnt < r:
            current_element = choice(vals)
            vals.remove(current_element)

            if not vals:
                for x in range(a, b + 1):
                    if x not in old_vals:
                        vals.append(x)

            old_vals.append(current_element)
            old_vals = sorted(old_vals)

            next_bigger = old_vals[old_vals.index(current_element) + 1]
            next_smaller = old_vals[old_vals.index(current_element) - 1]

            low_max_delta = current_element - next_smaller - 1
            up_max_delta = next_bigger - current_element - 1
            max_delta = low_max_delta + up_max_delta

            for delta in range(max_delta + 1):
                for low_delta in range(max(0, delta - up_max_delta), min(low_max_delta, delta) + 1):
                    lower_bound = current_element - low_delta
                    upper_bound = lower_bound + delta

                    assert (current_element - low_max_delta <= lower_bound)
                    assert (current_element + up_max_delta >= upper_bound)

                    query = (lower_bound, upper_bound)
                    if allow_empty or len(db.query(query)) > 0:
                        queries.append(query)
                    else:
                        ignored_queries_cnt += 1

        assert len(queries) == r - ignored_queries_cnt
        l_queries = len(queries)

        samples = beta_intervals(alpha, beta, l_queries)
        noisy_samples = [sample * random() / l_queries for sample in samples]
        tot_prob = sum(noisy_samples)
        normalised_samples = [elem / tot_prob for elem in noisy_samples]

        if amount == -1:
            amount = l_queries

        actual_queries = choice(l_queries, amount, p=normalised_samples, replace=allow_repetition)

        return [[queries[x] for x in actual_queries]]


class QueryLogRangeQuerySpace(RangeQuerySpace):
    """
    A range query space using the queries from a query log for all users (aggregated) in the query log.
    Pass non-existing bounds (denoted by None) as db.min() or db.max() respectively.
    """

    def __init__(self, db: RangeDatabase, amount: int = -1, allow_repetition: bool = True, allow_empty: bool = True,
                 **kwargs):
        """Do not resample, as space is created only once"""
        super().__init__(db, amount, allow_repetition, allow_empty, False, **kwargs)

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int, allow_repetition: bool = True, allow_empty: bool = True,
                    **kwargs) -> List[List[Tuple[int, int]]]:

        if "qlog" not in kwargs:
            raise ValueError("qlog must be supplied to create query log RangeQuerySpace")

        qlog: RangeQueryLog = kwargs["qlog"]

        if allow_repetition:
            queries = [query for item in qlog.queries_freq().items() for query in repeat(item[0], item[1])]
        else:
            queries = qlog.queries()

        if not allow_empty:
            queries = [query for query in queries if len(db.query(query)) > 0]

        if len(queries) > amount:
            queries = queries[:amount]

        q = []
        for query in queries:
            lower = query[0]
            upper = query[1]
            if lower is None:
                lower = db.get_min()
            else:
                lower -= db.get_original_min() - 1  # map to DB version scaled to [1...N]
            if upper is None:
                upper = db.get_max()
            else:
                upper -= db.get_original_min() - 1  # map to DB version scaled to [1...N]

            q.append((lower, upper))

        return [q]


class UserQueryLogRangeQuerySpace(QueryLogRangeQuerySpace):
    """
    A range query space using the queries from a query log for each user in the query log.
    """

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int, allow_repetition: bool = True, allow_empty: bool = True,
                    **kwargs) -> List[List[Tuple[int, int]]]:
        """
        :param db: The underlying range database.
        :param amount: The max number of queries to generate.
        :param allow_repetition: The same query may be chosen multiple times.
        :param allow_empty: A query with no replies may be chosen.
        :param kwargs: qlog: RangeQueryLog: The Query Log to take queries from
        :return: The generated queries."""

        if "qlog" not in kwargs:
            raise ValueError("qlog must be supplied to create query log RangeQuerySpace")

        qlog: RangeQueryLog = kwargs["qlog"]

        res = []

        for user_id in qlog.user_ids():
            if allow_repetition:
                queries = [query for item in qlog.queries_freq().items() for query in repeat(item[0], item[1])]
            else:
                queries = qlog.queries(user_id)

            if not allow_empty:
                queries = [query for query in queries if len(db.query(query)) > 0]

            if len(queries) > amount:
                queries = queries[:amount]

            q = []
            for lower, upper in queries:
                if lower is None:
                    lower = db.get_min()
                else:
                    lower -= db.get_original_min() - 1  # map to DB version scaled to [1...N]
                if upper is None:
                    upper = db.get_max()
                else:
                    upper -= db.get_original_min() - 1  # map to DB version scaled to [1...N]

                q.append((lower, upper))

            res.append(q)

        return res


class BoundedRangeQuerySpace(RangeQuerySpace):
    """
    Represent a QuerySpace such that b is an upper bound on the maximum number of discrete distinct values that might
    be returned by a query.

    Allow empty is always assumed to be False and thus ignored.
    """

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int, allow_repetition: bool = False, allow_empty: bool = False,
                    **kwargs) -> List[List[Tuple[int, int]]]:
        vals = sorted(set(db.get_numerical_values()))

        if allow_empty:
            raise ValueError("allow_empty is assumed to be false")

        bound = 3
        if "bound" in kwargs:
            bound = kwargs['bound']

        # This formula is obtained by calculating the difference between two fibonacci sums for n and (n - b)
        max_n = bound * len(vals) - bound * (bound - 1) // 2
        if amount == -1:
            amount = max_n

        queries = []
        for lower in range(0, len(vals)):
            for upper in range(lower, min(lower + bound, len(vals))):
                query = (vals[lower], vals[upper])
                assert len(db.query(query)) != 0

                if allow_repetition or query not in queries:
                    queries.append(query)

        actual_query_idx = choice(len(queries), amount, replace=allow_repetition)

        return [[queries[x] for x in actual_query_idx]]


class MissingBoundedRangeQuerySpace(BoundedRangeQuerySpace):
    """
    Drops k queries which are not "two-way" or single values queries.
    """

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int, allow_repetition: bool = False, allow_empty: bool = False,
                    **kwargs) -> List[List[Tuple[int, int]]]:
        k = 1
        if 'k' in kwargs:
            k = kwargs['k']

        if allow_empty:
            raise ValueError("allow_empty is assumed to be false")

        bound = 3
        if "bound" in kwargs:
            bound = kwargs['bound']

        assert k < bound - 1

        queries = BoundedRangeQuerySpace.gen_queries(db, -1, allow_repetition=False, allow_empty=False, bound=bound)[0]

        '''Drop k volumes per window:'''
        dropped = set()
        for i in reversed(range(db.get_min(), db.get_max() - bound - 1)):

            q = [(i, i + num) for num in range(2, bound + 1) if (i, i + num) in queries and (i, i + num) not in dropped]
            '''How many more queries need to be dropped for this window:'''
            k_p = min(max(0, k - len(set([(i + j, i + num) for num in range(bound)
                                          for j in range(num + 1)]).intersection(dropped))),
                      k)
            drops = choice(len(q), min(k_p, len(q)), replace=False)
            for drop in drops:
                queries.remove(q[drop])
                dropped.add(q[drop])

        for i in range(db.get_min(), db.get_max() + 1):
            assert (i, i) in queries
            if i < db.get_max() - 1:
                assert (i, i + 1) in queries
                if i < db.get_max() - bound - 1:
                    assert len({(i + j, i + num) for num in range(bound) for j in range(num + 1)
                                if (i + j, i + num) in queries}) >= bound * (bound + 1) / 2 - k

        if amount == -1 or amount > len(queries) and not allow_repetition:
            amount = len(queries)

        actual_query_idx = choice(len(queries), amount, replace=allow_repetition)

        return [[queries[x] for x in actual_query_idx]]


class ZipfRangeQuerySpace(RangeQuerySpace):
    """Creates a Zipf-like distribution of queries. Can be restricted to only consider a fraction of possible
    queries with the restrict_frac parameter"""

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int, allow_repetition: bool = True, allow_empty: bool = True,
                    **kwargs) -> List[List[Tuple[int, int]]]:
        """
        :param db: The underlying range database.
        :param amount: The max number of queries to generate.
        :param allow_repetition: The same query may be chosen multiple times.
        :param allow_empty: A query with no replies may be chosen.
        :param kwargs:  s: Zipf exponent
                        width: Optional max width of generated queries
                        restrict_frac: Fraction of possible queries (N(N+1)/2) to be considered
        :return: The generated queries."""
        s = 5
        if 's' in kwargs:
            s = kwargs['s']

        queries = UniformRangeQuerySpace.gen_queries(db, -1, False, allow_empty)[0]

        big_n = db.get_max() - db.get_min() + 1
        max_n = big_n * (big_n + 1) // 2

        if 'width' in kwargs:
            bound = kwargs['width']
            queries = [q for q in queries if q[1] - q[0] <= bound]

        if 'restrict_frac' in kwargs:
            num_unique_queries = int(kwargs['restrict_frac'] * max_n)
            if len(queries) > num_unique_queries > 0:
                queries = queries[:num_unique_queries]
            if num_unique_queries < 0:
                log.warning(f"Provided restriction fraction yielded empty query space; restriction is discarded.")

        weights = np.array(list(default_rng().zipf(s, len(queries))), dtype='float')
        weights /= weights.sum()

        if amount == -1:
            amount = len(queries)

        actual_query_idx = choice(len(queries), amount, p=weights, replace=allow_repetition)

        return [[queries[x] for x in actual_query_idx]]


class PermutedBetaRangeQuerySpace(RangeQuerySpace):

    @classmethod
    def gen_queries(cls, db: RangeDatabase, amount: int, allow_repetition: bool = True,
                    allow_empty: bool = True, **kwargs) \
            -> List[List[Tuple[int, int]]]:
        """
        Generate permuted beta queries as per KPT21.

        :param db: The underlying range database.
        :param amount: The number of queries to generate.
        :param allow_repetition: The same query may be chosen multiple times.
        :param allow_empty: A query with no replies may be chosen.
        :param kwargs: alpha: Alpha parameter for the Beta distribution that describes the probability of the queries.
                        beta: Beta parameter of the Beta distribution that describes the probability of the queries.
        :return: The generated queries.
        """

        alpha = 1
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        beta = 3
        if "beta" in kwargs:
            beta = kwargs["beta"]

        queries = UniformRangeQuerySpace.gen_queries(db, -1, False, allow_empty)[0]

        weights = np.array(beta_intervals(alpha, beta, len(queries)), dtype='float')
        weights /= weights.sum()

        np.random.shuffle(weights)

        if amount == -1:
            amount = len(queries)

        actual_query_idx = choice(len(queries), amount, p=weights, replace=allow_repetition)

        return [[queries[x] for x in actual_query_idx]]
