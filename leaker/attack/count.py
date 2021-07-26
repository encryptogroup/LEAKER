"""
For License information see the LICENSE file.

Authors: Tobias Stöckert, Amos Treiber

"""
import collections
import math
from itertools import starmap
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, Counter, Tuple, TypeVar, Type, Optional

from ..api import KeywordAttack, Dataset, LeakagePattern, Extension, RelationalDatabase, RelationalQuery, \
    RelationalKeyword
from ..extension import CoOccurrenceExtension, PandasExtension
from ..pattern import CoOccurrence
from ..sql_interface import SQLRelationalDatabase
from .relational_estimators.estimator import NaruRelationalEstimator

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class BasicCount(KeywordAttack):
    """
    Implements the basic Count attack for full DB knowledge from [CGPR15]. It uses the CoOccurrence and the
    ResponseLength patterns.
    """
    _known_keywords: Set[str]
    _known_coocc: CoOccurrenceExtension
    _known_unique_rlens: Dict[int, str]

    def __init__(self, known: Dataset):
        log.info(f"Setting up {self.name()} attack for {known.name()}. This might take some time.")
        super(BasicCount, self).__init__(known)

        self._known_keywords = known.keywords()

        if not known.has_extension(CoOccurrenceExtension):
            known.extend_with(CoOccurrenceExtension)
        self._known_coocc = known.get_extension(CoOccurrenceExtension)

        _known_response_length: Dict[str, int] = dict()
        for keyword in known.keywords():
            _known_response_length[keyword] = self._known_coocc.selectivity(keyword)

        known_keyword_count: Counter[int] = collections.Counter(_known_response_length.values())

        self._known_unique_rlens = {_known_response_length[keyword]: keyword for keyword in known.keywords()
                                    if known_keyword_count[_known_response_length[keyword]] == 1}
        log.info("Setup complete.")

    @classmethod
    def name(cls) -> str:
        return "BasicCount"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [CoOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {CoOccurrenceExtension}

    def _known_response_length(self, keyword: str) -> int:
        return self._known_coocc.co_occurrence(keyword, keyword)

    def __initialize_known_queries(self, queries: Iterable[str], rlens: List[int]) -> Dict[int, str]:
        return {i: self._known_unique_rlens[rlens[i]] for i, _ in enumerate(queries) if rlens[i]
                in self._known_unique_rlens}

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info("Running BasicCount")
        coocc = self.required_leakage()[0](dataset, queries)
        rlens = [coocc[i][i] for i, _ in enumerate(queries)]

        known_queries = self.__initialize_known_queries(queries, rlens)

        while True:
            unknown_queries = [i for i, _ in enumerate(queries) if i not in known_queries]
            old_size = len(known_queries)
            for i in unknown_queries:
                candidate_keywords = [k for k in self._known_keywords if k not in known_queries.values() and
                                      rlens[i] == self._known_response_length(k)]
                for s in candidate_keywords[:]:
                    for j, k in known_queries.items():
                        if coocc[i][j] != self._known_coocc.co_occurrence(s, k):
                            candidate_keywords.remove(s)
                            break
                if len(candidate_keywords) == 1:
                    known_queries[i] = candidate_keywords[0]
            if old_size >= len(known_queries):
                break

        uncovered = []
        for i, _ in enumerate(queries):
            if i in known_queries:
                uncovered.append(known_queries[i])
            else:
                uncovered.append("")

        log.info(f"Reconstruction completed.")

        return uncovered


class Countv2(BasicCount):
    """
    Implements the Count attack from [CGPR15]. It uses the CoOccurrence and the ResponseLength patterns.
    """

    _delta: float
    _n: int

    def __init__(self, known: Dataset):
        super(Countv2, self).__init__(known)

        self._delta = known.sample_rate()
        self._n = len(known.doc_ids())

    @classmethod
    def name(cls) -> str:
        return "Countv2"

    def _calculate_interval(self, c_ks: int) -> Tuple[float, float]:
        epsilon = math.sqrt(0.5 * (self._n - self._delta * self._n) * math.log2(40))
        lbk = c_ks / self._delta - epsilon
        ubk = c_ks / self._delta + epsilon
        return lbk, ubk

    def __calculate_candidates(self, queries: Iterable[str], rlens: List[int]) -> List[Set[str]]:
        return [set([w for w, (lbk, ubk) in
                     zip(self._known_keywords, map(self._calculate_interval,
                                                   map(self._known_response_length, self._known_keywords)))
                     if lbk <= rlens[i] <= ubk]) for i, _ in enumerate(queries)]

    def __fallback(self, query_id: int, known_queries: Dict[int, str], query_candidates: Dict[int, Set[str]]):
        """ Fall-back strategy if we have multiple candidates: Use keyword with highest selectivity"""
        candidate_keywords = [(w, self._known_response_length(w))
                              for w in query_candidates[query_id] if w not in known_queries.values()]
        return max(candidate_keywords, key=lambda i: i[1])[0]

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running Countv2 on {dataset.name()} for {self._delta:.3f}")
        coocc = self.required_leakage()[0](dataset, queries)
        rlens = [coocc[i][i] for i, _ in enumerate(queries)]

        query_candidates = self.__calculate_candidates(queries, rlens)
        known_queries: Dict[int, str] = {i: next(iter(query_candidates[i])) for i, _ in enumerate(queries)
                                         if len(query_candidates[i]) == 1}

        while True:
            unknown_queries = [i for i, _ in enumerate(queries) if i not in known_queries]
            old_size = len(known_queries)
            for i in unknown_queries:
                candidate_keywords = [w for w in query_candidates[i] if w not in known_queries.values()]
                for s in candidate_keywords[:]:
                    for j, k in known_queries.items():
                        lbk, ubk = self._calculate_interval(self._known_coocc.co_occurrence(s, k))
                        if not (lbk <= coocc[i][j] <= ubk):
                            candidate_keywords.remove(s)
                            break
                if len(candidate_keywords) == 1:
                    known_queries[i] = candidate_keywords[0]
            if old_size >= len(known_queries):
                break

        uncovered = []
        for i, _ in enumerate(queries):
            if i in known_queries:
                uncovered.append(known_queries[i])
            # elif len([w for w in query_candidates[i] if w not in known_queries.values()]) != 0:
            #   """We have multiple candidates => use fallback. Disabled for now."""
            #   uncovered.append(self.__fallback(i, known_queries, query_candidates))
            else:
                uncovered.append("")

        log.info(f"Reconstruction completed.")

        return uncovered


class EstimatorCount(Countv2):
    """
    Implements the Count attack with confidence intervals of different estimators. It uses the CoOccurrence and the
    ResponseLength patterns.
    """

    _table_n: Dict[int, int]
    _additional_methods: bool
    _query_candidates: List[Set[RelationalKeyword]]

    def __init__(self, known: RelationalDatabase):
        super(EstimatorCount, self).__init__(known)
        self._additional_methods = False
        self._table_n = dict()
        for t, ids in known._table_row_ids.items():
            self._table_n[t] = len(ids)

        self._query_candidates = []

    @classmethod
    def name(cls) -> str:
        return "Estimator-Count"

    def _calculate_interval(self, x: RelationalKeyword, n: int, x2: Optional[RelationalKeyword] = None,
                            m: int = 1) -> Tuple[float, float]:
        return super()._calculate_interval(self._known_response_length(x))

    def _calculate_candidates(self, queries: Iterable[RelationalQuery], rlens: List[float]) -> None:
        for i, q in enumerate(queries):
            self._query_candidates.append(set([w for w, (lbk, ubk) in
                                               zip(self._known_keywords, starmap(self._calculate_interval,
                                                                                 map(lambda x:
                                                                                     (x, self._table_n[x.table]),
                                                                                     self._known_keywords)))
                                               if lbk <= rlens[i] <= ubk and w.table == q.table]))

    def _calculate_known_queries(self, queries: Iterable[RelationalQuery], rlens: List[int],
                                 query_tables_length: List[int]) -> Dict[int, RelationalKeyword]:
        self._calculate_candidates(queries, [rlens[i] / query_tables_length[i] for i in range(len(rlens))])
        return {i: next(iter(self._query_candidates[i])) for i, _ in enumerate(queries)
                if len(self._query_candidates[i]) == 1}

    def recover(self, dataset: Dataset, queries: Iterable[RelationalQuery]) -> List[str]:
        log.info(f"Running {self.name()} on {dataset.name()} for {self._delta:.3f}")
        coocc = self.required_leakage()[0](dataset, queries)
        rlens = [coocc[i][i] for i, _ in enumerate(queries)]

        query_tables = [q.table for q in queries]  # we assume which table and their full length is leaked
        query_tables_length = [len(dataset._table_row_ids[t]) for t in query_tables]

        known_queries = self._calculate_known_queries(queries, rlens, query_tables_length)

        log.info(f"{self.name()}: Known queries {len(known_queries)}")
        c = 0
        while True:
            unknown_queries = [i for i, _ in enumerate(queries) if i not in known_queries]
            old_size = len(known_queries)
            for i in unknown_queries:
                candidate_keywords = [w for w in self._query_candidates[i] if w not in known_queries.values()]
                for s in candidate_keywords[:]:
                    for j, k in known_queries.items():
                        lbk, ubk = self._calculate_interval(s, n=self._table_n[s.table], x2=k, m=2)
                        if not (lbk <= coocc[i][j] / query_tables_length[i] <= ubk):
                            candidate_keywords.remove(s)
                            break

                if self._additional_methods:
                    for j, k in known_queries.items():
                        for s in candidate_keywords[:]:
                            if coocc[i][j] > 0 and s.attr == k.attr:
                                candidate_keywords.remove(s)

                if len(candidate_keywords) == 1:
                    known_queries[i] = candidate_keywords[0]
            if self.name() == "Naru-Count":
                log.info(f"Completed {c}")
            c += 1
            if old_size >= len(known_queries):
                break

        uncovered = []
        for i, _ in enumerate(queries):
            if i in known_queries:
                uncovered.append(known_queries[i])
            # elif len([w for w in query_candidates[i] if w not in known_queries.values()]) != 0:
            #   """We have multiple candidates => use fallback. Disabled for now."""
            #   uncovered.append(self.__fallback(i, known_queries, query_candidates))
            else:
                uncovered.append("")

        log.info(f"Reconstruction completed.")

        return uncovered


class RiondatoCount(EstimatorCount):

    @classmethod
    def name(cls) -> str:
        return "Riondato-Count"

    def _calculate_interval(self, x: RelationalKeyword, n: int, x2: Optional[RelationalKeyword] = None,
                            m: int = 1) -> Tuple[float, float]:
        if m == 1:
            c_ks = self._known_response_length(x)
            d = 2
        else:
            c_ks = self._known_coocc.co_occurrence(x, x2)
            d = 31

        epsilon = math.sqrt(0.5*d + 0.0256466) / math.sqrt(n)

        lbk = c_ks / n - epsilon
        ubk = c_ks / n + epsilon
        return lbk, ubk


class NaruCount(EstimatorCount):
    __est: NaruRelationalEstimator

    def __init__(self, known: SQLRelationalDatabase):
        super().__init__(known)
        full = known.parent()
        self.__est = NaruRelationalEstimator(sample=known, full=full)

        self._table_n = dict()
        for t, ids in full._table_row_ids.items():
            self._table_n[t] = len(ids)

    @classmethod
    def name(cls) -> str:
        return "Naru-Count"

    def _calculate_naru_interval(self, x: RelationalKeyword, n: int, x2: Optional[RelationalKeyword] = None,
                            m: int = 1) -> Tuple[float, float]:

        epsilon = 0.005
        c_ks = self.__est.estimate(x, x2)

        lbk = c_ks / self._table_n[x.table] - epsilon
        ubk = c_ks / self._table_n[x.table] + epsilon
        return lbk, ubk

    def _calculate_candidates(self, queries: Iterable[RelationalQuery], rlens: List[float]) -> None:
        """We first get a rough candidate set with Hoeffding, then refine it with Naru to prevent too many calls
        to the estimator"""
        super()._calculate_candidates(queries, rlens)
        for i, q in enumerate(queries):
            orig = len(self._query_candidates[i])
            contained = q in self._query_candidates[i]
            log.info(f"Calculating candidates for {i} with {orig} original {contained}")
            self._query_candidates[i] = set([w for w, (lbk, ubk) in
                                             zip(self._known_keywords, starmap(self._calculate_naru_interval,
                                                                               map(lambda x:
                                                                                   (x, self._table_n[x.table]),
                                                                                   self._known_keywords)))
                                             if lbk <= rlens[i] <= ubk and w.table == q.table])
            log.info(f"Calculated candidates for {i} with {orig} original_{contained} - "
                     f"now {len(self._query_candidates[i])}_{q in self._query_candidates[i]}")

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {PandasExtension}


class AddEstimatorCount(KeywordAttack):
    """Uses additional observations about relational databases"""
    _attck: EstimatorCount

    def __init__(self, known: RelationalDatabase, attck: Type[EstimatorCount]):
        super().__init__(known)
        self._attck = attck(known)
        self._attck._additional_methods = True

    def name(self) -> str:
        return f"Add{self._attck.name()}"
