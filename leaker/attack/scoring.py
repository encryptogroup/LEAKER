"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import collections
import random
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, Counter, TypeVar, Type

import numpy as np

from ..api import KeywordAttack, Dataset, LeakagePattern, Extension
from ..extension import CoOccurrenceExtension
from ..pattern import CoOccurrence

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class ScoringAttack(KeywordAttack):
    """
    Implements the basic Scoring attack from [DHP21]. If known_query_size == 0, they will be uncovered like in [CGPR15]
    """
    _known_keywords: List[str]
    _known_coocc: CoOccurrenceExtension
    _known_unique_rlens: Dict[int, str]
    _known_query_size: float

    def __init__(self, known: Dataset, known_query_size: float = 0.15):
        log.info(f"Setting up Scoring attack for {known.name()}. This might take some time.")
        super(ScoringAttack, self).__init__(known)

        self._known_keywords = list(known.keywords())
        self._known_query_size = known_query_size

        if not known.has_extension(CoOccurrenceExtension):
            known.extend_with(CoOccurrenceExtension)
        self._known_coocc = known.get_extension(CoOccurrenceExtension)

        _known_response_length: Dict[str, int] = dict()
        for keyword in known.keywords():
            _known_response_length[keyword] = self._known_coocc.co_occurrence(keyword, keyword)

        known_keyword_count: Counter[int] = collections.Counter(_known_response_length.values())

        self._known_unique_rlens = {_known_response_length[keyword]: keyword for keyword in known.keywords()
                                    if known_keyword_count[_known_response_length[keyword]] == 1}
        log.info("Setup complete.")

    @classmethod
    def name(cls) -> str:
        return "Scoring"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [CoOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {CoOccurrenceExtension}

    def _known_response_length(self, keyword: str) -> int:
        return self._known_coocc.selectivity(keyword)

    def __initialize_known_queries(self, queries: Iterable[str], rlens: List[int]) -> Dict[int, str]:
        return {i: self._known_unique_rlens[rlens[i]] for i, _ in enumerate(queries) if rlens[i]
                in self._known_unique_rlens}

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()}")
        queries = list(queries)
        coocc = self.required_leakage()[0](dataset, queries)
        rlens = [coocc[i][i] for i, _ in enumerate(queries)]

        if self._known_query_size == 0:
            known_queries = self.__initialize_known_queries(queries, rlens)
        else:
            known_query_ids = random.sample(range(len(queries)), int(self._known_query_size * len(list(queries))))
            known_queries = {i: queries[i] for i in known_query_ids}

        k = len(known_queries)
        known_queries_pos = {j: i for j, i in enumerate(known_queries.keys())}

        coocc_s_td = np.zeros((len(queries), k))
        for i in range(len(queries)):
            for j in range(k):
                coocc_s_td[i][j] = coocc[i][known_queries_pos[j]]

        coocc_s_kw = np.zeros((len(self._known_keywords), k))
        for i in range(len(self._known_keywords)):
            for j in range(k):
                coocc_s_kw[i][j] = self._known_coocc.co_occurrence(self._known_keywords[i],
                                                                   known_queries[known_queries_pos[j]])

        for i,  _ in enumerate(queries):
            if i not in known_queries:
                scores = coocc_s_kw - coocc_s_td[i].T
                scores = -np.log(np.linalg.norm(scores, axis=1))
                known_queries[i] = self._known_keywords[np.argmax(scores)]

        uncovered = []
        for i, _ in enumerate(queries):
            if i in known_queries:
                uncovered.append(known_queries[i])
            else:
                uncovered.append("")

        log.info(f"Reconstruction completed.")

        return uncovered