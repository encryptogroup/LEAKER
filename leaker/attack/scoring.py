"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import random
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, TypeVar, Type

import numpy as np

from .count import Countv2
from .relational_estimators.estimator import NaruRelationalEstimator, SamplingRelationalEstimator
from ..api import Dataset, LeakagePattern, Extension, RelationalDatabase, RelationalQuery
from ..extension import CoOccurrenceExtension
from ..pattern import CoOccurrence
from ..sql_interface import SQLRelationalDatabase
from ..sql_interface.sql_database import SampledSQLRelationalDatabase

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class ScoringAttack(Countv2):
    """
    Implements the basic Scoring attack from [DHP21]. If known_query_size == 0, they will be uncovered like in [CGPR15]
    """
    _known_query_size: float
    _known_keywords: List[str]

    def __init__(self, known: Dataset, known_query_size: float = 0.15):
        log.info(f"Setting up {self.name()} attack for {known.name()}. This might take some time.")
        super(ScoringAttack, self).__init__(known)
        self._countv2 = None
        self._known_keywords = list(self._known_keywords)
        self._known_query_size = known_query_size
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

    def _get_known_queries(self, dataset: Dataset, queries: List[str]) -> Dict[int, str]:
        if self._known_query_size == 0:
            uncovered = super(ScoringAttack, self).recover(dataset, queries)
            known_queries = {i: kw for i, kw in enumerate(uncovered) if kw != ""}
        else:
            known_query_ids = random.sample(range(len(queries)), int(self._known_query_size * len(list(queries))))
            known_queries = {i: queries[i] for i in known_query_ids}

        return known_queries

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()}")
        queries = list(queries)
        coocc = self.required_leakage()[0](dataset, queries)

        known_queries = self._get_known_queries(dataset, queries)

        k = len(known_queries)
        known_queries_pos = [i for i in known_queries.keys()]

        coocc_s_td = np.zeros((len(queries), k))
        for i in range(len(queries)):
            for j in range(k):
                coocc_s_td[i][j] = coocc[i][known_queries_pos[j]]

        coocc_s_kw = np.zeros((len(self._known_keywords), k))
        for i in range(len(self._known_keywords)):
            for j in range(k):
                coocc_s_kw[i][j] = self._known_coocc.co_occurrence(self._known_keywords[i],
                                                                   known_queries[known_queries_pos[j]])

        for i, _ in enumerate(queries):
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


class NaruScoringAttack(ScoringAttack):
    """
    Implements the Scoring attack from [DHP21]. Using naru co-occurrence estimates. If known_query_size == 0, they will be uncovered like in [CGPR15]
    """

    __est: NaruRelationalEstimator
    __est_sampling: SamplingRelationalEstimator

    ''' Set estimation lower limit absolute (e.g. 1 to skip sampling in 0 case) and upper limit relative (e.g. 0.5% as 
    in naru paper). Upper absolute limit will then be calculated based on the number of rows in the full dataset. '''
    __estimation_lower_limit = 1
    __estimation_upper_limit_relative = 0.005
    __estimation_upper_limit: int

    def __init__(self, known: SQLRelationalDatabase, known_query_size: float = 0.15):
        self.__est = NaruRelationalEstimator(known, known.parent())
        self.__est_sampling = SamplingRelationalEstimator(known, known.parent())
        self.__estimation_upper_limit = round(len(known.parent().queries()) * self.__estimation_upper_limit_relative)
        super(NaruScoringAttack, self).__init__(known, known_query_size)

    @classmethod
    def name(cls) -> str:
        return "NaruScoring"

    def __calculate_known_cooc(self, q1, q2) -> int:
        sampled_cooc = round(self.__est_sampling.estimate(q1, q2))
        if self.__estimation_lower_limit <= sampled_cooc <= self.__estimation_upper_limit:
            return round(self.__est.estimate(q1, q2))
        else:
            # use sampling
            return sampled_cooc

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()}")
        queries = list(queries)
        coocc = self.required_leakage()[0](dataset, queries)

        known_queries = self._get_known_queries(dataset, queries)

        k = len(known_queries)
        known_queries_pos = [i for i in known_queries.keys()]

        coocc_s_td = np.zeros((len(queries), k))
        for i in range(len(queries)):
            for j in range(k):
                coocc_s_td[i][j] = coocc[i][known_queries_pos[j]]

        coocc_s_kw = np.zeros((len(self._known_keywords), k))
        for i in range(len(self._known_keywords)):
            for j in range(k):
                coocc_s_kw[i][j] = self.__calculate_known_cooc(self._known_keywords[i],
                                                               known_queries[known_queries_pos[j]])

        for i, _ in enumerate(queries):
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


class RefinedScoringAttack(ScoringAttack):
    """
    Implements the refined Scoring attack from [DHP21]. If known_query_size == 0, they will be uncovered like in
    [CGPR15]
    """
    _ref_speed: int

    def __init__(self, known: Dataset, known_query_size: float = 0.15, ref_speed: int = 10):
        super(RefinedScoringAttack, self).__init__(known, known_query_size)
        self._ref_speed = ref_speed

    @classmethod
    def name(cls) -> str:
        return "RefinedScoring"

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()}")
        queries = list(queries)
        coocc = self.required_leakage()[0](dataset, queries)

        known_queries = self._get_known_queries(dataset, queries)

        k = len(known_queries)
        known_queries_pos = [i for i in known_queries.keys()]

        coocc_s_td = np.zeros((len(queries), k))
        for i in range(len(queries)):
            for j in range(k):
                coocc_s_td[i][j] = coocc[i][known_queries_pos[j]]

        coocc_s_kw = np.zeros((len(self._known_keywords), k))
        for i in range(len(self._known_keywords)):
            for j in range(k):
                coocc_s_kw[i][j] = self._known_coocc.co_occurrence(self._known_keywords[i],
                                                                   known_queries[known_queries_pos[j]])

        unknown_queries = [i for i, _ in enumerate(queries) if i not in known_queries]
        while len(unknown_queries) > 0:
            temp_pred = []  # %1 in Algo

            """%2 in Algo"""
            for i in unknown_queries:
                scores = coocc_s_kw - coocc_s_td[i].T
                scores = -np.log(np.linalg.norm(scores, axis=1))
                max_indices = np.argpartition(scores, -2)[-2:]  # top 2 argmax but result is unsorted
                cand_0 = max_indices[0] if scores[max_indices[0]] > scores[max_indices[1]] else max_indices[1]
                certainty = max(scores[max_indices[0]], scores[max_indices[1]]) \
                            - min(scores[max_indices[0]], scores[max_indices[1]])

                temp_pred.append((i, cand_0, certainty))

            """%3 in Algo"""
            if len(unknown_queries) < self._ref_speed:
                for i, kw, _ in temp_pred:
                    known_queries[i] = kw
                    unknown_queries = []
            else:
                """Enlarge matrices first"""
                old_coocc_s_td = coocc_s_td
                coocc_s_td = np.zeros((old_coocc_s_td.shape[0], old_coocc_s_td.shape[1] + self._ref_speed))
                coocc_s_td[:, :-self._ref_speed] = old_coocc_s_td

                old_coocc_s_kw = coocc_s_kw
                coocc_s_kw = np.zeros((old_coocc_s_kw.shape[0], old_coocc_s_kw.shape[1] + self._ref_speed))
                coocc_s_kw[:, :-self._ref_speed] = old_coocc_s_kw

                """Sort with certainties"""
                certainties = np.array([certainty for _, _, certainty in temp_pred])
                max_indices = np.argpartition(certainties, -self._ref_speed)[-self._ref_speed:]  # top ref_speed argmax
                for l in max_indices:
                    j = len(known_queries_pos)  # in len(known_queries)
                    j_q = temp_pred[l][0]  # in len(queries)
                    known_queries_pos.append(j_q)
                    j_kw = temp_pred[l][1]
                    known_queries[j_q] = self._known_keywords[j_kw]
                    """Add new columns"""
                    for i in range(len(queries)):
                        coocc_s_td[i][j] = coocc[i][j_q]
                    for i in range(len(self._known_keywords)):
                        coocc_s_kw[i][j] = self._known_coocc.co_occurrence(self._known_keywords[i],
                                                                           self._known_keywords[j_kw])

                unknown_queries = [i for i, _ in enumerate(queries) if i not in known_queries]

        uncovered = []
        for i, _ in enumerate(queries):
            if i in known_queries:
                uncovered.append(known_queries[i])
            else:
                uncovered.append("")

        log.info(f"Reconstruction completed.")

        return uncovered


class NaruRefinedScoringAttack(RefinedScoringAttack):
    """
    Implements the refined Scoring attack from [DHP21]. Using naru co-occurrence estimates.
    If known_query_size == 0, they will be uncovered like in [CGPR15]
    """
    __est: NaruRelationalEstimator
    __est_sampling: SamplingRelationalEstimator

    ''' Set estimation lower limit absolute (e.g. 1 to skip sampling in 0 case) and upper limit relative (e.g. 0.5% as 
    in naru paper). Upper absolute limit will then be calculated based on the number of rows in the full dataset. '''
    __estimation_lower_limit = 1
    __estimation_upper_limit_relative = 0.005
    __estimation_upper_limit: int

    def __init__(self, known: SampledSQLRelationalDatabase, known_query_size: float = 0.15, ref_speed: int = 10):
        self.__est = NaruRelationalEstimator(known, known.parent())
        self.__est_sampling = SamplingRelationalEstimator(known, known.parent())
        self.__estimation_upper_limit = round(len(known.parent().queries()) * self.__estimation_upper_limit_relative)
        super(NaruRefinedScoringAttack, self).__init__(known, known_query_size, ref_speed)
        self._ref_speed = ref_speed

    @classmethod
    def name(cls) -> str:
        return "NaruRefinedScoring"

    def __calculate_known_cooc(self, q1, q2):
        sampled_cooc = round(self.__est_sampling.estimate(q1, q2))
        if self.__estimation_lower_limit <= sampled_cooc <= self.__estimation_upper_limit:
            return round(self.__est.estimate(q1, q2))
        else:
            # use sampling
            return sampled_cooc

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()}")
        queries = list(queries)
        coocc = self.required_leakage()[0](dataset, queries)

        known_queries = self._get_known_queries(dataset, queries)

        k = len(known_queries)
        known_queries_pos = [i for i in known_queries.keys()]

        coocc_s_td = np.zeros((len(queries), k))
        for i in range(len(queries)):
            for j in range(k):
                coocc_s_td[i][j] = coocc[i][known_queries_pos[j]]

        coocc_s_kw = np.zeros((len(self._known_keywords), k))
        for i in range(len(self._known_keywords)):
            for j in range(k):
                coocc_s_kw[i][j] = self.__calculate_known_cooc(self._known_keywords[i],
                                                               known_queries[known_queries_pos[j]])

        unknown_queries = [i for i, _ in enumerate(queries) if i not in known_queries]
        while len(unknown_queries) > 0:
            temp_pred = []  # %1 in Algo

            """%2 in Algo"""
            for i in unknown_queries:
                scores = coocc_s_kw - coocc_s_td[i].T
                scores = -np.log(np.linalg.norm(scores, axis=1))
                max_indices = np.argpartition(scores, -2)[-2:]  # top 2 argmax but result is unsorted
                cand_0 = max_indices[0] if scores[max_indices[0]] > scores[max_indices[1]] else max_indices[1]
                certainty = max(scores[max_indices[0]], scores[max_indices[1]]) \
                            - min(scores[max_indices[0]], scores[max_indices[1]])

                temp_pred.append((i, cand_0, certainty))

            """%3 in Algo"""
            if len(unknown_queries) < self._ref_speed:
                for i, kw, _ in temp_pred:
                    known_queries[i] = kw
                    unknown_queries = []
            else:
                """Enlarge matrices first"""
                old_coocc_s_td = coocc_s_td
                coocc_s_td = np.zeros((old_coocc_s_td.shape[0], old_coocc_s_td.shape[1] + self._ref_speed))
                coocc_s_td[:, :-self._ref_speed] = old_coocc_s_td

                old_coocc_s_kw = coocc_s_kw
                coocc_s_kw = np.zeros((old_coocc_s_kw.shape[0], old_coocc_s_kw.shape[1] + self._ref_speed))
                coocc_s_kw[:, :-self._ref_speed] = old_coocc_s_kw

                """Sort with certainties"""
                certainties = np.array([certainty for _, _, certainty in temp_pred])
                max_indices = np.argpartition(certainties, -self._ref_speed)[-self._ref_speed:]  # top ref_speed argmax
                for l in max_indices:
                    j = len(known_queries_pos)  # in len(known_queries)
                    j_q = temp_pred[l][0]  # in len(queries)
                    known_queries_pos.append(j_q)
                    j_kw = temp_pred[l][1]
                    known_queries[j_q] = self._known_keywords[j_kw]
                    """Add new columns"""
                    for i in range(len(queries)):
                        coocc_s_td[i][j] = coocc[i][j_q]
                    for i in range(len(self._known_keywords)):
                        coocc_s_kw[i][j] = self.__calculate_known_cooc(self._known_keywords[i],
                                                                       self._known_keywords[j_kw])

                unknown_queries = [i for i, _ in enumerate(queries) if i not in known_queries]

        uncovered = []
        for i, _ in enumerate(queries):
            if i in known_queries:
                uncovered.append(known_queries[i])
            else:
                uncovered.append("")

        log.info(f"Reconstruction completed.")

        return uncovered
