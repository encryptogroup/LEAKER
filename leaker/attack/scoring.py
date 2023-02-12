"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import random
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, TypeVar, Type

import numpy as np

from .count import Countv2
from .relational_estimators.estimator import NaruRelationalEstimator, SamplingRelationalEstimator, RelationalEstimator, \
    PerfectRelationalEstimator
from .relational_estimators.eval import ErrorMetric
from .relational_estimators.uae_estimator import UaeRelationalEstimator
from ..api import Dataset, LeakagePattern, Extension, RelationalQuery
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
        super(ScoringAttack, self).__init__(known)
        self._countv2 = None
        self._known_keywords = list(self._known_keywords)
        self._known_query_size = known_query_size
        # log.info("Setup complete.")

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


class RelationalScoring(Countv2):
    """
    Basic Scoring attack that can be used with estimators
    """
    _known_query_size: float
    _known_keywords: List[str]
    _est: RelationalEstimator
    _full_cooc_ext: CoOccurrenceExtension

    def __init__(self, known: Dataset, known_query_size: float = 0.15):
        super(RelationalScoring, self).__init__(known)
        self._countv2 = None
        self._known_keywords = list(self._known_keywords)
        self._known_query_size = known_query_size

        if isinstance(known, SampledSQLRelationalDatabase):
            full = known.parent()
        else:
            # dataset is not sampled, therefore whole dataset is known
            full = known

        if not full.has_extension(CoOccurrenceExtension):
            full.extend_with(CoOccurrenceExtension)

        self._full_cooc_ext = full.get_extension(CoOccurrenceExtension)
        self._est = self._get_estimator(known, full)

    @classmethod
    def name(cls) -> str:
        return "RelationalScoring"

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
            uncovered = super(RelationalScoring, self).recover(dataset, queries)
            known_queries = {i: kw for i, kw in enumerate(uncovered) if kw != ""}
        else:
            known_query_ids = random.sample(range(len(queries)), int(self._known_query_size * len(list(queries))))
            known_queries = {i: queries[i] for i in known_query_ids}

        return known_queries

    def _get_estimator(self, known, full):
        return SamplingRelationalEstimator(known, full)

    def _build_cooc_s_kw_matrix(self, estimator: RelationalEstimator, k: int, known_queries, known_queries_pos):
        errors = []
        errors_without_zero_one = []

        coocc_s_kw = np.zeros((len(self._known_keywords), k))
        for i in range(len(self._known_keywords)):
            for j in range(k):
                est_cooc = estimator.estimate(self._known_keywords[i], known_queries[known_queries_pos[j]])
                act_cooc = self._full_cooc_ext.co_occurrence(self._known_keywords[i],
                                                              known_queries[known_queries_pos[j]])
                coocc_s_kw[i][j] = est_cooc
                errors.append(ErrorMetric(est_cooc, act_cooc))
                if not (est_cooc == 0 and act_cooc == 1) and not (est_cooc == 1 and act_cooc == 0):
                    errors_without_zero_one.append(ErrorMetric(est_cooc, act_cooc))

        import csv
        with open('error_logs/' + self.name() + '.csv', 'a') as fd:
            row = [np.median(errors), np.quantile(errors, 0.95), np.quantile(errors, .99), np.max(errors),
                   np.median(errors_without_zero_one), np.quantile(errors_without_zero_one, 0.95),
                   np.quantile(errors_without_zero_one, .99), np.max(errors_without_zero_one)]
            writer = csv.writer(fd)
            writer.writerow(row)

        return coocc_s_kw

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

        coocc_s_kw = self._build_cooc_s_kw_matrix(self._est, k, known_queries, known_queries_pos)

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


class ErrorSimulationRelationalScoring(RelationalScoring):
    """
    Basic Scoring attack that can be used with estimators
    """
    __mean_error: float

    def __init__(self, known: Dataset, mean_error: float, known_query_size: float = 0.15):
        super(ErrorSimulationRelationalScoring, self).__init__(known, known_query_size)
        self.__mean_error = mean_error

    @classmethod
    def name(cls) -> str:
        return "ErrorSimulationRelationalScoring"

    def _build_cooc_s_kw_matrix(self, estimator: RelationalEstimator, k: int, known_queries, known_queries_pos):
        coocc_s_kw = np.zeros((len(self._known_keywords), k))
        for i in range(len(self._known_keywords)):
            for j in range(k):
                perfect_cooc = self._full_cooc_ext.co_occurrence(self._known_keywords[i],
                                                              known_queries[known_queries_pos[j]])
                if self.__mean_error != 1.0:
                    gaussian_error = np.random.normal(loc=self.__mean_error, scale=0.1, size=1)
                    if random.choice([0, 1]) == 0:  # try to simulate max inversion
                        simulated_cooc = perfect_cooc * gaussian_error
                    else:
                        simulated_cooc = perfect_cooc / gaussian_error
                    coocc_s_kw[i][j] = simulated_cooc
                else:
                    coocc_s_kw[i][j] = perfect_cooc

        return coocc_s_kw


class NaruRelationalScoring(RelationalScoring):
    """
    Scoring with Naru
    """

    def __init__(self, known: SQLRelationalDatabase, known_query_size: float = 0.15):
        super(NaruRelationalScoring, self).__init__(known, known_query_size)

    @classmethod
    def name(cls) -> str:
        return "NaruRelationalScoring"

    def _get_estimator(self, known, full):
        return NaruRelationalEstimator(known, full)


class PerfectRelationalScoring(RelationalScoring):
    """
    Implements the Scoring attack from [DHP21]. Using perfect co-occurrences.
    If known_query_size == 0, they will be uncovered like in [CGPR15]
    """

    def __init__(self, known: SQLRelationalDatabase, known_query_size: float = 0.15):
        super(PerfectRelationalScoring, self).__init__(known, known_query_size)

    def _get_estimator(self, known, full):
        return PerfectRelationalEstimator(known, full)

    @classmethod
    def name(cls) -> str:
        return "PerfectScoring"


class UaeScoringAttack(ScoringAttack):
    """
    Implements the Scoring attack from [DHP21]. Using UAE-Q co-occurrence estimates. If known_query_size == 0, they will be uncovered like in [CGPR15]
    """

    __est: UaeRelationalEstimator

    def __init__(self, known: SQLRelationalDatabase, known_query_size: float = 0.15):
        super(UaeScoringAttack, self).__init__(known, known_query_size)

    @classmethod
    def name(cls) -> str:
        return "UaeScoring"

    def __calculate_known_cooc(self, q1, q2) -> int:
        return round(self.__est.estimate(q1, q2))

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()}")
        queries = list(queries)
        coocc = self.required_leakage()[0](dataset, queries)
        cooc_ext = dataset.get_extension(CoOccurrenceExtension)

        known_queries = self._get_known_queries(dataset, queries)

        # train uae estimator based on known queries
        train_queries = [[q1, q2] for q1 in known_queries.values() for q2 in known_queries.values()]
        self.__est = UaeRelationalEstimator(self._known(), dataset, train_queries)

        k = len(known_queries)
        known_queries_pos = [i for i in known_queries.keys()]

        coocc_s_td = np.zeros((len(queries), k))
        for i in range(len(queries)):
            for j in range(k):
                coocc_s_td[i][j] = coocc[i][known_queries_pos[j]]

        coocc_s_kw = np.zeros((len(self._known_keywords), k))
        for i in range(len(self._known_keywords)):
            for j in range(k):
                est_sel = self.__calculate_known_cooc(self._known_keywords[i],
                                                      known_queries[known_queries_pos[j]])
                act_sel = cooc_ext.co_occurrence(self._known_keywords[i],
                                                 known_queries[known_queries_pos[j]])
                coocc_s_kw[i][j] = est_sel

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


class LowUaeScoringAttack(ScoringAttack):
    """
    Implements the Scoring attack from [DHP21]. Using uae co-occurrence estimates in the selectivity-range 0<x<=0.005.
    If known_query_size == 0, they will be uncovered like in [CGPR15]
    """

    __est: UaeRelationalEstimator
    __est_sampling: SamplingRelationalEstimator

    ''' Set estimation lower limit absolute (e.g. 1 to skip sampling in 0 case) and upper limit relative (e.g. 0.5% as 
    in naru paper). Upper absolute limit will then be calculated based on the number of rows in the full dataset. '''
    __estimation_lower_limit = 1
    __estimation_upper_limit_relative = 0.005
    __estimation_upper_limit: int

    def __init__(self, known: SQLRelationalDatabase, known_query_size: float = 0.15):
        self.__est = UaeRelationalEstimator(known, known.parent())
        self.__est_sampling = SamplingRelationalEstimator(known, known.parent())
        self.__estimation_upper_limit = round(len(known.parent().queries()) * self.__estimation_upper_limit_relative)
        super(LowUaeScoringAttack, self).__init__(known, known_query_size)

    @classmethod
    def name(cls) -> str:
        return "LowUaeScoring"

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


class RelationalRefinedScoring(RefinedScoringAttack):
    """
    Implements the refined Scoring attack from [DHP21]. Using sampling co-occurrence estimates.
    If known_query_size == 0, they will be uncovered like in [CGPR15]
    """

    _est: RelationalEstimator
    _full_cooc_ext: CoOccurrenceExtension

    def __init__(self, known: SQLRelationalDatabase, known_query_size: float = 0.15, ref_speed: int = 10):
        super(RelationalRefinedScoring, self).__init__(known, known_query_size, ref_speed)

        if isinstance(known, SampledSQLRelationalDatabase):
            full = known.parent()
        else:
            # dataset is not sampled, therefore whole dataset is known
            full = known

        self._est = self._get_estimator(known, full)
        self._full_cooc_ext = full.get_extension(CoOccurrenceExtension)

    @classmethod
    def name(cls) -> str:
        return "RelationalRefinedScoring"

    def _get_estimator(self, known, full):
        return SamplingRelationalEstimator(known, full)

    def _estimate_coocc(self, estimator: RelationalEstimator, q1: RelationalQuery, q2: RelationalQuery):
        # estimator needs to output estimates based on the full dataset, otherwise we need to scale here
        return estimator.estimate(q1, q2)

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
                coocc_s_kw[i][j] = self._estimate_coocc(self._est, self._known_keywords[i],
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
                        coocc_s_kw[i][j] = self._estimate_coocc(self._est, self._known_keywords[i],
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


class PerfectRelationalRefinedScoring(RelationalRefinedScoring):
    """
    Implements the refined Scoring attack from [DHP21]. Using perfect co-occurrence estimates.
    If known_query_size == 0, they will be uncovered like in [CGPR15]
    """

    def __init__(self, known: SQLRelationalDatabase, known_query_size: float = 0.15, ref_speed: int = 10):
        super(PerfectRelationalRefinedScoring, self).__init__(known, known_query_size, ref_speed)

    @classmethod
    def name(cls) -> str:
        return "PerfectRelationalRefinedScoring"

    def _get_estimator(self, known, full):
        return PerfectRelationalEstimator(known, full)


class ErrorSimulationRelationalRefinedScoring(PerfectRelationalRefinedScoring):
    """
    Basic Scoring attack that can be used with estimators
    """
    __mean_error: float

    def __init__(self, known: SQLRelationalDatabase, mean_error: float, known_query_size: float = 0.15, ref_speed: int = 10):
        super(ErrorSimulationRelationalRefinedScoring, self).__init__(known, known_query_size, ref_speed)
        self.__mean_error = mean_error

    @classmethod
    def name(cls) -> str:
        return "ErrorSimulationRelationalRefinedScoring"

    def _estimate_coocc(self, estimator: RelationalEstimator, q1: RelationalQuery, q2: RelationalQuery):
        perfect_cooc = self._full_cooc_ext.co_occurrence(q1, q2)
        if self.__mean_error != 1.0:
            gaussian_error = np.random.normal(loc=self.__mean_error, scale=0.1, size=None)
            if random.choice([0, 1]) == 0:  # try to simulate max inversion
                simulated_cooc = perfect_cooc * gaussian_error
            else:
                simulated_cooc = perfect_cooc / gaussian_error
            return simulated_cooc
        else:
            return perfect_cooc
