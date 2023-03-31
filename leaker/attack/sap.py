"""Some Code in this file has been adapted from https://github.com/simon-oya/USENIX21-sap-code"""
import random
from calendar import week
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, TypeVar, Type, Optional
from typing import Iterable, List, Any, Dict, Set, TypeVar, Type, Union

import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment as hungarian

from leaker.extension import identity
from .relational_estimators.estimator import NaruRelationalEstimator, SamplingRelationalEstimator, RelationalEstimator, \
    KDERelationalEstimator, PerfectRelationalEstimator
from ..api import Extension, KeywordAttack, Dataset, LeakagePattern, RelationalDatabase, RelationalQuery, KeywordQueryLog
from ..extension import VolumeExtension, SelectivityExtension, IdentityExtension
from ..pattern import ResponseLength, TotalVolume, Frequency, ResponseIdentity
from ..sql_interface.sql_database import SampledSQLRelationalDatabase, SQLRelationalDatabase

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


def _log_binomial(n, beta):
    """Computes an approximation of log(binom(n, n*alpha)) for alpha < 1"""
    if beta == 0 or beta == 1:
        return 0
    elif beta < 0 or beta > 1:
        raise ValueError("beta cannot be negative or greater than 1 ({})".format(beta))
    else:
        entropy = -beta * np.log(beta) - (1 - beta) * np.log(1 - beta)
        return n * entropy - 0.5 * np.log(2 * np.pi * n * beta * (1 - beta))


def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities|x|observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    probabilities[probabilities == 0] = min(probabilities[
                                                probabilities > 0]) / 100  # To avoid numerical errors. An error would mean the adversary information is very off.
    probabilities[probabilities == 1] = max(probabilities[
                                                probabilities <1])  # To avoid numerical errors. An error would mean the adversary information is very off.
    log_binom_term = np.array([_log_binomial(ntrials, obs / ntrials) for obs in observations])  # ROW TERM
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = log_binom_term + np.array(observations) * column_term + last_term
    return log_matrix

def compute_frequencies(ql:KeywordQueryLog):
    kw_lst = ql.keywords_list()
    kw_counter = Counter(kw_lst).most_common()
    chosen_keywords = [kw for kw, _ in kw_counter]
    frequencies = np.array([count for _, count in kw_counter])[:,None]/sum(count for _, count in kw_counter)
    return frequencies, chosen_keywords


class Sap(KeywordAttack):
    """
    Implements the SAP attack from Oya & Kerschbaum.
    """
    _known_volume: Dict[str, int]
    _known_response_length: Dict[str, int]
    _known_keywords: Dict
    _delta: float
    _known_f_matrix: np.ndarray
    _chosen_keywords: List[str]
    _alpha: float

    def __init__(self, known: Dataset, known_frequencies: np.ndarray = None, chosen_keywords: List[str] = None, query_log: KeywordQueryLog = None, alpha:float = 0.0):
        super(Sap, self).__init__(known)

        self._delta = known.sample_rate()

        self._known_volume = dict()
        self._known_response_length = dict()
        self._known_keywords = dict()

        if not known.has_extension(VolumeExtension):
            known.extend_with(VolumeExtension)

        vol = known.get_extension(VolumeExtension)

        if known_frequencies is not None:
            assert chosen_keywords is not None, log.error("Auxiliary frequency knowledge and correstponding keywords needed for SAP frequency attack.")
            self._known_f_matrix = known_frequencies
            self._chosen_keywords = chosen_keywords
        elif query_log is not None:
            known_frequencies, chosen_keywords = compute_frequencies(query_log)
            self._known_f_matrix = known_frequencies
            self._chosen_keywords = chosen_keywords
        else:
            assert alpha == 0, log.error("Auxiliary frequency knowledge and correstponding keywords needed for SAP frequency attack.")
            if chosen_keywords is not None:
                self._chosen_keywords = chosen_keywords
            else:
                self._chosen_keywords = list(known.keywords())
        self._alpha = alpha

        i = 0
        for keyword in self._chosen_keywords:
            self._known_keywords[i] = keyword
            self._known_keywords[keyword] = i
            i += 1
            self._known_volume[keyword] = vol.total_volume(keyword)
            self._known_response_length[keyword] = vol.selectivity(keyword)



    @classmethod
    def name(cls) -> str:
        return "SAP"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [ResponseLength(), Frequency()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {VolumeExtension}

    def _build_cost_tvol(self, n: int, tvols: List[int]):
        vol = self._known().get_extension(VolumeExtension)
        kw_probs_train = [self._known_volume[self._known_keywords[i]] / vol.dataset_volume()
                          for i in range(len(self._known().keywords()))]
        log_prob_matrix = compute_log_binomial_probability_matrix(n, kw_probs_train,
                                                                  tvols)
        cost_vol = - log_prob_matrix
        return cost_vol

    def _build_cost_rlen(self, n: int, tag_info):
        kw_probs_train = [self._known_response_length[self._known_keywords[i]] / len(self._known().doc_ids())
                          for i in range(len(self._chosen_keywords))]
        kw_counts_test = [len(tag_info[tag]) for tag in tag_info]

        log_prob_matrix = compute_log_binomial_probability_matrix(n, kw_probs_train, kw_counts_test)
        cost_vol = - log_prob_matrix
        return cost_vol

    def _split_traces(self, queries:List[str],n_queries_per_week: int=5):
        weekly_queries = []
        for query_start_idx in range(0,len(queries),n_queries_per_week):
            weekly_queries.append(queries[query_start_idx:query_start_idx+n_queries_per_week])
        return weekly_queries

    def _process_traces(self,rid):
        tag_traces = []
        seen_tuples = {}
        tag_info = {}
        count = 0
        traces = self._split_traces(rid)
        for week in traces:
            weekly_tags = []
            for trace in week:
                obs_sorted = tuple(sorted(trace))
                if obs_sorted not in seen_tuples:
                    seen_tuples[obs_sorted] = count
                    tag_info[count] = obs_sorted
                    count += 1
                weekly_tags.append(seen_tuples[obs_sorted])
            tag_traces.append(weekly_tags)
        return tag_traces, tag_info

    def _build_trend_matrix(self,traces, n_tags):
        n_weeks = len(traces)
        tag_trend_matrix = np.zeros((n_tags, n_weeks))
        for i_week, weekly_tags in enumerate(traces):
            if len(weekly_tags) > 0:
                counter = Counter(weekly_tags)
                for key in counter:
                    tag_trend_matrix[key, i_week] = counter[key] / len(weekly_tags)
        return tag_trend_matrix

    def _build_cost_freq(self, trends, nq_per_week):
        log_c_matrix = np.zeros((len(self._known_f_matrix), len(trends)))
        for i_week, nq in enumerate(nq_per_week):
            probabilities = self._known_f_matrix[:, i_week].copy()
            probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100
            log_c_matrix += (nq * trends[:, i_week]) * np.log(np.array([probabilities]).T)
        return -log_c_matrix

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()} at {self._delta:.3f}")
        queries = list(queries)
        n_docs_test = len(dataset.doc_ids())
        rid = ResponseIdentity()
        tag_traces,tag_info = self._process_traces(rid(dataset, queries))
        tag_trends = self._build_trend_matrix(tag_traces,len(tag_info))
        nq_per_week = [len(trace) for trace in tag_traces]

        if self._alpha == 0:
            total_cost = self._build_cost_rlen(n_docs_test,tag_info)
        elif self._alpha == 1:
            weeks = self._known_f_matrix.shape[1]
            n_weeks = len(nq_per_week)
            if weeks == 1 and weeks != n_weeks:
                self._known_f_matrix = np.repeat(self._known_f_matrix,n_weeks,axis=1)

            total_cost = self._build_cost_freq(tag_trends, nq_per_week)
        else:
            weeks = self._known_f_matrix.shape[1]
            n_weeks = len(nq_per_week)
            if weeks == 1 and weeks != n_weeks:
                self._known_f_matrix = np.repeat(self._known_f_matrix,n_weeks,axis=1)
            freq = self._build_cost_freq(tag_trends, nq_per_week)
            rlen = self._build_cost_rlen(n_docs_test,tag_info)
            total_cost = self._alpha*freq + (1-self._alpha)*rlen

        row_ind, col_ind = hungarian(total_cost)

        query_predictions_for_each_tag = {}
        for tag, keyword in zip(col_ind, row_ind):
            query_predictions_for_each_tag[tag] = keyword
        query_predictions_for_each_obs = []
        for weekly_tags in tag_traces:
            query_predictions_for_each_obs.append([query_predictions_for_each_tag[tag_id] for tag_id in weekly_tags])
        pred = [self._chosen_keywords[kw] for week_kw in query_predictions_for_each_obs for kw in week_kw]

        # weekly_queries = self._split_traces(queries,5)
        # flat_real = [kw for week_kws in weekly_queries for kw in week_kws]
        # accuracy = np.mean(np.array([1 if real == prediction else 0 for real, prediction in zip(flat_real, pred)]))
        # print("SAP alpha",self._alpha,"=",accuracy)

        return pred


class RelationalSap(KeywordAttack):
    """
    Implements the SAP attack from Oya & Kerschbaum for relational data using only ResponseLength patterns (no volume patterns).
    """
    _known_response_length: Dict[str, int]
    _known_keywords: Dict
    _delta: float
    _full_id_extension: IdentityExtension
    _id_extension: IdentityExtension
    _full_size: int

    def __init__(self, known: SQLRelationalDatabase):
        super(RelationalSap, self).__init__(known)

        if not known.has_extension(IdentityExtension):
            known.extend_with(IdentityExtension)

        if isinstance(known, SampledSQLRelationalDatabase):
            full = known.parent()
        else:
            # dataset is not sampled, therefore whole dataset is known
            full = known

        self.id_extension = known.get_extension(IdentityExtension)
        self.full_id_extension = full.get_extension(IdentityExtension)
        self._full_size = len(full.row_ids())

        self._delta = known.sample_rate()

        self._known_response_length = dict()
        self._known_keywords = dict()

        self._perform_estimation(self._get_estimator(known, full), known, len(full.doc_ids()))

    def _get_estimator(self, known, full) -> RelationalEstimator:
        # estimator that returns rlen (based on full dataset size)
        return SamplingRelationalEstimator(known, full)

    def _perform_estimation(self, estimator: RelationalEstimator, known: SQLRelationalDatabase, n: int):
        i = 0
        for keyword in known.keywords():
            self._known_keywords[i] = keyword
            self._known_keywords[keyword] = i
            i += 1
            est_rlen = round(estimator.estimate(keyword))
            self._known_response_length[keyword] = est_rlen

    @classmethod
    def name(cls) -> str:
        return "Relational-SAP"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [ResponseLength()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {IdentityExtension}

    def _build_cost_rlen(self, n: int, rlens: List[int]):
        # estimation is always scaled to full size, therefore we normalize here by the full size (n)
        kw_probs_train = [self._known_response_length[self._known_keywords[i]] / n
                          for i in range(len(self._known().keywords()))]
        log_prob_matrix = compute_log_binomial_probability_matrix(n, kw_probs_train, rlens)
        cost_vol = - log_prob_matrix
        return cost_vol

    def recover(self, dataset: Dataset, queries: Iterable[RelationalQuery]) -> List[str]:
        log.info(f"Running {self.name()} at {self._delta:.3f}")
        queries = list(queries)
        leakage = list(self.required_leakage()[0](dataset, queries))

        rlen_cost = self._build_cost_rlen(len(dataset.doc_ids()), rlens=[l for l in leakage])
        total_cost = rlen_cost

        row_ind, col_ind = hungarian(total_cost)

        res = ["" for _ in range(len(leakage))]

        for i, j in zip(col_ind, row_ind):
            res[i] = self._known_keywords[j]

        return res


class ErrorSimulationRelationalSap(RelationalSap):
    """
    Implements the SAP attack from Oya & Kerschbaum for relational data using only ResponseLength patterns (no volume patterns).
    """
    __mean_error: float

    def __init__(self, known: SQLRelationalDatabase, mean_error: float):
        self.__mean_error = mean_error
        super(ErrorSimulationRelationalSap, self).__init__(known)

    @classmethod
    def name(cls) -> str:
        return "Error-Simulation-Relational-SAP"

    def _perform_estimation(self, estimator: RelationalEstimator, known: SQLRelationalDatabase, n: int):
        i = 0
        for keyword in known.keywords():
            self._known_keywords[i] = keyword
            self._known_keywords[keyword] = i
            i += 1
            perfect_rlen = len(self.full_id_extension.doc_ids(keyword))
            if self.__mean_error != 1.0:
                gaussian_error = np.random.normal(loc=self.__mean_error, scale=0.1, size=None)

                if random.choice([0, 1]) == 0:  # try to simulate max inversion
                    simulated_rlen = perfect_rlen * gaussian_error
                else:
                    simulated_rlen = perfect_rlen / gaussian_error

                if simulated_rlen > n:  # hungarian algorithm fails, if values are too large
                    self._known_response_length[keyword] = n
                else:
                    self._known_response_length[keyword] = simulated_rlen
            else:
                self._known_response_length[keyword] = perfect_rlen


class PerfectRelationalSap(RelationalSap):
    """
    Implements the SAP attack from Oya & Kerschbaum for relational data using only ResponseLength patterns (no volume patterns).
    Uses perfect estimates for rlen.
    """

    def __init__(self, known: SQLRelationalDatabase):
        super(PerfectRelationalSap, self).__init__(known)

    def _get_estimator(self, known, full) -> RelationalEstimator:
        # estimator that returns rlen (based on full dataset size)
        return PerfectRelationalEstimator(known, full)

    @classmethod
    def name(cls) -> str:
        return "Perfect-Relational-SAP"


class NaruRelationalSap(RelationalSap):
    """
    Implements the SAP attack from Oya & Kerschbaum for relational data using only ResponseLength patterns and the naru estimator
    """

    def __init__(self, known: SQLRelationalDatabase):
        """
        known : should be a SampledSQLRelationalDatabase object
                otherwise known dataset is assumed to be the full dataset
        """
        super().__init__(known)

    def _get_estimator(self, known, full) -> RelationalEstimator:
        # estimator that returns rlen (based on full dataset size)
        return NaruRelationalEstimator(known, full)

    @classmethod
    def name(cls) -> str:
        return "NaruRelationalSAP"
