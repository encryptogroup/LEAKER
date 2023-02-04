"""Some Code in this file has been adapted from https://github.com/simon-oya/USENIX21-sap-code"""
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, TypeVar, Type, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian

from .relational_estimators.estimator import NaruRelationalEstimator, SamplingRelationalEstimator, RelationalEstimator
from .relational_estimators.eval import ErrorMetric
from ..api import Extension, KeywordAttack, Dataset, LeakagePattern, RelationalDatabase, RelationalQuery
from ..extension import VolumeExtension, SelectivityExtension, IdentityExtension
from ..pattern import ResponseLength, TotalVolume
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
    probabilities[probabilities == 1.0] = 0.999999  # prevent log(0)
    log_binom_term = np.array([_log_binomial(ntrials, obs / ntrials) for obs in observations])  # ROW TERM
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = log_binom_term + np.array(observations) * column_term + last_term
    return log_matrix


class Sap(KeywordAttack):
    """
    Implements the SAP attack from Oya & Kerschbaum (the part using volume information).
    It uses the TotalVolume and ResponseLength patterns.
    """
    _known_volume: Dict[str, int]
    _known_response_length: Dict[str, int]
    _known_keywords: Dict
    _delta: float

    def __init__(self, known: Dataset):
        super(Sap, self).__init__(known)

        self._delta = known.sample_rate()

        self._known_volume = dict()
        self._known_response_length = dict()
        self._known_keywords = dict()

        if not known.has_extension(VolumeExtension):
            known.extend_with(VolumeExtension)

        vol = known.get_extension(VolumeExtension)
        i = 0
        for keyword in known.keywords():
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
        return [TotalVolume(), ResponseLength()]

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

    def _build_cost_rlen(self, n: int, rlens: List[int]):

        kw_probs_train = [self._known_response_length[self._known_keywords[i]] / len(self._known().doc_ids())
                          for i in range(len(self._known().keywords()))]
        log_prob_matrix = compute_log_binomial_probability_matrix(n, kw_probs_train,
                                                                  rlens)
        cost_vol = - log_prob_matrix
        return cost_vol

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()} at {self._delta:.3f}")
        queries = list(queries)
        leakage = list(zip(self.required_leakage()[0](dataset, queries), self.required_leakage()[1](dataset, queries)))
        dataset.extend_with(VolumeExtension)
        dtv = dataset.get_extension(VolumeExtension)

        tvol_cost = self._build_cost_tvol(dtv.dataset_volume(),
                                          tvols=[l[0] for l in leakage])  # * tv.dataset_volume() / dtv.dataset_volume()
        rlen_cost = self._build_cost_rlen(len(dataset.doc_ids()), rlens=[l[1] for l in leakage])
        total_cost = 0.5 * tvol_cost + 0.5 * rlen_cost

        row_ind, col_ind = hungarian(total_cost)

        res = ["" for _ in range(len(leakage))]

        for i, j in zip(col_ind, row_ind):
            res[i] = self._known_keywords[j]

        return res


class RelationalSap(KeywordAttack):
    """
    Implements the SAP attack from Oya & Kerschbaum for relational data using only ResponseLength patterns (no volume patterns).
    """
    _known_response_length: Dict[str, int]
    _known_keywords: Dict
    _delta: float

    def __init__(self, known: SQLRelationalDatabase):
        super(RelationalSap, self).__init__(known)

        if not known.has_extension(IdentityExtension):
            known.extend_with(IdentityExtension)
        id_extension = known.get_extension(IdentityExtension)
        full_id_extension = known.parent().get_extension(IdentityExtension)

        self._delta = known.sample_rate()

        self._known_response_length = dict()
        self._known_keywords = dict()

        errors = []
        errors_without_zero_one = []

        i = 0
        for keyword in known.keywords():
            self._known_keywords[i] = keyword
            self._known_keywords[keyword] = i
            i += 1
            est_rlen = len(id_extension.doc_ids(keyword))
            act_rlen = len(full_id_extension.doc_ids(keyword))
            errors.append(ErrorMetric(est_rlen, act_rlen))
            if not (est_rlen == 0 and act_rlen == 1) and not (est_rlen == 1 and act_rlen == 0):
                errors_without_zero_one.append(ErrorMetric(est_rlen, act_rlen))

            self._known_response_length[keyword] = len(id_extension.doc_ids(keyword))

        log.info(((np.median(errors), np.quantile(errors, 0.95), np.quantile(errors, .99), np.max(errors)),
                  (np.median(errors_without_zero_one), np.quantile(errors_without_zero_one, 0.95),
                   np.quantile(errors_without_zero_one, .99), np.max(errors_without_zero_one))))

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

        kw_probs_train = [self._known_response_length[self._known_keywords[i]] / len(self._known().doc_ids())
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


class PerfectRelationalSap(KeywordAttack):
    """
    Implements the SAP attack from Oya & Kerschbaum for relational data using only ResponseLength patterns (no volume patterns).
    Uses perfect estimates for rlen.
    """
    _known_response_length: Dict[str, int]
    _known_keywords: Dict
    _delta: float
    _full_size: int

    def __init__(self, known: SQLRelationalDatabase):
        super(PerfectRelationalSap, self).__init__(known)

        if not known.has_extension(IdentityExtension):
            known.extend_with(IdentityExtension)
        #id_extension = known.get_extension(IdentityExtension)
        full_id_extension = known.parent().get_extension(IdentityExtension)
        self._full_size = len(known.parent().doc_ids())

        self._delta = known.sample_rate()

        self._known_response_length = dict()
        self._known_keywords = dict()

        i = 0
        for keyword in known.keywords():
            self._known_keywords[i] = keyword
            self._known_keywords[keyword] = i
            i += 1
            self._known_response_length[keyword] = len(full_id_extension.doc_ids(keyword))

    @classmethod
    def name(cls) -> str:
        return "Perfect-Relational-SAP"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [ResponseLength()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {IdentityExtension}

    def _build_cost_rlen(self, n: int, rlens: List[int]):

        kw_probs_train = [self._known_response_length[self._known_keywords[i]] / self._full_size
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


class NaruRelationalSap(RelationalSap):
    """
    Implements the SAP attack from Oya & Kerschbaum for relational data using only ResponseLength patterns and the naru estimator
    """
    _known_response_length: Dict[str, int]
    _known_keywords: Dict
    _delta: float
    __est: NaruRelationalEstimator
    __est_sampling: SamplingRelationalEstimator

    ''' Set estimation lower limit absolute (e.g. 2 to skip sampling in 1 case) and upper limit relative (e.g. 0.5% as 
        in naru paper). Upper absolute limit will then be calculated based on the number of rows in the full dataset. '''
    __estimation_lower_limit = 0
    __estimation_upper_limit_relative = 1
    __estimation_upper_limit: int

    def __init__(self, known: SQLRelationalDatabase):
        """
        known : should be a SampledSQLRelationalDatabase object
                otherwise known dataset is assumed to be the full dataset
        """
        if not isinstance(known, SampledSQLRelationalDatabase):
            raise ValueError('Known dataset need to be of instance SampledSQLRelationalDatabase')

        super().__init__(known)

        if isinstance(known, SampledSQLRelationalDatabase):
            full = known.parent()
        else:
            # dataset is not sampled, therefore whole dataset is known
            full = known

        self.__est = NaruRelationalEstimator(sample=known, full=full)
        self.__est_sampling = SamplingRelationalEstimator(known, full)
        self.__estimation_upper_limit = round(len(known.parent().queries()) * self.__estimation_upper_limit_relative)

        log.debug('Start estimating known queries. This might take a while...')
        self._known_response_length = self._perform_estimation(self.__est, self.__est_sampling, known)
        log.debug('Finished estimating known queries')

    @classmethod
    def name(cls) -> str:
        return "Naru-SAP"

    def _perform_estimation(self, estimator: RelationalEstimator,
                            sampling_estimator: Optional[SamplingRelationalEstimator], known: SQLRelationalDatabase) \
            -> Dict[str, int]:
        """ Estimate rlen for all known queries.
        If sampled rlen is between lower and upper limit, then use naru estimator """

        known_response_length: Dict[str, int] = dict()
        for keyword in known.keywords():
            sampled_rlen = round(self.__est_sampling.estimate(keyword))
            if self.__estimation_lower_limit <= sampled_rlen <= self.__estimation_upper_limit:
                known_response_length[keyword] = round(estimator.estimate(keyword))
            else:
                known_response_length[keyword] = sampled_rlen
        return known_response_length

    def _build_cost_rlen(self, n: int, rlens: List[int]):

        kw_probs_train = [self._known_response_length[self._known_keywords[i]] / len(self._known().parent().doc_ids())
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


class NaruRelationalSapFast(NaruRelationalSap):
    """
    Adaption of NaruRelationalSap that uses basic sampling for queries with known rlen of 1 instead of naru estimation.
    This can improve the setup runtime.
    """

    def __init__(self, known: SQLRelationalDatabase):
        """
        known : should be a SampledSQLRelationalDatabase object
                otherwise known dataset is assumed to be the full dataset
        """
        super().__init__(known)

    def _perform_estimation(self, naru_estimator: NaruRelationalEstimator,
                            sampling_estimator: SamplingRelationalEstimator, known: SQLRelationalDatabase) \
            -> Dict[str, int]:
        """Perform estimation by sampling in case of known rlen is 1, otherwise use naru estimation"""
        known_response_length: Dict[str, int] = dict()
        for keyword in known.keywords():
            if self._known_response_length[keyword] != 1:
                # if known rlen is not 1, use naru estimation
                known_response_length[keyword] = int(naru_estimator.estimate(keyword))
            else:
                # if known rlen is 1, use sampling
                known_response_length[keyword] = int(sampling_estimator.estimate(keyword))
        return known_response_length

    @classmethod
    def name(cls) -> str:
        return "Naru-SAP-fast"
