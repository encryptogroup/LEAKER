"""Some Code in this file has been adapted from https://github.com/simon-oya/USENIX22-ihop-code"""
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, TypeVar, Type, Union

import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment as hungarian

from ..api import Extension, KeywordAttack, Dataset, LeakagePattern, KeywordQueryLog
from ..extension import CoOccurrenceExtension, IdentityExtension
from ..pattern import ResponseLength, Frequency, ResponseIdentity, CoOccurrence

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

def get_steady_state(markov_matrix):
    n = markov_matrix.shape[0]
    aux = np.vstack((markov_matrix - np.eye(n), np.ones((1, n))))
    return np.linalg.solve(aux.T @ aux, np.ones(n))

def compute_Vobs(token_info, n_docs_test):
    ntok = len(token_info)
    database_matrix = np.zeros((n_docs_test, ntok))
    for tag in token_info:
        for doc_id in token_info[tag]:
            database_matrix[doc_id, tag] = 1
    Vobs = np.matmul(database_matrix.T, database_matrix) / n_docs_test
    return Vobs

def compute_fobs(token_trace, n_tokens):
    counter = Counter(token_trace)
    fobs = np.array([counter[j] / len(token_trace) for j in range(n_tokens)])
    return fobs

def compute_Fobs(token_trace, n_tokens):
    Fobs = np.zeros((n_tokens, n_tokens))
    nq_per_tok = np.zeros(n_tokens)
    mj_test = np.histogram2d(token_trace[1:], token_trace[:-1], bins=(range(n_tokens + 1), range(n_tokens + 1)))[0] / (len(token_trace) - 1)
    for j in range(n_tokens):
        nq_per_tok[j] = np.sum(mj_test[:, j])
        if np.sum(mj_test[:, j]) > 0:
            Fobs[:, j] = mj_test[:, j] / np.sum(mj_test[:, j])
    return nq_per_tok, Fobs

def get_faux(freq,nkw):
    epsilon = 1e-20
    if freq is None:
        faux = np.ones(nkw) / nkw
    else:
        faux = (freq + epsilon / nkw) / (1 + epsilon * 2 / nkw)
    return faux

def get_Faux(freq,nkw):
    epsilon = 1e-20
    if freq is None:
        Faux = np.ones((nkw, nkw)) / nkw
    else:
        Faux = np.tile(((freq + epsilon / nkw) / (1 + 2 * epsilon / nkw)).reshape(nkw, 1), nkw)
    return Faux

def get_Fexp(freq,nkw):
    return get_Faux(freq,nkw)

def get_Vexp(dataset:Dataset, ids, chosen_keywords):
    ndocs = len(dataset)
    nkw = len(chosen_keywords)
    keywords = dataset.keywords()
    doc_map = {}
    i = 0
    for doc in dataset.documents():
        if isinstance(doc, tuple):
            # relational case
            doc_map[doc[1]] = i
        else:
            doc_map[doc.id()] = i
        i += 1
    binary_database_matrix = np.zeros((ndocs, nkw))
    for keyword in keywords:
        if keyword in chosen_keywords:
            resp = ids[keyword]
            i_kw = chosen_keywords.index(keyword)
            for doc in resp:
                try:
                    if isinstance(doc, tuple):
                        # relational case
                        binary_database_matrix[doc_map[doc[1]], i_kw] = 1
                    else:
                        binary_database_matrix[doc_map[doc], i_kw] = 1
                except IndexError as e:
                    log.warning(e)
    epsilon = 1e-20  # Value to control that there are no zero elements
    Vaux = (np.matmul(binary_database_matrix.T, binary_database_matrix) + epsilon) / (ndocs + 2 * epsilon)
    return Vaux

def get_all_freqs(known:Dataset, chosen_keywords: List[str], frequencies: np.ndarray):
    keywords = list(known.keywords())
    nkw = len(keywords)
    nweeks = frequencies.shape[1]
    freqs = np.zeros((nkw,nweeks))
    for i,kw in enumerate(keywords):
        if kw in chosen_keywords:
            freqs[i,:] = frequencies[chosen_keywords.index(kw),:]
    return keywords, freqs

class Ihop(KeywordAttack):
    """
    Implements the IHOP attack from Oya & Kerschbaum.
    """
    _known_volume: Dict[str, int]
    _known_response_length: Dict[str, int]
    _known_keywords: Dict
    _delta: float
    _known_f_matrix: np.ndarray
    _chosen_keywords: List[str]
    _alpha: float
    _nkw: int
    _niters: int
    _pct_free: float
    _known_ids: Dict[str,int]

    def __init__(self, known: Dataset, known_frequencies: np.ndarray = None, chosen_keywords: List[str] = None, query_log: KeywordQueryLog = None, alpha:float = 0.0, niters:int=1000, pct_free:float=0.25, modify: bool = False):
        super(Ihop, self).__init__(known)

        self._delta = known.sample_rate()
        self._known_volume = dict()
        self._known_response_length = dict()
        self._known_ids = dict()
        self._known_keywords = dict()
        self._niters = niters
        self._pct_free =  pct_free
        #if not known.has_extension(VolumeExtension):
        #    known.extend_with(VolumeExtension)

        #vol = known.get_extension(VolumeExtension)

        if not known.has_extension(IdentityExtension):
            known.extend_with(IdentityExtension)
        
        ide = known.get_extension(IdentityExtension)
        
        if known_frequencies is not None:
            assert chosen_keywords is not None, log.error("Auxiliary frequency knowledge and correstponding keywords needed for IHOP frequency attack.")
            if modify:
                chosen_keywords,known_frequencies = get_all_freqs(known,chosen_keywords,known_frequencies)
            self._known_f_matrix = np.mean(known_frequencies,axis=1)
            self._chosen_keywords = chosen_keywords
        elif query_log is not None:
            known_frequencies, chosen_keywords = compute_frequencies(query_log)
            self._known_f_matrix = np.mean(known_frequencies,axis=1)
            self._chosen_keywords = chosen_keywords
        else:
            assert alpha == 0, log.error("Auxiliary frequency knowledge and correstponding keywords needed for IHOP frequency attack.")
            if chosen_keywords is not None:
                self._chosen_keywords = chosen_keywords
            else:
                self._chosen_keywords = list(known.keywords())
            self._known_f_matrix = None
        self._alpha = alpha

        i = 0
        for keyword in self._chosen_keywords:
            self._known_keywords[i] = keyword
            self._known_keywords[keyword] = i
            i += 1
            #self._known_volume[keyword] = vol.total_volume(keyword)
            #self._known_response_length[keyword] = vol.selectivity(keyword)
            self._known_ids[keyword] = ide.doc_ids(keyword)

        self._nkw = len(self._chosen_keywords)

    @classmethod
    def name(cls) -> str:
        return "IHOP"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [ResponseLength(), Frequency(), CoOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        #return {VolumeExtension, CoOccurrenceExtension}
        return {IdentityExtension, CoOccurrenceExtension}
    
    def get_update_coefficients_functions(self, token_trace, token_info, ndocs, Vexp = None):
        """ pass Vexp to overwrite it """
        def _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):
            cost_vol = -compute_log_binomial_probability_matrix(ndocs, np.diagonal(Vexp)[free_keywords], np.diagonal(Vobs)[free_tags] * ndocs)
            for tag, kw in zip(fixed_tags, fixed_keywords):
                cost_vol -= compute_log_binomial_probability_matrix(ndocs, Vexp[kw, free_keywords], Vobs[tag, free_tags] * ndocs)
            return cost_vol

        def _build_cost_freq_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):
            cost_freq = - (nqr * fobs[free_tags]) * np.log(np.array([fexp[free_keywords]]).T)
            return cost_freq

        def _build_cost_Freq_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):

            cost_matrix = np.zeros((len(free_keywords), len(free_tags)))
            ss_aux = get_steady_state(Fexp)

            cost_matrix -= Fobs_counts[np.ix_(free_tags, free_tags)].diagonal() * np.log(np.array([Fexp[np.ix_(free_keywords, free_keywords)].diagonal()]).T)

            ss_from_others_train = (Fexp[np.ix_(free_keywords, free_keywords)] *
                                    (np.ones((len(free_keywords), len(free_keywords))) - np.eye(len(free_keywords)))) @ ss_aux[free_keywords]
            ss_from_others_train = ss_from_others_train / (np.sum(ss_aux[free_keywords]) - ss_aux[free_keywords])
            counts_from_others_test = Fobs_counts[np.ix_(free_tags, free_tags)].sum(axis=1) - Fobs_counts[np.ix_(free_tags, free_tags)].diagonal()
            cost_matrix -= counts_from_others_test * np.log(np.array([ss_from_others_train]).T)

            for tag, kw in zip(fixed_tags, fixed_keywords):
                cost_matrix -= Fobs_counts[free_tags, tag] * np.log(np.array([Fexp[free_keywords, kw]]).T)
                cost_matrix -= Fobs_counts[tag, free_tags] * np.log(np.array([Fexp[kw, free_keywords]]).T)

            return cost_matrix

        nqr = len(token_trace)
        rep_to_kw = rep_to_kw = {rep: rep for rep in range(self._nkw)}

        if self._alpha == 0:
            Vobs = compute_Vobs(token_info, ndocs)
            if Vexp is None:
                Vexp = get_Vexp(self._known(), self._known_ids, self._chosen_keywords)
            return _build_cost_Vol_some_fixed, rep_to_kw
        elif self._alpha == 1:  
            nq_per_tok, Fobs = compute_Fobs(token_trace, len(token_info))
            Fobs_counts = Fobs * nq_per_tok
            Fexp = get_Fexp(self._known_f_matrix,self._nkw)   
            return _build_cost_Freq_some_fixed, rep_to_kw
        else:
            Vobs = compute_Vobs(token_info, ndocs)
            fobs = compute_fobs(token_trace, len(token_info))
            if Vexp is None:
                Vexp = get_Vexp(self._known(), self._known_ids, self._chosen_keywords)
            fexp = get_faux(self._known_f_matrix, self._nkw)
            def compute_cost(free_keywords, free_tags, fixed_keywords, fixed_tags):
                return _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags) + \
                    _build_cost_freq_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags)
            return compute_cost, rep_to_kw


    def process_traces(self,traces,doc_to_id:dict):
        """tag_info is a dict [tag] -> AP (list of doc ids)"""
        token_trace = []
        seen_tuples = {}
        token_info = {}
        token_id = 0
        for trace in traces:
            ap_sorted = tuple(sorted(trace))
            if ap_sorted not in seen_tuples:
                seen_tuples[ap_sorted] = token_id
                ap_ids = []
                for ap in ap_sorted:
                    ap_ids.append(doc_to_id[ap])
                token_info[token_id] = tuple(ap_ids)
                token_id += 1
            token_trace.append(seen_tuples[ap_sorted])
        return token_trace, token_info



    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        ndocs = len(dataset)
        rid = ResponseIdentity()
        queries = list(queries)
        doc_ids = dataset.doc_ids()
        doc_to_id = {}
        id_to_doc = {}
        for id, doc in enumerate(doc_ids):
            doc_to_id[doc] = id
            id_to_doc[id] = doc
        token_trace, token_info = self.process_traces(rid(dataset, queries),doc_to_id)

        compute_coef_matrix, rep_to_kw = self.get_update_coefficients_functions(token_trace, token_info, ndocs)

        nrep = len(rep_to_kw)
        ntok = len(token_info)


        unknown_toks = [i for i in range(ntok)]
        unknown_reps = [i for i in range(nrep)]

        # First matching:
        c_matrix_original = compute_coef_matrix(unknown_reps, unknown_toks, [], [])
        row_ind, col_ind = hungarian(c_matrix_original)
        replica_predictions_for_each_token = {}
        for j, i in zip(col_ind, row_ind):
            try:
                replica_predictions_for_each_token[unknown_toks[j]] = unknown_reps[i]
            except KeyError:
                pass


        # Iterate using co-occurrence:
        n_free = int(self._pct_free * len(unknown_toks))
        assert n_free > 1, log.error(f"n_free = {n_free} is too small.")
        for _ in range(self._niters):
            random_unknown_tokens = list(np.random.permutation(unknown_toks))
            free_tokens = random_unknown_tokens[:n_free]
            fixed_tokens = random_unknown_tokens[n_free:]
            try:
                fixed_reps = [replica_predictions_for_each_token[token] for token in fixed_tokens]
            
                free_replicas = [rep for rep in unknown_reps if rep not in fixed_reps]

                c_matrix = compute_coef_matrix(free_replicas, free_tokens, fixed_reps, fixed_tokens)

                row_ind, col_ind = hungarian(c_matrix)
                for j, i in zip(col_ind, row_ind):
                    replica_predictions_for_each_token[free_tokens[j]] = free_replicas[i]
            except KeyError:
                pass

        keyword_predictions_for_each_query = []
        for token in token_trace:
            try:
                keyword_predictions_for_each_query.append(replica_predictions_for_each_token[token])
            except KeyError:
                pass

        pred = [self._chosen_keywords[kw_id] for kw_id in keyword_predictions_for_each_query]
        # accuracy = np.mean(np.array([1 if real == prediction else 0 for real, prediction in zip(queries, pred)]))
        # print("IHOP_accuracy =",accuracy)
        return pred


class PerfectIhop(Ihop):
    """
    Implements the IHOP attack from Oya & Kerschbaum for the relational setting
    """
    def __init__(self, known: Dataset, known_frequencies: np.ndarray = None, chosen_keywords: List[str] = None, query_log: KeywordQueryLog = None, alpha:float = 0.0, niters:int=1000, pct_free:float=0.25, modify: bool = False):
        super(PerfectIhop, self).__init__(known, known_frequencies, chosen_keywords, query_log, alpha, niters, pct_free, modify)

    @classmethod
    def name(cls) -> str:
        return "Perfect-IHOP"

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        ndocs = len(dataset)
        rid = ResponseIdentity()
        queries = list(queries)
        doc_ids = dataset.doc_ids()
        doc_to_id = {}
        id_to_doc = {}
        for id, doc in enumerate(doc_ids):
            doc_to_id[doc] = id
            id_to_doc[id] = doc
        token_trace, token_info = self.process_traces(rid(dataset, queries), doc_to_id)

        # Start relational estimator part
        if not dataset.get_extension(CoOccurrenceExtension):
            dataset.extend_with(CoOccurrenceExtension)
        ide = dataset.get_extension(CoOccurrenceExtension)

        epsilon = 1e-20  # Value to control that there are no zero elements
        known_keywords = list(self._known().keywords())
        nkw = len(known_keywords)
        Vexp = np.zeros((nkw, nkw))
        for kw1 in known_keywords:
            i_kw1 = known_keywords.index(kw1)
            for kw2 in known_keywords:
                i_kw2 = known_keywords.index(kw2)
                Vexp[i_kw1, i_kw2] = (ide.co_occurrence(kw1, kw2) + epsilon) / (ndocs + 2 * epsilon)

        compute_coef_matrix, rep_to_kw = self.get_update_coefficients_functions(token_trace, token_info, ndocs, Vexp)

        nrep = len(rep_to_kw)
        ntok = len(token_info)

        unknown_toks = [i for i in range(ntok)]
        unknown_reps = [i for i in range(nrep)]

        # First matching:
        c_matrix_original = compute_coef_matrix(unknown_reps, unknown_toks, [], [])
        row_ind, col_ind = hungarian(c_matrix_original)
        replica_predictions_for_each_token = {}
        for j, i in zip(col_ind, row_ind):
            try:
                replica_predictions_for_each_token[unknown_toks[j]] = unknown_reps[i]
            except KeyError:
                pass

        # Iterate using co-occurrence:
        n_free = int(self._pct_free * len(unknown_toks))
        assert n_free > 1, log.error(f"n_free = {n_free} is too small.")
        for _ in range(self._niters):
            random_unknown_tokens = list(np.random.permutation(unknown_toks))
            free_tokens = random_unknown_tokens[:n_free]
            fixed_tokens = random_unknown_tokens[n_free:]
            try:
                fixed_reps = [replica_predictions_for_each_token[token] for token in fixed_tokens]

                free_replicas = [rep for rep in unknown_reps if rep not in fixed_reps]

                c_matrix = compute_coef_matrix(free_replicas, free_tokens, fixed_reps, fixed_tokens)

                row_ind, col_ind = hungarian(c_matrix)
                for j, i in zip(col_ind, row_ind):
                    replica_predictions_for_each_token[free_tokens[j]] = free_replicas[i]
            except KeyError:
                pass

        keyword_predictions_for_each_query = []
        for token in token_trace:
            try:
                keyword_predictions_for_each_query.append(replica_predictions_for_each_token[token])
            except KeyError:
                pass

        pred = [self._chosen_keywords[kw_id] for kw_id in keyword_predictions_for_each_query]
        # accuracy = np.mean(np.array([1 if real == prediction else 0 for real, prediction in zip(queries, pred)]))
        # print("IHOP_accuracy =",accuracy)
        return pred
