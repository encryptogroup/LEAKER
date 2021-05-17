"""
For License information see the LICENSE file.

Authors: Tobias Stöckert

"""
import copy
import math
import random
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, Type, TypeVar

from ..api import Extension, KeywordAttack, Dataset, LeakagePattern
from ..extension import CoOccurrenceExtension
from ..pattern import CoOccurrence

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class Ikk(KeywordAttack):
    """
    Implements the attack from [IKK12]. It uses the CoOccurrence and the ResponseLength patterns
    """
    _known_keywords: Set[str]
    _known_coocc: CoOccurrenceExtension
    _init_temperature: float
    _cooling_rate: float
    _reject_threshold: int
    _known_query_size: float

    def __init__(self, known: Dataset, init_temperature: float = 200000.0, min_temperature: float = 1e-06,
                 cooling_rate: float = 0.99999,
                 reject_threshold: int = 10000, known_query_size: float = 0.15):
        log.info(f"Setting up Ikk attack for {known.name()}. This might take some time.")
        super(Ikk, self).__init__(known)

        if not known.has_extension(CoOccurrenceExtension):
            known.extend_with(CoOccurrenceExtension)
        self._known_coocc = known.get_extension(CoOccurrenceExtension)

        self._known_keywords = known.keywords()
        self._known_response_length = dict()
        self._init_temperature = init_temperature
        self._min_temperature = min_temperature
        self._cooling_rate = cooling_rate
        self._reject_threshold = reject_threshold
        self._known_query_size = known_query_size

        log.info("Setup complete.")

    @classmethod
    def name(cls) -> str:
        return "IKK"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [CoOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {CoOccurrenceExtension}

    def __anneal(self, queries: Iterable[str], init_state: Dict[int, str], keywords: Iterable[str],
                 m_c: List[List[int]]):

        current_state = copy.deepcopy(init_state)
        succ_reject = 0
        curr_t = self._init_temperature
        log.debug("Start Anneal")

        while curr_t > self._min_temperature and succ_reject < self._reject_threshold:
            current_cost = 0
            next_cost = 0
            next_state = copy.deepcopy(current_state)

            """generate next state"""
            x = random.choice(list(next_state.keys()))
            y = next_state[x]
            while True:
                y_prime = random.choice(keywords)
                if y_prime != y:
                    break
            next_state.pop(x)
            next_state[x] = y_prime
            if y_prime in list(init_state.values()):
                for key, value in init_state.items():
                    if value == y_prime and next_state[key] == y_prime:
                        next_state.pop(key)
                        next_state[key] = y
                        break

            """cost calculation"""
            for i, _ in enumerate(queries):
                for j, _ in enumerate(queries):
                    k = current_state[i]
                    k_prime = next_state[i]
                    l = current_state[j]
                    l_prime = next_state[j]
                    current_cost += pow((m_c[i][j] - self._known_coocc.co_occurrence(k, l)), 2)
                    next_cost += pow((m_c[i][j] - self._known_coocc.co_occurrence(k_prime, l_prime)), 2)
            e = next_cost - current_cost
            try:
                prob = math.exp(-e / curr_t)
            except OverflowError:
                if e > 0:
                    prob = 0.0
                else:
                    prob = float('inf')

            if e < 0 or prob >= random.random():
                succ_reject = 0
                current_state = copy.deepcopy(next_state)
            else:
                succ_reject += 1
            curr_t = self._cooling_rate * curr_t

        return current_state

    def __optimizer(self, queries: Iterable[str], keywords: List[str], known_query_keyword_pairs: Dict[int, str],
                    m_c: List[List[int]]):
        val_list: List[str] = keywords.copy()
        init_state: Dict[int, str] = dict()
        unknown_queries = [q for q, _ in enumerate(queries) if q not in list(known_query_keyword_pairs.keys())]
        for var_i in unknown_queries:
            val_i = random.choice(val_list)
            init_state[var_i] = val_i
            val_list.remove(val_i)
        init_state.update(known_query_keyword_pairs)
        return self.__anneal(queries, init_state, keywords, m_c)

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running Ikk on {dataset.name()} with query leakage of {self._known_query_size}.")
        coocc = self.required_leakage()[0](dataset, queries)
        known_queries = random.sample(list(queries), int(self._known_query_size*len(list(queries))))
        known_query_keyword_pairs = dict()
        keywords = []
        for kw in self._known_keywords:
            if kw in known_queries:
                known_query_keyword_pairs[list(queries).index(kw)] = kw
            else:
                keywords.append(kw)
        known_query_keyword_pairs = self.__optimizer(queries, keywords, known_query_keyword_pairs, coocc)
        uncovered = []

        for q, _ in enumerate(queries):
            if q in known_query_keyword_pairs:
                uncovered.append(known_query_keyword_pairs[q])
            else:
                uncovered.append("")

        log.info(f"Reconstruction completed.")

        return uncovered
