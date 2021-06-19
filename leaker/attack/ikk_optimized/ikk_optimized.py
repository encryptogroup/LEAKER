from logging import getLogger

from typing import List, Any, Set, Type, TypeVar, Iterable, Dict

from leaker.api import KeywordAttack, LeakagePattern, Extension, Dataset
from leaker.extension import CoOccurrenceExtension
from leaker.pattern import CoOccurrence

from ikk_roessink.ikk import IKK
from ikk_roessink.matrix_generation import MatrixGenerator

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class Ikk_optimized(KeywordAttack):
    """
    Implements the optimized version of the attack from [IKK12] by Groot Roessink.
    """
    _known_keywords: Set[str]
    _known_coocc: CoOccurrenceExtension
    _init_temperature: float
    _cooling_rate: float
    _reject_threshold: int
    _known_query_size: float
    _deterministic: bool
    _num_runs: int
    _ikk: IKK
    _background_index: Iterable
    _background_cooc: Iterable
    _gen: MatrixGenerator

    def __init__(self, known: Dataset, init_temperature: float = 200000.0, min_temperature: float = 1e-06,
                 cooling_rate: float = 0.99999, reject_threshold: int = 10000, known_query_size: float = 0.15,
                 deterministic: bool = False, num_runs: int = 1):
        log.info(f"Setting up Ikk_optimized attack for {known.name()}. This might take some time.")
        super(Ikk_optimized, self).__init__(known)

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

        self._deterministic = deterministic
        self._num_runs = num_runs
        self._ikk = IKK()

        self._gen = MatrixGenerator()

        self._background_knowledge = known
        # TODO ensure we are comparing the right types, result of query can be either string or document
        self._background_index = self._gen.generate_inverted_index(self._get_files_per_keyword(self._background_knowledge),
                                                             self._background_knowledge.documents())
        self._background_cooc = self._gen.generate_cooccurrence_matrix(self._background_index)

        log.info("Setup complete.")


    @classmethod
    def name(cls) -> str:
        return "IKK-Optimized"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        # not sure if this is sufficient due to usage of index and differenct cooc matrix
        return [CoOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {CoOccurrenceExtension}

    def _get_files_per_keyword(self, dataset: Dataset):
        kws = dataset.keywords()
        res = {kw: list(dataset.query(kw)) for kw in kws}
        return res

    # TODO check if we need new code to handle generation of server knowledge
    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        server_knowledge = dataset
        server_index = self._gen.generate_inverted_index(self._get_files_per_keyword(server_knowledge),
                                                         server_knowledge.documents())
        server_cooc = self._gen.generate_cooccurrence_matrix(server_index)

        states: List[Dict] = []  # The dict maps queries to keywords

        for i in range(self._num_runs):
            state, temperature, rejects, e, count_total, count_accepted = self._ikk.optimizer(
                server_knowledge_cooccurrence_matrix=server_cooc,
                background_knowledge_coocurrence_matrix=self._background_cooc, server_knowledge_index=server_index,
                background_knowledge_index=self._background_index, init_temperature=self._init_temperature,
                cooling_rate=self._cooling_rate, reject_threshold=self._reject_threshold,
                deterministic_ikk=self._deterministic)
            states.append(state)

        result = []

        # TODO check if queries and the queries of the result are always the same
        # is the case when queries only come from dataset that is passed to recover
        # entries from bg are used to obtain the values, server is used for variables
        for query in queries:
            state_results = [state[query] for state in states]

            max_count = 0
            best_candidate = None

            for candidate in set(state_results):
                count = state_results.count(candidate)
                if count > max_count:
                    max_count = count
                    best_candidate = candidate
                elif count == max_count:
                    best_candidate = None

            result.append(best_candidate)

        return result
