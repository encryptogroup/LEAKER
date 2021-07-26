"""
This file interfaces a LEAKER attack with the Roessink IKK implementation.

For License information see the LICENSE file.
Authors: Amos Treiber, Michael Yonli
"""

from logging import getLogger
from typing import List, Any, Set, Type, TypeVar, Iterable, Dict, Iterator, Tuple

from leaker.api import KeywordAttack, LeakagePattern, Extension, Dataset
from leaker.extension import CoOccurrenceExtension
from leaker.pattern import CoOccurrence
from .ikk_roessink.ikk import IKK

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class ListInterface(list):
    """To overload df.index.tolist()"""
    def tolist(self):
        return self


class LocInterface:
    """To overload df.loc[a,b]"""
    __ext: CoOccurrenceExtension

    def __init__(self, ext: CoOccurrenceExtension):
        self.__ext = ext

    def __getitem__(self, item):
        kw1, kw2 = item
        return self.__ext.co_occurrence(kw1, kw2)


class PandasInterface:
    """
    Interfaces with the Groot Roessink implementation and makes it compatible to LEAKER.
    Makes the called code invoke pre-computed extensions instead of pandas dataframes.
    It relies on the original implementation only making calls to .loc[a,b], .index.tolist(), .columns.tolist(),
    and .iterrows().
    """
    _ext: CoOccurrenceExtension
    columns: ListInterface
    index: ListInterface

    def __init__(self, dataset: Dataset, columns: ListInterface, index: ListInterface):
        if not dataset.has_extension(CoOccurrenceExtension):
            dataset.extend_with(CoOccurrenceExtension)

        self._ext = dataset.get_extension(CoOccurrenceExtension)
        self.columns = columns
        self.index = index


class IndexInterface(PandasInterface):
    """Corresponds to 'index' in the Groot Roessink implementation. Instead of a list of indices, it is a truth
    table whether a keyword appears in a document."""

    def __init__(self, dataset: Dataset, keywords: List[str]):
        super().__init__(dataset, ListInterface(dataset.doc_ids()), ListInterface(keywords))

    def iterrows(self) -> Iterator[Tuple[str, List[bool]]]:
        for kw in self.index:
            doc_ids = self._ext.doc_ids(kw)
            yield kw, [doc_id in doc_ids for doc_id in self.columns]


class CoOccInterface(IndexInterface):
    """Corresponds to 'cooccurrence' in the Groot Roessink implementation."""

    loc: LocInterface

    def __init__(self, dataset: Dataset, keywords: List[str]):
        super().__init__(dataset, keywords)
        self.loc = LocInterface(self._ext)

    def iterrows(self) -> Iterator[Tuple[str, List[bool]]]:
        """This should not be used"""
        assert False


class Ikkoptimized(KeywordAttack):
    """
    Implements the optimized version of the attack from [IKK12] by Groot Roessink.
    """
    _init_temperature: float
    _cooling_rate: float
    _reject_threshold: int
    _deterministic: bool
    _num_runs: int
    _ikk: IKK
    _background_index: IndexInterface
    _background_cooc: CoOccInterface

    def __init__(self, known: Dataset, init_temperature: float = 1.0, cooling_rate: float = 0.999,
                 reject_threshold: int = 5000, deterministic: bool = False, num_runs: int = 1):
        log.info(f"Setting up {self.name()} for {known.name()}. This might take some time.")
        super(Ikkoptimized, self).__init__(known)

        self._init_temperature = init_temperature
        self._cooling_rate = cooling_rate
        self._reject_threshold = reject_threshold
        self._deterministic = deterministic
        self._num_runs = num_runs
        self._ikk = IKK()

        self._background_index = IndexInterface(known, list(known.keywords()))
        self._background_cooc = CoOccInterface(known, list(known.keywords()))

        log.info("Setup complete.")

    @classmethod
    def name(cls) -> str:
        return "IKK-Optimized"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [CoOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {CoOccurrenceExtension}

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        server_knowledge = dataset
        queries = list(queries)
        log.info(f"Running {self.name()}")

        server_index = IndexInterface(server_knowledge, queries)
        server_cooc = CoOccInterface(server_knowledge, queries)

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
        log.info(f"Reconstruction completed.")
        return result
