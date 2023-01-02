"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from functools import reduce
from logging import getLogger
from multiprocessing.pool import Pool
from typing import List, Iterable, Union, Type, Dict, Optional, Iterator, Tuple, TypeVar

from .errors import Error, MAError
from ..api import AttackDefinition, Dataset, QuerySpace, KeywordQuerySpace, Selectivity, Attack, RangeDatabase,\
    Extension, KeywordQueryLog, RelationalDatabase

log = getLogger(__name__)

T = TypeVar("T", bound=Extension, covariant=True)


class EvaluationCase:
    """
    An evaluation case to use with the Evaluator. It consists of one or multiple attacks to evaluate, a data set to
    evaluate the attacks on and a number of runs to perform.

    Parameters
    ----------
    attacks: Union[AttackDefinition, Type[Attack], Iterable[Union[AttackDefinition, Type[Attack]]]]
        may be one or multiple elements of which each can be the type of an attack (i.e. a subclass of Attack) or an
        AttackDefinition (for example when specifying additional parameters)
    dataset: Union[Dataset, RangeDatabase, RelationalDatabase]
        the data set to evaluate the attacks on
    runs: int
        the number of runs of sampling new datasets for the kdr
        default: 1
    error: Type[Error]
        the error to use for a Range Attack EvaluationCase
        default: None
    base_restriction_rates: Iterable[float]
        the rates to restrict the base datasets to (base datasets are later used for an evaluation with sampling)
        default: None
    base_restrictions_repetitions: int
        how often an evaluation will be performed with fresh restricted base datasets
        => EvaluationCase.runs() runs for base_restrictions_repetitions new restricted base data sets in
        base_restriction_rates or max_keywords
        default: 1
    max_keywords: int
        The keyword size to restrict the base dataset to (will only be performed if no restriction_rates are set)
        default: 0
    selectivity: Selectivity
        If max_keywords is not 0, this determines the selectivity by which the keywords are chosen
    pickle_description: str
        If not None, use pickle_description as description string to load previously pickled extensions.
        Use "" if no description string was provided during pickling to load the exact previous extensions
        (should only be done when the evaluation takes place on the exact same dataset)
        default: None
    """
    __attacks: List[AttackDefinition]
    __datasets: List[Union[Dataset, RangeDatabase]]
    __full_dataset: Union[Dataset, RangeDatabase]

    __runs: int
    __base_restrictions_repetitions: int
    __error: Union[None, Type[Error]]

    def __init__(self, attacks: Union[AttackDefinition, Type[Attack], Iterable[Union[AttackDefinition, Type[Attack]]]],
                 dataset: Union[Dataset, RangeDatabase, RelationalDatabase], runs: int = 1, error: Type[Error] = None,
                 base_restriction_rates: Iterable[float] = None,
                 base_restrictions_repetitions: int = 1, max_keywords: int = 0,
                 selectivity: Selectivity = Selectivity.Independent, pickle_description: str = None):
        if runs < 1 or base_restrictions_repetitions < 1:
            raise ValueError("Run and repetition count must be at least 1")

        if isinstance(attacks, AttackDefinition):
            self.__attacks = [attacks]
        elif isinstance(attacks, type) and issubclass(attacks, Attack):
            self.__attacks = [attacks.definition()]
        else:
            self.__attacks = [a if isinstance(a, AttackDefinition) else a.definition() for a in attacks]

        self.__full_dataset = dataset

        if base_restriction_rates is not None:
            self.__datasets = [dataset.restrict_rate(rate) for rate in base_restriction_rates
                               for _ in range(base_restrictions_repetitions)]
        elif max_keywords != 0:
            self.__datasets = [dataset.restrict_keyword_size(max_keywords, selectivity)
                               for _ in range(base_restrictions_repetitions)]
        else:
            self.__datasets = [dataset]

        extensions = \
            [ext for ext in reduce(lambda a, b: a.union(b), [atk.required_extensions() for atk in self.__attacks])]
        min_extensions: List[Type[T]] = []  # Avoid building redundant extensions, like, e.g., Id and Sel
        for i, ext in enumerate(extensions):
            if not any(issubclass(ext1, ext) for j, ext1 in enumerate(extensions) if j != i):
                min_extensions.append(ext)

        log.debug(f"Computed minimal extensions {min_extensions} for this evaluation")

        if min_extensions:
            if pickle_description is None:
                for ext in min_extensions:
                    self.__datasets = [d.extend_with(ext) for d in self.__datasets]
            else:
                for ext in min_extensions:
                    self.__datasets = [d.extend_with_pickle(ext, pickle_description) for d in self.__datasets]

        self.__runs = runs
        self.__base_restrictions_repetitions = base_restrictions_repetitions
        self.__error = error

    def datasets(self) -> Iterable[Dataset]:
        yield from self.__datasets

    def full_dataset(self) -> Dataset:
        return self.__full_dataset

    def attacks(self) -> List[AttackDefinition]:
        return self.__attacks

    def runs(self) -> int:
        return self.__runs

    def error(self) -> Type[Error]:
        if self.__error is None:
            return MAError
        else:
            return self.__error

    def base_restrictions_repetitions(self) -> int:
        return self.__base_restrictions_repetitions


class DatasetSampler:
    """
    A tool to provide sampled data sets for evaluation. It samples data sets for the specified known data rate values
    and may reuse already sampled datasets or use monotonic sampling, i.e. sample data sets such that for each pair
    d1, d2 of sampled data sets if kdr1 < kdr2, then d1 is a subset of d2.

    Parameters
    ----------
    kdr_samples: Iterable[float]
        the known data rate values to sample the dataset to
    reuse : bool
        whether to only sample once for each run (False) or to sample #runs times new queries for each run (True)
        default: False
    monotonic : bool
        whether to apply monotonic sampling as described above
        default: False
    table_samples : Iterable[Union[str, int]]
        For relational evaluations: The names or identifiers of tables that should not be sampled, i.e., are known to
        the adversary in full.
    """
    __kdr_samples: Iterable[float]
    __table_samples: Iterable[Union[str, int]]

    __reuse: bool
    __sample_monotonic: bool

    __sample_cache: Dict[Tuple[Dataset, float], Dataset]

    def __init__(self, kdr_samples: Iterable[float], reuse: bool = False, monotonic: bool = False,
                 table_samples: Iterable[Union[str, int]] = None):
        self.__reuse = reuse
        self.__sample_monotonic = monotonic

        self.__kdr_samples = kdr_samples
        self.__table_samples = table_samples

        self.__sample_cache = dict()

    def __sample(self, dataset: Union[Dataset, RelationalDatabase], pool: Optional[Pool]) \
            -> Iterator[Tuple[float, Dataset]]:
        sorted_samples = sorted(self.__kdr_samples, reverse=True)
        prev: Union[Dataset, RelationalDatabase] = dataset

        if pool is None or self.__sample_monotonic or isinstance(dataset, RelationalDatabase):
            # TODO: parallel relational sampling, currently disabled due to MySQL concurrency problems
            for kdr in sorted_samples:
                if isinstance(dataset, RelationalDatabase):
                    kdr = (kdr, self.__table_samples)
                else:
                    kdr = (kdr,)
                if self.__sample_monotonic:
                    sampled = prev.sample(*kdr)
                    prev = sampled
                else:
                    sampled = dataset.sample(*kdr)

                yield kdr[0], sampled
        else:
            if isinstance(dataset, RelationalDatabase):
                yield from iter(pool.starmap(func=lambda rate: (rate, dataset.sample(rate, self.__table_samples)),
                                             iterable=map(lambda rate: (rate,), sorted_samples)))
            else:
                yield from iter(pool.starmap(func=lambda rate: (rate, dataset.sample(rate)),
                                             iterable=map(lambda rate: (rate,), sorted_samples)))

    def sample(self, datasets: Iterable[Dataset], pool: Optional[Pool] = None) \
            -> Iterator[Tuple[Dataset, float, Dataset]]:
        """
        Yields sub samples of the given data set for the configured known data rates.

        Parameters
        ----------
        datasets : Iterable[Dataset]
            the data sets to sample from
        pool : Optional[Pool]
            the thread pool for parallel processing if activated for the evaluation
        """
        if not self.__reuse or len(self.__sample_cache) == 0:
            for dataset in datasets:
                log.info(f"Sampling dataset '{dataset.name()}' to configured known data rates")
                for kdr, sampled in self.__sample(dataset, pool):
                    if self.__reuse:
                        self.__sample_cache[(dataset, kdr)] = sampled

                    yield dataset, kdr, sampled
        else:
            log.info("Reusing sampled datasets, no sampling necessary")
            yield from [(key[0], key[1], sampled) for key, sampled in self.__sample_cache.items()]

    def reuse(self) -> bool:
        return self.__reuse

    def set_reuse(self, reuse: bool) -> None:
        """Sets reuse and removes the prior sample cache"""
        self.__reuse = reuse
        self.__sample_cache = dict()


class QuerySelector:
    """
    Provides means to select query sequences from a specific type of QuerySpace with a specific selectivity.

    Parameters
    ----------
    query_space : Type[QuerySpace]
        the type of query space to use
    query_log : KeywordQueryLog
        the query log to fill the query space (if necessary)
        default: None
    selectivity : Selectivity
        the required selectivity of the keywords
        default: Selectivity.High
    query_space_size : int
        desired size of the query space
        default: 500
    queries : int
        desired length of the query sequence
        default: 150
    allow_repetition : bool
        if queries can appear multiple times
        default: False
    """
    __query_space: Type[KeywordQuerySpace]
    __query_space_size: int
    __query_log: KeywordQueryLog

    __selectivity: Selectivity
    __queries: int

    __query_space_cache: Dict[Tuple[int, int], QuerySpace]

    __allow_repetition: bool

    def __init__(self, query_space: Type[KeywordQuerySpace], query_log: KeywordQueryLog = None,
                 selectivity: Selectivity = Selectivity.High, query_space_size: int = 500, queries: int = 150,
                 allow_repetition: bool = False):
        self.__query_space = query_space
        self.__query_log = query_log
        self.__query_space_size = query_space_size
        self.__selectivity = selectivity

        self.__queries = queries

        self.__allow_repetition = allow_repetition

        self.__query_space_cache = dict()

    def user_count(self) -> int:
        """
        Shows how many users' queries are included in the query spaces.
        """
        if self.__query_log is None or not self.__query_space.is_multi_user():
            return 1
        else:
            return len(self.__query_log.user_ids())

    def select(self, full: Dataset, known: Dataset) -> Iterator[List[str]]:
        """
        Selects query sequences of the configured length, reusing query spaces that might have already been created.

        Parameters
        ----------
        full : Dataset
            the full data set
        known : Dataset
            the known data set

        Returns
        -------
        select : Iterator[List[str]]
            a sequence of queries from the query space, one for each user
        """
        key = (id(full), id(known))
        if key not in self.__query_space_cache:
            self.__query_space_cache[key] = self.__query_space.create(full, known, self.__selectivity,
                                                                      self.__query_space_size, self.__query_log,
                                                                      self.__allow_repetition)

        return self.__query_space_cache[key].select(self.__queries)
