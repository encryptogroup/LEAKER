"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber, Michael Yonli

"""
from abc import abstractmethod
from itertools import starmap
from logging import getLogger
from multiprocessing.pool import ThreadPool, Pool
from typing import List, Union, Iterable, Tuple, Iterator, Optional, Type

from .errors import Error
from .param import EvaluationCase, KnownDatasetSampler, QuerySelector, SampledDatasetSampler
from ..api import Attack, RangeAttack, KeywordAttack, Dataset, DataSink, RangeQuerySpace, RangeDatabase
from ..util.time import Stopwatch

log = getLogger(__name__)


class Evaluator:
    _evaluation_case: EvaluationCase
    _sinks: List[DataSink]
    _parallelism: int

    def __init__(self, evaluation_case: EvaluationCase, sinks: Union[DataSink, Iterable[DataSink]],
                 parallelism: int = 1):
        if isinstance(sinks, DataSink):
            self._sinks = [sinks]
        else:
            self._sinks = list(sinks)
        self._evaluation_case = evaluation_case
        self._parallelism = parallelism

    @abstractmethod
    def run(self):
        raise NotImplementedError


class KeywordAttackEvaluator(Evaluator):
    """
    A KeywordAttackEvaluator can be used to run a full evaluation of one or multiple attacks on a specific dataset. It is capable
    of running multiple attacks in parallel to speed up the evaluation.

    Parameters
    ----------
    evaluation_case : EvaluationCase
        the evaluation case to run, i. e. the attacks, the data set and the number of runs for each attack
    dataset_sampler : KnownDatasetSampler
        the data set sampling settings, including the known data rate values
    query_selector : QuerySelector
        the policies for selecting the query sequence including the selectivity, the type and size of the query space
        and the number of queries
    sinks : Union[DataSink, Iterable[DataSink]]
        one or multiple data sinks to write performance data to
    parallelism : int
        the number of parallel threads to use in the evaluation
        default: 1
    """
    __dataset_sampler: Union[KnownDatasetSampler,SampledDatasetSampler]
    __query_selector: QuerySelector

    def __init__(self, evaluation_case: EvaluationCase, dataset_sampler: Union[KnownDatasetSampler,SampledDatasetSampler], query_selector: QuerySelector,
                 sinks: Union[DataSink, Iterable[DataSink]], parallelism: int = 1):
        super().__init__(evaluation_case, sinks, parallelism)
        self.__dataset_sampler = dataset_sampler
        self.__query_selector = query_selector

    @staticmethod
    def __evaluate(dataset: Dataset, user: int, kdr: float, attack: KeywordAttack, queries: List[str]) -> \
            Tuple[str, int, float, float]:
        # recover queries using the given attack
        recovered = attack(dataset, queries)
        # count matches
        correct = [actual == recovered for actual, recovered in zip(queries, recovered)].count(True)

        return attack.name(), user, kdr, correct / len(queries)

    def __to_inputs(self, dataset: Dataset, kdr: float, known: Dataset) -> Iterator[Tuple[Dataset, int, float, Attack,
                                                                                          List]]:
        # yield input tuples for __evaluate for each attack on the given known data set and known data rate
        for i, queries in enumerate(self.__query_selector.select(dataset, known)):
            for attack in self._evaluation_case.attacks():
                yield dataset, i, kdr, attack.create(known), queries

    def __produce_input(self, pool: Optional[Pool] = None) \
            -> Iterator[Tuple[Dataset, int, float, KeywordAttack, List[str]]]:
        # yield all input tuples for __evaluate either by using parallel computation or sequential computation, based on
        # whether there is a multi threading pool
        datasets = self._evaluation_case.datasets()
        if pool is None:
            for inputs in starmap(self.__to_inputs, self.__dataset_sampler.sample(datasets)):
                yield from inputs
        else:
            for inputs in pool.starmap(self.__to_inputs, iterable=self.__dataset_sampler.sample(datasets, pool)):
                yield from inputs

    def run(self) -> None:
        """Runs the evaluation"""

        log.info(f"Running {self._evaluation_case.runs()} evaluation runs with parallelism {self._parallelism}")
        log.info("Evaluated Attacks:")

        # log and register all evaluated attacks with all sinks
        for attack in self._evaluation_case.attacks():
            log.info(f" - {attack.name()}")

            for sink in self._sinks:
                sink.register_series(attack.name())

        reuse: bool = self.__dataset_sampler.reuse()

        stopwatch = Stopwatch()
        stopwatch.start()

        # Perform desired number of runs
        for run in range(1, self._evaluation_case.runs() + 1):
            log.info("######################################################################################")
            log.info(f"# RUN {run}")
            log.info("######################################################################################")

            sample_runs: int = 1
            if reuse:
                sample_runs: int = self._evaluation_case.runs()
                self.__dataset_sampler.set_reuse(True)

            for sample_run in range(1, sample_runs + 1):
                log.info(f"Starting evaluation {run}-{sample_run} with new queries")
                if self._parallelism == 1:
                    # do evaluation sequentially
                    performances: List[Tuple[str, int, float, float]] = []
                    for dataset, user, kdr, attack, queries in self.__produce_input():
                        performances.append(KeywordAttackEvaluator.__evaluate(dataset, user, kdr, attack, queries))

                else:
                    # create thread pool and do evaluation in parallel
                    with ThreadPool(processes=self._parallelism) as pool:
                        performances = pool.starmap(func=KeywordAttackEvaluator.__evaluate,
                                                    iterable=self.__produce_input(pool))
                        log.info("All computations completed.")

                for series, user, kdr, result in performances:
                    for sink in self._sinks:
                        sink.offer_data(series, user, kdr, result)

            log.info(f"RUN {run} COMPLETED IN {stopwatch.lap()}")

        log.info("######################################################################################")
        log.info(f"Evaluation completed in {stopwatch.stop()}")

        for sink in self._sinks:
            sink.flush()


class RangeAttackEvaluator(Evaluator):
    """
       A RangeAttackEvaluator can be used to run a full evaluation of one or multiple attacks on a specific
       RangeDatabase. It is capable of running multiple attacks in parallel to speed up the evaluation.

       Parameters
       ----------
       evaluation_case : EvaluationCase
           the evaluation case to run, i. e. the attacks, the data set and the number of runs for each attack
       range_queries : RangeQuerySpace
           the policies for selecting the query sequence (query distribution)
       query_counts : List[int]
            the amount of queries the attack shall be evaluated on
       sinks : Union[DataSink, Iterable[DataSink]]
           one or multiple data sinks to write performance data to
       normalize : bool
            whether to normalize the displayed reconstruction errors
            default: True
       parallelism : int
           the number of parallel threads to use in the evaluation
           default: 1
       """

    __normalize: bool
    __queries: RangeQuerySpace
    __queries_n: List[int]

    def __init__(self, evaluation_case: EvaluationCase, range_queries: RangeQuerySpace, query_counts: List[int],
                 sinks: Union[DataSink, Iterable[DataSink]], normalize: bool = True, parallelism: int = 1):
        super().__init__(evaluation_case, sinks, parallelism)
        self.__normalize = normalize
        self.__queries = range_queries
        self.__queries_n = query_counts

    @staticmethod
    def _evaluate(db: RangeDatabase, attack: RangeAttack, queries: List[Tuple[int, int]], error: Type[Error],
                  normalize: bool, user: int) \
            -> Tuple[str, float, float, int]:
        recovered = attack.recover(queries)

        return attack.name(), len(queries), error.calc_error(db, recovered, normalize), user

    def _to_inputs(self, query_count: int) -> Iterator[Tuple[RangeDatabase, RangeAttack, List[Tuple[int, int]],
                                                             Type[Error], bool, int]]:
        # yield input tuples for __evaluate for each attack on the given known data set and known data rate
        for db in self._evaluation_case.datasets():
            for attack in self._evaluation_case.attacks():
                for i, queries in enumerate(self.__queries.select(query_count)):
                    yield db, attack.create(db), queries, self._evaluation_case.error(), self.__normalize, i

    def _produce_input(self, query_count: int, pool: Optional[Pool] = None) -> Iterator[Tuple[RangeDatabase,
                                                                                              RangeAttack,
                                                                                              List[Tuple[int, int]],
                                                                                              Type[Error], bool, int]]:
        # yield all input tuples for __evaluate either by using parallel computation or sequential computation, based on
        # whether there is a multi processing pool
        if pool is None:
            for inputs in map(self._to_inputs, [query_count for _ in range(self._evaluation_case.runs())]):
                yield from inputs
        else:
            for inputs in pool.map(self._to_inputs, [query_count for _ in range(self._evaluation_case.runs())]):
                yield from inputs

    def run(self) -> None:
        """Runs the evaluation"""
        log.info(f"Running {len(self.__queries_n)} query runs with parallelism {self._parallelism}")
        log.info("Evaluated Attacks:")

        # log and register all evaluated attacks with all sinks
        for attack in self._evaluation_case.attacks():
            log.info(f" - {attack.name()}")

            for sink in self._sinks:
                sink.register_series(attack.name())

        stopwatch = Stopwatch()
        stopwatch.start()

        # Perform desired number of runs for each #queries
        for query_count in self.__queries_n:
            log.info("######################################################################################")
            log.info(f"# Queries {query_count}")
            log.info("######################################################################################")

            if self._parallelism == 1:
                # do evaluation sequentially
                for db, attack, queries, error, normalize, user in self._produce_input(query_count):
                    attack_name, n_q, result, user = RangeAttackEvaluator._evaluate(db, attack, queries, error,
                                                                                    normalize, user)

                    for sink in self._sinks:
                        sink.offer_data(attack_name, user, n_q, result)
            else:
                # create pool and do evaluation of multiple runs in parallel
                with Pool(processes=self._parallelism) as pool:
                    results = pool.starmap(func=RangeAttackEvaluator._evaluate,
                                           iterable=self._produce_input(query_count, None))
                    log.info("All computations completed.")

                    for attack_name, n_q, result, user in results:
                        for sink in self._sinks:
                            sink.offer_data(attack_name, user, n_q, result)

            log.info(f"QUERY RUN {query_count} COMPLETED IN {stopwatch.lap()}")

        log.info("######################################################################################")
        log.info(f"Evaluation completed in {stopwatch.stop()}")

        for sink in self._sinks:
            sink.flush()
