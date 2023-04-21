"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber, Michael Yonli, Patrick Ehrler

"""
from abc import abstractmethod
from itertools import starmap
from logging import getLogger
from multiprocessing.pool import ThreadPool, Pool
from typing import List, Union, Iterable, Tuple, Iterator, Optional, Type

import pandas as pd

from .errors import Error
from .param import EvaluationCase, KnownDatasetSampler, QuerySelector, SampledDatasetSampler
from ..api import Attack, RangeAttack, KeywordAttack, Dataset, DataSink, RangeQuerySpace, RangeDatabase
from ..api.attack import L2KeywordDocumentAttack
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
    A KeywordAttackEvaluator can be used to run a full evaluation of one or multiple attacks on a specific dataset.
    It is capable of running multiple attacks in parallel to speed up the evaluation.

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
    _dataset_sampler: Union[KnownDatasetSampler, SampledDatasetSampler]
    _query_selector: QuerySelector

    def __init__(self, evaluation_case: EvaluationCase,
                 dataset_sampler: Union[KnownDatasetSampler, SampledDatasetSampler], query_selector: QuerySelector,
                 sinks: Union[DataSink, Iterable[DataSink]], parallelism: int = 1):
        super().__init__(evaluation_case, sinks, parallelism)
        self._dataset_sampler = dataset_sampler
        self._query_selector = query_selector

    @staticmethod
    def _evaluate(dataset: Dataset, user: int, kdr: float, attack: KeywordAttack, queries: List[str]) -> \
            Tuple[str, int, float, float]:
        # recover queries using the given attack
        recovered = attack(dataset, queries)
        # count matches
        matches = [actual == recovered for actual, recovered in zip(queries, recovered)]
        correct = matches.count(True)

        import itertools
        uncovered_queries = list(itertools.compress(recovered, matches))
        attributes = set([a.attr for a in dataset.keywords()])
        for attr in attributes:
            nr_total = 0
            nr_uncovered = 0
            for q in queries:
                if q.attr == attr:
                    nr_total = nr_total + 1
            for q in uncovered_queries:
                if q.attr == attr:
                    nr_uncovered = nr_uncovered + 1
            if nr_uncovered != 0:
                uncovered_percent = nr_uncovered / nr_total
            else:
                uncovered_percent = 0

            # write to csv
            attr_details = pd.DataFrame([[attack.name(), str(kdr), str(attr), str(nr_uncovered), str(nr_total), str(uncovered_percent)]])
            attr_details.to_csv('eval_attr_details.csv', mode='a', index=False, header=False)

        return attack.name(), user, kdr, correct / len(queries)

    def _to_inputs(self, dataset: Dataset, kdr: float, known: Dataset) -> Iterator[Tuple[Dataset, int, float, Attack,
    List]]:
        # yield input tuples for _evaluate for each attack on the given known data set and known data rate
        for i, queries in enumerate(self._query_selector.select(dataset, known)):
            for attack in self._evaluation_case.attacks():
                yield dataset, i, kdr, attack.create(known), queries

    def _produce_input(self, pool: Optional[Pool] = None) \
            -> Iterator[Tuple[Dataset, int, float, KeywordAttack, List[str]]]:
        # yield all input tuples for _evaluate either by using parallel computation or sequential computation, based on
        # whether there is a multi threading pool
        datasets = self._evaluation_case.datasets()
        if pool is None:
            for inputs in starmap(self._to_inputs, self._dataset_sampler.sample(datasets)):
                yield from inputs
        else:
            for inputs in pool.starmap(self._to_inputs, iterable=self._dataset_sampler.sample(datasets, pool)):
                yield from inputs

    def run(self) -> None:
        """Runs the evaluation"""

        log.info(f"Running {self._evaluation_case.runs()} evaluation runs with parallelism {self._parallelism}")
        log.info("Evaluated Attacks:")

        # delete output file for attr details
        import os
        if os.path.exists("eval_attr_details.csv"):
            os.remove("eval_attr_details.csv")

        # log and register all evaluated attacks with all sinks
        for attack in self._evaluation_case.attacks():
            log.info(f" - {attack.name()}")

            for sink in self._sinks:
                sink.register_series(attack.name())

        reuse: bool = self._dataset_sampler.reuse()

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
                self._dataset_sampler.set_reuse(True)

            for sample_run in range(1, sample_runs + 1):
                log.info(f"Starting evaluation {run}-{sample_run} with new queries")
                if self._parallelism == 1:
                    # do evaluation sequentially
                    performances: List[Tuple[str, int, float, float]] = []
                    for dataset, user, kdr, attack, queries in self._produce_input():
                        performances.append(KeywordAttackEvaluator._evaluate(dataset, user, kdr, attack, queries))

                else:
                    # create thread pool and do evaluation in parallel
                    with ThreadPool(processes=self._parallelism) as pool:
                        performances = pool.starmap(func=KeywordAttackEvaluator._evaluate,
                                                    iterable=self._produce_input(pool))
                        log.info("All computations completed.")

                for series, user, kdr, result in performances:
                    for sink in self._sinks:
                        sink.offer_data(series, user, kdr, result)

            log.info(f"RUN {run} COMPLETED IN {stopwatch.lap()}")

        # read attr info csv, perform aggregate, then write
        df = pd.read_csv('eval_attr_details.csv')
        df.columns = ['attack_name', 'kdr', 'attr_id', 'nr_uncovered', 'nr_total', 'uncovered_percent']
        df = df.groupby(['attack_name', 'kdr', 'attr_id'], as_index=False).mean()
        df.to_csv('eval_attr_details.csv', index=False, header=True)

        log.info("######################################################################################")
        log.info(f"Evaluation completed in {stopwatch.stop()}")

        for sink in self._sinks:
            sink.flush()


class L2KeywordDocumentAttackEvaluator(Evaluator):
    """
    A L2KeywordDocumentAttackEvaluator can be used to run a full evaluation of one or multiple attacks on a specific
    dataset. It is capable of running multiple attacks in parallel to speed up the evaluation.
    Parameters
    ----------
    evaluation_case : EvaluationCase
        the evaluation case to run, i. e. the attacks, the data set and the number of runs for each attack
    dataset_sampler : DatasetSampler
        the data set sampling settings, including the known data rate values
    sinks : Union[DataSink, Iterable[DataSink]]
        one or multiple data sinks to write performance data to
    parallelism : int
        the number of parallel threads to use in the evaluation
        default: 1
    """
    __dataset_sampler: SampledDatasetSampler

    def __init__(self, evaluation_case: EvaluationCase, dataset_sampler: SampledDatasetSampler,
                 sinks: Union[DataSink, Iterable[DataSink]], parallelism: int = 1):
        super().__init__(evaluation_case, sinks, parallelism)
        self.__dataset_sampler = dataset_sampler

    @staticmethod
    def __evaluate(known: Dataset, kdr: float, attack: L2KeywordDocumentAttack) -> \
            Tuple[str, float, float, float]:
        total_keywords = len(known.keywords())
        total_documents = len(known.doc_ids())

        # perform attack
        recovered_keywords, recovered_documents = attack(known, known.keywords())

        # count matches
        correct_keywords = [x[0] == x[1] for x in recovered_keywords].count(True)

        if recovered_documents:
            correct_documents = [x[0] == x[1] for x in recovered_documents].count(True)
        else:
            correct_documents = 0

        # calculate recovery rate
        keyword_recovery_rate = correct_keywords / total_keywords
        document_recovery_rate = correct_documents / total_documents

        return attack.name(), kdr, keyword_recovery_rate, document_recovery_rate

    def __to_inputs(self, dataset: Dataset, kdr: float, known: Dataset) -> Iterator[Tuple[Dataset, float, Attack]]:
        # yield input tuples for __evaluate for each attack on the given known data set and known data rate
        for attack in self._evaluation_case.attacks():
            yield known, kdr, attack.create(dataset)

    def __produce_input(self, pool: Optional[Pool] = None) \
            -> Iterator[Tuple[Dataset, float, L2KeywordDocumentAttack]]:
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
                    performances: List[Tuple[str, float, float, float]] = []
                    for known, kdr, attack in self.__produce_input():
                        performances.append(L2KeywordDocumentAttackEvaluator.
                                            __evaluate(known, kdr, attack))

                else:
                    # create thread pool and do evaluation in parallel
                    with ThreadPool(processes=self._parallelism) as pool:
                        performances = pool.starmap(func=L2KeywordDocumentAttackEvaluator.__evaluate,
                                                    iterable=self.__produce_input(pool))
                        log.info("All computations completed.")

                for series, kdr, keyword_recovery_rate, document_recovery_rate in performances:
                    for sink in self._sinks:
                        sink.offer_data(series_id=series, known_data_rate=kdr, user_id=0,
                                        recovery_rate=keyword_recovery_rate,
                                        document_recovery_rate=document_recovery_rate)

            log.info(f"RUN {run} COMPLETED IN {stopwatch.lap()}")

        log.info("######################################################################################")
        log.info(f"Evaluation completed in {stopwatch.stop()}")

        for sink in self._sinks:
            sink.flush()
            

RelationalAttackEvaluator = KeywordAttackEvaluator  # Works exactly the same as in the keyword case


class ErrorSimulationRelationalAttackEvaluator(KeywordAttackEvaluator):
    """
    A RelationalAttackEstimationErrorEvaluator can be used to run attacks with a specified estimation error.
    The attack init receives an additional parameter 'error_rate' on which it can mock the known data estimates.
    Important: only works properly for single known-data rate evaluations (x-axix is used for error-rates)
    It is capable of running multiple attacks in parallel to speed up the evaluation.
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
    error_rates : List[float]
        the q-error rates the attack should be evaluated on
    """

    _error_rates: List[float]

    def __init__(self, evaluation_case: EvaluationCase,
                 dataset_sampler: Union[KnownDatasetSampler, SampledDatasetSampler], query_selector: QuerySelector,
                 sinks: Union[DataSink, Iterable[DataSink]], parallelism: int = 1, error_rates: List[float] = [1.0]):
        super().__init__(evaluation_case, dataset_sampler, query_selector, sinks, parallelism)
        self._error_rates = error_rates

    def _to_inputs(self, dataset: Dataset, kdr: float, known: Dataset) -> \
            Iterator[Tuple[Dataset, int, float, Attack, List]]:
        # yield input tuples for _evaluate for each attack on the given known data set and known data rate
        for i, queries in enumerate(self._query_selector.select(dataset, known)):
            for attack in self._evaluation_case.attacks():
                for error_rate in self._error_rates:
                    yield dataset, i, error_rate, attack.create(known, error_rate), queries

    def run(self) -> None:
        """Runs the evaluation"""

        log.info(f"Running {self._evaluation_case.runs()} evaluation runs with parallelism {self._parallelism}")
        log.info("Evaluated Attacks:")

        # log and register all evaluated attacks with all sinks
        for attack in self._evaluation_case.attacks():
            log.info(f" - {attack.name()}")

            for sink in self._sinks:
                sink.register_series(attack.name())

        reuse: bool = self._dataset_sampler.reuse()

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
                self._dataset_sampler.set_reuse(True)

            for sample_run in range(1, sample_runs + 1):
                log.info(f"Starting evaluation {run}-{sample_run} with new queries")
                if self._parallelism == 1:
                    # do evaluation sequentially
                    performances: List[Tuple[str, int, float, float]] = []
                    for dataset, user, kdr, attack, queries in self._produce_input():
                        performances.append(
                                ErrorSimulationRelationalAttackEvaluator._evaluate(dataset, user, kdr, attack,
                                                                                    queries))

                else:
                    # create thread pool and do evaluation in parallel
                    with ThreadPool(processes=self._parallelism) as pool:
                        performances = pool.starmap(func=ErrorSimulationRelationalAttackEvaluator._evaluate,
                                                    iterable=self._produce_input(pool))
                        log.info("All computations completed.")

                for series, user, error_rate, result in performances:
                    for sink in self._sinks:
                        sink.offer_data(series, user, error_rate, result)

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
