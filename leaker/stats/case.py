"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from typing import List, Union, Iterable, Optional

from ..api import Dataset, KeywordQueryLog, RangeDatabase, QuerySpace, RangeQueryLog
from .types import StatisticsTypes


class StatisticsCase:
    """
    A statistic case to use with the StatisticsEvaluator. It consists of one or multiple StatisticTypes to evaluate, a
    query log and/or a data set necessary for the statistics, and restriction parameters for the data set.

    Parameters
    ----------
    statistics: Union[StatisticsType, Iterable[StatisticsType]]
        may be one or multiple elements of which each is a StatisticsType according to which statistics shall be
        gathered.
    query_data: Optional[Union[KeywordQueryLog, QuerySpace]]
        the query log or space to gather statistics from (necessity depends on the StatisticsType)
    dataset: Union[Dataset, RangeDatabase]
        the data set to gather statistics from (necessity depends on the StatisticsType)
    base_restriction_rates: Optional[Iterable[float]]
        the rates to restrict the base datasets to (useful if datasets are too large)
        default: None
    base_restrictions_repetitions: int
        how often a statistic will be gathered with fresh restricted base datasets
        default: 1
    """
    __statistics_types: List[StatisticsTypes]
    __query_data: Union[KeywordQueryLog, RangeQueryLog, QuerySpace]
    __datasets: List[Union[Dataset, RangeDatabase]]
    __full_dataset: Union[Dataset, RangeDatabase]

    __base_restrictions_repetitions: int

    def __init__(self, statistics: Union[StatisticsTypes, Iterable[StatisticsTypes]],
                 query_data: Optional[Union[KeywordQueryLog, RangeQueryLog, QuerySpace]] = None,
                 dataset: Optional[Union[Dataset, RangeDatabase]] = None,
                 base_restriction_rates: Optional[Iterable[float]] = None, base_restrictions_repetitions: int = 1):
        if base_restrictions_repetitions < 1:
            raise ValueError("Run and repetition count must be at least 1")

        if isinstance(statistics, StatisticsTypes):
            self.__statistics_types = [statistics]
        else:
            self.__statistics_types = list(statistics)

        self.__query_data = query_data
        self.__full_dataset = dataset

        if base_restriction_rates is not None:
            self.__datasets = [dataset.restrict_rate(rate) for rate in base_restriction_rates
                               for _ in range(base_restrictions_repetitions)]
        else:
            self.__datasets = [dataset]

    def query_data(self) -> Union[KeywordQueryLog, RangeQueryLog]:
        return self.__query_data

    def datasets(self) -> Iterable[Union[Dataset, RangeDatabase]]:
        yield from self.__datasets

    def full_dataset(self) -> Dataset:
        return self.__full_dataset

    def statistic_types(self) -> List[StatisticsTypes]:
        return self.__statistics_types

    def base_restrictions_repetitions(self) -> int:
        return self.__base_restrictions_repetitions
