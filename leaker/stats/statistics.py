"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import os

import dill as pickle

from collections import namedtuple
from logging import getLogger

from typing import List, Union

from ..api.constants import PICKLE_DIRECTORY
from ..api import QuerySpace, KeywordQueryLog, Dataset, RangeDatabase, RangeQueryLog
from leaker.stats import StatisticsCase

log = getLogger(__name__)


class Statistics:
    """
    An evaluation class to gather statistics specified by a StatisticsCase.

    Parameters
    ----------
    statistics_case: StatisticsCase
        the statistics case to run, i. e. which stats to gather on which data
    file_description: str
        description string for result pickle filenames
    parallelism: int
        the number of parallel threads to use in the evaluation (NOT IMPLEMENTED YET)
        default: 1
    """

    __statistics_case: StatisticsCase
    __parallelism: int
    __file_description: str

    __query_space: Union[None, QuerySpace]

    def __init__(self, statistics_case: StatisticsCase, file_description: str = "", parallelism: int = 1):
        self.__statistics_case = statistics_case

        self.__parallelism = parallelism

        self.__file_description = file_description

    def compute(self) -> List[Union[None, namedtuple]]:
        res: List[Union[None, namedtuple]] = []
        query_data = self.__statistics_case.query_data()
        for stat_type in self.__statistics_case.statistic_types():
            data: List[Union[Union[RangeQueryLog, KeywordQueryLog, QuerySpace], Union[Dataset, RangeDatabase]]] = []
            target: str = ""
            if query_data is not None:
                if isinstance(query_data, RangeQueryLog):
                    target += query_data.name()
                else:
                    target += str(query_data)
            if KeywordQueryLog in stat_type.required_input_data() or RangeQueryLog in stat_type.required_input_data()\
                    or QuerySpace in stat_type.required_input_data():
                data.append(query_data)
            if Dataset in stat_type.required_input_data() or RangeDatabase in stat_type.required_input_data():
                data.extend(self.__statistics_case.datasets())
                if query_data is not None:
                    target += " and "
                target += f"{[dataset.name() for dataset in self.__statistics_case.datasets()]}"

            log.info(f"Computing statistic {stat_type.name()} on {target}. This might take a while.")

            stat_type.offer_data(data)
            log.debug(f"Offered data to statistic {stat_type.name()}")
            res.append(stat_type.gather())
            log.info(f"Done computing {stat_type.name()}")

        if not os.path.exists(PICKLE_DIRECTORY):
            os.makedirs(PICKLE_DIRECTORY)

        fname = PICKLE_DIRECTORY + "results"
        if self.__file_description != "":
            fname += f"_{self.__file_description}"
        fname += ".pickle"
        log.info(f"Gathered all statistics. Dumping results in {fname}")
        pickle.dump(res, open(fname, "wb"))

        return res
