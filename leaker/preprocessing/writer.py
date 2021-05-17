"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from logging import getLogger

import dill as pickle
import numpy as np

from abc import ABC, abstractmethod

from typing import List, Union, Dict, Tuple

from ..api.constants import RANGE_PICKLE_ID, RANGE_QLOG_PICKLE_ID
from .pipeline import Source, Sink
from ..api import InputDocument, Data

log = getLogger(__name__)


class DatasetWriter(Sink[InputDocument], ABC):
    """
    A sink consuming a stream of `InputDocument` for writing it to a data set. Any subclass must implement the `write`
    and `flush` method according to the underlying technology.
    """

    @abstractmethod
    def write(self, document: InputDocument) -> None:
        """
        Writes the given document to the associated data set.

        Parameters
        ----------
        document: InputDocument
            The document to be written
        """
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """
        Finishes the writing process, finalizes the data set and frees up resources if applicable.
        """
        raise NotImplementedError

    def run(self, source: Source[InputDocument]) -> None:
        for doc in source:
            self.write(doc)
        self.flush()


class RangeDatabaseWriter(Sink[List[Union[float, int]]]):
    """
    A sink consuming a stream of Lists for writing it to a range database. Everything is combined to a list
    that is pickled under the given name.
    An optional scaling factor for converting float values into ints can be supplied.
    """

    __name: str
    __scale_factor: int

    def __init__(self, name: str, scale_factor=1):
        self.__name = name
        self.__scale_factor = scale_factor

    def run(self, source: Source[List[Union[float, int]]]) -> None:
        filename = Data.pickle_filename(self.__name, RANGE_PICKLE_ID)
        log.debug(f"Creating range index {self.__name}")
        vals = [int(np.around(float(value) * self.__scale_factor, decimals=0)) for values in source for value in values]
        log.debug(f"Storing {self.__name} in {filename}")
        pickle.dump(vals, open(filename, "wb"))


class RangeQueryLogWriter(Sink[Dict[str, Dict[Tuple[int, int], int]]]):
    """
    A sink consuming a stream of Lists for writing it to a range query log. Everything is combined to a Dict of users
    associated with a Counter of queries and their frequencies. The result is pickled under the given name.
    An optional scaling factor for converting float values into ints can be supplied.
    """

    __name: str
    __scale_factor: int

    def __init__(self, name: str, scale_factor=1):
        self.__name = name
        self.__scale_factor = scale_factor

    def run(self, source: Source[Tuple[List[Union[Tuple[Union[float, None]], Tuple[Union[int, None]]]], str]]) -> None:
        filename = Data.pickle_filename(self.__name, RANGE_QLOG_PICKLE_ID)
        log.debug(f"Creating range query log {self.__name}")

        query_dict: Dict[str, Dict[Tuple[int, int], int]] = dict()

        for queries, user_id in source:
            for query in queries:
                lower, upper = query
                if lower is not None:
                    lower = int(np.around(float(lower) * self.__scale_factor, decimals=0))
                if upper is not None:
                    upper = int(np.around(float(upper) * self.__scale_factor, decimals=0))
                query = (lower, upper)

                if user_id not in query_dict:
                    query_dict[user_id] = dict()
                if query not in query_dict[user_id]:
                    query_dict[user_id][query] = 1
                else:
                    query_dict[user_id][query] += 1

        log.debug(f"Storing {self.__name} in {filename}")
        pickle.dump(query_dict, open(filename, "wb"))
