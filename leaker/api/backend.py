"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
import os
import dill as pickle
from abc import ABC, abstractmethod
from typing import Set, Iterator, Union

from .constants import PICKLE_DIRECTORY, RANGE_PICKLE_ID, RANGE_QLOG_PICKLE_ID
from .dataset import Dataset, KeywordQueryLog, Data
from .range_database import RangeDatabase, RangeQueryLog


class Backend(ABC):
    """
    A backend for loading data sets and query logs. It encapsulates a specific method of storing and querying data.
    It would usually be paired with a specific type of Database or QueryLog.

    A backend is iterable and will yield all data sets in it.
    """

    @abstractmethod
    def has(self, name: str) -> bool:
        """
        Tests whether this backend has a dataset of the given name.

        Parameters
        ----------
        name : str
            the name of the data set to check
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, name: str) -> Union[RangeDatabase, Data]:
        """
        Loads the range database with the specified name, given it exists.
        Parameters
        ----------
        name : str
            the name of the range database to load
        """
        raise NotImplementedError

    @abstractmethod
    def data_sets(self) -> Set[str]:
        """
        Returns the names of all data sets in this backend.
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Union[RangeDatabase, Dataset, KeywordQueryLog]]:
        """
        Loads all data
        """
        raise NotImplementedError


class RangeBackend(Backend):
    """
    A `Backend` for loading Range pickles as RangeDataBases.
    """

    def has(self, name: str) -> bool:
        return name in self.data_sets() or name in self.query_logs()

    def load(self, name: str) -> RangeDatabase:
        return self.load_range_database(name)

    def load_range_database(self, name: str) -> RangeDatabase:
        if not self.has(name):
            raise FileNotFoundError('Index not found: ' + name)
        filename = Data.pickle_filename(name, RANGE_PICKLE_ID)
        return RangeDatabase(name, pickle.load(open(filename, "rb")))

    def load_range_querylog(self, name: str, min_user_count: int = 0, max_user_count: int = None,
                            reverse: bool = False) -> RangeQueryLog:
        """
        Loads the range query log, if it exists
        Parameters
        ----------
        min_user_count: int, max_user_count: int
            If given, only consider queries of most_freq_users[min_user_count:max_user_count]
        reverse: bool
            If True, consider queries of least_freq_users[min_user_count:max_user_count] with  min activity
            MIN_USER_QUERYLOG_ACTIVITY"""
        if not self.has(name):
            raise FileNotFoundError('Index not found: ' + name)
        filename = Data.pickle_filename(name, RANGE_QLOG_PICKLE_ID)
        return RangeQueryLog(name, pickle.load(open(filename, "rb")), min_user_count, max_user_count, reverse)

    def __data(self, pickle_str: str) -> Set[str]:
        return set([f[:-8 - len(pickle_str)] for f in os.listdir(PICKLE_DIRECTORY)
                    if os.path.isfile(os.path.join(PICKLE_DIRECTORY, f)) and pickle_str in f])

    def data_sets(self) -> Set[str]:
        return self.__data(RANGE_PICKLE_ID)

    def query_logs(self) -> Set[str]:
        return self.__data(RANGE_QLOG_PICKLE_ID)

    def __iter__(self) -> Iterator[RangeDatabase]:
        for name in self.data_sets():
            yield self.load_range_database(name)
