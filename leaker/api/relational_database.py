"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from abc import ABC, abstractmethod
from collections import namedtuple
from logging import getLogger
from typing import Union, Tuple, Iterator, Optional, Set

from ..api import Dataset, Selectivity

log = getLogger(__name__)


RelationalQuery = namedtuple("RelationalQuery", ["table", "attr", "value"])


class RelationalDatabase(Dataset):
    """
    A class encompassing relational queries of the type attr=val. Can be extended using an Extension to pre-compute
    results.
    """

    @abstractmethod
    def name(self) -> str:
        """Returns the name of this data instance"""
        raise NotImplementedError

    @abstractmethod
    def query(self, query: RelationalQuery) -> Iterator[int]:
        """
        Yields all matches for an attr=val query on a table.

        Parameters
        ----------
        query : RelationalQuery
            table : Union[str, int]
                the name or identifier of the table to search on.
            attr : Union[str, int]
                the attribute name or its identifier (within the table).
            val : Union[str, int]
                the value or its identifier.

        Returns
        -------
        query : Iterator[int]
            an iterator yielding all matches (row ids) for the query
        """
        raise NotImplementedError

    @abstractmethod
    def queries(self, table: Optional[Union[str, int]], attr: Optional[Union[str, int]], sel: Optional[Selectivity]) \
            -> Set[RelationalQuery]:
        """Yields all possible queries in this data instance (possibly restricted to a table and attribute or a
        selectivity. If attr is set, table needs to be set as well."""
        raise NotImplementedError

    @abstractmethod
    def row_ids(self) -> Set[Tuple[int, int]]:
        """Returns the unique identifiers (table_id, row_id) of all entries in this DB."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, rate: float, tables: Optional[Iterator[Union[str, int]]]) -> 'RelationalDatabase':
        """
        Samples this database to the given percentage. This method is used to sample base data sets to known data rates
        to simulate partial knowledge of the full database.
        For each table not in tables, rate*|table| rows will be sampled that are assumed to be known. All tables in
        in the Iterator tables are assumed to be known.

        Parameters
        ----------
        rate : float
            the sample rate in [0, 1]
        tables : Optional[Iterator[Union[str, int]]]
            tables or their identifiers that are assumed to be known in full to the adversary.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_rate(self) -> float:
        """The rate at which this data set was sampled, relative to the full data set"""
        raise NotImplementedError

    @abstractmethod
    def is_open(self) -> bool:
        """Returns true if this data instance was opened before"""
        raise NotImplementedError

    @abstractmethod
    def open(self) -> 'RelationalDatabase':
        """Opens the data instance, i. e. may allocate resources, if applicable"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Closes this data instance, i. e. may free resources, if applicable"""
        raise NotImplementedError

    def __call__(self, query: RelationalQuery) -> Iterator[int]:
        yield from self.query(query)

    def __len__(self) -> int:
        return len(self.row_ids())
