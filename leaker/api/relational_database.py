"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from abc import abstractmethod
from collections import namedtuple
from logging import getLogger
from typing import Union, Tuple, Iterator, Optional, Set, List

from ..api import Dataset, Selectivity

log = getLogger(__name__)

RelationalQuery = namedtuple("RelationalQuery", ["id", "table", "attr", "value"])


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
    def query(self, query: RelationalQuery) -> Iterator[Tuple[int, int]]:
        """
        Yields all matches for an attr=val query on a table.

        Parameters
        ----------
        query : RelationalQuery
            id : int
                if available, the id of the query in the backend (None otherwise)
            table : int
                the identifier of the table to search on.
            attr : int
                the attribute identifier (within the table).
            val : str
                the value.

        Returns
        -------
        query : Iterator[int]
            an iterator yielding all matches (table_id,row id) for the query
        """
        raise NotImplementedError

    @abstractmethod
    def queries(self, max_queries: Optional[int], table: Optional[int], attr: Optional[int],
                sel: Optional[Selectivity]) -> List[RelationalQuery]:
        """Yields all possible queries in this data instance (possibly restricted to max_queries queries, a table and
        attribute or a selectivity. If attr is set, table needs to be set as well."""
        raise NotImplementedError

    def keywords(self):
        return self.queries()

    @abstractmethod
    def row_ids(self) -> Set[Tuple[int, int]]:
        """Returns the unique identifiers (table_id, row_id) of all entries in this DB."""
        raise NotImplementedError

    def doc_ids(self):
        return self.row_ids()

    def documents(self):  # TODO: Refactor base class
        return self.row_ids()

    def pickle(self) -> None:
        """No pickling needed, everything happens in MySQL"""
        pass

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

    def __call__(self, query: RelationalQuery) -> Iterator[Tuple[int, int]]:
        yield from self.query(query)

    def __len__(self) -> int:
        return len(self.row_ids())
