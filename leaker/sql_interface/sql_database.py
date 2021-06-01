"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from logging import getLogger
from typing import Set, Iterator, Optional, Union, Tuple, List

from ..api import RelationalDatabase, RelationalQuery
from ..api.constants import MYSQL_IDENTIFIER, Selectivity
from . import SQLConnection


log = getLogger(__name__)


class SQLRelationalDatabase(RelationalDatabase):
    """
    A `RelationalDatabase` implementation relying on a MySQL index. This class should not be created directly but only
    loaded using the `SQLBackend`.

    When extending a data set of this type with an `Extension`, all sampled or restricted data sets that stem from this
    data set will automatically be extended with their respective sampled or restricted version of the extension, too.

    Parameters
    ----------
    name: str
        the name of the database on the MySQL server
    is_sampled_or_restricted: bool
        whether this data set is a sample or restriction of the full data set. Default: False
    """
    __name: str
    __backend_name: str

    _sql_connection: SQLConnection
    _row_ids: Set[str]

    _is_sampled_or_restricted: bool

    def __init__(self, name: str, is_sampled_or_restricted: bool = False):
        log.info(f"Loading {name}.")
        super(SQLRelationalDatabase, self).__init__()
        self._is_sampled_or_restricted = is_sampled_or_restricted
        self.__name = name
        self.__backend_name = f"{MYSQL_IDENTIFIER}_{name}"

        self._sql_connection = SQLConnection()
        log.info(f"Loading {name} completed.")

    def query(self, query: RelationalQuery) -> Iterator[int]:
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
            an iterator yielding all matches (row ids) for the query
        """
        if query.id is not None:
            ret, res = self._sql_connection.execute_query(f"SELECT row_id FROM queries_responses "
                                                          f"WHERE query_id = {query.id}", select=True)
        else:
            ret, res = self._sql_connection.execute_query(f"SELECT row_id FROM queries, queries_responses  "
                                                          f"WHERE queries_responses.query_id = queries.query_id AND "
                                                          f"queries.table_id = {query.table} AND "
                                                          f"queries.attr_id = {query.attr} AND "
                                                          f"queries.val = '{query.value}'", select=True)

        if res is not None:
            for r in res:
                yield r[0]

    def queries(self, max_queries: Optional[int] = None, table: Optional[int] = None, attr: Optional[int] = None,
                sel: Optional[Selectivity] = None) -> List[RelationalQuery]:
        """Yields all possible queries in this data instance (possibly restricted to max_queries queries, a table and
        attribute or a selectivity. If attr is set, table needs to be set as well."""
        stmt = f"SELECT query_id, table_id, attr_id, val FROM queries"
        if table is not None:
            stmt += f" WHERE table_id = {table}"
            if attr is not None:
                stmt += f" AND attr_id = {attr}"

        if sel is not None:
            if sel == Selectivity.High:
                stmt += f" ORDER BY selectivity DESC"
            elif sel == Selectivity.Low:
                stmt += f" ORDER BY selectivity ASC"

        if max_queries is not None:
            stmt += f" LIMIT {max_queries}"

        queries = []

        _, res = self._sql_connection.execute_query(stmt, select=True)
        if res is not None:
            for query_id, table_id, attr_id, val in res:
                queries.append(RelationalQuery(query_id, table_id, attr_id, val))
        return queries

    def row_ids(self) -> Set[Tuple[int, int]]:
        """Returns the unique identifiers (table_id, row_id) of all entries in this DB."""
        ret, res = self._sql_connection.execute_query(f"SELECT table_id, row_id FROM queries_responses", select=True)
        if res is not None:
            return set(res)
        else:
            return set()

    def name(self) -> str:
        return self.__backend_name

    def is_open(self) -> bool:
        return self._sql_connection.is_open()

    def open(self) -> 'SQLRelationalDatabase':
        self._sql_connection.open()
        self._sql_connection.execute_query(f"USE {self.__name}")
        return self

    def close(self) -> None:
        self._sql_connection.close()

    def restrict_keyword_size(self, max_keywords: int = 0,
                              selectivity: Selectivity = Selectivity.Independent) -> 'SQLRelationalDatabase':
        """TODO: Implement restriction"""
        pass

    def restrict_rate(self, rate: float) -> 'SQLRelationalDatabase':
        """TODO: Implement restriction"""
        pass

    def restriction_rate(self) -> float:
        return 1

    def sample(self, rate: float, tables: Optional[Iterator[Union[str, int]]]) -> 'SQLRelationalDatabase':
        """TODO: Implement sampling"""
        pass

    def sample_rate(self) -> float:
        return 1

    def selectivity(self,  query: RelationalQuery) -> int:
        if query.id is not None:
            ret, res = self._sql_connection.execute_query(f"SELECT selectivity FROM queries "
                                                          f"WHERE query_id = {query.id}", select=True)
        else:
            ret, res = self._sql_connection.execute_query(f"SELECT selectivity FROM queries "
                                                          f"WHERE table_id = {query.table} AND "
                                                          f"attr_id = {query.attr} AND "
                                                          f"val = '{query.value}'", select=True)

        if res is not None:
            return res[0][0]
