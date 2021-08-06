"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from logging import getLogger
from math import ceil
from random import sample
from typing import Set, Iterator, Optional, Union, Tuple, List, Dict, TypeVar, Type

from ..api import RelationalDatabase, RelationalQuery, Extension
from ..api.constants import MYSQL_IDENTIFIER, Selectivity
from . import SQLConnection
from ..extension import IdentityExtension, SelectivityExtension


log = getLogger(__name__)
T = TypeVar("T", bound=Extension, covariant=True)


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

    _is_sampled_or_restricted: bool

    _table_row_ids: Dict[int, Set[Tuple[int, int]]]
    _queries: Union[None, Dict[Tuple[int, int], List[RelationalQuery]]]  # maps row_id to queries returning it
    _tables: Set[int]
    _tables_ids: Dict[str, int]

    def __init__(self, name: str, is_sampled_or_restricted: bool = False):
        if not is_sampled_or_restricted:
            log.info(f"Loading {name}.")
        super(SQLRelationalDatabase, self).__init__()
        self._is_sampled_or_restricted = is_sampled_or_restricted
        self.__name = name
        self.__backend_name = f"{MYSQL_IDENTIFIER}_{name}"

        self._sql_connection = SQLConnection()
        self._queries = dict()

        self._tables_ids = dict()
        self._table_row_ids = dict()

        with self:
            self._tables = set(q.table for q in self.queries())
            for table_name in self.tables():
                ret, res = self._sql_connection.execute_query(f"SELECT table_id FROM tables "
                                                              f"WHERE table_name = '{table_name}'", select=True)
                self._tables_ids[table_name] = res[0][0]
                self._table_row_ids[self._tables_ids[table_name]] = set(r for r in self.row_ids() if r[0]
                                                                        == self._tables_ids[table_name])

        if not is_sampled_or_restricted:
            log.info(f"Loading {name} completed.")

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
        query : Iterator[Tuple[int, int]]
            an iterator yielding all matches (table_id,row id) for the query
        """
        use_ext = False
        if self.has_extension(IdentityExtension):
            if query in self.get_extension(IdentityExtension).get_identity_cache().keys():
                use_ext = True
                yield from self.get_extension(IdentityExtension).doc_ids(query)

        if not use_ext:
            if query.id is not None:
                ret, res = self._sql_connection.execute_query(f"SELECT table_id, row_id FROM queries_responses "
                                                              f"WHERE query_id = {query.id}", select=True)
            else:
                ret, res = self._sql_connection.execute_query(f"SELECT queries.table_id, row_id FROM "
                                                              f"queries, queries_responses  "
                                                              f"WHERE queries_responses.query_id = queries.query_id AND "
                                                              f"queries.table_id = {query.table} AND "
                                                              f"queries.attr_id = {query.attr} AND "
                                                              f"queries.val = '{query.value}'", select=True)

            if res is not None:
                for r in res:
                    yield r[0], r[1]

    def queries(self, max_queries: Optional[int] = None, table: Optional[int] = None, attr: Optional[int] = None,
                sel: Optional[Selectivity] = None) -> List[RelationalQuery]:
        """Yields all possible queries in this data instance (possibly restricted to max_queries queries, a table and
        attribute or a selectivity. If attr is set, table needs to be set as well."""
        no_restrictions = max_queries is None and table is None and attr is None and sel is None
        if len(self._queries) == 0 or not no_restrictions:
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
                elif sel == Selectivity.PseudoLow:
                    lb = int(round(0.015 * len(self)))
                    ub = int(round(0.02 * len(self)))
                    if table is not None:
                        stmt += f" AND selectivity BETWEEN {lb} AND {ub}"
                    else:
                        stmt += f" WHERE selectivity BETWEEN {lb} AND {ub}"

            if max_queries is not None:
                stmt += f" LIMIT {max_queries}"

            queries = []

            _, res = self._sql_connection.execute_query(stmt, select=True)
            if res is not None:
                for query_id, table_id, attr_id, val in res:
                    queries.append(RelationalQuery(query_id, table_id, attr_id, val))
            return queries
        else:
            return list(set(q for queries in self._queries.values() for q in queries))

    def tables(self) -> Iterator[str]:
        """
        :return: tables: names of the table in the database
        """
        ret, res = self._sql_connection.execute_query(f"SELECT DISTINCT table_name from tables", select=True)
        if res is not None:
            for r in res:
                yield r[0]

    def table_id(self, table_name: str) -> int:
        """
        :param table_name: String of the table
        :return: table_id: internal id of the table
        """
        return self._tables_ids[table_name]

    def row_ids(self, table_id: Optional[int] = None) -> Set[Tuple[int, int]]:
        """Returns the unique identifiers (table_id, row_id) of all entries in this DB. Can be restricted to a
        table_id"""
        if table_id is not None:
            return self._table_row_ids[table_id]
        if len(self._queries) == 0:
            ret, res = self._sql_connection.execute_query(f"SELECT queries.table_id, queries.query_id, attr_id, val, "
                                                          f"row_id "
                                                          f"FROM queries_responses "
                                                          f"INNER JOIN queries on "
                                                          f"queries.query_id = queries_responses.query_id",
                                                          select=True)
            if res is not None:
                for table_id, query_id, attr_id, val, row_id in res:
                    query = RelationalQuery(query_id, table_id, attr_id, val)
                    if (table_id, row_id) not in self._queries:
                        self._queries[(table_id, row_id)] = [query]
                    else:
                        self._queries[(table_id, row_id)].append(query)
            else:
                return set()
        return set(self._queries.keys())

    def name(self) -> str:
        return self.__name

    def is_open(self) -> bool:
        return self._sql_connection.is_open()

    def open(self) -> 'SQLRelationalDatabase':
        if not self.is_open():
            self._sql_connection.open()
            self._sql_connection.execute_query(f"USE {self.__backend_name}")
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

    def sample(self, rate: float, tables: Optional[Iterator[Union[str, int]]] = None) -> 'SQLRelationalDatabase':
        if rate > 1 or rate < 0:
            raise ValueError("Sample rate must be in (0, 1]")

        if rate == 1:
            return self

        ignored_tables = set()
        if tables is not None:
            for t in tables:
                if isinstance(t, str):
                    ignored_tables.add(self._tables_ids[t])
                else:
                    ignored_tables.add(t)

        sampled_table_row_ids = dict()
        for t in self._tables:
            if t in ignored_tables:
                sampled_table_row_ids[t] = self._table_row_ids[t]
            else:
                sample_size = ceil(len(self._table_row_ids[t]) * rate)
                sampled_table_row_ids[t] = sample(population=self._table_row_ids[t], k=sample_size)

        return SampledSQLRelationalDatabase(self, rate, sampled_table_row_ids)

    def sample_rate(self) -> float:
        return 1

    def selectivity(self,  query: RelationalQuery) -> int:
        if self.has_extension(SelectivityExtension):
            if query in self.get_extension(SelectivityExtension).get_identity_cache().keys():
                return self.get_extension(SelectivityExtension).selectivity(query)

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


class SampledSQLRelationalDatabase(SQLRelationalDatabase):
    """
    A sub sample of a `SQLRelationalDatabase`. It filters all query results by a set of document identifiers contained
    in this sub set. Instances of this class should not be created directly, but only by sampling from a
    `SQLRelationalDatabase`.

    When extending a data set of this type with an `Extension`, actually the parent (full) data set will be extended
    and thus, all other sampled data sets will be extended with their respective sampled versions of the extension, too.

    Parameters
    ----------
    parent: SQLRelationalDatabase
        the full data set
    rate: float
        the sample rate in (0,1)
    table_row_ids: Dict[int, Set[Tuple[int, int]]]
        the identifiers of all rows contained in this sub sample in a dict separated by table_ids
    """
    __parent: SQLRelationalDatabase

    __rate: float

    def __init__(self, parent: SQLRelationalDatabase, rate: float, table_row_ids: Dict[int, Set[Tuple[int, int]]]):
        log.info(f"Sampling SQL Index '{parent.name()}' at rate {rate:.3f}")

        self.__parent = parent
        self.__rate = rate

        super(SampledSQLRelationalDatabase, self).__init__(parent.name(), is_sampled_or_restricted=True)
        row_ids = set(row_id for row_ids in table_row_ids.values() for row_id in row_ids)
        self._queries = {row: queries for row, queries in self._queries.items() if row in row_ids}
        self._table_row_ids = table_row_ids

        log.info(f"Sampling extensions for '{self.name()}'.")
        self._set_extensions(map(lambda ext: ext.sample(self), parent._get_extensions()))

        log.info(f"Sampling SQL Index '{self.name()}' complete")

    def name(self) -> str:
        return f"{super(SampledSQLRelationalDatabase, self).name()}@{self.__rate}"

    def sample(self, rate: float, tables: Optional[Iterator[Union[str, int]]]) -> RelationalDatabase:
        if rate > 1 or rate < 0:
            raise ValueError("Sample rate must be in (0, 1]")

        if rate == 1:
            return self

        ignored_tables = set()
        if tables is not None:
            for t in tables:
                if isinstance(t, str):
                    ignored_tables.add(self._tables_ids[t])
                else:
                    ignored_tables.add(tables)

        sampled_table_row_ids = dict()
        for t in self._tables:
            if t in ignored_tables or self.__rate == rate:
                sampled_table_row_ids[t] = self._table_row_ids[t]
            else:
                sample_size = ceil(len(self._table_row_ids[t]) / self.__rate * rate)
                if rate < self.__rate:
                    sampled_table_row_ids[t] = sample(population=self._table_row_ids[t], k=sample_size)
                elif rate > self.__rate:
                    population = self.__parent._table_row_ids[t].difference(self._table_row_ids[t])
                    sampled_table_row_ids[t] = self._table_row_ids[t].union(
                        sample(population=population, k=sample_size - len(self._table_row_ids[t])))
        return SampledSQLRelationalDatabase(self, rate, sampled_table_row_ids)

    def sample_rate(self) -> float:
        return self.__rate

    def extend_with(self, extension: Type[T], **kwargs) -> 'SampledSQLRelationalDatabase':
        if not self.has_extension(extension):
            if not self.__parent.has_extension(extension):
                self.__parent.extend_with(extension, **kwargs)

            new_ext = self.__parent.get_extension(extension)

            extensions = self._get_extensions()
            extensions.append(new_ext.sample(self))
            self._set_extensions(extensions)
        return self

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
        query : Iterator[Tuple[int, int]]
            an iterator yielding all matches (table_id,row id) for the query
        """
        use_ext = False
        if self.has_extension(IdentityExtension):
            if query in self.get_extension(IdentityExtension).get_identity_cache().keys():
                use_ext = True
                yield from self.get_extension(IdentityExtension).doc_ids(query)

        if not use_ext:
            yield from set(self.__parent.query(query)).intersection(self._table_row_ids[query.table])

    def selectivity(self,  query: RelationalQuery) -> int:
        if self.has_extension(SelectivityExtension):
            if query in self.get_extension(SelectivityExtension).get_identity_cache().keys():
                return self.get_extension(SelectivityExtension).selectivity(query)
        return sum(1 for _ in self.query(query))
