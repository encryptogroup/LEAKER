"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from logging import getLogger
from typing import Tuple, List

from ..preprocessing import Sink, Source
from . import SQLConnection
from .sql import SQL_DATABASE_ALREADY_EXISTS
from ..api.constants import MYSQL_IDENTIFIER, SQL_WRITING_INTERVAL

log = getLogger(__name__)


class SQLRelationalDatabaseWriter(Sink[List[Tuple[str, List[str]]]]):
    """
    A sink consuming a stream of Lists for writing it to a relational database. For each entry, a table is created in a
    new database (given the name) and all values in the given list are inserted.
    """

    __name: str
    __backend_name: str

    def __init__(self, name: str):
        self.__name = f"{MYSQL_IDENTIFIER}_{name}"
        self.__backend_name = name

    def run(self, source: Source[Tuple[str, List[str]]]) -> None:
        log.info(f"Creating relational index {self.__backend_name}.")
        with SQLConnection() as sql_connection:
            """Create new MySQL Database for name"""
            if sql_connection.execute_query(f"CREATE DATABASE {self.__name}")[0] == SQL_DATABASE_ALREADY_EXISTS:
                log.warning(f"Overwriting existing DB {self.__name}!")
                sql_connection.execute_query(f"DROP DATABASE {self.__name}")
                sql_connection.execute_query(f"CREATE DATABASE {self.__name}")

            sql_connection.execute_query(f"USE {self.__name}")

            # Table to map table names to table ids
            sql_connection.execute_query("CREATE TABLE tables ("
                                         "table_id INT(8) UNSIGNED PRIMARY KEY, "
                                         "table_name VARCHAR(24))")

            # Table to store queries and their selectivity
            sql_connection.execute_query(f"CREATE TABLE queries ("
                                         "query_id INT(8) UNSIGNED PRIMARY KEY, "
                                         "table_id INT(8) UNSIGNED, "
                                         "FOREIGN KEY (table_id) REFERENCES tables(table_id), "
                                         "attr_id INT(8) UNSIGNED, "
                                         "val VARCHAR(32), "
                                         "selectivity INT(10) UNSIGNED)")

            # Table to store queries and their responses (one entry per response)
            sql_connection.execute_query("CREATE TABLE queries_responses ("
                                         "query_id INT(8) UNSIGNED, "
                                         "FOREIGN KEY (query_id) REFERENCES queries(query_id), "
                                         "table_id INT(8) UNSIGNED, "
                                         "FOREIGN KEY (table_id) REFERENCES tables(table_id), "
                                         "row_id INT(10) UNSIGNED)")

            prev_table_name = None
            current_table_id = -1
            current_table_queries = dict()  # Enumerate all possible queries and store their results
            current_row_id = 0  # within the table
            current_query_id = 0  # across all tables
            for table_name, values in source:
                table_name = table_name.replace('.', "").replace('/', "_")  # Causes issues with MySQL
                if prev_table_name != table_name:
                    if len(current_table_queries) > 0:
                        log.info(f"Indexing {len(current_table_queries)} queries for {prev_table_name}.")

                    items = list(current_table_queries.items())
                    items.sort(key=lambda t: len(t[1]), reverse=True)
                    queries = list(set(resp[0] for resp in items))

                    for pos in range(0, len(queries), SQL_WRITING_INTERVAL):
                        sql_connection.execute_query(
                            f"INSERT INTO queries VALUES " + ",".join(f"({current_query_id + pos + i}, "  # query id
                                                                      f"{current_table_id}, "
                                                                      f"{q[0]}, "
                                                                      f"'{q[1][:32]}', "  # ensure we don't exceed size
                                                                      f"{len(current_table_queries[q])})" for i, q in
                                                                      enumerate(queries[pos:pos +
                                                                                            SQL_WRITING_INTERVAL])))

                    responses = [(current_query_id + i, row_id) for i, q in enumerate(queries)
                                 for row_id in current_table_queries[q]]
                    current_query_id += len(queries)
                    for pos in range(0, len(responses), SQL_WRITING_INTERVAL):
                        sql_connection.execute_query(f"INSERT INTO queries_responses VALUES " +
                                                     ",".join(f"({i}, {current_table_id}, {row_id})"
                                                              for i, row_id in
                                                              responses[pos:pos + SQL_WRITING_INTERVAL]))

                    current_table_queries = dict()
                    current_row_id = 0

                    log.info(f"Indexing table {table_name}.")
                    current_table_id += 1
                    sql_connection.execute_query(f"INSERT INTO tables VALUES ({current_table_id}, '{table_name}')")

                    prev_table_name = table_name

                for i, val in enumerate(values):  # Add queries
                    if val == '':  # skip NULL values (we see them as invalid queries)
                        continue
                    val = val.replace("\'", "")  # may mess up SQL queries
                    if (i, val) not in current_table_queries.keys():
                        current_table_queries[(i, val)] = [current_row_id]
                    else:
                        current_table_queries[(i, val)].append(current_row_id)

                current_row_id += 1

            if len(current_table_queries) > 0:
                log.info(f"Indexing {len(current_table_queries)} queries for {table_name}.")
                items = list(current_table_queries.items())
                items.sort(key=lambda t: len(t[1]), reverse=True)

                # final flush of current_table_queries
                queries = list(set(resp[0] for resp in items))

                for pos in range(0, len(queries), SQL_WRITING_INTERVAL):
                    sql_connection.execute_query(
                        f"INSERT INTO queries VALUES " + ",".join(f"({current_query_id + pos + i}, "  # query id
                                                                  f"{current_table_id}, "
                                                                  f"{q[0]}, "
                                                                  f"'{q[1][:32]}', "  # ensure we don't exceed size
                                                                  f"{len(current_table_queries[q])})" for i, q in
                                                                  enumerate(queries[pos:pos +
                                                                                        SQL_WRITING_INTERVAL])))

                responses = [(current_query_id + i, row_id) for i, q in enumerate(queries)
                             for row_id in current_table_queries[q]]
                current_query_id += len(queries)
                for pos in range(0, len(responses), SQL_WRITING_INTERVAL):
                    sql_connection.execute_query(f"INSERT INTO queries_responses VALUES " +
                                                 ",".join(f"({i}, {current_table_id}, {row_id})"
                                                          for i, row_id in
                                                          responses[pos:pos + SQL_WRITING_INTERVAL]))

        log.info(f"Creation of relational index {self.__backend_name} completed.")
