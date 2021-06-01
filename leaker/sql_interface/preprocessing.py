"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from logging import getLogger
from typing import Tuple, List

from ..preprocessing import Sink, Source
from . import SQLConnection
from .sql import SQL_DATABASE_ALREADY_EXISTS
from ..api.constants import MYSQL_IDENTIFIER

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
                    for query, responses in current_table_queries.items():  # flush current_table_queries
                        sql_connection.execute_query(f"INSERT INTO queries VALUES "
                                                     f"({current_query_id}, {current_table_id}, {query[0]}, "
                                                     f"'{query[1]}', {len(responses)})")
                        for row_id in responses:
                            sql_connection.execute_query(f"INSERT INTO queries_responses VALUES "
                                                         f"({current_query_id}, {current_table_id}, {row_id})")

                        current_query_id += 1

                    current_table_queries = dict()
                    current_row_id = 0

                    log.info(f"Indexing table {table_name}.")
                    current_table_id += 1
                    sql_connection.execute_query(f"INSERT INTO tables VALUES ({current_table_id}, '{table_name}')")

                    attributes = ", ".join([f"attr_{i} VARCHAR(32)" for i in range(len(values))])
                    sql_connection.execute_query(f"CREATE TABLE {table_name} ("
                    "table_id INT(8), "
                    "row_id INT(8) UNSIGNED PRIMARY KEY, "
                    f"{attributes})")

                    prev_table_name = table_name

                # Add values
                val_str = f"{current_table_id}, {current_row_id}, " + ", ".join([f"'{v}'" for v in values])
                if sql_connection.execute_query(f"INSERT INTO {table_name} VALUES ({val_str})")[0] != 0:
                    log.warning(f"Skipping entry not compatible with MySQL!")
                else:
                    for i, val in enumerate(values):  # Add queries
                        if (i, val) not in current_table_queries.keys():
                            current_table_queries[(i, val)] = [current_row_id]
                        else:
                            current_table_queries[(i, val)].append(current_row_id)
                    current_row_id += 1

            if len(current_table_queries) > 0:
                log.info(f"Indexing {len(current_table_queries)} queries for {table_name}.")
            for query, responses in current_table_queries.items():  # final flush of current_table_queries
                sql_connection.execute_query(f"INSERT INTO queries VALUES "
                                                 f"({current_query_id}, {current_table_id}, {query[0]}, '{query[1]}', "
                                                 f"{len(responses)})")
                for row_id in responses:
                    sql_connection.execute_query(f"INSERT INTO queries_responses VALUES "
                                                     f"({current_query_id}, {current_table_id}, {row_id})")

                current_query_id += 1

        log.info(f"Creation of relational index {self.__backend_name} completed.")
