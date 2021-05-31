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
        log.debug(f"Creating relational index {self.__backend_name}.")
        with SQLConnection() as sql_connection:
            """Create new MySQL Database for name"""
            if sql_connection.execute_query(f"CREATE DATABASE {self.__name}") == SQL_DATABASE_ALREADY_EXISTS:
                log.warning(f"Overwriting existing DB {self.__name}!")
                sql_connection.execute_query(f"DROP DATABASE {self.__name}")
                sql_connection.execute_query(f"CREATE DATABASE {self.__name}")

            sql_connection.execute_query(f"USE {self.__name}")

            # Table to map table names to table ids
            sql_connection.execute_query(f"CREATE TABLE tables ("
                                         "table_id INT(8) UNSIGNED PRIMARY KEY, "
                                         "table_name VARCHAR(24))")

            prev_table_name = None
            current_table_id = -1
            for table_name, values in source:
                table_name = table_name.replace('.', "").replace('/', "_")  # Causes issues with MySQL
                if prev_table_name != table_name:
                    log.info(f"Indexing table {table_name}.")
                    current_table_id += 1
                    sql_connection.execute_query(f"INSERT INTO tables VALUES ({current_table_id}, '{table_name}')")

                    attributes = ", ".join([f"attr_{i} VARCHAR(16)" for i in range(len(values))])
                    sql_connection.execute_query(f"CREATE TABLE {table_name} ("
                    "table_id INT(8), "
                    "row_id INT(8) UNSIGNED AUTO_INCREMENT PRIMARY KEY, "
                    f"{attributes})")
                    prev_table_name = table_name

                val_str = f"{current_table_id}, default, " + ", ".join([f"'{v}'" for v in values])
                sql_connection.execute_query(f"INSERT INTO {table_name} VALUES ({val_str})")

        log.debug(f"Creation of relational index {self.__backend_name} completed.")
