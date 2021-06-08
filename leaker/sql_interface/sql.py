"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from logging import getLogger
from typing import Union, Tuple, List

import mysql.connector
from mysql.connector import Error, MySQLConnection

from ..api.constants import MYSQL_USER_NAME, MYSQL_USER_PASSWORD

log = getLogger(__name__)


"""MySQL Error Codes"""
SQL_DATABASE_ALREADY_EXISTS = 1007


class SQLConnection:
    """
    Class that opens and maintains a connection to a MySQL server.

    Parameters
        ----------
        host_name : str
            The name of the MySQL Server. Default: "localhost"
        user_name : str
            The username of the MySQL Server. Default: MYSQL_USER_NAME
        user_password : str
            The user password of the MySQL Server for user_name. Default: MYSQL_USER_PASSWORD
    """
    __connection: Union[MySQLConnection, None]
    __host_name: str
    __user_name: str
    __user_password: str

    def __init__(self, host_name: str = "localhost", user_name: str = MYSQL_USER_NAME,
                 user_password: str = MYSQL_USER_PASSWORD):
        self.__connection = None
        self.__host_name = host_name
        self.__user_password = user_password
        self.__user_name = user_name

    def execute_query(self, query: str, select: bool = False) -> Tuple[int, Union[None, List]]:
        """
        Executes a certain query. If results are expected, select has to be set to True. This is necessary because
        cursor.rowcount displays rows_affected for non-DQL (non-SELECT) queries, but we do not need them.
        :param query:
        :param select:
        :return: (ret, res): ret is the return value of the query (0 indicating success), and res is a list of the
                             results (or None if no results are returned or expected)
        """
        if not self.is_open():
            raise ConnectionError(f"MySQL Database connection to {self.__host_name} is not open!")
        res = None
        with self.__connection.cursor(buffered=True) as cursor:
            try:
                cursor.execute(query)
                if select and cursor.rowcount > 0:
                    res = [r for r in cursor.fetchall()]  # we need to consume all results before we can move on.
                ret = 0
                self.__connection.commit()
            except Error as err:
                log.debug(f"Error when performing query {query}: '{err.errno}'")
                ret = err.errno

        return ret, res

    def is_open(self) -> bool:
        if self.__connection is not None:
            return self.__connection.is_connected()
        else:
            return False

    def open(self) -> 'SQLConnection':
        try:
            self.__connection = mysql.connector.connect(
                host=self.__host_name,
                user=self.__user_name,
                passwd=self.__user_password,
                buffered=True
            )
            log.debug(f"MySQL Database connection to {self.__host_name} successful.")
        except Error as err:
            log.warning(f"MySQL Database connection to {self.__host_name} failed: {err}. Perhaps you did not set up"
                        f"a MySQL server or the LEAKER user yet.")
        return self

    def __enter__(self) -> 'SQLConnection':
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self):
        if self.is_open():
            self.__connection.close()
            self.__connection = None
