"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from typing import Set

from ..api import RelationalDatabase
from . import SQLConnection


class SQLRelationalDatabase(RelationalDatabase):
    """
    A `RelationalDatabase` implementation relying on a MySQL index. This class should not be created directly but only
    loaded using the `SQLBackend`.

    When extending a data set of this type with an `Extension`, all sampled or restricted data sets that stem from this
    data set will automatically be extended with their respective sampled or restricted version of the extension, too.

    Parameters
    ----------
    name: str
        the name of the database
    is_sampled_or_restricted: bool
        whether this data set is a sample or restriction of the full data set. Default: False
    """
    __name: str

    _sql_connection: SQLConnection
    _row_ids: Set[str]

    _is_sampled_or_restricted: bool

    def __init__(self, name: str, is_sampled_or_restricted: bool = False):
        super(SQLRelationalDatabase, self).__init__()
        self._is_sampled_or_restricted = is_sampled_or_restricted
        self.__name = name

        self._sql_connection = SQLConnection()

    def name(self) -> str:
        return self.__name

    def is_open(self) -> bool:
        return self._sql_connection.is_open()

    def open(self) -> 'SQLRelationalDatabase':
        self._sql_connection.open()
        return self

    def close(self) -> None:
        self._sql_connection.close()
