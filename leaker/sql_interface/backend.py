"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from typing import Set, Iterator

from . import SQLConnection, SQLRelationalDatabase
from ..api import Backend
from ..api.constants import MYSQL_IDENTIFIER


class SQLBackend(Backend):
    """
    A `Backend` for loading SQL data as a backend. It is associated with the `SQLRelationalDatabase` class.
    """

    def has(self, name: str) -> bool:
        with SQLConnection() as conn:
            ret, _ = conn.execute_query(f"USE {MYSQL_IDENTIFIER}_{name}")
            return ret == 0

    def load(self, name: str) -> SQLRelationalDatabase:
        return SQLRelationalDatabase(name)

    def data_sets(self) -> Set[str]:
        with SQLConnection() as conn:
            ret, res = conn.execute_query(f"SHOW DATABASES", select=True)
            if ret == 0 and res is not None:
                """Have to remove {MYSQL_IDENTIFIER}_"""
                return set(backend_name[len(MYSQL_IDENTIFIER) + 1:] for backend_name in
                           set(r[0] for r in res).difference({'information_schema'}))
            else:
                return set()

    def __iter__(self) -> Iterator[SQLRelationalDatabase]:
        for name in self.data_sets():
            yield self.load(name)
