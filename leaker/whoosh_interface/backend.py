"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
import os
from typing import Set

from whoosh.index import open_dir

from ..api.constants import WHOOSH_INDEX_DIRECTORY
from .dataset import WhooshDataset, WhooshKeywordQueryLog
from ..api import Backend


class WhooshBackend(Backend):
    """
    A `Backend` for loading Whoosh indices as data sets. It is associated with the `WhooshDataset` class.
    """

    def has(self, name: str) -> bool:
        return name in self.data_sets()

    def load(self, name: str) -> WhooshDataset:
        return self.load_dataset(name)

    def load_dataset(self, name: str, pickle_description: str = None) -> WhooshDataset:
        if not self.has(name):
            raise FileNotFoundError('Index not found:' + name)
        return WhooshDataset(name, open_dir(WHOOSH_INDEX_DIRECTORY + name), pickle_description=pickle_description)

    def load_querylog(self, name: str, pickle_description: str = None, min_user_count: int = 0,
                      max_user_count: int = None, reverse: bool = False) -> WhooshKeywordQueryLog:
        if not self.has(name):
            raise FileNotFoundError('Index not found:' + name)
        return WhooshKeywordQueryLog(name, open_dir(WHOOSH_INDEX_DIRECTORY + name),
                                     pickle_description=pickle_description, min_user_count=min_user_count,
                                     max_user_count=max_user_count, reverse=reverse)

    def data_sets(self) -> Set[str]:
        if os.path.exists(WHOOSH_INDEX_DIRECTORY):
            return set(next(os.walk(WHOOSH_INDEX_DIRECTORY))[1])
        else:
            return set()

    def __iter__(self) -> WhooshDataset:
        # TODO: Split this up between Dataset and KeywordQueryLog by adding a separate QueryLogBackend
        for name in self.data_sets():
            yield self.load_dataset(name)
