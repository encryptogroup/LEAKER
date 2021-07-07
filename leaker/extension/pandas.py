"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from abc import ABC
from logging import getLogger
from typing import Optional, Dict, Union, Tuple

import pandas as pd

from ..api import Extension, Dataset, RelationalKeyword, RelationalDatabase

log = getLogger(__name__)

Keyword = Union[str, RelationalKeyword]
Identifier = Union[str, Tuple[int, int]]


class PandasExtension(Extension, ABC):
    """
    An extension caching a relational database as a pandas dataframe

    Parameters
    ----------
    dataset : SQLRelationalDatabase
        the data set to build the cache on
    doc_ids : Optional[Set[Identifier]]
        (only used for sub sampling) the document identifiers contained in the sampled data set
    original_df : Optional[Dict[int, pd.DataFrame]]
        (only used for sub sampling) the full dict of dfs that needs to be subsampled
    """

    _df: Dict[int, pd.DataFrame]

    # noinspection PyMissingConstructor
    def __init__(self, dataset: RelationalDatabase, original_df: Optional[Dict[int, pd.DataFrame]] = None):

        self._df = dict()
        if original_df is not None:
            log.debug(f"Subsampling Pandas Cache for '{dataset.name()}'")

            if not dataset.is_open():
                log.debug("Opening dataset for caching")
                with dataset:
                    for table in dataset.tables():
                        table_id = dataset.table_id(table)
                        self._df[table_id] = original_df[table_id].iloc[list(map(lambda x: x[1],
                                                                                 dataset.row_ids(table_id)))]
            else:
                for table in dataset.tables():
                    table_id = dataset.table_id(table)
                    self._df[table_id] = original_df[table_id].iloc[
                        list(map(lambda x: x[1], dataset.row_ids(table_id)))]

            log.debug(f"Subsampling for '{dataset.name()}' complete")
        else:
            log.info(f"Creating Pandas Cache for '{dataset.name()}'. This might take a while.")
            if not dataset.is_open():
                log.debug("Opening dataset for caching")
                with dataset:
                    for table in dataset.tables():
                        table_id = dataset.table_id(table)
                        attr = set(f"attr_{x.attr}" for x in dataset.queries() if x.table == table_id)
                        index = list(map(lambda x: x[1], dataset.row_ids(table_id)))
                        self._df[table_id] = pd.DataFrame(index=index, columns=list(attr))

                    for kw in dataset.queries():
                        for table_id, row_id in dataset.query(kw):
                            self._df[table_id][f"attr_{kw.attr}"][row_id] = kw.value

            else:
                for table in dataset.tables():
                    table_id = dataset.table_id(table)
                    attr = set(f"attr_{x.attr}" for x in dataset.queries() if x.table == table_id)
                    index = list(map(lambda x: x[1], dataset.row_ids(table_id)))
                    self._df[table_id] = pd.DataFrame(index=index, columns=list(attr))

                for kw in dataset.queries():
                    for table_id, row_id in dataset.query(kw):
                        self._df[table_id][f"attr_{kw.attr}"][row_id] = kw.value

        log.info(f"Pandas loading for '{dataset.name()}' complete")

    def sample(self, dataset: Union[Dataset, RelationalDatabase]) -> 'PandasExtension':
        return PandasExtension(dataset, self._df)

    def get_df(self, table_id: int) -> pd.DataFrame:
        return self._df[table_id]

    def pickle(self, dataset: Union[Dataset, RelationalDatabase], description: Optional[str] = None) -> None:
        pass

    @classmethod
    def extend_with_pickle(cls, dataset: Union[Dataset, RelationalDatabase], description: Optional[str] = None) \
            -> 'PandasExtension':
        pass
