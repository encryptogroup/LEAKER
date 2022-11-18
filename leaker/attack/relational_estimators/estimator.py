"""
For License information see the LICENSE file.

Authors: Amos Treiber

This file provides interfacing to various cardinality estimator implementations.

"""

from abc import abstractmethod, ABC
from collections import Counter
from typing import Any, Dict, Union, Optional, Tuple
from sklearn.neighbors import KernelDensity
import torch
import numpy as np

from .naru.common import Table, CsvTable, TableDataset
from .naru.estimators import ProgressiveSampling, CardEst
from .naru.train_model import MakeMade, ReportModel, transformer, InitWeight, RunEpoch, Entropy, DEVICE
from ...api import RelationalDatabase, RelationalKeyword
from ...extension import PandasExtension
from ...sql_interface import SQLRelationalDatabase


class RelationalEstimator(ABC):
    """Trains an estimator using the supplied dataset as a sample. The model is then used to provide cardinality
    estimates for queries."""

    _dataset_sample: SQLRelationalDatabase
    _full: SQLRelationalDatabase
    _estimator: Any

    def __init__(self, sample: RelationalDatabase, full: RelationalDatabase):
        self._dataset_sample = sample
        self._estimator = None
        self._full = full

    @abstractmethod
    def _train(self) -> None:
        """Explicit method for training phase. Should be called from the constructor"""
        raise NotImplementedError

    @abstractmethod
    def estimate(self, kw: RelationalKeyword) -> float:
        """Uses the trained model to estimate the selectivity/relative cardinality of kw on the full dataset
        based on the sample provided at training time."""
        raise NotImplementedError


class NaruRelationalEstimator(RelationalEstimator):
    """Uses the Naru estimator of Yang et al.: https://github.com/naru-project/naru"""

    _table_dict: Dict[int, Tuple[Table, Table]]
    __epochs: int
    _estimator: Union[None, Dict[int, CardEst]] = None

    def __init__(self, sample: RelationalDatabase, full: RelationalDatabase, epochs: int = 20):
        self._table_dict = dict()
        self.__epochs = epochs
        super().__init__(sample, full)

        if not self._dataset_sample.has_extension(PandasExtension):
            self._dataset_sample.extend_with(PandasExtension)
        pd_ext = self._dataset_sample.get_extension(PandasExtension)
        full_pd_ext = self._full.get_extension(PandasExtension)

        if not self._dataset_sample.is_open():
            with self._dataset_sample:
                for table in self._dataset_sample.tables():
                    table_id = self._dataset_sample.table_id(table)
                    df = pd_ext.get_df(table_id)
                    full_df = full_pd_ext.get_df(table_id)
                    self._table_dict[table_id] = (CsvTable(table, df, df.columns),
                                                  CsvTable(table, full_df, full_df.columns))

                self._train()
        else:
            for table in self._dataset_sample.tables():
                table_id = self._dataset_sample.table_id(table)
                df = pd_ext.get_df(table_id)
                full_df = full_pd_ext.get_df(table_id)
                self._table_dict[table_id] = (
                    CsvTable(table, df, df.columns), CsvTable(table, full_df, full_df.columns))

            self._train()

    def _train(self) -> None:
        self._estimator = dict()
        for table_name in self._dataset_sample.tables():
            table_id = self._dataset_sample.table_id(table_name)

            table, full_table = self._table_dict[table_id]

            model = MakeMade(256, table.columns, None)
            ReportModel(model)

            if not isinstance(model, transformer.Transformer):
                model.apply(InitWeight)

            if isinstance(model, transformer.Transformer):
                opt = torch.optim.Adam(
                    list(model.parameters()),
                    2e-4,
                    betas=(0.9, 0.98),
                    eps=1e-9,
                )
            else:
                opt = torch.optim.Adam(list(model.parameters()), 2e-4)
            ReportModel(model)

            train_data = TableDataset(table)

            table_bits = Entropy(
                table,
                table.data.fillna(value=0).groupby([c.name for c in table.columns
                                                    ]).size(), [2])[0]

            for epoch in range(self.__epochs):
                mean_epoch_train_loss = RunEpoch('train',
                                                 model,
                                                 opt,
                                                 train_data=train_data,
                                                 val_data=train_data,
                                                 batch_size=1024,
                                                 epoch_num=epoch,
                                                 table_bits=table_bits,
                                                 return_losses=True
                                                 )

            estimator = ProgressiveSampling(model, table, len(self._dataset_sample.row_ids(table_id)),
                                            device=torch.device(DEVICE),
                                            cardinality=full_table.cardinality)
            self._estimator[table_id] = estimator

            print(f"Done.")
            ReportModel(model)

    def estimate(self, kw: RelationalKeyword, kw2: Optional[RelationalKeyword] = None) -> float:
        if self._estimator is None:
            self._train()

        table, _ = self._table_dict[kw.table]
        if kw2 is None:
            return self._estimator[kw.table].Query([c for c in table.Columns() if f"attr_{kw.attr}" in c.name], ["="],
                                                   [kw.value])
        else:
            if kw2.table != kw.table:
                return 0
            else:
                return self._estimator[kw.table].Query([c for c in table.Columns() if f"attr_{kw.attr}" in c.name] +
                                                       [c for c in table.Columns() if f"attr_{kw2.attr}" in c.name],
                                                       ["=", "="], [kw.value, kw2.value])


class KDERelationalEstimator(RelationalEstimator):
    """Uses a Kernel Density Estimator"""
    _estimator: Union[None, Dict[Tuple[int, str], KernelDensity]] = None
    _word_mapping: Union[None, Dict[Tuple[int, str, Union[str, int]], int]] = None
    _table_n: Dict[int, int]

    def __init__(self, sample: RelationalDatabase, full: RelationalDatabase):
        super().__init__(sample, full)

        self._table_n = dict()
        for t, ids in full._table_row_ids.items():
            self._table_n[t] = len(ids)

        if not self._dataset_sample.has_extension(PandasExtension):
            self._dataset_sample.extend_with(PandasExtension)

        self._train()

    def _train(self) -> None:
        self._estimator = dict()
        self._word_mapping = dict()
        pd_ext = self._dataset_sample.get_extension(PandasExtension)

        for t in self._table_n.keys():
            df = pd_ext.get_df(t)
            for a in df.columns:
                vals = []
                for i, w in enumerate(df[a].to_list()):
                    if (t, a, w) not in self._word_mapping:
                        self._word_mapping[(t, a, w)] = i
                    vals.append(self._word_mapping[(t, a, w)])
                vals = np.array(vals)
                self._estimator[(t, a)] = KernelDensity().fit(vals.reshape(-1, 1))

    def estimate(self, kw: RelationalKeyword, kw2: Optional[RelationalKeyword] = None) -> float:
        if self._estimator is None:
            self._train()
        if (kw.table, f"attr_{kw.attr}", kw.value) not in self._word_mapping:
            return 0

        ld = self._estimator[(kw.table, f"attr_{kw.attr}")].score(np.array([self._word_mapping[(kw.table,
                                                                                                f"attr_{kw.attr}",
                                                                                                kw.value)]]).reshape(
            1, -1))

        return np.exp(ld) * self._table_n[kw.table]


class NaiveRelationalEstimator(RelationalEstimator):
    """
        Uses the naive estimator of [HILM09]

    """
    _full_cardinality: Dict[int, int]

    def __init__(self, sample: RelationalDatabase, full: RelationalDatabase):
        super().__init__(sample, full)
        self._full_cardinality = self._calculate_full_cardinality(self._full)
        self._train()

    def _calculate_full_cardinality(self, dataset) -> Dict[int, int]:
        """calculates cardinality (number of documents) of each table (of the full dataset),
            returns {table0: nr_of_docs, ..., tableN: nr_of_docs}"""
        return dict((t, len([d for d in dataset.documents() if d[0] == t])) for t in
                    set(map(lambda x: x[0], dataset.documents())))

    def _calculate_column_cardinalities(self, dataset) -> Dict[int, Dict[int, int]]:
        """calculates the column cardinalities of each table and column,
            returns: {table0: {attr0: nr_of_unique_items, ..., attrN: nr_of_unique_items}, table1: {...}, ...}"""
        table_ids = set(map(lambda x: x.table, dataset.keywords()))
        attribute_ids = set(map(lambda x: (x.table, x.attr), dataset.keywords()))
        unique_values_dict = dict((t, dict((a[1], 0) for a in attribute_ids if a[0] == t)) for t in table_ids)
        for table_id, attr_value in unique_values_dict.items():
            for attr_id in attr_value:
                unique_values_dict[table_id][attr_id] = len(set(map(lambda x: x.value,
                                                                    [el for el in dataset.keywords() if
                                                                     el.attr == attr_id and el.table == table_id])))
        return unique_values_dict

    def _train(self) -> None:
        self._estimator = self._calculate_column_cardinalities(self._dataset_sample)

    def estimate(self, kw: RelationalKeyword, kw2: Optional[RelationalKeyword] = None) -> float:
        if kw2 is None:
            return self._full_cardinality[kw.table] / self._estimator[kw.table][kw.attr]
        else:
            return (self._full_cardinality[kw.table] / self._estimator[kw.table][kw.attr]) * \
                   (self._full_cardinality[kw2.table] / self._estimator[kw2.table][kw2.attr])


class SamplingRelationalEstimator(RelationalEstimator):
    """
        Uses the sampling estimator of ??
    """
    _full_cardinality: Dict[int, int]
    _sample_cardinality: Dict[int, int]

    def __init__(self, sample: RelationalDatabase, full: RelationalDatabase):
        super().__init__(sample, full)

        self._full_cardinality = self._calculate_cardinality(self._full)
        self._sample_cardinality = self._calculate_cardinality(self._dataset_sample)

        self._train()

    def _calculate_cardinality(self, dataset) -> Dict[int, int]:
        """calculates cardinality (number of documents) of each table,
            returns {table0: nr_of_docs, ..., tableN: nr_of_docs}"""
        return dict((t, len([d for d in dataset.documents() if d[0] == t])) for t in
                    set(map(lambda x: x[0], dataset.documents())))

    def _train(self) -> None:
        pass

    def estimate(self, kw: RelationalKeyword, kw2: Optional[RelationalKeyword] = None) -> float:
        if kw2 is None:
            rel_selectivity = (self._dataset_sample.selectivity(kw) / self._sample_cardinality[kw.table])
            return rel_selectivity * self._full_cardinality[kw.table]
        else:
            rlen_sample = len([i for i in self._dataset_sample(kw) if i in self._dataset_sample(kw2)])
            # if rlen>0, then kw and kw2 are from the same table; therefore sample cardinality is the same
            rel_selectivity = rlen_sample / self._sample_cardinality[kw.table]
            return rel_selectivity * self._full_cardinality[kw.table]
