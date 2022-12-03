import math
import random

import numpy as np
import torch
from typing import Dict, Tuple, Union, Optional

from leaker.api import RelationalDatabase, RelationalKeyword
from leaker.attack.relational_estimators.estimator import RelationalEstimator
from leaker.attack.relational_estimators.uae import made
from leaker.attack.relational_estimators.uae.common import Table, CsvTable, TableDataset
from leaker.attack.relational_estimators.uae.estimators import CardEst, DifferentiableProgressiveSampling, \
    ProgressiveSampling
from leaker.attack.relational_estimators.uae.train_uae import Entropy, ReportModel, InitWeight, DEVICE, \
    RunEpoch
from leaker.extension import PandasExtension


class UaeRelationalEstimator(RelationalEstimator):
    """Uses the UAE estimator of Qu et al.: https://github.com/pagegitss/UAE"""

    _table_dict: Dict[int, Tuple[Table, Table]]
    __epochs: int
    __batch_size: int
    __nr_train_queries: int
    __scale = 256
    __layers = 4
    __psample = 100
    __diff_psample = 100
    _estimator: Union[None, Dict[int, CardEst]] = None

    def __init__(self, sample: RelationalDatabase, full: RelationalDatabase, epochs: int = 20, batch_size: int = 1024,
                 nr_train_queries: int = 100):
        self._table_dict = dict()
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__nr_train_queries = nr_train_queries
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
            table_bits = Entropy(
                table,
                table.data.fillna(value=0).groupby([c.name for c in table.columns
                                                    ]).size(), [2])[0]

            table_train = table

            model = made.MADE(
                nin=len(table.columns),
                hidden_sizes=[self.__scale] * self.__layers,
                nout=sum([c.DistributionSize() for c in table.columns]),
                input_bins=[c.DistributionSize() for c in table.columns],
                embed_size=32,
                column_masking=True,
                input_encoding='binary',
            ).to(DEVICE)

            ReportModel(model)

            model.apply(InitWeight)

            opt = torch.optim.Adam(list(model.parameters()), 2e-4)

            train_data = TableDataset(table_train)
            n_cols = len(table.columns)

            # sample training queries randomly from full table
            query_list = random.sample(self._full.queries(table=table_id), self.__nr_train_queries)

            columns_list = []
            operators_list = []
            vals_list = []
            card_list = []

            for query in query_list:
                cols = [f'attr_{query.attr}']
                ops = ['=']
                vals = [query.value]
                # calculate selectivity of query based on full table
                sel = len([1 for x in self._full(query) if x[0] == table_id]) / \
                      len(list([1 for x in self._full.documents() if x[0] == table_id]))
                columns_list.append(cols)
                operators_list.append(ops)
                vals_list.append(vals)
                card_list.append(sel)

            total_query_num = len(card_list)

            num_steps = table.cardinality / self.__batch_size
            q_bs = math.ceil(total_query_num / num_steps)
            q_bs = int(q_bs)

            #self.__diff_psample = len(self._dataset_sample.row_ids(table_id))
            diff_estimator = DifferentiableProgressiveSampling(model=model,
                                                               table=table,
                                                               r=self.__diff_psample,
                                                               batch_size=q_bs,
                                                               device=DEVICE,
                                                               shortcircuit=True,
                                                               )

            wildcard_indicator, valid_i_list = diff_estimator.ProcessQuery(self._dataset_sample.name(), columns_list,
                                                                           operators_list, vals_list)

            valid_i_list = np.array(valid_i_list, dtype=object)
            card_list = torch.as_tensor(card_list, dtype=torch.float32)
            card_list = card_list.to(DEVICE)

            for epoch in range(self.__epochs):
                torch.set_grad_enabled(True)
                model.train()
                mean_epoch_train_loss = RunEpoch('train',
                                                 model,
                                                 diff_estimator,
                                                 valid_i_list,
                                                 wildcard_indicator,
                                                 card_list,
                                                 opt,
                                                 n_cols=n_cols,
                                                 train_data=train_data,
                                                 val_data=train_data,
                                                 batch_size=self.__batch_size,
                                                 q_bs=q_bs,
                                                 epoch_num=epoch,
                                                 log_every=10,
                                                 table_bits=table_bits)

            #self.__psample = len(self._dataset_sample.row_ids(table_id))
            estimator = ProgressiveSampling(model, table, self.__psample,
                                            device=torch.device(DEVICE),
                                            cardinality=full_table.cardinality)

            self._estimator[table_id] = estimator

    def estimate(self, kw: RelationalKeyword, kw2: Optional[RelationalKeyword] = None) -> float:
        if self._estimator is None:
            self._train()

        table, _ = self._table_dict[kw.table]
        if kw2 is None:
            return self._estimator[kw.table].Query([c for c in table.Columns() if f'attr_{kw.attr}' in c.name], ['='],
                                                   [kw.value])
        else:
            if kw2.table != kw.table:
                return 0
            else:
                return self._estimator[kw.table].Query([c for c in table.Columns() if f'attr_{kw.attr}' in c.name] +
                                                       [c for c in table.Columns() if f'attr_{kw2.attr}' in c.name],
                                                       ['=', '='], [kw.value, kw2.value])
