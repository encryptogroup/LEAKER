import math
import random

import numpy as np
import torch
from typing import Dict, Tuple, Union, Optional, List

from leaker.api import RelationalDatabase, RelationalKeyword, RelationalQuery
from leaker.attack.relational_estimators.estimator import RelationalEstimator
from leaker.attack.relational_estimators.uae import made
from leaker.attack.relational_estimators.uae.common import Table, CsvTable, TableDataset
from leaker.attack.relational_estimators.uae.estimators import CardEst, DifferentiableProgressiveSampling, \
    ProgressiveSampling
from leaker.attack.relational_estimators.uae.train_uae import Entropy, ReportModel, InitWeight, DEVICE, \
    RunEpoch, RunQueryEpoch
from leaker.extension import PandasExtension


class UaeRelationalEstimator(RelationalEstimator):
    """Uses the UAE estimator of Qu et al.: https://github.com/pagegitss/UAE"""

    _run_uaeq = True  # if true, only uae-q is used
    _q_bs = 100  # only used for uae-q only
    _table_dict: Dict[int, Tuple[Table, Table]]
    __epochs: int
    _estimator: Union[None, Dict[int, CardEst]] = None
    __hidden_size = [256] * 5  # or large MADE model: [512, 256, 512, 128, 1024]
    __psample = 200  # figure 4a, UAE paper
    __diff_psample = 200  # figure 4a, UAE paper
    __batch_size = 2048
    __column_masking = True
    __residual = True
    __direct_io = True
    __train_queries: List[List[RelationalQuery]]  # true cardinalities are calculated based on full database

    # TODO: In UAE github example: epochs=50
    #  batch-size=4096, here not possible because of but error
    def __init__(self, sample: RelationalDatabase, full: RelationalDatabase, train_queries: List[List[RelationalQuery]],
                 epochs: int = 20):
        self._table_dict = dict()
        self.__epochs = epochs
        self.__train_queries = train_queries
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
                hidden_sizes=self.__hidden_size,
                nout=sum([c.DistributionSize() for c in table.columns]),
                input_bins=[c.DistributionSize() for c in table.columns],
                input_encoding='binary',
                output_encoding='one_hot',
                embed_size=32,
                do_direct_io_connections=self.__direct_io,
                residual_connections=self.__residual,
                # fixed_ordering=fixed_ordering,
                column_masking=self.__column_masking,
            ).to(DEVICE)

            ReportModel(model)

            model.apply(InitWeight)

            opt = torch.optim.Adam(list(model.parameters()), 2e-4)

            train_data = TableDataset(table_train)
            n_cols = len(table.columns)

            columns_list = []
            operators_list = []
            vals_list = []
            card_list = []

            for query in self.__train_queries:
                cols = []
                ops = []
                vals = []
                sub_rids = []  # list with rids of sub-parts of query (each individual predicate)
                for query_item in query:
                    cols.append(f'attr_{query_item.attr}')
                    ops.append('=')
                    vals.append(query_item.value)
                    sub_rids.append([x for x in list(self._full.query(query_item)) if x[0] == table_id])
                # calculate selectivity of query based on full table
                rlen = len(set.intersection(*map(set, sub_rids)))
                sel = rlen / len([1 for _ in self._full.row_ids(table_id)])
                if sel > 0:
                    columns_list.append(cols)
                    operators_list.append(ops)
                    vals_list.append(vals)
                    card_list.append(sel)

            total_query_num = len(card_list)

            if self._run_uaeq:
                q_bs = self._q_bs
            else:
                num_steps = table.cardinality / self.__batch_size
                q_bs = math.ceil(total_query_num / num_steps)
                q_bs = int(q_bs)

            diff_estimator = DifferentiableProgressiveSampling(model=model,
                                                               table=full_table,  # seems to be only used for column information
                                                               r=self.__diff_psample,
                                                               batch_size=q_bs,
                                                               device=DEVICE,
                                                               shortcircuit=self.__column_masking,
                                                               )

            wildcard_indicator, valid_i_list = diff_estimator.ProcessQuery(self._dataset_sample.name(), columns_list,
                                                                           operators_list, vals_list)

            valid_i_list = np.array(valid_i_list, dtype=object)
            card_list = torch.as_tensor(card_list, dtype=torch.float32)
            card_list = card_list.to(DEVICE)

            for epoch in range(self.__epochs):
                torch.set_grad_enabled(True)
                model.train()

                if not self._run_uaeq:
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
                                                     return_losses=True,
                                                     table_bits=table_bits)
                else:
                    mean_epoch_train_loss = RunQueryEpoch('train',
                                                          model,
                                                          diff_estimator,
                                                          valid_i_list,
                                                          wildcard_indicator,
                                                          card_list,
                                                          full_table.cardinality,  # not used
                                                          opt,
                                                          n_cols=n_cols,
                                                          batch_size=q_bs,
                                                          epoch_num=epoch)

            ReportModel(model)
            model.eval()
            estimator = ProgressiveSampling(model, table, self.__psample,
                                            device=DEVICE,
                                            cardinality=full_table.cardinality,
                                            shortcircuit=self.__column_masking
                                            )

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
