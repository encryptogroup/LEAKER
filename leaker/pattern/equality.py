"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from typing import Iterable, List
from ..api import Dataset, LeakagePattern
from  collections import Counter
import numpy as np

class QueryEquality(LeakagePattern[List[int]]):
    def leak(self, dataset: Dataset, queries: Iterable[str]) -> List[List[int]]:
        return [[q == qp for qp in queries] for q in queries]

class Frequency(LeakagePattern[List[int]]):
    def leak(self, dataset: Dataset, queries: Iterable[str]) -> List[int]:
        n_docs = len(dataset.doc_ids())
        kw_list = list(dataset.keywords())
        n_kw = len(kw_list)
        if queries is not None and isinstance(queries[0],list):
            nweeks = len(queries)
            f_matrix = np.zeros((n_kw,nweeks))
            for i_week, week in enumerate(queries):
                count = Counter(week)
                for i_kw in range(n_kw):
                    f_matrix[i_kw,i_week] = count[kw_list[i_kw]]/n_docs
        else:
            nweeks = 1
            f_matrix = np.zeros((n_kw,nweeks))
            count = Counter(queries)
            for i_kw in range(n_kw):
                f_matrix[i_kw,0] = count[kw_list[i_kw]]/n_docs
        return f_matrix
