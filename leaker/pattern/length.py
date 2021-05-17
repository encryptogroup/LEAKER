"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from typing import List, Iterable, Union, Tuple

from ..api import Dataset, LeakagePattern, RangeDatabase
from ..extension import SelectivityExtension


class ResponseLength(LeakagePattern[int]):
    """
    The response length (rlen) leakage pattern leaking the number of documents matching any given query.
    """

    def leak(self, dataset: Union[Dataset, RangeDatabase], queries: Union[Iterable[str], Iterable[Tuple[int, int]]]) \
            -> List[int]:
        if isinstance(dataset, RangeDatabase):
            return [len(dataset.query(q)) for q in queries]
        else:
            if dataset.has_extension(SelectivityExtension):
                selectivity = dataset.get_extension(SelectivityExtension)
                return [selectivity.selectivity(q) for q in queries]
            else:
                return [dataset.selectivity(q) for q in queries]
