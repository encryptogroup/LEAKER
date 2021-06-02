"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from typing import List, Iterable, Union, Tuple

from ..api import Dataset, LeakagePattern, RangeDatabase, RelationalDatabase, RelationalQuery
from ..extension import SelectivityExtension


class ResponseLength(LeakagePattern[int]):
    """
    The response length (rlen) leakage pattern leaking the number of entries matching any given query.
    """

    def leak(self, dataset: Union[Dataset, RangeDatabase, RelationalDatabase],
             queries: Union[Iterable[str], Iterable[Tuple[int, int]], Iterable[RelationalQuery]])\
            -> List[int]:
        if isinstance(dataset, RangeDatabase):
            return [len(dataset.query(q)) for q in queries]
        else:
            if dataset.has_extension(SelectivityExtension):
                selectivity = dataset.get_extension(SelectivityExtension)
                return [selectivity.selectivity(q) for q in queries]
            else:
                return [dataset.selectivity(q) for q in queries]
