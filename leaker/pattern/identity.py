"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from typing import Iterable, List, Set, Union, Tuple

from ..api import Dataset, LeakagePattern, RangeDatabase, RelationalDatabase, RelationalQuery
from ..extension import IdentityExtension


class ResponseIdentity(LeakagePattern[Union[Set[int], Set[str], Set[Tuple[int, int]]]]):
    """
    The response identity (rid) leakage pattern leaking the set of identifiers matching any given query.
    """
    def leak(self, dataset: Union[Dataset, RangeDatabase, RelationalDatabase],
             queries: Union[Iterable[str], Iterable[Tuple[int, int]], Iterable[RelationalQuery]])\
            -> List[Union[Set[int], Set[str], Set[Tuple[int, int]]]]:
        if isinstance(dataset, RangeDatabase):
            return [set(dataset.query(q)) for q in queries]
        else:
            if dataset.has_extension(IdentityExtension):
                identity = dataset.get_extension(IdentityExtension)
                return [identity.doc_ids(q) for q in queries]
            else:
                if isinstance(dataset, RelationalDatabase):
                    return [set(dataset(q)) for q in queries]
                else:
                    return [set(map(lambda doc: doc.id(), dataset(q))) for q in queries]
