"""
For License information see the LICENSE file.

Authors: Abdelkarim Kati

"""
from typing import Iterable, List, Tuple
from ..api import LeakagePattern, RangeDatabase


class Rank(LeakagePattern[int]):
    """
    The rank leakage pattern for a given range query [x,y] leaks [a,b] such that a:=rank(x-1) and b:=rank(y)
    """

    def leak(self, db: RangeDatabase, queries: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [(db.get_rank(q[0] - 1), db.get_rank(q[1])) for q in queries]
