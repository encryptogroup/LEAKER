"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from logging import getLogger
from typing import List, Iterable, Set, Dict

import numpy as np

from ..pattern import ResponseIdentity
from ..api import RangeAttack, LeakagePattern

log = getLogger(__name__)


class GeneralizedKKNO(RangeAttack):
    """Implements the generalized KKNO attack from [GLMP19]"""

    @classmethod
    def name(cls) -> str:
        return "GeneralizedKKNO"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Set[int]]]:
        return [ResponseIdentity()]

    def __prob(self, k: int) -> float:
        big_n = self.db().get_max()
        return 2 * k * (big_n + 1 - k) / (big_n * (big_n + 1))

    def __get_estsymval(self, rids: List[Set[int]]) -> List[int]:
        big_n = self.db().get_max()
        est_symval: List[int] = []
        for r in range(len(self.db())):
            c = sum([1 for q in rids if r in q]) / len(rids)
            dist_array = np.absolute([self.__prob(k) - c for k in range(1, big_n // 2 + 1)])
            est_symval.append(np.argmin(dist_array))

        return est_symval

    def __get_estval(self, rids: List[Set[int]], est_symval: List[int]) -> List[int]:
        big_n = self.db().get_max()
        est_val: Dict[int, int] = dict()

        dist_array = np.absolute([val - big_n / 4 for val in est_symval])
        r_a = np.argmin(dist_array)
        est_val[r_a] = est_symval[r_a]

        for r in [r for r in range(len(self.db())) if r != r_a]:
            c_p = sum([1 for records in rids if r in records and r_a in records]) / len(rids)
            if c_p > min(est_val[r_a], est_symval[r]) / big_n:
                est_val[r] = est_symval[r] + 1
            else:
                est_val[r] = big_n - est_symval[r]

        return [est_val[r] for r in range(len(self.db()))]

    def recover(self, queries: Iterable[Iterable[int]]) -> List[int]:
        log.info(f"Starting {self.name()}.")

        rids = self.required_leakage()[0](self.db(), queries)

        est_symval = self.__get_estsymval(rids)

        est_val = self.__get_estval(rids, est_symval)

        log.info(f"Reconstruction completed.")

        return est_val
