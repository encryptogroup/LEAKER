"""
For License information see the LICENSE file.

Authors: Amos Treiber, Michael Yonli

"""
from logging import getLogger
from typing import List, Any, Iterable, Tuple

import numpy as np
from scipy.optimize import minimize

from .arr.estimators import modular_estimator
from ..api import RangeAttack, LeakagePattern, RegularRangeDatabase
from ..pattern import ResponseLength

log = getLogger(__name__)


class Apa(RangeAttack):
    __m: int
    __n: int
    __alpha: int
    __beta: int
    __big_n: int

    def __init__(self, db: RegularRangeDatabase, m: int = 10):
        """:param m: Amount of reconstructions"""
        if not isinstance(db, RegularRangeDatabase):
            raise ValueError(f"{self.name()} requires a RegularRangeDatabase")

        super().__init__(db)
        self.__m = m
        self.__n = len(db)
        self.__alpha = db.get_min()
        self.__beta = db.get_max()
        self.__big_n = self.__beta - self.__alpha + 1

    @classmethod
    def name(cls) -> str:
        """
        :return: name of the attack
        """
        return "APA"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [ResponseLength()]

    def recover(self, queries: Iterable) -> List[float]:
        queries = list(queries)
        log.info(f"Starting {self.name()}")

        big_d = [(query, self.required_leakage()[0].leak(self.db(), [query])[0]) for query in queries]

        result = self._agnostic_parameterised_attack(self.db(), big_d, self.__alpha, self.__m, self.__n)
        log.info(f"Reconstruction completed")
        return result

    @staticmethod
    def _group_search_tk_by_vol(d: List[Tuple]) -> dict:
        """
        This functions groups search tokens based on the volume.
        :param d: A List of Tuples [(token_n: volume_n) ...]
        :return: A dict {(volume: matching tokens ..)...}
        """
        vols = set(map(lambda x: x[1], d))

        result = {vol: [] for vol in vols}
        for token, vol in d:
            result[vol].append(token)

        return result

    @staticmethod
    def _get_random_initial_point(big_n: int, count: int):
        idx = np.random.randint(0, big_n + 1, count + 1)
        idx[0] = 0
        idx[count] = big_n
        idx.sort()

        big_l = np.diff(idx)
        assert sum(big_l) == big_n
        return big_l

    def _agnostic_parameterised_attack(self, db: RegularRangeDatabase, big_d: List[Tuple], alpha: int, m: int, n: int):
        vol_token_map = self._group_search_tk_by_vol(big_d)

        w = np.zeros(n + 1)
        theta = np.zeros(n + 1)

        for i, d_i in vol_token_map.items():
            w[i] = len(d_i) ** 2
            theta[i] = modular_estimator(d_i)

        big_q = db.num_canonical_queries()
        sum_theta = np.sum(theta)

        diff = big_q - sum_theta
        if abs(diff) > 0:
            pdf = [(theta_i + 1) / (n + 1 + sum_theta) for theta_i in theta]
            to_modify = np.random.choice(n + 1, round(abs(diff)), replace=True, p=pdf)

            indices, counts = np.unique(to_modify, return_counts=True)
            for idx, count in zip(indices, counts):
                if diff < 0:
                    theta[idx] -= count
                else:
                    theta[idx] += count

        sols = np.zeros((m, n))

        for j in range(m):
            big_l = self._get_random_initial_point(self.__big_n, n + 1)
            log.info(f"j={j}")

            # sum = N represented as inequalities
            sum_constraint1 = {'type': 'ineq', 'fun': lambda x: np.ones((1, n + 1)).dot(x) - self.__big_n}
            sum_constraint2 = {'type': 'ineq', 'fun': lambda x: self.__big_n - np.ones((1, n + 1)).dot(x)}
            bound_constraint = {'type': 'ineq', 'fun': lambda x: min(x)}  # L_i > 0

            res = minimize(db.loss, method='COBYLA', x0=big_l, args=(theta, w),
                           constraints=[sum_constraint1, sum_constraint2, bound_constraint])

            v = res['x']
            sols[j][0] = alpha + v[0]
            for i, x in enumerate(v[1:-1]):
                sols[j][i + 1] = sols[j][i] + x
                if sols[j][i + 1] > self.__beta:
                    log.warning(f"Encountered value >beta at {j, i + 1}.")

        err = np.zeros(m)

        def calc_abs_sol_diff(a, b):
            d = np.abs(a - b)
            di = np.abs(a - np.flip(b))
            md = min(np.sum(d), np.sum(di))

            return md

        min_err = float('inf')
        min_idx = 0

        if m > 1:
            for j in range(m):
                diffs = [calc_abs_sol_diff(sols[j], sols[i]) for i in range(m)]
                err[j] = 1 / (m - 1) / n * sum(diffs)

                if err[j] < min_err:
                    min_err = err[j]
                    min_idx = j

        return sols[min_idx]
