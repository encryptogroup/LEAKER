"""
For License information see the LICENSE file.

Authors: Abdelkarim Kati

"""
import itertools
from functools import reduce
from logging import getLogger
from typing import List, Iterable, Set, Tuple, Dict

import numpy as np
from scipy.stats import binom

from ..pattern import ResponseIdentity, Rank
from ..api import RangeDatabase, RangeAttack, LeakagePattern

log = getLogger(__name__)


class LMPrank(RangeAttack):
    """
    Implements the Full data reconstruction Range attack from [LMP17] based on Access pattern & Rank leakage.
    PS. The dataset should always be dense and as a best practice N should be a multiple of 4
    """

    @classmethod
    def name(cls) -> str:
        return "LMP-rank"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Set[int]]]:
        return [ResponseIdentity(), Rank()]

    def __partition(self, rids: List[Set[int]], rank: List[Tuple[int, int]]) \
            -> Dict[int, int]:
        leakage = list(zip(rank, rids))
        m_r: Dict[int, int] = dict()
        for r in range(len(self.db())):
            intersect_cand = [np.arange(_[0], _[1] + 1) for _, R in leakage if r in R]
            union_cand = [np.arange(_[0], _[1] + 1) for _, R in leakage if r not in R]
            intersect_set = reduce(np.intersect1d, intersect_cand) if len(intersect_cand) > 0 else []
            union_set = reduce(np.union1d, union_cand) if len(union_cand) > 0 else []
            set_diff = np.setdiff1d(intersect_set, union_set, assume_unique=True)
            m_r[r] = np.amax(set_diff) if len(set_diff) > 0 else np.amax(intersect_set) if len(
                intersect_set) > 0 else np.amin(union_set)

        return m_r

    def __sorting(self, rids: List[Set[int]], rank: List[Tuple[int, int]]) \
            -> Dict[int, int]:
        val_r: Dict[int, int] = dict()
        big_n = self.db().get_num_of_values()
        m_r = self.__partition(rids, rank)
        big_m = sorted(set(m_r.values()))
        if len(big_m) < big_n:
            log.warning(
                f"{self.name()} Failed, The number of recreated partitions is not enough... Return min(DB) n times")
            val_r = {_: self.db().get_min() for _ in range(len(self.db()))}
        else:
            for r in range(len(self.db())):
                val_r[r + 1] = big_m.index(m_r[r]) + 1
        return val_r

    def recover(self, queries: Iterable[Tuple[int, int]]) -> List[int]:
        log.info(f"Starting {self.name()}.")

        rids = self.required_leakage()[0](self.db(), queries)

        rank = self.required_leakage()[1](self.db(), queries)

        val = self.__sorting(rids, rank)

        log.info(f"Reconstruction completed.")

        return [val[i + 1] for i in range(len(val.keys()))]


class LMPrid(RangeAttack):
    """
    Implements the Full data reconstruction Range attack from [LMP17] based on Access Pattern leakage.
    PS. The dataset should always be dense and as a best practice N should be a multiple of 4
    """

    @classmethod
    def name(cls) -> str:
        return "LMP-rid"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Set[int]]]:
        return [ResponseIdentity()]

    def partitioning(self, rids: List[Set[int]]) -> Dict[int, Set[int]]:
        leakage = set(tuple(sorted(row)) for row in rids)
        p_r: Dict[int, Set[int]] = dict()
        for r in range(len(self.db())):
            intersect_cand = [c for c in leakage if r in c]
            union_cand = [c for c in leakage if r not in c]
            intersect_set = reduce(np.intersect1d, intersect_cand) if len(intersect_cand) > 0 else []
            union_set = reduce(np.union1d, union_cand) if len(union_cand) > 0 else []
            p_r[r] = set(np.setdiff1d(intersect_set, union_set, assume_unique=True))
        return p_r

    def sorting(self, rids: List[Set[int]]) -> Dict[int, int]:

        val_r: Dict[int, int] = dict()
        big_i: Dict[int, Set[int]] = dict()
        big_n = self.db().get_num_of_values()
        big_r = self.db().get_n()
        p_r = self.partitioning(rids)

        leakage = set(tuple(sorted(row)) for row in rids)
        points_set = set(tuple(sorted(row)) for row in p_r.values())
        # set of distinct points; in which each point contains a set of records

        if len(points_set) < big_n:
            log.warning(f"{self.name()} Failed, The Set of recreated points are not enough... Return min(DB) n times")
            val_r = {_: self.db().get_min() for _ in range(len(self.db()))}

        else:
            s = [a for a in leakage if len(a) < len(self.db())].pop(0)

            for q in leakage:
                if (len(np.intersect1d(q, s, assume_unique=True)) > 0
                        and len(np.setdiff1d(q, s, assume_unique=True)) > 0
                        and len(np.union1d(q, s)) < big_r):
                    s = np.union1d(s, q)

            r_s = tuple(np.setdiff1d(np.arange(big_r), s, assume_unique=True))

            if r_s not in points_set:
                log.warning(
                    f"{self.name()} failed, The set diffrence of records R\S doesn't match a single point..."
                    f" Return min(DB) n times")
                val_r = {_: self.db().get_min() for _ in range(len(self.db()))}
                return val_r

            else:
                big_i[1] = r_s

                for i in range(1, big_n):
                    q_prim = set()

                    for q in leakage:
                        if (len(np.intersect1d(q, big_i[i], assume_unique=True)) > 0
                                and len(np.setdiff1d(q, big_i[i], assume_unique=True)) > 0):
                            q_prim.add(q)

                    big_t = np.setdiff1d(reduce(np.intersect1d, q_prim), big_i[i], assume_unique=True)

                    for q in leakage:
                        if (len(np.intersect1d(q, big_t, assume_unique=True)) > 0
                                and len(np.setdiff1d(q, np.union1d(big_t, big_i[i]), assume_unique=True)) > 0
                                and len(np.setdiff1d(big_t, q)) > 0):
                            big_t = np.setdiff1d(big_t, q)

                    if tuple(big_t) not in points_set:
                        log.warning(
                            f"{self.name()} Failed, The |Val(T)|!=1... Return min(DB) n times")
                        "No need to check the next big_i[i+1] nor set it since the attack should fail and return ⊥"
                        val_r = {_: self.db().get_min() for _ in range(len(self.db()))}
                        return val_r
                    else:
                        big_i[i + 1] = np.union1d(big_t, big_i[i])

                for r in range(len(self.db())):
                    val_r[r] = min([key for (key, value) in big_i.items() if r in value])

        return val_r

    def recover(self, queries: Iterable[Iterable[int]]) -> List[int]:
        log.info(f"Starting {self.name()}.")

        rids = self.required_leakage()[0](self.db(), queries)

        val = self.sorting(rids)

        log.info(f"Reconstruction completed.")

        return [val[i] for i in range(len(val.keys()))]


class LMPappRec(RangeAttack):
    """
    Implements the Access Pattern Approximate reconstruction Range attack from [LMP17].
    PS. The dataset should always be dense; Max accepted error is 75% and N should be a multiple of 4 for optimal result
    If return_mid_point is True, the attack will return the mid-point of the calculated interval. If False, the real
    value will be returned if it is in the interval, i.e., if the interval was correct.
    """

    __return_mid_point: bool
    __error: float

    def __init__(self, db: RangeDatabase, return_mid_point: bool = True, error=0.25):
        super().__init__(db)
        self.__return_mid_point = return_mid_point
        self.__error = error

    @classmethod
    def name(cls) -> str:
        return "LMP-approx"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Set[int]]]:
        return [ResponseIdentity()]

    @classmethod
    def __partitioning(cls, r: int, rids: List[Set[int]]) -> List[Set[int]]:
        leakage = set(tuple(sorted(row)) for row in rids)

        halves = dict()
        halves_l = dict()
        halves_r = dict()

        intersect_cand = [c for c in leakage if r in c]
        big_m = set(reduce(np.intersect1d, intersect_cand)) if len(intersect_cand) > 0 else set()

        for a, b in itertools.combinations(leakage, 2):
            if big_m == set(np.intersect1d(a, b, assume_unique=True)) and len(a) > 1 and len(b) > 1:
                halves.update({len(np.union1d(a, b)): [a, b]})

        max_length = max(halves.keys())
        q_l = halves[max_length][0]
        q_r = halves[max_length][1]

        for q in leakage:
            if (len(np.intersect1d(q, q_l, assume_unique=True)) > 0
                    and set(np.intersect1d(q, q_r, assume_unique=True)).issubset(big_m)):
                halves_l.update({len(np.union1d(q, q_l)): [q, q_l]})

            if (len(np.intersect1d(q, q_r, assume_unique=True)) > 0
                    and set(np.intersect1d(q, q_l, assume_unique=True)).issubset(big_m)):
                halves_r.update({len(np.union1d(q, q_r)): [q, q_r]})

        max_length_r = max(halves_r.keys())
        max_length_l = max(halves_l.keys())
        q_l_prime = halves_l[max_length_l][0]
        q_r_prime = halves_r[max_length_r][0]

        return [q_l_prime, q_l, q_r, q_r_prime, big_m]

    def __sorting(self, error: float, rids: List[Set[int]]) -> Dict[int, int]:

        val_r: Dict[int, int] = dict()

        big_n = self.db().get_num_of_values()
        big_r = set(range(self.db().get_n()))
        leakage = set(tuple(sorted(row)) for row in rids)
        flag = False

        for i in range(len(self.db())):
            if flag is True:
                break
            try:
                p_r = self.__partitioning(i, rids)
            except ValueError:
                log.debug(f"Encountered ValueError")
                continue
            *union_cand, big_m = p_r  # get the first 4 elements in union_cand and the last element in big_m
            union_set = reduce(np.union1d, union_cand) if len(union_cand) > 0 else set()

            if set(union_set) == big_r:
                q_l_prime = p_r[0]
                q_l = p_r[1]
                q_r = p_r[2]
                q_r_prime = p_r[3]
                coupon_l = set()
                coupon_r = set()
                half_l = np.union1d(q_l_prime, q_l)
                half_r = np.union1d(q_r_prime, q_r)

                for q in leakage:
                    if big_m.issubset(q):
                        coupon_l.add(frozenset(np.setdiff1d(q, half_r)))
                        coupon_r.add(frozenset(np.setdiff1d(q, half_l)))

                coupon_l = list(filter(None, coupon_l))
                coupon_r = list(filter(None, coupon_r))
                n_l = len(coupon_l)
                n_r = len(coupon_r)

                if (big_n - (n_l + n_r + 1)) <= (error * big_n):
                    log.debug(f"Approximate reconstruction succeeded with precision ɛN={(error * big_n)}... ")
                    coupon_l = sorted(coupon_l, key=len)
                    coupon_r = sorted(coupon_r, key=len)

                    for r in range(len(self.db())):
                        min_val_r = n_l + 1
                        if r in half_l and len([coupon_l.index(_) + 1 for _ in coupon_l if r in _]) > 0:
                            min_val_r = n_l + 1 - min([coupon_l.index(_) + 1 for _ in coupon_l if r in _])

                        elif r in big_m:
                            min_val_r = n_l + 1

                        elif r in half_r and len([coupon_r.index(_) + 1 for _ in coupon_r if r in _]) > 0:
                            min_val_r = n_l + 1 + min([coupon_r.index(_) + 1 for _ in coupon_r if r in _])

                        max_val_r = min_val_r + (big_n - (n_l + n_r + 1))
                        """Return either the exact value if val_r[r]∈[minVal_r, minVal_r+k] (or its reflection)
                        or mid-point of the recovered interval"""
                        if self.db().__getitem__(r) in range(min_val_r, max_val_r + 1) and not self.__return_mid_point:
                            val_r[r] = self.db().__getitem__(r)
                        elif big_n - self.db().__getitem__(r) + 1 in range(min_val_r, max_val_r + 1) and \
                                not self.__return_mid_point:
                            val_r[r] = big_n - self.db().__getitem__(r) + 1
                        else:
                            val_r[r] = min(min_val_r, max_val_r) + abs(min_val_r - max_val_r) // 2

                    flag = True

        if flag is False:
            log.warning(f"{self.name()} The approximate reconstruction has Failed... Return min(DB) n times")
            val_r = {_: self.db().get_min() for _ in range(len(self.db()))}

        return val_r

    def recover(self, queries: Iterable[Iterable[int]]) -> List[int]:
        log.info(f"Starting {self.name()} with {self.__return_mid_point}, {self.__error}.")

        rids = self.required_leakage()[0](self.db(), queries)

        val = self.__sorting(error=self.__error, rids=rids)

        log.info(f"Reconstruction completed.")

        return [val[i] for i in range(len(val.keys()))]


class LMPaux(RangeAttack):
    """
    Implements the data reconstruction Range attack from [LMP17] using auxiliary distribution for the target dataset
    based on Access pattern & Rank leakage.
    """

    @classmethod
    def name(cls) -> str:
        return "LMP-aux"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Set[int]]]:
        return [ResponseIdentity(), Rank()]

    def __partition(self, rids: List[Set[int]], rank: List[Tuple[int, int]]) \
            -> Dict[int, Tuple[int, int]]:
        leakage = list(zip(rank, rids))
        s_r: Dict[int, Tuple[int, int]] = dict()
        for r in range(len(self.db())):
            intersect_cand = [np.arange(_[0] + 1, _[1] + 1) for _, R in leakage if r in R]
            union_cand = [np.arange(_[0] + 1, _[1] + 1) for _, R in leakage if r not in R]
            intersect_set = reduce(np.intersect1d, intersect_cand) if len(intersect_cand) > 0 else []
            union_set = reduce(np.union1d, union_cand) if len(union_cand) > 0 else []
            max_pos = np.amax(np.setdiff1d(intersect_set, union_set, assume_unique=True))
            min_pos = np.amin(np.setdiff1d(intersect_set, union_set, assume_unique=True))
            s_r[r] = (min_pos, max_pos)

        return s_r

    def __sorting(self, rids: List[Set[int]], rank: List[Tuple[int, int]]) \
            -> Dict[int, int]:
        val_r: Dict[int, int] = dict()
        big_r = self.db().get_n()
        # computing the minimal intervals containing the position of each record
        try:
            s_r = self.__partition(rids, rank)
        except ValueError:
            log.warning(f"{self.name()} Partition Failed... Return min(DB) n times")
            val_r = {_: self.db().get_min() for _ in range(1, len(self.db()) + 1)}
            return val_r
        prob_dist = self.db().get_weights()
        prob_dist = dict(sorted(prob_dist.items()))
        big_z = list(prob_dist.keys())  # distinct values occuring in the DB
        pdf = list(prob_dist.values())
        cdf = np.cumsum(pdf)  # cumulative distribution function of the weights of values in DB

        for r in range(len(self.db())):
            a = s_r[r][0] - 1
            b = s_r[r][1]
            # x-1 = 1/rank(a)= z or z+1 we pick the optimal one based on it's probability
            #  check if z+1 ϵ [1,N] and also P_r[z+1]>P_r[z]
            p_ra = [(z, binom.pmf(k=a, n=big_r, p=cdf[big_z.index(z)]))
                    for z in big_z]  # list[Tuple(z, prob mass func(z))]
            est_z = max(p_ra, key=lambda t: t[1])  # return max(p_ra) based on second emnt which is pmf
            if est_z[0] + 1 in big_z and est_z[1] < binom.pmf(k=a, n=big_r, p=cdf[big_z.index(est_z[0] + 1)]):
                x = est_z[0] + 2
            else:
                x = est_z[0] + 1
            # y = 1/rank(b)= z or z+1 we pick the optimal one based on it's probability
            p_rb = [(z, binom.pmf(k=b, n=big_r, p=cdf[big_z.index(z)])) for z in big_z]
            est_z = max(p_rb, key=lambda t: t[1])
            if est_z[0] + 1 in big_z and est_z[1] < binom.pmf(k=b, n=big_r, p=cdf[big_z.index(est_z[0] + 1)]):
                y = est_z[0] + 1
            else:
                y = est_z[0]

            if x > y:
                x, y = y, x  # switch x & y values if x>y to generate range[x,y]

            val_r[r + 1] = np.round(sum([i * prob_dist[i] for i in range(x, y + 1) if i in big_z]
                                        ) / sum([prob_dist[i] for i in range(x, y + 1) if i in big_z]))

        return val_r

    def recover(self, queries: Iterable[Tuple[int, int]]) -> List[int]:
        log.info(f"Starting {self.name()}.")

        rids = self.required_leakage()[0](self.db(), queries)

        rank = self.required_leakage()[1](self.db(), queries)

        val = self.__sorting(rids, rank)

        log.info(f"Reconstruction completed.")

        return [val[i + 1] for i in range(len(val.keys()))]
