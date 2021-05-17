"""
For License information see the LICENSE file.

Authors: Michael Yonli

"""
from logging import getLogger
from typing import List, Iterable, Union, Any, Set, Tuple, FrozenSet
import numpy as np

from ..pattern import ResponseLength
from ..api import RangeAttack, LeakagePattern, RangeDatabase, AbortException

log = getLogger(__name__)


class GJWbasic(RangeAttack):
    """
    Implements the Volume attack from GJW19

    Only recovers the volumes of the entries in the db.
    It is assumed that ALL queries with bound b are issued.
    """

    _bound: int

    def __init__(self, db: RangeDatabase, bound: int = 3):
        self._bound = bound
        super().__init__(db)

    @classmethod
    def name(cls) -> str:
        return "GJW-Basic"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [ResponseLength()]

    def recover(self, queries: Iterable[Iterable[Union[int, float]]]) -> List[int]:
        log.debug(f"Starting with {self.name()}.")
        volumes: List[int] = self.required_leakage()[0](self.db(), queries)

        recovered = self._postprocess(self._attack(set(volumes), self._bound,
                                                   self.db().get_max() - self.db().get_min() + 1))

        log.debug(f"Reconstruction completed.")

        return recovered

    @staticmethod
    def _big_l(v: Tuple[int], b: int) -> Set[int]:
        big_w = {sum(v[start:start + num]) for start in range(len(v)) for num in range(1, b + 1)}

        return big_w

    @staticmethod
    def _abort_check(big_si: Set[Tuple[int]], big_w: Set[int]) -> None:
        if len(big_si) >= len(big_w) ** 3:
            log.warning(f"Aborted due to size surpassing |W|^3!")
            raise AbortException

    def _postprocess(self, big_sn: Set[Tuple[int]]) -> List[int]:
        big_n = self.db().get_max()
        if len(big_sn) == 0:
            """Fallback: Each value appears len(db)/max times"""
            log.warning(f"{self.name()} could not find a solution!")
            recovered = [len(self.db()) // big_n for _ in range(big_n)]
        else:
            candidate: Union[Tuple[int], None] = None
            for c in big_sn:
                if sum(c) == len(self.db()):
                    candidate = c

            if candidate is None:
                log.warning(f"Did not find any candidate creating the observed volumes!")
                candidate = big_sn.pop()
            else:
                big_sn.remove(candidate)

            """Remove reflection from big_sn"""
            r_candidate = tuple(reversed(candidate))
            if r_candidate in big_sn:
                big_sn.remove(r_candidate)

            recovered = list(candidate)

            if len(big_sn) > 0:
                log.warning(f"{self.name()} found more than one solution ({len(big_sn) + 1})! Using the first one.")
            else:
                log.debug(f"{self.name()} found a unique solution.")

            if len(recovered) > big_n:
                recovered = recovered[:big_n]
            elif len(recovered) < big_n:
                """Fallback: Each remaining value appears (len(db) - #recovered)/N times"""
                missing_value_count = big_n - len(recovered)
                recovered.extend([max(0, (len(self.db()) - sum(recovered)) // big_n)
                                  for _ in range(missing_value_count)])

        return recovered

    @classmethod
    def _initial_solution(cls, big_w: Set[int], b: int) -> Set[Tuple[int]]:
        """
        Implements initial_solution Algorithm 1
        :param big_w: Leakage
        :param b: bound on queries
        :return: Initial solution
        """
        assert b >= 3
        big_c: Set[int] = {v for v in big_w if max(big_w) - v in big_w}
        big_c.add(max(big_w))

        last_big_g: Set[FrozenSet] = {frozenset([v, max(big_w)]) for v in big_c if v != max(big_w)}

        for i in range(3, b + 1):
            big_gi: Set[FrozenSet] = set()
            for g in last_big_g:
                for v in big_c:
                    temp = {abs(h - v) for h in g}
                    if temp <= big_w:
                        frozen = set(g)
                        frozen.add(v)
                        big_gi.add(frozenset(frozen))
            if not big_gi:
                log.error("Clique failure")
                raise Exception
            last_big_g = big_gi

        big_s: Set[Tuple[int]] = set()
        for g in last_big_g:
            g = sorted(list(g))
            s = [g[0]]
            for i, x in enumerate(g[1:]):
                s.append(x - g[i])
            big_s.add(tuple(s))

        return big_s

    def _attack(self, big_w: Set[int], b: int, big_n: int) -> Set[Tuple[int]]:
        last_big_s = self._initial_solution(big_w, b)
        log.debug(f"Found initial solution.")
        try:
            for _ in range(big_n - b):
                big_si = self._extend_left(last_big_s, big_w, b).union(self._extend_right(last_big_s, big_w, b))
                last_big_s = big_si
                self._abort_check(big_si, big_w)
            big_sn = {s for s in last_big_s if self._big_l(s, b) == big_w}
        except AbortException:
            big_sn = set()

        return big_sn

    @staticmethod
    def _extend_left(big_si: Set[Tuple[int]], big_w: Set[int], b: int) -> Set:
        big_s = set()
        for s in big_si:
            for w in big_w:
                temp = {w + sum(s[:x]) for x in range(1, b)}
                if temp <= big_w:
                    big_s.add(tuple([w]) + s)
        return big_s

    @staticmethod
    def _extend_right(big_si: Set[Tuple[int]], big_w: Set[int], b: int) -> Set:
        big_s = set()
        for s in big_si:
            for w in big_w:
                temp = {w + sum(list(reversed(s))[:x]) for x in range(1, b)}
                if temp <= big_w:
                    big_s.add(s + tuple([w]))
        return big_s


class GJWspurious(GJWbasic):
    """Implements 3.1 from GJW19"""

    __noise_alpha: float

    def __init__(self, db: RangeDatabase, bound: int = 3, noise_alpha: float = 0.5):
        self.__noise_alpha = noise_alpha
        super().__init__(db, bound)

    @classmethod
    def name(cls) -> str:
        return "GJW-spurious"

    def recover(self, queries: Iterable[Iterable[Union[int, float]]]) -> List[int]:
        log.info(f"Starting with {self.name()}.")
        volumes: Set[int] = set(self.required_leakage()[0](self.db(), queries))

        n_noise = min(int(len(volumes) * self.__noise_alpha), max(volumes) - min(volumes) + 1)
        noise = np.random.choice(range(min(volumes), max(volumes) + 1), n_noise, replace=False)
        volumes = volumes.union(set(noise))

        recovered = self._postprocess(self._attack(volumes, self._bound,
                                                   self.db().get_max() - self.db().get_min() + 1))

        log.info(f"Reconstruction completed.")

        return recovered

    def _initial_solution(self, big_w: Set[int], b: int) -> Set[Tuple[int]]:
        big_w = sorted(list(big_w))
        # Should be more efficient if failure to identify correct results only depends on max(w)
        volumes = [big_w[:x] for x in range(1, len(big_w))]
        initial_solutions = set()

        try:
            for volume in volumes:
                volume = set(volume)
                try:
                    sol = super()._initial_solution(volume, b)
                except Exception:
                    continue
                if sol:
                    initial_solutions = initial_solutions.union(sol)
                    self._abort_check(initial_solutions, big_w)
        except AbortException:
            log.warning(f"Initial solution aborted.")
            pass

        return initial_solutions

    def _attack(self, big_w: Set[int], b: int, big_n: int) -> Set[Tuple[int]]:
        last_big_s = self._initial_solution(big_w, b)
        try:
            for _ in range(big_n - b):
                big_si = self._extend_left(last_big_s, big_w, b).union(self._extend_right(last_big_s, big_w, b))
                last_big_s = big_si
                self._abort_check(big_si, big_w)
            big_sn = {s for s in last_big_s if self._big_l(s, b) <= big_w}
        except AbortException:
            big_sn = set()
            log.warning(f"Attack aborted.")

        return big_sn


class GJWmissing(GJWbasic):

    _k: int

    @classmethod
    def name(cls) -> str:
        return "GJW-missing"

    def __init__(self, db: RangeDatabase, bound: int = 3, k: int = 1):
        self._k = k
        super().__init__(db, bound)

    def _attack(self, big_w: Set[int], b: int, big_n: int) -> Set[Tuple[int]]:
        last_big_s = {(min(big_w),)}
        try:
            for _ in range(big_n - 1):
                big_si = self._extend_left(last_big_s, big_w, b).union(self._extend_right(last_big_s, big_w, b))
                last_big_s = big_si
                self._abort_check(big_si, big_w)
            big_sn = last_big_s
        except AbortException:
            big_sn = set()

        return big_sn

    def _extend_left(self, big_si: Set[Tuple[int]], big_w: Set[int], b: int) -> Set:
        big_s = set()
        for s in big_si:
            for w0 in big_w:
                m = 0
                i = len(s)
                s = tuple([w0]) + s
                for y in range(2, min(b, i + 1)):
                    for x in range(y):
                        if sum(s[x:y]) not in big_w:
                            m += 1
                            if m > self._k:
                                continue
                    else:
                        continue
                    break
                if m <= self._k:
                    big_s.add(s)
        return big_s

    def _extend_right(self, big_si: Set[Tuple[int]], big_w: Set[int], b: int) -> Set:
        big_s = set()
        for s in big_si:
            for w in big_w:
                m = 0
                i = len(s)
                s_p = tuple([0]) + s + tuple([w])
                for x in range(max(1, i - b + 1), len(s_p) - 1):
                    for y in range(x + 2, len(s_p) + 1):
                        if sum(s_p[x:y]) not in big_w:
                            m += 1
                            if m > self._k:
                                break
                    else:
                        continue
                    break
                if m <= self._k:
                    big_s.add(s + tuple([w]))
        return big_s


class GJWbounded(GJWbasic):
    # WARNING: Untested Attack

    @classmethod
    def name(cls) -> str:
        return "GJW-bounded"

    def _big_l(self, a: int, b: int, v: Tuple[int]) -> Set[int]:
        big_w = {sum(v[start:start + num]) for start in range(len(v)) for num in range(a, b + 1)}

        return big_w

    def _attack(self, big_w: Set[int], a: int, b: int, big_n: int) -> Set[Tuple[int]]:
        assert b // a > 1  # Attack relies on this constraint

        big_x = set()

        big_s = self._initial_solution(big_w, b // a, big_n // a)
        for s in big_s:
            big_s_primed = self._offset_solutions(s, big_w, b // a, big_n // a)
            big_s_primed = self._merge_solutions(big_s_primed, s, big_w, a, b, big_n)
            for s_primed in big_s_primed:
                s_primed = self._finalise_solution(s_primed, big_w, a, b, big_n)
                big_x.add(s_primed)

        big_x = {x for x in big_x if self._big_l(a, b, x) == big_w}

        return big_x

    def _initial_solution(self, big_w: Set[int], b: int, big_n: int) -> Set[Tuple[int]]:
        big_s_0 = {tuple([w]) for w in big_w}
        last_big_s = big_s_0

        for i in range(1, big_n):
            big_s_i = self._extend_left(last_big_s, big_w, b).union(self._extend_right(last_big_s, big_w, b))
            last_big_s = big_s_i

        for s in last_big_s:
            if s[::-1] in last_big_s:
                last_big_s.remove(s)

        return last_big_s

    def _offset_solutions(self, s: Tuple[int], big_w: Set[int], b: int, big_n: int):
        big_s_0 = set()

        for w in big_w:
            if s[0] < w < s[0] + s[1] \
                    and {sum(s[:k]) - w for k in range(3, b + 2)} <= big_w:
                big_s_0 = big_s_0.union(tuple([w]))

        last_big_s = big_s_0
        for i in range(1, big_n - 2):
            big_s_i = set()
            for s_primed in last_big_s:
                for w in big_w:
                    if sum(s[:i + 1]) < w + sum(s_primed) < sum(s[:i + 2]) \
                            and {w + sum(s_primed[max(0, i - b + 1): i])} <= big_w \
                            and {sum(s[:k]) - sum(s_primed) - w for k in range(i + 3, min(len(s), i + b + 1))} <= big_w:
                        # pseudocode for above says + 1 but I think that is incorrect (code corresponds to + 0)
                        big_s_i.add(s_primed + tuple([w]))
            last_big_s = big_s_i

            big_s_n_1 = set()
            for s_primed in last_big_s:
                if {sum(s) - sum(s_primed[:j]) for j in range(big_n - b, big_n - 1)} <= big_w:
                    big_s_n_1.add(frozenset(s_primed + tuple(sum(s) - sum(s_primed))))

            return big_s_n_1

    #  This function has not been checked for correctness.
    def _merge_solutions(self, big_s: Set[Tuple[int]], s: Tuple[int], big_w: Set[int], a: int, b: int, big_n: int):
        big_s_0 = set(tuple())
        last_big_s = big_s_0

        for i in range(1, big_n // a - 1):
            big_s_i = {s_primed + sum(s[:i + 1]) for s_primed in last_big_s}
            big_x = {sum(s_primed[:i + 1]) for s_primed in big_s}

            for x in big_x:
                for s_primed in big_s_i:
                    if {x + sum(s_primed[a - 2: b])} <= big_w \
                            and {sum(s[:k]) - x for k in range(i + 3, min(len(s) - 1, i + b // a + 2))} <= big_w:
                        big_s_i.add(s_primed + tuple(x))

            big_s_i = {s_primed for s_primed in big_s_i if len(s_primed) == i * a}
            last_big_s = big_s_i

        big_s_primed = set()
        for s_primed in last_big_s:
            temp = [s_primed[0]]
            for i in range(1, big_n - a + 1):
                temp.append(s_primed[i] - s_primed[0])
            temp.append(sum(s) - s[big_n // a - 1] - s_primed[big_n - a])
            temp.append(s[big_n // a - 1] - s[big_n // a])

            big_s_primed.add(temp)

        return big_s_primed

    def _finalise_solution(self, s: Tuple[int], big_w: Set[int], a: int, b: int, big_n: int):
        """
        Assumes that first and last elements of s are compound volumes of size a and attempts to recover the
        elementary volumes of these compound volumes.
        :param s:
        :param big_w:
        :param a:
        :param b:
        :param big_n:
        :return:
        """
        if not s:
            return set()

        big_s_primed = {s[1:-1]}
        for i in range(a):
            big_s_tmp = set()

            for x in big_s_primed:
                for w in big_w:
                    if {w + sum(x[a: k]) for k in range(a, b)} <= big_w:
                        big_s_tmp.add(tuple(w - sum(x[:a])) + x)

            big_s_primed = big_s_tmp

        # I don't see why a reversal would be needed, pseudocode wrong?
        # big_s_primed = {x[::-1] for x in big_s_primed}

        for i in range(a):
            big_s_tmp = set()
            for x in big_s_primed:
                for w in big_w:
                    # Pseudocode appears to be wrong, should be  |x| - b + 2
                    if {w + sum(x[k: len(x) - a + 2]) for k in range(len(x) - b + 2, len(x) - a + 2)} <= big_w:
                        big_s_tmp.add({x + tuple(w - sum(x[- (a - 2 + 1):]))})
            big_s_primed = big_s_tmp
        return big_s_primed


class GJWpartial(GJWbasic):

    @classmethod
    def name(cls) -> str:
        return "GJW-partial"

    def recover(self, queries: Iterable[Iterable[Union[int, float]]]) -> List[int]:
        log.info(f"Starting with {self.name()}.")
        volumes: List[int] = self.required_leakage()[0](self.db(), queries)

        recovered = self._postprocess(self._attack(set(volumes), self._bound,
                                                   self.db().get_max() - self.db().get_min() + 1))

        log.info(f"Reconstruction completed.")

        return recovered

    def _postprocess(self, big_sn: FrozenSet[Tuple[int]]) -> List[int]:
        big_n = self.db().get_max()

        assert len(big_sn) < 2

        if len(big_sn) == 0:
            """Fallback: Each value appears len(db)/max times"""
            log.warning(f"{self.name()} could not find a solution!")
            recovered = [len(self.db()) // big_n for _ in range(big_n)]
        else:
            recovered = list(list(big_sn)[0])
        return recovered

    def _attack(self, big_w: Set[int], b: int, big_n: int) -> FrozenSet[Tuple[int]]:
        big_s_b = self._initial_solution(big_w, b)
        list_big_s = list(big_s_b)

        big_s_i: List[FrozenSet] = [frozenset()] * b

        log.debug(f"Found initial solution.")
        try:
            for idx in range(len(list_big_s)):
                if list_big_s[idx][::-1] in big_s_b:
                    big_s_b.remove(list_big_s[idx])

            big_s_i.append(frozenset(big_s_b))
            for i in range(b + 1, big_n + 1):
                big_s_i.append(self._extend_left(big_s_i[i - 1], big_w, b))
                self._abort_check(big_s_i[i], big_w)

            biggest_i = 0
            for i, big_s in enumerate(big_s_i):
                if len(big_s) == 1:
                    biggest_i = i

            j = biggest_i

            for i in range(j + 1, big_n):
                big_s_i.append(self._extend_right(big_s_i[i - 1], big_w, b))
                self._abort_check(big_s_i[i], big_w)

            for i in range(j + 1, len(big_s_i)):
                if len(big_s_i[i]) == 1:
                    biggest_i = i

            if biggest_i == 0:
                return frozenset()

        except AbortException:
            return frozenset()

        return big_s_i[biggest_i]
