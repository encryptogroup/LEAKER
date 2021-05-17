"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import itertools
from logging import getLogger

import networkx as nx
import sys

from ..api.constants import PYTHON_DIST_PACKAGES_DIRECTORY

from math import ceil, sqrt
from typing import List, Set, Iterable, Tuple, FrozenSet

from scipy.special import comb

from ..api import RangeAttack, LeakagePattern
from ..pattern import ResponseLength

log = getLogger(__name__)

# To make graph-tool work, since it is not given as a classic python module
sys.path.append(PYTHON_DIST_PACKAGES_DIRECTORY)
try:
    from graph_tool.all import *
    import_graph_tool_success = True
except ImportError:
    log.debug(f"graph_tool could not be imported, will use NetworkX instead.")
    import_graph_tool_success = False


class GLMP18(RangeAttack):
    """
    Implements the Range attack from [GLMP18].
    """

    @classmethod
    def name(cls) -> str:
        return "GLMP18"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[int]]:
        return [ResponseLength()]

    @staticmethod
    def __augment_nec(v_cand: Set[int], v_nec: Set[int], o_v_comp: Set[int], n_min: int) -> Set[int]:
        if len(v_cand) == n_min:
            return v_cand.copy()

        v_nec = v_nec.copy()  # since we alter v_nec here

        for e in o_v_comp:
            for v in v_cand.difference(v_nec):
                diff = v_cand.difference({v})
                if all(abs(prod[0] - prod[1]) != e for prod in itertools.combinations(diff, 2)):
                    v_nec.add(v)

        for v in v_cand.difference(v_nec):
            diff = v_cand.difference({v})
            if all(abs(prod[0] - prod[1]) != v for prod in itertools.combinations(diff, 2)):
                v_nec.add(v)

        return v_nec

    @staticmethod
    def __reduce_cand(v_cand: Set[int], v_nec: Set[int], rlens: Set[int]) -> Set[int]:
        v_cand = v_cand.copy()  # since we alter v_cand here

        for v in v_cand.difference(v_nec):
            if any(abs(v - vn) not in rlens for vn in v_nec):
                v_cand.remove(v)

        return v_cand

    @classmethod
    def __graph_preproc(cls, big_n: int, rlens: Set[int]) -> Tuple[Set[int], Set[int]]:
        rlens = rlens.copy()
        r = max(rlens)
        if 0 in rlens:
            rlens.remove(0)
            n_min = int(ceil(-0.5 + 0.5 * sqrt(1 + 8 * len(rlens))))
        else:
            n_min = big_n

        v_comp = {l for l in rlens if r - l in rlens}
        v_comp.add(r)
        o_v_comp = rlens.difference(v_comp)
        v_min = min(v_comp)
        v_nec = {v_min, r}
        v_cand = v_comp.copy()

        all_processed = False
        while not all_processed:
            s_v_nec = cls.__augment_nec(v_cand, v_nec, o_v_comp, n_min)
            s_v_cand = cls.__reduce_cand(v_cand, s_v_nec, rlens)

            all_processed = s_v_cand == v_cand and s_v_nec == v_nec

            v_nec = s_v_nec.copy()
            v_cand = s_v_cand.copy()

        return v_cand, v_nec

    @classmethod
    def __find_maximal_cliques(cls, cand_nn: Set[int], m_min: int, v_all: Set[int], v_nec: Set[int]) -> List[Set[int]]:
        edges = [(a, b) for a, b in itertools.combinations(cand_nn, 2) if abs(a - b) in v_all]

        if len(cand_nn) <= 20 or not import_graph_tool_success:
            log.debug(f"Using NetworkX")
            g = nx.Graph()
            g.add_edges_from(edges)
            cliques = list(nx.find_cliques(g))
        else:
            log.debug(f"Using graph-tool")
            ref_list = list(cand_nn)  # we have to map to 0...|cand_nn| - 1
            g = Graph(directed=False)
            g.add_vertex(len(cand_nn))
            for s, t in [(ref_list.index(a), ref_list.index(b)) for a, b in edges]:
                g.add_edge(s, t)

            cliques = []
            i = 0
            for clique in max_cliques(g):
                if i >= 10**4:
                    log.debug(f"Stopped at 10^4 cliques!")
                    break
                if len(clique) > m_min:
                    cliques.append(set(ref_list[v] for v in clique))  # convert back to original representation
                i += 1

            if len(cliques) == 0:
                log.warning(f"Could not find any cliques!")
            elif not any(cls.__gen_all_volumes(v_nec.union(v_k), v_all) for v_k in cliques):
                log.warning(f"Could not find any clique that generates all volumes!")

        log.debug(f"Got {len(cliques)} cliques")
        return cliques

    @classmethod
    def __gen_all_volumes(cls, v_nodes: Set[int], v_all: Set[int]) -> bool:
        for v in v_all.difference(v_nodes):  # if v in v_nodes it is already generated
            if all(abs(perm[0] - perm[1]) != v for perm in itertools.combinations(v_nodes, 2)):
                return False

        return True

    @classmethod
    def __gen_exact_volumes(cls, v_nodes: Set[int], v_all: Set[int]) -> bool:
        if v_nodes.issubset(v_all) and cls.__gen_all_volumes(v_nodes, v_all):
            for prod in itertools.combinations(v_nodes, 2):
                if abs(prod[1] - prod[0]) not in v_all:
                    return False
            return True
        else:
            return False

    @classmethod
    def __min_subcliques(cls, v_k: Set[int], v_all: Set[int], m_min: int, m_max: int, v_nec: Set[int]) \
            -> Set[FrozenSet[int]]:
        possibilities = sum(comb(len(v_k), m) for m in range(m_min, m_max + 1))
        if possibilities <= 5e4:
            log.debug(f"Returning all subcliques")
            return cls.__all_subcliques_p(v_k, v_all, m_min, m_max, v_nec)
        elif possibilities <= 1e8:
            log.debug(f"Trying to find one subclique... "
                      f"(#combinations {possibilities} exceeds 50k)")
            i = 0
            for m in range(m_min, min(m_max, len(v_k)) + 1):
                for v_sk in itertools.combinations(v_k, m):
                    if cls.__gen_exact_volumes(v_nec.union(v_sk), v_all):
                        log.debug(f"... success! Found a subclique")
                        return {frozenset(v_nec.union(v_sk))}
                    elif i >= 1e8:
                        log.warning(f"... failed!")
                        break
                    i += 1
                return set(frozenset())
        else:
            log.debug(f"#combinations {possibilities} is too large to try and find a subclique.")
            return set(frozenset())

    @classmethod
    def __all_subcliques_p(cls, v_k: Set[int], v_all: Set[int], m_min: int, m_max: int, v_nec: Set[int]) \
            -> Set[FrozenSet[int]]:
        subcliques: Set[FrozenSet[int]] = set()
        for m in range(m_min, min(m_max, len(v_k)) + 1):
            for v_sk in itertools.combinations(v_k, m):
                if cls.__gen_exact_volumes(v_nec.union(v_sk), v_all):
                    subcliques.add(frozenset(v_nec.union(v_sk)))

        return subcliques

    def __get_elem_volumes(self, big_n: int, v_cand: Set[int], v_nec: Set[int], rlens: Set[int]) \
            -> FrozenSet[int]:
        if len(v_cand) == len(v_nec):
            log.debug(f"Preprocessing already found a solution")
            return frozenset(v_nec)
        else:
            log.debug(f"Preprocessing did not find a solution (differs by {len(v_cand.symmetric_difference(v_nec))})")

        if 0 in rlens:
            rlens.remove(0)
            n_min = int(ceil(-0.5 + 0.5 * sqrt(1 + 8 * len(rlens))))
            n_max = min(big_n - 1, len(v_cand))
        else:
            n_min = big_n
            n_max = big_n

        m_min = max(0, n_min - len(v_nec))
        m_max = n_max - len(v_nec)

        cand_nn = v_cand.difference(v_nec)
        cliques = self.__find_maximal_cliques(cand_nn, m_min, rlens, v_nec)
        solutions = []

        for v_k in cliques:
            if len(v_k) < m_min:
                continue
            if self.__gen_all_volumes(v_nec.union(v_k), rlens):
                solutions.append(self.__min_subcliques(set(v_k), rlens, m_min, m_max, v_nec))

        res = set()
        found = False  # if no one fitting solution exists, we just use the first one
        for solution in solutions:
            if len(solution) == 1:
                if not found:
                    res = solution.pop()
                    found = True
                else:
                    log.warning(f"Found more than one valid solution. Using the first one...")
                    break

        if not found:
            warn = f"Found no valid solution,"
            if len(solutions) == 0:
                warn += " using default fallback (no solutions exist)."
            else:
                sol = solutions.pop()
                for s in sol:
                    u_l = list(sorted(s))
                    if not found and u_l[0] + sum(y - x for x, y in zip(u_l, u_l[1:])) == len(self.db()):
                        warn += f" using first fitting one."
                        res = s
                        found = True
                        break
                if not found:
                    warn += " using default fallback (solutions exist but do not fit)."
            log.warning(warn)

        return res

    def recover(self, queries: Iterable[Tuple[int, int]]) -> List[int]:
        log.info(f"Starting with {self.name()}")
        big_n = self.db().get_max()

        rlens = set(self.required_leakage()[0](self.db(), queries))

        v_cand, v_nec = self.__graph_preproc(big_n, rlens)

        solutions = self.__get_elem_volumes(big_n, v_cand, v_nec, rlens)

        res = [len(self.db()) // big_n for _ in range(big_n)]

        if len(solutions) > 0:
            solutions = list(sorted(solutions))
            res = [solutions[0]]
            for k in range(1, len(solutions)):
                res.append(solutions[k] - solutions[k - 1])

            if len(res) > big_n:
                res = res[:big_n]
            elif len(res) < big_n:
                """Fallback: Each remaining value appears (len(db) - #recovered)/N times"""
                missing_value_count = big_n - len(res)
                res.extend([max(0, (len(self.db()) - sum(res)) // big_n)
                            for _ in range(missing_value_count)])

        log.info(f"Reconstruction completed")

        return res
