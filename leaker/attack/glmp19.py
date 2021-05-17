"""
For License information see the LICENSE file.

Authors: Abdelkarim Kati

"""
import itertools
from logging import getLogger
from typing import List, Iterable, Set, Dict, Any

import numpy as np

from ..pattern import ResponseIdentity
from ..api import RangeAttack, LeakagePattern, RangeDatabase

# pq-trees lib after being added as a submodule in leaker dir
__import__("leaker.pq-trees")  # automatic compilation
import sys
from pathlib import Path

path = Path(__file__).resolve().parents[1]  # here path.parents[1]` is the same as `path.parent.parent
sys.path.insert(1, str(path) + '/pq-trees/build')

from pqtree_cpp import PQTree, PQNode, PQNodeArray, PQNodeDict

log = getLogger(__name__)


class ApproxValue(RangeAttack):
    """Implements the ɛ-approximate database reconstruction (ɛ-ADR) attack from [GLMP19]"""

    @classmethod
    def name(cls) -> str:
        return "ApproxValue"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Set[int]]]:
        return [ResponseIdentity()]

    def prob(self, k: int) -> float:
        big_n = self.db().get_max()
        return 2 * k * (big_n + 1 - k) / (big_n * (big_n + 1))

    def anchor_dist(self, v_a: int, k: int) -> float:
        big_n = self.db().get_max()
        if k <= v_a:
            dist = 2 * k * (big_n + 1 - v_a) / (big_n * (big_n + 1))
        else:
            dist = 2 * v_a * (big_n + 1 - k) / (big_n * (big_n + 1))
        return dist

    def __get_anchor(self, rids: List[Set[int]]) -> List[int]:
        big_n = self.db().get_max()
        v_tilde: List[int] = []
        est_val: Dict[int, int] = dict()
        c: Dict[int, float] = dict()

        for r in range(len(self.db())):
            c[r] = sum([1 for q in rids if r in q]) / len(rids)
            dist_array = np.absolute([c[r] - self.prob(k) for k in range(1, big_n + 1)])
            v_tilde.append(np.argmin(dist_array))

        dist_array = np.absolute([val - big_n / 4 for val in v_tilde])
        r_a = np.argmin(dist_array)
        v_a = v_tilde[r_a] + 1  # map back to 1...n

        for r in range(len(self.db())):
            c_p = sum([1 for records in rids if r in records and r_a in records]) / len(rids)
            dist_l = np.absolute([self.anchor_dist(v_a, k) - c_p for k in range(1, v_a + 1)])
            w_l = np.argmin(dist_l) + 1  # map back to 1...v_a
            dist_r = np.absolute([self.anchor_dist(v_a, k) - c_p for k in range(v_a, big_n + 1)])
            w_r = np.argmin(dist_r) + v_a  # map back to v_a ... N

            if c[r] < (self.prob(w_l) + self.prob(w_r)) / 2:
                est_val[r] = w_l
            else:
                est_val[r] = w_r

        return [est_val[r] for r in range(len(self.db()))]

    def recover(self, queries: Iterable[Iterable[int]]) -> List[int]:
        log.info(f"Starting {self.name()}.")

        rids = self.required_leakage()[0](self.db(), queries)

        est_val = self.__get_anchor(rids)

        log.info(f"Reconstruction completed.")

        return est_val


class ApproxOrder(RangeAttack):
    """Implements the ɛ-approximate Order reconstruction (ɛ-AOR) attack from [GLMP19]"""

    __attempt_val_rec: bool
    __bucket_error_rec: bool

    def __init__(self, db: RangeDatabase, attempt_val_rec: bool = False, bucket_error_rec: bool = False):
        self.__attempt_val_rec = attempt_val_rec
        self.__bucket_error_rec = bucket_error_rec
        super(ApproxOrder, self).__init__(db)

    @classmethod
    def name(cls) -> str:
        return "ApproxOrder"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Set[int]]]:
        return [ResponseIdentity()]

    def __find_node_t(self, tree: PQTree, node: PQNode) -> PQNode:
        root: PQNode = tree.Root()
        leaves = PQNodeDict()
        root.FindLeaves(leaves)
        big_r = len(leaves)

        children = PQNodeArray()
        node.Children(children)

        for i in range(len(children)):
            leaf_address = PQNodeDict()
            child: PQNode = children[i]
            child.FindLeaves(leaf_address)
            if len(leaf_address) > big_r // 2:
                return self.__find_node_t(tree, child)

        return node

    def __get_aor(self, rids: List[Set[int]]) -> List[Any]:
        big_r = self.db().get_n()
        ground_set = set(range(big_r))
        buckets: List[Any] = []
        tree = PQTree(ground_set)
        for rid in rids:
            tree.SafeReduce(rid)
        root: PQNode = tree.Root()
        node_t: PQNode = self.__find_node_t(tree, root)

        leaf_address = PQNodeDict()
        node_t.FindLeaves(leaf_address)
        children = PQNodeArray()
        node_t.Children(children)
        '''now lets get the leaves of each child of the result'''
        for i in range(len(children)):
            leaves = PQNodeDict()
            child: PQNode = children[i]
            child.FindLeaves(leaves)
            buckets.append([item for item in leaves])

        return buckets

    def recover(self, queries: Iterable[Iterable[int]]) -> List[Any]:
        log.info(f"Starting {self.name()}.")

        rids = self.required_leakage()[0](self.db(), queries)

        buckets = self.__get_aor(rids)

        '''
        If indicated by __attempt_val_rec, check if the DB is dense so we can recreate the exact values up to a
        reflection based on the order
        
        If indicated by __bucket_error_rec, we shall return the recreated buckets to be evaluated by the MaxABucketError
        
        Else we shall return the recreated order coupled with a fallback in case we don't have sufficient #Queries
        The Fallback append a shuffled array of missing indices to the recover array
        '''
        density = self.db().get_density()
        big_r = self.db().get_n()
        db_min = self.db().get_min()
        first_bucket = self.db().__getitem__(buckets[0][0])

        if density == 1 and self.__attempt_val_rec:

            if first_bucket != db_min:
                # remove the reflection
                buckets.reverse()
            op = {i: buckets[i] for i in range(0, len(buckets))}
            est_val: Dict[int, int] = dict()
            for k, v in op.items():
                est_val.update({_: k for _ in v})
            recover = [val[1] + 1 for val in sorted(est_val.items())]
            if len(recover) != big_r:
                # Fallback for low #Queries
                missing = [self.db().get_min() for _ in range(len(recover), len(self.db()))]
                recover += missing
            return recover

        elif self.__bucket_error_rec:
            return buckets

        recover = list(itertools.chain(*buckets))  # map the buckets to an array of ordered indices,
        if len(recover) != big_r:
            '''
            The recovered array length is not equal to the number of items in the DB
            since the deepest PQNode doesn't contain len(recs)>R/2
            '''
            missing = list(filter(lambda x: x not in recover, list(range(0, big_r))))
            np.random.shuffle(missing)
            recover += missing

        log.info(f"Reconstruction completed.")
        return recover
