"""
For License information see the LICENSE file.

Authors: Michael Yonli

"""
from logging import getLogger
from typing import List, Any, Iterable

from leaker.api import RangeDatabase
from .agnostic_reconstruction import arr
from ..glmp19 import ApproxOrder
from ...api import RangeAttack, LeakagePattern
from ...pattern import ResponseIdentity, Rank

log = getLogger(__name__)


class Arr(RangeAttack):
    """
    Implements Agnostic Reconstruction Range from [KPT20] with an error function allowing for repeated values.
    Reconstructs the database using range query tokens.
    The ApproxOrder algorithm from [GLMP19] will be used to get the ordering.

    :param processes: the number of processes to use (all available CPUs if None) - Can only be reliably used if
        parallelism of the Evaluator is 1
    :param e: minimum ARR weight
    """

    _leak_order: bool
    __processes: int
    __e: float

    def __init__(self, db: RangeDatabase, processes=1, e=0.01):
        super().__init__(db)
        assert processes is None or processes > 0
        self.__processes = processes
        self.__e = e
        self._leak_order = False

    @classmethod
    def name(cls) -> str:
        """
        :return: name of the attack
        """
        return "ARR"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [ResponseIdentity(), Rank()]

    def recover(self, queries: Iterable) -> List[float]:
        """
        Runs the Agnostic Reconstruction Range attack from KPT19.
        :param queries: An iterable representing the queries that the attacker has access to.
        :return: A list of reconstructed database values.
        """
        log.info(f"Starting {self.name()} with {self.__processes} processes")
        if self.__processes != 1:
            log.warning(f"Using multiprocessing for {self.name()} within a parallel Evaluator is unsafe and can lead"
                        f" to concurrency problems.")
        db = self.db()

        a = db.get_min()
        b = db.get_max()
        n = db.get_n()

        order = [None] * n
        if self._leak_order:
            for entry in db.get_numerical_values():
                rank = db.get_rank(entry) - 1  # Rank is never 0 for db entries
                rid = list(ResponseIdentity().leak(db, [(entry, entry)])[0])
                num = len(rid)
                for x in range(num):
                    order[rank - x] = rid[x]
        else:
            order_rec = ApproxOrder(db, attempt_val_rec=False, bucket_error_rec=False)
            for i, val in enumerate(order_rec.recover(queries)):
                order[i] = val

        assert (None not in order)

        tokens = []
        for q in queries:
            result = tuple(ResponseIdentity().leak(db, [q])[0])
            num_token = q[0] * (b + 1) + q[1]
            tokens.append((num_token, result))

        recovered = arr(tokens, order, a, b, n, processes=self.__processes, e=self.__e)

        log.info(f"Reconstruction completed")

        return recovered


class Arrorder(Arr):
    """Implements the case if the order is explicitly leaked"""

    def __init__(self, db: RangeDatabase, processes=1, e=0.01):
        super().__init__(db, processes, e)
        self._leak_order = True

    @classmethod
    def name(cls) -> str:
        """
        :return: name of the attack
        """
        return "ARRorder"
