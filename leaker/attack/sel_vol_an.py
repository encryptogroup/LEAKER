"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from itertools import starmap
from logging import getLogger
from math import floor
from typing import Iterable, List, Any, Dict, Set, TypeVar, Type

from ..api import KeywordAttack, Dataset, LeakagePattern, Extension
from ..extension import VolumeExtension
from ..pattern import ResponseLength, TotalVolume

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class SelVolAn(KeywordAttack):
    """
    Implements the SelVolAn attack from "Revisiting Leakage Abuse Attacks". It uses the TotalVolume and the
    ResponseLength patterns.

    Other Parameters
    ----------------
    epsilon : int
        the epsilon error parameter
        default: 1
    """
    __known_volume: Dict[str, int]
    __known_response_length: Dict[str, int]

    __delta: float
    __epsilon: int = 100
    __theta: int

    def __init__(self, known: Dataset, epsilon: int = 1):
        super(SelVolAn, self).__init__(known)

        if epsilon < 1:
            raise ValueError("epsilon must not be less than 1.")

        self.__known_volume = dict()
        self.__known_response_length = dict()

        if not known.has_extension(VolumeExtension):
            known.extend_with(VolumeExtension)

        vol = known.get_extension(VolumeExtension)
        for keyword in known.keywords():
            self.__known_volume[keyword] = vol.total_volume(keyword)
            self.__known_response_length[keyword] = vol.selectivity(keyword)

        self.__delta = known.sample_rate()
        self.__epsilon = epsilon
        self.__theta = vol.dataset_volume() // len(known.doc_ids())

    @classmethod
    def name(cls) -> str:
        return "SelVolAn"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [TotalVolume(), ResponseLength()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {VolumeExtension}

    def __fallback(self, bound: int) -> str:
        less_than_delta_volume = filter(lambda item: item[1] <= bound, self.__known_volume.items())
        max_volume_item = max(less_than_delta_volume, key=lambda item: item[1], default=("", 0))

        return max_volume_item[0]

    def __window_matching(self, tvol: int) -> Set[str]:
        # determine all elements in the volume window
        lower_volume = floor(self.__delta * tvol)
        inside_window = set(map(lambda item: item[0], filter(lambda item: lower_volume <= item[1] <= tvol,
                                                             self.__known_volume.items())))

        if len(inside_window) > 0:
            return inside_window
        else:
            log.debug("No candidate in window, using fallback.  ")
            return {self.__fallback(lower_volume)}

    def __selectivity_filtering(self, candidates: Set[str], tvol: int, rlen: int) -> Set[str]:
        filtered: Set[str] = set()
        for w in candidates:
            # apply the filtering formula from the paper
            _lambda = (1 - self.__delta) * rlen / self.__epsilon
            expected_rlen = rlen - (tvol - self.__known_volume[w]) / self.__theta

            if expected_rlen - _lambda <= self.__known_response_length[w] <= rlen:
                filtered.add(w)
        return filtered

    def __recover_query(self, tvol: int, rlen: int) -> str:
        candidates = self.__window_matching(tvol)

        if len(candidates) == 1:
            # if already only one candidate left, we do not need to apply selectivity filtering
            return candidates.pop()

        candidates = self.__selectivity_filtering(candidates, tvol, rlen)
        if len(candidates) == 0:
            # if selectivity filtering leaves no candidate, we can still fall back to the candidate with the highest
            # known volume
            log.debug("No matching candidate found - falling back to VolAn method")
            return self.__fallback(tvol)

        return max(candidates, key=self.__known_volume.get)

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running SelVolAn at {self.__delta:.3f}.")
        leakage = zip(self.required_leakage()[0](dataset, queries), self.required_leakage()[1](dataset, queries))
        res = list(starmap(self.__recover_query, leakage))
        log.info(f"Reconstruction completed.")
        return res
