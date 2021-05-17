"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from logging import getLogger
from typing import Iterable, List, Any, Dict, Set, Type, TypeVar

from ..api import KeywordAttack, Dataset, LeakagePattern, Extension
from ..extension import VolumeExtension
from ..pattern import TotalVolume

E = TypeVar("E", bound=Extension, covariant=True)

log = getLogger(__name__)


class VolAn(KeywordAttack):
    """
    Implements the VolAn attack from "Revisiting Leakage Abuse Attacks". It uses the TotalVolume pattern.
    """

    __known_volume: Dict[str, int]
    __delta: float

    def __init__(self, known: Dataset):
        super(VolAn, self).__init__(known)

        self.__known_volume = dict()

        if not known.has_extension(VolumeExtension):
            known.extend_with(VolumeExtension)

        vol = known.get_extension(VolumeExtension)
        for keyword in known.keywords():
            self.__known_volume[keyword] = vol.total_volume(keyword)

        self.__delta = known.sample_rate()

    @classmethod
    def name(cls) -> str:
        return "VolAn"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [TotalVolume()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {VolumeExtension}

    def __recover_query(self, tvol: int) -> str:
        # get item with maximum volume still below the observed volume
        less_than_observed_volume = filter(lambda item: item[1] <= tvol, self.__known_volume.items())
        max_volume_item = max(less_than_observed_volume, key=lambda item: item[1], default=("", 0))

        return max_volume_item[0]

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running VolAn at {self.__delta:.3f}.")
        tvols = self.required_leakage()[0](dataset, queries)
        res = [self.__recover_query(tvol) for tvol in tvols]
        log.info(f"Reconstruction completed.")
        return res
