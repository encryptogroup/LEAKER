"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from typing import Iterable, List

from ..api import LeakagePattern, Dataset
from ..extension import VolumeExtension


class Volume(LeakagePattern[List[int]]):
    """
    The volume (vol) leakage pattern leaking the volume of every document matching any given query.
    """

    def leak(self, dataset: Dataset, keywords: Iterable[str]) -> List[List[int]]:
        if dataset.has_extension(VolumeExtension):
            volume = dataset.get_extension(VolumeExtension)
            return [volume.volumes(q) for q in keywords]
        else:
            return [list(map(lambda doc: doc.length(), dataset(q))) for q in keywords]


class TotalVolume(LeakagePattern[int]):
    """
    The total volume (tvol) leakage pattern leaking the total volume of the documents matching any given query.
    """

    def leak(self, dataset: Dataset, queries: Iterable[str]) \
            -> List[int]:
        if dataset.has_extension(VolumeExtension):
            volume = dataset.get_extension(VolumeExtension)
            return [volume.total_volume(q) for q in queries]
        else:
            return [sum(map(lambda doc: doc.length(), dataset(q))) for q in queries]
