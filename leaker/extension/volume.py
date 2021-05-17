"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from collections import Counter
from logging import getLogger
from typing import Set, Dict, Optional, List

from .selectivity import SelectivityExtension
from ..api import Dataset
from ..cache import Cache

log = getLogger(__name__)


class VolumeExtension(SelectivityExtension):
    """
    An extension caching the document identifiers together with the respective document volume of all documents
    matching every keyword. This extension is a subclass of SelectivityExtension and thus also of IdentityExtension
    and also provides their functionality.

    Parameters
    ----------
    dataset : Dataset
        the data set to build the cache on
    doc_ids : Optional[Set[str]]
        (only used for sub sampling) the document identifiers contained in the sampled data set
    keywords : Optional[Set[str]]
        (only used for sub sampling) the keywords contained in the sampled data set
    original_cache : Optional[Cache[str, Set[str]]]
        (only used for sub sampling) the full cache that needs to be subsampled
    original_cache : Optional[Dict[str, int]]
        (only used for sub sampling) the full document volume dictionary
    pickle_filename : str
        the volume cache filename to be loaded via pickle
    """

    __volume_cache: Cache[str, Dict[str, int]]
    __total_volume: int

    # noinspection PyMissingConstructor
    def __init__(self, dataset: Dataset, doc_ids: Optional[Set[str]] = None, keywords: Optional[Set[str]] = None,
                 original_cache: Optional[Cache[str, Dict[str, int]]] = None, pickle_filename: str = None):
        if pickle_filename is None:
            _doc_ids: Set[str] = set() if doc_ids is None else doc_ids
            _keywords: Set[str] = set() if keywords is None else keywords

            if original_cache is not None:
                log.debug(f"Subsampling Volume Cache for '{dataset.name()}'")
                new_volume_cache: Dict[str, Dict[str, int]] = dict()
                for keyword, volumes in filter(lambda item: item[0] in _keywords, original_cache.items()):
                    new_volume_cache[keyword] = dict(filter(lambda item: item[0] in _doc_ids, volumes.items()))

                self.__volume_cache = Cache(new_volume_cache,
                                            lambda key: dict(map(lambda doc: (doc.id(), doc.length()), dataset(key))))
                log.debug(f"Subsampling for '{dataset.name()}' complete")
            else:
                log.info(f"Creating Volume Cache for '{dataset.name()}'. This might take a while.")
                if not dataset.is_open():
                    log.debug("Opening dataset for caching")
                    with dataset:
                        self.__volume_cache = Cache.build(
                            lambda kw: dict(map(lambda doc: (doc.id(), doc.length()), dataset(kw))),
                            dataset.keywords())
                else:
                    self.__volume_cache = Cache.build(
                        lambda kw: dict(map(lambda doc: (doc.id(), doc.length()), dataset(kw))),
                        dataset.keywords())

                log.info(f"Volume Cache for '{dataset.name()}' complete")
        else:
            self.__volume_cache = Cache.load_pickle(
                lambda kw: dict(map(lambda doc: (doc.id(), doc.length()), dataset(kw))), pickle_filename)
            log.info(f"Loading Volume Cache for '{dataset.name()}' complete")

        used_doc_ids = set()
        self.__total_volume = 0
        for d in self.__volume_cache.values():
            for item in d.items():
                if item[0] in dataset.doc_ids() and item[0] not in used_doc_ids:
                    used_doc_ids.add(item[0])
                    self.__total_volume += item[1]

    def sample(self, dataset: Dataset) -> 'VolumeExtension':
        return VolumeExtension(dataset, dataset.doc_ids(), dataset.keywords(), self.__volume_cache)

    def volumes(self, keyword: str) -> List[int]:
        """
        Returns the volumes of all documents matching the given keyword.

        Parameters
        ----------
        keyword : str
            the keyword to look up

        Returns
        -------
        volumes : List[int]
            the volumes of all documents matching the keyword
        """
        return list(self.__volume_cache[keyword].values())

    def total_volume(self, keyword: str) -> int:
        """
        Returns the total volume the documents matching the given keyword.

        Parameters
        ----------
        keyword : str
            the keyword to look up

        Returns
        -------
        total_volume : int
            the total volume (i.e. the sum of the individual volumes) of the documents matching the keyword
        """
        return sum(self.__volume_cache[keyword].values())

    def dataset_volume(self) -> int:
        """
        Returns the total volume of the data set, i.e. the sum of the volumes of all documents.

        Returns
        -------
        the total volume of the data set
        """
        return self.__total_volume

    def doc_ids(self, keyword: str) -> Set[str]:
        return set(self.__volume_cache[keyword].keys())

    def selectivity(self, keyword: str) -> int:
        return len(self.__volume_cache[keyword])

    def max_selectivity(self) -> int:
        return max(map(len, self.__volume_cache.values()))

    def min_selectivity(self) -> int:
        return min(map(len, self.__volume_cache.values()))

    def selectivity_distribution(self) -> Dict[int, int]:
        return Counter(map(len, self.__volume_cache.values()))

    def pickle(self, dataset: 'Dataset', description: Optional[str] = None) -> None:
        filename = self.pickle_filename(self.key(), dataset.name(), description)
        self.__volume_cache.pickle(filename)
        log.info(f"Stored extension pickle in {filename}")

    @classmethod
    def extend_with_pickle(cls, dataset: 'Dataset', description: Optional[str] = None) -> 'VolumeExtension':
        filename = cls.pickle_filename(cls.key(), dataset.name(), description)
        log.info(f"Loading Selectivity Cache for '{dataset.name()}' with pickle in {filename}")
        return cls(dataset, pickle_filename=filename)

    def get_identity_cache(self) -> Cache[str, Set[str]]:
        return Cache.build(lambda kw: self.doc_ids(kw), self.__volume_cache.keys())
