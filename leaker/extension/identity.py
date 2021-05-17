"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from abc import ABC
from copy import deepcopy
from logging import getLogger
from typing import Set, Optional, Dict, Any

from ..api import Extension, Dataset
from ..cache import Cache

log = getLogger(__name__)


class IdentityExtension(Extension, ABC):
    """
    An extension caching the document identifiers of all documents matching every keyword.

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
    original_identity_extension: IdentityExtension
        If an IdentityExtension has already been created, use this as base instead of building it again.
        This will ignore all other parameters.
    pickle_filename : str
        the document identity cache filename to be loaded via a pickle
    """

    _identity_cache: Cache[str, Set[str]]

    # noinspection PyMissingConstructor
    def __init__(self, dataset: Dataset, doc_ids: Optional[Set[str]] = None, keywords: Optional[Set[str]] = None,
                 original_cache: Optional[Cache[str, Set[str]]] = None, original_identity_extension: Any = None,
                 pickle_filename: str = None):
        if pickle_filename is None:
            if original_identity_extension is None:
                _doc_ids: Set[str] = set() if doc_ids is None else doc_ids
                _keywords: Set[str] = set() if keywords is None else keywords

                if original_cache is not None:
                    log.debug(f"Subsampling Identity Cache for '{dataset.name()}'")
                    new_identity_cache: Dict[str, Set[str]] = dict()
                    for keyword, original_doc_ids in filter(lambda item: item[0] in _keywords, original_cache.items()):
                        new_identity_cache[keyword] = original_doc_ids.intersection(_doc_ids)

                    self._identity_cache = Cache(new_identity_cache,
                                                 lambda kw: set(map(lambda doc: doc.id(), dataset(kw))))
                    log.debug(f"Subsampling for '{dataset.name()}' complete")
                else:
                    log.info(f"Creating Identity Cache for '{dataset.name()}'. This might take a while.")
                    if not dataset.is_open():
                        log.debug("Opening dataset for caching")
                        with dataset:
                            self._identity_cache = Cache.build(lambda kw: set(map(lambda doc: doc.id(), dataset(kw))),
                                                               dataset.keywords())
                    else:
                        self._identity_cache = Cache.build(lambda kw: set(map(lambda doc: doc.id(), dataset(kw))),
                                                           dataset.keywords())
                    log.info(f"Identity Cache for '{dataset.name()}' complete")
            else:
                self._identity_cache = deepcopy(original_identity_extension.get_identity_cache())
        else:
            self._identity_cache = Cache.load_pickle(lambda kw: set(map(lambda doc: doc.id(), dataset(kw))),
                                                     pickle_filename)
            log.info(f"Loading Cache for '{dataset.name()}' complete")

    def sample(self, dataset: Dataset) -> 'IdentityExtension':
        return IdentityExtension(dataset, dataset.doc_ids(), dataset.keywords(), self._identity_cache)

    def get_identity_cache(self) -> Cache[str, Set[str]]:
        return self._identity_cache

    def set_identity_cache(self, identity_cache: Cache[str, Set[str]]) -> None:
        self._identity_cache = identity_cache

    def doc_ids(self, keyword: str) -> Set[str]:
        """
        Returns the identifiers of all documents matching the given keyword.

        Parameters
        ----------
        keyword : str
            the keyword to look up

        Returns
        -------
        doc_ids : Set[str]
            the identifiers of all documents matching the keyword
        """
        return self._identity_cache[keyword]

    def pickle(self, dataset: 'Dataset', description: Optional[str] = None) -> None:
        filename = self.pickle_filename(self.key(), dataset.name(), description)
        self._identity_cache.pickle(filename)
        log.info("Stored extension pickle in " + filename)

    @classmethod
    def extend_with_pickle(cls, dataset: 'Dataset', description: Optional[str] = None) -> 'IdentityExtension':
        filename = cls.pickle_filename(cls.key(), dataset.name(), description)
        log.info(f"Loading Identity Cache for '{dataset.name()}' with pickle {filename}")
        return cls(dataset, pickle_filename=filename)
