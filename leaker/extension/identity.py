"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from abc import ABC
from copy import deepcopy
from logging import getLogger
from typing import Set, Optional, Dict, Any, Union, Tuple

from ..api import Extension, Dataset, RelationalKeyword, RelationalDatabase
from ..cache import Cache

log = getLogger(__name__)

Keyword = Union[str, RelationalKeyword]
Identifier = Union[str, Tuple[int, int]]


class IdentityExtension(Extension, ABC):
    """
    An extension caching the document identifiers of all documents matching every keyword.

    Parameters
    ----------
    dataset : Union[Dataset, RelationalDatabase]
        the data set to build the cache on
    doc_ids : Optional[Set[Identifier]]
        (only used for sub sampling) the document identifiers contained in the sampled data set
    keywords : Optional[Set[Keyword]]
        (only used for sub sampling) the keywords contained in the sampled data set
    original_cache : Optional[Cache[Keyword, Set[Identifier]]]
        (only used for sub sampling) the full cache that needs to be subsampled
    original_identity_extension: IdentityExtension
        If an IdentityExtension has already been created, use this as base instead of building it again.
        This will ignore all other parameters.
    pickle_filename : str
        the document identity cache filename to be loaded via a pickle
    """

    _identity_cache: Cache[Keyword, Set[Identifier]]
    extension_name = 'Identity Cache'

    # noinspection PyMissingConstructor
    def __init__(self, dataset: Union[Dataset, RelationalDatabase], doc_ids: Optional[Set[Identifier]] = None,
                 keywords: Optional[Set[Keyword]] = None,
                 original_cache: Optional[Cache[Keyword, Set[Identifier]]] = None,
                 original_identity_extension: Any = None, pickle_filename: str = None):

        if pickle_filename is None or isinstance(dataset, RelationalDatabase):
            if original_identity_extension is None:
                _doc_ids: Set[Identifier] = set() if doc_ids is None else doc_ids
                _keywords: Set[Keyword] = set() if keywords is None else keywords

                if original_cache is not None and not isinstance(dataset, RelationalDatabase):
                    log.debug(f"Subsampling {self.extension_name} for '{dataset.name()}'")
                    new_identity_cache: Dict[Keyword, Set[Identifier]] = dict()
                    for keyword, original_doc_ids in filter(lambda item: item[0] in _keywords, original_cache.items()):
                        new_identity_cache[keyword] = original_doc_ids.intersection(_doc_ids)

                    if isinstance(dataset, RelationalDatabase):
                        self._identity_cache = Cache(new_identity_cache, lambda kw: set(dataset.query(kw)))
                    else:
                        self._identity_cache = Cache(new_identity_cache,
                                                     lambda kw: set(map(lambda doc: doc.id(), dataset(kw))))
                    log.debug(f"Subsampling for '{dataset.name()}' complete")
                else:
                    log.info(f"Creating {self.extension_name} for '{dataset.name()}'. This might take a while.")
                    if not dataset.is_open():
                        log.debug("Opening dataset for caching")
                        with dataset:
                            if isinstance(dataset, RelationalDatabase):
                                self._identity_cache = Cache.build(lambda kw: set(dataset.query(kw)),
                                                                   dataset.keywords())
                            else:
                                self._identity_cache = Cache.build(lambda kw: set(map(lambda doc: doc.id(),
                                                                                      dataset(kw))),
                                                                   dataset.keywords())
                    else:
                        if isinstance(dataset, RelationalDatabase):
                            self._identity_cache = Cache.build(lambda kw: set(dataset.query(kw)), dataset.keywords())
                        else:
                            self._identity_cache = Cache.build(lambda kw: set(map(lambda doc: doc.id(), dataset(kw))),
                                                               dataset.keywords())
                    log.info(f"{self.extension_name} for '{dataset.name()}' complete")
            else:
                self._identity_cache = deepcopy(original_identity_extension.get_identity_cache())
        else:
            self._identity_cache = Cache.load_pickle(lambda kw: set(map(lambda doc: doc.id(), dataset(kw))),
                                                         pickle_filename)
            log.info(f"Loading Cache for '{dataset.name()}' complete")

    def sample(self, dataset: Union[Dataset, RelationalDatabase]) -> 'IdentityExtension':
        if isinstance(dataset, RelationalDatabase):
            #return IdentityExtension(dataset, dataset.doc_ids(), dataset.keywords(), self._identity_cache)
            return IdentityExtension(dataset)  # building a new extension is cheaper than sampling it.
        else:
            return IdentityExtension(dataset, dataset.doc_ids(), dataset.keywords(), self._identity_cache)

    def get_identity_cache(self) -> Cache[Keyword, Set[Identifier]]:
        return self._identity_cache

    def set_identity_cache(self, identity_cache: Cache[Keyword, Set[Identifier]]) -> None:
        self._identity_cache = identity_cache

    def doc_ids(self, keyword: Keyword) -> Set[Identifier]:
        """
        Returns the identifiers of all documents matching the given keyword.

        Parameters
        ----------
        keyword : Keyword
            the keyword to look up

        Returns
        -------
        doc_ids : Set[Identifier]
            the identifiers of all entries matching the keyword
        """
        return self._identity_cache[keyword]

    def pickle(self, dataset: Union[Dataset, RelationalDatabase], description: Optional[str] = None) -> None:
        if not isinstance(dataset, RelationalDatabase):
            filename = self.pickle_filename(self.key(), dataset.name(), description)
            self._identity_cache.pickle(filename)
            log.info("Stored extension pickle in " + filename)

    @classmethod
    def extend_with_pickle(cls, dataset: Union[Dataset, RelationalDatabase], description: Optional[str] = None) \
            -> 'IdentityExtension':
        if isinstance(dataset, RelationalDatabase):
            filename = None
        else:
            filename = cls.pickle_filename(cls.key(), dataset.name(), description)
            log.info(f"Loading Identity Cache for '{dataset.name()}' with pickle {filename}")
        return cls(dataset, pickle_filename=filename)
