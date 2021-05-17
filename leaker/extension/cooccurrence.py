"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import dill as pickle
from logging import getLogger
from typing import Set, Optional, Dict, Tuple

from .selectivity import SelectivityExtension
from .identity import IdentityExtension
from ..api import Dataset
from ..cache import Cache

log = getLogger(__name__)


class CoOccurrenceExtension(SelectivityExtension):
    """
    An extension caching the length of the intersection of all document ids for each keyword pair. This class is a
    subclass of IdentityExtension and also provides its functionality. It builds a cache indexed by (keyword0, keyword1)
    pairs and avoids redundant storage by using the symmetric property of the co-occurrence matrix.

    Parameters
    ----------
    dataset : Dataset
        the data set to build the cache on
    doc_ids : Optional[Set[str]]
        (only used for sub sampling) the document identifiers contained in the sampled data set
    keywords : Optional[Set[str]]
        (only used for sub sampling) the keywords contained in the sampled data set
    original_id_cache : Optional[Cache[str, Set[str]]]
        (only used for sub sampling) the full document id cache that needs to be subsampled
    original_coocc_cache : Cache[Tuple[str,str], Set[int]]
        (only used for sub sampling) the full document id cooccurrence cache that needs to be subsampled
    pickle_id_filename : str
        the document id cache filename to be loaded via pickle
    pickle_co_filename : str
        the document cooccurrence cache filename to be loaded via pickle
    original_doc_id_dict : Dict[str, int]
        Dict of original doc id strings to their enumeration
    """

    __coocc_cache: Cache[Tuple[str, str], Set[int]]
    __sampled_coocc_cache: Cache[Tuple[str, str], int]
    __is_sampled: bool
    __doc_id_dict: Dict[str, int]  # used to store int doc ids instead of str in coocc_cache

    def __init__(self, dataset: Dataset, doc_ids: Optional[Set[str]] = None, keywords: Optional[Set[str]] = None,
                 original_id_cache: Optional[Cache[str, Set[str]]] = None,
                 original_coocc_cache: Cache[Tuple[str, str], Set[int]] = None,
                 original_identity_extension: IdentityExtension = None,
                 pickle_id_filename: str = None, pickle_co_filename: str = None,
                 original_doc_id_dict: Dict[str, int] = None):
        if pickle_co_filename is None or pickle_id_filename is None:
            if dataset.has_extension(IdentityExtension):
                original_identity_extension = dataset.get_extension(IdentityExtension)

            super(CoOccurrenceExtension, self).__init__(dataset, doc_ids, keywords, original_id_cache,
                                                        original_identity_extension)
            _doc_ids: Set[str] = set() if doc_ids is None else doc_ids
            _keywords: Set[str] = set() if keywords is None else keywords

            if not original_doc_id_dict:
                self.__doc_id_dict = {d: i for i, d in enumerate(list(dataset.doc_ids()))}
            else:
                self.__doc_id_dict = original_doc_id_dict

            if original_coocc_cache is not None:
                log.debug(f"Subsampling CoOccurrence Cache for '{dataset.name()}'")
                self.__coocc_cache = original_coocc_cache
                self.__is_sampled = True
                _doc_ids_int: Set[int] = set([self.__doc_id_dict[d] for d in _doc_ids])
                new_coocc_cache: Dict[Tuple[str, str], int] = dict(set())
                for keys, coocc in filter(lambda item: item[0][0] in _keywords and item[0][1] in _keywords,
                                          original_coocc_cache.items()):
                    new_coocc_cache[keys] = len(coocc.intersection(_doc_ids_int))

                self.__sampled_coocc_cache = \
                    Cache(new_coocc_cache,
                          lambda key: len(set(map(lambda d: self.__doc_id_dict[d],
                                                  self.doc_ids(key[0]).intersection(self.doc_ids(key[1]))))),
                          max_elements=original_coocc_cache.max_elements())
                log.debug(f"Subsampling for '{dataset.name()}' complete.")
            else:
                log.info(f"Creating CoOccurrence Cache for '{dataset.name()}'. This might take a while.")
                self.__is_sampled = False
                max_elements = int(len(dataset.keywords()) / 8)
                self.__coocc_cache = \
                    Cache.build(lambda key: set(map(lambda d: self.__doc_id_dict[d],
                                                    self.doc_ids(key[0]).intersection(self.doc_ids(key[1])))),
                                max_elements=max_elements)

                for key0 in dataset.keywords():
                    kw_iter = iter(dataset.keywords())
                    for _ in range(len(dataset.keywords())):
                        key1 = next(kw_iter)
                        if (key0, key1) not in self.__coocc_cache.keys() and \
                                (key1, key0) not in self.__coocc_cache.keys():
                            self.__coocc_cache.compute_if_absent((key0, key1))
                        if len(self.__coocc_cache) >= max_elements:
                            break
                    else:
                        continue
                    break

                log.info(f"CoOccurrence Cache for '{dataset.name()}' complete.")
        else:
            self.__is_sampled = False
            self.set_identity_cache(Cache.load_pickle(lambda kw: set(map(lambda doc: doc.id(), dataset(kw))),
                                                      pickle_id_filename))
            self.__coocc_cache = \
                Cache.load_pickle(lambda key: set(map(lambda d: self.__doc_id_dict[d],
                                                      self.doc_ids(key[0]).intersection(self.doc_ids(key[1])))),
                                  pickle_co_filename)
            log.info(f"CoOccurrence Cache for '{dataset.name()}' loaded")

    def sample(self, dataset: Dataset) -> 'CoOccurrenceExtension':
        return CoOccurrenceExtension(dataset, dataset.doc_ids(), dataset.keywords(), self._identity_cache,
                                     self.__coocc_cache, original_doc_id_dict=self.__doc_id_dict)

    def co_occurrence(self, key0: str, key1: str) -> int:
        """
        Returns the cooccurrence counts of the given keyword.

        Parameters
        ----------
        key0 : str
            the first keyword to look up
        key1 : str
            the second keyword to look up

        Returns
        -------
        co_occurrence : int
            the cooccurrence count of key0 with key1
        """
        if self.__is_sampled:
            if (key0, key1) in self.__sampled_coocc_cache:
                return self.__sampled_coocc_cache[(key0, key1)]
            else:
                return self.__sampled_coocc_cache[(key1, key0)]
        else:
            if (key0, key1) in self.__coocc_cache:
                return len(self.__coocc_cache[(key0, key1)])
            else:
                return len(self.__coocc_cache[(key1, key0)])

    def pickle(self, dataset: 'Dataset', description: Optional[str] = None) -> None:
        id_filename = self.pickle_filename(IdentityExtension.key(), dataset.name(), description)
        self.get_identity_cache().pickle(id_filename)
        co_filename = self.pickle_filename(self.key(), dataset.name(), description)
        self.__coocc_cache.pickle(co_filename)
        dict_filename = self.pickle_filename("__doc_id_dict", dataset.name(), description)
        pickle.dump(self.__doc_id_dict, open(dict_filename, "wb"))
        log.info(f"Stored extension pickle in {id_filename}, {dict_filename}, and {co_filename}")

    @classmethod
    def extend_with_pickle(cls, dataset: 'Dataset', description: Optional[str] = None) -> 'CoOccurrenceExtension':
        id_filename = cls.pickle_filename(IdentityExtension.key(), dataset.name(), description)
        co_filename = cls.pickle_filename(cls.key(), dataset.name(), description)
        dict_filename = cls.pickle_filename("__doc_id_dict", dataset.name(), description)
        log.info(f"Loading CoOccurrence Cache for '{dataset.name()}' with pickle in {id_filename}, {dict_filename} "
                 f"and {co_filename}")
        return cls(dataset, pickle_id_filename=id_filename, pickle_co_filename=co_filename,
                   original_doc_id_dict=pickle.load(open(dict_filename, "rb")))
