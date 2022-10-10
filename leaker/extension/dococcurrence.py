"""
For License information see the LICENSE file.

Authors: Patrick Ehrler

"""
import dill as pickle
from logging import getLogger
from typing import Set, Optional, Dict, Tuple

from .selectivity import SelectivityExtension
from .identity import IdentityExtension
from ..api import Dataset
from ..cache import Cache

log = getLogger(__name__)


class DocOccurrenceExtension(SelectivityExtension):
    """
    An extension caching the length of the intersection of all keywords for each document pair. This class is a
    subclass of IdentityExtension and also provides its functionality.
    It builds a cache indexed by (document0, document1) pairs and avoids redundant storage by using the
    symmetric property of the co-occurrence matrix.

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
    original_dococc_cache : Cache[Tuple[str,str], Set[int]]
        (only used for sub sampling) the full document id dococcurrence cache that needs to be subsampled
    pickle_id_filename : str
        the document id cache filename to be loaded via pickle
    pickle_co_filename : str
        the document dococcurrence cache filename to be loaded via pickle
    original_keyword_dict : Dict[str, int]
        Dict of original keyword strings to their enumeration
    """

    __dococc_cache: Cache[Tuple[str, str], Set[int]]
    __sampled_dococc_cache: Cache[Tuple[str, str], int]
    __is_sampled: bool
    __doc_id_dict: Dict[str, int]
    __keyword_dict: Dict[str, int]  # used to store int keywords instead of str in dococc_cache

    def __init__(self, dataset: Dataset, doc_ids: Optional[Set[str]] = None, keywords: Optional[Set[str]] = None,
                 original_id_cache: Optional[Cache[str, Set[str]]] = None,
                 original_dococc_cache: Cache[Tuple[str, str], Set[int]] = None,
                 original_identity_extension: IdentityExtension = None,
                 pickle_id_filename: str = None, pickle_co_filename: str = None,
                 original_keyword_dict: Dict[str, int] = None):
        if pickle_co_filename is None or pickle_id_filename is None:
            if dataset.has_extension(IdentityExtension):
                original_identity_extension = dataset.get_extension(IdentityExtension)

            super(DocOccurrenceExtension, self).__init__(dataset, doc_ids, keywords, original_id_cache,
                                                         original_identity_extension)
            _doc_ids: Set[str] = set() if doc_ids is None else doc_ids
            _keywords: Set[str] = set() if keywords is None else keywords

            if not original_keyword_dict:
                self.__keyword_dict = {d: i for i, d in enumerate(list(dataset.keywords()))}
            else:
                if keywords:
                    _new_keyword_dict: Dict[str, int] = dict()
                    for key, kid in original_keyword_dict.items():
                        if key in keywords:
                            _new_keyword_dict[key] = kid
                    self.__keyword_dict = _new_keyword_dict
                else:
                    self.__keyword_dict = original_keyword_dict

            if original_dococc_cache is not None:
                log.info(f"Subsampling DocOccurrence Cache for '{dataset.name()}'")
                self.__dococc_cache = original_dococc_cache
                self.__is_sampled = True
                _keywords_int: Set[int] = set([self.__keyword_dict[d] for d in _keywords])
                new_dococc_cache: Dict[Tuple[str, str], int] = dict(set())
                for keys, dococc in filter(lambda item: item[0][0] in _doc_ids and item[0][1] in _doc_ids,
                                           original_dococc_cache.items()):
                    new_dococc_cache[keys] = len(dococc.intersection(_keywords_int))

                self.__sampled_dococc_cache = \
                    Cache(new_dococc_cache,
                          lambda doc: len(set(map(lambda key_id: self.__keyword_dict[key_id],
                                                  [k for k in self.__keyword_dict if
                                                   (doc[0] in self.doc_ids(k) and doc[1]
                                                    in self.doc_ids(k))]))),
                          max_elements=original_dococc_cache.max_elements())
                log.info(f"Subsampling for '{dataset.name()}' complete.")
            else:
                log.info(f"Creating DocOccurrence Cache for '{dataset.name()}'. This might take a while.")
                self.__is_sampled = False
                max_elements = int(len(dataset.doc_ids()) / 2)
                self.__dococc_cache = \
                    Cache.build(lambda doc: set(map(lambda key_id: self.__keyword_dict[key_id],
                                                    [k for k in self.__keyword_dict if
                                                     (doc[0] in self.doc_ids(k) and doc[1] in self.doc_ids(k))])),
                                max_elements=max_elements)

                for doc0 in dataset.doc_ids():
                    docs_iter = iter(dataset.doc_ids())
                    for _ in range(len(dataset.doc_ids())):
                        doc1 = next(docs_iter)
                        if (doc0, doc1) not in self.__dococc_cache.keys() and \
                                (doc1, doc0) not in self.__dococc_cache.keys():
                            self.__dococc_cache.compute_if_absent((doc0, doc1))
                        if len(self.__dococc_cache) >= max_elements:
                            break
                    else:
                        continue
                    break

                log.info(f"DocOccurrence Cache for '{dataset.name()}' complete.")
        else:
            self.__is_sampled = False
            self.set_identity_cache(Cache.load_pickle(lambda kw: set(map(lambda doc: doc.id(), dataset(kw))),
                                                      pickle_id_filename))
            self.__dococc_cache = \
                Cache.load_pickle(lambda doc: set(map(lambda key_id: self.__keyword_dict[key_id],
                                                      [k for k in self.__keyword_dict if
                                                       (doc[0] in self.doc_ids(k) and doc[1] in
                                                        self.doc_ids(k))])),
                                  pickle_co_filename)
            log.info(f"DocOccurrence Cache for '{dataset.name()}' loaded")

    def sample(self, dataset: Dataset) -> 'DocOccurrenceExtension':
        return DocOccurrenceExtension(dataset, dataset.doc_ids(), dataset.keywords(), self._identity_cache,
                                      self.__dococc_cache, original_keyword_dict=self.__keyword_dict)

    def doc_occurrence(self, doc0: str, doc1: str) -> int:
        """
        Returns the dococcurrence counts of the given documents.

        Parameters
        ----------
        doc0 : str
            the first document to look up
        doc1 : str
            the second document to look up

        Returns
        -------
        doc_occurrence : int
            the dococcurrence count of doc0 with doc1
        """
        if self.__is_sampled:
            if (doc0, doc1) in self.__sampled_dococc_cache:
                return self.__sampled_dococc_cache[(doc0, doc1)]
            else:
                return self.__sampled_dococc_cache[(doc1, doc0)]
        else:
            if (doc0, doc1) in self.__dococc_cache:
                return len(self.__dococc_cache[(doc0, doc1)])
            else:
                return len(self.__dococc_cache[(doc1, doc0)])

    def pickle(self, dataset: 'Dataset', description: Optional[str] = None) -> None:
        id_filename = self.pickle_filename(IdentityExtension.key(), dataset.name(), description)
        self.get_identity_cache().pickle(id_filename)
        co_filename = self.pickle_filename(self.key(), dataset.name(), description)
        self.__dococc_cache.pickle(co_filename)
        dict_filename = self.pickle_filename("__keyword_dict", dataset.name(), description)
        pickle.dump(self.__keyword_dict, open(dict_filename, "wb"))
        log.info(f"Stored extension pickle in {id_filename}, {dict_filename}, and {co_filename}")

    @classmethod
    def extend_with_pickle(cls, dataset: 'Dataset', description: Optional[str] = None) -> 'DocOccurrenceExtension':
        id_filename = cls.pickle_filename(IdentityExtension.key(), dataset.name(), description)
        co_filename = cls.pickle_filename(cls.key(), dataset.name(), description)
        dict_filename = cls.pickle_filename("__keyword_dict", dataset.name(), description)
        log.info(f"Loading DocOccurrence Cache for '{dataset.name()}' with pickle in {id_filename}, {dict_filename} "
                 f"and {co_filename}")
        return cls(dataset, pickle_id_filename=id_filename, pickle_co_filename=co_filename,
                   original_keyword_dict=pickle.load(open(dict_filename, "rb")))
