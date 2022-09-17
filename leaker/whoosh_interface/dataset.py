"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from itertools import chain, repeat

import dill as pickle

from collections import Counter
from functools import reduce
from logging import getLogger
from math import ceil
from os import path
from random import sample, shuffle
from typing import Set, Iterator, Optional, List, TypeVar, Type, Any, Dict

from whoosh.index import FileIndex
from whoosh.lang import porter
from whoosh.qparser import QueryParser
from whoosh.query import Query, Term, Every, Or
from whoosh.searching import Searcher, Hit, Results

from ..api.constants import MIN_USER_QUERYLOG_ACTIVITY, Selectivity
from .__setup import DataSetIndexSchema, QueryLogIndexSchema, keyword_analyzer
from ..api import Dataset, Document, Extension, KeywordQueryLog, Data
from ..cache import Cache
from ..extension import IdentityExtension, SelectivityExtension

log = getLogger(__name__)
T = TypeVar("T", bound=Extension, covariant=True)


class WhooshDataset(Dataset):
    """
    A `Dataset` implementation relying on a Whoosh index. This class should not be created directly but only loaded
    using the `WhooshBackend`.

    It can be used in a context manager to keep an `IndexSearcher` open for the whole time which saves time when doing
    multiple queries.

    It keeps track of already performed queries in a query cache that is stored in a pickle file for future use.

    When extending a data set of this type with an `Extension`, all sampled or restricted data sets that stem from this
    data set will automatically be extended with their respective sampled or restricted version of the extension, too.

    Parameters
    ----------
    name: str
        the name of the data set (index)
    index: FileIndex
        the opened Whoosh index
    is_sampled_or_restricted: bool
        whether this data set is a sample or restriction of the full data set
    pickle_description: str
        If a specific pickle file should be used for the query cache (identified by the description).
    """

    __name: str

    _index: FileIndex
    __searcher: Optional[Searcher]

    _doc_ids: Set[str]
    __keywords: Set[str]

    _keyword_cache: Cache[str, Set[str]]
    _query_cache: Cache[str, List[Document]]
    __initial_query_cache_size: int

    _is_sampled_or_restricted: bool
    __pickle_filename: str

    def __init__(self, name: str, index: FileIndex, is_sampled_or_restricted: bool = False,
                 pickle_description: str = None):
        super(WhooshDataset, self).__init__()

        self._is_sampled_or_restricted = is_sampled_or_restricted
        self.__name = name
        self._index = index

        self.__searcher = None
        self.__query_parser = QueryParser("content", DataSetIndexSchema())

        if not self._is_sampled_or_restricted:
            log.info(f"Loading Whoosh Index '{name}'")
            self.__pickle_filename = Dataset.pickle_filename(name, pickle_description)
            # Cache already performed queries (using string representation, i.e., q is of the form "field:word"
            if path.exists(self.__pickle_filename):
                log.info(f"Found existing query cache for {name} in {self.__pickle_filename}. Loading that...")
                self._query_cache = Cache.load_pickle(
                    lambda q: [d for d in self.__query_docs_internal(Term(q.split(':')[0], q.split(':')[1]))],
                    self.__pickle_filename)
            else:
                self._query_cache = Cache.build(
                    lambda q: [d for d in self.__query_docs_internal(Term(q.split(':')[0], q.split(':')[1]))])

            self.__initial_query_cache_size = len(self._query_cache)

            with self:
                # Cache document IDs
                log.debug('Building Document ID Cache')
                # noinspection Mypy
                self._doc_ids = set(map(lambda bytestr: bytestr.decode("utf-8"), self.__searcher.lexicon("doc_id")))

                # Cache keywords
                log.debug('Building Keyword Cache')
                keyword_cache: Cache[str, List[str]] = Cache.build(
                    lambda doc_id: list(map(porter.stem, self.__doc_by_id(doc_id)['keywords'])), self._doc_ids)

                self.__keywords = reduce(lambda a, b: a.union(b), keyword_cache.values(), set())
                self._keyword_cache = Cache({str(doc_id): set(keyword_cache[doc_id]) for doc_id in self._doc_ids},
                                            lambda doc_id: set([k for k in keyword_cache[doc_id]]))

            log.info(f"Loading Whoosh Index '{name}' complete")

        else:
            self.__keywords = reduce(lambda a, b: a.union(b), self._keyword_cache.values(), set())

    def name(self) -> str:
        return self.__name

    def query(self, keyword: str, stemmed: bool = True) -> Iterator[Document]:
        yield from self._query_cache[str(self.__query(keyword, stemmed))]

    def documents(self) -> Iterator[Document]:
        yield from self.__query_docs_internal(Every())

    def keywords(self) -> Set[str]:
        return self.__keywords

    def doc_ids(self) -> Set[str]:
        return self._doc_ids

    def selectivity(self, keyword: str) -> int:
        if self.has_extension(SelectivityExtension):
            return self.get_extension(SelectivityExtension).selectivity(keyword)

        return len([doc for doc in self.query(keyword)])

    def sample(self, rate: float) -> Dataset:
        if rate > 1 or rate < 0:
            raise ValueError("Sample rate must be in [0, 1]")

        if rate == 1:
            return self

        sample_size = ceil(len(self._doc_ids) * rate)

        return SampledWhooshDataset(self, rate, doc_ids=set(sample(population=self._doc_ids, k=sample_size)))
    
    def sample_test_training(self, rate: float) -> Dataset:
        if rate > 0.9 or rate < 0.1:
            raise ValueError("Sample rate must be in [0.1, 0.9]")

        sample_size = ceil(len(self._doc_ids) * rate)

        training_ids = set(sample(population=self._doc_ids, k=sample_size))
        test_ids = set(self._doc_ids).difference(training_ids)

        return (SampledWhooshDataset(self, rate, doc_ids=training_ids), SampledWhooshDataset(self, 1-rate, doc_ids=test_ids))

    def sample_rate(self) -> float:
        return 1.

    def pickle(self) -> None:
        if len(self._query_cache) > self.__initial_query_cache_size:
            self._query_cache.pickle(self.__pickle_filename)
            self.__initial_query_cache_size = len(self._query_cache)
            log.info(f"Stored query cache for {self.name()} in {self.__pickle_filename}")

    def is_open(self) -> bool:
        return self.__searcher is not None

    def open(self) -> 'WhooshDataset':
        self.__searcher = self._index.searcher(closereader=False)
        return self

    def close(self) -> None:
        if not self._is_sampled_or_restricted:
            self.pickle()
        if self.__searcher is not None:
            self.__searcher.close()
            self.__searcher = None

    def extend_with(self, extension: Type[T], **kwargs) -> 'WhooshDataset':
        if not self.has_extension(extension):
            super(WhooshDataset, self).extend_with(extension, **kwargs)

        return self

    def restrict_keyword_size(self, max_keywords: int = 0, selectivity: Selectivity = Selectivity.Independent) \
            -> 'WhooshDataset':
        if max_keywords > len(self.keywords()):
            raise ValueError(f"Max keyword restriction for {self.name()} cannot be larger than its keyword size "
                             f"{len(self.keywords())}")
        return RestrictedWhooshDataset(self, max_keywords=max_keywords, selectivity=selectivity)

    def restrict_rate(self, rate: float) -> 'WhooshDataset':
        if rate > 1 or rate < 0:
            raise ValueError("Restriction rate must be in [0, 1]")
        return RestrictedWhooshDataset(self, restriction_rate=rate)

    def restriction_rate(self) -> float:
        return 1

    def __del__(self):
        self.__exit__(None, None, None)

    def __len__(self) -> int:
        return len(self._doc_ids)

    def __query(self, keyword: str, stemmed: bool) -> Query:
        return self.__query_parser.parse(keyword) if not stemmed else Term('content', keyword)

    def __query_docs_internal(self, query: Query) -> Iterator[Document]:
        for hit in self.__query_internal(query):
            if WhooshDataset.__doc(hit).id() in self._doc_ids:
                yield WhooshDataset.__doc(hit)

    def __query_internal(self, query: Query) -> Iterator[Hit]:
        yield from self._query_results_internal(query)

    def __doc_by_id(self, doc_id: str) -> Dict[str, Any]:
        if self.__searcher is None:
            with self._index.searcher(closereader=False) as s:
                return s.document(doc_id=doc_id)
        else:
            return self.__searcher.document(doc_id=doc_id)

    def _query_results_internal(self, query: Query) -> Results:
        if self.__searcher is None:
            with self._index.searcher(closereader=False) as s:
                return self._results(s, query)
        else:
            return self._results(self.__searcher, query)

    def _results(self, searcher: Searcher, query: Query) -> Results:
        return searcher.search(query, limit=None, scored=False, sortedby=None)

    @staticmethod
    def __doc(hit: Hit) -> Document:
        return Document(hit['doc_id'], hit['length'])


class RestrictedWhooshDataset(WhooshDataset):
    """
    A restricted sample of a `WhooshDataset`. It restricts all results to a random subset at a given rate, or a
    specified maximum amount of keywords.

    When extending a data set of this type with an `Extension`, only this data set will be extended.
    This distinguishes a RestrictedWhooshDataset from a sampled one: the former is used as an actual restricted basis
    for sampling to reduce big data sets, while the latter is used to simulate restricted knowledge of a given data set
    (but still computes on the whole data set).

    Parameters
    ----------
    parent: WhooshDataset
        the full data set
    max_keywords: int
        If not 0, restrict the keyword space to max_keywords keywords using random shuffling of the set of keywords (if
        random_shuffling) or most common max_keywords
    selectivity: Selectivity
        If max_keywords is not 0, this determines the selectivity by which the keywords are chosen
    restriction_rate: float
        The rate at which this dataset shall be restricted (only use a random restriction_rate fraction as basis for the
        dataset, extensions, and subsequent sampling).
    restriction_rate : float
        the sample rate in (0,1)
    """
    __parent: WhooshDataset

    __restriction_rate: float

    def __init__(self, parent: WhooshDataset, max_keywords: int = 0, selectivity: Selectivity = Selectivity.Independent,
                 restriction_rate: float = 1):
        if max_keywords != 0 and restriction_rate != 1:
            raise ValueError("Cannot restrict a WhooshDataset to both max keywords and a rate!")

        self._samples = []
        self._restricted = []

        self.__restriction_rate = restriction_rate

        if max_keywords <= 0:
            log.info(f'Restricting Dataset to {restriction_rate * 100}%')
            name = f'{parent.name()}%{restriction_rate}'
        else:
            log.info(f'Restricting Dataset to {max_keywords} keywords')
            name = f'{parent.name()}|{max_keywords}'
        self.__parent = parent

        self._doc_ids = set(sample(parent._doc_ids, ceil(self.__restriction_rate * len(parent._doc_ids))))

        self._keyword_cache = Cache({doc_id: set(self.__parent._keyword_cache[doc_id]) for doc_id in self._doc_ids},
                                    lambda doc_id: set([k for k in self.__parent._keyword_cache[doc_id]]))

        if max_keywords != 0:
            keywords: List[str] = [k for ks in self._keyword_cache.values() for k in ks]
            if selectivity == Selectivity.High:
                keywords_restricted = set([k for k, _ in Counter(keywords).most_common(max_keywords)])
            elif selectivity == Selectivity.Low:
                keywords_restricted = set([k for k, _ in Counter(keywords).most_common()[:-max_keywords - 1:-1]])
            elif selectivity == Selectivity.PseudoLow:
                keywords_restricted = set(sorted(filter(lambda key: 10 <= self.__parent.selectivity(key), keywords),
                                                 key=self.__parent.selectivity)[:max_keywords])
            else:  # selectivity == Selectivity.Independent:
                keywords = list(set(keywords))
                shuffle(keywords)
                keywords_restricted = set(keywords[:max_keywords])

            # We also have to restrict the keyword cache and doc_ids set *again* to only include the relevant keywords
            self._keyword_cache = Cache({doc_id: set(self._keyword_cache[doc_id]).intersection(keywords_restricted)
                                         for doc_id in self._doc_ids if
                                         len(set(self._keyword_cache[doc_id]).intersection(keywords_restricted)) > 0},
                                        lambda doc_id: set([k for k in self.__parent._keyword_cache[doc_id] if
                                                            k in self.keywords()]))
            self._doc_ids = set(self._keyword_cache.keys())

        super(RestrictedWhooshDataset, self).__init__(name, parent._index, True)

        self._set_extensions(map(lambda ext: ext.sample(self), parent._get_extensions()))

        log.info(f"Restricting Whoosh Index '{parent.name()}' complete")

    def restriction_rate(self) -> float:
        return self.__restriction_rate

    def extend_with(self, extension: Type[T], **kwargs) -> 'RestrictedWhooshDataset':
        if not self.has_extension(extension):
            if not self.__parent.has_extension(extension):
                self.__parent.extend_with(extension, **kwargs)

            new_ext = self.__parent.get_extension(extension)

            extensions = self._get_extensions()
            extensions.append(new_ext.sample(self))
            self._set_extensions(extensions)
        return self

    def query(self, keyword: str, stemmed: bool = True) -> Iterator[Document]:
        for doc in self.__parent.query(keyword, stemmed):
            if doc.id() in self.doc_ids():
                yield doc

    def _results(self, searcher: Searcher, query: Query) -> Results:
        if not self._is_sampled_or_restricted:
            return self.__parent._results(searcher, query)
        else:
            return searcher.search(query, limit=None, scored=False, sortedby=None,
                                   filter=Or([Term('doc_id', doc_id) for doc_id in self._doc_ids]))


class SampledWhooshDataset(WhooshDataset):
    """
    A sub sample of a `WhooshDataset`. It filters all query results by a set of document identifiers contained in this
    sub set. Instances of this class should not be created directly, but only by sampling from a `WhooshDataset`.

    When extending a data set of this type with an `Extension`, actually the parent (full) data set will be extended
    and thus, all other sampled data sets will be extended with their respective sampled versions of the extension, too.

    Parameters
    ----------
    parent: WhooshDataset
        the full data set
    rate: float
        the sample rate in (0,1)
    doc_ids: Set[str]
        the identifiers of all documents contained in this sub sample
    """
    __parent: WhooshDataset

    __sampled_documents: Results
    __rate: float

    def __init__(self, parent: WhooshDataset, rate: float, doc_ids: Set[str]):
        log.info(f"Sampling Whoosh Index '{parent.name()}' at rate {rate:.3f}")

        self.__parent = parent

        self._doc_ids = doc_ids
        self._keyword_cache = Cache(dict(filter(lambda item: item[0] in self._doc_ids,
                                                self.__parent._keyword_cache.items())),
                                    lambda doc_id: set(next(self.__query_internal(Term('doc_id', doc_id)))[
                                                           'keywords']).intersection(self.__parent.keywords()))

        sample_query = Or([Term('doc_id', doc_id) for doc_id in doc_ids])
        with parent._index.searcher() as s:
            self.__sampled_documents = parent._results(s, sample_query)

        super(SampledWhooshDataset, self).__init__(parent.name(), parent._index, is_sampled_or_restricted=True)

        self.__rate = rate

        self._set_extensions(map(lambda ext: ext.sample(self), parent._get_extensions()))

        # We also have to restrict the doc_ids set
        if not self.has_extension(IdentityExtension):
            self.extend_with(IdentityExtension)
        identity = self.get_extension(IdentityExtension)
        self._doc_ids = set([doc_id for keyword in self.keywords() for doc_id in identity.doc_ids(keyword)])

        log.info(f"Sampling Whoosh Index '{parent.name()}' complete")

    def name(self) -> str:
        return f"{super(SampledWhooshDataset, self).name()}@{self.__rate}"

    def sample(self, rate: float) -> Dataset:
        if rate > 1 or rate < 0:
            raise ValueError("Sample rate must be in [0, 1]")

        if rate == 1:
            return self.__parent

        if rate < self.__rate:
            sample_size = ceil(len(self._doc_ids) / self.__rate * rate)

            return SampledWhooshDataset(self.__parent, rate,
                                        doc_ids=set(sample(population=self._doc_ids, k=sample_size)))
        elif rate > self.__rate:
            sample_size = ceil(len(self._doc_ids) / self.__rate * rate)
            population = self.__parent._doc_ids.difference(self._doc_ids)

            doc_ids = self._doc_ids.union(sample(population=population, k=sample_size - len(self._doc_ids)))

            return SampledWhooshDataset(self.__parent, rate, doc_ids=doc_ids)
        return self

    def sample_rate(self) -> float:
        return self.__rate

    def extend_with(self, extension: Type[T], **kwargs) -> 'SampledWhooshDataset':
        if not self.has_extension(extension):
            if not self.__parent.has_extension(extension):
                self.__parent.extend_with(extension, **kwargs)

            new_ext = self.__parent.get_extension(extension)

            extensions = self._get_extensions()
            extensions.append(new_ext.sample(self))
            self._set_extensions(extensions)
        return self

    def query(self, keyword: str, stemmed: bool = True) -> Iterator[Document]:
        for doc in self.__parent.query(keyword, stemmed):
            if doc in self.__sampled_documents:
                yield doc


class WhooshKeywordQueryLog(KeywordQueryLog):
    """
    A `KeywordQueryLog` implementation relying on a Whoosh index. This class should not be created directly but only loaded
    using the `WhooshBackend`.

    It can be used in a context manager to keep an `IndexSearcher` open for the whole time which saves time when doing
    multiple queries.

    It keeps track of already performed queries in a query cache that is stored in a pickle file for future use.

    Parameters
    ----------
    name: str
        the name of the query log (index)
    index: FileIndex
        the opened Whoosh index
    pickle_description: str
        If a specific pickle file should be used for the query cache (identified by the description).
    min_user_count: int, max_user_count: int
        If given, only consider queries of most_freq_users[min_user_count:max_user_count]
    reverse: bool
        If True, consider queries of least_freq_users[min_user_count:max_user_count] with
        min activity MIN_USER_QUERYLOG_ACTIVITY
    """

    __name: str

    __index: FileIndex
    __searcher: Optional[Searcher]

    __doc_ids: Set[str]
    __keywords_list: List[str]
    __user_ids: List[str]

    __query_cache: Cache[str, List[Document]]
    __initial_query_cache_size: int
    __keyword_cache: Cache[str, List[str]]

    __pickle_filename: str

    def __init__(self, name: str, index: FileIndex, pickle_description: str = None, min_user_count: int = 0,
                 max_user_count: int = None, reverse: bool = False):
        super(WhooshKeywordQueryLog, self).__init__()

        self.__name = name
        self.__index = index

        self.__searcher = None
        self.__query_parser = QueryParser("user_id", QueryLogIndexSchema())

        log.info(f"Loading Whoosh Query Log '{name}'")

        self.__pickle_filename = Data.pickle_filename(name, pickle_description)
        # Cache already performed queries (using string representation, i.e., q is of the form "field:word"
        if path.exists(self.__pickle_filename):
            log.info(f"Found existing query cache for {name} in {self.__pickle_filename}. Loading that...")
            self.__query_cache = Cache.load_pickle(
                lambda q: [d for d in self.__query_docs_internal(Term(q.split(':')[0], q.split(':')[1]))],
                self.__pickle_filename)
        else:
            self.__query_cache = Cache.build(
                lambda q: [d for d in self.__query_docs_internal(Term(q.split(':')[0], q.split(':')[1]))])

        self.__initial_query_cache_size = len(self.__query_cache)

        with self:
            self.__user_ids = self.__get_freq_users(min_user_count, max_user_count, reverse)

            # Cache document IDs
            log.debug('Building Document ID and Query Cache. This might take a while...')

            self.__doc_ids = set([doc.id() for docs in list(map(lambda user_id: self.query_docs(user_id),
                                                                self.__user_ids))
                                  for doc in docs])

            # Cache keywords
            log.debug('Building Keyword Cache')
            _keyword_analyzer = keyword_analyzer()
            self.__keyword_cache: Cache[str, List[str]] = Cache.build(
                lambda user_id: [keyword for doc_id in [doc.id() for doc in self.query_docs(user_id)]
                                 for keyword in list(map(lambda token: token.text,
                                                         _keyword_analyzer(self.__doc_by_id(doc_id)['content'],
                                                                           mode='query')))],
                set(self.__user_ids))

            self.__keywords_list = [kw for keywords in self.__keyword_cache.values() for kw in keywords]

            log.info(f"Loading Whoosh Index '{name}' complete")

    def name(self) -> str:
        return self.__name

    def query(self, user_id: str) -> Iterator[str]:
        yield from self.__keyword_cache[user_id]

    def query_docs(self, keyword: str) -> Iterator[Document]:
        yield from self.__query_cache[str(self.__query(keyword))]

    def documents(self) -> Iterator[Document]:
        yield from self.__query_docs_internal(Every())

    def keywords_list(self, user_id: str = None) -> List[str]:
        if user_id is None:
            return self.__keywords_list
        else:
            return list(self.query(user_id))

    def doc_ids(self) -> Set[str]:
        return self.__doc_ids

    def user_ids(self) -> List[str]:
        return self.__user_ids

    def __get_freq_users(self, min_user_count: int = 0, max_user_count: int = None, reverse: bool = False) -> List[str]:
        log.info("Fetching all users...")
        user_frequencies_filename = self.__pickle_filename[:-7] + "_freq_users.pickle"
        if path.exists(user_frequencies_filename):
            log.info(f"Found {user_frequencies_filename}, loading that...")
            users: Counter[str] = pickle.load(open(user_frequencies_filename, "rb"))
        else:
            log.info(f"Did not find user frequencies for {self.name()}, creating them from scratch. "
                     f"This might take a while ...")
            users = Counter()
            results = self.__results(self.__searcher, Every('user_id'))
            log.debug("Computing frequencies...")
            for res in results:
                users.update([res['user_id']])

            log.debug("Done.")
            pickle.dump(users, open(user_frequencies_filename, "wb"))
            log.info(f"Generated and stored user frequencies in {user_frequencies_filename}.")

        if reverse:
            user_list = list(dict.fromkeys(list(chain.from_iterable(repeat(i, c) for i, c in users.most_common()[::-1]
                                                                    if c > MIN_USER_QUERYLOG_ACTIVITY))).keys())
        else:
            user_list = list(dict.fromkeys(list(chain.from_iterable(repeat(i, c) for i, c in
                                                                    users.most_common()))).keys())
        if max_user_count is None:
            max_user_count = len(user_list)

        user_list = user_list[min_user_count:max_user_count]

        if reverse:
            user_list.reverse()
        return user_list

    def pickle(self) -> None:
        if len(self.__query_cache) > self.__initial_query_cache_size:
            self.__query_cache.pickle(self.__pickle_filename)
            self.__initial_query_cache_size = len(self.__query_cache)
            log.info(f"Stored query cache for {self.name()} in {self.__pickle_filename}")

    def is_open(self) -> bool:
        return self.__searcher is not None

    def open(self) -> 'WhooshKeywordQueryLog':
        self.__searcher = self.__index.searcher(closereader=False)
        return self

    def close(self) -> None:
        self.pickle()
        if self.__searcher is not None:
            self.__searcher.close()
            self.__searcher = None

    def sample(self, sample_rate):
        return self.__keyword_cache.sample(sample_rate)

    def __del__(self):
        self.__exit__(None, None, None)

    def __len__(self) -> int:
        return len(self.__doc_ids)

    def __query(self, user_id: str) -> Query:
        return self.__query_parser.parse(user_id)

    def __query_docs_internal(self, query: Query) -> Iterator[Document]:
        for hit in self.__query_internal(query):
            yield WhooshKeywordQueryLog.__doc(hit)

    def __query_internal(self, query: Query) -> Iterator[Hit]:
        yield from self.__query_results_internal(query)

    def __doc_by_id(self, doc_id: str) -> Dict[str, Any]:
        if self.__searcher is None:
            with self.__index.searcher(closereader=False) as s:
                return s.document(doc_id=doc_id)
        else:
            return self.__searcher.document(doc_id=doc_id)

    def __query_results_internal(self, query: Query) -> Results:
        if self.__searcher is None:
            with self.__index.searcher(closereader=False) as s:
                return self.__results(s, query)
        else:
            return self.__results(self.__searcher, query)

    @classmethod
    def __results(cls, searcher: Searcher, query: Query) -> Results:
        return searcher.search(query, limit=None, scored=False, sortedby=None)

    @staticmethod
    def __doc(hit: Hit) -> Document:
        return Document(hit['doc_id'], -1)
