"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
import os
import random
import numpy as np
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Iterator, Set, TypeVar, Type, Dict, List, Iterable, Optional, Union, Tuple

from .constants import PICKLE_DIRECTORY, Selectivity
from .document import Document
from ..cache import Cache

log = getLogger(__name__)


class Extension(ABC):
    """
    An extension for a data set. To be used efficiently, it must be feasible to compute a sub-sample of the data stored
    by the extension for a sampled data set.
    """

    # noinspection PyUnusedLocal
    @abstractmethod
    def __init__(self, dataset: 'Dataset', **kwargs):
        raise NotImplementedError

    @classmethod
    def extend(cls, dataset: 'Dataset', **kwargs) -> 'Extension':
        """
        Extends a given dataset with this extension.

        Parameters
        ----------
        dataset : Dataset
            the dataset to extend
        **kwargs
            (optional) arguments for the extension

        Returns
        -------
        extend : Extension
            the extension instance
        """
        extension = cls(dataset, **kwargs)
        return extension

    @classmethod
    def key(cls) -> str:
        """
        The unique key of the extension type.

        Returns
        -------
        key : str
            unique name for the extension
        """
        return f"{cls.__module__}.{cls.__qualname__}"

    @abstractmethod
    def sample(self, dataset: 'Dataset') -> 'Extension':
        """
        Samples the data in this extension with respect to the given data set.

        Parameters
        ----------
        dataset: Dataset
            the sampled data set

        Returns
        -------
        sample : Extension
            the sampled extension
        """
        raise NotImplementedError

    @staticmethod
    def pickle_filename(key: str, dataset_name: str, description: Optional[str] = None) -> str:
        """
        Creates filename PICKLE_DIRECTORY + {dataset.name}_{key}[_description].pickle

        Parameters
        ----------
        key: str
            the extension key
        dataset_name: str
            the name of the data set
        description : Optional[str]
            further qualifier of the extension instance

        Returns
        -------
        filename : str
            the pickle filename
        """
        desc = ""
        if description is not None:
            desc = "_" + description
        if not os.path.exists(PICKLE_DIRECTORY):
            os.makedirs(PICKLE_DIRECTORY)
        return PICKLE_DIRECTORY + dataset_name + "_" + key + desc + ".pickle"

    @abstractmethod
    def pickle(self, dataset: 'Dataset', description: Optional[str] = None) -> None:
        """
        Stores the current extension instance *and its parent instances* in
        PICKLE_DIRECTORY + {dataset.name}_{ext.key}[_description].pickle for ext in self and all parents

        Parameters
        ----------
        dataset: Dataset
            the extended data set
        description : Optional[str]
            further qualifier of the extension instance
        """
        raise NotImplementedError

    @classmethod
    def extend_with_pickle(cls, dataset: 'Dataset', description: Optional[str] = None) -> 'Extension':
        """
        Loads the current extension instance and all its parents in
        PICKLE_DIRECTORY + {ext.key}[_description].pickle for ext in cls and all parents

        Parameters
        ----------
        dataset : Dataset
            the dataset to extend
        description : Optional[str]
            further qualifier of the extension instance
        """
        raise NotImplementedError


class Data(ABC):
    """
    A class encompassing the types of indexed data that can be queried. It offers pickling of already performed queries
    for improved performance
    """

    @abstractmethod
    def name(self) -> str:
        """Returns the name of this data instance"""
        raise NotImplementedError

    @abstractmethod
    def query(self, keyword: str) -> Union[Iterator[Document], Iterator[str]]:
        """
        Yields all matches for a keyword.

        Parameters
        ----------
        keyword : str
            the keyword to search for

        Returns
        -------
        query : Union[Iterator[Document], Iterator[str]]
            an iterator yielding all matches (documents or strings) for the query
        """
        raise NotImplementedError

    @abstractmethod
    def documents(self) -> Iterator[Document]:
        """Yields all documents in this data instance."""
        raise NotImplementedError

    @abstractmethod
    def keywords(self) -> Set[str]:
        """Returns all keywords in this data instance."""
        raise NotImplementedError

    @abstractmethod
    def doc_ids(self) -> Set[str]:
        """Returns the unique identifiers of all documents in this data instance."""
        raise NotImplementedError

    @abstractmethod
    def is_open(self) -> bool:
        """Returns true if this data instance was opened before"""
        raise NotImplementedError

    @abstractmethod
    def open(self) -> 'Dataset':
        """Opens the data instance, i. e. may allocate resources, if applicable"""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Closes this data instance, i. e. may free resources, if applicable"""
        raise NotImplementedError

    @abstractmethod
    def pickle(self) -> None:
        """
        Stores the data set instance in "self.name[_description].pickle"
        """
        raise NotImplementedError

    @staticmethod
    def pickle_filename(dataset_name: str, description: Optional[str] = None) -> str:
        """
        Creates filename PICKLE_DIRECTORY + {dataset.name}[_description].pickle

        Parameters
        ----------
        dataset_name: str
            the name of the data set
        description : Optional[str]
            further qualifier of the data set instance

        Returns
        -------
        filename : str
            the pickle filename
        """
        desc = ""
        if description is not None:
            desc = "_" + description
        if not os.path.exists(PICKLE_DIRECTORY):
            os.makedirs(PICKLE_DIRECTORY)
        return PICKLE_DIRECTORY + dataset_name + desc + ".pickle"

    def __call__(self, query: str) -> Iterator[Document]:
        yield from self.query(query)

    def __len__(self) -> int:
        return len(list(self.documents()))

    def __enter__(self) -> 'Dataset':
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


T = TypeVar("T", bound=Extension, covariant=True)


class Dataset(Data):
    """
    A specific dataset. It will usually be obtained from a Backend.

    A data set can be sampled to a given known data rate. It is callable and will return all matches for a query if
    called on a keyword. By implementing the open and close methods, a data set can be used with a context manager.
    """
    __extensions: Dict[str, Extension]

    def __init__(self):
        self.__extensions = dict()

    @abstractmethod
    def selectivity(self, keyword: str) -> int:
        """
        Determines the selectivity of the given keyword on this data set.

        Parameters
        ----------
        keyword : str
            the keyword to get the selectivity for
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, rate: float) -> 'Dataset':
        """
        Samples this data set to the given percentage. The rate must be in [0, 1]. Other values must be rejected by
        this method. Furthermore, the method should always act relative to the full data set, i. e. when calling it
        on a sampled data set, the sampled fraction returned should be calculated with respect to the size of the
        full data set. This method is used to sample base data sets to known data rates to simulate partial knowledge of
        the full data set.

        Parameters
        ----------
        rate : float
            the sample rate in [0, 1]
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_test_training(self, rate: float) -> Tuple['Dataset','Dataset']:
        """
        Samples this data set to the given percentage. The rate must be in [0.1, 0.9]. Other values must be rejected by
        this method. This method is used to generate disjunct test and training data sets from a base data sets to a given rate
        to simulate knowledge of a statisitcally close dataset.

        Parameters
        ----------
        rate : float
            the sample rate in [0, 1]
        """
        raise NotImplementedError

    @abstractmethod
    def sample_rate(self) -> float:
        """The rate at which this data set was sampled, relative to the full data set"""
        raise NotImplementedError

    @abstractmethod
    def restrict_keyword_size(self, max_keywords: int = 0,
                              selectivity: Selectivity = Selectivity.Independent) -> 'Dataset':
        """
        Restricts this data set to the given amount of keywords. Contrary to sampling, the restriction method returns a
        full data set that acts accordingly, i.e., that is not yet sampled. This method is used to restrict big data
        sets to subsets used as basis for evaluations.

        Parameters
        ----------
        max_keywords : int
            the keyword set size to restrict to
        selectivity: Selectivity
            determines the selectivity by which the keywords are chosen
         """
        raise NotImplementedError

    @abstractmethod
    def restrict_rate(self, rate: float) -> 'Dataset':
        """
        Restricts this data set to the given percentage. The rate must be in [0, 1]. Other values must be rejected by
        this method. Contrary to sampling, the restriction method returns a full data set that acts accordingly, i.e.,
        that is not yet sampled. This method is used to restrict big data sets to representative subsets used as basis
        for evaluations.

        Parameters
        ----------
        rate : float
            the restriction rate in [0, 1]
        """
        raise NotImplementedError

    @abstractmethod
    def restriction_rate(self) -> float:
        """The rate at which this data set was restricted"""
        raise NotImplementedError

    def extend_with(self, extension: Type[T], **kwargs) -> 'Dataset':
        """
        Extends this dataset with an extension of the given type. Additional arguments may be specified.

        Parameters
        ----------
        extension: Type[Extension]
            the extension type
        **kwargs
            (optional) additional parameters for the extension

        Returns
        -------
        extend_with : Dataset
            the extended dataset, for chaining
        """
        if not self.has_extension(extension):
            self.__extensions[extension.key()] = extension.extend(self, **kwargs)
        return self

    def extend_with_pickle(self, extension: Type[T], description: Optional[str]) -> 'Dataset':
        """
        Loads an extension instance with the specified pickle file(s), given it exists.
        Parameters
        ----------
        extension: Type[Extension]
            the extension type
        description : Optional[str]
            the further qualifier of the extension instance
        """
        if not self.has_extension(extension):
            self.__extensions[extension.key()] = extension.extend_with_pickle(self, description)
        return self

    def pickle_extensions(self, description: Optional[str]) -> None:
        """
        Stores all current extension instances in PICKLE_DIRECTORY + {extension.key}[_description].pickle.

        Parameters
        ----------
        description : Optional[str]
            Further qualifier of the extension instance
        """
        for extension in self._get_extensions():
            extension.pickle(self, description)

    def has_extension(self, extension: Type[T]) -> bool:
        """
        Checks whether this dataset has an extension of the given type. It will not only match the specific extension
        type but also any subtype.

        Parameters
        ----------
        extension : Type[Extension]
            the extension type to check

        Returns
        -------
        has_extension : bool
            True, if this data set is extended with the given extension type or a subtype of it
            False, otherwise
        """
        return extension.key() in self.__extensions or any(
            isinstance(ext, extension) for ext in self.__extensions.values())

    # noinspection Mypy
    def get_extension(self, extension: Type[T]) -> T:
        """
        Returns the extension of the given type.

        Parameters
        ----------
        extension : Type[Extension]
            the extension type to get

        Raises
        ------
        KeyError
            if the extension does not exist

        Returns
        -------
        get_extension : T
            the extension of the given type
        """
        if extension.key() in self.__extensions:
            return self.__extensions[extension.key()]
        elif any(isinstance(ext, extension) for ext in self.__extensions.values()):
            for ext in self.__extensions.values():
                if isinstance(ext, extension):
                    return ext
        else:
            raise KeyError(f"This dataset does not have an extension {extension.key()}")

    def _get_extensions(self) -> List[Extension]:
        """Returns all extensions"""
        return list(self.__extensions.values())

    def _set_extensions(self, extensions: Iterable[Extension]) -> None:
        """Sets the extension"""
        for ext in extensions:
            self.__extensions[ext.key()] = ext


class KeywordQueryLog(Data):
    """
    A log of real-world queries issued by users, used for statistics and query space generation. It behaves like a
    dataset, where a document is a query and its content are the keywords with an additional field indicating the
    id of the user that issued the query.
    """

    @abstractmethod
    def user_ids(self) -> List[str]:
        """Returns the unique identifiers of all users in this query log, ordered according to activity (descending,
        or ascending if the query log is reversed)."""
        raise NotImplementedError

    def keywords(self, user_id: str = None) -> Set[str]:
        """Returns the set of all queries in this query log."""
        return set(self.keywords_list(user_id))

    @abstractmethod
    def keywords_list(self, user_id: str = None) -> List[str]:
        """Returns the multiset of all queries in this query log (restricted to a user_id)."""
        raise NotImplementedError

    def __call__(self, user_id: str) -> Iterator[str]:
        yield from self.keywords_list(user_id)


class DummyKeywordQueryLogFromTrends(KeywordQueryLog):
    """Simulates a log by taking a list of queries"""
    __name: str
    __keywords_list: List[str]
    __keywords_trends: dict
    __frequencies: np.ndarray
    __n_kw: int
    __weeks: Tuple[int]
    __offset: int
    __selectivity: Selectivity
    __n_queries_per_week: int
    __chosen_keywords: List[str] = None
    __p: float

    def __init__(self, name: str, keywords_trends: dict, n_kw: int, weeks: Tuple[int], offset:int=0, n_queries_per_week: int = 5, selectivity: Selectivity=Selectivity.High, p: float = 1.0):
        self.__name = name
        self.__keywords_list = list(keywords_trends.keys())
        self.__keywords_trends = keywords_trends
        assert offset >=0 and weeks[0]-offset >=0 and weeks[0] < weeks[1] and weeks[1] <= len(keywords_trends), f"The weeks can only range from 0 to {len(keywords_trends)}"
        self.__weeks = weeks
        self.__offset = offset
        self.__selectivity = selectivity
        self.__n_queries_per_week = n_queries_per_week
        assert n_kw <= len(self.__keywords_list), f"The number of keywords to choose has to be smaller than {len(self.__keywords_list)}."
        self.__n_kw = n_kw
        self.__frequencies = self._trend_matrix()
        self.__p = p
        
        
        

    def name(self) -> str:
        return self.__name

    def user_ids(self) -> List[str]:
        return ["0"]

    def keywords_list(self, user_id: str = None, remove_endstates: bool = False) -> List[str]:
        # Ignores user_id and remove_endstates and instead just yield the initially supplied list.
        return self.__keywords_list
    
    def chosen_keywords(self):
        return self.__chosen_keywords
    
    def _choose_keywords(self):
        if self.__selectivity == Selectivity.High:
            chosen_keywords = sorted(self.__keywords_trends.keys(), key=lambda x: self.__keywords_trends[x]['count'], reverse=True)[:self.__n_kw]
        elif self.__selectivity == Selectivity.Low:
            chosen_keywords = sorted(self.__keywords_trends.keys(), key=lambda x: self.__keywords_trends[x]['count'])[:self.__n_kw]
        elif self.__selectivity == Selectivity.Independent:
            chosen_keywords = random.sample(self.__keywords_trends.keys(),self.__n_kw)
        else:
            log.error(f"Unknown selectivity: {self.__selectivity}")
        self.__chosen_keywords = chosen_keywords
        return chosen_keywords
    
    def _trend_matrix(self,chosen_keywords:list=None):
        if chosen_keywords is None:
            chosen_keywords = self._choose_keywords()
            self.__chosen_keywords = chosen_keywords
        trend_matrix = np.array([self.__keywords_trends[kw]['trend'] for kw in chosen_keywords])
        n_kw, n_weeks = trend_matrix.shape
        for i_col in range(n_weeks):
            if sum(trend_matrix[:,i_col]) == 0:
                trend_matrix[:,i_col] = 1/n_kw
            else:
                trend_matrix[:,i_col] = trend_matrix[:,i_col] / sum(trend_matrix[:,i_col])
        return trend_matrix

    def generate_queries(self):
        queries = []
        if self.__p < 1:
            self.__n_kw = int(100/self.__p)
            chosen_keywords = self._choose_keywords()
        elif self.__chosen_keywords is None:
            chosen_keywords = self._choose_keywords()
        else:
            chosen_keywords = self.__chosen_keywords
        for week in range(*self.__weeks):
            query_prob = self._trend_matrix(chosen_keywords)[:,week]
            queries += list(np.random.choice(chosen_keywords, self.__n_queries_per_week, p=query_prob))
        return queries

    def frequencies(self):
        return self.__frequencies[:,self.__weeks[0]-self.__offset:self.__weeks[1]-self.__offset]

    def open(self) -> 'Dataset':
        return self

    def close(self) -> None:
        pass

    def doc_ids(self) -> Set[str]:
        return []

    def documents(self) -> Iterator[Document]:
        pass

    def is_open(self) -> bool:
        return True

    def pickle(self) -> None:
        pass

    def query(self, keyword: str) -> Union[Iterator[Document], Iterator[str]]:
        pass


class SampledKeywordQueryLog(KeywordQueryLog):

    __name: str
    __keyword_cache: Cache[str, List[str]]
    __keywords_list: List[str]
    __user_ids: List[str]

    def __init__(self, name: str, keyword_cache: Cache):
        self.__name = name
        self.__keyword_cache = keyword_cache
        self.__keywords_list = [kw for keywords in self.__keyword_cache.values() for kw in keywords]
        self.__user_ids = self.__keyword_cache.keys()
        super().__init__()
    
    def name(self) -> str:
        return self.__name

    def query(self, user_id: str) -> Iterator[str]:
        yield from self.__keyword_cache[user_id]

    def user_ids(self) -> List[str]:
        return self.__user_ids

    def keywords_list(self, user_id: str = None) -> List[str]:
        if user_id is None:
            return self.__keywords_list
        else:
            return list(self.query(user_id))
