"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
import dill as pickle
from typing import Generic, TypeVar, Dict, Iterator, Mapping, Callable, Optional, Set, Hashable

T = TypeVar("T")
A = TypeVar("A", bound=Hashable)


class Cache(Generic[A, T], Mapping[A, T]):
    """A utility class to provide means of caching precomputed data or data computed on the fly with hashable keys.
    Results can be pickled and loaded between usages."""
    __cache: Dict[A, T]
    __max_elements: int

    def __init__(self, cache: Dict[A, T], accessor: Callable[[A], T], max_elements: int = 0):
        self.__cache = cache

        self.__max_elements = max_elements

        self.__accessor: Callable[[A], T] = accessor

    def compute_if_absent(self, key: A) -> T:
        if key not in self.__cache:
            val = self.__accessor(key)
            if self.__max_elements == 0 or len(self.__cache.keys()) < self.max_elements():
                self.__cache[key] = val
        else:
            val = self.__cache[key]

        return val

    def __getitem__(self, key: A) -> T:
        return self.compute_if_absent(key)

    def __iter__(self) -> Iterator[A]:
        return iter(self.__cache)

    def __len__(self) -> int:
        return len(self.__cache)

    @staticmethod
    def __build(accessor: Callable[[A], T], keywords: Set[A], max_elements: int) -> Dict[A, T]:
        cache: Dict[A, T] = dict()

        for keyword in keywords:
            cache[keyword] = accessor(keyword)
            if max_elements != 0 and len(cache) >= max_elements:
                break

        return cache

    @classmethod
    def build(cls, accessor: Callable[[A], T], keys: Optional[Set[A]] = None, max_elements: int = 0) -> 'Cache[A, T]':
        """
        Creates a cache for the given accessor method. All values can be precomputed by supplying the full set
        of cache key words.

        Parameters
        ----------
        accessor : Callable[[A], T]
            the accessor function to cache results for
        keys : Optional[Set[A]]
            the keys to precompute values for
        max_elements: int
            the maximum size of the cache

        Returns
        -------
        build : Cache[A, T]
            the cache
        """
        if keys is None:
            return cls(dict(), accessor)

        return cls(cls.__build(accessor, keys, max_elements), accessor)

    def pickle(self, filename: str) -> None:
        """
        Stores the cache dict object

        Parameters
        ----------
        filename : str
            the filename
        """
        pickle.dump((self.__cache, self.__max_elements), open(filename, "wb"))

    @classmethod
    def load_pickle(cls, accessor: Callable[[A], T], filename: str,) -> 'Cache[A, T]':
        """
        Loads the cache object

        Parameters
        ----------
        filename : str
            the pickle filename
        accessor : Callable[[A], T]
            the accessor function to cache results for
        """
        items = pickle.load(open(filename, "rb"))
        return cls(items[0], accessor, max_elements=items[1])

    def max_elements(self) -> int:
        return self.__max_elements
