"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from logging import getLogger
from typing import List, Union, Iterable, Generic, TypeVar

from .pipeline import Sink, Source

log = getLogger(__name__)

T = TypeVar("T", covariant=True)


class Preprocessor(Generic[T]):
    """
    The main component of the preprocessing library. It is capable of running preprocessing of a single source to
    an arbitrary number of sinks. The sinks may be preceded by filters.

    Note that, when using multiple sinks, the source needs to be capable of yielding the
    elements more than once.

    Parameters
    ----------
    source: Source[T]
        The source to preprocess elements from
    sinks: Union[Sink[T], Iterable[Sink[T]]]
        One or multiple sinks to pipe the elements to.
    """

    __source: Source[T]
    __sinks: List[Sink[T]]

    def __init__(self, source: Source[T], sinks: Union[Sink[T], Iterable[Sink[T]]]):
        self.__source = source

        if isinstance(sinks, Sink):
            self.__sinks = [sinks]
        else:
            self.__sinks = list(sinks)

    def run(self):
        """
        Lets the sinks consume the elements produced by the source sequentially, i.e. one sink at a time.
        """
        for sink in self.__sinks:
            self.__source >> sink
