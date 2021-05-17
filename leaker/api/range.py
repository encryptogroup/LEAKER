"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from typing import Iterator

import numpy as np


class Range:
    """Works like the python built-in range(...), but with float values."""

    __start: float
    __end: float
    __step: float
    __inclusive: bool

    def __init__(self, start: float, end: float, step: float, inclusive: bool = True):
        """
        Creates a new range.

        Parameters
        ----------
        start : float
            the first value (inclusive)
        end : float
            the last value (inclusive/exclusive)
        step : float
            the step size
        inclusive : bool
            whether the end of the range should be inclusive
        """
        if (start < end and step < 0) or (start > end and step > 0):
            raise ValueError("Can never reach the end")

        self.__start = start
        self.__end = end
        self.__step = step
        self.__inclusive = inclusive

    def __iter__(self) -> Iterator[float]:
        for value in np.arange(self.__start, self.__end, self.__step):
            yield value
        if self.__inclusive:
            yield self.__end
