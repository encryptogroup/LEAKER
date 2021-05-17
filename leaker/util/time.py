"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from math import trunc
from time import process_time, perf_counter
from typing import Tuple, Optional


class Duration:
    """
    A utility class for representing durations that can be pretty-printed. The string format will be
    "[..h] [..min] [..s] [...ms]" where "." is a placeholder for a digit and parts may be left out if their value
    is zero.

    When creating an instance of this class, all values will be propagated left as far as possible. For example,
    Duration(millis=3662500) will actually return Duration(hours=1, minutes=1, seconds=2, millis=500).

    Parameters
    ----------
    hours: int
        The number of hours
    minutes: int
        The number of minutes
    seconds: int
        The number of seconds
    millis: int
        The number of milliseconds
    """

    __hours: int
    __minutes: int
    __seconds: int
    __millis: int

    def __init__(self, hours: int = 0, minutes: int = 0, seconds: int = 0, millis: int = 0):
        self.__hours, self.__minutes, self.__seconds, self.__millis = Duration.__shift_left(hours, minutes, seconds,
                                                                                            millis)

    @staticmethod
    def __shift_left(hours: int, minutes: int, seconds: int, millis: int) -> Tuple[int, int, int, int]:
        seconds += millis // 1000
        millis %= 1000

        minutes += seconds // 60
        seconds %= 60

        hours += minutes // 60
        minutes %= 60

        return hours, minutes, seconds, millis

    def hours(self) -> int:
        """
        Returns the hour part

        Returns
        -------
        hours: int
            the hour part
        """
        return self.__hours

    def minutes(self) -> int:
        """
        Returns the minute part

        Returns
        -------
        minutes: int
            the minute part
        """
        return self.__minutes

    def seconds(self) -> int:
        """
        Returns the second part

        Returns
        -------
        seconds: int
            the second part
        """
        return self.__seconds

    def millis(self) -> int:
        """
        Returns the millisecond part

        Returns
        -------
        millis: int
            the millisecond part
        """
        return self.__millis

    def __repr__(self):
        if self.__millis == 0 and self.__seconds == 0 and self.__minutes == 0 and self.__hours == 0:
            return "0s"

        r = ""

        if self.__millis != 0:
            r = f"{self.__millis:03}ms"

        if self.__seconds != 0 or ((self.__hours != 0 or self.__minutes != 0) and self.__millis != 0):
            r = f"{self.__seconds:02}s {r}"

        if self.__minutes != 0 or (self.__hours != 0 and (self.__seconds != 0 or self.__millis != 0)):
            r = f"{self.__minutes:02}min {r}"

        if self.__hours != 0:
            r = f"{self.__hours:02}h {r}"

        return r

    @staticmethod
    def from_fractional_seconds(seconds: float) -> 'Duration':
        """
        Creates a duration from a fractional number of seconds. The number will be truncated to milliseconds precision.

        Parameters
        ----------
        seconds: float
            the fractional number of seconds

        Returns
        -------
        from_fractional_seconds: Duration
            the Duration representing the time span expressed by seconds
        """
        millis = trunc(seconds * 1000)
        return Duration(millis=millis)


class Stopwatch:
    """
    A utility class for measuring durations using `time.perf_counter` or `time.process_time`. It is capable of
    measuring lap times just like a real stop watch.

    Parameters
    ----------
    use_process_time: bool
        Whether to use `time.process_time` in place of `time.perf_counter`,
        default: False
    """
    __process_time: bool
    __start_time: Optional[float]
    __last_lap_time: Optional[float]

    def __init__(self, use_process_time: bool = False):
        self.__process_time = use_process_time
        self.__start_time = None
        self.__last_lap_time = None

    def __measurement(self) -> float:
        return process_time() if self.__process_time else perf_counter()

    def start(self) -> 'Stopwatch':
        """
        Starts the stop watch, i.e. stores the current value of the measurement function as the reference point.

        Returns
        -------
        start: Stopwatch
            this object
        """
        if self.__start_time is not None:
            raise RuntimeError("This stopwatch is already running")

        self.__start_time = self.__measurement()
        return self

    def stop(self) -> Duration:
        """
        Stops the stop watch, i.e. deletes the reference point, and returns the `Duration` from the reference point
        to the current value of the measurement function.

        Returns
        -------
        stop: Duration
            the amount of time the Stopwatch was running
        """
        if self.__start_time is None:
            return Duration()

        end_time = self.__measurement()
        duration = end_time - self.__start_time

        self.__start_time = None

        return Duration.from_fractional_seconds(duration)

    def lap(self) -> Duration:
        """
        Returns the current lap time, i.e. the amount of time between the last point where `lap` was called (or the
        start point if it was never called before) and the current point in time.

        Returns
        -------
        lap: Duration
            the current lap duration
        """
        if self.__start_time is None:
            return Duration()

        current_time = self.__measurement()
        reference_point = self.__start_time if self.__last_lap_time is None else self.__last_lap_time
        duration = current_time - reference_point

        self.__last_lap_time = current_time

        return Duration.from_fractional_seconds(duration)
