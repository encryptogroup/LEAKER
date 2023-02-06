"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
import os
from abc import ABC
from typing import List, Dict, Optional, Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..api.constants import FIGURE_DIRECTORY
from ..api import DataSink


class MatPlotLibSink(DataSink, ABC):
    """
    A data sink for plotting the performance data with matplotlib.

    Parameters
    ----------
    out_file : Optional[str]
        if set, the plot will be written to the specified file
        default: None
    markers : Optional[List[str]]
        the actual point markers to use, as defined by matplotlib.
        the markers are used in order.
        if not specified, a set of six default markers will be used.
        default: None
    """
    __markers: List[str]
    _metric: Dict[float, float]

    _out_file: Optional[str]
    __data: Dict[str, Dict[int, Dict[float, List[float]]]]

    def __init__(self, out_file: Optional[str] = None, markers: Optional[List[str]] = None, metric: Optional[Dict[float,float]] = []):
        self.__markers = markers or ['x', 'o', 's', 'D', '|', '+']
        self._metric = metric
        self._out_file = out_file
        if self._out_file is not None:
            if self._out_file.count('.') > 1:
                self._out_file = self._out_file.replace('.', '_', 1)
            self._out_file = FIGURE_DIRECTORY + self._out_file
        self.__data = {}

    def register_series(self, series_id: str) -> None:
        self.__data[series_id] = {}

    def offer_data(self, series_id: str, user_id: int, known_data_rate: float, recovery_rate: float) -> None:
        if user_id not in self.__data[series_id]:
            self.__data[series_id][user_id] = {}
        if known_data_rate not in self.__data[series_id][user_id]:
            self.__data[series_id][user_id][known_data_rate] = []
        self.__data[series_id][user_id][known_data_rate].append(recovery_rate)

    def yield_plotpoints(self, use_mean: bool = False) \
            -> Iterator[Tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray, str]]:
        err = np.mean if use_mean else np.median
        for series_id in self.__data.keys():
            if len(self.__data[series_id]) == 0:
                continue

            series_data: Dict[int, Dict[float, List[float]]] = self.__data[series_id]

            x = np.array(sorted(series_data[0].keys()))
            y = np.array([err([err(series_data[user_id][kdr]) for user_id in series_data.keys()])
                          for kdr in x])

            y_max = np.array(
                [np.max([series_data[user_id][kdr] for user_id in series_data.keys()]) for kdr in x]) - y
            y_min = y - np.array(
                [np.min([series_data[user_id][kdr] for user_id in series_data.keys()]) for kdr in x])

            yield (x, y, series_id, y_min, y_max, self.__markers.pop(0))


class KeywordMatPlotLibSink(MatPlotLibSink):
    """Uses the median of reconstruction rates"""
    def flush(self) -> None:
        plt.figure(dpi=300)

        plt.xlabel('Partial Knowledge in %')
        plt.xlim(0, 105)
        plt.xticks(np.linspace(0, 100, num=11))

        plt.ylabel('Recovery Rate')
        plt.ylim(0, 1.05)
        plt.yticks(np.linspace(0, 1, num=6))

        plt.grid(True)

        for x, y, series_id, y_min, y_max, marker in self.yield_plotpoints():

            if not y_max.any() and not y_min.any():
                plt.plot(x * 100, y, label=series_id, marker=marker, linewidth=1)
            else:
                plt.errorbar(x * 100, y, yerr=[y_min, y_max], label=series_id, marker=marker,
                             linewidth=1, capsize=3)

        plt.legend()

        if self._out_file is not None:
            if not os.path.exists(FIGURE_DIRECTORY):
                os.makedirs(FIGURE_DIRECTORY)
            plt.savefig(self._out_file)
            import tikzplotlib
            tikzplotlib.save(f"{self._out_file[:-4]}.tikz")
        else:
            plt.show()

class SampledMatPlotLibSink(MatPlotLibSink):
    """Uses the median of reconstruction rates"""
    def flush(self) -> None:
        plt.figure(dpi=300)

        plt.xlabel('Sample Size in %')
        plt.xlim(-2, 52)
        plt.xticks(np.linspace(0, 50, num=6))

        plt.ylabel('Recovery Rate')
        plt.ylim(0, 1.05)
        plt.yticks(np.linspace(0, 1, num=6))

        plt.grid(True)

        for x, y, series_id, y_min, y_max, marker in self.yield_plotpoints():
            if len(self._metric) > 0:
                assert(len(x) == len(self._metric))
                x = np.array([self._metric[key] for key in x])
                plt.xlabel('Statistical Closeness in %')
                plt.xlim(45,105)
                plt.xticks(np.linspace(50, 100, num=6))

            if not y_max.any() and not y_min.any():
                plt.plot(x * 100, y, label=series_id, marker=marker, linewidth=1)
            else:
                plt.errorbar(x * 100, y, yerr=[y_min, y_max], label=series_id, marker=marker,
                             linewidth=1, capsize=3)

        plt.legend()

        if self._out_file is not None:
            if not os.path.exists(FIGURE_DIRECTORY):
                os.makedirs(FIGURE_DIRECTORY)
            plt.savefig(self._out_file)
            import tikzplotlib
            tikzplotlib.save(f"{self._out_file[:-4]}.tikz")
        else:
            plt.show()


class RangeMatPlotLibSink(MatPlotLibSink):
    """
    A data sink for plotting the performance data with matplotlib that uses the mean of provided errors.

    Parameters
    ----------
    out_file : Optional[str]
        if set, the plot will be written to the specified file
        default: None
    markers : Optional[List[str]]
        the actual point markers to use, as defined by matplotlib.
        the markers are used in order.
        if not specified, a set of six default markers will be used.
        default: None
    log_y : Optional[bool]
        whether the y axis should have a logarithmic y-scale (usually used if the provided errors are not normalized)
        default: False
    """

    __log_y: bool

    def __init__(self, out_file: Optional[str] = None, markers: Optional[List[str]] = None,
                 log_y: Optional[bool] = False):

        super().__init__(out_file, markers)
        self.__log_y = log_y

    def flush(self) -> None:
        plt.figure(dpi=300)

        plt.xlabel('#Q')
        plt.xscale('log')

        plt.ylabel('Error')
        if self.__log_y:
            plt.yscale('symlog')

        plt.grid(True)

        for x, y, series_id, y_min, y_max, marker in self.yield_plotpoints(use_mean=True):

            if not y_max.any() and not y_min.any():
                plt.plot(x, y, label=series_id, marker=marker, linewidth=1)
            else:
                plt.errorbar(x, y, yerr=[y_min, y_max], label=series_id, marker=marker,
                             linewidth=1, capsize=3)

        plt.legend()

        if self._out_file is not None:
            if not os.path.exists(FIGURE_DIRECTORY):
                os.makedirs(FIGURE_DIRECTORY)
            plt.savefig(self._out_file)
            import tikzplotlib
            tikzplotlib.save(f"{self._out_file[:-4]}.tikz")
        else:
            plt.show()
