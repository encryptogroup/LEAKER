"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from abc import ABC, abstractmethod
from collections import Counter, namedtuple
from logging import getLogger

import powerlaw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Union, Iterable, Tuple

from numpy.core.multiarray import ndarray
from scipy.stats import stats

from ..util import fit_power_law_curve

log = getLogger(__name__)


class StatisticsPlotter(ABC):
    """
    A class for plotting different statistics gathered from the Statistics Module.
    """
    __filename: str

    __title: str
    __xlabel: str
    __ylabel: str

    _f: Figure
    _ax: Axes
    _plotted: bool

    def __init__(self, filename: str = None, title: str = "Statistics Plot",
                 xlabel: str = "x", ylabel: str = "y"):
        self.__filename = filename

        self.__title = title
        self.__xlabel = xlabel
        self.__ylabel = ylabel

        self._f, self._ax = plt.subplots()
        self._plotted = False

    def filename(self) -> str:
        return self.__filename

    def title(self) -> str:
        return self.__title

    def xlabel(self) -> str:
        return self.__xlabel

    def ylabel(self) -> str:
        return self.__ylabel

    def flush(self) -> None:
        if not self._plotted:
            self._f.clf()
            self._plotted = True

    @abstractmethod
    def gen(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError


PowerLawFittingResults = namedtuple("PowerLawFittingResults", ["freq_exponent", "alpha", "R", "p"])


class FrequencyPlotter(StatisticsPlotter):
    """
    Frequency distribution plot and scatter plot of frequencies of provided keywords + MLE fitted curve
    (so far:only fit power law). Alternatively, just the frequencies can be provided. If both types of data are
    provided (keywords + frequencies), just the keywords will be considered.
    Credit: https://stats.stackexchange.com/questions/331219 and
    https://www.digitalocean.com/community/tutorials/how-to-graph-word-frequency-using-matplotlib-with-python-3 and
    https://arxiv.org/pdf/1305.0215.pdf
    """
    __keywords = List[str]

    __fitted_exponent: Union[None, float]
    __fitted_alpha: Union[None, float]
    __fitted_r: Union[None, float]
    __fitted_p: Union[None, float]
    __fitted_curve: Union[None, ndarray]
    __topic_counts: Union[None, ndarray]
    __freq_of_topic_counts: Union[None, ndarray]
    __occurrences: Union[List[int], List[float]]
    __fit: Union[None, powerlaw.Fit]
    __invalid: bool
    __max_xval: Union[int, None]
    __normalize: bool

    def __init__(self, filename: str = None, title: str = "Distribution of Frequencies",
                 xlabel: str = "frequency", ylabel: str = "occurrences", max_xval: Union[int, None] = None,
                 normalize: bool = False):
        self.__keywords = []

        super(FrequencyPlotter, self).__init__(filename, title, xlabel, ylabel)

        self.__topic_counts = None
        self.__freq_of_topic_counts = None
        self.__occurrences = []

        self.__fitted_exponent = None
        self.__fitted_curve = None
        self.__fitted_alpha = None
        self.__fitted_p = None
        self.__fitted_r = None
        self.__fit = None
        self.__invalid = False
        self.__max_xval = max_xval
        self.__normalize = normalize

    def offer_keywords(self, keywords: List[str]) -> None:
        self.__keywords.extend(keywords)

    def offer_occurrences(self, occurrences: List[int]) -> None:
        """
        Offer occurrences instead of keywords. This data will only be used in gen() if no keywords are provided
        """
        self.__occurrences.extend(occurrences)

    def gen(self) -> None:
        if len(self.__keywords) > 0:
            self.__occurrences = [count for _, count in Counter(self.__keywords).most_common()]
        else:
            self.__occurrences = sorted(self.__occurrences, reverse=True)
        if self.__normalize:
            denom = sum(self.__occurrences)
            if denom != 0:
                self.__occurrences = [occ/denom for occ in self.__occurrences]

        counter_of_counts = Counter(self.__occurrences)
        self.__topic_counts = np.array(list(counter_of_counts.keys()))
        self.__freq_of_topic_counts = np.array(list(counter_of_counts.values()))

        if len(self.__topic_counts) == 0 or len(self.__freq_of_topic_counts) == 0:
            log.warning(f"Not enough data for {self.filename()}. Skipping that.")
            self.__invalid = True
        elif not self.__normalize:
            self.__fitted_exponent, self.__fitted_curve = \
                fit_power_law_curve(self.__topic_counts, self.__freq_of_topic_counts)

            self.__fit = powerlaw.Fit(self.__occurrences, discrete=True)
            self.__fitted_alpha = self.__fit.power_law.alpha
            self.__fitted_r, self.__fitted_p = self.__fit.distribution_compare('power_law', 'exponential')

    def plot(self):
        if self.__topic_counts is None and not self.__invalid:
            self.gen()

        if not self.__invalid and not self.__normalize:
            """1st plot: Scatter frequencies of frequencies"""
            self._ax.scatter(self.__topic_counts, self.__freq_of_topic_counts, label="Original Data", color='blue')
            self._ax.set_xlabel(self.xlabel())
            self._ax.set_ylabel(self.ylabel())
            self._ax.set_title(self.title())
            self._ax.set_xscale('log')
            self._ax.set_yscale('log')
            self._ax.set_ylim(bottom=0.9)

            self._ax.plot(self.__topic_counts, self.__fitted_curve, '--', color="orange",
                          lw=3, label=f"Fitted PL-MLE x={self.__fitted_exponent}")
            self._ax.legend()

            self._f.savefig(f"{self.filename()[:-4]}_scatter.png")

            import tikzplotlib

            tikzplotlib.save(f"{self.filename()[:-4]}_scatter.tikz")

            self._f.clf()

            """2nd plot: Complementary cumulative distribution function"""

            self._f, self._ax = plt.subplots()
            self._ax.set_title(self.title())
            self._ax.set_ylabel(r'$P(X\geq x)$')
            self._ax.set_xlabel("Word frequency")
            self._ax = self.__fit.plot_ccdf(linewidth=3, label="Original Data")

            self.__fit.power_law.plot_ccdf(ax=self._ax, color='r', linestyle='--',
                                           label=f"Power Law Fit $\\alpha = {round(self.__fit.power_law.alpha, 4)}$")
            self.__fit.lognormal.plot_ccdf(ax=self._ax, color='g', linestyle='--',
                                           label=f"Lognormal Fit $\\mu = {round(self.__fit.lognormal.mu, 4)}, "
                                                 f"\\sigma = {round(self.__fit.lognormal.sigma, 4)}$")
            self._ax.legend()
            self._f.savefig(f"{self.filename()[:-4]}_ccdf.png")
            tikzplotlib.save(f"{self.filename()[:-4]}_ccdf.tikz")

            self._f.clf()
        if not self.__invalid:
            """3rd plot: Log frequencies of ranked keywords"""

            self._f, self._ax = plt.subplots()
            self._ax.set_title(self.title())
            if not self.__normalize:
                self._ax.set_ylabel("Total number of occurrences")
            else:
                self._ax.set_ylabel("Fraction of occurrences")
            self._ax.set_xlabel("Rank of query")

            if self.__max_xval is not None:
                if len(self.__occurrences) < self.__max_xval:
                    self._ax.set_xlim(1, self.__max_xval)
                    self.__occurrences.append(0)
            self._ax.set_xscale("log")
            if self.__normalize:
                self._ax.set_yscale('linear')
                self._ax.set_ylim(0, max(max(self.__occurrences), 0.05))
            else:
                self._ax.set_yscale('log')
            self._ax.plot(range(1, len(self.__occurrences) + 1), sorted(self.__occurrences, reverse=True))

            self._ax.legend()

            self._f.savefig(f"{self.filename()[:-4]}_freq.png")
            import tikzplotlib
            tikzplotlib.save(f"{self.filename()[:-4]}_freq.tikz")
            self.flush()

    def fitted_exponent(self) -> PowerLawFittingResults:
        """
        Fit curve without plotting

        Returns
        ---------
        fitted_exponent: float
        """
        if self.__fitted_exponent is None:
            self.gen()
        return PowerLawFittingResults(freq_exponent=self.__fitted_exponent, alpha=self.__fitted_alpha,
                                      R=self.__fitted_r, p=self.__fitted_p)

    def __call__(self, keywords: List[str] = None) -> PowerLawFittingResults:
        if keywords is not None:
            self.offer_keywords(keywords)
        self.plot()
        return self.fitted_exponent()


class SelectivityPlotter(StatisticsPlotter):
    """
    Plots correlation between queries' frequencies and their selectivities
    """
    __selectivities: List[int]
    __frequencies: List[int]
    __corr_co: Union[None, float]
    __regr_line: Union[None, ndarray]
    __invalid: bool

    def __init__(self, filename: str = None, title: str = "Frequency-Selectivity Correlation",
                 xlabel: str = "Query frequency", ylabel: str = "Query selectivity"):
        super(SelectivityPlotter, self).__init__(filename, title, xlabel, ylabel)

        self.__selectivities = []
        self.__frequencies = []
        self.__corr_co = None
        self.__invalid = False

    def offer_data(self, data: Iterable[Tuple[int, int]]) -> None:
        """
        Parameters
        ---------
        data: Iterable[Tuple[int, int]]
            an Iterable of (selectivity of query, query frequency) data points
        """
        for sel, freq in data:
            self.__selectivities.append(sel)
            self.__frequencies.append(freq)

    def gen(self) -> None:
        """
        Computes the (Pearson) correlation between selectivities and frequencies
        """
        if len(self.__selectivities) == 0 or len(self.__frequencies) == 0:
            log.warning(f"Not enough data for {self.filename()}. Skipping that.")
            self.__invalid = True
        else:
            slope, intercept, r, p, stderr = stats.linregress(self.__selectivities, self.__frequencies)
            self.__corr_co = r
            self.__regr_line: ndarray = intercept + slope * np.array(range(max(self.__selectivities)))

    def plot(self) -> None:
        if (self.__corr_co is None or self.__regr_line is None) and not self.__invalid:
            self.gen()

        if not self.__invalid:
            self._ax.set_title(self.title())
            self._ax.set_ylabel("Selectivity of query")
            self._ax.set_xlabel("Frequency of query")
            self._ax.scatter(self.__selectivities, self.__frequencies, color='blue')
            self._ax.plot(range(max(self.__selectivities)), self.__regr_line, '--', color="orange",
                          lw=3, label=f"Pearson's r={self.__corr_co}")
            self._ax.legend()

            self._f.savefig(self.filename())
            import tikzplotlib
            tikzplotlib.save(f"{self.filename()[:-4]}.tikz")
            self.flush()

    def correlation_coefficient(self) -> float:
        if (self.__corr_co is None or self.__regr_line is None) and not self.__invalid:
            self.gen()

        return self.__corr_co

    def __call__(self, data: Iterable[Tuple[int, int]] = None) -> float:
        if data is not None:
            self.offer_data(data)
        self.plot()

        return self.correlation_coefficient()


class HeatMapPlotter(StatisticsPlotter):
    """
    A class for plotting heatmaps.
    """
    __filename: str

    __queries: ndarray
    __max_val: int
    __creation_failed: bool

    def __init__(self, max_val: int, filename: str = None, title: str = "Occurrences", xlabel: str = "upper bound",
                 ylabel: str = "lower bound"):
        super().__init__(filename, title, xlabel, ylabel)
        self.__max_val = max_val
        self.__creation_failed = False
        try:
            self.__queries = np.zeros((self.__max_val + 1, self.__max_val + 1), dtype=np.int64)
        except MemoryError:
            log.warning(f"Could not allocate heatmap of size {self.__max_val + 1}x{self.__max_val + 1}. "
                        f"Will only print out basic statistics instead.")
            self.__creation_failed = True

    def offer_data(self, queries: Iterable[Tuple[int, int]]) -> None:
        if not self.__creation_failed:
            for query in queries:
                if 0 < query[0] <= self.__max_val and 0 < query[1] <= self.__max_val:
                    self.__queries[self.__max_val - query[0], query[1]] += 1

    def gen(self) -> None:
        pass

    def plot(self) -> None:
        if not self.__creation_failed:
            im = self._ax.imshow(self.__queries)
            ticks = list(range(self.__max_val + 1))
            self._ax.set_xticks(ticks)
            self._ax.set_yticks(ticks)

            self._ax.set_yticklabels(reversed(ticks))
            self._f.savefig(self.filename())
            import tikzplotlib
            tikzplotlib.save(f"{self.filename()[:-4]}.tikz")
            self.flush()


class RangesPlotter(StatisticsPlotter):
    """
    A class for plotting ranges.
    """
    __filename: str

    __queries: List[Tuple[int, int]]
    __max_val: int

    def __init__(self, max_val: int, filename: str = None, title: str = "Ranges", xlabel: str = "Queries",
                 ylabel: str = "Ranges"):
        super().__init__(filename, title, xlabel, ylabel)
        self.__max_val = max_val
        self.__creation_failed = False
        self.__queries = []

    def offer_data(self, queries: Iterable[Tuple[int, int]]) -> None:
        if not self.__creation_failed:
            for query in queries:
                lower = max(1, query[0])
                upper = min(self.__max_val, query[1])
                self.__queries.append((lower, upper))

    def gen(self) -> None:
        pass

    def plot(self) -> None:
        self._ax.set_title(self.title())
        self._ax.set_ylabel(self.ylabel())
        self._ax.set_xlabel(self.xlabel())

        queries = Counter(self.__queries).most_common()

        yerr = [(item[1] - item[0]) / 2 for item, _ in queries]
        y = [item[0] + (item[1] - item[0]) / 2 for item, _ in queries]

        self._ax.errorbar(list(range(1, len(queries) + 1)), y, yerr=yerr)
        self._f.savefig(self.filename())
        import tikzplotlib
        tikzplotlib.save(f"{self.filename()[:-4]}.tikz")
        self.flush()
