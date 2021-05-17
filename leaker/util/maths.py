"""
For License information see the LICENSE file.

Authors: Amos Treiber, Michael Yonli

"""
import numpy as np
from numpy.core.multiarray import ndarray
from scipy.optimize import minimize
from scipy.stats import beta
from typing import Tuple


def power_law_llh(b: float, counts: ndarray, freq_of_counts: ndarray) -> float:
    """
    LogLikelihood of a potential power law curve (with frequencies and their frequencies).
    Credit: https://stats.stackexchange.com/questions/331219

    Parameters
    ----------
    b: float
        power law exponent
    counts: ndarray
        frequencies of elements
    frerq_of_counts: ndarray
        frequencies of elements within counts

    Returns
    ----------
    -llh: float
        (-1)*loglikelihood
    """
    # Power law function
    probabilities = counts ** (-b)

    # Normalized
    probabilities = probabilities / probabilities.sum()

    lvector = np.log(probabilities) * freq_of_counts
    llh = lvector.sum()
    # We want to minimize (-1)*LogLikelihood
    return -llh


def fit_power_law_curve(counts: ndarray, freq_of_counts: ndarray) -> Tuple[float, ndarray]:
    """
    Fit power law curve (with frequencies and their frequencies).
    Credit: https://stats.stackexchange.com/questions/331219

    Parameters
    ----------
    counts: ndarray
        frequencies of elements
    frerq_of_counts: ndarray
        frequencies of elements within counts

    Returns
    ----------
    s_best.x, fitted_curve: Tuple[float, ndarray]
        fitted exponent and points of the fitted curve
    """
    s_best = minimize(power_law_llh, np.array([2]), (counts, freq_of_counts))
    return s_best.x, np.max(freq_of_counts) * counts ** -s_best.x


def beta_intervals(a: int, b: int, n: int) -> np.ndarray:
    """
    A Beta pdf with the parameters alpha = a and beta = b is discretized by splitting the interval from 0 to 1 into n
    equally long intervals and calculating the probability of each segment with the pdf.

    These probabilities are then returned.
    :param a: alpha of beta pdf
    :param b: beta of beta pdf
    :param n: number of intervals for discretization
    :return: probabilities of the n segments
    """
    segment_edges = np.linspace(0, 1, n + 1)
    edge_probs = beta.cdf(segment_edges, a, b)
    segment_probs = np.diff(edge_probs)

    return segment_probs
