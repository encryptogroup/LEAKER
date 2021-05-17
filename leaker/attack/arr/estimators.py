"""
This file contains the implementation of various estimators that are described in KPT19.
These are Valiant-Valiant (= unseen), Jackknife and the modular estimator.

For License information see the LICENSE file.
Authors: Michael Yonli
"""
from logging import getLogger
from math import ceil, sqrt, log, factorial

import numba
import numpy as np
from mpmath import mp, binomial
from scipy.optimize import linprog
from scipy.stats import poisson, norm

logger = getLogger(__name__)

bino_table = dict()


def bino_cdf(k: int, n: int, p: float) -> float:
    """
    Implementation of a binomial cumulative distribution function.
    The scipy implementation caused issues due to limited precision.
    Coefficients are stored in bino_table for performance benefits.

    :param k: k out of n
    :param n: k out of n
    :param p: the probability of success
    :return: probability of at most k successes
    """
    k = int(k)
    n = int(n)

    if p == 0 != k or p == 1 and k != n or k > n:
        return 0

    if n not in bino_table:
        bino_table[n] = dict()
    if k not in bino_table[n]:
        bino_table[n][k] = binomial(n, k)

    bino_cof = bino_table[n][k]

    return float(bino_cof * pow(p, k) * pow(1 - p, n - k))


@numba.njit
def numba_hist(samples, bins, range):
    """
    Wrapper to optimise numpy's histogram function with numba.
    """
    return np.histogram(samples, bins, range)


def get_fingerprint(samples: np.ndarray) -> np.ndarray:
    """
    Calculates a fingerprint as used by KPT20.

    Input are samples from some distribution.
    Returns a fingerprint (i.e. a histogram of a histogram) s.t. f[x] is the number of samples occurring x times.
    :param samples: samples of some distribution
    :return: fingerprint
    """
    if type(samples) == np.ndarray:
        samples = samples.astype(int)
    else:
        samples = np.array(samples)

    m_samp = samples.max() + 1
    fp, _ = numba_hist(samples, bins=int(m_samp), range=(0, m_samp))

    m_fp = fp.max() + 1
    fp, _ = numba_hist(fp, bins=int(m_fp), range=(0, m_fp))

    fp[0] = 0
    fp = np.trim_zeros(fp, "b")
    return fp


def unseen(fingerprint):
    """
    An implementation of the unseen estimator also known as Valiant-Valiant.
    https://theory.stanford.edu/~valiant/papers/unseenJournal.pdf
    https://theory.stanford.edu/~valiant/code/unseenCode.zip

    :param fingerprint: a fingerprint as returned by get_fingerprint
    :return: (x, hist) such that hist[i] is the number of elements that occur with probability x[i]
    """
    k = 0  # number of samples
    for i in range(len(fingerprint)):
        k += i * fingerprint[i]

    gridFactor = 1.05
    alpha = 0.5

    min_i = 0
    while fingerprint[min_i] == 0:
        min_i += 1

    #  set smallest possible probability
    if min_i > 1:
        xLPmin = min_i / k
    else:
        xLPmin = 1 / (k * max(10, k))

    x = [0]  # list of probabilities
    histx = [0]  # list of counts
    fLP = []

    fingerprint = fingerprint[1:]
    for i, f_i in enumerate(fingerprint):
        if f_i:
            wind = [max(0, i - ceil(sqrt(i + 1))), min(i + ceil(sqrt(i + 1)) + 1, len(fingerprint))]
            if sum(fingerprint[wind[0]:wind[1]]) < sqrt(i + 1):
                x.append((i + 1) / k)
                histx.append(f_i)
                fLP.append(0)
            else:
                fLP.append(f_i)
        else:
            fLP.append(0)

    fLP_nonzero = [i for i in range(len(fLP)) if fLP[i] > 0]

    if not fLP_nonzero:
        logger.debug("All fLP are zero. Returning data from fingerprints without linear programming.")
        return x[1:], histx[1:]

    fmax = max(fLP_nonzero)

    LPmass = 1 - sum(map(lambda count, p: count * p, histx, x))

    fLP = fLP[:fmax + 1] + [0] * ceil(sqrt(fmax + 1))
    szLPf = len(fLP)

    xLPmax = (fmax + 1) / k
    # corresponds to x_1 (= xLPmin), ... x_l (= xLPmax or slightly greater) in the paper
    xLP = [xLPmin * gridFactor ** p for p in range(ceil(log(xLPmax / xLPmin) / log(gridFactor)) + 1)]
    szLPx = len(xLP)

    # the term to minimise
    objf = [0] * (szLPx + 2 * szLPf)
    # x[0:szLPx] represents counts and x[szLPx:szLPf * 2] is set to satisfy the constraint Ax <= b in a manner which
    # represents the difference between the expected fingerprint sum(x[:szLPx] * poi(i, k * xLP)) and fLP[i]
    objf[szLPx + 0::2] = [1 / sqrt(s + 1) for s in fLP]
    objf[szLPx + 1::2] = [1 / sqrt(s + 1) for s in fLP]

    # Ax <= b
    A = np.zeros((2 * szLPf, szLPx + 2 * szLPf))
    b = np.zeros(2 * szLPf)

    for i in range(szLPf):
        A[2 * i, :szLPx] = poisson.pmf(i + 1, k * np.array(xLP))
        A[2 * i + 1, :szLPx] = (-1) * A[2 * i, :szLPx]

        A[2 * i, szLPx + 2 * i] = -1
        A[2 * i + 1, szLPx + 2 * i + 1] = -1

        b[2 * i] = fLP[i]
        b[2 * i + 1] = - fLP[i]

    # ensures that product of counts and probabilities is equal to the unexplained probability
    Aeq = np.zeros((1, szLPx + 2 * szLPf))
    Aeq[:, :szLPx] = xLP
    beq = LPmass

    maxLPIters = 1000
    options = {'maxiter': maxLPIters, 'disp': False, 'autoscale': True}

    logger.debug("Starting first linear programming.")
    # Default bounds are (0, inf)
    res = linprog(objf, A, b, Aeq, beq, options=options)
    sol = res['x']
    exitflag = res['success']

    logger.debug(f"Done exitflag: {exitflag}")

    fval = res['fun']

    if not exitflag:
        logger.warning("Failed to find a solution for LP1")
        logger.warning(res['message'])
    else:
        logger.debug(res['message'])

    # We now run a second iteration to find the simplest solution

    # Just minimise the sum of all counts
    objf2 = float(0) * np.array(objf)
    objf2[:szLPx] += 1.0

    # Solve the same problem, but ensure that solution is at most alpha worse
    A2 = np.append(A, np.asarray([objf]), axis=0)
    b2 = np.append(b, [fval + alpha])

    logger.debug("Starting second linear programming.")
    res = linprog(objf2, A2, b2, Aeq, beq, options=options)
    sol2 = res['x']
    exitflag = res['success']
    fval2 = res['fun']

    logger.debug(f"Done exitflag: {exitflag}")
    logger.debug(f"fval diff: {fval2 - fval}")

    if not exitflag:
        logger.warning("Failed to find a solution for LP2")
        logger.warning(res['message'])
    else:
        logger.debug(res['message'])

    x = np.append(np.array(x), np.array(xLP))
    histx = np.append(np.array(histx), sol2)

    ind = np.argsort(x)
    x = x.take(ind)
    histx = histx.take(ind)

    ind = histx > 0
    x = np.extract(ind, x)
    histx = np.extract(ind, histx)

    return histx, x


def to_native_type(data):
    """
    Converts data from numpy types to python types.
    This is important because numpy integers for example are limited to 64 bits, which may cause overflow errors when
    dealing with very big numbers.

    :param data: type
    :return: data cast to python type
    """
    if type(data).__module__ == np.__name__:
        return data.item()
    else:
        return data


def get_jackk_coeffs(f, m):
    """
    Calculates the jackknife coefficients.
    We only calculate the coefficients for up to the 10th order estimator.

    :param f: a fingerprint
    :param m: number of search queries issued
    :return: a list l s.t. l[x][y] contains the yth coefficient for the estimator of order x for all x, y.
    """
    mp.dps = 50
    d = to_native_type(sum(f))

    m = to_native_type(m)

    coeffs = []

    for k in range(1, 11):
        cur_coeff = [0] * 11
        cur_coeff[0] = m ** k * d

        for j in range(1, k + 1):
            cur_coeff[0] += (-1) ** j * (binomial(k, j)) * (m - j) ** k * d

            for t in range(1, j + 1):
                # if m is very small, we might into div by 0 here
                try:
                    cur_coeff[t] -= (-1) ** j * (binomial(k, j)) * (m - j) ** k * (binomial(m - t, j - t)) \
                                    / (binomial(m, j))
                except:
                    logger.warning("Division by zero in get_jack_coeffs")

        for l in range(k + 1):
            cur_coeff[l] /= factorial(k)

        cur_coeff = list(map(float, cur_coeff))
        coeffs.append(cur_coeff)

    return coeffs


def jackknife_selftune(fingerprint, m):
    """
    Support size estimator as used by KPT19.

    :param fingerprint: A fingerprint, f[0] should be set to 0.
    :param m: The total number of search queries issued.
    :return: Estimated support size
    """
    d = sum(fingerprint)
    if d == 1:
        return 1
    logger.debug(f"d: {d}")
    coeffs = get_jackk_coeffs(fingerprint, m)

    if len(fingerprint) > 11:
        fingerprint = np.array(fingerprint[:11])
    elif len(fingerprint) < 11:
        fingerprint = np.array(list(fingerprint) + [0] * (11 - len(fingerprint)))

    for i in range(9):
        a0 = np.array(coeffs[i])
        a1 = np.array(coeffs[i + 1])

        b = a1 - a0
        dN = sum(np.array(fingerprint) * b)

        var = d / (d - 1) * (sum(b * b * np.array(fingerprint)) - dN * dN / d)
        T = dN / sqrt(abs(var))

        N = sum(a0 * np.array(fingerprint)) + d
        P = 2 * (1 - norm.cdf(abs(T)))

        var_n = sum((a0 + 1) * (a0 + 1) * np.array(fingerprint)) - N
        logger.debug(f"N_{i}: {N}, se: {sqrt(abs(var_n))}")
        logger.debug(f"Hypothesis {i}: T: {T}, P: {P}")

        if P > 0.1:
            logger.debug(f"Hypothesis {i}: Accepted")
            return N

    logger.warning("Could not accept any hypothesis")

    return N


def modular_estimator(tokens):
    """
    An estimator which uses either unseen of jackknife based on a small test.

    :param tokens: the query tokens
    :return: the estimated support size
    """
    logger.debug("Modular estimator called")
    fp = get_fingerprint(tokens)
    hist, _ = unseen(fp)
    Nv = sum(hist)
    c = len(tokens) - len(set(tokens))
    e = 1 / Nv

    if c / max(binomial(len(tokens), 2), 1) <= (1 + 2 * e ** 2) / Nv:
        logger.debug("Using result from unseen")
        return Nv
    else:
        logger.debug("Using result from jackknife")
        return jackknife_selftune(fp, len(tokens))
