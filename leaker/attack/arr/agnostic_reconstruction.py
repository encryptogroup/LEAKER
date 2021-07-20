"""
This file contains the implementation of ARR itself.

For License information see the LICENSE file.
Authors: Michael Yonli
"""
from itertools import repeat
from multiprocessing import Pool

import numba
import numpy as np
from scipy.optimize import least_squares

from .estimators import modular_estimator


def calc_weight_and_size(uniq, token_result_pairs, ordering, e: float):
    """
    A helper function used by Arr. It calculates the weight and sizes as used by arr.

    :param uniq: the unique response to consider
    :param token_result_pairs: all token result pairs
    :param ordering: the ordering of the db
    :param e: the minimum weight
    :return: A tuple (i, j + 1, w, L), i and j + 1  are indices, w ist the weight and L is the size
    """
    # needed in case generalARR has been called before
    cleaned_list = np.array([ordering.index(token) for token in uniq if token in ordering])

    i = cleaned_list.min()
    j = cleaned_list.max()

    D = [y for (y, x) in token_result_pairs if x == uniq]
    w = max(e, len(D) ** 2)
    L = modular_estimator(D)

    return i, j + 1, w, L


def arr(token_result_pairs, ordering, a, b, n, processes=1, e=0.01):
    """
    The implementation of ARR as in KPT19.

    :param token_result_pairs:
    :param ordering:
    :param a: minimum of db
    :param b: maximum of db
    :param n: size of database
    :param processes: number of processes
    :param e: minimum weight
    :return: reconstructed database values
    """
    assert n == len(ordering)

    responses = [y for (_, y) in token_result_pairs if len(y) > 0]
    uniq_resp = set(responses)

    weights = np.zeros((n + 1, n + 1)) + e
    sizes = np.zeros((n + 1, n + 1))

    if processes == 1:
        for uniq in uniq_resp:
            i, j_, w, L = calc_weight_and_size(uniq, token_result_pairs, ordering, e)
            if not np.isfinite(L):
                L = 0
            weights[i][j_] = w
            sizes[i][j_] = L
    else:
        with Pool(processes) as p:
            results = p.starmap(calc_weight_and_size, zip(uniq_resp, repeat(token_result_pairs), repeat(ordering),
                                                          repeat(e)))
            for i, j_, w, L in results:
                weights[i][j_] = w
                if not np.isfinite(L):
                    L = 0
                sizes[i][j_] = L

    @numba.njit
    def loss_function(x, weights, sizes, n):
        res = 0
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                res += weights[i][j] * (x[i] * x[j] - sizes[i][j]) ** 2
        return res

    res = least_squares(loss_function, x0=[1] * (n + 1), args=(weights, sizes, n), bounds=(0, np.inf), method='dogbox')
    x_opt = res.x

    scale_f = sum(x_opt) / (b - a + 1)
    lengths = [x / scale_f for x in x_opt]

    vals = np.zeros(n)
    last = a - 1
    for y in range(n):
        last = last + lengths[y]
        vals[y] = last

    perm = np.argsort(ordering)
    return vals.take(perm)
