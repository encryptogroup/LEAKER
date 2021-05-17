"""
This file contains the implementation of ARR itself.

For License information see the LICENSE file.
Authors: Michael Yonli
"""
from itertools import repeat
from math import log
from multiprocessing import Pool

import numpy as np

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
    sizes = np.zeros((n + 1, n + 1)) + 1

    if processes == 1:
        for uniq in uniq_resp:
            i, j_, w, L = calc_weight_and_size(uniq, token_result_pairs, ordering, e)
            weights[i][j_] = w
            sizes[i][j_] = L
    else:
        with Pool(processes) as p:
            results = p.starmap(calc_weight_and_size, zip(uniq_resp, repeat(token_result_pairs), repeat(ordering),
                                                          repeat(e)))
            for i, j_, w, L in results:
                weights[i][j_] = w
                if np.isnan(L):
                    L = 1
                sizes[i][j_] = L

    n_p = n
    A_eq = np.zeros((n_p + 1, n_p + 1))
    b_eq = np.zeros(n_p + 1)

    for x_p in range(n_p + 1):
        for y_p in range(n_p + 1):
            if x_p > y_p:
                A_eq[x_p][x_p] += 2 * weights[y_p][x_p]
                A_eq[x_p][y_p] = 2 * weights[y_p][x_p]
                b_eq[x_p] += 2 * weights[y_p][x_p] * log(sizes[y_p][x_p], 2)
            elif y_p > x_p:
                A_eq[x_p][y_p] = 2 * weights[x_p][y_p]
                A_eq[x_p][x_p] += 2 * weights[x_p][y_p]
                b_eq[x_p] += 2 * weights[x_p][y_p] * log(sizes[x_p][y_p], 2)

    res = np.linalg.lstsq(A_eq, b_eq)
    x_opt = res[0]

    lengths = [2 ** x for x in x_opt]
    scale_f = sum(lengths) / (b - a + 1)
    lengths = [x / scale_f for x in lengths]

    vals = np.zeros(n)
    last = a - 1
    for y in range(n_p):
        last = last + lengths[y]
        vals[y] = last

    return vals


def general_arr(token_result_pairs, ordering, a, b, n, processes=1, e=0.01, minw=1):
    """
    This attack extends arr to handle databases with repeating values.

    :param token_result_pairs: list of tuples [(t0, r0), (t1, r1)...] t = search token, r = record tokens
    :param ordering: list of all possible ids s.t. the index is its order
    :param a: min element of db
    :param b: max element of db
    :param n: number of db entries
    :param processes: the number of processes to use (all available CPUs if None)
    :param e: minimum ARR weight
    :param minw: minimum repetition identification weight
    :return: values of db
    """
    assert n == len(ordering)

    responses = [y for (_, y) in token_result_pairs]
    uniq_resp = set(responses)

    reduction_weight = dict()

    # contain indices of the ordering
    minima = set()
    maxima = set()

    for uniq in uniq_resp:
        # in case no values were returned
        if len(uniq) == 0:
            continue
        count = responses.count(uniq)
        for record_token in uniq:
            if record_token in reduction_weight:
                reduction_weight[record_token] += count
            else:
                reduction_weight[record_token] = count

        idx_list = np.array([ordering.index(token) for token in uniq])
        minima.add(idx_list.min())
        maxima.add(idx_list.max())

    big_n = set(range(len(ordering)))
    never_min = big_n - minima  # elements that are never the minimum
    never_max = big_n - maxima  # elements that are never the maximum

    cand_list = []
    # find continuous lists
    # never_max indicates start of repetition, which is stopped if not in never_min
    never_max = sorted(list(never_max))

    starting_points = [x for x in never_max if x not in never_min]

    cur_list = starting_points[:1]
    if starting_points:
        i = never_max.index(starting_points[0]) + 1
    else:
        i = len(never_max)

    while i < len(never_max):
        # check if cur_list can be extended
        if never_max[i] - cur_list[-1] == 1 and never_max[i] in never_min:
            cur_list.append(never_max[i])
            i += 1
            continue

        # else check if cur_list can be ended
        if never_max[i - 1] + 1 in never_min:
            cur_list.append(never_max[i - 1] + 1)
            cand_list.append(cur_list)

        # find new start of cur_list
        if i < len(never_max):
            cur_list = [never_max[i]]
        i += 1

    # see if we can end cur_list
    if cur_list and never_max and never_max[i - 1] + 1 in never_min:
        cur_list.append(never_max[i - 1] + 1)
        cand_list.append(cur_list)

    to_rem = dict()
    for subset in cand_list:
        weight = 0
        if ordering[subset[0]] in reduction_weight:
            weight = reduction_weight[ordering[subset[0]]]
        if weight > minw:
            to_rem[subset[0]] = subset[-1] - subset[0]

    n_p = n - sum(to_rem.values())  # n without repetitions

    new_order = []
    i = 0
    while len(new_order) < n_p:
        elem = ordering[i]
        new_order.append(elem)

        if i in to_rem:
            i += to_rem[i]
        i += 1

    # run on DB without repetitions
    u_vals = arr(token_result_pairs, new_order, a, b, n_p, processes, e)

    vals = np.zeros(n)

    u_i = 0
    for i, v in enumerate(u_vals):
        idx = i + u_i

        if idx in to_rem:
            vals[idx:idx + to_rem[idx] + 1] = v
            u_i += to_rem[idx]
        else:
            vals[idx] = v

    assert i + u_i + 1 == n

    perm = np.argsort(ordering)
    return vals.take(perm), to_rem
