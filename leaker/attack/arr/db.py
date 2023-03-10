"""
For License information see the LICENSE file.

Authors: Michael Yonli

This module contains helper coder for testing ARR. It is only used for testing and not required for ARR to work.
TODO: merge this code with testing code
"""
from random import randint, random

from numpy.random import choice

from ...util import beta_intervals


def generate_db(a, b, density):
    n = b - a + 1
    n_vals = round(n * density)
    vals = set()
    while len(vals) < n_vals:
        vals.add(randint(a, b))

    return sorted(list(vals))


def generate_queries_uniform(db, q):
    vals, a, b = db
    queries = []

    for _ in range(q):
        l0 = randint(a, b)
        l1 = randint(a, b)

        lower = min(l0, l1)
        upper = max(l0, l1)
        b_code = upper * b + lower

        results = tuple(sorted(filter(lambda x: lower <= x <= upper, vals)))

        queries.append((b_code, results))

    return queries


def generate_queries_short(db, q, a_b, b_b):
    vals, a, b = db
    queries = []

    R = int((b - a + 1) * (b - a + 2) / 2)

    l_queries = R
    alpha = a_b
    beta = b_b
    samples = beta_intervals(alpha, beta, l_queries)
    noisy_samples = [sample * random() / l_queries for sample in samples]
    tot_prob = sum(noisy_samples)
    normalised_samples = [elem / tot_prob for elem in noisy_samples]

    l_b = u_b = a
    delta = 0

    while True:

        results = tuple(sorted(filter(lambda x: l_b <= x <= u_b, vals)))
        queries.append((u_b * b + l_b, results))

        if l_b + delta + 1 <= b:
            l_b += 1
            u_b = l_b + delta
        elif len(queries) < R:
            delta += 1
            l_b = a
            u_b = l_b + delta
        else:
            break

    assert len(queries) == R

    actual_queries = choice(len(queries), q, p=normalised_samples)

    return [queries[x] for x in actual_queries]
