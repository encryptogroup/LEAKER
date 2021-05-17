"""
For License information see the LICENSE file.

Authors: Michael Yonli

"""
from leaker.attack import ShortRangeQuerySpace, ValueCenteredRangeQuerySpace, BoundedRangeQuerySpace,\
    UniformRangeQuerySpace
from leaker.api import RandomRangeDatabase, RangeDatabase
import numpy as np
import pytest
import matplotlib.pyplot as plt


# Almost instant
def test_smoke_uniform():
    db = RandomRangeDatabase("test", 1, 10**3, 1, 10**3, False)

    qsp = UniformRangeQuerySpace(db, 10 ** 4, False, False)

    queries = [q for q in next(qsp.select(10**4))]


# 10 seconds
def test_shortrange():
    db = RandomRangeDatabase("test", 1, 10**3, 1, 10**3, False)

    expected = [142, 90, 23]  # Data comes from KPT20 around Fig. 10
    for idx, beta in enumerate([3, 5, 20]):
        qsp = ShortRangeQuerySpace(db, 10**4, False, True, alpha=1, beta=beta)
        queries = [q for q in next(qsp.select(10**4))]
        sum_diff = sum(map(lambda x: x[1] - x[0], queries))
        mean = sum_diff / len(queries)

        result_diff = abs(expected[idx] - mean)
        assert result_diff / expected[idx] < 0.20


# 5 seconds
def test_smoke_valuecentered():
    db = RandomRangeDatabase("test", 1, 10**3, 1, 10**3, False)

    qsp = ValueCenteredRangeQuerySpace(db, 10 ** 4, False, False, alpha=1, beta=3)

    queries = [q for q in next(qsp.select(10**4))]


# Takes about a minute
def test_smoke_valuecentred_sparse():
    db = RandomRangeDatabase("test", 1, 10**3, 0.5, int(10**3/2), False)

    qsp = ValueCenteredRangeQuerySpace(db, 10 ** 4, False, False, alpha=1, beta=3)

    queries = [q for q in next(qsp.select(10**4))]


# Takes about a minute
def test_smoke_valuecentered_repeated():
    db = RandomRangeDatabase("test", 1, 10**3, length=2*10**3, allow_repetition=True)

    qsp = ValueCenteredRangeQuerySpace(db, 10 ** 4, False, False, alpha=1, beta=3)

    queries = [q for q in next(qsp.select(10**4))]


@pytest.mark.skip("Test should only be run manually")
def test_valuecentred_graphic():
    # Run test and compare graphic with graphic from Fig 10 of [KPT20]
    db = RandomRangeDatabase("test", 1, 50, density=1, allow_repetition=False)

    qsp = ValueCenteredRangeQuerySpace(db, 10 ** 6, True, True, alpha=1, beta=3)

    queries = [q for q in next(qsp.select(10**4))]

    res = np.zeros((51, 51))

    for x in queries:
        res[50-x[0], x[1]] += 1

    fig, ax = plt.subplots()
    im = ax.imshow(res)
    ticks = list(range(51))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_yticklabels(reversed(ticks))
    plt.show()


# Almost instant
def test_bounded_rangequery_space():
    db = RangeDatabase("test", list(range(10)))
    qsp = BoundedRangeQuerySpace(db, bound=1, allow_empty=False, allow_repetition=False)
    queries = next(qsp.select())

    assert len(queries) == 10
    for x in queries:
        assert x[0] == x[1]

    qsp2 = BoundedRangeQuerySpace(db, allow_empty=False, allow_repetition=False)
    queries2 = next(qsp2.select())
    count = 0
    for lower_bound in range(1, 11):
        for upper_bound in range(lower_bound, min(11, lower_bound + 3)):
            assert (lower_bound, upper_bound) in queries2
            count += 1

    assert count == len(queries2)
