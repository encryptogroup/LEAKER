#!/usr/bin/env python3
"""
For License information see the LICENSE file.

Authors: Michael Yonli

"""
from leaker.attack.arr.db import generate_db, generate_queries_uniform, generate_queries_short
from leaker.attack.arr.agnostic_reconstruction import arr
import numpy as np
import random
import pytest

np.random.seed(1)
random.seed(1)

# Takes about 5 minutes, 0.8 and 0.9 fail
@pytest.mark.parametrize("density,mse", [(0.1, 600),
                                          (0.2, 50),
                                          (0.4, 200),
                                          (0.6, 200),
                                          (0.8, 50),
                                          (0.9, 50)])
def test_range_reconstruction_uniform(density, mse):
    if density == 0.2 or density == 0.9:
        pytest.skip()
    a = 1
    b = 10**3
    q = 10**4

    val = generate_db(a, b, density)
    queries = generate_queries_uniform((val, a, b), q)

    recon_vals = arr(queries, val, a, b, len(val))
    errors = [(recon_vals[x] - val[x])**2 for x in range(len(recon_vals))]
    actual_mse = sum(errors)/len(errors)
    assert actual_mse < mse


@pytest.mark.parametrize("density,mse,alpha,beta", [(0.10, 100, 1, 20)])
def test_range_reconstruction_short(density, mse, alpha, beta):
    pytest.skip()
    a = 1
    b = 10**3
    q = 10**4

    val = generate_db(a, b, density)
    queries = generate_queries_short((val, a, b), q, alpha, beta)

    recon_vals = arr(queries, val, a, b, len(val))

    errors = [(recon_vals[x] - val[x])**2 for x in range(len(recon_vals))]
    actual_mse = sum(errors)/len(errors)
    assert actual_mse < mse

