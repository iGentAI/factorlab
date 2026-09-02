import math

import gmpy2
from sympy import isprime, primerange

from factorlab.experiments.average_case_pm1 import (
    THETA_DEFAULT, THETA_FRIEDLANDER, average_case_point, largest_prime_factor, pm1_deterministic,
    stage1_bound, stage1_exponent,
)


def test_stage1_exponent_contains_every_smooth_shifted_prime():
    N = 10 ** 6
    B = 20
    E = stage1_exponent(N, B)
    # every prime m <= 10^4 with P+(m-1) <= 20 has m-1 | E; a non-smooth one does not
    for m in primerange(3, 10 ** 4):
        if largest_prime_factor(m - 1) <= B:
            assert E % (m - 1) == 0
    assert E % (10007 - 1) != 0 or largest_prime_factor(10006) <= B
    # the exponent of 2 is floor(log N / log 2) exactly
    assert E % 2 ** 19 == 0 and E % 2 ** 20 != 0


def test_theta_constants():
    assert abs(THETA_FRIEDLANDER - 0.303265) < 1e-6
    assert THETA_DEFAULT > THETA_FRIEDLANDER          # Proposition PP needs a strict inequality
    assert THETA_DEFAULT < 0.5


def test_pm1_returns_smooth_side_only():
    # p - 1 = 2^4 * 3^2 * 5 * 7 = 5040 -> p = 5041? not prime; search a smooth p and a rough q
    p = None
    for cand in range(5000, 20000):
        if isprime(cand) and largest_prime_factor(cand - 1) <= 7:
            p = cand
    assert p is not None
    # q with q - 1 having a prime factor above B (q ~ p)
    q = None
    for cand in range(p + 2, 4 * p):
        if isprime(cand) and largest_prime_factor(cand - 1) > 200 and cand != p:
            q = cand
            break
    N = p * q
    # choose theta so that B >= 7 but B < P+(q-1): B = ceil(N^{theta/2})
    theta = 2 * math.log(50) / math.log(N)
    B, g = pm1_deterministic(N, theta)
    assert 7 <= B < largest_prime_factor(q - 1)
    assert g == p


def test_bookkeeping_consistency_small_run():
    row = average_case_point(32, 60, THETA_DEFAULT, seed=3)
    assert row["success"] + row["two_sided"] + row["fail"] == 60
    # success is one-sided smoothness plus the rare order anomaly on the rough side
    # (an anomaly turns a would-be success into a two-sided outcome, or a failure into a success)
    assert abs(row["success"] - row["one_sided_smooth"]) <= row["order_anomaly"]
    assert 0.0 <= row["dickman_prediction"] <= 1.0
    phat = row["success"] / 60
    assert abs(row["success_se"] - math.sqrt(phat * (1 - phat) / 60)) < 1e-12
    assert row["B_min"] >= stage1_bound(2 ** 31, THETA_DEFAULT)
