"""Tests for factorlab.experiments.common_factor_curve."""
import math

from gmpy2 import gcd, is_prime, powmod

from factorlab.experiments.common_factor_curve import (fermat_progression_search, moduli_with_common_factor, progression, route_b,
                                                       route_c_trial)


def test_progression_contains_p_plus_q_and_the_fermat_congruence_holds():
    for row in moduli_with_common_factor(36, 0.15, 4, seed=3):
        N, p, q, g = row["N"], row["p"], row["q"], row["g"]
        r, t_lo, t_hi = progression(N, g)
        S = p + q
        assert (S - r) % (g * g) == 0
        t = (S - r) // (g * g)
        assert t_lo <= t <= t_hi
        for a in (2, 3, 5):
            if gcd(a, N) == 1:
                assert powmod(a, S, N) == powmod(a, N + 1, N)


def test_route_b_factors_and_uses_about_sqrt_T_work():
    for row in moduli_with_common_factor(40, 0.12, 5, seed=5):
        N, p, q, g = row["N"], row["p"], row["q"], row["g"]
        res = route_b(N, g)
        assert res["factor"] in (p, q)
        assert res["work"] <= 3 * math.sqrt(res["T"]) + 3


def test_route_c_factors_within_T_trials():
    for row in moduli_with_common_factor(36, 0.2, 3, seed=7):
        N, p, q, g = row["N"], row["p"], row["q"], row["g"]
        res = route_c_trial(N, g)
        assert res["factor"] in (p, q) and res["work"] <= res["T"]


def test_baby_collision_is_reported_not_guessed():
    # a base whose g^2-th power has tiny order modulo N: alpha = N - 1 (order 2) gives beta = 1 when g^2 is even
    row = moduli_with_common_factor(36, 0.15, 1, seed=11)[0]
    N, g = row["N"], row["g"]
    res = fermat_progression_search(N, g, alpha=N - 1)
    assert res["baby_collision"] and res["factor"] is None and res["work"] == 2
    # a base that is 0 modulo N is unusable, not a factor
    res0 = fermat_progression_search(N, g, alpha=N)
    assert res0["factor"] is None and not res0["usable"]
    # route_b moves on and reports cumulative work
    rb = route_b(N, g, bases=(N - 1, 2, 3))
    assert rb["factor"] in (row["p"], row["q"]) and rb["work"] >= 2 + 1


def test_progression_bound_is_exact_for_large_moduli():
    # the upper bound must not depend on float precision: a 400-bit modulus with a small common factor
    import random
    rng = random.Random(1)
    g = 6
    while True:
        x = rng.getrandbits(197) | (1 << 196)
        p = 1 + g * x
        if is_prime(p):
            break
    while True:
        y = rng.getrandbits(197) | (1 << 196)
        q = 1 + g * y
        if is_prime(q) and q != p and max(p, q) < 2 * min(p, q):
            break
    N = p * q
    r, t_lo, t_hi = progression(N, g)
    t = (p + q - r) // (g * g)
    assert (p + q - r) % (g * g) == 0 and t_lo <= t <= t_hi


def test_progression_bound_holds_for_a_balance_ratio_near_one():
    # C = 1 + 1e-7 as a float: its nearest rational with denominator <= 10^6 is exactly 1, which would collapse the range to the
    # neighbourhood of 2 sqrt N and exclude p + q; the exact rational value of the float keeps it.  Primes p = 1 + g x, q = 1 + g (x + delta)
    # with p about 2^60 and g delta about 6e10 have q/p < 1 + 1e-7 and p + q - 2 sqrt N of order (g delta)^2 / (4p), in the hundreds.
    from fractions import Fraction
    from gmpy2 import isqrt
    import random
    rng = random.Random(4)
    g, C = 6, 1.0000001
    assert Fraction(C).limit_denominator(10 ** 6) == 1          # the approximation the code must not use
    while True:
        x = rng.getrandbits(58) | (1 << 57)
        p = 1 + g * x
        if not is_prime(p):
            continue
        y = x + (10 ** 10 + rng.getrandbits(20))
        q = 1 + g * y
        if is_prime(q) and q < C * p:
            break
    N = p * q
    assert p + q - 2 * isqrt(N) > 100                             # far enough above 2 sqrt N for the collapsed range to miss it
    r, t_lo, t_hi = progression(N, g, C=C)
    t = (p + q - r) // (g * g)
    assert t_lo <= t <= t_hi
    S_hi_collapsed = isqrt(4 * N) + 1                            # the upper bound the collapsed C = 1 would give
    assert (S_hi_collapsed - r) // (g * g) < t                     # which would exclude the true candidate


def test_any_common_divisor_works_not_only_the_gcd():
    row = moduli_with_common_factor(40, 0.2, 1, seed=17)[0]
    N, p, q, g = row["N"], row["p"], row["q"], row["g"]
    g2 = g // 2 if g % 4 == 0 else 2         # a proper common divisor
    assert (p - 1) % g2 == 0 and (q - 1) % g2 == 0
    res = route_b(N, g2)
    assert res["factor"] in (p, q)
