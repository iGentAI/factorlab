"""Tests for E51 (Harvey's search on a residue class)."""
import math

import pytest
from gmpy2 import mpz

from factorlab.experiments.harvey_residue import decode, harvey_residue_factor, parameters, window_int
from factorlab.experiments.order_selection import select_order_element
from factorlab.gen import make_semiprime


def _modulus(bits: int, index: int):
    sp = make_semiprime(bits, "rsa", 12, index)
    return int(sp.N), int(sp.p), int(sp.q)


def test_window_and_decode():
    N = 10 ** 12 + 39
    for r in (10, 100):
        for ab in (1, 6, 50):
            W = window_int(N, r, ab)
            assert (4 * r * W) ** 2 * ab >= N > (4 * r * (W - 1)) ** 2 * ab
    N, p, q = _modulus(36, 0)
    a, b = 1, 1
    assert decode(a * q + b * p, a * b, N) in (p, q)
    assert decode(a * q + b * p + 1, a * b, N) is None or decode(a * q + b * p + 1, a * b, N) in (p, q)


def test_plain_harvey_factors():
    N, p, q = _modulus(36, 1)
    r, m = parameters(N, 1)
    sel = select_order_element(N, m)
    assert sel["outcome"] == "large order"
    res = harvey_residue_factor(N, sel["alpha"], r, m)
    assert res["factor"] in (p, q)
    assert res["work"]["babies"] == m and res["work"]["giants"] > 0


def test_residue_restricted_harvey_factors_with_fewer_tests():
    N, p, q = _modulus(36, 2)
    M = 16
    r, m = parameters(N, M)
    sel = select_order_element(N, M * m)
    assert sel["outcome"] == "large order"
    res = harvey_residue_factor(N, sel["alpha"], r, m, M, p % M)
    assert res["factor"] in (p, q)
    r1, m1 = parameters(N, 1)
    sel1 = select_order_element(N, m1)
    res1 = harvey_residue_factor(N, sel1["alpha"], r1, m1)
    assert res["work"]["tested"] < res1["work"]["tested"] / 4  # about 1/M of the tested integers


def test_preconditions():
    N, p, q = _modulus(36, 3)
    with pytest.raises(ValueError):
        harvey_residue_factor(N, 2, 5, 5, 0, 0)
    # a residue sharing a divisor with M that also divides N returns that factor
    assert harvey_residue_factor(N, 2, 5, 5, p * 2, p)["factor"] == p


def test_common_factor_attack_on_constructed_moduli():
    import math

    from gmpy2 import is_prime, powmod
    from sympy import n_order, primerange

    from factorlab.experiments.harvey_residue import common_factor_attack, order_collision_factor

    # p = 1 + k g, q = 1 + l g with g = 2 * prime and gcd(k, l) = 1: gcd(N-1, p-1) = g and beta = alpha^{N-1} has coprime
    # component orders dividing k and l, so the collision search with bound about sqrt(N)/g finds a factor.
    found = 0
    for g0 in primerange(5000, 40000):
        g = 2 * g0
        for k, l in ((1, 3), (2, 5), (3, 4), (5, 6)):
            p, q = 1 + k * g, 1 + l * g
            if is_prime(p) and is_prime(q):
                N = p * q
                res = common_factor_attack(N)
                assert res["factor"] in (p, q), (N, res)
                alpha = res["alpha"]
                beta = int(powmod(alpha, N - 1, N))
                if beta % p != 1 and beta % q != 1:
                    e = min(n_order(beta % p, p), n_order(beta % q, q))
                    assert e <= min(k, l) and res["horizon"] >= e and res["how"] == "collision"
                    # boundary: a bound whose searched range J*h lies strictly below e finds nothing
                    D = e - 1
                    while D >= 1:
                        f, searched = order_collision_factor(N, beta, D)
                        if searched < e:
                            assert f is None, (D, searched, e)
                            break
                        D //= 2
                    f, searched = order_collision_factor(N, beta, e)
                    assert f in (p, q) and searched >= e
                found += 1
        if found >= 6:
            break
    assert found >= 6
