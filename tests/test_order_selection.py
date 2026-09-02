"""Tests for E50 (order-element selection for every D)."""
from gmpy2 import is_prime
from sympy import n_order

from factorlab.experiments.order_selection import (
    common_divisor_moduli,
    order_le_D,
    recover_from_common_divisor,
    select_order_element,
)

# L = 10159 (prime, = 1 mod 3): p = 1 + 4L = 40637 and q = 1 + 12L = 121909 are prime, 2 is a cubic residue mod q, so
# ord_p(2) = ord_q(2) = 4L = 40636 and alpha = 2 fails; 4L exceeds N^{1/3} (N = 4954016033, N^{1/3} = 1704), so the recovery fires.
L0, P0, Q0 = 10159, 40637, 121909


def test_order_le_D_exact():
    N = 1000003 * 1000033
    for alpha in (2, 3, 5, 7):
        for D in (50, 5000):
            e = order_le_D(alpha, N, D)
            true = n_order(alpha, N)
            assert (e == true) if true <= D else (e is None)


def test_recovery_from_common_divisor():
    assert is_prime(P0) and is_prime(Q0)
    assert recover_from_common_divisor(P0 * Q0, 4 * L0) == (P0, Q0)   # p = 1 + 1*(4L), q = 1 + 3*(4L)
    assert recover_from_common_divisor(91, 6) == (7, 13)              # 7 = 1 + 6, 13 = 1 + 2*6
    assert recover_from_common_divisor(1000003 * 1000033, 7) is None  # a wrong L returns None, never a bogus factorisation


def test_adversarial_two_fails_three_succeeds():
    # 2^47 - 1 = 2351 * 4513 * 13264529: ord_2351(2) = ord_4513(2) = 47, so alpha = 2 fails (L = 47 < N^{1/3}) and 3 succeeds.
    N = 2351 * 4513
    r = select_order_element(N, 1000)
    assert r["trace"][0] == (2, "fail", 47)
    assert r["outcome"] in ("large order", "factor") and r["alpha"] == 3


def test_divisor_gcd_factors_when_orders_differ():
    # p = 7, q = 13: ord_7(2) = 3, ord_13(2) = 12 -> ord_91(2) = 12 <= D; the prime divisor r = 2 or 3 of 12 gives a proper gcd.
    r = select_order_element(91, 20, threshold=10 ** 9)
    assert r["outcome"] == "factor" and r["factor"] in (7, 13)


def test_fail_then_recover_path():
    N = P0 * Q0
    assert n_order(2, P0) == n_order(2, Q0) == 4 * L0
    r = select_order_element(N, N)
    assert r["trace"][0] == (2, "fail", 4 * L0)
    assert r["trace"][1] == (2, "recover", 4 * L0)
    assert r["outcome"] == "factor" and r["factor"] == P0
    # the deterministic construction reproduces this modulus first
    assert common_divisor_moduli(1)[0][:3] == (L0, P0, Q0)


def test_skip_path():
    # Seed the lcm with L = 4 L0 (a known common divisor of p - 1 and q - 1) and disable the recovery threshold: 2 lies in the
    # subgroup {x^L = 1} modulo N and is skipped; 3 has ord_p(3) = 40636 != ord_q(3) = 30477, so the prime divisor 2 of
    # ord_N(3) = 121908 gives the proper gcd q.
    N = P0 * Q0
    r = select_order_element(N, N, threshold=10 ** 30, initial_L=4 * L0)
    assert r["trace"] == [(2, "skip", 4 * L0), (3, "divisor gcd", (121908, 2, Q0))]
    assert r["outcome"] == "factor" and r["factor"] == Q0 and r["skipped"] == 1
