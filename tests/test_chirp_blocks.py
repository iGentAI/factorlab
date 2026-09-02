"""Tests for E52 (block decomposition of multiplicative chirps)."""
from fractions import Fraction

from factorlab.experiments.chirp_blocks import (
    beatty_chirp_blocked,
    beatty_chirp_direct,
    beatty_pochhammer_blocked,
    beatty_pochhammer_direct,
    bivariate_P,
    chirp_blocked,
    chirp_direct,
    coefficient_count,
)

N = 1000003 * 1000033


def test_block_identity_quadratic_exponent():
    for (a, b, c) in ((3, 5, 7), (1, 0, 0), (2, -3, 11)):
        for n, a1 in ((24, 4), (36, 6), (60, 5)):
            assert chirp_blocked(2, N, a, b, c, n, a1) == chirp_direct(2, N, a, b, c, n)
    P = bivariate_P(2, N, 3, 5, 7, 8)
    assert max(s for s, t in P) == 8 and max(t for s, t in P) == 8 * 7 // 2
    assert len(P) <= coefficient_count(8)


def test_block_identity_with_beatty_rounding():
    rho = Fraction(6180339887, 10 ** 10)  # a generic rational slope (golden-ratio digits); k rho is never an integer for 0 < k < n
    for n, a1 in ((30, 5), (48, 6), (64, 8)):
        val, npolys = beatty_chirp_blocked(3, N, 2, 1, 5, rho, n, a1)
        assert val == beatty_chirp_direct(3, N, 2, 1, 5, rho, n)
        assert npolys <= a1 + 1
    # q need not be a unit modulo N: q = 1000003 divides N
    q = 1000003
    assert chirp_blocked(q, N, 3, 5, 7, 24, 4) == chirp_direct(q, N, 3, 5, 7, 24)
    val, _ = beatty_chirp_blocked(q, N, 2, 1, 5, rho, 30, 5)
    assert val == beatty_chirp_direct(q, N, 2, 1, 5, rho, 30)


def test_beatty_pochhammer_block_identity():
    rho = Fraction(6180339887, 10 ** 10)
    for M, a1 in ((40, 5), (63, 7), (96, 8)):
        assert beatty_pochhammer_blocked(5, 7, 11, N, rho, M, a1) == beatty_pochhammer_direct(5, 7, 11, N, rho, M)
    # z need not be a unit: z = 1000033 divides N
    assert beatty_pochhammer_blocked(5, 7, 1000033, N, rho, 40, 5) == beatty_pochhammer_direct(5, 7, 1000033, N, rho, 40)
