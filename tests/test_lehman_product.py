"""Tests for E47 (Lehman product, localised family, Wronskian constants)."""
import math
from fractions import Fraction

import numpy as np
from gmpy2 import iroot, mpz

from factorlab.experiments.lehman_product import (
    ceil_2sqrt,
    covering_cell,
    in_localised_family,
    lehman_product_stats,
    localised_family,
    standard_family,
    window_int,
    wronskian_matrix,
)
from factorlab.gen import make_semiprime


def test_window_int_is_exact_ceiling():
    N = 10 ** 12 + 39
    for r in (100, 1000):
        for ab in (1, 7, 50, 1000):
            W = window_int(N, r, ab)
            assert (4 * r * W) ** 2 * ab >= N and (4 * r * (W - 1)) ** 2 * ab < N
            assert abs(W - math.ceil(math.sqrt(N) / (4 * r * math.sqrt(ab)))) <= 1


def test_ceil_2sqrt_exact():
    for k, N in ((1, 25), (2, 13), (3, 1000003)):
        c = ceil_2sqrt(k, N)
        assert (c - 1) ** 2 < 4 * k * N <= c ** 2


def test_mirror_symmetry_makes_full_product_gcd_N():
    # Lemma (mirror symmetry): true hits and mirrored hits are equal in number, so gcd(Pi_r, N) = N; the half-family avoids mirrors.
    for i in range(4):
        sp = make_semiprime(28, "rsa", 21, i)
        N, p, q = int(sp.N), int(sp.p), int(sp.q)
        r = int(iroot(mpz(N), 3)[0])
        full = lehman_product_stats(N, p, q, r, 2)
        assert full["hits_p"] >= 1  # Lehman coverage
        true = mirror = 0
        for a, b, c, W in standard_family(N, r):
            u, um = a * q + b * p, a * p + b * q
            if c <= u <= c + W - 1 and u != um:
                true += 1
            if c <= um <= c + W - 1 and u != um:
                mirror += 1
        assert true == mirror
        if true > 0:
            assert full["gcd_kind"] == "N"


def test_localised_predicate_exact_and_family_covers():
    sp = make_semiprime(36, "rsa", 4, 0)
    N, p, q = int(sp.N), int(sp.p), int(sp.q)
    L = int(N ** 0.47)
    r = math.ceil((N / L) ** 0.4)
    I_lo, I_hi = Fraction(p - L // 3), Fraction(p - L // 3 + L)
    fam = localised_family(N, I_lo, I_hi, r)
    cov = covering_cell(N, p, q, r)
    assert cov is not None and any((a, b) == cov for a, b, _ in fam)
    R_lo, R_hi = I_lo * I_lo / N, I_hi * I_hi / N
    assert all(in_localised_family(a, b, R_lo, R_hi, r) for a, b, _ in fam)
    # strict boundary: with r = 49 (sqrt r = 7) and b = 7 the tolerance is 2/49; a/b = 3/7 sits at distance exactly 2/49
    # below x_lo = 3/7 + 2/49 = 23/49, so it must be excluded, while moving x_lo down by 1/1000 admits it.
    x_lo = Fraction(23, 49)
    assert in_localised_family(3, 7, x_lo, x_lo, 49) is False
    assert in_localised_family(3, 7, x_lo - Fraction(1, 1000), x_lo, 49) is True
    assert in_localised_family(1, 3, Fraction(1, 3), Fraction(1, 2), 49) is True  # inside the interval


def test_wronskian_matrix_invertible_on_interval():
    for M in (1, 2, 3):
        for t in np.linspace(2 ** -0.5, 1.0, 7):
            s = np.linalg.svd(wronskian_matrix(t, M), compute_uv=False)
            assert s[-1] > 1e-6
