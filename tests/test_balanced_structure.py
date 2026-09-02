"""Tests for E53 (additive structure of the balanced sub-family)."""
from math import gcd, sqrt

import numpy as np
from gmpy2 import isqrt, mpz

from factorlab.experiments.balanced_structure import (
    balanced_cells,
    ceil_2sqrt,
    cluster_spectrum,
    dmax,
    dmax_tol,
    energy_form_bound,
    equal_product_pairs,
    excision_census,
    lemma_d_bound,
    primitive_cells,
    resonant_cells,
    shell_cells,
    exponent_set,
    unique_product_cells,
    window,
)
from factorlab.gen import make_semiprime


def test_cells_and_windows():
    N = int(make_semiprime(30, "rsa", 5, 0).N)
    from gmpy2 import iroot

    r = int(iroot(mpz(N), 3)[0])
    cells = balanced_cells(N, r, 2)
    assert all(a <= b <= 2 * a and a * b <= r for a, b in cells)
    assert len(set(cells)) == len(cells)
    # every balanced pair with ab <= r is present
    brute = {(a, b) for a in range(1, r + 1) for b in range(a, 2 * a + 1) if a * b <= r}
    assert set(cells) == brute
    prim = primitive_cells(cells, r, r // 4)
    assert all(gcd(a, b) == 1 and a * b > r // 4 for a, b in prim)
    # windows: exact ceiling of sqrt N / (4 r sqrt(ab)), at least 1
    for a, b in ((1, 1), (3, 5), cells[-1]):
        W = window(N, r, a * b)
        assert W >= 1 and (W - 1) ** 2 * 16 * r * r * a * b < N <= W * W * 16 * r * r * a * b or W == 1
    assert shell_cells(10) == [(1, 6), (1, 7), (1, 8), (1, 9), (1, 10)]
    assert ceil_2sqrt(4) == 4 and ceil_2sqrt(5) == 5 and ceil_2sqrt(2) == 3


def test_cluster_statistics_exact_on_small_sets():
    Z = np.array([0, 1, 3, 4, 10, 11, 13], dtype=np.int64)
    # differences: 1 occurs for (0,1),(3,4),(10,11): 3 times; 3 occurs for (0,3),(1,4),(10,13): 3 times; 10 for (0,10),(1,11),(3,13): 3
    D, t = dmax(Z)
    assert D == 3 and t in (1, 3, 10)
    assert dmax_tol(Z, 1) == dmax(Z)
    # with tolerance W = 2 the differences {3, 4} (0->3, 0->4, 1->3, 1->4 ... ) cluster: count pairs with |d - t| < 2
    DW, tW = dmax_tol(Z, 2)
    diffs = sorted(int(b - a) for i, a in enumerate(Z) for b in Z[i + 1 :])
    best = max(sum(1 for d in diffs if abs(d - t) < 2) for t in range(min(diffs), max(diffs) + 1))
    assert DW == best
    assert abs(lemma_d_bound(100, 1, 1) - 2 * (10000 / 4) ** (1 / 3)) < 1e-9


def test_energy_form_bound_invariants():
    """The energy form never falls below the D_max form (Sigma_bar_m <= D_max), equals it when all clusters are equal, and is a
    minimum over every m (so a skewed certificate cannot make it invalid)."""
    n = 400
    # skewed spectrum: one large cluster, many small ones (each listed twice, as for h and -h)
    spec = np.sort(np.repeat(np.array([50] + [2] * 300 + [1] * 2000), 2))[::-1]
    K, m, sbar = energy_form_bound(n, 1, spec)
    D = int(spec[0])
    assert K >= lemma_d_bound(n, 1, D) - 1e-9
    assert 1 <= m <= len(spec) and sbar <= D
    # brute-force check of the minimisation over m
    cum = np.cumsum(spec, dtype=float)
    phi = [mm + (n ** 2 / (4 * cum[mm - 1] / mm)) ** (2 / 3) / mm for mm in range(1, len(spec) + 1)]
    assert abs(K - min(n / 2, min(phi))) < 1e-9
    # flat spectrum: the energy form equals the D_max form up to the discreteness of m (the continuous minimiser is m = sqrt(A),
    # with minimum value 2 sqrt(A))
    flat = np.full(1000, 3)
    K2, _, _ = energy_form_bound(n, 1, flat)
    assert lemma_d_bound(n, 1, 3) - 1e-9 <= K2 <= lemma_d_bound(n, 1, 3) + 1
    # spectrum of a small explicit set lists each positive difference cluster twice; only W = 1 is supported
    Z = np.array([0, 1, 3, 4, 10, 11, 13], dtype=np.int64)
    sp = cluster_spectrum(Z, 1)
    assert sp[0] == sp[1] == 3 and sp.sum() == 2 * (7 * 6 // 2)
    import pytest

    with pytest.raises(ValueError):
        cluster_spectrum(Z, 2)


def test_equal_product_lemma_and_unique_product_family():
    """Pairs (q lam, p(lam+m)) -> (q(lam+m), p lam) have equal products and start differences exactly m(qN - p); with
    (p, q, m) = (3, 2, 6) they are primitive for lam odd, 3 !| lam, so D_max of the primitive family is >= their number.  Keeping
    one cell per product removes them, and the unique-product family's maximiser is the half-offset line 2b = 3a + 1 with
    consecutive-b pairs (increment (u/sqrt6)(1 + 1/(72 a^2) + ...))."""
    from gmpy2 import iroot

    N = int(make_semiprime(44, "rsa", 5, 0).N)
    r = int(iroot(mpz(N), 3)[0])
    pairs = equal_product_pairs(3, 2, 6, r)
    assert len(pairs) >= 10
    Nm = mpz(N)
    E = lambda a, b: int(a * Nm + b - ceil_2sqrt(a * b * Nm))
    for (a, b), (a2, b2) in pairs:
        assert a * b == a2 * b2 and a2 - a == 12 and b2 - b == -18
        assert E(a2, b2) - E(a, b) == 12 * N - 18
        lam = a // 2
        assert lam % 2 == 1 and lam % 3 != 0
    prim = primitive_cells(balanced_cells(N, r, 2), r)
    S = exponent_set(N, prim, r, with_windows=False)
    D, t = dmax(S)
    assert D >= len(pairs) and t == 12 * N - 18
    # unique products: the equal-product structure is gone and the maximiser is the (3, 2) half-offset family
    uniq = unique_product_cells(prim)
    assert len(set(a * b for a, b in uniq)) == len(uniq)
    Su = exponent_set(N, uniq, r, with_windows=False)
    Du, tu = dmax(Su)
    assert Du < D and Du <= 12
    Eu = {E(a, b): (a, b) for a, b in uniq}
    hits = [(Eu[v], Eu[v + tu]) for v in Eu if v + tu in Eu]
    assert len(hits) == Du
    for (a, b), (a2, b2) in hits:
        assert a2 == a and b2 == b - 1 and 2 * b - 3 * a == 1
    # the half-offset increment: u sqrt(a)/(sqrt b + sqrt(b-1)) = (u/sqrt6)(1 + 1/(72 a^2) + O(a^-4)) on 2b = 3a + 1
    from math import sqrt

    for a in (77, 101, 201):
        b = (3 * a + 1) // 2
        lhs = sqrt(a) / (sqrt(b) + sqrt(b - 1))
        rhs = (1 + 1 / (72 * a * a)) / sqrt(6)
        assert abs(lhs - rhs) < 1e-3 / a ** 4


def test_half_offset_families_lemma():
    """Lemma (half-offset families): for m = 1 (mod 4), k = (m+1)/2, d = m-1, alpha = (m-1)/4, beta = (m-3)/2, the cells
    P = (k lam + alpha, d lam + beta) and P' = (k lam + alpha + 1, d lam + beta + 2) satisfy k beta - d alpha = -1 and
    k(beta+2) - d(alpha+1) = +1 (hence both primitive), n(lam) = kd (lam + s)^2 - 1/(4kd) with s = (k beta + d alpha)/(2kd), and the
    twist increment u(sqrt n' - sqrt n) = (u m/sqrt(kd)) (1 + 1/(8 (kd)^2 (lam+s)(lam+s')) + O(lam^-3)).  Over the top half of a
    family the increment varies by V = (u m/sqrt(kd)) (1/(8 (kd)^2)) (1/lam1^2 - 1/lam_max^2), so the rounded differences take at
    most V + 3 values and some difference occurs >= count/(V + 3) times (checked at 48 bits for m = 13, 17, 21)."""
    from fractions import Fraction
    from gmpy2 import iroot

    for m in (5, 9, 13, 17, 21, 25, 29):
        k, d = (m + 1) // 2, m - 1
        al, be = (m - 1) // 4, (m - 3) // 2
        assert gcd(k, d) == 1 and k * be - d * al == -1 and k * (be + 2) - d * (al + 1) == 1
        s = Fraction(k * be + d * al, 2 * k * d)
        for lam in (1, 7, 30):
            n = (k * lam + al) * (d * lam + be)
            assert Fraction(n) == k * d * (lam + s) ** 2 - Fraction(1, 4 * k * d)
            assert gcd(k * lam + al, d * lam + be) == 1 and gcd(k * lam + al + 1, d * lam + be + 2) == 1
    # increment expansion check in floating point at large lam
    m = 9
    k, d, al, be = 5, 8, 2, 3
    s = (k * be + d * al) / (2 * k * d)
    s2 = s + m / (k * d)
    for lam in (50, 200):
        n = (k * lam + al) * (d * lam + be)
        n2 = (k * lam + al + 1) * (d * lam + be + 2)
        inc = sqrt(n2) - sqrt(n)
        pred = (m / sqrt(k * d)) * (1 + 1 / (8 * (k * d) ** 2 * (lam + s) * (lam + s2)))
        assert abs(inc - pred) < 1e-2 / lam ** 3
    # coherence at 48 bits: over the top half of the family the increment varies by V = (u m/sqrt(kd)) (1/(8 (kd)^2)) (1/lam1^2 -
    # 1/lam_max^2), so the rounded differences take at most V + 3 values (the rounding adds at most two)
    N = int(make_semiprime(48, "rsa", 5, 0).N)
    r = int(iroot(mpz(N), 3)[0])
    Nm = mpz(N)
    u = 2 * sqrt(N)
    E = lambda a, b: int(a * Nm + b - ceil_2sqrt(a * b * Nm))
    for m in (13, 17, 21):
        k, d, al, be = (m + 1) // 2, m - 1, (m - 1) // 4, (m - 3) // 2
        fam = []
        lam = 1
        while (k * lam + al + 1) * (d * lam + be + 2) <= r:
            fam.append(lam)
            lam += 1
        lam1, lam_max = max(1, fam[-1] // 2), fam[-1]
        top = [l for l in fam if l >= lam1]
        V = (u * m / sqrt(k * d)) * (1 / (8 * (k * d) ** 2)) * (1 / lam1 ** 2 - 1 / lam_max ** 2)
        diffs = {E(k * l + al + 1, d * l + be + 2) - E(k * l + al, d * l + be) for l in top}
        assert len(top) >= 6 and len(diffs) <= V + 3
        # and the largest cluster within the top half is at least len(top)/(V + 3)
        from collections import Counter

        cnt = Counter(E(k * l + al + 1, d * l + be + 2) - E(k * l + al, d * l + be) for l in top)
        assert max(cnt.values()) >= len(top) / (V + 3)


def test_resonant_cells_and_excision():
    """resonant_cells marks the (2,3)/(0,-1) half-offset family (window 4kd/(Delta^2 |X|) = 12) and the ((m+1)/2, m-1)/(1,2) families
    (window 4kd/(2m)); excising every family with window >= 8 at 42 bits removes a minority of the unique-product cells and leaves a
    statistic <= 8 that does not exceed the unexcised one."""
    from gmpy2 import iroot

    N = int(make_semiprime(42, "rsa", 5, 0).N)
    r = int(iroot(mpz(N), 3)[0])
    marked = resonant_cells(r, 8)
    lam = 38
    assert marked[(2 * lam + 1, 3 * lam + 2)] >= 12 - 1e-9 and marked[(2 * lam + 1, 3 * lam + 1)] >= 12 - 1e-9
    m = 17
    k, d, al, be = 9, 16, 4, 7
    assert marked[(k * 5 + al, d * 5 + be)] >= 4 * k * d / (2 * m) - 1e-9
    res = excision_census(N, Ms=(8,))
    row = res["rows"][0]
    assert 0.1 < row["removed_fraction"] < 0.5 and row["D_max_thinned"] <= 8 and row["D_max_thinned"] <= res["D_max"]


def test_diagonal_ray_lemma():
    """D_max(F_bal, W) >= floor(sqrt r) - 1 for W >= 2, attained by consecutive diagonal cells (m, m), whose start differences are
    N + 1 - ceil((m+1) theta) + ceil(m theta) in {N + 1 - ceil(theta), N + 1 - floor(theta)}, theta = 2 sqrt N."""
    from gmpy2 import iroot

    for idx in range(2):
        N = int(make_semiprime(30, "rsa", 5, idx).N)
        r = int(iroot(mpz(N), 3)[0])
        m_max = int(isqrt(r))
        diag = [(m, m) for m in range(1, m_max + 1)]
        S = exponent_set(N, diag, r, with_windows=False)
        DW, tW = dmax_tol(S, 2)
        assert DW == m_max - 1
        c = ceil_2sqrt(N)  # ceil(2 sqrt N) = ceil(theta)
        assert tW in (N + 1 - c, N + 2 - c, N - c)
        cells = balanced_cells(N, r, 2)
        Sb = exponent_set(N, cells, r, with_windows=False)
        assert dmax_tol(Sb, 2)[0] >= m_max - 1
