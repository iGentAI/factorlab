"""Tests for latticelab.certify_audit: exact Gram-Schmidt norms and the certified per-basis identity."""
from fractions import Fraction

import numpy as np
from flint import arb, ctx, fmpz_mat

from latticelab.certify_audit import certify_basis, exact_gso_norms


def _dense_basis(d, seed, bound=40):
    """A random dense nonsingular integer basis; its Gram-Schmidt squared norms are non-integral rationals in general."""
    rng = np.random.default_rng(seed)
    while True:
        B = rng.integers(-bound, bound + 1, size=(d, d))
        rows = [[int(x) for x in row] for row in B]
        if fmpz_mat(rows).det() != 0:
            return rows


def _frac(s):
    n, den = s.split("/")
    return Fraction(int(n), int(den))


def test_exact_gso_matches_float_qr_and_is_nonintegral():
    rows = _dense_basis(10, 5)
    r = exact_gso_norms(rows)
    Q, R = np.linalg.qr(np.array(rows, dtype=float).T)
    float_norms = np.diag(R) ** 2
    nonintegral = 0
    for k in range(10):
        fr = Fraction(int(r[k].p), int(r[k].q))
        assert abs(float(fr) - float_norms[k]) <= 1e-9 * max(1.0, float_norms[k])
        nonintegral += fr.denominator != 1
    assert nonintegral >= 5
    # the product of the squared norms is the Gram determinant
    prod = Fraction(1)
    for x in r:
        prod *= Fraction(int(x.p), int(x.q))
    G = fmpz_mat(rows) * fmpz_mat(rows).transpose()
    assert prod == Fraction(int(G.det()))


def test_certified_identity_and_enclosures():
    rows = _dense_basis(10, 7)
    res = certify_basis(rows, beta=4, prec=128)
    gap = res["head_minus_floor"]
    wd = res["weighted_deficit"]
    lo, hi = _frac(gap["lower"]), _frac(gap["upper"])
    wlo, whi = _frac(wd["lower"]), _frac(wd["upper"])
    assert lo <= hi and wlo <= whi
    # the identity ell_1 - S/d - h(0) = -sum_i y_i nu_i: the interval sum encloses zero
    assert lo + wlo <= 0 <= hi + whi
    assert hi - lo < Fraction(1, 10**30)
    # the display midpoint lies within the exact endpoints up to double rounding
    mid = Fraction(gap["approx_mid"])
    tol = Fraction(1, 10**12) * max(1, abs(mid))
    assert lo - tol <= mid <= hi + tol
    # the decision flag is the directed comparison of the upper endpoint with zero
    assert res["below_floor_rigorously"] == (hi < 0)
    # the log-volume enclosure contains an independent higher-precision value of (1/2) log det G
    G = fmpz_mat(rows) * fmpz_mat(rows).transpose()
    saved_prec = ctx.prec
    try:
        ctx.prec = 512
        ref = arb(int(G.det())).log() / 2
        m, e = ref.mid().man_exp()
    finally:
        ctx.prec = saved_prec
    vlo, vhi = _frac(res["log_vol"]["lower"]), _frac(res["log_vol"]["upper"])
    ref_mid = Fraction(int(m)) * (Fraction(2) ** int(e))
    assert vlo <= ref_mid <= vhi
