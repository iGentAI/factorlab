"""E52: the block decomposition of a multiplicative chirp.

A multiplicative chirp is  C(n) = prod_{k < n} (1 - q^{f(k)})  in Z/NZ with f(k) = a k^2 + b k + c integer-valued.  Writing
k = k1 + a' k2 (0 <= k1 < a', 0 <= k2 < b', a' b' = n),

    f(k) = f(k1) + (2 a a' k1) k2 + (a a'^2 k2^2 + b a' k2),

so  q^{f(k)} = q^{f(k1)} W^{k1} Z  with  W = q^{2 a a' k2},  Z = q^{a a'^2 k2^2 + b a' k2}  depending on k2 only, and

    C(n) = prod_{k2 < b'} P(Z_{k2}, W_{k2}),     P(Z, W) := prod_{k1 < a'} (1 - q^{f(k1)} Z W^{k1}).

P is one bivariate polynomial of degree a' in Z and at most a'(a'-1)/2 in W (about a'^3/2 coefficients), and the chirp is
P evaluated at b' points and multiplied out.  By Kedlaya-Umans multivariate multipoint evaluation over Z/NZ (SIAM J. Comput.
40 (2011)), evaluating P at b' points costs (d^2 + b')^{1+o(1)} bit operations with d = a'^2/2, so with a' = n^{1/5} the chirp
costs n^{4/5+o(1)} in that form.  With a Beatty rounding term, f(k) + ceil(k rho), the identity ceil(x + y) = ceil(x) + ceil(y) - [ {x} + {y} <= 1 ]
(x, y non-integers) makes the extra factor q^{-1} apply to the k1 whose fractional part {k1 rho} is at most a threshold
1 - {a' k2 rho} depending on k2; sorting the k1 by {k1 rho} gives a' prefix/suffix polynomials.  Splitting the W-exponent into two digits below a' + 1 makes all
degrees at most a', so Kedlaya-Umans in three variables costs (a'^3 + b')^{1+o(1)}: the chirp costs n^{3/4+o(1)} and the
Beatty chirp n^{4/5+o(1)} (the two-variable form gives 4/5 and 5/6).

This module checks the two identities exactly (naive evaluation of P; it does not implement Kedlaya-Umans).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Tuple

from gmpy2 import mpz, powmod


def chirp_direct(q: int, N: int, a: int, b: int, c: int, n: int) -> int:
    """prod_{k<n} (1 - q^{a k^2 + b k + c}) mod N, directly."""
    N = mpz(N)
    out = mpz(1)
    for k in range(n):
        out = out * (1 - powmod(q, a * k * k + b * k + c, N)) % N
    return int(out)


def bivariate_P(q: int, N: int, a: int, b: int, c: int, a1: int, extra: Dict[int, int] | None = None) -> Dict[Tuple[int, int], int]:
    """Coefficients of P(Z, W) = prod_{k1 < a1} (1 - q^{f(k1)} e_{k1} Z W^{k1}) mod N as a dict {(s, t): coeff}; e_{k1} = extra
    multiplier for k1 (default 1)."""
    N = mpz(N)
    P: Dict[Tuple[int, int], int] = {(0, 0): 1}
    for k1 in range(a1):
        coef = powmod(q, a * k1 * k1 + b * k1 + c, N) * (extra.get(k1, 1) if extra else 1) % N
        newP: Dict[Tuple[int, int], int] = {}
        for (s, t), v in P.items():
            newP[(s, t)] = (newP.get((s, t), 0) + v) % N
            key = (s + 1, t + k1)
            newP[key] = (newP.get(key, 0) - v * coef) % N
        P = newP
    return {k: int(v) for k, v in P.items()}


def eval_P(P: Dict[Tuple[int, int], int], Z: int, W: int, N: int) -> int:
    N = mpz(N)
    Zp = {}
    Wp = {}
    out = mpz(0)
    for (s, t), v in P.items():
        if s not in Zp:
            Zp[s] = powmod(Z, s, N)
        if t not in Wp:
            Wp[t] = powmod(W, t, N)
        out = (out + v * Zp[s] % N * Wp[t]) % N
    return int(out)


def chirp_blocked(q: int, N: int, a: int, b: int, c: int, n: int, a1: int) -> int:
    """The same product through the block decomposition (naive evaluation of P at the b' points)."""
    assert n % a1 == 0
    b1 = n // a1
    P = bivariate_P(q, N, a, b, c, a1)
    N_ = mpz(N)
    out = mpz(1)
    for k2 in range(b1):
        W = powmod(q, 2 * a * a1 * k2, N_)
        Z = powmod(q, a * a1 * a1 * k2 * k2 + b * a1 * k2, N_)
        out = out * eval_P(P, int(Z), int(W), N) % N_
    return int(out)


def frac_ceil(m: int, rho: Fraction) -> Tuple[Fraction, int]:
    """({m rho}, ceil(m rho)) for a rational rho (the exact-arithmetic stand-in for an irrational slope)."""
    x = m * rho
    fl = x.numerator // x.denominator
    return x - fl, (fl if x == fl else fl + 1)


def beatty_chirp_direct(q: int, N: int, a: int, b: int, c: int, rho: Fraction, n: int) -> int:
    """prod_{k<n} (1 - q^{f(k) + ceil(k rho)}) mod N, directly."""
    N_ = mpz(N)
    out = mpz(1)
    for k in range(n):
        _, ce = frac_ceil(k, rho)
        out = out * (1 - powmod(q, a * k * k + b * k + c + ce, N_)) % N_
    return int(out)


def beatty_chirp_blocked(q: int, N: int, a: int, b: int, c: int, rho: Fraction, n: int, a1: int) -> Tuple[int, int]:
    """Blocked form with the carry handled by prefix/suffix polynomials; returns (value, number of distinct polynomials used).
    Requires rho > 0 and k rho non-integral for 0 < k < n (generic slope); q may be any residue modulo N when the exponents f(k) are non-negative (a, b, c >= 0), else q must be a unit."""
    assert n % a1 == 0
    b1 = n // a1
    N_ = mpz(N)
    fr = [frac_ceil(k1, rho) for k1 in range(a1)]  # ({k1 rho}, ceil(k1 rho))
    order = sorted(range(a1), key=lambda k1: fr[k1][0])  # k1 sorted by fractional part
    base_extra = {k1: int(powmod(q, fr[k1][1], N_)) for k1 in range(a1)}  # q^{ceil(k1 rho)}
    carry_extra = {k1: int(powmod(q, fr[k1][1] - 1, N_)) for k1 in range(1, a1)}  # q^{ceil(k1 rho) - 1}, exponent >= 0
    polys: Dict[int, Dict[Tuple[int, int], int]] = {}

    out = mpz(1)
    for k2 in range(b1):
        fy, cy = frac_ceil(a1 * k2, rho)  # ({a' k2 rho}, ceil(a' k2 rho))
        # ceil(x + y) = ceil(x) + ceil(y) - [{x} + {y} <= 1] for non-integral x, y; the k1 = 0 term (x = 0) and the k2 = 0
        # block (y = 0) never carry.
        thr = None if k2 == 0 else 1 - fy
        P = carry_polynomial(thr, order, fr, base_extra, carry_extra, q, N, a, b, c, a1, polys)
        W = powmod(q, 2 * a * a1 * k2, N_)
        Z = powmod(q, a * a1 * a1 * k2 * k2 + b * a1 * k2 + cy, N_)
        out = out * eval_P(P, int(Z), int(W), N) % N_
    return int(out), len(polys)


def carry_polynomial(thr, order, fr, base_extra, carry_extra, q, N, a, b, c, a1, polys):
    """The polynomial P in which the k1 >= 1 with fractional part {k1 rho} at most thr (a prefix of the k1 sorted by fractional
    part) carry the factor q^{ceil(k1 rho) - 1} instead of q^{ceil(k1 rho)}; thr = None means no carry.  Cached by the prefix."""
    if thr is None:
        key = 0  # the same polynomial as an empty prefix at any threshold
        carriers: List[int] = []
    else:
        carriers = [k1 for k1 in order if k1 >= 1 and fr[k1][0] <= thr]
        key = len(carriers)
    if key not in polys:
        extra = dict(base_extra)
        for k1 in carriers:
            extra[k1] = carry_extra[k1]
        polys[key] = bivariate_P(q, N, a, b, c, a1, extra)
    return polys[key]


def beatty_pochhammer_direct(x: int, y: int, z: int, N: int, rho: Fraction, M: int) -> int:
    """prod_{m<M} (1 - x y^m z^{ceil(m rho)}) mod N, directly (the Beatty-rounded q-Pochhammer product)."""
    N_ = mpz(N)
    out = mpz(1)
    for m in range(M):
        _, ce = frac_ceil(m, rho)
        out = out * (1 - x * powmod(y, m, N_) % N_ * powmod(z, ce, N_)) % N_
    return int(out)


def beatty_pochhammer_blocked(x: int, y: int, z: int, N: int, rho: Fraction, M: int, a1: int) -> int:
    """The same product in blocks m = m1 + a1 m2: for fixed m2 the block is the UNIVARIATE polynomial
    prod_{m1<a1} (1 - X d_{m1}) at X = x y^{a1 m2} z^{ceil(a1 m2 rho)}, where the carrying m1 >= 1 (a prefix of the m1 sorted by
    {m1 rho}) use d_{m1} = y^{m1} z^{ceil(m1 rho) - 1} and the others y^{m1} z^{ceil(m1 rho)}.  Requires rho > 0 and m rho
    non-integral for 0 < m < M; z may be any residue.  Naive evaluation (the proposition's prefix/suffix product trees are not
    implemented)."""
    assert M % a1 == 0
    b1 = M // a1
    N_ = mpz(N)
    fr = [frac_ceil(m1, rho) for m1 in range(a1)]
    d = {m1: powmod(y, m1, N_) * powmod(z, fr[m1][1], N_) % N_ for m1 in range(a1)}
    d_carry = {m1: powmod(y, m1, N_) * powmod(z, fr[m1][1] - 1, N_) % N_ for m1 in range(1, a1)}  # exponent >= 0
    out = mpz(1)
    for m2 in range(b1):
        fy, cy = frac_ceil(a1 * m2, rho)
        X = x * powmod(y, a1 * m2, N_) % N_ * powmod(z, cy, N_) % N_
        thr = None if m2 == 0 else 1 - fy
        val = mpz(1)
        for m1 in range(a1):
            carries = thr is not None and m1 >= 1 and fr[m1][0] <= thr
            val = val * (1 - X * (d_carry[m1] if carries else d[m1])) % N_
        out = out * val % N_
    return int(out)


def coefficient_count(a1: int) -> int:
    """Number of monomials of P(Z, W) for a block of length a1 (upper bound (a1 + 1)(a1(a1-1)/2 + 1))."""
    return (a1 + 1) * (a1 * (a1 - 1) // 2 + 1)
