"""N1: Frobenius-defect gcd leakage in R_r = Z_N[X]/(X^r - 1).

F_a(X) = (X + a)^N - X^N - a.  Over F_p, since N = pq and Frobenius,
    F_a = (X^p + a)^q - X^{pq} - a = G_{q,a}(X^p),   G_{m,a}(Y) = (Y+a)^m - Y^m - a.
Over F_q the roles of p and q swap.  Compute the Euclidean remainder sequence
of (F_a mod (X^r-1), X^r-1) over Z_N.  If at some step the leading coefficient
is a zero divisor (gcd with N nontrivial), a factor leaks.  Equivalently the
degree profile of gcd(F_a, X^r-1) differs modulo p and modulo q.

Diagnostic (uses secret p, q): compute deg gcd(F_a, X^r - 1) over F_p and F_q
directly with flint nmod_poly / fmpz_mod_poly and compare; the leakage event is
exactly degree mismatch (or mismatch at any intermediate remainder).

Prediction: roots of G_{q,a} on mu_r(F_p) occur with probability ~ gcd(r, p-1)/p
per unit of structure, so leakage ~ r/p per trial -> total cost N^{1/2}.
"""

from __future__ import annotations

import time

import flint

from ..numth import mpz, gcd


def _poly_mod_field(P: int, coeffs):
    return flint.nmod_poly(coeffs, P) if P < 2**62 else flint.fmpz_mod_poly(coeffs, flint.fmpz_mod_poly_ctx(P))


def frobenius_defect_poly(modulus: int, N: int, a: int, r: int):
    """F_a mod (X^r - 1) over Z/modulus, via pow_mod."""
    if modulus < 2**62:
        ctx = None
        X = flint.nmod_poly([0, 1], modulus)
        xr1 = flint.nmod_poly([-1 % modulus] + [0] * (r - 1) + [1], modulus)
        base = flint.nmod_poly([a % modulus, 1], modulus)
    else:
        ctx = flint.fmpz_mod_poly_ctx(modulus)
        X = ctx([0, 1])
        xr1 = ctx([-1] + [0] * (r - 1) + [1])
        base = ctx([a, 1])
    t1 = base.pow_mod(N, xr1)
    t2 = X.pow_mod(N, xr1)
    return (t1 - t2 - a) % xr1, xr1


def gcd_degree_profile(P: int, N: int, a: int, r: int) -> int:
    """deg gcd(F_a, X^r - 1) over the field F_P."""
    F, xr1 = frobenius_defect_poly(P, N, a, r)
    g = F.gcd(xr1)
    return int(g.degree())


def leakage_over_ZN(N: int, a: int, r: int):
    """Run Euclid on (X^r - 1, F_a) over Z_N with explicit leading-coefficient
    unit checks.  Returns (leaked_factor_or_None, steps)."""
    N = int(N)
    ctx = flint.fmpz_mod_poly_ctx(N)
    F, xr1 = frobenius_defect_poly(N, N, a, r)
    A, B = xr1, F
    steps = 0
    while True:
        if B.is_zero():
            return None, steps
        lc = int(B.leading_coefficient())
        g = gcd(mpz(lc), mpz(N))
        if 1 < g < N:
            return int(g), steps
        if g == N:
            # leading coefficient is 0 mod N -- cannot happen for normalised polys
            return None, steps
        # lc is a unit: ordinary division step
        try:
            _, R = divmod(A, B)
        except (ZeroDivisionError, ValueError):
            # flint could not invert the leading coefficient -> zero divisor
            return int(g) if 1 < g < N else None, steps
        A, B = B, R
        steps += 1


def experiment(N, p, q, r_values, a_values):
    """For each (r, a): degrees over F_p, F_q; whether Z_N Euclid leaks."""
    rows = []
    for r in r_values:
        for a in a_values:
            t0 = time.perf_counter()
            dp = gcd_degree_profile(int(p), int(N), a, r)
            dq = gcd_degree_profile(int(q), int(N), a, r)
            leaked, steps = leakage_over_ZN(int(N), a, r)
            rows.append({"r": r, "a": a, "deg_p": dp, "deg_q": dq, "mismatch": dp != dq,
                         "leaked": leaked is not None and leaked in (int(p), int(q)),
                         "euclid_steps": steps, "wall": time.perf_counter() - t0,
                         "gcd_r_pm1": int(gcd(mpz(r), mpz(p) - 1)), "gcd_r_qm1": int(gcd(mpz(r), mpz(q) - 1))})
    return rows
