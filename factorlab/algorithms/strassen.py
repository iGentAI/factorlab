"""Pollard-Strassen deterministic O~(N^{1/4}) factoring via product trees and
multipoint evaluation over Z/NZ (python-flint fmpz_mod_poly).

Given a bound M (default ceil(N^{1/2})), let d = ceil(sqrt(M)).  Compute
f(x) = (x+1)(x+2)...(x+d) in Z_N[x] and evaluate at x = 0, d, 2d, ..., (d-1)d.
f(jd) = (jd+1)...(jd+d); a nontrivial gcd with N isolates a block containing a
prime factor, which is then found by direct scan of d numbers.

All degree work is counted in ``poly_deg`` (sum of degrees of polynomials
multiplied plus evaluation points), which is Theta(d log d) ~ Theta(N^{1/4})
up to logs; wall time is the honest measure here since flint does the work.
"""

from __future__ import annotations

import time

import flint

from ..numth import mpz, gcd, isqrt_ceil
from ..registry import register
from ..result import Work, success, failure


def product_tree(ctx, roots, w: Work | None = None):
    """prod (x - r) over roots, as an fmpz_mod_poly.  Roots given as python ints/mpz."""
    level = [ctx([(-int(r)) % int(ctx.modulus()), 1]) for r in roots]
    if not level:
        return ctx([1])
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(level[i] * level[i + 1])
            if w is not None:
                w.add("poly_deg", level[i].degree() + level[i + 1].degree())
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def shifted_factorial_poly(ctx, d: int, w: Work | None = None):
    """(x+1)(x+2)...(x+d) in Z_N[x]."""
    return product_tree(ctx, [-(i) for i in range(1, d + 1)], w)


@register("pollard_strassen", primary_key="poly_deg", description="deterministic N^{1/4+o(1)} via product tree + multipoint evaluation (flint)")
def pollard_strassen(N, M=None, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    if N % 2 == 0:
        return success("pollard_strassen", N, 2, w, "poly_deg", time.perf_counter() - t0)
    M = isqrt_ceil(N) if M is None else mpz(M)
    d = int(isqrt_ceil(M))
    ctx = flint.fmpz_mod_poly_ctx(int(N))
    f = shifted_factorial_poly(ctx, d, w)
    pts = [j * d for j in range(d)]
    vals = f.multipoint_evaluate(pts)
    w.add("poly_deg", len(pts))
    for j, v in enumerate(vals):
        g = gcd(mpz(int(v)), N)
        w.add("gcd")
        if g != 1:
            # a factor lies in (jd, jd+d]
            for i in range(1, d + 1):
                w.add("division")
                cand = j * d + i
                if cand > 1 and N % cand == 0:
                    return success("pollard_strassen", N, cand, w, "poly_deg", time.perf_counter() - t0, block=j, block_size=d)
    return failure("pollard_strassen", N, w, "poly_deg", time.perf_counter() - t0, block_size=d)
