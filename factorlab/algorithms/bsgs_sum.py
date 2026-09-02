"""Baby-step giant-step for s = p + q using alpha^{N+1} = alpha^{p+q} (mod N).

This is the McKee-Pinch / Hittmeir starting point.  For balanced N with
p,q ~ sqrt(N), s - 2 sqrt(N) = (sqrt q - sqrt p)^2 lies in [0, W) with
W = (q-p)^2 / (4 sqrt N) roughly; for random balanced primes W ~ N^{1/2}.
BSGS over an interval of width W costs O(W^{1/2}) mulmods, i.e. N^{1/4} for
generic balanced semiprimes (Hittmeir's 2018 refinement squeezes the constant
using residues of s modulo small primes).

Parameters
----------
W : interval width to search above s_min = ceil(2 sqrt N) (default N^{1/2}).
alpha : base (default 2).

Work counted in ``mulmod`` (plus dictionary lookups, which are free here).
"""

from __future__ import annotations

import time

from ..numth import mpz, isqrt, isqrt_ceil, is_square, gcd, powmod, invert
from ..registry import register
from ..result import Work, success, failure


def factor_from_sum(N, s):
    """Given s = p + q, return p (or None)."""
    disc = s * s - 4 * N
    if disc < 0 or not is_square(disc):
        return None
    r = isqrt(disc)
    if (s - r) % 2:
        return None
    p = (s - r) // 2
    if 1 < p < N and N % p == 0:
        return p
    return None


@register("bsgs_sum", primary_key="mulmod", description="BSGS for p+q via alpha^{p+q} = alpha^{N+1} mod N over width W")
def bsgs_sum(N, W=None, alpha=2, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    alpha = mpz(alpha)
    g = gcd(alpha, N)
    if 1 < g < N:
        return success("bsgs_sum", N, g, w, "mulmod", time.perf_counter() - t0)
    s_min = isqrt_ceil(4 * N)  # p+q >= 2 sqrt N
    W = isqrt(N) + 1 if W is None else mpz(W)
    m = int(isqrt_ceil(W))
    # target: alpha^{N+1-s_min} = alpha^{x}, x = s - s_min in [0, W)
    target = powmod(alpha, N + 1 - s_min, N)
    w.add("mulmod", 2 * int(N).bit_length())
    # baby steps alpha^i, i in [0, m)
    table = {}
    cur = mpz(1)
    for i in range(m):
        table.setdefault(int(cur), i)
        cur = cur * alpha % N
        w.add("mulmod")
    # giant steps: target * alpha^{-jm}
    step = invert(cur, N)  # alpha^{-m}
    w.add("mulmod", 2 * int(N).bit_length())
    y = target
    for j in range(m + 1):
        i = table.get(int(y))
        if i is not None:
            x = i + j * m
            s = s_min + x
            p = factor_from_sum(N, s)
            w.add("sqrt_test")
            if p is not None:
                return success("bsgs_sum", N, p, w, "mulmod", time.perf_counter() - t0, s=int(s), x=int(x))
        y = y * step % N
        w.add("mulmod")
    return failure("bsgs_sum", N, w, "mulmod", time.perf_counter() - t0, W=int(W))
