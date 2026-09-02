"""Quadratic sieve with a single polynomial and an optional single large prime.

Q(x) = (s + x)^2 - N with s = isqrt(N) + 1, x in [-M, M).  The factor base is
-1, 2 and the odd primes l <= B with (N | l) = 1, for which Q(x) = 0 (mod l)
iff s + x = +-r_l (mod l), r_l = sqrt(N) mod l.  A numpy log-sieve adds log l
at those positions (prime powers are ignored; the acceptance slack compensates);
the 2-adic contribution follows the residue of N mod 8 (v_2(Q) = 3+, 2, 1 for
odd s + x when N = 1, 5, 3 or 7 mod 8).  Candidates whose sieve value exceeds
log|Q(x)| - slack are trial-divided exactly.  Full relations, and pairs of
partial relations sharing one large prime < B^2, are collected until their
number exceeds the factor-base size; dependencies over GF(2) are found by
Gaussian elimination on Python-int bit rows, and each gives X = prod (s + x_i),
Y = sqrt(prod Q(x_i)) (mod N) and a gcd test.

Work counters: ``sieve`` (log additions, the primary cost), ``candidate``
(trial divisions), ``relation``, ``partial``, ``dependency``.  The result's
meta records the parameters and the yield (full relations per sieved x), which
E11 compares with the Dickman prediction.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..numth import mpz, isqrt, gcd, jacobi, small_primes, sqrt_mod_prime
from ..registry import register
from ..result import Work, success, failure


def default_parameters(N) -> tuple[int, int]:
    """(B, M): B = exp(0.55 sqrt(ln N ln ln N)) clipped to [60, 20000], M = 30 B."""
    lnN = math.log(int(N))
    B = int(round(math.exp(0.55 * math.sqrt(lnN * math.log(lnN)))))
    B = max(60, min(B, 20000))
    return B, 30 * B


def factor_base(N, B: int):
    """Odd primes l <= B with (N | l) = 1 and their roots sqrt(N) mod l."""
    N = mpz(N)
    primes, roots = [], []
    for l in small_primes(B + 1):
        if l == 2:
            continue
        if N % l == 0:
            continue
        if jacobi(N, l) == 1:
            r = int(sqrt_mod_prime(N, l))
            primes.append(l)
            roots.append(r)
    return primes, roots


def expected_valuation_shift(N, B: int) -> float:
    """sum_l log(l) (E[v_l(Q(x))] - 1/(l-1)) over primes l <= B: the local excess
    of small-prime content of Q(x) over a random integer (Knuth-Schroeppel type).
    Split odd primes contribute 2/(l-1) - 1/(l-1); inert ones -1/(l-1); 2
    contributes E[v_2] - 1 with E[v_2] = 2, 1, 1/2 for N = 1, 5, {3,7} mod 8."""
    N = mpz(N)
    s = 0.0
    for l in small_primes(B + 1):
        if l == 2:
            m = int(N % 8)
            e2 = 2.0 if m == 1 else (1.0 if m == 5 else 0.5)
            s += math.log(2) * (e2 - 1.0)
        elif N % l == 0:
            continue
        else:
            ev = 2.0 / (l - 1) if jacobi(N, l) == 1 else 0.0
            s += math.log(l) * (ev - 1.0 / (l - 1))
    return s


def _trial_divide(Q, primes):
    """Exact factorisation of |Q| over the factor base; returns (exponents dict, cofactor)."""
    q = abs(int(Q))
    ex = {}
    if q == 0:
        return ex, 0
    e = 0
    while q % 2 == 0:
        q //= 2
        e += 1
    if e:
        ex[2] = e
    for l in primes:
        if q % l == 0:
            e = 0
            while q % l == 0:
                q //= l
                e += 1
            ex[l] = e
        if q == 1:
            break
    return ex, q


def _gf2_dependencies(rows: list[int], max_deps: int = 32) -> list[int]:
    """Rows are bitmasks over columns; returns combinations (bitmasks over row
    indices) whose XOR is zero, by incremental elimination."""
    pivots: dict[int, tuple[int, int]] = {}  # pivot bit -> (reduced row, combination)
    deps = []
    for i, r in enumerate(rows):
        comb = 1 << i
        while r:
            b = r.bit_length() - 1
            if b in pivots:
                pr, pc = pivots[b]
                r ^= pr
                comb ^= pc
            else:
                pivots[b] = (r, comb)
                break
        if r == 0:
            deps.append(comb)
            if len(deps) >= max_deps:
                break
    return deps


@register("qs", primary_key="sieve",
          description="single-polynomial quadratic sieve with one large prime; B, M default from N")
def quadratic_sieve(N, B=None, M=None, slack=1.5, large_prime=True, max_blocks=400, extra=8, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    if N % 2 == 0:
        return success("qs", N, 2, w, "sieve", time.perf_counter() - t0)
    r = isqrt(N)
    if r * r == N:
        return success("qs", N, r, w, "sieve", time.perf_counter() - t0)
    if B is None or M is None:
        B0, M0 = default_parameters(N)
        B = B0 if B is None else int(B)
        M = M0 if M is None else int(M)
    B, M = int(B), int(M)
    for l in small_primes(B + 1):
        if N % l == 0:
            return success("qs", N, l, w, "sieve", time.perf_counter() - t0)
    primes, roots = factor_base(N, B)
    s = isqrt(N) + 1
    s_int = int(s)
    cols = {-1: 0, 2: 1}
    for i, l in enumerate(primes):
        cols[l] = i + 2
    ncols = len(cols)
    logs = np.log(np.array(primes, dtype=np.float64))
    m8 = int(N % 8)
    log2_contrib = math.log(2) * (3.0 if m8 == 1 else (2.0 if m8 == 5 else 1.0))
    thr_slack = slack * math.log(B)

    relations = []  # (xs: list[int], exponents: dict, large: int)
    partials: dict[int, tuple[int, dict]] = {}
    n_full = 0
    n_pairs = 0
    sieved = 0
    needed = ncols + extra
    s_f = float(s_int)

    def block_bounds(bi):
        # blocks: 0 -> [-M, M); then alternating [M,3M), [-3M,-M), ...
        if bi == 0:
            return -M, M
        k = (bi + 1) // 2
        if bi % 2 == 1:
            return (2 * k - 1) * M, (2 * k + 1) * M
        return -(2 * k + 1) * M, -(2 * k - 1) * M

    for bi in range(max_blocks):
        x0, x1 = block_bounds(bi)
        L = x1 - x0
        arr = np.zeros(L, dtype=np.float64)
        # 2-adic: Q(x) even iff s + x odd
        start2 = (1 - (s_int + x0) % 2) % 2
        arr[start2::2] += log2_contrib
        w.add("sieve", L // 2)
        for l, rt, lg in zip(primes, roots, logs):
            for root in (rt, l - rt):
                start = (root - s_int - x0) % l
                arr[start::l] += lg
            w.add("sieve", 2 * (L // l + 1))
        sieved += L
        xs = np.arange(x0, x1, dtype=np.float64)
        logQ = np.log(np.maximum(np.abs(xs * xs + 2.0 * s_f * xs + float(s * s - N)), 1.0))
        cand = np.nonzero(arr >= logQ - thr_slack)[0]
        for ci in cand:
            x = int(x0 + ci)
            Q = (s + x) * (s + x) - N
            w.add("candidate")
            ex, cof = _trial_divide(Q, primes)
            if Q < 0:
                ex[-1] = 1
            if cof == 1:
                relations.append(([x], ex, 0))
                n_full += 1
                w.add("relation")
            elif large_prime and cof < B * B and cof > B:
                if cof in partials:
                    x2, ex2 = partials.pop(cof)
                    comb = dict(ex)
                    for k_, v_ in ex2.items():
                        comb[k_] = comb.get(k_, 0) + v_
                    relations.append(([x, x2], comb, cof))
                    n_pairs += 1
                    w.add("relation")
                else:
                    partials[cof] = (x, ex)
                    w.add("partial")
        if len(relations) >= needed:
            break

    meta = {"B": B, "M": M, "factor_base": ncols, "sieved": sieved, "full": n_full, "pairs": n_pairs,
            "partials_unmatched": len(partials), "yield_full_per_x": n_full / sieved if sieved else 0.0,
            "local_shift": expected_valuation_shift(N, B)}
    if len(relations) < ncols + 1:
        return failure("qs", N, w, "sieve", time.perf_counter() - t0, **meta)
    rows = []
    for xs_, ex, _L in relations:
        bits = 0
        for l, e in ex.items():
            if e & 1:
                bits |= 1 << cols[l]
        rows.append(bits)
    for comb in _gf2_dependencies(rows):
        w.add("dependency")
        X = mpz(1)
        tot = {}
        larges = mpz(1)
        i = 0
        c = comb
        while c:
            if c & 1:
                xs_, ex, Lp = relations[i]
                for x in xs_:
                    X = X * (s + x) % N
                for l, e in ex.items():
                    tot[l] = tot.get(l, 0) + e
                if Lp:
                    larges = larges * Lp % N
            c >>= 1
            i += 1
        Y = larges
        for l, e in tot.items():
            if l == -1:
                continue
            Y = Y * pow(mpz(l), e // 2, N) % N
        for cand_g in (gcd(X - Y, N), gcd(X + Y, N)):
            if 1 < cand_g < N:
                return success("qs", N, cand_g, w, "sieve", time.perf_counter() - t0, **meta)
    return failure("qs", N, w, "sieve", time.perf_counter() - t0, **meta)
