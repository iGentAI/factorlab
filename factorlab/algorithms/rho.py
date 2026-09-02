"""Probabilistic / smoothness-based methods: Pollard rho (Brent), p-1, Williams p+1."""

from __future__ import annotations

import math
import random
import time

from ..numth import mpz, gcd, powmod, small_primes
from ..registry import register
from ..result import Work, success, failure


@register("pollard_rho", primary_key="mulmod", description="Brent-variant rho with batched gcd", deterministic=False)
def pollard_rho(N, seed=0, max_iter=10**8, batch=128, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    if N % 2 == 0:
        return success("pollard_rho", N, 2, w, "mulmod", time.perf_counter() - t0)
    rng = random.Random(seed)
    while True:
        c = mpz(rng.randrange(1, int(N) - 1))
        y = mpz(rng.randrange(0, int(N)))
        m = batch
        g = r = q = mpz(1)
        x = ys = y
        iters = 0
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % N
                w.add("mulmod")
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % N
                    q = q * abs(x - y) % N
                    w.add("mulmod", 2)
                g = gcd(q, N)
                w.add("gcd")
                k += m
            r *= 2
            iters += r
            if iters > max_iter:
                return failure("pollard_rho", N, w, "mulmod", time.perf_counter() - t0)
        if g == N:
            g = mpz(1)
            while g == 1:
                ys = (ys * ys + c) % N
                g = gcd(abs(x - ys), N)
                w.add("mulmod")
                w.add("gcd")
        if 1 < g < N:
            return success("pollard_rho", N, g, w, "mulmod", time.perf_counter() - t0, c=int(c))
        # else retry with new c


@register("pollard_pm1", primary_key="mulmod", description="Pollard p-1 stage 1 with bound B1 (prime powers <= B1)")
def pollard_pm1(N, B1=10**5, base=2, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    a = mpz(base)
    for p in small_primes(int(B1) + 1):
        e = int(math.log(B1) / math.log(p))
        a = powmod(a, p ** e, N)
        w.add("mulmod", int(e * math.log2(p)) + 1)
    g = gcd(a - 1, N)
    w.add("gcd")
    if 1 < g < N:
        return success("pollard_pm1", N, g, w, "mulmod", time.perf_counter() - t0, B1=B1)
    return failure("pollard_pm1", N, w, "mulmod", time.perf_counter() - t0, B1=B1, g=int(g))


def lucas_v(P, n, N):
    """Correct Lucas V_n(P,1) mod N using the standard ladder (v_k, v_{k+1})."""
    N = mpz(N)
    P = mpz(P) % N
    vk, vk1 = mpz(2), P
    for bit in bin(int(n))[2:]:
        if bit == "1":
            vk, vk1 = (vk * vk1 - P) % N, (vk1 * vk1 - 2) % N
        else:
            vk, vk1 = (vk * vk - 2) % N, (vk * vk1 - P) % N
    return vk


@register("williams_pp1", primary_key="mulmod", description="Williams p+1 stage 1 with bound B1")
def williams_pp1(N, B1=10**5, P0=3, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    V = mpz(P0)
    for p in small_primes(int(B1) + 1):
        e = int(math.log(B1) / math.log(p))
        V = lucas_v(V, p ** e, N)
        w.add("mulmod", 2 * (int(e * math.log2(p)) + 1))
    g = gcd(V - 2, N)
    w.add("gcd")
    if 1 < g < N:
        return success("williams_pp1", N, g, w, "mulmod", time.perf_counter() - t0, B1=B1)
    return failure("williams_pp1", N, w, "mulmod", time.perf_counter() - t0, B1=B1)
