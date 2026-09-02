"""Lenstra's elliptic curve method, stage 1, on Montgomery curves.

A smoothness method with *internal* randomness: the curve is chosen at random
(Suyama's parametrisation by sigma), its group order over F_p is a random-like
integer in the Hasse interval, and stage 1 with bound B1 succeeds on that curve
iff the order of the starting point divides lcm{l^e : l^e <= B1}.  In contrast
to p-1 and p+1, whose success is a fixed property of N, repeating ECM with fresh
curves makes the failure probability decay geometrically for every N.

Work is counted in modular multiplications (x-only Montgomery ladder: 5 per
doubling, 6 per differential addition).
"""

from __future__ import annotations

import random
import time

from ..numth import mpz, gcd, invert, small_primes
from ..registry import register
from ..result import Work, success, failure


def suyama_curve(sigma, N):
    """Montgomery curve B y^2 = x^3 + A x^2 + x with a point, from Suyama's sigma.

    Returns ``((a24, X0, Z0), None)`` with ``a24 = (A + 2)/4`` and a projective
    point (X0 : Z0), or ``(None, g)`` when the construction meets a non-unit
    (then ``g = gcd(den, N) > 1``, possibly a factor).
    """
    N = mpz(N)
    sigma = mpz(sigma) % N
    u = (sigma * sigma - 5) % N
    v = (4 * sigma) % N
    X0 = pow(u, 3, N)
    Z0 = pow(v, 3, N)
    num = (pow(v - u, 3, N) * (3 * u + v)) % N
    den = (16 * X0 * v) % N  # 4 u^3 v * 4: a24 = (A + 2)/4 = (v-u)^3 (3u+v) / (16 u^3 v)
    g = gcd(den, N)
    if g != 1:
        return None, g
    a24 = (num * invert(den, N)) % N
    return (a24, X0, Z0), None


def xdbl(X, Z, a24, N, w: Work):
    """x-only doubling on a Montgomery curve."""
    t1 = (X + Z) % N
    t1 = t1 * t1 % N
    t2 = (X - Z) % N
    t2 = t2 * t2 % N
    t3 = (t1 - t2) % N
    X2 = t1 * t2 % N
    Z2 = t3 * ((t2 + a24 * t3) % N) % N
    w.add("mulmod", 5)
    return X2, Z2


def xadd(XP, ZP, XQ, ZQ, Xd, Zd, N, w: Work):
    """x-only differential addition: P + Q given P - Q = (Xd : Zd)."""
    u = (XP - ZP) * (XQ + ZQ) % N
    v = (XP + ZP) * (XQ - ZQ) % N
    s = (u + v) % N
    d = (u - v) % N
    X = Zd * (s * s % N) % N
    Z = Xd * (d * d % N) % N
    w.add("mulmod", 6)
    return X, Z


def ladder(k, X, Z, a24, N, w: Work):
    """Montgomery ladder: the x-coordinate of k*(X : Z), k >= 1."""
    k = int(k)
    if k == 1:
        return X, Z
    R0 = (X, Z)
    R1 = xdbl(X, Z, a24, N, w)
    for bit in bin(k)[3:]:
        if bit == "1":
            R0 = xadd(R1[0], R1[1], R0[0], R0[1], X, Z, N, w)
            R1 = xdbl(R1[0], R1[1], a24, N, w)
        else:
            R1 = xadd(R0[0], R0[1], R1[0], R1[1], X, Z, N, w)
            R0 = xdbl(R0[0], R0[1], a24, N, w)
    return R0


def stage1_exponents(B1: int):
    """Prime powers l^e <= B1, one per prime l <= B1."""
    out = []
    for l in small_primes(int(B1) + 1):
        pe = l
        while pe * l <= B1:
            pe *= l
        out.append(pe)
    return out


@register("ecm", primary_key="mulmod",
          description="ECM stage 1 on Montgomery curves (Suyama), bound B1, up to `curves` random curves",
          deterministic=False)
def ecm(N, B1=1000, curves=20, seed=0, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    if N % 2 == 0:
        return success("ecm", N, 2, w, "mulmod", time.perf_counter() - t0)
    rng = random.Random(seed)
    exps = stage1_exponents(int(B1))
    for ci in range(int(curves)):
        sigma = rng.randrange(6, 2 ** 31)
        curve, g = suyama_curve(sigma, N)
        w.add("curve")
        if curve is None:
            if 1 < g < N:
                return success("ecm", N, g, w, "mulmod", time.perf_counter() - t0,
                               curve=ci, sigma=sigma, B1=int(B1))
            continue
        a24, X, Z = curve
        for pe in exps:
            X, Z = ladder(pe, X, Z, a24, N, w)
        g = gcd(Z, N)
        w.add("gcd")
        if 1 < g < N:
            return success("ecm", N, g, w, "mulmod", time.perf_counter() - t0,
                           curve=ci, sigma=sigma, B1=int(B1))
    return failure("ecm", N, w, "mulmod", time.perf_counter() - t0, B1=int(B1), curves=int(curves))
