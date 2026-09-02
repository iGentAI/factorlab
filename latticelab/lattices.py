"""Lattice constructions and the Gaussian heuristic.

Bases are fpylll IntegerMatrix objects (rows are basis vectors).  Families:
  qary(d, k, q, seed)      -- a random q-ary lattice of dimension d and index q^k (LWE/SIS type): rows [q I_k | 0 ; A | I_{d-k}]
  knapsack(d, bits, seed)  -- the classical knapsack lattice of d random `bits`-bit integers (dimension d)
  ntru(n, q, seed)         -- the NTRU module lattice of dimension 2n over Z[X]/(X^n + 1), n a power of two: rows [q I | 0 ; rot(h) | I]
                              with h = g f^{-1} mod (q, X^n + 1), f, g ternary; contains the 2n rotations of (g, f) of norm ~ sqrt(4n/3)
All constructions are seeded and reproducible.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
from fpylll import GSO, IntegerMatrix, LLL


def qary(d: int, k: int, q: int, seed: int) -> IntegerMatrix:
    """Random q-ary lattice: the first k rows q e_i, the last d - k rows (a_i | e_i) with a_i uniform in [0, q)^k."""
    rng = random.Random(seed)
    A = IntegerMatrix(d, d)
    for i in range(k):
        A[i, i] = q
    for i in range(k, d):
        for j in range(k):
            A[i, j] = rng.randrange(q)
        A[i, i] = 1
    return A


def knapsack(d: int, bits: int, seed: int) -> IntegerMatrix:
    """Knapsack lattice: rows (a_i, e_i) for random `bits`-bit a_i, in dimension d (the first column carries the weights)."""
    rng = random.Random(seed)
    A = IntegerMatrix(d, d + 1)
    for i in range(d):
        A[i, 0] = rng.getrandbits(bits) | (1 << (bits - 1))
        A[i, i + 1] = 1
    return A


def _poly_inverse_mod(f: List[int], n: int, q: int) -> List[int] | None:
    """Inverse of f in Z_q[X]/(X^n + 1) for prime q, by the extended Euclidean algorithm over GF(q)[X]; None if not invertible."""
    from flint import nmod_poly

    F = nmod_poly(f, q)
    M = nmod_poly([1] + [0] * (n - 1) + [1], q)
    g, s, t = F.xgcd(M)  # s F + t M = g
    if g.degree() != 0:
        return None
    inv = (s * nmod_poly([pow(int(g[0]), -1, q)], q)) % M
    return [int(inv[i]) for i in range(n)]


def ntru(n: int, q: int, seed: int) -> Tuple[IntegerMatrix, List[int], List[int]]:
    """NTRU module lattice of dimension 2n: rows [q I_n | 0 ; rot(h) | I_n], rot(h)_{i, j} the coefficient of X^j in X^i h mod X^n + 1.
    Returns (basis, f, g) with f, g ternary, f invertible mod q.  Requires q an odd prime and n a power of two."""
    from gmpy2 import is_prime

    if n <= 0 or n & (n - 1) != 0:
        raise ValueError(f"n must be a positive power of two (got {n})")
    if q < 3 or not is_prime(q):
        raise ValueError(f"q must be an odd prime (got {q})")
    rng = random.Random(seed)
    while True:
        f = [rng.choice((-1, 0, 1)) for _ in range(n)]
        g = [rng.choice((-1, 0, 1)) for _ in range(n)]
        finv = _poly_inverse_mod([c % q for c in f], n, q)
        if finv is not None and any(g):
            break
    # h = g * finv mod (q, X^n + 1)
    h = [0] * n
    for i, gi in enumerate(g):
        if gi == 0:
            continue
        for j, fj in enumerate(finv):
            k = i + j
            sgn = 1
            if k >= n:
                k -= n
                sgn = -1
            h[k] = (h[k] + sgn * gi * fj) % q
    B = IntegerMatrix(2 * n, 2 * n)
    for i in range(n):
        B[i, i] = q
    for i in range(n):
        # row n + i: X^i h in the first n coordinates, e_i in the last n
        for j in range(n):
            k = i + j
            sgn = 1
            if k >= n:
                k -= n
                sgn = -1
            B[n + i, k] = (sgn * h[j]) % q
        B[n + i, n + i] = 1
    return B, f, g


def log_volume(A: IntegerMatrix) -> float:
    """log of the lattice volume from the Gram-Schmidt norms (exact up to floating point)."""
    M = GSO.Mat(A)
    M.update_gso()
    return 0.5 * sum(math.log(M.get_r(i, i)) for i in range(A.nrows))


def gaussian_heuristic(d: int, log_vol: float) -> float:
    """GH(d, vol) = (vol / V_d)^{1/d}, V_d the volume of the unit d-ball: the expected length of the shortest vector of a random
    lattice of dimension d and volume vol."""
    log_Vd = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)
    return math.exp((log_vol - log_Vd) / d)


def gh_count(d: int, log_vol: float, R: float) -> float:
    """Gaussian-heuristic count of nonzero lattice vectors of norm <= R: V_d R^d / vol."""
    log_Vd = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)
    return math.exp(log_Vd + d * math.log(R) - log_vol)


def cap_fraction(d: int, theta: float) -> float:
    """Fraction of the unit sphere S^{d-1} within angle theta of a fixed point (exact, by the regularised incomplete beta function):
    I_{sin^2 theta}((d-1)/2, 1/2) / 2 for theta <= pi/2."""
    from scipy.special import betainc

    assert 0 < theta <= math.pi / 2
    return 0.5 * betainc((d - 1) / 2, 0.5, math.sin(theta) ** 2)


def lll(A: IntegerMatrix) -> IntegerMatrix:
    B = IntegerMatrix.from_matrix(A)
    LLL.reduction(B)
    return B


def to_numpy(A: IntegerMatrix) -> np.ndarray:
    return np.array([[int(A[i, j]) for j in range(A.ncols)] for i in range(A.nrows)], dtype=object)


def negacyclic_rotate(v, k: int):
    """X^k acting on a vector of Z^{2n} viewed as a pair (g, f) of elements of Z[X]/(X^n + 1): each half is rotated negacyclically by k
    ((X h)_0 = -h_{n-1}, (X h)_i = h_{i-1}).  X has order 2n (X^n = -1), so the orbit of a nonzero vector under <X> has exactly 2n
    elements: X^k - 1 is invertible over Q unless 2n divides k, the eigenvalues of X being primitive 2n-th roots of unity."""
    v = [int(x) for x in v]
    if len(v) % 2:
        raise ValueError("need an even-length vector (two ring elements)")
    n = len(v) // 2
    out = []
    for half in (0, 1):
        h = v[half * n:(half + 1) * n]
        for i in range(n):
            src = (i - k) % (2 * n)
            out.append(h[src] if src < n else -h[src - n])
    return out


def orbit_angles(v, theta: float = math.pi / 3) -> Dict:
    """The exact inner products c_k = <v, X^k v>, 0 < k < 2n (negacyclic autocorrelations of the pair; c_{k+n} = -c_k), the largest
    |cosine| within the orbit, and the number of ordered pairs (X^a v, X^b v), a != b, within angle theta of each other -- the angular
    excess the 2n-element orbit forces on a G-invariant list, normalised by (2n)^2 cap_{2n}(theta).  For theta = 60 degrees the count is
    exact (the integer test 2|c_k| >= ||v||^2); otherwise it compares with a floating-point cosine."""
    v = [int(x) for x in v]
    n2 = len(v)
    n = n2 // 2
    if n < 2:
        raise ValueError("need n >= 2: for n = 1 the orbit of v is {v, -v} and there is no non-antipodal rotation")
    if not (0 < theta <= math.pi / 2):
        raise ValueError("theta must lie in (0, pi/2]")
    norm2 = sum(x * x for x in v)
    if norm2 == 0:
        raise ValueError("zero vector")
    cs = [sum(a * b for a, b in zip(v, negacyclic_rotate(v, k))) for k in range(1, 2 * n)]
    if any(cs[k - 1] != -cs[k + n - 1] for k in range(1, n)) or cs[n - 1] != -norm2:
        raise AssertionError("c_{k+n} = -c_k or c_n = -||v||^2 failed")
    # angle(v, X^k v) <= theta  iff  c_k >= ||v||^2 cos(theta) (signed: the antipode -v = X^n v is at 180 degrees, not a close pair);
    # for each k exactly one of X^k v, -X^k v can be within theta < 90 degrees.  Exact integer tests at 60 and 90 degrees.
    if abs(math.cos(theta) - 0.5) < 1e-15:
        close = sum(1 for c in cs if 2 * c >= norm2)
    elif abs(theta - math.pi / 2) < 1e-15:
        close = sum(1 for c in cs if c >= 0)
    else:
        close = sum(1 for c in cs if c >= norm2 * math.cos(theta))
    ordered_pairs_close = 2 * n * close  # every orbit element has `close` partners within theta (the orbit is a single G-orbit)
    cap = cap_fraction(n2, theta)
    max_abs_cos = max(abs(c) for c in cs[:n - 1]) / norm2  # over X^k v, 0 < k < n (k = n is the antipode; -X^k v covers the sign)
    return {"n": n, "norm2": norm2, "autocorrelations": cs[:n - 1], "max_abs_cos": max_abs_cos,
            "min_angle_deg": math.degrees(math.acos(min(1.0, max_abs_cos))), "pairs_within_theta": ordered_pairs_close,
            "angular_excess": ordered_pairs_close / ((2 * n) ** 2 * cap), "cap_fraction": cap,
            "pairs_within_theta_uniform_expectation": (2 * n) * (2 * n - 1) * cap}
