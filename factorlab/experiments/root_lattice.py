"""E15: the coefficient floor of known-root polynomials.

An NFS-type algorithm needs a polynomial f of degree d together with a root
m of f modulo N (the ring homomorphism Z[alpha] -> Z/N).  The coefficient
vectors of all such f form the lattice
    L_{N,m} = {(f_0, ..., f_d) in Z^{d+1} : sum f_i m^i = 0 (mod N)},
of determinant N.  Minkowski's theorem gives a nonzero vector of sup-norm
<= N^{1/(d+1)}; the Gaussian heuristic for a lattice of this determinant and
dimension puts the shortest vector at about sqrt((d+1)/(2 pi e)) N^{1/(d+1)},
and for generic (N, m) nothing much shorter exists.  Since the algebraic norm
of a - b alpha is F(a, b) = sum f_i a^i b^{d-i}, the auxiliary numbers of any
known-root NFS are at least of size ~ N^{1/(d+1)} A^d on the algebraic side and
A N^{1/d} on the rational side, which after balancing gives L_N[2/3] and the
exponent 1/3.  The exception is N of special form (SNFS): N = 2^k - c has the
polynomial x^d - c with root 2^{k/d}, far below the floor.

This module computes the *certified* shortest vector of L_{N,m} (LLL followed
by Fincke-Pohst enumeration, exact in dimension d+1 <= 7) with PARI, for
random RSA-style N with random m and with m = round(N^{1/d}) (the base-m
choice), and for SNFS-form N, and reports it relative to N^{1/(d+1)}.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np

from ..gen import make_semiprime
from ..numth import mpz, iroot

_pari = None


def pari():
    global _pari
    if _pari is None:
        import cypari2
        _pari = cypari2.Pari()
        _pari.allocatemem(1024 * 1024 * 1024)
    return _pari


def root_lattice_columns(N: int, m: int, d: int) -> list[list[int]]:
    """Basis of L_{N,m} as column vectors: (N, 0, ..., 0) and e_i - m^i e_0."""
    cols = [[N] + [0] * d]
    for i in range(1, d + 1):
        col = [0] * (d + 1)
        col[0] = -pow(m, i, N)
        col[i] = 1
        cols.append(col)
    return cols


def lll_shortest(N: int, m: int, d: int) -> tuple[int, list[int]]:
    """Shortest vector of the LLL-reduced basis of L_{N,m} (squared L2 norm, vector).

    An upper bound on the true minimum; kept for comparison with exact_shortest.
    """
    P = pari()
    cols = root_lattice_columns(N, m, d)
    n = d + 1
    entries = [cols[j][i] for i in range(n) for j in range(n)]  # row-major
    M = P.matrix(n, n, entries)
    T = P.qflll(M)
    R = M * T
    best = None
    for j in range(n):
        v = [int(R[i, j]) for i in range(n)]
        nv = sum(x * x for x in v)
        if nv > 0 and (best is None or nv < best[0]):
            best = (nv, v)
    return best


def exact_shortest(N: int, m: int, d: int) -> tuple[int, list[int]]:
    """Certified shortest nonzero vector of L_{N,m} (squared L2 norm, vector).

    LLL-reduces the basis, then runs Fincke-Pohst enumeration (PARI qfminim,
    flag 2 for large integer entries) on the Gram matrix.  Exact in the
    dimensions used here (d + 1 <= 7).
    """
    P = pari()
    cols = root_lattice_columns(N, m, d)
    n = d + 1
    entries = [cols[j][i] for i in range(n) for j in range(n)]  # row-major
    M = P.matrix(n, n, entries)
    R = M * P.qflll(M)
    G = R.mattranspose() * R
    cnt, min_norm, vecs = P.qfminim(G, None, None, 2)
    if int(cnt) == 0:
        raise RuntimeError(f"qfminim returned no minimal vectors for N={N}, m={m}, d={d}")
    v_coeff = [int(vecs[i, 0]) for i in range(n)]
    v = [sum(int(R[i, j]) * v_coeff[j] for j in range(n)) for i in range(n)]
    return int(min_norm), v


def gaussian_constant(n: int) -> float:
    """Exact Gaussian-heuristic constant V_n^{-1/n} = Gamma(n/2+1)^{1/n} / sqrt(pi)."""
    return math.gamma(n / 2 + 1) ** (1.0 / n) / math.sqrt(math.pi)


def root_lattice_experiment(bits: Sequence[int] = (64, 96, 128), ds: Sequence[int] = (2, 3, 4, 5, 6),
                            count: int = 20, seed: int = 97) -> dict:
    rng = random.Random(seed)
    out = {"gaussian_heuristic": {str(d): gaussian_constant(d + 1) for d in ds}, "rows": []}
    for nbits in bits:
        for d in ds:
            ratios_rand, ratios_basem, ratios_snfs = [], [], []
            for i in range(count):
                inst = make_semiprime(nbits, "rsa", seed, i)
                N = int(inst.N)
                floor = N ** (1.0 / (d + 1))
                m_rand = rng.randrange(2, N - 1)
                nv, _ = exact_shortest(N, m_rand, d)
                ratios_rand.append(math.sqrt(nv) / floor)
                m_base = int(iroot(mpz(N), d)[0])
                nv, _ = exact_shortest(N, m_base, d)
                ratios_basem.append(math.sqrt(nv) / floor)
                # SNFS form: N' = 2^k - c, k a multiple of d, m = 2^{k/d}, so x^d - c has root m
                k = d * ((nbits + d - 1) // d)
                c = rng.randrange(3, 1000, 2)
                Ns = (1 << k) - c
                nv, v = exact_shortest(Ns, 1 << (k // d), d)
                ratios_snfs.append(math.sqrt(nv) / Ns ** (1.0 / (d + 1)))
            out["rows"].append({
                "nbits": nbits, "d": d,
                "random_m_ratio_mean": float(np.mean(ratios_rand)), "random_m_ratio_min": float(np.min(ratios_rand)),
                "base_m_ratio_mean": float(np.mean(ratios_basem)), "base_m_ratio_min": float(np.min(ratios_basem)),
                "snfs_ratio_mean": float(np.mean(ratios_snfs)),
                "gaussian": gaussian_constant(d + 1),
            })
    return out
