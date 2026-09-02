"""E23b: the cost-quality frontier of free-root polynomial selection (linear pair).

For the standard NFS pair (f of degree d with root m, g = x - m) the unskewed
figure of merit is the norm product P = ||f||_oo * m (E18, notes_beyond_gnfs
section 2b).  Two constructions at a trial budget T are compared here.

  * Base-m leading-coefficient window ("Kleinjung scale" below refers only to the
    coefficient scale a_d ~ m ~ N^{1/(d+1)} shared with practical selection): a_d runs
    through T consecutive integers near N^{1/(d+1)},
    m = round((N/a_d)^{1/d}) and the lower coefficients are the balanced base-m
    digits of N - a_d m^d.  The digit a_{d-1} is automatically O(a_d) and every
    coefficient is O(m), so P ~ a_d m ~ N^{2/(d+1)} whatever T is: the leading
    coefficient dominates the sup-norm and more trials cannot lower it.  (This is
    the raw base-m scale, not a model of Kleinjung's algorithm or of practical
    size optimisation; nothing about those methods follows from this experiment.)
  * Free-root search (E18's leading-coefficient search): a_d = 1, ..., T with
    the same digit construction.  With a_d ~ T the root is m ~ (N/T)^{1/d} and
    the d-1 lower digits behave like independent uniforms on (-m/2, m/2]; the
    best of T trials has H ~ (m/2) T^{-1/(d-1)} and
        P(T) ~ max(T, H) * m .
    P decreases with T until H = T, i.e. until T = N^{h*}, h* = (d-1)/(d^2+d-1),
    where P reaches the joint floor N^{(2d-1)/(d^2+d-1)} (Proposition J).
    Matching the Kleinjung scale P ~ N^{2/(d+1)} needs
        T_x = N^{2(d-1)/((d+1)(3d-2))}
    trials (N^{1/7}, N^{0.12}, N^{0.103}, N^{0.089} for d = 3, 4, 5, 6).

Skew.  Over |a| <= A sqrt(s), |b| <= A / sqrt(s) the algebraic norm is at most
A^d sum_i |f_i| s^{i-d/2} and the rational norm at most A (sqrt(s) + m/sqrt(s)); the
skew-optimised product Q(f, m) = min_s (sum_i |f_i| s^{i-d/2}) (sqrt(s) + m/sqrt(s))
divided by A^{d+1} is the second figure of merit reported.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..gen import make_semiprime
from ..numth import mpz, iroot
from .poly_floor import _balanced_digits, is_irreducible, floor_exponents


def crossover_exponent(d: int) -> float:
    """log_N of the trial budget at which the free-root search matches P ~ N^{2/(d+1)}."""
    return 2.0 * (d - 1) / ((d + 1) * (3 * d - 2))


def theory_P(N: float, d: int, T: float) -> float:
    """Heuristic frontier of the free-root search: max(T, (m/2) T^{-1/(d-1)}) * m, m = (N/T)^{1/d}."""
    m = (N / T) ** (1.0 / d)
    H = 0.5 * m * T ** (-1.0 / (d - 1))
    return max(T, H) * m


def skewed_product(f: Sequence[int], m: int, grid: int = 241) -> tuple[float, float]:
    """min over s in [1, m] of (sum |f_i| s^{i-d/2}) (sqrt(s) + m/sqrt(s)); returns (Q, s).

    Evaluated on a geometric grid of ``grid`` points in the log domain; the grid
    minimum is an upper bound on the continuous one (within the grid spacing).
    """
    d = len(f) - 1
    lf = np.array([math.log(abs(int(c))) if c != 0 else -math.inf for c in f])
    ls = np.linspace(0.0, math.log(float(m)), grid)
    lm = math.log(float(m))
    expo = (np.arange(d + 1) - d / 2.0)[None, :] * ls[:, None] + lf[None, :]
    alg = np.logaddexp.reduce(expo, axis=1)
    rat = np.logaddexp(0.5 * ls, lm - 0.5 * ls)
    v = alg + rat
    j = int(np.argmin(v))
    return float(math.exp(v[j])), float(math.exp(ls[j]))


def _candidate(N: int, d: int, ad: int):
    """Best (P, f, m) over the two nearest roots for leading coefficient ad, or None."""
    m0 = int(iroot(mpz(N // ad), d)[0])
    best = None
    for m in (m0, m0 + 1):
        if m < 2:
            continue
        digits, left = _balanced_digits(N - ad * m ** d, m, d)
        if left != 0:
            continue
        f = digits + [ad]
        P = max(abs(c) for c in f) * m
        if best is None or P < best[0]:
            best = (P, f, m)
    return best


def frontier(N: int, d: int, a_start: int, T_max: int, checkpoints: Sequence[int]) -> dict:
    """Running minima of P and Q over a_d = a_start, ..., a_start + T_max - 1 at the checkpoints."""
    N = int(N)
    bestP, bestQ = None, None
    outP, outQ, outHm = [], [], []
    cps = sorted(set(int(c) for c in checkpoints if c <= T_max))
    ci = 0
    for t in range(1, T_max + 1):
        cand = _candidate(N, d, a_start + t - 1)
        if cand is not None:
            P, f, m = cand
            if (bestP is None or P < bestP[0]) and is_irreducible(f):
                bestP = (P, f, m)
            Q, s = skewed_product(f, m)  # evaluated for every candidate: no pruning by P
            if (bestQ is None or Q < bestQ[0]) and is_irreducible(f):
                bestQ = (Q, f, m, s)
        while ci < len(cps) and t == cps[ci]:
            outP.append(None if bestP is None else math.log2(bestP[0]))
            outQ.append(None if bestQ is None else math.log2(bestQ[0]))
            outHm.append(None if bestP is None else [math.log2(max(abs(c) for c in bestP[1])), math.log2(bestP[2])])
            ci += 1
    return {"checkpoints": cps, "log2_P": outP, "log2_Q": outQ, "log2_H_m": outHm,
            "best_P": None if bestP is None else {"P": str(bestP[0]), "f": [str(c) for c in bestP[1]], "m": str(bestP[2])},
            "best_Q": None if bestQ is None else {"Q": bestQ[0], "f": [str(c) for c in bestQ[1]], "m": str(bestQ[2]), "skew": bestQ[3]}}


def frontier_experiment(d: int, bits: Sequence[int], count: int, T_max: int = 1 << 18, seed: int = 23,
                        family: str = "rsa") -> dict:
    cps = [1 << k for k in range(2, T_max.bit_length())]
    if cps[-1] != T_max:
        cps.append(T_max)
    ex = floor_exponents(d)
    rows = []
    for nbits in bits:
        per = []
        for i in range(count):
            inst = make_semiprime(nbits, family, seed, i)
            N = int(inst.N)
            fr = frontier(N, d, 1, T_max, cps)
            A0 = max(1, int(round(float(N) ** (1.0 / (d + 1)))))
            kl = frontier(N, d, A0, T_max, cps)
            th = [math.log2(theory_P(float(N), d, float(T))) for T in fr["checkpoints"]]
            per.append({"N": str(N), "free": fr, "kleinjung_scale": kl, "theory_log2_P": th})
        ck = per[0]["free"]["checkpoints"]
        def col(key, sub):
            return [float(np.mean([r[sub][key][j] for r in per])) for j in range(len(ck))]
        logN = float(np.mean([math.log2(float(r["N"])) for r in per]))
        rows.append({"nbits": nbits, "count": count, "mean_log2_N": logN, "checkpoints": ck,
                     "free_log2_P": col("log2_P", "free"), "free_log2_Q": col("log2_Q", "free"),
                     "kleinjung_log2_P": col("log2_P", "kleinjung_scale"), "kleinjung_log2_Q": col("log2_Q", "kleinjung_scale"),
                     "theory_log2_P": [float(np.mean([r["theory_log2_P"][j] for r in per])) for j in range(len(ck))],
                     "log2_base_m_scale": 2.0 / (d + 1) * logN,
                     "log2_joint_floor": ex["product"] * logN,
                     "log2_T_floor": ex["coeff"] * logN, "log2_T_crossover": crossover_exponent(d) * logN,
                     "instances": per})
    return {"d": d, "T_max": T_max, "crossover_exponent": crossover_exponent(d),
            "floor_exponent": ex["product"], "floor_trials_exponent": ex["coeff"], "rows": rows}
