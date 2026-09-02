"""E17: a Dickman-based cost calculator for the relation paradigm.

The size-exponent principle: if an algorithm must find B-smooth values among
auxiliary integers of size X = L_N[beta, .], then with B = L_N[beta/2, .] the
smoothness probability is L_N[beta/2, .]^{-1}, the number of relations needed
is ~B, and the cost is L_N[beta/2, .].  Random squares and the quadratic sieve
have X ~ N^{1/2 + o(1)} (beta = 1), hence L[1/2]; the number field sieve has
X ~ N^{2/d} A^{d+1} with d ~ (log N / log log N)^{1/3}, i.e. X = L[2/3], hence
L[1/3].  This module evaluates the crude cost model numerically:

* QS: Q(x) ~ 2 |x| sqrt(N) for |x| <= M; relations needed pi(B); cost M + pi(B)^2;
* NFS(d): rational side |a - b m| <= A N^{1/d}, algebraic side
  |F(a,b)| <= (d+1) N^{1/d} A^d (base-m coefficients), A^2 points, relations
  needed 2 pi(B), independent smoothness, cost A^2 + (2 pi(B))^2;

and reports the optimal parameters, the crossover, and least-squares fits of
log(cost) to (ln N)^{1/2}(ln ln N)^{1/2} and (ln N)^{1/3}(ln ln N)^{2/3}.
Constants are not meaningful (no sieve speed, sparse linear algebra, large
primes, skew); exponents are.  It also tabulates the size of the pure-power
family |a^d - N b^d| along Dirichlet approximants of N^{1/d}, which is
~ d N^{(d-1)/d} b^{d-2}: best at d = 2 (CFRAC/QS), worse for every d > 2.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _lnrho(u: float) -> float:
    """ln rho(u) from the table, or de Bruijn's asymptotic -u (ln u + ln ln u - 1) beyond it."""
    from .smooth_profiles import dickman_rho, _RHO_UMAX
    if u < _RHO_UMAX - 1:
        r = dickman_rho(u)
        if r > 1e-300:
            return math.log(r)
    return -u * (math.log(u) + math.log(math.log(u)) - 1.0)


def _pi(B: float) -> float:
    return B / math.log(B)


def qs_cost(nbits: int) -> dict:
    lnN = nbits * math.log(2)
    best = None
    for lnB in np.linspace(math.log(50), lnN / 2, 400):
        lnR = lnB - math.log(lnB)  # ln pi(B)
        # solve ln M = ln R - ln rho((lnN/2 + ln 2 + ln M)/lnB) by fixed point
        lnM = lnR
        for _ in range(80):
            u = (lnN / 2 + math.log(2) + lnM) / lnB
            lnM_new = lnR - _lnrho(u)
            if abs(lnM_new - lnM) < 1e-9:
                break
            lnM = 0.5 * (lnM + lnM_new)
        ln_cost = float(np.logaddexp(lnM, 2 * lnR))
        if best is None or ln_cost < best["ln_cost"]:
            best = {"ln_cost": ln_cost, "log2_cost": ln_cost / math.log(2), "lnB": lnB, "log2_B": lnB / math.log(2),
                    "log2_M": lnM / math.log(2), "u": u}
    return best


def nfs_cost(nbits: int, d: int) -> dict:
    lnN = nbits * math.log(2)
    best = None
    for lnB in np.linspace(math.log(50), lnN / 3, 400):
        lnR = math.log(2) + lnB - math.log(lnB)  # ln 2 pi(B)
        lnA = lnR / 2
        for _ in range(100):
            ur = (lnN / d + lnA) / lnB
            ua = (math.log(d + 1) + lnN / d + d * lnA) / lnB
            lnpr = _lnrho(ur) + _lnrho(ua)
            lnA_new = 0.5 * (lnR - lnpr)
            if abs(lnA_new - lnA) < 1e-9:
                break
            lnA = 0.5 * (lnA + lnA_new)
        ln_cost = float(np.logaddexp(2 * lnA, 2 * lnR))
        if best is None or ln_cost < best["ln_cost"]:
            best = {"ln_cost": ln_cost, "log2_cost": ln_cost / math.log(2), "d": d, "log2_B": lnB / math.log(2),
                    "log2_A": lnA / math.log(2), "u_rational": ur, "u_algebraic": ua}
    return best


def paradigm_table(bits_list: Sequence[int] = (96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048),
                   ds: Sequence[int] = (2, 3, 4, 5, 6, 7, 8)) -> dict:
    rows = []
    for nb in bits_list:
        q = qs_cost(nb)
        nfs = min((nfs_cost(nb, d) for d in ds), key=lambda r: r["ln_cost"])
        rows.append({"nbits": nb, "qs_log2_cost": q["log2_cost"], "qs_log2_B": q["log2_B"], "qs_u": q["u"],
                     "nfs_log2_cost": nfs["log2_cost"], "nfs_d": nfs["d"], "nfs_log2_B": nfs["log2_B"],
                     "nfs_log2_A": nfs["log2_A"]})
    lnN = np.array([r["nbits"] * math.log(2) for r in rows])
    x_half = np.sqrt(lnN * np.log(lnN))
    x_third = lnN ** (1 / 3) * np.log(lnN) ** (2 / 3)
    y_qs = np.array([r["qs_log2_cost"] * math.log(2) for r in rows])
    y_nfs = np.array([r["nfs_log2_cost"] * math.log(2) for r in rows])
    fit = {
        "qs_vs_L_half_slope": float(np.polyfit(x_half, y_qs, 1)[0]),
        "qs_vs_L_third_slope": float(np.polyfit(x_third, y_qs, 1)[0]),
        "nfs_vs_L_third_slope": float(np.polyfit(x_third, y_nfs, 1)[0]),
        "nfs_vs_L_half_slope": float(np.polyfit(x_half, y_nfs, 1)[0]),
        "nfs_heuristic_constant": (64 / 9) ** (1 / 3),
    }
    cross = None
    for r in rows:
        if r["nfs_log2_cost"] < r["qs_log2_cost"]:
            cross = r["nbits"]
            break
    return {"rows": rows, "fits": fit, "first_bits_where_nfs_cheaper": cross}


def pure_power_sizes(nbits: int, ds: Sequence[int] = (2, 3, 4, 5, 6)) -> dict:
    """Exponent of |a^d - N b^d| ~ d N^{(d-1)/d} b^{d-2} for a Dirichlet approximant
    a/b of N^{1/d} with b = N^{1/6}, as a fraction of log N (QS residue: 1/2 + 1/6)."""
    out = {}
    for d in ds:
        out[str(d)] = (d - 1) / d + (d - 2) / 6.0
    out["qs_M_N16"] = 0.5 + 1 / 6
    return out
