"""BKZ Gram-Schmidt profiles against the Geometric Series Assumption (Layer A of docs/lattice_barrier_plan.md).

For a reduced basis with Gram-Schmidt norms ||b_i^*||, the GSA says log ||b_i^*|| is linear in i with slope -2 log delta_beta,
delta_beta = ((beta/(2 pi e)) (pi beta)^{1/beta})^{1/(2(beta-1))} (the Chen-Nguyen form, an asymptotic constant meaningful only for
beta of a few tens or more).  We record the profile, the least-squares slope and its implied delta_fit = exp(-slope/2),
first_ratio = ||b_1|| / vol^{1/d} (under a linear profile this equals delta^{d-1}), the root-Hermite factor
root_hermite = first_ratio^{1/(d-1)} in that convention, the deviation from the fitted line (how linear the profile is), and for
beta >= 40 the deviation from the GSA line through the volume.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from fpylll import BKZ, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction


def delta_gsa(beta: int) -> float:
    """Root-Hermite factor predicted for BKZ-beta by the Gaussian heuristic on blocks (Chen-Nguyen)."""
    return ((beta / (2 * math.pi * math.e)) * (math.pi * beta) ** (1 / beta)) ** (1 / (2 * (beta - 1)))


def _gso(A: IntegerMatrix, float_type: str | None = None) -> GSO.Mat:
    """A Gram-Schmidt object on a copy of A; double precision below d = 150 and long double from there (fpylll's double-precision GSO and
    LLL fail on the 225-dimensional q-ary bases of the census: 'infinite loop in babai', non-positive r_ii), unless `float_type` is given."""
    ft = float_type if float_type is not None else ("ld" if A.nrows >= 150 else "d")
    M = GSO.Mat(IntegerMatrix.from_matrix(A), float_type=ft)
    M.update_gso()
    return M


def gs_profile(A: IntegerMatrix, float_type: str | None = None) -> np.ndarray:
    """log ||b_i^*|| for the basis A."""
    M = _gso(A, float_type)
    return np.array([0.5 * math.log(M.get_r(i, i)) for i in range(A.nrows)])


def bkz(A: IntegerMatrix, beta: int, tours: int | None = None) -> IntegerMatrix:
    """BKZ 2.0 reduction (fpylll) with default strategies; `tours` bounds the number of tours (None: auto-abort)."""
    B = IntegerMatrix.from_matrix(A)
    LLL.reduction(B)
    if beta <= 2:
        return B
    flags = BKZ.AUTO_ABORT | BKZ.GH_BND
    kwargs = {}
    if tours is not None:
        flags |= BKZ.MAX_LOOPS
        kwargs["max_loops"] = tours
    try:
        par = BKZ.Param(block_size=beta, strategies=BKZ.DEFAULT_STRATEGY, flags=flags, **kwargs)
    except RuntimeError:  # the pip wheel may not ship the pruning-strategy file: unpruned enumeration
        par = BKZ.Param(block_size=beta, flags=flags, **kwargs)
    BKZReduction(B)(par)
    return B


def block_gh_ratios(A: IntegerMatrix, beta: int, max_blocks: int | None = None, float_type: str | None = None) -> Dict:
    """For each block B_i = pi_i(L(b_i..b_{i+beta_i-1})), beta_i = min(beta, d-i+1), of the basis A: the exact lambda_1(B_i) (unpruned
    enumeration on the projected block), its Gaussian-heuristic length GH(B_i) = chat(beta_i) vol(B_i)^{1/beta_i}, and the ratio
    ||b_i^*|| / lambda_1(B_i).  The admissibility parameter of the profile floor is eps = max_i log(GH(B_i)/lambda_1(B_i))^+ (0 if every
    block's shortest vector is at least its GH length).  Blocks of size < 2 are skipped; `max_blocks` limits the number of blocks (from
    the head) for cost."""
    from fpylll import Enumeration, EnumerationError

    from latticelab.profile_floor import log_chat

    d = A.nrows
    if beta < 2 or d < 2:
        raise ValueError("need beta >= 2 and a basis of dimension >= 2")
    if max_blocks is not None and max_blocks < 0:
        raise ValueError("max_blocks must be nonnegative")
    M = _gso(A, float_type)
    r = [M.get_r(i, i) for i in range(d)]
    rows = []
    n_blocks = d - 1 if max_blocks is None else min(d - 1, max_blocks)
    for i in range(n_blocks):
        bi = min(beta, d - i)
        if bi < 2:
            continue
        log_vol = 0.5 * sum(math.log(r[j]) for j in range(i, i + bi))
        gh = math.exp(log_chat(bi) + log_vol / bi)
        radius2 = (1.05 * gh) ** 2  # search a little beyond GH; enlarge until a vector is found
        lam2 = None
        while lam2 is None:
            try:
                sols = Enumeration(M).enumerate(i, i + bi, radius2, 0, pruning=None)
                lam2 = min(s[0] for s in sols)
            except EnumerationError:
                radius2 *= 1.5
        lam = math.sqrt(lam2)
        rows.append({"i": i + 1, "beta_i": bi, "lambda1": lam, "gh": gh, "log_gh_over_lambda1": math.log(gh / lam),
                     "b_star_over_lambda1": math.sqrt(r[i]) / lam})
    if not rows:
        return {"d": d, "beta": beta, "blocks": [], "eps_needed": 0.0, "worst_block": None, "mean_log_gh_over_lambda1": float("nan"),
                "frac_blocks_with_bstar_shortest": float("nan")}
    eps_needed = max(0.0, max(row["log_gh_over_lambda1"] for row in rows))
    return {"d": d, "beta": beta, "blocks": rows, "eps_needed": eps_needed,
            "worst_block": max(rows, key=lambda row: row["log_gh_over_lambda1"])["i"],
            "mean_log_gh_over_lambda1": float(np.mean([row["log_gh_over_lambda1"] for row in rows])),
            "frac_blocks_with_bstar_shortest": float(np.mean([abs(row["b_star_over_lambda1"] - 1) < 1e-9 for row in rows]))}


def profile_stats(A: IntegerMatrix, beta: int, gsa_min_beta: int = 40) -> Dict:
    """Profile, fitted slope and its implied delta, root-Hermite factor, deviation from the fitted line (linearity of the profile),
    and -- only for beta >= gsa_min_beta, where the asymptotic Chen-Nguyen constant is meaningful -- the deviation from the GSA line."""
    d = A.nrows
    p = gs_profile(A)
    log_vol = float(p.sum())
    i = np.arange(d)
    slope, intercept = np.polyfit(i, p, 1)
    fit_line = slope * i + intercept
    first_ratio = math.exp(p[0] - log_vol / d)  # ||b_1|| / vol^{1/d}
    # under a linear profile of slope -2 log delta through the volume, ||b_1|| / vol^{1/d} = delta^{d-1}
    rhf = math.exp((p[0] - log_vol / d) / (d - 1)) if d > 1 else float("nan")
    out = {"d": d, "beta": beta, "profile": p.tolist(), "slope": float(slope), "delta_fit": math.exp(-slope / 2),
           "first_ratio": first_ratio, "root_hermite": rhf,
           "max_dev_fit": float(np.max(np.abs(p - fit_line))), "head_dev_fit": float(p[0] - fit_line[0]),
           "tail_dev_fit": float(p[-1] - fit_line[-1])}
    if beta >= gsa_min_beta:
        delta = delta_gsa(beta)
        gsa_slope = -2 * math.log(delta)
        c = (log_vol - gsa_slope * i.sum()) / d  # GSA line with the correct volume
        gsa_line = gsa_slope * i + c
        out.update({"delta_gsa": delta, "gsa_slope": gsa_slope, "first_ratio_gsa": delta ** (d - 1),
                    "max_dev_gsa": float(np.max(np.abs(p - gsa_line)))})
    return out
