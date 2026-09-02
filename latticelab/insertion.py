"""L6 -- the physical completion and the per-basis head identity (docs/notes_lattice_barrier.md, section 7).

Per-basis identity.  For any basis, l_i >= log lambda_1(B_i) = log chat(beta_i) - eps_i(B) + avg_{block i} l with eps_i(B) := log(GH(B_i)/lambda_1(B_i))
(signed), so the profile satisfies (A_i) with c_i = log chat(beta_i) - eps_i(B) and the dual certificate gives
        l_1 - S/d  >=  h_{d,beta}(0) - sum_i y_i eps_i(B),
with equality iff every b_i^* is the shortest vector of its block (a strictly BKZ-beta-reduced basis).  `weighted_subgh_mass` evaluates
both sides on a real basis (exact lambda_1 by unpruned enumeration, exact multipliers) and returns the residual (>= 0).

Physical insertion.  `single_insertion` performs one exact SVP insertion at block kappa of a basis with fpylll's basic BKZ machinery
(`fpylll.algorithms.bkz.BKZReduction.svp_reduction`: LLL on the block, unpruned enumeration with radius ||b_kappa^*||, insertion of the
solution, LLL on the block to complete the basis) and records the log-profile change Delta_r = l'_r - l_r: its support, the removed mass
-Delta_kappa, the cumulative flow C_t = sum_{r<=t} Delta_r (rightward flow <=> C_t <= 0), the admissibility violations (GH_j - l_j)^+ of
the profile before and after at the blocks overlapping the change, and the change in the exact per-block eps_j(B) there.  fpylll's
enumeration radius is delta_LLL * ||b_kappa^*||^2 with delta_LLL = 0.99, so a block whose shortest vector is within 0.5 % of ||b_kappa^*|| is
left unchanged: repeated insertions make a basis SVP-tight up to that slack (R(B) <= (-log 0.99 / 2) sum_i y_i), the standard BKZ convention.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from fpylll import BKZ, GSO, IntegerMatrix
from fpylll.algorithms.bkz import BKZReduction as BKZBase

from latticelab.profile import block_gh_ratios, gs_profile
from latticelab.profile_floor import block_sizes, dual_certificate, floor_l1_float, log_chat


def weighted_subgh_mass(A: IntegerMatrix, beta: int) -> Dict:
    """Both sides of  l_1 - S/d >= h(0) - sum_i y_i eps_i(B)  on the basis A at test blocksize beta, with the exact per-block eps_i(B)."""
    d = A.nrows
    st = block_gh_ratios(A, beta)
    eps = np.array([row["log_gh_over_lambda1"] for row in st["blocks"]])  # blocks i = 1..d-1 in order (all sizes >= 2)
    assert len(eps) == d - 1
    y, _ = dual_certificate(d, beta)
    yf = np.array([float(v) for v in y])
    p = gs_profile(A)
    S = float(p.sum())
    h0 = floor_l1_float(d, beta)["l1_floor"]
    lhs = float(p[0]) - S / d
    mass_signed = float(np.dot(yf, eps))
    mass_pos = float(np.dot(yf, np.maximum(eps, 0.0)))
    residual = lhs - (h0 - mass_signed)
    tight = np.array([abs(row["b_star_over_lambda1"] - 1) < 1e-9 for row in st["blocks"]])
    return {"d": d, "beta": beta, "head_minus_mean": lhs, "floor_h0": h0, "weighted_eps_signed": mass_signed, "weighted_eps_positive": mass_pos,
            "residual": residual, "frac_tight": float(tight.mean()), "eps_needed": st["eps_needed"], "eps_max_weighted_block": float(eps[np.argmax(yf * np.maximum(eps, 0))]) if mass_pos > 0 else 0.0,
            "y1_eps1": float(yf[0] * eps[0]), "eps": eps.tolist(), "y": yf.tolist()}


def single_insertion(A: IntegerMatrix, kappa: int, beta: int) -> Dict:
    """One exact SVP insertion at 0-based block position kappa (size min(beta, d - kappa)) on a copy of A; returns the new basis and the
    profile-change statistics.  Uses fpylll's basic (unpruned) BKZ svp_reduction, whose enumeration radius is delta_LLL * ||b_kappa^*||^2
    with delta_LLL = 0.99: the block's shortest vector is found and inserted whenever it is shorter than sqrt(0.99) ||b_kappa^*||, and the
    block is left unchanged when b_kappa^* is already within that 0.5 % slack of the block minimum."""
    d = A.nrows
    size = min(beta, d - kappa)
    if size < 2:
        raise ValueError("block of size < 2")
    B = IntegerMatrix.from_matrix(A)
    M = GSO.Mat(B)
    M.update_gso()
    p0 = gs_profile(A)
    bkz = BKZBase(M)
    par = BKZ.Param(block_size=size, max_loops=1)
    bkz.svp_reduction(kappa, size, par)
    p1 = gs_profile(B)
    delta = p1 - p0
    support = np.where(np.abs(delta) > 1e-9)[0]
    changed = len(support) > 0
    cum = np.cumsum(delta)
    out = {"kappa": kappa, "size": size, "changed": changed, "removed_mass": float(-delta[kappa]),
           "support_within_block": bool(support.size == 0 or (support.min() >= kappa and support.max() <= kappa + size - 1)),
           "sum_delta_block": float(delta[kappa:kappa + size].sum()),
           "cumulative_flow_max": float(cum[kappa:kappa + size - 1].max()) if changed else 0.0,  # > 0 means mass moved left within the block
           "cumulative_flow_min": float(cum[kappa:kappa + size - 1].min()) if changed else 0.0,
           "delta": delta[kappa:kappa + size].tolist()}
    return out, B


def insertion_census(A: IntegerMatrix, beta: int, kappas: List[int]) -> Dict:
    """Single insertions at each kappa (independently, each from the same starting basis A): the profile change, and the admissibility
    violations and exact per-block eps_j before and after on the blocks overlapping the change."""
    d = A.nrows
    before = block_gh_ratios(A, beta)
    eps0 = {row["i"] - 1: row["log_gh_over_lambda1"] for row in before["blocks"]}
    p0 = gs_profile(A)
    rows = []
    for kappa in kappas:
        r, B = single_insertion(A, kappa, beta)
        if not r["changed"]:
            rows.append({**r, "note": "no change"})
            continue
        lo, hi = max(0, kappa - beta + 1), min(d - 2, kappa + r["size"] - 1)
        after = block_gh_ratios(B, beta, max_blocks=hi + 1)
        eps1 = {row["i"] - 1: row["log_gh_over_lambda1"] for row in after["blocks"]}
        p1 = gs_profile(B)
        # profile-level violations (GH_j - l_j)^+ before and after at the overlapping blocks, their increase, and the exact eps_j change
        def viol(p, j):
            bj = min(beta, d - j)
            return max(0.0, log_chat(bj) + float(p[j:j + bj].mean()) - float(p[j]))

        viol_before = [viol(p0, j) for j in range(lo, hi + 1)]
        viol_after = [viol(p1, j) for j in range(lo, hi + 1)]
        increase = [a - b for a, b in zip(viol_after, viol_before)]
        d_eps = {j: eps1[j] - eps0[j] for j in range(lo, hi + 1) if j in eps1 and j in eps0}
        earlier = [d_eps[j] for j in d_eps if j < kappa]
        later = [d_eps[j] for j in d_eps if j > kappa]
        rows.append({**r, "overlap_range": [lo, hi], "max_profile_violation_before": max(viol_before), "max_profile_violation_after": max(viol_after),
                     "max_violation_increase": max(increase), "argmax_violation_increase": lo + int(np.argmax(increase)),
                     "violations_created": int(sum(1 for a, b in zip(viol_after, viol_before) if a > 1e-9 and b <= 1e-9)),
                     "eps_kappa_after": eps1.get(kappa), "max_eps_increase_earlier": max(earlier) if earlier else None,
                     "max_eps_increase_later": max(later) if later else None, "mean_eps_change_earlier": float(np.mean(earlier)) if earlier else None,
                     "mean_eps_change_later": float(np.mean(later)) if later else None})
    return {"d": d, "beta": beta, "eps_needed_before": before["eps_needed"], "rows": rows}
