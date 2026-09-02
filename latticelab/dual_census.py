"""L6, continued: exact dual-block ratios of real bases, the two-sided per-basis decomposition, and a strict-tour census
(docs/notes_lattice_barrier.md, sections 4, 6 and 7).

Dual blocks by the reversed dual basis.  For a lattice containing q Z^d (every q-ary lattice of `latticelab.lattices.qary`, and the NTRU
lattices) the dual basis D = B^{-T} (row j = the dual vector d_j with <b_i, d_j> = delta_ij) satisfies q D in Z^{d x d}.  Let the rows of q D be
taken in reversed order.  The projected block of that basis starting at position i' = d - e + 1 (1-based) with size n = min(beta, e) is q
times the dual of the primal block B_e = pi_{e-n+1}(L(b_{e-n+1}, ..., b_e)) ending at e: the dual basis vectors d_l, e-n+1 <= l <= e, projected
onto span(b_1, ..., b_e), pair to delta_{lk} with the block basis pi_{e-n+1} b_k (d_l is orthogonal to b_1..b_{e-n} for l > e-n, and
pi_{e-n+1} b_k differs from b_k by a vector in span(b_1..b_{e-n})), so they form the dual basis of the block; and that projection is exactly
the Gram-Schmidt projection of the reversed dual basis at positions d-e+1, ..., d-e+n (orthogonal to d_{e+1}, ..., d_d, which span
span(b_1..b_e)^perp).  Hence `latticelab.profile.block_gh_ratios` on the reversed dual basis returns the exact lambda_1(B_e^vee) of every
dual block (the scaling by q cancels in every ratio), the signed dual ratio  eps^vee_e(B) := log(GH(B_e^vee) / lambda_1(B_e^vee)), and the dual
non-tightness  r_e := -log(||b_e^*|| lambda_1(B_e^vee)) >= 0  (the last dual basis vector of the block is b_e^*/||b_e^*||^2, of norm 1/||b_e^*||).

Two-sided per-basis decomposition.  With any exact two-family certificate e_1 = sum_i y_i a_i + sum_e w_e d_e + z 1 (y, w >= 0;
`latticelab.dual_floor.combined_certificate`), pairing with the profile gives, for every basis,
    l_1 - S/d  =  h^{two}(0) - sum_i y_i eps_i(B) - sum_e w_e eps^vee_e(B) + R^{two}(B),
    R^{two}(B) = sum_i y_i log(||b_i^*||/lambda_1(B_i)) + sum_e w_e r_e  >=  0,
since <a_i, l> = log chat(beta_i) - eps_i + log(||b_i^*||/lambda_1(B_i)) and <d_e, l> = avg - l_e = log chat(n) - eps^vee_e + r_e.  `two_sided_mass`
evaluates every term on a real basis with exact enumerations and exact multipliers and reports the identity gap (roundoff).

Strict tours.  `strict_tours_census` runs fpylll's basic BKZ (exact, unpruned enumeration in every block; insertion whenever the block minimum
is below sqrt(0.99) ||b_kappa^*||) tour by tour from an LLL start and evaluates the primal per-basis identity at checkpoints.  After a *clean*
tour (no insertion changed anything) every block is SVP-tight up to the 0.99 slack and R(B) <= (-log 0.99 / 2) sum_i y_i; before that, later
insertions of a tour un-tighten earlier blocks and R(B) can be large, so the signed mass, its positive part and R(B) are reported separately.
The state of the dynamics is the integer basis (the Gram-Schmidt data are recomputed from it), so a census can be RESUMED exactly from an
archived basis at its recorded tour count: the CLI continues every archived (d, beta, seed) that has neither reached a clean tour nor the
requested number of tours, and archives after every checkpoint.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from fpylll import BKZ, GSO, IntegerMatrix, LLL
from fpylll.algorithms.bkz import BKZReduction as BKZBase

from latticelab.insertion import weighted_subgh_mass
from latticelab.profile import block_gh_ratios, gs_profile
from latticelab.profile_floor import log_chat


def reversed_dual_basis(A: IntegerMatrix, q: int) -> IntegerMatrix:
    """The rows of q A^{-T} in reversed order (exact rational inverse via python-flint).  Requires q A^{-T} integral, i.e. q Z^d subset L;
    raises ValueError otherwise."""
    from flint import fmpz_mat

    d = A.nrows
    if A.ncols != d:
        raise ValueError("need a square basis")
    M = fmpz_mat([[int(A[i, j]) for j in range(d)] for i in range(d)])
    Minv = M.inv()  # fmpq_mat; column j is the dual vector d_j
    rows = []
    for j in range(d):
        row = []
        for i in range(d):
            x = Minv[i, j] * q
            if int(x.q) != 1:
                raise ValueError(f"q A^(-T) is not integral at ({i}, {j}): the lattice does not contain q Z^d")
            row.append(int(x.p))
        rows.append(row)
    rows.reverse()
    return IntegerMatrix.from_matrix(rows)


def dual_block_ratios(A: IntegerMatrix, beta: int, q: int) -> Dict:
    """Exact eps^vee_e(B) and r_e for every dual block ending at e = 2..d (size min(beta, e)), via the reversed dual basis."""
    d = A.nrows
    D = reversed_dual_basis(A, q)
    dual = block_gh_ratios(D, beta)
    eps_D, r_D, n_D = {}, {}, {}
    for row in dual["blocks"]:
        e = d - row["i"] + 1
        if row["beta_i"] != min(beta, e):
            raise AssertionError("dual block size mismatch")
        eps_D[e], r_D[e], n_D[e] = row["log_gh_over_lambda1"], math.log(row["b_star_over_lambda1"]), row["beta_i"]
    return {"eps": eps_D, "r": r_D, "n": n_D}


def two_sided_mass(A: IntegerMatrix, beta: int, q: int, dual_mode: str = "full", primal_stats: Dict | None = None) -> Dict:
    """Every term of the two-sided per-basis decomposition on the basis A at test blocksize beta with the dual family `dual_mode`: exact
    primal ratios (block_gh_ratios on A, or `primal_stats` if supplied), exact dual ratios (reversed dual basis), the exact two-family
    multipliers of combined_certificate, the residual R^{two} >= 0 and the identity gap.  Also the full-size dual family's exact violations
    (eps^vee_e > 0, e >= beta), its profile-level violations l_e - (avg - log chat(n)), the dual-tight fraction, and the head family's eps."""
    from latticelab.dual_floor import combined_certificate

    d = A.nrows
    prim = primal_stats if primal_stats is not None else block_gh_ratios(A, beta)
    dl = dual_block_ratios(A, beta, q)
    eps_D, r_D, n_D = dl["eps"], dl["r"], dl["n"]
    eps_A = {row["i"]: row["log_gh_over_lambda1"] for row in prim["blocks"]}
    r_A = {row["i"]: math.log(row["b_star_over_lambda1"]) for row in prim["blocks"]}
    cert = combined_certificate(d, beta, dual_mode=dual_mode)
    yA, wD = cert["multipliers_A"], cert["multipliers_D"]
    p = gs_profile(A)
    S = float(p.sum())
    lhs = float(p[0]) - S / d
    mass_A = sum(float(y) * eps_A[i] for i, y in yA.items())
    mass_D = sum(float(w) * eps_D[e] for e, w in wD.items())
    R = sum(float(y) * r_A[i] for i, y in yA.items()) + sum(float(w) * r_D[e] for e, w in wD.items())
    h_two = cert["l1_bound"]
    identity_gap = lhs - (h_two - mass_A - mass_D + R)
    full = [e for e in eps_D if n_D[e] == beta]
    viol_prof = {e: float(p[e - 1]) - (float(p[e - beta:e].mean()) - log_chat(beta)) for e in full}
    exact_viol = {e: eps_D[e] for e in full if eps_D[e] > 0}
    prof_viol = {e: v for e, v in viol_prof.items() if v > 1e-12}
    return {"d": d, "beta": beta, "q": q, "dual_mode": dual_mode, "head_minus_mean": lhs, "two_sided_floor_h0": h_two,
            "primal_floor_h0": cert["primal_floor"], "shift": cert["shift"], "mass_A_active": mass_A, "mass_D_active": mass_D,
            "residual_two_sided": R, "identity_gap": identity_gap, "n_active_A": cert["n_active_A"], "n_active_D": cert["n_active_D"],
            "active_D_positions": cert["active_D_positions"], "n_full_dual_blocks": len(full),
            "dual_eps_needed_full": max([eps_D[e] for e in full] + [0.0]),
            "dual_worst_block_full": max(full, key=lambda e: eps_D[e]) if full else None,
            "dual_mean_eps_full": float(np.mean([eps_D[e] for e in full])) if full else float("nan"),
            "dual_n_exact_violations_full": len(exact_viol), "dual_n_profile_violations_full": len(prof_viol),
            "dual_max_profile_violation_full": max(prof_viol.values()) if prof_viol else 0.0,
            "dual_eps_needed_head": max([eps_D[e] for e in eps_D if n_D[e] < beta] + [0.0]),
            "dual_frac_tight_full": float(np.mean([abs(r_D[e]) < 1e-9 for e in full])) if full else float("nan"),
            "primal_eps_needed": prim["eps_needed"], "primal_frac_tight": prim["frac_blocks_with_bstar_shortest"],
            "primal_mean_eps": prim["mean_log_gh_over_lambda1"],
            "eps_dual": [eps_D[e] for e in range(2, d + 1)], "r_dual": [r_D[e] for e in range(2, d + 1)],
            "eps_primal": [eps_A[i] for i in range(1, d)]}


def gso_float_type(d: int, float_type: str = "auto") -> str:
    """'auto' selects double precision below d = 150 and long double from there (fpylll's double-precision LLL and GSO fail on the census's
    225-dimensional q-ary bases: 'infinite loop in babai', non-positive r_ii), matching `latticelab.profile._gso`; any other value is passed
    through to fpylll."""
    if float_type == "auto":
        return "ld" if d >= 150 else "d"
    return float_type


def strict_tours_census(d: int, beta: int, q: int, seed: int, checkpoints: List[int], float_type: str = "auto", start_basis=None,
                        tours_done: int = 0, on_checkpoint=None) -> Dict:
    """fpylll's basic BKZ-beta (exact unpruned SVP in every block, insertion below sqrt(0.99) ||b_kappa^*||) tour by tour from an LLL start on
    qary(d, d//2, q, seed); at each tour count in `checkpoints` the primal per-basis identity is evaluated (weighted_subgh_mass).  Stops
    running tours once a tour changes nothing (the basis is then stable up to the 0.99 slack), records that tour at the next checkpoint and
    evaluates no further checkpoints.  `float_type` is the
    GSO floating-point type ('auto': double below d = 150, long double from there).

    Resume: with `start_basis` -- the archived basis of this same (d, beta, seed) after `tours_done` tours -- the tours continue from that
    basis (no LLL is re-run), only the checkpoints above `tours_done` are evaluated, and each row records `resumed_from`.  This is an exact
    continuation of the trajectory: the state of the dynamics is the integer basis.  `on_checkpoint(row, basis_rows)`, if given, is called
    after every checkpoint with the row and the current basis (as integer rows), for incremental archiving."""
    import time

    from latticelab.lattices import qary

    ft = gso_float_type(d, float_type)
    if start_basis is None:
        if tours_done:
            raise ValueError("tours_done > 0 requires start_basis")
        A = qary(d, d // 2, q, seed=seed)
        B = IntegerMatrix.from_matrix(A)
        LLL.reduction(B)  # fplll's wrapper method: it escalates its own precision as needed (it succeeded on the d = 225 bases; the failures
        # that motivate `float_type` were in the BKZ object's LLL and GSO below, which are constructed from `ft`); kept as the archived
        # trajectories' starting point
        resumed_from = None
    else:
        if tours_done < 0:
            raise ValueError("tours_done must be nonnegative")
        B = IntegerMatrix.from_matrix(start_basis)
        if B.nrows != d or B.ncols != d:
            raise ValueError(f"start_basis is {B.nrows} x {B.ncols}, expected {d} x {d}")
        resumed_from = tours_done
    M = GSO.Mat(B, float_type=ft)
    M.update_gso()
    bkz = BKZBase(M)
    par = BKZ.Param(block_size=beta, max_loops=1)
    rows, t0, stable_at = [], time.time(), None
    for T in sorted(checkpoints):
        if T <= tours_done:
            continue
        while tours_done < T and stable_at is None:
            clean = bkz.tour(par)
            tours_done += 1
            if clean:
                stable_at = tours_done
        w = weighted_subgh_mass(B, beta)
        row = {**{k: v for k, v in w.items() if k not in ("eps", "y")}, "tours": tours_done, "requested_tours": T, "seed": seed,
               "stable_at_tour": stable_at, "seconds": time.time() - t0, "q": q, "float_type": ft, "resumed_from": resumed_from}
        rows.append(row)
        if on_checkpoint is not None:
            on_checkpoint(row, [[int(B[i, j]) for j in range(d)] for i in range(d)])
        if stable_at is not None:
            break  # the basis no longer changes: later checkpoints would only repeat this row
    return {"d": d, "beta": beta, "q": q, "seed": seed, "rows": rows, "stable_at_tour": stable_at, "resumed_from": resumed_from,
            "basis": [[int(B[i, j]) for j in range(d)] for i in range(d)]}


def primal_from_reversed_dual(D: IntegerMatrix, q: int) -> IntegerMatrix:
    """Inverse of `reversed_dual_basis`: given the rows of q B^{-T} in reversed order (possibly transformed by a unimodular matrix, i.e. another
    basis of q L^vee), return the primal basis B = q (P D)^{-T} (P reverses the rows), which is integral and generates the same lattice L."""
    from flint import fmpz_mat

    d = D.nrows
    rows = [[int(D[d - 1 - i, j]) for j in range(d)] for i in range(d)]  # P D: rows q d_1, ..., q d_d
    M = fmpz_mat(rows)
    Minv = M.inv()
    B = []
    for i in range(d):
        row = []
        for j in range(d):
            x = Minv[j, i] * q  # B = q (P D)^{-T}: B[i][j] = q * (Minv)[j][i]
            if int(x.q) != 1:
                raise ValueError("q (P D)^(-T) is not integral")
            row.append(int(x.p))
        B.append(row)
    return IntegerMatrix.from_matrix(B)


def self_dual_tours_census(d: int, beta: int, q: int, seed: int, checkpoints: List[int], primal_first: bool = True) -> Dict:
    """A self-dual schedule: each round is one exact (unpruned, 0.99-threshold) BKZ-beta tour on the primal basis followed by one on the reversed
    dual basis q B^{-T} (transformed back to a primal basis by `primal_from_reversed_dual`), from an LLL start on qary(d, d//2, q, seed).  At each
    round count in `checkpoints` the two-sided decomposition (`two_sided_mass`, full-size dual family) and the primal identity are evaluated:
    the dual tour makes the dual blocks dual-SVP-tight (r_e small) and the question is whether the two-sided residual R^two shrinks while the
    head stays above or falls below the two-sided floor.  Stops the tours once both tours of a round are clean."""
    import time

    from latticelab.lattices import qary

    A = qary(d, d // 2, q, seed=seed)
    B = IntegerMatrix.from_matrix(A)
    LLL.reduction(B)
    par = BKZ.Param(block_size=beta, max_loops=1)
    rows, t0, stable_at, rounds_done = [], time.time(), None, 0

    def tour(M_basis):
        M = GSO.Mat(M_basis)
        M.update_gso()
        return BKZBase(M).tour(par)

    for T in sorted(checkpoints):
        while rounds_done < T and stable_at is None:
            if primal_first:
                clean_p = tour(B)
                D = reversed_dual_basis(B, q)
                clean_d = tour(D)
                B = primal_from_reversed_dual(D, q)
            else:
                D = reversed_dual_basis(B, q)
                clean_d = tour(D)
                B = primal_from_reversed_dual(D, q)
                clean_p = tour(B)
            rounds_done += 1
            if clean_p and clean_d:
                stable_at = rounds_done
        prim = block_gh_ratios(B, beta)
        w = weighted_subgh_mass(B, beta)
        ts = two_sided_mass(B, beta, q, "full", primal_stats=prim)
        rows.append({**{k: v for k, v in ts.items() if k not in ("eps_dual", "r_dual", "eps_primal")}, "rounds": rounds_done, "requested_rounds": T,
                     "seed": seed, "stable_at_round": stable_at, "seconds": time.time() - t0, "primal_mass_signed": w["weighted_eps_signed"],
                     "primal_mass_positive": w["weighted_eps_positive"], "primal_residual": w["residual"], "primal_floor_h0": w["floor_h0"],
                     "eps_dual": ts["eps_dual"], "r_dual": ts["r_dual"], "eps_primal": ts["eps_primal"]})
    return {"d": d, "beta": beta, "q": q, "seed": seed, "rows": rows, "stable_at_round": stable_at,
            "basis": [[int(B[i, j]) for j in range(d)] for i in range(d)]}


def main(argv=None):
    """CLI.  `python -m latticelab.dual_census --dual-census --out results/lattice_l6_dual.json` evaluates the two-sided per-basis
    decomposition (exact primal and dual block ratios) on BKZ-2.0 outputs (8 tours) at fixed d/beta = 2.5, three seeds, archiving the bases;
    `--strict-census --out results/lattice_l6_strict.json` runs the strict-tour census.  Rows already present are not recomputed; a strict
    census whose archived rows stop below the largest requested checkpoint without a clean tour is RESUMED from its archived basis (exact
    continuation), and every checkpoint is archived as soon as it is evaluated."""
    import argparse
    import json
    import os
    import time

    from latticelab.lattices import qary
    from latticelab.profile import bkz

    ap = argparse.ArgumentParser(description="L6: exact dual-block ratios, two-sided decomposition, strict-tour census")
    ap.add_argument("--dual-census", action="store_true")
    ap.add_argument("--strict-census", action="store_true")
    ap.add_argument("--self-dual-census", action="store_true", help="alternating exact primal and dual tours; checkpoints count rounds")
    ap.add_argument("--points", nargs="+", default=["50,20", "75,30", "100,40", "125,50"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[31, 32, 33])
    ap.add_argument("--tours", type=int, default=8, help="BKZ 2.0 tours for the dual census")
    ap.add_argument("--checkpoints", nargs="+", type=int, default=[2, 8, 32])
    ap.add_argument("--q", type=int, default=2 ** 16 + 1)
    ap.add_argument("--float-type", default="auto", help="GSO float type for the strict census: 'auto' (double below d = 150, long double from there), 'd', 'ld'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"rows": [], "bases": {}}
    out.setdefault("rows", [])
    out.setdefault("bases", {})

    def save():
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    def have(kind, d, beta, seed):
        return any(r.get("kind") == kind and r["d"] == d and r["beta"] == beta and r["seed"] == seed for r in out["rows"])

    for pt in a.points:
        d, beta = (int(x) for x in pt.split(","))
        for seed in a.seeds:
            key = f"{d},{beta},{seed}"
            if a.dual_census and not have("dual", d, beta, seed):
                out["note_dual"] = ("two-sided per-basis decomposition on BKZ-2.0 outputs: exact primal and dual block ratios (unpruned "
                                    "enumeration on the basis and on the reversed dual basis q B^(-T)), exact two-family multipliers "
                                    "(full-size dual family), residual R^two >= 0; the reduced bases are archived under 'bases'")
                t0 = time.time()
                A = qary(d, d // 2, a.q, seed=seed)
                B = bkz(A, beta, tours=a.tours)
                prim = block_gh_ratios(B, beta)
                w1 = weighted_subgh_mass(B, beta)
                ts = two_sided_mass(B, beta, a.q, "full", primal_stats=prim)
                row = {"kind": "dual", "seed": seed, "tours": a.tours, "seconds": time.time() - t0,
                       "primal_mass_signed": w1["weighted_eps_signed"], "primal_mass_positive": w1["weighted_eps_positive"],
                       "primal_residual": w1["residual"], "primal_floor_h0_check": w1["floor_h0"], **ts}
                out["rows"].append(row)
                out["bases"][key] = [[int(B[i, j]) for j in range(d)] for i in range(d)]
                save()
                print(f"dual ({d},{beta}) seed {seed}: head-mean {ts['head_minus_mean']:+.4f}, two-sided floor {ts['two_sided_floor_h0']:.4f} "
                      f"(shift {ts['shift']:+.4f}); active mass A {ts['mass_A_active']:+.4f} D {ts['mass_D_active']:+.4f}; R^two {ts['residual_two_sided']:.4f}; "
                      f"gap {ts['identity_gap']:.1e}; dual eps_needed(full) {ts['dual_eps_needed_full']:.4f} at e={ts['dual_worst_block_full']} "
                      f"({ts['dual_n_exact_violations_full']}/{ts['n_full_dual_blocks']} exact, {ts['dual_n_profile_violations_full']} profile, "
                      f"max {ts['dual_max_profile_violation_full']:.4f}); dual mean eps {ts['dual_mean_eps_full']:+.4f}, dual tight "
                      f"{ts['dual_frac_tight_full']:.2f}; head dual eps_needed {ts['dual_eps_needed_head']:.4f}; primal eps_needed "
                      f"{ts['primal_eps_needed']:.4f}, primal signed mass {w1['weighted_eps_signed']:+.4f} [{row['seconds']:.0f}s]", flush=True)
            if a.strict_census:
                existing = [r for r in out["rows"] if r.get("kind") == "strict" and r["d"] == d and r["beta"] == beta and r["seed"] == seed]
                done = max((r["tours"] for r in existing), default=0)
                stable = any(r.get("stable_at_tour") is not None for r in existing)
                start, resume_from, run = None, 0, True
                if existing:
                    if stable or max(a.checkpoints) <= done:
                        run = False  # already at a clean tour, or nothing new requested
                    elif ("strict," + key) not in out["bases"]:
                        print(f"strict ({d},{beta}) seed {seed}: rows to {done} tours archived without a basis; cannot resume", flush=True)
                        run = False
                    else:
                        start, resume_from = out["bases"]["strict," + key], done
                        print(f"strict ({d},{beta}) seed {seed}: resuming from the archived basis after {done} tours", flush=True)
                if run:
                    out["note_strict"] = ("strict-tour census: fpylll basic BKZ (exact unpruned SVP per block, insertion below sqrt(0.99)||b*||) "
                                          "from an LLL start, primal per-basis identity at tour checkpoints; the basis after the last checkpoint is "
                                          "archived, and later runs resume from it (rows carry resumed_from)")

                    def on_checkpoint(r, basis_rows, key=key, d=d, beta=beta, seed=seed):
                        out["rows"].append({"kind": "strict", **r})
                        out["bases"]["strict," + key] = basis_rows
                        save()
                        print(f"strict ({d},{beta}) seed {seed} tours {r['tours']}: mass {r['weighted_eps_signed']:+.4f} (pos {r['weighted_eps_positive']:.4f}) "
                              f"R {r['residual']:.4f} tight {r['frac_tight']:.2f} head-floor {r['head_minus_mean'] - r['floor_h0']:+.4f} "
                              f"stable_at {r['stable_at_tour']} [{r['seconds']:.0f}s]", flush=True)

                    strict_tours_census(d, beta, a.q, seed, a.checkpoints, a.float_type, start_basis=start, tours_done=resume_from,
                                        on_checkpoint=on_checkpoint)
            if a.self_dual_census and not have("selfdual", d, beta, seed):
                out["note_selfdual"] = ("self-dual census: rounds of one exact primal BKZ tour and one exact tour on the reversed dual basis "
                                        "q B^(-T) (transformed back), from an LLL start; two-sided decomposition (full-size dual family) and the "
                                        "primal identity at round checkpoints; final bases archived")
                c = self_dual_tours_census(d, beta, a.q, seed, a.checkpoints)
                for r in c["rows"]:
                    out["rows"].append({"kind": "selfdual", **r})
                    print(f"selfdual ({d},{beta}) seed {seed} rounds {r['rounds']}: head-two {r['head_minus_mean'] - r['two_sided_floor_h0']:+.4f} "
                          f"head-primal {r['head_minus_mean'] - r['primal_floor_h0']:+.4f} active A {r['mass_A_active']:+.4f} D {r['mass_D_active']:+.4f} "
                          f"R^two {r['residual_two_sided']:.4f} primal R {r['primal_residual']:.4f} tight P {r['primal_frac_tight']:.2f} D {r['dual_frac_tight_full']:.2f} "
                          f"eps_needed P {r['primal_eps_needed']:.4f} D {r['dual_eps_needed_full']:.4f} ({r['dual_n_exact_violations_full']}/{r['n_full_dual_blocks']}) "
                          f"stable_at {r['stable_at_round']} [{r['seconds']:.0f}s]", flush=True)
                out["bases"]["selfdual," + key] = c["basis"]
                save()
    print("CENSUS_DONE", flush=True)


if __name__ == "__main__":
    main()
