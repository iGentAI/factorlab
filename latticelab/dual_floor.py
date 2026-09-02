"""The two-sided profile floor: primal block-GH constraints (A_i) together with the dual block-GH constraints (D_e) of
docs/notes_lattice_barrier.md, section 4, with an exact certificate.

Constraints on l in R^d (log Gram-Schmidt norms), all written as <v, l> >= c:
  (A_i)  (1 - 1/beta_i) l_i - (1/beta_i) sum_{j=i+1}^{i+beta_i-1} l_j  >=  L(beta_i) - eps,        i = 1..d-1, beta_i = min(beta, d-i+1)
  (D_e)  (1/n) sum_{j=e-n+1}^{e} l_j - l_e  >=  L(n) - eps,                                        e = 2..d,   n = min(beta, e)
  (S)    sum_j l_j = S,
where L(n) = log chat(n).  (D_e) is the block-GH lower bound on the dual of the block ending at e: the vector dual to its last projected
basis vector has norm 1/||b_e^*|| and the dual block has volume 1/vol(B), so lambda_1(B^vee) >= e^{-eps} chat(n) vol(B^vee)^{1/n} gives
l_e <= avg - L(n) + eps.  Two dual families are distinguished by `dual_mode`: 'full' keeps only the full-size dual blocks (e >= beta) --
the blocks a self-dual BKZ tour actually dual-reduces -- while 'all' also imposes the shrinking dual head blocks [1, e], e < beta, i.e.
GH on the duals of the head sublattices L(b_1..b_e); a basis with a flat GSA head violates the latter by about L(e) - (e-1) L(beta)/(beta-1)
(0.4 at e = 50, beta = 403), so the 'all' floor describes a stronger two-sided reduction notion than BKZ-type outputs satisfy.

Certificate.  A double-precision LP (scipy HiGHS) minimising l_1 identifies the active constraints; the multiplier system
  e_1 = sum_{i in P} y_i a_i + sum_{e in D} w_e d_e + z 1
restricted to the active set is solved exactly in rational arithmetic (python-flint fmpq_mat); if it is uniquely solvable with y, w >= 0
exactly, then for every profile satisfying all constraints  l_1 >= sum y_i (L(beta_i) - eps) + sum w_e (L(n_e) - eps) + z S,  a rigorous lower
bound whose value is evaluated as an arb ball.  Dual feasibility is exact; no primal solution is needed (the bound is valid whether or not
the active set is the true optimal one; it equals the LP optimum when it is).

Primal witness.  The dual bound is a LOWER bound on the two-sided minimum, so it certifies that a beta FAILS a target (bound above the
target) but not that it PASSES.  `primal_witness` supplies the other side: the double-precision LP solution, taken as an exact rational
vector and refined on its tight rows, is shifted along the steepening direction w_j = (d+1)/2 - j, which raises the slack of every
constraint of block size n by exactly t (n-1)/2 (both families) and l_1 - S/d by t (d-1)/2; the least exact rational t making every
slack's arb enclosure nonnegative gives a rigorously feasible profile whose head is an exact rational UPPER bound on the minimum.  The
two together enclose the two-sided minimum; `two_sided_beta_floor` decides each beta by whichever side is conclusive.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

from latticelab.profile_floor import _arb_log_chat, block_sizes, floor_l1, log_chat


def constraint_rows(d: int, beta: int, dual_mode: str = "all") -> Tuple[List[Tuple[str, int, int, List[Fraction]]], List[float]]:
    """All (A_i) rows and the (D_e) rows of the chosen dual family ('all': e = 2..d; 'full': e = beta..d; 'none') as exact rational
    coefficient vectors, with their double-precision constants L(n).  Returns [(kind, index, n, coeffs)], [constant]."""
    if dual_mode not in ("all", "full", "none"):
        raise ValueError(dual_mode)
    bs = block_sizes(d, beta)
    rows, consts = [], []
    for i in range(1, d):
        bi = bs[i - 1]
        v = [Fraction(0)] * d
        v[i - 1] = 1 - Fraction(1, bi)
        for j in range(i + 1, i + bi):
            v[j - 1] = -Fraction(1, bi)
        rows.append(("A", i, bi, v)); consts.append(log_chat(bi))
    for e in range(2, d + 1):
        n = min(beta, e)
        if dual_mode == "none" or (dual_mode == "full" and n < beta):
            continue
        v = [Fraction(0)] * d
        for j in range(e - n + 1, e + 1):
            v[j - 1] += Fraction(1, n)
        v[e - 1] -= 1
        rows.append(("D", e, n, v)); consts.append(log_chat(n))
    return rows, consts


def float_lp(d: int, beta: int, eps: float = 0.0, dual_mode: str = "all", primal: bool = True):
    """Double-precision LP: minimise l_1 subject to (A_i) (unless primal=False), the (D_e) of the chosen dual family, and sum l = 0.
    Returns the scipy result and the row list."""
    from scipy.optimize import linprog

    rows, consts = constraint_rows(d, beta, dual_mode)
    if not primal:
        keep = [k for k, r in enumerate(rows) if r[0] == "D"]
        rows = [rows[k] for k in keep]; consts = [consts[k] for k in keep]
    if not rows:
        raise ValueError("no constraints")
    A_ub = -np.array([[float(x) for x in r[3]] for r in rows])  # <v, l> >= c  <=>  -<v, l> <= -c
    b_ub = -np.array(consts) + eps
    res = linprog(c=np.eye(d)[0], A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, d)), b_eq=[0.0], bounds=[(None, None)] * d, method="highs")
    return res, rows, consts


def combined_certificate(d: int, beta: int, eps=0, prec: int = 256, tol: float = 1e-9, dual_mode: str = "all") -> Dict:
    """Exact two-family certificate for the two-sided floor at (d, beta, eps) with the dual family `dual_mode` ('all' or 'full').  The
    active set is read off the double-precision LP (the dual support, rank-filtered and completed to d - 1 independent rows from the tight
    rows by pivoted QR); the (d x d) multiplier system (d - 1 active rows plus the all-ones row) is solved exactly; the certificate is
    accepted only if the solution is unique and every multiplier is >= 0 exactly.  Returns the multipliers, the rigorous bound (arb ball) on
    l_1 - S/d, and its shift above the primal-only floor (arb ball), or raises ValueError if no certificate of this form is found."""
    from flint import arb, ctx, fmpq, fmpq_mat

    res, rows, consts = float_lp(d, beta, float(Fraction(eps)) if not isinstance(eps, str) else float(eps), dual_mode=dual_mode)
    if res.status != 0:
        raise ValueError(f"float LP failed: {res.message}")
    A = np.array([[float(x) for x in r[3]] for r in rows])
    slack = A @ res.x - (np.array(consts) - (float(Fraction(eps)) if not isinstance(eps, str) else float(eps)))
    marg = -res.ineqlin.marginals  # multipliers of the '>=' constraints (nonnegative at optimum)
    # active set: the dual support first (rows with a positive multiplier), completed to d - 1 linearly independent rows by pivoted QR
    # over the remaining tight rows (degenerate vertices have more tight rows than the dimension)
    support = [k for k in range(len(rows)) if marg[k] > 1e-10]
    tight = [k for k in range(len(rows)) if slack[k] < tol and k not in support]
    from scipy.linalg import qr

    def independent(cands, base):
        """Greedy completion of `base` by rows of `cands`, in the order given by pivoted QR of the candidate rows projected off `base`."""
        chosen = list(base)
        if not cands:
            return chosen
        Mb = A[chosen] if chosen else np.zeros((0, d))
        C = A[cands]
        if len(chosen):
            Q, _ = np.linalg.qr(Mb.T)  # orthonormal basis of the row space of the chosen rows
            C = C - (C @ Q) @ Q.T
        _, R, piv = qr(C.T, pivoting=True)
        rank = int(np.sum(np.abs(np.diag(R)) > 1e-9))
        for p in piv[:rank]:
            if len(chosen) >= d - 1:
                break
            chosen.append(cands[p])
        return chosen

    active = independent(sorted(support, key=lambda k: -marg[k]), [])  # rank-filtered support, largest multipliers first
    if len(active) < d - 1:
        active = independent(tight, active)
    if len(active) < d - 1:  # still short: allow non-tight rows with tiny slack (float noise)
        rest = [k for k in np.argsort(slack) if k not in active]
        active = independent(list(rest), active)
    if len(active) != d - 1:
        raise ValueError(f"could not assemble d-1 = {d-1} independent active rows (got {len(active)})")
    # exact multiplier system: columns = active rows' vectors and the all-ones vector; solve M^T [y; z] = e_1
    M = fmpq_mat(d, d)
    for col, k in enumerate(active):
        for r_, x in enumerate(rows[k][3]):
            if x:
                M[r_, col] = fmpq(x.numerator, x.denominator)
    for r_ in range(d):
        M[r_, d - 1] = fmpq(1)
    rhs = fmpq_mat(d, 1)
    rhs[0, 0] = fmpq(1)
    try:
        sol = M.solve(rhs)
    except Exception as exc:  # singular active set
        raise ValueError(f"active set singular: {exc}")
    mult = [Fraction(int(sol[k, 0].p), int(sol[k, 0].q)) for k in range(d - 1)]
    z = Fraction(int(sol[d - 1, 0].p), int(sol[d - 1, 0].q))
    if any(m < 0 for m in mult):
        raise ValueError(f"a multiplier is negative (min {float(min(mult)):.3e}); the float active set does not certify")
    # exact identity check
    coeff = [z] * d
    for m, k in zip(mult, active):
        for r_, x in enumerate(rows[k][3]):
            coeff[r_] += m * x
    assert coeff[0] == 1 and all(c == 0 for c in coeff[1:]), "certificate identity failed"
    # rigorous bound on l_1 - S/d (S = 0 here; z S adds nothing): sum mult * (L(n) - eps)
    ctx.prec = prec
    eps_a = arb(eps) if isinstance(eps, str) else arb(fmpq(Fraction(eps).numerator, Fraction(eps).denominator))
    total = arb(0)
    cache = {}
    for m, k in zip(mult, active):
        n = rows[k][2]
        if n not in cache:
            cache[n] = _arb_log_chat(n, prec)
            ctx.prec = prec
        total += arb(fmpq(m.numerator, m.denominator)) * (cache[n] - eps_a)
    primal = floor_l1(d, beta, eps, 0, prec)
    ctx.prec = prec
    shift = total - primal["l1_floor_ball"]
    rhf_ball = (total / (d - 1)).exp()  # rigorous enclosure of the two-sided floor on the root-Hermite factor (S = 0)
    yA = {rows[k][1]: m for m, k in zip(mult, active) if rows[k][0] == "A"}
    wD = {rows[k][1]: m for m, k in zip(mult, active) if rows[k][0] == "D"}
    return {"d": d, "beta": beta, "eps": eps, "dual_mode": dual_mode, "n_active_A": len(yA), "n_active_D": len(wD), "active_D_positions": sorted(wD),
            "multipliers_A": yA, "multipliers_D": wD,  # exact Fractions keyed by the primal start i and the dual end e
            "z": z, "l1_bound_ball": total, "l1_bound": float(total.mid()), "primal_floor": primal["l1_floor"],
            "shift_ball": shift, "shift": float(shift.mid()), "shift_certified_positive": bool(shift.lower() > 0),
            "float_lp_value": float(res.fun), "rhf_bound_ball": rhf_ball, "rhf_bound": float(rhf_ball.mid())}


def _target_ball(target, prec: int):
    """An arb ball for a target root-Hermite factor given as an arb ball, a Fraction/int, or a decimal string."""
    from flint import arb, ctx, fmpq

    ctx.prec = prec
    if isinstance(target, str):
        return arb(target)
    if hasattr(target, "lower"):
        return target
    ft = Fraction(target)
    return arb(fmpq(ft.numerator, ft.denominator))


def _exact_fraction(a) -> Fraction:
    """The exact rational value of an exact (zero-radius) arb, such as `.mid()`, `.lower()` or `.upper()` of a ball."""
    man, ex = a.man_exp()
    return Fraction(int(man)) * Fraction(2) ** int(ex)


def _exact_row_dots(xhat: List[Fraction], d: int, beta: int, dual_mode: str) -> List[Tuple[str, int, int, Fraction]]:
    """Exact <v, xhat> for every row of constraint_rows(d, beta, dual_mode), in the same order, by prefix sums over a common denominator:
    <a_i, x> = x_i - avg_{[i, i+n-1]} x  and  <d_e, x> = avg_{[e-n+1, e]} x - x_e."""
    den = 1
    for f in xhat:
        den = den * f.denominator // math.gcd(den, f.denominator)
    X = [f.numerator * (den // f.denominator) for f in xhat]
    P = [0]
    for v in X:
        P.append(P[-1] + v)  # P[k] = X_1 + ... + X_k (1-based)
    bs = block_sizes(d, beta)
    out = []
    for i in range(1, d):
        n = bs[i - 1]
        out.append(("A", i, n, Fraction(n * X[i - 1] - (P[i + n - 1] - P[i - 1]), n * den)))
    for e in range(2, d + 1):
        n = min(beta, e)
        if dual_mode == "none" or (dual_mode == "full" and n < beta):
            continue
        out.append(("D", e, n, Fraction((P[e] - P[e - n]) - n * X[e - 1], n * den)))
    return out


def primal_witness(d: int, beta: int, eps=0, prec: int = 256, dual_mode: str = "full", x=None, refine_iters: int = 2,
                   tight_tol: float = 1e-8) -> Dict:
    """A rigorously feasible profile for the primal constraints (A_i) and the dual family `dual_mode`, whose head is an exact rational
    UPPER bound on the two-sided LP minimum of l_1 - S/d (the dual certificate of `combined_certificate` is the lower bound).

    The double-precision LP solution x (recomputed unless given) is taken as the exact rational vector xhat of its binary values, and
    refined: with the constants L(n) replaced by exact rationals of their `prec`-bit midpoints, the residuals of every violated row and of
    every row with slack below `tight_tol` (i.e. all rows with slack < tight_tol) are computed exactly and corrected by a least-squares
    step in double precision, `refine_iters` times, which drives the violations from the LP's feasibility tolerance to roundoff.  Then
    xhat is shifted along w_j = (d+1)/2 - j: this direction has zero sum and satisfies <a_i, w> = <d_e, w> = (n-1)/2 for every
    constraint of block size n, so every slack rises by t (n-1)/2 while l_1 - S/d rises by t (d-1)/2.  Each slack is
    <v, xhat> - L(n) + eps + t (n-1)/2  with <v, xhat> exact and L(n) an arb ball; t is the exact rational value of the arb upper bound of
    the least admissible shift (a directed rounding, magnitude-independent), enlarged by a tiny margin, and the shifted profile is
    re-verified row by row.
    Returns q = xhat_1 - sum(xhat)/d + t (d-1)/2 (Fraction), its root-Hermite ball exp(q/(d-1)), t, and diagnostics.  Raises ValueError
    if the LP fails or the verification does not close."""
    from flint import arb, ctx, fmpq

    eps_fr = Fraction(eps)
    if x is None:
        res, _, _ = float_lp(d, beta, float(eps_fr), dual_mode)
        if res.status != 0:
            raise ValueError(f"float LP failed: {res.message}")
        x = res.x
    xhat = [Fraction(float(v)) for v in x]
    rows, _ = constraint_rows(d, beta, dual_mode)
    ctx.prec = prec
    Lball = {}
    for n in {r[2] for r in rows}:
        Lball[n] = _arb_log_chat(n, prec)
    ctx.prec = prec
    Lfr = {}
    for n, b in Lball.items():
        Lfr[n] = _exact_fraction(b.mid())
    A_np = None
    for _ in range(refine_iters):
        dots = _exact_row_dots(xhat, d, beta, dual_mode)
        slack = [dot - Lfr[n] + eps_fr for (_, _, n, dot) in dots]
        tight = [k for k, s in enumerate(slack) if s < tight_tol]  # the tight rows and every violated row
        if not tight:
            break
        if A_np is None:
            A_np = np.array([[float(v) for v in r[3]] for r in rows])
        delta, *_ = np.linalg.lstsq(A_np[tight], -np.array([float(slack[k]) for k in tight]), rcond=None)
        xhat = [xi + Fraction(float(dl)) for xi, dl in zip(xhat, delta)]
    dots = _exact_row_dots(xhat, d, beta, dual_mode)
    # the least t: every slack  dot - L(n) + eps + t (n-1)/2  must be >= 0, i.e. t >= 2 (L(n) - dot - eps)/(n-1); take the upper bounds
    need = arb(0)
    max_violation = 0.0
    for (_, _, n, dot) in dots:
        v = Lball[n] - arb(fmpq((dot + eps_fr).numerator, (dot + eps_fr).denominator))
        max_violation = max(max_violation, float(v.upper()))
        need = need.max(v * 2 / (n - 1))
    up = need.upper()
    t = Fraction(0)
    if up > 0:
        t = _exact_fraction(up)
        t = t * (1 + Fraction(1, 2 ** 40)) + Fraction(1, 2 ** 100)  # strictly above the enclosure's upper bound, at any magnitude
    min_slack = None
    for (kind, idx, n, dot) in dots:
        val = dot + eps_fr + t * (n - 1) / 2
        s = arb(fmpq(val.numerator, val.denominator)) - Lball[n]
        if not (s.lower() >= 0):
            raise ValueError(f"witness verification failed at row {kind}_{idx}: slack {s}")
        lo = float(s.lower())
        min_slack = lo if min_slack is None else min(min_slack, lo)
    S = sum(xhat, Fraction(0))
    q = xhat[0] - S / d + t * (d - 1) / 2
    ctx.prec = prec
    q_ball = arb(fmpq(q.numerator, q.denominator))
    rhf = (q_ball / (d - 1)).exp()
    return {"d": d, "beta": beta, "eps": eps, "dual_mode": dual_mode, "q": q, "l1_upper": float(q), "t": t, "t_float": float(t),
            "head_shift": float(t * (d - 1) / 2), "max_violation_before_shift": max_violation, "min_slack_lower": min_slack,
            "rhf_witness_ball": rhf, "rhf_witness": float(rhf.mid()), "n_rows": len(dots), "refine_iters": refine_iters}


def two_sided_beta_floor(d: int, target, beta_lo: int, beta_hi: int, eps=0, dual_mode: str = "full", prec: int = 256,
                         max_prec: int = 4096, log=None) -> Dict:
    """The least beta in [beta_lo, beta_hi] whose two-sided floor -- the minimum of l_1 - S/d under the primal constraints (A_i) and the
    dual family `dual_mode` -- on the root-Hermite factor is <= the target, with every beta from beta_lo up to the first passing one
    decided rigorously.  A beta FAILS when the exact two-family certificate's lower bound (an arb ball, divided by d - 1 and exponentiated)
    lies above the target ball; it PASSES when the primal witness's upper bound (`primal_witness`: a rigorously feasible profile) lies at or
    below it.  If neither is conclusive the precision is doubled (capped at `max_prec`); if the target lies strictly inside the certified
    enclosure [lower bound, witness] no precision helps and a ValueError reports the enclosure.  Leastness is relative to the range.  Since
    the two-sided feasible set is contained in the primal one, the two-sided floor dominates the primal floor at every beta, so every beta
    rigorously excluded by the primal floor (`profile_floor.beta_floor_for_target` in exact_all mode) is excluded here as well; that is
    how a scan starting at beta_lo above the primal crossing is extended downwards.  `target` is an arb ball (e.g.
    `profile_floor.gsa_delta_ball`), a Fraction/int, or a decimal string.  Returns the per-beta decisions and `beta_floor` (None if no
    beta in the range passes).  `log`, if given, is called with one line per decided beta."""
    if not (2 <= beta_lo <= beta_hi <= d):
        raise ValueError(f"need 2 <= beta_lo <= beta_hi <= d (got {beta_lo}, {beta_hi}, {d})")
    if prec > max_prec:
        raise ValueError(f"prec {prec} exceeds max_prec {max_prec}")
    from flint import ctx

    decisions: List[Dict] = []
    out = {"d": d, "eps": eps, "dual_mode": dual_mode, "beta_lo": beta_lo, "beta_hi": beta_hi, "decisions": decisions, "beta_floor": None}
    for beta in range(beta_lo, beta_hi + 1):
        p = prec
        while True:
            c = combined_certificate(d, beta, eps, p, dual_mode=dual_mode)
            ctx.prec = p
            t = _target_ball(target, p)
            fb = c["rhf_bound_ball"]
            rec = {"beta": beta, "rhf_two_sided_lower": c["rhf_bound"], "shift": c["shift"], "shift_certified_positive": c["shift_certified_positive"],
                   "prec": p}
            if fb.lower() > t.upper():
                rec.update({"reaches": False, "certified_by": "dual bound"})
                break
            w = primal_witness(d, beta, eps, p, dual_mode)
            wb = w["rhf_witness_ball"]
            from flint import arb, fmpq

            width = (arb(fmpq(w["q"].numerator, w["q"].denominator)) - c["l1_bound_ball"]).upper()  # rigorous: q >= true minimum >= bound
            rec.update({"rhf_witness": w["rhf_witness"], "witness_head_shift": w["head_shift"], "witness_t": w["t_float"],
                        "enclosure_width_l1": math.nextafter(float(width), math.inf)})  # rounded upwards: the float remains an upper bound
            if wb.upper() <= t.lower():
                rec.update({"reaches": True, "certified_by": "primal witness"})
                break
            if fb.upper() <= t.lower() and wb.lower() > t.upper():
                raise ValueError(f"undecidable at beta={beta}: the target {t} lies inside the certified enclosure [{fb}, {wb}] of the two-sided floor")
            if p >= max_prec:
                raise ValueError(f"threshold undecidable at beta={beta} up to {max_prec} bits: floor in [{fb}, {wb}] vs target {t}")
            p = min(2 * p, max_prec)
        decisions.append(rec)
        if log is not None:
            log(f"two-sided d={d} beta={beta}: {'PASSES' if rec['reaches'] else 'fails'} rhf_lower={c['rhf_bound']:.13f}"
                + (f" rhf_witness={rec['rhf_witness']:.13f}" if "rhf_witness" in rec else "") + f" shift={c['shift']:+.5f} ({rec['certified_by']})")
        if rec["reaches"]:
            out["beta_floor"] = beta
            out["minimality"] = (f"certified within [{beta_lo}, {beta}]: every failing beta excluded by the exact two-family dual bound and the passing "
                                 f"beta attained by a rigorously feasible primal witness (arb enclosures, directed comparisons); exclusion below "
                                 f"{beta_lo} rests on the primal floor, which the two-sided floor dominates")
            return out
    out["note"] = f"no beta in [{beta_lo}, {beta_hi}] reaches the target under the two-sided constraints (every beta decided rigorously)"
    return out


def main(argv=None):
    """CLI: `python -m latticelab.dual_floor --d 1003 --beta-spec 403 --lo 410 --hi 430 --out results/lattice_two_sided_scan.json`
    scans the two-sided floor against the pure-GSA target of `--beta-spec` and stores (or replaces) the row keyed by (d, eps, dual_mode)."""
    import argparse
    import json
    import os
    import time

    from latticelab.profile_floor import gsa_delta_ball

    ap = argparse.ArgumentParser(description="two-sided (primal + dual block-GH) profile-floor blocksize scan with exact certificates")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--beta-spec", type=int, required=True, help="the target is the pure-GSA root-Hermite factor at this blocksize")
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--eps", default="0", help="uniform slack (decimal string)")
    ap.add_argument("--dual-mode", default="full", choices=("full", "all"))
    ap.add_argument("--out", default=None, help="JSON archive to update")
    a = ap.parse_args(argv)
    t0 = time.time()
    row = two_sided_beta_floor(a.d, gsa_delta_ball(a.beta_spec), a.lo, a.hi, a.eps, a.dual_mode, log=lambda s: print(s, flush=True))
    row.update({"beta_spec": a.beta_spec, "seconds": time.time() - t0})
    print(f"two-sided d={a.d}: beta_floor={row['beta_floor']} ({row['seconds']:.0f}s)", flush=True)
    if a.out:
        out = json.load(open(a.out)) if os.path.exists(a.out) else {"note": "two-sided floor blocksize scan: every failing beta excluded by the exact "
                                                                     "two-family dual bound, the passing beta attained by a rigorously feasible primal "
                                                                     "witness, both as arb balls against the pure-GSA target ball", "rows": []}
        key = (a.d, str(a.eps), a.dual_mode)
        out["rows"] = [r for r in out["rows"] if (r["d"], str(r.get("eps", "0")), r.get("dual_mode", "full")) != key] + [row]
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
