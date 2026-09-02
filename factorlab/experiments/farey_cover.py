"""Localised covering families built from Farey fractions (the in-model landscape below L = N^{3/8}).

Setting.  N = pq is balanced, J_C = [sqrt(N/C), sqrt N], and I = [p0 - L/2, p0 + L/2] is an interval inside J_C known to
contain the smaller factor.  An oblivious covering family of I is a set of forms (a, b), a <= b <= Ca, each testing a window of W
consecutive integers around g(p*) = 2 sqrt(abN), g(p) := aN/p + bp, whose covered sets {p : g(p*) <= g(p) < g(p*) + W} exhaust I.  Its
M1 cost is P + 2 sqrt(Sigma_W) up to the babystep balance, P the number of forms and Sigma_W the number of tested integers.

The Farey family of order Q.  The critical point p* = sqrt(aN/b) of a form lies in I iff b/a lies in [xi_1, xi_2] :=
[N/(p0 + L/2)^2, N/(p0 - L/2)^2].  Take every reduced fraction b/a with a <= Q in that interval, together with the nearest such fraction
on each side of it, order them by their critical points, and assign every gap between consecutive critical points to the endpoint of
smaller denominator (ties to the left); each form's window is the least one whose covered set contains the part of its assigned gaps
inside I.

Windows and coverage.  A form with W tested integers n_0, ..., n_0 + W - 1, n_0 = ceil(g(p*)), has candidate set containing
[g(p*), g(p*) + W - 1]; we use that conservative covered set {p : g(p*) <= g(p) <= g(p*) + W - 1} throughout, an interval around p*
by convexity, and give each form the least W for which it contains the required points: W = ceil(need) + 1, need := max g(e) - g(p*)
over the required points e.  The true candidate set is larger (it extends to the neighbourhoods of the tested integers), so every cost
reported here is an upper bound on the family's cost under this convention, not the least cost of a family of these forms.  Coverage is
verified independently by recomputing every covered interval from (a, b, W) by bisection on brackets found by halving and doubling
from p*, and checking that the union covers I.

With Q = round((N/L)^{1/5}) the family has O(L^{3/5} N^{-1/10}) forms and tested integers of the same order squared, for every interval;
intervals within distance L of a critical point of a form with small ab are covered by that single form at cost O(L N^{-1/4}).

All arithmetic on positions is floating point (relative error 1e-16 on positions of size sqrt N, far below the scale L); the window
sizes are integers and the coverage check allows a slack of 1e-9 L at the joints.
"""
from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from math import gcd, isqrt
from typing import Dict, List, Sequence, Tuple

from factorlab.gen import make_semiprime


def g(a: int, b: int, N: int, p: float) -> float:
    return a * N / p + b * p


def critical_point(a: int, b: int, N: int) -> float:
    return math.sqrt(a * N / b)


def covered_interval(a: int, b: int, N: int, W: int, lo: float = 0.0, hi: float = 0.0) -> Tuple[float, float]:
    """The conservative covered set {p > 0 : g(p*) <= g(p) <= g(p*) + W - 1} of the form (a, b) with W tested integers, as an interval
    [p_-, p_+] around p* = sqrt(aN/b); g is convex on (0, oo) with g -> oo at both ends, so each endpoint is found by bisection on a
    bracket obtained by halving (left) or doubling (right) from p*.  The arguments lo, hi are accepted for interface compatibility and
    not used: the interval does not depend on the region being covered."""
    ps = critical_point(a, b, N)
    target = g(a, b, N, ps) + (W - 1)

    def root(direction: int) -> float:
        inner = ps
        outer = ps / 2 if direction < 0 else 2 * ps
        while g(a, b, N, outer) <= target:
            outer = outer / 2 if direction < 0 else 2 * outer
        for _ in range(200):
            mid = 0.5 * (inner + outer)
            if g(a, b, N, mid) <= target:
                inner = mid
            else:
                outer = mid
        return inner

    return root(-1), root(+1)


def farey_in_interval(xi1: Fraction, xi2: Fraction, Q: int) -> List[Tuple[int, int]]:
    """Reduced fractions b/a with 1 <= a <= Q and xi1 <= b/a <= xi2, as (a, b)."""
    out = []
    for a in range(1, Q + 1):
        b_lo = math.ceil(xi1 * a)
        b_hi = math.floor(xi2 * a)
        for b in range(b_lo, b_hi + 1):
            if gcd(a, b) == 1:
                out.append((a, b))
    return out


def farey_neighbours(xi1: Fraction, xi2: Fraction, Q: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """The largest reduced fraction of order Q below xi1 and the smallest above xi2."""
    best_lo, best_hi = None, None
    for a in range(1, Q + 1):
        b = math.floor(xi1 * a)
        if Fraction(b, a) < xi1:
            if best_lo is None or Fraction(b, a) > Fraction(*reversed(best_lo)):
                best_lo = (a, b)
        b = math.ceil(xi2 * a)
        if Fraction(b, a) > xi2:
            if best_hi is None or Fraction(b, a) < Fraction(*reversed(best_hi)):
                best_hi = (a, b)
    return best_lo, best_hi


def farey_family(N: int, p0: float, L: float, Q: int, C: float = 2.0, verify: bool = True) -> Dict:
    """The Farey covering family of order Q of I = [p0 - L/2, p0 + L/2].  Returns forms, P, Sigma_W, cost and the coverage check."""
    lo, hi = p0 - L / 2, p0 + L / 2
    xi1 = Fraction(N) / (Fraction(hi).limit_denominator(1 << 40) ** 2)
    xi2 = Fraction(N) / (Fraction(lo).limit_denominator(1 << 40) ** 2)
    inside = farey_in_interval(xi1, xi2, Q)
    nb_lo, nb_hi = farey_neighbours(xi1, xi2, Q)
    frs = list(inside)
    if nb_lo is not None:
        frs.append(nb_lo)
    if nb_hi is not None:
        frs.append(nb_hi)
    # order by critical point (decreasing in b/a); keep only wide forms a <= b <= (C+1) a (all fractions in [1, C] are)
    pts = sorted(((critical_point(a, b, N), a, b) for a, b in frs), key=lambda t: t[0])
    # assign gaps: gap between consecutive critical points goes to the endpoint of smaller denominator
    assigned = {i: [] for i in range(len(pts))}  # index -> list of endpoints (positions) it must reach
    for i in range(len(pts) - 1):
        (x0, a0, _), (x1, a1, _) = pts[i], pts[i + 1]
        seg_lo, seg_hi = max(x0, lo), min(x1, hi)
        if seg_lo > seg_hi:
            continue
        owner = i if a0 <= a1 else i + 1
        assigned[owner].extend([seg_lo, seg_hi])
    forms = []
    for i, (ps, a, b) in enumerate(pts):
        ends = assigned[i]
        if not ends:
            continue
        gmin = g(a, b, N, ps)
        need = max(g(a, b, N, e) for e in ends) - gmin
        W = int(math.ceil(need)) + 1  # least W with W - 1 >= need: the conservative covered set contains every assigned endpoint
        forms.append({"a": a, "b": b, "p_star": ps, "W": W})
    P = len(forms)
    SW = sum(f["W"] for f in forms)
    res = {"N": N, "p0": p0, "L": L, "Q": Q, "C": C, "n_fractions_inside": len(inside), "P": P, "Sigma_W": SW,
           "cost": P + 2 * math.sqrt(SW), "D_I": min((a for a, _ in inside), default=None), "forms": forms}
    if verify:
        ivs = sorted(covered_interval(f["a"], f["b"], N, f["W"]) for f in forms)
        reach = lo
        ok = True
        for x0, x1 in ivs:
            if x0 > reach + 1e-9 * L:
                ok = False
                break
            reach = max(reach, x1)
            if reach >= hi:
                break
        res["covers"] = bool(ok and reach >= hi - 1e-9 * L)
    return res


def single_form_cover(N: int, p0: float, L: float, a: int, b: int) -> Dict:
    """The window a single form (a, b) needs to cover I on its own, and the resulting cost 1 + 2 sqrt(W)."""
    lo, hi = p0 - L / 2, p0 + L / 2
    ps = critical_point(a, b, N)
    gmin = g(a, b, N, ps)
    need = max(g(a, b, N, lo), g(a, b, N, hi)) - gmin
    W = int(math.ceil(need)) + 1
    x0, x1 = covered_interval(a, b, N, W)
    return {"a": a, "b": b, "p_star": ps, "W": W, "cost": 1 + 2 * math.sqrt(W), "covers": bool(x0 <= lo + 1e-9 * L and x1 >= hi - 1e-9 * L)}


def best_single_form(N: int, p0: float, L: float, Q: int, C: float = 2.0) -> Dict:
    """The cheapest single-form cover of I among wide forms with a <= Q whose critical point lies within 2L of I."""
    lo, hi = p0 - 2 * L, p0 + 2 * L
    xi1 = Fraction(N) / (Fraction(hi).limit_denominator(1 << 40) ** 2)
    xi2 = Fraction(N) / (Fraction(lo).limit_denominator(1 << 40) ** 2)
    best = None
    for a, b in farey_in_interval(xi1, xi2, Q):
        if not (a <= b <= C * a + 1):
            continue
        r = single_form_cover(N, p0, L, a, b)
        if best is None or r["cost"] < best["cost"]:
            best = r
    return best


def landscape_experiment(bits: int, lam: float, count: int, seed: int = 17, C: float = 2.0, special: bool = True) -> Dict:
    """Farey families of order Q = round((N/L)^{1/5}) for `count` random centres in J_C on one modulus per index, plus the
    Fermat-type centre p0 = sqrt N - L (a critical point of the form (1, 1) within L of I).  Records P, Sigma_W and cost against
    P0 := L^{3/5} N^{-1/10} and L N^{-1/4}, the least denominator D(I) inside I, and the cheapest single form."""
    rng = random.Random(seed)
    rows = []
    for idx in range(count):
        sp = make_semiprime(bits, "rsa", seed, idx)
        N = int(sp.N)
        L = N ** lam
        Q = max(2, round((N / L) ** 0.2))
        lo_J, hi_J = math.sqrt(N / C), math.sqrt(N)
        p0 = rng.uniform(lo_J + L, hi_J - L)
        fam = farey_family(N, p0, L, Q, C)
        sing = best_single_form(N, p0, L, Q, C)
        P0 = L ** 0.6 * N ** (-0.1)
        row = {"N": N, "p0": p0, "L": L, "Q": Q, "P": fam["P"], "Sigma_W": fam["Sigma_W"], "cost": fam["cost"], "covers": fam["covers"],
               "D_I": fam["D_I"], "n_inside": fam["n_fractions_inside"], "P0": P0, "cost_over_P0": fam["cost"] / P0,
               "L_N14": L * N ** (-0.25), "best_single_cost": sing["cost"] if sing else None, "best_single_a": sing["a"] if sing else None,
               "best_single_over_P0": (sing["cost"] / P0) if sing else None}
        rows.append(row)
    out = {"bits": bits, "lambda": lam, "count": count, "seed": seed, "C": C, "rows": rows,
           "all_cover": all(r["covers"] for r in rows),
           "cost_over_P0_min": min(r["cost_over_P0"] for r in rows), "cost_over_P0_max": max(r["cost_over_P0"] for r in rows),
           "single_over_P0_median": sorted(r["best_single_over_P0"] for r in rows if r["best_single_over_P0"])[len(rows) // 2]}
    if special:
        sp = make_semiprime(bits, "rsa", seed, 0)
        N = int(sp.N)
        L = N ** lam
        p0 = math.sqrt(N) - L  # the Fermat form (1, 1) has its critical point sqrt N at distance L/2 from I
        fermat = single_form_cover(N, p0, L, 1, 1)
        Q = max(2, round((N / L) ** 0.2))
        fam = farey_family(N, p0, L, Q, C)
        out["fermat_type"] = {"N": N, "p0": p0, "L": L, "fermat_W": fermat["W"], "fermat_cost": fermat["cost"], "fermat_covers": fermat["covers"],
                              "L_N14": L * N ** (-0.25), "fermat_cost_over_LN14": fermat["cost"] / (L * N ** (-0.25)),
                              "farey_cost": fam["cost"], "farey_P": fam["P"], "farey_covers": fam["covers"]}
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[40, 50, 60])
    ap.add_argument("--lam", type=float, nargs="+", default=[0.26, 0.3, 0.34, 0.375])
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="results/e55_farey_cover.json")
    args = ap.parse_args()
    results = {}
    for bits in args.bits:
        for lam in args.lam:
            r = landscape_experiment(bits, lam, args.count, args.seed)
            results[f"{bits},{lam}"] = r
            ft = r.get("fermat_type", {})
            print(f"bits={bits} lam={lam}: all cover={r['all_cover']} cost/P0 in [{r['cost_over_P0_min']:.2f}, {r['cost_over_P0_max']:.2f}] "
                  f"single/P0 median {r['single_over_P0_median']:.2f}; Fermat-type: cost/(L N^-1/4) = {ft.get('fermat_cost_over_LN14', float('nan')):.2f} "
                  f"(covers={ft.get('fermat_covers')}), Farey cost there {ft.get('farey_cost', float('nan')):.1f}", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("FAREY_DONE")
