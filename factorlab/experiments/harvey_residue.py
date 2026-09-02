"""E51: Harvey's babystep-giantstep search restricted to a residue class of the tested integers.

Given N = pq (balanced), an element alpha of large order, the radius r, the baby count m and -- optionally -- the residue
p0 = p (mod M), the standard Lehman family (a, b), ab <= r, is searched as in Harvey's Algorithm 4.2 (arXiv:2010.05450):
for the covering cell the true value u = aq + bp lies in the window T_ab = {ceil(2 sqrt(abN)) + j : 0 <= j < delta_ab},
delta_ab = sqrt N / (4 r sqrt(ab)), and alpha^u = alpha^{aN + b} (mod p) by Fermat.  Knowing p0 gives q = N p0^{-1} (mod M) and
hence u = a N p0^{-1} + b p0 =: omega_ab (mod M), so only every M-th integer of the window is tested: u = c_ab + j0 + M t.
Writing t = i + m k, the babies are beta^i with beta = alpha^M and the giants g_{ab,k} = alpha^{aN + b - c_ab - j0} beta^{-mk};
a collision beta^i = g (mod p) is detected for all giants at once by evaluating F(X) = prod_i (X - beta^i) at the giants
(product tree + multipoint evaluation modulo N) and taking gcds (Harvey's Algorithm 4.1).  Exact matches modulo N are
resolved by sorting, and the candidate u is decoded by the square test u^2 - 4abN = square (Harvey's Lemma 3.1).

Work is counted in babies, giants and cells; with r = m = N^{1/5} M^{-2/5} the total is N^{1/5 + o(1)} M^{-2/5}.

Run:  python -m factorlab.experiments.harvey_residue --bits 40 --count 10 --moduli 1 4 16 64 256 --out results/e51_harvey_residue.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from gmpy2 import gcd, invert, iroot, is_square, isqrt, mpz, powmod

from factorlab.experiments.order_selection import select_order_element
from factorlab.gen import make_semiprime


def ceil_2sqrt(k: int, N: int) -> int:
    v = 4 * k * N
    s = int(isqrt(mpz(v)))
    return s if s * s == v else s + 1


def window_int(N: int, r: int, ab: int) -> int:
    """W_ab = ceil(sqrt N / (4 r sqrt(ab))): least W with (4 r W)^2 ab >= N."""
    lo, hi = 1, 1
    while (4 * r * hi) ** 2 * ab < N:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if (4 * r * mid) ** 2 * ab >= N:
            hi = mid
        else:
            lo = mid + 1
    return lo


def decode(u: int, ab: int, N: int) -> Optional[int]:
    """Harvey's Lemma 3.1: if u = aq + bp then u^2 - 4abN = (aq - bp)^2; return a proper factor or None."""
    d = u * u - 4 * ab * N
    if d < 0 or not is_square(mpz(d)):
        return None
    s = int(isqrt(mpz(d)))
    for v in (u + s, u - s):  # 2aq or 2bp
        g = int(gcd(mpz(v), mpz(N)))
        if 1 < g < N:
            return g
    return None


def _product_tree_eval(roots: List[int], points: List[int], N: int) -> List[int]:
    """Values of F(X) = prod (X - root) at the points, modulo N (python-flint; division-free remainder tree)."""
    from flint import fmpz_mod_ctx, fmpz_mod_poly_ctx

    mctx = fmpz_mod_ctx(N)
    ctx = fmpz_mod_poly_ctx(mctx)
    # product tree
    level = [ctx([(-rt) % N, 1]) for rt in roots]
    while len(level) > 1:
        nxt = [level[i] * level[i + 1] for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    F = level[0]
    return [int(v) for v in F.multipoint_evaluate([mctx(x) for x in points])]


def harvey_residue_factor(N: int, alpha: int, r: int, m: int, M: int = 1, p0: int = 0) -> Dict:
    """Run the residue-restricted Harvey search; return the factor found (or None) and work counters.
    Preconditions: M >= 1; for M > 1, p0 = p (mod M) with gcd(p0, M) = 1 (a common divisor of p0 and M divides p, so
    gcd(N, gcd(p0, M)) is then returned directly as a factor when proper, and a precondition error is raised otherwise)."""
    N = int(N)
    if M < 1:
        raise ValueError("M must be a positive integer")
    work = {"babies": 0, "giants": 0, "cells": 0, "tested": 0}
    if M > 1:
        g0 = math.gcd(p0 % M, M)
        if g0 > 1:
            g = math.gcd(N, g0)
            if 1 < g < N:
                return {"factor": g, "how": "gcd(p0, M)", "work": work}
            raise ValueError("p0 must be coprime to M (p0 = p mod M with p prime > M)")
    beta = int(powmod(alpha, M, N))
    # Step (1): babies and the order check gcd(beta^i - 1, N)
    babies: List[int] = []
    x = 1
    for i in range(m):
        babies.append(x)
        if i > 0:
            g = int(gcd(mpz(x - 1), mpz(N)))
            if 1 < g < N:
                return {"factor": g, "how": "order", "work": work}
        x = x * beta % N
    work["babies"] = m
    inv_p0 = int(invert(mpz(p0), mpz(M))) if M > 1 else 0
    beta_inv_m = int(powmod(beta, -m, N))
    giants: List[int] = []
    labels: List[Tuple[int, int, int, int]] = []  # (a, b, j0, k)
    for a in range(1, r + 1):
        for b in range(1, r // a + 1):
            ab = a * b
            c = ceil_2sqrt(ab, N)
            W = window_int(N, r, ab)
            work["cells"] += 1
            if M > 1:
                omega = (a * (N % M) * inv_p0 + b * p0) % M
                j0 = (omega - c) % M
            else:
                j0 = 0
            if j0 >= W:
                continue  # no tested integer of the class in this window (cannot happen for the covering cell)
            K = (W - j0 + M - 1) // M  # tested integers in the class
            work["tested"] += K
            g = int(powmod(alpha, a * N + b - c - j0, N))  # alpha^{aN+b} alpha^{-(c+j0)}
            for k in range((K + m - 1) // m):
                giants.append(g)
                labels.append((a, b, j0, k))
                g = g * beta_inv_m % N
    work["giants"] = len(giants)
    # Step (3): exact matches modulo N -> decode directly
    baby_index = {v: i for i, v in enumerate(babies)}
    remaining_g, remaining_l = [], []
    for g, (a, b, j0, k) in zip(giants, labels):
        i = baby_index.get(g)
        if i is not None:
            u = ceil_2sqrt(a * b, N) + j0 + M * (i + m * k)
            f = decode(u, a * b, N)
            if f:
                return {"factor": f, "how": "exact", "work": work}
        else:
            remaining_g.append(g)
            remaining_l.append((a, b, j0, k))
    # Step (4): collisions modulo p or q via the product tree
    if remaining_g:
        vals = _product_tree_eval(babies, remaining_g, N)
        for v, g, (a, b, j0, k) in zip(vals, remaining_g, remaining_l):
            gg = int(gcd(mpz(v), mpz(N)))
            if 1 < gg < N:
                return {"factor": gg, "how": "collision", "work": work}
            if gg == N:  # collisions modulo both primes at different babies: isolate
                for i, bb in enumerate(babies):
                    gi = int(gcd(mpz(g - bb), mpz(N)))
                    if 1 < gi < N:
                        return {"factor": gi, "how": "collision-isolated", "work": work}
    return {"factor": None, "how": "none", "work": work}


def parameters(N: int, M: int) -> Tuple[int, int]:
    """r = m = round(N^{1/5} M^{-2/5}), at least 2."""
    v = max(2, round(N ** 0.2 * M ** -0.4))
    return v, v


def order_collision_factor(N: int, beta: int, D: int) -> Tuple[Optional[int], int]:
    """Harvey's Algorithm 4.1 on the powers of beta.  With h = isqrt(D) + 1 and J = ceil(D/h) giants the exponents jh - i cover
    exactly [1, Jh] (Jh >= D); returns (factor, Jh) where factor is a proper factor of N if some e <= Jh has gcd(beta^e - 1, N)
    proper -- found directly, or from the exact order of beta when the collision is an exact equality modulo N and the
    component orders differ -- and None otherwise (no such e <= Jh, or equal component orders)."""
    N = int(N)
    h = math.isqrt(D) + 1
    J = (D + h - 1) // h
    searched = J * h
    babies: List[int] = []
    x = 1
    for i in range(h):
        babies.append(x)
        if i > 0:
            g = int(gcd(mpz(x - 1), mpz(N)))
            if 1 < g < N:
                return g, searched
        x = x * beta % N
    step = int(powmod(beta, h, N))
    giants, js = [], []
    y = step
    for j in range(1, J + 1):
        giants.append(y)
        js.append(j)
        y = y * step % N
    baby_index = {v: i for i, v in enumerate(babies)}
    for gnt, j in zip(giants, js):  # exact equalities modulo N: the exact order, then the divisor gcds (Hittmeir's Lemma 2.3)
        i = baby_index.get(gnt)
        if i is not None and j * h - i > 0:
            from sympy import factorint

            E = j * h - i
            for pr, k in factorint(E).items():
                for _ in range(k):
                    if powmod(beta, E // int(pr), N) == 1:
                        E //= int(pr)
                    else:
                        break
            for pr in factorint(E):
                g = int(gcd(powmod(beta, E // int(pr), N) - 1, N))
                if 1 < g < N:
                    return g, searched
            return None, searched  # ord_p(beta) = ord_q(beta)
    vals = _product_tree_eval(babies, giants, N)
    for gnt, v in zip(giants, vals):
        g = int(gcd(mpz(v), mpz(N)))
        if 1 < g < N:
            return g, searched
        if g == N:  # this giant agrees with one baby modulo p and with another modulo q: isolate (Harvey's Algorithm 4.1, steps 6-8)
            for b in babies:
                gi = int(gcd(mpz(gnt - b), mpz(N)))
                if 1 < gi < N:
                    return gi, searched
    return None, searched


def common_factor_attack(N: int, max_alpha: int = 1000) -> Dict:
    """Lemma (a large common factor of p - 1 and q - 1): with g = gcd(p-1, q-1), beta = alpha^{N-1} has orders dividing the
    coprime numbers (p-1)/g and (q-1)/g modulo p and q, so a collision search on beta with doubling bound finds a factor in
    O~(N^{1/4}/sqrt g) operations for every alpha with alpha^{N-1} != 1 (mod N).

    Returns {"factor", "alpha", "liars", "how", "horizon"}: `liars` is the number of bases with alpha^{N-1} = 1 skipped before
    the successful one; `how` is "gcd(alpha, N)", "gcd(beta - 1, N)" (a component order equal to 1) or "collision" (the doubling
    search, including its exact-order and isolation branches); `horizon` is J*h of the final search (the largest exponent the
    search covered), NOT the least exponent e with gcd(beta^e - 1, N) proper -- that exponent is a property of (p, q, alpha) and
    is recomputed from the factors by the experiment driver in `common_factor.py`."""
    N = int(N)
    liars = 0
    for alpha in range(2, max_alpha):
        g = int(gcd(mpz(alpha), mpz(N)))
        if 1 < g < N:
            return {"factor": g, "alpha": alpha, "liars": liars, "how": "gcd(alpha, N)", "horizon": 0}
        beta = int(powmod(alpha, N - 1, N))
        if beta == 1:
            liars += 1
            continue  # a Fermat liar: alpha lies in the subgroup of order g
        g = int(gcd(mpz(beta - 1), mpz(N)))
        if 1 < g < N:
            return {"factor": g, "alpha": alpha, "liars": liars, "how": "gcd(beta - 1, N)", "horizon": 1}
        D = 4
        horizon = 0
        while D < N:
            f, horizon = order_collision_factor(N, beta, D)
            if f:
                return {"factor": f, "alpha": alpha, "liars": liars, "how": "collision", "horizon": horizon}
            D *= 4
        return {"factor": None, "alpha": alpha, "liars": liars, "how": "none", "horizon": horizon}
    return {"factor": None, "alpha": None, "liars": liars, "how": "none", "horizon": 0}


def experiment(bits: int, count: int, moduli: List[int], seed: int = 12) -> Dict:
    rows = []
    for i in range(count):
        sp = make_semiprime(bits, "rsa", seed, i)
        N, p, q = int(sp.N), int(sp.p), int(sp.q)
        row = {"N": N, "runs": []}
        for M in moduli:
            r, m = parameters(N, M)
            sel = select_order_element(N, M * m)
            if sel["outcome"] == "factor":
                row["runs"].append({"M": M, "factored_by_selection": True})
                continue
            res = harvey_residue_factor(N, sel["alpha"], r, m, M, p % M if M > 1 else 0)
            ok = res["factor"] in (p, q)
            w = res["work"]
            row["runs"].append({"M": M, "r": r, "m": m, "ok": ok, "how": res["how"], **w, "total": w["babies"] + w["giants"] + w["cells"]})
        rows.append(row)
    summary = {}
    for M in moduli:
        runs = [rr for row in rows for rr in row["runs"] if rr["M"] == M and "total" in rr]
        if runs:
            summary[str(M)] = {"successes": sum(rr["ok"] for rr in runs), "runs": len(runs),
                               "mean_total_work": sum(rr["total"] for rr in runs) / len(runs),
                               "mean_tested": sum(rr["tested"] for rr in runs) / len(runs)}
    return {"bits": bits, "count": count, "moduli": moduli, "summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, default=40)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--moduli", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    ap.add_argument("--out", default="results/e51_harvey_residue.json")
    args = ap.parse_args()
    if any(M < 1 for M in args.moduli):
        ap.error("--moduli must be positive integers")
    res = experiment(args.bits, args.count, args.moduli)
    base = res["summary"].get("1", {}).get("mean_total_work")
    for M, sm in res["summary"].items():
        ratio = sm["mean_total_work"] / base if base else float("nan")
        print(f"M={M}: {sm['successes']}/{sm['runs']} factored; mean work {sm['mean_total_work']:.0f} (ratio to M=1: {ratio:.3f}, "
              f"M^(-2/5) = {int(M) ** -0.4:.3f}); mean tested {sm['mean_tested']:.0f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
