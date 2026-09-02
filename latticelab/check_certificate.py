"""Independent checker for the profile-floor certificates.

This script deliberately shares no code with `latticelab.profile_floor`: the dual multipliers are recomputed by the direct O(d beta)
accumulation with an explicit block-coverage test (no sliding window), the certificate identity e_1 = sum_i y_i a_i + z 1 and y >= 0 are
re-verified exactly in rational arithmetic, the value h(eps) = sum_i y_i (log chat(beta_i) - eps) is re-enclosed in arb ball arithmetic, and
the comparison with the target is re-decided with directed endpoints.  Two modes:

  head:    python -m latticelab.check_certificate --d 1003 --beta 413 --eps 0 --gsa-beta 403
           decides whether the zero-slack root-Hermite floor at (d, beta) is <= delta_GSA(403) = chat(403)^{1/402}.
  detect:  python -m latticelab.check_certificate --detect --d 1030 --b 417 --m 517 --k 2 --eta1 3
           decides whether log(sigma sqrt b) <= l^tight_{d-b+1}(d, b, 0; S = m log q) with q = 3329, sigma^2 = eta1/2 (the prefix-volume
           form of the Kyber round-3 primal condition).

A decision is printed only when the two balls are disjoint at some precision up to --max-prec; otherwise the script reports 'undecided'.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Tuple


def block_sizes(d: int, beta: int) -> List[int]:
    return [min(beta, d - i + 1) for i in range(1, d)]


def covers(i: int, k: int, bs: List[int]) -> bool:
    """Block i (1-based, size bs[i-1]) covers position k > i."""
    return i < k <= i + bs[i - 1] - 1


def solve_multipliers(d: int, beta: int, rhs: List) -> Tuple[List, object]:
    """Exact (w_1..w_{d-1}, z) with rhs = sum_i w_i a_i + z 1, by the direct forward recurrence with an explicit coverage test."""
    from flint import fmpq

    bs = block_sizes(d, beta)
    z = sum(rhs, fmpq(0)) / d
    w: List = []
    for k in range(1, d):
        incoming = fmpq(0)
        for i in range(1, k):
            if covers(i, k, bs):
                incoming += w[i - 1] / bs[i - 1]
        w.append((rhs[k - 1] - z + incoming) / (1 - fmpq(1, bs[k - 1])))
    return w, z


def verify_identity(d: int, beta: int, w: List, z, rhs: List) -> bool:
    """Exact re-check of every coordinate of rhs = sum_i w_i a_i + z 1 by explicit accumulation of every block's entries."""
    from flint import fmpq

    bs = block_sizes(d, beta)
    coeff = [fmpq(z) for _ in range(d)]
    for i in range(1, d):
        n = bs[i - 1]
        coeff[i - 1] += w[i - 1] * (1 - fmpq(1, n))
        for k in range(i + 1, i + n):
            coeff[k - 1] -= w[i - 1] / n
    return all(coeff[k] == rhs[k] for k in range(d))


def log_chat(n: int, prec: int):
    from flint import arb, ctx, fmpq

    ctx.prec = prec
    if n == 1:
        return -arb(2).log()
    h = arb(fmpq(n, 2))
    return -(h * arb.pi().log() - (h + 1).lgamma()) / n


def enclose_value(d: int, beta: int, w: List, z, eps, log_vol, prec: int):
    """arb ball of sum_i w_i (log chat(beta_i) - eps) + z log_vol."""
    from flint import arb, ctx, fmpq

    ctx.prec = prec
    bs = block_sizes(d, beta)
    cache = {}
    total = arb(0)
    for wi, n in zip(w, bs):
        if wi == 0:
            continue
        if n not in cache:
            cache[n] = log_chat(n, prec) - arb(eps)
            ctx.prec = prec
        total += arb(wi) * cache[n]
    return total + arb(z) * arb(log_vol)


def decide(lhs_fn, rhs_fn, prec: int, max_prec: int) -> Tuple[str, int]:
    """Directed decision lhs <= rhs on balls, doubling the precision until disjoint."""
    from flint import ctx

    p = prec
    while True:
        ctx.prec = p
        L, R = lhs_fn(p), rhs_fn(p)
        if L.upper() <= R.lower():
            return "passes (lhs <= rhs rigorously)", p
        if L.lower() > R.upper():
            return "fails (lhs > rhs rigorously)", p
        if p >= max_prec:
            return "undecided", p
        p = min(2 * p, max_prec)


def main(argv=None) -> int:
    from flint import arb, fmpq

    ap = argparse.ArgumentParser(description="independent checker for profile-floor certificates")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--beta", type=int, help="head mode: blocksize of the class")
    ap.add_argument("--eps", default="0", help="slack, an exact rational or decimal string")
    ap.add_argument("--gsa-beta", type=int, help="head mode: target delta_GSA(gsa_beta)")
    ap.add_argument("--detect", action="store_true", help="detection mode")
    ap.add_argument("--b", type=int); ap.add_argument("--m", type=int); ap.add_argument("--k", type=int); ap.add_argument("--eta1", type=int)
    ap.add_argument("--q", type=int, default=3329)
    ap.add_argument("--prec", type=int, default=256); ap.add_argument("--max-prec", type=int, default=4096)
    a = ap.parse_args(argv)
    if a.d < 2:
        ap.error("--d must be at least 2")
    if a.prec < 16 or a.max_prec < a.prec:
        ap.error("need 16 <= --prec <= --max-prec")
    if a.q < 2:
        ap.error("--q must be at least 2")
    t0 = time.time()
    eps = fmpq(a.eps) if "/" in a.eps or a.eps.lstrip("-").isdigit() else a.eps
    if a.detect:
        if a.b is None or a.m is None or a.k is None or a.eta1 is None:
            ap.error("detect mode needs --b, --m, --k and --eta1")
        d, b = a.d, a.b
        if not 2 <= b <= d:
            ap.error("need 2 <= --b <= --d")
        if 2 * b >= d + 1:
            ap.error("detection mode needs 2b < d + 1 (the domain of the primal condition)")
        if a.k < 1 or a.eta1 < 1 or a.m < 0:
            ap.error("need --k >= 1, --eta1 >= 1 and --m >= 0")
        kpos = d - b + 1
        rhs = [fmpq(1 if i == kpos - 1 else 0) for i in range(d)]
        w, z = solve_multipliers(d, b, rhs)
        ok = verify_identity(d, b, w, z, rhs)
        print(f"entry certificate for l_{kpos} at (d, b) = ({d}, {b}): identity {'verified' if ok else 'FAILED'} exactly; z = {z}")
        if not ok:
            return 1

        def lhs(p):  # log(sigma sqrt b) = (1/2) log(eta1 b / 2)
            from flint import ctx
            ctx.prec = p
            return arb(fmpq(a.eta1 * b, 2)).log() / 2

        def rhs_fn(p):
            from flint import ctx
            ctx.prec = p
            return enclose_value(d, b, w, z, eps, arb(a.m) * arb(a.q).log(), p)

        verdict, p = decide(lhs, rhs_fn, a.prec, a.max_prec)
        print(f"detection: log(sigma sqrt b) = {lhs(p)} vs l^tight_{kpos} = {rhs_fn(p)} -> {verdict} at {p} bits [{time.time()-t0:.1f}s]")
        return 0
    d, beta = a.d, a.beta
    if beta is None or a.gsa_beta is None:
        ap.error("head mode needs --beta and --gsa-beta")
    if not 2 <= beta <= d or not 2 <= a.gsa_beta <= d:
        ap.error("need 2 <= --beta <= --d and 2 <= --gsa-beta <= --d")
    rhs = [fmpq(1 if i == 0 else 0) for i in range(d)]
    y, z = solve_multipliers(d, beta, rhs)
    ok = verify_identity(d, beta, y, z, rhs)
    pos = all(v > 0 for v in y)
    print(f"head certificate at (d, beta) = ({d}, {beta}): identity {'verified' if ok else 'FAILED'} exactly; y > 0: {pos}; z = {z}; "
          f"y_1 = {float(y[0]):.6f}, min y = {float(min(y)):.3e}, sum y = {float(sum(y)):.4f}")
    if not (ok and pos):
        return 1

    def lhs(p):
        from flint import ctx
        ctx.prec = p
        return (enclose_value(d, beta, y, z, eps, 0, p) / (d - 1)).exp()

    def rhs_fn(p):
        from flint import ctx
        ctx.prec = p
        return (log_chat(a.gsa_beta, p) / (a.gsa_beta - 1)).exp()

    verdict, p = decide(lhs, rhs_fn, a.prec, a.max_prec)
    print(f"floor delta_0 = {lhs(p)} vs target delta_GSA({a.gsa_beta}) = {rhs_fn(p)} -> {verdict} at {p} bits [{time.time()-t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
