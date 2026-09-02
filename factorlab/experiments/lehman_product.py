"""E47: three direct tests around the Lehman product and the localised family.

(a) Spurious collisions.  Pi_r(alpha) = prod_{ab<=r} prod_{i<W_ab} (alpha^{c_ab+i} - alpha^{aN+b}) mod N is divisible by p
    whenever the standard family covers p.  It may also be divisible by q (a spurious collision), in which case
    gcd(Pi_r, N) = N and the scalar product alone gives no factor.  We measure, on random rsa moduli, how often that
    happens for alpha = 2 (and whether alpha = 3 separates), and we verify that a mod-p hit always exists.

(b) Localised family.  F(I, r) = {(a, b): ab <= r, dist(a/b, {x^2/N : x in I}) < 2/(b sqrt r)}.  For intervals I of
    length L = N^lambda containing p we check that the covering cell of the standard family lies in F(I, r), and compare
    |F| and Sigma_W with the bounds of the localised-count lemma.

(c) Wronskian constants.  c_M(C) = (2M)^{-1/2} min_{t in [C^{-1/2}, 1]} sigma_min(W(t)) for the matrix W(t)_{k,m} =
    (m)_k t^{m-k}, k = 1..2M, m in {+-1..+-M}: the uniform derivative bound of the bounded-degree theorem.  Computed on a grid.

Run:  python -m factorlab.experiments.lehman_product --spurious --bits 36 --count 60 --localised --wronskian --out results/e47_lehman_product.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import numpy as np
from gmpy2 import gcd, iroot, isqrt, mpz, powmod

from factorlab.gen import make_semiprime


def ceil_2sqrt(k: int, N: int) -> int:
    v = 4 * k * N
    s = int(isqrt(mpz(v)))
    return s if s * s == v else s + 1


def window_int(N: int, r: int, ab: int) -> int:
    """W^int_ab = ceil(sqrt N / (4 r sqrt(ab))), computed exactly: smallest W with (4 r W)^2 ab >= N."""
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


def standard_family(N: int, r: int):
    """Yield (a, b, c_ab, W_ab) over the standard Lehman family at radius r."""
    for a in range(1, r + 1):
        for b in range(1, r // a + 1):
            yield a, b, ceil_2sqrt(a * b, N), window_int(N, r, a * b)


def lehman_product_stats(N: int, p: int, q: int, r: int, alpha: int = 2, half: bool = False) -> Dict:
    """Per-factor divisibility of Pi_r(alpha) by p and q (known here), and gcd(Pi_r, N).
    With half=True only the cells a < b are used (the half-family, which excludes the mirrored cells a/b ~ q/p).

    A factor alpha^{c+i} - alpha^{aN+b} is a *geometric hit* modulo p when c + i = aq + bp (the cell covers p), and a
    *mirrored geometric hit* modulo q when c + i = ap + bq (the mirror cell (b, a) covers p); every other zero modulo p or
    q is a *coincidental* zero (a relation in the multiplicative group).  All four counts are recorded, with the gap q - p."""
    N, p, q = mpz(N), mpz(p), mpz(q)
    hits_p = hits_q = both = 0
    geo_p = geo_q = 0
    factors = 0
    prod = mpz(1)
    for a, b, c, W in standard_family(int(N), r):
        if half and a >= b:
            continue
        target = powmod(alpha, a * N + b, N)
        x = powmod(alpha, c, N)
        u_true = a * q + b * p  # the cell's value at (p, q)
        u_mirror = a * p + b * q  # the value of the mirror cell (b, a), zero modulo q by the mirror lemma
        for i in range(W):
            f = (x - target) % N
            factors += 1
            dp = f % p == 0
            dq = f % q == 0
            hits_p += dp
            hits_q += dq
            both += dp and dq
            if c + i == u_true:
                geo_p += 1
            if c + i == u_mirror:
                geo_q += 1
            prod = prod * f % N if f else mpz(0)
            x = x * alpha % N
    g = int(gcd(prod, N))
    return {"alpha": alpha, "half": half, "factors": factors, "hits_p": hits_p, "hits_q": hits_q, "hits_both": both,
            "geo_hits_p": geo_p, "geo_hits_q": geo_q, "coinc_p": hits_p - geo_p, "coinc_q": hits_q - geo_q,
            "gap": int(q - p), "gcd_kind": {1: "1", int(p): "p", int(q): "q", int(N): "N"}.get(g, "other")}


def spurious_experiment(bits: int, count: int, seed: int = 21) -> Dict:
    rows = []
    for i in range(count):
        sp = make_semiprime(bits, "rsa", seed, i)
        N, p, q = int(sp.N), int(sp.p), int(sp.q)
        r = int(iroot(mpz(N), 3)[0])
        s2 = lehman_product_stats(N, p, q, r, 2)
        h2 = lehman_product_stats(N, p, q, r, 2, half=True)
        row = {"N": N, "p": p, "q": q, "r": r, "alpha2": s2, "half_alpha2": h2}
        if s2["gcd_kind"] == "N":
            row["alpha3"] = lehman_product_stats(N, p, q, r, 3)
        rows.append(row)
    n = len(rows)
    half_gcdN = [rw for rw in rows if rw["half_alpha2"]["gcd_kind"] == "N"]
    half_nop = [rw for rw in rows if rw["half_alpha2"]["hits_p"] == 0]
    return {
        "bits": bits, "moduli": n, "seed": seed,
        "no_mod_p_hit": sum(rw["alpha2"]["hits_p"] == 0 for rw in rows),
        "gcd_N_alpha2": sum(rw["alpha2"]["gcd_kind"] == "N" for rw in rows),
        "gcd_p_alpha2": sum(rw["alpha2"]["gcd_kind"] == "p" for rw in rows),
        "any_mod_q_hit_alpha2": sum(rw["alpha2"]["hits_q"] > 0 for rw in rows),
        "mean_hits_p": float(np.mean([rw["alpha2"]["hits_p"] for rw in rows])),
        "mean_hits_q": float(np.mean([rw["alpha2"]["hits_q"] for rw in rows])),
        "mean_factors": float(np.mean([rw["alpha2"]["factors"] for rw in rows])),
        "geo_hits_equal_mirrors": sum(rw["alpha2"]["geo_hits_p"] == rw["alpha2"]["geo_hits_q"] for rw in rows),
        "mean_geo_hits_p": float(np.mean([rw["alpha2"]["geo_hits_p"] for rw in rows])),
        "mean_coinc_p": float(np.mean([rw["alpha2"]["coinc_p"] for rw in rows])),
        "mean_coinc_q": float(np.mean([rw["alpha2"]["coinc_q"] for rw in rows])),
        "alpha3_separates": sum(1 for rw in rows if "alpha3" in rw and rw["alpha3"]["gcd_kind"] in ("p", "q")),
        "half_mean_factors": float(np.mean([rw["half_alpha2"]["factors"] for rw in rows])),
        "half_no_mod_p_hit": len(half_nop),
        "half_no_mod_p_hit_gaps": sorted(rw["half_alpha2"]["gap"] for rw in half_nop),
        "half_gcd_p": sum(rw["half_alpha2"]["gcd_kind"] == "p" for rw in rows),
        "half_gcd_N": len(half_gcdN),
        "half_any_mod_q_hit": sum(rw["half_alpha2"]["hits_q"] > 0 for rw in rows),
        "half_mean_hits_p": float(np.mean([rw["half_alpha2"]["hits_p"] for rw in rows])),
        "half_mean_hits_q": float(np.mean([rw["half_alpha2"]["hits_q"] for rw in rows])),
        "half_geo_hits_q": sum(rw["half_alpha2"]["geo_hits_q"] for rw in rows),
        "half_mean_coinc_q_given_gcd_N": float(np.mean([rw["half_alpha2"]["coinc_q"] for rw in half_gcdN])) if half_gcdN else None,
        "half_max_coinc_q": max(rw["half_alpha2"]["coinc_q"] for rw in rows),
        "max_zeros_any_alpha": max(max(rw["alpha2"]["hits_p"] + rw["alpha2"]["hits_q"] - rw["alpha2"]["hits_both"],
                                       rw.get("alpha3", {"hits_p": 0, "hits_q": 0, "hits_both": 0})["hits_p"]
                                       + rw.get("alpha3", {"hits_p": 0, "hits_q": 0, "hits_both": 0})["hits_q"]
                                       - rw.get("alpha3", {"hits_p": 0, "hits_q": 0, "hits_both": 0})["hits_both"]) for rw in rows),
        "rows": rows,
    }


# ------------------------------------------------------------------ (b) localised family

def in_localised_family(a: int, b: int, R_lo: Fraction, R_hi: Fraction, r: int) -> bool:
    """Exact test of dist(a/b, [R_lo, R_hi]) < 2/(b sqrt r) in rational arithmetic (square the positive inequality)."""
    x = Fraction(a, b)
    if R_lo <= x <= R_hi:
        return True
    d = (R_lo - x) if x < R_lo else (x - R_hi)  # positive rational distance
    return (d * b) ** 2 * r < 4


def localised_family(N: int, I_lo: Fraction, I_hi: Fraction, r: int) -> List[Tuple[int, int, int]]:
    """Cells (a, b, W_ab) of F(I, r): ab <= r and dist(a/b, R(I)) < 2/(b sqrt r), R(I) = [I_lo^2/N, I_hi^2/N].
    A slightly widened float range generates candidates; each is then filtered by the exact rational predicate."""
    R_lo, R_hi = I_lo * I_lo / N, I_hi * I_hi / N
    cells = []
    sqrt_r = math.sqrt(r)
    for b in range(1, r + 1):
        tol = 2.0 / (b * sqrt_r) * 1.001 + 1e-12
        a_lo = math.ceil((float(R_lo) - tol) * b) - 1
        a_hi = math.floor((float(R_hi) + tol) * b) + 1
        for a in range(max(1, a_lo), a_hi + 1):
            if a * b > r:
                break
            if in_localised_family(a, b, R_lo, R_hi, r):
                cells.append((a, b, window_int(N, r, a * b)))
    return cells


def covering_cell(N: int, p: int, q: int, r: int):
    """A standard cell covering p in the model's sense (|aq + bp - n| < 1 for a tested n), or None."""
    for a, b, c, W in standard_family(N, r):
        u = a * q + b * p
        if c <= u <= c + W - 1:
            return (a, b)
    return None


def localised_check(bits: int, count: int, lambdas: Sequence[float] = (0.5, 0.47, 0.44), C: float = 2.0, seed: int = 4) -> Dict:
    rows = []
    for i in range(count):
        sp = make_semiprime(bits, "rsa", seed, i)
        N, p, q = int(sp.N), int(sp.p), int(sp.q)
        rng = np.random.default_rng(seed + i)
        for lam in lambdas:
            L = int(N ** lam)
            off = int(rng.integers(0, L))
            I_lo, I_hi = Fraction(p - off), Fraction(p - off + L)
            r = math.ceil((N / L) ** 0.4)
            fam = localised_family(N, I_lo, I_hi, r)
            cov = covering_cell(N, p, q, r)
            in_family = cov is not None and any((a, b) == cov for a, b, _ in fam)
            P = len(fam)
            SW = sum(W for _, _, W in fam)
            sqrtN = math.sqrt(N)
            bound_P = 2 * C * L * r / sqrtN + 3 * math.sqrt(2 * C * r)
            bound_SW = C * L / math.sqrt(r) + math.sqrt(2 * C * N) * math.log(6 * C * r) / (2 * r) + P
            rows.append({"N": N, "lambda": lam, "L": L, "r": r, "covering_cell": cov, "in_family": in_family,
                         "P": P, "P_bound": bound_P, "Sigma_W": SW, "Sigma_W_bound": bound_SW,
                         "full_family_cells": sum(1 for _ in standard_family(N, r))})
    return {"bits": bits, "C": C, "rows": rows,
            "all_covered": all(rw["in_family"] for rw in rows),
            "all_bounds_hold": all(rw["P"] <= rw["P_bound"] and rw["Sigma_W"] <= rw["Sigma_W_bound"] for rw in rows)}


# ------------------------------------------------------------------ (c) Wronskian constants

def wronskian_matrix(t: float, M: int) -> np.ndarray:
    ms = [m for m in range(-M, M + 1) if m != 0]
    W = np.zeros((2 * M, 2 * M))
    for col, m in enumerate(ms):
        for k in range(1, 2 * M + 1):
            ff = 1.0
            for j in range(k):
                ff *= (m - j)
            W[k - 1, col] = ff * t ** (m - k)
    return W


def wronskian_constant(M: int, C: float = 2.0, grid: int = 4001) -> Dict:
    ts = np.linspace(C ** -0.5, 1.0, grid)
    sig = np.array([np.linalg.svd(wronskian_matrix(t, M), compute_uv=False)[-1] for t in ts])
    i = int(np.argmin(sig))
    return {"M": M, "C": C, "min_sigma": float(sig[i]), "argmin_t": float(ts[i]), "c_M": float(sig[i] / math.sqrt(2 * M))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spurious", action="store_true")
    ap.add_argument("--localised", action="store_true")
    ap.add_argument("--wronskian", action="store_true")
    ap.add_argument("--bits", type=int, default=36)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--out", default="results/e47_lehman_product.json")
    args = ap.parse_args()
    res: Dict = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            res = json.load(fh)
    if args.spurious:
        res[f"spurious_{args.bits}"] = s = spurious_experiment(args.bits, args.count)
        print(f"spurious at {args.bits} bits: moduli={s['moduli']} no_mod_p_hit={s['no_mod_p_hit']} gcd=N for alpha=2: {s['gcd_N_alpha2']} "
              f"(alpha=3 separates {s['alpha3_separates']}); any mod-q hit: {s['any_mod_q_hit_alpha2']}; mean hits_p={s['mean_hits_p']:.2f} "
              f"hits_q={s['mean_hits_q']:.3f} factors={s['mean_factors']:.0f}", flush=True)
        print(f"  half-family (a<b): no_mod_p_hit={s['half_no_mod_p_hit']} gcd=p: {s['half_gcd_p']} gcd=N: {s['half_gcd_N']} "
              f"any mod-q hit: {s['half_any_mod_q_hit']} mean hits_p={s['half_mean_hits_p']:.2f} hits_q={s['half_mean_hits_q']:.3f}", flush=True)
    if args.localised:
        res[f"localised_{args.bits}"] = lc = localised_check(args.bits, min(args.count, 12))
        print(f"localised at {args.bits} bits: all covered={lc['all_covered']} all bounds hold={lc['all_bounds_hold']}", flush=True)
        for rw in lc["rows"][:6]:
            print(f"  lambda={rw['lambda']}: r={rw['r']} P={rw['P']} (bound {rw['P_bound']:.0f}, full family {rw['full_family_cells']}) "
                  f"Sigma_W={rw['Sigma_W']} (bound {rw['Sigma_W_bound']:.0f}) covering cell in family: {rw['in_family']}", flush=True)
    if args.wronskian:
        res["wronskian"] = [wronskian_constant(M) for M in (1, 2, 3, 4)]
        for w in res["wronskian"]:
            print(f"M={w['M']}: min sigma={w['min_sigma']:.4g} at t={w['argmin_t']:.3f}, c_M={w['c_M']:.4g}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
