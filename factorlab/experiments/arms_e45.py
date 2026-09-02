"""E45: two direct attacks on named-only arms of the escape audit.

(a) Class count for even periods.  A drift-free family s = A d^2 + C has members k_-+(d) = (A d^2 -+ d + C)/2 on the
    admissible set  A = {d : both members are integers}.  The sharpened class-count lemma gives n <= 2^{omega(q)} classes
    modulo the least period q, including even q (the earlier bound allowed 2^{omega(q)+1} for even q).  We enumerate every family with alpha = A q^2, gamma = C q^2 in
    a box, compute the admissible set exactly (period divides 4 q^2), its least period and class count, and look for
    n > 2^{omega(q_least)}.

(b) Hit classification.  A Lehman hit of cell (a, b), ab <= r, is  0 <= aq + bp - 2 sqrt(abN) < W_ab := sqrt(N)/(4 r sqrt(ab)).
    It implies |p/q - a/b| < (1 + eta)/b^2 with eta = O(1/r); Fatou's theorem classifies fractions with |xi - a/b| < 1/b^2
    as convergents or intermediate fractions.  We search for hits in the sliver 1/b^2 <= |xi - a/b| that are neither a
    convergent nor a mediant (h_{n-1} + k h_n)/(k_{n-1} + k k_n), 0 <= k <= a_{n+1}, of xi = p/q.

All numbers are finite computations at the stated ranges.

Run:  python -m factorlab.experiments.arms_e45 --classes --q-max 60 --alpha-max 400 --gamma-max 400
      python -m factorlab.experiments.arms_e45 --hits --bits 36 --count 20000
"""
from __future__ import annotations

import argparse
import json
import math
import os
from fractions import Fraction
from math import gcd, isqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
from gmpy2 import invert, iroot, is_prime, mpz
from sympy import factorint

from factorlab.gen import make_semiprime


def r_of(N: int) -> int:
    """r = floor(N^(1/3)), exact."""
    return int(iroot(mpz(N), 3)[0])


# ----------------------------------------------------------------------------- (a) class counts

def omega(n: int) -> int:
    return len(factorint(n)) if n > 1 else 0


def admissible_mask(alpha: int, gamma: int, q: int) -> np.ndarray:
    """Boolean mask over d in [0, 4 q^2): both q^2 (A d^2 -+ d + C) = alpha d^2 -+ q^2 d + gamma are = 0 mod 2 q^2."""
    m = 4 * q * q
    d = np.arange(m, dtype=np.int64)
    base = (alpha % (2 * q * q)) * (d * d % (2 * q * q)) % (2 * q * q)
    shift = (q * q % (2 * q * q)) * d % (2 * q * q)
    g = gamma % (2 * q * q)
    minus = (base - shift + g) % (2 * q * q)
    plus = (base + shift + g) % (2 * q * q)
    return (minus == 0) & (plus == 0)


def least_period(mask: np.ndarray) -> Optional[int]:
    """Least positive period of a periodic boolean sequence given over one full period of length len(mask)."""
    m = len(mask)
    if not mask.any():
        return None
    for t in range(1, m + 1):
        if m % t == 0 and np.array_equal(mask, np.roll(mask, t)):
            return t
    return m


def class_count(alpha: int, gamma: int, q: int) -> Optional[Tuple[int, int]]:
    """(least period, number of admissible classes modulo it) for the family alpha/q^2 d^2 + gamma/q^2, or None if empty."""
    mask = admissible_mask(alpha, gamma, q)
    t = least_period(mask)
    if t is None:
        return None
    return t, int(mask[:t].sum())


def class_count_search(q_max: int, alpha_max: int, gamma_max: int) -> Dict:
    """Enumerate families (alpha, gamma) with A = alpha/q^2 > 0 for q <= q_max; report max n / 2^omega(q_least)."""
    seen = set()
    worst = 0.0
    worst_case = None
    even_q_families = 0
    violations: List[Dict] = []
    for q in range(1, q_max + 1):
        for alpha in range(1, alpha_max + 1):
            for gamma in range(-gamma_max, gamma_max + 1):
                A = Fraction(alpha, q * q)
                C = Fraction(gamma, q * q)
                if (A, C) in seen:
                    continue
                seen.add((A, C))
                res = class_count(alpha, gamma, q)
                if res is None:
                    continue
                t, n = res
                if t % 2 == 0:
                    even_q_families += 1
                ratio = n / 2 ** omega(t)
                if ratio > worst:
                    worst, worst_case = ratio, {"A": str(A), "C": str(C), "q_least": t, "n": n}
                if n > 2 ** omega(t):
                    violations.append({"A": str(A), "C": str(C), "q_least": t, "n": n, "omega": omega(t)})
    return {
        "box": {"q_max": q_max, "alpha_max": alpha_max, "gamma_max": gamma_max},
        "distinct_families": len(seen),
        "even_least_period_families": even_q_families,
        "max_ratio_n_over_2pow_omega": worst,
        "argmax": worst_case,
        "violations_n_gt_2pow_omega": violations[:50],
        "n_violations": len(violations),
    }


# ----------------------------------------------------------------------------- (b) hit classification

def continued_fraction(p: int, q: int) -> List[int]:
    cf = []
    while q:
        a, r = divmod(p, q)
        cf.append(a)
        p, q = q, r
    return cf


def convergents_and_mediants(cf: List[int], b_max: int) -> Tuple[set, set]:
    """Sets of reduced fractions (num, den) with den <= b_max: convergents, and all mediants (h_{n-1} + k h_n)/(k_{n-1} + k k_n)
    for 1 <= k < a_{n+1} (intermediate fractions)."""
    h_prev, k_prev, h, k = 1, 0, cf[0], 1
    conv = {(h, k)}
    med = set()
    for i in range(1, len(cf)):
        a = cf[i]
        for j in range(1, a):
            num, den = h_prev + j * h, k_prev + j * k
            if den <= b_max:
                med.add((num, den))
        h_prev, k_prev, h, k = h, k, a * h + h_prev, a * k + k_prev
        if k <= b_max:
            conv.add((h, k))
    return conv, med


def is_hit(a: int, b: int, p: int, q: int, r: int) -> bool:
    """Exact test of 0 <= aq + bp - 2 sqrt(abN) < sqrt(N)/(4 r sqrt(ab)) via squared integer inequalities."""
    N = p * q
    u = a * q + b * p                      # integer
    ab = a * b
    # 2 sqrt(abN) <= u  always (AM-GM); the hit condition is  u - 2 sqrt(abN) < W  <=>  u - W < 2 sqrt(abN).
    # With W = sqrt(N)/(4 r sqrt(ab)):  u - W < 2 sqrt(abN)  <=>  4 r sqrt(ab) u - sqrt(N) < 8 r ab sqrt(N)
    #                                   <=>  4 r sqrt(ab) u < sqrt(N) (8 r ab + 1)  <=>  16 r^2 ab u^2 < N (8 r ab + 1)^2.
    return 16 * r * r * ab * u * u < N * (8 * r * ab + 1) ** 2


def classify_hits(N: int, p: int, q: int, r: int) -> Dict:
    """All hits (a, b) with ab <= r via the necessary condition ||b xi|| < 2/b, classified against the CF of p/q."""
    if p > q:
        p, q = q, p
    xi_num, xi_den = p, q
    cf = continued_fraction(p, q)
    b_max = isqrt(r * q // p) + 2
    conv, med = convergents_and_mediants(cf, b_max)
    out = {"convergent": 0, "intermediate": 0, "neither_sliver": [], "neither_nonsliver": [], "hits": 0, "sliver": 0}
    for b in range(1, b_max + 1):
        a0 = (b * p) // q
        for a in (a0, a0 + 1):
            if a < 1 or a * b > r:
                continue
            if not is_hit(a, b, p, q, r):
                continue
            g = gcd(a, b)
            ar, br = a // g, b // g
            out["hits"] += 1
            # Fatou sliver: |xi - a/b| >= 1/b^2  <=>  |a q - b p| * b >= q  (reduced fraction)
            in_sliver = abs(ar * q - br * p) * br >= q
            if in_sliver:
                out["sliver"] += 1
            if (ar, br) in conv:
                out["convergent"] += 1
            elif (ar, br) in med:
                out["intermediate"] += 1
            else:
                rec = {"a": a, "b": b, "reduced": [ar, br], "|aq-bp|*b/q": abs(ar * q - br * p) * br / q}
                # outside the sliver Fatou's theorem forbids 'neither'; such a record would indicate a classification bug
                (out["neither_sliver"] if in_sliver else out["neither_nonsliver"]).append(rec)
    return out


def hit_search(bits: int, count: int, seed: int = 5) -> Dict:
    """Classify all hits at r = floor(N^(1/3)) over `count` rsa-family moduli of the given size."""
    tot = {"moduli": 0, "hits": 0, "convergent": 0, "intermediate": 0, "sliver": 0, "neither_sliver": [], "neither_nonsliver": []}
    for i in range(count):
        sp = make_semiprime(bits, "rsa", seed, i)
        N, p, q = sp.N, sp.p, sp.q
        r = r_of(N)
        res = classify_hits(N, p, q, r)
        tot["moduli"] += 1
        for k in ("hits", "convergent", "intermediate", "sliver"):
            tot[k] += res[k]
        for key in ("neither_sliver", "neither_nonsliver"):
            for item in res[key]:
                item.update({"N": N, "p": p, "q": q, "r": r})
                tot[key].append(item)
    tot["bits"] = bits
    tot["n_neither_sliver"] = len(tot["neither_sliver"])
    tot["n_neither_nonsliver"] = len(tot["neither_nonsliver"])
    tot["neither_sliver"] = tot["neither_sliver"][:50]
    tot["neither_nonsliver"] = tot["neither_nonsliver"][:50]
    return tot


def construct_sliver_hits(b_lo: int, b_hi: int, rho2_lo: float = 1.0, rho2_hi: float = 2.0, max_found: int = 20) -> Dict:
    """Search for Lehman hits with |xi - a/b| >= 1/b^2 (outside Fatou's hypothesis).

    For fixed b and prime q, a hit with |aq - bp| = m and aq - bp = s m (s = +-1) needs p = -s m b^{-1} mod q; the sliver
    needs m b >= q and the hit needs m^2 < N/r + W^2, so only m in {ceil(q/b), ceil(q/b)+1} can work.  The sliver lives at
    the top of the shell, b ~ sqrt(r q/p) with r ~ N^{1/3}, which forces q ~ b^3 (p/q); with rho^2 = q/p in [rho2_lo, rho2_hi]
    we scan the primes q in [b^3/rho2_hi, b^3/rho2_lo], solve for p, and keep the prime p in (q/2, q) whose cell (a, b) is an
    exact hit in the sliver.  Each find is classified against the CF of p/q.  At most max_found records are returned.
    """
    found: List[Dict] = []
    scanned = 0
    for b in range(b_lo, b_hi + 1):
        if len(found) >= max_found:
            break
        q_lo, q_hi = math.ceil(b ** 3 / rho2_hi), int(b ** 3 / rho2_lo)
        q = q_lo | 1
        while q <= q_hi and len(found) < max_found:
            if is_prime(q):
                scanned += 1
                binv = int(invert(mpz(b), mpz(q)))
                m0 = -(-q // b)  # ceil(q/b)
                for m in (m0, m0 + 1):
                    for sgn in (1, -1):
                        if len(found) >= max_found:
                            break
                        p = (-sgn * m * binv) % q
                        if p <= q // 2 or p >= q or not is_prime(p):
                            continue
                        a = (b * p + sgn * m) // q
                        if (b * p + sgn * m) % q or a < 1:
                            continue
                        N = p * q
                        r = r_of(N)
                        if a * b > r or not is_hit(a, b, p, q, r):
                            continue
                        g = gcd(a, b)
                        ar, br = a // g, b // g
                        if abs(ar * q - br * p) * br < q:
                            continue  # not in the sliver after reduction
                        cf = continued_fraction(p, q)
                        conv, med = convergents_and_mediants(cf, br + 1)
                        kind = "convergent" if (ar, br) in conv else ("intermediate" if (ar, br) in med else "neither")
                        found.append({"N": N, "p": p, "q": q, "r": r, "a": a, "b": b, "m": m, "reduced": [ar, br],
                                      "|aq-bp|*b/q": abs(ar * q - br * p) * br / q, "kind": kind})
            q += 2
    kinds = {k: sum(f["kind"] == k for f in found) for k in ("convergent", "intermediate", "neither")}
    return {"b_range": [b_lo, b_hi], "rho2_range": [rho2_lo, rho2_hi], "primes_q_scanned": scanned, "found": found, "kinds": kinds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", action="store_true")
    ap.add_argument("--hits", action="store_true")
    ap.add_argument("--construct", action="store_true")
    ap.add_argument("--b-lo", type=int, default=40)
    ap.add_argument("--b-hi", type=int, default=90)
    ap.add_argument("--q-max", type=int, default=60)
    ap.add_argument("--alpha-max", type=int, default=400)
    ap.add_argument("--gamma-max", type=int, default=400)
    ap.add_argument("--bits", type=int, default=36)
    ap.add_argument("--count", type=int, default=20000)
    ap.add_argument("--out", default="results/e45_arms.json")
    args = ap.parse_args()
    res: Dict = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            res = json.load(fh)
    if args.classes:
        res["class_count"] = class_count_search(args.q_max, args.alpha_max, args.gamma_max)
        c = res["class_count"]
        print(f"class count: {c['distinct_families']} families, {c['even_least_period_families']} with even least period, "
              f"max n/2^omega = {c['max_ratio_n_over_2pow_omega']:.3f} at {c['argmax']}, violations = {c['n_violations']}")
    if args.hits:
        res[f"hits_{args.bits}"] = hit_search(args.bits, args.count)
        h = res[f"hits_{args.bits}"]
        print(f"hits at {args.bits} bits: moduli={h['moduli']} hits={h['hits']} convergent={h['convergent']} "
              f"intermediate={h['intermediate']} sliver(|xi-a/b|>=1/b^2)={h['sliver']} "
              f"neither_in_sliver={h['n_neither_sliver']} neither_outside_sliver(bug if >0)={h['n_neither_nonsliver']}")
        for item in h["neither_sliver"][:10]:
            print("  neither (sliver):", item)
    if args.construct:
        res["construct"] = construct_sliver_hits(args.b_lo, args.b_hi)
        c = res["construct"]
        print(f"construction: scanned {c['primes_q_scanned']} primes q, sliver hits found = {len(c['found'])}, kinds = {c['kinds']}")
        for item in c["found"][:10]:
            print("  ", item)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
