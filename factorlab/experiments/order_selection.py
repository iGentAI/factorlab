"""E50: deterministic selection of an element of large order modulo N = pq, for every bound D.

The procedure is Hittmeir's Algorithm 6.2 (Math. Comp. 87 (2018), Section 6) with two changes in the analysis and one in
the recovery step.  For alpha = 2, 3, ... with L = 1 initially:
  (a) if gcd(alpha, N) > 1 return the factor;
  (a') if alpha^L = 1 (mod N), skip alpha (it lies in the subgroup already known);
  (b) compute by baby-step giant-step the least E <= D with alpha^E = 1 (mod N); if none, return alpha (ord_N(alpha) > D);
  (c) for each prime r | E test gcd(alpha^{E/r} - 1, N) (Hittmeir's Lemma 2.3); a proper gcd is a factor; otherwise
      ord_p(alpha) = ord_q(alpha) = E, so E | p - 1 and E | q - 1; set L = lcm(L, E) (at least doubling L);
  (d) if L >= N^{1/3}: p = 1 + kL, q = 1 + lL with k + l < L, so k + l = ((N - 1)/L) mod L and kl = ((N - 1)/L - (k + l))/L;
      solve the quadratic and return the factors.
Every candidate processed before termination lies in the subgroup H_L = {x : x^L = 1} of F_p^*, which has exactly L
elements; if all alpha <= y were processed, every y-smooth integer below p lies in H_L, so Psi(p - 1, y) <= L < N^{1/3},
impossible for y = (log N)^4 and N large.  Hence O((log N)^4) candidates, O(log N) of them non-skipped (L doubles), total
cost O~(sqrt D) + polylog -- with no lower restriction on D (Hittmeir's Theorem 6.3 needs D >= N^{2/5}).

Run:  python -m factorlab.experiments.order_selection --bits 36 44 --count 30 --out results/e50_order_selection.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from gmpy2 import gcd, iroot, is_prime, isqrt, mpz, powmod
from sympy import factorint

from factorlab.gen import make_semiprime


def order_le_D(alpha: int, N: int, D: int) -> Optional[int]:
    """Exact multiplicative order of alpha modulo N if it is <= D, else None (baby-step giant-step, O~(sqrt D))."""
    N = mpz(N)
    m = math.isqrt(D) + 1
    baby: Dict[int, int] = {}
    x = mpz(1)
    for i in range(m):
        baby.setdefault(int(x), i)
        x = x * alpha % N
    step = powmod(alpha, m, N)
    y = mpz(1)
    best = None
    for j in range(m + 1):
        i = baby.get(int(y))
        if i is not None:
            e = j * m - i
            if e > 0 and (best is None or e < best):
                best = e
        y = y * step % N
    if best is None or best > D:
        return None
    e = best  # a positive multiple of the order; reduce to the exact order
    for pr, k in factorint(e).items():
        for _ in range(k):
            if powmod(alpha, e // int(pr), N) == 1:
                e //= int(pr)
            else:
                break
    return e if e <= D else None


def recover_from_common_divisor(N: int, L: int) -> Optional[Tuple[int, int]]:
    """If N = pq with p = 1 + kL, q = 1 + lL and k + l < L, return (p, q); else None."""
    if (N - 1) % L:
        return None
    t = (N - 1) // L            # = (k + l) + k l L
    s = t % L                   # = k + l  when k + l < L
    if (t - s) % L:
        return None
    prod = (t - s) // L         # = k l
    disc = s * s - 4 * prod
    if disc < 0:
        return None
    rt = int(isqrt(mpz(disc)))
    if rt * rt != disc or (s + rt) % 2:
        return None
    k, l = (s - rt) // 2, (s + rt) // 2
    p, q = 1 + k * L, 1 + l * L
    if 1 < p < N and p * q == N:
        return (min(p, q), max(p, q))
    return None


def select_order_element(N: int, D: int, max_candidates: int = 10 ** 6, threshold: Optional[int] = None, initial_L: int = 1) -> Dict:
    """Run the modified Algorithm 6.2; return the outcome and the trace of candidates.  `initial_L` seeds the accumulated lcm (a
    known common divisor of p - 1 and q - 1), which is how the skip step is exercised in tests."""
    N = int(N)
    T0 = threshold if threshold is not None else int(iroot(mpz(N), 3)[0]) + 1
    L = initial_L
    trace: List[Tuple[int, str, object]] = []
    skipped = 0
    for alpha in range(2, max_candidates + 2):
        g = int(gcd(alpha, N))
        if g > 1:
            trace.append((alpha, "gcd", g))
            return {"outcome": "factor", "factor": g, "alpha": alpha, "candidates": alpha - 1, "skipped": skipped, "L": L, "trace": trace}
        if L > 1 and powmod(alpha, L, N) == 1:
            skipped += 1
            trace.append((alpha, "skip", L))
            continue
        E = order_le_D(alpha, N, D)
        if E is None:
            trace.append((alpha, "ord>D", None))
            return {"outcome": "large order", "alpha": alpha, "candidates": alpha - 1, "skipped": skipped, "L": L, "trace": trace}
        for r in factorint(E):
            gg = int(gcd(powmod(alpha, E // int(r), N) - 1, N))
            if 1 < gg < N:
                trace.append((alpha, "divisor gcd", (E, int(r), gg)))
                return {"outcome": "factor", "factor": gg, "alpha": alpha, "candidates": alpha - 1, "skipped": skipped, "L": L, "trace": trace}
        L = L * E // math.gcd(L, E)  # ord_p(alpha) = ord_q(alpha) = E, so E | p - 1, q - 1
        trace.append((alpha, "fail", E))
        if L >= T0:
            rec = recover_from_common_divisor(N, L)
            if rec is not None:
                trace.append((alpha, "recover", L))
                return {"outcome": "factor", "factor": rec[0], "alpha": alpha, "candidates": alpha - 1, "skipped": skipped, "L": L, "trace": trace}
    return {"outcome": "exhausted", "candidates": max_candidates, "skipped": skipped, "L": L, "trace": trace}


def adversarial_moduli() -> List[Tuple[int, int, int]]:
    """(d, p, q): two prime factors of 2^d - 1 with ord_p(2) = ord_q(2) = d, so that alpha = 2 fails for D >= d."""
    out = []
    for d in range(5, 64):
        f = factorint(2 ** d - 1)
        ps = [int(pp) for pp in f if pow(2, d, int(pp)) == 1 and all(pow(2, d // r, int(pp)) != 1 for r in factorint(d))]
        if len(ps) >= 2:
            ps.sort()
            out.append((d, ps[0], ps[1]))
    return out


def common_divisor_moduli(count: int, L_lo: int = 10 ** 4, L_hi: int = 10 ** 7) -> List[Tuple[int, int, int, int, int]]:
    """(L, p, q, k, l) with p = 1 + 4L, q = 1 + 12L prime, L prime, L = 1 (mod 3), and 2 a cubic residue mod q.  Then p, q = 5
    (mod 8), so 2 is a non-residue modulo both and ord_p(2) = ord_q(2) = 4L: alpha = 2 fails, and its order 4L >= N^{1/3} triggers
    the recovery step at once.  Deterministic (the first `count` such L above L_lo)."""
    from sympy import primerange

    out: List[Tuple[int, int, int, int, int]] = []
    for L in primerange(L_lo, L_hi):
        if L % 3 != 1:
            continue
        p, q = 1 + 4 * L, 1 + 12 * L
        if is_prime(p) and is_prime(q) and pow(2, 4 * L, q) == 1:
            out.append((int(L), p, q, 1, 3))  # p = 1 + 1*(4L), q = 1 + 3*(4L) in terms of the recovered L' = 4L
            if len(out) >= count:
                break
    if len(out) < count:
        raise RuntimeError(f"only {len(out)} recovery-path moduli found below L = {L_hi}")
    return out


def experiment(bits_list, count: int, exponent_den: int = 6, seed: int = 3) -> Dict:
    res: Dict = {"adversarial": [], "common_divisor": [], "random": []}
    for d, p, q in adversarial_moduli():
        N = p * q
        r = select_order_element(N, max(d, 1000))
        res["adversarial"].append({"d": d, "p": p, "q": q, "N": N, "D": max(d, 1000), "outcome": r["outcome"], "alpha": r.get("alpha"),
                                   "candidates": r["candidates"], "L": r["L"], "trace": [(a, k, str(v)) for a, k, v in r["trace"]]})
    for L, p, q, k, l in common_divisor_moduli(40):
        N = p * q
        r = select_order_element(N, N)  # D = N: every order is found, so alpha = 2 must fail with order 4L and trigger the recovery
        recovered = any(t[1] == "recover" for t in r["trace"])
        if not (r["outcome"] == "factor" and r.get("factor") in (p, q) and recovered and r["trace"][0][1] == "fail"):
            raise AssertionError(f"recovery path not exercised as constructed for N = {N}: {r['trace']}")
        res["common_divisor"].append({"L": L, "k": k, "l": l, "p": p, "q": q, "N": N, "outcome": r["outcome"], "alpha": r.get("alpha"),
                                      "factor_ok": True, "recovered": True, "candidates": r["candidates"],
                                      "trace": [(a, kk, str(v)) for a, kk, v in r["trace"]]})
    for bits in bits_list:
        first = later = factored = 0
        for i in range(count):
            sp = make_semiprime(bits, "rsa", seed, i)
            N = int(sp.N)
            D = int(iroot(mpz(N), exponent_den)[0])
            r = select_order_element(N, D)
            if r["outcome"] == "factor":
                factored += 1
            elif r["alpha"] == 2:
                first += 1
            else:
                later += 1
        res["random"].append({"bits": bits, "count": count, "D": f"N^(1/{exponent_den})", "alpha2_accepted": first, "later_candidate": later, "factored": factored})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, nargs="+", default=[36, 44])
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out", default="results/e50_order_selection.json")
    args = ap.parse_args()
    res = experiment(args.bits, args.count)
    for a in res["adversarial"]:
        print(f"2^{a['d']}-1: N={a['p']}*{a['q']} D={a['D']}: {a['outcome']} after {a['candidates']} candidate(s): {a['trace']}", flush=True)
    cd = res["common_divisor"]
    print(f"recovery-path moduli (p = 1 + 4L, q = 1 + 12L, alpha = 2 of order 4L modulo both): {len(cd)} moduli, all recovered by the"
          f" k+l identity after the first candidate failed; L from {cd[0]['L']} to {cd[-1]['L']}", flush=True)
    for rr in res["random"]:
        print(f"{rr['bits']} bits, D={rr['D']}: alpha=2 accepted {rr['alpha2_accepted']}/{rr['count']}, later {rr['later_candidate']}, factored {rr['factored']}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
