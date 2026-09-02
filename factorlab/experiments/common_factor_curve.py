"""The common-factor class from N alone: the Fermat-progression search and the elementary trial.

Let N = pq be balanced and let g be a common divisor of p - 1 and q - 1 (any one; the gcd is the best).  Writing p = 1 + g x, q = 1 + g y,
(N - 1)/g = x + y + g x y, so x + y == (N - 1)/g (mod g) and

    p + q = 2 + g (x + y) = r + g^2 t,      r := 2 + g * (((N - 1)/g) mod g),  t >= 0.

For balanced p, q the sum S = p + q lies in [2 sqrt N, (sqrt C + 1/sqrt C) sqrt N], so t ranges over T = O(sqrt N / g^2) candidates.  Harvey's
Fermat cell gives the exact congruence alpha^{p+q} == alpha^{N+1} (mod N) for every alpha coprime to N, so the true t satisfies
beta^t == alpha^{N+1-r} (mod N) with beta := alpha^{g^2}, and a babystep-giantstep over t finds it in O(sqrt T) group operations (exact matches
modulo N; each match is verified by the square test on S^2 - 4N).  The babies must be pairwise distinct modulo N, which holds when
ord_N(alpha) > g^2 sqrt T; the search reports a baby collision instead of guessing, and the caller may change the base.

Route C is the elementary alternative: try t = 0, 1, ..., T directly (x y = ((N-1)/g - x - y)/g must be an integer and the quadratic must
have integer roots), O(T) operations.  Route A is the report's large-common-factor lemma (a collision search on alpha^{N-1}, O(N^{1/4}/sqrt g)),
implemented in harvey_residue.common_factor_attack.

Work is counted in machine-independent units: babies and giants for Route B, trials for Route C; exponentiations, the inversion, the
hash-table look-ups and the square tests are not counted.  This module does not implement the Pollard-Strassen recovery of g from N - 1,
the dovetailing over budgets, or the Harvey-Hittmeir order selection: the experiment supplies g (the gcd, and the constructed proper
divisor when it differs) and tries the small bases 2, 3, 5, ... until one has distinct babies, to measure the search itself; the theory
(docs) accounts for finding g and for the order selection.  The match structure is a hash table, an experimental stand-in for the sorted
lists of the bit-complexity model.

CLI:  python -m factorlab.experiments.common_factor_curve --bits 40 48 --gammas 0.083 0.125 0.167 0.2 --count 6 --out results/e57_common_factor_curve.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from fractions import Fraction
from typing import Dict, List

from gmpy2 import gcd, invert, is_prime, is_square, isqrt, mpz, powmod

from factorlab.numth import isqrt_ceil


def moduli_with_common_factor(bits: int, gamma: float, count: int, seed: int = 13, C: float = 2.0) -> List[Dict]:
    """Balanced semiprimes N = pq of about `bits` bits with p = 1 + g x, q = 1 + g y for an even g of about gamma * bits bits.  The
    recorded g is the exact gcd(p - 1, q - 1), which may exceed the constructed g when x and y share a factor."""
    rng = random.Random(seed)
    gbits = max(2, round(gamma * bits))
    xbits = max(2, bits // 2 - gbits)
    out: List[Dict] = []
    while len(out) < count:
        g = 2 * rng.getrandbits(gbits - 1) | 2
        x = rng.getrandbits(xbits) | (1 << (xbits - 1))
        y = rng.getrandbits(xbits) | (1 << (xbits - 1))
        p, q = 1 + g * x, 1 + g * y
        if p == q or not (is_prime(p) and is_prime(q)):
            continue
        p, q = min(p, q), max(p, q)
        if q >= C * p:
            continue
        N = p * q
        out.append({"N": int(N), "p": int(p), "q": int(q), "g_constructed": int(g), "g": int(gcd(p - 1, q - 1)),
                    "gamma": math.log(int(gcd(p - 1, q - 1))) / math.log(N), "bits": N.bit_length()})
    return out


def progression(N: int, g: int, C: float = 2.0):
    """(r, t_lo, t_hi): p + q = r + g^2 t with t in [t_lo, t_hi] for balanced p, q with q/p < C.  The upper bound sqrt(N) (sqrt C + 1/sqrt C)
    = sqrt(N (a+b)^2/(ab)) is computed in exact integer arithmetic with C = a/b the exact rational value of the given float, so it never
    undershoots; the range may be empty (t_hi < t_lo)."""
    N, g = mpz(N), mpz(g)
    if (N - 1) % g:
        raise ValueError("g must divide N - 1")
    c = ((N - 1) // g) % g
    r = 2 + g * c
    g2 = g * g
    S_lo = isqrt_ceil(4 * N)                                        # ceil(2 sqrt N)
    Cf = Fraction(C)                                                # the exact rational value of the float C: never below C
    a, b = Cf.numerator, Cf.denominator
    S_hi = isqrt((N * (a + b) ** 2) // (a * b)) + 1                 # >= sqrt(N) (sqrt C + 1/sqrt C)
    t_lo = max(0, -((r - S_lo) // g2))                                # ceil((S_lo - r)/g^2), at least 0
    t_hi = (S_hi - r) // g2
    return int(r), int(t_lo), int(t_hi)


def fermat_progression_search(N: int, g: int, C: float = 2.0, alpha: int = 2) -> Dict:
    """Route B with the base alpha.  Returns the factor (or None), the work (babies + giants generated), the candidate count T, the number
    of exact matches examined, and whether the babies collided modulo N (in which case no search was run).  A base sharing a proper
    factor with N returns that factor; a base that is 0 modulo N is reported unusable."""
    N = mpz(N)
    d0 = gcd(alpha, N)
    if 1 < d0 < N:
        return {"factor": int(d0), "work": 0, "T": 0, "matches": 0, "baby_collision": False, "alpha": alpha, "usable": True}
    if d0 == N:
        return {"factor": None, "work": 0, "T": 0, "matches": 0, "baby_collision": True, "alpha": alpha, "usable": False}
    r, t_lo, t_hi = progression(N, g, C)
    T = max(0, t_hi - t_lo + 1)
    if T == 0:
        return {"factor": None, "work": 0, "T": 0, "matches": 0, "baby_collision": False, "alpha": alpha, "usable": True}
    m = int(isqrt_ceil(T))                                             # ceil(sqrt T)
    g2 = mpz(g) * mpz(g)
    beta = powmod(alpha, g2, N)
    babies: Dict[int, int] = {}
    b = mpz(1)
    for i in range(m):
        key = int(b)
        if key in babies:
            return {"factor": None, "work": i + 1, "T": T, "matches": 0, "baby_collision": True, "alpha": alpha, "usable": True,
                    "collision_order": i - babies[key]}
        babies[key] = i
        b = (b * beta) % N
    S0 = mpz(r) + g2 * t_lo
    giant = powmod(alpha, N + 1 - S0, N)                # alpha^{N+1-S0}: the target for u = t - t_lo = 0
    step = invert(powmod(beta, m, N), N)                # beta^{-m}
    K = (T + m - 1) // m                                # ceil(T/m) giant blocks
    matches = 0
    for k in range(K):
        i = babies.get(int(giant))
        if i is not None:
            u = i + m * k
            if u < T:                                   # indices beyond the range belong to the padded last block
                matches += 1
                S = S0 + g2 * u
                disc = S * S - 4 * N
                if disc >= 0 and is_square(disc):
                    p = (S - isqrt(disc)) // 2
                    if p > 1 and N % p == 0:
                        return {"factor": int(p), "work": m + k + 1, "T": T, "matches": matches, "baby_collision": False, "alpha": alpha,
                                "usable": True, "t": int(u + t_lo), "m": m, "K": K}
        giant = (giant * step) % N
    return {"factor": None, "work": m + K, "T": T, "matches": matches, "baby_collision": False, "alpha": alpha, "usable": True, "m": m, "K": K}


def route_b(N: int, g: int, C: float = 2.0, bases=(2, 3, 5, 7, 11, 13)) -> Dict:
    """Route B over a list of bases, moving to the next base when the babies collide modulo N or the base is unusable; the returned work
    is cumulative over all bases tried."""
    total = 0
    res: Dict = {}
    for a in bases:
        res = fermat_progression_search(N, g, C, a)
        total += res["work"]
        if res["factor"] is not None or not res["baby_collision"]:
            break
    res["work"] = total
    return res


def route_c_trial(N: int, g: int, C: float = 2.0) -> Dict:
    """The elementary trial over x + y = c + g t: xy = ((N-1)/g - (x+y))/g integral and z^2 - (x+y) z + xy with integer roots."""
    N, g = mpz(N), mpz(g)
    Mp = (N - 1) // g
    c = Mp % g
    _, t_lo, t_hi = progression(int(N), int(g), C)
    T = max(0, t_hi - t_lo + 1)
    for t in range(t_lo, t_hi + 1):
        s = c + g * t
        if (Mp - s) % g:
            continue
        xy = (Mp - s) // g
        if xy <= 0:
            continue
        disc = s * s - 4 * xy
        if disc >= 0 and is_square(disc):
            u = isqrt(disc)
            if (s - u) % 2 == 0:
                x = (s - u) // 2
                p = 1 + g * x
                if p > 1 and N % p == 0:
                    return {"factor": int(p), "work": t - t_lo + 1, "T": T}
    return {"factor": None, "work": T, "T": T}


def experiment(bits_list, gammas, count: int, seed: int = 13, C: float = 2.0) -> Dict:
    rows = []
    for bits in bits_list:
        for gamma in gammas:
            for row in moduli_with_common_factor(bits, gamma, count, seed, C):
                N, g = row["N"], row["g"]
                rb = route_b(N, g, C)
                rc = route_c_trial(N, g, C) if rb["T"] <= 2_000_000 else {"factor": None, "work": None, "T": rb["T"]}
                rb_con = route_b(N, row["g_constructed"], C) if row["g_constructed"] != g else None
                rows.append({**row, "route_b": rb, "route_c": rc, "route_b_constructed_divisor": rb_con, "sqrt_T": math.sqrt(max(rb["T"], 1)),
                             "pred_N14_over_g": float(N) ** 0.25 / g, "factored_b": rb["factor"] in (row["p"], row["q"]),
                             "factored_c": rc["factor"] in (row["p"], row["q"]),
                             "factored_b_constructed": (rb_con["factor"] in (row["p"], row["q"])) if rb_con else None})
    return {"bits": list(bits_list), "gammas": list(gammas), "count": count, "seed": seed, "C": C, "rows": rows,
            "note": "route_b work = babies + giants generated until the match, cumulative over the bases tried; route_c work = trials; T = number of "
                    "candidates in the progression; route_b_constructed_divisor reruns Route B with the constructed proper divisor when it is not the gcd"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", nargs="+", type=int, default=[40, 48])
    ap.add_argument("--gammas", nargs="+", type=float, default=[1 / 12, 1 / 8, 1 / 6, 1 / 5])
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="results/e57_common_factor_curve.json")
    a = ap.parse_args()
    res = experiment(a.bits, a.gammas, a.count, a.seed)
    json.dump(res, open(a.out, "w"), indent=1)
    ok_b = sum(r["factored_b"] for r in res["rows"])
    ok_c = sum(r["factored_c"] for r in res["rows"])
    con = [r for r in res["rows"] if r["route_b_constructed_divisor"]]
    print(f"{len(res['rows'])} moduli: route B factored {ok_b}, route C factored {ok_c}; proper-divisor reruns {len(con)}, factored "
          f"{sum(1 for r in con if r['factored_b_constructed'])}")
    ratios = [r["route_b"]["work"] / r["sqrt_T"] for r in res["rows"] if r["route_b"]["T"] >= 100]
    if ratios:
        print(f"  work / sqrt(T) over the {len(ratios)} moduli with T >= 100: min {min(ratios):.2f}, max {max(ratios):.2f}")
    for r in res["rows"]:
        rb = r["route_b"]
        print(f"  {r['bits']}b gamma={r['gamma']:.3f} g={r['g']}: T={rb['T']} sqrtT={r['sqrt_T']:.1f} routeB work={rb['work']} (alpha={rb['alpha']}, "
              f"matches={rb['matches']}, collision={rb['baby_collision']}) routeC work={r['route_c']['work']} N^(1/4)/g={r['pred_N14_over_g']:.1f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
