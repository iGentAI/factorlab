"""Checks of Theorems W and W' and of the prime onset on the Theorem-W family (E56).

The start differences and the shell membership are exact: ceil(2 sqrt(kN)) = isqrt(4kN) + [4kN not a square], and the shell radius is
floor(N^{1/3}) // C with an exact integer cube root.  The resonance centres d_*, t_* and the window half-widths are irrational; they are
evaluated at 256-bit precision, so the integer window [ceil(centre - L), floor(centre + L)] is the intended one unless the true endpoint lies
within 2^{-200} of an integer.  The start of the cell (1, k) is ceil(2 sqrt(kN)) - N - k; the start difference of the pair (k_-, k_+) is
ceil(2 sqrt(k_+ N)) - ceil(2 sqrt(k_- N)) - (k_+ - k_-).

Theorem W: family k_-(d) = (d^2 - d + 2)/2, k_+(d) = (d^2 + d + 2)/2 at r = floor(N^{1/3}); resonance window |d - d_*| <= lambda sqrt(d_*),
d_* = (7 sqrt N/(2 sqrt 2))^{1/3}.  Theorem W': families k_-+(t) = n t^2 -+ m t + 1 at r = floor(N^{1/3})/C, window |t - t_*| <=
lambda sqrt(t_*/(3m)), t_*^3 = u delta/(8 n^{5/2}), delta = 4n - m^2, u = 2 sqrt N.  Prime onset: on the Theorem-W window, the number of d
with both k_-(d), k_+(d) prime that share the most frequent start difference among the prime pairs.

Moduli: factorlab.gen.make_semiprime(bits, 'rsa', seed, 0).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

from gmpy2 import get_context, iroot, is_prime, isqrt, mpfr, mpz
from gmpy2 import sqrt as gmpy2_sqrt

from factorlab.gen import make_semiprime

get_context().precision = 256


def _window(centre, half_width):
    """Integer window [ceil(centre - half_width), floor(centre + half_width)] from 256-bit mpfr values."""
    import gmpy2
    lo = int(gmpy2.ceil(centre - half_width))
    hi = int(gmpy2.floor(centre + half_width))
    return lo, hi


def icbrt(N: int) -> int:
    """floor(N^{1/3}) exactly."""
    r, _ = iroot(mpz(N), 3)
    return int(r)


def ceil_2sqrt(k: int, N: int) -> int:
    x = mpz(4) * k * N
    s = isqrt(x)
    return int(s) if s * s == x else int(s) + 1


def start_difference(km: int, kp: int, N: int) -> int:
    return ceil_2sqrt(kp, N) - ceil_2sqrt(km, N) - (kp - km)


def theorem_w_check(N: int, lam: float = 0.8) -> Dict:
    N = int(N)
    d_star = (mpfr(7) * gmpy2_sqrt(mpfr(N)) / (mpfr(2) * gmpy2_sqrt(mpfr(2)))) ** (mpfr(1) / 3)
    L = mpfr(lam) * gmpy2_sqrt(d_star)
    lo, hi = _window(d_star, L)
    counts: Counter = Counter()
    for d in range(lo, hi + 1):
        km, kp = (d * d - d + 2) // 2, (d * d + d + 2) // 2
        counts[start_difference(km, kp, N)] += 1
    members = hi - lo + 1
    top_value, top_count = counts.most_common(1)[0]
    return {"N": str(N), "bits": N.bit_length(), "lambda": lam, "d_star": float(d_star), "d_lo": lo, "d_hi": hi, "members": members,
            "distinct_values": len(counts), "most_frequent_count": top_count, "most_frequent_share": top_count / members,
            "most_frequent_value": str(top_value), "guaranteed_share": 1.0 / 3.0}


def theorem_wprime_check(N: int, C: int, n: int, m: int, lam: float = 0.8) -> Dict:
    N = int(N)
    u = mpfr(2) * gmpy2_sqrt(mpfr(N))
    delta = 4 * n - m * m
    t_star = (u * delta / (mpfr(8) * mpfr(n) ** mpfr(2.5))) ** (mpfr(1) / 3)
    L = mpfr(lam) * gmpy2_sqrt(t_star / (3 * m))
    lo, hi = _window(t_star, L)
    r = icbrt(N) // C
    counts: Counter = Counter()
    in_shell = 0
    for t in range(lo, hi + 1):
        km, kp = n * t * t - m * t + 1, n * t * t + m * t + 1
        if r // 2 < km and kp <= r:
            in_shell += 1
        counts[start_difference(km, kp, N)] += 1
    members = hi - lo + 1
    top_value, top_count = counts.most_common(1)[0]
    z_over_r = float(n * t_star * t_star / r)
    return {"N": str(N), "bits": N.bit_length(), "C": C, "n": n, "m": m, "rho": delta / n, "t_star": float(t_star), "r": r,
            "z_over_r": z_over_r, "members": members, "members_in_shell": in_shell, "distinct_values": len(counts),
            "most_frequent_count": top_count, "most_frequent_share": top_count / members,
            "most_frequent_over_N112": top_count / float(mpfr(N) ** (mpfr(1) / 12))}


def prime_onset(N: int, lam: float = 0.8) -> Dict:
    N = int(N)
    d_star = (mpfr(7) * gmpy2_sqrt(mpfr(N)) / (mpfr(2) * gmpy2_sqrt(mpfr(2)))) ** (mpfr(1) / 3)
    L = mpfr(lam) * gmpy2_sqrt(d_star)
    lo, hi = _window(d_star, L)
    counts: Counter = Counter()
    prime_pairs = 0
    for d in range(lo, hi + 1):
        km, kp = (d * d - d + 2) // 2, (d * d + d + 2) // 2
        if is_prime(km) and is_prime(kp):
            prime_pairs += 1
            counts[start_difference(km, kp, N)] += 1
    top = counts.most_common(1)[0] if counts else (None, 0)
    return {"N": str(N), "bits": N.bit_length(), "lambda": lam, "window_members": hi - lo + 1, "prime_pairs": prime_pairs,
            "prime_pairs_sharing_most_frequent": top[1], "most_frequent_value": str(top[0])}


def experiment(w_bits: List[int], wp_bits: List[int], onset_bits: List[int], seed: int = 7, lam: float = 0.8) -> Dict:
    out = {"seed": seed, "lambda": lam, "theorem_W": [], "theorem_Wprime": [], "prime_onset": []}
    for b in w_bits:
        N = int(make_semiprime(b, "rsa", seed, 0).N)
        out["theorem_W"].append(theorem_w_check(N, lam))
    families = {1: (1, 1), 2: (3, 3), 3: (7, 5), 4: (13, 7)}
    for b in wp_bits:
        N = int(make_semiprime(b, "rsa", seed, 0).N)
        for C, (n, m) in families.items():
            out["theorem_Wprime"].append(theorem_wprime_check(N, C, n, m, lam))
    for b in onset_bits:
        N = int(make_semiprime(b, "rsa", seed, 0).N)
        out["prime_onset"].append(prime_onset(N, lam))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-bits", nargs="*", type=int, default=[48, 64, 96, 128, 160, 200])
    ap.add_argument("--wp-bits", nargs="*", type=int, default=[64, 96, 128])
    ap.add_argument("--onset-bits", nargs="*", type=int, default=[96, 128, 160, 200, 256])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--out", default="results/e56_theorem_checks.json")
    args = ap.parse_args()
    res = experiment(args.w_bits, args.wp_bits, args.onset_bits, args.seed, args.lam)
    json.dump(res, open(args.out, "w"), indent=1)
    for x in res["theorem_W"]:
        print(f"W  {x['bits']} bits: {x['members']} members, {x['distinct_values']} values, top {x['most_frequent_count']} ({x['most_frequent_share']:.2f})")
    for x in res["theorem_Wprime"]:
        print(f"W' {x['bits']} bits C={x['C']} (n,m)=({x['n']},{x['m']}): rho={x['rho']:.3f} z/r={x['z_over_r']:.2f} members {x['members']} (in shell {x['members_in_shell']}), values {x['distinct_values']}, top {x['most_frequent_count']} = {x['most_frequent_over_N112']:.2f} N^(1/12)")
    for x in res["prime_onset"]:
        print(f"onset {x['bits']} bits: {x['prime_pairs']} prime pairs on {x['window_members']} members, {x['prime_pairs_sharing_most_frequent']} share the most frequent difference")
    print("E56_DONE")
