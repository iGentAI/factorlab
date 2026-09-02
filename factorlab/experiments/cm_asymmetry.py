"""E44: complex-multiplication asymmetry -- does the inert/split asymmetry help?

For a fundamental discriminant D with chi_D(N) = -1 exactly one of the primes p, q of N = pq is
inert in Q(sqrt D), so a curve with CM by the maximal order is supersingular modulo one prime
(order p + 1) and ordinary modulo the other.  This module measures, for the CM curve

    E : y^2 = x^3 - x        (CM by Z[i], D = -4; inert primes p = 3 mod 4, split primes p = 1 mod 4)

the frequency with which the group order #E(F_p) is p^theta-smooth at inert and at split primes,
against a control of random orders in the Hasse interval, and checks the structural facts behind the
numbers: every order at a split prime is a norm from Z[i] (a sum of two squares); the curve's full
rational 2-torsion forces 4 | #E at every prime, and at split primes (0, 0) is halvable because -1 is a
square, so E(F_p) contains Z/2 x Z/4 and 8 | #E.

The point count is exact (Euler-criterion character sum, vectorised in numpy; primes < 2^31).
Standard errors are binomial.  All numbers are finite measurements at the stated prime range.

Run:  python -m factorlab.experiments.cm_asymmetry --lo 14 --hi 15 --out results/e44_cm_asymmetry.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np
from sympy import factorint, primerange


def order_x3_minus_x(p: int) -> int:
    """#E(F_p) for E: y^2 = x^3 - x, by the character sum 1 + sum_x (1 + chi(x^3 - x)).

    Exact for odd primes p < 2^31 (int64 products stay below 2^62).
    """
    if p < 3 or p >= 2 ** 31:
        raise ValueError("p must be an odd prime below 2^31")
    x = np.arange(p, dtype=np.int64)
    f = (x * x % p * x - x) % p
    r = np.ones(p, dtype=np.int64)
    b = f.copy()
    e = (p - 1) // 2
    while e:
        if e & 1:
            r = r * b % p
        b = b * b % p
        e >>= 1
    chi = np.where(f == 0, 0, np.where(r == 1, 1, -1))
    return p + 1 + int(chi.sum())


def order_brute(p: int) -> int:
    """Brute-force point count of y^2 = x^3 - x over F_p (for tests)."""
    squares: Dict[int, int] = {}
    for y in range(p):
        squares[y * y % p] = squares.get(y * y % p, 0) + 1
    return 1 + sum(squares.get((x * x * x - x) % p, 0) for x in range(p))


def largest_prime_factor(n: int) -> int:
    return max(int(q) for q in factorint(n)) if n > 1 else 1


def is_norm_from_gaussian_integers(n: int) -> bool:
    """n is a sum of two squares iff every prime = 3 mod 4 divides it to an even power."""
    return all(e % 2 == 0 for q, e in factorint(n).items() if int(q) % 4 == 3)


def is_theta_smooth(p: int, o: int, theta: Fraction) -> bool:
    """Exact test of P^+(o) <= p**theta for rational theta = a/b: compare P^+(o)**b <= p**a in integers."""
    theta = Fraction(theta)
    a, b = theta.numerator, theta.denominator
    return largest_prime_factor(o) ** b <= p ** a


def _summary(rows: List[Tuple[int, int]], theta: Fraction) -> Dict[str, float]:
    """Fraction of orders whose largest prime factor is <= p**theta (exact integer comparison)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "p_smooth": None, "se": None}
    hits = sum(is_theta_smooth(p, o, theta) for p, o in rows)
    ph = hits / n
    return {"n": n, "p_smooth": ph, "se": math.sqrt(ph * (1 - ph) / n)}


def _excess_in_se(a: Dict[str, float], b: Dict[str, float]) -> float | None:
    """(a - b) in units of the combined binomial standard error; None if undefined."""
    if not a["n"] or not b["n"]:
        return None
    se = math.hypot(a["se"], b["se"])
    return (a["p_smooth"] - b["p_smooth"]) / se if se > 0 else None


def derive_verdict(inert: Dict[str, float], split: Dict[str, float], control: Dict[str, float]) -> Dict:
    """Descriptive statistics of the run: excess of each CM population over the control, and over each other,
    in combined-SE units, plus a two-SE flag for each comparison.  No interpretation is hardcoded."""
    e_i = _excess_in_se(inert, control)
    e_s = _excess_in_se(split, control)
    e_si = _excess_in_se(split, inert)
    return {
        "inert_minus_control_se": e_i,
        "split_minus_control_se": e_s,
        "split_minus_inert_se": e_si,
        "inert_exceeds_control_2se": (e_i is not None and e_i > 2),
        "split_exceeds_control_2se": (e_s is not None and e_s > 2),
        "split_exceeds_inert_2se": (e_si is not None and e_si > 2),
    }


def cm_asymmetry_experiment(log2_lo: int, log2_hi: int, theta: Fraction = Fraction(1, 3), seed: int = 1) -> Dict:
    """Smoothness of #E(F_p) at inert and split primes in [2^lo, 2^hi) against a random-order control."""
    theta = Fraction(theta)
    rng = np.random.default_rng(seed)
    inert: List[Tuple[int, int]] = []
    split: List[Tuple[int, int]] = []
    for p in primerange(2 ** log2_lo, 2 ** log2_hi):
        if p % 4 == 3:
            inert.append((p, p + 1))
        else:
            split.append((p, order_x3_minus_x(p)))
    control = []
    for p, _ in split:
        s = math.isqrt(p)
        t = int(rng.integers(-2 * s, 2 * s + 1))
        control.append((p, p + 1 - t))
    inert_s, split_s, control_s = _summary(inert, theta), _summary(split, theta), _summary(control, theta)
    out = {
        "curve": "y^2 = x^3 - x (CM by Z[i], D = -4)",
        "range_log2": [log2_lo, log2_hi],
        "theta": str(theta),
        "inert_supersingular": inert_s,
        "split_ordinary": split_s,
        "control_random_hasse_order": control_s,
        "dickman_rho_1_over_theta": None,
        "split_orders_norms_from_Zi": sum(is_norm_from_gaussian_integers(o) for _, o in split),
        "inert_orders_norms_from_Zi": sum(is_norm_from_gaussian_integers(o) for _, o in inert),
        "inert_all_divisible_by_4": all(o % 4 == 0 for _, o in inert),
        "split_fraction_divisible_by_8": (sum(o % 8 == 0 for _, o in split) / len(split)) if split else None,
        "comparison": derive_verdict(inert_s, split_s, control_s),
    }
    try:
        from factorlab.experiments.smooth_profiles import dickman_rho

        out["dickman_rho_1_over_theta"] = float(dickman_rho(float(1 / theta)))
    except Exception:  # pragma: no cover - optional cross-check
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lo", type=int, default=14)
    ap.add_argument("--hi", type=int, default=15)
    ap.add_argument("--theta", type=Fraction, default=Fraction(1, 3), help="rational, e.g. 1/3")
    ap.add_argument("--out", default="results/e44_cm_asymmetry.json")
    args = ap.parse_args()
    res = cm_asymmetry_experiment(args.lo, args.hi, args.theta)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    for key in ("inert_supersingular", "split_ordinary", "control_random_hasse_order"):
        s = res[key]
        print(f"{key}: n={s['n']} P[p^theta-smooth]={s['p_smooth']:.4f} (SE {s['se']:.4f})")
    print("split orders that are norms:", res["split_orders_norms_from_Zi"], "/", res["split_ordinary"]["n"])
    print("dickman rho(1/theta):", res["dickman_rho_1_over_theta"])
    print("comparison (SE units):", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["comparison"].items()})


if __name__ == "__main__":
    main()
