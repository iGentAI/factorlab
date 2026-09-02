"""E40 -- the positive-density success theorem for the deterministic ``p - 1`` method.

Proposition PP (``docs/notes_probabilistic.md`` section 5.10).  Fix ``theta`` with
``1/(2 sqrt e) = 0.30327... < theta < 1/2`` and a sufficiently small balance parameter
``eps`` (``theta (1 - eps) > 1/(2 sqrt e)``).  The base-2 ``p - 1`` method with the
stage-1 exponent

    E_B(N) = prod_{l <= B} l^{floor(log N / log l)},      B = ceil(N^{theta/2}),

is deterministic, costs ``O(N^{theta/2})`` multiplications modulo ``N``, and
returns the factor ``p`` of ``N = pq`` whenever ``p - 1`` is ``B``-smooth and
``ord_q(2)`` is not.  Friedlander (1989) gives a positive proportion of primes
with ``P^+(p-1) <= x^theta`` for every ``theta > 1/(2 sqrt e)``; Fouvry (1985)
gives a positive proportion with ``P^+(q-1) >= q^{0.6687}``, and for those the
small-order exceptions ``ord_q(2) < x^{0.3313}`` number at most ``x^{0.6626}``.
Hence a positive relative proportion of the semiprimes ``N = pq`` with
``x^{1-eps} < p, q <= x`` are factored deterministically in ``O(N^{theta/2})``
multiplications -- threshold exponent ``1/(4 sqrt e) = 0.15163...``.  This is a
success theorem for a partial algorithm, not a worst-case or almost-all statement.

This module measures the finite-size proportion on the ``rsa`` family by
running the algorithm itself, checks that its outcome is exactly the one-sided
smoothness predicate of the proof (the order anomaly -- ``q - 1`` not
``B``-smooth but ``ord_q(2)`` ``B``-smooth -- is counted separately), and
quotes the Dickman heuristic ``rho(1/theta)`` for comparison.  The semismooth
column (a stage 2 with ``B_2 = B^2``) is a heuristic companion only: no
positive-proportion theorem for semismooth shifted primes is used.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Sequence

import gmpy2
from sympy import factorint, primerange

from factorlab.experiments.smooth_profiles import dickman_rho
from factorlab.gen import make_semiprime

THETA_FRIEDLANDER = 1.0 / (2.0 * math.sqrt(math.e))   # 0.30326..., the threshold: the theorem needs theta > this
THETA_DEFAULT = 0.31                                  # theorem-valid default (exponent theta/2 = 0.155)
ALPHA_FOUVRY = 0.6687


def stage1_bound(N, theta: float = THETA_DEFAULT) -> int:
    """``B = ceil(N^{theta/2})``."""
    return int(math.ceil(math.exp(0.5 * theta * math.log(int(N)))))


def stage1_exponent(N, B: int) -> int:
    """``E_B(N) = prod_{l <= B} l^{floor(log N / log l)}``.

    Every ``m - 1`` with ``m <= N`` and ``P^+(m - 1) <= B`` divides ``E``.
    """
    logN = math.log(int(N))
    E = 1
    for ell in primerange(2, B + 1):
        e = int(logN / math.log(ell))
        # guard the float: l^e <= N < l^(e+1)
        while ell ** (e + 1) <= N:
            e += 1
        while e > 0 and ell ** e > N:
            e -= 1
        E *= ell ** e
    return E


def pm1_deterministic(N, theta: float = THETA_DEFAULT):
    """Run the base-2 ``p - 1`` method at ``B = ceil(N^{theta/2})``.

    Returns ``(B, g)`` with ``g = gcd(2^E - 1, N)``; ``1 < g < N`` is success.
    """
    N = gmpy2.mpz(N)
    B = stage1_bound(N, theta)
    E = stage1_exponent(N, B)
    g = gmpy2.gcd(gmpy2.powmod(2, E, N) - 1, N)
    return B, int(g)


def largest_prime_factor(n: int) -> int:
    return max(int(x) for x in factorint(int(n)))


def second_largest_prime_factor(n: int) -> int:
    """Second-largest prime factor with multiplicity (1 if ``n`` is a prime power)."""
    fac = factorint(int(n))
    flat: List[int] = []
    for pr, mult in fac.items():
        flat.extend([int(pr)] * mult)
    flat.sort()
    return flat[-2] if len(flat) >= 2 else 1


def smooth_order_divides(E: int, q: int) -> bool:
    """``ord_q(2) | E``, i.e. ``q | 2^E - 1``."""
    return gmpy2.powmod(2, E, q) == 1


def average_case_point(bits: int, count: int, theta: float = THETA_DEFAULT,
                       seed: int = 11) -> Dict:
    """Run the algorithm on ``count`` rsa-family moduli of ``bits`` bits."""
    n_success = n_two_sided = n_fail = 0
    n_one_sided_smooth = n_both_smooth = n_order_anomaly = 0
    n_semismooth_one_sided = 0
    pred_one_sided = 0.0
    B_values: List[int] = []
    for idx in range(count):
        sp = make_semiprime(bits, "rsa", seed, idx)
        N, p, q = int(sp.N), int(sp.p), int(sp.q)
        B = stage1_bound(N, theta)
        B_values.append(B)
        E = stage1_exponent(N, B)
        g = int(gmpy2.gcd(gmpy2.powmod(2, E, N) - 1, N))
        if 1 < g < N:
            n_success += 1
        elif g == N:
            n_two_sided += 1
        else:
            n_fail += 1
        sp_ = largest_prime_factor(p - 1) <= B
        sq_ = largest_prime_factor(q - 1) <= B
        if sp_ != sq_:
            n_one_sided_smooth += 1
        if sp_ and sq_:
            n_both_smooth += 1
        # order anomaly: the rough side still has 2^E = 1 (ord_q(2) smooth, q-1 not)
        for r, sm in ((p, sp_), (q, sq_)):
            if not sm and smooth_order_divides(E, r):
                n_order_anomaly += 1
        # heuristic stage-2 companion: (B, B^2)-semismooth on exactly one side
        ss_p = second_largest_prime_factor(p - 1) <= B and largest_prime_factor(p - 1) <= B * B
        ss_q = second_largest_prime_factor(q - 1) <= B and largest_prime_factor(q - 1) <= B * B
        if ss_p != ss_q:
            n_semismooth_one_sided += 1
        rp = dickman_rho(math.log(p) / math.log(B))
        rq = dickman_rho(math.log(q) / math.log(B))
        pred_one_sided += rp * (1 - rq) + rq * (1 - rp)
    n = float(count)
    phat = n_success / n
    return {
        "bits": bits,
        "count": count,
        "theta": theta,
        "B_min": min(B_values),
        "B_max": max(B_values),
        "success": n_success,
        "two_sided": n_two_sided,
        "fail": n_fail,
        "success_fraction": phat,
        "success_se": math.sqrt(phat * (1.0 - phat) / n),
        "one_sided_smooth": n_one_sided_smooth,
        "both_smooth": n_both_smooth,
        "order_anomaly": n_order_anomaly,
        "semismooth_one_sided_fraction": n_semismooth_one_sided / n,
        "dickman_prediction": pred_one_sided / n,
        "u": math.log(2.0 ** (bits / 2.0)) / math.log(2.0 ** (bits * theta / 2.0)),
    }


def average_case_experiment(bits_list: Sequence[int] = (40, 48, 56, 64), count: int = 2000,
                            theta: float = THETA_DEFAULT, seed: int = 11) -> Dict:
    rows = [average_case_point(b, count, theta, seed) for b in bits_list]
    return {
        "experiment": "E40 average-case deterministic p-1 (Proposition PP)",
        "theta": theta,
        "theta_threshold_friedlander": THETA_FRIEDLANDER,
        "exponent_N": theta / 2.0,
        "alpha_fouvry": ALPHA_FOUVRY,
        "family": "rsa",
        "seed": seed,
        "rows": rows,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[40, 48, 56, 64])
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--theta", type=float, default=THETA_DEFAULT)
    ap.add_argument("--out", default=os.path.join("results", "e40_average_case_pm1.json"))
    args = ap.parse_args()
    if args.theta <= THETA_FRIEDLANDER:
        print(f"note: theta = {args.theta} is not above the Friedlander threshold {THETA_FRIEDLANDER:.6f}; "
              "the run is a heuristic benchmark, not an instance of Proposition PP")
    res = average_case_experiment(args.bits, args.count, args.theta)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res["rows"]:
        print(f"{r['bits']:3d} bits  B={r['B_min']}-{r['B_max']}  success {r['success_fraction']:.4f} "
              f"(+-{r['success_se']:.4f})  two-sided {r['two_sided']}  one-sided-smooth {r['one_sided_smooth']}  "
              f"anomaly {r['order_anomaly']}  semismooth {r['semismooth_one_sided_fraction']:.4f}  "
              f"Dickman {r['dickman_prediction']:.4f}  u={r['u']:.3f}")
