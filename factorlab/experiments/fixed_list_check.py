"""E21: the fixed-list deterministic ECM on N, against the E20 simulation.

Two checks.  (1) Certificate: E20 found that the Suyama curves sigma = 6..44
with B1 = 64, B2 = 4096 have distinct one-large-prime signatures on every prime
in [2^18, 2^19); by Proposition V the algorithm `fixed_list_ecm` must factor
every N = pq with distinct p, q in that range within 39 curves.  This module
runs it on random pairs and on the pairs that were last to separate in E20,
and compares the curve index at which it succeeds with the simulated
separation index of the pair (the algorithm may succeed earlier -- its gcds
see one-sided relations that the collapsed bit hides -- never later).
(2) Cost: on RSA moduli with N-derived parameters (u, C), the work counters
and the number of curves, with a fit of log(mulmod) against log N.
"""

from __future__ import annotations

import json
import math
import os
import resource
import time
from typing import Sequence

import numpy as np

from ..gen import make_semiprime
from ..registry import get_algorithm
from .ecm_hitting import primes_in_range
from .hitting_sets import ecm_success


def simulated_separation_index(p: int, q: int, B1: int, B2: int, max_curves: int = 120,
                               sigma0: int = 6):
    """First curve (1-based) whose collapsed one-large-prime bits differ on p and q, or None."""
    ps = np.array([p, q], dtype=np.int64)
    for i in range(max_curves):
        s1, s2 = ecm_success(sigma0 + i, ps, B1, B2)
        bits = s1 | s2
        if bool(bits[0]) != bool(bits[1]):
            return i + 1
    return None


def certificate_test(bits: int = 18, B1: int = 64, B2: int = 4096, count: int = 500, seed: int = 3,
                     tail_pairs_path: str = "results/e20_last_pairs.json", max_curves: int = 60,
                     bound: int = 39) -> dict:
    algo = get_algorithm("fixed_list_ecm")
    primes = primes_in_range(1 << bits, 1 << (bits + 1))
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < count:
        i, j = (int(v) for v in rng.integers(0, primes.size, 2))
        if i != j:
            pairs.add((int(primes[min(i, j)]), int(primes[max(i, j)])))
    tail = []
    if os.path.exists(tail_pairs_path):
        d = json.load(open(tail_pairs_path))
        for key, val in d.items():
            if key.startswith(f"{bits}:"):
                tail += [tuple(int(v) for v in pr) for pr in (val.get("pairs") or [])]
                tail += [tuple(int(v) for v in pr) for pr in (val.get("last_pairs") or [])]
    tail = sorted(set(tail))
    rows = []
    for (p, q) in sorted(pairs | set(tail)):
        res = algo(p * q, B1=B1, B2=B2, max_curves=max_curves)
        sim = simulated_separation_index(p, q, B1, B2, max_curves)
        rows.append({"p": p, "q": q, "found": bool(res.found), "curves": int(res.meta.get("curve", max_curves)),
                     "stage": res.meta.get("stage"), "detail": res.meta.get("detail"),
                     "two_sided": int(res.meta.get("two_sided", 0)), "sim_index": sim,
                     "mulmod": int(res.work.get("mulmod", 0)), "is_tail_pair": (p, q) in set(tail)})
    curves = np.array([r["curves"] for r in rows if r["found"]])
    sims = np.array([r["sim_index"] if r["sim_index"] is not None else -1 for r in rows])
    ok_sim = np.array([r["sim_index"] is not None for r in rows])
    earlier = [r for r in rows if r["found"] and r["sim_index"] is not None and r["curves"] < r["sim_index"]]
    later = [r for r in rows if r["found"] and r["sim_index"] is not None and r["curves"] > r["sim_index"]]
    return {"bits": bits, "B1": B1, "B2": B2, "n_pairs": len(rows), "n_tail_pairs": len(tail),
            "all_found": bool(all(r["found"] for r in rows)),
            "all_within_bound": bool(all(r["found"] and r["curves"] <= bound for r in rows)),
            "max_curves_used": int(curves.max()) if curves.size else None,
            "mean_curves_used": float(curves.mean()) if curves.size else None,
            "curve_histogram": {str(k): int(v) for k, v in zip(*np.unique(curves, return_counts=True))} if curves.size else {},
            "stage_counts": {s: sum(1 for r in rows if r["stage"] == s) for s in ("den", "1", "2")},
            "fraction_earlier_than_simulation": len(earlier) / max(1, int(ok_sim.sum())),
            "n_later_than_simulation": len(later),
            "mean_sim_index": float(sims[ok_sim].mean()) if ok_sim.any() else None,
            "tail_pairs": [r for r in rows if r["is_tail_pair"]],
            "rows": rows}


def modeled_u_profile(nbits: int, us: Sequence[float] = tuple(2.5 + 0.25 * i for i in range(19))) -> dict:
    """Bach-Peralta model for choosing u on a random balanced semiprime.

    A prime is exposed with ideal probability G(1/u, 2/u).  The conservative
    binary probability that exactly one of p,q is exposed is 2G(1-G); exact
    residual-order labels can only improve it.  The score is expected work
    B1 / (2G(1-G)), in log2 units, with B1 ~ N^{1/(2u)}.
    """
    from .smooth_profiles import semismooth_G
    rows = []
    for u in us:
        g = semismooth_G(1.0 / u, min(2.0 / u, 1.0))
        p_one = 2.0 * g * (1.0 - g)
        log2_B1 = nbits / (2.0 * u)
        rows.append({"u": float(u), "G": g, "binary_one_sided_probability": p_one,
                     "log2_B1": log2_B1,
                     "log2_expected_work": log2_B1 - math.log2(max(p_one, 1e-300)),
                     "expected_binary_curves": 1.0 / max(p_one, 1e-300)})
    best = min(rows, key=lambda r: r["log2_expected_work"])
    return {"nbits": int(nbits), "best_u": best["u"], "best": best, "rows": rows,
            "qualification": "random-integer semismooth model; exact curve-order labels and fixed-family correlations omitted"}


def scalability_probe(nbits: int, index: int = 0, u: float = 3.0, C: float = 2.0,
                      seed: int = 211, family: str = "rsa", max_curves: int = 400) -> dict:
    """One fresh-process scalability point: algorithm work/wall and process peak RSS.

    Invoke through run_fixed_scale so ru_maxrss belongs to one bit size (apart
    from imports).  This measures random-instance implementation scaling, not
    the asymptotic worst-case list length of Conjecture E.
    """
    from ..algorithms.fixed_list_ecm import fixed_list_parameters
    inst = make_semiprime(int(nbits), family, seed, int(index))
    B1, B2 = fixed_list_parameters(inst.N, u, C)
    algo = get_algorithm("fixed_list_ecm")
    t0 = time.perf_counter()
    res = algo(inst.N, u=u, C=C, max_curves=max_curves)
    elapsed = time.perf_counter() - t0
    if not res.found:
        raise RuntimeError(f"fixed_list_ecm failed at {nbits} bits: {res.meta}")
    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"nbits": int(nbits), "index": int(index), "N": str(inst.N),
            "B1": int(B1), "B2": int(B2), "stage2_degree": int(math.isqrt(B2)) + 1,
            "found": True, "curve": int(res.meta["curve"]), "stage": res.meta["stage"],
            "two_sided": int(res.meta.get("two_sided", 0)),
            "mulmod": int(res.work.get("mulmod", 0)), "poly_deg": int(res.work.get("poly_deg", 0)),
            "gcd": int(res.work.get("gcd", 0)), "wall": float(elapsed),
            "peak_rss_kb": rss_kb, "u": u, "C": C, "family": family}


def cost_scaling(bits: Sequence[int] = (32, 40, 48, 56, 64), count: int = 20, u: float = 3.0, C: float = 2.0,
                 seed: int = 9, family: str = "rsa", max_curves: int = 400) -> dict:
    algo = get_algorithm("fixed_list_ecm")
    rows = []
    for nbits in bits:
        per = []
        for i in range(count):
            inst = make_semiprime(nbits, family, seed, i)
            res = algo(inst.N, u=u, C=C, max_curves=max_curves)
            assert res.found and {int(res.p), int(res.q)} == {int(inst.p), int(inst.q)}, (inst, res.meta)
            per.append({"N": str(inst.N), "curves": int(res.meta["curve"]), "stage": res.meta["stage"],
                        "B1": int(res.meta["B1"]), "B2": int(res.meta["B2"]), "two_sided": int(res.meta["two_sided"]),
                        "mulmod": int(res.work.get("mulmod", 0)), "poly_deg": int(res.work.get("poly_deg", 0)),
                        "gcd": int(res.work.get("gcd", 0)), "wall": res.wall})
        mm = np.array([r["mulmod"] for r in per], dtype=float)
        cv = np.array([r["curves"] for r in per], dtype=float)
        b1s = np.array([r["B1"] for r in per], dtype=float)
        b2s = np.array([r["B2"] for r in per], dtype=float)
        rows.append({"nbits": nbits, "count": count, "B1": per[0]["B1"], "B2": per[0]["B2"],
                     "B1_range": [int(b1s.min()), int(b1s.max())], "B2_range": [int(b2s.min()), int(b2s.max())],
                     "mean_mulmod": float(mm.mean()), "median_mulmod": float(np.median(mm)),
                     "mean_curves": float(cv.mean()), "max_curves": int(cv.max()),
                     "mean_mulmod_per_curve_over_B1": float(np.mean(mm / (cv * b1s))),
                     "mean_poly_deg": float(np.mean([r["poly_deg"] for r in per])),
                     "mean_wall": float(np.mean([r["wall"] for r in per])),
                     "stage_counts": {s: sum(1 for r in per if r["stage"] == s) for s in ("den", "1", "2")},
                     "instances": per})
    x = np.array([r["nbits"] for r in rows], dtype=float)  # log2 N
    y = np.array([math.log2(r["mean_mulmod"]) for r in rows])
    yc = np.array([math.log2(r["mean_curves"]) for r in rows])
    slope, icpt = np.polyfit(x, y, 1)
    slope_c, _ = np.polyfit(x, yc, 1)
    return {"u": u, "C": C, "family": family, "rows": rows,
            "fit": {"mulmod_exponent": float(slope), "intercept": float(icpt), "curves_exponent": float(slope_c),
                    "predicted_exponent": 1.0 / (2 * u)}}
