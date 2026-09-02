"""Experiments L1 and L2 of docs/lattice_barrier_plan.md.

L1 -- BKZ Gram-Schmidt profiles against the GSA on random q-ary and NTRU lattices: fitted delta, root-Hermite factor, deviation from
      the fitted line; for beta >= 40 also the deviation from the asymptotic GSA line.
L2 -- Gauss-sieve statistics: saturated list size against (4/3)^{n/2}, the shortest vector found against the exact SVP (unpruned
      enumeration, n <= 44) and the Gaussian heuristic, work counters, and the coverage statistic of Theorem T2 (fraction of uniform
      directions within 60 degrees of the list) against its uniform-sphere prediction.
Run: python -m latticelab.experiments --l1 --l2 --out results/lattice_l1_l2.json
"""
from __future__ import annotations

import json
import math
from typing import Dict, List

from fpylll import SVP, IntegerMatrix

from latticelab.lattices import gaussian_heuristic, lll, log_volume, ntru, qary
from latticelab.profile import bkz, profile_stats
from latticelab.sieve import GaussSieve, angle_histogram, coverage, predicted_coverage


def exact_svp_norm(B: IntegerMatrix) -> float:
    """Exact shortest-vector norm by unpruned enumeration (pruning is unsafe on unique-SVP instances with a large gap)."""
    v = SVP.shortest_vector(IntegerMatrix.from_matrix(B), pruning=False)
    return math.sqrt(sum(int(x) ** 2 for x in v))


def l1_profile(d: int, betas=(2, 10, 20, 30), q: int = 2 ** 20 + 7, seed: int = 1, tours: int | None = None) -> List[Dict]:
    """Profiles of a random q-ary lattice and of an NTRU lattice of the same dimension (d must be twice a power of two for NTRU)."""
    rows = []
    fams = [("qary", qary(d, d // 2, q, seed))]
    n = d // 2
    if n > 0 and n & (n - 1) == 0:
        fams.append(("ntru", ntru(n, 12289, seed)[0]))
    for fam, A in fams:
        for beta in betas:
            st = profile_stats(bkz(A, beta, tours), beta)
            st.pop("profile")
            st["family"] = fam
            rows.append(st)
    return rows


def l2_sieve(B: IntegerMatrix, label: str, seed: int = 3, max_collisions: int = 300, n_dirs: int = 4000, exact_max_dim: int = 44) -> Dict:
    """Sieve statistics: list size, exactness, work, the angle histogram of the final list (mean angle against the uniform-sphere mean
    conditioned on >= 60 degrees is not available in closed form, so the raw mean and the fraction within 65 degrees are recorded), and
    the coverage curve cov(L[:k]) against the uniform prediction at k = |L|/8, |L|/4, |L|/2, |L|."""
    n = B.nrows
    Bl = lll(B)
    lv = log_volume(Bl)
    gh = gaussian_heuristic(n, lv)
    gs = GaussSieve(Bl, seed=seed, max_collisions=max_collisions)
    res = gs.run(max_samples=500000)
    st = res["stats"]
    shortest = math.sqrt(res["shortest_norm2"])
    exact = exact_svp_norm(Bl) if n <= exact_max_dim else None
    L = gs.L
    ang = angle_histogram(L)
    curve = []
    for frac in (1 / 8, 1 / 4, 1 / 2, 1.0):
        k = max(2, int(len(L) * frac))
        curve.append({"k": k, "coverage_60": coverage(L[:k], math.pi / 3, n_dirs, seed=1),
                      "predicted": predicted_coverage(k, n, math.pi / 3)})
    return {"label": label, "n": n, "list_size": len(L), "kissing_heuristic": (4 / 3) ** (n / 2),
            "list_ratio": len(L) / (4 / 3) ** (n / 2), "shortest": shortest, "exact": exact, "gh": gh,
            "found_exact": (exact is not None and abs(shortest - exact) < 1e-6),
            "samples": st.samples, "inner_products": st.inner_products, "reductions": st.reductions, "collisions": st.collisions,
            "angle_mean_deg": float(ang.mean() * 180 / math.pi), "angle_min_deg": float(ang.min() * 180 / math.pi),
            "angle_frac_within_65": float((ang <= 65 * math.pi / 180).mean()),
            "coverage_curve": curve, "coverage_60": curve[-1]["coverage_60"], "coverage_60_predicted": curve[-1]["predicted"]}


def experiment(l1: bool, l2: bool, dims_l1=(60, 64), dims_l2=(30, 36), ntru_n=(16,), seed: int = 1) -> Dict:
    out: Dict = {"l1": [], "l2": []}
    if l1:
        for d in dims_l1:
            out["l1"].extend(l1_profile(d, seed=seed))
    if l2:
        for n in dims_l2:
            out["l2"].append(l2_sieve(qary(n, n // 2, 2 ** 14 + 27, seed + 1), "qary"))
        for n in ntru_n:
            B, f, g = ntru(n, 257, seed + 4)
            row = l2_sieve(B, f"ntru(n={n},q=257)")
            row["target_norm"] = math.sqrt(sum(x * x for x in f) + sum(x * x for x in g))
            out["l2"].append(row)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", action="store_true")
    ap.add_argument("--l2", action="store_true")
    ap.add_argument("--out", default="results/lattice_l1_l2.json")
    args = ap.parse_args()
    res = experiment(args.l1, args.l2)
    for r in res["l1"]:
        print(f"L1 {r['family']:5s} d={r['d']} beta={r['beta']:2d} delta_fit={r['delta_fit']:.4f} rhf={r['root_hermite']:.4f} maxdev_fit={r['max_dev_fit']:.2f}"
              + (f" delta_gsa={r['delta_gsa']:.4f} maxdev_gsa={r['max_dev_gsa']:.2f}" if "delta_gsa" in r else ""))
    for r in res["l2"]:
        print(f"L2 {r['label']} n={r['n']}: |L|={r['list_size']} ratio={r['list_ratio']:.2f} shortest={r['shortest']:.1f} exact={r['exact']} "
              f"GH={r['gh']:.1f} found_exact={r['found_exact']} ips={r['inner_products']} angle_mean={r['angle_mean_deg']:.1f} "
              f"curve={[(c['k'], round(c['coverage_60'], 3), round(c['predicted'], 3)) for c in r['coverage_curve']]}")
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
