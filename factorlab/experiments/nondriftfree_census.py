"""E43 -- non-drift-free two-progression families (B != 0) with a stationary speed in the shell.

A pair family s = k_- + k_+ = A d^2 + B d + C (d = k_+ - k_-) with B != 0 has speed drift ~ B/(2 sqrt(2A) w)
to first order, independent of d, so away from stationary points of the speed a residue class holds O(1) pairs
in a window of half-width theta rho_r.  The loophole left open by Theorems V and V_P is a stationary point of
the speed inside the shell: with Delta := 1/4 - AC the first-order stationary point is d_* = 2(B^2 + Delta)/(AB),
and there a class can hold ~ theta sqrt(2q/|beta|) pairs (B = beta/q), independent of r.  This census places
d_* inside the shell by construction -- for each (q, alpha, beta) and a target d_* = c sqrt(r/A) it sets
C = (1/4 + B^2 - (B c/2) sqrt(rA)) / A and rounds gamma = C q^2 to nearby integers -- enumerates the admissible
residues mod 2q^2, collects the shell members, and computes each family's exact largest window (all pairs and
squarefree pairs).  The maximum over the box is compared with r^{1/3} and with the drift-free census maxima.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from fractions import Fraction

import numpy as np

from factorlab.experiments.resonant_census import squarefree_mask
from factorlab.experiments.sidon_bucketed import rho


def admissible_residues(q: int, alpha: int, beta: int, gamma: int) -> np.ndarray:
    """Residues d0 mod L = 2q^2 with k_-(d) = (alpha d^2 + (beta q - q^2) d + gamma) / (2 q^2) an integer."""
    L = 2 * q * q
    d0 = np.arange(L, dtype=np.int64)
    num = alpha * d0 * d0 + (beta * q - q * q) * d0 + gamma
    return d0[num % L == 0]


def family_members(r: int, q: int, alpha: int, beta: int, gamma: int):
    """Shell pairs (k_-, k_+) of the family: r/2 < k_- < k_+ <= r."""
    res = admissible_residues(q, alpha, beta, gamma)
    if len(res) == 0:
        return None
    L = 2 * q * q
    A = alpha / (q * q)
    dmax = int(math.isqrt(int(2 * r / A) + 1)) + L if A > 0 else r
    dmax = min(dmax, r)
    m = np.arange(0, dmax // L + 2, dtype=np.int64)
    d = (res[:, None] + L * m[None, :]).ravel()
    d = d[(d >= 1) & (d <= r)]
    if len(d) == 0:
        return None
    num = alpha * d * d + (beta * q - q * q) * d + gamma      # = 2 q^2 k_-
    km = num // L
    kp = km + d
    keep = (km > r // 2) & (kp <= r) & (km >= 1)
    km, kp = km[keep], kp[keep]
    if len(km) < 2:
        return None
    return km, kp


def family_cluster(r: int, q: int, alpha: int, beta: int, gamma: int, theta: float = 1.0):
    mem = family_members(r, q, alpha, beta, gamma)
    if mem is None:
        return None
    km, kp = mem
    v = (kp - km) / (np.sqrt(kp.astype(float)) + np.sqrt(km.astype(float)))
    w = 2.0 * theta * rho(r)
    vs = np.sort(v)
    c_all = int((np.searchsorted(vs, vs + w, side="left") - np.arange(len(vs))).max())
    sf = squarefree_mask(km) & squarefree_mask(kp)
    vsf = np.sort(v[sf])
    c_sf = int((np.searchsorted(vsf, vsf + w, side="left") - np.arange(len(vsf))).max()) if len(vsf) else 0
    return {"q": q, "alpha": alpha, "beta": beta, "gamma": gamma, "A": str(Fraction(alpha, q * q)),
            "B": str(Fraction(beta, q)), "C": str(Fraction(gamma, q * q)), "members": int(len(km)),
            "members_sf": int(sf.sum()), "cluster_all": c_all, "cluster_sf": c_sf}


def driftfree_baseline(r: int, path: str = "results/e39_modfree_census.json"):
    """The drift-free census maxima at radius r (E39: max_sf, largest cluster_all among its top families, sym_sf),
    or None when the radius is not in the file."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = json.load(f)
    for row in rows:
        if int(row["r"]) == int(r):
            stored = row.get("max_all")
            top_all = max((x.get("cluster_all", 0) for x in row.get("top", [])), default=None)
            return {"driftfree_max_sf": row.get("max_sf"),
                    "driftfree_max_all": stored if stored is not None else top_all,
                    "driftfree_max_all_source": "stored max_all" if stored is not None else "largest cluster_all among the top-by-squarefree list (a lower bound)",
                    "driftfree_sym_sf": row.get("sym_sf")}
    return None


def census(r: int, q_max: int = 24, betas=(-3, -2, -1, 1, 2, 3), A_grid: int = 48, A_lo: float = 2.0 ** -10,
           A_hi: float = 4.0, d_targets=(0.8, 1.0, 1.2), gamma_offsets=(-2, -1, 0, 1, 2), theta: float = 1.0,
           verbose: bool = False, baseline_path: str = "results/e39_modfree_census.json"):
    t0 = time.time()
    best_sf, best_all = [], []
    n_fam = 0
    seen = set()
    for q in range(1, q_max + 1):
        for beta in betas:
            B = beta / q
            for A_t in np.exp(np.linspace(math.log(A_lo), math.log(A_hi), A_grid)):
                alpha = int(round(A_t * q * q))
                if alpha < 1:
                    continue
                A = alpha / (q * q)
                for c in d_targets:
                    # stationary point at d_* = c sqrt(r/A): C = (1/4 + B^2 - (B c / 2) sqrt(r A)) / A
                    C_t = (0.25 + B * B - 0.5 * B * c * math.sqrt(r * A)) / A
                    g0 = int(round(C_t * q * q))
                    for off in gamma_offsets:
                        gamma = g0 + off
                        key = (q, alpha, beta, gamma)
                        if key in seen:
                            continue
                        seen.add(key)
                        res = family_cluster(r, q, alpha, beta, gamma, theta)
                        n_fam += 1
                        if res is None:
                            continue
                        best_sf.append(res)
                        best_all.append(res)
    best_sf.sort(key=lambda x: -x["cluster_sf"])
    best_all.sort(key=lambda x: -x["cluster_all"])
    out = {"r": int(r), "q_max": q_max, "betas": list(betas), "A_grid": A_grid, "A_range": [A_lo, A_hi],
           "families_tried": n_fam, "families_with_pairs": len(best_sf),
           "max_sf": best_sf[0]["cluster_sf"] if best_sf else 0, "max_all": best_all[0]["cluster_all"] if best_all else 0,
           "r_cuberoot": r ** (1 / 3), "top_sf": best_sf[:8], "top_all": best_all[:8], "time_s": time.time() - t0}
    base = driftfree_baseline(r, baseline_path)
    out["driftfree_baseline"] = base
    out["ratio_sf_over_driftfree"] = (out["max_sf"] / base["driftfree_max_sf"]) if base and base.get("driftfree_max_sf") else None
    out["ratio_all_over_driftfree"] = (out["max_all"] / base["driftfree_max_all"]) if base and base.get("driftfree_max_all") else None
    out["max_sf_over_r_cuberoot"] = out["max_sf"] / out["r_cuberoot"]
    if verbose:
        print(f"r=2^{r.bit_length() - 1}: B!=0 families tried={n_fam} with pairs={len(best_sf)} max_sf={out['max_sf']} "
              f"max_all={out['max_all']} r^(1/3)={out['r_cuberoot']:.1f} driftfree={base} "
              f"ratio_sf={out.get('ratio_sf_over_driftfree')} ({out['time_s']:.0f}s)", flush=True)
        for x in out["top_sf"][:4]:
            print("   ", {k: x[k] for k in ("A", "B", "C", "q", "members", "cluster_sf", "cluster_all")}, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, nargs="+", default=[2 ** 16, 2 ** 18, 2 ** 20])
    ap.add_argument("--q-max", type=int, default=24)
    ap.add_argument("--A-grid", type=int, default=48)
    ap.add_argument("--out", default="results/e43_nondriftfree_census.json")
    a = ap.parse_args()
    rows = [census(r, q_max=a.q_max, A_grid=a.A_grid, verbose=True) for r in a.r]
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1, default=str)
