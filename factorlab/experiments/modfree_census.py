"""E39: modulus-free census of the drift-free two-progression families.

The integrality lemma parametrises the drift-free families ((A d^2 - d + C)/2, (A d^2 + d + C)/2) on the
classes d = D_0 (mod q) by integers alpha = A q^2, gamma = C q^2 with 4 alpha gamma = q^2 (q^2 - M),
M = (q^4 - 4 alpha gamma)/q^2.  The symmetric family is q = 1, M = 1 (C = 0).  To first order a family
with n classes has modulus-free capacity proportional to n q^{-1/3} |M|^{-1/3} r^{1/3} at its balance point
A* = ((sqrt2 - 1)|M|/(4 sqrt2 theta))^{2/3} r^{1/3} q^{-4/3}, so multi-class families with |M| = 1 and
q = 15, 105 could exceed the symmetric family by a constant factor (n q^{-1/3} = 1.62, 1.70).  The exact
2^18 sweep of E37 returned such a family (A = 7/15, C = 8/15: q = 15, M = 1, four classes) with 31 pairs
against the symmetric family's 25.  This census enumerates the families (q <= q_max, |M| <= m_max,
alpha in a wide window around A* q^2) and computes each family's exact cluster: its members in the shell
(all classes pooled), the speeds (k' - k)/(sqrt k' + sqrt k), and the largest number of squarefree pairs
(and of all pairs) in one window of width 2 theta rho_r.  The maximum over the census is a lower bound
for D*_theta(r); equality with the exact sweep at 2^18 validates the enumeration.

Run:  python -m factorlab.experiments.modfree_census --r 262144 1048576 4194304 [--q-max 120] [--m-max 64]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from factorlab.experiments.resonant_census import (integral_classes, family_members, divisors_in_range,
                                                   squarefree_mask)
from factorlab.experiments.sidon_bucketed import rho


def balance_A(r: int, q: int, M: int, theta: float = 1.0) -> float:
    """First-order balance point of population and capacity for the family (q, M)."""
    return ((math.sqrt(2) - 1) * abs(M) / (4 * math.sqrt(2) * theta)) ** (2 / 3) * r ** (1 / 3) / q ** (4 / 3)


def family_cluster(r: int, alpha: int, gamma: int, q: int, theta: float = 1.0):
    """Exact cluster of the family (alpha, gamma, q) at resolution rho_r: members of all integral classes
    in the shell, speeds, and the largest number of squarefree pairs / of all pairs in one window."""
    classes = integral_classes(alpha, gamma, q)
    if not classes:
        return None
    km, kp = [], []
    for D0 in classes:
        for d, a, b in family_members(r, alpha, gamma, q, D0):
            km.append(a)
            kp.append(b)
    if len(km) < 2:
        return None
    km = np.asarray(km, dtype=np.int64)
    kp = np.asarray(kp, dtype=np.int64)
    # distinct pairs only (classes may overlap when q is not the primitive period)
    pairs = np.unique(np.stack([km, kp], axis=1), axis=0)
    km, kp = pairs[:, 0], pairs[:, 1]
    v = (kp - km) / (np.sqrt(kp.astype(float)) + np.sqrt(km.astype(float)))
    w = 2.0 * theta * rho(r)

    def cluster(vals):
        if len(vals) == 0:
            return 0, None
        vs = np.sort(vals)
        ends = np.searchsorted(vs, vs + w, side="left")
        counts = ends - np.arange(len(vs))
        i = int(np.argmax(counts))
        return int(counts[i]), float(vs[i] + w / 2)

    sf = squarefree_mask(km) & squarefree_mask(kp)
    c_all, tau_all = cluster(v)
    c_sf, tau_sf = cluster(v[sf])
    return {"alpha": int(alpha), "gamma": int(gamma), "q": int(q), "classes": [int(c) for c in classes],
            "n_classes": len(classes), "A": f"{alpha}/{q * q}", "C": f"{gamma}/{q * q}",
            "M": int((q ** 4 - 4 * alpha * gamma) // (q * q)), "members": int(len(km)),
            "members_sf": int(sf.sum()), "cluster_all": c_all, "cluster_sf": c_sf,
            "tau_sf": tau_sf, "speed": float(1 / math.sqrt(2 * alpha / (q * q)))}


def enumerate_box(r: int, q_max: int = 120, m_max: int = 64, theta: float = 1.0, A_window: float = 6.0):
    """Yield (q, M, alpha, gamma, A_star) over the census box: q <= q_max, 0 < |M| <= m_max, alpha a divisor
    of q^2 (q^2 - M)/4 in [A* q^2 / A_window, A* q^2 A_window], plus every C = 0 family of period q = 1."""
    for q in range(1, q_max + 1):
        q2 = q * q
        Ms = [M for M in range(-m_max, m_max + 1) if M != 0 and M != q2 and (q2 * (q2 - M)) % 4 == 0]
        if q == 1:
            Ms.append(1)  # C = 0, the symmetric families
        for M in Ms:
            A_star = balance_A(r, q, M, theta)
            lo, hi = A_star * q2 / A_window, A_star * q2 * A_window
            if M == q2:
                alphas = list(range(max(1, math.ceil(lo)), math.floor(hi) + 1))
                gammas = [0] * len(alphas)
            else:
                n_int = q2 * (q2 - M) // 4
                alphas = divisors_in_range(abs(n_int), lo, hi)
                gammas = [n_int // a for a in alphas]
            for alpha, gamma in zip(alphas, gammas):
                yield q, M, alpha, gamma, A_star


def family_pairs(r: int, alpha: int, gamma: int, q: int):
    """Distinct shell pairs (k_-, k_+) of the family over all its integral classes, or None."""
    classes = integral_classes(alpha, gamma, q)
    if not classes:
        return None
    km, kp = [], []
    for D0 in classes:
        for d, a, b in family_members(r, alpha, gamma, q, D0):
            km.append(a)
            kp.append(b)
    if len(km) < 2:
        return None
    pairs = np.unique(np.stack([np.asarray(km, dtype=np.int64), np.asarray(kp, dtype=np.int64)], axis=1), axis=0)
    return pairs


def pooled_cluster(r: int, q_max: int = 120, m_max: int = 64, theta: float = 1.0, A_window: float = 6.0):
    """E41(b): pile-up test inside the enumerated class.  Pool the distinct pairs of every family of the
    census box and take the largest window of width 2 theta rho_r over the union (all pairs, and squarefree
    pairs); compare with the largest single-family cluster.  pooled > max_family means several families
    share one window (a pile-up); equality means the class's maximum is carried by one family."""
    t0 = time.time()
    all_pairs = []
    best_sf = 0
    best_all = 0
    n_fam = 0
    for q, M, alpha, gamma, A_star in enumerate_box(r, q_max, m_max, theta, A_window):
        pairs = family_pairs(r, alpha, gamma, q)
        n_fam += 1
        if pairs is None:
            continue
        all_pairs.append(pairs)
        km, kp = pairs[:, 0], pairs[:, 1]
        v = (kp - km) / (np.sqrt(kp.astype(float)) + np.sqrt(km.astype(float)))
        w = 2.0 * theta * rho(r)
        vs = np.sort(v)
        c_all = int((np.searchsorted(vs, vs + w, side="left") - np.arange(len(vs))).max())
        sf = squarefree_mask(km) & squarefree_mask(kp)
        vsf = np.sort(v[sf])
        c_sf = int((np.searchsorted(vsf, vsf + w, side="left") - np.arange(len(vsf))).max()) if len(vsf) else 0
        best_sf = max(best_sf, c_sf)
        best_all = max(best_all, c_all)
    if not all_pairs:
        return {"r": int(r), "theta": theta, "q_max": q_max, "m_max": m_max, "A_window": A_window,
                "families": n_fam, "pairs_pooled": 0, "pairs_pooled_sf": 0, "pooled_sf": 0, "pooled_all": 0,
                "tau_sf": None, "pooled_sf_bracket": [0, 0], "pooled_all_bracket": [0, 0], "speed_eps": 0.0,
                "max_family_sf": 0, "max_family_all": 0, "pileup_sf": False,
                "pileup_all": False, "time_s": time.time() - t0}
    pairs = np.unique(np.concatenate(all_pairs, axis=0), axis=0)
    km, kp = pairs[:, 0], pairs[:, 1]
    v = (kp - km) / (np.sqrt(kp.astype(float)) + np.sqrt(km.astype(float)))
    w = 2.0 * theta * rho(r)

    def cluster(vals, width=w):
        if len(vals) == 0:
            return 0, None
        vs = np.sort(vals)
        counts = np.searchsorted(vs, vs + width, side="left") - np.arange(len(vs))
        i = int(np.argmax(counts))
        return int(counts[i]), float(vs[i] + width / 2)

    sf = squarefree_mask(km) & squarefree_mask(kp)
    p_all, tau_all = cluster(v)
    p_sf, tau_sf = cluster(v[sf])
    # two-tolerance bracket: |v_float - v_true| <= eps for every pooled pair (4u times the largest speed),
    # so the true count in any window lies between the float counts at widths w -/+ 2 eps
    eps = 4.0 * 2.0 ** -53 * float(np.abs(v).max()) * 1.01
    p_sf_lo, _ = cluster(v[sf], w - 2 * eps)
    p_sf_hi, _ = cluster(v[sf], w + 2 * eps)
    p_all_lo, _ = cluster(v, w - 2 * eps)
    p_all_hi, _ = cluster(v, w + 2 * eps)
    return {"r": int(r), "theta": theta, "q_max": q_max, "m_max": m_max, "A_window": A_window,
            "families": n_fam, "pairs_pooled": int(len(km)), "pairs_pooled_sf": int(sf.sum()),
            "pooled_sf": p_sf, "pooled_all": p_all, "tau_sf": tau_sf,
            "pooled_sf_bracket": [p_sf_lo, p_sf_hi], "pooled_all_bracket": [p_all_lo, p_all_hi],
            "speed_eps": eps,
            "max_family_sf": best_sf, "max_family_all": best_all,
            "pileup_sf": p_sf > best_sf, "pileup_all": p_all > best_all, "time_s": time.time() - t0}


def in_census_box(A, C, r: int, q_max: int = 120, m_max: int = 64, theta: float = 1.0,
                  A_window: float = 6.0) -> bool:
    """Whether the drift-free family with reduced coefficients A, C (Fractions) is enumerated by
    ``enumerate_box``: some q <= q_max makes alpha = A q^2 and gamma = C q^2 integers with
    M = (q^4 - 4 alpha gamma)/q^2 an integer, either (M = q^2 = 1, the symmetric C = 0 case) or
    (M != 0, M != q^2, |M| <= m_max), and alpha inside [A* q^2 / A_window, A* q^2 A_window]."""
    from fractions import Fraction
    A = Fraction(A)
    C = Fraction(C)
    for q in range(1, q_max + 1):
        q2 = q * q
        alpha, gamma = A * q2, C * q2
        if alpha.denominator != 1 or gamma.denominator != 1 or alpha <= 0:
            continue
        alpha, gamma = int(alpha), int(gamma)
        M = Fraction(q2 * q2 - 4 * alpha * gamma, q2)
        if M.denominator != 1:
            continue
        M = int(M)
        if M == 0:
            continue
        if M == q2:
            if q != 1:
                continue          # C = 0 with q > 1 is not enumerated
        elif abs(M) > m_max:
            continue
        A_star = balance_A(r, q, M, theta)
        if A_star * q2 / A_window <= alpha <= A_star * q2 * A_window:
            return True
    return False


def census(r: int, q_max: int = 120, m_max: int = 64, theta: float = 1.0, A_window: float = 6.0,
           verbose: bool = False):
    """Enumerate the families with q <= q_max and 0 < |M| <= m_max, alpha a divisor of q^2 (q^2 - M)/4 in
    [A* q^2 / A_window, A* q^2 A_window], together with every C = 0 family of period q = 1 (M = 1, alpha any
    integer in the window: the symmetric families, enumerated exhaustively so that the baseline is exact).
    C = 0 families with q > 1 (M = q^2) are omitted: their first-order potential n q^{-1/3} |M|^{-1/3} = n/q
    never exceeds the symmetric family's.  Returns the best families by squarefree cluster."""
    t0 = time.time()
    best = []
    n_fam = 0
    for q, M, alpha, gamma, A_star in enumerate_box(r, q_max, m_max, theta, A_window):
        res = family_cluster(r, alpha, gamma, q, theta)
        n_fam += 1
        if res is None:
            continue
        res["A_star"] = A_star
        best.append(res)
    # merge non-primitive representations (same reduced A and C), keeping the smallest q
    from fractions import Fraction
    seen = {}
    for x in best:
        key = (Fraction(x["alpha"], x["q"] ** 2), Fraction(x["gamma"], x["q"] ** 2))
        if key not in seen or x["q"] < seen[key]["q"]:
            seen[key] = x
    best = list(seen.values())
    for x in best:
        x["density_sf"] = x["members_sf"] / x["members"] if x["members"] else None
    best.sort(key=lambda x: (-x["cluster_sf"], -x["cluster_all"]))
    sym = [x for x in best if x["q"] == 1 and x["gamma"] == 0]
    out = {"r": int(r), "theta": theta, "q_max": q_max, "m_max": m_max, "families": n_fam,
           "distinct_families": len(best),
           "A_window": A_window, "C0_convention": "C = 0 enumerated exhaustively for q = 1 only",
           "r_cuberoot": r ** (1 / 3), "top": best[:25],
           "max_sf": best[0]["cluster_sf"] if best else 0,
           "max_all": max((x["cluster_all"] for x in best), default=0),
           "sym_sf": sym[0]["cluster_sf"] if sym else None, "sym_top": sym[0] if sym else None,
           "time_s": time.time() - t0}
    out["ratio_max_over_sym"] = (out["max_sf"] / out["sym_sf"]) if out["sym_sf"] else None
    if verbose:
        b = best[0] if best else {}
        print(f"r=2^{r.bit_length() - 1}: families={n_fam} max_sf={out['max_sf']} "
              f"(A={b.get('A')}, C={b.get('C')}, q={b.get('q')}, M={b.get('M')}, n={b.get('n_classes')}, "
              f"members_sf={b.get('members_sf')}) sym_sf={out['sym_sf']} ratio={out['ratio_max_over_sym']} "
              f"max/r^(1/3)={out['max_sf'] / r ** (1 / 3):.3f} ({out['time_s']:.0f}s)", flush=True)
    return out


def run(rs, q_max, m_max, theta, out_path, A_window=6.0):
    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    for r in rs:
        res = census(r, q_max=q_max, m_max=m_max, theta=theta, A_window=A_window, verbose=True)
        results = [x for x in results if x["r"] != r] + [res]
        results.sort(key=lambda x: x["r"])
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1, default=str)
        os.replace(tmp, out_path)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, nargs="+", required=True)
    ap.add_argument("--q-max", type=int, default=120)
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--A-window", type=float, default=6.0)
    ap.add_argument("--out", default="results/e39_modfree_census.json")
    a = ap.parse_args()
    run(a.r, a.q_max, a.m_max, a.theta, a.out, a.A_window)
