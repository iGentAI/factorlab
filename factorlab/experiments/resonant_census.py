"""E32: a census of offset-resonant two-progression families with a modulus.

The exact-start statistic of the a = 1 cells in the planar regime (E31, notes_barrier 7.11) is carried, above the
null, by two-progression families whose start difference D(d) = u v(d) - d is stationary inside the shell.  The
exact pair sweep that found them is quadratic in the shell size and stops at 48 bits.  This module enumerates the
drift-free candidates directly from the integrality lemma of 7.10: a family is ((A d^2 - d + C)/2, (A d^2 + d + C)/2)
on a residue class D_0 modulo q with alpha = A q^2 and gamma = C q^2 integers, 4 alpha gamma = q^2 (q^2 - M),
Delta_d = M/(4 q^2); its stationary point lies in the shell (to first order in the expansion of v) iff
    alpha in [|M| W_real/(2 sqrt2), |M| W_real),      W_real = sqrt N/(2 sqrt2 r^{3/2}),  1/(2 sqrt2) = 0.3536,
so for each (q, M < 0) the admissible alpha are the divisors of q^2 (q^2 - M)/4 in that range (widened by a
tolerance), gamma follows, and the classes D_0 are those for which the two members are integers (a three-point
test).  Representations of one algebraic family (A, C) by several (q, D_0) are merged.  For each family the members
in the shell, their exact start differences on N, the largest window of 2W - 1 consecutive values (the family's
contribution to D_max) on the full shell, on the squarefree members and on the prime members, and the detuning
at the median d are computed.  Non-drift-free (B != 0) resonances at small d (7.11) are outside this census.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import numpy as np
import gmpy2

from .sidon_scaling import _ceil_2sqrt, lemma_d_window


def w_real(N: int, r: int) -> float:
    """Harvey's real window of the cell (1, r/2): sqrt N/(4 r sqrt(r/2)) = sqrt N/(2 sqrt2 r^{3/2})."""
    return math.sqrt(N) / (2 * math.sqrt(2) * r ** 1.5)


def divisors_in_range(n: int, lo: float, hi: float) -> list[int]:
    """Divisors alpha of n with lo <= alpha <= hi, by trial division up to sqrt n."""
    out = []
    a = 1
    root = math.isqrt(n)
    while a <= root:
        if n % a == 0:
            b = n // a
            if lo <= a <= hi:
                out.append(a)
            if b != a and lo <= b <= hi:
                out.append(b)
        a += 1
    return sorted(out)


def integral_classes(alpha: int, gamma: int, q: int) -> list[int]:
    """Residues D_0 modulo q for which both (alpha d^2 -+ q^2 d + gamma)/(2 q^2) are integers on d = D_0 (mod q)
    (a quadratic integer-valued at three consecutive points of the progression is integer-valued on all of it)."""
    m = 2 * q * q
    out = []
    for D0 in range(q):
        ok = True
        for d in (D0, D0 + q, D0 + 2 * q):
            base = alpha * d * d + gamma
            if (base - q * q * d) % m or (base + q * q * d) % m:
                ok = False
                break
        if ok:
            out.append(D0)
    return out


def family_members(r: int, alpha: int, gamma: int, q: int, D0: int) -> list[tuple[int, int, int]]:
    """(d, k_-, k_+) for d = D_0 (mod q), d >= 1, with r/2 < k_- and k_+ <= r, where
    k_-+ = (alpha d^2 -+ q^2 d + gamma)/(2 q^2).  The condition k_+ <= r bounds d above; k_- > r/2 holds for
    d below the smaller root or above the larger root of k_-(d) = r/2 (when C = gamma/q^2 > r the small-d
    branch is non-empty), so both branches are scanned and every candidate is checked exactly."""
    q2 = q * q
    # k_+ <= r  <=>  alpha d^2 + q^2 d + gamma <= 2 r q^2: real root gives the upper end
    disc = q2 * q2 - 4 * alpha * (gamma - 2 * r * q2)
    if disc < 0:
        return []
    d_hi = int((-q2 + math.isqrt(disc)) // (2 * alpha)) + 2
    # k_- > r/2  <=>  alpha d^2 - q^2 d + gamma > r q^2 (integer r//2: k_- >= r//2 + 1)
    disc2 = q2 * q2 - 4 * alpha * (gamma - (2 * (r // 2) + 2) * q2)
    if disc2 < 0:
        ranges = [(1, d_hi)]                      # k_- > r/2 for every d
    else:
        root_lo = (q2 - math.isqrt(disc2)) // (2 * alpha)
        root_hi = (q2 + math.isqrt(disc2)) // (2 * alpha)
        ranges = [(1, min(root_lo + 2, d_hi)), (max(1, root_hi - 2), d_hi)]
    out = []
    seen = set()
    for lo, hi in ranges:
        d = max(1, lo)
        d += (D0 - d) % q
        while d <= hi:
            if d not in seen:
                km = (alpha * d * d - q2 * d + gamma) // (2 * q2)
                kp = (alpha * d * d + q2 * d + gamma) // (2 * q2)
                if kp > r:
                    break
                if km > r // 2:
                    out.append((d, km, kp))
                    seen.add(d)
            d += q
    out.sort()
    return out


def window_cluster(values: Sequence[int], W: int) -> tuple[int, int]:
    """Largest number of values in a window of 2W - 1 consecutive integers, and the window's lower edge."""
    if len(values) == 0:
        return 0, 0
    v = np.sort(np.asarray(values, dtype=np.int64))
    hi = np.searchsorted(v, v + (2 * W - 1), side="left")
    counts = hi - np.arange(v.size)
    i = int(np.argmax(counts))
    return int(counts[i]), int(v[i])


def squarefree_mask(ks: np.ndarray, p_max: int | None = None) -> np.ndarray:
    """Squarefreeness of the given integers by vectorised trial division by p^2 for primes p <= sqrt(max k)."""
    ks = np.asarray(ks, dtype=np.int64)
    if ks.size == 0:
        return np.zeros(0, dtype=bool)
    top = math.isqrt(int(ks.max())) if p_max is None else p_max
    sieve = np.ones(top + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, math.isqrt(top) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    mask = np.ones(ks.size, dtype=bool)
    for p in np.nonzero(sieve)[0]:
        pp = int(p) * int(p)
        mask &= (ks % pp) != 0
    return mask


def squarefree_stats(D: np.ndarray, km: np.ndarray, kp: np.ndarray, W: int) -> tuple[int, int]:
    sf = squarefree_mask(km) & squarefree_mask(kp)
    return int(sf.sum()), window_cluster(D[sf], W)[0]


def prime_stats(D: np.ndarray, km: np.ndarray, kp: np.ndarray, W: int) -> tuple[int, int]:
    pr = np.array([bool(gmpy2.is_prime(int(a))) and bool(gmpy2.is_prime(int(b))) for a, b in zip(km, kp)], dtype=bool)
    return int(pr.sum()), window_cluster(D[pr], W)[0]


def analyse_family(N: int, r: int, W: int, alpha: int, gamma: int, q: int, classes: Sequence[int],
                   want_masks: bool = True, members: list[tuple[int, int, int]] | None = None) -> dict | None:
    if members is None:
        members = []
        for D0 in classes:
            members.extend(family_members(r, alpha, gamma, q, D0))
        members = sorted(set(members))
    if len(members) < 2:
        return None
    ds = np.array([m[0] for m in members], dtype=np.int64)
    km = np.array([m[1] for m in members], dtype=np.int64)
    kp = np.array([m[2] for m in members], dtype=np.int64)
    D = np.array([_ceil_2sqrt(int(b), N) - _ceil_2sqrt(int(a), N) - int(b - a) for a, b in zip(km, kp)], dtype=object)
    base = int(min(D))
    D = np.array([int(x - base) for x in D], dtype=np.int64)
    cl_full, edge = window_cluster(D, W)
    out = {"A": Fraction(alpha, q * q), "C": Fraction(gamma, q * q), "q": q, "classes": list(classes), "alpha": alpha,
           "gamma": gamma, "M": (q ** 4 - 4 * alpha * gamma) // (q * q), "members": len(members), "d_min": int(ds.min()),
           "d_max": int(ds.max()), "cluster_full": cl_full, "_D": D, "_km": km, "_kp": kp}
    # the intrinsic invariant Delta_d = 1/4 - AC = M/(4 q^2) is representation-free; report M for the smallest q at
    # which the family has an integral class (the primitive period of its d-progression)
    out["delta_d"] = Fraction(1, 4) - out["A"] * out["C"]
    sel = (D >= edge) & (D <= edge + 2 * W - 2)
    out["d_cluster_range"] = (int(ds[sel].min()), int(ds[sel].max()))
    out.update({"squarefree_members": None, "cluster_squarefree": None, "prime_members": None, "cluster_prime": None})
    if want_masks:
        out["squarefree_members"], out["cluster_squarefree"] = squarefree_stats(D, km, kp, W)
        out["prime_members"], out["cluster_prime"] = prime_stats(D, km, kp, W)
    # detuning u v'(d) - 1 (analytic derivative) at the median d of the selected cluster; the stationary point is
    # where it vanishes, so a value near 0 means the cluster sits on the resonance
    A, C = float(out["A"]), float(out["C"])
    d0 = float(np.median(ds[sel]))
    s, sp = A * d0 * d0 + C, 2 * A * d0
    u = 2 * math.sqrt(N)
    out["detuning_cluster_median"] = u * ((sp + 1) / (4 * math.sqrt((s + d0) / 2)) - (sp - 1) / (4 * math.sqrt((s - d0) / 2))) - 1
    out["detuning"] = out["detuning_cluster_median"]
    return out


def enumerate_resonant_families(N: int, r: int, q_max: int = 48, M_max: int = 400, tol: tuple[float, float] = (0.7, 1.4),
                                min_population: int = 3, want_masks: bool = True, mask_threshold: int = 4) -> list[dict]:
    """All drift-free families with Delta_d < 0 whose alpha lies in [0.35 tol_lo |M| W_real, tol_hi |M| W_real] for
    some q <= q_max and 1 <= |M| <= M_max, merged by (A, C), with at least min_population members in the shell
    (exact count), analysed on the modulus N.  Sorted by the full-shell cluster.  Complete within the searched box
    (q, |M|) and the tolerance; families outside it are not enumerated.  The squarefree and prime masks are
    evaluated, in decreasing order of the full-shell cluster, for every family whose full cluster is at least
    mask_threshold and beyond that for every family whose full cluster could still beat the running squarefree or
    prime maximum (the full cluster bounds both); families without a computed mask carry None.  The resonance
    condition depends on N and r only through W_real = 0.354 sqrt N r^{-3/2}, so at fixed r/N^{1/3} the list of
    candidate families is the same for every N; only the clusters change."""
    W = lemma_d_window(N, r)
    Wr = w_real(N, r)
    found: dict[tuple[Fraction, Fraction], dict] = {}
    for q in range(1, q_max + 1):
        q2 = q * q
        for Mneg in range(1, M_max + 1):
            M = -Mneg
            n4 = q2 * (q2 - M)
            if n4 % 4:
                continue
            n = n4 // 4
            lo = 0.354 * tol[0] * Mneg * Wr
            hi = tol[1] * Mneg * Wr
            if hi < 1:
                continue
            for alpha in divisors_in_range(n, max(lo, 1.0), hi):
                gamma = n // alpha
                classes = integral_classes(alpha, gamma, q)
                if not classes:
                    continue
                key = (Fraction(alpha, q2), Fraction(gamma, q2))
                rec = found.setdefault(key, {"reps": []})
                rec["reps"].append((q, alpha, gamma, tuple(classes)))
    out = []
    for key, rec in found.items():
        # members are the union over all representations (a representation with period q' = t q lists one class in t)
        members = set()
        for q, alpha, gamma, classes in rec["reps"]:
            for D0 in classes:
                members.update(family_members(r, alpha, gamma, q, D0))
        members = sorted(members)
        if len(members) < min_population:          # exact count, not an estimate
            continue
        q, alpha, gamma, classes = min(rec["reps"], key=lambda t: t[0])   # primitive representation
        z = analyse_family(N, r, W, alpha, gamma, q, classes, want_masks=False, members=members)
        if z is None:
            continue
        z["representations"] = rec["reps"]
        out.append(z)
    out.sort(key=lambda z: -z["cluster_full"])
    if want_masks:
        best_sf = best_pr = 0
        for z in out:
            if z["cluster_full"] >= mask_threshold or z["cluster_full"] > best_sf:
                z["squarefree_members"], z["cluster_squarefree"] = squarefree_stats(z["_D"], z["_km"], z["_kp"], W)
                best_sf = max(best_sf, z["cluster_squarefree"])
            if z["cluster_full"] >= mask_threshold or z["cluster_full"] > best_pr:
                z["prime_members"], z["cluster_prime"] = prime_stats(z["_D"], z["_km"], z["_kp"], W)
                best_pr = max(best_pr, z["cluster_prime"])
    for z in out:
        for k in ("_D", "_km", "_kp"):
            z.pop(k, None)
    return out


def census_point(N: int, r: int, thresholds: Sequence[int] = (4, 6, 8, 10), rel_thresholds: Sequence[float] = (0.25, 0.5), **kw) -> dict:
    n112 = N ** (1 / 12)
    levels = [(f"ge_{t}", t) for t in thresholds] + [(f"ge_{c}N112", c * n112) for c in rel_thresholds]
    derived = max(1, int(math.ceil(min(t for _, t in levels))))
    kw["mask_threshold"] = min(kw.get("mask_threshold", derived), derived)   # never above the lowest requested level
    fams = enumerate_resonant_families(N, r, **kw)
    W = lemma_d_window(N, r)
    R_shell = r - r // 2

    def val(z, key):
        v = z.get(key)
        return 0 if v is None else v

    summary = {"N_bits": N.bit_length(), "r": r, "log_r_over_log_N": math.log(r) / math.log(N), "W": W,
               "W_real": w_real(N, r), "families": len(fams), "shell_cells": R_shell,
               "N_1_12": n112, "N_1_11": N ** (1 / 11), "mask_threshold": kw["mask_threshold"],
               "max_cluster_full": fams[0]["cluster_full"] if fams else 0,
               "max_cluster_squarefree": max((val(z, "cluster_squarefree") for z in fams), default=0),
               "max_cluster_prime": max((val(z, "cluster_prime") for z in fams), default=0)}
    for name, t in levels:
        sel = [z for z in fams if z["cluster_full"] >= t]
        summary[f"families_{name}"] = len(sel)
        summary[f"members_{name}"] = sum(z["members"] for z in sel)
        sel_sf = [z for z in fams if z["cluster_squarefree"] is not None and z["cluster_squarefree"] >= t]
        summary[f"families_sf_{name}"] = len(sel_sf)
        summary[f"members_sf_{name}"] = sum(z["squarefree_members"] for z in sel_sf)
    summary["top"] = [{k: (str(v) if isinstance(v, Fraction) else v) for k, v in z.items() if k != "representations"} for z in fams[:8]]
    return summary


def theorem_w_check(N: int, lam: float = 0.8, count_primes: bool = False) -> dict:
    """The resonance lower bound made rigorous for the full a = 1 shell at r = floor(N^{1/3}) (W = 1), on the family
    (k_-, k_+) = ((d^2 - d + 2)/2, (d^2 + d + 2)/2), q = 1, Delta_d = -7/4.  The smooth phase f(d) = u v(d) - d has
    f'(d) = A d^{-3}(1 - (17/8) d^{-2} + ...) - 1 with A = 7u/(4 sqrt2); the nominal point d_0 = A^{1/3} used to
    centre the window is O(1/d_0) from the exact zero hat d of f', and f''(d) = -3/d_0 (1 + O(L/d_0)) on the window,
    so over |d - d_0| <= L = lam sqrt(d_0) the phase varies by (3/2) lam^2 + o(1) < 1 for lam < sqrt(2/3); the integer
    start differences D(d) lie within one of f(d), hence take at most three values, and one value is shared by at
    least ceil(M/3) of the M >= 2L - 1 members.  In the theorem's regime (window inside the shell, i.e. N beyond
    about 2^46) the conclusions are asserted; otherwise the clipped window is only reported.  With count_primes the
    members of the most frequent value whose two cells are both prime are counted (a lower bound for the prime
    shell's D_max at this r)."""
    assert 0 < lam < math.sqrt(2 / 3), lam
    r = int(gmpy2.iroot(gmpy2.mpz(N), 3)[0])
    W = lemma_d_window(N, r)
    assert W == 1, (N.bit_length(), r, W)
    u = 2 * math.sqrt(N)
    d_star = (7 * u / (4 * math.sqrt(2))) ** (1 / 3)
    L = lam * math.sqrt(d_star)
    lo, hi = int(math.ceil(d_star - L)), int(math.floor(d_star + L))
    vals, ds, clipped = [], [], 0
    for d in range(lo, hi + 1):
        km, kp = (d * d - d + 2) // 2, (d * d + d + 2) // 2
        if not (r // 2 < km and kp <= r):        # the window leaves the shell only for N below ~2^46
            clipped += 1
            continue
        vals.append(_ceil_2sqrt(kp, N) - _ceil_2sqrt(km, N) - d)
        ds.append(d)
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    M = len(vals)
    t_best = max(counts, key=counts.get)
    out = {"N_bits": N.bit_length(), "r": r, "W": W, "d_star": d_star, "window": (lo, hi), "members": M, "clipped": clipped,
           "distinct_values": len(counts), "largest_multiplicity": counts[t_best], "t": t_best,
           "pigeonhole_bound": -(-M // 3), "asymptotic_bound": (2 * L - 1) / 3,
           "asymptotic_bound_over_N112": (2 * L - 1) / 3 / N ** (1 / 12),
           "range_law": 1.94 * 7 ** -0.25 * N ** (1 / 12)}
    if clipped == 0:                               # the theorem's regime: assert its conclusions
        assert M >= 2 * L - 1 and all(v != 0 for v in vals) and len(counts) <= 3, out
        assert counts[t_best] >= -(-M // 3) and counts[t_best] >= (2 * L - 1) / 3, out
    if count_primes:
        pr = 0
        for d, v in zip(ds, vals):
            if v == t_best:
                km, kp = (d * d - d + 2) // 2, (d * d + d + 2) // 2
                if gmpy2.is_prime(km) and gmpy2.is_prime(kp):
                    pr += 1
        out["prime_pairs_at_t"] = pr
        out["prime_pairs_all"] = sum(1 for d in ds if gmpy2.is_prime((d * d - d + 2) // 2) and gmpy2.is_prime((d * d + d + 2) // 2))
        lnr2 = math.log(r) ** 2
        out["bateman_horn_constant"] = out["prime_pairs_all"] * lnr2 / M if M else None
    return out


def bounded_w_family_check(N: int, r: int, n: int, m: int, lam: float = 0.8) -> dict | None:
    """Theorem W' (notes_barrier 7.12): on the family k_-+(t) = n t^2 -+ m t + 1 (delta = 4n - m^2 > 0) the start
    difference D(t) = ceil(u sqrt k_+) - ceil(u sqrt k_-) - 2mt has smooth phase f(t) = u v(t) - 2mt with
    v = (m/sqrt n)(1 - delta/(8 n^2 t^2) + O(t^-4)), stationary at t_*^3 = u delta/(8 n^{5/2}) with f''(t_*) = -6m/t_*;
    the stationary point lies in the shell iff delta/n in (1/(2 W_real), sqrt2/W_real), and then the members with
    |t - t_*| <= lam sqrt(t_*/(3m)) take at most three start differences, one shared by >= ceil(M/3) of them.
    Returns None if the window leaves the shell (N too small for the family); otherwise the counts and the ratio test,
    asserting the theorem's conclusions (at most three values, a value shared by >= ceil(M/3) members, all values > W - 1)."""
    assert 0 < lam < 1 and 4 * n > m * m
    u = 2 * math.sqrt(N)
    W = lemma_d_window(N, r)
    w = math.sqrt(N) / (2 * math.sqrt(2) * r ** 1.5)
    delta = 4 * n - m * m
    t_star = (u * delta / (8 * n ** 2.5)) ** (1 / 3)
    L = lam * math.sqrt(t_star / (3 * m))
    vals = []
    for t in range(int(math.ceil(t_star - L)), int(math.floor(t_star + L)) + 1):
        km, kp = n * t * t - m * t + 1, n * t * t + m * t + 1
        if not (r // 2 < km and kp <= r):
            return None
        vals.append(_ceil_2sqrt(kp, N) - _ceil_2sqrt(km, N) - 2 * m * t)
    if not vals:
        return None
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    M = len(vals)
    assert len(counts) <= 3 and max(counts.values()) >= -(-M // 3) and min(vals) > W - 1, (counts, W)
    return {"N_bits": N.bit_length(), "r": r, "W": W, "W_real": w, "ratio": delta / n, "ratio_interval": (1 / (2 * w), math.sqrt(2) / w),
            "ratio_inside": 1 / (2 * w) < delta / n < math.sqrt(2) / w, "z_over_r": n * t_star ** 2 / r, "members": M,
            "t_window": (int(math.ceil(t_star - L)), int(math.floor(t_star + L))), "min_D": min(vals),
            "distinct_values": len(counts), "largest_multiplicity": max(counts.values()), "pigeonhole_bound": -(-M // 3),
            "per_N112": max(counts.values()) / N ** (1 / 12)}


if __name__ == "__main__":  # python -m factorlab.experiments.resonant_census [--quick] [--bits B] [--relative-only]
    # --relative-only drops the fixed levels 4, 6, 8, 10, so the mask threshold is the lowest relative level
    # (masks are still evaluated below it where a family could affect the squarefree or prime maximum).
    import json
    import os
    import sys

    from ..bench import RESULTS_DIR
    from ..gen import make_semiprime

    quick = "--quick" in sys.argv
    rel_only = "--relative-only" in sys.argv
    one_bits = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else None
    res = []
    if one_bits is not None:
        configs = ((one_bits, 0),)
    else:
        configs = ((40, 0), (48, 0), (56, 0)) if quick else ((40, 0), (40, 1), (48, 0), (48, 1), (56, 0), (64, 0), (72, 0), (80, 0), (96, 0))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_name = "e32_resonant_census" + (f"_{one_bits}" if one_bits is not None else "") + ".json"
    for bits, idx in configs:
        N = int(make_semiprime(bits, "rsa", 7, idx).N)
        for label, r in (("N^(1/3)", int(round(N ** (1 / 3)))), ("1.44 N^(3/11)", int(round(1.44 * N ** (3 / 11)))),
                         ("N^(0.3)", int(round(N ** 0.3)))):
            kw = {"q_max": 32 if quick else 48, "M_max": 200 if quick else 400}
            if rel_only:
                kw["thresholds"] = ()
            z = census_point(N, r, **kw)
            z["label"] = label
            res.append(z)
            ge6 = f">=6: {z['families_ge_6']} fam, {z['members_ge_6']} members of {z['shell_cells']} ({z['members_ge_6'] / z['shell_cells']:.2e}); sf>=6: {z['families_sf_ge_6']} fam, {z['members_sf_ge_6']} members | " if not rel_only else ""
            print(f"{bits}b#{idx} r={label:13s} r={r:10d} W={z['W']:5d} W_real={z['W_real']:.3f} | families={z['families']:4d} | "
                  f"max cluster full/sf/prime = {z['max_cluster_full']:3d}/{z['max_cluster_squarefree']:3d}/{z['max_cluster_prime']:2d} | "
                  f"N^(1/12)={z['N_1_12']:.1f} N^(1/11)={z['N_1_11']:.1f} | {ge6}"
                  f">=0.5N^(1/12): {z['families_ge_0.5N112']} fam, {z['members_ge_0.5N112']} members; >=0.25N^(1/12): {z['families_ge_0.25N112']} fam, {z['members_ge_0.25N112']} members (shell {z['shell_cells']})")
            for t in z["top"][:4]:
                print(f"      A={t['A']} C={t['C']} Delta_d={t['delta_d']} q={t['q']} M={t['M']} members={t['members']} cluster full={t['cluster_full']} sf={t.get('cluster_squarefree')} prime={t.get('cluster_prime')} detuning={t['detuning']:+.3f} d in {t['d_cluster_range']}")
            tmp = os.path.join(RESULTS_DIR, out_name + ".tmp")
            with open(tmp, "w") as fh:     # incremental, atomic: a long run never loses or corrupts its rows
                json.dump(res, fh, indent=1, default=str)
            os.replace(tmp, os.path.join(RESULTS_DIR, out_name))
