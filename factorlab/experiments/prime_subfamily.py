"""E30: the prime sub-family of the Lehman-Harvey a = 1 cells and the two-progression hierarchy.

Theorem M3 applies Lemma D to any disjoint sub-family of the a = 1 windows with that sub-family's own
approximate-Sidon statistic, and its exponent (1 - eta)/(5 - 4 eta) depends on the envelope eta of that
statistic.  The squarefree family has eta >= 1/3 (Theorem Q'): the symmetric pairs ((j t^2 - t)/2, (j t^2 + t)/2)
form clusters of size ~ 0.38 r^{1/3}.  Their members t (j t -+ 1)/2 are composite for t >= 3, so the family of
*prime* cells {(1, k) : k prime in (r/2, r]} -- r/(2 ln r) cells, a polylogarithmic thinning -- does not carry them.

What it does carry.  (i) e = 1 chains: a prime k with 8 j k + 1 = M^2 has M = a k +- 1, hence 8 j = a (a k +- 2);
two primes k < k' of the shell on one chain satisfy a^2 k - a'^2 k' = 2 (eps' a' - eps a): either a = a' = 2 and
k' = k + 2 (twin primes, j = (k + 1)/2, the t = 2 symmetric pair) or a > a' with (a/a')^2 < 2, a pair on the line
k' = (a/a')^2 k + O(1) -- along which the speed (a/a' - 1) sqrt k + O(1/sqrt k) changes by >> rho_r per unit of k, so
no two such pairs share a window (Proposition R; brute force below).
(ii) Two-progression pairs (P_1(n), P_2(n)) = (a n^2 + b_1 n + c_1, a n^2 + b_2 n + c_2) with the drift-free
condition 4 a (c_1 - c_2) = b_1^2 - b_2^2 (section 7.8 of notes_barrier.md); in the normal form
(a t^2 - b t + c, a t^2 + b t + c), Delta = b^2 - 4 a c, the speed is (b/sqrt a)(1 + Delta/(8 a^2 t^2) + O(t^-4)), the
pairs of the shell fit in one window of half-width theta rho_r once a >= (|Delta| b sqrt2/(4 theta))^{2/3} r^{1/3}
(the transition), and the occupancy there is 0.29 sqrt(r/a).  In the difference parameter d = k' - k every such
family is a parabola s = k + k' = A d^2 + C with Delta_d = 1/4 - A C = M/(4 q^2) (M an integer, q the period of the
admissible d); with C = 0 the members factor as d (A d -+ 1)/2 and q = 1 gives the symmetric r^{1/3} law; prime
values need a non-square M (so C != 0), and the integrality bound alpha = A q^2 <= q |q^2 - M|/2 caps the
first-order occupancy of such families at ~ 1.2 |M|^{-3/7} r^{2/7} (section 7.10 of notes_barrier.md): the r^{2/7}
law, times the prime-pair density.

This module enumerates two-progression families and their clusters in any masked shell, sweeps the prime shell
against a phase-randomised null, identifies the family behind a maximising window at the pair-set level
(d = k' - k is affine in n, s = k + k' quadratic), peels identified families iteratively, and checks Proposition R
by brute force.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import numpy as np

from .lehman_cover import squarefree_flags
from .sidon_scaling import rho, cluster_max, pair_speed


def prime_flags(n: int) -> np.ndarray:
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, math.isqrt(n) + 1):
        if s[p]:
            s[p * p::p] = False
    return s


def prime_shell(r: int) -> np.ndarray:
    pf = prime_flags(r)
    ks = np.arange(r // 2 + 1, r + 1, dtype=np.int64)
    return ks[pf[ks]]


def d_star_for(ks: np.ndarray, r: int, half_width: float | None = None, block: int = 2048, chunk: int = 1 << 22) -> dict:
    """Cluster maximum of the ordered-pair speeds of an arbitrary set ks of shell members at half-width rho(r)
    (default), with the speeds computed stably as (k' - k)/(sqrt k' + sqrt k) in row blocks.  The speed set is
    symmetric under k <-> k', so the maximum is attained by a negative window and its positive reflection; the
    maximising pairs are recovered with the sweep's own half-open window [L_i, L_i + 2h) on the same floats and
    reported as (min, max), and tau as |tau|.  The R(R-1) speeds are materialised once (8 R^2 bytes), sorted in
    place, and the sliding-window counts are taken in chunks, so the peak memory is that single array plus
    O(block R + chunk) temporaries."""
    h = rho(r) if half_width is None else half_width
    ks = np.asarray(ks, dtype=np.int64)
    R = ks.size
    if R < 2:
        return {"R": int(R), "pairs": 0, "D_star": 0, "tau": 0.0, "tau_sq": 0.0, "pairs_at_max": []}
    s = np.sqrt(ks.astype(np.float64))
    L = np.empty(R * (R - 1), dtype=np.float64)
    pos = 0
    for i0 in range(0, R, block):
        i1 = min(R, i0 + block)
        num = (ks[None, :] - ks[i0:i1, None]).astype(np.float64)
        den = s[None, :] + s[i0:i1, None]
        blk = num / den
        keep = np.arange(R)[None, :] != np.arange(i0, i1)[:, None]
        vals = blk[keep]
        L[pos:pos + vals.size] = vals
        pos += vals.size
        del num, den, blk, keep, vals
    L.sort()
    D, i_best = 0, 0
    for c0 in range(0, L.size, chunk):
        c1 = min(L.size, c0 + chunk)
        hi = np.searchsorted(L, L[c0:c1] + 2 * h, side="left")
        counts = hi - np.arange(c0, c1)
        j = int(np.argmax(counts))
        if int(counts[j]) > D:
            D, i_best = int(counts[j]), c0 + j
        del hi, counts
    lo_edge = float(L[i_best])
    last = int(np.searchsorted(L, lo_edge + 2 * h, side="left")) - 1
    tau = float(0.5 * (L[i_best] + L[last]))
    del L
    # recover the pairs with the sweep's own half-open window [L_i, L_i + 2h), evaluated on the same floats
    pairs = []
    for i0 in range(0, R, block):
        i1 = min(R, i0 + block)
        num = (ks[None, :] - ks[i0:i1, None]).astype(np.float64)
        den = s[None, :] + s[i0:i1, None]
        blk = num / den
        ii, jj = np.nonzero((blk >= lo_edge) & (blk < lo_edge + 2 * h))
        for a, b in zip(ii.tolist(), jj.tolist()):
            if i0 + a != b:
                k1, k2 = int(ks[i0 + a]), int(ks[b])
                pairs.append((min(k1, k2), max(k1, k2)))
    assert len(pairs) == D, (len(pairs), D)
    return {"R": int(R), "pairs": int(R * (R - 1)), "D_star": D, "tau": abs(tau), "tau_sq": tau * tau,
            "pairs_at_max": sorted(pairs)}


def phase_null_for(ks: np.ndarray, r: int, rng: np.random.Generator, block: int = 2048) -> int:
    """The phase-randomised null of E26 for an arbitrary member set: every delta-class (delta = k' - k) is shifted by
    an independent uniform fraction of its minimal spacing |delta|/(4 r^{3/2}).  Computed in row blocks into one
    flat array of the R(R-1) shifted speeds (8 R^2 bytes), the shifts looked up in a table indexed by delta."""
    ks = np.asarray(ks, dtype=np.int64)
    R = ks.size
    if R < 2:
        return 0
    s = np.sqrt(ks.astype(np.float64))
    span = int(ks.max() - ks.min())
    deltas = np.arange(-span, span + 1, dtype=np.int64)
    shift = rng.uniform(0.0, 1.0, size=deltas.size) * (np.abs(deltas) / (4.0 * r ** 1.5))
    L = np.empty(R * (R - 1), dtype=np.float64)
    pos = 0
    for i0 in range(0, R, block):
        i1 = min(R, i0 + block)
        num = ks[None, :] - ks[i0:i1, None]
        blk = num.astype(np.float64) / (s[None, :] + s[i0:i1, None]) + shift[num + span]
        keep = np.arange(R)[None, :] != np.arange(i0, i1)[:, None]
        vals = blk[keep]
        L[pos:pos + vals.size] = vals
        pos += vals.size
    return cluster_max(L, rho(r))[0]


# ---------------------------------------------------------------------------------------------------------------
# two-progression pair families
# ---------------------------------------------------------------------------------------------------------------

def two_progression_members(r: int, a2: int, b2: int, c: int) -> list[tuple[int, int, int]]:
    """Normal-form family P_-+(t) = (a2 t^2 -+ b2 t)/2 + c (a = a2/2, b = b2/2), integer-valued iff a2 = b2 (mod 2):
    the (t, k_-, k_+) with both values in the shell (r/2, r]."""
    if a2 <= 0 or b2 <= 0 or (a2 - b2) % 2:
        raise ValueError("need a2, b2 > 0 of equal parity")
    out = []
    t_hi = math.isqrt(4 * r // a2) + 3
    for t in range(1, t_hi + 1):
        km = (a2 * t * t - b2 * t) // 2 + c
        kp = (a2 * t * t + b2 * t) // 2 + c
        if km <= r // 2 or kp > r:
            continue
        out.append((t, km, kp))
    return out


def two_progression_prediction(r: int, a2: int, b2: int, c: int, theta: float = 1.0) -> dict:
    """Speed b/sqrt a, discriminant Delta = b^2 - 4 a c, transition a* = (|Delta| b sqrt2/(4 theta))^{2/3} r^{1/3},
    occupancy 0.29 sqrt(r/a) if a >= a*, else the linearised top-of-shell capacity 1.41 a/(b |Delta|) pairs."""
    a, b = a2 / 2, b2 / 2
    delta = b * b - 4 * a * c
    a_star = (abs(delta) * b * math.sqrt(2) / (4 * theta)) ** (2 / 3) * r ** (1 / 3) if delta else 0.0
    members = (1 - 1 / math.sqrt(2)) * math.sqrt(r / a)
    cap = 8 * rho(r) * r ** 1.5 * a / (b * abs(delta)) if delta else float("inf")
    return {"a2": a2, "b2": b2, "c": c, "speed": b / math.sqrt(a), "tau_sq": b * b / a, "delta": delta,
            "a_star": a_star, "fits": a >= a_star, "members_predicted": members,
            "occupancy_predicted": min(members, cap)}


def two_progression_cluster(r: int, a2: int, b2: int, c: int, mask: np.ndarray) -> dict:
    """Exact cluster of the family's pairs whose two members both satisfy mask (squarefree or prime), with the
    prediction and the measured speed spread in windows."""
    mem = two_progression_members(r, a2, b2, c)
    speeds_all = [pair_speed(km, kp) for _, km, kp in mem]
    speeds = [pair_speed(km, kp) for _, km, kp in mem if mask[km] and mask[kp]]
    Dm = cluster_max(np.array(speeds), rho(r))[0] if speeds else 0
    z = two_progression_prediction(r, a2, b2, c)
    z.update({"r": r, "members": len(mem), "links": len(speeds), "cluster": Dm,
              "density": len(speeds) / len(mem) if mem else 0.0,
              "spread_over_window": (max(speeds_all) - min(speeds_all)) / (2 * rho(r)) if len(speeds_all) > 1 else 0.0})
    return z


def find_two_progression_families(a2_lo: int, a2_hi: int, c_values: Sequence[int], delta4_values: Sequence[int]) -> list[tuple[int, int, int]]:
    """All (a2, b2, c) with a2 in [a2_lo, a2_hi], c in c_values and 4 Delta = b2^2 - 8 a2 c in delta4_values, b2 > 0 of
    the parity of a2."""
    out = []
    for a2 in range(a2_lo, a2_hi + 1):
        for c in c_values:
            for d4 in delta4_values:
                v = 8 * a2 * c + d4
                if v <= 0:
                    continue
                b2 = math.isqrt(v)
                if b2 * b2 == v and b2 > 0 and (a2 - b2) % 2 == 0:
                    out.append((a2, b2, c))
    return out


def identify_two_progression(pairs: Sequence[tuple[int, int]], min_support: int = 4) -> dict | None:
    """Find the two-progression pair family containing the most of the given ordered pairs (k, k'), k' > k.

    For any pair of quadratic progressions with equal leading coefficient, (P_1(n), P_2(n)) = (a n^2 + b_1 n + c_1,
    a n^2 + b_2 n + c_2) with b_1 != b_2, the sum s = k + k' is an exact quadratic function of the difference
    d = k' - k:  s = A d^2 + B d + C with A = 2a/(b_2 - b_1)^2 > 0, and B = 0 exactly when the pair is drift-free
    (4 a (c_1 - c_2) = b_1^2 - b_2^2, section 7.8 of notes_barrier.md).  Conversely any set of pairs on one such
    quadratic is a two-progression family in the parameter d.  The identification is therefore complete: the
    quadratic through any three pairs of the family (distinct d) is fitted exactly in integer arithmetic
    (Lagrange with a common denominator) and its support among all pairs counted; the best support >= min_support
    wins.  Three pairs always fit, so min_support >= 4 is required.  Returned with the family: the limiting speed
    1/sqrt(2A) (the pairs' speed is (1/sqrt(2A))(1 + Delta_d/(2 A^2 d^2) + O(d^-4)) when B = 0), the discriminant
    Delta_d = 1/4 - A C of the normal form ((s - d)/2, (s + d)/2), and the top-of-shell capacity 1.41 A/|Delta_d|
    consecutive d-values per window; the family's d-values need not be all integers (the consecutive e = 1 links
    of all classes of the chain at j satisfy s = d^2/j + (j^2 - 1)/(4j), A = 1/j, Delta_d = 1/(4 j^2), capacity
    5.66 j, on 2^omega(j) residues of d modulo j; the symmetric pairs have A = j, C = 0, Delta_d = 1/4)."""
    P = sorted(set((min(int(k), int(kp)), max(int(k), int(kp))) for k, kp in pairs if kp != k))
    m = len(P)
    if m < min_support:
        return None
    d = [kp - k for k, kp in P]
    s = [kp + k for k, kp in P]
    # one representative pair per distinct d (pairs with equal d and different s cannot share a quadratic in d)
    by_d: dict[int, list[int]] = {}
    for i in range(m):
        by_d.setdefault(d[i], []).append(i)
    dvals = sorted(by_d)
    if len(dvals) < 3:
        return None
    best = None
    for x in range(len(dvals)):
        for y in range(x + 1, len(dvals)):
            for z in range(y + 1, len(dvals)):
                n1, n2, n3 = dvals[x], dvals[y], dvals[z]
                for i1 in by_d[n1]:
                    for i2 in by_d[n2]:
                        for i3 in by_d[n3]:
                            s1, s2, s3 = s[i1], s[i2], s[i3]
                            d12, d13, d23 = n1 - n2, n1 - n3, n2 - n3
                            Dn = d12 * d13 * d23
                            # leading coefficient A = (s1 d23 - s2 d13 + s3 d12)/Dn must be positive
                            A_num = s1 * d23 - s2 * d13 + s3 * d12
                            if A_num * Dn <= 0:
                                continue
                            support = [i for i in range(m)
                                       if s[i] * Dn == s1 * (d[i] - n2) * (d[i] - n3) * d23
                                       - s2 * (d[i] - n1) * (d[i] - n3) * d13
                                       + s3 * (d[i] - n1) * (d[i] - n2) * d12]
                            if len(support) >= min_support and (best is None or len(support) > best[0]):
                                best = (len(support), support, (n1, s1), (n2, s2), (n3, s3))
    if best is None:
        return None
    sup, support, (n1, s1), (n2, s2), (n3, s3) = best
    A = Fraction(s1 * (n2 - n3) - s2 * (n1 - n3) + s3 * (n1 - n2), (n1 - n2) * (n1 - n3) * (n2 - n3))
    B = Fraction(s1 - s2, n1 - n2) - A * (n1 + n2)
    C = Fraction(s1) - A * n1 * n1 - B * n1
    delta_d = Fraction(1, 4) - A * C
    ds = sorted(d[i] for i in support)
    out = {"support": sup, "of": m, "A": A, "B": B, "C": C, "drift_free": B == 0, "delta_d": delta_d,
           "speed": 1.0 / math.sqrt(2.0 * float(A)), "tau_sq": float(1 / (2 * A)),
           # the expansion and the capacity 1.41 A/|Delta_d| hold for drift-free families only
           "capacity_d": ((8 * 0.17677669529663687 * float(A) / abs(float(delta_d))) if delta_d != 0 else float("inf")) if B == 0 else None,
           "d_min": ds[0], "d_max": ds[-1], "d_density": sup / (ds[-1] - ds[0] + 1),
           "pairs": [P[i] for i in support]}
    return {k: (str(v) if isinstance(v, Fraction) else v) for k, v in out.items()}


def peel(ks: np.ndarray, r: int, rounds: int = 6, min_support: int = 4, stop_at: int | None = None) -> list[dict]:
    """Iteratively: cluster maximum of ks, identify the family behind the maximising window, remove the second
    member of every explained pair, repeat.  Stops when no family with min_support pairs is identified (the
    maximiser is then 'generic') or when D* <= stop_at."""
    ks = np.array(ks, dtype=np.int64)
    out = []
    for it in range(rounds):
        z = d_star_for(ks, r)
        fam = identify_two_progression(z["pairs_at_max"], min_support=min_support)
        row = {"round": it, "R": int(ks.size), "D_star": z["D_star"], "tau_sq": z["tau_sq"],
               "family": None if fam is None else {k: v for k, v in fam.items() if k != "pairs"}}
        out.append(row)
        if fam is None or (stop_at is not None and z["D_star"] <= stop_at):
            break
        drop = set(kp for _, kp in fam["pairs"])
        ks = ks[~np.isin(ks, list(drop))]
        row["removed"] = len(drop)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Proposition R: prime-prime e = 1 chain pairs are twin pairs
# ---------------------------------------------------------------------------------------------------------------

def chain_representations(k: int, a_max: int) -> dict[int, tuple[int, int]]:
    """For a prime k: j -> (a, eps) with 8 j = a (a k + 2 eps), a <= a_max, i.e. the chains 8 j k + 1 = (a k + eps)^2.
    (Every solution has M = +-1 mod k since k is prime.)"""
    reps = {}
    for a in range(1, a_max + 1):
        for eps in (1, -1):
            v = a * (a * k + 2 * eps)
            if v > 0 and v % 8 == 0:
                reps[v // 8] = (a, eps)
    return reps


def prime_chain_pairs_brute(r: int, a_max: int = 64) -> dict:
    """All pairs of distinct primes (k < k') in the shell lying on a common e = 1 chain (some j with 8 j k + 1 and
    8 j k' + 1 both squares), through the M = a k + eps parametrisation with a <= a_max.  Checks Proposition R:
    a^2 k - a'^2 k' = 2 (eps' a' - eps a); a = a' only for twins (a = 2, k' = k + 2); and, for the cluster question,
    that no two distinct chain pairs have speeds within 2 rho_r of each other.  The search is truncated: k' <= r
    gives no bound on a (the relation is of Pell type; k = 3, k' = 5 share the chain j = 7812 with a = 144,
    a' = 112), so the counts are those with a, a' <= a_max."""
    ks = [int(k) for k in prime_shell(r)]
    reps = {k: chain_representations(k, a_max) for k in ks}
    pairs = []
    for i, k in enumerate(ks):
        for kp in ks[i + 1:]:
            common = set(reps[k]) & set(reps[kp])
            for j in sorted(common):
                a, e = reps[k][j]
                ap, ep = reps[kp][j]
                pairs.append({"k": k, "kp": kp, "j": j, "a": a, "a_prime": ap, "eps": e, "eps_prime": ep,
                              "relation": a * a * k - ap * ap * kp == 2 * (ep * ap - e * a),
                              "twin": kp == k + 2})
    relation_ok = all(z["relation"] for z in pairs)
    same_a = [z for z in pairs if z["a"] == z["a_prime"]]
    same_a_are_twins = all(z["twin"] and z["a"] == 2 and z["j"] == (z["k"] + 1) // 2 for z in same_a)
    speeds = sorted({(z["k"], z["kp"]): pair_speed(z["k"], z["kp"]) for z in pairs}.values())
    min_gap = min((b - a for a, b in zip(speeds, speeds[1:])), default=float("inf"))
    return {"r": r, "primes": len(ks), "chain_pairs": len(pairs), "distinct_pairs": len({(z["k"], z["kp"]) for z in pairs}),
            "twin_pairs": sum(1 for z in pairs if z["twin"]), "same_a_pairs": len(same_a),
            "relation_holds_for_all": relation_ok, "same_a_are_twins": same_a_are_twins,
            "a_ratio_max": max((z["a"] / z["a_prime"] for z in pairs), default=0.0),
            "min_speed_gap_over_2rho": min_gap / (2 * rho(r)), "examples": pairs[:6]}


# ---------------------------------------------------------------------------------------------------------------
# offset resonance in the planar regime (with a modulus)
# ---------------------------------------------------------------------------------------------------------------

def offset_resonance_check(N: int, frac: float = 0.87, W: int | None = None) -> dict:
    """The family (((m-1)^2 + 1)/2, ((m+1)^2 + 1)/2), m even (A = 1/4, C = 2, q = 4, Delta_d = -1/4, M = -16), at
    r = frac N^{1/3}: exact start differences D(m) = ceil(2 sqrt(k_+ N)) - ceil(2 sqrt(k_- N)) - (k_+ - k_-) of its
    members in the shell, the largest number of members sharing one value of D and the largest number inside an
    integer window of 2W - 1 consecutive values, against the stationary point of u v(d) - d (v the speed,
    u = 2 sqrt N): d_*^3 = 4 sqrt2 u, m_* = d_*/2, which lies in the shell for r in about [0.63, 1.26] N^{1/3}."""
    from .sidon_scaling import _ceil_2sqrt, lemma_d_window
    r = int(round(frac * N ** (1 / 3)))
    if W is None:
        W = lemma_d_window(N, r)
    u = 2 * math.sqrt(N)
    m_star = (4 * math.sqrt(2) * u) ** (1 / 3) / 2
    k_star = (m_star ** 2 + 1) / 2
    Ds = {}
    m = 2
    while ((m + 1) ** 2 + 1) // 2 <= r:
        km, kp = ((m - 1) ** 2 + 1) // 2, ((m + 1) ** 2 + 1) // 2
        if km > r // 2:
            Ds[m] = _ceil_2sqrt(kp, N) - _ceil_2sqrt(km, N) - (kp - km)
        m += 2
    if not Ds:
        return {"N_bits": N.bit_length(), "r": r, "W": W, "members": 0, "same_D": 0, "window": 0, "m_star": m_star,
                "k_star_in_shell": r // 2 < k_star <= r, "N_1_12": N ** (1 / 12)}
    counts: dict[int, list[int]] = {}
    for mm, v in Ds.items():
        counts.setdefault(v, []).append(mm)
    best_v = max(counts, key=lambda v: len(counts[v]))
    vals = sorted(Ds.values())
    window = max(sum(1 for w in vals if v <= w <= v + 2 * W - 2) for v in vals)
    return {"N_bits": N.bit_length(), "r": r, "W": W, "members": len(Ds), "same_D": len(counts[best_v]),
            "m_range_same_D": (min(counts[best_v]), max(counts[best_v])), "window": window, "m_star": m_star,
            "k_star_in_shell": r // 2 < k_star <= r, "N_1_12": N ** (1 / 12)}


# ---------------------------------------------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------------------------------------------

def families_near_transition(r: int, c_values: Sequence[int], delta4_values: Sequence[int], a2_max: int | None = None,
                             lo: float = 1.0, hi: float = 2.0, per_delta: int = 4) -> list[tuple[int, int, int]]:
    """Two-progression families (a2, b2, c) with 4 Delta in delta4_values whose a = a2/2 lies in [lo a*, hi a*] of their
    own transition a* (so that all pairs of the shell fit in one window and the occupancy is near its maximum)."""
    if a2_max is None:
        a2_max = int(12 * math.sqrt(r))
    out = []
    for d4 in delta4_values:
        found = []
        for (a2, b2, c) in find_two_progression_families(1, a2_max, c_values, (d4,)):
            z = two_progression_prediction(r, a2, b2, c)
            if z["a_star"] > 0 and lo * z["a_star"] <= a2 / 2 <= hi * z["a_star"]:
                found.append((a2, b2, c))
        out.extend(found[:per_delta])
    return out


def prime_subfamily_experiment(rs: Sequence[int] = (4096, 8192, 16384, 32768, 65536, 131072, 262144), null_samples: int = 3,
                               seed: int = 30) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for r in rs:
        ks = prime_shell(r)
        z = d_star_for(ks, r)
        fam = identify_two_progression(z["pairs_at_max"])
        nulls = [phase_null_for(ks, r, rng) for _ in range(null_samples)] if ks.size <= 11000 else []
        rows.append({"r": r, "R": int(ks.size), "pairs": z["pairs"], "D_star": z["D_star"], "tau_sq": z["tau_sq"],
                     "eta_pointwise": math.log(max(z["D_star"], 1)) / math.log(r), "null": nulls,
                     "maximiser_family": None if fam is None else {k: v for k, v in fam.items() if k != "pairs"},
                     "pairs_at_max": z["pairs_at_max"],
                     "r27_over_ln2": r ** (2 / 7) / math.log(r) ** 2})
    return {"rows": rows}


if __name__ == "__main__":  # python -m factorlab.experiments.prime_subfamily [--quick]
    import json
    import os
    import sys

    from ..bench import RESULTS_DIR

    quick = "--quick" in sys.argv
    res = {"proposition_R": [prime_chain_pairs_brute(r) for r in ((512, 2048) if quick else (512, 2048, 8192))]}
    print("== E30: Proposition R (prime-prime e = 1 chain pairs: a^2 k - a'^2 k' = 2(eps' a' - eps a); a = a' only for twins) ==")
    for z in res["proposition_R"]:
        print(f"  r={z['r']}: primes={z['primes']} chain pairs={z['chain_pairs']} twins={z['twin_pairs']} same-a pairs={z['same_a_pairs']} "
              f"relation holds={z['relation_holds_for_all']} same-a are twins={z['same_a_are_twins']} max a/a'={z['a_ratio_max']:.3f} "
              f"min speed gap / 2rho={z['min_speed_gap_over_2rho']:.1f}")
    res["prime_shell"] = prime_subfamily_experiment(rs=(4096, 8192, 16384, 32768) if quick else (4096, 8192, 16384, 32768, 65536, 131072, 262144))
    print("== prime shell: D*(r) against the phase-randomised null; maximiser identification (support >= 4) ==")
    for row in res["prime_shell"]["rows"]:
        f = row["maximiser_family"]
        fs = (f"family A={f['A']} B={f['B']} C={f['C']} Delta_d={f['delta_d']} explains {f['support']}/{f['of']} drift_free={f['drift_free']}") if f else "generic (no family with >= 4 pairs)"
        print(f"  r={row['r']:6d}: R={row['R']:5d} D*={row['D_star']:2d} at tau^2={row['tau_sq']:.5f} null={row['null']} r^(2/7)/ln^2 r={row['r27_over_ln2']:.2f} | {fs}")
    # the general two-progression law with c != 0 near the transition, on the squarefree and on the prime shell
    res["two_progression"] = []
    print("== two-progression families with c != 0 at their transition: squarefree shell (and prime shell) ==")
    for r in ((2 ** 20,) if quick else (2 ** 20, 2 ** 22, 2 ** 24)):
        sf = squarefree_flags(r)
        pf = prime_flags(r)
        for (a2, b2, c) in families_near_transition(r, (1, 2), (1, -7, 17), per_delta=3):
            z = two_progression_cluster(r, a2, b2, c, sf)
            zp = two_progression_cluster(r, a2, b2, c, pf)
            z["prime_links"], z["prime_cluster"] = zp["links"], zp["cluster"]
            res["two_progression"].append(z)
            print(f"  r=2^{int(math.log2(r))} (a2,b2,c)=({a2},{b2},{c}) 4Delta={round(4*z['delta'])}: a={a2/2:.0f} a*={z['a_star']:.0f} fits={z['fits']} spread/window={z['spread_over_window']:.2f} "
                  f"| members={z['members']} (pred {z['members_predicted']:.1f}) squarefree links={z['links']} cluster={z['cluster']} density={z['density']:.2f} "
                  f"| prime links={z['prime_links']} cluster={z['prime_cluster']}")
    # peeling the squarefree shell: is every maximiser a two-progression family until the null is reached?
    res["peel_squarefree"] = {}
    print("== peeling identified families from the squarefree shell ==")
    from .sidon_scaling import shell
    for r, rounds in (((8192, 6),) if quick else ((8192, 40), (16384, 40))):
        rows = peel(shell(r, True), r, rounds=rounds, stop_at=5)
        res["peel_squarefree"][str(r)] = rows
        for z in rows:
            f = z["family"]
            fs = (f"A={f['A']} C={f['C']} Delta_d={f['delta_d']} capacity_d={f['capacity_d'] if f['capacity_d'] is None else round(f['capacity_d'], 1)} "
                  f"d-density={f['d_density']:.3f} drift_free={f['drift_free']} support {f['support']}/{f['of']}") if f else "generic"
            print(f"  r={r} round {z['round']:2d}: R={z['R']} D*={z['D_star']:2d} tau^2={z['tau_sq']:.4f} | {fs} | removed {z.get('removed', 0)}")
        removed = sum(z.get("removed", 0) for z in rows)
        print(f"  r={r}: {len(rows)} rounds, removed {removed} members in total ({removed / rows[0]['R']:.3%}), final D*={rows[-1]['D_star']}")
    # offset resonance in the planar regime, on real moduli
    from ..gen import make_semiprime
    res["offset_resonance"] = []
    print("== offset resonance: the family (((m-1)^2+1)/2, ((m+1)^2+1)/2) near r = 0.87 N^(1/3) on RSA moduli ==")
    for bits, idx in (((64, 0), (72, 0)) if quick else ((64, 0), (64, 1), (72, 0), (80, 0), (96, 0))):
        N = int(make_semiprime(bits, "rsa", 7, idx).N)
        for frac in (0.6, 0.87, 1.0):
            z = offset_resonance_check(N, frac)
            res["offset_resonance"].append(z)
            print(f"  {bits}b idx{idx} r={frac:.2f}N^(1/3) W={z['W']}: members={z['members']} m*={z['m_star']:.0f} (k* in shell {z['k_star_in_shell']}) | "
                  f"same start difference: {z['same_D']} members {z.get('m_range_same_D', '')} | window count {z['window']} | N^(1/12)={z['N_1_12']:.1f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e30_prime_subfamily.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
