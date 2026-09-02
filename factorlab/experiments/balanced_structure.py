"""E53: additive structure of the BALANCED sub-family.

The difference-cover floors of the covers paper (Lemma D, Theorem M3) were computed from the additive structure of the a = 1 shell
(1, k), r/2 < k <= r, whose cells cover unbalanced p ~ sqrt(N/k).  A family covering only J_C = [sqrt(N/C), sqrt N] may omit that
shell: its cells are the balanced ones, a <= b <= C a.  This module computes, exactly, the cluster statistic

    D_max(Z) = max_{t != 0} #{(z, z') in Z^2 : z' - z = t}

for the balanced START set  S_bal = { a N + b - ceil(2 sqrt(abN)) : a <= b <= C a, ab <= r }  at r = N^{1/3} (one exponent per cell;
the window offsets i < W_ab only add the trivial runs of consecutive integers, which Lemma D prices through its factor W), for the
a = 1 shell at the same radius, and for random sets of the same size and range (control), together with Lemma D's lower bound
|B| + |G| >= min( n'/(2W), 2 (n'^2 / (4 W D_max))^{1/3} )  for a difference cover of each family (with the family's maximal window
W_ab = max(1, ceil( sqrt N / (4 r sqrt(ab)) ))).  The windowed set's D_max is reported too; it is attained at t = 1 and equals
|Z| - #cells.

Exactness: all exponents and differences are formed in int64.  This is exact for N < 2^50 (then a <= sqrt(r) < 2^9, so every
exponent is below 2^59 and every difference below 2^60); the module refuses larger N.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

import numpy as np
from gmpy2 import iroot, isqrt, mpz

from factorlab.gen import make_semiprime


def ceil_2sqrt(m: int) -> int:
    """ceil(2 sqrt(m)) = ceil(sqrt(4m)) exactly."""
    s = isqrt(4 * m)
    return int(s if s * s == 4 * m else s + 1)


def window(N: int, r: int, ab: int) -> int:
    """Standard window length max(1, ceil(sqrt N / (4 r sqrt(ab)))), computed exactly: ceil(sqrt(N / (16 r^2 ab)))."""
    den = 16 * r * r * ab
    # ceil(sqrt(N/den)) = least w with w^2 den >= N
    w = isqrt(N // den)
    while w * w * den < N:
        w += 1
    return max(1, int(w))


def balanced_cells(N: int, r: int, C: int = 2) -> List[Tuple[int, int]]:
    """Cells (a, b) with a <= b <= C a and ab <= r (the cells whose stationary point sqrt(aN/b) lies in J_C)."""
    cells = []
    a = 1
    while a * a <= r:
        for b in range(a, min(C * a, r // a) + 1):
            cells.append((a, b))
        a += 1
    return cells


N_MAX = 1 << 50  # int64 exactness bound, see the module docstring


def exponent_set(N: int, cells: List[Tuple[int, int]], r: int, with_windows: bool = True) -> np.ndarray:
    """The tested exponents aN + b - ceil(2 sqrt(abN)) - i (0 <= i < W_ab if with_windows, else i = 0 only) as a sorted int64
    array.  Requires N < N_MAX so that int64 arithmetic is exact (asserted)."""
    if N >= N_MAX:
        raise ValueError(f"N must be below 2^50 for exact int64 arithmetic (got {N.bit_length()} bits)")
    vals: List[int] = []
    N_ = mpz(N)
    for a, b in cells:
        e = int(a * N_ + b - ceil_2sqrt(a * b * N_))
        assert e < (1 << 60)
        W = window(N, r, a * b) if with_windows else 1
        vals.extend(e - i for i in range(W))
    assert len(set(vals)) == len(vals), "tested exponents are not distinct"
    return np.array(sorted(vals), dtype=np.int64)


def shell_cells(r: int) -> List[Tuple[int, int]]:
    """The a = 1 shell (1, k), r/2 < k <= r."""
    return [(1, k) for k in range(r // 2 + 1, r + 1)]


DIFF_PAIR_BUDGET = 100_000_000
"""Largest number of pairs n(n-1)/2 for which dmax and dmax_tol materialise every positive difference at once (an int64 array of that
length, sorted).  Above it the differences are scanned in value buckets built by `_difference_buckets`, each holding at most
max(budget, n - 1) differences; dmax_tol's buckets carry in addition the differences within 2W - 2 above their range, which is checked
to be at most the budget as well (see dmax_tol).  Both scans require Z to consist of distinct integers, which every start set of this
module does; the results, including the tie-break, are those of the direct scan."""


def _check_distinct(Z: np.ndarray) -> None:
    if len(Z) > 1 and not bool(np.all(np.diff(np.sort(Z)) > 0)):
        raise ValueError("Z must consist of distinct integers")


def _positive_differences(Z: np.ndarray, chunk: int, lo: int | None = None, hi: int | None = None) -> np.ndarray:
    """All positive differences z' - z of the int64 array Z, optionally restricted to lo <= z' - z < hi, as one array."""
    n = len(Z)
    parts = []
    for s in range(0, n, chunk):
        blk = Z[s : s + chunk]
        d = Z[None, :] - blk[:, None]
        m = d > 0
        if lo is not None:
            m &= d >= lo
        if hi is not None:
            m &= d < hi
        parts.append(d[m])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def _bucket_occupancy(Z: np.ndarray, chunk: int, edges: np.ndarray) -> np.ndarray:
    """The exact number of positive differences of Z in each range [edges[b], edges[b+1]), counted chunk by chunk without
    materialising more than one chunk of differences."""
    n = len(Z)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for s in range(0, n, chunk):
        blk = Z[s : s + chunk]
        d = Z[None, :] - blk[:, None]
        d = d[d > 0]
        idx = np.searchsorted(edges, d, side="right") - 1
        counts += np.bincount(idx, minlength=len(edges) - 1)[: len(edges) - 1]
    return counts


def _difference_buckets(Z: np.ndarray, budget: int, chunk: int = 2000, n_sample: int = 1_000_000, max_rounds: int = 64):
    """Value ranges [edges[b], edges[b+1]) covering every positive difference of Z, each containing at most max(budget, n - 1)
    of them.  The edges start at the quantiles of a sample of `n_sample` random pair differences (about budget/2 differences per
    bucket in expectation); the occupancy of every bucket is then counted exactly, and any bucket over the budget that spans more than
    one value is split at its midpoint, repeatedly.  This terminates: a single difference value t is realised by at most n - 1 pairs
    (z, z + t), so a one-value bucket is always within max(budget, n - 1).  Requires distinct values.  Returns (edges, occupancy)."""
    Zs = np.sort(np.asarray(Z, dtype=np.int64))
    _check_distinct(Zs)
    n = len(Zs)
    span = int(Zs[-1] - Zs[0]) + 1
    n_pairs = n * (n - 1) // 2
    B = max(2, -(-n_pairs // max(1, budget // 2)))
    rs = np.random.default_rng(0)
    i, j = rs.integers(0, n, n_sample), rs.integers(0, n, n_sample)
    samp = np.abs(Zs[i] - Zs[j])
    samp = samp[samp > 0]
    inner = np.floor(np.quantile(samp, np.linspace(0, 1, B + 1)[1:-1])).astype(np.int64) if samp.size else np.zeros(0, np.int64)
    edges = np.unique(np.concatenate(([1], inner[(inner > 1) & (inner <= span)], [span + 1])).astype(np.int64))
    occ = _bucket_occupancy(Zs, chunk, edges)
    for _ in range(max_rounds):
        over = [b for b in range(len(edges) - 1) if occ[b] > budget and edges[b + 1] - edges[b] > 1]
        if not over:
            break
        edges = np.unique(np.concatenate((edges, [(int(edges[b]) + int(edges[b + 1])) // 2 for b in over])).astype(np.int64))
        occ = _bucket_occupancy(Zs, chunk, edges)
    assert int(occ.max()) <= max(budget, n - 1), "bucket splitting did not reach the occupancy bound"
    return edges, occ


def dmax(Z: np.ndarray, chunk: int = 2000, pair_budget: int = DIFF_PAIR_BUDGET) -> Tuple[int, int]:
    """(D_max, t*) : the maximal number of ordered pairs (z, z') with z' - z = t over t > 0, and the smallest maximising t.  Exact;
    differences are formed in chunks and aggregated with numpy, in value buckets (`_difference_buckets`) when the pair count exceeds
    `pair_budget`.  Z must consist of distinct integers."""
    Z = np.asarray(Z, dtype=np.int64)
    _check_distinct(Z)
    n = len(Z)
    n_pairs = n * (n - 1) // 2
    if n_pairs <= pair_budget:
        diffs = []
        for s in range(0, n, chunk):
            blk = Z[s : s + chunk]
            d = Z[None, :] - blk[:, None]  # z' - z for z in blk
            d = d[d > 0]
            diffs.append(d)
        alld = np.concatenate(diffs)
        vals, counts = np.unique(alld, return_counts=True)
        k = int(np.argmax(counts))
        return int(counts[k]), int(vals[k])
    best = (0, 0)
    edges, occ = _difference_buckets(Z, pair_budget, chunk)
    for b in range(len(edges) - 1):
        if occ[b] == 0:
            continue
        d = _positive_differences(Z, chunk, int(edges[b]), int(edges[b + 1]))
        vals, counts = np.unique(d, return_counts=True)
        k = int(np.argmax(counts))
        if int(counts[k]) > best[0]:
            best = (int(counts[k]), int(vals[k]))
    return best


def dmax_tol(Z: np.ndarray, W: int, chunk: int = 2000, pair_budget: int = DIFF_PAIR_BUDGET) -> Tuple[int, int]:
    """Lemma D's statistic: max over t of #{(z, z') : |z' - z - t| < W} over z' > z (exact; W = 1 is dmax).  Returns (count, t) where
    t = v + W - 1 is the centre of the window [v, v + 2W - 2] at the smallest difference v attaining the maximum (for W = 1 the smallest
    maximising t; for W > 1 other centres can attain the same count).  Z must consist of distinct integers.

    Above `pair_budget` pairs the scan runs in the value buckets of `_difference_buckets`; a bucket's array also holds its halo, the
    differences within 2W - 2 above its range, which its windows need.  The halo of every bucket is counted exactly first and must not
    exceed the budget, so at most max(budget, n - 1) + budget differences are ever materialised; a window so wide that a halo alone
    exceeds the budget is refused with ValueError (raise `pair_budget`, or use the direct scan), except that a window covering the whole
    difference span needs no scan: every pair lies in it and the answer is (n(n-1)/2, min difference + W - 1)."""
    Z = np.asarray(Z, dtype=np.int64)
    _check_distinct(Z)
    n = len(Z)
    n_pairs = n * (n - 1) // 2
    ext = 2 * W - 2
    if n_pairs <= pair_budget:
        diffs = []
        for s in range(0, n, chunk):
            blk = Z[s : s + chunk]
            d = Z[None, :] - blk[:, None]
            diffs.append(d[d > 0])
        alld = np.sort(np.concatenate(diffs))
        # for each difference v, count differences in [v, v + 2W - 2] (all within < W of t = v + W - 1)
        hi = np.searchsorted(alld, alld + ext, side="right")
        lo = np.arange(len(alld))
        counts = hi - lo
        k = int(np.argmax(counts))
        return int(counts[k]), int(alld[k] + W - 1)
    Zs = np.sort(Z)
    min_diff = int(np.min(np.diff(Zs)))
    if ext >= int(Zs[-1] - Zs[0]) - min_diff:
        # the window at the smallest difference reaches the largest one: every pair is counted, no scan is needed
        return n_pairs, min_diff + W - 1
    edges, occ = _difference_buckets(Z, pair_budget, chunk)
    # exact halo occupancies: the differences in [hi_b, hi_b + ext) for every bucket b, from one histogram over the refined edges
    fine = np.unique(np.concatenate((edges, np.minimum(edges[1:] + ext, edges[-1]))))
    fine_occ = _bucket_occupancy(Zs, chunk, fine)
    cum = np.concatenate(([0], np.cumsum(fine_occ)))
    best = (0, 0)
    for b in range(len(edges) - 1):
        if occ[b] == 0:
            continue
        lo_v, hi_v = int(edges[b]), int(edges[b + 1])
        i0, i1 = int(np.searchsorted(fine, hi_v)), int(np.searchsorted(fine, min(hi_v + ext, int(edges[-1]))))
        halo = int(cum[i1] - cum[i0])
        if halo > pair_budget:
            raise ValueError(f"tolerance window 2W - 2 = {ext} too wide for the bounded scan at pair_budget = {pair_budget}: a bucket's "
                             f"halo holds {halo} differences; raise pair_budget or use W = 1")
        # the windows starting in [lo_v, hi_v) see differences up to hi_v + ext - 1
        d = np.sort(_positive_differences(Z, chunk, lo_v, hi_v + ext))
        m = int(np.searchsorted(d, hi_v, side="left"))  # the starts: differences below hi_v
        starts = d[:m]
        counts = np.searchsorted(d, starts + ext, side="right") - np.arange(m)
        k = int(np.argmax(counts))
        if int(counts[k]) > best[0]:
            best = (int(counts[k]), int(starts[k] + W - 1))
    return best


def primitive_cells(cells: List[Tuple[int, int]], r: int, ab_min: int = 0) -> List[Tuple[int, int]]:
    """Cells with gcd(a, b) = 1 (no ray of length >= 2 through them) and ab > ab_min."""
    from math import gcd

    return [(a, b) for a, b in cells if gcd(a, b) == 1 and a * b > ab_min]


def cluster_spectrum(Z: np.ndarray, W: int = 1, chunk: int = 2000) -> np.ndarray:
    """The values D(h) = #{(z, z') : z' - z = h}, z' > z, for every positive difference h, each listed twice (for h and -h, since
    D(-h) = D(h) over ordered pairs and Lemma D's Sigma_m sums the m largest values over ALL h != 0), sorted in decreasing order.
    Only the exact statistic (W = 1) is supported: for W > 1 the maximising centres need not be differences, so this enumeration
    would be incomplete; such a W is rejected."""
    if W != 1:
        raise ValueError("cluster_spectrum supports only the exact statistic W = 1")
    Z = np.asarray(Z, dtype=np.int64)
    n = len(Z)
    diffs = []
    for s in range(0, n, chunk):
        blk = Z[s : s + chunk]
        d = Z[None, :] - blk[:, None]
        diffs.append(d[d > 0])
    alld = np.concatenate(diffs)
    _, pos = np.unique(alld, return_counts=True)
    return np.sort(np.repeat(pos, 2))[::-1]


def energy_form_bound(n_windows: int, W: int, spectrum: np.ndarray) -> Tuple[float, int, float]:
    """The energy form of Lemma D.  For a cover with K1 = |B|, K2 = |G| and m = min(K1, K2): K1 K2 >= (n'^2/(4 W Sigma_bar_m))^{2/3}
    (unless K1 or K2 exceeds n'/(2W), which gives |B| + |G| >= n'/(2W) directly), where Sigma_bar_m is the mean of the m largest
    values D(h).  Hence |B| + |G| >= m + (n'^2/(4 W Sigma_bar_m))^{2/3}/m, and since m is not controlled by the analyst the floor
    valid for every cover is the minimum over m >= 1.  Returns (bound, minimising m, Sigma_bar_m at that m)."""
    cum = np.cumsum(spectrum, dtype=np.float64)
    ms = np.arange(1, len(spectrum) + 1, dtype=np.float64)
    sbar = cum / ms
    A = (n_windows ** 2 / (4 * W * sbar)) ** (2 / 3)
    phi = ms + A / ms
    k = int(np.argmin(phi))
    bound = min(n_windows / (2 * W), float(phi[k]))
    return bound, k + 1, float(sbar[k])


def unique_product_cells(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """One cell per product n = ab (the one with the smallest a).  Two divisor pairs of the same n share the twist ceil(2 sqrt(nN)), so
    their starts differ by (a' - a) N + (b' - b) exactly (Lemma (equal products)); keeping one cell per product removes that structure."""
    best: Dict[int, Tuple[int, int]] = {}
    for a, b in cells:
        n = a * b
        if n not in best or a < best[n][0]:
            best[n] = (a, b)
    return sorted(best.values())


def equal_product_pairs(p: int, q: int, m: int, r: int, C: int = 2) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """The pairs (q lam, p (lam + m)) -> (q (lam + m), p lam) of primitive balanced cells with equal products pq lam (lam + m) <= r,
    for coprime q < p < C q.  Their start differences are exactly m (q N - p) for every lam (Lemma (equal products))."""
    from math import gcd

    assert gcd(p, q) == 1 and q < p < C * q
    out = []
    lam = 1
    while p * q * lam * (lam + m) <= r:
        a, b = q * lam, p * (lam + m)
        a2, b2 = q * (lam + m), p * lam
        if a <= b <= C * a and a2 <= b2 <= C * a2 and gcd(a, b) == 1 and gcd(a2, b2) == 1:
            out.append(((a, b), (a2, b2)))
        lam += 1
    return out


def half_offset_family(N: int, r: int, m: int, uniq: set | None = None, C: int = 2) -> Dict:
    """The half-offset family of Lemma (half-offset families): for m = 1 (mod 4), k = (m+1)/2, d = m-1, alpha = (m-1)/4,
    beta = (m-3)/2 (so k beta - d alpha = -1), the cells P = (k lam + alpha, d lam + beta) and P' = (k lam + alpha + 1, d lam + beta + 2)
    with both products <= r.  Returns the family size, the number of lam for which BOTH cells are the unique-product representatives
    (when `uniq`, the set of representatives, is supplied), the maximal exact cluster of the start differences s(P') - s(P) over the
    family and over its top half, and the predicted variation V of the twist increment over the top half."""
    from collections import Counter
    from math import gcd, sqrt

    assert m % 4 == 1 and m >= 5
    k, d, al, be = (m + 1) // 2, m - 1, (m - 1) // 4, (m - 3) // 2
    assert gcd(k, d) == 1 and k * be - d * al == -1 and k * (be + 2) - d * (al + 1) == 1
    Nm = mpz(N)

    def start(a: int, b: int) -> int:
        return int(a * Nm + b - ceil_2sqrt(a * b * Nm))

    lams = []
    lam = 1
    while (k * lam + al + 1) * (d * lam + be + 2) <= r:
        a, b, a2, b2 = k * lam + al, d * lam + be, k * lam + al + 1, d * lam + be + 2
        assert a <= b <= C * a and a2 <= b2 <= C * a2
        lams.append(lam)
        lam += 1
    diffs = {l: start(k * l + al + 1, d * l + be + 2) - start(k * l + al, d * l + be) for l in lams}
    in_uniq = None
    if uniq is not None:
        in_uniq = sum(1 for l in lams if (k * l + al, d * l + be) in uniq and (k * l + al + 1, d * l + be + 2) in uniq)
    top = [l for l in lams if lams and l >= lams[-1] / 2]
    lam1, lam_max = (top[0], lams[-1]) if top else (0, 0)
    V = 0.0
    if lam1 > 0:
        u = 2 * sqrt(N)
        V = (u * m / sqrt(k * d)) * (1 / (8 * (k * d) ** 2)) * (1 / lam1 ** 2 - 1 / lam_max ** 2)
    return {"m": m, "k": k, "d": d, "alpha": al, "beta": be, "family": len(lams), "in_uniq": in_uniq,
            "max_cluster": max(Counter(diffs.values()).values()) if diffs else 0,
            "max_cluster_top_half": max(Counter(diffs[l] for l in top).values()) if top else 0,
            "top_half_values": len({diffs[l] for l in top}), "V_top_half": V, "lam_max": lam_max}


def half_offset_census(N: int, ms=(5, 9, 13, 17, 21, 25), C: int = 2) -> Dict:
    """half_offset_family for each m at r = floor(N^{1/3}), with the unique-product representatives of the balanced primitive family."""
    r = int(iroot(mpz(N), 3)[0])
    uniq = set(unique_product_cells(primitive_cells(balanced_cells(N, r, C), r)))
    return {"N": N, "r": r, "N_pow_1_12": N ** (1 / 12), "rows": [half_offset_family(N, r, m, uniq, C) for m in ms]}


def resonant_cells(r: int, M: float, C: int = 2) -> Dict[Tuple[int, int], float]:
    """Cells lying on a half-offset family whose linear-regime coherent window 4kd/(Delta^2 |X|) is >= M.  A family is a lattice line
    (k lam + alpha, d lam + beta), gcd(k, d) = 1, k <= d <= C k, paired with a fixed shift (k', d'); X = k d' + d k', Y = k d' - d k'
    (both even, nonzero), Delta = |Y|/2, and the half-offset condition k beta - d alpha = -Y/2 (which kills the 1/lam term of the twist
    increment; the 1/lam^2 coefficient is then Delta^2 X / (16 (kd)^{5/2}), so the increment stays within one unit over about
    4kd/(Delta^2 |X|) cells at r = N^{1/3}).  Returns {cell: largest window of a family through it}, over cells and partners that are
    balanced with products <= r.  Directions with kd > 4r/M^2 cannot carry a window >= M with M/2 cells and are skipped."""
    from math import gcd, sqrt

    marked: Dict[Tuple[int, int], float] = {}
    k = 1
    while k * k <= 4 * r / (M * M) + 4:
        for d in range(k, C * k + 1):
            K = k * d
            if gcd(k, d) != 1 or K > 4 * r / (M * M):
                continue
            Xmax = int(4 * K / M) + 1
            for X in range(-Xmax, Xmax + 1):
                if X == 0 or X % 2:
                    continue
                Ymax = int(sqrt(16 * K / (M * abs(X)))) + 1
                for Y in range(-Ymax, Ymax + 1):
                    if Y == 0 or Y % 2 or (X + Y) % (2 * k) or (X - Y) % (2 * d):
                        continue
                    dp, kp = (X + Y) // (2 * k), (X - Y) // (2 * d)
                    window = 4 * K / ((Y / 2) ** 2 * abs(X))
                    if window < M:
                        continue
                    c = -Y // 2  # k beta - d alpha = c
                    al = next(a0 for a0 in range(k) if (c + d * a0) % k == 0)
                    be = (c + d * al) // k
                    lam = -min(al // k, be // d) - 2
                    while True:
                        a, b = k * lam + al, d * lam + be
                        if a > 0 and a * b > r:
                            break
                        if a >= 1 and a <= b <= C * a and a * b <= r:
                            a2, b2 = a + kp, b + dp
                            if a2 >= 1 and a2 <= b2 <= C * a2 and a2 * b2 <= r:
                                marked[(a, b)] = max(marked.get((a, b), 0.0), window)
                                marked[(a2, b2)] = max(marked.get((a2, b2), 0.0), window)
                        lam += 1
        k += 1
    return marked


def excision_census(N: int, Ms=(3, 4, 6, 8), C: int = 2) -> Dict:
    """The thinned statistic: D_max of the unique-product family after removing the cells of resonant_cells(r, M), for each M, with the
    removed fraction and the a-shift and sample pairs of the residual maximiser."""
    r = int(iroot(mpz(N), 3)[0])
    uniq = unique_product_cells(primitive_cells(balanced_cells(N, r, C), r))
    S = exponent_set(N, uniq, r, with_windows=False)
    D0, _ = dmax(S)
    rows = []
    for M in Ms:
        marked = resonant_cells(r, M, C)
        keep = [c for c in uniq if c not in marked]
        S2 = exponent_set(N, keep, r, with_windows=False)
        D2, t2 = dmax(S2)
        # exponent_set sorts its output, so associate starts with cells explicitly
        Nm = mpz(N)
        start_of = {int(a * Nm + b - ceil_2sqrt(a * b * Nm)): (a, b) for a, b in keep}
        pairs = sorted((start_of[z], start_of[z + t2]) for z in start_of if z + t2 in start_of)
        assert len(pairs) == D2
        a_shift = pairs[0][1][0] - pairs[0][0][0] if pairs else None
        rows.append({"M": M, "removed": len(uniq) - len(keep), "removed_fraction": (len(uniq) - len(keep)) / len(uniq), "kept": len(keep),
                     "D_max_thinned": D2, "a_shift": a_shift, "b_shifts": sorted({p[1][1] - p[0][1] for p in pairs}),
                     "lemma_d_starts": lemma_d_bound(len(keep), 1, D2), "sample_pairs": pairs[:6]})
    return {"N": N, "r": r, "cells": len(uniq), "D_max": D0, "lemma_d_starts": lemma_d_bound(len(uniq), 1, D0), "rows": rows}


ENERGY_PAIR_BUDGET = 50_000_000
"""Largest number of pairs n(n-1)/2 for which family_stats builds the per-difference cluster spectrum: the spectrum has one entry per
distinct difference (nearly one per pair) and energy_form_bound takes several float64 copies of it, so beyond a few 10^7 pairs it does
not fit in a few GB.  Above the budget the energy fields are recorded as not computed (None); the exact and tolerance-W statistics,
the Lemma D floors and the controls are unaffected."""


def family_stats(N: int, cells: List[Tuple[int, int]], r: int, rng: random.Random, controls: int, label: str,
                 energy_pair_budget: int = ENERGY_PAIR_BUDGET) -> Dict:
    """Exact and tolerance-W cluster statistics of the start set of a cell family, its Lemma D floor, and random controls.  The energy
    form (cluster spectrum) is computed only when the family's pair count is at most `energy_pair_budget`; otherwise its fields are
    None and `energy_computed` is False."""
    S = exponent_set(N, cells, r, with_windows=False)
    Wmax = max(window(N, r, a * b) for a, b in cells)
    n_prime = sum(window(N, r, a * b) for a, b in cells)
    D1, t1 = dmax(S)
    DW, tW = dmax_tol(S, Wmax)
    ctrl = [dmax_tol(random_control(int(S[0]), int(S[-1]), len(S), rng), Wmax)[0] for _ in range(controls)]
    k_star, t_res = divmod(tW, N)
    if t_res > N // 2:
        k_star, t_res = k_star + 1, t_res - N
    n_pairs = len(S) * (len(S) - 1) // 2
    if n_pairs <= energy_pair_budget:
        spec = cluster_spectrum(S, 1)
        K_e, m_e, sbar_e = energy_form_bound(len(cells), 1, spec)
        spec_top = [int(x) for x in spec[:8]]
        energy_computed = True
    else:
        K_e = m_e = sbar_e = spec_top = None
        energy_computed = False
    return {"label": label, "cells": len(cells), "n_prime": n_prime, "W_max": Wmax, "D_exact": D1, "D_W": DW,
            "t_star": tW, "a_diff": k_star, "t_residual": t_res, "random_D_W": ctrl,
            "lemma_d": lemma_d_bound(n_prime, Wmax, DW), "trivial": 2 * n_prime ** 0.5,
            # singleton starts: keep only the first exponent of every cell (a cover of Z covers this subset), so W = 1
            "lemma_d_starts": lemma_d_bound(len(cells), 1, D1), "trivial_starts": 2 * len(cells) ** 0.5,
            # energy form on the singleton starts: mean of the m largest signed clusters, minimised over the unknown cover
            # parameter m = min(|B|, |G|); None when the pair count exceeds the budget
            "energy_starts": K_e, "energy_m": m_e, "energy_sigma_bar": sbar_e,
            "spectrum_top": spec_top, "energy_pairs": n_pairs, "energy_computed": energy_computed}


def lemma_d_bound(n_windows: int, W: int, D: int) -> float:
    """Lemma D: |B| + |G| >= min(n'/(2W), 2 (n'^2/(4 W D_max))^{1/3})."""
    return min(n_windows / (2 * W), 2 * (n_windows ** 2 / (4 * W * D)) ** (1 / 3))


def random_control(lo: int, hi: int, n: int, rng: random.Random) -> np.ndarray:
    """n distinct integers in [lo, hi]."""
    s = set()
    while len(s) < n:
        s.add(rng.randint(lo, hi))
    return np.array(sorted(s), dtype=np.int64)


def census_point(N: int, C: int = 2, seed: int = 5, controls: int = 3) -> Dict:
    r = int(iroot(mpz(N), 3)[0])
    rng = random.Random(seed)
    cells = balanced_cells(N, r, C)
    prim = primitive_cells(cells, r)
    fams = {
        "balanced": family_stats(N, cells, r, rng, controls, "balanced a<=b<=Ca, ab<=r"),
        "primitive": family_stats(N, prim, r, rng, controls, "balanced, gcd(a,b)=1"),
        "primitive_short": family_stats(N, primitive_cells(cells, r, r // 4), r, rng, controls, "balanced, gcd=1, ab>r/4 (W=1 here)"),
        "unique_product": family_stats(N, unique_product_cells(prim), r, rng, controls, "balanced, gcd=1, one cell per product ab"),
        "shell": family_stats(N, shell_cells(r), r, rng, controls, "a=1 shell r/2<k<=r"),
    }
    Zb = exponent_set(N, cells, r, with_windows=True)
    Dz, tz = dmax(Zb)
    return {"N": N, "r": r, "C": C, "families": fams,
            "balanced_windowed": {"size": int(len(Zb)), "D_max": Dz, "t_star": tz},
            "N_pow": {"1/12": N ** (1 / 12), "1/6": N ** (1 / 6), "1/5": N ** (1 / 5), "2/9": N ** (2 / 9)}}


def unique_product_maximisers(N: int, C: int = 2) -> Dict:
    """Every maximiser of the exact statistic of the unique-product family at r = floor(N^{1/3}): each positive difference t attaining
    D_max(S) for S the family's singleton starts, with the D_max ordered pairs of cells realising it (starts kept attached to their cells,
    since exponent_set sorts) and whether every one of those pairs is a consecutive-cell pair (a, b), (a, b - 1) on the line 2b = 3a + 1
    (the first cell of a pair has the smaller start).  The maximum can be attained by several differences; `n_on_line` counts the
    maximisers all of whose pairs lie on that line and `line_maximiser_exists` says whether at least one does.  Requires N < N_MAX (int64
    exactness of the starts and differences), like every statistic of this module."""
    if N >= N_MAX:
        raise ValueError(f"N must be below 2^50 for exact int64 arithmetic (got {N.bit_length()} bits)")
    if N < 8:
        raise ValueError("N must be at least 8 (r = floor(N^(1/3)) >= 2 is needed for a nonempty family)")
    r = int(iroot(mpz(N), 3)[0])
    cells = unique_product_cells(primitive_cells(balanced_cells(N, r, C), r))
    Nm = mpz(N)
    start_of = {int(a * Nm + b - ceil_2sqrt(a * b * Nm)): (a, b) for a, b in cells}
    assert len(start_of) == len(cells)
    S = np.array(sorted(start_of), dtype=np.int64)
    n = len(S)
    diffs = []
    for s in range(0, n, 2000):
        d = S[None, :] - S[s : s + 2000, None]
        diffs.append(d[d > 0])
    vals, counts = np.unique(np.concatenate(diffs), return_counts=True)
    D = int(counts.max())
    maximisers = []
    for t in (int(v) for v in vals[counts == D]):
        pairs = sorted((start_of[z], start_of[z + t]) for z in start_of if z + t in start_of)
        assert len(pairs) == D
        on_line = all(a1 == a2 and b2 == b1 - 1 and 2 * b1 == 3 * a1 + 1 for (a1, b1), (a2, b2) in pairs)
        maximisers.append({"t": t, "on_line": on_line, "pairs": pairs})
    n_on = sum(1 for m in maximisers if m["on_line"])
    return {"N": N, "bits": N.bit_length(), "r": r, "cells": len(cells), "D": D, "n_maximisers": len(maximisers),
            "n_on_line": n_on, "line_maximiser_exists": n_on > 0, "maximisers": maximisers}


def experiment(bits_list=(32, 36, 40), count: int = 2, seed: int = 5) -> Dict:
    rows = []
    for bits in bits_list:
        for idx in range(count):
            sp = make_semiprime(bits, "rsa", seed, idx)
            rows.append(census_point(int(sp.N), seed=seed + idx))
    return {"bits": list(bits_list), "count": count, "seed": seed, "rows": rows}


def merge_census_archive(existing: Dict | None, res: Dict) -> Dict:
    """Merge the census result `res` (as `experiment` returns it) into `existing` (a previous archive of the same form, or None).  Rows
    are keyed by the modulus N: a rerun of a modulus replaces its row in place, new moduli are appended in order.  The top level keeps
    `bits` as the sorted union, `count` when every run used the same count (otherwise the list of counts, one per run), `seed`, the
    list `runs` of the (bits, count) pairs merged, and `rows`.  A census can thus be assembled from several runs, e.g. two moduli at each
    of several sizes and one at another size."""
    if existing is None or not existing.get("rows"):
        runs = [{"bits": list(res["bits"]), "count": res["count"]}]
        return {"bits": sorted(set(res["bits"])), "count": res["count"], "seed": res["seed"], "runs": runs, "rows": list(res["rows"])}
    if existing.get("seed", res["seed"]) != res["seed"]:
        raise ValueError(f"cannot merge censuses of different seeds ({existing.get('seed')} and {res['seed']})")
    rows = list(existing["rows"])
    index = {int(row["N"]): i for i, row in enumerate(rows)}
    for row in res["rows"]:
        N = int(row["N"])
        if N in index:
            rows[index[N]] = row
        else:
            index[N] = len(rows)
            rows.append(row)
    runs = list(existing.get("runs") or [{"bits": list(existing["bits"]), "count": existing["count"]}])
    runs.append({"bits": list(res["bits"]), "count": res["count"]})
    counts = [run["count"] for run in runs]
    return {"bits": sorted(set(existing["bits"]) | set(res["bits"])), "count": counts[0] if len(set(counts)) == 1 else counts,
            "seed": res["seed"], "runs": runs, "rows": rows}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[32, 36, 40], help="modulus sizes; must be below 50 (int64 exactness)")
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--out", default="results/e53_balanced_structure.json")
    ap.add_argument("--half-offset", action="store_true", help="run the half-offset census (Lemma (half-offset families)) instead")
    ap.add_argument("--excision", action="store_true", help="run the excision census (thinned statistic) instead")
    ap.add_argument("--maximisers", action="store_true", help="enumerate every maximiser of the unique-product family's exact statistic instead")
    ap.add_argument("--moduli", type=int, nargs="*", default=None, help="--maximisers: explicit moduli (default: the --bits x --count moduli of seed 5)")
    args = ap.parse_args()
    if max(args.bits) >= 50:
        raise SystemExit("--bits must be below 50 (int64 exactness bound)")
    if args.maximisers:
        Ns = args.moduli or [int(make_semiprime(b, "rsa", 5, i).N) for b in args.bits for i in range(args.count)]
        bad = [N for N in Ns if N >= N_MAX or N < 8]
        if bad:
            raise SystemExit(f"--moduli must be integers in [8, 2^50) for exact int64 arithmetic; rejected: {bad}")
        out = []
        for N in Ns:
            res = unique_product_maximisers(N)
            out.append(res)
            print(f"N={N} ({res['bits']} bits) r={res['r']} cells={res['cells']} D={res['D']}: {res['n_maximisers']} maximising differences, "
                  f"{res['n_on_line']} with every pair on the line 2b = 3a + 1", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=int)
        raise SystemExit(0)
    if args.excision:
        out = {"bits": list(args.bits), "count": args.count, "rows": []}
        for bits in args.bits:
            for idx in range(args.count):
                res = excision_census(int(make_semiprime(bits, "rsa", 5, idx).N))
                out["rows"].append(res)
                print(f"N~2^{bits} r={res['r']} cells={res['cells']} D_max={res['D_max']} LemmaD={res['lemma_d_starts']:.0f}")
                for row in res["rows"]:
                    print(f"   M={row['M']}: removed {row['removed']} ({100*row['removed_fraction']:.1f}%) D_max(thinned)={row['D_max_thinned']} "
                          f"a_shift={row['a_shift']} b_shifts={row['b_shifts']} LemmaD={row['lemma_d_starts']:.0f}")
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=int)
        raise SystemExit(0)
    if args.half_offset:
        out = {"bits": list(args.bits), "count": args.count, "rows": []}
        for bits in args.bits:
            for idx in range(args.count):
                res = half_offset_census(int(make_semiprime(bits, "rsa", 5, idx).N))
                out["rows"].append(res)
                print(f"N~2^{bits} r={res['r']} N^(1/12)={res['N_pow_1_12']:.1f}")
                for row in res["rows"]:
                    print(f"   m={row['m']:2d} (k,d)=({row['k']:2d},{row['d']:2d}) family={row['family']:3d} in_uniq={row['in_uniq']:3d} "
                          f"max_cluster={row['max_cluster']:2d} top_half: cluster={row['max_cluster_top_half']:2d} values={row['top_half_values']} "
                          f"V={row['V_top_half']:.2f} lam_max={row['lam_max']}")
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=int)
        raise SystemExit(0)
    res = experiment(tuple(args.bits), args.count)
    for row in res["rows"]:
        P = row["N_pow"]
        print(f"N~2^{row['N'].bit_length()} r={row['r']}  N^(1/6)={P['1/6']:.0f} N^(1/5)={P['1/5']:.0f} N^(2/9)={P['2/9']:.0f}")
        for key, f in row["families"].items():
            energy = (f"energy={f['energy_starts']:.1f} (m={f['energy_m']}, Sbar={f['energy_sigma_bar']:.2f}, top {f['spectrum_top']})"
                      if f["energy_computed"] else f"energy not computed ({f['energy_pairs']} pairs exceed the budget)")
            print(f"   {key:16s} cells={f['cells']:6d} n'={f['n_prime']:6d} W={f['W_max']:3d} D_exact={f['D_exact']:4d} D_W={f['D_W']:4d} "
                  f"(random {f['random_D_W']}) a_diff={f['a_diff']} t_res={f['t_residual']}  LemmaD={f['lemma_d']:.1f} trivial={f['trivial']:.1f} "
                  f"| singleton starts: LemmaD={f['lemma_d_starts']:.1f} trivial={f['trivial_starts']:.1f} " + energy)
    import os

    existing = None
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
    merged = merge_census_archive(existing, res)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1, default=int)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {len(merged['rows'])} moduli over bits {merged['bits']} ({len(merged['runs'])} run(s) merged)")
