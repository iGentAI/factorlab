"""E24: a difference-cover lower bound for oblivious collision algorithms from the
approximate-Sidon structure of the Lehman window starts (notes_barrier.md, section 7.5).

In the explicit-difference-representation model of notes_barrier.md section 7.1 an
algorithm represents every candidate exponent z of its covering family as z = beta - gamma
with beta in a baby set B and gamma in a giant set G, at cost >= |B| + |G|.  Let
Z' = union of R' disjoint integer windows I = [s_I, s_I + W_I), W_I <= W, be the candidates
of a sub-family (a cover of Z covers Z').  Choosing one representation per element gives a
bipartite graph on B and G with n' = sum W_I edges.  A baby beta shared by two edges
(beta, gamma) in window I and (beta, gamma') in window I' is an 'overlap'; Jensen gives at
least n'^2/(2|B|) - n'/2 overlaps.  Overlaps with I = I' number at most sum_I C(W_I, 2)
<= (W - 1) n'/2 (two elements of the same window through the same baby).  An overlap with
I != I' forces |s_I - s_{I'} - (gamma' - gamma)| < W, and a fixed (I, I', gamma, gamma')
admits at most W babies; hence the cross overlaps number at most W |G|^2 D_max with
    D_max := max_{t in Z, t != 0} #{(I, I') : I != I', |s_I - s_{I'} - t| < W}.
If |B| <= n'/(2W) this gives (n'^2/(2|B|) - W n'/2 >= n'^2/(4|B|))
    |B| |G|^2 >= n'^2 / (4 W D_max),
and the same count at the giants (two distinct edges at a common giant have distinct
babies, and a cross-window pair forces |s_I - s_{I'} - (beta - beta')| < W) gives the
inequality with B and G exchanged, so
    |B| + |G| >= min( n'/(2W), 2 (n'^2 / (4 W D_max))^{1/3} ).
For the Lehman-Harvey family at parameter r, three short-window sub-families are used.
The cells with ab in (r/2, r] and all a carry the exact lattice coincidences of cells
sharing k = ab (s_{a,(a+1)t} - s_{a+1,at} = N - t for every admissible a); the cells (1, k),
r/2 < k <= r, carry the Beatty chains c_{j m^2} = ceil(m 2 sqrt(jN)) on square-multiple k,
whose Q - 1 consecutive-square differences have gaps in {1, 2, 3} and hence alone force
D_max >= min(Q - 1, floor((2W - 1)/3)), about min(Q, W) in practice; the cells (1, k) with k
squarefree carry neither (rounding coincidences remain).  If the squarefree family's D_max is
polylogarithmic the bound is ~ N^{1/6} r^{1/6}, which together with the counting bound
~ N^{1/4} r^{-1/4} is minimised at r = N^{1/5}: Harvey's exponent, from below, for explicit
representations of this family.  This module computes D_max exactly for the actual window
starts (pairwise differences grouped by the forced value of a - a', since
|s_I - s_{I'} + (a - a')N| < N/2 and groups are separated by more than a window when
N - 2 max|reduced| > 2W - 2, which is asserted), the resulting bound, the counting bound,
Harvey's exact materialised-giant cost, and a random-start control.  Sub-families larger
than ``max_cells`` are truncated (a1 modes) or restricted to a dyadic a-range (all-cell
mode); the records say so (``truncated``): a truncated sub-family still gives a valid lower
bound but not a statistic of the full family.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..gen import make_semiprime
from ..numth import mpz, isqrt


def lehman_cells(r: int, lo: int = 0, a_range: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Coprime cells (a, b) with lo < ab <= r (and a in [a_range) if given)."""
    out = []
    for a in range(1, r + 1):
        if a_range is not None and not (a_range[0] <= a < a_range[1]):
            continue
        for b in range(max(1, lo // a + 1), r // a + 1):
            if a * b > lo and math.gcd(a, b) == 1:
                out.append((a, b))
    return out


def ceil_2sqrt(k: int, N: int) -> int:
    """ceil(2 sqrt(k N)) exactly."""
    c = int(isqrt(mpz(4 * k * N)))
    if c * c < 4 * k * N:
        c += 1
    return c


def window_length(N: int, r: int, a: int, b: int) -> int:
    """W_ab = ceil(sqrt N / (4 r sqrt(ab))) exactly: least W >= 1 with 16 r^2 ab W^2 >= N."""
    denom = 16 * r * r * a * b
    W = int(isqrt(mpz(N // denom)))
    while denom * W * W < N:
        W += 1
    return max(W, 1)


def window_start(N: int, a: int, b: int) -> int:
    """s_ab = ceil(2 sqrt(abN)) - aN - b: the exponent of the first tested integer of the cell."""
    return ceil_2sqrt(a * b, N) - a * N - b


def family_candidates(N: int, r: int) -> tuple[int, int]:
    """(number of cells, Sigma_W) of the full Lehman-Harvey family at parameter r."""
    cells = lehman_cells(r)
    return len(cells), sum(window_length(N, r, a, b) for a, b in cells)


def harvey_cost(N: int, r: int) -> dict:
    """Materialised-giant cost m + sum ceil(W_i/m) minimised exactly over m (model M1), and 2 sqrt(Sigma_W).

    f(m) >= m, so once f(m0) is known only m <= f(m0) can improve on it; all those m are tried.
    """
    cells = lehman_cells(r)
    Ws = np.array([window_length(N, r, a, b) for a, b in cells], dtype=np.int64)
    S = int(Ws.sum())
    P = int(Ws.size)
    m0 = max(1, int(round(math.sqrt(S))))
    f0 = m0 + int(((Ws + m0 - 1) // m0).sum())
    best = (f0, m0)
    # f(m) = m + sum ceil(W_i/m) >= m + P, so only m <= f_best - P can improve on f_best
    ms = np.arange(1, max(1, best[0] - P) + 1, dtype=np.int64)
    chunk = max(1, (1 << 22) // max(1, P))
    for lo in range(0, ms.size, chunk):
        mm = ms[lo:lo + chunk]
        if mm[0] > best[0] - P:
            break
        costs = mm + ((Ws[None, :] + mm[:, None] - 1) // mm[:, None]).sum(axis=1)
        j = int(np.argmin(costs))
        if int(costs[j]) < best[0]:
            best = (int(costs[j]), int(mm[j]))
    return {"cells": P, "Sigma_W": float(S), "M1_cost": float(best[0]), "m": best[1],
            "counting_bound": 2.0 * math.sqrt(S)}


def approx_sidon(starts_a: Sequence[int], starts_c: Sequence[int], starts_b: Sequence[int], W: int,
                 N: int | None = None) -> dict:
    """D_max = max over integers t != 0 of #{ordered (I, I'), I != I' : |s_I - s_I' - t| < W} for
    s = c - aN - b, computed exactly from the reduced differences (c - c') - (b - b') grouped
    by a - a'.  Differences with different a-differences are at least N - 2 max|reduced|
    apart, so the grouping is exact when N - 2 max|reduced| > 2W - 2 (asserted when N is
    supplied).  Groups are placed on a common axis at spacing S = 2 (max|reduced| + W) + 1,
    so a window never straddles two groups.  For integer t the condition |d - t| < W selects
    the 2W - 1 consecutive integers [t-W+1, t+W-1]; the maximum over t is attained at a
    window whose left end is a difference, or at the boundary t = 1 of the excluded t = 0
    (only the a-difference-0 group can have t = 0); t <= -1 follows from t >= 1 by the
    symmetry d -> -d of ordered pairs.
    Returns D_max, the a-difference and reduced t where it is attained, and the number of
    ordered pairs whose difference has another pair's difference within distance 2W - 2 on
    either side.
    """
    a = np.asarray(starts_a, dtype=np.int64)
    c = np.asarray(starts_c, dtype=np.int64)
    b = np.asarray(starts_b, dtype=np.int64)
    n = a.size
    if n < 2:
        return {"D_max": 0, "pairs": 0}
    # a-priori bound on |reduced difference| so that the group spacing is known before encoding
    M0 = int(c.max() - c.min()) + int(b.max() - b.min())
    if N is not None:
        assert N - 2 * M0 > 2 * W - 2, "a-difference groups are not separated by more than a window"
    S = 2 * (M0 + W) + 1
    assert (int(np.abs(a).max()) * 2 + 1) * S < (1 << 62), "encoding overflow"
    vals = np.empty(n * (n - 1), dtype=np.int64)
    pos = 0
    rows = max(1, (1 << 21) // n)
    for i0 in range(0, n, rows):
        i1 = min(n, i0 + rows)
        dd = (c[i0:i1, None] - c[None, :]) - (b[i0:i1, None] - b[None, :])
        kk = a[i0:i1, None] - a[None, :]
        enc = (kk * S + dd).ravel()
        keep = np.ones((i1 - i0, n), dtype=bool)
        keep[np.arange(i1 - i0), np.arange(i0, i1)] = False  # drop I = I'
        enc = enc[keep.ravel()]
        vals[pos:pos + enc.size] = enc
        pos += enc.size
    assert pos == vals.size
    vals.sort()
    L = 2 * W - 1  # number of integers d with |d - t| < W
    idx = np.arange(vals.size)
    hi = np.searchsorted(vals, vals + L, side="left")
    counts = hi - idx  # differences in [v, v + L), i.e. t = v + W - 1
    # in the a-difference-0 group (encoded value = reduced d, |d| <= M0 < S/2) exclude t <= 0
    group0 = np.abs(vals) <= M0
    counts = np.where(group0 & (vals + W - 1 <= 0), 0, counts)
    i = int(np.argmax(counts))
    D, t_enc = int(counts[i]), int(vals[i] + W - 1)
    # boundary window t = 1 of group 0: d in [2 - W, W]
    c1 = int(np.searchsorted(vals, W + 1, side="left") - np.searchsorted(vals, 2 - W, side="left"))
    if c1 > D:
        D, t_enc = c1, 1
    lo = np.searchsorted(vals, vals - (L - 1), side="left")  # neighbours at distance <= L - 1 = 2W - 2 below
    crowded = int(np.count_nonzero((hi - idx > 1) | (idx - lo > 0)))
    k0 = int(np.floor_divide(t_enc + S // 2, S))
    return {"D_max": D, "at_a_diff": k0, "at_reduced": int(t_enc - k0 * S), "pairs": int(vals.size),
            "pairs_in_crowded_windows": crowded}


def squarefree_flags(limit: int) -> np.ndarray:
    """Boolean array sf[k] for 0 <= k <= limit (sf[0] = False)."""
    sf = np.ones(limit + 1, dtype=bool)
    sf[0] = False
    p = 2
    while p * p <= limit:
        sf[p * p::p * p] = False
        p += 1
    return sf


def short_window_subfamily(N: int, r: int, max_cells: int = 8000, mode: str = "a1") -> dict:
    """Short-window sub-family of the Lehman-Harvey family at r.

    mode 'a1': the cells (1, k), r/2 < k <= r (R = r/2, pure chirp start differences; the
    cells with k = j m^2 form Beatty chains c_{j m^2} = ceil(m 2 sqrt(jN)) whose consecutive
    differences have gaps in {1, 2, 3}, which forces D_max >= min(Q - 1, floor((2W-1)/3)) with
    Q the number of squares in the shell, about min(Q, W) in practice).
    mode 'a1sqfree': the cells (1, k) with k squarefree in (r/2, r] (no Beatty chains and no
    identities among the unrounded radicals; rounding coincidences remain).
    mode 'all': every coprime cell with ab in (r/2, r]; if there are more than max_cells,
    a is restricted to a dyadic range chosen to keep the count below max_cells (any
    disjoint sub-family is admissible for the bound).
    The record reports ``requested_R`` (the size before any truncation) and ``truncated``.
    """
    if mode in ("a1", "a1sqfree"):
        ks = range(r // 2 + 1, r + 1)
        if mode == "a1sqfree":
            sf = squarefree_flags(r)
            ks = [k for k in ks if sf[k]]
        cells = [(1, k) for k in ks]
        requested = len(cells)
        if len(cells) > max_cells:
            cells = cells[:max_cells]
        a_range = (1, 2)
    else:
        cells = lehman_cells(r, lo=r // 2)
        requested = len(cells)
        a_range = None
        if len(cells) > max_cells:
            A = 1
            while True:
                sub = [cd for cd in cells if A <= cd[0] < 2 * A]
                if 0 < len(sub) <= max_cells:
                    cells, a_range = sub, (A, 2 * A)
                    break
                A *= 2
                if A > r:
                    raise RuntimeError("no dyadic a-range small enough")
    a = [cd[0] for cd in cells]
    b = [cd[1] for cd in cells]
    c = [ceil_2sqrt(x * y, N) for x, y in cells]
    Ws = [window_length(N, r, x, y) for x, y in cells]
    return {"cells": cells, "a": a, "b": b, "c": c, "W": Ws, "a_range": a_range, "mode": mode,
            "R": len(cells), "requested_R": requested, "truncated": len(cells) < requested,
            "n": int(sum(Ws)), "W_max": int(max(Ws)), "W_min": int(min(Ws))}


def cover_lower_bound(N: int, r: int, rng: np.random.Generator | None = None, max_cells: int = 8000,
                      mode: str = "a1") -> dict:
    """The difference-cover lower bound of section 7.5 for the Lehman-Harvey family at r,
    from the sub-family selected by ``mode``."""
    N, r = int(N), int(r)
    sub = short_window_subfamily(N, r, max_cells, mode)
    R, n, W = sub["R"], sub["n"], sub["W_max"]
    sid = approx_sidon(sub["a"], sub["c"], sub["b"], W, N=N)
    D = max(1, sid["D_max"])
    lb_sidon = 2.0 * (n * n / (4.0 * W * D)) ** (1.0 / 3.0)
    lb_windows = n / (2.0 * W)  # if |B| or |G| exceeds n/(2W) the cost already exceeds it
    lb = min(lb_windows, lb_sidon)
    hc = harvey_cost(N, r)
    out = {"N": str(N), "r": r, "log2_N": math.log2(N), "log_r_over_log_N": math.log(r) / math.log(N),
           "subfamily": {"mode": mode, "R": R, "requested_R": sub["requested_R"], "truncated": sub["truncated"],
                         "n": n, "W_max": W, "W_min": sub["W_min"], "a_range": sub["a_range"]},
           "D_max": sid["D_max"], "D_max_at": {"a_diff": sid.get("at_a_diff"), "reduced": sid.get("at_reduced")},
           "pairs": sid["pairs"], "pairs_in_crowded_windows": sid.get("pairs_in_crowded_windows", 0),
           "lower_bound": lb, "lower_bound_sidon_branch": lb_sidon, "lower_bound_window_branch": lb_windows,
           "counting_bound_full_family": hc["counting_bound"], "harvey_M1_cost": hc["M1_cost"], "harvey_m": hc["m"],
           "Sigma_W_full": hc["Sigma_W"], "cells_full": hc["cells"],
           "log_bound_over_log_N": math.log(max(lb, hc["counting_bound"])) / math.log(N),
           "log_harvey_over_log_N": math.log(hc["M1_cost"]) / math.log(N)}
    if rng is not None:
        # control: reduced starts c - b replaced by uniform random integers of the same spread (same a, W)
        a = np.asarray(sub["a"], dtype=np.int64)
        x = np.asarray(sub["c"], dtype=np.int64) - np.asarray(sub["b"], dtype=np.int64)
        lo, hi = int(x.min()), int(x.max()) + 1
        x_rand = rng.integers(lo, hi, size=a.size, dtype=np.int64)
        sid_r = approx_sidon(a, x_rand, np.zeros(a.size, dtype=np.int64), W)
        out["control_random_starts_D_max"] = sid_r["D_max"]
    return out


def harvey_cover(N: int, r: int, sub: dict | None = None) -> tuple[list[int], list[int]]:
    """Harvey-type explicit cover of the sub-family: B = [0, m), G = {s_I + j m}; Z' subset of B - G
    is a consequence of the construction (used as a sanity check of the inequality's direction)."""
    sub = sub or short_window_subfamily(N, r, mode="a1")
    m = max(1, int(round(math.sqrt(sub["n"]))))
    B = list(range(m))
    G = set()
    for (a, b), c, W in zip(sub["cells"], sub["c"], sub["W"]):
        s = c - a * N - b
        for j in range(0, W, m):
            G.add(-(s + j))  # z = s + j + j' = beta - gamma with beta = j' in B, gamma = -(s + j)
    return B, sorted(G)


def lehman_cover_experiment(bits: Sequence[int] = (32, 40, 48, 56, 64), exponents: Sequence[float] = (1 / 6, 1 / 5, 2 / 9, 1 / 4, 0.3, 1 / 3),
                            seed: int = 3, max_cells: int = 8000, count: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for nbits in bits:
        for i in range(count):
            N = int(make_semiprime(nbits, "rsa", seed, i).N)
            for e in exponents:
                r = max(4, int(round(float(N) ** e)))
                # skip configurations whose full family is too large to enumerate
                if r * math.log(r + 1) > 3e5:
                    continue
                row = cover_lower_bound(N, r, rng=rng, max_cells=max_cells, mode="a1sqfree")
                row["target_exponent"] = e
                row["instance"] = i
                for other in ("a1", "all"):
                    row[f"{other}_subfamily"] = {k: v for k, v in cover_lower_bound(N, r, rng=rng, max_cells=max_cells, mode=other).items()
                                                 if k in ("subfamily", "D_max", "D_max_at", "pairs", "pairs_in_crowded_windows",
                                                          "lower_bound", "lower_bound_sidon_branch", "lower_bound_window_branch",
                                                          "control_random_starts_D_max")}
                rows.append(row)
    return {"rows": rows}
