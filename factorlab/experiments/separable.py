"""E1: can the Lehman window offsets be made *separable*?

A Lehman-type family has, for each cell (a, b) with ab <= r, a window of
width w_ab = sqrt(N) / (4 r sqrt(ab)) that must sit near the minimum
c_ab = 2 sqrt(abN) of g(p) = aN/p + bp to obtain the square-root coverage gain
(Lemma A of docs/notes_barrier.md).  Separable offsets A(a) + B(b) would allow
an O~(sqrt(P)) difference cover of the candidate set and hence an N^{1/6}
algorithm (notes, section 7).  Cell (a, b) is K-aligned if

    |A(a) + B(b) - c_ab| <= K w_ab .

With x_a = A(a) and y_b = -B(b) this is the pair of difference constraints

    x_a - y_b <= c_ab + K w_ab ,      y_b - x_a <= -(c_ab - K w_ab),

feasible iff the constraint graph has no negative cycle.  We insert cells
greedily (by decreasing Lemma-A coverage bound) and keep a cell iff the system
stays feasible.

Incremental feasibility (Ramalingam-Reps style).  We maintain a feasible
potential pi (pi[v] <= pi[u] + c for every kept constraint x_v - x_u <= c).
Adding (u, v, c) with pi[v] > pi[u] + c forces the decrease
dec_v = pi[v] - (pi[u] + c) > 0 at v.  Along an existing edge x -> y with cost
c_xy the induced decrease is dec_y = dec_x - (pi[x] + c_xy - pi[y]) where the
reduced cost pi[x] + c_xy - pi[y] >= 0, so decreases are non-increasing along
paths and the largest decrease must be settled first (Dijkstra on reduced
costs).  If u itself would be decreased, the new edge is violated again and a
negative cycle through it exists: the constraint is rejected.

Outputs: |G| (aligned cells kept), the forest bound #a + #b - 1, whether G is
K_{2,2}-free, and the *exact* measure of the union of the p-intervals covered
by the kept cells under their actual separable windows
[x_a - y_b, x_a - y_b + w_ab), as a share of L = |J|.  For reference the same
union measure is reported for (i) the kept cells with Lehman windows
[c_ab, c_ab + w_ab) and (ii) all considered cells with Lehman windows.

Decision rule (notes, section 9): |G| - (#a + #b - 1) = O(1) and aligned
coverage share -> 0 as r grows supports Conjecture F (separable offsets cost
N^{1/4}); a large feasible G whose separable-window coverage share stays
bounded below would be a signal for N^{1/6}.

Feasibility decisions are exact for the rationalised window endpoints (square
roots replaced by 200-bit rational approximations, error < 2^-200, negligible
against window widths >= 1/4); coverage measures use doubles on
cancellation-free formulas.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from fractions import Fraction

from ..numth import mpz, isqrt


def _sqrt_frac(n: int, prec_bits: int = 200) -> Fraction:
    """Rational approximation of sqrt(n) with error < 2^-prec_bits."""
    s = isqrt(mpz(n) << (2 * prec_bits))
    return Fraction(int(s), 1 << prec_bits)


class DifferenceConstraints:
    """Incremental feasibility of a system of constraints x_v - x_u <= c."""

    def __init__(self) -> None:
        self.adj: dict = defaultdict(list)  # u -> list of (v, c)
        self.pi: dict = {}

    def _ensure(self, node) -> None:
        if node not in self.pi:
            self.pi[node] = Fraction(0)

    def try_add(self, u, v, c: Fraction) -> bool:
        """Try to add x_v - x_u <= c.  Returns True and keeps it if feasible."""
        self._ensure(u)
        self._ensure(v)
        if u == v:
            # x_u - x_u <= c  is feasible iff c >= 0 (and then vacuous)
            return c >= 0
        pi = self.pi
        if pi[v] <= pi[u] + c:
            self.adj[u].append((v, c))
            return True
        new_pi = {v: pi[u] + c}
        # max-heap on the decrease dec[x] = pi[x] - new_pi[x]
        heap = [(-(pi[v] - new_pi[v]), v)]
        settled = set()
        while heap:
            negdec, x = heapq.heappop(heap)
            dec = -negdec
            if x in settled or dec != pi[x] - new_pi[x]:
                continue  # stale entry
            settled.add(x)
            for y, cxy in self.adj[x]:
                cand = new_pi[x] + cxy
                cur = new_pi.get(y, pi[y])
                if cand < cur:
                    if y == u:
                        return False  # u decreased -> new edge violated -> negative cycle
                    new_pi[y] = cand
                    heapq.heappush(heap, (-(pi[y] - cand), y))
        for x, val in new_pi.items():
            pi[x] = val
        self.adj[u].append((v, c))
        return True


def lehman_cells(N, r: int):
    """Yield (a, b, c_ab (Fraction), w_ab (Fraction), coverage_bound (float))."""
    sqrtN = _sqrt_frac(int(N))
    sqrtN_f = math.sqrt(float(N))
    for a in range(1, r + 1):
        for b in range(1, r // a + 1):
            c_ab = 2 * _sqrt_frac(a * b * int(N))
            w_ab = sqrtN / (4 * r * _sqrt_frac(a * b))
            cov = 2 * math.sqrt(float(w_ab) * sqrtN_f / a)
            yield a, b, c_ab, w_ab, cov


def window_preimage(N, a: int, b: int, delta_lo: float, width: float, p_lo: float, p_hi: float):
    """Intervals of p in [p_lo, p_hi] with g(p) = aN/p + bp in [c_min + delta_lo, c_min + delta_lo + width),
    c_min = 2 sqrt(abN).  Returns a list of (x0, x1)."""
    Nf = float(N)
    cmin = 2 * math.sqrt(a * b * Nf)

    def roots(delta):
        if delta <= 0:
            return None
        s = math.sqrt(delta * (2 * cmin + delta))
        t = cmin + delta
        return (2 * a * Nf / (t + s), (t + s) / (2 * b))  # (p-, p+)

    outer = roots(delta_lo + width)
    if outer is None:
        return []
    inner = roots(delta_lo)
    pieces = [outer] if inner is None else [(outer[0], inner[0]), (inner[1], outer[1])]
    out = []
    for x0, x1 in pieces:
        x0, x1 = max(x0, p_lo), min(x1, p_hi)
        if x1 > x0:
            out.append((x0, x1))
    return out


def union_measure(intervals) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


def greedy_separable_alignment(N, r: int, K: float = 1.0, C: float = 4.0,
                               order: str = "coverage") -> dict:
    """Greedy maximal K-aligned separable subfamily of the Lehman cells with
    critical point in [sqrt(N/C), sqrt N] (i.e. a <= b <= C a)."""
    cells = [c for c in lehman_cells(N, r) if c[0] <= c[1] <= C * c[0]]
    if order == "coverage":
        cells.sort(key=lambda t: -t[4])
    elif order == "ab":
        cells.sort(key=lambda t: (t[0] * t[1], t[0]))
    sysd = DifferenceConstraints()
    kept = []
    Kf = Fraction(K).limit_denominator(10**6)
    for a, b, c_ab, w_ab, cov in cells:
        hi = c_ab + Kf * w_ab
        lo = c_ab - Kf * w_ab
        ua, vb = ("a", a), ("b", b)
        snap_pi = dict(sysd.pi)
        snap_len = {k: len(v) for k, v in sysd.adj.items()}
        # x_a - y_b <= hi  (edge vb -> ua, weight hi);  y_b - x_a <= -lo  (edge ua -> vb, weight -lo)
        ok = sysd.try_add(vb, ua, hi) and sysd.try_add(ua, vb, -lo)
        if ok:
            kept.append((a, b, c_ab, w_ab, cov))
        else:
            sysd.pi = snap_pi
            for k in list(sysd.adj.keys()):
                n0 = snap_len.get(k, 0)
                del sysd.adj[k][n0:]
    A_vals = {a for a, *_ in kept}
    B_vals = {b for _, b, *_ in kept}
    forest_bound = len(A_vals) + len(B_vals) - 1 if kept else 0
    by_a = defaultdict(set)
    for a, b, *_ in kept:
        by_a[a].add(b)
    a_list = sorted(by_a)
    rect_free = True
    for i in range(len(a_list)):
        for j in range(i + 1, len(a_list)):
            if len(by_a[a_list[i]] & by_a[a_list[j]]) >= 2:
                rect_free = False
                break
        if not rect_free:
            break
    # acyclicity of the bipartite incidence graph (union-find)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    is_forest = True
    for a, b, *_ in kept:
        ra, rb = find(("a", a)), find(("b", b))
        if ra == rb:
            is_forest = False
            break
        parent[ra] = rb
    # exact coverage measures
    p_hi = math.sqrt(float(N))
    p_lo = p_hi / math.sqrt(C)
    L = p_hi - p_lo
    sep_intervals, lehman_kept_intervals, lehman_all_intervals = [], [], []
    max_abs_offset_over_w = 0.0
    for a, b, c_ab, w_ab, _ in kept:
        offset = sysd.pi[("a", a)] - sysd.pi[("b", b)]
        delta = float(offset - c_ab)
        max_abs_offset_over_w = max(max_abs_offset_over_w, abs(delta) / float(w_ab))
        sep_intervals += window_preimage(N, a, b, delta, float(w_ab), p_lo, p_hi)
        lehman_kept_intervals += window_preimage(N, a, b, 0.0, float(w_ab), p_lo, p_hi)
    for a, b, c_ab, w_ab, _ in cells:
        lehman_all_intervals += window_preimage(N, a, b, 0.0, float(w_ab), p_lo, p_hi)
    return {
        "r": r, "K": K, "C": C, "cells_considered": len(cells),
        "kept": len(kept), "distinct_a": len(A_vals), "distinct_b": len(B_vals),
        "forest_bound": forest_bound, "excess_over_forest": len(kept) - forest_bound,
        "K22_free": rect_free,
        "is_forest": is_forest,
        "max_offset_deviation_in_w": max_abs_offset_over_w,
        "sep_window_coverage_share": union_measure(sep_intervals) / L,
        "lehman_window_coverage_share_kept": union_measure(lehman_kept_intervals) / L,
        "lehman_window_coverage_share_all": union_measure(lehman_all_intervals) / L,
        "kept_cells_sample": [(a, b) for a, b, *_ in kept[:12]],
    }


# ---------------------------------------------------------------------------
# E2: linear complexity of the chirp sequence
# ---------------------------------------------------------------------------

def berlekamp_massey(seq: list[int], p: int) -> int:
    """Linear complexity of ``seq`` over F_p (length of shortest LFSR)."""
    n = len(seq)
    C = [1] + [0] * n
    B = [1] + [0] * n
    L, m, b = 0, 1, 1
    for i in range(n):
        d = seq[i]
        for j in range(1, L + 1):
            d = (d + C[j] * seq[i - j]) % p
        if d == 0:
            m += 1
            continue
        coef = d * pow(b, p - 2, p) % p
        T = C[:]
        for j in range(m, n + 1):
            C[j] = (C[j] - coef * B[j - m]) % p
        if 2 * L <= i:
            L, B, b, m = i + 1 - L, T, d, 1
        else:
            m += 1
    return L


def chirp_linear_complexity(N, r: int, ell: int, alpha: int = 3) -> dict:
    """Linear complexity over F_ell of s_k = alpha^{ceil(2 sqrt(kN))} mod ell,
    k = 1..r.  This tests the Prony / sparse-interpolation route only: a low
    value would mean the sequence satisfies a short linear recurrence.

    Controls: (i) a sum of two geometric sequences (LFSR order 2) -- genuinely
    low complexity; (ii) an exact quadratic chirp alpha^{7k^2+3k+1}, which is
    Bluestein-exploitable but is *not* an LFSR sequence and is expected to have
    generic complexity ~ r/2.  The comparison shows that linear complexity does
    not detect chirp structure; the chirp-hull / second-difference analysis
    (barrier.py) is the test for that.
    """
    N = mpz(N)
    seq = []
    for k in range(1, r + 1):
        c = isqrt(4 * k * N)
        if c * c < 4 * k * N:
            c += 1
        seq.append(pow(alpha, int(c) % (ell - 1), ell))
    Lc = berlekamp_massey(seq, ell)
    low = [(pow(alpha, k, ell) + pow(5, k, ell)) % ell for k in range(1, r + 1)]
    Llow = berlekamp_massey(low, ell)
    chirp = [pow(alpha, (7 * k * k + 3 * k + 1) % (ell - 1), ell) for k in range(1, r + 1)]
    Lchirp = berlekamp_massey(chirp, ell)
    return {"r": r, "ell": ell, "linear_complexity": Lc, "generic_expectation": r / 2,
            "ratio": Lc / (r / 2), "control_lfsr_order2_complexity": Llow,
            "control_exact_quadratic_chirp_complexity": Lchirp}
