"""E46: sampling the difference-cover avenue directly.

Inside the model, an N^{1/6} algorithm needs an explicit representation  Z' subset B - G  of the tested exponents with
|B| + |G| = N^{1/6+o(1)}; Lemma D bounds |B| + |G| from below through the approximate-Sidon statistic D_max, and
Theorem M3 turns that into the certificate.  Nothing in those theorems says what covers are actually ACHIEVABLE.
This module searches for them: for the a = 1 shell, whose shifted exponent starts are s_k = ceil(2 sqrt(kN)) - N - k for
r/2 < k <= r (the set Z' of Lemma D; the constant -N is immaterial, the term -k is not), with windows of length W, we
minimise  W |B'| + |G|  over offset sets B' (containing 0) and giant sets G with S subset B' + G, by greedy set cover
for G given B' and a local search over B' among the frequent start differences.  The search is a heuristic: it returns
UPPER bounds on the minimal cover, to be compared with the trivial cover (B' = {0}, G = S; cost W + |S|), with Lemma D's
lower bound, and with the same search run on a random set of the same size in the same range (control).

Run:  python -m factorlab.experiments.cover_search --bits 21 24 27 30 33 36 --out results/e46_cover_search.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from gmpy2 import iroot, isqrt, mpz

from factorlab.gen import make_semiprime


def ceil_2sqrt(k: int, N: int) -> int:
    """ceil(2 sqrt(k N)) exactly."""
    v = 4 * k * N
    s = int(isqrt(mpz(v)))
    return s if s * s == v else s + 1


def lehman_starts(N: int, r: int) -> List[int]:
    """Shifted exponent starts s_k = ceil(2 sqrt(kN)) - N - k of the a = 1 shell cells k in (r/2, r].

    These are the starts of the windows of Lemma D (the exponents U - aN - b with U the first tested integer); the
    twists ceil(2 sqrt(kN)) alone differ from them by the k-dependent term -k, which changes the difference structure."""
    return [ceil_2sqrt(k, N) - N - k for k in range(r // 2 + 1, r + 1)]


def difference_multiplicities(S: Sequence[int], W: int = 1) -> Counter:
    """D(t) = #{(i != j) : |s_j - s_i - t| < W} for positive t, computed as multiplicities of exact differences and then
    smeared over the window (W = 1: exact multiplicities)."""
    arr = np.array(sorted(S), dtype=np.int64)
    n = len(arr)
    if n < 2:
        return Counter()
    diffs = (arr[None, :] - arr[:, None])
    pos = diffs[diffs > 0]
    c = Counter(pos.tolist())
    if W == 1:
        return c
    out: Counter = Counter()
    for t, m in c.items():
        for u in range(t - W + 1, t + W):
            if u > 0:
                out[u] += m
    return out


def lemma_d_lower_bound(n_prime: int, W: int, d_max: int) -> float:
    """Lemma D: |B| + |G| >= min(n'/(2W), 2 (n'^2 / (4 W D_max))^{1/3})."""
    d = max(d_max, 1)
    return min(n_prime / (2 * W), 2 * (n_prime ** 2 / (4 * W * d)) ** (1 / 3))


def greedy_G(S: Sequence[int], B: Sequence[int]) -> List[int]:
    """Greedy set cover: smallest G found such that S subset B + G (each s = b + g with b in B).
    Lazy-greedy: gains only decrease, so a max-heap with re-evaluation on pop is exact greedy."""
    import heapq

    Bl = list(B)
    cover: Dict[int, Set[int]] = {}
    for s in S:
        for b in Bl:
            cover.setdefault(s - b, set()).add(s)
    uncovered: Set[int] = set(S)
    heap = [(-len(cs), g) for g, cs in cover.items()]
    heapq.heapify(heap)
    G: List[int] = []
    while uncovered:
        neg_gain, g = heapq.heappop(heap)
        gain = len(cover[g] & uncovered)
        if gain != -neg_gain:
            if gain > 0:
                heapq.heappush(heap, (-gain, g))
            continue
        G.append(g)
        uncovered -= cover[g]
    return G


def cover_cost(B: Sequence[int], G: Sequence[int], W: int) -> int:
    return W * len(B) + len(G)


def verify_cover(S: Sequence[int], B: Sequence[int], G: Sequence[int]) -> bool:
    Bs = set(B)
    Gs = set(G)
    return all(any((s - b) in Gs for b in Bs) for s in S)


def local_search(S: Sequence[int], W: int, candidates: Sequence[int], max_offsets: int = 12, verbose: bool = False) -> Tuple[List[int], List[int], int]:
    """Start from B = {0}, G = S. Repeatedly add the candidate offset that lowers W|B| + |G| the most (recomputing G
    greedily), or remove an offset if that lowers the cost; stop at a local minimum or at max_offsets offsets."""
    B: List[int] = [0]
    G = list(S)
    best = cover_cost(B, G, W)
    improved = True
    while improved:
        improved = False
        trial_best: Optional[Tuple[int, List[int], List[int]]] = None
        if len(B) < max_offsets:
            for t in candidates:
                if t in B:
                    continue
                B2 = B + [t]
                G2 = greedy_G(S, B2)
                c = cover_cost(B2, G2, W)
                if c < best and (trial_best is None or c < trial_best[0]):
                    trial_best = (c, B2, G2)
        for t in B[1:]:
            B2 = [b for b in B if b != t]
            G2 = greedy_G(S, B2)
            c = cover_cost(B2, G2, W)
            if c < best and (trial_best is None or c < trial_best[0]):
                trial_best = (c, B2, G2)
        if trial_best is not None:
            best, B, G = trial_best
            improved = True
            if verbose:
                print(f"  cost {best}: |B|={len(B)} |G|={len(G)}")
    return B, G, best


def random_control(lo: int, hi: int, n: int, rng: np.random.Generator) -> List[int]:
    """n distinct integers in [lo, hi], uniformly without replacement; the large-range branch samples until n distinct."""
    if hi - lo + 1 < n:
        raise ValueError("range too small for n distinct values")
    if hi - lo + 1 < 5_000_000:
        R = sorted(int(x) for x in rng.choice(np.arange(lo, hi + 1), size=n, replace=False))
    else:
        seen: Set[int] = set()
        while len(seen) < n:
            for x in rng.integers(lo, hi + 1, size=max(16, n - len(seen))):
                seen.add(int(x))
                if len(seen) == n:
                    break
        R = sorted(seen)
    assert len(R) == n and len(set(R)) == n
    return R


def cover_point(N: int, r: int, W: int, top_candidates: int = 25, max_offsets: int = 12, seed: int = 0) -> Dict:
    N = int(N)
    S = lehman_starts(N, r)
    n = len(S)
    mult = difference_multiplicities(S, W)
    d_max = max(mult.values()) if mult else 0
    cands = [t for t, _ in mult.most_common(top_candidates)]
    B, G, cost = local_search(S, W, cands, max_offsets=max_offsets)
    assert verify_cover(S, B, G)
    trivial = W + n
    lb = lemma_d_lower_bound(n * W, W, d_max)
    # control: random distinct integers in the same range, same size
    rng = np.random.default_rng(seed)
    lo, hi = min(S), max(S)
    R = random_control(lo, hi, n, rng)
    multR = difference_multiplicities(R, W)
    d_maxR = max(multR.values()) if multR else 0
    candsR = [t for t, _ in multR.most_common(top_candidates)]
    BR, GR, costR = local_search(R, W, candsR, max_offsets=max_offsets)
    assert verify_cover(R, BR, GR)
    return {
        "N": N, "r": r, "W": W, "shell_size": n, "D_max": d_max, "trivial_cost": trivial,
        "lemma_D_lower_bound": lb, "best_cost": cost, "best_over_trivial": cost / trivial, "best_over_lower_bound": cost / lb,
        "B_offsets": len(B), "G_size": len(G), "B": sorted(B),
        "control": {"D_max": d_maxR, "best_cost": costR, "best_over_trivial": costR / trivial, "B_offsets": len(BR), "G_size": len(GR)},
    }


def cover_experiment(bits_list: Sequence[int], regimes: Sequence[str] = ("third",), seed: int = 7, top_candidates: int = 25, max_offsets: int = 12) -> List[Dict]:
    rows = []
    for bits in bits_list:
        sp = make_semiprime(bits, "rsa", seed, 0)
        N = int(sp.N)
        for regime in regimes:
            if regime == "third":
                r = int(iroot(mpz(N), 3)[0]); W = 1
            elif regime == "quarter":
                r = int(iroot(mpz(N), 4)[0]); W = max(1, math.ceil(math.sqrt(N) / (4 * r ** 1.5)))
            else:
                raise ValueError(regime)
            row = cover_point(N, r, W, top_candidates=top_candidates, max_offsets=max_offsets)
            row["bits"], row["regime"] = bits, regime
            rows.append(row)
            print(f"{bits} bits, r={r}, W={W}, |S|={row['shell_size']}, D_max={row['D_max']}: trivial {row['trivial_cost']}, "
                  f"best {row['best_cost']} (ratio {row['best_over_trivial']:.3f}, |B'|={row['B_offsets']}), Lemma D lb {row['lemma_D_lower_bound']:.1f}; "
                  f"control best {row['control']['best_cost']} (ratio {row['control']['best_over_trivial']:.3f}, D_max {row['control']['D_max']})", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, nargs="+", default=[21, 24, 27, 30, 33])
    ap.add_argument("--regimes", nargs="+", default=["third"])
    ap.add_argument("--top-candidates", type=int, default=25)
    ap.add_argument("--max-offsets", type=int, default=12)
    ap.add_argument("--out", default="results/e46_cover_search.json")
    args = ap.parse_args()
    rows = cover_experiment(args.bits, args.regimes, top_candidates=args.top_candidates, max_offsets=args.max_offsets)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
