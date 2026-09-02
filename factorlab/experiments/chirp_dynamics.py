"""E25: the Lehman chirp as a linear point system, and the average case of hypothesis (S).

For the cells (1, k), r/2 < k <= r, of the Lehman-Harvey family the window starts are
s_k = ceil(2 sqrt(kN)) - N - k, so a start difference is, up to the two ceilings,
    s_k - s_k' = u L - delta,   u = 2 sqrt N,   L = sqrt k - sqrt k',   delta = k - k'.
The r^2/4 ordered pairs are therefore points moving linearly in the single parameter u
(speed L, offset -delta), and D_max(N) of notes_barrier.md section 7.5 is the largest
number of points inside a window of length 2W - 1 at time u(N).

Two pairs crowd *persistently* (for all u) iff they have the same speed; the same-k layer
is speed 0 and the Beatty chain k = j m^2, k' = j (m-1)^2 is speed sqrt j for every m.  For
squarefree k, k', k'', k''' the relation sqrt k - sqrt k' = sqrt k'' - sqrt k''' forces
{k, k'} = {k'', k'''} (linear independence of square roots of distinct squarefree integers
over Q), so all speeds are distinct and clusters are transient: pairs i, j meet only for u in
an interval of length (2W-1)/|L_i - L_j| around (delta_i - delta_j)/(L_i - L_j).

First moment [proven, elementary]: for u uniform on an interval of length U,
    E_u D(t) <= sum_i (2W-1)/(|L_i| U) <= (2W-1) 2 sqrt r / U * sum_{k != k'} 1/|k - k'|
             = O(log r)   for W ~ sqrt N / (2.83 r^{3/2}) and U ~ 0.83 sqrt N.
Higher moments depend on the Diophantine configuration of the meeting times; the module
measures the distribution of D_max over an ensemble of random N (RSA semiprimes, random odd
integers, and synthetic uniform u without any rounding) to test the generic-position
hypothesis behind (S) in the average case.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..gen import make_semiprime
from ..numth import mpz, isqrt
from .lehman_cover import short_window_subfamily, approx_sidon, squarefree_flags, ceil_2sqrt, window_length


def speed_census(r: int, squarefree: bool) -> dict:
    """Exact count of repeated speeds sqrt k - sqrt k' over ordered pairs of the shell (r/2, r].

    Write k = j m^2 with j squarefree.  sqrt k - sqrt k' = m sqrt j - m' sqrt j' is a formal
    Z-combination of the independent radicals sqrt j, so two ordered pairs have exactly the
    same speed iff these formal combinations agree; the census keys each pair by its reduced
    combination.  For squarefree k (m = m' = 1, j != j') a speed determines the pair, so every
    class has size one; repeated classes in the full shell are the Beatty chains (j = j').
    """
    sf = squarefree_flags(r)
    ks = [k for k in range(r // 2 + 1, r + 1) if (not squarefree) or sf[k]]
    # speed sqrt k - sqrt k' = m sqrt j - m' sqrt j' with k = j m^2 (j squarefree part); equal speeds
    # between distinct pairs require equal squarefree parts (j = j'' and j' = j''' or the cross case)
    def core(k):
        m = 1
        while (m + 1) ** 2 <= k:
            m += 1
        for mm in range(m, 0, -1):
            if k % (mm * mm) == 0:
                return k // (mm * mm), mm
        return k, 1
    cores = {k: core(k) for k in ks}
    from collections import Counter
    cnt = Counter()
    for k in ks:
        jk, mk = cores[k]
        for kp in ks:
            if kp == k:
                continue
            jp, mp = cores[kp]
            # canonical exact representation of the speed as a formal combination of radicals
            if jk == jp:
                key = ((jk, mk - mp),)
            else:
                key = tuple(sorted([(jk, mk), (jp, -mp)]))
            cnt[key] += 1
    repeated = {key: c for key, c in cnt.items() if c > 1}
    return {"pairs": len(ks) * (len(ks) - 1), "distinct_speeds": len(cnt),
            "repeated_speed_classes": len(repeated), "pairs_in_repeated_speeds": sum(repeated.values()),
            "largest_speed_class": max(cnt.values()) if cnt else 0}


def _integer_centre_cluster_max(pts: np.ndarray, W: int) -> int:
    """max over integers t != 0 of #{i : |p_i - t| < W} for real points p_i (sorted).

    The count is piecewise constant in t and can only increase when t passes p_i - W, so the
    maximum over integers is attained at some t = ceil(p_i + W) - 1 (the largest integer
    with t < p_i + W, exact also when p_i + W is an integer) or at the boundaries t = +-1 of
    the excluded centre.
    """
    cands = np.ceil(pts + W).astype(np.int64) - 1
    cands = np.concatenate([cands, np.array([1, -1], dtype=np.int64)])
    cands = cands[cands != 0]
    lo = np.searchsorted(pts, cands - W, side="right")
    hi = np.searchsorted(pts, cands + W, side="left")
    return int((hi - lo).max()) if cands.size else 0


def moving_point_cluster_max(N: int, r: int, cells_k: Sequence[int], W: int) -> int:
    """D_max of the rounding-free point system: max over integers t != 0 of the number of
    unrounded points u L_i - delta_i (u = 2 sqrt N, ordered pairs) with |p_i - t| < W -- the
    same centre domain and window as ``approx_sidon`` applied to the exact starts."""
    k = np.array(cells_k, dtype=np.float64)
    u = 2.0 * math.sqrt(float(N))
    L = np.sqrt(k)[:, None] - np.sqrt(k)[None, :]
    D = k[:, None] - k[None, :]
    pts = (u * L - D)[~np.eye(k.size, dtype=bool)]
    pts.sort()
    return _integer_centre_cluster_max(pts, W)


def first_moment_bound(N: int, r: int, cells_k: Sequence[int], W: int, U: float) -> float:
    """sum_i (2W-1)/(|L_i| U) over ordered pairs: an upper bound on E_u D(t) for u uniform on a
    length-U interval (each pair's window condition holds on a u-interval of length (2W-1)/|L_i|)."""
    k = np.array(cells_k, dtype=np.float64)
    L = np.abs(np.sqrt(k)[:, None] - np.sqrt(k)[None, :])[~np.eye(k.size, dtype=bool)]
    return float(((2 * W - 1) / (L * U)).sum())


def d_max_for_N(N: int, r: int, mode: str = "a1sqfree") -> dict:
    sub = short_window_subfamily(N, r, mode=mode)
    W = sub["W_max"]
    sid = approx_sidon(sub["a"], sub["c"], sub["b"], W, N=N)
    mp = moving_point_cluster_max(N, r, sub["b"], W)
    return {"D_max": sid["D_max"], "moving_point_max": mp, "R": sub["R"], "W": W,
            "crowded_fraction": sid["pairs_in_crowded_windows"] / max(1, sid["pairs"])}


def random_odd_integer(rng: np.random.Generator, bits: int) -> int:
    lo, hi = 1 << (bits - 1), 1 << bits
    x = int(rng.integers(lo // 2, hi // 2)) * 2 + 1
    return x


def ensemble(bits: int, e: float, count: int, seed: int = 7) -> dict:
    """D_max over random RSA semiprimes, random odd integers, and synthetic uniform u (no
    rounding) at fixed bit size and r = N^e; the squarefree a = 1 family."""
    rng = np.random.default_rng(seed)
    out = {"bits": bits, "e": e, "count": count, "semiprime": [], "integer": [], "synthetic": [],
           "moving_point_agreement": 0, "moving_point_within_one": 0, "first_moment_bound": None}
    fm = []
    for i in range(count):
        N = int(make_semiprime(bits, "rsa", seed, i).N)
        r = max(4, int(round(float(N) ** e)))
        d = d_max_for_N(N, r)
        out["semiprime"].append(d["D_max"])
        out["moving_point_agreement"] += d["D_max"] == d["moving_point_max"]
        out["moving_point_within_one"] += abs(d["D_max"] - d["moving_point_max"]) <= 1
        # synthetic: u uniform in [2 sqrt(2^{bits-1}), 2 sqrt(2^bits)) with the same cell set and W
        sub = short_window_subfamily(N, r, mode="a1sqfree")
        u = 2.0 * math.sqrt(float(rng.uniform(1 << (bits - 1), 1 << bits)))
        k = np.array(sub["b"], dtype=np.float64)
        pts = ((u * (np.sqrt(k)[:, None] - np.sqrt(k)[None, :])) - (k[:, None] - k[None, :]))[~np.eye(k.size, dtype=bool)]
        pts.sort()
        out["synthetic"].append(_integer_centre_cluster_max(pts, sub["W_max"]))
        U = 2.0 * (math.sqrt(float(1 << bits)) - math.sqrt(float(1 << (bits - 1))))
        fm.append(first_moment_bound(N, r, sub["b"], sub["W_max"], U))
        M = random_odd_integer(rng, bits)
        rM = max(4, int(round(float(M) ** e)))
        out["integer"].append(d_max_for_N(M, rM)["D_max"])
    out["first_moment_bound"] = float(np.mean(fm))
    for key in ("semiprime", "integer", "synthetic"):
        v = np.array(out[key])
        out[key + "_stats"] = {"mean": float(v.mean()), "max": int(v.max()), "min": int(v.min()),
                               "tail": {str(K): int((v >= K).sum()) for K in range(int(v.min()), int(v.max()) + 1)}}
    return out


def chirp_dynamics_experiment(configs=((40, 0.2), (40, 0.25), (48, 0.2), (48, 0.25)), count: int = 200, seed: int = 7) -> dict:
    res = {"speed_census": {}, "ensembles": []}
    for r in (256, 1024, 4096):
        res["speed_census"][str(r)] = {"all_a1": speed_census(r, False), "squarefree": speed_census(r, True)}
    for bits, e in configs:
        res["ensembles"].append(ensemble(bits, e, count, seed))
    return res


if __name__ == "__main__":  # python -m factorlab.experiments.chirp_dynamics
    import json
    import os
    import sys

    from ..bench import RESULTS_DIR

    quick = "--quick" in sys.argv
    res = chirp_dynamics_experiment(count=40 if quick else 200)
    print("== E25: speeds of the chirp point system (ordered pairs of the shell (r/2, r]) ==")
    for r, c in res["speed_census"].items():
        a, s = c["all_a1"], c["squarefree"]
        print(f"  r={r}: all a=1 cells: {a['pairs']} pairs, {a['distinct_speeds']} distinct speeds, {a['repeated_speed_classes']} repeated classes "
              f"covering {a['pairs_in_repeated_speeds']} pairs, largest class {a['largest_speed_class']} | squarefree: {s['pairs']} pairs, "
              f"{s['distinct_speeds']} distinct speeds, repeated classes {s['repeated_speed_classes']}")
    print("== E25: D_max over random N (squarefree a = 1 family) ==")
    for en in res["ensembles"]:
        sp, it, sy = en["semiprime_stats"], en["integer_stats"], en["synthetic_stats"]
        print(f"  {en['bits']} bits r=N^{en['e']}: n={en['count']} | semiprimes mean {sp['mean']:.2f} max {sp['max']} tail {sp['tail']} | "
              f"odd integers mean {it['mean']:.2f} max {it['max']} | synthetic uniform u (no rounding) mean {sy['mean']:.2f} max {sy['max']} | "
              f"moving-point max equals exact D_max in {en['moving_point_agreement']}/{en['count']} (within one: {en['moving_point_within_one']}) | "
              f"first-moment bound on E_u D(t): {en['first_moment_bound']:.2f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e25_chirp_dynamics.json"), "w") as fh:
        json.dump(res, fh, indent=1)
