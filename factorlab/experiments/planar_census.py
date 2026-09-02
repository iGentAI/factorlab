"""E31: the planar regime with real moduli -- the exact-start statistic of the a = 1 sub-families.

For N^{1/5} << r <= N^{1/3} the offsets k' - k exceed the window and the modulus-free statistic D*(r) of
E26-E30 no longer describes the Lemma D statistic
    D_max = max over integers t != 0 of #{(k, k') : |s_k - s_k' - t| < W},   s_k = ceil(2 sqrt(k N)) - N - k,
of a sub-family of cells (1, k), k in (r/2, r].  Two mechanisms are known there: the planar cap (7.9: a family with
Delta_d > 0 has start differences decreasing by more than one per step, at most 2W/q per window) and offset
resonance (7.10: the start difference D(d) = u v(d) - d of a two-progression family, v(d) = d/(sqrt k' + sqrt k)
with k + k' = A d^2 + B d + C, is stationary where u v'(d) = 1, and all members near the stationary point share one
window).  For drift-free families (B = 0) the condition is u |Delta_d|/(sqrt(2A) A^2 d^3) = 1 and needs
Delta_d < 0; for B != 0 at small d the B/d and C/d^2 terms of v are comparable and the condition can be met by
families whose speeds drift modulus-free -- e.g. (33 d^2 - 3, 33 d^2 + d - 3), d = 13..17, five members with
identical start differences on a 40-bit modulus at r = N^{1/3}.  This module computes D_max exactly for the
squarefree and prime shells at 40-56 bits across r = N^{1/4} .. N^{1/3}, identifies the maximising window's
family by the parabola s = A d^2 + B d + C of Proposition U, reports the family's exact detuning u v'(d) - 1 at
the window (finite differences along the fitted family), peels, and compares with a random-start null and with the
predicted envelopes.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import numpy as np

from .lehman_cover import squarefree_flags
from .prime_subfamily import prime_flags, identify_two_progression, d_star_for
from .sidon_scaling import _ceil_2sqrt, lemma_d_window


def shell_starts(N: int, r: int, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cells (1, k), r/2 < k <= r, with mask[k], and their exact window starts s_k = ceil(2 sqrt(kN)) - N - k
    (Python integers reduced to int64 after subtracting a common offset; only differences are used)."""
    ks = np.arange(r // 2 + 1, r + 1, dtype=np.int64)
    ks = ks[mask[ks]]
    starts = [_ceil_2sqrt(int(k), N) - N - int(k) for k in ks]
    base = min(starts)
    s = np.array([x - base for x in starts], dtype=np.int64)
    return ks, s


def exact_dmax(ks: np.ndarray, s: np.ndarray, W: int, block: int = 2048, chunk: int = 1 << 22) -> dict:
    """Lemma D's statistic for the cells: the largest number of ordered pairs (k < k') whose start differences
    s_k' - s_k lie in one window of 2W - 1 consecutive integers, with the window and its pairs.  Differences are
    taken for k < k' only: the starts increase with k, so every unordered pair has one positive and one negative
    difference, the negative copies populate the window at -t, and the maximum over t of the ordered count equals
    the maximum over the positive differences (not twice it); t > 0 is then automatic.  One maximising window is
    returned (the first in sorted order); ties are not enumerated.  The R(R-1)/2 differences are materialised once
    (int64, 4 R^2 bytes: 1.9 GB at R = 22000), sorted in place, and the sliding-window counts are taken in chunks
    of `chunk` positions, so the peak memory is that single array plus O(block R + chunk) temporaries."""
    R = ks.size
    if R < 2:
        return {"D_max": 0, "t": 0, "pairs": []}
    n_pairs = R * (R - 1) // 2
    diffs = np.empty(n_pairs, dtype=np.int64)
    pos = 0
    for i0 in range(0, R, block):
        i1 = min(R, i0 + block)
        blk = s[None, :] - s[i0:i1, None]                 # s_k' - s_k for rows k in block, all k'
        upper = np.arange(R)[None, :] > np.arange(i0, i1)[:, None]
        vals = blk[upper]
        diffs[pos:pos + vals.size] = vals
        pos += vals.size
        del blk, upper, vals
    diffs.sort()
    D, i_best = 0, 0
    width = 2 * W - 1
    for c0 in range(0, n_pairs, chunk):
        c1 = min(n_pairs, c0 + chunk)
        hi = np.searchsorted(diffs, diffs[c0:c1] + width, side="left")
        counts = hi - np.arange(c0, c1)
        j = int(np.argmax(counts))
        if int(counts[j]) > D:
            D, i_best = int(counts[j]), c0 + j
        del hi, counts
    lo_edge = int(diffs[i_best])
    del diffs
    pairs = []
    for i0 in range(0, R, block):
        i1 = min(R, i0 + block)
        blk = s[None, :] - s[i0:i1, None]
        upper = np.arange(R)[None, :] > np.arange(i0, i1)[:, None]
        sel = upper & (blk >= lo_edge) & (blk <= lo_edge + width - 1)
        ii, jj = np.nonzero(sel)
        for a, b in zip(ii.tolist(), jj.tolist()):
            pairs.append((int(ks[i0 + a]), int(ks[b])))
    assert len(pairs) == D, (len(pairs), D)
    return {"D_max": D, "t": lo_edge + W - 1, "pairs": sorted(pairs)}


def random_start_null(s: np.ndarray, W: int, rng: np.random.Generator) -> int:
    """D_max of as many distinct integer starts drawn uniformly from the span of s (the E24 random-start control;
    distinct so that no zero difference, which Lemma D excludes, can enter the count)."""
    lo, hi = int(s.min()), int(s.max()) + 1
    fake = np.unique(rng.integers(lo, hi, size=s.size, dtype=np.int64))
    while fake.size < s.size:
        fake = np.unique(np.concatenate([fake, rng.integers(lo, hi, size=s.size - fake.size, dtype=np.int64)]))
    return exact_dmax(np.arange(fake.size, dtype=np.int64), fake, W)["D_max"]


def family_detuning(N: int, fam: dict, pairs: Sequence[tuple[int, int]]) -> dict:
    """Detuning of an identified family at its window.  Along the fitted parabola s(d) = A d^2 + B d + C the pair is
    (k, k') = ((s - d)/2, (s + d)/2), its speed is v(d) = d/(sqrt k' + sqrt k) = sqrt k' - sqrt k, and the start
    difference is u v(d) - d + O(1); the slope u v'(d) - 1, with the analytic derivative
        v'(d) = (s'(d) + 1)/(4 sqrt k') - (s'(d) - 1)/(4 sqrt k),   s'(d) = 2 A d + B,
    measures how fast the start differences move per unit of d: near 0 is offset resonance, near -1 the planar-cap
    regime (speed term negligible), large positive the regime where the speed term dominates.  Evaluated in
    floating point from the exact rational fit at the median, smallest and largest d of the pairs."""
    A, B, C = float(Fraction(fam["A"])), float(Fraction(fam["B"])), float(Fraction(fam["C"]))
    ds = sorted(kp - k for k, kp in pairs)
    u = 2 * math.sqrt(N)

    def slope(d: float) -> float:
        s = A * d * d + B * d + C
        sp = 2 * A * d + B
        return u * ((sp + 1) / (4 * math.sqrt((s + d) / 2)) - (sp - 1) / (4 * math.sqrt((s - d) / 2))) - 1

    dd = Fraction(1, 4) - Fraction(fam["A"]) * Fraction(fam["C"])
    return {"u_dv_dd_minus_1": slope(ds[len(ds) // 2]), "d_median": ds[len(ds) // 2],
            "detuning_at_d_min": slope(ds[0]), "detuning_at_d_max": slope(ds[-1]), "d_min": ds[0], "d_max": ds[-1],
            "sign_delta_d": (dd > 0) - (dd < 0)}


def planar_point(N: int, r: int, mask: np.ndarray, rng: np.random.Generator, nulls: int = 2, identify: bool = True) -> dict:
    W = lemma_d_window(N, r)
    ks, s = shell_starts(N, r, mask)
    z = exact_dmax(ks, s, W)
    fam = identify_two_progression(z["pairs"]) if identify else None
    out = {"N_bits": N.bit_length(), "r": r, "log_r_over_log_N": math.log(r) / math.log(N), "R": int(ks.size), "W": W,
           "D_max": z["D_max"], "t": z["t"], "null": [random_start_null(s, W, rng) for _ in range(nulls)],
           "cap_symmetric": min(0.38 * r ** (1 / 3), 2 * W),
           # first-order resonance laws for the best single-class family with |M| = 4 (q = 2, n = 1), top-of-shell tuning
           "resonance_members_limited": 1.17 * r ** 1.25 * N ** -0.25 / 2.0,
           "resonance_range_limited": 1.94 * N ** 0.125 * r ** -0.125 / (2 ** 0.5 * 4 ** 0.25),
           "pairs_at_max": z["pairs"]}
    if fam is not None:
        out["family"] = {k: v for k, v in fam.items() if k != "pairs"}
        out["family"].update(family_detuning(N, fam, fam["pairs"]))
    else:
        out["family"] = None
    return out


def peel_exact(N: int, r: int, mask: np.ndarray, rounds: int = 60, stop_at: int | None = None) -> list[dict]:
    """Peel the exact-start statistic: identify the maximiser's family, remove the second member of each of its pairs,
    repeat until D_max <= stop_at or the maximiser is generic (no family with four pairs).  `rounds` is a safety
    bound only; if it is reached the last row carries exhausted = True."""
    W = lemma_d_window(N, r)
    ks, s = shell_starts(N, r, mask)
    out = []
    for it in range(rounds):
        z = exact_dmax(ks, s, W)
        fam = identify_two_progression(z["pairs"])
        row = {"round": it, "R": int(ks.size), "D_max": z["D_max"], "family": None, "exhausted": False}
        if fam is not None:
            row["family"] = {k: v for k, v in fam.items() if k != "pairs"}
            row["family"].update(family_detuning(N, fam, fam["pairs"]))
        out.append(row)
        if fam is None or (stop_at is not None and z["D_max"] <= stop_at):
            break
        drop = np.isin(ks, [kp for _, kp in fam["pairs"]])
        row["removed"] = int(drop.sum())
        ks, s = ks[~drop], s[~drop]
    else:
        out[-1]["exhausted"] = True
    return out


def planar_census(N: int, exponents: Sequence[float], masks: dict[str, np.ndarray], r_cap: int, seed: int = 31) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for name, mask in masks.items():
        for e in exponents:
            r = int(round(N ** e))
            if r > r_cap:
                continue
            R = int(mask[r // 2 + 1: r + 1].sum())
            if R > 22000:
                continue
            z = planar_point(N, r, mask, rng)
            z["family_name"] = name
            if R <= 14000:       # the modulus-free comparison materialises 8 R^2 bytes
                free = d_star_for(np.arange(r // 2 + 1, r + 1, dtype=np.int64)[mask[r // 2 + 1: r + 1]], r)
                z["D_star_free"] = free["D_star"]
            else:
                z["D_star_free"] = None
            rows.append(z)
    return rows


if __name__ == "__main__":  # python -m factorlab.experiments.planar_census [--quick]
    import json
    import os
    import sys

    from ..bench import RESULTS_DIR
    from ..gen import make_semiprime

    quick = "--quick" in sys.argv
    exps = (0.25, 3 / 11, 0.3, 1 / 3)
    res = {"points": [], "peel": []}
    configs = ((40, 0), (48, 0)) if quick else ((40, 0), (40, 1), (48, 0), (48, 1), (56, 0))   # two moduli at 40 and 48 bits, one at 56
    for bits, idx in configs:
        N = int(make_semiprime(bits, "rsa", 7, idx).N)
        r_max = int(round(N ** (1 / 3)))
        sf = squarefree_flags(r_max + 1)
        pf = prime_flags(r_max + 1)
        masks = {"squarefree": sf, "prime": pf} if bits <= 48 else {"prime": pf}
        rows = planar_census(N, exps, masks, r_cap=r_max)
        res["points"].extend(rows)
        print(f"== {bits}-bit modulus #{idx}: exact D_max of the a = 1 sub-families in the planar regime ==")
        for z in rows:
            f = z["family"]
            fs = (f"family A={f['A']} B={f['B']} C={f['C']} Delta_d={f['delta_d']} drift_free={f['drift_free']} support {f['support']}/{f['of']} "
                  f"detuning u v' - 1 = {f['u_dv_dd_minus_1']:+.3f} at d={f['d_median']} (range {f['detuning_at_d_min']:+.3f}..{f['detuning_at_d_max']:+.3f} over d {f['d_min']}..{f['d_max']})") if f else "generic (no family with >= 4 pairs)"
            print(f"  {z['family_name']:10s} r=N^{z['log_r_over_log_N']:.3f} r={z['r']:7d} R={z['R']:6d} W={z['W']:5d} | D_max={z['D_max']:3d} null={z['null']} free D*={z['D_star_free']} | "
                  f"cap={z['cap_symmetric']:.1f} res(members)={z['resonance_members_limited']:.1f} res(range)={z['resonance_range_limited']:.1f} | {fs}")
    # peeling the exact statistic of the squarefree shell at r = N^{0.3} and N^{1/3}, two moduli per size
    print("== peeling the exact statistic (squarefree shell) until the null ==")
    peel_configs = ((40, 0),) if quick else ((40, 0), (40, 1), (48, 0), (48, 1))
    for bits, idx in peel_configs:
        N = int(make_semiprime(bits, "rsa", 7, idx).N)
        r_max = int(round(N ** (1 / 3)))
        sf = squarefree_flags(r_max + 1)
        for e in (0.3, 1 / 3):
            r = int(round(N ** e))
            rng = np.random.default_rng(31)
            W = lemma_d_window(N, r)
            _, s0 = shell_starts(N, r, sf)
            null = max(random_start_null(s0, W, rng) for _ in range(2))
            rows = peel_exact(N, r, sf, rounds=4 if quick else 60, stop_at=null)
            res["peel"].append({"N_bits": bits, "idx": idx, "r": r, "exponent": e, "W": W, "null": null, "rows": rows})
            for z in rows:
                f = z["family"]
                fs = (f"A={f['A']} B={f['B']} C={f['C']} Delta_d={f['delta_d']} drift_free={f['drift_free']} detuning={f['u_dv_dd_minus_1']:+.3f} "
                      f"(range {f['detuning_at_d_min']:+.3f}..{f['detuning_at_d_max']:+.3f}) d_med={f['d_median']} support {f['support']}/{f['of']}") if f else "generic"
                print(f"  {bits}b#{idx} r=N^{e:.3f} W={W} null={null} round {z['round']:2d}: R={z['R']} D_max={z['D_max']:2d} | {fs} | removed {z.get('removed', 0)}")
            removed = sum(z.get("removed", 0) for z in rows)
            tag = " [ROUND BOUND REACHED]" if rows[-1]["exhausted"] else ""
            print(f"  {bits}b#{idx} r=N^{e:.3f}: {len(rows)} rounds, removed {removed} of {rows[0]['R']} ({removed / rows[0]['R']:.2%}), final D_max={rows[-1]['D_max']} (null {null}){tag}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e31_planar_census.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
