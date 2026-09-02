"""E37: memory-lean exact computation of the two approximate-Sidon statistics of the a = 1 shell.

Modulus-free (E26's statistic, Section 4 of the paper):

    D*_theta(r) = max_tau #{(k, k') : k < k' in the shell, |sqrt(k') - sqrt(k) - tau| < theta * rho_r},
    rho_r = 1 / (4 sqrt 2 r^{3/2})   (Harvey's resolution for the cell (1, r/2)).

Exact-start (Lemma D's statistic with a modulus, E31):

    D_max = max_{t != 0} #{(k, k') : k < k', |s_k' - s_k - t| < W},   s_k = ceil(2 sqrt(kN)) - N - k,

i.e. the largest number of start differences in a window of 2W - 1 consecutive integers.  E26 and
E31 materialised all R^2 pair values and were limited to r ~ 2^16 and 48 bits on a 6 GB machine.
Here the values are streamed once into a coarse histogram with about 4e8 bins (mean occupancy
~ 2), so that a bin pair with a large count is structured rather than a Poisson fluctuation, and
only candidate bins are refined exactly.

Exactness contract.  A window meets at most two adjacent coarse bins (the bin width exceeds the
window span), so its count is at most hist[b] + hist[b+1] for some b.  The refinement descends
through the adjacent-bin sums level by level from the largest value, refining every bin of a level
exactly (all pairs in those bins are re-enumerated and the sliding window is computed on them);
once a level T has been completed with best >= T, every unrefined bin has sum <= T - 1 < best, so
the maximum is certified and `exact` is True.  If the refinement budget is exhausted first -- which
happens only when the true maximum is small compared with the mean occupancy, e.g. the prime
shell -- a collective refinement marks every bin adjacent to a sum >= T_c, where T_c is the
smallest threshold in [best + 1, T] whose marked bins hold at most `collect_budget` pairs, streams
the values a second time keeping only those in marked bins, sorts them and sweeps exactly.  Any
window with count >= T_c lies entirely in marked bins, so a maximum >= T_c found this way is
exact, and otherwise D <= T_c - 1 is certified; the result then carries `exact` = False and the
bounds D_star <= D <= upper.  Speeds are computed as (k' - k)/(sqrt k' + sqrt k), accurate to
relative 1e-16 (the naive difference of square roots loses the window at r ~ 2^25).  The
maximising window's pairs are identified as a two-progression family with the exact parabola fit
of E30.

Run:  python -m factorlab.experiments.sidon_bucketed --r 65536 131072 --prime-r 524288 [--theta 1.0]
      python -m factorlab.experiments.sidon_bucketed --planar 56 --planar-r 399153 55095 --kinds squarefree full
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np


def rho(r: int) -> float:
    return 1.0 / (4.0 * math.sqrt(2.0) * r ** 1.5)


def _primes_upto(n: int) -> np.ndarray:
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.nonzero(sieve)[0]


def squarefree_mask(r: int) -> np.ndarray:
    """Boolean array m of length r + 1 with m[k] iff k is squarefree (m[0] = False)."""
    m = np.ones(r + 1, dtype=bool)
    m[0] = False
    for p in _primes_upto(int(math.isqrt(r))):
        m[::int(p) * int(p)] = False
    return m


def prime_mask(r: int) -> np.ndarray:
    m = np.zeros(r + 1, dtype=bool)
    m[_primes_upto(r)] = True
    return m


def squarefree_shell(r: int) -> np.ndarray:
    """Squarefree integers in (r/2, r]."""
    ks = np.arange(r // 2 + 1, r + 1, dtype=np.int64)
    return ks[squarefree_mask(r)[ks]]


def prime_shell(r: int) -> np.ndarray:
    """Primes in (r/2, r]."""
    ps = _primes_upto(r)
    return ps[ps > r // 2].astype(np.int64)


def _range_pairs(key: np.ndarray, lo, hi):
    """All index pairs (i, j), i < j, with key[j] - key[i] in [lo, hi], for a sorted key array."""
    R = len(key)
    jlo = np.searchsorted(key, key + lo, side="left")
    jhi = np.searchsorted(key, key + hi, side="right")
    jlo = np.maximum(jlo, np.arange(R) + 1)
    cnt = np.maximum(jhi - jlo, 0)
    total = int(cnt.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ii = np.repeat(np.arange(R), cnt)
    offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    jj = np.repeat(jlo, cnt) + offs
    return ii, jj


def _window_max(vs: np.ndarray, span, side: str):
    """Largest number of sorted values in a window: values in [v, v + span) (side 'left', open real
    window) or in [v, v + span] (side 'right', closed integer window), with the index range."""
    if len(vs) == 0:
        return 0, 0, 0
    ends = np.searchsorted(vs, vs + span, side=side)
    counts = ends - np.arange(len(vs))
    a = int(np.argmax(counts))
    return int(counts[a]), a, int(ends[a])


def _bucketed_max(key, stream, values, labels, span, side, is_int, nbins_target, block_pairs,
                  max_refine, collect_budget, verbose, tag, vlo=None, vhi=None):
    """Shared core.  key: sorted array such that a pair's value equals key[j] - key[i] (exactly for the
    integer statistic, to rounding for speeds); stream(i): values of the pairs (i, j > i);
    values(ii, jj): values of given pairs; labels: the cell labels k.  span/side define the window
    (see _window_max).  vlo, vhi restrict the histogram to values in [vlo, vhi] (the maximum over
    windows inside that range).  Returns the result dict."""
    t0 = time.time()
    R = len(key)
    origin = 0 if vlo is None else vlo
    top = float(key[-1] - key[0]) + float(span) + 1.0 if vhi is None else float(vhi)
    vmax = top - float(origin) + float(span) + 1.0
    if is_int:
        B = max(2 * (int(span) + 1), int(math.ceil(vmax / nbins_target)))
    else:
        B = max(4.0 * span, vmax / nbins_target)
    nbins = int(vmax / B) + 3
    hist = np.zeros(nbins, dtype=np.uint16)
    total_pairs = R * (R - 1) // 2
    restricted = vlo is not None or vhi is not None

    def stream_r(i):
        v = stream(i)
        if restricted:
            v = v[(v >= origin) & (v <= top)]
        return v

    def bin_index(v):
        return ((v - origin) // B).astype(np.int64) if is_int else np.floor((v - origin) / B).astype(np.int64)

    buf, buf_n = [], 0

    def flush():
        nonlocal buf, buf_n
        if not buf:
            return
        idx = bin_index(np.concatenate(buf))
        u, c = np.unique(idx, return_counts=True)
        new = hist[u].astype(np.int64) + c
        if new.max() > 65000:
            raise OverflowError("coarse bin overflow; increase nbins_target")
        hist[u] = new.astype(np.uint16)
        buf, buf_n = [], 0

    for i in range(R - 1):
        v = stream_r(i)
        buf.append(v)
        buf_n += len(v)
        if buf_n >= block_pairs:
            flush()
    flush()
    t1 = time.time()
    in_range = int(hist.sum())

    def refine(vlo, vhi):
        ii, jj = _range_pairs(key, vlo, vhi)
        if len(ii) == 0:
            return 0, [], None
        vs = values(ii, jj)
        o = np.argsort(vs, kind="stable")
        vs, ii, jj = vs[o], ii[o], jj[o]
        cnt, a, b = _window_max(vs, span, side)
        pairs = [(int(labels[ii[m]]), int(labels[jj[m]])) for m in range(a, b)]
        centre = None
        if b > a:
            # integer window [v_a, v_a + 2W - 2] has centre v_a + W - 1; the real window reports its midpoint
            centre = int(vs[a]) + int(span) // 2 if is_int else float((vs[a] + vs[b - 1]) / 2)
        return cnt, pairs, centre

    # certified descent through the adjacent-bin sums
    if int(hist.max()) <= 32767:
        sums = hist[:-1] + hist[1:]  # uint16 without overflow
    else:
        sums = hist[:-1].astype(np.int32) + hist[1:].astype(np.int32)
    T = int(sums.max())
    best, best_pairs, best_centre, refined, exact = 0, [], None, 0, False
    margin = span
    while T >= 1:
        cand = np.nonzero(sums == T)[0]
        done = 0
        for b in cand:
            if refined >= max_refine:
                break
            cnt, pairs, centre = refine(origin + b * B - margin, origin + (b + 2) * B + margin)
            refined += 1
            done += 1
            if cnt > best:
                best, best_pairs, best_centre = cnt, pairs, centre
        if done < len(cand):
            break  # level T interrupted by the budget: unrefined bins have sums <= T
        if best >= T:
            exact = True
            break  # level complete and every remaining sum is <= T - 1 < best
        T -= 1
    if T < 1:
        exact = True
    upper = best if exact else max(best, T)
    exact = exact or upper == best  # coinciding certified bounds determine the maximum
    T_collect, collected = None, 0
    if not exact and collect_budget > 0:
        for Tc in range(best + 1, T + 1):
            hi = np.nonzero(sums >= Tc)[0]
            mask = np.zeros(nbins, dtype=bool)
            mask[hi] = True
            mask[hi + 1] = True
            if int(hist[mask].sum()) <= collect_budget:
                T_collect = Tc
                break
        if T_collect is not None:
            vs = []
            for i in range(R - 1):
                v = stream_r(i)
                sel = mask[bin_index(v)]
                if sel.any():
                    vs.append(v[sel])
            vs = np.sort(np.concatenate(vs)) if vs else np.empty(0)
            collected = int(len(vs))
            cnt, a, b = _window_max(vs, span, side)
            if cnt > best:
                cnt2, pairs, centre = refine(vs[a] - margin, vs[a] + 2 * margin + (1 if is_int else 0))
                if cnt2 >= cnt:
                    best, best_pairs, best_centre = cnt2, pairs, centre
            if best >= T_collect:
                exact, upper = True, best
            else:
                upper = max(best, T_collect - 1)
                exact = upper == best
            del mask, vs
    t2 = time.time()
    out = {
        "R": int(R), "pairs_total": int(total_pairs), "pairs_in_range": in_range, "nbins": int(nbins),
        "vrange": None if not restricted else [float(origin), float(top)],
        "bin_over_span": float(B) / (float(span) if span else 1.0),
        "mean_occupancy": in_range / nbins, "max_bin": int(hist.max()),
        "refined": refined, "T_collect": T_collect, "collected_pairs": collected,
        "D": int(best), "exact": bool(exact), "upper": int(upper), "centre": best_centre,
        "pairs": best_pairs, "time_hist_s": t1 - t0, "time_refine_s": t2 - t1,
    }
    if verbose:
        print(f"{tag} R={R} bins={nbins} occ={out['mean_occupancy']:.2f} refined={refined} "
              f"collect(T={T_collect}, n={collected}) D={best} exact={exact} upper={upper} "
              f"centre={best_centre} ({t1 - t0:.0f}s + {t2 - t1:.0f}s)", flush=True)
    return out


def _split_max(core, vmin, vmax, span, parts, is_int, verbose, tag):
    """Run core(vlo, vhi) on `parts` equal sub-ranges of [vmin, vmax], each extended by one window
    span on both sides so that every window lies inside some part, and merge."""
    if parts <= 1:
        return core(None, None)
    edges = np.linspace(float(vmin), float(vmax), parts + 1)
    best, summaries = None, []
    t0 = time.time()
    for p in range(parts):
        lo, hi = edges[p] - span, edges[p + 1] + span
        if is_int:
            lo, hi = int(math.floor(lo)), int(math.ceil(hi))
        res = core(lo, hi)
        summaries.append({k: res[k] for k in ("vrange", "pairs_in_range", "mean_occupancy", "refined",
                                                 "T_collect", "collected_pairs", "D", "exact", "upper")})
        if best is None or res["D"] > best["D"]:
            best = dict(res)
    best["upper"] = max(s["upper"] for s in summaries)
    best["exact"] = all(s["exact"] for s in summaries) or best["upper"] == best["D"]
    best["parts"] = summaries
    best["time_total_s"] = time.time() - t0
    if verbose:
        print(f"{tag} merged over {parts} parts: D={best['D']} exact={best['exact']} upper={best['upper']} "
              f"({best['time_total_s']:.0f}s)", flush=True)
    return best


def d_star_bucketed(ks: np.ndarray, r: int, theta: float = 1.0, nbins_target: float = 4e8,
                    block_pairs: int = 4_000_000, max_refine: int = 20_000,
                    collect_budget: float = 6e7, verbose: bool = False, parts: int = 1):
    """Certified computation of the modulus-free D*_theta(r) over the member set ks (sorted, distinct).

    Returns a dict with `D_star` (the best exact window count found), `exact` (True when certified
    as the maximum), `upper` (a certified upper bound, equal to D_star when exact), the maximising
    pairs (k, k') and speed tau, the bucketing parameters and timings.  `parts` > 1 splits the speed
    range into that many sub-ranges bucketed separately (more effective bins at the same memory).
    """
    ks = np.asarray(ks, dtype=np.int64)
    sq = np.sqrt(ks.astype(np.float64))
    w = 2.0 * theta * rho(r)

    def stream(i):
        return (ks[i + 1:] - ks[i]) / (sq[i + 1:] + sq[i])

    def values(ii, jj):
        return (ks[jj] - ks[ii]) / (sq[jj] + sq[ii])

    def core(vlo, vhi):
        return _bucketed_max(sq, stream, values, ks, w, "left", False, nbins_target, block_pairs,
                             max_refine, collect_budget, verbose, f"r={r}", vlo, vhi)

    out = _split_max(core, 0.0, float(sq[-1] - sq[0]), w, parts, False, verbose, f"r={r}")
    out.update({"r": int(r), "theta": theta, "rho": rho(r), "window": w, "D_star": out["D"],
                "tau": out["centre"], "tau_sq": None if out["centre"] is None else out["centre"] ** 2})
    return out


def dmax_bucketed(ks: np.ndarray, s: np.ndarray, W: int, nbins_target: float = 4e8,
                  block_pairs: int = 4_000_000, max_refine: int = 20_000,
                  collect_budget: float = 6e7, verbose: bool = False, parts: int = 1):
    """Certified computation of Lemma D's exact-start statistic: the largest number of ordered pairs
    (k < k') whose start differences s_k' - s_k lie in a window of 2W - 1 consecutive integers.  The
    starts s must be strictly increasing (they are, on the a = 1 shell), so every difference is
    positive and t > 0 is automatic.  Returns `D_max`, `exact`, `upper`, `t` (the window centre) and
    the maximising pairs.  `parts` as in d_star_bucketed."""
    ks = np.asarray(ks, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    assert np.all(np.diff(s) > 0), "starts must be strictly increasing"
    span = 2 * int(W) - 2

    def stream(i):
        return s[i + 1:] - s[i]

    def values(ii, jj):
        return s[jj] - s[ii]

    def core(vlo, vhi):
        return _bucketed_max(s, stream, values, ks, span, "right", True, nbins_target, block_pairs,
                             max_refine, collect_budget, verbose, f"W={W}", vlo, vhi)

    out = _split_max(core, 0, int(s[-1] - s[0]), span, parts, True, verbose, f"W={W}")
    out.update({"W": int(W), "D_max": out["D"], "t": out["centre"]})
    return out


def identify(pairs):
    """Exact two-progression identification of a window's pairs (E30), JSON-safe."""
    try:
        from factorlab.experiments.prime_subfamily import identify_two_progression
        fam = identify_two_progression([(min(a, b), max(a, b)) for a, b in pairs], min_support=4)
        return json.loads(json.dumps(fam, default=str))
    except Exception as e:  # identification is diagnostic; never fail the sweep on it
        return {"error": repr(e)}


def planar_exact_point(N: int, r: int, kind: str = "squarefree", **kw):
    """Exact-start statistic of the squarefree, prime or full a = 1 shell (r/2, r] of the modulus N at
    Lemma D's window W = W(N, r) (the widest a = 1 window, from sidon_scaling.lemma_d_window)."""
    from factorlab.experiments.planar_census import shell_starts
    from factorlab.experiments.sidon_scaling import lemma_d_window
    if kind == "squarefree":
        mask = squarefree_mask(r)
    elif kind == "prime":
        mask = prime_mask(r)
    elif kind == "full":
        mask = np.ones(r + 1, dtype=bool)
    else:
        raise ValueError(kind)
    W = int(lemma_d_window(N, r))
    ks, s = shell_starts(N, r, mask)
    res = dmax_bucketed(ks, s, W, **kw)
    res.update({"N_bits": int(N).bit_length(), "N": str(N), "r": int(r), "kind": kind,
                "N_1_12": float(N) ** (1 / 12), "family": identify(res["pairs"])})
    return res


def _save(out_path, results):
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1, default=str)
    os.replace(tmp, out_path)


def _load(out_path, default):
    if os.path.exists(out_path):
        with open(out_path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return default


def _parts_for(parts, n: int) -> list:
    """Broadcast a scalar `parts` to n radii, or validate a sequence of exactly n (or 1) values."""
    if isinstance(parts, (int, np.integer)):
        return [int(parts)] * n
    parts = [int(p) for p in parts]
    if len(parts) == 1:
        return parts * n
    if len(parts) != n:
        raise ValueError(f"parts has {len(parts)} values for {n} radii")
    return parts


def run(rs, prime_rs, theta, out_path, nbins_target, max_refine, collect_budget=6e7, parts=1):
    """`parts`: an int applied to every radius, or one value per radius of rs + prime_rs."""
    results = _load(out_path, {"theta": theta, "squarefree": [], "prime": []})
    plist = _parts_for(parts, len(rs) + len(prime_rs))
    pi = 0
    for kind, rlist, shell in (("squarefree", rs, squarefree_shell), ("prime", prime_rs, prime_shell)):
        for r in rlist:
            ks = shell(r)
            res = d_star_bucketed(ks, r, theta=theta, nbins_target=nbins_target, max_refine=max_refine,
                                  collect_budget=collect_budget, verbose=True, parts=plist[pi])
            pi += 1
            res["shell"] = kind
            res["r_cuberoot"] = r ** (1 / 3)
            res["D_over_r13"] = res["D_star"] / r ** (1 / 3)
            res["family"] = identify(res["pairs"])
            results[kind] = [x for x in results.get(kind, []) if x["r"] != r] + [res]
            results[kind].sort(key=lambda x: x["r"])
            _save(out_path, results)
    return results


def run_planar(bits, rs, kinds, out_path, nbins_target, max_refine, collect_budget=6e7, index=0, parts=1):
    """Exact planar points for the E32 modulus make_semiprime(bits, 'rsa', 7, index).  `parts`: an int
    applied to every radius, or one value per radius of rs."""
    from factorlab.gen import make_semiprime
    N = int(make_semiprime(bits, "rsa", 7, index).N)
    results = _load(out_path, [])
    plist = _parts_for(parts, len(rs))
    for r, p in zip(rs, plist):
        for kind in kinds:
            res = planar_exact_point(N, r, kind, nbins_target=nbins_target, max_refine=max_refine,
                                     collect_budget=collect_budget, verbose=True, parts=p)
            res["index"] = index
            results = [x for x in results if not (x["N"] == res["N"] and x["r"] == r and x["kind"] == kind)]
            results.append(res)
            _save(out_path, results)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, nargs="*", default=[])
    ap.add_argument("--prime-r", type=int, nargs="*", default=[])
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--nbins", type=float, default=4e8)
    ap.add_argument("--max-refine", type=int, default=20_000)
    ap.add_argument("--collect-budget", type=float, default=6e7)
    ap.add_argument("--out", default="results/e37_sidon_bucketed.json")
    ap.add_argument("--planar", type=int, default=None, help="bits of the E32 modulus")
    ap.add_argument("--planar-index", type=int, default=0)
    ap.add_argument("--planar-r", type=int, nargs="*", default=[])
    ap.add_argument("--kinds", nargs="*", default=["squarefree"])
    ap.add_argument("--planar-out", default="results/e37_planar_exact.json")
    ap.add_argument("--parts", type=int, nargs="+", default=[1],
                    help="split the value range into this many parts: one value, or one per radius")
    a = ap.parse_args()
    if a.planar is not None:
        run_planar(a.planar, a.planar_r, a.kinds, a.planar_out, a.nbins, a.max_refine, a.collect_budget,
                   a.planar_index, a.parts)
    if a.r or a.prime_r:
        run(a.r, a.prime_r, a.theta, a.out, a.nbins, a.max_refine, a.collect_budget, a.parts)
