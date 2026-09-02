"""E41 -- three checks asked for by the devil's-advocate review of the conjectural architecture
(`.maestro/complex_reasoning/conjectural_architecture_devils_advocate.md`, sections 12-14).

(a) Boundary certification of the modulus-free maxima of E37.  The sweep works with float64 speeds
    v = (k'-k)/(sqrt k' + sqrt k); each of the two square roots, the sum and the division is correctly
    rounded, so the relative error is at most 4 u (u = 2^-53), i.e. 4.5e-16, and the speeds on the shell
    are at most (r/2)/(sqrt r + sqrt(r/2)) < 0.3 sqrt r; ``speed_error_bound(r)`` is this rigorous
    absolute bound (6.6e-14 at r = 2^18, 4.3e-12 at r = 2^30) and the certification uses
    eps = 2 * speed_error_bound(r).  If |v_float - v_true| <= eps for every pair then, with
    delta = eps/rho_r,

        D*_float(theta - delta) <= D*_true(theta) <= D*_float(theta + delta),

    because a true window of half-width theta rho_r contains float speeds within a window of half-width
    theta rho_r + eps, and conversely.  Equality of the two float sweeps certifies the true value.
(b) Pile-up test inside the enumerated class (``modfree_census.pooled_cluster``): pool the distinct pairs of
    every family of the E39 box and take the largest window over the union; compare with the largest
    single-family cluster.
(c) Frozen out-of-sample test: the E39 box (q <= 120, |M| <= 64, A-window 6), fixed before the run, at
    non-dyadic r, against the certified exact sweep at the same r; the exact maximiser is identified and
    checked against the box.
(d) High-precision recheck of the census counts.  The census clusters are computed in float64; near speed
    tau the absolute error is 4u tau, which at r = 2^30 (rho_r = 5e-15) is a tenth of the window half-width
    for tau ~ 1.  ``high_precision_cluster`` recomputes a family's cluster with 200-bit mpfr speeds and an
    exact-rational window, and ``recheck_census`` does so for the top family and the symmetric family of
    every radius recorded in results/e39_modfree_census.json; ``symmetric_family_recheck`` does the same for the
    E27 symmetric families (alpha, gamma, q) = (j, 0, 1) at the (r, j) maximisers of results/e27_cross_class_pairs.json
    and results/e27_big.json, against their float64 census clusters (results/e41_e27_recheck.json).

Run:  python -m factorlab.experiments.pileup_certify --certify 65536 131072 262144 --oos 150000 200000 --out OUT.json
      python -m factorlab.experiments.pileup_certify --pool 65536 262144 1048576 --recheck --pool-hp 67108864 --out OUT.json
      python -m factorlab.experiments.pileup_certify --recheck-e27 14:9 16:15 18:15 --out OUT.json
The list flags take explicit radii (a bare flag is rejected); --recheck-e27 writes a bare list and runs on its own.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from factorlab.experiments.modfree_census import census, family_pairs, in_census_box, pooled_cluster
from factorlab.experiments.resonant_census import squarefree_mask
from factorlab.experiments.sidon_bucketed import d_star_bucketed, identify, rho, squarefree_shell


def speed_error_bound(r: int) -> float:
    """Rigorous absolute error of the float64 speed on the shell (r/2, r]: 4u times the largest speed."""
    u = 2.0 ** -53
    vmax = (r / 2.0) / (np.sqrt(r) + np.sqrt(r / 2.0))
    return 4.0 * u * vmax * 1.01


def parts_for_r(r: int) -> int:
    return 4 if r >= 2 ** 18 else (2 if r >= 2 ** 17 else 1)


def certify_boundary(r: int, theta: float = 1.0, eps: float | None = None, verbose: bool = False, **kw):
    """E41(a).  Returns the two float sweeps at theta -/+ eps/rho_r and the certification verdict.
    ``eps`` defaults to twice the rigorous float error bound at r; a caller-supplied eps must exceed it."""
    bound = speed_error_bound(r)
    if eps is None:
        eps = 2.0 * bound
    assert eps >= bound, "eps below the rigorous float error bound"
    ks = squarefree_shell(r)
    delta = eps / rho(r)
    parts = kw.pop("parts", parts_for_r(r))
    t0 = time.time()
    lo = d_star_bucketed(ks, r, theta=theta - delta, verbose=verbose, parts=parts, **kw)
    hi = d_star_bucketed(ks, r, theta=theta + delta, verbose=verbose, parts=parts, **kw)
    both_exact = bool(lo["exact"]) and bool(hi["exact"])
    out = {"r": int(r), "theta": theta, "eps": eps, "delta_theta": delta,
           "speed_error_bound": bound,
           "D_lo": int(lo["D_star"]), "exact_lo": bool(lo["exact"]), "upper_lo": int(lo["upper"]),
           "D_hi": int(hi["D_star"]), "exact_hi": bool(hi["exact"]), "upper_hi": int(hi["upper"]),
           "certified": both_exact and lo["D_star"] == hi["D_star"],
           "D_true_bounds": [int(lo["D_star"]), int(hi["upper"])],
           "time_s": time.time() - t0}
    if verbose:
        print(f"certify r={r}: D_lo={out['D_lo']} D_hi={out['D_hi']} certified={out['certified']} "
              f"({out['time_s']:.0f}s)", flush=True)
    return out


def out_of_sample(r: int, q_max: int = 120, m_max: int = 64, theta: float = 1.0, A_window: float = 6.0,
                  verbose: bool = False, **kw):
    """E41(c).  Frozen census box against the exact sweep at a non-dyadic r."""
    t0 = time.time()
    cen = census(r, q_max=q_max, m_max=m_max, theta=theta, A_window=A_window, verbose=verbose)
    ks = squarefree_shell(r)
    parts = kw.pop("parts", parts_for_r(r))
    ex = d_star_bucketed(ks, r, theta=theta, verbose=verbose, parts=parts, **kw)
    fam = identify(ex["pairs"]) if ex.get("pairs") else None
    top = cen["top"][0] if cen["top"] else None
    in_box = None
    if isinstance(fam, dict) and "error" not in fam and fam.get("A") is not None and fam.get("drift_free"):
        try:
            in_box = in_census_box(fam["A"], fam["C"], r, q_max=q_max, m_max=m_max, theta=theta, A_window=A_window)
        except (ValueError, TypeError):
            in_box = None
    out = {"r": int(r), "theta": theta, "census_max_sf": cen["max_sf"], "census_sym_sf": cen["sym_sf"],
           "census_top": {k: top[k] for k in ("A", "C", "q", "M", "n_classes", "cluster_sf", "members")} if top else None,
           "exact_D_star": int(ex["D_star"]), "exact": bool(ex["exact"]), "exact_upper": int(ex["upper"]),
           "exact_tau_sq": ex["tau_sq"], "exact_family": fam, "exact_maximiser_in_census_box": in_box,
           "agree": bool(ex["exact"]) and int(ex["D_star"]) == int(cen["max_sf"]),
           "r_cuberoot": r ** (1 / 3), "time_s": time.time() - t0}
    if verbose:
        print(f"oos r={r}: census={out['census_max_sf']} exact={out['exact_D_star']} (exact={out['exact']}) "
              f"agree={out['agree']} ({out['time_s']:.0f}s)", flush=True)
    return out


def high_precision_cluster(r: int, alpha: int, gamma: int, q: int, theta: float = 1.0, bits: int = 200):
    """E41(d).  The family's largest window count (squarefree pairs, and all pairs) with speeds computed in
    ``bits``-bit mpfr and the window width 2 theta rho_r = theta/(2 sqrt2 r^{3/2}) in the same precision.
    Returns None when the family has fewer than two shell pairs."""
    import gmpy2
    pairs = family_pairs(r, alpha, gamma, q)
    if pairs is None:
        return None
    km, kp = pairs[:, 0], pairs[:, 1]
    sf = squarefree_mask(km) & squarefree_mask(kp)
    ctx = gmpy2.get_context()
    old = ctx.precision
    ctx.precision = bits
    try:
        w = gmpy2.mpfr(theta) / (2 * gmpy2.sqrt(gmpy2.mpfr(2)) * gmpy2.sqrt(gmpy2.mpfr(int(r))) ** 3)
        v = [(gmpy2.mpfr(int(b) - int(a)) / (gmpy2.sqrt(gmpy2.mpfr(int(b))) + gmpy2.sqrt(gmpy2.mpfr(int(a)))))
             for a, b in zip(km.tolist(), kp.tolist())]

        def cluster(vals):
            vs = sorted(vals)
            best, j = 0, 0
            for i in range(len(vs)):
                hi = vs[i] + w
                while j < len(vs) and vs[j] < hi:
                    j += 1
                best = max(best, j - i)
            return best

        c_all = cluster(v)
        c_sf = cluster([x for x, f in zip(v, sf.tolist()) if f])
    finally:
        ctx.precision = old
    return {"cluster_sf": c_sf, "cluster_all": c_all, "members": int(len(km)), "members_sf": int(sf.sum()), "bits": bits}


def recheck_census(path: str = "results/e39_modfree_census.json", theta: float = 1.0, bits: int = 200, top_n: int = 3):
    """Recompute, in high precision, the clusters of the ``top_n`` census families and of the symmetric family
    at every radius of the E39 results file; report float64 versus mpfr values."""
    with open(path) as f:
        rows = json.load(f)
    out = []
    for row in rows:
        r = int(row["r"])
        fams = row["top"][:top_n] + ([row["sym_top"]] if row.get("sym_top") else [])
        checked = []
        for fam in fams:
            hp = high_precision_cluster(r, int(fam["alpha"]), int(fam["gamma"]), int(fam["q"]), theta, bits)
            checked.append({"A": fam["A"], "C": fam["C"], "q": fam["q"], "M": fam["M"],
                            "float_sf": fam["cluster_sf"], "float_all": fam["cluster_all"],
                            "mpfr_sf": hp["cluster_sf"] if hp else None, "mpfr_all": hp["cluster_all"] if hp else None,
                            "agree_sf": (hp is not None and hp["cluster_sf"] == fam["cluster_sf"]),
                            "agree_all": (hp is not None and hp["cluster_all"] == fam["cluster_all"])})
        out.append({"r": r, "rho": rho(r), "speed_error_near_1": 4 * 2.0 ** -53, "families": checked,
                    "all_agree": all(c["agree_sf"] and c["agree_all"] for c in checked)})
        print(f"recheck r=2^{r.bit_length() - 1}: " + "; ".join(
            f"({c['A']},{c['C']}) sf {c['float_sf']}->{c['mpfr_sf']} all {c['float_all']}->{c['mpfr_all']}" for c in checked), flush=True)
    return out


def symmetric_family_recheck(r: int, j: int, theta: float = 1.0, bits: int = 200) -> dict:
    """E41(d) for the symmetric family of E27, (alpha, gamma, q) = (j, 0, 1), whose shell pairs are
    ((j t^2 - t)/2, (j t^2 + t)/2).  ``float_D_sym`` is the float64 census cluster of the squarefree pairs exactly as
    ``sidon_scaling.symmetric_pair_cluster`` computes it (speeds (k' - k)/(sqrt k' + sqrt k), windows
    [v, v + 2 theta rho_r)); ``mpfr_sf`` and ``mpfr_all`` are the ``bits``-bit clusters of the squarefree pairs and of all
    pairs from ``high_precision_cluster``; ``members`` counts the family's distinct shell pairs.  One row of
    results/e41_e27_recheck.json."""
    from factorlab.experiments.sidon_scaling import cluster_max
    pairs = family_pairs(r, j, 0, 1)
    if pairs is None:
        raise ValueError(f"the symmetric family j = {j} has fewer than two shell pairs at r = {r}")
    km, kp = pairs[:, 0], pairs[:, 1]
    sf = squarefree_mask(km) & squarefree_mask(kp)
    a, b = km.astype(np.float64), kp.astype(np.float64)
    v = (b - a) / (np.sqrt(b) + np.sqrt(a))
    d_float = int(cluster_max(v[sf], theta * rho(r))[0]) if sf.any() else 0
    hp = high_precision_cluster(r, j, 0, 1, theta, bits)
    log2_r = r.bit_length() - 1 if r & (r - 1) == 0 else math.log2(r)
    return {"log2_r": log2_r, "j": int(j), "float_D_sym": d_float,
            "mpfr_sf": hp["cluster_sf"], "mpfr_all": hp["cluster_all"], "members": hp["members"]}


def recheck_e27_symmetric(entries, theta: float = 1.0, bits: int = 200, verbose: bool = True) -> list:
    """``symmetric_family_recheck`` at every (r, j) of ``entries``, in order.  The archived list
    (results/e41_e27_recheck.json) used the E27 symmetric-census maximisers, as log2 r : j,
    14:9 15:9 16:15 18:15 20:21 22:45 24:75 26:105 28:165 30:255 32:405."""
    rows = []
    for r, j in entries:
        row = symmetric_family_recheck(int(r), int(j), theta, bits)
        rows.append(row)
        if verbose:
            print(f"recheck-e27 r=2^{row['log2_r']} j={j}: float {row['float_D_sym']} -> mpfr sf {row['mpfr_sf']} "
                  f"all {row['mpfr_all']} (members {row['members']})", flush=True)
    return rows


def parse_log2r_j(token: str) -> tuple[int, int]:
    """'14:9' -> (2**14, 9): the LOG2R:J form taken by --recheck-e27."""
    try:
        e, j = (int(x) for x in token.split(":"))
    except ValueError:
        raise ValueError(f"expected LOG2R:J, got {token!r}") from None
    if e < 1 or j < 1:
        raise ValueError(f"need log2 r >= 1 and j >= 1, got {token!r}")
    return 2 ** e, j


def pooled_high_precision(r: int, q_max: int = 120, m_max: int = 64, theta: float = 1.0, A_window: float = 6.0,
                          bits: int = 200):
    """E41(b) in high precision: the largest window over the union of all census families, with
    ``bits``-bit mpfr speeds and an exact-precision window; for the radii where the float64 bracket of
    ``pooled_cluster`` is not tight.  Also returns the largest single-family cluster at the same precision."""
    import gmpy2
    from factorlab.experiments.modfree_census import enumerate_box
    t0 = time.time()
    fam_pairs = []
    for q, M, alpha, gamma, A_star in enumerate_box(r, q_max, m_max, theta, A_window):
        pairs = family_pairs(r, alpha, gamma, q)
        if pairs is not None:
            fam_pairs.append(pairs)
    if not fam_pairs:
        return {"r": int(r), "bits": bits, "pairs_pooled": 0, "pairs_pooled_sf": 0, "pooled_sf_hp": 0,
                "pooled_all_hp": 0, "max_family_sf_hp": 0, "max_family_all_hp": 0, "pileup_sf": False,
                "pileup_all": False, "time_s": time.time() - t0}
    allp = np.unique(np.concatenate(fam_pairs, axis=0), axis=0)
    km, kp = allp[:, 0], allp[:, 1]
    sf = squarefree_mask(km) & squarefree_mask(kp)
    ctx = gmpy2.get_context()
    old = ctx.precision
    ctx.precision = bits
    try:
        w = gmpy2.mpfr(theta) / (2 * gmpy2.sqrt(gmpy2.mpfr(2)) * gmpy2.sqrt(gmpy2.mpfr(int(r))) ** 3)

        def speeds(a_arr, b_arr):
            return [gmpy2.mpfr(int(b) - int(a)) / (gmpy2.sqrt(gmpy2.mpfr(int(b))) + gmpy2.sqrt(gmpy2.mpfr(int(a))))
                    for a, b in zip(a_arr.tolist(), b_arr.tolist())]

        def cluster(vs):
            vs.sort()
            best, j = 0, 0
            for i in range(len(vs)):
                hi = vs[i] + w
                while j < len(vs) and vs[j] < hi:
                    j += 1
                best = max(best, j - i)
            return best

        pooled_sf = cluster(speeds(km[sf], kp[sf]))
        pooled_all = cluster(speeds(km, kp))
        best_sf = best_all = 0
        for pairs in fam_pairs:
            a, b = pairs[:, 0], pairs[:, 1]
            f_sf = squarefree_mask(a) & squarefree_mask(b)
            best_all = max(best_all, cluster(speeds(a, b)))
            if f_sf.any():
                best_sf = max(best_sf, cluster(speeds(a[f_sf], b[f_sf])))
    finally:
        ctx.precision = old
    return {"r": int(r), "bits": bits, "pairs_pooled": int(len(km)), "pairs_pooled_sf": int(sf.sum()),
            "pooled_sf_hp": pooled_sf, "pooled_all_hp": pooled_all,
            "max_family_sf_hp": best_sf, "max_family_all_hp": best_all,
            "pileup_sf": pooled_sf > best_sf, "pileup_all": pooled_all > best_all, "time_s": time.time() - t0}


def _save(out_path, key, rows):
    data = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    old = [x for x in data.get(key, []) if x["r"] not in {y["r"] for y in rows}]
    data[key] = sorted(old + rows, key=lambda x: x["r"])
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E41: boundary certification, pile-up tests, out-of-sample radii and 200-bit rechecks; "
                                             "every list flag takes explicit radii")
    ap.add_argument("--certify", type=int, nargs="+", metavar="R", default=[], help="radii for the boundary certification (a)")
    ap.add_argument("--pool", type=int, nargs="+", metavar="R", default=[], help="radii for the float64 pile-up test (b)")
    ap.add_argument("--oos", type=int, nargs="+", metavar="R", default=[], help="non-dyadic radii for the frozen out-of-sample test (c)")
    ap.add_argument("--pool-hp", type=int, nargs="+", metavar="R", default=[], help="radii for the 200-bit pile-up test (b)")
    ap.add_argument("--recheck", action="store_true", help="200-bit recheck of the E39 census families (d); rows under the key 'recheck'")
    ap.add_argument("--recheck-e27", nargs="+", metavar="LOG2R:J", default=[],
                    help="200-bit recheck of the E27 symmetric families at the given (log2 r, j); writes a bare list to --out "
                         "and cannot be combined with the other modes")
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--q-max", type=int, default=120)
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--A-window", type=float, default=6.0)
    ap.add_argument("--out", default="results/e41_pileup_certify.json")
    a = ap.parse_args()
    keyed_modes = bool(a.certify or a.pool or a.oos or a.pool_hp or a.recheck)
    if not keyed_modes and not a.recheck_e27:
        ap.error("nothing to do: give at least one of --certify R..., --pool R..., --oos R..., --pool-hp R..., --recheck, "
                 "--recheck-e27 LOG2R:J...")
    if a.recheck_e27 and keyed_modes:
        ap.error("--recheck-e27 writes a bare list and must be run on its own")
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if a.recheck_e27:
        try:
            entries = [parse_log2r_j(t) for t in a.recheck_e27]
        except ValueError as exc:
            ap.error(str(exc))
        rows = recheck_e27_symmetric(entries, theta=a.theta)
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rows, f, indent=1)
        os.replace(tmp, a.out)
        raise SystemExit(0)
    for r in a.certify:
        _save(a.out, "certify", [certify_boundary(r, theta=a.theta, verbose=True)])
    for r in a.pool:
        row = pooled_cluster(r, q_max=a.q_max, m_max=a.m_max, theta=a.theta, A_window=a.A_window)
        print(f"pool r={r}: pooled_sf={row['pooled_sf']} bracket={row['pooled_sf_bracket']} max_family_sf={row['max_family_sf']} "
              f"pooled_all={row['pooled_all']} bracket={row['pooled_all_bracket']} max_family_all={row['max_family_all']} "
              f"pairs={row['pairs_pooled']} pileup_sf={row['pileup_sf']} ({row['time_s']:.0f}s)", flush=True)
        _save(a.out, "pool", [row])
    for r in a.oos:
        _save(a.out, "oos", [out_of_sample(r, q_max=a.q_max, m_max=a.m_max, theta=a.theta,
                                           A_window=a.A_window, verbose=True)])
    if a.recheck:
        _save(a.out, "recheck", recheck_census(theta=a.theta))
    for r in a.pool_hp:
        row = pooled_high_precision(r, q_max=a.q_max, m_max=a.m_max, theta=a.theta, A_window=a.A_window)
        print(f"pool-hp r={r}: pooled_sf={row['pooled_sf_hp']} max_family_sf={row['max_family_sf_hp']} "
              f"pooled_all={row['pooled_all_hp']} max_family_all={row['max_family_all_hp']} "
              f"pileup_sf={row['pileup_sf']} pileup_all={row['pileup_all']} ({row['time_s']:.0f}s)", flush=True)
        _save(a.out, "pool_hp", [row])
