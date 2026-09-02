"""Simulator-based re-evaluation of the ML-KEM detection chain.

The report's detection chain reads the attack's success condition -- the planted vector's projection, of norm about sigma sqrt(b), is no
longer than the Gaussian-heuristic length of the last block L/F_{d-b} -- off three profiles: the GSA line (the specification's own
model, condition (9)), the extremal profile of the admissible class (the floor-side form), and here the outputs of the two standard BKZ
simulators, Chen-Nguyen [CN11] and the probabilistic Bai-Stehle-Wen variant [BSW18], as shipped in fpylll.tools.bkz_simulator.

For each Kyber set (k, eta1) and each (b, m) the lattice has dimension d = m + kn + 1 and log-volume S = m log q.  The simulators need a
starting profile; we use the standard Z-shape model of an LLL-reduced q-ary basis: l_i = clamp(a - s (i - 1), 0, log q) with slope
s = 2 log delta_LLL, delta_LLL = 1.0219, and the level a fixed by sum_i l_i = S.  The CN11 simulator is deterministic and stops on its
own progress criterion (typically after 40-60 tours at these sizes), so its output is a converged profile; the BSW18 simulator is
randomised and its stopping rule does not fire, so it is run with a fixed tour budget (50 by default) and its output is budget-limited,
not established as converged -- the budget is reported with every reading.  Two readings of the condition are
recorded: the floor-side form  log(sigma sqrt b) <= log GH_b(L/F_{d-b}) = log c_hat_b + (S - P_{d-b}) / b  (a function of the prefix
volume only), and the entry form  log(sigma sqrt b) <= l_{d-b+1}  used by the 2016 estimate.  The least b for which some admissible m
passes is the simulator-based crossing.

Everything here is heuristic: the simulators are predictions of BKZ output, the Z-shape start is a model, and double precision is used
throughout (sense (iv) of the report).  The purpose is comparison with the GSA crossings 406, 624, 874 and the certified floor-side
crossings 417, 642, 900.

CLI:  python -m latticelab.simulator_chain --sets Kyber512 Kyber768 Kyber1024 --models cn bsw --m-stride 4 --out results/lattice_simulator_chain.json
      python -m latticelab.simulator_chain --fixed-point Kyber512:417:520 Kyber768:642:700 --intervals Kyber512:417 Kyber768:642 Kyber1024:900 \
          --out results/lattice_cn_fixed_point.json

The fixed-point mode (`fixed_point_check`) tests the converged CN11 output against the proposition that a profile is a fixed point of the
tour iff l_k <= log GH(B_k) at every position, with fpylll's own constants (exact Gaussian heuristic above block dimension 45, its
tabulated HKZ constants at or below), and compares it with the head-clipped tight profile of latticelab.qceiling over the first d - 45
entries, where the two conventions' constants coincide.  `passing_interval` lists every admissible sample count passing the detection
condition at a blocksize (double precision), with the maximum-margin count the certified witnesses use.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, List, Optional, Sequence

from fpylll import BKZ
from fpylll.tools.bkz_simulator import _lg_gh, simulate, simulate_prob

from latticelab.profile_floor import log_chat
from latticelab.spec_chain import KYBER, N_RING, Q

DELTA_LLL = 1.0219


def zshape_profile(d: int, log_vol: float, log_q: float, delta_lll: float = DELTA_LLL) -> List[float]:
    """Z-shape model of an LLL-reduced q-ary profile: l_i = clamp(a - s(i-1), 0, log_q), s = 2 log delta_lll, with sum l_i = log_vol."""
    if not 0 <= log_vol <= d * log_q:
        raise ValueError("log_vol must lie in [0, d log q]")
    s = 2.0 * math.log(delta_lll)

    def total(a: float) -> float:
        return sum(min(log_q, max(0.0, a - s * i)) for i in range(d))

    lo, hi = -s * d, log_q + s * d
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < log_vol:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    return [min(log_q, max(0.0, a - s * i)) for i in range(d)]


def simulate_profile(ell0: Sequence[float], b: int, model: str, max_tours: int = 2000, seed: int = 0xDEADBEEF) -> Dict:
    """Run the CN11 ('cn') or BSW18 ('bsw') simulator on the profile ell0 (log-norms) with blocksize b; returns the output log-norms and
    the number of tours performed.  For 'cn' that count is where the simulator's own progress criterion stopped it (a converged profile
    when it is below max_tours); for 'bsw' the criterion does not fire and the count is the exhausted budget max_tours."""
    r = [math.exp(2.0 * x) for x in ell0]
    par = BKZ.Param(block_size=b, max_loops=max_tours)
    if model == "cn":
        r_out, tours = simulate(r, par)
    elif model == "bsw":
        r_out, tours = simulate_prob(r, par, prng_seed=seed)
    else:
        raise ValueError("model must be 'cn' or 'bsw'")
    return {"ell": [0.5 * math.log(x) for x in r_out], "tours": int(tours)}


def detection_readings(ell: Sequence[float], b: int, sigma2: float) -> Dict:
    """Both readings of the detection condition on a profile: floor-side (last-block GH from the prefix volume) and entry form."""
    d = len(ell)
    S = float(sum(ell))
    P = float(sum(ell[: d - b]))
    log_gh_last = log_chat(b) + (S - P) / b
    entry = float(ell[d - b])
    lhs = 0.5 * math.log(sigma2 * b)
    return {"lhs": lhs, "log_gh_last": log_gh_last, "entry": entry, "margin_gh": log_gh_last - lhs, "margin_entry": entry - lhs}


def point(k: int, eta1: int, b: int, m: int, model: str, max_tours: int = 2000, seed: int = 0xDEADBEEF) -> Dict:
    """One (b, m) evaluation: build the Z-shape start of the Kyber lattice, simulate BKZ-b, read the condition both ways."""
    d = m + k * N_RING + 1
    if 2 * b >= d + 1:
        raise ValueError("outside the condition's domain: need 2b < d + 1")
    log_q = math.log(Q)
    ell0 = zshape_profile(d, m * log_q, log_q)
    sim = simulate_profile(ell0, b, model, max_tours, seed)
    out = detection_readings(sim["ell"], b, eta1 / 2.0)
    out.update({"b": b, "m": m, "d": d, "tours": sim["tours"], "model": model})
    return out


def simulator_chain(name: str, model: str, b_lo: int, b_hi: int, m_stride: int = 4, m_range: Optional[Sequence[int]] = None,
                    max_tours: int = 2000, seed: int = 0xDEADBEEF, log=print) -> Dict:
    """Scan b in [b_lo, b_hi] and admissible m (stride m_stride over [0, (k+1) n], or m_range) and return, for each reading of the condition,
    the least passing b with the m of largest margin, plus the per-b best margins."""
    k, eta1 = KYBER[name]["k"], KYBER[name]["eta1"]
    lo_m, hi_m = (0, (k + 1) * N_RING) if m_range is None else (int(m_range[0]), int(m_range[1]))
    per_b = []
    least = {"gh": None, "entry": None}
    t0 = time.time()
    for b in range(b_lo, b_hi + 1):
        best = {"gh": None, "entry": None}
        for m in range(lo_m, hi_m + 1, m_stride):
            d = m + k * N_RING + 1
            if 2 * b >= d + 1:
                continue
            p = point(k, eta1, b, m, model, max_tours, seed)
            for key, mk in (("gh", "margin_gh"), ("entry", "margin_entry")):
                if best[key] is None or p[mk] > best[key]["margin"]:
                    best[key] = {"m": m, "d": d, "margin": p[mk], "tours": p["tours"]}
        row = {"b": b, "best": best}
        per_b.append(row)
        for key in ("gh", "entry"):
            if least[key] is None and best[key] is not None and best[key]["margin"] >= 0:
                least[key] = {"b": b, **best[key]}
        if log:
            log(f"{name} {model} b={b}: gh margin {best['gh']['margin']:+.5f} at m={best['gh']['m']}; "
                f"entry margin {best['entry']['margin']:+.5f} at m={best['entry']['m']}  [{time.time() - t0:.0f}s]")
        if least["gh"] is not None and least["entry"] is not None and b >= max(least["gh"]["b"], least["entry"]["b"]) + 2:
            break
    return {"set": name, "model": model, "k": k, "eta1": eta1, "b_range": [b_lo, b_hi], "m_range": [lo_m, hi_m], "m_stride": m_stride,
            "max_tours": max_tours, "delta_lll": DELTA_LLL, "least_passing": least, "per_b": per_b,
            "note": "heuristic simulator outputs from a Z-shape start; 'gh' reads the last block's GH from the prefix volume (floor-side form), "
                    "'entry' reads l_{d-b+1} (2016 estimate); least_passing is over the scanned m only"}


def block_gh_fpylll(ell: Sequence[float], k: int, b: int) -> float:
    """log GH of the block starting at 0-based position k of the profile ell (natural log-norms), of dimension min(b, d - k), with the
    constant fpylll's simulator uses for that dimension (exact GH above 45, the tabulated HKZ constant at or below)."""
    d = len(ell)
    f = min(k + b, d)
    n = f - k
    return sum(ell[k:f]) / n + _lg_gh(n) * math.log(2.0)


def fixed_point_check(name: str, b: int, m: int, max_tours: int = 2000, tol: float = 1e-6) -> Dict:
    """Run the CN11 simulator from the Z-shape start of the Kyber lattice at (b, m) and test its output against the fixed-point
    condition  l_k <= log GH(B_k) for every k <= d - 1  (fpylll's constants).  Reports: the tour count and whether the run stopped by the
    simulator's own criterion (a fixed point) rather than the budget; the number of entries equal to log q and the largest excess over it;
    the deficits log GH(B_k) - l_k over all k <= d - 2 (min, max) and the positions where they exceed tol; and the comparison with the
    head-clipped tight profile of latticelab.qceiling -- the largest entrywise difference over the first d - 45 entries (where the constants
    of the two conventions agree) and that profile's own deficits over the same range.  Double precision."""
    from latticelab.qceiling import clipped_tight_profile

    k_set, eta1 = KYBER[name]["k"], KYBER[name]["eta1"]
    d = m + k_set * N_RING + 1
    if 2 * b >= d + 1:
        raise ValueError("outside the condition's domain: need 2b < d + 1")
    log_q = math.log(Q)
    S = m * log_q
    ell0 = zshape_profile(d, S, log_q)
    sim = simulate_profile(ell0, b, "cn", max_tours)
    ell = sim["ell"]
    deficits = [block_gh_fpylll(ell, k, b) - ell[k] for k in range(d - 1)]
    above = [k for k, x in enumerate(deficits) if x > tol]
    n_at_q = sum(1 for x in ell if abs(x - log_q) < 1e-9)
    k_clip, clip = clipped_tight_profile(d, b, S, log_q)
    head = d - 45
    diff_head = max(abs(ell[i] - clip[i]) for i in range(head))
    clip_def = [block_gh_fpylll(list(clip), k, b) - float(clip[k]) for k in range(head)]
    return {"set": name, "b": b, "m": m, "d": d, "tours": sim["tours"], "stopped_by_criterion": sim["tours"] < max_tours,
            "max_tours": max_tours, "n_entries_at_log_q": n_at_q, "max_excess_over_log_q": max(ell) - log_q,
            "deficit_min": min(deficits), "deficit_max": max(deficits), "positions_deficit_above_tol": above, "tol": tol,
            "deficits_at_those_positions": [deficits[k] for k in above], "clip_depth": int(k_clip),
            "max_abs_diff_to_clipped_tight_first_d_minus_45": diff_head,
            "clipped_tight_deficits_at_clipped_positions": clip_def[: int(k_clip)],
            "clipped_tight_deficit_max_elsewhere_first_d_minus_45": max(clip_def[int(k_clip):]) if k_clip < head else None,
            "clipped_tight_deficit_min_first_d_minus_45": min(clip_def),
            "note": "deficits are log GH(B_k) - l_k with fpylll's simulator constants; a fixed point has all deficits >= 0 (Proposition, "
                    "fixed points of the Chen-Nguyen tour); the comparison with the head-clipped tight profile is over the first d - 45 "
                    "entries, the tail constants of the two conventions differing"}


def passing_interval(name: str, b: int, eps: float = 0.0, model: str = "tight") -> Dict:
    """Every sample count m in [0, (k+1) n] with 2b < d + 1 at which the detection condition  log(sigma sqrt b) <= detection_entry(d, b,
    m log q, model, eps)  holds, for the Kyber set `name` at blocksize b: the passing interval, its count and contiguity, and the
    maximum-margin count (the certified witnesses' convention).  Double precision."""
    from latticelab.spec_chain import detection_entry

    k_set, eta1 = KYBER[name]["k"], KYBER[name]["eta1"]
    lhs = 0.5 * math.log((eta1 / 2.0) * b)
    rows = []
    for mm in range(0, (k_set + 1) * N_RING + 1):
        d = mm + k_set * N_RING + 1
        if d + 1 - 2 * b <= 0:
            continue
        rows.append((mm, detection_entry(d, b, mm * math.log(Q), model, eps) - lhs))
    passing = [mm for mm, mg in rows if mg >= 0]
    best = max(rows, key=lambda r: r[1]) if rows else None
    return {"set": name, "b": b, "eps": eps, "model": model, "passing": passing, "count": len(passing),
            "m_lo": min(passing) if passing else None, "m_hi": max(passing) if passing else None,
            "contiguous": passing == list(range(min(passing), max(passing) + 1)) if passing else True,
            "m_max_margin": best[0] if best else None, "d_max_margin": best[0] + k_set * N_RING + 1 if best else None,
            "margin_max": best[1] if best else None,
            "note": "m_max_margin is the sample count of largest margin among all admissible m, passing or not; the certified witnesses use it"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["Kyber512", "Kyber768", "Kyber1024"])
    ap.add_argument("--models", nargs="+", default=["cn", "bsw"])
    ap.add_argument("--b-lo", type=int, default=None, help="default: the printed blocksize minus 8")
    ap.add_argument("--b-hi", type=int, default=None, help="default: the printed blocksize plus 40")
    ap.add_argument("--m-stride", type=int, default=4)
    ap.add_argument("--max-tours", type=int, default=2000, help="tour budget for the CN11 simulator, which stops on its own criterion")
    ap.add_argument("--bsw-tours", type=int, default=50, help="tour budget for the BSW18 simulator, whose stopping rule does not fire")
    ap.add_argument("--bsw-m-stride", type=int, default=None, help="m stride for the BSW18 scan (default: --m-stride)")
    ap.add_argument("--out", default="results/lattice_simulator_chain.json")
    ap.add_argument("--fixed-point", nargs="*", metavar="SET:B:M", default=None,
                    help="fixed-point mode: run fixed_point_check at each SET:B:M and write the results to --out instead of the scan")
    ap.add_argument("--intervals", nargs="*", metavar="SET:B", default=None,
                    help="fixed-point mode: also record passing_interval at each SET:B")
    a = ap.parse_args()
    if a.fixed_point is not None:
        out = {"fixed_point": {}, "passing_intervals": {}}
        for spec in a.fixed_point:
            name, b, m = spec.split(":")
            r = fixed_point_check(name, int(b), int(m), a.max_tours)
            out["fixed_point"][spec] = r
            print(f"{spec}: tours={r['tours']} fixed point={r['stopped_by_criterion']} entries at log q={r['n_entries_at_log_q']} "
                  f"deficits in [{r['deficit_min']:.1e}, {r['deficit_max']:.4f}] above tol at {r['positions_deficit_above_tol']} "
                  f"|sim - clipped tight| <= {r['max_abs_diff_to_clipped_tight_first_d_minus_45']:.2e} (clip depth {r['clip_depth']})")
        for spec in a.intervals or []:
            name, b = spec.split(":")
            r = passing_interval(name, int(b))
            out["passing_intervals"][spec] = r
            if r["count"]:
                print(f"{spec}: passing m in [{r['m_lo']}, {r['m_hi']}] ({r['count']}, contiguous={r['contiguous']}), max-margin m={r['m_max_margin']}")
            else:
                print(f"{spec}: no admissible m passes; best margin {r['margin_max']} at m={r['m_max_margin']}")
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)
        return
    results = {}
    for name in a.sets:
        b_print = KYBER[name]["printed"][1]
        b_lo = a.b_lo if a.b_lo is not None else b_print - 8
        b_hi = a.b_hi if a.b_hi is not None else b_print + 40
        for model in a.models:
            tours = a.max_tours if model == "cn" else a.bsw_tours
            stride = a.m_stride if (model == "cn" or a.bsw_m_stride is None) else a.bsw_m_stride
            results[f"{name},{model}"] = simulator_chain(name, model, b_lo, b_hi, stride, max_tours=tours)
            json.dump(results, open(a.out, "w"), indent=1)
            lp = results[f"{name},{model}"]["least_passing"]
            print(f"== {name} {model}: least passing b (gh) = {lp['gh']}, (entry) = {lp['entry']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
