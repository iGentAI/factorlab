"""The q-ary ceiling, the clipped extremal profile, and the dual route under the head floor.

A q-ary lattice contains vectors of length exactly q, and reduced bases of it keep such vectors at the head until the reduction's slope
allows shorter ones (the Z-shape), so every realisable Gram-Schmidt profile satisfies l_i <= log q.  The report's admissible class has no
such ceiling; its extremal (all-tight) profile can therefore have a head above log q, at which point it is not the profile of any basis
of the lattice.  For blocks whose first vector IS a q-vector the Gaussian-heuristic block inequality is false (q is far below the block's
Gaussian heuristic), so the natural q-aware axiom is  l_i >= min(log GH(B_i), log q) - eps.  Its extremal profile for a prefix-volume
objective is the CLIPPED tight profile: k entries at log q followed by the all-tight profile of dimension d - k and volume S - k log q,
with k the least value for which that suffix's head is at most log q (the suffix head decreases in k).  This module computes

  * clipped_tight_profile(d, b, S, log_q): the clipped extremal profile and its k;
  * qaware_chain(name, ...): the least blocksize at which the clipped extremal profile passes the detection condition over all admissible
    sample counts (compare the certified crossings 417/642/900 of the unclipped class);
  * ceiling_status(name, b, m): whether the unclipped extremal head at (b, m) is below log q;
  * dual_route(name, ...): the specification script's dual-attack cost model (LWE_dual_cost of pq-crystals/security-estimates) evaluated
    with three first-vector lengths -- the unclipped GSA (the script's 'randomized' shape), the class's head floor l_1 >= S/d + h_{d,b}(0),
    and min(head floor, log q) -- minimised over (b, m); the head floor is a certified lower bound on l_1 for admissible profiles, and the
    dual cost is increasing in the first vector's length, so the floor's minimum cost is a lower bound on the model's cost over the class.

Everything is double precision (sense (iv)); the head floors at the reported optima are certified separately with profile_floor.floor_l1.

CLI:  python -m latticelab.qceiling --qaware --dual --sets Kyber512 Kyber768 Kyber1024 --out results/lattice_qceiling.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, Sequence, Tuple

import numpy as np

from latticelab.profile import delta_gsa
from latticelab.profile_floor import floor_l1_float, log_chat, tight_profile
from latticelab.spec_chain import KYBER, N_RING, Q

LOG_Q = math.log(Q)
C_SVP, C_SIEVE = math.log(math.sqrt(3.0 / 2.0), 2), math.log(math.sqrt(4.0 / 3.0), 2)   # the script's svp_classical and nvec_sieve slopes


def tight_head(d: int, b: int, S: float) -> float:
    """Head of the all-tight profile of dimension d, blocksize b, log-volume S: S/d + h_{d,b}(0)."""
    return floor_l1_float(d, b, 0.0, S)["l1_floor"]


def _suffix_head(d: int, b: int, S: float, k: int, log_q: float) -> float:
    """Head of the q-aware suffix after k q-entries: dimension d - k, blocksize min(b, d - k), log-volume S - k log_q."""
    dp, Sp = d - k, S - k * log_q
    if dp == 1:
        return Sp
    if dp > b:
        return tight_head(dp, b, Sp)
    return float(tight_profile(dp, dp, 0.0, Sp)[0])


def _suffix(d: int, b: int, S: float, k: int, log_q: float) -> np.ndarray:
    dp, Sp = d - k, S - k * log_q
    if dp == 1:
        return np.array([Sp])
    return tight_profile(dp, min(b, dp), 0.0, Sp)


def clipped_tight_profile(d: int, b: int, S: float, log_q: float = LOG_Q) -> Tuple[int, np.ndarray]:
    """HEAD-clipped extremal profile: k entries at log_q followed by the tight profile of the remaining dimension d - k (blocksize
    min(b, d - k)) and volume S - k log_q, with k the least value for which that suffix's HEAD is at most log_q.  The tail of a tight
    profile rises (its last entry is the previous one plus 2|log c_hat_2|), so near the ceiling the last entries of the suffix may exceed
    log_q; use n_above_ceiling to check the whole profile.  For blocksizes b <= 12 the tight head lies below the mean (log c_hat_n < 0
    there), so the head criterion never clips and only the entrywise check is informative.  S = d log_q (to machine precision) gives the profile with every entry S/d;
    S > d log_q beyond machine roundoff is infeasible.  The suffix head decreases in k, so k is found by bisection and the least-k property
    is verified afterwards; the returned profile sums to S."""
    if not 2 <= b < d:
        raise ValueError("need 2 <= b < d")
    full = d * log_q
    if S > full and not math.isclose(S, full, rel_tol=1e-13, abs_tol=0.0):
        raise ValueError("infeasible: the log-volume exceeds d log q, so no profile fits under the ceiling")
    if math.isclose(S, full, rel_tol=1e-13, abs_tol=0.0):
        return d, np.full(d, S / d)
    if _suffix_head(d, b, S, 0, log_q) <= log_q:
        return 0, tight_profile(d, b, 0.0, S)
    lo, hi = 0, d - 1                              # head(0) > log_q; head(d-1) = S - (d-1) log_q < log_q
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _suffix_head(d, b, S, mid, log_q) <= log_q:
            hi = mid
        else:
            lo = mid
    k = hi
    if not (_suffix_head(d, b, S, k, log_q) <= log_q < _suffix_head(d, b, S, k - 1, log_q)):
        raise RuntimeError("the suffix head is not monotone in the clipping depth here; least k not established")
    return k, np.concatenate([np.full(k, log_q), _suffix(d, b, S, k, log_q)])


def n_above_ceiling(prof: Sequence[float], log_q: float = LOG_Q, tol: float = 1e-9) -> int:
    """Number of entries of a profile above log_q (beyond tol)."""
    return int(sum(1 for x in prof if x > log_q + tol))


def detection_margin(ell: Sequence[float], b: int, sigma2: float) -> float:
    """Floor-side reading: log GH_b(L/F_{d-b}) - log(sigma sqrt b), the last block's GH from the prefix volume."""
    d = len(ell)
    S = float(sum(ell))
    P = float(sum(ell[: d - b]))
    return log_chat(b) + (S - P) / b - 0.5 * math.log(sigma2 * b)


def ceiling_status(name: str, b: int, m: int) -> Dict:
    """Unclipped extremal head at (b, m) against log q, and the clipping depth the q-aware extremal needs there."""
    k = KYBER[name]["k"]
    d = m + k * N_RING + 1
    S = m * LOG_Q
    head = tight_head(d, b, S)
    kq, prof = clipped_tight_profile(d, b, S)
    return {"set": name, "b": b, "m": m, "d": d, "tight_head": head, "log_q": LOG_Q, "head_minus_log_q": head - LOG_Q, "clip_depth": kq,
            "n_above_ceiling_clipped": n_above_ceiling(prof), "n_above_ceiling_unclipped": n_above_ceiling(tight_profile(d, b, 0.0, S))}


def qaware_chain(name: str, b_lo: int, b_hi: int, m_stride: int = 1, log=print) -> Dict:
    """Least b in [b_lo, b_hi] at which the clipped (q-aware) extremal profile passes the detection condition for some admissible m,
    with the per-b best margin, the m attaining it and its clipping depth; also the unclipped extremal's best margin for comparison."""
    k, eta1 = KYBER[name]["k"], KYBER[name]["eta1"]
    sigma2 = eta1 / 2.0
    per_b, least = [], None
    t0 = time.time()
    for b in range(b_lo, b_hi + 1):
        best, best_un = None, None
        for m in range(0, (k + 1) * N_RING + 1, m_stride):
            d = m + k * N_RING + 1
            if 2 * b >= d + 1:
                continue
            S = m * LOG_Q
            kq, prof = clipped_tight_profile(d, b, S)
            mg = detection_margin(prof, b, sigma2)
            if best is None or mg > best["margin"]:
                best = {"m": m, "d": d, "margin": mg, "clip_depth": kq, "n_above_ceiling": n_above_ceiling(prof)}
            if kq == 0:
                mu = mg
            else:
                mu = detection_margin(tight_profile(d, b, 0.0, S), b, sigma2)
            if best_un is None or mu > best_un["margin"]:
                best_un = {"m": m, "d": d, "margin": mu}
        per_b.append({"b": b, "best_qaware": best, "best_unclipped": best_un})
        if least is None and best is not None and best["margin"] >= 0:
            least = {"b": b, **best}
        if log:
            log(f"{name} q-aware b={b}: margin {best['margin']:+.5f} at m={best['m']} (clip {best['clip_depth']}); unclipped best {best_un['margin']:+.5f} at m={best_un['m']}  [{time.time()-t0:.0f}s]")
        if least is not None and b >= least["b"] + 2:
            break
    return {"set": name, "b_range": [b_lo, b_hi], "m_stride": m_stride, "least_passing_qaware": least, "per_b": per_b,
            "note": "clipped extremal = k entries at log q then the tight profile of the rest; margins are the floor-side reading in double precision"}


def dual_cost_bits(log_l: float, b: int, sigma2: float) -> Dict:
    """The script's LWE_dual_cost with first-vector log-length log_l: tau = l s / q, log2 eps = -2 pi^2 tau^2 / ln 2,
    log2 R = max(0, -2 log2 eps - c_sieve b), cost = c_svp b + log2 R (classical core-SVP)."""
    tau = math.exp(log_l) * math.sqrt(sigma2) / Q
    log2_eps = -2.0 * math.pi ** 2 * tau ** 2 / math.log(2.0)
    log2_R = max(0.0, -2.0 * log2_eps - C_SIEVE * b)
    return {"log_l": log_l, "l": math.exp(log_l), "tau": tau, "log2_eps": log2_eps, "log2_R": log2_R, "cost_bits": C_SVP * b + log2_R}


def dual_route(name: str, b_lo: int = 300, b_hi: int = 1100, m_stride: int = 1, log=print) -> Dict:
    """Minimise the script's dual cost over (b, m), m in [1, (k+1) n], d = kn + m, volume q^{kn}, for three first lengths: the unclipped GSA
    (the script's randomized shape), the class head floor, and min(head floor, log q)."""
    k, eta1 = KYBER[name]["k"], KYBER[name]["eta1"]
    n = k * N_RING
    sigma2 = eta1 / 2.0
    best = {"gsa": None, "floor": None, "floor_capped": None}
    t0 = time.time()
    for b in range(b_lo, b_hi + 1):
        for m in range(1, (k + 1) * N_RING + 1, m_stride):
            d = n + m
            if b >= d:
                continue
            S = n * LOG_Q
            log_l_gsa = S / d + (d - 1) * math.log(delta_gsa(b))
            log_l_floor = tight_head(d, b, S)
            for key, log_l in (("gsa", log_l_gsa), ("floor", log_l_floor), ("floor_capped", min(log_l_floor, LOG_Q))):
                c = dual_cost_bits(log_l, b, sigma2)
                if best[key] is None or c["cost_bits"] < best[key]["cost_bits"]:
                    best[key] = {"b": b, "m": m, "d": d, **c, "head_minus_log_q": log_l_floor - LOG_Q}
        if log and b % 50 == 0:
            log(f"{name} dual b={b}: gsa best {best['gsa']['cost_bits']:.2f} (b={best['gsa']['b']}), floor best {best['floor']['cost_bits']:.2f} (b={best['floor']['b']})  [{time.time()-t0:.0f}s]")
    return {"set": name, "b_range": [b_lo, b_hi], "m_stride": m_stride, "best": best,
            "note": "the script's LWE_dual_cost (classical core-SVP 0.292 b, sieve reuse 0.2075 b) with the first length from the unclipped GSA, "
                    "the class head floor, and the floor capped at log q; the floor's cost is a lower bound over admissible profiles; double precision"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["Kyber512", "Kyber768", "Kyber1024"])
    ap.add_argument("--qaware", action="store_true")
    ap.add_argument("--dual", action="store_true")
    ap.add_argument("--b-margin", type=int, default=12, help="q-aware scan from printed b - b_margin to printed b + 3 b_margin")
    ap.add_argument("--m-stride", type=int, default=1)
    ap.add_argument("--dual-m-stride", type=int, default=2)
    ap.add_argument("--out", default="results/lattice_qceiling.json")
    a = ap.parse_args()
    res: Dict = {}
    for name in a.sets:
        b_print = KYBER[name]["printed"][1]
        if a.qaware:
            res[f"{name},qaware"] = qaware_chain(name, b_print - a.b_margin, b_print + 3 * a.b_margin, a.m_stride)
            json.dump(res, open(a.out, "w"), indent=1)
            print(f"== {name}: least passing b, q-aware extremal = {res[f'{name},qaware']['least_passing_qaware']}")
        if a.dual:
            res[f"{name},dual"] = dual_route(name, m_stride=a.dual_m_stride)
            json.dump(res, open(a.out, "w"), indent=1)
            for key, v in res[f"{name},dual"]["best"].items():
                print(f"== {name} dual [{key}]: b={v['b']} m={v['m']} d={v['d']} l={v['l']:.1f} log2eps={v['log2_eps']:.2f} log2R={v['log2_R']:.2f} cost={v['cost_bits']:.2f} bits (head - log q = {v['head_minus_log_q']:+.4f})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
