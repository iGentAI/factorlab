"""E11: quadratic-sieve relation yield against the Dickman prediction with the
local shift, and the cost scaling.

The heuristic behind the quadratic sieve is that Q(x) = (s + x)^2 - N is
B-smooth as often as a random integer of the same size, corrected for its local
structure: the expected l-adic valuation of Q(x) is 2/(l-1) for split primes,
0 for inert primes, and 2, 1, 1/2 for l = 2 according to N mod 8, against
1/(l-1) for a random integer.  The correction is applied as a shift of the
logarithmic size (Silverman's form of the Knuth-Schroeppel heuristic):
    Pr[Q(x) is B-smooth] ~ rho((log|Q(x)| - Delta_N) / log B),
    Delta_N = sum_{l <= B} log(l) (E[v_l(Q)] - 1/(l-1)).
This module runs the harness QS, records the number of full relations per
sieved x, and compares it with the prediction summed over the sieved interval,
with and without the shift.  It also fits the primary work (sieve additions +
trial divisions) against sqrt(ln N ln ln N).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..gen import make_semiprime
from ..numth import mpz, isqrt
from ..registry import get_algorithm
from ..algorithms.qs import default_parameters, expected_valuation_shift


def predicted_full_relations(N, B: int, M: int, sieved: int, shift: bool = True) -> float:
    """sum over the sieved x of rho((log|Q(x)| - Delta)/log B), on a grid."""
    from .smooth_profiles import dickman_rho
    N = mpz(N)
    s = isqrt(N) + 1
    delta = expected_valuation_shift(N, B) if shift else 0.0
    half = sieved // 2
    xs = np.linspace(1, half, 400)
    # |Q(x)| ~ 2 s |x| for 1 <= |x| << s (both signs)
    logQ = np.log(2.0 * float(s) * xs)
    u = (logQ - delta) / math.log(B)
    vals = np.array([dickman_rho(float(t)) for t in u])
    per_x = float(np.trapezoid(vals, xs)) / max(half - 1, 1)
    return 2.0 * half * per_x


def exhaustive_smooth_count(N, B: int, half: int, shift: float = 0.0) -> dict:
    """Trial-divide every Q(x), |x| <= half, over the factor base: the true
    number of B-smooth values, together with the Dickman sums over the actual
    magnitudes, sum_x rho(log|Q(x)|/log B) and the same with the local shift."""
    from ..algorithms.qs import factor_base, _trial_divide
    from .smooth_profiles import dickman_rho
    N = mpz(N)
    s = isqrt(N) + 1
    primes, _ = factor_base(N, B)
    smooth = 0
    n = 0
    pred_plain = 0.0
    pred_shift = 0.0
    lB = math.log(B)
    for x in range(-half, half):
        Q = (s + x) * (s + x) - N
        if Q == 0:
            continue
        n += 1
        lq = math.log(abs(int(Q)))
        pred_plain += dickman_rho(lq / lB)
        pred_shift += dickman_rho((lq - shift) / lB)
        _ex, cof = _trial_divide(Q, primes)
        if cof == 1:
            smooth += 1
    return {"smooth": smooth, "n": n, "interval": 2 * half, "pred_plain": pred_plain, "pred_shift": pred_shift}


def sieve_efficiency(nbits_list: Sequence[int] = (40, 48, 56), count: int = 8, seed: int = 61, family: str = "rsa") -> dict:
    """For small N: exhaustive count of smooth Q(x) on the first block versus the
    sieve's full relations on the same block and the exact Dickman sums, plus a
    control: for x uniform on the block, a random integer within +-1% of the
    actual |Q(x)|, tested for B-smoothness over all primes <= B, with its own
    Dickman sum over the sampled magnitudes."""
    import random
    from sympy import factorint
    from .smooth_profiles import dickman_rho
    qs = get_algorithm("qs")
    rng = random.Random(seed)
    rows = []
    for nbits in nbits_list:
        for i in range(count):
            inst = make_semiprime(nbits, family, seed, i)
            B, M = default_parameters(inst.N)
            shift = expected_valuation_shift(inst.N, B)
            res = qs(inst.N, B=B, M=M, large_prime=False, max_blocks=1, extra=10 ** 9)  # one block only
            ex = exhaustive_smooth_count(inst.N, B, M, shift=shift)
            s = isqrt(mpz(inst.N)) + 1
            ctrl_smooth = 0
            ctrl_pred = 0.0
            n_ctrl = 0
            lB = math.log(B)
            for _ in range(2 * M):
                x = rng.randrange(-M, M)
                qmag = abs(int((s + x) * (s + x) - inst.N))
                if qmag == 0:
                    continue
                lo = max(1, int(qmag * 0.99))
                hi = max(lo + 1, int(qmag * 1.01) + 1)
                r = rng.randrange(lo, hi)
                n_ctrl += 1
                ctrl_pred += dickman_rho(math.log(r) / lB)
                f = factorint(r)
                if all(int(p_) <= B for p_ in f):
                    ctrl_smooth += 1
            rows.append({"nbits": nbits, "B": B, "M": M, "sieve_full": res.meta["full"], "exhaustive_smooth": ex["smooth"],
                         "exhaustive_n": ex["n"],
                         "pred_shift": ex["pred_shift"], "pred_plain": ex["pred_plain"],
                         "random_control_smooth": ctrl_smooth, "random_control_pred": ctrl_pred, "random_control_n": n_ctrl,
                         "local_shift_nats": shift})
    agg = {}
    for nb in nbits_list:
        sub = [r for r in rows if r["nbits"] == nb]
        sv, exh = sum(r["sieve_full"] for r in sub), sum(r["exhaustive_smooth"] for r in sub)
        ctrl = sum(r["random_control_smooth"] for r in sub)
        n_exh = sum(r["exhaustive_n"] for r in sub)
        n_ctrl = sum(r["random_control_n"] for r in sub)
        agg[str(nb)] = {"sieve_over_exhaustive": sv / max(exh, 1), "exhaustive_over_pred_shift": exh / sum(r["pred_shift"] for r in sub),
                        "exhaustive_over_pred_plain": exh / sum(r["pred_plain"] for r in sub), "exhaustive_total": exh, "sieve_total": sv,
                        "random_control_total": ctrl,
                        "Q_smooth_rate": exh / n_exh, "random_control_smooth_rate": ctrl / max(n_ctrl, 1),
                        "Q_over_random_control": (exh / n_exh) / max(ctrl / max(n_ctrl, 1), 1e-12),
                        # control scaled per modulus by its own local-shift factor pred_shift/pred_plain
                        "Q_over_shift_adjusted_control": exh / max(sum(r["random_control_smooth"] * (r["exhaustive_n"] / max(r["random_control_n"], 1))
                                                                       * r["pred_shift"] / r["pred_plain"] for r in sub), 1e-12),
                        "random_control_over_its_pred": ctrl / sum(r["random_control_pred"] for r in sub),
                        "mean_shift_nats": float(np.mean([r["local_shift_nats"] for r in sub]))}
    return {"rows": rows, "by_bits": agg}


def yield_experiment(nbits_list: Sequence[int] = (48, 56, 64, 72, 80), count: int = 4, seed: int = 61,
                     family: str = "rsa") -> dict:
    qs = get_algorithm("qs")
    rows = []
    for nbits in nbits_list:
        for i in range(count):
            inst = make_semiprime(nbits, family, seed, i)
            B, M = default_parameters(inst.N)
            res = qs(inst.N, B=B, M=M, large_prime=True)
            meta = res.meta
            pred_shift = predicted_full_relations(inst.N, B, M, meta["sieved"], shift=True)
            pred_plain = predicted_full_relations(inst.N, B, M, meta["sieved"], shift=False)
            rows.append({
                "nbits": nbits, "i": i, "found": bool(res.found), "B": B, "M": M, "factor_base": meta["factor_base"],
                "sieved": meta["sieved"], "full": meta["full"], "pairs": meta["pairs"],
                "pred_full_shift": pred_shift, "pred_full_plain": pred_plain,
                "local_shift_nats": meta["local_shift"],
                "work_sieve": int(res.work.get("sieve", 0)), "work_candidate": int(res.work.get("candidate", 0)),
                "wall": res.wall,
            })
    # aggregate ratios and cost fit
    by_bits = {}
    for nb in nbits_list:
        sub = [r for r in rows if r["nbits"] == nb]
        full = sum(r["full"] for r in sub)
        by_bits[str(nb)] = {
            "found": sum(r["found"] for r in sub), "count": len(sub),
            "full_total": full,
            "ratio_obs_over_pred_shift": full / max(sum(r["pred_full_shift"] for r in sub), 1e-9),
            "ratio_obs_over_pred_plain": full / max(sum(r["pred_full_plain"] for r in sub), 1e-9),
            "mean_work": float(np.mean([r["work_sieve"] + r["work_candidate"] for r in sub])),
            "mean_wall": float(np.mean([r["wall"] for r in sub])),
            "mean_shift_nats": float(np.mean([r["local_shift_nats"] for r in sub])),
        }
    xs = np.array([math.sqrt(nb * math.log(2) * math.log(nb * math.log(2))) for nb in nbits_list])
    ys = np.array([math.log(by_bits[str(nb)]["mean_work"]) for nb in nbits_list])
    slope, intercept = np.polyfit(xs, ys, 1)
    return {"rows": rows, "by_bits": by_bits,
            "L_half_fit": {"slope_c": float(slope), "intercept": float(intercept),
                           "comment": "work ~ exp(c sqrt(ln N ln ln N)); QS heuristic c = 1 asymptotically"}}
