"""E19: quadratic polynomial selection is a bounded near-square search.

For f(x) = a x^2 + b x + c with f(m) = kN, put y = 2am + b.  Then

    y^2 - 4akN = b^2 - 4ac = disc(f).

Conversely, given (a, k, y) and any b == y (mod 2a), m = (y - b)/(2a) and
c = (b^2 - (y^2 - 4akN))/(4a) are integers with f(m) = kN.  This is an exact
bijection between pairs (f, m) with a > 0 and triples (a, k, y) together with
a representative b of y modulo 2a (the translates b -> b + 2at, m -> m - t
are the shifts f(x) -> f(x + t)).  If ||f||_oo <= H then |disc f| <= 5H^2,
so every bounded known-root quadratic is a near-square 4akN whose signed
residual is O(H^2).  At the unskewed joint floor of E18 (H = N^{1/5},
m = N^{2/5}, y ~ 2 sqrt(aN) ~ N^{3/5}) degree-2 polynomial selection is
therefore: search a <= N^{1/5} for near-squares 4akN with residual O(N^{2/5}).
The search exponent 1/5 coincides with that of Harvey's deterministic
factoring algorithm, but the target is a *small* discriminant, not a square
one, and by itself does not factor N.

The module provides the two maps (the forward map takes the unsigned root
|m| and selects the sign that is a root, so it inverts the backward map on
pairs with a fixed sign convention); an exhaustive scan of all pairs (f, m)
with f irreducible over Q, 1 <= a <= H, |b|, |c| <= H, f(m) = N and m > 0
(k = 1), by enumerating every integer y of either sign with
|y^2 - 4aN| <= 5H^2 and every b == y (mod 2a) in [-H, H] (each pair has one
y, so each pair appears exactly once; a polynomial appears once per positive
root, and has at most one when N > H); and a population experiment measuring

* the number of such pairs at H = C N^{1/5} against the random-phase
  prediction lambda(H) = sum_{a<=H} (2H+1)^2 / (2 sqrt(aN)) ~ 4 H^{5/2}/sqrt N
  = 4 C^{5/2} (for fixed (a, b) the values a m^2 + b m near N are spaced
  ~ 2 sqrt(aN), so the expected number of m with |f(m) - N| <= H is
  (2H+1)/(2 sqrt(aN))).  The pairs sharing a near-square (a, y) are the
  translates f(x + t), so pair counts are compound; the translation-collapsed
  events are the distinct (a, y), and these are still dependent through the
  scaling families (a t^2, y t, Delta t^2), so the scaling-primitive classes
  (a/t^2, y/t) with t maximal are counted as well; means, variances and zero
  fractions of all three are reported;
* the signed nearest-square phase (y^2 - 4aN)/(2y) = d - d^2/(2y) with
  d = y - 2 sqrt(aN) (so it equals d up to O(1/y)) of the chirp 2 sqrt(aN),
  a <= H, against the null 'uniform on [-1/2, 1/2], no lag-1 correlation'
  (a diagnostic for structure in the residuals, calibrated as if i.i.d.);
* the best k = 1 product ||f||_oo m relative to N^{3/5}, conditional on the
  modulus having a candidate;
* exact round trips of every selected quadratic E18 witness.

The discriminant/near-square identity is classical (Lehman's method, MPQS's
(ax+b)^2 - N = aQ(x), Montgomery's two-quadratics selection all use close
relatives); the explicit bijection and its reading of the E18 floor were not
found stated in the sources checked
(`.maestro/perplexity/quadratic_poly_small_square_bridge.md`).  No priority
claim is made.
"""

from __future__ import annotations

import json
import math
import random
from typing import Sequence

import numpy as np
from scipy import stats

from ..gen import make_semiprime
from ..numth import is_square, isqrt, mpz


def centered_residue(x: int, modulus: int) -> int:
    """Representative of x modulo modulus in (-modulus/2, modulus/2]."""
    r = int(x) % int(modulus)
    if 2 * r > modulus:
        r -= modulus
    return r


def quadratic_to_small_square(N: int, f: Sequence[int], m: int) -> dict:
    """Map f = [c, b, a] (low -> high) and an unsigned root m > 0 to (a, k, y, delta).

    The sign of the root is selected here (+m first, then -m); the returned
    signed root ``m`` is what ``small_square_to_quadratic`` reproduces.  Raises
    unless f(+m) or f(-m) is a nonzero multiple of N.
    """
    N = int(N)
    c, b, a = map(int, f)
    if a <= 0 or m <= 0:
        raise ValueError("require a > 0 and m > 0")
    vals = [(m, a * m * m + b * m + c), (-m, a * m * m - b * m + c)]
    roots = [(sm, v // N) for sm, v in vals if v != 0 and v % N == 0]
    if not roots:
        raise ValueError("neither sign is a nonzero root modulo N")
    sm, k = roots[0]
    y = 2 * a * sm + b
    delta = y * y - 4 * a * k * N
    assert delta == b * b - 4 * a * c
    H = max(a, abs(b), abs(c))
    return {"a": a, "b": b, "c": c, "m": sm, "k": k, "y": y, "delta": delta, "H": H, "P": H * abs(sm)}


def small_square_to_quadratic(N: int, a: int, k: int, y: int, b: int | None = None) -> dict:
    """Inverse map; b defaults to the centered representative of y modulo 2a."""
    N, a, k, y = int(N), int(a), int(k), int(y)
    if a <= 0 or k == 0:
        raise ValueError("require a > 0 and k != 0")
    if b is None:
        b = centered_residue(y, 2 * a)
    b = int(b)
    if (y - b) % (2 * a):
        raise ValueError("b must be congruent to y modulo 2a")
    m = (y - b) // (2 * a)
    delta = y * y - 4 * a * k * N
    num = b * b - delta
    assert num % (4 * a) == 0, "the congruence b == y (mod 2a) forces 4a | b^2 - delta"
    c = num // (4 * a)
    assert a * m * m + b * m + c == k * N
    H = max(a, abs(b), abs(c))
    return {"f": [c, b, a], "m": m, "k": k, "y": y, "delta": delta, "H": H, "P": H * abs(m)}


def _ceil_sqrt(n: int) -> int:
    if n <= 0:
        return 0
    r = int(isqrt(mpz(n)))
    return r if r * r == n else r + 1


def bounded_k_quadratics(N: int, H: int, k: int = 1) -> list[dict]:
    """All pairs (f, m), f = [c, b, a] irreducible over Q, 1 <= a <= H, |b|, |c| <= H, f(m) = kN, m > 0.

    Requires N > 0, H >= 1, k >= 1.  Complete and duplicate-free as a set of
    pairs: such an f has |disc f| <= 5H^2, so y = 2am + b (of either sign) lies
    in {y : |y^2 - 4akN| <= 5H^2}; both signs are enumerated, every b == y
    (mod 2a) in [-H, H] is tried, and |c| <= H, m > 0 are checked.  A pair has
    exactly one y, so it appears once.  A polynomial appears once per positive
    root; when kN > H it has at most one (two positive roots m1 < m2 would give
    kN = c - a m1 m2 <= H).  The mirror (f(-x), -m) is excluded by m > 0.
    Reducible quadratics (nonnegative square discriminant) are dropped.
    """
    N, H, k = int(N), int(H), int(k)
    if N <= 0 or H < 1 or k < 1:
        raise ValueError("require N > 0, H >= 1 and k >= 1")
    out = []
    R = 5 * H * H
    for a in range(1, H + 1):
        center = 4 * a * k * N
        lo = _ceil_sqrt(max(0, center - R))
        hi = int(isqrt(mpz(center + R)))
        for ay in range(lo, hi + 1):
            for y in ((ay, -ay) if ay else (0,)):
                delta = y * y - center
                b0 = centered_residue(y, 2 * a)
                tlo = -((H + b0) // (2 * a))  # ceil((-H - b0) / (2a))
                thi = (H - b0) // (2 * a)     # floor((H - b0) / (2a))
                for t in range(tlo, thi + 1):
                    b = b0 + 2 * a * t
                    rec = small_square_to_quadratic(N, a, k, y, b)
                    if abs(rec["f"][0]) > H or rec["m"] <= 0:
                        continue
                    if delta >= 0 and is_square(mpz(delta)):
                        continue
                    out.append(rec)
    return out


def bounded_k1_quadratics(N: int, H: int) -> list[dict]:
    """The k = 1 slice of bounded_k_quadratics (see there for the completeness argument)."""
    return bounded_k_quadratics(N, H, 1)


def predicted_k_count(N: int, H: int, k: int = 1) -> float:
    """Random-phase expected number of pairs with f(m) = kN: sum_{a<=H} (2H+1)^2 / (2 sqrt(akN))."""
    N, H, k = float(N), int(H), float(k)
    a = np.arange(1, H + 1, dtype=np.float64)
    return float((2 * H + 1) ** 2 * np.sum(1.0 / (2.0 * np.sqrt(a * k * N))))


def predicted_k1_count(N: int, H: int) -> float:
    """Random-phase expected number of pairs with f(m) = N: sum_{a<=H} (2H+1)^2 / (2 sqrt(aN))."""
    return predicted_k_count(N, H, 1)


def primitive_event(a: int, y: int) -> tuple[int, int]:
    """(a/t^2, y/t) for the largest t with t^2 | a and t | y: the root of the scaling family."""
    a, y = int(a), int(y)
    t = int(isqrt(mpz(a)))
    while t > 1:
        if a % (t * t) == 0 and y % t == 0:
            break
        t -= 1
    return a // (t * t), y // t


def k_scaling_experiment(bits: Sequence[int] = (40, 56), count: int = 100, ks: Sequence[int] = (1, 2, 3, 4),
                         C: float = 1.5, seed: int = 131, family: str = "rsa") -> dict:
    """Mean number of bounded pairs with f(m) = kN at H = C N^{1/5}, against lambda_1 / sqrt k.

    ``ks`` must contain 1, the baseline of the reported ratios.
    """
    ks = [int(k) for k in ks]
    if 1 not in ks:
        raise ValueError("ks must contain the baseline k = 1")
    rows = []
    for nbits in bits:
        counts = {k: [] for k in ks}
        preds = {k: [] for k in ks}
        for i in range(count):
            N = int(make_semiprime(nbits, family, seed, i).N)
            H = max(1, int(round(C * N ** 0.2)))
            for k in ks:
                counts[k].append(len(bounded_k_quadratics(N, H, k)))
                preds[k].append(predicted_k_count(N, H, k))
        base = float(np.mean(counts[1]))
        per_k = {}
        for k in ks:
            mc, mp = float(np.mean(counts[k])), float(np.mean(preds[k]))
            per_k[str(k)] = {"mean_count": mc, "se_count": float(np.std(counts[k]) / math.sqrt(count)),
                             "predicted": mp, "ratio_to_k1_mean": (mc / base) if base > 0 else None,
                             "predicted_ratio": 1.0 / math.sqrt(k)}
        rows.append({"nbits": nbits, "count": count, "C": C, "per_k": per_k})
    return {"bits": list(bits), "count": count, "ks": list(ks), "rows": rows}


def nearest_square_phases(N: int, A: int) -> np.ndarray:
    """(y^2 - 4aN)/(2y) for the nearest integer y to 2 sqrt(aN), a = 1..A.

    With d = y - 2 sqrt(aN) in [-1/2, 1/2] this equals d - d^2/(2y), i.e. the
    signed distance to the nearest integer up to O(1/y).
    """
    out = np.empty(int(A), dtype=np.float64)
    N = int(N)
    for i, a in enumerate(range(1, int(A) + 1)):
        z = 4 * a * N
        y0 = int(isqrt(mpz(z)))
        y = y0 if z - y0 * y0 <= (y0 + 1) * (y0 + 1) - z else y0 + 1
        out[i] = (y * y - z) / (2.0 * y)
    return out


def bridge_e18(path: str = "results/e18_poly_floor.json") -> dict:
    """Round-trip every selected quadratic E18 witness through the two maps."""
    with open(path) as fh:
        data = json.load(fh)
    checked = 0
    max_ratio = 0.0
    ks = []
    for row in data["2"]["rows"]:
        for inst in row["instances"]:
            N, f, m = int(inst["N"]), [int(c) for c in inst["f"]], int(inst["m"])
            sq = quadratic_to_small_square(N, f, m)
            back = small_square_to_quadratic(N, sq["a"], sq["k"], sq["y"], sq["b"])
            assert back["f"] == f and back["m"] == sq["m"] and abs(sq["m"]) == m
            assert abs(sq["delta"]) <= 5 * sq["H"] ** 2
            checked += 1
            max_ratio = max(max_ratio, abs(sq["delta"]) / sq["H"] ** 2)
            ks.append(sq["k"])
    return {"checked": checked, "max_abs_delta_over_H2": max_ratio,
            "k_values": {str(k): ks.count(k) for k in sorted(set(ks))}}


def quadratic_bridge_experiment(bits: Sequence[int] = (40, 56, 72, 80), count: int = 200,
                                scales: Sequence[float] = (0.75, 1.0, 1.5), seed: int = 131,
                                family: str = "rsa", e18_path: str = "results/e18_poly_floor.json") -> dict:
    """Population study of bounded k = 1 near-square pairs and of the chirp phases.

    Best-product statistics are conditional on the modulus having at least one
    candidate (``n_with_candidates`` records how many did).
    """
    rng = random.Random(seed)
    rows = []
    phase_sample = []
    for nbits in bits:
        per = {str(c): {"counts": [], "events": [], "primitive": [], "pred": [], "best": []} for c in scales}
        phase_sum = phase_sq = 0.0
        phase_n = 0
        lags = []
        for i in range(count):
            inst = make_semiprime(nbits, family, seed, i)
            N = int(inst.N)
            Hs = {str(c): max(1, int(round(c * N ** 0.2))) for c in scales}
            Hmax = max(Hs.values())
            cands = bounded_k1_quadratics(N, Hmax)
            phases = nearest_square_phases(N, Hmax)
            phase_sum += float(phases.sum())
            phase_sq += float(np.dot(phases, phases))
            phase_n += phases.size
            if phases.size > 2:
                lags.append(float(np.corrcoef(phases[:-1], phases[1:])[0, 1]))
            stride = max(1, phases.size // 50)
            phase_sample.extend(phases[::stride][:50].tolist())
            for c in scales:
                key, H = str(c), Hs[str(c)]
                keep = [r for r in cands if r["H"] <= H]
                per[key]["counts"].append(len(keep))
                per[key]["events"].append(len({(r["f"][2], r["y"]) for r in keep}))
                per[key]["primitive"].append(len({primitive_event(r["f"][2], r["y"]) for r in keep}))
                per[key]["pred"].append(predicted_k1_count(N, H))
                per[key]["best"].append(None if not keep else min(r["P"] for r in keep) / N ** 0.6)
        out_scales = {}
        for c in scales:
            z = per[str(c)]
            counts = np.array(z["counts"], dtype=float)
            events = np.array(z["events"], dtype=float)
            prim = np.array(z["primitive"], dtype=float)
            lam = float(np.mean(z["pred"]))
            best = [v for v in z["best"] if v is not None]
            out_scales[str(c)] = {
                "mean_count": float(counts.mean()), "var_count": float(counts.var()),
                "predicted_mean_count": lam,
                "mean_events": float(events.mean()), "var_events": float(events.var()),
                "mean_primitive_events": float(prim.mean()), "var_primitive_events": float(prim.var()),
                "zero_fraction": float(np.mean(counts == 0)),
                "poisson_zero_prediction_from_events": math.exp(-float(events.mean())),
                "poisson_zero_prediction_from_primitive": math.exp(-float(prim.mean())),
                "poisson_zero_prediction_from_polynomials": math.exp(-lam),
                "mean_best_P_over_N35": None if not best else float(np.mean(best)),
                "median_best_P_over_N35": None if not best else float(np.median(best)),
                "n_with_candidates": len(best),
            }
        rows.append({"nbits": nbits, "count": count, "scales": out_scales,
                     "phase_mean": phase_sum / phase_n, "phase_variance": phase_sq / phase_n - (phase_sum / phase_n) ** 2,
                     "phase_lag1_mean": float(np.mean(lags)) if lags else None,
                     "phase_lag1_se": float(np.std(lags) / math.sqrt(len(lags))) if lags else None})
    phases = np.array(phase_sample)
    ks = stats.kstest(phases + 0.5, "uniform")
    control = np.array([rng.random() - 0.5 for _ in range(phases.size)])
    ksc = stats.kstest(control + 0.5, "uniform")
    return {"bits": list(bits), "count": count, "scales": list(scales), "rows": rows,
            "phase_check": {"n": int(phases.size), "mean": float(phases.mean()), "variance": float(phases.var()),
                            "uniform_variance": 1.0 / 12.0, "ks": float(ks.statistic), "p": float(ks.pvalue),
                            "control_ks": float(ksc.statistic), "control_p": float(ksc.pvalue)},
            "e18_roundtrip": bridge_e18(e18_path)}
