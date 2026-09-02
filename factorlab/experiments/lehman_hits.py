"""E7: the Lehman hits of a modulus are the continued-fraction convergents of p/q.

For N = pq and parameter r, the cell (a, b) (gcd(a, b) = 1, ab <= r) is a *hit*
if Harvey's window condition holds:
    0 <= a q + b p - 2 sqrt(a b N) < sqrt(N) / (4 r sqrt(a b)).
Since aq + bp - 2 sqrt(abN) = v^2 / (sqrt(aq) + sqrt(bp))^2 with v = |aq - bp|,
the condition is (up to the factor (sqrt(aq)+sqrt(bp))^2 / (4 sqrt(abN)) >= 1)
v^2 < N / r, i.e. |p/q - a/b| < sqrt(N/r) / (bq).  Dirichlet's theorem gives
such a fraction with b <= sqrt(rq/p), and the convergent h_n/k_n of p/q with
k_n <= sqrt(rq/p) < k_{n+1} satisfies it.  This module enumerates all hits,
classifies them as convergents / intermediate fractions / other, and records
the distribution of the number of hits over random moduli.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..gen import make_semiprime
from ..numth import mpz, iroot


def continued_fraction(p: int, q: int) -> list[int]:
    """Partial quotients of p/q (p, q > 0)."""
    out = []
    while q:
        a, r = divmod(p, q)
        out.append(a)
        p, q = q, r
    return out


def convergents(cf: list[int]) -> list[tuple[int, int]]:
    """(h_n, k_n) for n = 0, 1, ..."""
    h0, k0, h1, k1 = 0, 1, 1, 0  # h_{-2}/k_{-2}, h_{-1}/k_{-1}
    out = []
    for a in cf:
        h0, k0, h1, k1 = h1, k1, a * h1 + h0, a * k1 + k0
        out.append((h1, k1))
    return out


def intermediate_fractions(cf: list[int], max_den: int) -> dict[tuple[int, int], tuple[int, int]]:
    """All fractions (h_{n-1} + j h_n)/(k_{n-1} + j k_n), 1 <= j < a_{n+1},
    with denominator <= max_den; maps (h, k) -> (n, j)."""
    conv = convergents(cf)
    out = {}
    for n in range(len(conv) - 1):
        hn, kn = conv[n]
        hp, kp = conv[n - 1] if n >= 1 else (1, 0)
        a_next = cf[n + 1]
        for j in range(1, a_next):
            h, k = hp + j * hn, kp + j * kn
            if k > max_den:
                break
            out[(h, k)] = (n, j)
    return out


def lehman_hits(N, p, q, r: int | None = None) -> list[dict]:
    """All coprime cells (a, b) with ab <= r satisfying Harvey's window condition.

    Float64 is sufficient here: the window width is >= 1/4 at r = N^{1/3}
    while the rounding error of 2 sqrt(abN) is far below 1e-6 for N < 2^52.
    """
    N, p, q = int(N), int(p), int(q)
    if r is None:
        r = int(iroot(mpz(N), 3)[0])
    sqrtN = math.sqrt(N)
    hits = []
    for a in range(1, r + 1):
        bmax = r // a
        b = np.arange(1, bmax + 1, dtype=np.float64)
        U = a * q + b * p
        x = 2.0 * np.sqrt(a * b * N)
        delta = sqrtN / (4.0 * r * np.sqrt(a * b))
        ok = np.nonzero(U - x < delta)[0]
        for i in ok:
            bb = int(b[i])
            if math.gcd(a, bb) == 1:
                v = abs(a * q - bb * p)
                hits.append({"a": a, "b": bb, "ab": a * bb, "v": v,
                             "offset_over_width": float((U[i] - x[i]) / delta[i])})
    return hits


def classify_hits(p: int, q: int, r: int, hits: list[dict]) -> dict:
    """Classify hits against the continued fraction of p/q."""
    cf = continued_fraction(p, q)
    conv = convergents(cf)
    conv_index = {hk: n for n, hk in enumerate(conv)}
    inter = intermediate_fractions(cf, r)
    kinds = Counter()
    for h in hits:
        key = (h["a"], h["b"])
        if key in conv_index:
            h["kind"] = "convergent"
            h["n"] = conv_index[key]
            h["next_partial_quotient"] = cf[h["n"] + 1] if h["n"] + 1 < len(cf) else None
        elif key in inter:
            h["kind"] = "intermediate"
            h["n"], h["j"] = inter[key]
        else:
            h["kind"] = "other"
        kinds[h["kind"]] += 1
    # the Dirichlet convergent: last k_n <= sqrt(r q / p)
    bound = math.sqrt(r * q / p)
    n_star = max(n for n, (_, k) in enumerate(conv) if k <= bound)
    h_star, k_star = conv[n_star]
    star_is_hit = any(h["a"] == h_star and h["b"] == k_star for h in hits)
    return {"kinds": dict(kinds), "n_star": n_star, "star_cell": (h_star, k_star), "star_is_hit": star_is_hit,
            "k_next_over_bound": conv[n_star + 1][1] / bound if n_star + 1 < len(conv) else None,
            "cf_len": len(cf)}


def hits_experiment(nbits: int = 40, count: int = 100, seed: int = 23, family: str = "rsa",
                    r_exponent: float = 1 / 3) -> dict:
    counts = Counter()
    kinds = Counter()
    star_hits = 0
    min_ab_is_star = 0
    ratios = []
    offsets = []
    for i in range(count):
        inst = make_semiprime(nbits, family, seed, i)
        r = int(round(float(inst.N) ** r_exponent))
        hits = lehman_hits(inst.N, inst.p, inst.q, r)
        cl = classify_hits(int(inst.p), int(inst.q), r, hits)
        counts[len(hits)] += 1
        kinds.update(cl["kinds"])
        star_hits += cl["star_is_hit"]
        if hits:
            best = min(hits, key=lambda h: h["ab"])
            min_ab_is_star += (best["a"], best["b"]) == cl["star_cell"]
        if cl["k_next_over_bound"] is not None:
            ratios.append(cl["k_next_over_bound"])
        offsets.extend(h["offset_over_width"] for h in hits)
    return {"nbits": nbits, "count": count, "r_exponent": r_exponent,
            "hit_count_distribution": dict(sorted(counts.items())),
            "mean_hits": sum(k * v for k, v in counts.items()) / count,
            "kinds": dict(kinds), "star_is_hit_fraction": star_hits / count,
            "min_ab_hit_is_star_fraction": min_ab_is_star / count,
            "k_next_over_bound_quantiles": [float(x) for x in np.quantile(ratios, [0.1, 0.5, 0.9])] if ratios else None,
            "offset_over_width_mean": float(np.mean(offsets)) if offsets else None}
