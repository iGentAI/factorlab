"""E33: the k-only candidate set of Lehman/Hart -- 'batched squares'.

Collapsing the cells (a, b) with ab = k, Lehman's test becomes: is Delta_k = ceil(2 sqrt(kN))^2 - 4kN a perfect
square for some k <= r?  With the single candidate x = c_k := ceil(2 sqrt(kN)) per k this is Hart's one-line
factoring (OLF) in Lehman's normalisation; exclusion 7 of notes_barrier.md asks whether the r squareness tests can
be batched in o(r) time.  This module records what the candidate set looks like.

Lemma (no false positives; every hit is a cell) [proven].  Let 1 <= k <= N/16, N = pq >= 39, and Delta_k = t^2.
Then (c_k - t)(c_k + t) = 4kN, and 1 < gcd(c_k - t, N) < N.  Proof: Delta_k < 4 sqrt(kN) + 1 gives
t < 2 (kN)^{1/4} + 1, so c_k - t > 2 sqrt(kN) - 2 (kN)^{1/4} - 1 > 4k for k <= N/16 (2 sqrt(kN) - 4k =
2 sqrt k (sqrt N - 2 sqrt k) >= sqrt(kN) >= 2 (kN)^{1/4} + 1 once kN >= 39); if gcd(c_k - t, N) = 1 then
c_k - t | 4k, contradicting c_k - t > 4k; if N | c_k - t then N <= c_k < 2 sqrt(kN) + 1, i.e. k > N/4 - o(1).
Hence one of c_k -+ t is divisible by exactly one prime: {c_k - t, c_k + t} = {2 b p, 2 a q} with ab = k (the two
factors have equal parity and product 4kN; which of the two carries p depends on the sign of aq - bp, e.g.
N = 101 * 199, k = 2 has c - t = 2 q and c + t = 4 p), so c_k = aq + bp and t = |aq - bp| -- every hit is a
Lehman cell (a, b) with (sqrt(aq) - sqrt(bp))^2 = aq + bp - 2 sqrt(kN) < 1.

Hit count [first-order lattice-area model; its ensemble mean verified].  The hits by k <= r are the lattice points
(a, b) with ab <= r and |aq - bp| < sqrt(aq) + sqrt(bp); for fixed b the admissible a fill a real interval of
length 4 sqrt(bp)/q, and summing the lengths over b <= sqrt(rq/p) gives
    lambda(r) = (8/3) r^{3/4} N^{-1/4}      (leading coefficient independent of the balance p/q),
i.e. 8/3 at r = N^{1/3}.  Treating Delta_k instead as a random integer of its size (probability 1/(2 sqrt Delta)
of being a square, eps_k uniform) gives sum_k 1/(2 (kN)^{1/4}) ~ (2/3) r^{3/4} N^{-1/4}: the two first-order
models differ by a factor four in their leading coefficients.  The first hit is at k ~ N^{1/3}; its tail is
much heavier than Poisson in the samples: a modulus whose continued fraction has a convergent denominator well
below N^{1/6} followed by a large partial quotient has no hit until far beyond N^{1/3} (Gauss-Kuzmin, heuristic),
which is what Lehman's window of width sqrt N/(4 r sqrt(ab)) per cell repairs.

Structure test: the linear complexity of Delta_k mod ell over k <= r (Berlekamp-Massey), against the E2 controls.
"""

from __future__ import annotations

import math

import gmpy2
import numpy as np

from .separable import berlekamp_massey
from .sidon_scaling import _ceil_2sqrt


def delta_k(N: int, k: int) -> tuple[int, int]:
    c = _ceil_2sqrt(k, N)
    return c, c * c - 4 * k * N


def olf_hits(N: int, r: int, p: int | None = None, q: int | None = None) -> list[dict]:
    """All k <= r with Delta_k a perfect square, each with the factor gcd(c_k - t, N), whether it is proper, and
    (when p, q are given) the cell (a, b) with c_k = aq + bp, ab = k."""
    out = []
    N = gmpy2.mpz(N)
    for k in range(1, r + 1):
        v = 4 * k * N
        c = gmpy2.isqrt(v)
        if c * c < v:
            c += 1
        D = c * c - v
        if gmpy2.is_square(D):
            t = gmpy2.isqrt(D)
            g = int(gmpy2.gcd(c - t, N))
            rec = {"k": k, "c": int(c), "t": int(t), "gcd": g, "proper": 1 < g < int(N)}
            if p is not None and q is not None:
                cell = None
                for a in range(1, k + 1):
                    if k % a == 0 and a * q + (k // a) * p == int(c):
                        cell = (a, k // a)
                        break
                rec["cell"] = cell
            out.append(rec)
    return out


def lattice_parameter(N: int, r: int) -> float:
    """lambda(r) = (8/3) r^{3/4} N^{-1/4}: the expected number of cells (a, b), ab <= r, with |sqrt(aq) - sqrt(bp)| < 1."""
    return (8 / 3) * r ** 0.75 * float(N) ** -0.25


def random_square_parameter(N: int, r: int) -> float:
    """sum_{k <= r} 1/(2 (kN)^{1/4}): the expectation if Delta_k were a random integer of its size."""
    ks = np.arange(1, r + 1, dtype=np.float64)
    return float(np.sum(0.5 / (ks * float(N)) ** 0.25))


def olf_experiment(bits: int, count: int, r_mult: float = 8.0, seed: int = 7) -> dict:
    """Hit counts by k <= N^{1/3} and by k <= r_mult N^{1/3}, the first-hit distribution, and the two model parameters."""
    from ..gen import make_semiprime

    hits_13, hits_r, firsts, improper, non_cell = [], [], [], 0, 0
    for idx in range(count):
        sp = make_semiprime(bits, "rsa", seed, idx)
        N, p, q = int(sp.N), int(sp.p), int(sp.q)
        r13 = int(round(N ** (1 / 3)))
        r = int(round(r_mult * N ** (1 / 3)))
        hs = olf_hits(N, r, p, q)
        improper += sum(1 for h in hs if not h["proper"])
        non_cell += sum(1 for h in hs if h["cell"] is None)
        hits_13.append(sum(1 for h in hs if h["k"] <= r13))
        hits_r.append(len(hs))
        firsts.append(hs[0]["k"] / N ** (1 / 3) if hs else math.inf)
    N0 = int(make_semiprime(bits, "rsa", seed, 0).N)
    firsts = np.array(firsts)
    h13, hr = np.array(hits_13), np.array(hits_r)
    cs = [c for c in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0) if c <= r_mult]
    lam13, lamr = lattice_parameter(N0, int(round(N0 ** (1 / 3)))), lattice_parameter(N0, int(round(r_mult * N0 ** (1 / 3))))
    return {"bits": bits, "count": count, "r_mult": r_mult, "improper_hits": improper, "hits_not_cells": non_cell,
            "mean_hits_N13": float(h13.mean()), "se_hits_N13": float(h13.std() / math.sqrt(count)), "lambda_lattice_N13": lam13,
            "lambda_random_N13": random_square_parameter(N0, int(round(N0 ** (1 / 3)))),
            "mean_hits_r": float(hr.mean()), "se_hits_r": float(hr.std() / math.sqrt(count)), "lambda_lattice_r": lamr,
            "lambda_random_r": random_square_parameter(N0, int(round(r_mult * N0 ** (1 / 3)))),
            "var_over_mean_r": float(hr.var() / hr.mean()) if hr.mean() else None,
            "frac_no_hit_N13": float(np.mean(h13 == 0)), "poisson_no_hit_N13": math.exp(-lam13),
            "first_hit_tail": {c: float(np.mean(firsts > c)) for c in cs},
            "first_hit_tail_poisson": {c: math.exp(-(8 / 3) * c ** 0.75) for c in cs},
            "median_first_over_N13": float(np.median(firsts[np.isfinite(firsts)])) if np.isfinite(firsts).any() else None}


def convergent_denominators(p: int, q: int) -> list[int]:
    """Denominators of the convergents of p/q."""
    h0, h1, k0, k1 = 0, 1, 1, 0
    a, b = p, q
    ks = []
    while b:
        t = a // b
        a, b = b, a - t * b
        h0, h1 = h1, t * h1 + h0
        k0, k1 = k1, t * k1 + k0
        ks.append(k1)
    return ks


def tail_diagnostic(bits: int, count: int, c: float = 4.0, seed: int = 7) -> dict:
    """For each modulus: the largest ratio b_{n+1}/b_n between consecutive convergent denominators of p/q with
    b_n <= 2 N^{1/6}.  Reported separately for the moduli whose first OLF hit exceeds c N^{1/3} (the tail) and the
    others; the heuristic of the module docstring predicts a much larger jump in the tail."""
    from ..gen import make_semiprime

    tail, rest = [], []
    for idx in range(count):
        sp = make_semiprime(bits, "rsa", seed, idx)
        N, p, q = int(sp.N), int(sp.p), int(sp.q)
        r = int(round(c * N ** (1 / 3)))
        hs = olf_hits(N, r)
        ks = convergent_denominators(p, q)
        lim = 2 * N ** (1 / 6)
        jumps = [ks[i + 1] / ks[i] for i in range(len(ks) - 1) if ks[i] <= lim]
        jmax = max(jumps) if jumps else 1.0
        (tail if not hs else rest).append(jmax)
    return {"bits": bits, "c": c, "tail_count": len(tail), "rest_count": len(rest),
            "tail_jumps": sorted(tail), "rest_jumps": sorted(rest), "tail_min_jump": min(tail) if tail else None,
            "rest_median_jump": float(np.median(rest)) if rest else None,
            "rest_frac_jump_ge_tail_min": float(np.mean(np.array(rest) >= min(tail))) if tail and rest else None}


def delta_linear_complexity(N: int, r: int, ell: int) -> dict:
    seq = [int(delta_k(N, k)[1] % ell) for k in range(1, r + 1)]
    Lc = berlekamp_massey(seq, ell)
    low = [(pow(3, k, ell) + pow(5, k, ell)) % ell for k in range(1, r + 1)]
    return {"r": r, "ell": ell, "linear_complexity": Lc, "generic": r / 2, "ratio": Lc / (r / 2),
            "control_lfsr_order2": berlekamp_massey(low, ell)}


if __name__ == "__main__":  # python -m factorlab.experiments.olf_squares [--quick]
    import json
    import os
    import sys

    from ..bench import RESULTS_DIR
    from ..gen import make_semiprime

    quick = "--quick" in sys.argv
    res = {"olf": [], "linear_complexity": []}
    for bits, count, mult in (((32, 100, 8.0), (40, 60, 8.0)) if quick else ((32, 300, 64.0), (40, 200, 16.0), (48, 60, 8.0))):
        z = olf_experiment(bits, count, r_mult=mult)
        res["olf"].append(z)
        print(f"{bits} bits, {count} moduli: improper hits {z['improper_hits']}, hits that are not cells {z['hits_not_cells']} | "
              f"hits by N^(1/3): {z['mean_hits_N13']:.3f} +- {z['se_hits_N13']:.3f} (lattice {z['lambda_lattice_N13']:.3f}, random-square {z['lambda_random_N13']:.3f}); "
              f"by {mult:g} N^(1/3): {z['mean_hits_r']:.2f} +- {z['se_hits_r']:.2f} (lattice {z['lambda_lattice_r']:.2f}, random-square {z['lambda_random_r']:.2f}, var/mean {z['var_over_mean_r']:.2f}) | "
              f"no hit by N^(1/3): {z['frac_no_hit_N13']:.3f} (Poisson {z['poisson_no_hit_N13']:.3f}) | median first hit / N^(1/3) = {z['median_first_over_N13']:.3f}")
        print("      P[first hit > c N^(1/3)]: " + ", ".join(f"c={c:g}: {z['first_hit_tail'][c]:.3f} (Poisson {z['first_hit_tail_poisson'][c]:.1e})" for c in z["first_hit_tail"]))
    N = int(make_semiprime(48, "rsa", 7, 0).N)
    for r, ell in ((512, 1009), (1024, 65537)):
        z = delta_linear_complexity(N, r, ell)
        res["linear_complexity"].append(z)
        print(f"linear complexity of Delta_k mod {ell}, k <= {r}: {z['linear_complexity']} (generic {z['generic']:.0f}; LFSR control {z['control_lfsr_order2']})")
    res["tail"] = []
    for bits, count, c in (((40, 60, 2.0),) if quick else ((40, 200, 4.0), (48, 60, 4.0))):
        z = tail_diagnostic(bits, count, c)
        res["tail"].append(z)
        fmt = lambda v, f: (f % v) if v is not None else "n/a"   # noqa: E731
        print(f"{bits} bits: first hit > {c:g} N^(1/3) in {z['tail_count']}/{count}; largest convergent-denominator jump below 2 N^(1/6): "
              f"tail {[round(x, 1) for x in z['tail_jumps']]} (min {fmt(z['tail_min_jump'], '%.1f')}), rest median {fmt(z['rest_median_jump'], '%.2f')}, "
              f"fraction of rest with jump >= tail min {fmt(z['rest_frac_jump_ge_tail_min'], '%.3f')}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e33_olf_squares.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
