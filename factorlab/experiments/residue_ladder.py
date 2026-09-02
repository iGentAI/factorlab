"""E12: the residue ladder of the random-squares paradigm.

Three classes of residues r with x^2 = r (mod N) drive the rigorous and
heuristic random-squares algorithms:

* Dixon: x uniform in [1, N), r = x^2 mod N, size ~ N (rigorous, uniform);
* Vallee: x in B = {x : x^2 mod N < 4 N^{2/3}}, i.e. x within
  4N^{2/3}/(2 sqrt(kN)) of sqrt(kN) for some k <= N (rigorous quasi-uniform
  generation; L[1/2, 1.515]); the Lehman-cell candidate set of the barrier
  note is the sub-family k <= N^{1/3} of the same windows;
* QS: x = s + y, |y| <= M, r = |x^2 - N| <= 2 M sqrt(N) (heuristic).

For each class this module samples residues, factors them, and reports the
fraction that is B-smooth and (B, B^2)-semismooth against the Dickman
prediction for their size, with and without the local shift.  Residues of
Vallee type have no local shift on average: the multiplier k varies, and
(kN | l) = 1 for half the k, so E[v_l] averages to 1/(l-1).  QS residues
(fixed k = 1) carry the Knuth-Schroeppel-type shift of E11.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
from sympy import factorint

from ..gen import make_semiprime
from ..numth import mpz, isqrt, isqrt_ceil
from ..algorithms.qs import expected_valuation_shift


def _top_two(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    ps = sorted(((int(p), int(e)) for p, e in factorint(int(n)).items()), reverse=True)
    l1 = ps[0][0]
    if ps[0][1] >= 2:
        return l1, l1
    return l1, (ps[1][0] if len(ps) > 1 else 1)


def sample_dixon(rng, N):
    x = rng.randrange(1, int(N))
    return x, int((mpz(x) * x) % N)


def sample_vallee(rng, N, kmax=None):
    """x with x^2 - kN in [0, 4 N^{2/3}) for k ~ k^{-1/2} on [1, kmax]; returns (x, r)."""
    N = int(N)
    kmax = N if kmax is None else int(kmax)
    bound = 4.0 * N ** (2.0 / 3.0)
    while True:
        u = rng.random()
        k = int(math.ceil((u * math.sqrt(kmax)) ** 2))
        k = min(max(k, 1), kmax)
        kN = mpz(k) * N
        x0 = int(isqrt_ceil(kN))
        w = bound / (2.0 * math.sqrt(float(kN)))
        cnt = int(math.floor(math.sqrt(float(kN)) + w)) - x0 + 1
        if cnt <= 0:
            continue
        x = x0 + rng.randrange(cnt)
        r = int(mpz(x) * x - kN)
        if 0 <= r < bound:
            return x, r


def sample_qs(rng, N, M):
    s = isqrt(mpz(N)) + 1
    y = rng.randrange(-int(M), int(M) + 1)
    x = int(s) + y
    return x, abs(int(mpz(x) * x - N))


def residue_ladder(nbits: int = 48, count: int = 100, samples: int = 200, c: float = 1 / 6, seed: int = 71,
                   family: str = "rsa") -> dict:
    from .smooth_profiles import dickman_rho, semismooth_G
    rng = random.Random(seed)
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    B = 2.0 ** (c * nbits)
    B2 = B * B
    M_big = int(round(2.0 ** (nbits / 6)))   # residues ~ 2 M sqrt N ~ N^{2/3}
    M_small = int(round(2.0 ** (nbits / 12)))  # residues ~ N^{7/12}
    classes = {
        "dixon": lambda rng, inst: sample_dixon(rng, inst.N),
        "vallee_k_le_N": lambda rng, inst: sample_vallee(rng, inst.N),
        "lehman_k_le_N13": lambda rng, inst: sample_vallee(rng, inst.N, kmax=int(round(float(inst.N) ** (1 / 3)))),
        "qs_M_N16": lambda rng, inst: sample_qs(rng, inst.N, M_big),
        "qs_M_N112": lambda rng, inst: sample_qs(rng, inst.N, M_small),
    }
    out = {"nbits": nbits, "count": count, "samples_per_modulus": samples, "c": c, "B_bits": c * nbits, "classes": {}}
    for name, sampler in classes.items():
        logs, smooth, semi, shifts = [], [], [], []
        for inst in insts:
            shift = expected_valuation_shift(inst.N, int(B)) if name.startswith("qs") else 0.0
            for _ in range(samples):
                x, r = sampler(rng, inst)
                if r <= 1:
                    continue
                l1, l2 = _top_two(r)
                logs.append(math.log(r))
                smooth.append(l1 <= B)
                semi.append(l2 <= B and l1 <= B2)
                shifts.append(shift)
        logs = np.array(logs)
        mean_log = float(logs.mean())
        mean_shift = float(np.mean(shifts))
        u_plain = mean_log / math.log(B)
        u_shift = (mean_log - mean_shift) / math.log(B)
        alpha_p, alpha_s = 1.0 / u_plain, 1.0 / u_shift
        out["classes"][name] = {
            "n": int(len(logs)), "mean_log2_r": mean_log / math.log(2), "mean_log2_r_over_log2_N": mean_log / (nbits * math.log(2)),
            "smooth_obs": float(np.mean(smooth)), "smooth_pred_plain": dickman_rho(u_plain), "smooth_pred_shift": dickman_rho(u_shift),
            "semismooth_obs": float(np.mean(semi)),
            "semismooth_pred_plain": semismooth_G(alpha_p, min(2 * alpha_p, 1.0)) if alpha_p < 0.5 else 1.0,
            "semismooth_pred_shift": semismooth_G(alpha_s, min(2 * alpha_s, 1.0)) if alpha_s < 0.5 else 1.0,
            "mean_shift_nats": mean_shift,
        }
    return out
