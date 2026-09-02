"""E48: Schnorr-type lattice relation finding, measured on the beta scale.

The prime-number lattice: columns b_i = (f_i e_i, round(K ln p_i)) for the factor base p_1..p_n and the target
t = (0, ..., 0, round(K ln N)).  A close vector sum e_i b_i ~ t gives sum e_i ln p_i ~ ln N, i.e. u = prod_{e_i>0} p_i^{e_i}
close to v N with v = prod_{e_i<0} p_i^{-e_i}; the candidate relation is u = u - vN (mod N) and it is useful iff the
residue |u - vN| is smooth over the factor base (a 'fac-relation').  We run LLL (PARI qflll) with the Kannan embedding on
lattices with randomly chosen diagonal scales in {1, 2, 3}, read off the short vectors that use the target exactly once, and record
beta = log|u - vN| / log N and the smoothness of the residue.  This places lattice relation finding on the size-exponent
scale of the smoothness paradigm; it does not test Schnorr's asymptotic claims, only what LLL delivers at these sizes.

Run:  python -m factorlab.experiments.schnorr_lattice --bits 40 --n 20 --trials 200 --out results/e48_schnorr_lattice.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
from sympy import factorint, primerange

from factorlab.gen import make_semiprime


def _pari():
    import cypari2

    return cypari2.Pari()


def largest_prime_factor(n: int) -> int:
    return max(int(x) for x in factorint(n)) if n > 1 else 1


def schnorr_trial(pari, N: int, primes: List[int], scales: List[int], K: int, M_embed: int) -> List[Dict]:
    """One LLL reduction of the embedded prime-number lattice; returns the relations read off the reduced basis."""
    n = len(primes)
    # columns: b_i (i < n): scales[i] at row i, round(K ln p_i) at row n, 0 at row n+1; target column: round(K ln N) at row n, M at row n+1
    cols = []
    for i, p in enumerate(primes):
        col = [0] * (n + 2)
        col[i] = scales[i]
        col[n] = int(round(K * math.log(p)))
        cols.append(col)
    tcol = [0] * (n + 2)
    tcol[n] = int(round(K * math.log(N)))
    tcol[n + 1] = M_embed
    cols.append(tcol)
    mat = pari.matrix(n + 2, n + 1, [cols[j][i] for i in range(n + 2) for j in range(n + 1)])
    T = pari.qflll(mat)
    red = mat * T
    out = []
    for j in range(int(pari.matsize(red)[1])):
        v = [int(red[i, j]) for i in range(n + 2)]
        if abs(v[n + 1]) != M_embed:
            continue  # uses the target 0 or >=2 times
        sgn = -1 if v[n + 1] > 0 else 1  # v = sum e_i b_i - t  => coefficients e_i = v_i / scale_i when v[n+1] = -M
        e = [sgn * v[i] // scales[i] for i in range(n)]
        if any(sgn * v[i] % scales[i] for i in range(n)):
            continue
        u = 1
        vv = 1
        for ei, p in zip(e, primes):
            if ei > 0:
                u *= p ** ei
            elif ei < 0:
                vv *= p ** (-ei)
        R = abs(u - vv * N)
        if R == 0:
            continue
        B = primes[-1]
        lpf = largest_prime_factor(R)
        out.append({"beta": math.log(R) / math.log(N), "smooth_B": lpf <= B, "smooth_B2": lpf <= B * B,
                    "u": int(u), "v": int(vv), "R": int(R),
                    "log2_u": math.log2(u), "log2_v": math.log2(vv) if vv > 1 else 0.0})
    return out


def schnorr_experiment(bits: int, n: int, trials: int, seed: int = 8, K_bits: int = 40) -> Dict:
    pari = _pari()
    sp = make_semiprime(bits, "rsa", seed, 0)
    N = int(sp.N)
    primes = list(primerange(2, 10 ** 6))[:n]
    rng = np.random.default_rng(seed)
    rels: List[Dict] = []
    for _ in range(trials):
        scales = [int(x) for x in rng.integers(1, 4, size=n)]  # random diagonal weights in {1,2,3} (Ducas-style randomisation)
        K = 2 ** K_bits
        rels.extend(schnorr_trial(pari, N, primes, scales, K, M_embed=2 ** (K_bits // 2)))
    betas = np.array([r["beta"] for r in rels]) if rels else np.array([])
    return {
        "bits": bits, "N": N, "n_primes": n, "B": primes[-1], "trials": trials, "relations_read": len(rels),
        "beta_mean": float(betas.mean()) if len(betas) else None,
        "beta_min": float(betas.min()) if len(betas) else None,
        "beta_quantiles": [float(x) for x in np.quantile(betas, [0.1, 0.5, 0.9])] if len(betas) else None,
        "fac_relations_B": int(sum(r["smooth_B"] for r in rels)),
        "fac_relations_B2": int(sum(r["smooth_B2"] for r in rels)),
        "dixon_reference_beta": 1.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", type=int, nargs="+", default=[40])
    ap.add_argument("--n", type=int, nargs="+", default=[20])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", default="results/e48_schnorr_lattice.json")
    args = ap.parse_args()
    rows = []
    for bits in args.bits:
        for n in args.n:
            res = schnorr_experiment(bits, n, args.trials)
            rows.append(res)
            print(f"{bits} bits, n={n} (B={res['B']}): {res['relations_read']} relations from {args.trials} LLL runs; "
                  f"beta mean={res['beta_mean']}, quantiles={res['beta_quantiles']}; fac-relations B-smooth={res['fac_relations_B']}, "
                  f"B^2-smooth={res['fac_relations_B2']}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
