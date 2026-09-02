"""Exact certification of the profile statements of the applicability audit.

For an integer basis B (rows) the Gram-Schmidt squared norms are r_k = det(G_k)/det(G_{k-1}) with G_k the leading k x k principal
minor of the Gram matrix G = B B^T, exact rationals.  Hence the profile l_k = (1/2) log r_k, the log-volume S = (1/2) log det G, the
zero-slack floor S/d + h_{d,beta}(0) (exact dual multipliers, ball-arithmetic constants L(n)) and the head deficit
nu_1 = L(beta) + avg_{k <= beta} l_k - l_1 are all enclosed in balls from exact rational inputs, with no enumeration and none of the
non-rigorous floating-point Gram-Schmidt data used for enumeration: the statement
'the basis lies below the zero-slack floor' is decided rigorously.  The positional deficits nu_i and the weighted deficit sum y_i nu_i are
enclosed likewise.

Usage: python -m latticelab.certify_audit --archive results/lattice_l6_strict.json --keys strict,75,30,31 ... --out results/....json
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Dict, List

from flint import arb, ctx, fmpq, fmpz_mat

from latticelab.profile_floor import _arb_log_chat, block_sizes, dual_certificate


def exact_gso_norms(rows: List[List[int]]) -> List[fmpq]:
    """Exact squared Gram-Schmidt norms of an integer basis given by rows, via leading principal minors of the Gram matrix."""
    d = len(rows)
    B = fmpz_mat(rows)
    G = B * B.transpose()
    dets = []
    for k in range(1, d + 1):
        Gk = fmpz_mat([[int(G[i, j]) for j in range(k)] for i in range(k)])
        dets.append(Gk.det())
    r = [fmpq(dets[0], 1)]
    for k in range(1, d):
        r.append(fmpq(dets[k], 1) / fmpq(dets[k - 1], 1))
    return r


def certify_basis(rows: List[List[int]], beta: int, prec: int = 256) -> Dict:
    d = len(rows)
    ctx.prec = prec
    r = exact_gso_norms(rows)
    ell = [arb(x).log() / 2 for x in r]
    S = sum(ell[1:], ell[0])
    y, z = dual_certificate(d, beta)
    bs = block_sizes(d, beta)
    h0 = arb(0)
    for i in range(d - 1):
        yi = Fraction(y[i])
        h0 += arb(fmpq(yi.numerator, yi.denominator)) * _arb_log_chat(bs[i], prec)
    floor = S / d + h0
    head_gap = ell[0] - floor
    # positional deficits nu_i = L(beta_i) + avg over block i of ell - ell_i, i = 1..d-1
    nus = []
    for i in range(d - 1):
        n = bs[i]
        avg = sum(ell[i + 1:i + n], ell[i]) / n
        nus.append(_arb_log_chat(n, prec) + avg - ell[i])
    weighted = arb(0)
    for i in range(d - 1):
        yi = Fraction(y[i])
        weighted += arb(fmpq(yi.numerator, yi.denominator)) * nus[i]

    def ball(x: arb) -> Dict:
        """Lossless record of an enclosure: the exact binary lower and upper bounds as rationals (outward-valid by construction, since
        arb.lower()/upper() are exact floating-point numbers enclosing the ball), the full Arb string, and an approximate midpoint
        for display only."""
        def exact(v: arb) -> str:
            m, e = v.man_exp()
            fr = Fraction(int(m)) * (Fraction(2) ** int(e))
            return f"{fr.numerator}/{fr.denominator}"
        return {"lower": exact(x.lower()), "upper": exact(x.upper()), "arb": str(x), "approx_mid": float(x.mid())}

    return {"d": d, "beta": beta, "prec": prec, "log_vol": ball(S), "ell_1": ball(ell[0]), "h0": ball(h0), "floor": ball(floor),
            "head_minus_floor": ball(head_gap), "below_floor_rigorously": bool(head_gap.upper() < 0),
            "nu_1": ball(nus[0]), "y_1": float(Fraction(y[0])), "weighted_deficit": ball(weighted),
            "rest_weighted_deficit": ball(weighted - arb(fmpq(Fraction(y[0]).numerator, Fraction(y[0]).denominator)) * nus[0])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="results/lattice_l6_strict.json")
    ap.add_argument("--keys", nargs="+", default=["strict,75,30,31", "strict,75,30,32", "strict,75,30,33",
                                                  "strict,100,40,31", "strict,100,40,32", "strict,100,40,33"])
    ap.add_argument("--prec", type=int, default=256)
    ap.add_argument("--out", default="results/lattice_audit_heads_certified.json")
    args = ap.parse_args()
    arch = json.load(open(args.archive))
    out = {}
    for key in args.keys:
        rows = arch["bases"][key]
        beta = int(key.split(",")[2])
        res = certify_basis(rows, beta, args.prec)
        out[key] = res
        print(f"{key}: ell_1 - floor = {res['head_minus_floor']['arb'][:40]} below={res['below_floor_rigorously']}; "
              f"nu_1 = {res['nu_1']['arb'][:32]}; weighted deficit = {res['weighted_deficit']['arb'][:32]}; rest = {res['rest_weighted_deficit']['arb'][:32]}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("AUDIT_CERTIFIED")
