"""Head-slack robustness of the prefix-volume detection crossings.

In the profile class with slack eps_1 at the head only (every other block inequality tight or satisfied), the prefix identity
P_m(l) = P_m(tight) - sum_i w_i^{(m)} nu_i with nu_1 <= eps_1 and nu_i <= 0 (i >= 2) gives P_m(l) >= P_m(tight) - w_1^{(m)} eps_1, and the
first multiplier has the closed form w_1^{(m)} = (d - m) beta / (d (beta - 1)) (coordinate 1 of 1_{<= m} = sum_i w_i a_i + (m/d) 1, no
earlier block covering position 1).  With m = d - b the detection bound becomes

    log GH_b(L / F_{d-b}) <= l^tight_{d-b+1} + b eps_1 / (d (b - 1)).

This script re-decides, for each parameter set and every b in [b_GSA - 3, b* - 1] and every admissible m, whether
log(sigma sqrt b) > l^tight_{d-b+1} + b eps_1/(d(b-1)) rigorously (ball arithmetic, directed), and records any (b, m) that no longer fails.
Results are appended per b to the JSON so a long run can be read while it proceeds.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from fractions import Fraction

from flint import arb, ctx, fmpq

from latticelab.profile_floor import tight_entry

Q = 3329
N_RING = 256
SETS = {"kyber512": {"k": 2, "eta1": 3, "b_gsa": 406, "b_star": 417},
        "kyber768": {"k": 3, "eta1": 2, "b_gsa": 624, "b_star": 642},
        "kyber1024": {"k": 4, "eta1": 2, "b_gsa": 874, "b_star": 900}}


def lhs_ball(b: int, eta1: int, prec: int) -> arb:
    ctx.prec = prec
    return (arb(eta1) / 2).log() / 2 + arb(b).log() / 2


def decide_set(name: str, eps1: Fraction, prec: int, out_path: str, log) -> dict:
    cfg = SETS[name]
    k, eta1, b_lo, b_star = cfg["k"], cfg["eta1"], cfg["b_gsa"] - 3, cfg["b_star"]
    res = {"set": name, "eps1": str(eps1), "prec": prec, "b_range": [b_lo, b_star - 1], "rows": []}
    for b in range(b_lo, b_star):
        t0 = time.time()
        n_fail = n_dom = 0
        non_failing = []
        for m in range(0, (k + 1) * N_RING + 1):
            d = m + k * N_RING + 1
            if not (2 * b < d + 1):
                continue
            n_dom += 1
            ctx.prec = prec
            S = arb(m) * arb(Q).log()
            e = tight_entry(d, b, d - b + 1, 0, 0, prec)
            ball = e["ball"] if isinstance(e, dict) else e
            ball = ball + S / d  # tight_entry at log_vol 0 shifted by the volume term S/d (the tight profile is affine in S)
            fr = Fraction(b, 1) * eps1 / (d * (b - 1))
            slack = arb(fmpq(fr.numerator, fr.denominator))
            rhs = ball + slack
            lhs = lhs_ball(b, eta1, prec)
            if lhs.lower() > rhs.upper():
                n_fail += 1
            else:
                non_failing.append(m)
        row = {"b": b, "admissible_m": n_dom, "rigorous_fail": n_fail, "n_non_failing": len(non_failing), "non_failing_m": non_failing, "seconds": round(time.time() - t0, 1)}
        res["rows"].append(row)
        log(f"{name} eps1={eps1} b={b}: {n_fail}/{n_dom} admissible m fail rigorously; non-failing: {non_failing[:5]} [{row['seconds']}s]")
        json.dump(res, open(out_path, "w"), indent=1)
    res["all_fail"] = all(r["rigorous_fail"] == r["admissible_m"] for r in res["rows"])
    json.dump(res, open(out_path, "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["kyber512", "kyber768", "kyber1024"])
    ap.add_argument("--eps1", default="1")
    ap.add_argument("--prec", type=int, default=256)
    ap.add_argument("--out-prefix", default="results/lattice_head_slack_")
    args = ap.parse_args()
    eps1 = Fraction(args.eps1)
    summary = {}
    for name in args.sets:
        r = decide_set(name, eps1, args.prec, f"{args.out_prefix}{name}.json", lambda s: print(s, flush=True))
        summary[name] = r["all_fail"]
        print(f"{name}: every b in [{r['b_range'][0]}, {r['b_range'][1]}] fails at every admissible m under head slack {eps1}: {r['all_fail']}", flush=True)
    print("HEAD_SLACK_DONE", summary)
