"""Uniform-slack sensitivity of the prefix-volume detection crossings.

In the profile class with uniform slack eps (every block inequality (A_i) relaxed by eps), the prefix-volume floor gives

    log GH_b(L / F_{d-b}) <= l^tight_{d-b+1}(d, b, eps; S) + eps,

so the detection condition log(sigma sqrt b) <= log GH_b can hold for a profile of the class only if it holds against that bound.  This
module screens, in double precision (a pre-screen, not a certificate), the least b in a range for which some admissible m passes, and
certifies, in ball arithmetic with directed comparisons, that every (b, m) below a stated crossing fails and that the crossing passes.

Usage:
  python -m latticelab.uniform_slack --screen --sets kyber512 kyber768 kyber1024 --eps 0.01 0.03
  python -m latticelab.uniform_slack --certify --sets kyber512 --eps 0.01 [--b-lo 403 --b-star 411] --out-prefix results/lattice_uniform_slack
  (without --b-star the crossing is taken from the double-precision screen; without --b-lo the scan starts at min(b_GSA - 3, b* - 8))
"""
from __future__ import annotations

import argparse
import json
import math
import time
from fractions import Fraction

from flint import arb, ctx, fmpq

from latticelab.profile_floor import tight_entry, tight_entry_float

Q = 3329
N_RING = 256
SETS = {"kyber512": {"k": 2, "eta1": 3, "b_gsa": 406, "b_star": 417},
        "kyber768": {"k": 3, "eta1": 2, "b_gsa": 624, "b_star": 642},
        "kyber1024": {"k": 4, "eta1": 2, "b_gsa": 874, "b_star": 900}}


def lhs_float(b: int, eta1: int) -> float:
    return 0.5 * math.log(eta1 / 2) + 0.5 * math.log(b)


def lhs_ball(b: int, eta1: int, prec: int) -> arb:
    ctx.prec = prec
    return (arb(eta1) / 2).log() / 2 + arb(b).log() / 2


def screen(name: str, eps: float, b_lo: int, b_hi: int) -> dict:
    """Least b in [b_lo, b_hi] with some admissible m passing the uniform-slack bound, in double precision."""
    cfg = SETS[name]
    k, eta1 = cfg["k"], cfg["eta1"]
    logq = math.log(Q)
    for b in range(b_lo, b_hi + 1):
        lhs = lhs_float(b, eta1)
        best = None
        for m in range(0, (k + 1) * N_RING + 1):
            d = m + k * N_RING + 1
            if 2 * b >= d + 1:
                continue
            bound = tight_entry_float(d, b, d - b + 1, eps, m * logq) + eps
            if bound - lhs > (best or (None, -1e9))[1]:
                best = (m, bound - lhs)
            if lhs <= bound:
                return {"set": name, "eps": eps, "crossing": b, "m": m, "d": d, "margin": bound - lhs, "b_range": [b_lo, b_hi]}
    return {"set": name, "eps": eps, "crossing": None, "b_range": [b_lo, b_hi]}


def certify(name: str, eps: Fraction, b_lo: int, b_star: int, prec: int, out_path: str, log) -> dict:
    """Every (b, m) with b in [b_lo, b_star - 1] fails rigorously, and some m passes rigorously at b_star, under uniform slack eps."""
    cfg = SETS[name]
    k, eta1 = cfg["k"], cfg["eta1"]
    ctx.prec = prec
    eps_arb = arb(fmpq(eps.numerator, eps.denominator))
    logq = arb(Q).log()
    out = {"set": name, "eps": str(eps), "prec": prec, "b_range": [b_lo, b_star - 1], "b_star": b_star, "rows": [], "crossing": None}
    t0 = time.time()
    for b in list(range(b_lo, b_star - 1 + 1)) + [b_star]:
        lhs = lhs_ball(b, eta1, prec)
        fails = 0
        admissible = 0
        non_failing = []
        passing = []
        for m in range(0, (k + 1) * N_RING + 1):
            d = m + k * N_RING + 1
            if 2 * b >= d + 1:
                continue
            admissible += 1
            e = tight_entry(d, b, d - b + 1, eps, 0, prec)
            bound = e + m * logq / d + eps_arb
            if lhs.lower() > bound.upper():
                fails += 1
            else:
                non_failing.append(m)
                if lhs.upper() < bound.lower():
                    passing.append(m)
        row = {"b": b, "admissible_m": admissible, "rigorous_fail": fails, "non_failing_m": non_failing, "rigorous_pass_m": passing,
               "seconds": time.time() - t0}
        if b == b_star:
            out["crossing"] = {"b": b, "passing_m": passing, "passes": bool(passing)}
        else:
            out["rows"].append(row)
        json.dump(out, open(out_path, "w"), indent=1)
        log(f"{name} eps={eps} b={b}: admissible {admissible}, rigorous fails {fails}, non-failing {non_failing[:5]}{'...' if len(non_failing) > 5 else ''}, passes {len(passing)} ({time.time() - t0:.0f}s)")
    out["all_fail_below"] = all(r["rigorous_fail"] == r["admissible_m"] for r in out["rows"])
    json.dump(out, open(out_path, "w"), indent=1)
    log(f"{name} eps={eps}: every b in [{b_lo}, {b_star - 1}] fails at every admissible m: {out['all_fail_below']}; b* = {b_star} passes: {out['crossing']['passes']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["kyber512", "kyber768", "kyber1024"])
    ap.add_argument("--eps", nargs="+", type=str, default=["0.01", "0.03"])
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--b-lo", type=int, default=None)
    ap.add_argument("--b-star", type=int, default=None)
    ap.add_argument("--prec", type=int, default=256)
    ap.add_argument("--out-prefix", default="results/lattice_uniform_slack")
    args = ap.parse_args()
    if args.screen:
        res = []
        for name in args.sets:
            cfg = SETS[name]
            for e in args.eps:
                r = screen(name, float(e), cfg["b_gsa"] - 20, cfg["b_star"] + 5)
                res.append(r)
                print(f"{name} eps={e}: least passing b in [{r['b_range'][0]}, {r['b_range'][1]}] (double precision) = {r['crossing']} at m={r.get('m')}, d={r.get('d')}", flush=True)
        json.dump(res, open(args.out_prefix + "_screen.json", "w"), indent=1)
    if args.certify:
        for name in args.sets:
            cfg = SETS[name]
            for e in args.eps:
                eps = Fraction(e)
                b_star = args.b_star
                if b_star is None:
                    scr = screen(name, float(e), cfg["b_gsa"] - 40, cfg["b_star"] + 5)
                    b_star = scr["crossing"]
                    if b_star is None:
                        raise SystemExit(f"{name} eps={e}: no crossing found in the screened range; give --b-star explicitly")
                b_lo = args.b_lo if args.b_lo is not None else min(cfg["b_gsa"] - 3, b_star - 8)
                if not (1 < b_lo <= b_star):
                    raise SystemExit(f"{name} eps={e}: need 1 < b_lo <= b_star, got b_lo={b_lo}, b_star={b_star}")
                path = f"{args.out_prefix}_{name}_eps{e.replace('.', 'p')}.json"
                certify(name, eps, b_lo, b_star, args.prec, path, lambda s: print(s, flush=True))
    print("UNIFORM_SLACK_DONE")
