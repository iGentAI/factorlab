"""Driver for D1/N1/N2 experiments; writes JSON to results/ and prints a summary.

    python -m factorlab.experiments.run_d1_n1_n2 [--quick]
"""

from __future__ import annotations

import argparse
import json
import os

from ..bench import run_suite, summarize, print_summary, RESULTS_DIR
from ..gen import make_semiprime
from . import hull, frobenius_defect, cf_dag


def d1(quick: bool):
    print("== D1 hull locator: exponent and divisor-vertex rank ==")
    bits = [24, 32, 40, 48, 56] if quick else [24, 32, 40, 48, 56, 64, 72]
    recs = run_suite("hull_locator", bits, count=4, family="balanced", seed=0, experiment="d1_hull",
                     verbose=True, timeout_per_instance=60)
    print_summary(summarize(recs))
    # rank statistics on moderate sizes
    stats = []
    for nb in ([32, 40] if quick else [32, 40, 48]):
        for i in range(4):
            inst = make_semiprime(nb, "balanced", 1, i)
            s = hull.hull_statistics(inst.N, int(inst.p))
            stats.append({"nbits": nb, **s})
            if s["divisor_index"] is not None:
                print(f"  {nb}b  vertices={s['n_vertices']:6d} mean_dx={s['mean_dx']:.1f}  "
                      f"divisor dx_prev/dx_next={s['divisor_features']['dx_prev']}/{s['divisor_features']['dx_next']}  "
                      f"rank_small(dx_min)={s['rank_small']['dx_min']} rank_large(dx_min)={s['rank_large']['dx_min']} "
                      f"rank_small(det)={s['rank_small']['det']}")
    with open(os.path.join(RESULTS_DIR, "d1_hull_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    return stats


def n1(quick: bool):
    print("== N1 Frobenius-defect gcd leakage ==")
    out = []
    sizes = [24, 32] if quick else [24, 32, 40]
    r_values = [5, 7, 11, 13, 16, 17, 19, 23, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 128]
    a_values = [1, 2, 3, 5, 7]
    for nb in sizes:
        mism = leak = tot = 0
        enriched = 0
        for i in range(6 if quick else 10):
            inst = make_semiprime(nb, "balanced", 2, i)
            rows = frobenius_defect.experiment(inst.N, inst.p, inst.q, r_values, a_values)
            for row in rows:
                row.update({"nbits": nb, "index": i})
            out.extend(rows)
            mism += sum(r["mismatch"] for r in rows)
            leak += sum(r["leaked"] for r in rows)
            tot += len(rows)
            enriched += sum(1 for r in rows if r["gcd_r_pm1"] > 1 or r["gcd_r_qm1"] > 1)
        print(f"  {nb}b: trials={tot} degree-mismatch={mism} leaked={leak}  (r | p-1 or q-1 in {enriched} trials)")
        # conditional rates
        sub = [r for r in out if r["nbits"] == nb]
        g1 = [r for r in sub if r["gcd_r_pm1"] > 1 or r["gcd_r_qm1"] > 1]
        g0 = [r for r in sub if not (r["gcd_r_pm1"] > 1 or r["gcd_r_qm1"] > 1)]
        if g1:
            print(f"      P(mismatch | gcd(r,p-1)>1 or gcd(r,q-1)>1) = {sum(r['mismatch'] for r in g1)/len(g1):.3f}")
        if g0:
            print(f"      P(mismatch | gcd = 1 both)                  = {sum(r['mismatch'] for r in g0)/len(g0):.3f}")
    with open(os.path.join(RESULTS_DIR, "n1_frobenius_defect.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def n2(quick: bool):
    print("== N2 continued-fraction state sharing over sqrt(kN) ==")
    out = []
    for nb in ([40, 60] if quick else [40, 60, 80]):
        inst = make_semiprime(nb, "balanced", 3, 0)
        for K in ([1000, 10000] if quick else [1000, 10000, 100000]):
            r = cf_dag.prefix_sharing(inst.N, K, depth=8)
            r["nbits"] = nb
            r["entropy_a1_bits"] = cf_dag.quotient_entropy(inst.N, K, 1)
            out.append(r)
            print(f"  {nb}b K={K:6d}: distinct prefixes by depth {r['distinct_prefixes']}  intervals {r['interval_count']}  H(a1)={r['entropy_a1_bits']:.2f} bits")
    with open(os.path.join(RESULTS_DIR, "n2_cf_dag.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="d1,n1,n2")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name in args.only.split(","):
        {"d1": d1, "n1": n1, "n2": n2}[name](args.quick)
