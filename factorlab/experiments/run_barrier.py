"""Driver: Frobenius-degree identity, Lehman covering scaling, chirp hull complexity."""

from __future__ import annotations

import json
import math
import os

from ..bench import RESULTS_DIR, fit_exponent
from ..gen import make_semiprime
from . import barrier


def frobenius_degree():
    print("== Frobenius defect: degree q-p on F_p (finite-difference test) ==")
    out = []
    for nb in (20, 24, 28, 32):
        for i in range(5):
            inst = make_semiprime(nb, "balanced", 31, i)
            if int(inst.q - inst.p) > 60000:
                continue
            r = barrier.frobenius_degree_check(inst.N, inst.p, inst.q)
            r.update({"nbits": nb, "index": i, "p": int(inst.p), "q": int(inst.q)})
            out.append(r)
            print(f"  {nb}b d=q-p={r['d']:6d} identity={r['identity_holds']} degree_exact={r['degree_exact']}")
    # also close primes: gap small -> cheap
    for i in range(3):
        inst = make_semiprime(80, "close", 31, i, gap_bits=12)
        r = barrier.frobenius_degree_check(inst.N, inst.p, inst.q)
        r.update({"nbits": 80, "index": i, "p": int(inst.p), "q": int(inst.q), "family": "close"})
        out.append(r)
        print(f"  80b close d={r['d']:6d} identity={r['identity_holds']} degree_exact={r['degree_exact']}")
    with open(os.path.join(RESULTS_DIR, "barrier_frobenius_degree.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    ok = all(r["identity_holds"] for r in out)
    print(f"  identity holds in {sum(r['identity_holds'] for r in out)}/{len(out)}; degree exact in {sum(r['degree_exact'] for r in out)}/{len(out)}")
    return ok


def lehman_covering():
    print("== Lehman covering sum Sigma_w vs P (prediction Sigma_w ~ sqrt(N) log r /(2 sqrt r), P ~ r log r) ==")
    inst = make_semiprime(120, "balanced", 32, 0)
    rows = []
    for r in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        row = barrier.lehman_covering(inst.N, r)
        rows.append(row)
        print(f"  r={r:5d} P={row['P']:6d} Sigma_w/sqrtN={row['sigma_over_sqrtN']:.5f}  "
              f"exact sum/(4r)={row['analytic_sigma_over_sqrtN']:.5f}  leading log r/(2 sqrt r)={row['analytic_leading']:.5f}  "
              f"ratio to sqrtN/sqrtP = {row['ratio_to_prediction']:.3f}  (log r)^1.5/2={math.log(r)**1.5/2:.3f}")
    xs = [math.log(row["P"]) for row in rows]
    ys = [math.log(row["sigma_over_sqrtN"]) for row in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    print(f"  raw log-log slope: Sigma_w ~ P^{slope:.3f}  (log factors dominate at these r; compare ratio column)")
    with open(os.path.join(RESULTS_DIR, "barrier_lehman_covering.json"), "w") as fh:
        json.dump({"rows": rows, "slope": slope}, fh, indent=1)
    return slope


def coverage_lemma():
    print("== Per-cell p-coverage length vs convexity bound 2 sqrt(w sqrt N / a) ==")
    import random
    rng = random.Random(7)
    inst = make_semiprime(100, "balanced", 34, 0)
    worst = None
    worst_mono = None
    rows = []
    for _ in range(40000):
        a = rng.randrange(1, 200)
        kind = rng.random()
        if kind < 0.5:
            b = rng.randrange(a, 4 * a + 1)          # critical: p* in [sqrt(N/4), sqrt N]
        elif kind < 0.75:
            b = rng.randrange(1, a + 1)              # monotone decreasing (b/a <= 1)
        else:
            b = rng.randrange(4 * a, 40 * a + 1)     # monotone increasing (b/a >= 4)
        w = 10 ** rng.uniform(0, 12)
        delta0 = rng.choice([0.0, rng.uniform(0, 10 * w)])
        row = barrier.cell_coverage_length(inst.N, a, b, delta0, w)
        row["critical"] = a <= b <= 4 * a
        rows.append(row)
        if worst is None or row["ratio"] > worst["ratio"]:
            worst = row
        if not row["critical"] and (worst_mono is None or row["ratio"] > worst_mono["ratio"]):
            worst_mono = row
    print(f"  cells tested: {len(rows)}   max length/bound = {worst['ratio']:.4f}  (lemma requires <= 1)")
    print(f"  worst case: a={worst['a']} b={worst['b']} delta0={worst['delta0']:.3g} w={worst['w']:.3g} "
          f"length={worst['length']:.4g} bound={worst['bound']:.4g} critical={worst['critical']}")
    print(f"  worst monotone cell: ratio={worst_mono['ratio']:.4f} a={worst_mono['a']} b={worst_mono['b']}")
    with open(os.path.join(RESULTS_DIR, "barrier_coverage_lemma.json"), "w") as fh:
        json.dump({"max_ratio": worst["ratio"], "worst": worst, "worst_monotone": worst_mono, "n": len(rows)}, fh, indent=1)
    return worst["ratio"]


def chirp_hull():
    print("== Hull complexity of c(k) = ceil(2 sqrt(kN)) ==")
    out = []
    for nb in (30, 45, 60):
        inst = make_semiprime(nb, "balanced", 33, 0)
        K_third = int(round(float(inst.N) ** (1 / 3)))
        for K in sorted({K_third // 8, K_third // 2, K_third, 2 * K_third}):
            if K < 8:
                continue
            row = barrier.chirp_hull_complexity(inst.N, K)
            out.append(row)
            print(f"  {nb}b K={K:7d} (N^1/3={K_third}) upper-hull vertices={row['upper_hull_vertices']:7d} "
                  f"({row['upper_hull_vertices']/K:.3f} K) lower={row['lower_hull_vertices']:7d}  d2 hist={row['second_difference_hist']}")
    with open(os.path.join(RESULTS_DIR, "barrier_chirp_hull.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frobenius_degree()
    lehman_covering()
    coverage_lemma()
    chirp_hull()
