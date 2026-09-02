"""Driver for E20: smoothness hitting sets across prime sizes."""

from __future__ import annotations

import json
import os
import sys

from ..bench import RESULTS_DIR
from .hitting_sets import hitting_scaling_experiment, residual_scaling_experiment


def _print(res: dict) -> None:
    print(f"== E20: hitting sets at u = {res['u']} (B1 = x^(1/u), B2 = x^(2/u)) ==")
    for row in res["rows"]:
        d, e, th = row["with_deterministic_stage1"], row["ecm_only_stage1"], row["theta"]
        ds2, es2 = row["with_deterministic_stage2"], row["ecm_only_stage2"]
        if ds2 is not None and es2 is not None:
            s2 = (f", one-large-prime Kcover det/ecm={ds2['K_star']}/{es2['K_star']}, "
                  f"Kseparate={ds2['K_separate']}/{es2['K_separate']}")
        else:
            s2 = ""
        print(f"  x=2^{row['log2_x']}: {row['n_primes']} primes, B1={row['B1']}, B2={row['B2']} | "
              f"stage1 Kcover det/ecm={d['K_star_stage1_only']}/{e['K_star_stage1_only']}, "
              f"Kseparate det/ecm={d['K_separate_stage1']}/{e['K_separate_stage1']}{s2} | "
              f"theta(stage1+LP): mean {th['theta_mean']:.3f}, min {th['theta_min']:.3f}, 1%-quantile {th['theta_quantiles'][0]:.3f}, "
              f"zero fraction {th['fraction_theta_zero']:.4f}; predicted Kcover Jeffreys {th['predicted_K_star_independent_jeffreys']} "
              f"(stage1 {th['predicted_K_star_stage1_jeffreys']}), predicted Kseparate continuation/stage1 "
              f"{th['predicted_K_separate_independent_jeffreys']}/{th['predicted_K_separate_stage1_jeffreys']} | "
              f"latest-covered primes (first hit): {d['hardest_primes'][:3]}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {}
    res3 = hitting_scaling_experiment(log2_xs=(14, 16, 18) if quick else (14, 16, 18, 20, 22), u=3.0, max_curves=60)
    _print(res3)
    out["u3"] = res3
    res4 = hitting_scaling_experiment(log2_xs=(14, 16) if quick else (14, 16, 18), u=4.0, max_curves=300)
    _print(res4)
    out["u4"] = res4
    for u, key, xs in ((3.0, "u3_residual", (14, 16) if quick else (14, 16, 18, 20)),
                       (4.0, "u4_residual", (14,) if quick else (14, 16, 18))):
        rr = residual_scaling_experiment(log2_xs=xs, u=u, max_curves=80 if u == 3.0 else 300)
        out[key] = rr
        print(f"== E20b: Proposition V' separation (exposure labels) at u = {u} ==")
        for row in rr["rows"]:
            h = row["history"]
            print(f"  x=2^{row['log2_x']}: {row['n_primes']} primes, B1={row['B1']}, B2={row['B2']} | "
                  f"Kcov={row['K_cov']}, Ksep(labels)={row['K_sep_residual']} | first-curve exposed rate "
                  f"{h[0]['exposed_rate_evaluated']:.3f}; unresolved pairs: {[z['unresolved_pairs'] for z in h[:6]]}; "
                  f"equal-exposure pairs: {[z['unresolved_pairs_equal_exposure'] for z in h[:6]]}")
    with open(os.path.join(RESULTS_DIR, "e20_hitting_sets.json"), "w") as fh:
        json.dump(out, fh, indent=1)
