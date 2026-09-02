"""Driver for the probabilistic campaign: E6 (success profiles) and E7 (Lehman hits)."""

from __future__ import annotations

import json
import os
import sys

from ..bench import RESULTS_DIR
from .smooth_profiles import (semismooth_profile, collision_profile, ecm_profile, dickman_rho,
                              semismooth_G, brute_force_G, rho_k_profile)
from .lehman_hits import hits_experiment


def main(quick: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {}
    print("== Dickman / semismooth table ==")
    table = []
    for c in (1/8, 1/6, 1/5, 2/9, 1/4):
        G = semismooth_G(2 * c, min(4 * c, 1.0))
        table.append({"c": c, "rho_stage1": dickman_rho(1 / (2 * c)), "G_stage2": G,
                      "any_of_four_indep": 1 - (1 - G) ** 4})
        print(f"  c={c:.4f}  rho(1/2c)={table[-1]['rho_stage1']:.4f}  G(2c,4c)={G:.4f}  1-(1-G)^4={table[-1]['any_of_four_indep']:.4f}")
    out["theory"] = table
    X = 2 * 10 ** 5 if quick else 3 * 10 ** 6
    bf = {f"{a:.3f},{b:.3f}": {"brute": brute_force_G(X, a, b), "G": semismooth_G(a, b)}
          for a, b in ((1/3, 2/3), (0.4, 0.8), (0.5, 1.0), (1/3, 1/3))}
    print(f"  brute-force check at X={X}: {bf}")
    out["brute_force_check"] = {"X": X, "values": bf}

    print("== E6a: semismoothness of p-1, p+1, q-1, q+1 (RSA family) ==")
    out["semismooth"] = []
    for nbits, count in ((48, 500 if quick else 6000), (64, 300 if quick else 4000)):
        res = semismooth_profile(nbits, count)
        out["semismooth"].append(res)
        print(f"  nbits={nbits} count={count}")
        for row in res["rows"]:
            print(f"   c={row['c']:.4f} pred G={row['pred_G']:.3f}  p-1/q-1={row['minus_pooled']:.3f} "
                  f"p+1/q+1={row['plus_pooled']:.3f} ctrl_even={row['ctrl_even']:.3f} ctrl_int={row['ctrl_int']:.3f} "
                  f"any4={row['any_of_four']:.3f} (indep pred {row['pred_any_of_four_indep']:.3f}) corr={row['corr_minus_plus']:+.3f}")
            print(f"            exact: p-1 base2={row['exact_minus_pooled_base2']:.3f} (disagree {row['exact_minus_vs_ideal_disagree']:.4f}, n={row['exact_minus_disagree_count']}) "
                  f"Williams base3={row['exact_plus_pooled_base3']:.3f} 3 bases={row['exact_plus_pooled_3bases']:.3f} "
                  f"exact any4={row['exact_any_of_four']:.3f} both-exposed={row['exact_minus_both_exposed']:.3f} equal-order={row['exact_minus_both_exposed_equal_order']:.4f} "
                  f"(n={row['degenerate_count']}, recoverable by descent={row['degenerate_recoverable_by_descent']})")

    print("== E6b: rho collision times and Fermat steps ==")
    out["collision"] = []
    for nbits, count in ((40, 300 if quick else 2000), (48, 100 if quick else 800)):
        res = collision_profile(nbits, count)
        out["collision"].append(res)
        print(f"  nbits={nbits}: E[(mu+lambda)/sqrt p]={res['rho_length_over_sqrt_p_mean']:.3f} "
              f"(random mapping {res['random_mapping_mean']:.3f})  min over p,q: {res['rho_min_over_sqrt_p_mean']:.3f}")
        for row in res["rows"]:
            print(f"   c={row['c']:.4f} rho success={row['rho_success']:.3f} (Rayleigh pred {row['rho_pred_rayleigh']:.3f})  fermat={row['fermat_success']:.3f}")

    print("== E6d: rho length of x^k + c mod p versus sqrt(pi p/(2(k-1))) ==")
    res = rho_k_profile(pbits=20 if quick else 22, count=60 if quick else 400)
    out["rho_k"] = res
    if "random_mapping_control" in res:
        rc = res["random_mapping_control"]
        print(f"  random mapping control: {rc['mean']:.3f} +- {rc['stderr']:.3f} (pred {rc['pred']:.3f})")
    for row in res["rows"]:
        print(f"   k={row['k']:2d} E[rho]/sqrt p = {row['mean']:.3f} +- {row['stderr']:.3f}  "
              f"Arney-Bender {row['pred_arney_bender']:.3f}  image-only {row['naive_image_only']:.3f}")

    print("== E6c: ECM stage 1 at B1 = N^c, 8 curves ==")
    res = ecm_profile(40, 60 if quick else 300)
    out["ecm"] = res
    for row in res["rows"]:
        print(f"   c={row['c']:.4f} B1={row['B1']} success={row['success']:.3f} per-curve={row['per_curve_success_est']:.3f} "
              f"(rho pred/prime {row['pred_rho_per_prime']:.3f}) mean mulmod={row['mean_mulmod']:.0f}")

    print("== E7: Lehman hits vs continued fraction of p/q ==")
    out["hits"] = []
    for nbits, count in ((32, 30 if quick else 200), (40, 10 if quick else 60)):
        res = hits_experiment(nbits, count)
        out["hits"].append(res)
        print(f"  nbits={nbits}: hits dist={res['hit_count_distribution']} mean={res['mean_hits']:.2f} kinds={res['kinds']} "
              f"star hit={res['star_is_hit_fraction']:.3f} min-ab hit is star={res['min_ab_hit_is_star_fraction']:.3f} "
              f"k_next/bound quantiles={res['k_next_over_bound_quantiles']}")

    with open(os.path.join(RESULTS_DIR, "e6_e7_probabilistic.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
