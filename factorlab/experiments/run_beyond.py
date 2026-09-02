"""Driver for the beyond-GNFS experiments: E15 root-lattice floor, E17 paradigm calculator,
E18 joint polynomial-selection floor, E19 quadratic near-square bridge."""

from __future__ import annotations

import json
import os
import sys

from ..bench import RESULTS_DIR
from .root_lattice import root_lattice_experiment
from .paradigm_calculator import paradigm_table, pure_power_sizes
from .poly_floor import poly_floor_experiment
from .quadratic_bridge import quadratic_bridge_experiment, k_scaling_experiment


def main(quick: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("== E15: shortest vectors of the root lattice {f : f(m) = 0 mod N} versus N^(1/(d+1)) ==")
    rl = root_lattice_experiment(count=6 if quick else 20)
    for r in rl["rows"]:
        print(f"  {r['nbits']:4d} bits d={r['d']}: random m ratio mean {r['random_m_ratio_mean']:.3f} (min {r['random_m_ratio_min']:.3f}) "
              f"base-m ratio mean {r['base_m_ratio_mean']:.3f} (min {r['base_m_ratio_min']:.3f}) Gaussian {r['gaussian']:.3f} | SNFS ratio {r['snfs_ratio_mean']:.2e}")
    with open(os.path.join(RESULTS_DIR, "e15_root_lattice.json"), "w") as fh:
        json.dump(rl, fh, indent=1)

    print("== E17: Dickman cost model, QS versus NFS(d) ==")
    pt = paradigm_table()
    for r in pt["rows"]:
        print(f"  {r['nbits']:5d} bits: QS log2 cost {r['qs_log2_cost']:6.1f} (log2 B {r['qs_log2_B']:.1f}, u {r['qs_u']:.2f}) | "
              f"NFS log2 cost {r['nfs_log2_cost']:6.1f} (d={r['nfs_d']}, log2 B {r['nfs_log2_B']:.1f}, log2 A {r['nfs_log2_A']:.1f})")
    print(f"  fits: QS slope vs L[1/2] {pt['fits']['qs_vs_L_half_slope']:.3f}; NFS slope vs L[1/3] {pt['fits']['nfs_vs_L_third_slope']:.3f} "
          f"(heuristic (64/9)^(1/3) = {pt['fits']['nfs_heuristic_constant']:.3f}); first size where NFS is cheaper: {pt['first_bits_where_nfs_cheaper']} bits")
    pp = pure_power_sizes(128)
    print(f"  pure-power |a^d - N b^d| size exponents at b = N^(1/6): {pp}")
    with open(os.path.join(RESULTS_DIR, "e17_paradigm_calculator.json"), "w") as fh:
        json.dump({"table": pt, "pure_power": pp}, fh, indent=1)

    print("== E18: exact joint minimum of ||f||_oo * m over irreducible degree-d polynomials with a root mod N ==")
    if quick:
        plans = {2: ((24, 28, 32), (3, 3, 3)), 3: ((16, 20), (3, 3))}
    else:
        plans = {2: ((24, 28, 32, 36, 40, 44), (10, 10, 10, 10, 8, 3)), 3: ((16, 20, 24, 28, 32), (10, 10, 10, 8, 6))}
    e18 = {}
    for d, (bits, counts) in plans.items():
        res = poly_floor_experiment(d, bits, counts)
        e18[str(d)] = res
        ex = res["predicted"]
        print(f"  d={d}: predicted exponents product {ex['product']:.4f} (base-m {ex['base_m_product']:.4f}), "
              f"coeff {ex['coeff']:.4f}, root {ex['root']:.4f}")
        for row in res["rows"]:
            print(f"    {row['nbits']:3d} bits (n={row['count']:2d}): log P/log N {row['mean_log_P_over_log_N']:.4f} "
                  f"(coeff {row['mean_log_Hf_over_log_N']:.3f}, root {row['mean_log_m_over_log_N']:.3f}) | "
                  f"P/pred {row['mean_P_over_pred']:.2f} (crude {row['mean_P_over_pred_crude']:.2f}) | "
                  f"log2 P {row['mean_log2_P']:.2f} +- {row['std_log2_P']:.2f}, base-m {row['mean_log2_base_m_P']:.2f}, "
                  f"search K* {row['mean_log2_search_P_Kstar']:.2f}, 8K* {row['mean_log2_search_P_8Kstar']:.2f} "
                  f"(reaches optimum {row['search_reaches_optimum_fraction']:.2f})")
        print(f"    fit: slope {res['fit']['slope']:.4f} +- {res['fit']['slope_se']:.4f} (predicted {ex['product']:.4f}); "
              f"base-m slope {res['fit']['base_m_slope']:.4f} +- {res['fit']['base_m_slope_se']:.4f} (predicted {ex['base_m_product']:.4f})")
        with open(os.path.join(RESULTS_DIR, "e18_poly_floor.json"), "w") as fh:
            json.dump(e18, fh, indent=1)

    print("== E19: quadratic polynomial selection as a bounded near-square search ==")
    e18_path = os.path.join(RESULTS_DIR, "e18_poly_floor.json")
    if quick:
        e19 = quadratic_bridge_experiment(bits=(40, 56), count=40, e18_path=e18_path)
    else:
        e19 = quadratic_bridge_experiment(e18_path=e18_path)
    for row in e19["rows"]:
        parts = []
        for c, z in row["scales"].items():
            parts.append(f"C={c}: pairs {z['mean_count']:.2f} (pred {z['predicted_mean_count']:.2f}, var {z['var_count']:.1f}); "
                         f"events {z['mean_events']:.2f} (var {z['var_events']:.2f}); primitive {z['mean_primitive_events']:.2f} "
                         f"(var {z['var_primitive_events']:.2f}); zero {z['zero_fraction']:.2f} "
                         f"(Poisson on events {z['poisson_zero_prediction_from_events']:.2f}, on primitive {z['poisson_zero_prediction_from_primitive']:.2f}); "
                         f"best P/N^0.6 median {z['median_best_P_over_N35'] if z['median_best_P_over_N35'] is None else round(z['median_best_P_over_N35'], 3)} "
                         f"over {z['n_with_candidates']} with candidates")
        print(f"  {row['nbits']:3d} bits: " + " | ".join(parts))
        print(f"           phase mean {row['phase_mean']:+.4f}, variance {row['phase_variance']:.4f} (uniform 0.0833), "
              f"lag-1 corr {row['phase_lag1_mean']:+.4f} +- {row['phase_lag1_se']:.4f}")
    pc = e19["phase_check"]
    print(f"  phases pooled: n={pc['n']}, KS {pc['ks']:.4f} (p {pc['p']:.3f}); uniform control KS {pc['control_ks']:.4f} (p {pc['control_p']:.3f})")
    rt = e19["e18_roundtrip"]
    print(f"  E18 quadratic witnesses round-tripped: {rt['checked']}, max |disc|/H^2 = {rt['max_abs_delta_over_H2']:.3f}, k values {rt['k_values']}")
    e19["k_scaling"] = k_scaling_experiment(bits=(40, 56), count=40 if quick else 100)
    for row in e19["k_scaling"]["rows"]:
        parts = [f"k={k}: {z['mean_count']:.2f} +- {z['se_count']:.2f} (pred {z['predicted']:.2f}; ratio to k=1 {z['ratio_to_k1_mean']:.3f} vs {z['predicted_ratio']:.3f})"
                 for k, z in row["per_k"].items()]
        print(f"  k-scaling {row['nbits']} bits (C={row['C']}, n={row['count']}): " + " | ".join(parts))
    with open(os.path.join(RESULTS_DIR, "e19_quadratic_bridge.json"), "w") as fh:
        json.dump(e19, fh, indent=1)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
