"""Driver for E23: the constant-level whitespace inside the relation paradigm.

E23a  exact (d, d) pair floors at small N against the resultant bound, the tiny-f
      route at larger N, and the geometric-progression constructions (Montgomery
      for d = 2; root-lift GP for d = 3, 4) with their exponents 2(d-1)/d^2.
E23b  the cost-quality frontier of free-root selection for the linear pair at
      d = 3, 4 against the Kleinjung scale and the theoretical curve.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

from ..bench import RESULTS_DIR
from ..gen import make_semiprime
from .nonlinear_pairs import pair_floor_experiment, tiny_f_scaling, gp_pair
from .selection_frontier import frontier_experiment


def gp_construction_experiment(ds=(2, 3, 4), bits=(40, 64, 96, 128), count=10, seed=41) -> dict:
    rng = random.Random(seed)
    out = {"rows": []}
    for d in ds:
        xs, ys = [], []
        for nbits in bits:
            exps, ratios, minima_spread = [], [], []
            for i in range(count):
                N = int(make_semiprime(nbits, "rsa", seed, i).N)
                r = gp_pair(N, d, rng)
                if r is None:
                    continue
                exps.append(r["log_P2_over_log_N"])
                ratios.append(r["gp_max_over_N_1_minus_1_over_d"])
                minima_spread.append(math.log2(r["admissible_basis_norms"][-1] / r["admissible_basis_norms"][0]))
                xs.append(math.log2(float(N)))
                ys.append(math.log2(r["P2"]))
            out["rows"].append({"d": d, "nbits": nbits, "count": len(exps),
                                "mean_log_P2_over_log_N": float(np.mean(exps)),
                                "mean_gp_max_ratio": float(np.mean(ratios)),
                                "mean_log2_minima_spread": float(np.mean(minima_spread))})
        (s, _), cov = np.polyfit(xs, ys, 1, cov=True)
        out[f"fit_d{d}"] = {"slope": float(s), "slope_se": float(math.sqrt(cov[0, 0])),
                            "predicted": 2.0 * (d - 1) / d ** 2, "floor": 1.0 / d}
    return out


def main(quick: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("== E23a: geometric-progression constructions of (d, d) pairs ==")
    gp = gp_construction_experiment(count=4 if quick else 10)
    for r in gp["rows"]:
        print(f"  d={r['d']} {r['nbits']:4d} bits: log P2/log N {r['mean_log_P2_over_log_N']:.3f}, GP max / N^(1-1/d) {r['mean_gp_max_ratio']:.2f}, "
              f"log2 spread of admissible basis norms {r['mean_log2_minima_spread']:.2f}")
    for d in (2, 3, 4):
        f = gp[f"fit_d{d}"]
        print(f"  d={d}: slope {f['slope']:.4f} +- {f['slope_se']:.4f} (predicted 2(d-1)/d^2 = {f['predicted']:.4f}; floor 1/d = {f['floor']:.4f})")

    print("== E23a: exact (d, d) pair floors (certified) ==")
    if quick:
        plans = {2: ((16, 20), (3, 3)), 3: ((16, 18), (3, 3))}
    else:
        plans = {2: ((16, 20, 24, 28), (6, 6, 4, 2)), 3: ((16, 18, 20, 22, 24), (6, 6, 4, 3, 2))}
    exact = {}
    for d, (bits, counts) in plans.items():
        res = pair_floor_experiment(d, bits, counts)
        exact[str(d)] = res
        for row in res["rows"]:
            print(f"  d={d} {row['nbits']:3d} bits (n={row['count']}): exact log P2/log N {row['mean_log_P2_over_log_N']:.3f}, P2/floor {row['mean_P2_over_floor']:.2f}; "
                  f"construction log P2/log N {row['mean_construction_log_P2_over_log_N']:.3f}, construction/exact {row['mean_construction_over_exact']:.1f}")
            norms = [tuple(r["exact_norms"]) for r in row["instances"]]
            print(f"        minimiser norms (||f||, ||g||): {norms}")
        print(f"  d={d}: exact slope {res['fit']['exact_slope']:.3f} +- {res['fit']['exact_slope_se']:.3f} (floor {res['floor_exponent']:.3f}); "
              f"construction slope {res['fit']['construction_slope']:.3f} +- {res['fit']['construction_slope_se']:.3f} (predicted {res['gp_construction_exponent']:.3f})")

    print("== E23a: the tiny-f route at larger N ==")
    tiny = {}
    for d in (2, 3, 4):
        bits = (24, 40, 64) if quick else (24, 32, 40, 48, 64, 80, 96)
        t = tiny_f_scaling(d, bits, 3 if quick else 6)
        tiny[str(d)] = t
        for row in t["rows"]:
            print(f"  d={d} {row['nbits']:3d} bits (n={row['count']}, complete={row['all_complete']}): log P2/log N {row['mean_log_P2_over_log_N']:.3f}, P2/floor {row['mean_P2_over_floor']:.2f}, "
                  f"log2 P2 - log2 N/d {row['mean_log2_P2_minus_log2_N_over_d']:+.2f}")
        print(f"  d={d}: slope {t['fit']['slope']:.4f} +- {t['fit']['slope_se']:.4f} (floor 1/d = {1/d:.4f})")
    with open(os.path.join(RESULTS_DIR, "e23_nonlinear_pairs.json"), "w") as fh:
        json.dump({"gp_constructions": gp, "exact_floors": exact, "tiny_f": tiny}, fh, indent=1)

    print("== E23b: the cost-quality frontier of free-root selection (linear pair) ==")
    T_max = 1 << 14 if quick else 1 << 18
    fr = {}
    plans = {3: (48, 64, 80, 96), 4: (64, 96, 128)} if not quick else {3: (48, 64), 4: (64,)}
    for d, bits in plans.items():
        res = frontier_experiment(d, bits, 2 if quick else 3, T_max=T_max)
        fr[str(d)] = res
        print(f"  d={d}: crossover exponent {res['crossover_exponent']:.4f}, floor exponent {res['floor_exponent']:.4f} at T = N^{res['floor_trials_exponent']:.4f}")
        for row in res["rows"]:
            print(f"   {row['nbits']:3d} bits: base-m scale log2 P {row['log2_base_m_scale']:.1f}, joint floor {row['log2_joint_floor']:.1f}, "
                  f"log2 T_x {row['log2_T_crossover']:.1f}, log2 T_floor {row['log2_T_floor']:.1f}")
            for j, T in enumerate(row["checkpoints"]):
                if T & (T - 1) == 0 and (T.bit_length() - 1) % 2 == 0 or T == T_max:
                    print(f"      T=2^{T.bit_length()-1:2d}: free P {row['free_log2_P'][j]:.1f} Q {row['free_log2_Q'][j]:.1f} | theory P {row['theory_log2_P'][j]:.1f} | "
                          f"Kleinjung-scale P {row['kleinjung_log2_P'][j]:.1f} Q {row['kleinjung_log2_Q'][j]:.1f}")
    with open(os.path.join(RESULTS_DIR, "e23_selection_frontier.json"), "w") as fh:
        json.dump(fr, fh, indent=1)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
