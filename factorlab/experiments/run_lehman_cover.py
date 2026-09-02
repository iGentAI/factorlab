"""Driver for E24: the difference-cover lower bound from the approximate-Sidon structure
of the Lehman window starts (notes_barrier.md, section 7.5)."""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from ..bench import RESULTS_DIR
from .lehman_cover import lehman_cover_experiment


def main(quick: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bits = (32, 40) if quick else (32, 40, 48, 56, 64)
    res = lehman_cover_experiment(bits=bits, count=1 if quick else 2)
    print("== E24: approximate-Sidon statistic of the squarefree a = 1 Lehman windows and the cover lower bound ==")
    for row in res["rows"]:
        sub = row["subfamily"]
        a1 = row["a1_subfamily"]
        alt = row["all_subfamily"]
        lb = max(row["lower_bound"], row["counting_bound_full_family"])
        tr = lambda s: "*" if s["subfamily"]["truncated"] or s["subfamily"].get("a_range") not in (None, (1, 2), [1, 2]) else ""
        print(f"  {row['log2_N']:.0f} bits #{row['instance']} r=N^{row['target_exponent']:.3f}={row['r']:6d}: sqfree R={sub['R']:5d}{'*' if sub['truncated'] else ''} W={sub['W_min']}-{sub['W_max']} "
              f"D_max={row['D_max']:3d} (random {row['control_random_starts_D_max']}; all a=1 {a1['D_max']}{tr(a1)}; all cells {alt['D_max']}{tr(alt)}) | "
              f"bound: sidon {row['lower_bound_sidon_branch']:.0f}, windows {row['lower_bound_window_branch']:.0f}, min {row['lower_bound']:.0f}, counting {row['counting_bound_full_family']:.0f} -> {lb:.0f} = N^{math.log(lb)/math.log(float(row['N'])):.3f} | "
              f"Harvey M1 {row['harvey_M1_cost']:.0f} = N^{row['log_harvey_over_log_N']:.3f} (m={row['harvey_m']})")
    print("  (* = sub-family truncated or restricted to a dyadic a-range; its bound is valid, its D_max is not the full family's)")
    # slope of the Sidon branch at fixed r = N^e against log2 N (predicted (1 + e)/6)
    for e in (1 / 5, 1 / 4):
        xs = [row["log2_N"] for row in res["rows"] if abs(row["target_exponent"] - e) < 1e-9]
        ys = [math.log2(row["lower_bound_sidon_branch"]) for row in res["rows"] if abs(row["target_exponent"] - e) < 1e-9]
        if len(set(xs)) > 2:
            (s, _), cov = np.polyfit(xs, ys, 1, cov=True)
            res[f"sidon_branch_slope_e{e:.3f}"] = {"slope": float(s), "se": float(math.sqrt(cov[0, 0])), "predicted": 1 / 6 + e / 6, "n": len(xs)}
            print(f"  Sidon-branch slope at r = N^{e:.3f}: {s:.4f} +- {math.sqrt(cov[0,0]):.4f} (predicted (1 + {e:.3f})/6 = {1/6 + e/6:.4f})")
    dmax = [row["D_max"] for row in res["rows"]]
    ctrl = [row["control_random_starts_D_max"] for row in res["rows"]]
    res["D_max_summary"] = {"sqfree_max": max(dmax), "sqfree_mean": float(np.mean(dmax)), "control_max": max(ctrl), "control_mean": float(np.mean(ctrl)),
                            "a1_max": max(row["a1_subfamily"]["D_max"] for row in res["rows"]),
                            "all_cells_max": max(row["all_subfamily"]["D_max"] for row in res["rows"])}
    print(f"  D_max summary: squarefree a=1 family max {max(dmax)} mean {np.mean(dmax):.1f}; random control max {max(ctrl)} mean {np.mean(ctrl):.1f}; "
          f"all a=1 cells max {res['D_max_summary']['a1_max']}; all cells max {res['D_max_summary']['all_cells_max']}")
    with open(os.path.join(RESULTS_DIR, "e24_lehman_cover.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
