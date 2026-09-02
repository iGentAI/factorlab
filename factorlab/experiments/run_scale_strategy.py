"""E22 strategy driver: modeled random-input u and exact all-prime label entropy across u."""

from __future__ import annotations

import json
import os
import sys

from ..bench import RESULTS_DIR
from .fixed_list_check import modeled_u_profile
from .hitting_sets import label_entropy_profile, greedy_label_schedule


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bits = (96, 128, 160, 180, 200, 240, 300)
    modeled = [modeled_u_profile(n) for n in bits]
    entropy = [label_entropy_profile(lx, us=(2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
                                     max_curves=250 if quick else 500)
               for lx in ((14,) if quick else (14, 16))]
    out = {"modeled_random_input": modeled, "exact_label_entropy": entropy}
    greedy = [greedy_label_schedule(lx, us=(2.5, 3.0, 3.5, 4.0, 4.5),
                                    curves_per_u=8 if quick else 12)
              for lx in ((14,) if quick else (14, 16))]
    out["greedy_multi_u"] = greedy
    with open(os.path.join(RESULTS_DIR, "e22_scale_strategy.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("== E22 random-input model ==")
    for r in modeled:
        b = r["best"]
        print(f"  N={r['nbits']} bits: u*={r['best_u']:.2f}, log2 B1={b['log2_B1']:.2f}, "
              f"G={b['G']:.4g}, expected binary curves={b['expected_binary_curves']:.1f}, "
              f"log2 work={b['log2_expected_work']:.2f}")
    print("== E22 exact-label separator profile ==")
    for r in entropy:
        print(f"  x=2^{r['log2_x']} ({r['n_primes']} primes), best observed u={r['best_observed_u']}")
        for z in r["rows"]:
            print(f"    u={z['u']:.1f}: B1={z['B1']}, Rlabel={z['R_label']:.4f}, I/B1={z['information_per_B1']:.4g}, "
                  f"Ksep={z['K_sep_label']}, Ksep*B1={z['separation_work_proxy']}, binary Ksep={z['K_sep_binary']}")
    print("== E22 greedy multi-u separating schedule ==")
    for r in greedy:
        print(f"  x=2^{r['log2_x']} ({r['n_primes']} primes): separated={r['separated']}, "
              f"steps={r['n_steps']}, total B1 proxy={r['total_cost_proxy']}, "
              f"best fixed={r['best_fixed_u']}, ratio={r['greedy_over_best_fixed_cost']}; "
              f"schedule={[(z['u'], z['sigma'], z['B1']) for z in r['steps']]}")
