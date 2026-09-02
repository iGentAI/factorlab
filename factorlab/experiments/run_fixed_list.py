"""Driver for E21: the fixed-list deterministic ECM against the E20 certificate, and its cost."""

from __future__ import annotations

import json
import os
import sys

from ..bench import RESULTS_DIR
from .fixed_list_check import certificate_test, cost_scaling


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {}
    print("== E21: certificate test, p, q in [2^18, 2^19), B1 = 64, B2 = 4096, bound 39 curves ==")
    cert = certificate_test(count=100 if quick else 500)
    out["certificate"] = cert
    print(f"  pairs {cert['n_pairs']} (tail pairs {cert['n_tail_pairs']}): all found {cert['all_found']}, "
          f"all within bound {cert['all_within_bound']}, max curves {cert['max_curves_used']}, mean {cert['mean_curves_used']:.2f}, "
          f"stages {cert['stage_counts']}, earlier than simulation {cert['fraction_earlier_than_simulation']:.3f}, "
          f"later {cert['n_later_than_simulation']}, mean simulated index {cert['mean_sim_index']:.2f}")
    for r in cert["tail_pairs"]:
        print(f"    tail pair ({r['p']}, {r['q']}): curve {r['curves']} stage {r['stage']} {r['detail'] or ''} simulated {r['sim_index']}")
    print("== E21: cost scaling on RSA moduli, u = 3, C = 2 ==")
    cs = cost_scaling(bits=(32, 40, 48) if quick else (32, 40, 48, 56, 64), count=10 if quick else 20)
    out["cost"] = cs
    for r in cs["rows"]:
        print(f"  {r['nbits']} bits: B1={r['B1']}, B2={r['B2']}, curves mean {r['mean_curves']:.1f} (max {r['max_curves']}), "
              f"mulmod mean {r['mean_mulmod']:.3g}, per curve / B1 = {r['mean_mulmod_per_curve_over_B1']:.1f}, "
              f"stages {r['stage_counts']}, wall {r['mean_wall']:.2f}s")
    print(f"  fit: log2 mulmod slope {cs['fit']['mulmod_exponent']:.3f} (predicted 1/(2u) = {cs['fit']['predicted_exponent']:.3f} plus logs), "
          f"curves slope {cs['fit']['curves_exponent']:.3f}")
    with open(os.path.join(RESULTS_DIR, "e21_fixed_list.json"), "w") as fh:
        json.dump(out, fh, indent=1)
