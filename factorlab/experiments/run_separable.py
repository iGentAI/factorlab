"""Driver for E1 (separable alignment) and E2 (chirp linear complexity)."""

from __future__ import annotations

import json
import os

from ..bench import RESULTS_DIR
from ..gen import make_semiprime
from . import separable


def e1(quick: bool = False, large: bool = False):
    print("== E1: greedy maximal separably-aligned Lehman subfamily (difference constraints) ==")
    out = []
    nb = 60
    inst = make_semiprime(nb, "balanced", 51, 0)
    if large:
        rs = [2048, 4096, 8192]
    else:
        rs = [32, 64, 128, 256] if quick else [32, 64, 128, 256, 512, 1024]
    for K in ((1.0,) if large else (1.0, 4.0)):
        for r in rs:
            row = separable.greedy_separable_alignment(inst.N, r, K=K, C=4.0)
            row["nbits"] = nb
            out.append(row)
            print(f"  K={K:3.0f} r={r:5d}: cells={row['cells_considered']:6d} kept={row['kept']:5d} "
                  f"forest_bound={row['forest_bound']:5d} excess={row['excess_over_forest']:4d} "
                  f"K22free={row['K22_free']} forest={row['is_forest']}  union coverage/L: separable windows={row['sep_window_coverage_share']:.4f} "
                  f"kept@Lehman={row['lehman_window_coverage_share_kept']:.4f} all@Lehman={row['lehman_window_coverage_share_all']:.4f}  "
                  f"max|offset-c|/w={row['max_offset_deviation_in_w']:.2f}")
    with open(os.path.join(RESULTS_DIR, "e1_separable_alignment_large.json" if large else "e1_separable_alignment.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    return out


def e2(quick: bool = False):
    print("== E2: linear complexity of alpha^{ceil(2 sqrt(kN))} mod ell ==")
    out = []
    inst = make_semiprime(60, "balanced", 52, 0)
    for r in ([256, 1024] if quick else [256, 1024, 4096]):
        for ell in (1000003, 998244353):
            row = separable.chirp_linear_complexity(inst.N, r, ell)
            out.append(row)
            print(f"  r={r:5d} ell={ell}: L={row['linear_complexity']:5d}  r/2={row['generic_expectation']:.0f}  "
                  f"ratio={row['ratio']:.3f}  controls: LFSR-2={row['control_lfsr_order2_complexity']} "
                  f"exact-quadratic-chirp={row['control_exact_quadratic_chirp_complexity']}")
    with open(os.path.join(RESULTS_DIR, "e2_chirp_linear_complexity.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--large-e1", action="store_true", help="only E1 at r = 2048..8192")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.large_e1:
        e1(large=True)
    else:
        e1(args.quick)
        e2(args.quick)
