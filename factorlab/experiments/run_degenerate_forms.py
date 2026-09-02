"""Driver for E3."""

from __future__ import annotations

import json
import os

from ..bench import RESULTS_DIR
from ..gen import make_semiprime
from . import degenerate_forms as dfm


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    inst = make_semiprime(60, "balanced", 61, 0)
    N = int(inst.N)
    print("== E3: cancellation at local minima of small-coefficient polynomial forms (256-bit) ==")
    out = {"surveys": [], "coverage": []}
    for e, A, trials, exhaustive in ((2, 2, 0, True), (2, 4, 0, True), (2, 16, 20000, False),
                                     (3, 1, 0, True), (3, 2, 20000, False), (3, 8, 20000, False)):
        s = dfm.survey(N, e, A, trials, seed=e * 100 + A, exhaustive=exhaustive)
        out["surveys"].append(s)
        print(f"  e={e} A={A:2d} forms={s['forms']:6d} local minima in J={s['local_minima_in_J']:5d}  "
              f"eta2 (cancellation ratio of g''): min={s['eta2_min']:.3e} median={s['eta2_median']:.3f}  "
              f"P(<0.1)={s['frac_eta2_below_1e-1']:.4f} P(<1e-2)={s['frac_eta2_below_1e-2']:.4f} P(<1e-4)={s['frac_eta2_below_1e-4']:.5f}  "
              f"minima with eta2<1e-2: {s['minima_with_eta2_below_1e-2']} of which perfect-power top: {s['of_which_perfect_power_top']} "
              f"(perfect-power minima total: {s['minima_with_perfect_power_top']})")
        for b in s["most_degenerate"][:2]:
            print(f"      eta2={b['eta2']:.3e} eta3={b['eta3']:.3f} asym_fraction={b['asym_fraction']:.3f} "
                  f"perfect_power_top={b['perfect_power_top']} p*/sqrtN={b['pstar_over_sqrtN']:.7f}  {b['coeffs']}")
        if s["most_degenerate"]:
            b = s["most_degenerate"][0]
            ws = [1e3, 1e6, 1e9, 1e12]
            rows = dfm.coverage_comparison(N, b["coeffs"], b["pstar_over_sqrtN"], ws)
            for row in rows:
                pred_s = (f"Hoelder(order {row['hoelder_order_used']})={row['hoelder_prediction']:.4g} cov/pred={row['coverage_over_prediction']:.3f}"
                          if row["coverage_over_prediction"] is not None else "Hoelder prediction unavailable")
                print(f"        w={row['w']:.0e}: coverage={row['coverage']:.4g}  Lehman(1,1)={row['lehman11_coverage']:.4g}  "
                      f"ratio={row['ratio_to_lehman']:.3e}  {pred_s}  well_depth={row['well_depth']:.3g} (w>depth: {row['w_exceeds_well_depth']})  "
                      f"cubic ref={row['cubic_reference']:.4g}")
            out["coverage"].append({"e": e, "A": A, "cell": b, "rows": rows})
    with open(os.path.join(RESULTS_DIR, "e3_degenerate_forms.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
