"""Positional profile deficits of a basis: nu_i = log GH(B_i) - ell_i = L(beta_i) - <a_i, ell>, the quantity the per-basis
identity ell_1 - S/d = h(0) - sum_i y_i nu_i pairs with the head multipliers.  Needs only the Gram-Schmidt profile (no
enumeration): the deficit is a property of the profile, the sub-Gaussian ratio eps_i = nu_i + log(||b_i^*|| / lambda_1(B_i)) needs
the block minima.

Run on the archived strict-census bases:
  python -m latticelab.deficits --archive results/lattice_l6_strict.json --keys "strict,100,40,31" "strict,100,40,32" ...
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence

import numpy as np

from latticelab.profile import gs_profile
from latticelab.profile_floor import block_sizes, dual_float, log_chat


def profile_deficits(ell: Sequence[float], beta: int) -> Dict:
    """nu_i, the head multipliers y_i (double precision), y_i nu_i, and the split of the weighted deficit into position 1 and the rest."""
    ell = np.asarray(ell, dtype=float)
    d = len(ell)
    bs = block_sizes(d, beta)
    y = np.asarray(dual_float(d, beta), dtype=float)
    nu = np.empty(d - 1)
    for i in range(d - 1):
        n = bs[i]
        avg = float(ell[i:i + n].mean())
        nu[i] = log_chat(n) + avg - ell[i]  # log GH(B_i) - ell_i
    ynu = y[: d - 1] * nu
    return {"nu": nu.tolist(), "y": y[: d - 1].tolist(), "y_nu": ynu.tolist(), "weighted_deficit": float(ynu.sum()),
            "nu1": float(nu[0]), "y1": float(y[0]), "y1_nu1": float(ynu[0]), "rest": float(ynu[1:].sum())}


def deficits_of_basis(A, beta: int) -> Dict:
    return profile_deficits(gs_profile(A), beta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default="results/lattice_l6_strict.json")
    ap.add_argument("--keys", nargs="+", required=True)
    args = ap.parse_args()
    from fpylll import IntegerMatrix

    arc = json.load(open(args.archive))
    for key in args.keys:
        rows = arc["bases"][key]
        B = IntegerMatrix.from_matrix(rows)
        beta = int(key.split(",")[2])
        r = deficits_of_basis(B, beta)
        print(key, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
