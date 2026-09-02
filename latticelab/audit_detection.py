"""Prefix-weighted deficits of the archived audit bases at their detection position.

The report's applicability audit measures, on archived BKZ output bases of shape (d, beta), the per-block deficits
nu_i := log GH(B_i) - l_i (Gaussian-heuristic length of the projected block minus the actual Gram-Schmidt log-norm).  The prefix-volume
floor of the report comes with an exact dual certificate w^{(m)} >= 0, z = m/d for each prefix length m, and the linear identity

    P_m(l) = P_m(l^tight) - sum_i w^{(m)}_i nu_i          (for every profile l of the same log-volume S)

holds exactly.  The detection chain reads the Gaussian-heuristic length of the last block off the prefix volume P_{d-b}, so the quantity that
measures how far a real basis moves the detection bound is the weighted deficit  W_m := sum_i w^{(m)}_i nu_i  at m = d - beta, and the
implied shift of the last block's log-GH bound is W_m / beta.  This module evaluates W_m on the archived bases, splits off the head term
w_1 nu_1, and checks the identity above to floating-point accuracy as a guard on the computation.

Status of the numbers: the Gram-Schmidt norms are exact rationals (certify_audit.exact_gso_norms); their logarithms, the deficits and the
weighted sums are double precision.  These are measured values of finite bases (sense (iv) of the report), not certificates.  Sign
convention: admissibility is l_i >= log GH(B_i) - eps, so nu_i <= eps; a positive nu_i is a violation (the block's first vector is shorter
than its Gaussian heuristic, as at the head), a negative nu_i is slack (longer than the heuristic, as in the shrinking tail).  A negative W
means the prefix volume exceeds the extremal profile's, which LOWERS the last block's Gaussian-heuristic bound.

CLI:  python -m latticelab.audit_detection --archive results/lattice_l6_strict.json --out results/lattice_audit_detection_deficits.json
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Sequence

from latticelab.certify_audit import exact_gso_norms
from latticelab.deficits import profile_deficits
from latticelab.profile_floor import prefix_volume_certificate, tight_profile


def log_norms_from_rows(rows: Sequence[Sequence[int]]) -> List[float]:
    """Gram-Schmidt log-norms l_i = (1/2) log ||b_i^*||^2 of an integer basis, from the exact rational squared norms."""
    sq = exact_gso_norms([list(map(int, r)) for r in rows])
    return [0.5 * (math.log(int(x.p)) - math.log(int(x.q))) for x in sq]


def weighted_deficits(ell: Sequence[float], beta: int, m: int) -> Dict:
    """W_m = sum_i w^{(m)}_i nu_i for the profile ell at prefix length m, with the head term, the implied last-block shift W_m / beta, and
    the exact-identity gap  P_m(ell) - P_m(ell_tight) + W_m  (zero up to rounding)."""
    d = len(ell)
    if not 1 <= m <= d:
        raise ValueError("m must lie in [1, d]")
    if not 2 <= beta < d:
        raise ValueError("beta must lie in [2, d)")
    nu = profile_deficits(ell, beta)["nu"]
    w, z = prefix_volume_certificate(d, beta, m)
    wf = [float(x) for x in w]
    terms = [wi * ni for wi, ni in zip(wf, nu)]
    W = sum(terms)
    S = float(sum(ell))
    tight = tight_profile(d, beta, 0.0, S)
    gap = float(sum(ell[:m])) - float(sum(tight[:m])) + W
    body = sum(terms[1:m])          # blocks 2..m (1-based): full-size blocks starting inside the prefix
    tail = sum(terms[m:])           # blocks starting beyond the prefix, including the shrinking tail
    return {"d": d, "beta": beta, "m": m, "W": W, "head_term": terms[0], "body_term": body, "tail_term": tail, "w1": wf[0], "nu1": nu[0],
            "last_block_shift": W / beta, "identity_gap": gap, "z": float(z), "n_nonzero_w": sum(1 for x in wf if x != 0.0),
            # admissibility is  l_i >= log GH(B_i) - eps, i.e. nu_i <= eps: a violation (at eps = 0) is a POSITIVE deficit
            "n_violated_blocks": sum(1 for x in nu if x > 0.0), "n_violated_in_prefix": sum(1 for x in nu[:m] if x > 0.0),
            "max_nu": max(nu), "argmax_nu": 1 + nu.index(max(nu)), "min_nu": min(nu), "argmin_nu": 1 + nu.index(min(nu))}


def audit_archive(path: str, keys: Sequence[str] | None = None) -> Dict:
    """Evaluate W_{d-beta} on every basis of a strict-census archive (keys 'strict,d,beta,seed'), or on the given keys."""
    arch = json.load(open(path))
    bases = arch["bases"]
    out = {}
    for key in (keys or sorted(bases)):
        _, d_s, beta_s, seed_s = key.split(",")
        d, beta = int(d_s), int(beta_s)
        ell = log_norms_from_rows(bases[key])
        if len(ell) != d:
            raise ValueError(f"{key}: basis has {len(ell)} rows, expected {d}")
        row = weighted_deficits(ell, beta, d - beta)
        row["seed"] = int(seed_s)
        out[key] = row
    return {"archive": path, "rows": out,
            "note": "W = sum_i w^{(d-beta)}_i nu_i on the archived basis; last_block_shift = W/beta is the implied change of the last block's "
                    "log-GH bound; identity_gap checks P_m(l) - P_m(l_tight) + W = 0; double precision from exact GSO norms."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default="results/lattice_l6_strict.json")
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--out", default="results/lattice_audit_detection_deficits.json")
    a = ap.parse_args()
    res = audit_archive(a.archive, a.keys)
    json.dump(res, open(a.out, "w"), indent=1)
    for key, r in res["rows"].items():
        print(f"{key}: m={r['m']} W={r['W']:+.4f} = head {r['head_term']:+.4f} + body {r['body_term']:+.4f} + tail {r['tail_term']:+.4f} "
              f"(w1={r['w1']:.4f}, nu1={r['nu1']:+.4f}; violated blocks (nu>0) {r['n_violated_blocks']}, of which {r['n_violated_in_prefix']} in "
              f"the prefix; max nu {r['max_nu']:+.3f} at i={r['argmax_nu']}, min nu {r['min_nu']:+.3f} at i={r['argmin_nu']}) "
              f"shift={r['last_block_shift']:+.2e} gap={r['identity_gap']:.1e}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
