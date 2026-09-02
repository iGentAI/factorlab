"""Flags of sublattices, the canonical forward tour, and the LLL conveyor (docs/notes_lattice_barrier.md, section 10).

A basis b_1..b_d of L determines the complete flag  L_k := L(b_1, ..., b_k)  of primitive sublattices (k = 0..d), with log vol L_k = P_k =
l_1 + ... + l_k, so l_k = P_k - P_{k-1}; the projected block at position k of size n is the lattice L_{k+n-1}/L_{k-1} (realised as
pi_k(L(b_k, ..., b_{k+n-1}))), so every oracle query and every Gram-Schmidt norm is a function of the flag alone.  An exact-SVP insertion at
position j with block size n replaces L_j by the FORCED sublattice L_{j-1} + Z v, v any lift of the block's shortest vector (lift-independent,
and primitive because a shortest vector is primitive), and L_{j+1}, ..., L_{j+n-2} by an arbitrary chain of primitive sublattices between
L_j^new and L_{j+n-1} -- the completion; L_k for k < j and for k >= j+n-1 is unchanged PROVIDED the completion is block-supported (a
unimodular transformation of rows j..j+n-1, plus size reduction against earlier rows, which changes no L_k with k >= j).

Theorem (canonical forward tour; block-supported completions).  Assume every block met along the tour has a shortest vector unique up to sign
(ties have probability zero for Haar-random lattices and are non-generic, not impossible, for the q-ary bases used here).  A complete forward tour -- positions 1, 2, ..., d-1 in this order, each an exact-SVP insertion
(insert whenever the block minimum is strictly shorter than ||b_k^*||) with ANY block-supported completion -- produces the flag
        L_k' = L_{k-1}' + Z v_k,        v_k the shortest vector of  L_{k+n_k-1} / L_{k-1}',        n_k = min(beta, d-k+1),
where L_{k+n_k-1} is the INPUT flag's member: at step k the block's numerator has not been touched (a step j < k modifies only L_j..L_{j+n_j-2},
and j + n_j - 2 <= (k-1) + n_{k-1} - 2 <= k + n_k - 2 < k + n_k - 1, since j + n_j = min(j + beta, d + 1) is non-decreasing in j and
n_{k-1} <= n_k + 1) and its denominator is the forced L_{k-1}'; if the current b_k^* is already shortest no insertion happens
and, by uniqueness, L_{k-1}' + Z b_k = L_{k-1}' + Z v_k.  Hence l_k' = log lambda_1(L_{k+n_k-1}/L_{k-1}') for every k, and BOTH THE OUTPUT FLAG
AND THE OUTPUT PROFILE ARE INDEPENDENT OF THE COMPLETIONS.  Without uniqueness the no-insertion branch keeps the completion's vector as the
next block's denominator and later minima can differ, so the hypothesis is needed for the profile too.  Reverse
tours are not canonical: at step k the numerator L_{k+n-1} was set by the completion of step k+1.  A single insertion at kappa (0-based)
with a full block can change only the blocks with starts s in [kappa-beta+2, kappa-1] and [kappa+2, kappa+beta-1] (those whose numerator
or denominator is a completion-chosen flag member); every other block -- in particular the forced starts kappa-beta+1 (numerator
L_kappa'), kappa (L_{kappa+beta-1}/L_{kappa-1}) and kappa+1 (L_{kappa+beta}/L_kappa') -- is completion-independent.

The LLL conveyor.  Classical BKZ (Schnorr-Euchner; fpylll without BKZ.BOUNDED_LLL) is NOT in this class: before each enumeration it runs
LLL on the whole prefix b_1..b_{kappa+n-1}, so a freshly inserted short vector migrates backwards by Lovasz swaps, replacing L_{kappa-1},
L_{kappa-2}, ... by sublattices that contain it -- a non-oracle operation that lowers earlier Gram-Schmidt norms 'for free'.  With
BKZ.BOUNDED_LLL (LLL on the block only) the tour is block-supported and canonical.  `variant` selects: 'bounded_lll' (block LLL after the
insertion), 'bounded_hkz' (block LLL, then exact SVP insertions on the shrinking residuals kappa+1, ..., kappa+n-2: the HKZ completion,
realisable with the same oracle), 'conveyor' (classical BKZ: prefix LLL, fpylll's default).

Strict insertion.  fpylll inserts only when the block minimum is below delta_LLL ||b_kappa^*||^2 (delta = 0.99); `_StrictBKZ.svp_call`
replaces that threshold by (1 - tie_tol) ||b_kappa^*||^2 with tie_tol = 1e-12, independent of the LLL delta used by the completion:
the block minimum is inserted whenever it is shorter than the current vector by more than a relative 1e-12 -- the non-uniqueness the
theorem excludes -- so the tours realise the theorem's hypothesis up to floating-point ties.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
from fpylll import BKZ, GSO, IntegerMatrix
from fpylll.algorithms.bkz import BKZReduction as BKZBase
from fpylll.tools.bkz_stats import dummy_tracer

from latticelab.profile import block_gh_ratios, gs_profile

VARIANTS = ("bounded_lll", "bounded_hkz", "conveyor")
TIE_TOL = 1e-12


class _StrictBKZ(BKZBase):
    """fpylll's basic BKZ with the insertion threshold (1 - tie_tol) ||b_kappa^*||^2 in place of delta_LLL ||b_kappa^*||^2."""

    tie_tol = TIE_TOL

    def svp_call(self, kappa, block_size, params, tracer=None):
        from fpylll import Enumeration, EnumerationError

        radius = self.M.get_r(kappa, kappa) * (1 - self.tie_tol)
        try:
            sols = Enumeration(self.M).enumerate(kappa, kappa + block_size, radius, 0, pruning=None)
        except EnumerationError:
            return None
        if not sols:
            return None
        return min(sols, key=lambda s: s[0])[1]


def _params(size: int, bounded: bool) -> BKZ.Param:
    return BKZ.Param(block_size=size, max_loops=1, flags=(BKZ.BOUNDED_LLL if bounded else BKZ.DEFAULT))


def _reducer(A: IntegerMatrix):
    B = IntegerMatrix.from_matrix(A)
    M = GSO.Mat(B)
    M.update_gso()
    return _StrictBKZ(M), B, M


def _insert(bkz: BKZBase, kappa: int, size: int, variant: str, record: Dict | None = None) -> None:
    """Exact SVP insertion at kappa (block of `size`) with the completion of `variant`, in fpylll's three phases: preprocessing (LLL on the
    block if bounded, on the whole prefix in the 'conveyor' variant), enumeration, postprocessing (insertion), then size reduction of rows
    0..kappa -- exactly `svp_reduction`'s sequence.  If `record` is given, the queried block's log-volume is written into it AFTER the
    preprocessing, i.e. for the block that is actually enumerated."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    bounded = variant != "conveyor"
    par = _params(size, bounded)
    bkz.svp_preprocessing(kappa, size, par, dummy_tracer)
    if record is not None:
        bkz.M.update_gso()
        record["log_vol_block"] = 0.5 * sum(math.log(bkz.M.get_r(j, j)) for j in range(kappa, kappa + size))
    solution = bkz.svp_call(kappa, size, par, dummy_tracer)
    bkz.svp_postprocessing(kappa, size, solution, dummy_tracer)
    bkz.lll_obj.size_reduction(0, kappa + 1)
    if variant == "bounded_hkz":
        for t in range(1, size - 1):
            bkz.svp_reduction(kappa + t, size - t, _params(size - t, True))


def _tour(A: IntegerMatrix, beta: int, order: Iterable[int], variant: str) -> Tuple[IntegerMatrix, List[Dict]]:
    d = A.nrows
    if beta < 2 or d < 2:
        raise ValueError("need beta >= 2 and d >= 2")
    bkz, B, M = _reducer(A)
    rec = []
    for kappa in order:
        size = min(beta, d - kappa)
        r = {"kappa": kappa, "size": size}
        _insert(bkz, kappa, size, variant, record=r)
        M.update_gso()
        r["l_after"] = 0.5 * math.log(M.get_r(kappa, kappa))
        rec.append(r)
    return B, rec


def forward_tour(A: IntegerMatrix, beta: int, variant: str = "bounded_lll") -> Tuple[IntegerMatrix, List[Dict]]:
    """One complete forward tour (positions 0..d-2) of strict exact-SVP insertions with the completion of `variant`; returns the new basis
    and, per step, the log-volume of the block actually enumerated (recorded after fpylll's preprocessing: the block LLL leaves it
    unchanged, the conveyor's prefix LLL may not) and log ||b_kappa^*|| after the step (= log lambda_1 of that block up to the tie tolerance)."""
    return _tour(A, beta, range(A.nrows - 1), variant)


def reverse_tour(A: IntegerMatrix, beta: int, variant: str = "bounded_lll") -> Tuple[IntegerMatrix, List[Dict]]:
    """One complete reverse tour (positions d-2 down to 0)."""
    return _tour(A, beta, range(A.nrows - 2, -1, -1), variant)


def same_sublattice(rows1: List[List[int]], rows2: List[List[int]]) -> bool:
    """Exact test that two integer row sets of equal length generate the same lattice: equal Gram determinants and every row of the second
    an integer combination of the first (containment plus equal volume gives equality)."""
    from flint import fmpq_mat, fmpz_mat

    k = len(rows1)
    if k == 0 or len(rows2) != k:
        return k == len(rows2)
    A = fmpz_mat(rows1)
    Bm = fmpz_mat(rows2)
    GA = A * A.transpose()
    if GA.det() != (Bm * Bm.transpose()).det():
        return False
    X = fmpq_mat(GA).solve(fmpq_mat(A * Bm.transpose()))  # column j: coefficients of row j of B in the rows of A, if it lies in their span
    if X.transpose() * fmpq_mat(A) != fmpq_mat(Bm):
        return False
    return all(int(X[i, j].q) == 1 for i in range(k) for j in range(k))


def _rows(B: IntegerMatrix) -> List[List[int]]:
    return [[int(B[i, j]) for j in range(B.ncols)] for i in range(B.nrows)]


def compare_variants(A: IntegerMatrix, beta: int, tour, v1: str, v2: str, exact_flags: bool = True) -> Dict:
    """Run `tour` from A with variants v1 and v2 and compare: the output profiles, the per-step block records (log-volume before the query and
    the minimum found), and -- exactly -- how many members L_1..L_{d-1} of the two output flags coincide."""
    d = A.nrows
    B1, r1 = tour(A, beta, v1)
    B2, r2 = tour(A, beta, v2)
    p1, p2 = gs_profile(B1), gs_profile(B2)
    out = {"variants": [v1, v2], "max_profile_diff": float(np.max(np.abs(p1 - p2))), "n_positions_differ_1e-6": int(np.sum(np.abs(p1 - p2) > 1e-6)),
           "max_block_logvol_diff": max(abs(a["log_vol_block"] - b["log_vol_block"]) for a, b in zip(r1, r2)),
           "max_block_min_diff": max(abs(a["l_after"] - b["l_after"]) for a, b in zip(r1, r2)),
           "head_diff": float(p1[0] - p2[0]), "profile_1": p1.tolist(), "profile_2": p2.tolist()}
    if exact_flags:
        rows1, rows2 = _rows(B1), _rows(B2)
        eq = [same_sublattice(rows1[:k], rows2[:k]) for k in range(1, d)]
        out["n_flag_members_equal"] = int(sum(eq))
        out["first_flag_member_differing"] = next((k + 1 for k, e in enumerate(eq) if not e), None)
    return out


def canonical_tour_check(d: int, beta: int, q: int, seed: int, exact_flags: bool = True) -> Dict:
    """From the LLL basis of qary(d, d//2, q, seed): forward tours with the two block-bounded completions must agree (profile, block records,
    flags); reverse bounded tours and the classical conveyor variant generically do not."""
    from latticelab.lattices import lll, qary

    A = lll(qary(d, d // 2, q, seed=seed))
    return {"d": d, "beta": beta, "q": q, "seed": seed, "tie_tol": TIE_TOL,
            "forward_bounded": compare_variants(A, beta, forward_tour, "bounded_lll", "bounded_hkz", exact_flags),
            "reverse_bounded": compare_variants(A, beta, reverse_tour, "bounded_lll", "bounded_hkz", exact_flags),
            "forward_conveyor_vs_bounded": compare_variants(A, beta, forward_tour, "conveyor", "bounded_lll", exact_flags)}


def single_insertion_dependence(A: IntegerMatrix, kappa: int, beta: int) -> Dict:
    """One strict insertion at 0-based `kappa` (full block: kappa + beta <= d) with each bounded completion; for every emitted block start s
    the difference of the exact log(GH/lambda_1) between the two completions.  Predicted: zero exactly outside
    [kappa-beta+2, kappa-1] u [kappa+2, kappa+beta-1] (in particular at the forced starts kappa-beta+1, kappa, kappa+1); inside that set the
    two completions generically differ, but need not at every start."""
    d = A.nrows
    if not (0 <= kappa and kappa + beta <= d):
        raise ValueError("need a full block: 0 <= kappa and kappa + beta <= d")
    outs = {}
    for var in ("bounded_lll", "bounded_hkz"):
        bkz, B, M = _reducer(A)
        _insert(bkz, kappa, beta, var)
        outs[var] = B
    eps = {var: {row["i"] - 1: row["log_gh_over_lambda1"] for row in block_gh_ratios(outs[var], beta)["blocks"]} for var in outs}
    starts = sorted(eps["bounded_lll"])
    diff = {s: abs(eps["bounded_lll"][s] - eps["bounded_hkz"][s]) for s in starts}
    dependent = sorted(s for s in starts if kappa - beta + 2 <= s <= kappa - 1 or kappa + 2 <= s <= kappa + beta - 1)
    forced = [s for s in starts if s not in set(dependent)]
    p_l, p_h = gs_profile(outs["bounded_lll"]), gs_profile(outs["bounded_hkz"])
    return {"kappa": kappa, "beta": beta, "d": d, "eps_diff_by_start": diff, "predicted_dependent_starts": dependent,
            "max_diff_forced": max((diff[s] for s in forced), default=0.0), "max_diff_dependent": max((diff[s] for s in dependent), default=0.0),
            "n_dependent_starts_differing_1e-6": int(sum(1 for s in dependent if diff[s] > 1e-6)),
            "profile_positions_differing_1e-6": [int(i) for i in np.where(np.abs(p_l - p_h) > 1e-6)[0]],
            "inserted": bool(abs(p_l[kappa] - gs_profile(A)[kappa]) > 1e-9)}


def conveyor_tours(A: IntegerMatrix, beta: int, tours: int, mass_at: Iterable[int] = ()) -> Dict:
    """Run `tours` forward tours from A in the 'bounded_lll' and 'conveyor' variants side by side, recording after every tour the head
    l_1 - S/d of each and, at the tour counts in `mass_at`, the y-weighted sub-GH mass (insertion.weighted_subgh_mass) and residual."""
    from latticelab.insertion import weighted_subgh_mass

    d = A.nrows
    S = float(gs_profile(A).sum())
    state = {}
    for var in ("bounded_lll", "conveyor"):
        state[var] = _reducer(A)
    heads = {var: [] for var in state}
    masses = {var: {} for var in state}
    for T in range(1, tours + 1):
        for var, (bkz, B, M) in state.items():
            for kappa in range(d - 1):
                _insert(bkz, kappa, min(beta, d - kappa), var)
            heads[var].append(float(gs_profile(B)[0]) - S / d)
            if T in set(mass_at):
                w = weighted_subgh_mass(B, beta)
                masses[var][T] = {k: w[k] for k in ("weighted_eps_signed", "weighted_eps_positive", "residual", "frac_tight", "head_minus_mean", "floor_h0")}
    return {"d": d, "beta": beta, "tours": tours, "heads": heads, "masses": masses,
            "head_diff_conveyor_minus_bounded": [c - b for c, b in zip(heads["conveyor"], heads["bounded_lll"])]}


def main(argv=None):
    """CLI: `python -m latticelab.flag --points 40,8 60,20 80,30 --seeds 1 2 3 --out results/lattice_canonical_tour.json` archives the
    canonical-tour checks (bounded forward agreement, bounded reverse disagreement, conveyor vs bounded), the single-insertion dependence
    pattern, and (`--conveyor-tours T`) the multi-tour conveyor comparison."""
    import argparse
    import json
    import os
    import time

    from latticelab.lattices import lll, qary

    ap = argparse.ArgumentParser(description="canonical forward tour and the LLL conveyor")
    ap.add_argument("--points", nargs="+", default=["40,8", "60,20", "80,30"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--q", type=int, default=2 ** 16 + 1)
    ap.add_argument("--kappas", nargs="+", type=int, default=[5, 25])
    ap.add_argument("--conveyor-tours", type=int, default=0)
    ap.add_argument("--mass-at", nargs="+", type=int, default=[2, 8, 32])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = {"note": "canonical forward tour: strict exact-SVP tours (insertion below (1 - 1e-12) ||b*||^2) with block-bounded LLL and block-bounded HKZ "
                   "completions from the same LLL start must agree in profile, queried-block records and output flag; reverse bounded tours and the "
                   "classical prefix-LLL conveyor (fpylll default) differ; single insertions differ only at the predicted block starts",
           "tours": [], "single": [], "conveyor_tours": []}
    for pt in a.points:
        d, beta = (int(x) for x in pt.split(","))
        for seed in a.seeds:
            t0 = time.time()
            r = canonical_tour_check(d, beta, a.q, seed)
            r["seconds"] = time.time() - t0
            out["tours"].append(r)
            f, b, c = r["forward_bounded"], r["reverse_bounded"], r["forward_conveyor_vs_bounded"]
            print(f"({d},{beta}) seed {seed}: bounded forward max|dprofile| {f['max_profile_diff']:.2e} blocks vol {f['max_block_logvol_diff']:.2e} "
                  f"min {f['max_block_min_diff']:.2e} flags equal {f.get('n_flag_members_equal')}/{d-1}; bounded reverse max|dprofile| {b['max_profile_diff']:.3f} "
                  f"({b['n_positions_differ_1e-6']} positions) flags equal {b.get('n_flag_members_equal')}/{d-1}; conveyor vs bounded forward: "
                  f"max|dprofile| {c['max_profile_diff']:.3f}, head diff {c['head_diff']:+.4f}, flags equal {c.get('n_flag_members_equal')}/{d-1} [{r['seconds']:.0f}s]", flush=True)
            A = lll(qary(d, d // 2, a.q, seed=seed))
            for kappa in a.kappas:
                if kappa + beta <= d:
                    s = single_insertion_dependence(A, kappa, beta)
                    s.update({"seed": seed})
                    out["single"].append(s)
                    print(f"   single insertion kappa={kappa}: inserted {s['inserted']}; max diff at forced starts {s['max_diff_forced']:.2e}, "
                          f"{s['n_dependent_starts_differing_1e-6']}/{len(s['predicted_dependent_starts'])} predicted-dependent starts differ "
                          f"(max {s['max_diff_dependent']:.3f}); profile differs at {s['profile_positions_differing_1e-6']}", flush=True)
            if a.conveyor_tours:
                t0 = time.time()
                ct = conveyor_tours(A, beta, a.conveyor_tours, a.mass_at)
                ct.update({"seed": seed, "seconds": time.time() - t0})
                out["conveyor_tours"].append(ct)
                hd = ct["head_diff_conveyor_minus_bounded"]
                print(f"   conveyor vs bounded over {a.conveyor_tours} tours: head diff after 1/2/4/8/.../{a.conveyor_tours} tours "
                      + " ".join(f"{hd[t-1]:+.4f}" for t in [1, 2, 4, 8, 16, 32, 64] if t <= a.conveyor_tours)
                      + "; masses " + "; ".join(f"T={T}: bounded {ct['masses']['bounded_lll'][T]['weighted_eps_signed']:+.4f} (R {ct['masses']['bounded_lll'][T]['residual']:.3f}) "
                                                 f"conveyor {ct['masses']['conveyor'][T]['weighted_eps_signed']:+.4f} (R {ct['masses']['conveyor'][T]['residual']:.3f})"
                                                 for T in sorted(ct["masses"]["conveyor"])) + f" [{ct['seconds']:.0f}s]", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
