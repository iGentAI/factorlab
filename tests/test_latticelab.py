"""Tests for latticelab (the lattice barrier programme's computation arm)."""
import math

import numpy as np
import pytest

from latticelab.lattices import cap_fraction, gaussian_heuristic, gh_count, lll, log_volume, ntru, qary
from latticelab.profile import bkz, delta_gsa, gs_profile, profile_stats
from latticelab.sieve import GaussSieve, angular_excess, coverage, predicted_coverage
from latticelab.experiments import exact_svp_norm


def test_constructions_and_volume():
    A = qary(20, 10, 101, seed=1)
    assert abs(log_volume(A) - 10 * math.log(101)) < 1e-6  # det of [qI|0; A|I] is q^k
    n, q = 8, 257
    B, f, g = ntru(n, q, seed=2)
    assert B.nrows == 2 * n and abs(log_volume(B) - n * math.log(q)) < 1e-6
    # (g | f) lies in the lattice: it equals sum_i f_i row_{n+i} plus a q-multiple of the first n unit rows, i.e. the first n
    # coordinates of sum_i f_i row_{n+i} are congruent to g mod q and the last n coordinates equal f
    Bn = np.array([[int(B[i, j]) for j in range(2 * n)] for i in range(2 * n)], dtype=object)
    w = np.array(f, dtype=object) @ Bn[n:]
    assert all(int(w[n + i]) == f[i] for i in range(n))
    assert all((int(w[j]) - g[j]) % q == 0 for j in range(n))
    # and it is a shortest vector of this small instance
    target = math.sqrt(sum(x * x for x in f) + sum(x * x for x in g))
    assert abs(exact_svp_norm(lll(B)) - target) < 1e-9
    with pytest.raises(ValueError):
        ntru(12, 257, 1)
    with pytest.raises(ValueError):
        ntru(8, 256, 1)


def test_gaussian_heuristic_and_caps():
    # GH for the integer lattice Z^d (vol 1): (1/V_d)^{1/d}; d = 2: 1/sqrt(pi)
    assert abs(gaussian_heuristic(2, 0.0) - 1 / math.sqrt(math.pi)) < 1e-12
    assert abs(gh_count(2, 0.0, 1.0) - math.pi) < 1e-12
    # cap fractions: hemisphere = 1/2 exactly; in d = 3 the cap of angle theta has fraction (1 - cos theta)/2
    assert abs(cap_fraction(3, math.pi / 2) - 0.5) < 1e-12
    assert abs(cap_fraction(3, math.pi / 3) - (1 - math.cos(math.pi / 3)) / 2) < 1e-12
    # asymptotics: cap(60 degrees) ~ (3/4)^{d/2} up to polynomial factors
    d = 200
    assert abs(math.log(cap_fraction(d, math.pi / 3)) / d - 0.5 * math.log(3 / 4)) < 0.03


def test_profile_statistics():
    A = qary(40, 20, 2 ** 16 + 1, seed=3)
    B = bkz(A, 10)
    st = profile_stats(B, 10)
    p = np.array(gs_profile(B))
    assert abs(p.sum() - log_volume(B)) < 1e-6
    assert 1.005 < st["root_hermite"] < 1.03 and st["slope"] < 0
    assert abs(st["first_ratio"] - math.exp(p[0] - p.sum() / 40)) < 1e-9
    assert "delta_gsa" not in st  # asymptotic constant not reported for small beta
    assert 1.010 < delta_gsa(50) < 1.014  # Chen-Nguyen constant at beta = 50 is 1.0121


def test_gauss_sieve_finds_shortest_and_coverage_model():
    A = lll(qary(24, 12, 2 ** 12 + 3, seed=5))
    gs = GaussSieve(A, seed=1, max_collisions=100)
    res = gs.run()
    assert abs(math.sqrt(res["shortest_norm2"]) - exact_svp_norm(A)) < 1e-9
    # the list is pairwise reduced: no pair within 60 degrees among vectors of equal norm, hence angular excess 0 at 60 degrees
    assert angular_excess(gs.L, math.pi / 3) == 0.0
    cov = coverage(gs.L, math.pi / 3, 2000, seed=2)
    assert abs(cov - predicted_coverage(len(gs.L), 24, math.pi / 3)) < 0.1
    assert res["stats"].inner_products > 0 and res["stats"].samples > 0
    with pytest.raises(ValueError):
        GaussSieve(A, max_collisions=0)


def test_block_gh_ratios_definition_and_edge_cases():
    """block_gh_ratios: lambda_1 of each projected block by exact enumeration, GH from the block volume, eps = max log(GH/lambda_1)^+;
    on an HKZ-like fully reduced small basis every b_i^* is its block's shortest vector; edge cases are rejected or well-defined."""
    from latticelab.profile import block_gh_ratios
    from latticelab.profile_floor import log_chat

    A = bkz(qary(24, 12, 2 ** 12 + 3, seed=7), 12)
    st = block_gh_ratios(A, 12)
    assert len(st["blocks"]) == 23 and st["eps_needed"] >= 0 and st["worst_block"] in range(1, 24)
    for row in st["blocks"]:
        # GH(B_i) = chat(beta_i) vol(B_i)^{1/beta_i} recomputed from the reported lambda_1 and log ratio is consistent
        assert abs(math.log(row["gh"] / row["lambda1"]) - row["log_gh_over_lambda1"]) < 1e-12
        assert row["b_star_over_lambda1"] >= 1 - 1e-9  # b_i^* is a vector of the block, so it is at least the block's minimum
    # the first block of a BKZ-12 basis with beta = 12: b_1 is the shortest vector of the whole first block
    assert abs(st["blocks"][0]["b_star_over_lambda1"] - 1) < 1e-9
    assert block_gh_ratios(A, 12, max_blocks=0)["worst_block"] is None
    with pytest.raises(ValueError):
        block_gh_ratios(A, 1)
    with pytest.raises(ValueError):
        block_gh_ratios(A, 12, max_blocks=-1)
    assert log_chat(2) < 0 < log_chat(20)


def test_profile_floor_certificate_matches_lp():
    """The exact dual certificate reproduces the LP minimum of l_1 over (beta, eps)-admissible profiles (independent scipy solve), the
    tight profile is admissible and attains it, all multipliers are positive for beta < d, z = 1/d, and the eps-loss is sum(y) * eps."""
    from fractions import Fraction

    from scipy.optimize import linprog

    from latticelab.profile_floor import (_dual_certificate_windowed, _verify_certificate_windowed, block_sizes, dual_certificate, floor_l1,
                                          floor_l1_float, log_chat, tight_profile, verify_certificate, beta_floor_for_target)

    # the O(d) sliding-window dual and verification agree exactly with the original O(d beta) forms, including beta > d/2, and the
    # verification rejects a perturbed certificate
    for d, beta in ((30, 6), (60, 10), (60, 40), (60, 59), (100, 51)):
        y, z = dual_certificate(d, beta)
        yw, zw = _dual_certificate_windowed(d, beta)
        assert y == yw and z == zw and verify_certificate(d, beta, y, z) and _verify_certificate_windowed(d, beta, y, z)
        bad = list(y)
        bad[len(bad) // 2] += Fraction(1, 10 ** 9)
        assert not verify_certificate(d, beta, bad, z) and not _verify_certificate_windowed(d, beta, bad, z)
        assert not verify_certificate(d, beta, y[:-1], z)

    for d, beta in ((30, 6), (60, 10), (80, 25)):
        y, z = dual_certificate(d, beta)
        assert z == Fraction(1, d) and len(y) == d - 1 and all(v > 0 for v in y)
        assert verify_certificate(d, beta, y, z)
        assert float(y[0]) == pytest.approx(beta * (d - 1) / (d * (beta - 1)))
        assert float(y[1]) == pytest.approx(beta * (d - beta) / (d * (beta - 1) ** 2))
        bs = block_sizes(d, beta)
        A = np.zeros((d - 1, d))
        b = np.zeros(d - 1)
        for i in range(1, d):
            bi = bs[i - 1]
            A[i - 1, i - 1] = -(1 - 1 / bi)
            A[i - 1, i:i + bi - 1] = 1 / bi
            b[i - 1] = -log_chat(bi)
        res = linprog(c=np.eye(d)[0], A_ub=A, b_ub=b, A_eq=np.ones((1, d)), b_eq=[0.0], bounds=[(None, None)] * d, method="highs")
        assert res.status == 0
        r = floor_l1(d, beta)
        assert abs(res.x[0] - r["l1_floor"]) < 1e-7
        tp = tight_profile(d, beta)
        assert abs(tp.sum()) < 1e-8 and abs(tp[0] - r["l1_floor"]) < 1e-8
        assert np.all(A @ tp - b <= 1e-8)  # admissible (all tight)
        # eps-loss is linear with slope sum(y)
        r1 = floor_l1(d, beta, eps=0.05)
        assert abs((r["l1_floor"] - r1["l1_floor"]) - 0.05 * r["dual_sum"]) < 1e-9
        # float dual agrees with the exact one to rounding
        assert abs(floor_l1_float(d, beta)["l1_floor"] - r["l1_floor"]) < 1e-9
    # degenerate case beta = d: single block, y_1 = 1, y_i = 0 for i >= 2, floor = log chat(d)
    y, z = dual_certificate(20, 20)
    assert y[0] == 1 and all(v == 0 for v in y[1:]) and verify_certificate(20, 20, y, z)
    assert abs(floor_l1(20, 20)["l1_floor"] - log_chat(20)) < 1e-12
    # Theorem (positivity of the certificate): the two-term recurrence y_k = ((beta_{k+1}-1)/beta_{k+1}) y_{k+1} + [k >= beta] y_{k-beta+1}/beta
    # (y_d := 0) and y_1 = 1 + ((beta-1)/beta) y_2 hold exactly; the rows of T sum to at most (beta-1)/beta in the head (less when the
    # shrinking tail overlaps it, beta > d/2), exactly 1 in the body, and < 1 in the tail
    for d, beta in ((30, 6), (100, 20), (60, 59)):
        y, z = dual_certificate(d, beta)
        bs = block_sizes(d, beta)
        Y = {k: y[k - 1] for k in range(1, d)}
        Y[d] = Fraction(0)
        B = lambda k: bs[k - 1] if k <= d - 1 else 1
        assert Y[1] == 1 + Fraction(B(2) - 1, B(2)) * Y[2]
        for k in range(2, d):
            rhs = Fraction(B(k + 1) - 1, B(k + 1)) * Y[k + 1] + (Y[k - beta + 1] / B(k - beta + 1) if k >= beta else 0)
            assert Y[k] == rhs
        rows = {k: Fraction(B(k + 1) - 1, B(k + 1)) + (Fraction(1, B(k - beta + 1)) if k >= beta else 0) for k in range(2, d)}
        assert all(rows[k] == 1 for k in rows if beta <= k <= d - beta)
        assert all(rows[k] < 1 for k in rows if k < beta or k > d - beta)
        assert Y[d - 1] == Y[d - beta] / beta and all(v > 0 for v in y)
        # exact identity sum (beta_i - 1) y_i = d - 1 (pair the certificate with the profile l_j = j)
        assert sum((Fraction(b - 1) * yi for yi, b in zip(y, bs)), Fraction(0)) == d - 1
        # closed form of the tail correction kappa = g - h(0)
        c_b = log_chat(beta)
        kappa_formula = -sum(float(yi) * (log_chat(b) - (b - 1) * c_b / (beta - 1)) for yi, b in zip(y, bs) if b < beta)
        kappa_direct = (d - 1) * c_b / (beta - 1) - floor_l1_float(d, beta)["l1_floor"]
        assert abs(kappa_formula - kappa_direct) < 1e-9
    # the floor is not monotone in beta: for small beta chat(beta) < 1 and the axiom is nearly vacuous, so the floor rises with beta up
    # to about beta = 36 (the exact integer maximum of log chat(beta)/(beta-1)) and decreases beyond; the scan therefore never assumes monotonicity.  Record the small-beta behaviour:
    assert floor_l1(60, 14)["root_hermite_floor"] < floor_l1(60, 15)["root_hermite_floor"] < floor_l1(60, 40)["root_hermite_floor"]
    # blocksize floor in the decreasing regime: target strictly between the floors at beta = 60 and beta = 59 (d = 100)
    d = 100
    target = floor_l1(d, 60)["root_hermite_floor"] + 1e-9
    assert floor_l1(d, 59)["root_hermite_floor"] > target
    bf = beta_floor_for_target(d, target, beta_lo=50, beta_hi=70)
    assert bf["beta_floor"] == 60 and bf["dual_exact"] and bf["predecessor_fails_rigorous"] and "decided rigorously" in bf["minimality"]
    bfe = beta_floor_for_target(d, target, beta_lo=57, beta_hi=63, exact_all=True)
    assert bfe["beta_floor"] == 60 and bfe["minimality"].startswith("certified")
    # a ball target (the pure-GSA delta at beta = 60 is above the floor at 60 and below the floor at 59? check and use whichever holds)
    from latticelab.profile_floor import gsa_delta_ball, decide_floor_vs_target

    tb = gsa_delta_ball(60)
    reaches60, r60 = decide_floor_vs_target(d, 60, tb)
    reaches59, _ = decide_floor_vs_target(d, 59, tb)
    assert r60["root_hermite_floor_ball"].rad() < 1e-60 and isinstance(reaches60, bool) and isinstance(reaches59, bool)
    # exact_all ignores the band (it does not use the pre-screen)
    assert beta_floor_for_target(d, target, beta_lo=59, beta_hi=61, exact_all=True, band=0.0)["beta_floor"] == 60
    # no candidate: a target below every floor in the range yields None with a double-precision-scan note
    none = beta_floor_for_target(d, 0.5, beta_lo=50, beta_hi=55)
    assert none["beta_floor"] is None and "double-precision" in none["note"]
    # an excessive float/rigorous discrepancy must fail certification on the no-candidate path: the target lies below every floor in
    # the range, the patched float at beta_lo falls inside the band (so it is decided rigorously and rejected), and the post-loop check raises
    import latticelab.profile_floor as pf

    real_float = pf.floor_l1_float
    try:
        pf.floor_l1_float = lambda dd, bb, ee=0.0, log_vol=0.0: {**real_float(dd, bb, ee, log_vol),
                                                                  "root_hermite_floor": real_float(dd, bb, ee, log_vol)["root_hermite_floor"] + 5e-6}
        lowest = min(floor_l1(d, b)["root_hermite_floor"] for b in range(62, 66))
        assert lowest == floor_l1(d, 65)["root_hermite_floor"]  # decreasing regime: the last beta has the lowest floor
        with pytest.raises(ValueError):
            pf.beta_floor_for_target(d, lowest - 4e-6, beta_lo=62, beta_hi=65)
    finally:
        pf.floor_l1_float = real_float
    with pytest.raises(ValueError):
        beta_floor_for_target(d, target, beta_lo=6, beta_hi=30, band=0.0)
    with pytest.raises(ValueError):
        block_sizes(10, 12)


def test_simulator_invariance_convergence_and_completions():
    """Insertion dynamics on profiles: every completion conserves the block volume; the uniform completion preserves admissibility along
    any schedule (Proposition), so l_1 never crosses the certified floor and all-block tours converge to the all-tight profile (Theorem);
    the HKZ completion creates violations and lets tours cross the floor; tail-first switches to forward tours exactly once stable."""
    import random

    from latticelab.profile_floor import floor_l1_float, tight_profile
    from latticelab.simulator import (consistency_census, gh_of_block, insert, lll_like_profile, make_random, make_tail_first,
                                      make_tours, run_schedule, violations)

    d, beta = 60, 10
    l0 = lll_like_profile(d)
    assert abs(l0.sum()) < 1e-9 and violations(l0, beta).max() == 0.0  # the LLL line is admissible at this beta
    for comp in ("uniform", "hkz", "hkz13"):
        new, changed = insert(l0, 5, beta, comp)
        assert changed and abs(new[5:15].sum() - l0[5:15].sum()) < 1e-9 and np.all(new[:5] == l0[:5]) and np.all(new[15:] == l0[15:])
        assert abs(new[5] - gh_of_block(l0, 5, beta)) < 1e-12  # the inserted position is set to the block's GH value
        assert insert(new, 5, beta, comp)[1] is False  # and is now tight: re-inserting changes nothing
    with pytest.raises(ValueError):
        insert(l0, 0, beta, "nonsense")
    floor = floor_l1_float(d, beta)["l1_floor"]
    # invariance: random schedule, violations stay zero and l_1 stays above the floor
    r = run_schedule(l0, beta, make_random(d, beta), 1500, "uniform", seed=3)
    assert r["max_violation"] < 1e-12 and r["min_l1"] >= floor - 1e-9 and r["changes"] > 100
    # convergence: all-block tours reach the all-tight profile
    r = run_schedule(l0, beta, make_tours(d, beta), 20000, "uniform")
    assert np.max(np.abs(r["final"] - tight_profile(d, beta))) < 1e-8 and abs(r["final_l1"] - floor) < 1e-8
    # the HKZ completion violates admissibility and drives l_1 below the floor (at (100, 20): tours end about 1.7 below)
    d2, b2 = 100, 20
    r = run_schedule(lll_like_profile(d2), b2, make_tours(d2, b2), 2500, "hkz")
    assert r["max_violation"] > 0.1 and r["min_l1"] < floor_l1_float(d2, b2)["l1_floor"] - 0.5
    # tail-first: reverse tours until a full reverse tour changes nothing, then forward tours starting at 0, 1, 2, ...
    d3, b3 = 40, 8
    sched, l, last, seq, rng = make_tail_first(d3, b3), lll_like_profile(d3), False, [], random.Random(0)
    for _ in range(20000):
        i = sched(l, b3, rng, last)
        seq.append(i)
        l, last = insert(l, i, b3, "uniform")
    m = d3 - 1
    k = next(t for t in range(1, len(seq)) if seq[t] == 1 and seq[t - 1] == 0)
    assert seq[k - 1:k - 1 + m] == list(range(m)) and seq[k - 3:k - 1] == [1, 0]  # ... preceded by the end of a reverse tour
    cen = consistency_census(d3, b3, 300, completions=("uniform",), seeds=(0,))
    assert len(cen["rows"]) == 5 and all(row["min_l1_minus_floor"] >= -1e-9 and row["max_violation"] < 1e-12 for row in cen["rows"])


def test_insertion_identity_and_physical_insertion():
    """Physical single insertions (exact SVP + insert + LLL on the block) conserve the block volume and stay within the block; exact
    insertions over all blocks until stable give an SVP-tight basis, on which the per-basis identity  l_1 - S/d = h(0) - sum y_i eps_i(B)
    holds with zero residual and every block tight; the census reports before/after violations."""
    from latticelab.insertion import insertion_census, single_insertion, weighted_subgh_mass
    from latticelab.profile import block_gh_ratios

    d, beta = 30, 10
    A = lll(qary(d, d // 2, 2 ** 16 + 1, seed=17))
    r, B = single_insertion(A, 3, beta)
    assert r["support_within_block"] and abs(r["sum_delta_block"]) < 1e-8 and abs(log_volume(B) - log_volume(A)) < 1e-6
    if r["changed"]:
        assert r["removed_mass"] > 0 and len(r["delta"]) == beta
    with pytest.raises(ValueError):
        single_insertion(A, d - 1, beta)
    # exact tours until stable: SVP-tight everywhere, residual zero, identity exact
    cur, stable, tours = A, False, 0
    while not stable and tours < 60:
        stable, tours = True, tours + 1
        for kappa in range(d - 1):
            rr, cur = single_insertion(cur, kappa, beta)
            stable = stable and not rr["changed"]
    assert stable
    w = weighted_subgh_mass(cur, beta)
    # fpylll's svp_reduction enumerates with radius delta_LLL * ||b_kappa^*||^2 (delta = 0.99), so a block is left alone when its shortest
    # vector is within 0.5 % of ||b_kappa^*||: SVP-tight up to that slack, residual R(B) <= (-log(0.99)/2) * sum(y)
    slack = -0.5 * math.log(0.99)
    assert max(math.log(row["b_star_over_lambda1"]) for row in block_gh_ratios(cur, beta)["blocks"]) <= slack + 1e-9
    assert 0 <= w["residual"] <= slack * sum(w["y"]) + 1e-9 and w["frac_tight"] > 0.8
    assert abs(w["head_minus_mean"] - (w["floor_h0"] - w["weighted_eps_signed"] + w["residual"])) < 1e-8
    assert w["weighted_eps_positive"] >= max(0.0, w["weighted_eps_signed"]) - 1e-12 and len(w["eps"]) == d - 1
    # on the LLL basis the residual is nonnegative and the identity is an inequality
    w0 = weighted_subgh_mass(A, beta)
    assert w0["residual"] >= -1e-9 and w0["head_minus_mean"] >= w0["floor_h0"] - w0["weighted_eps_signed"] - 1e-9
    c = insertion_census(A, beta, [2, 12])
    for row in c["rows"]:
        if row["changed"]:
            assert {"max_profile_violation_before", "max_profile_violation_after", "max_violation_increase", "violations_created"} <= row.keys()
            assert row["violations_created"] >= 0 and row["max_profile_violation_before"] >= 0


def test_two_sided_floor_certificate():
    """The two-sided (primal + dual block-GH) floor: the exact two-family certificate reproduces the double-precision LP optimum, its shift
    above the primal floor is a rigorously positive arb ball, the degenerate vertex at (200, 40) is handled, the pure GSA line is primal- and
    dual-tight in the body, and the dual-only LP is bounded but far below the primal floor."""
    from latticelab.dual_floor import combined_certificate, constraint_rows, float_lp, primal_witness
    from latticelab.profile_floor import floor_l1_float, log_chat

    for d, beta in ((60, 10), (200, 40)):
        c = combined_certificate(d, beta)
        assert abs(c["l1_bound"] - c["float_lp_value"]) < 1e-7 and c["shift_certified_positive"] and c["shift"] > 0
        assert c["n_active_A"] + c["n_active_D"] == d - 1 and c["shift_ball"].lower() > 0
        assert abs(c["primal_floor"] - floor_l1_float(d, beta)["l1_floor"]) < 1e-9
        # the primal witness: a rigorously feasible profile whose head is an exact rational upper bound on the LP minimum, enclosing the
        # minimum together with the dual bound to within the shift t (d-1)/2 (roundoff after refinement)
        w = primal_witness(d, beta, dual_mode="all")
        assert w["min_slack_lower"] >= 0 and w["t"] >= 0 and w["n_rows"] == (d - 1) + (d - 1)
        from flint import arb, fmpq

        qa = arb(fmpq(w["q"].numerator, w["q"].denominator))  # exact head of the witness
        assert c["l1_bound_ball"].lower() <= qa and 0 <= float((qa - c["l1_bound_ball"]).upper()) < 1e-7
        assert abs(w["rhf_witness"] - c["rhf_bound"]) < 1e-9 and w["max_violation_before_shift"] < 1e-9
    # the full-size dual family alone shifts the floor by less than (or as much as) the full family
    cf = combined_certificate(200, 40, dual_mode="full")
    assert cf["dual_mode"] == "full" and 0 < cf["shift"] <= combined_certificate(200, 40)["shift"] + 1e-12 and all(e >= 40 for e in cf["active_D_positions"])
    # the pure GSA line of decrement s = 2 L(beta)/(beta-1) makes every full primal and full dual constraint tight
    d, beta = 80, 20
    rows, consts = constraint_rows(d, beta)
    s = 2 * log_chat(beta) / (beta - 1)
    l = [-(j - 1) * s for j in range(1, d + 1)]
    for (kind, idx, n, v), cst in zip(rows, consts):
        if n == beta:
            assert abs(sum(float(x) * lj for x, lj in zip(v, l)) - cst) < 1e-9
    # the two-sided LP lies above the primal floor; the dual-only LP is bounded and lies far below it
    res_two, _, _ = float_lp(d, beta)
    res_dual_only, _, _ = float_lp(d, beta, primal=False)
    fp = floor_l1_float(d, beta)["l1_floor"]
    assert res_two.status == 0 and res_two.fun > fp
    assert res_dual_only.status == 0 and res_dual_only.fun < fp - 0.1
    with pytest.raises(ValueError):
        constraint_rows(d, beta, "sideways")


def test_two_sided_blocksize_scan():
    """two_sided_beta_floor decides every beta in its range rigorously -- failing betas by the exact dual bound, the passing beta by the
    primal witness (disjoint arb balls) -- returns the first passing beta, and never crosses before the primal scan does (the two-sided
    floor dominates the primal floor at every beta); the CLI archives a row."""
    import json

    from latticelab.dual_floor import combined_certificate, main, two_sided_beta_floor
    from latticelab.profile_floor import beta_floor_for_target, floor_l1_float, gsa_delta_ball

    d = 100
    # decreasing regime (beta > 36): a target strictly between the two-sided floors at beta = 60 and beta = 59
    c60, c59 = combined_certificate(d, 60, dual_mode="full"), combined_certificate(d, 59, dual_mode="full")
    assert c60["rhf_bound_ball"].rad() < 1e-60 and c59["rhf_bound"] > c60["rhf_bound"]
    target = c60["rhf_bound"] + 1e-9
    lines = []
    r = two_sided_beta_floor(d, target, 57, 63, log=lines.append)
    assert r["beta_floor"] == 60 and [x["beta"] for x in r["decisions"]] == [57, 58, 59, 60] and len(lines) == 4
    assert all(not x["reaches"] and x["certified_by"] == "dual bound" for x in r["decisions"][:-1])
    last = r["decisions"][-1]
    assert last["reaches"] and last["certified_by"] == "primal witness" and 0 <= last["enclosure_width_l1"] < 1e-7
    assert last["rhf_witness"] >= last["rhf_two_sided_lower"] and "primal witness" in r["minimality"]
    # the reported width is an upper bound on the exact enclosure width (directed rounding), not a nearest-rounded difference
    from flint import arb, fmpq
    from latticelab.dual_floor import primal_witness as _pw

    w60 = _pw(d, 60)
    exact_width = (arb(fmpq(w60["q"].numerator, w60["q"].denominator)) - c60["l1_bound_ball"]).upper()
    assert arb(last["enclosure_width_l1"]) >= exact_width
    assert all(x["shift_certified_positive"] and x["shift"] > 0 for x in r["decisions"])
    # dominance: against the pure-GSA target at beta_spec = 45 the two-sided crossing is at or after the primal exact-all crossing
    tb = gsa_delta_ball(45)
    primal = beta_floor_for_target(d, tb, beta_lo=40, beta_hi=60, exact_all=True)
    two = two_sided_beta_floor(d, tb, 40, 60)
    assert primal["beta_floor"] is not None and two["beta_floor"] is not None and two["beta_floor"] >= primal["beta_floor"]
    for x in two["decisions"]:  # every decided beta's two-sided floor lies above the primal floor on the root-Hermite factor
        assert x["rhf_two_sided_lower"] > floor_l1_float(d, x["beta"])["root_hermite_floor"]
    # undecidable: a target strictly inside the certified enclosure [dual bound, witness] at the passing beta
    from latticelab.dual_floor import primal_witness

    w = primal_witness(d, two["beta_floor"])
    cb = combined_certificate(d, two["beta_floor"], dual_mode="full")
    if w["rhf_witness_ball"].lower() > cb["rhf_bound_ball"].upper():  # a nonempty gap exists at double precision
        mid = (cb["rhf_bound_ball"].upper() + w["rhf_witness_ball"].lower()) / 2
        with pytest.raises(ValueError):
            two_sided_beta_floor(d, mid, two["beta_floor"], two["beta_floor"])
    # no beta passes: an unreachable target
    none = two_sided_beta_floor(d, "1.0", 58, 59)
    assert none["beta_floor"] is None and len(none["decisions"]) == 2 and "no beta" in none["note"]
    with pytest.raises(ValueError):
        two_sided_beta_floor(d, target, 70, 60)
    with pytest.raises(ValueError):
        two_sided_beta_floor(d, target, 58, 60, prec=8192, max_prec=4096)
    # a witness from a caller-supplied badly infeasible vector (an increasing line: every constraint violated) still closes, with a large
    # shift t rounded upwards exactly at any magnitude
    from latticelab.dual_floor import primal_witness

    bad = primal_witness(d, 60, x=np.linspace(-3.0, 3.0, d), refine_iters=0)
    assert bad["min_slack_lower"] >= 0 and bad["t"] > 0.05 and float(bad["q"]) > combined_certificate(d, 60, dual_mode="full")["l1_bound"]
    # CLI archives a row keyed by (d, eps, dual_mode)
    out = "/tmp/artifacts/two_sided_test.json"
    main(["--d", str(d), "--beta-spec", "45", "--lo", str(two["beta_floor"]), "--hi", str(two["beta_floor"] + 1), "--out", out])
    main(["--d", str(d), "--beta-spec", "45", "--lo", str(two["beta_floor"]), "--hi", str(two["beta_floor"] + 1), "--out", out])
    arch = json.load(open(out))
    assert len(arch["rows"]) == 1 and arch["rows"][0]["beta_floor"] == two["beta_floor"] and arch["rows"][0]["beta_spec"] == 45


def test_poisson_world_tail_identity_call_price_and_budget_bounds():
    """Lemma (tail multipliers): y_{d-m} = (1/(m beta)) sum_{j<=min(m,d-beta)} j y_{d-beta+1-j} exactly, including beta > d/2; A = sum y_i/beta_i
    lies in [(d-1)/(beta(beta-1)), 2(d-1)/(beta(beta-1))] (equality below at beta = 2 only); every y_k <= y_1 and rho = y_1/beta exactly;
    the entropy form A(log(N/2)+gamma-H(w)) equals the water-filling optimum when no floor binds and dominates the uniform allocation; the
    adaptive bound exceeds it by (1-gamma)A; the Wald bound holds for the first-exceedance stopping rule in Monte Carlo."""
    from fractions import Fraction

    from latticelab.poisson_world import (EULER_GAMMA, adaptive_bound, call_price, calls_to_consume, concentration, expected_mass,
                                          gumbel_stopped_max_check, optimal_fixed_mass, tail_identity_check, waterfilling, weight_entropy)

    for d, beta in ((30, 6), (60, 10), (60, 40), (60, 59), (100, 51)):
        assert tail_identity_check(d, beta)
        cp = call_price(d, beta)
        y, bs = cp["y"], cp["block_sizes"]
        assert all(yk <= y[0] for yk in y) and cp["rho"] == y[0] / beta == Fraction(d - 1, d * (beta - 1))
        assert cp["lower"] < cp["A"] <= cp["upper"]
        assert cp["A"] == sum((yi / b for yi, b in zip(y, bs)), Fraction(0))
    assert call_price(20, 2)["A"] == Fraction(19, 2)  # beta = 2: A = (d-1)/(beta(beta-1)) exactly
    d, beta = 60, 10
    we = weight_entropy(d, beta)
    assert 0 < we["min_entropy"] <= we["entropy"] <= we["max_entropy"] + 1e-12
    N = 4 * we["N_floor"]
    wf = waterfilling(d, beta, N)
    assert abs(wf["expected_mass"] - optimal_fixed_mass(d, beta, N)) < 1e-9 and wf["floor_binding"] == 0
    assert wf["uniform_mass"] < wf["expected_mass"] <= wf["fixed_allocation_bound"]
    assert abs(adaptive_bound(d, beta, N) - optimal_fixed_mass(d, beta, N) - (1 - EULER_GAMMA) * we["A"]) < 1e-12
    assert adaptive_bound(d, beta, N, with_entropy=False) > adaptive_bound(d, beta, N)
    # calls to consume a mass invert the entropy optimum
    c = calls_to_consume(d, beta, optimal_fixed_mass(d, beta, N))
    assert abs(c["calls"] - N) < 1e-6 * N and c["valid"]
    # a small allocation: expected_mass is the exact Poisson formula
    alloc = [1.0] * (d - 1)
    assert abs(expected_mass(d, beta, alloc) - we["A"] * (math.log(0.5) + EULER_GAMMA)) < 1e-12
    con = concentration(d, beta, 0.5)
    assert con["variance"] <= con["variance_bound"] + 1e-15 and 0 <= con["tail_bound"] <= 1
    # Wald: the stopped maximum of the first-exceedance rule lies below log(E tau) + 1 (up to Monte Carlo error) and above log n + gamma
    g = gumbel_stopped_max_check(50, trials=100000, seed=3)
    se = g["sd_max"] / math.sqrt(100000)
    assert g["mean_max"] <= g["wald_bound"] + 5 * se and g["mean_max"] > g["fixed_n_formula"] + 0.2
    with pytest.raises(ValueError):
        tail_identity_check(10, 10)
    with pytest.raises(ValueError):
        waterfilling(d, beta, d - 2)
    # adaptive concentration: P[M >= A log(N/2) + t] <= e^{1 - t/(2 rho)} for every adaptive rule; two rules, small (d, beta)
    from latticelab.poisson_world import adaptive_tail_bound, simulate_adaptive

    d, beta, N = 20, 5, 400
    for rule in ("greedy", "chase"):
        sim = simulate_adaptive(d, beta, N, trials=400, seed=11, rule=rule)
        for t, rec in sim["exceedances"].items():
            assert rec["empirical"] <= rec["bound"] + 0.05  # Monte Carlo slack on 400 trials
        assert sim["mean_M"] <= adaptive_tail_bound(d, beta, N, 0.0)["expectation_bound_uniform_level"] + 0.02
    tb = adaptive_tail_bound(d, beta, N, 0.3)
    assert tb["tail_bound"] == min(1.0, math.exp(1 - 0.3 / (2 * tb["rho"])))
    # the consistent uniform-completion world: pathwise sum y_i v_i^+ <= sum y_i (best draw)^+ along every run, the profile identity holds,
    # and the head deficit obeys the adaptive tail bound in Monte Carlo (three schedules, including the adaptive 'head' rule)
    from latticelab.poisson_world import uniform_world_run, uniform_world_tail_check

    for sched in ("random", "tours", "head"):
        r = uniform_world_run(40, 8, 600, seed=2, schedule=sched)
        assert r["pathwise_worst_gap"] <= 1e-9 and r["coordinatewise_worst_gap"] <= 1e-9 and abs(r["identity_gap"]) < 1e-9
        assert r["floor_h0"] - r["head_minus_mean"] <= r["mass_best_positive"] + 1e-9  # the head beats the floor by at most the best draws
    chk = uniform_world_tail_check(20, 5, 300, trials=200, seed=5, schedule="head")
    assert chk["pathwise_worst_gap"] <= 1e-9
    for t, rec in chk["exceedances"].items():
        assert rec["empirical"] <= rec["bound"] + 0.05
    with pytest.raises(ValueError):
        adaptive_tail_bound(20, 5, 400.5, 0.1)
    with pytest.raises(ValueError):
        uniform_world_run(40, 8, 10, schedule="nonsense")
    with pytest.raises(ValueError):
        uniform_world_run(40, 8, -1)
    with pytest.raises(ValueError):
        uniform_world_run(40, 8, 10, check_every=0)
    for bad_N in (0, 1):
        with pytest.raises(ValueError):
            uniform_world_tail_check(20, 5, bad_N, trials=2)
    with pytest.raises(ValueError):
        uniform_world_tail_check(20, 5, 10, trials=0)
    for bad in (1.5, float("nan"), float("inf"), Fraction(18014398509481985, 2), True):
        with pytest.raises(ValueError):
            uniform_world_run(20, 5, bad)
        with pytest.raises(ValueError):
            uniform_world_run(20, 5, 10, check_every=bad)
    with pytest.raises(ValueError):
        uniform_world_tail_check(20, 5, 2.5, trials=1)
    with pytest.raises(ValueError):
        uniform_world_tail_check(20, 5, Fraction(18014398509481985, 2), trials=1)
    assert uniform_world_run(20, 5, 10.0, check_every=2.0)["N"] == 10  # integral floats are accepted and normalised
    r0 = uniform_world_run(20, 5, 0)  # zero queries: the admissible start, no draws, head above the floor
    assert r0["queried_positions"] == 0 and r0["mass_best_positive"] == 0 and r0["head_minus_floor"] >= -1e-12


def test_kyber_spec_chain_reproduction_and_targets():
    """The specification's primal chain (condition (9), Chen-Nguyen delta, m optimised over [0, (k+1)n]) reproduces the round-3 blocksizes to
    within three (406/624/874 against the printed 403/625/877); the arb target ball agrees with its double-precision form; blocksizes with
    2b >= d + 1 for every m are handled; a floor above the GSA delta can only raise the chain's blocksize."""
    import math

    from latticelab.spec_chain import KYBER, N_RING, chain, gsa_chain, log_delta_req, target_ball
    from latticelab.profile import delta_gsa

    for name, (b_exp, d_exp) in (("Kyber512", (406, 1026)), ("Kyber768", (624, 1427)), ("Kyber1024", (874, 1867))):
        p = KYBER[name]
        g = gsa_chain(p["k"], p["eta1"], b_lo=b_exp - 5, b_hi=b_exp + 5)
        assert g["b"] == b_exp and abs(g["b"] - p["printed"][1]) <= 3 and g["m_range"] == [0, (p["k"] + 1) * N_RING]
        assert g["d_best"] == g["m_best"] + p["k"] * N_RING + 1
    from fractions import Fraction

    for b, m, k in ((406, 513, 2), (624, 658, 3), (877, 842, 4)):
        t = target_ball(b, m, k, Fraction(3 if k == 2 else 2, 2))
        assert abs(math.log(float(t.mid())) - log_delta_req(b, m, k, Fraction(3 if k == 2 else 2, 2))) < 1e-12 and t.rad() < 1e-60
    with pytest.raises(ValueError):
        log_delta_req(600, 0, 2, Fraction(3, 2))  # d = 513 < 2b: direction change
    # a synthetic floor 5e-5 above the GSA delta shifts the chain by a few blocksizes, never downwards
    f = chain(2, 3, lambda d, b: math.log(delta_gsa(b)) + 5e-5, b_lo=400, b_hi=430)
    assert f["b"] is not None and f["b"] > 406
    none = chain(2, 3, lambda d, b: 0.05, b_lo=400, b_hi=402)  # an absurd floor: nothing passes, no crash
    assert none["b"] is None and "no b" in none["note"]


def test_reversed_dual_basis_and_two_sided_identity():
    """The reversed dual basis q B^{-T} is integral for q-ary lattices, its profile is log q minus the reversed primal profile, its projected
    block at position d-e+1 is q times the dual of the primal block ending at e (checked against a brute-force exact minimum of the dual
    block's rational Gram matrix on a tiny lattice), and the two-sided per-basis decomposition closes to roundoff on a real BKZ basis with
    nonnegative residuals; the strict-tour census evaluates the identity at its checkpoints up to the first clean tour, where the residual
    lies within the 0.99 slack, and emits no row for later checkpoints (the basis no longer changes)."""
    import itertools
    from fractions import Fraction

    from latticelab.dual_census import dual_block_ratios, reversed_dual_basis, strict_tours_census, two_sided_mass

    # tiny lattice: exact projected block Gram matrix, its inverse (the dual block's Gram matrix), brute-force minimum
    q, d, beta = 97, 6, 3
    A = lll(qary(d, 3, q, seed=4))
    Bn = [[Fraction(int(A[i, j])) for j in range(d)] for i in range(d)]
    # exact Gram-Schmidt to get the projections pi_s(b_k), s = e - n + 1 (1-based) for the block ending at e
    e, n = 5, min(beta, 5)
    s = e - n  # 0-based index of the first block vector
    gs = []
    for k in range(d):
        v = Bn[k][:]
        for g in gs:
            mu = sum(a * b for a, b in zip(Bn[k], g)) / sum(a * a for a in g)
            v = [a - mu * b for a, b in zip(v, g)]
        gs.append(v)
    def proj(vec):  # orthogonal to b_1..b_s
        v = vec[:]
        for g in gs[:s]:
            mu = sum(a * b for a, b in zip(vec, g)) / sum(a * a for a in g)
            v = [a - mu * b for a, b in zip(v, g)]
        return v
    block = [proj(Bn[k]) for k in range(s, e)]
    G = [[sum(a * b for a, b in zip(u, w)) for w in block] for u in block]
    # inverse of G by Gauss-Jordan in Fractions
    M = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(G)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    Ginv = [row[n:] for row in M]
    best = min(sum(x[i] * Ginv[i][j] * x[j] for i in range(n) for j in range(n))
               for x in itertools.product(range(-6, 7), repeat=n) if any(x))
    dl = dual_block_ratios(A, beta, q)
    D = reversed_dual_basis(A, q)
    from latticelab.profile import block_gh_ratios as bgr

    row = next(r for r in bgr(D, beta)["blocks"] if r["i"] == d - e + 1)
    assert row["beta_i"] == n and abs(row["lambda1"] ** 2 / q ** 2 - float(best)) < 1e-6 * float(best)
    assert abs(dl["eps"][e] - row["log_gh_over_lambda1"]) < 1e-12 and dl["n"][e] == n
    # profile duality and the two-sided identity on a real BKZ basis
    q, d, beta = 2 ** 16 + 1, 30, 8
    B = bkz(qary(d, d // 2, q, seed=6), beta, tours=4)
    D = reversed_dual_basis(B, q)
    p, pd = gs_profile(B), gs_profile(D)
    assert max(abs(pd[i] - (math.log(q) - p[d - 1 - i])) for i in range(d)) < 1e-9
    ts = two_sided_mass(B, beta, q, "full")
    assert abs(ts["identity_gap"]) < 1e-9 and ts["residual_two_sided"] >= -1e-12 and min(ts["r_dual"]) >= -1e-9
    assert ts["n_active_A"] + ts["n_active_D"] == d - 1 and ts["n_full_dual_blocks"] == d - beta + 1
    assert abs((ts["head_minus_mean"] - ts["two_sided_floor_h0"]) - (-ts["mass_A_active"] - ts["mass_D_active"] + ts["residual_two_sided"])) < 1e-9
    with pytest.raises(ValueError):
        reversed_dual_basis(B, q + 2)  # (q+2) B^{-T} is not integral
    checkpoints = [1, 3, 40]
    c = strict_tours_census(20, 6, 2 ** 12 + 3, 3, checkpoints)
    assert c["stable_at_tour"] is not None and c["stable_at_tour"] <= 40 and len(c["basis"]) == 20
    # rows are emitted for the checkpoints up to and including the first one at or beyond the clean tour and for no later checkpoint (the
    # basis no longer changes, so a later row would only repeat the last one); the last row records the clean tour as its tour count
    first_stable_checkpoint = next(T for T in checkpoints if T >= c["stable_at_tour"])
    assert [r["requested_tours"] for r in c["rows"]] == [T for T in checkpoints if T <= first_stable_checkpoint]
    assert all(r["tours"] == r["requested_tours"] and r["stable_at_tour"] is None for r in c["rows"][:-1])
    assert c["rows"][-1]["tours"] == c["stable_at_tour"] == c["rows"][-1]["stable_at_tour"] and c["rows"][-1]["resumed_from"] is None
    from latticelab.profile_floor import dual_certificate

    slack = -0.5 * math.log(0.99) * sum(float(v) for v in dual_certificate(20, 6)[0])
    assert all(r["residual"] >= -1e-9 for r in c["rows"])
    # after a clean tour every block is SVP-tight up to the 0.99 slack, so the residual is within the slack bound
    assert c["rows"][-1]["residual"] <= slack + 1e-9


def test_negacyclic_orbit_angles():
    """X has order 2n on nonzero vectors, X^n = -1, rotations preserve the norm; orbit_angles counts ordered pairs within theta exactly (signed
    cosine, the antipode never counted) and agrees with a direct pairwise computation; the 90-degree endpoint uses the exact test c >= 0."""
    import itertools
    import random

    from latticelab.lattices import negacyclic_rotate, orbit_angles

    v = [1, 0, 0, 0]  # g = 1, f = 0 in Z[X]/(X^2 + 1)
    assert negacyclic_rotate(v, 1) == [0, 1, 0, 0] and negacyclic_rotate(v, 2) == [-1, 0, 0, 0] and negacyclic_rotate(v, 4) == v
    oa = orbit_angles(v, math.pi / 2)
    assert oa["pairs_within_theta"] == 2 * 2 * 2 and oa["max_abs_cos"] == 0 and abs(oa["min_angle_deg"] - 90) < 1e-12  # X v and X^3 v are orthogonal
    assert orbit_angles(v)["pairs_within_theta"] == 0
    rng = random.Random(5)
    n = 8
    w = [rng.choice((-1, 0, 1)) for _ in range(2 * n)]
    w[0] = 1
    orbit = [negacyclic_rotate(w, k) for k in range(2 * n)]
    assert len({tuple(u) for u in orbit}) == 2 * n and all(sum(x * x for x in u) == sum(x * x for x in w) for u in orbit)
    assert orbit[n] == [-x for x in w]
    norm2 = sum(x * x for x in w)
    direct = sum(1 for a, b in itertools.permutations(range(2 * n), 2) if 2 * sum(x * y for x, y in zip(orbit[a], orbit[b])) >= norm2)
    oa = orbit_angles(w)
    assert oa["pairs_within_theta"] == direct and len(oa["autocorrelations"]) == n - 1
    assert oa["angular_excess"] == direct / ((2 * n) ** 2 * oa["cap_fraction"])
    with pytest.raises(ValueError):
        orbit_angles([1, 1])
    with pytest.raises(ValueError):
        orbit_angles([0, 0, 0, 0])
    with pytest.raises(ValueError):
        orbit_angles(w, 2.0)


def test_canonical_forward_tour_flags_and_conveyor():
    """Theorem (canonical forward tour): with block-supported completions the output flag of a complete forward tour is determined by the
    successive block minima -- bounded-LLL and bounded-HKZ completions give the same profile, the same queried-block records and exactly the
    same flag; reverse tours and the classical prefix-LLL conveyor do not; a single insertion's completion changes only blocks with
    starts in [kappa-beta+2, kappa-1] u [kappa+2, kappa+beta-1] (that set is an upper bound; the complement, including the three forced
    starts, is unchanged exactly)."""
    from latticelab.flag import VARIANTS, canonical_tour_check, forward_tour, same_sublattice, single_insertion_dependence

    d, beta, q, seed = 30, 6, 2 ** 12 + 3, 4
    r = canonical_tour_check(d, beta, q, seed)
    f = r["forward_bounded"]
    assert f["max_profile_diff"] < 1e-8 and f["max_block_logvol_diff"] < 1e-8 and f["max_block_min_diff"] < 1e-8
    assert f["n_flag_members_equal"] == d - 1 and f["first_flag_member_differing"] is None
    assert r["reverse_bounded"]["n_positions_differ_1e-6"] > 0 and r["reverse_bounded"]["n_flag_members_equal"] < d - 1
    assert r["forward_conveyor_vs_bounded"]["n_positions_differ_1e-6"] > 0 and r["forward_conveyor_vs_bounded"]["n_flag_members_equal"] < d - 1
    A = lll(qary(d, d // 2, q, seed=seed))
    inserted = []
    for kappa in (3, 8, 14):
        s = single_insertion_dependence(A, kappa, beta)
        lo, hi = kappa - beta + 2, kappa + beta - 1
        assert s["predicted_dependent_starts"] == [t for t in range(0, d - 1) if lo <= t <= kappa - 1 or kappa + 2 <= t <= hi]
        assert s["max_diff_forced"] < 1e-8 and s["n_dependent_starts_differing_1e-6"] >= 1
        assert set(s["profile_positions_differing_1e-6"]) <= set(range(kappa + 1, kappa + beta))  # l_kappa forced, outside the block untouched
        inserted.append(s["inserted"])
    assert any(inserted)
    # the strict tour inserts: after one bounded forward tour the head is at most the LLL head, and every recorded minimum is at most the
    # block's first Gram-Schmidt norm before the step (the block record is the enumerated block)
    B, rec = forward_tour(A, beta, "bounded_lll")
    assert gs_profile(B)[0] <= gs_profile(A)[0] + 1e-12 and len(rec) == d - 1 and all(x["size"] == min(beta, d - x["kappa"]) for x in rec)
    assert same_sublattice([[1, 0], [0, 1]], [[1, 1], [0, 1]]) and not same_sublattice([[1, 0], [0, 2]], [[2, 0], [0, 1]])
    assert not same_sublattice([[1, 0], [0, 1]], [[1, 0], [0, 2]]) and same_sublattice([], [])
    assert set(VARIANTS) == {"bounded_lll", "bounded_hkz", "conveyor"}
    with pytest.raises(ValueError):
        forward_tour(A, beta, "nonsense")
    with pytest.raises(ValueError):
        single_insertion_dependence(A, d - beta + 1, beta)


def test_strict_census_resume_is_exact():
    """The strict census resumes from an archived basis exactly: the state of the dynamics is the integer basis, so running to 4 tours
    straight and running to 2 then resuming to 4 give the same basis and the same statistics; only the checkpoints above the resume point are
    evaluated, rows carry the float type and the resume point, the checkpoint callback sees every row, and 'auto' selects double below
    d = 150 and long double from there."""
    from latticelab.dual_census import gso_float_type, strict_tours_census

    d, beta, q, seed = 30, 8, 2 ** 12 + 3, 5
    straight = strict_tours_census(d, beta, q, seed, [2, 4])
    seen = []
    first = strict_tours_census(d, beta, q, seed, [2], on_checkpoint=lambda row, basis: seen.append((row["tours"], basis)))
    assert [t for t, _ in seen] == [2] and seen[0][1] == first["basis"] and first["rows"][0]["resumed_from"] is None
    assert first["basis"] != straight["basis"]  # two more tours change the basis at this size
    resumed = strict_tours_census(d, beta, q, seed, [2, 4], start_basis=first["basis"], tours_done=2)
    assert resumed["basis"] == straight["basis"] and [r["requested_tours"] for r in resumed["rows"]] == [4]
    assert resumed["rows"][0]["resumed_from"] == 2 and resumed["rows"][0]["float_type"] == "d" and resumed["resumed_from"] == 2
    for key in ("weighted_eps_signed", "weighted_eps_positive", "residual", "head_minus_mean", "frac_tight"):
        assert abs(resumed["rows"][0][key] - straight["rows"][1][key]) < 1e-9
    assert gso_float_type(149) == "d" and gso_float_type(150) == "ld" and gso_float_type(30, "ld") == "ld"
    with pytest.raises(ValueError):
        strict_tours_census(d, beta, q, seed, [4], tours_done=2)
    with pytest.raises(ValueError):
        strict_tours_census(d, beta, q, seed, [4], start_basis=first["basis"][:-1], tours_done=2)


def test_residual_cap_at_forced_neighbour():
    """Lemma (residual cap): after an insertion at kappa the forced neighbour block [kappa+1, kappa+beta+1) contains the queried block's HKZ
    residual [kappa+1, kappa+beta) as a primitive sublattice, so lambda_1(Q/v) <= lambda_1(P/v); the neighbour's signed ratio decomposes
    exactly as gh_shift + res_ratio + gap with gap >= 0 and agrees with block_gh_ratios on the new basis; in a bounded forward tour the
    minimum recorded for the forced neighbour is what the next step finds; the GSA-tight deterministic term changes sign at 36; the random
    control returns one row per lattice."""
    from latticelab.profile import block_gh_ratios
    from latticelab.profile_floor import log_chat
    from latticelab.residual import (control_dependence, forced_neighbour_decomposition, gsa_tight_gh_shift, inheritance_stats, residual_census,
                                     residual_ratio_random)

    d, beta, q, seed = 30, 6, 2 ** 12 + 3, 4
    A = lll(qary(d, d // 2, q, seed=seed))
    kappa = 3
    r, B = forced_neighbour_decomposition(A, kappa, beta)
    a = r["after"]
    assert r["cap_holds"] and a["gap"] >= -1e-12 and abs(a["gh_shift"] + a["res_ratio"] + a["gap"] - a["eps_neighbour"]) < 1e-12
    assert abs(r["created_ratio"] - (a["eps_neighbour"] - r["before"]["eps_neighbour"])) < 1e-12
    row = next(x for x in block_gh_ratios(B, beta)["blocks"] if x["i"] == kappa + 2)  # 1-based start kappa+2 = 0-based kappa+1
    assert abs(row["log_gh_over_lambda1"] - a["eps_neighbour"]) < 1e-9 and abs(row["lambda1"] - a["lambda1_neighbour"]) < 1e-9
    res = next(x for x in block_gh_ratios(B, beta - 1)["blocks"] if x["i"] == kappa + 2)  # the residual: size beta-1 at the same start
    assert abs(res["lambda1"] - a["lambda1_residual"]) < 1e-9 and abs(res["log_gh_over_lambda1"] - a["res_ratio"]) < 1e-9
    # the GSA-tight form of the deterministic term reproduces the closed form on an exactly GSA-tight profile: gh_shift depends on the
    # profile only through l_{kappa+beta} and log vol(P/v)
    assert abs(a["gh_shift"] - (log_chat(beta) - log_chat(beta - 1) + a["l_kappa_plus_beta"] / beta - a["log_vol_residual"] / (beta * (beta - 1)))) < 1e-12
    c = residual_census(A, beta, 2)
    assert c["next_query_consistent"] and c["summary"]["n"] == 2 * (d - beta)
    assert all(s["gap"] >= -1e-12 and abs(s["gh_shift"] + s["res_ratio"] + s["gap"] - s["eps_neighbour"]) < 1e-12 for s in c["steps"])
    assert c["summary"]["frac_cap_bound_positive"] <= c["summary"]["frac_eps_neighbour_positive"] + 1e-12  # the cap bounds eps from below
    assert c["final_profile"][0] <= gs_profile(A)[0] + 1e-12 and set(c["per_tour"]) == {1, 2}
    assert gsa_tight_gh_shift(30) > 0 > gsa_tight_gh_shift(40) and abs(gsa_tight_gh_shift(36)) < 1e-4
    assert gsa_tight_gh_shift(26) > 0.005 > gsa_tight_gh_shift(27) > 0  # the |gh_shift| < 0.005 threshold starts at beta = 27
    # the post-processing: adjacent steps of one tour pair the block queried at kappa+1 (eps_neighbour at kappa) with its residual ratio
    st = inheritance_stats(c["steps"])
    assert st["n_pairs"] == 2 * (d - beta - 1) and -1 <= st["corr_eps_res"] <= 1 and st["n_dense"] <= st["n_pairs"]
    assert inheritance_stats(c["steps"][:2]) == {"n_pairs": 1}
    rr = residual_ratio_random(8, 5, q, seed0=7)
    assert len(rr["rows"]) == 5 and abs(rr["mean_res_ratio"] - sum(x["res_ratio"] for x in rr["rows"]) / 5) < 1e-12
    dp = control_dependence(rr["rows"], 8)
    assert dp["n"] == 5 and abs(dp["volume_only_slope"] - 1 / 7) < 1e-12 and dp["n_dense"] == sum(1 for x in rr["rows"] if x["eps_P"] > 0)
    with pytest.raises(ValueError):
        forced_neighbour_decomposition(A, d - beta, beta)  # the neighbour block would not be full
    with pytest.raises(ValueError):
        forced_neighbour_decomposition(A, 0, beta, "conveyor")
    with pytest.raises(ValueError):
        residual_census(A, 2, 1)


def test_prefix_volume_floor_and_detection_chain():
    """Theorem (prefix-volume floor): every prefix sum of an admissible profile is at least the all-tight profile's -- checked on random admissible
    profiles and by the LP; the general linear certificate reproduces the head certificate at e_1 and gives every tight entry and prefix volume
    (exactly nonnegative multipliers for the prefix objectives) to roundoff against the dense solve; no tail entry is bounded above (the LP for
    max l_{d-beta+1} is unbounded).  The detection chain reproduces the round-3 blocksize with the GSA line and gives 417 with the tight profile at
    Kyber512; its certification passes the crossing and refuses a full-range certificate from a sampled stride."""
    from scipy.optimize import linprog

    from latticelab.profile_floor import (block_sizes, dual_certificate, linear_certificate, log_chat, prefix_volume_certificate, tight_entry,
                                          tight_entry_float, tight_prefix_volume, tight_profile)
    from latticelab.spec_chain import certify_detection_chain, detection_chain, detection_entry

    rng = np.random.default_rng(3)
    for d, beta in ((30, 6), (60, 20)):
        tp = tight_profile(d, beta)
        w1, z1 = linear_certificate(d, beta, [1] + [0] * (d - 1))
        assert (w1, z1) == dual_certificate(d, beta)
        assert max(abs(float(tight_entry(d, beta, k).mid()) - tp[k - 1]) for k in range(1, d + 1)) < 1e-9
        assert max(abs(tight_entry_float(d, beta, k) - tp[k - 1]) for k in range(1, d + 1)) < 1e-9
        for m in (1, d // 3, d - beta, d - 1):
            w, z = prefix_volume_certificate(d, beta, m)  # raises if any multiplier is negative
            assert all(v >= 0 for v in w) and abs(float(tight_prefix_volume(d, beta, m).mid()) - tp[:m].sum()) < 1e-9
        # random admissible profiles of the same volume: every prefix sum at least the tight one (the maximum principle)
        bs = block_sizes(d, beta)
        for _ in range(30):
            s = rng.exponential(0.05, size=d - 1) * (rng.random(d - 1) < 0.5)
            l = np.zeros(d)
            for i in range(d - 2, -1, -1):
                n = bs[i]
                l[i] = (l[i + 1:i + n].sum() / n + log_chat(n) + s[i]) / (1 - 1 / n)
            l -= l.mean()
            assert np.all(np.cumsum(l - tp)[:-1] >= -1e-9)
    # LP: min P_{d-beta} is attained by the tight profile, max l_{d-beta+1} is unbounded
    d, beta = 60, 20
    A = np.zeros((d - 1, d)); b = np.zeros(d - 1)
    for i, bi in enumerate(block_sizes(d, beta), start=1):
        A[i - 1, i - 1] = -(1 - 1 / bi); A[i - 1, i:i + bi - 1] = 1 / bi; b[i - 1] = -log_chat(bi)
    c = np.zeros(d); c[:d - beta] = 1.0
    res = linprog(c, A_ub=A, b_ub=b, A_eq=np.ones((1, d)), b_eq=[0.0], bounds=[(None, None)] * d, method="highs")
    assert res.status == 0 and abs(res.fun - tight_profile(d, beta)[:d - beta].sum()) < 1e-6
    c2 = np.zeros(d); c2[d - beta] = -1.0
    assert linprog(c2, A_ub=A, b_ub=b, A_eq=np.ones((1, d)), b_eq=[0.0], bounds=[(None, None)] * d, method="highs").status == 3
    for bad in (("len",), None):
        with pytest.raises(ValueError):
            linear_certificate(d, beta, [1] * (d - 1)) if bad else prefix_volume_certificate(d, beta, d)
    with pytest.raises(ValueError):
        tight_entry(d, beta, 0)
    # the detection chain at Kyber512 (k = 2, eta1 = 3): GSA line 406 (the round-3 chain in this form), tight profile 417; the GSA entry is condition
    # (9)'s right side and the tight entry is the tight profile's l_{d-b+1}
    g = detection_chain(2, 3, "gsa", b_lo=404, b_hi=408, m_range=(480, 560))
    t = detection_chain(2, 3, "tight", 0.0, b_lo=414, b_hi=419, m_range=(480, 560))
    assert g["b"] == 406 and t["b"] == 417 and t["m_best"] >= 480
    dd, S = 1030, 517 * math.log(3329)
    assert abs(detection_entry(dd, 417, S, "tight") - (tight_profile(dd, 417)[dd - 417] + S / dd)) < 1e-9
    assert abs(detection_entry(dd, 417, S, "gsa") - ((2 * 417 - dd - 1) * log_chat(417) / 416 + S / dd)) < 1e-12
    with pytest.raises(ValueError):
        detection_entry(dd, 417, S, "sideways")
    cert = certify_detection_chain(2, 3, t, b_lo=t["b"])  # no earlier b: the crossing alone, rigorously
    assert cert["crossing"] == "passes" and cert["certified"] and cert["earlier_b"] == {}
    sampled = certify_detection_chain(2, 3, t, b_lo=t["b"], m_stride=2)
    assert not sampled["certified"] and sampled["certified_sampled"] and "SAMPLED" in sampled["certification_scope"]
    with pytest.raises(ValueError):
        certify_detection_chain(2, 3, t, b_lo=t["b"], m_stride=0)


def test_thinned_residual_model_and_undercut_law():
    """The thinned residual: the lift constraint gives lambda_1(P/v) >= (sqrt 3/2) lambda_1(P), checked on random lattices through the deterministic
    cap on res_ratio; the unthinned law reproduces the sign-pair constants (E = -(log 2 - gamma)/(beta-1), sd = pi/(sqrt 6 (beta-1)),
    P[res > 0] = 1 - e^{-1/2}); thinning lowers the mean, confines the support to u >= sqrt 3/2 and creates a positive dependence on eps(P) where
    the unthinned law has exactly none; a control of any size scores, including one row; the undercut law vanishes when the layer height
    exceeds the residual minimum, is monotone in it and is bounded by log(u*/h); gap_model_check runs on census steps."""
    from latticelab.residual import (SQRT3_2, _sample_thinned_U, _thinned_survival_grid, compare_control_with_model, gap_model_check, residual_census,
                                     residual_gh_log_offset, residual_ratio_cap, residual_ratio_random, thinned_pit, thinned_residual_law,
                                     thinned_residual_model, thinning_factor, undercut_law)

    g = thinning_factor([0.5, SQRT3_2, 0.95, 1.0, 1.5])
    assert g[0] == 0 and abs(g[1]) < 1e-12 and abs(g[2] - (1 - 2 * math.sqrt(1 - 0.95 ** 2))) < 1e-12 and g[3] == 1 and g[4] == 1
    for beta in (8, 20, 40):
        un = thinned_residual_law(beta, 0.0, thin=False)
        assert abs(un["E_res_ratio"] + (math.log(2) - np.euler_gamma) / (beta - 1)) < 1e-12  # the unthinned Poisson constants, analytic
        assert abs(un["sd_res_ratio"] - math.pi / (math.sqrt(6) * (beta - 1))) < 1e-12
        assert abs(un["P_res_ratio_positive"] - (1 - math.exp(-0.5))) < 1e-12 and abs(thinned_pit(beta, 0.0, un["c"], thin=False) - (1 - math.exp(-0.5))) < 1e-12
        t, S, lc = _thinned_survival_grid(beta, 0.0, 40000, False)  # the quadrature machinery against the analytic survival e^{-1/2} at u = c
        assert abs(float(np.interp(lc, t, S)) - math.exp(-0.5)) < 1e-4
        th = thinned_residual_law(beta, 0.0)
        assert th["E_res_ratio"] < un["E_res_ratio"] - 0.005 and th["E_log_U"] >= math.log(SQRT3_2) - 1e-12
        assert th["P_res_ratio_positive"] < un["P_res_ratio_positive"] and abs(th["cap"] - residual_ratio_cap(beta, 0.0)) < 1e-12 and th["cap"] > th["E_res_ratio"]
    assert abs(residual_gh_log_offset(20) + 0.0311) < 5e-4
    m = thinned_residual_model(20, n_quad=60, grid=5000)
    mu = thinned_residual_model(20, n_quad=60, grid=5000, thin=False)
    assert -0.056 < m["E_res_ratio"] < -0.046 and 0.25 < m["slope_res_on_eps"] < 0.40 and abs(mu["slope_res_on_eps"]) < 1e-6
    assert m["E_res_given_dense"] > m["E_res_given_sparse"] and m["P_pos_given_dense"] > m["P_pos_given_sparse"] and abs(m["P_dense"] - (1 - math.exp(-0.5))) < 0.02
    # scoring a control of five lattices, and of a single row (a `--random 1` archive must not crash: slope and correlation undefined); the
    # inserted vector is the independently enumerated block minimum on every row
    rr = residual_ratio_random(8, 5, 2 ** 12 + 3, seed0=7)
    assert rr["max_svp_mismatch"] < 1e-9 and all("svp_mismatch" in row for row in rr["rows"])
    cm = compare_control_with_model(rr["rows"], 8, grid=5000, n_boot=50)
    assert cm["n"] == 5 and cm["cap_holds_all"] and cm["min_cap_slack"] >= 0 and math.isfinite(cm["model"]["E_res_ratio"]) and math.isfinite(cm["bias_z"])
    assert math.isfinite(cm["se_discrepancy_null"]) and math.isfinite(cm["sign_z"]) and cm["pit_eps_spearman"] is not None and cm["ks_thinned_by_eps_tercile"] == []
    assert math.isfinite(cm["measured"]["bootstrap_se"]["slope_res_on_eps"]) and cm["max_svp_mismatch"] < 1e-9
    one = compare_control_with_model(rr["rows"][:1], 8, grid=5000)
    assert one["n"] == 1 and math.isnan(one["model"]["slope_res_on_eps"]) and math.isfinite(one["model"]["E_res_ratio"]) and one["cap_holds_all"]
    with pytest.raises(ValueError):
        compare_control_with_model([], 8)
    # the sampler draws from the law it is meant to: the PITs of its draws are uniform
    from scipy.stats import kstest

    rng = np.random.default_rng(11)
    draws = [_sample_thinned_U(20, 0.0, rng) for _ in range(1500)]
    pits = [thinned_pit(20, 0.0, u, grid=8000) for u in draws]
    assert kstest(pits, "uniform").pvalue > 1e-3 and min(draws) >= SQRT3_2 - 1e-12
    # the undercut law
    assert undercut_law(20, 0.9, 1.0, 0.97) == {"P_gap_positive": 0.0, "E_gap": 0.0, "count_at_u_star": 0.0}
    u1, u2 = undercut_law(20, 1.05, 0.6, 0.97), undercut_law(20, 1.15, 0.6, 0.97)
    assert 0 < u1["P_gap_positive"] < u2["P_gap_positive"] < 1 and 0 < u1["E_gap"] < u2["E_gap"] <= math.log(1.15 / 0.6) + 1e-12
    assert abs(u1["count_at_u_star"] - ((1.05 ** 2 - 0.6 ** 2) / 0.97 ** 2) ** 9.5) < 1e-12  # one layer below u*
    with pytest.raises(ValueError):
        undercut_law(2, 1.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        thinned_residual_law(20, 0.0, grid=10)
    A = lll(qary(30, 15, 2 ** 12 + 3, seed=4))
    c = residual_census(A, 6, 1, final_ratios=True)
    assert all({"l_kappa", "l_kappa_plus_beta", "log_vol_numerator"} <= s.keys() for s in c["steps"])
    # the output block at kappa+1 contains the queried block's shortest vector: its ratio exceeds the queried one by the volume drift plus a
    # nonnegative non-tightness, exactly
    o = c["output_vs_queried"]
    assert o["n"] == 30 - 6 and o["max_decomposition_gap"] < 1e-9 and o["min_delta_minimum_part"] >= -1e-9
    assert abs(o["mean_delta"] - (o["mean_delta_volume_part"] + o["mean_delta_minimum_part"])) < 1e-9 and len(c["final_basis"]) == 30
    # every position, head block and shrinking tail included: the forced entry is log lambda_1 of the queried block, nu_k = eps(Q_k) + Delta^vol_k,
    # and the dual-weighted head identity l_1 - S/d = h(0) - sum y_k nu_k
    ap = c["all_positions"]
    assert ap["n"] == 29 and ap["max_forced_entry_gap"] < 1e-9 and ap["max_nu_identity_gap"] < 1e-9 and ap["head_identity_gap"] < 1e-9
    assert abs(ap["head_minus_floor"] - (ap["minus_sum_y_eps_query"] + ap["minus_sum_y_dvol"])) < 1e-9 and len(ap["eps_query"]) == 29
    assert "output_vs_queried" not in residual_census(A, 6, 1)
    gm = gap_model_check(c["steps"], 6)
    assert gm["n"] == len(c["steps"]) and 0 <= gm["mean_predicted_P_gap_positive"] <= 1 and gm["mean_predicted_E_gap"] >= 0 and len(gm["calibration_terciles"]) == 3
    assert gap_model_check([{"gap": 0.0}], 6) == {"n": 0}
    # the thinned-residual world: a scalar Markov chain on the queried block's ratio whose profile quantities are tied by the exact identity
    # gh_shift = L(beta) - ((beta-1)/beta) L(beta-1) + (log h_bar - log c)/beta (checked on every census step); two starts agree in the
    # stationary mean; the solved fixed point of the conditional means sits near it; heights can be drawn per step from census values
    from latticelab.residual import census_h_values, census_profile_pairs, gh_shift_from_profile, thinned_world_chain, world_fixed_point

    for s in c["steps"]:
        v = math.exp(s["l_kappa"])
        u_star, h_bar = s["lambda1_residual"] / v, math.exp(s["l_kappa_plus_beta"]) / v
        assert abs(gh_shift_from_profile(6, h_bar, u_star * math.exp(s["res_ratio"])) - s["gh_shift"]) < 1e-9
    w1 = thinned_world_chain(20, 600, h_bar=0.6, seed=1, burn=100, solve_fixed_point=False)
    w2 = thinned_world_chain(20, 600, h_bar=0.6, seed=2, burn=100, eps0=0.3, solve_fixed_point=False)
    assert abs(w1["mean_eps"] - w2["mean_eps"]) < 0.02 and -0.15 < w1["mean_eps"] < 0.05 and 0 <= w1["P_gap_positive"] <= 1 and w1["mean_gap"] >= 0
    assert not w1["from_h_values"] and w1["h_bar"] == 0.6 and -1 <= w1["lag1_corr_eps"] <= 1 and math.isfinite(w1["post_hoc_affine_balance"])
    fp = world_fixed_point(20, 0.6)
    assert math.isfinite(fp["fixed_point"]) and abs(fp["fixed_point"] - w1["mean_eps"]) < 0.04
    hv = census_h_values(c["steps"])
    assert hv.shape == (len(c["steps"]),) and np.all(hv > 0) and census_profile_pairs(c["steps"]).shape == (len(c["steps"]), 2)
    w3 = thinned_world_chain(6, 200, seed=3, burn=50, h_values=hv, solve_fixed_point=False)  # height-driven mode needs no fixed constant
    assert w3["from_h_values"] and abs(w3["h_bar"] - hv.mean()) < 1e-12
    for args, kw in (((20, 10), {"h_bar": 0.0}), ((20, 0), {"h_bar": 0.6}), ((20, 10), {"h_bar": 0.6, "burn": -1}), ((20, 10), {}),
                     ((20, 10), {"h_values": np.zeros(0)}), ((20, 10), {"h_values": [0.0]})):
        with pytest.raises(ValueError):
            thinned_world_chain(*args, **kw)
    with pytest.raises(ValueError):
        gh_shift_from_profile(20, 0.0, 1.0)
    # across tours: the final ratio at every position is exactly the queried ratio at the last entry change plus the block's GH drift since
    # minus the entry drift; the head identity holds at the end; at a clean tour under the 0.99 rule the output ratios equal the queried ones
    from latticelab.residual import tour_bookkeeping

    tb = tour_bookkeeping(A, 6, 40, change_tol=0.005, insert_tol=0.01)
    assert tb["identity_gap"] < 1e-9 and tb["head_identity_gap"] < 1e-9 and tb["insert_tol"] == 0.01 and len(tb["last_change"]) == 29
    assert abs(tb["mass_final"] - (tb["mass_at_last_change"] + tb["mass_drift"] - tb["mass_entry_drift"])) < 1e-9
    assert len(tb["entries_moved_per_tour"]) == 40 and len(tb["insertions_per_tour"]) == 40 and tb["entries_moved_per_tour"][0] > 0
    assert tb["clean_tour"] is not None and tb["clean_tour"] <= 10  # this fixture is clean from tour 6 under the 0.99 rule
    assert tb["nu_final_equals_eps_final_gap"] < 1e-9 and tb["entries_moved_per_tour"][tb["clean_tour"] - 1] == 0 and all(m == 0 for m in tb["entries_moved_per_tour"][tb["clean_tour"] - 1:])
    for args, kw in (((A, 6, 0), {}), ((A, 6, 2), {"insert_tol": 1.5}), ((A, 2, 2), {}), ((A, 6, 2), {"variant": "conveyor"})):
        with pytest.raises(ValueError):
            tour_bookkeeping(*args, **kw)


def test_mass_ledger_and_schedule_game():
    """The across-tours mass ledger: mass^(t+1) - mass^(t) = R^(t+1) + sum y (D + Delta^vol) at every tour to roundoff, with R >= 0; its cumulative
    forms are endpoint functionals -- the drift sum telescopes to sum_k (y_k/n_k)(P^fin - P^0) and the mass change equals (head^0 - head^fin) +
    (R^fin - R^0) by the per-basis identity.  The schedule game: every rule runs from the same basis, the head never increases under any rule (row 0
    is changed only by an insertion at position 0, which shortens it), the omniscient potential rule's first step is the largest one-step potential
    decrease of all rules, marks close at the last call with consistent fields, the GH-greedy never re-queries a fruitless position before the
    basis changes, and bad arguments are rejected.  (The LLL potential itself is NOT monotone under the physical bounded-LLL completion: only F_k
    is forced smaller and F_{k+n-1} is fixed, the intermediate flag members can grow -- observed at (20, 5).)"""
    from latticelab.profile_floor import block_sizes, dual_certificate
    from latticelab.residual import tour_bookkeeping
    from latticelab.schedule_game import RULES, compare, play

    d, beta, T = 30, 8, 6
    A = lll(qary(d, d // 2, 2 ** 12 + 3, seed=4))
    tb = tour_bookkeeping(A, beta, T, insert_tol=0.01, change_tol=0.005, ledger=True)
    L = tb["ledger"]
    assert L["max_ledger_gap"] < 1e-9 and len(L["mass_per_tour"]) == T + 1 and all(r >= -1e-9 for r in L["R_per_tour"])
    assert abs(L["mass_change"] - (L["cumulative_R"] + L["cumulative_sum_yD"] + L["cumulative_sum_yDvol"])) < 1e-9
    y = np.array([float(v) for v in dual_certificate(d, beta)[0]])
    ell0, ellT = np.array(tb["ell"][0]), np.array(tb["ell"][T])
    S = float(sum(ell0)) + float(gs_profile(A)[-1])  # the volume: d-1 recorded entries plus the last one
    P = lambda e, j: S if j >= d else float(e[:j].sum())
    tele = sum(y[k] / n * (P(ellT, k + n) - P(ell0, k + n)) for k, n in enumerate(block_sizes(d, beta)))
    assert abs(L["cumulative_sum_yDvol"] - tele) < 1e-9
    assert abs(L["mass_change"] - ((ell0[0] - ellT[0]) + (L["R_per_tour"][-1] - L["R_per_tour"][0]))) < 1e-9
    assert tb["ledger"] is not None and tour_bookkeeping(A, beta, 1)["ledger"] is None
    # the game on a small instance
    c = compare(20, 5, 2 ** 12 + 3, 1, 38)
    assert set(c["runs"]) == set(RULES)
    for r, run in c["runs"].items():
        assert len(run["heads"]) == run["calls"] <= 38 and run["marks"][-1]["calls"] == run["calls"] and np.all(np.diff(run["heads"]) <= 1e-9)
        last = run["marks"][-1]
        assert last["head_minus_floor"] == run["final_head_minus_floor"] == run["heads"][-1] and math.isfinite(last["mass_signed"]) and last["residual"] >= -1e-9
        assert 0 <= last["frac_tight"] <= 1 and last["mass_positive"] >= max(0.0, last["mass_signed"]) - 1e-12
        assert run["changed"] <= run["calls"] and all(0 <= k <= 18 for k in run["positions"])
    g = c["runs"]["gh_greedy"]
    # between two changes of the basis no position is queried twice (the no-repeat-until-change rule); a run that ends early does so at a fixed
    # point, i.e. after every position has been queried fruitlessly since the last change
    since_change: set = set()
    for k, mv in zip(g["positions"], g["moved"]):
        assert k not in since_change
        since_change = set() if mv else since_change | {k}
    assert g["ended_early_at"] is None or (g["ended_early_at"] == g["calls"] and len(since_change) == 19)
    assert sum(g["moved"]) == g["changed"]
    # the omniscient potential rule applies the largest one-step decrease, so after the first call its potential is the least of all rules'
    om = c["runs"]["omniscient_phi"]
    assert all(om["phis"][0] <= c["runs"][r]["phis"][0] + 1e-9 for r in RULES)
    with pytest.raises(ValueError):
        play(A, beta, 5, "nonsense")
    with pytest.raises(ValueError):
        play(A, beta, 5, "tours", mass_every=0)
    with pytest.raises(ValueError):
        play(A, beta, 0, "tours")
