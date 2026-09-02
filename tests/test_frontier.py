import itertools
import math
import random

import numpy as np
import pytest

from factorlab.experiments.nonlinear_pairs import (
    gp_pair, exact_pair_floor, tiny_f_pairs, resultant_floor, poly_eval_mod, is_irreducible_q, sup,
    orthogonal_lattice, roots_mod_N,
)
from factorlab.experiments.selection_frontier import (
    frontier, theory_P, crossover_exponent, skewed_product, _candidate,
)
from factorlab.gen import make_semiprime


def brute_pair_floor(N: int, d: int, Hb: int):
    """Independent minimum of ||f|| ||g|| over coprime irreducible degree-d pairs (leading
    coefficients > 0, coefficients bounded by Hb) with a common root r in [1, N-1], found by
    evaluating both polynomials at every residue (no lattices, no factorisation of N)."""
    ms = np.arange(N, dtype=np.int64)

    def values(f):
        v = np.zeros(N, dtype=np.int64)
        for c in reversed(f):
            v = (v * ms + c) % N
        return v

    rng = range(-Hb, Hb + 1)
    polys = []
    for top in itertools.product(range(1, Hb + 1), *[rng] * (d - 1)):
        for f0 in rng:
            if f0 == 0:
                continue
            f = [f0] + list(reversed(top))
            if math.gcd(*[abs(c) for c in f]) == 1 and is_irreducible_q(f):
                polys.append((f, values(f)))
    best = (math.inf, None)
    for i in range(len(polys)):
        f, vf = polys[i]
        zf = vf == 0
        zf[0] = False
        if not zf.any():
            continue
        for j in range(i + 1, len(polys)):
            g, vg = polys[j]
            P = sup(f) * sup(g)
            if P >= best[0]:
                continue
            if not (zf & (vg == 0)).any():
                continue
            from flint import fmpz_poly
            if fmpz_poly(f).gcd(fmpz_poly(g)).degree() > 0:
                continue
            best = (P, (f, g))
    return best


@pytest.mark.parametrize("p,q", [(11, 13), (13, 17)])
def test_exact_pair_floor_matches_brute_force(p, q):
    N = p * q
    r = exact_pair_floor(N, p, q, 2, rng=random.Random(2))
    assert r["certified"] and is_irreducible_q(r["f"]) and is_irreducible_q(r["g"])
    common = [x for x in roots_mod_N(r["f"], p, q) if x != 0 and poly_eval_mod(r["g"], x, N) == 0]
    assert common
    # every pair with a smaller product has both sup-norms <= P2 - 1
    b = brute_pair_floor(N, 2, r["P2"] - 1)
    assert b[0] >= r["P2"], (b, r)
    # and the exact value is attained by some bounded pair
    b2 = brute_pair_floor(N, 2, max(r["norms"]))
    assert b2[0] == r["P2"], (b2, r)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_gp_pair_has_common_root_and_predicted_size(d):
    rng = random.Random(7)
    N = int(make_semiprime(56, "rsa", 9, 0).N)
    r = gp_pair(N, d, rng)
    assert r is not None
    for v in (r["f"], r["g"]):
        assert len(v) == d + 1 and v[-1] > 0 and is_irreducible_q(v)
        assert poly_eval_mod(v, r["theta"], N) == 0
    # GP of length d+1 with every entry of absolute value at most 2^d N^{1-1/d}
    assert r["gp_max_over_N_1_minus_1_over_d"] <= 2 ** d
    # the orthogonal lattice has rank d
    gp = [int(x) for x in r["gp"]]
    assert len(orthogonal_lattice(gp)) == d
    assert len(r["admissible_basis_norms"]) >= 2
    # product exponent near 2(d-1)/d^2 and never below the resultant floor
    assert r["P2"] >= resultant_floor(N, d)
    assert abs(r["log_P2_over_log_N"] - 2 * (d - 1) / d ** 2) < 0.08


def test_tiny_f_route_equals_the_exact_floor_for_tiny_minimisers():
    for idx in (0, 1):
        inst = make_semiprime(18, "rsa", 5, idx)
        N, p, q = int(inst.N), int(inst.p), int(inst.q)
        ex = exact_pair_floor(N, p, q, 3, rng=random.Random(1))
        t = tiny_f_pairs(N, p, q, 3, Hf=1)
        assert t["P2"] is not None and t["norms"][0] == 1 and len(t["g"]) == 4 and t["g"][-1] > 0
        assert poly_eval_mod(t["g"], t["root"], N) == 0 and poly_eval_mod(t["f"], t["root"], N) == 0
        assert is_irreducible_q(t["g"])
        # the exact (3,3) minimum is over all f, so it is at most the tiny route's value, and
        # when its minimiser has ||f|| = 1 the two certified routines must agree exactly
        assert t["complete"]
        assert ex["P2"] <= t["P2"]
        if ex["norms"][0] == 1:
            assert ex["P2"] == t["P2"], (ex, t)
        # fixed-f relation between the reduced (degree < 3) minimum lambda and the degree-3
        # minimum mu: every degree-3 partner reduces to a vector of sup-norm <= (1 + ||f||) mu
        assert t["P2_reduced"] <= 2 * t["P2"]
        # brute force: no nonzero g of degree < 3 below the reduced minimum vanishes at the root
        r = t["root"]
        s = t["P2_reduced"]
        for g0 in range(-s + 1, s):
            for g1 in range(-s + 1, s):
                for g2 in range(-s + 1, s):
                    if (g0, g1, g2) != (0, 0, 0):
                        assert (g0 + g1 * r + g2 * r * r) % N != 0


def test_hessian_covariant_does_not_vanish_at_a_known_root():
    from factorlab.experiments.covariants import syzygy_constants, covariants_at_root
    assert syzygy_constants() == (-1, -432)
    # a cubic with a root modulo N = 1009 * 1013: f = x^3 + 2x^2 - x + c0 with c0 chosen so that f(m) = 0 mod N
    N, m = 1009 * 1013, 12345
    c0 = (-(m ** 3 + 2 * m * m - m)) % N
    r = covariants_at_root([c0, -1, 2, 1], m, N)
    assert not r["H_vanishes"] and r["syzygy_congruence_mod_N2"]


def test_frontier_monotone_and_theory_shape():
    N = int(make_semiprime(48, "rsa", 23, 0).N)
    cps = [4, 16, 64, 256, 1024]
    fr = frontier(N, 3, 1, 1024, cps)
    P = fr["log2_P"]
    Q = fr["log2_Q"]
    assert all(P[i + 1] <= P[i] + 1e-12 for i in range(len(P) - 1))
    assert all(Q[i + 1] <= Q[i] + 1e-12 for i in range(len(Q) - 1))
    # Q <= 2P for the same pair: check on the recorded best-P candidate
    f = [int(c) for c in fr["best_P"]["f"]]
    m = int(fr["best_P"]["m"])
    Qv, s = skewed_product(f, m)
    assert Qv <= 2.0 * max(abs(c) for c in f) * m * (1 + 1e-9) and 1.0 <= s <= m
    # every candidate has a genuine root
    for ad in (1, 2, 5, 17):
        c = _candidate(N, 3, ad)
        if c is not None:
            Pv, fc, mc = c
            assert poly_eval_mod(fc, mc, N) == 0 and Pv == max(abs(x) for x in fc) * mc
    # the theoretical frontier decreases until T = N^{h*} and increases after
    h = 2 / 11
    Tstar = float(N) ** h
    assert theory_P(N, 3, Tstar / 4) > theory_P(N, 3, Tstar) < theory_P(N, 3, Tstar * 4)
    assert abs(crossover_exponent(3) - 1 / 7) < 1e-12 and abs(crossover_exponent(5) - 8 / 78) < 1e-12
