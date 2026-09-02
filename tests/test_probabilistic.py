import math

import pytest

from factorlab import algorithms  # noqa: F401
from factorlab.registry import get_algorithm
from factorlab.gen import make_semiprime
from factorlab.numth import mpz
from factorlab.result import Work
from factorlab.algorithms.ecm import suyama_curve, ladder, xdbl, xadd, stage1_exponents
from factorlab.experiments.smooth_profiles import (dickman_rho, semismooth_G, brute_force_G, rho_length,
                                                   semismooth_profile, collision_profile, stage1_exponent,
                                                   multiplicative_order, lucas_order, exact_success, _factor)
from factorlab.experiments.lehman_hits import (continued_fraction, convergents, lehman_hits, classify_hits,
                                               hits_experiment)


def _proj_equal(P, Q, N):
    return (P[0] * Q[1] - P[1] * Q[0]) % N == 0


def test_ladder_consistency():
    N = mpz(1000003) * mpz(1000033)
    (a24, X, Z), g = suyama_curve(11, N)
    assert g is None
    w = Work()
    P2 = ladder(2, X, Z, a24, N, w)
    P3 = ladder(3, X, Z, a24, N, w)
    P5 = ladder(5, X, Z, a24, N, w)
    P6 = ladder(6, X, Z, a24, N, w)
    assert _proj_equal(P6, xdbl(P3[0], P3[1], a24, N, w), N)
    assert _proj_equal(P5, xadd(P3[0], P3[1], P2[0], P2[1], X, Z, N, w), N)
    assert _proj_equal(P2, xdbl(X, Z, a24, N, w), N)
    P7 = ladder(7, X, Z, a24, N, w)
    P13 = ladder(13, X, Z, a24, N, w)
    assert _proj_equal(P13, xadd(P7[0], P7[1], P6[0], P6[1], X, Z, N, w), N)


def test_stage1_exponents():
    assert stage1_exponents(10) == [8, 9, 5, 7]


@pytest.mark.parametrize("i", range(3))
def test_ecm_factors_balanced(i):
    inst = make_semiprime(40, "balanced", 21, i)
    res = get_algorithm("ecm")(inst.N, B1=2000, curves=200, seed=i)
    assert res.found, res
    assert {int(res.p), int(res.q)} == {int(inst.p), int(inst.q)}
    assert res.work["mulmod"] > 0


def test_dickman_values():
    assert dickman_rho(0.5) == 1.0
    assert abs(dickman_rho(2.0) - (1 - math.log(2))) < 1e-12
    assert abs(dickman_rho(2.5) - 0.130320) < 2e-5
    assert abs(dickman_rho(3.0) - 0.0486084) < 2e-5
    assert abs(dickman_rho(4.0) - 0.0049109) < 2e-5
    assert abs(dickman_rho(5.0) - 3.5472e-4) < 2e-5


def test_semismooth_special_cases():
    assert abs(semismooth_G(0.5, 1.0) - 1.0) < 1e-6
    assert abs(semismooth_G(1/3, 1/3) - dickman_rho(3.0)) < 1e-12
    G = semismooth_G(1/3, 2/3)
    assert 0.40 < G < 0.50, G  # ~0.45
    assert 0.70 < semismooth_G(0.4, 0.8) < 0.76


def test_semismooth_brute_force():
    # finite-size deviations at X = 2e5 are a few percent; the asymptotic G must be close
    X = 2 * 10 ** 5
    for a, b in ((1/3, 2/3), (0.4, 0.8), (0.5, 1.0)):
        bf = brute_force_G(X, a, b)
        assert abs(bf - semismooth_G(a, b)) < 0.08, (a, b, bf, semismooth_G(a, b))


def test_rho_length_small_prime():
    p = 101
    for c in (1, 2, 3):
        mu, lam = rho_length(p, c, 2)
        assert lam >= 1 and mu >= 0 and mu + lam <= p
        # check the cycle: f^{mu+lam}(x0) == f^{mu}(x0)
        x = 2
        seq = [x]
        for _ in range(mu + lam):
            x = (x * x + c) % p
            seq.append(x)
        assert seq[mu] == seq[mu + lam]
        assert len(set(seq[:mu + lam])) == mu + lam


def test_profiles_run_small():
    res = semismooth_profile(32, 40, exponents=(1/6, 1/4))
    assert res["rows"][1]["pred_G"] == 1.0
    for row in res["rows"]:
        for k in ("minus_pooled", "plus_pooled", "ctrl_even", "any_of_four", "exact_any_of_four"):
            assert 0.0 <= row[k] <= 1.0
    # with B2 = p the BSGS stage 2 always succeeds: exact predicate is 1 at c = 1/4
    assert res["rows"][1]["exact_minus_pooled_base2"] == 1.0
    res = collision_profile(32, 20, exponents=(1/4,))
    assert res["rho_length_over_sqrt_p_mean"] > 0


def test_exact_predicates_against_algorithms():
    # p = 30029: p+1 = 2*3*5*7*11*13 is 13-smooth, p-1 = 4 * 7507
    p, q = 30029, 1000003
    fm, fp = _factor(p - 1), _factor(p + 1)
    assert stage1_exponent(10) == 2 ** 3 * 3 ** 2 * 5 * 7
    o1 = int(multiplicative_order(2, p, fm))
    assert (p - 1) % o1 == 0 and pow(2, o1, p) == 1
    assert all(pow(2, o1 // l, p) != 1 for l in _factor(o1))  # minimality
    from factorlab.numth import jacobi
    P0 = next(P for P in range(3, 200) if jacobi(P * P - 4, p) == -1)
    o, s = lucas_order(P0, p, fp, fm)
    assert s == -1 and (p + 1) % int(o) == 0
    L = stage1_exponent(50)
    assert exact_success(o, L, 1)  # p+1 is 13-powersmooth: stage 1 alone suffices
    # the harness's williams_pp1 agrees
    res = get_algorithm("williams_pp1")(p * q, B1=50, P0=P0)
    assert res.found and int(res.p) == p
    # p-1 = 4 * 7507 with base 2: stage 1 at B1 = 50 fails, stage 2 to B2 >= 7507 succeeds
    o2 = multiplicative_order(2, p, fm)
    assert not exact_success(o2, L, 1000) and exact_success(o2, L, 7507)
    res = get_algorithm("pollard_pm1")(p * q, B1=50, base=2)
    assert not res.found


def test_continued_fraction_and_convergents():
    cf = continued_fraction(415, 93)
    assert cf == [4, 2, 6, 7]
    conv = convergents(cf)
    assert conv[-1] == (415, 93)
    assert conv == [(4, 1), (9, 2), (58, 13), (415, 93)]


@pytest.mark.parametrize("i", range(3))
def test_lehman_hits_contain_dirichlet_convergent(i):
    inst = make_semiprime(32, "rsa", 23, i)
    r = int(round(float(inst.N) ** (1 / 3)))
    hits = lehman_hits(inst.N, inst.p, inst.q, r)
    assert hits, "Lehman's lemma guarantees a hit"
    for h in hits:  # every hit satisfies Harvey's condition exactly (recheck in high precision)
        import gmpy2
        gmpy2.get_context().precision = 200
        a, b = h["a"], h["b"]
        U = gmpy2.mpfr(a * int(inst.q) + b * int(inst.p))
        x = 2 * gmpy2.sqrt(gmpy2.mpfr(a * b * int(inst.N)))
        delta = gmpy2.sqrt(gmpy2.mpfr(int(inst.N))) / (4 * r * gmpy2.sqrt(gmpy2.mpfr(a * b)))
        assert 0 <= U - x < delta * (1 + 1e-9)
    cl = classify_hits(int(inst.p), int(inst.q), r, hits)
    assert cl["star_is_hit"], cl


def test_hits_experiment_runs():
    res = hits_experiment(28, 5)
    assert res["mean_hits"] >= 1.0
