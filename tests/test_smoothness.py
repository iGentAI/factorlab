import math

import numpy as np
import pytest

from factorlab.experiments.local_joint import (joint_valuation_model, theoretical_corr, pairing_type,
                                               local_joint_test, pairing_score)
from factorlab.experiments.ecm_hitting import (primes_in_range, vec_suyama, vec_ladder, stage1_success,
                                               vec_inv)
from factorlab.algorithms.ecm import suyama_curve, ladder, stage1_exponents
from factorlab.result import Work
from factorlab.numth import mpz, is_prime


def test_joint_model_is_distribution_and_pairing():
    for l, j in ((5, 1), (7, 1), (3, 2), (2, 3)):
        m = l ** j
        for n in range(1, m):
            if math.gcd(n, l) != 1:
                continue
            model = joint_valuation_model(l, j, n)
            assert abs(sum(model.values()) - 1.0) < 1e-12
            # marginal probability that l^j | p-1 is 1/phi(l^j), independent of n
            phi = (l - 1) * l ** (j - 1)
            assert abs(sum(v for t, v in model.items() if t[0] >= j) - 1 / phi) < 1e-12
            both = sum(v for t, v in model.items() if t[0] >= j and t[1] >= j)
            if pairing_type(n, m) == "same":
                assert abs(both - 1 / phi) < 1e-12
            else:
                assert both == 0.0


def test_theoretical_corr_matches_model():
    l, j = 7, 1
    for n in range(1, 7):
        model = joint_valuation_model(l, j, n)
        a = np.array([t[0] >= 1 for t in model]).astype(float)
        b = np.array([t[1] >= 1 for t in model]).astype(float)
        w = np.array(list(model.values()))
        ma, mb = (a * w).sum(), (b * w).sum()
        cov = ((a - ma) * (b - mb) * w).sum()
        corr = cov / math.sqrt(((a - ma) ** 2 * w).sum() * ((b - mb) ** 2 * w).sum())
        assert abs(corr - theoretical_corr(l, j, n)) < 1e-9


def test_local_joint_small_run():
    res = local_joint_test(32, 300, moduli=((3, 1), (5, 1)))
    for row in res["per_modulus"].values():
        assert row["outside_support"] == 0
    assert pairing_score(1 + 8 * 3 * 5 * 7) > pairing_score(2 * 3 * 5 * 7 + 3)


def test_primes_in_range():
    ps = primes_in_range(100, 200)
    assert ps.tolist() == [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]


def test_vectorised_ladder_matches_scalar():
    ps = np.array([1000003, 1000033, 999983, 2147483587], dtype=np.int64)
    a24v, Xv, Zv = vec_suyama(11, ps)
    inv_check = (vec_inv(np.array([3, 5, 7, 11], dtype=np.int64), ps) * np.array([3, 5, 7, 11])) % ps
    assert (inv_check == 1).all()
    for i, p in enumerate(ps):
        (a24, X, Z), g = suyama_curve(11, mpz(int(p)))
        assert g is None
        assert int(a24) == int(a24v[i]) and int(X) == int(Xv[i]) and int(Z) == int(Zv[i])
        w = Work()
        for k in (2, 3, 97, 1000):
            Xs, Zs = ladder(k, X, Z, a24, mpz(int(p)), w)
            Xk, Zk = vec_ladder(k, Xv, Zv, a24v, ps)
            assert (int(Xs) * int(Zk[i]) - int(Zs) * int(Xk[i])) % int(p) == 0


def test_stage1_success_consistent_with_order():
    # For p = 1000003 find a sigma whose stage-1 succeeds at B1 = 2000 and verify
    # with the scalar ladder that Z = 0 mod p at the end.
    ps = primes_in_range(1000000, 1000100)
    B1 = 2000
    S = np.array([stage1_success(s, ps, B1) for s in range(6, 26)])
    assert S.shape == (20, len(ps))
    assert S.any()
    s_idx, p_idx = np.argwhere(S)[0]
    p = int(ps[p_idx])
    (a24, X, Z), _ = suyama_curve(6 + int(s_idx), mpz(p))
    w = Work()
    for pe in stage1_exponents(B1):
        X, Z = ladder(pe, X, Z, a24, mpz(p), w)
    assert int(Z) % p == 0


def test_e20_pm1_williams_match_exact_orders():
    from sympy import factorint
    from factorlab.experiments.hitting_sets import pm1_success, williams_success
    from factorlab.experiments.smooth_profiles import stage1_exponent, multiplicative_order, lucas_order
    ps = primes_in_range(1000, 1200)
    B1, B2 = 20, 200
    L = int(stage1_exponent(B1))
    s1, s2 = pm1_success(ps, B1, B2, base=2)
    for i, pv in enumerate(ps):
        p = int(pv)
        facm = {int(l): int(e) for l, e in factorint(p - 1).items()}
        order = int(multiplicative_order(2, p, facm))
        residual = order // math.gcd(order, L)
        assert bool(s1[i]) == (residual == 1)
        assert bool(s2[i]) == (residual > B1 and residual <= B2 and bool(is_prime(residual)))
    for P0 in (3, 5, 11):
        s1, s2 = williams_success(ps, B1, B2, P0=P0)
        for i, pv in enumerate(ps):
            p = int(pv)
            facm = {int(l): int(e) for l, e in factorint(p - 1).items()}
            facp = {int(l): int(e) for l, e in factorint(p + 1).items()}
            order, symbol = lucas_order(P0, p, facp, facm)
            assert symbol != 0
            residual = int(order) // math.gcd(int(order), L)
            assert bool(s1[i]) == (residual == 1)
            assert bool(s2[i]) == (residual > B1 and residual <= B2 and bool(is_prime(residual)))


def test_e20_ecm_matches_scalar_stage2_and_mask():
    from factorlab.experiments.hitting_sets import ecm_success, stage2_primes
    ps = np.array([31, 101, 103, 107, 109, 127], dtype=np.int64)
    B1, B2 = 10, 80
    for sigma in (6, 11):
        s1, s2 = ecm_success(sigma, ps, B1, B2)
        for i, pv in enumerate(ps):
            p = int(pv)
            curve, g = suyama_curve(sigma, mpz(p))
            if curve is None:
                assert int(g) == p and bool(s1[i]) and not bool(s2[i])
                continue
            a24, X, Z = curve
            w = Work()
            for pe in stage1_exponents(B1):
                X, Z = ladder(pe, X, Z, a24, mpz(p), w)
            expected1 = int(Z) % p == 0
            assert bool(s1[i]) == expected1
            expected2 = False
            if not expected1:
                for l in stage2_primes(B1, B2):
                    _, Zl = ladder(l, X, Z, a24, mpz(p), w)
                    expected2 |= int(Zl) % p == 0
            assert bool(s2[i]) == expected2
        mask = np.arange(ps.size) % 2 == 0
        m1, m2 = ecm_success(sigma, ps, B1, B2, mask=mask)
        assert np.array_equal(m1, s1)
        assert not m2[~mask].any() and np.array_equal(m2[mask], s2[mask])


def test_e20_separation_metrics_and_schema(capsys):
    from factorlab.experiments.hitting_sets import cover, hitting_scaling_experiment, _refine_classes, theta_sample
    from factorlab.experiments.run_hitting import _print
    classes = np.zeros(4, dtype=np.int64)
    classes, sizes = _refine_classes(classes, np.array([0, 0, 1, 1], dtype=bool))
    assert sizes.tolist() == [2, 2]
    classes, sizes = _refine_classes(classes, np.array([0, 1, 0, 1], dtype=bool))
    assert sizes.tolist() == [1, 1, 1, 1]
    ps = np.array([101, 103, 107], dtype=np.int64)
    empty = cover(ps, 5, None, False, 0)
    assert empty["K_separate"] is None and empty["K_separate_stage1"] is None
    assert empty["unresolved_prime_pairs_at_end"] == 3
    assert empty["stage1_unresolved_prime_pairs_at_end"] == 3
    assert empty["signature_information_lower_bound"] == 2
    assert all(hit is None for _, hit in empty["hardest_primes"])
    out = cover(ps, 10, None, True, 8)
    pairs = [h["stage1_unresolved_prime_pairs"] for h in out["history"]]
    classes_n = [h["stage1_signature_classes"] for h in out["history"]]
    assert pairs == sorted(pairs, reverse=True)
    assert classes_n == sorted(classes_n)
    if out["K_separate_stage1"] is not None:
        assert out["history"][out["K_separate_stage1"] - 1]["stage1_signature_classes"] == len(ps)
    z = theta_sample(np.array([], dtype=np.int64), 5, 10)
    assert z["predicted_K_separate_independent_jeffreys"] == 0
    assert z["predicted_K_separate_stage1_jeffreys"] == 0
    assert theta_sample(np.array([101], dtype=np.int64), 5, 10, n_curves=3)["predicted_K_separate_stage1_jeffreys"] == 0
    with pytest.raises(ValueError):
        theta_sample(ps, 5, 10, sample=1)
    smoke = hitting_scaling_experiment(log2_xs=(8,), u=3.0, max_curves=6, sample=20,
                                       stage2_through_bits=None)
    _print(smoke)
    assert "Kseparate" in capsys.readouterr().out


def test_e20_cover_lengths_match_independent_signatures():
    """cover() on curves only, reproduced from scalar-ladder signatures and set logic."""
    from factorlab.experiments.hitting_sets import cover, stage2_primes
    ps = primes_in_range(1000, 1300)
    B1, B2, max_curves = 10, 60, 40
    sigs = {int(p): [] for p in ps}
    for sigma in range(6, 6 + max_curves):
        for pv in ps:
            p = int(pv)
            curve, g = suyama_curve(sigma, mpz(p))
            if curve is None:
                sigs[p].append(1)
                continue
            a24, X, Z = curve
            w = Work()
            for pe in stage1_exponents(B1):
                X, Z = ladder(pe, X, Z, a24, mpz(p), w)
            bit = int(Z) % p == 0
            if not bit:
                for l in stage2_primes(B1, B2):
                    _, Zl = ladder(l, X, Z, a24, mpz(p), w)
                    if int(Zl) % p == 0:
                        bit = True
                        break
            sigs[p].append(int(bit))
    K_cov = next((t for t in range(1, max_curves + 1) if all(any(s[:t]) for s in sigs.values())), None)
    K_sep = next((t for t in range(1, max_curves + 1)
                  if len({tuple(s[:t]) for s in sigs.values()}) == len(sigs)), None)
    out = cover(ps, B1, B2, False, max_curves)
    assert out["K_star"] == K_cov and out["K_separate"] == K_sep
    assert K_sep is not None and out["K_separate"] >= math.ceil(math.log2(len(ps)))
    if K_sep is not None:
        sizes = {}
        for s in sigs.values():
            key = tuple(s[:K_sep - 1])
            sizes[key] = sizes.get(key, 0) + 1
        pairs = sum(c * (c - 1) // 2 for c in sizes.values())
        assert out["history"][K_sep - 2]["unresolved_prime_pairs"] == pairs


def test_e20_plain_family_matches_scalar_and_ordering():
    from factorlab.experiments.hitting_sets import ecm_plain_success, stage2_primes, family
    from factorlab.numth import invert
    ps = np.array([1009, 1013, 1019, 1021, 1031, 1033], dtype=np.int64)
    B1, B2 = 10, 80
    for A in (7, 11):
        s1, s2 = ecm_plain_success(A, ps, B1, B2)
        for i, pv in enumerate(ps):
            p = int(pv)
            a24 = (A + 2) * int(invert(4, p)) % p
            X, Z = mpz(3), mpz(1)
            w = Work()
            for pe in stage1_exponents(B1):
                X, Z = ladder(pe, X, Z, mpz(a24), mpz(p), w)
            expected1 = int(Z) % p == 0
            assert bool(s1[i]) == expected1
            expected2 = False
            if not expected1:
                for l in stage2_primes(B1, B2):
                    _, Zl = ladder(l, X, Z, mpz(a24), mpz(p), w)
                    expected2 |= int(Zl) % p == 0
            assert bool(s2[i]) == expected2
    names = [n for n, _ in family(ps, B1, B2, False, 6, curves="mixed")]
    assert names == ["ecm_6", "plain_7", "ecm_7", "plain_8", "ecm_8", "plain_9"]
    assert [n for n, _ in family(ps, B1, B2, False, 3, curves="plain")] == ["plain_7", "plain_8", "plain_9"]
    with pytest.raises(ValueError):
        family(ps, B1, B2, False, 2, curves="edwards")


def test_e20b_labels_match_scalar_orders_and_algorithm():
    """Exposure labels against scalar orders, and Proposition V' end to end: fixed_list_ecm on
    N = pq run for one curve succeeds iff the labels of p and q differ."""
    from factorlab.experiments.hitting_sets import residual_order_labels
    from factorlab.registry import get_algorithm
    from factorlab.algorithms.ecm import xadd
    ps = primes_in_range(1000, 1400)
    B1, B2 = 10, 120
    algo = get_algorithm("fixed_list_ecm")
    checked_pairs = 0
    for sigma in (6, 7, 11, 13):
        lab = residual_order_labels(sigma, ps, B1, B2)
        for i, pv in enumerate(ps):
            p = int(pv)
            curve, g = suyama_curve(sigma, mpz(p))
            if curve is None:
                assert lab[i] == -1
                continue
            a24, X, Z = curve
            if int(a24) % p == 0:
                assert lab[i] == -2  # singular, A = -2
                continue
            if int(a24) % p == 1:
                assert lab[i] == -3  # singular, A = 2
                continue
            w = Work()
            for pe in stage1_exponents(B1):
                X, Z = ladder(pe, X, Z, a24, mpz(p), w)
            if int(Z) % p == 0:
                assert lab[i] == 1
                continue
            order = 0
            prev, cur = (X, Z), None
            for j in range(2, B2 + 1):
                if j == 2:
                    from factorlab.algorithms.ecm import xdbl
                    cur = xdbl(X, Z, a24, mpz(p), w)
                else:
                    cur, prev = xadd(cur[0], cur[1], X, Z, prev[0], prev[1], mpz(p), w), cur
                if int(cur[1]) % p == 0:
                    order = j
                    break
            assert int(lab[i]) == order, (sigma, p, int(lab[i]), order)
        # Proposition V': one curve factors pq iff labels differ
        for a in range(0, min(12, ps.size - 1)):
            p, q = int(ps[a]), int(ps[a + 1])
            res = algo(p * q, B1=B1, B2=B2, max_curves=1, sigma0=sigma)
            assert res.found == (lab[a] != lab[a + 1]), (sigma, p, q, int(lab[a]), int(lab[a + 1]), res.meta)
            if res.found:
                assert {int(res.p), int(res.q)} == {p, q}
            checked_pairs += 1
    assert checked_pairs >= 40


def test_e20b_cover_residual_cumulative_accounting():
    from factorlab.experiments.hitting_sets import cover_residual, residual_order_labels
    ps = primes_in_range(1000, 1300)
    B1, B2, curves = 10, 80, 8
    out = cover_residual(ps, B1, B2, curves)
    classes = np.zeros(ps.size, dtype=np.int64)
    classes_bin = np.zeros(ps.size, dtype=np.int64)
    ever = np.zeros(ps.size, dtype=bool)
    for i, h in enumerate(out["history"]):
        lab = residual_order_labels(6 + i, ps, B1, B2)
        ever |= lab != 0
        key = classes * (B2 + 4) + (lab + 3)
        _, classes = np.unique(key, return_inverse=True)
        sizes = np.bincount(classes)
        _, classes_bin = np.unique(classes_bin * 2 + (lab != 0).astype(np.int64), return_inverse=True)
        sizes_bin = np.bincount(classes_bin)
        assert h["classes_binary"] == sizes_bin.size
        assert h["unresolved_pairs_binary"] == int(np.sum(sizes_bin * (sizes_bin - 1) // 2))
        assert h["classes"] == sizes.size
        unex = eq = 0
        for c, size in enumerate(sizes):
            members = np.nonzero(classes == c)[0]
            assert np.all(ever[members] == ever[members[0]])  # exposure history is constant in a class
            pairs = int(size * (size - 1) // 2)
            if ever[members[0]]:
                eq += pairs
            else:
                unex += pairs
        assert unex == h["unresolved_pairs_unexposed"]
        assert eq == h["unresolved_pairs_equal_exposure"]
        assert unex + eq == h["unresolved_pairs"]


def test_e22_greedy_label_schedule_small():
    from factorlab.experiments.hitting_sets import greedy_label_schedule
    row = greedy_label_schedule(8, us=(3.0, 4.0), curves_per_u=5)
    assert row["separated"] and row["termination"] == "separated"
    assert row["n_steps"] <= 10 and row["total_cost_proxy"] > 0
    pairs = [z["unresolved_pairs"] for z in row["steps"]]
    assert pairs == sorted(pairs, reverse=True) and pairs[-1] == 0
    assert all(z["gain"] > 0 and z["gain_per_B1"] > 0 for z in row["steps"])
    assert row["best_fixed_u"] is not None and row["greedy_over_best_fixed_cost"] is not None
    # Two copies of the same single coordinate: the second has zero gain, so
    # stop rather than charging it; one coordinate cannot separate this population.
    stalled = greedy_label_schedule(8, us=(3.0, 3.0), curves_per_u=1)
    assert not stalled["separated"] and stalled["termination"] == "no_positive_gain"
    assert stalled["n_steps"] == 1 and stalled["unresolved_pairs"] > 0
    assert all(z["gain"] > 0 for z in stalled["steps"])


def test_classgroup_basics():
    from factorlab.experiments.classgroup import (discriminant, class_number, omega, cohen_lenstra_div_prob,
                                                  cohen_lenstra_expected_valuation, squarefree_multipliers)
    N = 1009 * 1013
    D = discriminant(1, N)
    h, method = class_number(D)
    assert method == "qfbclassno" and h > 0
    # genus theory: 2^{omega(D)-1} divides h
    assert h % (2 ** (omega(D) - 1)) == 0
    # cross-check qfbclassno against quadclassunit, also on a large discriminant
    from factorlab.experiments.classgroup import pari
    assert int(pari().quadclassunit(D)[0]) == h
    D2 = discriminant(3, 1000003 * 1000033)
    assert int(pari().quadclassunit(D2)[0]) % (2 ** (omega(D2) - 1)) == 0
    assert abs(cohen_lenstra_div_prob(3) - 0.43987) < 1e-4
    assert 0.6 < cohen_lenstra_expected_valuation(3) < 0.75
    assert squarefree_multipliers(N, 6) == [1, 2, 3, 5, 6, 7]


@pytest.mark.parametrize("nbits,i", [(44, 0), (52, 1), (60, 2)])
def test_quadratic_sieve_factors(nbits, i):
    from factorlab import algorithms  # noqa: F401
    from factorlab.registry import get_algorithm
    from factorlab.gen import make_semiprime
    inst = make_semiprime(nbits, "rsa", 77, i)
    res = get_algorithm("qs")(inst.N)
    assert res.found, res.meta
    assert {int(res.p), int(res.q)} == {int(inst.p), int(inst.q)}
    assert res.work["sieve"] > 0 and res.meta["full"] > 0


def test_qs_helpers():
    from factorlab.algorithms.qs import factor_base, expected_valuation_shift, _gf2_dependencies, _trial_divide
    N = mpz(1000003) * 1000033
    primes, roots = factor_base(N, 100)
    for l, r in zip(primes, roots):
        assert (r * r - N) % l == 0
    s = expected_valuation_shift(N, 100)
    assert isinstance(s, float)
    deps = _gf2_dependencies([0b101, 0b011, 0b110])  # row0 ^ row1 ^ row2 = 0
    assert deps and deps[0] == 0b111
    ex, cof = _trial_divide(2 ** 3 * 7 * 11 * 101, [3, 5, 7, 11])
    assert ex == {2: 3, 7: 1, 11: 1} and cof == 101


def test_qs_small_odd_factor():
    from factorlab import algorithms  # noqa: F401
    from factorlab.registry import get_algorithm
    q = 1000003
    for small in (3, 5, 97):
        res = get_algorithm("qs")(small * q, B=200, M=2000)
        assert res.found and int(res.p) == small


def test_schnorr_lenstra_factors_and_classifies():
    from factorlab import algorithms  # noqa: F401
    from factorlab.registry import get_algorithm
    from factorlab.gen import make_semiprime
    from factorlab.algorithms.classgroup_factor import (discriminant, identity_form, odd_stage1_exponent,
                                                      ambiguous_divisor, pari)
    assert odd_stage1_exponent(10) == 9 * 5 * 7
    sl = get_algorithm("schnorr_lenstra")
    found = 0
    for i in range(12):
        inst = make_semiprime(32, "rsa", 91, i)
        res = sl(inst.N, B1=2000, B2=4 * 10 ** 6, k=1, seed=i, forms=3)
        if res.found:
            found += 1
            assert {int(res.p), int(res.q)} == {int(inst.p), int(inst.q)}
        else:
            assert set(res.meta["reasons"]) <= {"trivial_2_part", "odd_part_not_semismooth", "useless_ambiguous_form", "order_not_2_power"}
    assert found >= 6, found
    # the 2-part stripping exponent exceeds log2 h(D) on actual class numbers
    from factorlab.algorithms.classgroup_factor import two_part_bound
    P = pari()
    for i in range(20):
        inst = make_semiprime(32, "rsa", 93, i)
        for k in (1, 2, 3, 5, 6, 7, 30, 210):
            D = discriminant(k, int(inst.N))
            assert two_part_bound(D) > int(P.qfbclassno(D)).bit_length()
    # stage-2 regression: a modulus whose h_odd has exactly one prime factor r in
    # (B1, B2] and all others <= B1; stage 1 alone must fail with
    # order_not_2_power (for a form whose order involves r) and stage 2 must factor it
    from sympy import factorint
    B1 = 50
    L_odd = odd_stage1_exponent(B1)
    target = None
    for i in range(300):
        inst = make_semiprime(32, "rsa", 93, i)
        D = discriminant(1, int(inst.N))
        h = int(P.qfbclassno(D))
        fac = {int(a): int(b) for a, b in factorint(h).items()}
        odd = sorted((a, b) for a, b in fac.items() if a != 2)
        if not odd:
            continue
        r, e = odd[-1]
        h_odd = h >> fac.get(2, 0)
        # exactly one residual prime r > B1 (to the first power) and the rest divides L_odd exactly
        if e == 1 and r > 300 and L_odd % (h_odd // r) == 0:
            target = (inst, r)
            break
    assert target is not None
    inst, r = target
    stage1_only = [sl(inst.N, B1=B1, B2=None, k=1, seed=s, forms=1) for s in range(4)]
    assert not any(res.found for res in stage1_only)
    assert any("order_not_2_power" in res.meta["reasons"] for res in stage1_only)
    res = sl(inst.N, B1=B1, B2=r + 10, k=1, seed=0, forms=6)
    assert res.found and {int(res.p), int(res.q)} == {int(inst.p), int(inst.q)}, res.meta
    # identity and ambiguous-form divisor sanity
    D = discriminant(1, 1009 * 1013)
    e = identity_form(D)
    assert int(e[0]) == 1
    amb = pari().Qfb(1009, 0, 1013 * 4)
    assert 1009 in ambiguous_divisor(amb)


def test_adaptive_selection_shapes():
    from factorlab.experiments.ecm_hitting import adaptive_selection
    rng = np.random.default_rng(0)
    Sp = rng.random((12, 400)) < 0.3
    Sq = rng.random((12, 400)) < 0.3
    Nres = rng.integers(0, 3, 400)
    out = adaptive_selection(Sp, Sq, Nres, k_select=5)
    assert 0 <= out["first_k_curves"] <= 1 and 0 <= out["adaptive_per_class"] <= 1
    assert out["n_test"] == 200


def test_root_lattice_floor_and_snfs():
    from factorlab.experiments.root_lattice import root_lattice_columns, lll_shortest, pari
    from factorlab.gen import make_semiprime
    inst = make_semiprime(64, "rsa", 5, 0)
    N = int(inst.N)
    d = 4
    cols = root_lattice_columns(N, 12345, d)
    P = pari()
    M = P.matrix(d + 1, d + 1, [cols[j][i] for i in range(d + 1) for j in range(d + 1)])
    assert int(P.matdet(M)) == N
    nv, v = lll_shortest(N, 12345, d)
    assert sum(v[i] * pow(12345, i, N) for i in range(d + 1)) % N == 0  # a root-lattice vector
    ratio = math.sqrt(nv) / N ** (1 / (d + 1))
    assert 0.2 < ratio < 3.0, ratio
    # SNFS form: x^4 - c has root 2^16 modulo 2^64 - c
    c = 17
    Ns = (1 << 64) - c
    nv, v = lll_shortest(Ns, 1 << 16, 4)
    assert math.sqrt(nv) <= math.sqrt(1 + c * c) + 1e-9


def test_paradigm_calculator_exponents():
    from factorlab.experiments.paradigm_calculator import qs_cost, nfs_cost, paradigm_table, pure_power_sizes
    q256, q512 = qs_cost(256), qs_cost(512)
    assert q512["log2_cost"] > q256["log2_cost"] > 0
    n = nfs_cost(1024, 5)
    assert n["log2_cost"] > 0 and n["u_algebraic"] > 1
    pt = paradigm_table(bits_list=(128, 256, 512, 1024, 2048), ds=(3, 4, 5, 6))
    assert 0.8 < pt["fits"]["qs_vs_L_half_slope"] < 1.3
    assert 1.3 < pt["fits"]["nfs_vs_L_third_slope"] < 2.6
    assert pt["first_bits_where_nfs_cheaper"] is None or pt["first_bits_where_nfs_cheaper"] >= 128
    pp = pure_power_sizes(128)
    assert pp["2"] == pytest.approx(0.5) and pp["3"] > pp["2"]


def test_degeneracy_recoverability_field():
    from factorlab.experiments.smooth_profiles import semismooth_profile
    res = semismooth_profile(32, 60, exponents=(1 / 4,))
    row = res["rows"][0]
    assert row["degenerate_count"] >= 0
    if row["degenerate_recoverable_by_descent"] is not None:
        assert 0.0 <= row["degenerate_recoverable_by_descent"] <= 1.0


def test_residue_samplers():
    import random
    from factorlab.experiments.residue_ladder import sample_vallee, sample_qs, sample_dixon
    from factorlab.gen import make_semiprime
    inst = make_semiprime(40, "rsa", 3, 0)
    rng = random.Random(1)
    N = int(inst.N)
    for _ in range(50):
        x, r = sample_vallee(rng, N)
        assert 0 <= r < 4 * N ** (2 / 3) + 1 and (x * x - r) % N == 0
        x, r = sample_vallee(rng, N, kmax=int(N ** (1 / 3)))
        assert 0 <= r < 4 * N ** (2 / 3) + 1 and (x * x - r) % N == 0
        x, r = sample_qs(rng, N, 1000)
        assert (x * x - N) % N == 0 or (x * x + r) % N == 0 or (x * x - r) % N == 0
        x, r = sample_dixon(rng, N)
        assert r == (x * x) % N
