import math

import numpy as np
import pytest

from factorlab.experiments.sidon_scaling import (
    rho, shell, speeds_and_deltas, cluster_max, d_star, window_membership_exact, verify_chain, e1_chain_pairs,
    j_census, omega, chain_of_maximiser, near_square_multiple_mask, e1_chain_classes, chain_census_fast,
    g_link_cluster, symmetric_pair_cluster, symmetric_pair_census, link_density_model, squarefree_flags,
    symmetric_cluster_with_modulus, lemma_d_window, identify_window_as_class_pairs, resonance_partition,
    symmetric_density_model, KAPPA_0, symmetric_family_window_fit, theorem_q_prime_check, sieve_lower_bound,
    symmetric_pigeonhole_bound,
)
from factorlab.gen import make_semiprime


def test_cluster_max_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(40):
        pts = rng.uniform(0, 100, size=int(rng.integers(2, 40)))
        h = float(rng.uniform(0.2, 5))
        D, tau = cluster_max(pts, h)
        # brute force over all centres at midpoints of pairs and points
        cands = list(pts) + [0.5 * (a + b) for a in pts for b in pts]
        brute = max(int(np.sum(np.abs(pts - t) < h)) for t in cands)
        assert D == brute
        assert int(np.sum(np.abs(pts - tau) < h)) == D  # the returned centre realises the count


def test_membership_formula_and_small_d_star():
    r = 300
    ks = shell(r, True)
    rng = np.random.default_rng(4)
    for _ in range(500):
        k, kp = int(rng.choice(ks)), int(rng.choice(ks))
        if k == kp:
            continue
        tau = math.sqrt(k) - math.sqrt(kp) + rng.uniform(-3, 3) * rho(r)
        assert window_membership_exact(k, k - kp, tau, r) == (abs(math.sqrt(k) - math.sqrt(kp) - tau) < rho(r))
    row = d_star(r, True)
    L, _ = speeds_and_deltas(ks)
    assert int(np.sum(np.abs(L - row["tau"]) < rho(r))) == row["D_star"]


def test_e1_chain_integrality_and_speed_expansion():
    # j = 15: M = 731 is odd with 731^2 = 1 (mod 120); the chain (M^2 - 1)/120 with step 30
    v = verify_chain(8192, 120, 1, 30, 731)
    assert v["members"][:4] == [4453, 4826, 5214, 5617] and abs(v["tau0_sq"] - 7.5) < 1e-12
    for d, p in zip(v["deviation_over_rho"], v["predicted_over_rho"]):
        assert abs(d - p) < 0.02
    with pytest.raises(ValueError):
        verify_chain(8192, 120, 1, 0, 731)
    with pytest.raises(ValueError):
        verify_chain(8192, 120, 1, 31, 731)
    # all odd M with M^2 = 1 (mod 8j) give integral members; the classes modulo 2j number 2^omega(j)
    for j in (15, 21, 105):
        classes = {M % (2 * j) for M in range(1, 8 * j, 2) if (M * M - 1) % (8 * j) == 0}
        assert len(classes) == 2 ** omega(j)
        pairs = e1_chain_pairs(32768, j)
        for k, kp in pairs:
            assert (8 * j * k + 1) == math.isqrt(8 * j * k + 1) ** 2 and (8 * j * kp + 1) == math.isqrt(8 * j * kp + 1) ** 2
            assert 32768 // 2 < k < kp <= 32768


def test_census_is_consistent_with_the_global_maximum():
    r = 4096
    row = d_star(r, True)
    cen = j_census(r)
    best = max(cen, key=lambda z: z["cluster"])
    # the global maximum is attained by the best e = 1 chain at this r (j = 15, omega = 2)
    assert best["j"] == 15 and best["cluster"] == row["D_star"] == 9
    mc = chain_of_maximiser(r, abs(row["tau"]))
    assert mc is not None and mc["j"] == 15 and mc["omega"] == 2
    # the chain's links all sit within the window capacity: 4 classes x 5.66 links at the top of the shell
    assert best["links"] == 10 and best["cluster"] <= 4 * 5.66
    # near-square-multiple mask catches the chain members for c = 8j, e = 1
    ks = shell(r, True)
    mask = near_square_multiple_mask(ks, 120, 1)
    for k, kp in e1_chain_pairs(r, 15):
        assert mask[np.searchsorted(ks, k)] and mask[np.searchsorted(ks, kp)]


def test_class_pairs_admissibility_lemma():
    # N_g(j) = 2^{#{p | j : p | g}} if g = 0, +-2 (mod p) for every prime p | j, else 0
    sf = squarefree_flags(200)
    for j in range(1, 200, 2):
        if not sf[j]:
            continue
        cls = e1_chain_classes(j)
        assert len(cls) == 2 ** omega(j) and all(M % 2 == 1 and (M * M - 1) % (8 * j) == 0 for M in cls)
        cset = set(cls)
        primes = [p for p in range(3, j + 1) if j % p == 0 and all(p % q for q in range(2, int(p ** 0.5) + 1))]
        for g in range(2, 2 * j + 1, 2):
            N = sum(1 for M0 in cls if (M0 + g) % (2 * j) in cset)
            adm = all((g % p) in (0, 2, p - 2) for p in primes)
            assert N == (2 ** sum(1 for p in primes if g % p == 0) if adm else 0)
    with pytest.raises(ValueError):
        e1_chain_classes(10)
    # non-squarefree odd j: roots modulo p^a are +-1, so 2^omega(j) classes and the lemma with p^a in place of p
    for j in (9, 45, 75):
        cls = e1_chain_classes(j)
        assert len(cls) == 2 ** omega(j) and all(M % 2 == 1 and (M * M - 1) % (8 * j) == 0 for M in cls)
    with pytest.raises(ValueError):
        g_link_cluster(1000, 15, 0, sf)


def test_g_links_explain_the_off_resonance_clusters():
    # r = 2^14: the window at tau^2 = 5/6 holds 10 pairs, all (M, M + 10) members of the j = 15 chain
    r = 16384
    sf = squarefree_flags(r)
    g10 = g_link_cluster(r, 15, 10, sf)
    assert g10["class_pairs"] == 2 and abs(g10["tau_sq"] - 5 / 6) < 1e-12 and g10["cluster"] == 10
    ks = shell(r, True)
    L, _ = speeds_and_deltas(ks)
    near = L[np.abs(L * L - 5 / 6) < 1e-3]
    assert cluster_max(near, rho(r))[0] == 10
    # r = 2^12: tau^2 = 10/3 is the (j, g) = (15, 20) family
    g20 = g_link_cluster(4096, 15, 20, squarefree_flags(4096))
    assert abs(g20["tau_sq"] - 10 / 3) < 1e-12 and g20["cluster"] == 5


def test_symmetric_pairs_and_the_cube_root_law():
    # integrality and speed expansion of ((j t^2 - t)/2, (j t^2 + t)/2)
    for j in (3, 5, 15):
        for t in range(50, 60):
            km, kp = (j * t * t - t) // 2, (j * t * t + t) // 2
            assert (j * t * t - t) % 2 == 0 and 8 * j * km + 1 == (2 * j * t - 1) ** 2 and 8 * j * kp + 1 == (2 * j * t + 1) ** 2
            sp = math.sqrt(kp) - math.sqrt(km)
            tau0 = 1 / math.sqrt(2 * j)
            assert abs((sp - tau0) - tau0 / (8 * j * j * t * t)) < 3 * tau0 / (8 * j * j * t * t) ** 2 * 10
    # census values pinned at the corrected resolution; the law 0.3-0.4 r^{1/3}
    s14 = symmetric_pair_census(16384, sf=squarefree_flags(16384))
    assert s14["D_symmetric"] == 8 and s14["best"]["j"] == 9
    s18 = symmetric_pair_census(2 ** 18, sf=squarefree_flags(2 ** 18))
    assert s18["D_symmetric"] == 25 and s18["best"]["j"] == 15
    assert 0.25 < s18["D_symmetric"] / (2 ** 18) ** (1 / 3) < 0.45
    # empty census is well defined
    assert symmetric_pair_census(20, j_max=1, sf=squarefree_flags(20))["best"] is None


def test_fast_census_matches_sweep_census_and_density_model():
    r = 4096
    slow = {z["j"]: (z["links"], z["cluster"]) for z in j_census(r)}
    fast = {z["j"]: (z["links"], z["cluster"]) for z in chain_census_fast(r)["rows"]}
    assert slow == fast
    d1, d3, d15 = link_density_model(1), link_density_model(3), link_density_model(15)
    assert 0.28 < d1 < 0.34 and 0.24 < d3 < 0.29 and 0.33 < d15 < 0.40


def test_planar_cap_and_exact_transfer_with_a_modulus():
    N = int(make_semiprime(48, "rsa", 7, 0).N)
    # at r = N^{1/5} the offsets are negligible: the cluster with the modulus equals the modulus-free one
    r = int(round(N ** 0.2))
    sf = squarefree_flags(r)
    W = lemma_d_window(N, r)
    assert 16 * r * r * (r // 2 + 1) * W * W >= N > 16 * r * r * (r // 2 + 1) * (W - 1) ** 2
    for j in (3, 5, 7, 9, 15):
        z = symmetric_cluster_with_modulus(N, r, j, sf, W)
        free = symmetric_pair_cluster(r, j, sf)
        assert z["monotone"] and z["links"] == free["links"] and z["cluster"] == free["cluster"]
    # in the planar regime the offset t caps every family at 2W while the modulus-free value is far above
    r = int(round(N ** 0.32))
    sf = squarefree_flags(r)
    W = lemma_d_window(N, r)
    for j in (3, 5, 7, 9, 15, 21):
        z = symmetric_cluster_with_modulus(N, r, j, sf, W)
        assert z["monotone"] and z["cluster"] <= 2 * W
    best_free = symmetric_pair_census(r, sf=sf)["D_symmetric"]
    assert best_free > 2 * W
    # the off-resonance window identification is exact at the pair-set level (window centre from the partition:
    # the class-pair cluster sits a few rho above the limiting speed sqrt(10/3))
    p = resonance_partition(4096)
    res = identify_window_as_class_pairs(4096, math.sqrt(p["tau_sq_off"]), 15, 20)
    assert res["all_identified"] and res["pairs_in_window"] == 5 == p["D_off_resonance"]


def test_theorem_x_symmetric_pigeonhole_bound_at_the_crossing():
    # the exact symmetric clusters with the modulus (full and squarefree) exceed the proven bound at r = N^{3/11}
    N = int(make_semiprime(64, "rsa", 7, 0).N)
    r = int(round(N ** (3 / 11)))
    W = lemma_d_window(N, r)
    sf = squarefree_flags(r)
    full = np.ones(r + 1, dtype=bool)
    b = symmetric_pigeonhole_bound(N, r, 0.4)
    zf = symmetric_cluster_with_modulus(N, r, b["j"], full, W)
    zs = symmetric_cluster_with_modulus(N, r, b["j"], sf, W)
    assert zf["links"] >= b["L"] and zf["cluster"] >= b["bound"] > 0.2 * N ** (1 / 11)
    assert zs["cluster"] >= KAPPA_0 * b["bound"]
    # regimes: below the crossing (W >> L) the bound is a fraction of r^{1/3} approaching 0.438 (finite-size deficit
    # from the -3 in L); above it (W << L) it approaches 2W - 1
    r_small = int(round(N ** 0.21))
    b_small = symmetric_pigeonhole_bound(N, r_small, 0.397)
    assert b_small["W_real"] > 20 * b_small["L"] and 0.3 < b_small["bound"] / r_small ** (1 / 3) < 0.438
    r_big = int(round(N ** 0.29))
    b_big = symmetric_pigeonhole_bound(N, r_big, 0.4)
    assert b_big["L"] > 5 * b_big["W"] and b_big["bound"] > 0.6 * (2 * b_big["W"] - 1)


def test_theorem_q_prime_density_and_construction():
    # kappa_j: (5/8) prod_{p|j}(1 - 1/p^2) prod_{p not | 2j}(1 - 3/p^2)
    assert abs(symmetric_density_model(1) - KAPPA_0) < 2e-3 and 0.31 < KAPPA_0 < 0.32
    assert 0.45 < symmetric_density_model(15) < 0.46 and symmetric_density_model(15) > symmetric_density_model(7) > symmetric_density_model(1) - 0.02
    # measured densities at r = 2^20 match kappa_j within sampling error (binomial sd <= 0.08 at >= 40 members)
    r = 2 ** 20
    sf = squarefree_flags(r)
    for j in (1, 3, 5, 15, 21, 105):
        z = symmetric_family_window_fit(r, j, sf)
        assert z["members"] > 40
        sd = math.sqrt(0.25 / z["members"])
        assert abs(z["squarefree_links"] / z["members"] - symmetric_density_model(j)) < 3 * sd
        # the spread of the speeds matches the first-order prediction r^{1/2}/(8 j^{3/2}) within 3%
        assert abs(z["spread_over_window"] / z["predicted_spread"] - 1) < 0.03
    # the theorem's construction (delta = 2/r): all pairs fit, the exact Lemma S3 bound holds, and the squarefree
    # count is near kappa_j L
    for theta in (1.0, 0.5):
        z = theorem_q_prime_check(r, theta=theta, sf=sf)
        assert z["fits"] and z["spread_over_window"] / theta <= z["exact_fit_bound"] <= 1
        assert abs(z["squarefree_links"] - z["kappa_L"]) < 3 * math.sqrt(z["kappa_L"]) + 2
    # the explicit sieve bound is kappa_j L minus the stated error terms (and is weak at this size)
    lb = sieve_lower_bound(1000, 2000, 15, 7)
    assert lb < symmetric_density_model(15) * 1000
