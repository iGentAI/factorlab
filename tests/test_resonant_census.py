import json
import math
from fractions import Fraction

import numpy as np

from factorlab.experiments.resonant_census import (
    divisors_in_range, integral_classes, family_members, analyse_family, enumerate_resonant_families, w_real, window_cluster,
    squarefree_mask, theorem_w_check, bounded_w_family_check,
)
from factorlab.experiments.sidon_scaling import lemma_d_window
from factorlab.experiments.lehman_cover import squarefree_flags
from factorlab.gen import make_semiprime


def test_divisors_classes_and_members():
    assert divisors_in_range(360, 5, 20) == [5, 6, 8, 9, 10, 12, 15, 18, 20]
    # A = 1/2, C = 2 with q = 2: alpha = 2, gamma = 8; integral exactly on even d
    assert integral_classes(2, 8, 2) == [0]
    # A = 29/3, C = 4/3 with q = 3: alpha = 87, gamma = 12; integral on d = +-1 (mod 3)
    assert integral_classes(87, 12, 3) == [1, 2]
    mem = family_members(10000, 2, 8, 2, 0)
    assert len(mem) > 20
    for d, km, kp in mem:
        assert d % 2 == 0 and kp - km == d and 5000 < km and kp <= 10000
        m = d // 2
        assert (km, kp) == (m * m - m + 1, m * m + m + 1)
    # squarefree mask agrees with the sieve
    ks = np.arange(5001, 10001, dtype=np.int64)
    sf = squarefree_flags(10000)
    assert np.array_equal(squarefree_mask(ks), sf[ks])
    assert window_cluster([3, 4, 4, 5, 9, 10], 1) == (2, 4) and window_cluster([3, 4, 4, 5, 9, 10], 2) == (4, 3)
    # both branches of k_- > r/2: with C = 248 > r = 180 the members sit at small d (the referee's case)
    mem = family_members(180, 1, 155000, 25, 0)
    assert mem == [(25, 112, 137), (50, 101, 151), (75, 91, 166)]
    # brute force agreement on a family with both branches and on an ordinary one
    for (r, alpha, gamma, q, D0) in ((180, 1, 155000, 25, 0), (10000, 2, 8, 2, 0), (3000, 87, 12, 3, 1)):
        brute = []
        for d in range(D0 if D0 else q, 4 * r + 1, q):
            num_m, num_p = alpha * d * d - q * q * d + gamma, alpha * d * d + q * q * d + gamma
            if num_m % (2 * q * q) or num_p % (2 * q * q):
                continue
            km, kp = num_m // (2 * q * q), num_p // (2 * q * q)
            if km > r // 2 and kp <= r:
                brute.append((d, km, kp))
        assert family_members(r, alpha, gamma, q, D0) == brute


def test_merged_representations_and_validation_against_e31_results():
    # A = 1/2, C = 2 appears with q = 2 (primitive) and with its refinements; the merged member set is the q = 2 one
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    r = int(round(N ** (1 / 3)))
    fams = enumerate_resonant_families(N, r, q_max=32, M_max=600, min_population=2, mask_threshold=1)
    half = [z for z in fams if z["A"] == Fraction(1, 2) and z["C"] == 2][0]
    qs = sorted(t[0] for t in half["representations"])
    assert half["q"] == 2 == qs[0] and len(qs) > 1 and all(t % 2 == 0 for t in qs)
    assert half["members"] == len(family_members(r, 2, 8, 2, 0))
    # every census family in every published run had C < r, so the small-d branch was empty there
    assert all(z["C"] < r for z in fams)
    # the census's squarefree maximum equals E31's exact D_max at every common (bits, r) of the stored results
    with open("results/e31_planar_census.json") as fh:
        e31 = json.load(fh)
    with open("results/e32_resonant_census.json") as fh:
        e32 = json.load(fh)
    exact = {(p["N_bits"], p["r"]): p["D_max"] for p in e31["points"] if p["family_name"] == "squarefree"}
    common = 0
    for z in e32:
        key = (z["N_bits"], z["r"])
        if key in exact:
            common += 1
            assert z["max_cluster_squarefree"] == exact[key], (key, z["max_cluster_squarefree"], exact[key])
    assert common >= 8          # four moduli (40#0, 40#1, 48#0, 48#1) at r = N^{0.3} and N^{1/3}


def test_theorem_w_pigeonhole_mechanics():
    # at most three distinct start differences in the resonance window, and a value shared by >= ceil(M/3) of its
    # M members (M >= 2L - 1); the window lies inside the shell from 48 bits (2L/d_* < 0.094 needs N > 2^46)
    for bits, idx in ((48, 1), (64, 0), (80, 0), (96, 0)):
        N = int(make_semiprime(bits, "rsa", 7, idx).N)
        z = theorem_w_check(N, lam=0.8)
        assert z["W"] == 1 and z["clipped"] == 0 and z["distinct_values"] <= 3
        assert z["largest_multiplicity"] >= z["pigeonhole_bound"] == -(-z["members"] // 3)
        assert z["members"] >= 2 * 0.8 * math.sqrt(z["d_star"]) - 1 and z["largest_multiplicity"] >= z["asymptotic_bound"]
    z = theorem_w_check(int(make_semiprime(40, "rsa", 7, 0).N), lam=0.8)
    assert z["clipped"] > 0 and z["distinct_values"] <= 3
    # prime pairs sharing the most frequent value: 11 of the 12 prime pairs of the window at 128 bits (E32)
    z = theorem_w_check(int(make_semiprime(128, "rsa", 7, 0).N), lam=0.8, count_primes=True)
    assert z["prime_pairs_all"] == 12 and z["prime_pairs_at_t"] == 11 and 3 < z["bateman_horn_constant"] < 4


def test_theorem_w_prime_bounded_window_families():
    # Theorem W': at r = N^{1/3}/C the families (n, m) = (1,1), (3,3), (7,5), (13,7) have delta/n inside
    # (1/(2 W_real), sqrt2/W_real), a stationary point interior to the shell, at most three start differences on the
    # resonance window and a value shared by >= ceil(M/3) members
    import gmpy2
    for bits in (64, 96):
        N = int(make_semiprime(bits, "rsa", 7, 0).N)
        r13 = int(gmpy2.iroot(gmpy2.mpz(N), 3)[0])
        for C, (n, m) in ((1, (1, 1)), (2, (3, 3)), (3, (7, 5)), (4, (13, 7))):
            z = bounded_w_family_check(N, r13 // C, n, m)
            assert z is not None and z["ratio_inside"] and 0.55 < z["z_over_r"] < 0.9
            assert z["distinct_values"] <= 3 and z["largest_multiplicity"] >= z["pigeonhole_bound"]
            assert z["members"] >= 5 and z["W"] >= 1


def test_a_half_family_reproduces_the_exact_planar_maximum():
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    r = int(round(N ** (1 / 3)))
    W = lemma_d_window(N, r)
    z = analyse_family(N, r, W, 2, 8, 2, [0])
    assert z["A"] == Fraction(1, 2) and z["C"] == 2 and z["M"] == -12
    # E31: D_max = 8 at this (N, r), attained by this family's squarefree members, with d_med = 182 near d* = (3u)^{1/3} = 180
    assert z["cluster_squarefree"] == 8 and z["cluster_full"] >= 8
    lo, hi = z["d_cluster_range"]
    d_star = (3 * 2 * math.sqrt(N)) ** (1 / 3)
    assert lo <= d_star <= hi and abs(z["detuning"]) < 0.1
    # resonance range: alpha = 2 lies in [|M| W_real/(2 sqrt2), |M| W_real)
    Wr = w_real(N, r)
    assert 12 * Wr / (2 * math.sqrt(2)) <= 2 < 12 * Wr


def test_census_contains_the_peeled_maximisers_of_e31():
    # every drift-free family peeled from the squarefree shell at 40 bits in E31 must be in the census with a
    # squarefree cluster at least the peeled count: the census window is taken on the intact squarefree shell and
    # peeling only removes pairs, so the peeled round's window (all of whose pairs were identified as this family)
    # is a subset of one of the family's windows on the intact shell
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    with open("results/e31_planar_census.json") as fh:
        e31 = json.load(fh)
    for block in e31["peel"]:
        if block["N_bits"] != 40 or block["idx"] != 0:
            continue
        r = block["r"]
        fams = enumerate_resonant_families(N, r, q_max=32, M_max=600, min_population=2, mask_threshold=1)
        table = {(str(z["A"]), str(z["C"])): z for z in fams}
        for row in block["rows"]:
            f = row["family"]
            if f is None or not f["drift_free"] or Fraction(f["delta_d"]) >= 0:
                continue
            key = (f["A"], f["C"])
            assert key in table, (r, key)
            assert table[key]["cluster_squarefree"] >= row["D_max"]
        # the census maximum over squarefree clusters equals the exact D_max of the first round when it is drift-free
        f0 = block["rows"][0]["family"]
        if f0 is not None and f0["drift_free"] and Fraction(f0["delta_d"]) < 0:
            assert max(z["cluster_squarefree"] for z in fams) == block["rows"][0]["D_max"]
