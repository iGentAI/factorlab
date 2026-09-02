import math
from fractions import Fraction

import numpy as np

from factorlab.experiments.prime_subfamily import (
    identify_two_progression, two_progression_members, two_progression_prediction, two_progression_cluster, d_star_for,
    prime_chain_pairs_brute, peel, prime_shell, phase_null_for, offset_resonance_check,
)
from factorlab.gen import make_semiprime
from factorlab.experiments.sidon_scaling import e1_chain_pairs, shell, d_star, rho, pair_speed, symmetric_pair_cluster, squarefree_flags


def test_identifier_recovers_known_families_and_rejects_generic_pairs():
    # symmetric family j = 15: s = 15 d^2, Delta_d = 1/4, drift-free
    mem = two_progression_members(2 ** 14, 15, 1, 0)
    f = identify_two_progression([(km, kp) for _, km, kp in mem])
    assert Fraction(f["A"]) == 15 and Fraction(f["B"]) == 0 and Fraction(f["C"]) == 0 and Fraction(f["delta_d"]) == Fraction(1, 4)
    assert f["support"] == f["of"] == len(mem) and f["drift_free"] and abs(f["tau_sq"] - 1 / 30) < 1e-12
    # consecutive e = 1 links of all classes of the chain at j = 15: s = d^2/15 + (15^2 - 1)/60, capacity 5.66 j
    f = identify_two_progression(e1_chain_pairs(2 ** 13, 15))
    assert Fraction(f["A"]) == Fraction(1, 15) and Fraction(f["C"]) == Fraction(224, 60) and Fraction(f["delta_d"]) == Fraction(1, 900)
    assert abs(f["tau_sq"] - 7.5) < 1e-12 and abs(f["capacity_d"] - 4 * math.sqrt(2) * 15) < 1e-9
    # a c != 0 family in the normal form (a2, b2, c) = (153, 35, 1): s = (153/35^2) d^2 + 2
    mem = two_progression_members(2 ** 20, 153, 35, 1)
    f = identify_two_progression([(km, kp) for _, km, kp in mem])
    assert Fraction(f["A"]) == Fraction(153, 1225) and Fraction(f["C"]) == 2 and f["drift_free"] and f["support"] == len(mem)
    # orientation-insensitive
    g = identify_two_progression([(kp, km) for _, km, kp in mem])
    assert g["A"] == f["A"] and g["support"] == f["support"]
    # four pairs not on one quadratic in d are generic
    assert identify_two_progression([(10, 20), (11, 25), (13, 31), (17, 40)]) is None
    assert identify_two_progression([(10, 20), (11, 25), (13, 31)]) is None  # three pairs never suffice


def test_d_star_for_matches_d_star_and_recovers_the_window():
    r = 4096
    ks = shell(r, True)
    z = d_star_for(ks, r)
    w = d_star(r)
    assert z["D_star"] == w["D_star"] == 9 and abs(z["tau_sq"] - w["tau"] ** 2) < 1e-9
    assert len(z["pairs_at_max"]) == z["D_star"]
    sp = [pair_speed(k, kp) for k, kp in z["pairs_at_max"]]
    assert max(sp) - min(sp) < 2 * rho(r) and all(k < kp for k, kp in z["pairs_at_max"])
    # the null is well defined on an arbitrary member set
    rng = np.random.default_rng(3)
    assert 1 <= phase_null_for(prime_shell(2048), 2048, rng) <= 6


def test_two_progression_prediction_and_cluster():
    r = 2 ** 16
    sf = squarefree_flags(r)
    # the symmetric family (a2, b2, c) = (j, 1, 0) has transition a* = 0.125 r^{1/3}, i.e. j* = r^{1/3}/4
    z = two_progression_prediction(r, 15, 1, 0)
    assert abs(z["a_star"] / r ** (1 / 3) - (math.sqrt(2) / 32) ** (2 / 3)) < 1e-12 and abs(z["delta"] - 0.25) < 1e-12
    assert abs(z["tau_sq"] - 1 / 30) < 1e-12
    # and its cluster agrees with the E27 enumerator
    for j in (9, 15, 21):
        a = two_progression_cluster(r, j, 1, 0, sf)
        b = symmetric_pair_cluster(r, j, sf)
        assert a["members"] == b["members"] and a["links"] == b["links"] and a["cluster"] == b["cluster"]
    # integrality: a2 and b2 must have equal parity
    try:
        two_progression_members(r, 4, 1, 0)
        assert False
    except ValueError:
        pass


def test_proposition_R_prime_chain_pairs_by_brute_force():
    for r in (512, 1024, 2048):
        z = prime_chain_pairs_brute(r, a_max=48)
        assert z["relation_holds_for_all"] and z["same_a_are_twins"]
        assert z["chain_pairs"] >= z["twin_pairs"] >= 1 and z["a_ratio_max"] < math.sqrt(2)
        # no two chain pairs share a window
        assert z["min_speed_gap_over_2rho"] > 1


def test_peel_removes_identified_pairs():
    r = 4096
    rows = peel(shell(r, True), r, rounds=2)
    assert rows[0]["D_star"] == 9 and Fraction(rows[0]["family"]["A"]) == Fraction(1, 15) and rows[0]["removed"] == 9
    assert rows[1]["R"] == rows[0]["R"] - 9 and rows[1]["D_star"] <= 9


def test_offset_resonance_on_a_real_modulus():
    # with the modulus, the family (((m-1)^2+1)/2, ((m+1)^2+1)/2) has a stationary start difference at m* = (4 sqrt2 u)^{1/3}/2;
    # when k* lies in the shell ~0.45 N^{1/12} members share one value, otherwise only a few
    N = int(make_semiprime(64, "rsa", 7, 0).N)
    z = offset_resonance_check(N, 0.87)
    assert z["k_star_in_shell"] and z["W"] == 1 and z["same_D"] >= 10
    lo, hi = z["m_range_same_D"]
    assert lo <= z["m_star"] <= hi and 0.25 * z["N_1_12"] < z["same_D"] < 0.8 * z["N_1_12"]
    w = offset_resonance_check(N, 0.6)
    assert not w["k_star_in_shell"] and w["same_D"] <= 5
