from fractions import Fraction

import numpy as np

from factorlab.experiments.planar_census import shell_starts, exact_dmax, random_start_null, family_detuning, planar_point, peel_exact
from factorlab.experiments.prime_subfamily import identify_two_progression
from factorlab.experiments.lehman_cover import window_start, squarefree_flags
from factorlab.experiments.sidon_scaling import lemma_d_window, _ceil_2sqrt
from factorlab.gen import make_semiprime


def test_exact_dmax_matches_brute_force():
    rng = np.random.default_rng(5)
    for _ in range(30):
        R = int(rng.integers(3, 40))
        s = np.sort(rng.integers(0, 2000, size=R)).astype(np.int64)
        ks = np.arange(R, dtype=np.int64)
        W = int(rng.integers(1, 6))
        z = exact_dmax(ks, s, W)
        diffs = [int(s[j] - s[i]) for i in range(R) for j in range(i + 1, R)]
        brute = max(sum(1 for d in diffs if abs(d - t) < W) for t in range(min(diffs) - W, max(diffs) + W + 1))
        assert z["D_max"] == brute
        # the recovered pairs all lie in the reported window, and there are exactly D of them
        assert len(z["pairs"]) == z["D_max"]
        for i, j in z["pairs"]:
            assert i < j and abs(int(s[j] - s[i]) - z["t"]) < W


def test_shell_starts_match_window_start():
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    r = 3000
    sf = squarefree_flags(r + 1)
    ks, s = shell_starts(N, r, sf)
    ref = np.array([window_start(1, int(k), N) for k in ks], dtype=object)
    assert all(int(ref[i] - ref[0]) == int(s[i] - s[0]) for i in range(len(ks)))


def test_planar_maximisers_of_the_40_bit_modulus():
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    rng = np.random.default_rng(1)
    r_max = int(round(N ** (1 / 3)))
    sf = squarefree_flags(r_max + 1)
    # r = N^{1/3}, W = 1: the resonant family ((d^2/2 - d + 2)/2, (d^2/2 + d + 2)/2), i.e. (m^2 - m + 1, m^2 + m + 1), eight pairs
    z = planar_point(N, r_max, sf, rng, nulls=1)
    f = z["family"]
    assert z["W"] == 1 and z["D_max"] == 8 and f is not None and f["support"] == 8
    assert Fraction(f["A"]) == Fraction(1, 2) and Fraction(f["C"]) == 2 and Fraction(f["delta_d"]) == Fraction(-3, 4)
    assert abs(f["u_dv_dd_minus_1"]) < 0.05
    # r = N^{0.3}: resonant family A = 1/10, C = 29/10, seven pairs
    r = int(round(N ** 0.3))
    z = planar_point(N, r, sf, rng, nulls=1)
    f = z["family"]
    assert z["D_max"] == 7 and Fraction(f["A"]) == Fraction(1, 10) and abs(f["u_dv_dd_minus_1"]) < 0.05
    # r = N^{3/11}: the maximiser is at the null level and is the symmetric j = 17 family in the cap regime
    # (Delta_d = 1/4 > 0: the speed term and the offset both decrease, detuning ~ -1)
    r = int(round(N ** (3 / 11)))
    z = planar_point(N, r, sf, rng, nulls=2)
    f = z["family"]
    assert z["D_max"] <= max(z["null"]) + 1 and f is not None and f["drift_free"]
    assert Fraction(f["A"]) == 17 and Fraction(f["C"]) == 0 and Fraction(f["delta_d"]) == Fraction(1, 4)
    assert f["u_dv_dd_minus_1"] < -0.8


def test_non_drift_free_family_resonates_with_the_modulus():
    # (33 d^2 - 3, 33 d^2 + d - 3), d = 13..17: B = 1 != 0, so modulus-free its speeds drift like 1/d, yet on the 40-bit
    # modulus all five exact start differences coincide and the exact detuning is small
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    pairs = [(33 * d * d - 3, 33 * d * d + d - 3) for d in range(13, 18)]
    D = [_ceil_2sqrt(kp, N) - _ceil_2sqrt(k, N) - (kp - k) for k, kp in pairs]
    assert len(set(D)) == 1
    f = identify_two_progression(pairs)
    assert Fraction(f["A"]) == 66 and Fraction(f["B"]) == 1 and Fraction(f["C"]) == -6 and not f["drift_free"]
    det = family_detuning(N, f, pairs)
    assert abs(det["u_dv_dd_minus_1"]) < 0.25


def test_peel_exact_removes_and_null_is_small():
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    r = int(round(N ** 0.3))
    sf = squarefree_flags(r + 1)
    rows = peel_exact(N, r, sf, rounds=2)
    assert rows[0]["D_max"] == 7 and rows[0]["removed"] == 7 and rows[1]["R"] == rows[0]["R"] - 7
    ks, s = shell_starts(N, r, sf)
    W = lemma_d_window(N, r)
    assert 1 <= random_start_null(s, W, np.random.default_rng(2)) <= 8
