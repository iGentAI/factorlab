import math

import numpy as np

from factorlab.experiments.chirp_dynamics import (
    speed_census, moving_point_cluster_max, _integer_centre_cluster_max, first_moment_bound, d_max_for_N,
)
from factorlab.experiments.lehman_cover import approx_sidon, short_window_subfamily, squarefree_flags
from factorlab.gen import make_semiprime


def _exact_equal_speed(k1, k2, k3, k4):
    """sqrt k1 - sqrt k2 == sqrt k3 - sqrt k4 exactly, via squaring with exact integers."""
    # a := sqrt k1 + sqrt k4, b := sqrt k2 + sqrt k3; equality iff a == b, i.e. a^2 == b^2 and same sign (both positive)
    # a^2 = k1 + k4 + 2 sqrt(k1 k4), b^2 = k2 + k3 + 2 sqrt(k2 k3): equal iff k1 + k4 - k2 - k3 == 2(sqrt(k2 k3) - sqrt(k1 k4))
    s = k1 + k4 - k2 - k3
    # 2(sqrt(k2k3) - sqrt(k1k4)) = s  ->  4 k2 k3 + 4 k1 k4 - s^2 = 8 sqrt(k1 k2 k3 k4) ... check exactly
    lhs = 4 * k2 * k3 + 4 * k1 * k4 - s * s
    if lhs < 0:
        return False
    prod = 64 * k1 * k2 * k3 * k4
    if lhs * lhs != prod:
        return False
    # also need the sign: sqrt(k2k3) - sqrt(k1k4) has the sign of s
    return (k2 * k3 - k1 * k4) * s >= 0 if s != 0 else k2 * k3 == k1 * k4


def test_equal_speeds_are_exactly_the_beatty_chains():
    r = 100  # shell (50, 100]: squares 64, 81, 100 -> consecutive pairs (81,64), (100,81) share speed 1
    ks = list(range(r // 2 + 1, r + 1))
    pairs = [(a, b) for a in ks for b in ks if a != b]
    repeated = 0
    for i, (k1, k2) in enumerate(pairs):
        for (k3, k4) in pairs[i + 1:]:
            if _exact_equal_speed(k1, k2, k3, k4):
                repeated += 1
                # both pairs on a common Beatty chain k = j m^2 with the same step
                def core(k):
                    m = int(math.isqrt(k))
                    while k % (m * m):
                        m -= 1
                    return k // (m * m), m
                j1, m1 = core(k1); j2, m2 = core(k2); j3, m3 = core(k3); j4, m4 = core(k4)
                assert j1 == j2 == j3 == j4 and m1 - m2 == m3 - m4
    census = speed_census(r, squarefree=False)
    # classes +1 and -1, two ordered pairs each: two unordered coincidences in the brute force
    assert repeated == 2 and census["repeated_speed_classes"] == 2 and census["pairs_in_repeated_speeds"] == 4
    assert census["largest_speed_class"] == 2
    sf = speed_census(r, squarefree=True)
    assert sf["repeated_speed_classes"] == 0 and sf["distinct_speeds"] == sf["pairs"]
    assert _exact_equal_speed(81, 64, 100, 81) and _exact_equal_speed(50, 32, 50, 32)
    assert not _exact_equal_speed(35, 33, 37, 35) and not _exact_equal_speed(81, 64, 64, 81)


def test_integer_centre_statistic_matches_exact_definition_on_integer_points():
    rng = np.random.default_rng(3)
    for _ in range(30):
        m = int(rng.integers(3, 12))
        W = int(rng.integers(1, 5))
        c = [int(x) for x in rng.integers(0, 60, size=m)]
        exact = approx_sidon([1] * m, c, [0] * m, W)["D_max"]
        pts = np.array(sorted(ci - cj for i, ci in enumerate(c) for j, cj in enumerate(c) if i != j), dtype=np.float64)
        assert _integer_centre_cluster_max(pts, W) == exact


def test_moving_points_track_the_exact_starts():
    N = int(make_semiprime(40, "rsa", 3, 0).N)
    for e in (0.2, 0.25):
        r = int(round(N ** e))
        d = d_max_for_N(N, r)
        assert abs(d["D_max"] - d["moving_point_max"]) <= 2
        sub = short_window_subfamily(N, r, mode="a1sqfree")
        U = 2.0 * (math.sqrt(2.0 ** 40) - math.sqrt(2.0 ** 39))
        fm = first_moment_bound(N, r, sub["b"], sub["W_max"], U)
        assert 0 < fm < 3 * math.log(r) + 3
