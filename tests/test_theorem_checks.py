"""Tests for the theorem-check experiment (E56): exact ceilings and the archived 48-bit Theorem W values."""
import json
import math
import os

from factorlab.experiments.theorem_checks import ceil_2sqrt, icbrt, start_difference, theorem_w_check
from factorlab.gen import make_semiprime


def test_ceil_2sqrt_and_icbrt_exact():
    for k in range(1, 60):
        for N in (91, 1001, 2 ** 31 - 1, 3 * 10 ** 12 + 7):
            v = ceil_2sqrt(k, N)
            assert (v - 1) ** 2 < 4 * k * N <= v ** 2
    for N in (7, 8, 26, 27, 28, 10 ** 18, 10 ** 18 + 1, 2 ** 90 - 1):
        c = icbrt(N)
        assert c ** 3 <= N < (c + 1) ** 3


def test_start_difference_matches_definition():
    N = 964754165423
    for d in range(10, 40):
        km, kp = (d * d - d + 2) // 2, (d * d + d + 2) // 2
        s_m = ceil_2sqrt(km, N) - N - km
        s_p = ceil_2sqrt(kp, N) - N - kp
        assert start_difference(km, kp, N) == s_p - s_m


def test_theorem_w_check_48_bits_matches_archive():
    N = int(make_semiprime(48, "rsa", 7, 0).N)
    res = theorem_w_check(N, 0.8)
    assert res["members"] == 28 and res["distinct_values"] == 3 and res["most_frequent_count"] == 16
    assert res["most_frequent_share"] >= 1.0 / 3.0
    arch = "results/e56_theorem_checks.json"
    if os.path.exists(arch):
        row = [x for x in json.load(open(arch))["theorem_W"] if x["bits"] == 48][0]
        assert row["members"] == res["members"] and row["most_frequent_count"] == res["most_frequent_count"]
        assert math.isclose(row["d_star"], res["d_star"], rel_tol=1e-12)
