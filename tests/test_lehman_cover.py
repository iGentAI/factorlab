import math

import numpy as np

from factorlab.experiments.lehman_cover import (
    ceil_2sqrt, window_length, window_start, approx_sidon, harvey_cost, harvey_cover, short_window_subfamily,
    cover_lower_bound, lehman_cells,
)
from factorlab.gen import make_semiprime


def test_exact_window_arithmetic():
    N = int(make_semiprime(40, "rsa", 3, 0).N)
    for k in (1, 7, 100, 12345):
        c = ceil_2sqrt(k, N)
        assert (c - 1) ** 2 < 4 * k * N <= c * c
    r = 237
    for a, b in ((1, 1), (1, 200), (3, 70), (15, 15)):
        W = window_length(N, r, a, b)
        assert W >= 1 and 16 * r * r * a * b * W * W >= N and (W == 1 or 16 * r * r * a * b * (W - 1) ** 2 < N)
        assert abs(W - math.ceil(math.sqrt(N) / (4 * r * math.sqrt(a * b)))) <= 1
        assert window_start(N, a, b) == ceil_2sqrt(a * b, N) - a * N - b
    cells = lehman_cells(20, lo=10)
    assert all(10 < a * b <= 20 and math.gcd(a, b) == 1 for a, b in cells) and (1, 11) in cells and (2, 5) not in cells


def test_approx_sidon_progression_and_group_separation():
    W = 10
    # window starts in arithmetic progression with common difference 1000: every consecutive pair
    # has the same difference, so D_max >= R - 1 (ordered pairs: both signs counted separately)
    R = 30
    c = [1000 * i for i in range(R)]
    s = approx_sidon([1] * R, c, [0] * R, W)
    assert s["D_max"] >= R - 1 and s["at_a_diff"] == 0 and abs(abs(s["at_reduced"]) - 1000) <= 2 * W
    # random starts far apart: no window of length 2W holds many differences
    rng = np.random.default_rng(0)
    c = sorted(set(int(x) for x in rng.integers(0, 10 ** 9, size=200)))
    s = approx_sidon([1] * len(c), c, [0] * len(c), W)
    assert s["D_max"] <= 3
    # group separation: two cells with a-difference 1 whose reduced difference is near +M and two
    # cells with a-difference 0 whose reduced difference is near -M must not be counted together
    a = [1, 2, 5, 5]
    c = [0, 10 ** 6, 0, 10 ** 6]
    b = [0, 0, 0, 0]
    s = approx_sidon(a, c, b, W)
    assert s["D_max"] == 1
    # coincident starts at W = 1: the only t with |0 - t| < 1 is t = 0, which is excluded
    assert approx_sidon([1, 1], [0, 0], [0, 0], 1)["D_max"] == 0
    # crowded statistic is inclusive on both sides: starts 0, 100, 200 + (2W - 2) give the ordered
    # differences +-100 and +-(100 + 2W - 2), which are exactly 2W - 2 apart, so all four of
    # these differences (and the two +-(200 + 2W - 2) ones are isolated) count as crowded
    W = 3
    s = approx_sidon([1, 1, 1], [0, 100, 200 + 2 * W - 2], [0, 0, 0], W)
    assert s["pairs_in_crowded_windows"] == 4 and s["D_max"] == 2
    # the grouping by a-difference is certified only when N separates the groups by more than a window
    import pytest
    with pytest.raises(AssertionError):
        approx_sidon([1, 2], [0, 10], [0, 0], 3, N=20)
    assert approx_sidon([1, 2], [0, 10], [0, 0], 3, N=10 ** 6)["D_max"] == 1
    # three starts 0, 3, 6 at W = 2: differences +-3 (x4 ordered... two pairs each) and +-6; t = 3
    # captures the two ordered pairs with difference 3 (|3 - 3| = 0 < 2) and nothing else
    assert approx_sidon([1, 1, 1], [0, 3, 6], [0, 0, 0], 2)["D_max"] == 2
    # brute force against the definition on random small instances
    rng = np.random.default_rng(5)
    for _ in range(20):
        m = int(rng.integers(3, 9))
        W = int(rng.integers(1, 4))
        c = [int(x) for x in rng.integers(0, 40, size=m)]
        aa = [int(x) for x in rng.integers(1, 3, size=m)]
        bb = [int(x) for x in rng.integers(0, 5, size=m)]
        Nbig = 10 ** 6
        s_vals = [ci - ai * Nbig - bi for ai, ci, bi in zip(aa, c, bb)]
        diffs = [s_vals[i] - s_vals[j] for i in range(m) for j in range(m) if i != j]
        brute = 0
        for d0 in diffs:
            for t in range(d0 - W + 1, d0 + W):
                if t != 0:
                    brute = max(brute, sum(1 for d in diffs if abs(d - t) < W))
        assert approx_sidon(aa, c, bb, W)["D_max"] == brute


def test_harvey_cover_satisfies_the_inequality_and_covers():
    N = int(make_semiprime(36, "rsa", 3, 0).N)
    r = int(round(N ** 0.2))
    sub = short_window_subfamily(N, r, mode="a1")
    assert not sub["truncated"] and sub["requested_R"] == sub["R"]
    B, G = harvey_cover(N, r, sub)
    Bs = set(B)
    for (a, b), c, W in zip(sub["cells"], sub["c"], sub["W"]):
        s = c - a * N - b
        for j in range(W):
            assert any((s + j + g) in Bs for g in G)
    row = cover_lower_bound(N, r, mode="a1")
    n, W, D = sub["n"], sub["W_max"], max(1, row["D_max"])
    K1, K2 = len(B), len(G)
    assert K1 * K2 >= n  # counting
    if K1 <= n / (2 * W):
        assert K1 * K2 ** 2 >= n * n / (4 * W * D)
    assert K1 + K2 >= row["lower_bound"]


def test_harvey_cost_is_exact():
    N = int(make_semiprime(32, "rsa", 3, 0).N)
    r = 40
    hc = harvey_cost(N, r)
    cells = lehman_cells(r)
    Ws = [window_length(N, r, a, b) for a, b in cells]
    brute = min(m + sum((w + m - 1) // m for w in Ws) for m in range(1, 4 * int(math.sqrt(sum(Ws))) + 50))
    assert hc["M1_cost"] == brute and hc["cells"] == len(cells) and hc["Sigma_W"] == sum(Ws)
