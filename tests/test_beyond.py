import itertools
import math

import numpy as np
import pytest

from factorlab.experiments.poly_floor import (
    quad_floor, cubic_floor, is_irreducible, floor_exponents, expected_pairs, predicted_mean,
    predicted_mean_crude, refined_pairs, base_m_construction, leading_coefficient_search, _eval,
    poisson_check, _m_lo, _tuple_bound, _shell_bound, _f0_cap,
)


from factorlab.experiments.quadratic_bridge import (
    quadratic_to_small_square, small_square_to_quadratic, bounded_k1_quadratics, predicted_k1_count,
    nearest_square_phases, primitive_event, bounded_k_quadratics, predicted_k_count,
)


def brute_floor(N: int, d: int, Hb: int):
    """Independent minimum of ||f||_oo * min(m, N-m) over irreducible f with max coeff <= Hb.

    Evaluates f_d m^d + ... + f_1 m at every residue m modulo N directly (no
    factorisation of N, no CRT, no square-root tables) and reads off f_0.
    """
    ms = np.arange(N, dtype=np.int64)
    best = (math.inf, None, None)
    rng = range(-Hb, Hb + 1)
    for top in itertools.product(range(1, Hb + 1), *[rng] * (d - 1)):  # (f_d, ..., f_1)
        g = np.zeros(N, dtype=np.int64)
        for c in top:
            g = (g * ms + c) % N
        g = (g * ms) % N  # f_d m^d + ... + f_1 m
        small = np.nonzero((g <= Hb) | (g >= N - Hb))[0]
        for m in small:
            v = int(g[m])
            f0 = -v if v <= Hb else N - v
            f = [f0] + list(reversed(top))
            mm = min(int(m), N - int(m))
            P = max(abs(c) for c in f) * mm
            if P < best[0] and mm > 0 and is_irreducible(f):
                best = (P, f, mm)
    return best


@pytest.mark.parametrize("p,q", [(11, 13), (17, 19)])
def test_quad_floor_matches_brute_force(p, q):
    N = p * q
    r = quad_floor(N, p, q)
    assert r["certified"]
    assert is_irreducible(r["f"])
    assert _eval(r["f"], r["m"]) % N == 0 or _eval(r["f"], -r["m"]) % N == 0
    # Independent global bound: every improving pair has m >= 1, hence ||f||_oo <= P-1.
    # This does not reuse the enumerator's shell/cap calculations.
    b = brute_floor(N, 2, r["P"] - 1)
    assert b[0] >= r["P"], (b, r)


@pytest.mark.parametrize("p,q", [(11, 13), (13, 17)])
def test_cubic_floor_matches_brute_force(p, q):
    N = p * q
    r = cubic_floor(N, p, q)
    assert r["certified"]
    assert is_irreducible(r["f"])
    assert _eval(r["f"], r["m"]) % N == 0 or _eval(r["f"], -r["m"]) % N == 0
    # Independent global coefficient bound, not one derived from the pruning logic.
    b = brute_floor(N, 3, r["P"] - 1)
    assert b[0] >= r["P"], (b, r)


def test_prediction_constants():
    e2 = floor_exponents(2)
    assert abs(e2["C"] - 54.0 / 5) < 1e-9 and abs(e2["e"] - 5) < 1e-12 and abs(e2["g"] - 3) < 1e-12
    assert abs(e2["product"] - 0.6) < 1e-12 and abs(e2["coeff"] - 0.2) < 1e-12 and abs(e2["root"] - 0.4) < 1e-12
    e3 = floor_exponents(3)
    assert abs(e3["C"] - 512.0 / 33) < 1e-9 and abs(e3["e"] - 5.5) < 1e-12 and abs(e3["g"] - 2.5) < 1e-12
    assert abs(e3["product"] - 5 / 11) < 1e-12
    N = 2 ** 40
    x = predicted_mean_crude(N, 2)
    # the mean of exp(-count) distribution: count at the scale equals 1 up to Gamma
    scale = x / math.gamma(1.2)
    assert abs(expected_pairs(scale, N, 2) - 1.0) < 1e-9
    # the refined count is smaller than the crude one (larger admissible-root cutoff)
    assert refined_pairs(scale, N, 2) < 1.0
    # continuum estimate for d = 2: (2/3) x^5/N^3 versus (54/5) x^5/N^3,
    # so the refined mean is about 1.75x the mirror-consistent crude one.
    ratio = predicted_mean(N, 2) / x
    assert 1.5 < ratio < 2.0, ratio
    for best in (3, 17, 100, 1000):
        for d in (2, 3):
            cap = _f0_cap(best, 143, d)
            assert 143 * cap ** (d - 1) <= (d + 1) * (best - 1) ** d
            assert 143 * (cap + 1) ** (d - 1) > (d + 1) * (best - 1) ** d


@pytest.mark.parametrize("d", [2, 3])
def test_constructions_have_roots(d):
    N = 1000003 * 1000033
    b = base_m_construction(N, d)
    assert _eval(b["f"], b["m"]) % N == 0 and len(b["f"]) == d + 1 and b["f"][-1] != 0
    s = leading_coefficient_search(N, d, 50)
    assert s is not None and _eval(s["f"], s["m"]) % N == 0 and is_irreducible(s["f"])
    assert s["P"] <= b["P"] * 4  # the search at small K is in the same regime as base-m


def test_pruning_bounds_are_lower_envelopes():
    """The integer product max(h,F) m_lo(fd,h,F) is not monotone (N = 7, fd = h = 2: 4 then 3),
    so the tuple and shell bounds must sit below it for every F <= N/2, fd <= h, h' >= h."""
    assert 2 * _m_lo(2, 2, 2, 7, 2) == 4 and 3 * _m_lo(2, 2, 3, 7, 2) == 3
    assert _tuple_bound(2, 2, 7, 2) <= 3
    rng = np.random.default_rng(5)
    for _ in range(300):
        d = int(rng.integers(2, 4))
        N = int(rng.integers(50, 20000))
        h = int(rng.integers(1, 30))
        fd = int(rng.integers(1, h + 1))
        tb = _tuple_bound(fd, h, N, d)
        for F in range(0, min(N // 2, 80) + 1):
            assert tb <= max(h, F) * _m_lo(fd, h, F, N, d), (d, N, h, fd, F)
        sb = _shell_bound(h, N, d)
        assert sb <= max(h, 0) * _m_lo(fd, h, 0, N, d)
        for hp in range(h, h + 4):
            for fdp in range(1, hp + 1):
                for F in range(0, min(N // 2, 40) + 1, 3):
                    assert sb <= max(hp, F) * _m_lo(fdp, hp, F, N, d), (d, N, h, hp, fdp, F)


def test_bridge_round_trips_random_quadratics():
    rng = np.random.default_rng(11)
    n_done = 0
    while n_done < 200:
        a = int(rng.integers(1, 30)); b = int(rng.integers(-30, 31)); c = int(rng.integers(-30, 31))
        m = int(rng.integers(1, 60))
        if not is_irreducible([c, b, a]):
            continue
        v = a * m * m + b * m + c
        if v == 0:
            continue
        k = int(rng.choice([1, -1, 2, -3]))
        if v % k:
            continue
        N = v // k
        if N <= 0:
            N, k = -N, -k
        sq = quadratic_to_small_square(N, [c, b, a], m)
        assert sq["delta"] == b * b - 4 * a * c and sq["y"] ** 2 - 4 * a * sq["k"] * N == sq["delta"]
        back = small_square_to_quadratic(N, sq["a"], sq["k"], sq["y"], sq["b"])
        assert back["f"] == [c, b, a] and back["m"] == sq["m"] and abs(sq["m"]) == m and back["k"] == sq["k"]
        n_done += 1


def test_bounded_scan_includes_negative_y_witness():
    # f = x^2 - 3x + 3, f(1) = f(2) = 1 = N; at m = 1, y = 2am + b = -1 < 0; disc = -3 so f is irreducible.
    # With N <= H a polynomial may have two positive roots and is then emitted once per root.
    recs = bounded_k1_quadratics(1, 3)
    assert any(r["f"] == [3, -3, 1] and r["m"] == 1 for r in recs)
    assert any(r["f"] == [3, -3, 1] and r["m"] == 2 for r in recs)
    assert len(recs) == len({(r["f"][0], r["f"][1], r["f"][2], r["m"]) for r in recs})  # pairs are not duplicated
    for r in recs:
        c, b, a = r["f"]
        assert a * r["m"] ** 2 + b * r["m"] + c == 1 and r["m"] > 0 and r["H"] <= 3


def _brute_pairs(N: int, H: int, k: int = 1) -> set:
    out = set()
    target = k * N
    for a in range(1, H + 1):
        for b in range(-H, H + 1):
            for c in range(-H, H + 1):
                if not is_irreducible([c, b, a]):
                    continue
                m = 1
                while a * m * m - H * m - H <= target:
                    if a * m * m + b * m + c == target:
                        out.add((c, b, a, m))
                    m += 1
    return out


@pytest.mark.parametrize("N,H,k", [(143, 6, 1), (221, 5, 1), (37, 4, 1), (1000003 * 7, 3, 1), (143, 5, 2), (91, 4, 3)])
def test_bounded_scan_matches_brute_force(N, H, k):
    recs = bounded_k_quadratics(N, H, k)
    scan = [(r["f"][0], r["f"][1], r["f"][2], r["m"]) for r in recs]
    assert len(scan) == len(set(scan))  # no duplicate pairs
    assert set(scan) == _brute_pairs(N, H, k)
    if k * N > H:  # each polynomial at most once
        polys = [(r["f"][0], r["f"][1], r["f"][2]) for r in recs]
        assert len(polys) == len(set(polys))
    assert abs(predicted_k_count(2 ** 40, 100, 4) - predicted_k1_count(2 ** 40, 100) / 2) < 1e-9


def test_predicted_count_scale_and_phases():
    N, H = 2 ** 60, 4096
    lam = predicted_k1_count(N, H)
    assert abs(lam / (4 * H ** 2.5 / math.sqrt(N)) - 1) < 0.05
    # finite sanity at 40 bits: the mean scan count over 20 moduli is within a factor 2 of lambda
    from factorlab.gen import make_semiprime
    counts, lams = [], []
    for i in range(20):
        Ni = int(make_semiprime(40, "rsa", 3, i).N)
        Hi = int(round(Ni ** 0.2))
        counts.append(len(bounded_k1_quadratics(Ni, Hi)))
        lams.append(predicted_k1_count(Ni, Hi))
    assert 0.5 < np.mean(counts) / np.mean(lams) < 2.0, (np.mean(counts), np.mean(lams))
    ph = nearest_square_phases(2 ** 40 + 15, 500)
    assert ph.size == 500 and np.all(np.abs(ph) <= 0.5 + 1e-6)
    assert primitive_event(12, 10) == (3, 5) and primitive_event(7, 9) == (7, 9) and primitive_event(36, 12) == (1, 2)


def test_poisson_check_self_consistent():
    """Minima generated from the model (u = count(P) at Exp(1) quantiles) pass the check."""
    d, N = 2, 2 ** 30
    us = -np.log(1.0 - (np.arange(1, 41) - 0.5) / 40.0)  # Exp(1) quantiles
    insts = []
    for u in us:
        lo, hi = 1.0, float(N)
        for _ in range(200):  # bisection on the monotone count
            mid = math.sqrt(lo * hi)
            if refined_pairs(mid, N, d) < u:
                lo = mid
            else:
                hi = mid
        insts.append({"N": str(N), "P": hi})
    chk = poisson_check({"d": d, "rows": [{"instances": insts}]})
    assert abs(chk["refined"]["mean_u"] - float(us.mean())) < 0.02
    assert chk["refined"]["p"] > 0.5
