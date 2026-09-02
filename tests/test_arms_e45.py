"""Tests for E45 (class counts for even periods; Lehman hit classification)."""
from factorlab.experiments.arms_e45 import (
    class_count,
    classify_hits,
    construct_sliver_hits,
    continued_fraction,
    convergents_and_mediants,
    is_hit,
    omega,
    r_of,
)


def test_class_count_known_families():
    # The E39 maximiser A = 7/15, C = 8/15 (q = 15, M = 1): alpha = A q^2 = 105, gamma = C q^2 = 120; four classes = 2^omega(15).
    t, n = class_count(105, 120, 15)
    assert t == 15 and n == 4 == 2 ** omega(15)
    # A = 1/2, C = 2 (Theorem W's family, q = 1, M = -7) represented with q = 2: alpha = 2, gamma = 8.
    res = class_count(2, 8, 2)
    assert res is not None
    t, n = res
    assert n <= 2 ** (omega(t) + 1)


def test_cf_and_mediants_and_hit_predicate():
    p, q = 487, 977
    cf = continued_fraction(p, q)
    conv, med = convergents_and_mediants(cf, 10_000)
    assert (487, 977) in conv  # the fraction itself is its last convergent
    # the mediant (h_{-1} + h_0)/(k_{-1} + k_0) of 487/977 = [0; 2, 243, 2]: h_0/k_0 = 0/1, h_{-1}/k_{-1} = 1/0 -> 1/1
    assert (1, 1) in med or (1, 1) in conv
    # exact hit predicate against a float evaluation on a tiny example
    N = p * q
    r = int(round(N ** (1 / 3)))
    for a in range(1, 6):
        for b in range(1, 6):
            if a * b > r:
                continue
            u = a * q + b * p
            W = N ** 0.5 / (4 * r * (a * b) ** 0.5)
            approx = u - 2 * (a * b * N) ** 0.5 < W
            assert is_hit(a, b, p, q, r) == approx


def test_construction_finds_known_sliver_hit_and_stern_brocot_bound_holds():
    # b = 50: the E45 run found N = 2464897889 = 36497 * 67537 with the hit (27, 50), b^2 |xi - a/b| = 1.00019 > 1.
    res = construct_sliver_hits(50, 50, max_found=5)
    Ns = {f["N"] for f in res["found"]}
    assert 2464897889 in Ns
    for f in res["found"]:
        p, q, a, b = f["p"], f["q"], f["a"], f["b"]
        assert p < q and a * b <= f["r"] == r_of(p * q) and is_hit(a, b, p, q, f["r"])
        m = abs(a * q - b * p)
        assert m * b >= q                      # outside Fatou's hypothesis ...
        assert m * (b - 1) < q                 # ... but inside the Stern-Brocot parent criterion (Proposition hits (iii))
        assert f["kind"] in ("convergent", "intermediate")


def test_hit_implies_m_squared_below_N_over_r_plus_W_squared():
    # Proposition hits (i), checked exactly on random moduli: m^2 < N/r + W^2  <=>  m^2 * 16 r^2 ab < 16 r ab N + N (as W^2 = N/(16 r^2 ab)).
    from factorlab.gen import make_semiprime

    for i in range(20):
        sp = make_semiprime(32, "rsa", 9, i)
        p, q, N = sp.p, sp.q, sp.N
        r = r_of(N)
        b_max = int((r * q / p) ** 0.5) + 2
        for b in range(1, b_max + 1):
            a0 = (b * p) // q
            for a in (a0, a0 + 1):
                if a < 1 or a * b > r or not is_hit(a, b, p, q, r):
                    continue
                m = abs(a * q - b * p)
                assert 16 * r * r * a * b * m * m < 16 * r * a * b * N + N


def test_classify_hits_has_no_nonsliver_neither():
    # Fatou's theorem: a hit with |xi - a/b| < 1/b^2 is a convergent or intermediate fraction; the classifier must agree.
    from factorlab.gen import make_semiprime

    for i in range(30):
        sp = make_semiprime(30, "rsa", 3, i)
        r = int(round(sp.N ** (1 / 3)))
        res = classify_hits(sp.N, sp.p, sp.q, r)
        assert res["neither_nonsliver"] == []
        assert res["hits"] == res["convergent"] + res["intermediate"] + len(res["neither_sliver"])
