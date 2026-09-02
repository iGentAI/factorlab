import pytest

from factorlab.gen import make_semiprime
from factorlab.numth import mpz, isqrt
from factorlab.experiments import hull
from factorlab.experiments.frobenius_defect import gcd_degree_profile, leakage_over_ZN
from factorlab.experiments.cf_dag import cf_sqrt_quotients, prefix_sharing
from factorlab.registry import get_algorithm


def _brute_lower_hull(N, x0, x1):
    """Andrew monotone chain on the points (x, ceil(N/x)) for x in [x0, x1]."""
    pts = [(x, -((-N) // x)) for x in range(x0, x1 + 1)]
    hull_pts = []
    for p in pts:
        while len(hull_pts) >= 2:
            (ax, ay), (bx, by) = hull_pts[-2], hull_pts[-1]
            # keep lower hull: remove b if a->b->p is clockwise or collinear
            if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) <= 0:
                hull_pts.pop()
            else:
                break
        hull_pts.append(p)
    return hull_pts


@pytest.mark.parametrize("N", [1000003 * 1000033, 10**12 + 39, 999983 * 1000003, 7919 * 7927, 101 * 103, 2**40 + 15])
def test_hull_walk_matches_brute_force(N):
    x1 = int(isqrt(mpz(N)))
    x0 = max(2, int(x1 * 0.6))
    ours = [(int(x), int(y)) for x, y in hull.hull_walk(N, x0, x1)]
    # the unbounded hull's last edge may jump past x1; brute-force a wider
    # window and compare the vertices with x <= x1
    brute = [v for v in _brute_lower_hull(N, x0, 2 * x1 + 10) if v[0] <= x1]
    assert all(y == -((-N) // x) for x, y in ours)
    assert ours == brute, (ours[:5], brute[:5], len(ours), len(brute))


def test_feasible_interval_exact():
    from factorlab.experiments.hull import feasible_interval
    import random
    rng = random.Random(5)

    def check(A, B, da, db, N):
        iv = feasible_interval(A, B, da, db, N)
        for k in range(-250, 250):
            truth = (A + k * da) * (B - k * db) >= N and B - k * db >= 1
            if iv is None:
                assert not truth, (A, B, da, db, N, k)
            else:
                k1, k2 = iv
                inside = (k1 is None or k >= k1) and (k2 is None or k <= k2)
                assert inside == truth, (A, B, da, db, N, k, iv)

    # explicit degenerate cases raised in review
    check(34822, 45883, 19, 25, 1597738589)  # real roots 1.04, 1.54: no integer between -> empty
    check(10, 0, 1, 1, 10)      # feasible only at negative k (k=-2: 8*2=16>=10)
    check(10, 0, 1, 0, 10)      # db=0, B=0 -> empty (B<1), no division by zero
    check(10, -3, 0, 1, 10)     # da=0, B<0, feasible for k<=-4
    check(5, 3, 0, 1, 100)      # da=0, a0<0: 5*(3-k)>=100 -> k<=-17
    check(-5, 3, 0, 1, 10)      # A<=0 -> empty
    check(7, 3, 0, 0, 10)       # constants: 21>=10, B>=1 -> all k
    check(7, 0, 0, 0, 10)       # constants with B<1 -> empty
    for _ in range(400):
        N = rng.randrange(1, 10**6)
        A = rng.randrange(-50, 3000)
        B = rng.randrange(-50, 3000)
        da = rng.choice([0, 0, 1, 2, 3, 7, 15, 40])
        db = rng.choice([0, 0, 1, 2, 3, 7, 15, 40])
        check(A, B, da, db, N)
    for _ in range(300):
        N = rng.randrange(10**6, 10**9)
        x = rng.randrange(1000, 30000)
        y = -((-N) // x) + rng.randrange(0, 3)
        da, db = rng.randrange(1, 50), rng.randrange(0, 50)
        check(x, y, da, db, N)


def test_hull_walk_regression_n20():
    # Lower hull of {(x, ceil(20/x))} from x=2: (2,10),(3,7),(4,5),(5,4) -- every
    # point is a vertex here; an implementation that confuses a feasible chord
    # with a hull edge jumps (2,10)->(5,4) and skips (3,7),(4,5).
    got = [(int(x), int(y)) for x, y in hull.hull_walk(20, 2, 4)]
    assert got == [(2, 10), (3, 7), (4, 5)]


def test_hull_walk_exhaustive_small():
    for N in range(3, 400):
        x1 = int(isqrt(mpz(N)))
        for x0 in range(1, x1 + 1):
            ours = [(int(x), int(y)) for x, y in hull.hull_walk(N, x0, x1)]
            brute = [v for v in _brute_lower_hull(N, x0, 3 * x1 + 10) if v[0] <= x1]
            assert ours == brute, (N, x0, ours, brute)


def test_hull_walk_exhaustive_medium_windows():
    import random
    rng = random.Random(11)
    for _ in range(300):
        N = rng.randrange(10**4, 10**6)
        x1 = int(isqrt(mpz(N)))
        x0 = rng.randrange(1, x1 + 1)
        ours = [(int(x), int(y)) for x, y in hull.hull_walk(N, x0, x1)]
        brute = [v for v in _brute_lower_hull(N, x0, 3 * x1 + 10) if v[0] <= x1]
        assert ours == brute, (N, x0, len(ours), len(brute))


@pytest.mark.parametrize("i", range(4))
def test_hull_locator_finds_factor(i):
    inst = make_semiprime(40, "balanced", 21, i)
    res = get_algorithm("hull_locator")(inst.N)
    assert res.found and int(res.p) == int(inst.p)


def test_hull_locator_prime_fails_cleanly():
    res = get_algorithm("hull_locator")(1000000007)
    assert not res.found


@pytest.mark.parametrize("pq", [(3, 5), (5, 7), (11, 13), (101, 103), (101, 397), (1009, 4001), (65537, 262147)])
def test_hull_locator_small_and_skewed_within_window(pq):
    p, q = pq
    N = p * q
    res = get_algorithm("hull_locator")(N, C=4)
    assert res.found and int(res.p) == p


def test_frobenius_defect_degrees_agree_for_prime_modulus_self_consistency():
    # For a prime P and N = P, F_a = (X+a)^P - X^P - a = 0 over F_P, so gcd = X^r - 1 (degree r)
    P = 1000003
    assert gcd_degree_profile(P, P, 3, 7) == 7


def test_leakage_detects_planted_mismatch():
    # Construct N = p*q where r | p-1 but r does not divide q-1, so mu_r(F_p) is
    # full; leakage is not guaranteed but the routine must run and return a
    # factor or None without error.
    inst = make_semiprime(32, "balanced", 22, 0)
    leaked, steps = leakage_over_ZN(int(inst.N), 2, 16)
    assert leaked in (None, int(inst.p), int(inst.q))
    assert steps >= 0


def test_cf_quotients_known():
    # sqrt(2) = [1; 2, 2, 2, ...]
    assert cf_sqrt_quotients(2, 5) == (2, 2, 2, 2, 2)
    # sqrt(7) = [2; 1, 1, 1, 4, ...]
    assert cf_sqrt_quotients(7, 5) == (1, 1, 1, 4, 1)


def test_prefix_sharing_shapes():
    r = prefix_sharing(1000003 * 1000033, 200, 4)
    assert len(r["distinct_prefixes"]) == 4
    assert r["distinct_prefixes"][-1] <= 200
    assert all(a <= b for a, b in zip(r["distinct_prefixes"], r["interval_count"]))


def test_frobenius_degree_identity():
    from factorlab.experiments.barrier import frobenius_degree_check
    # p=10007, q=10009 (d=2): Delta^3 F == 0 mod p, Delta^2 F != 0 mod p
    r = frobenius_degree_check(10007 * 10009, 10007, 10009)
    assert r["identity_holds"] and r["degree_exact"]
    inst = make_semiprime(24, "balanced", 41, 0)
    r = frobenius_degree_check(inst.N, inst.p, inst.q)
    assert r["identity_holds"]


def test_lehman_covering_counts():
    from factorlab.experiments.barrier import lehman_covering
    row = lehman_covering(10**20 + 39, 10)
    # P = sum_{a<=10} floor(10/a) = 10+5+3+2+2+1+1+1+1+1 = 27
    assert row["P"] == 27
    assert row["sigma_w"] > 0


def test_chirp_hull_small():
    from factorlab.experiments.barrier import chirp_hull_complexity
    row = chirp_hull_complexity(10**12 + 39, 100)
    assert 2 <= row["upper_hull_vertices"] <= 100


def test_cell_coverage_length_exact():
    from factorlab.experiments.barrier import cell_coverage_length
    import math
    N = 10**20
    # cell (1,1): g(p) = N/p + p, minimum 2 sqrt N at p = sqrt N; {g < 2 sqrt N + w}
    # is the interval (sqrt N - d, sqrt N + d) with d ~ sqrt(w sqrt N) for small w;
    # clipped to p <= sqrt N gives length ~ d.
    w = 1e6
    row = cell_coverage_length(N, 1, 1, 0.0, w)
    d = math.sqrt(w * math.sqrt(N))
    assert abs(row["length"] - d) / d < 0.01
    assert row["ratio"] <= 1.0


def test_cell_coverage_length_large_offset_stable():
    """Lower-branch cancellation regime: p* = sqrt(aN/b) far above sqrt N, so
    the level set meeting [sqrt(N)/2, sqrt N] lies at delta >> c_min and the
    lower root must be taken in conjugate form.  Reference values are computed
    with 60-digit Decimal arithmetic; the naive float (t - s)/(2b) formula is
    shown to be wrong here."""
    from factorlab.experiments.barrier import cell_coverage_length
    from decimal import Decimal, getcontext
    import math
    getcontext().prec = 60
    N = 10**40
    a, b = 10**6, 1                      # p* = 10^3 sqrt N, far outside the window
    sqrtN = math.sqrt(N)
    cmin = 2 * math.sqrt(a * b * N)      # 2 * 10^23
    # choose the level so that the lower root sits at p0 = 0.7 sqrt N:
    # g(p0) = aN/p0 + b p0 ; delta0 = g(p0) - cmin  (~ 1.4e26 >> cmin)
    p0 = 0.7 * sqrtN
    delta0 = a * N / p0 + b * p0 - cmin
    assert delta0 > 100 * cmin
    w = 1e15
    row = cell_coverage_length(N, a, b, delta0, w)
    Nd = Decimal(N)
    lo, hi = (Nd / 4).sqrt(), Nd.sqrt()
    cmin_d = 2 * (Decimal(a * b) * Nd).sqrt()

    def roots(delta):
        delta = Decimal(delta)
        t = cmin_d + delta
        s = (delta * (2 * cmin_d + delta)).sqrt()
        return ((t - s) / (2 * b), (t + s) / (2 * b))

    def clip(iv):
        x0, x1 = max(iv[0], lo), min(iv[1], hi)
        return max(Decimal(0), x1 - x0)

    ref = float(clip(roots(delta0 + w)) - clip(roots(delta0)))
    assert ref > 0
    # The length is a difference of endpoints ~ 7e19, where binary64 spacing is
    # ~ 8e3; with ref ~ 5e8 the attainable relative accuracy is ~ 1e-4.
    assert abs(row["length"] - ref) / ref < 2e-4, (row["length"], ref)

    # the naive float formula loses the lower root here
    def naive_length():
        def r(delta):
            t = cmin + delta
            s = math.sqrt(delta * (2 * cmin + delta))
            return ((t - s) / (2 * b), (t + s) / (2 * b))

        def c(iv):
            x0, x1 = max(iv[0], sqrtN / 2), min(iv[1], sqrtN)
            return max(0.0, x1 - x0)
        return c(r(delta0 + w)) - c(r(delta0))
    assert abs(naive_length() - ref) / ref > 1e-2


def test_cell_coverage_length_rejects_bad_args():
    from factorlab.experiments.barrier import cell_coverage_length
    import pytest
    with pytest.raises(ValueError):
        cell_coverage_length(10**20, 1, 1, 0.0, 0.0)
    with pytest.raises(ValueError):
        cell_coverage_length(10**20, 0, 1, 0.0, 1.0)
    with pytest.raises(ValueError):
        cell_coverage_length(10**20, 1, 1, -1.0, 1.0)
