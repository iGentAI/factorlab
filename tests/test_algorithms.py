import pytest

from factorlab import algorithms  # noqa: F401  (registers)
from factorlab.registry import ALGORITHMS, get_algorithm
from factorlab.gen import make_semiprime
from factorlab.numth import mpz


def _check(res, inst):
    assert res.found, f"{res.algorithm} failed on {inst}"
    assert {int(res.p), int(res.q)} == {int(inst.p), int(inst.q)}
    assert res.primary_work >= 0


@pytest.mark.parametrize("family", ["balanced", "rsa", "skew", "close"])
def test_trial_division(family):
    inst = make_semiprime(30, family, 1, 0)
    _check(get_algorithm("trial_division")(inst.N), inst)


def test_fermat_close():
    inst = make_semiprime(80, "close", 1, 0, gap_bits=20)
    _check(get_algorithm("fermat")(inst.N), inst)


@pytest.mark.parametrize("i", range(4))
def test_lehman(i):
    inst = make_semiprime(40, "balanced", 3, i)
    _check(get_algorithm("lehman")(inst.N), inst)
    inst = make_semiprime(40, "skew", 3, i)
    _check(get_algorithm("lehman")(inst.N), inst)


@pytest.mark.parametrize("i", range(4))
def test_hart_olf(i):
    inst = make_semiprime(44, "balanced", 4, i)
    _check(get_algorithm("hart_olf")(inst.N), inst)


@pytest.mark.parametrize("i", range(6))
def test_squfof(i):
    inst = make_semiprime(56, "balanced", 5, i)
    _check(get_algorithm("squfof")(inst.N), inst)


@pytest.mark.parametrize("i", range(4))
def test_pollard_rho(i):
    inst = make_semiprime(64, "balanced", 6, i)
    _check(get_algorithm("pollard_rho")(inst.N), inst)


def test_pollard_pm1_smooth():
    inst = make_semiprime(100, "smooth_pm1", 7, 0, B=2000)
    _check(get_algorithm("pollard_pm1")(inst.N, B1=2000), inst)


def test_williams_pp1_known():
    # p = 2*3*5*7*11*13 - 1 = 30029 is prime and p+1 is 13-smooth
    p = 30029
    q = 1000003
    from factorlab.numth import is_prime, jacobi
    assert is_prime(p) and is_prime(q)
    # Williams' method finds p when (P0^2 - 4 | p) = -1 (order of the Lucas
    # sequence then divides p+1).  Choose such a P0 explicitly -- the test
    # knows p, the algorithm does not.
    P0 = next(P for P in range(3, 200) if jacobi(P * P - 4, p) == -1)
    res = get_algorithm("williams_pp1")(p * q, B1=50, P0=P0)
    assert res.found and int(res.p) == p, (P0, res)
    # and a P0 with symbol +1 whose order divides p-1 = 4 * 7507 must fail at B1=50
    P1 = next(P for P in range(3, 200) if jacobi(P * P - 4, p) == 1)
    res = get_algorithm("williams_pp1")(p * q, B1=50, P0=P1)
    assert not res.found


@pytest.mark.parametrize("i", range(3))
def test_pollard_strassen(i):
    inst = make_semiprime(48, "balanced", 8, i)
    _check(get_algorithm("pollard_strassen")(inst.N), inst)
    inst = make_semiprime(48, "skew", 8, i)
    _check(get_algorithm("pollard_strassen")(inst.N), inst)


@pytest.mark.parametrize("i", range(4))
def test_bsgs_sum(i):
    inst = make_semiprime(48, "balanced", 9, i)
    _check(get_algorithm("bsgs_sum")(inst.N), inst)


def test_registry_has_all():
    for name in ["trial_division", "fermat", "lehman", "hart_olf", "squfof", "pollard_rho",
                 "pollard_pm1", "williams_pp1", "pollard_strassen", "bsgs_sum", "ecm", "fixed_list_ecm"]:
        assert name in ALGORITHMS


def test_fixed_list_stage2_blocked_matches_single_block():
    from factorlab.algorithms.fixed_list_ecm import stage2_bsgs
    from factorlab.algorithms.ecm import suyama_curve, ladder, stage1_exponents
    from factorlab.result import Work
    from factorlab.numth import gcd
    p, q, sigma = 262147, 262187, 12
    N, B1, B2 = mpz(p * q), 64, 4096
    curve, g = suyama_curve(sigma, N)
    assert curve is not None
    a24, X, Z = curve
    w = Work()
    for pe in stage1_exponents(B1):
        X, Z = ladder(pe, X, Z, a24, N, w)
    assert gcd(Z, N) == 1
    g_full, _ = stage2_bsgs(X, Z, a24, N, B2, Work(), poly_block=4096)
    g_block, _ = stage2_bsgs(X, Z, a24, N, B2, Work(), poly_block=3)
    assert 1 < g_full < N and g_block == g_full


def test_fixed_list_stage2_shared_giant_identity_regression():
    """Residual orders 24 and 72 with m = 9 both first divide giant 8m = 72.
    The two-sided giant identity must be filtered; the order-24 collision remains one-sided."""
    import numpy as np
    from factorlab.experiments.hitting_sets import residual_order_labels
    p, q = 1097, 7019
    B1, B2, sigma = 5, 80, 6
    labels = residual_order_labels(sigma, np.array([p, q], dtype=np.int64), B1, B2)
    assert labels.tolist() == [24, 72]
    res = get_algorithm("fixed_list_ecm")(p * q, B1=B1, B2=B2, sigma0=sigma, max_curves=1)
    assert res.found and {int(res.p), int(res.q)} == {p, q}, res.meta
    assert res.meta["stage"] == "2"


def test_fixed_list_scalability_probe_small():
    from factorlab.experiments.fixed_list_check import scalability_probe
    row = scalability_probe(32, 0, u=4.0, C=2.0, max_curves=50)
    assert row["found"] and row["nbits"] == 32 and row["u"] == 4.0
    assert row["B1"] >= 5 and row["B2"] >= row["B1"]
    assert row["stage2_degree"] == int(row["B2"] ** 0.5) + 1 or row["stage2_degree"] == int(row["B2"] ** 0.5)
    assert row["curve"] >= 1 and row["wall"] >= 0 and row["peak_rss_kb"] > 0


def test_result_rejects_wrong_factor():
    from factorlab.result import FactorResult, Work
    with pytest.raises(AssertionError):
        FactorResult("x", mpz(15), True, mpz(4), mpz(4), 0.0, Work(), "candidate")


@pytest.mark.parametrize("i", range(3))
def test_fixed_list_ecm_rsa(i):
    inst = make_semiprime(40, "rsa", 21, i)
    res = get_algorithm("fixed_list_ecm")(inst.N)
    _check(res, inst)
    assert res.meta["stage"] in ("den", "1", "2") and res.meta["curve"] >= 1


def test_fixed_list_ecm_certificate_pairs():
    """E20 certificate: sigma = 6..44 at B1 = 64, B2 = 4096 separate all primes in [2^18, 2^19);
    the algorithm on N = pq must succeed within 39 curves and no later than the simulated
    separation index of the pair (its gcds see finer one-sided relations than the collapsed bit)."""
    import numpy as np
    from factorlab.experiments.ecm_hitting import primes_in_range
    from factorlab.experiments.fixed_list_check import simulated_separation_index
    algo = get_algorithm("fixed_list_ecm")
    pairs = [(407699, 442447)]  # the last pair to separate in E20 at x = 2^18 (simulated index 39)
    primes = primes_in_range(1 << 18, 1 << 19)
    rng = np.random.default_rng(4)
    while len(pairs) < 20:
        i, j = (int(v) for v in rng.integers(0, primes.size, 2))
        if i != j:
            pairs.append((int(primes[min(i, j)]), int(primes[max(i, j)])))
    for p, q in pairs:
        res = algo(p * q, B1=64, B2=4096, max_curves=60)
        assert res.found and {int(res.p), int(res.q)} == {p, q}, (p, q, res.meta)
        sim = simulated_separation_index(p, q, 64, 4096)
        assert sim is not None and res.meta["curve"] <= sim <= 39, (p, q, res.meta, sim)


def test_fixed_list_stage2_matches_simulation():
    """stage2_bsgs on N = pq against the simulated one-large-prime bits of p and q."""
    import numpy as np
    from factorlab.algorithms.fixed_list_ecm import stage2_bsgs
    from factorlab.algorithms.ecm import suyama_curve, ladder, stage1_exponents
    from factorlab.experiments.ecm_hitting import primes_in_range
    from factorlab.experiments.hitting_sets import ecm_success
    from factorlab.numth import gcd
    from factorlab.result import Work
    ps_all = primes_in_range(1 << 18, (1 << 18) + 400)
    p, q = int(ps_all[0]), int(ps_all[3])
    N = mpz(p * q)
    B1, B2 = 64, 4096
    ps = np.array([p, q], dtype=np.int64)
    one_sided = 0
    for sigma in range(6, 46):
        s1, s2 = ecm_success(sigma, ps, B1, B2)
        if s1.any():
            continue  # a stage-1 exposure: stage 2 is not reached for that prime
        curve, g = suyama_curve(sigma, N)
        assert curve is not None
        a24, X, Z = curve
        w = Work()
        for pe in stage1_exponents(B1):
            X, Z = ladder(pe, X, Z, a24, N, w)
        assert gcd(Z, N) == 1
        g, detail = stage2_bsgs(X, Z, a24, N, B2, w)
        if bool(s2[0]) != bool(s2[1]):
            # differing collapsed bits guarantee a one-sided cell; the BSGS predicate (residual
            # order <= B2, including small residual powers) may expose the *other* prime first
            assert int(g) in (p, q), (sigma, int(g), s2.tolist())
            one_sided += 1
        else:
            # both bits equal: a factor may still appear (a composite residual order <= B2, or
            # different cells), but never a wrong value
            assert int(g) in (1, p, q, int(N)), (sigma, int(g))
    assert one_sided >= 3
