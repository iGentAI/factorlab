from fractions import Fraction

import math
import numpy as np
import pytest

from factorlab.experiments.separable import (
    DifferenceConstraints, berlekamp_massey, greedy_separable_alignment, chirp_linear_complexity,
    window_preimage, union_measure,
)


def _brute_feasible(edges):
    """Bellman-Ford negative-cycle check for constraints x_v - x_u <= c."""
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    dist = {n: Fraction(0) for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for u, v, c in edges:
            if dist[u] + c < dist[v]:
                dist[v] = dist[u] + c
                changed = True
        if not changed:
            return True
    for u, v, c in edges:
        if dist[u] + c < dist[v]:
            return False
    return True


def _check_potentials(sysd, accepted):
    for u, v, c in accepted:
        assert sysd.pi[v] <= sysd.pi[u] + c, (u, v, c, sysd.pi)


def test_difference_constraints_matches_bellman_ford():
    import random
    rng = random.Random(3)
    for trial in range(300):
        n = rng.randrange(2, 8)
        sysd = DifferenceConstraints()
        accepted = []
        for _ in range(rng.randrange(1, 20)):
            u, v = rng.randrange(n), rng.randrange(n)
            c = Fraction(rng.randrange(-6, 9))
            ok = sysd.try_add(u, v, c)
            brute = _brute_feasible(accepted + [(u, v, c)])
            assert ok == brute, (trial, accepted, (u, v, c))
            if ok and u != v:
                accepted.append((u, v, c))
            _check_potentials(sysd, accepted)


def test_difference_constraints_smaller_decrease_first_then_larger():
    """y is reachable from v by a direct edge (small decrease) and via x (large
    decrease); z hangs off y.  A min-decrease-first relaxation settles y and z
    with the small decrease and leaves z stale."""
    sysd = DifferenceConstraints()
    u, v, x, y, z = "u", "v", "x", "y", "z"
    pre = [(v, y, Fraction(5)), (v, x, Fraction(0)), (x, y, Fraction(0)), (y, z, Fraction(0))]
    for a, b, c in pre:
        assert sysd.try_add(a, b, c)
    assert sysd.try_add(u, v, Fraction(-10))
    accepted = pre + [(u, v, Fraction(-10))]
    _check_potentials(sysd, accepted)
    assert sysd.pi[y] == sysd.pi[v] and sysd.pi[z] <= sysd.pi[y]
    # now a constraint closing a negative cycle must be rejected and leave potentials intact
    snap = dict(sysd.pi)
    assert not sysd.try_add(z, u, Fraction(5))   # u - z <= 5 but u = 0, z = -10 -> u - z = 10 > 5 -> forces cycle u->v->x->y->z->u of weight -10+0+0+0+5 < 0
    assert sysd.pi == snap


def test_difference_constraints_self_loops():
    sysd = DifferenceConstraints()
    assert sysd.try_add("a", "b", Fraction(3))
    snap = dict(sysd.pi)
    assert sysd.try_add("a", "a", Fraction(0))
    assert sysd.try_add("a", "a", Fraction(2))
    assert not sysd.try_add("a", "a", Fraction(-1))
    assert sysd.pi == snap
    # no self-loop edges stored
    assert all(v != u for u, lst in sysd.adj.items() for v, _ in lst)


def test_berlekamp_massey_known():
    p = 1000003
    f = [1, 1]
    for _ in range(60):
        f.append((f[-1] + f[-2]) % p)
    assert berlekamp_massey(f, p) == 2
    g = [pow(5, i, p) for i in range(40)]
    assert berlekamp_massey(g, p) == 1
    two = [(pow(3, k, p) + pow(5, k, p)) % p for k in range(1, 80)]
    assert berlekamp_massey(two, p) == 2


def test_window_preimage_and_union():
    N = 10**20
    a, b = 1, 1
    p_hi = math.sqrt(N)
    p_lo = p_hi / 2
    w = 1e6
    iv = window_preimage(N, a, b, 0.0, w, p_lo, p_hi)
    # single interval ending at sqrt N of length ~ sqrt(w sqrt N)
    assert len(iv) == 1
    d = math.sqrt(w * math.sqrt(N))
    assert abs((iv[0][1] - iv[0][0]) - d) / d < 0.01
    # window strictly above the minimum gives two pieces (one may be clipped)
    iv2 = window_preimage(N, a, b, 1e6, w, 0.0, 2 * p_hi)
    assert len(iv2) == 2
    assert union_measure([(0, 1), (0.5, 2), (3, 4)]) == 3.0


def test_greedy_alignment_small_is_sane():
    N = 1000003 * 1000033
    row = greedy_separable_alignment(N, 16, K=1.0, C=4.0)
    assert 1 <= row["kept"] <= row["cells_considered"]
    assert row["K22_free"]
    assert isinstance(row["is_forest"], bool)
    if row["is_forest"]:
        assert row["kept"] <= row["forest_bound"]
    assert row["max_offset_deviation_in_w"] <= 1.0 + 1e-9
    assert 0.0 <= row["sep_window_coverage_share"] <= 1.0 + 1e-9
    assert 0.0 <= row["lehman_window_coverage_share_all"] <= 1.0 + 1e-9


def test_chirp_linear_complexity_runs():
    row = chirp_linear_complexity(1000003 * 1000033, 64, 1000003)
    assert 1 <= row["linear_complexity"] <= 64
    assert row["control_lfsr_order2_complexity"] == 2


def test_null_suite_model_distribution():
    from factorlab.experiments.null_suite import model_distribution
    # l = 7, n = 2 (QR mod 7: 3^2 = 2): support (7+1)/2 = 4 values; two of them
    # (t = +-3) have half weight 1/6, the rest 2/6.
    m = model_distribution(7, 2, "sum")
    assert len(m) == 4 and abs(sum(m.values()) - 1) < 1e-12
    assert sorted(round(v, 6) for v in m.values()) == [round(1 / 6, 6), round(1 / 6, 6), round(1 / 3, 6), round(1 / 3, 6)]
    # n = 3 is a non-residue mod 7: support (7-1)/2 = 3, uniform
    m = model_distribution(7, 3, "sum")
    assert len(m) == 3 and all(abs(v - 1 / 3) < 1e-12 for v in m.values())


def test_null_suite_small_run_passes():
    from factorlab.experiments.null_suite import conditional_residue_test
    res = conditional_residue_test(nbits=40, count=3000, moduli=(3, 5, 7, 11), family="balanced", seed=5)
    assert res["pass"], res["per_modulus"]


def test_degenerate_forms_lehman_cell():
    from factorlab.experiments.degenerate_forms import Form, hoelder_prediction
    N = 10**20
    f = Form({(0, 1): 1, (1, 0): 1}, N)  # g = N/p + p
    hi = math.sqrt(N)
    mins = f.local_minima(hi / 2, hi * 1.0000001)
    assert mins and abs(float(mins[0]) - hi) / hi < 1e-12
    assert abs(float(f.deriv(hi, 2)) - 2 / hi) / (2 / hi) < 1e-12
    assert abs(f.cancellation_ratio(hi, 2) - 1.0) < 1e-12
    w = 1e6
    cov = f.coverage(mins[0], w, hi / 2, 2 * hi)
    exact = math.sqrt(4 * hi * w + w * w)  # roots of p^2 - (2s + w) p + N
    assert abs(cov - exact) / exact < 1e-9
    pred = hoelder_prediction(f, mins[0], w)
    assert pred["available"] and pred["order_used"] == 2
    assert abs(pred["prediction"] - 2 * math.sqrt(w * hi)) / (2 * math.sqrt(w * hi)) < 1e-9


def test_degenerate_forms_degree2_has_no_degenerate_critical_points():
    """Homogeneous degree-2 forms: g'' = 8c at every critical point (Lemma C consequence)."""
    from factorlab.experiments.degenerate_forms import Form
    import random
    N = 10**20
    hi = math.sqrt(N)
    rng = random.Random(2)
    for _ in range(50):
        a, c = rng.randrange(1, 10), rng.randrange(1, 10)
        f = Form({(0, 2): a, (2, 0): c}, N)
        cps = f.all_critical_points(hi / 4, 4 * hi)
        assert cps
        for p in cps:
            assert abs(float(f.deriv(p, 2)) - 8 * c) / (8 * c) < 1e-9


def test_degenerate_forms_minima_exclude_maxima():
    from factorlab.experiments.degenerate_forms import Form
    N = 10**20
    hi = math.sqrt(N)
    fmin = Form({(0, 1): 1, (1, 0): 1}, N)
    fmax = Form({(0, 1): -1, (1, 0): -1}, N)
    assert len(fmin.local_minima(hi / 2, 2 * hi)) == 1
    assert len(fmax.all_critical_points(hi / 2, 2 * hi)) == 1
    assert fmax.local_minima(hi / 2, 2 * hi) == []


def test_log_coordinate_identities():
    """h(t) = g(s e^t): h' = p g', h'' = p^2 g'' + p g'.  For an antisymmetric
    form h is odd, so h''(0) = 0, which means s^2 g''(s) = -s g'(s) -- not
    g''(s) = 0.  At a critical point, however, g'' = h''/p^2 exactly."""
    from factorlab.experiments.degenerate_forms import Form
    N = 10**20
    s = math.sqrt(N)
    f = Form({(0, 3): 1, (3, 0): -1, (1, 2): 2, (2, 1): -2}, N)  # antisymmetric
    assert abs(float(f.h_deriv(0, 2))) <= 1e-40 * abs(float(f.h_deriv(0, 1)))
    g1, g2 = float(f.deriv(s, 1)), float(f.deriv(s, 2))
    assert abs(s * s * g2 + s * g1) <= 1e-30 * abs(s * g1)
    assert g1 == pytest.approx(-10 * s * s, rel=1e-12) and g2 == pytest.approx(10 * s, rel=1e-12)
    assert f.cancellation_ratio(s, 2) == pytest.approx(5 / 11, rel=1e-12)
    # identity at a critical point, on a form that has one
    g = Form({(0, 1): 1, (1, 0): 2}, N)  # Lehman (1, 2)
    p = g.local_minima(s / 2, s)[0]
    t = math.log(float(p) / s)
    assert float(g.h_deriv(t, 2)) == pytest.approx(float(p) ** 2 * float(g.deriv(p, 2)), rel=1e-12)


def test_perfect_cube_cell_mechanism():
    """Top-degree part 2 (q - p)^3 with lower terms: h ~ -16 s^3 t^3 - 30 s^2 t^2
    - 2 s^2 t + ..., minimum at t* ~ -sqrt(2 s^2 / (48 s^3)) = -0.204 / sqrt s,
    h''(t*) ~ 19.6 s^{5/2} against a natural scale ~ 48 s^3: cancellation ratio
    of order s^{-1/2}; coverage far below the Lehman cell."""
    from factorlab.experiments.degenerate_forms import Form
    N = 1152921504606846999
    s = math.sqrt(N)
    co = {(0, 1): 8, (0, 2): -8, (0, 3): 2, (1, 0): 4, (1, 2): -6, (2, 0): -7, (2, 1): 6, (3, 0): -2}
    f = Form(co, N)
    assert f.top_degree_perfect_power() == (2, 3)
    lo, hi2 = s / 2, s * 1.01
    mins = f.local_minima(lo, hi2)
    assert len(mins) == 1
    p = mins[0]
    t = math.log(float(p) / s)
    assert t == pytest.approx(-math.sqrt(2 / (48 * s)), rel=0.05)
    eta2 = f.cancellation_ratio(p, 2)
    assert 1e-6 < eta2 < 1e-4
    assert float(f.deriv(p, 2)) == pytest.approx(float(f.h_deriv(t, 2)) / float(p) ** 2, rel=1e-9)
    w = 1e6
    cov = f.coverage(p, w, lo, hi2)
    lehman = Form({(0, 1): 1, (1, 0): 1}, N).coverage(s, w, lo, hi2)
    assert 0 < cov < 1e-6 * lehman
    # within the well: Hoelder order-2 prediction matches the exact coverage
    from factorlab.experiments.degenerate_forms import hoelder_prediction
    pred = hoelder_prediction(f, p, w)
    assert pred["available"] and pred["order_used"] == 2
    assert cov == pytest.approx(pred["prediction"], rel=0.05)
    assert f.well_depth(p, lo, hi2) > w


def test_perfect_power_detector_requires_k_at_least_3():
    from factorlab.experiments.degenerate_forms import Form
    N = 10**20
    # (q - p)^2 = q^2 - 2pq + p^2 : quadratic top, curvature at natural scale -> not flagged
    assert Form({(0, 2): 1, (2, 0): 1, (1, 1): -2, (1, 0): 3}, N).top_degree_perfect_power() is None
    # q - p : linear top -> not flagged
    assert Form({(0, 1): 1, (1, 0): -1}, N).top_degree_perfect_power() is None
    # (q - p)^3 and -(q - p)^4 are flagged with the right (c, k)
    assert Form({(0, 3): 1, (1, 2): -3, (2, 1): 3, (3, 0): -1, (1, 0): 5}, N).top_degree_perfect_power() == (1, 3)
    assert Form({(0, 4): -1, (1, 3): 4, (3, 1): 4, (4, 0): -1, (2, 2): -6}, N).top_degree_perfect_power() == (-1, 4)
    # a cubic top that is not a perfect cube
    assert Form({(0, 3): 1, (1, 2): -3, (2, 1): 2, (3, 0): -1}, N).top_degree_perfect_power() is None


def test_high_precision_resolves_cancelling_monomials():
    """Near-antisymmetric degree-3 cell from the E3 survey: monomials ~1e27,
    g ~ 1e19.  Coverage of a width-1e6 window must be finite and positive and
    the component endpoint must lie on the level set to high precision."""
    from factorlab.experiments.degenerate_forms import Form
    from gmpy2 import mpfr
    N = 1152921504606846999
    hi = math.sqrt(N)
    co = {(0, 1): 8, (0, 2): -8, (0, 3): 2, (1, 0): 4, (1, 2): -6, (2, 0): -7, (2, 1): 6, (3, 0): -2}
    f = Form(co, N)
    lo, hi2 = hi / 2, hi * 1.01
    mins = f.local_minima(lo, hi2)
    assert mins
    p = mins[0]
    assert float(f.term_scale(p, 0)) > 1e6 * abs(float(f.value(p)))  # massive cancellation
    w = 1e6
    cov = f.coverage(p, w, lo, hi2)
    assert 0 < cov < hi
    level = f.value(p) + mpfr(w)
    right_cps = [c for c in f.all_critical_points(lo, hi2) if c > p]
    a, b = p, (right_cps[0] if right_cps else mpfr(hi2))
    if f.value(b) >= level:
        for _ in range(200):
            mid = (a + b) / 2
            if f.value(mid) < level:
                a = mid
            else:
                b = mid
        assert abs(float(f.value(a) - level)) <= 1e-30 * float(f.term_scale(a, 0))


def test_degenerate_forms_coverage_does_not_skip_excursions():
    """A non-monotone g with two minima separated by a hump higher than the
    window: the component of the first minimum must stop at the hump even
    though a second inside region exists further right.

    Construction: with N = 1 the form u = q^2 + p^2 - c (p + q) restricts to
    g(p) = p^{-2} + p^2 - c (p + p^{-1}) = x^2 - c x - 2 with x = p + 1/p >= 2,
    symmetric under p -> 1/p.  The vertex x = c/2 exceeds 2 iff c > 4; for
    c = 5 the minima are the roots of p + 1/p = 5/2, i.e. p = 1/2 and p = 2,
    with g = -33/4 there, and p = 1 (x = 2) is the intervening local maximum
    with g(1) = -8.  Window width 0.1 at the left minimum (level -8.15) must
    not reach the right well."""
    from factorlab.experiments.degenerate_forms import Form
    form = Form({(0, 2): 1, (2, 0): 1, (1, 0): -5, (0, 1): -5}, 1)
    lo, hi = 0.2, 5.0
    mins = form.local_minima(lo, hi)
    assert len(mins) == 2
    pstar = mins[0]
    assert abs(float(pstar) - 0.5) < 1e-12 and abs(float(mins[1]) - 2.0) < 1e-12
    assert abs(float(form.value(pstar)) + 33 / 4) < 1e-12 and abs(float(form.value(1.0)) + 8) < 1e-12
    # hump at p = 1 lies above the window level; the right well lies below it
    w = 0.1
    level = float(form.value(pstar)) + w
    assert float(form.value(1.0)) > level and float(form.value(mins[1])) < level
    cov = form.coverage(pstar, w, lo, hi)
    grid = np.linspace(lo, hi, 2_000_001)
    gv = 1 / grid ** 2 + grid ** 2 - 5 * (grid + 1 / grid)
    i0 = int(np.searchsorted(grid, float(pstar)))
    j = i0
    while j + 1 < len(grid) and gv[j + 1] < level:
        j += 1
    i = i0
    while i - 1 >= 0 and gv[i - 1] < level:
        i -= 1
    brute = grid[j] - grid[i]
    assert abs(cov - brute) <= 2 * (grid[1] - grid[0]) + 1e-9 * brute
    # and the component indeed stops before the hump at p = 1
    assert float(pstar) + cov / 2 < 1.0 and cov < 1.0 - float(pstar)
    assert form.well_depth(pstar, lo, hi) == pytest.approx(0.25, rel=1e-9)


def test_hoelder_selector_orders():
    """(i) small positive g2 with large negative g4: order 2 must still be
    usable when its own term dominates at h_2, else unavailable -- never an
    infinite 'order 4'.  (ii) a pure quartic well (g2 = 0, g4 > 0) selects
    order 4 with prediction 2 (24 w / g4)^{1/4}."""
    from factorlab.experiments.degenerate_forms import Form, hoelder_prediction
    from gmpy2 import mpfr

    class Poly(Form):
        """g(p) = sum c_k (p - 1)^k, for direct control of derivatives at p* = 1."""
        def __init__(self, cs):
            self.cs = cs
            self.coeffs = {"dummy": 1}
            self.N = mpfr(1)
            self._Npow = {}

        def deriv(self, p, order):
            p = mpfr(p)
            tot = mpfr(0)
            for k, c in enumerate(self.cs):
                if k >= order:
                    ff = math.factorial(k) // math.factorial(k - order)
                    tot += mpfr(c) * ff * (p - 1) ** (k - order)
            return tot

    # (i) g = 1e-6 (p-1)^2 - 10 (p-1)^4 : at small w the quadratic dominates
    f = Poly([0, 0, 1e-6, 0, -10])
    pred = hoelder_prediction(f, 1.0, 1e-14)
    assert pred["available"] and pred["order_used"] == 2
    h = math.sqrt(2 * 1e-14 / 2e-6)
    assert abs(pred["prediction"] - 2 * h) / (2 * h) < 1e-12
    # at larger w the negative quartic competes: must be reported unavailable, not order 4
    pred = hoelder_prediction(f, 1.0, 1e-3)
    assert not pred["available"] and pred["order_used"] is None and pred["prediction"] is None
    # (ii) pure quartic well g = 5 (p-1)^4
    f = Poly([0, 0, 0, 0, 5])
    pred = hoelder_prediction(f, 1.0, 1e-2)
    assert pred["available"] and pred["order_used"] == 4
    g4 = 5 * 24
    assert abs(pred["prediction"] - 2 * (24 * 1e-2 / g4) ** 0.25) < 1e-12
    # (iii) quadratic plus positive quartic: order 2 at small w, order 4 never preferred while 2 dominates
    f = Poly([0, 0, 1.0, 0, 1.0])
    pred = hoelder_prediction(f, 1.0, 1e-6)
    assert pred["order_used"] == 2
