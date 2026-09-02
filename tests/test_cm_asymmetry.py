"""Tests for E44 (CM asymmetry)."""
from fractions import Fraction

from sympy import primerange

from factorlab.experiments.cm_asymmetry import (
    _summary,
    cm_asymmetry_experiment,
    derive_verdict,
    is_norm_from_gaussian_integers,
    is_theta_smooth,
    order_brute,
    order_x3_minus_x,
)


def test_point_count_matches_brute_force_and_supersingular_law():
    for p in primerange(5, 200):
        n = order_x3_minus_x(p)
        assert n == order_brute(p)
        if p % 4 == 3:
            assert n == p + 1  # supersingular at inert primes of Z[i]
        else:
            assert n % 8 == 0 and is_norm_from_gaussian_integers(n)  # Z/2 x Z/4 torsion; order is a norm


def test_smoothness_threshold_is_exact_at_integer_boundaries():
    third = Fraction(1, 3)
    # 11^3 = 1331: an order with largest prime factor 11 is exactly p^(1/3)-smooth at p = 1331 ...
    assert is_theta_smooth(1331, 11, third) is True
    assert _summary([(1331, 11)], third)["p_smooth"] == 1.0
    # ... and not at p = 1330 (11^3 > 1330), nor with a larger prime at p = 1331.
    assert is_theta_smooth(1330, 11, third) is False
    assert is_theta_smooth(1331, 13, third) is False
    # p = 1300: p^(1/3) = 10.91; a rounded threshold (11) would wrongly accept largest factor 11.
    assert _summary([(1300, 11)], third)["p_smooth"] == 0.0
    assert _summary([(1300, 10)], third)["p_smooth"] == 1.0
    # a non-unit-numerator rational: theta = 2/3 at p = 1331 gives threshold 121 exactly.
    assert is_theta_smooth(1331, 121, Fraction(2, 3)) is True   # P^+(121) = 11 <= 1331^(2/3) = 121: smooth
    assert is_theta_smooth(1331, 127, Fraction(2, 3)) is False  # 127 > 121
    assert is_theta_smooth(1331, 113, Fraction(2, 3)) is True   # 113 <= 121


def test_derived_verdict_is_consistent_with_data():
    a = {"n": 400, "p_smooth": 0.20, "se": 0.02}
    b = {"n": 400, "p_smooth": 0.10, "se": 0.015}
    c = {"n": 400, "p_smooth": 0.05, "se": 0.011}
    v = derive_verdict(b, a, c)  # inert = b, split = a, control = c
    assert v["inert_exceeds_control_2se"] and v["split_exceeds_control_2se"] and v["split_exceeds_inert_2se"]
    v2 = derive_verdict(c, c, c)
    assert not v2["inert_exceeds_control_2se"] and v2["split_minus_inert_se"] == 0.0
    empty = {"n": 0, "p_smooth": None, "se": None}
    assert derive_verdict(empty, a, c)["inert_minus_control_se"] is None


def test_experiment_schema_and_structure():
    res = cm_asymmetry_experiment(8, 10)
    for key in ("inert_supersingular", "split_ordinary", "control_random_hasse_order"):
        assert res[key]["n"] > 0 and 0.0 <= res[key]["p_smooth"] <= 1.0
    assert res["split_orders_norms_from_Zi"] == res["split_ordinary"]["n"]
    assert res["inert_all_divisible_by_4"] is True
    assert res["split_fraction_divisible_by_8"] == 1.0
    assert set(res["comparison"]) >= {"inert_minus_control_se", "split_minus_control_se", "split_minus_inert_se"}
