"""Tests for the common-factor experiment: the constructor's arithmetic and the attack on moduli that exercise the collision search."""
import math

from gmpy2 import is_prime

from factorlab.experiments.common_factor import check_modulus, common_factor_moduli, multiplicative_order


def test_constructor_arithmetic():
    rows = common_factor_moduli(4, 30, 34, seed=5)
    assert len(rows) == 4
    for r in rows:
        p, q, g, k, l = r["p"], r["q"], r["g"], r["k"], r["l"]
        assert is_prime(p) and is_prime(q) and p < q
        assert p == 1 + k * g and q == 1 + l * g and math.gcd(k, l) == 1 and k < l < 2 * k
        assert g % 2 == 0 and is_prime(g // 2)
        assert math.gcd(p - 1, q - 1) == g
        assert 30 <= r["N"].bit_length() <= 34


def test_attack_factors_and_least_exponent_bounded():
    for r in common_factor_moduli(6, 30, 36, seed=11):
        out = check_modulus(r)
        assert out["factor_ok"], out
        assert out["how"] in ("collision", "gcd(beta - 1, N)")
        if "least_exponent" in out:
            e = out["least_exponent"]
            assert 1 <= e <= min(r["k"], r["l"])
            assert out["ratio_e_over_sqrtN_g"] < 1.0  # e < sqrt N / g
            # the horizon of the doubling search covers the least exponent when the collision branch ran
            if out["how"] == "collision" and e > 1:
                assert out["horizon"] >= e


def test_multiplicative_order():
    assert multiplicative_order(2, 7) == 3
    assert multiplicative_order(3, 7) == 6
    assert multiplicative_order(1, 13) == 1
