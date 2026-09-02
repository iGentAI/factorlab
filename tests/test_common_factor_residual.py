"""The large common-factor class (companion paper, Corollary 5.2): moduli with g = gcd(p-1, q-1) > (p-1)^{3/4} are factored by
trying successive bases alpha until alpha^{N-1} != 1 mod N (Burgess bounds the first such base) and then one collision search of
cost sqrt(min(k, l)).  The constructed moduli p = 1 + 4L, q = 1 + 12L have g = 4L, k = 1, l = 3: every base is a Fermat liar
modulo p, and a base is a non-liar modulo q exactly when it is not a cube modulo q, so gcd(alpha^{N-1} - 1, N) = p for the
first non-cube base."""
from gmpy2 import gcd, powmod

from factorlab.experiments.harvey_residue import common_factor_attack
from factorlab.experiments.order_selection import common_divisor_moduli


def test_residual_common_factor_class_factors_with_few_bases():
    moduli = common_divisor_moduli(6)
    assert len(moduli) == 6
    for L, p, q, k, l in moduli:
        N = p * q
        g = int(gcd(p - 1, q - 1))
        assert (k, l) == (1, 3) and g == 4 * L and (p - 1) // g == 1 and (q - 1) // g == 3
        assert g ** 4 > (p - 1) ** 3  # in the residual class g > (p-1)^{3/4}
        res = common_factor_attack(N)
        assert res["factor"] in (p, q)
        # the first non-liar base is the first non-cube modulo q; Burgess bounds it, here it is tiny
        alpha = res["alpha"]
        assert alpha <= 20
        beta = powmod(alpha, N - 1, N)
        assert beta % p == 1 and beta % q != 1
