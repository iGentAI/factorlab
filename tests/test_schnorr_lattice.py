"""Tests for E48 (Schnorr-type lattice relations)."""
import math

from sympy import primerange

from factorlab.experiments.schnorr_lattice import _pari, largest_prime_factor, schnorr_trial


def test_relations_are_valid_and_beta_in_range():
    pari = _pari()
    N = 1000003 * 1000033
    primes = list(primerange(2, 100))[:12]
    rels = schnorr_trial(pari, N, primes, [1] * len(primes), K=2 ** 40, M_embed=2 ** 20)
    assert len(rels) > 0
    for r in rels:
        u, v, R = r["u"], r["v"], r["R"]
        assert u >= 1 and v >= 1 and R == abs(u - v * N) > 0          # the relation invariant: residue of u modulo N up to sign
        assert all(largest_prime_factor(x) <= primes[-1] for x in (u, v))  # u and v are factor-base smooth by construction
        assert abs(r["beta"] - math.log(R) / math.log(N)) < 1e-12 and 0 < r["beta"] < 1.5
    assert largest_prime_factor(2 * 3 * 5 * 7) == 7 and largest_prime_factor(1) == 1
