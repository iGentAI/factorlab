import random

import gmpy2
import pytest

from factorlab import numth
from factorlab.gen import (
    random_prime, random_prime_in, make_semiprime, semiprime_suite, FAMILIES, rng_from_seed,
)


def test_isqrt_ceil_and_square():
    for n in [0, 1, 2, 3, 4, 15, 16, 17, 10**20, 10**20 + 1]:
        r = numth.isqrt_ceil(n)
        assert r * r >= n and (r - 1) * (r - 1) < n or n == 0
    assert numth.is_square(144) and not numth.is_square(145)


def test_crt():
    x = numth.crt(2, 3, 3, 5)
    assert x % 3 == 2 and x % 5 == 3 and 0 <= x < 15


def test_sqrt_mod_prime():
    rng = random.Random(1)
    for _ in range(50):
        p = random_prime(rng, 40)
        a = rng.randrange(1, int(p))
        r = numth.sqrt_mod_prime(a, p)
        if numth.legendre(a, p) == 1:
            assert r is not None and r * r % p == a % p
        else:
            assert r is None


def test_random_prime_exact_bits():
    rng = rng_from_seed(0)
    for nb in [3, 8, 17, 64, 129]:
        for _ in range(5):
            p = random_prime(rng, nb)
            assert numth.bits(p) == nb and numth.is_prime(p)


def test_random_prime_in_uniform_small_interval():
    """Empirical uniformity over the primes of a tiny interval (the overseer case)."""
    rng = rng_from_seed(7)
    lo, hi = 5, 8  # primes 5 and 7
    counts = {5: 0, 7: 0}
    n = 4000
    for _ in range(n):
        counts[int(random_prime_in(rng, lo, hi))] += 1
    # binomial(4000, 1/2): 5 sigma is ~158
    assert abs(counts[5] - counts[7]) < 320
    # interval containing 2
    rng = rng_from_seed(8)
    seen = {int(random_prime_in(rng, 2, 6)) for _ in range(200)}
    assert seen == {2, 3, 5}


def test_random_prime_in_bounds():
    rng = rng_from_seed(3)
    lo, hi = 10**15, 10**15 + 10**6
    for _ in range(20):
        p = random_prime_in(rng, lo, hi)
        assert lo <= p < hi and numth.is_prime(p)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_families_produce_valid_semiprimes(family):
    nb = 40
    for s in semiprime_suite(nb, 4, family, seed=11):
        assert s.p < s.q and s.p * s.q == s.N
        assert numth.is_prime(s.p) and numth.is_prime(s.q)
        assert abs(numth.bits(s.N) - nb) <= 1


def test_rsa_family_exact_bits_and_gap():
    for s in semiprime_suite(64, 10, "rsa", seed=2):
        assert numth.bits(s.N) == 64
        assert numth.bits(s.p) == 32 and numth.bits(s.q) == 32


def test_skew_family_sorted_even_when_equal_bits():
    for s in semiprime_suite(40, 10, "skew", seed=5, p_bits=20):
        assert s.p < s.q


def test_reproducibility_and_independence():
    a = make_semiprime(48, "balanced", 123, 4)
    b = make_semiprime(48, "balanced", 123, 4)
    c = make_semiprime(48, "balanced", 123, 5)
    assert a.N == b.N and a.N != c.N


def test_smooth_family_is_smooth():
    s = make_semiprime(60, "smooth_pm1", 0, 0, B=500)
    m = int(s.p - 1)
    for q in numth.small_primes(500):
        while m % q == 0:
            m //= q
    assert m == 1
