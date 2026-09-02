"""Thin, fast number-theory primitives built on gmpy2.

Everything here operates on ``gmpy2.mpz`` (or Python ints, which are coerced).
The functions are deliberately tiny wrappers so that the rest of the code base
has a single place to swap implementations.
"""

from __future__ import annotations

import gmpy2
from gmpy2 import mpz, isqrt, is_square, gcd, powmod, invert, jacobi, iroot

__all__ = [
    "mpz", "isqrt", "isqrt_ceil", "is_square", "gcd", "powmod", "invert",
    "jacobi", "iroot", "is_prime", "bits", "crt", "int_nth_root_ceil",
    "small_primes", "sqrt_mod_prime", "legendre",
]


def is_prime(n, reps: int = 30) -> bool:
    """Probabilistic primality (Miller-Rabin, ``reps`` rounds) via gmpy2.

    gmpy2.is_prime wraps mpz_probab_prime_p which performs a BPSW-style test
    followed by ``reps`` Miller-Rabin rounds; false positives are negligible.
    """
    return bool(gmpy2.is_prime(mpz(n), reps))


def bits(n) -> int:
    """Bit length of ``abs(n)``."""
    return int(abs(mpz(n))).bit_length()


def isqrt_ceil(n):
    """Smallest integer r with r*r >= n (n >= 0)."""
    n = mpz(n)
    r = isqrt(n)
    return r if r * r == n else r + 1


def int_nth_root_ceil(n, k: int):
    """Smallest integer r with r**k >= n (n >= 0, k >= 1)."""
    n = mpz(n)
    r, exact = iroot(n, k)
    return r if exact else r + 1


def crt(r1, m1, r2, m2):
    """Chinese remainder: x with x = r1 (mod m1), x = r2 (mod m2), gcd(m1,m2)=1.

    Returns x in [0, m1*m2).
    """
    r1, m1, r2, m2 = mpz(r1), mpz(m1), mpz(r2), mpz(m2)
    inv = invert(m1 % m2, m2)
    t = ((r2 - r1) * inv) % m2
    return (r1 + m1 * t) % (m1 * m2)


def legendre(a, p) -> int:
    """Legendre symbol (a/p) for odd prime p, returned as -1, 0, 1."""
    return int(jacobi(mpz(a), mpz(p)))


def sqrt_mod_prime(a, p):
    """Tonelli-Shanks: a square root of a modulo odd prime p, or None."""
    a, p = mpz(a) % mpz(p), mpz(p)
    if a == 0:
        return mpz(0)
    if jacobi(a, p) != 1:
        return None
    if p % 4 == 3:
        return powmod(a, (p + 1) // 4, p)
    # factor p-1 = q * 2^s
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = mpz(2)
    while jacobi(z, p) != -1:
        z += 1
    m, c, t, r = s, powmod(z, q, p), powmod(a, q, p), powmod(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = powmod(c, mpz(1) << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


_SMALL_PRIME_CACHE: dict[int, list[int]] = {}


def small_primes(limit: int) -> list[int]:
    """All primes < limit (simple sieve, cached)."""
    if limit in _SMALL_PRIME_CACHE:
        return _SMALL_PRIME_CACHE[limit]
    if limit < 3:
        return []
    sieve = bytearray([1]) * limit
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, limit, i)))
    out = [i for i in range(limit) if sieve[i]]
    _SMALL_PRIME_CACHE[limit] = out
    return out
