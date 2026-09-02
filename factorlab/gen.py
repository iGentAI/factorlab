"""Seeded, unbiased generation of primes and semiprime test instances.

Why rejection sampling and not ``next_prime``
---------------------------------------------
``next_prime(x)`` for uniform ``x`` returns a prime with probability
proportional to the gap *preceding* it, so primes after large gaps are
over-represented (a size-biased sample of gaps).  This is a real, measurable
bias (see ``factorlab.audit.next_prime_gap_bias``) and it is exactly the kind
of systematic artefact that could fool a factoring experiment (gap structure
leaks into ``p - q`` and ``p + q`` statistics).  We therefore draw candidates
uniformly from the target interval and accept the first prime.

Families
--------
Each family is a function ``(rng, nbits, **kw) -> (p, q)`` with ``p < q``.
Register new families by adding to ``FAMILIES``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from .numth import mpz, is_prime, bits, isqrt, small_primes

__all__ = [
    "random_odd", "random_prime", "random_prime_in", "Semiprime",
    "make_semiprime", "semiprime_suite", "FAMILIES", "rng_from_seed",
]


def rng_from_seed(seed) -> random.Random:
    """A ``random.Random`` instance; accepts an int or an existing Random."""
    if isinstance(seed, random.Random):
        return seed
    return random.Random(seed)


def random_odd(rng: random.Random, nbits: int):
    """Uniform odd integer with exactly ``nbits`` bits (top and bottom bits set)."""
    if nbits < 2:
        raise ValueError("need nbits >= 2")
    x = rng.getrandbits(nbits)
    x |= (1 << (nbits - 1)) | 1
    return mpz(x)


def random_prime(rng: random.Random, nbits: int, reps: int = 30):
    """Uniformly random prime with exactly ``nbits`` bits, by rejection sampling."""
    if nbits == 2:
        return mpz(rng.choice([2, 3]))
    while True:
        x = random_odd(rng, nbits)
        if is_prime(x, reps):
            return x


def random_prime_in(rng: random.Random, lo, hi, reps: int = 30):
    """Uniformly random prime in the half-open interval [lo, hi) by rejection.

    Candidates are drawn uniformly from the *odd* integers of the interval (so
    every odd integer has exactly one preimage), and the prime 2 is handled
    explicitly when it lies in the interval.  The accepted prime is therefore
    uniform over the primes of [lo, hi).
    """
    lo, hi = mpz(lo), mpz(hi)
    if hi <= lo:
        raise ValueError("empty interval")
    first_odd = lo if lo % 2 else lo + 1
    n_odd = int((hi - first_odd + 1) // 2) if hi > first_odd else 0
    has_two = lo <= 2 < hi
    if n_odd == 0 and not has_two:
        raise ValueError("interval contains no odd integers")
    while True:
        # Each candidate integer (odd ones plus 2 if present) is equally likely.
        k = rng.randrange(n_odd + (1 if has_two else 0))
        if has_two and k == n_odd:
            return mpz(2)
        x = first_odd + 2 * k
        if is_prime(x, reps):
            return x


# --------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------

def _balanced(rng, nbits, **kw):
    """Two independent uniform (nbits//2)-bit primes (p != q)."""
    h = nbits // 2
    while True:
        p, q = random_prime(rng, h), random_prime(rng, nbits - h)
        if p != q:
            return (p, q) if p < q else (q, p)


def _rsa(rng, nbits, min_gap_bits: Optional[int] = None, **kw):
    """FIPS-186 style: p, q in [sqrt(2)*2^(h-1), 2^h) so N has exactly nbits bits,
    and |p - q| > 2^(h - 100) (or 2^(h - min_gap_bits) for small sizes)."""
    h = nbits // 2
    if nbits % 2:
        raise ValueError("rsa family needs even nbits")
    lo = isqrt(mpz(2) << (2 * h - 2)) + 1  # ceil(sqrt(2) * 2^(h-1))
    hi = mpz(1) << h
    gap_bits = min_gap_bits if min_gap_bits is not None else min(100, max(1, h // 2))
    gap = mpz(1) << max(0, h - gap_bits)
    while True:
        p, q = random_prime_in(rng, lo, hi), random_prime_in(rng, lo, hi)
        if abs(p - q) > gap:
            return (p, q) if p < q else (q, p)


def _skew(rng, nbits, p_bits: Optional[int] = None, ratio: float = 0.35, **kw):
    """Unbalanced: p has ``p_bits`` bits (default ratio*nbits), q the remainder."""
    pb = p_bits if p_bits is not None else max(3, int(round(ratio * nbits)))
    qb = nbits - pb
    if qb < pb:
        pb, qb = qb, pb
    while True:
        p, q = random_prime(rng, pb), random_prime(rng, qb)
        if p != q:
            return (p, q) if p < q else (q, p)


def _close(rng, nbits, gap_bits: Optional[int] = None, **kw):
    """q - p is roughly 2^gap_bits (default nbits//4): Fermat-friendly instances."""
    h = nbits // 2
    g = gap_bits if gap_bits is not None else max(2, nbits // 4)
    while True:
        p = random_prime(rng, h)
        lo = p + (mpz(1) << (g - 1))
        hi = p + (mpz(1) << g)
        q = random_prime_in(rng, lo, hi)
        if bits(q) == nbits - h or bits(q) == h:
            return p, q


def _smooth_pm1(rng, nbits, B: int = 1000, **kw):
    """p - 1 is B-smooth (Pollard p-1 friendly); q is a uniform prime."""
    h = nbits // 2
    primes = small_primes(B)
    while True:
        # build p-1 = 2 * product of random small prime powers until ~h bits
        m = mpz(2)
        while bits(m) < h - 1:
            m *= rng.choice(primes)
        p = m + 1
        if bits(p) == h and is_prime(p):
            break
    q = random_prime(rng, nbits - h)
    while q == p:
        q = random_prime(rng, nbits - h)
    return (p, q) if p < q else (q, p)


def _safe(rng, nbits, **kw):
    """Both primes are safe primes p = 2p' + 1."""
    h = nbits // 2

    def safe_prime(b):
        while True:
            pp = random_prime(rng, b - 1)
            p = 2 * pp + 1
            if bits(p) == b and is_prime(p):
                return p

    while True:
        p, q = safe_prime(h), safe_prime(nbits - h)
        if p != q:
            return (p, q) if p < q else (q, p)


FAMILIES: dict[str, Callable] = {
    "balanced": _balanced,
    "rsa": _rsa,
    "skew": _skew,
    "close": _close,
    "smooth_pm1": _smooth_pm1,
    "safe": _safe,
}


@dataclass
class Semiprime:
    """A test instance with provenance."""

    N: object
    p: object
    q: object
    nbits: int
    family: str
    seed: int
    index: int
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.N, self.p, self.q = mpz(self.N), mpz(self.p), mpz(self.q)
        assert self.p * self.q == self.N
        assert self.p < self.q

    def to_json(self) -> dict:
        return {
            "N": str(self.N), "p": str(self.p), "q": str(self.q),
            "nbits": self.nbits, "family": self.family, "seed": self.seed,
            "index": self.index, "params": self.params,
        }

    def __repr__(self) -> str:
        return f"Semiprime({self.family}, {self.nbits}b, seed={self.seed}, i={self.index}, N={self.N})"


def make_semiprime(nbits: int, family: str = "balanced", seed: int = 0, index: int = 0,
                   **params) -> Semiprime:
    """Deterministically build instance ``index`` of a (nbits, family, seed) stream.

    Instance ``index`` is generated by an rng seeded with a hash of
    (seed, family, nbits, index), so each instance is independently
    reproducible without generating its predecessors.
    """
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    rng = random.Random((seed, family, nbits, index, tuple(sorted(params.items()))).__repr__())
    p, q = FAMILIES[family](rng, nbits, **params)
    return Semiprime(p * q, p, q, nbits, family, seed, index, dict(params))


def semiprime_suite(nbits: int, count: int, family: str = "balanced", seed: int = 0,
                    **params) -> Iterator[Semiprime]:
    """Yield ``count`` reproducible instances."""
    for i in range(count):
        yield make_semiprime(nbits, family, seed, i, **params)
