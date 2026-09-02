"""factorlab: a research harness for integer factorisation experiments.

Design goals
------------
* Reproducible: every random choice flows from an explicit seed.
* Unbiased: primes are drawn uniformly from dyadic intervals by rejection
  sampling (never via ``next_prime``), and ``factorlab.audit`` can verify this
  statistically.
* Uniform: every algorithm implements ``factor(N, **params) -> FactorResult``
  and reports machine-independent *work* counters (modular multiplications,
  candidates tested, polynomial degrees ...) alongside wall time, so that
  scaling exponents can be fitted independently of interpreter overhead.
* Extensible: new algorithms register themselves with a decorator; new
  semiprime families are one function each.
"""

from .numth import (
    mpz,
    isqrt,
    isqrt_ceil,
    is_square,
    gcd,
    powmod,
    invert,
    jacobi,
    is_prime,
    bits,
    iroot,
    crt,
)
from .gen import (
    random_prime,
    random_odd,
    Semiprime,
    make_semiprime,
    semiprime_suite,
    FAMILIES,
)
from .result import FactorResult, Work
from .registry import ALGORITHMS, register, get_algorithm

__all__ = [
    "mpz", "isqrt", "isqrt_ceil", "is_square", "gcd", "powmod", "invert",
    "jacobi", "is_prime", "bits", "iroot", "crt",
    "random_prime", "random_odd", "Semiprime", "make_semiprime",
    "semiprime_suite", "FAMILIES",
    "FactorResult", "Work", "ALGORITHMS", "register", "get_algorithm",
]
