"""Result and work-accounting types shared by all algorithms."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from .numth import mpz


class Work(Counter):
    """Machine-independent operation counters.

    Keys are free-form but algorithms should use the conventional names
    ``mulmod`` (modular multiplications), ``gcd``, ``sqrt_test`` (integer square
    tests), ``division`` (trial divisions), ``candidate`` (search candidates
    examined), ``poly_deg`` (total degree of polynomial products/evaluations).
    ``primary`` is the counter used for scaling fits; each algorithm declares
    which key is primary via ``FactorResult.primary_key``.
    """

    def add(self, key: str, n: int = 1) -> None:
        self[key] += n


@dataclass
class FactorResult:
    """Outcome of one factoring attempt."""

    algorithm: str
    N: Any
    found: bool
    p: Optional[Any] = None
    q: Optional[Any] = None
    wall: float = 0.0
    work: Work = field(default_factory=Work)
    primary_key: str = "candidate"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.N = mpz(self.N)
        if self.p is not None:
            self.p = mpz(self.p)
        if self.q is not None:
            self.q = mpz(self.q)
        if self.found:
            assert self.p is not None and self.q is not None
            assert self.p * self.q == self.N, "claimed factors do not multiply to N"
            assert 1 < self.p < self.N, "trivial factor claimed"
            if self.p > self.q:
                self.p, self.q = self.q, self.p

    @property
    def primary_work(self) -> int:
        return int(self.work.get(self.primary_key, 0))

    def to_json(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "N": str(self.N),
            "found": self.found,
            "p": None if self.p is None else str(self.p),
            "q": None if self.q is None else str(self.q),
            "wall": self.wall,
            "work": dict(self.work),
            "primary_key": self.primary_key,
            "meta": {k: (str(v) if isinstance(v, (int,)) and abs(v) > 2**62 else v)
                     for k, v in self.meta.items()},
        }


def success(algorithm: str, N, d, work: Work, primary_key: str, wall: float = 0.0,
            **meta) -> FactorResult:
    """Build a successful result from a nontrivial divisor ``d`` of ``N``."""
    N, d = mpz(N), mpz(d)
    assert N % d == 0 and 1 < d < N
    return FactorResult(algorithm, N, True, d, N // d, wall, work, primary_key, dict(meta))


def failure(algorithm: str, N, work: Work, primary_key: str, wall: float = 0.0,
            **meta) -> FactorResult:
    return FactorResult(algorithm, N, False, None, None, wall, work, primary_key, dict(meta))
