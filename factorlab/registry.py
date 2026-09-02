"""Algorithm registry.

An algorithm is any callable ``f(N, **params) -> FactorResult``.  Register with

    @register("name", primary_key="mulmod", family_hint="generic")
    def my_algo(N, **params): ...

The registry records the work counter that should be used for scaling fits
(``primary_key``) and a short description, so the benchmark driver can be
completely generic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class AlgorithmInfo:
    name: str
    fn: Callable
    primary_key: str
    description: str = ""
    deterministic: bool = True

    def __call__(self, N, **params):
        return self.fn(N, **params)


ALGORITHMS: dict[str, AlgorithmInfo] = {}


def register(name: str, primary_key: str, description: str = "", deterministic: bool = True):
    def deco(fn):
        if name in ALGORITHMS:
            raise KeyError(f"algorithm {name!r} already registered")
        ALGORITHMS[name] = AlgorithmInfo(name, fn, primary_key, description, deterministic)
        return fn
    return deco


def get_algorithm(name: str) -> AlgorithmInfo:
    # import side-effect modules lazily so the registry is populated
    from . import algorithms  # noqa: F401
    from . import experiments  # noqa: F401
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; known: {sorted(ALGORITHMS)}")
    return ALGORITHMS[name]
