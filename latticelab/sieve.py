"""A Gauss sieve (Micciancio-Voulgaris) with exact work counters and the obstruction statistics of the lattice barrier programme.

The list L is kept pairwise reduced: for no v, w in L is ||v -+ w|| < max(||v||, ||w||).  Fresh vectors are sampled from a reduced
basis by randomised Babai rounding of a random target, reduced against L, then used to reduce L.  Work counters: samples,
inner products (each comparison of a pair counts one), reductions, collisions.  Termination: `max_collisions` consecutive
collisions (the sampled vector reduces to zero or to a list vector).

Statistics (the obstruction statistic of Layer B in docs/lattice_barrier_plan.md):
  angular_excess(L, theta)   -- #{ordered pairs within angle theta} / (|L|(|L|-1) cap_fraction(d, theta)): 1 for uniform directions
  coverage(L, theta, U)      -- fraction of unit directions U within angle theta of some list vector: the covering statistic of T2
  angle_histogram(L)         -- pairwise angles, to be compared with the uniform-sphere density ~ sin^{d-2} theta
Vectors are numpy int64 arrays; norms and inner products are exact in int64 for the dimensions used here (coordinates < 2^20).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fpylll import GSO, IntegerMatrix

from latticelab.lattices import cap_fraction


@dataclass
class SieveStats:
    samples: int = 0
    inner_products: int = 0
    reductions: int = 0
    collisions: int = 0
    list_size_trace: List[int] = field(default_factory=list)


class GaussSieve:
    def __init__(self, B: IntegerMatrix, seed: int = 0, max_collisions: int = 200):
        if max_collisions < 1:
            raise ValueError("max_collisions must be positive")
        self.d = B.nrows
        self.B = np.array([[int(B[i, j]) for j in range(B.ncols)] for i in range(B.nrows)], dtype=np.int64)
        M = GSO.Mat(IntegerMatrix.from_matrix(B))
        M.update_gso()
        self.mu = np.array([[M.get_mu(i, j) if j < i else (1.0 if i == j else 0.0) for j in range(self.d)] for i in range(self.d)])
        self.rr = np.array([M.get_r(i, i) for i in range(self.d)])
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        self.max_collisions = max_collisions
        self.L: List[np.ndarray] = []
        self.norms: List[int] = []
        self.stats = SieveStats()

    # -- sampling: randomised Babai rounding of a random real target -------------------------------------------------------------
    def sample(self) -> np.ndarray:
        self.stats.samples += 1
        # target t = sum c_i b_i with c_i uniform in [-1/2, 1/2) scaled by a spread s, then round coordinate-wise in the GS basis
        c = self.nprng.uniform(-3.0, 3.0, size=self.d)
        coeffs = np.zeros(self.d, dtype=np.int64)
        for i in range(self.d - 1, -1, -1):
            x = c[i] - sum(coeffs[j] * self.mu[j, i] for j in range(i + 1, self.d))
            # randomised rounding: floor or ceil with probability from the fractional part
            fl = math.floor(x)
            coeffs[i] = fl + (1 if self.rng.random() < x - fl else 0)
        v = coeffs @ self.B
        if not v.any():
            return self.sample()
        return v

    @staticmethod
    def _norm2(v: np.ndarray) -> int:
        return int(np.dot(v, v))

    def _reduce_against_list(self, v: np.ndarray, n2: int):
        """Reduce v by list vectors until no list vector shortens it.  Returns (v, n2)."""
        changed = True
        while changed and n2 > 0:
            changed = False
            for w, w2 in zip(self.L, self.norms):
                if w2 > n2:
                    continue
                self.stats.inner_products += 1
                ip = int(np.dot(v, w))
                if 2 * abs(ip) > w2:
                    v = v - w if ip > 0 else v + w
                    n2 = n2 + w2 - 2 * abs(ip)
                    self.stats.reductions += 1
                    changed = True
                    if n2 == 0:
                        break
        return v, n2

    def run(self, target_norm2: Optional[int] = None, max_samples: int = 10 ** 6) -> Dict:
        """Run until `max_collisions` consecutive collisions or a vector of squared norm <= target_norm2 is found."""
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        stack: List[np.ndarray] = []
        consecutive = 0
        while consecutive < self.max_collisions and self.stats.samples < max_samples:
            v = stack.pop() if stack else self.sample()
            n2 = self._norm2(v)
            v, n2 = self._reduce_against_list(v, n2)
            if n2 == 0:
                self.stats.collisions += 1
                consecutive += 1
                continue
            # reduce the list by v; moved vectors go to the stack
            keep_L, keep_n = [], []
            for w, w2 in zip(self.L, self.norms):
                if w2 <= n2:
                    keep_L.append(w)
                    keep_n.append(w2)
                    continue
                self.stats.inner_products += 1
                ip = int(np.dot(w, v))
                if 2 * abs(ip) > n2:
                    w = w - v if ip > 0 else w + v
                    self.stats.reductions += 1
                    if w.any():
                        stack.append(w)
                else:
                    keep_L.append(w)
                    keep_n.append(w2)
            self.L, self.norms = keep_L, keep_n
            # collision: v equals +-w for some w already in the list
            if any(w2 == n2 and (np.array_equal(w, v) or np.array_equal(w, -v)) for w, w2 in zip(self.L, self.norms)):
                self.stats.collisions += 1
                consecutive += 1
                continue
            consecutive = 0
            self.L.append(v)
            self.norms.append(n2)
            self.stats.list_size_trace.append(len(self.L))
            if target_norm2 is not None and n2 <= target_norm2:
                break
        if not self.L:
            return {"shortest": None, "shortest_norm2": None, "list_size": 0, "stats": self.stats}
        i = int(np.argmin(self.norms))
        return {"shortest": self.L[i], "shortest_norm2": self.norms[i], "list_size": len(self.L), "stats": self.stats}


def unit_directions(L: List[np.ndarray]) -> np.ndarray:
    A = np.array(L, dtype=np.float64)
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def angle_histogram(L: List[np.ndarray]) -> np.ndarray:
    """All pairwise angles (radians, in [0, pi/2] using |cos|) of the list directions."""
    U = unit_directions(L)
    C = np.abs(U @ U.T)
    iu = np.triu_indices(len(L), 1)
    return np.arccos(np.clip(C[iu], -1, 1))


def angular_excess(L: List[np.ndarray], theta: float) -> float:
    """#{unordered pairs within angle theta (up to sign)} divided by its expectation for uniform directions,
    C(|L|, 2) * 2 cap_fraction(d, theta) (the factor 2 because of the sign identification)."""
    d = len(L[0])
    ang = angle_histogram(L)
    expected = len(ang) * 2 * cap_fraction(d, theta)
    return float((ang <= theta).sum()) / expected


def coverage(L: List[np.ndarray], theta: float, n_dirs: int, seed: int = 0) -> float:
    """Fraction of n_dirs uniform random unit directions within angle theta (up to sign) of some list vector."""
    d = len(L[0])
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_dirs, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    U = unit_directions(L)
    C = np.abs(X @ U.T)
    return float((C.max(axis=1) >= math.cos(theta)).mean())


def predicted_coverage(list_size: int, d: int, theta: float) -> float:
    """Coverage predicted for a list of uniform directions: 1 - (1 - 2 cap)^{|L|}."""
    return 1 - (1 - 2 * cap_fraction(d, theta)) ** list_size
