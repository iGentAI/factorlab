"""Insertion dynamics on log-profiles (the primal exact-SVP insertion model of docs/notes_lattice_barrier.md, section 7, in its
simulator-world idealisation), and the consistency question: can an insertion schedule drive l_1 below the certified profile floor?

State: l in R^d (log Gram-Schmidt norms), sum conserved.  Insertion at block i (size beta_i = min(beta, d - i + 1)): the block's shortest
vector is taken to have length exactly GH, g = L(beta_i) + avg(l_i..l_{i+beta_i-1}) with L(n) = log chat(n); if g < l_i then l_i <- g and
the removed mass m = l_i - g is redistributed over the other beta_i - 1 positions of the block by a *completion rule*, preserving the
block volume exactly; otherwise nothing changes.  Completion rules:
  uniform -- every other position gains m/(beta_i - 1).  Admissibility (l_j >= GH_j for all blocks) is INVARIANT under this rule
             (Proposition (uniform completion) in the notes): the inserted position becomes tight, later positions in the block gain
             more than their block's GH does, and earlier overlapping blocks lose mass.  Hence every reachable profile satisfies the
             certified floor, and tours converge to its unique minimiser.
  hkz     -- the GH-modelled HKZ residual: l_{i+k} <- L(beta_i - k) + (remaining)/(beta_i - k), k = 1..beta_i-1.  Front-loaded relative
             to the GSA line: the last positions of the block fall below their own full-block GH (the surplus identity of the notes), so
             this completion CREATES violations, increasingly so in the small residual dimensions where chat(n) < 1 and GH is not a
             model of HKZ shapes.
  hkz13   -- the HKZ residual for residual dimensions >= 13 and uniform completion of the last 12 positions: isolates how much of the
             hkz behaviour is the small-dimension artefact.

Schedules (factories returning stateful callables) range over every block position 1..d-1 (sizes beta_i, as a BKZ tour does): tours
(i = 1..d-1 repeated), reverse tours, random positions, greedy (the
insertion lowering l_1 most; if none does, the one lowering the potential sum_i (d - i) l_i most), tail-first (reverse tours until a full
reverse tour changes nothing, then forward tours).  For every trajectory we record min_t (l_1(t) - floor), with floor the certified LP
value at (d, beta, eps = 0) for the same volume, and the largest violation max_i (GH_i - l_i)^+ along the way.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, Tuple

import numpy as np

from latticelab.profile_floor import floor_l1_float, log_chat


def gh_of_block(l: np.ndarray, i: int, size: int) -> float:
    """log GH of the block starting at 0-based position i with the given size."""
    return log_chat(size) + float(l[i:i + size].mean())


def insert(l: np.ndarray, i: int, beta: int, completion: str = "uniform") -> Tuple[np.ndarray, bool]:
    """One insertion at 0-based block position i with the given completion rule.  Returns (new profile, changed)."""
    d = len(l)
    size = min(beta, d - i)
    if size < 2:
        return l, False
    g = gh_of_block(l, i, size)
    if g >= l[i] - 1e-15:
        return l, False
    new = l.copy()
    removed = float(l[i]) - g
    new[i] = g
    if completion == "uniform":
        new[i + 1:i + size] += removed / (size - 1)
        return new, True
    if completion not in ("hkz", "hkz13"):
        raise ValueError(completion)
    remaining = float(l[i:i + size].sum()) - g
    cutoff = 13 if completion == "hkz13" else 1
    for k in range(1, size):
        n = size - k  # residual dimension at position i + k
        if n < cutoff:  # uniform completion of the remaining positions
            new[i + k:i + size] = remaining / n
            break
        if n == 1:
            new[i + k] = remaining
        else:
            v = log_chat(n) + remaining / n
            new[i + k] = v
            remaining -= v
    return new, True


def violations(l: np.ndarray, beta: int) -> np.ndarray:
    """(GH_i - l_i)^+ for every block i (0-based, sizes min(beta, d - i) >= 2): the admissibility violations of the profile."""
    d = len(l)
    out = np.zeros(d - 1)
    for i in range(d - 1):
        out[i] = max(0.0, gh_of_block(l, i, min(beta, d - i)) - l[i])
    return out


def lll_like_profile(d: int, log_vol: float = 0.0, delta: float = 1.0219) -> np.ndarray:
    """A linear profile with root-Hermite factor delta (LLL-like) and the given log volume."""
    s = -2 * math.log(delta)
    l = s * np.arange(d)
    return l - l.mean() + log_vol / d


# -- schedule factories: each call returns a fresh stateful schedule(l, beta, rng, last_changed) -> i ---------------------------------
def make_tours(d: int, beta: int):
    st = {"i": 0}
    m = d - 1  # every block, including the shrinking tail (BKZ processes the blocks [i, min(i + beta - 1, d)] for all i)

    def sched(l, b, rng, last_changed):
        i = st["i"] % m
        st["i"] += 1
        return i

    return sched


def make_reverse_tours(d: int, beta: int):
    st = {"i": 0}
    m = d - 1

    def sched(l, b, rng, last_changed):
        i = (m - 1) - (st["i"] % m)
        st["i"] += 1
        return i

    return sched


def make_random(d: int, beta: int):
    return lambda l, b, rng, last_changed: rng.randrange(0, d - 1)


def make_greedy(d: int, beta: int, completion: str):
    weights = np.arange(d, 0, -1)

    def sched(l, b, rng, last_changed):
        best_i, best_val, best_pot = 0, float(l[0]), None
        for i in range(0, d - 1):
            new, changed = insert(l, i, beta, completion)
            if not changed:
                continue
            if new[0] < best_val - 1e-15:
                best_i, best_val = i, float(new[0])
            elif best_val >= l[0] - 1e-15:  # nothing lowers l_1 yet: use the potential (move mass right)
                pot = float(np.dot(weights, new))
                if best_pot is None or pot < best_pot:
                    best_i, best_pot = i, pot
        return best_i

    return sched


def make_tail_first(d: int, beta: int):
    """Reverse tours until a complete reverse tour changes nothing, then forward tours."""
    st = {"phase": 0, "i": 0, "changed_in_tour": False}
    m = d - 1

    def sched(l, b, rng, last_changed):
        if st["phase"] == 0:
            if st["i"] > 0:
                st["changed_in_tour"] = st["changed_in_tour"] or last_changed
            if st["i"] > 0 and st["i"] % m == 0:
                if not st["changed_in_tour"]:
                    st["phase"], st["i"] = 1, 1  # this call is the first forward insertion (position 0); forward tours continue at 1
                    return 0
                st["changed_in_tour"] = False
            i = (m - 1) - (st["i"] % m)
            st["i"] += 1
            return i
        i = st["i"] % m
        st["i"] += 1
        return i

    return sched


SCHEDULES = {"tours": make_tours, "reverse": make_reverse_tours, "random": make_random, "tail_first": make_tail_first}


def run_schedule(l0: np.ndarray, beta: int, schedule: Callable, steps: int, completion: str = "uniform", seed: int = 0) -> Dict:
    """Apply `steps` insertions chosen by `schedule(l, beta, rng, last_changed)`; record the minimum of l_1 and the maximal violation."""
    rng = random.Random(seed)
    l = l0.copy()
    min_l1, argmin_t = float(l[0]), 0
    max_viol = float(violations(l, beta).max())
    changes, last_changed = 0, False
    for t in range(1, steps + 1):
        i = schedule(l, beta, rng, last_changed)
        l, last_changed = insert(l, i, beta, completion)
        changes += last_changed
        if l[0] < min_l1:
            min_l1, argmin_t = float(l[0]), t
        if last_changed:
            max_viol = max(max_viol, float(violations(l, beta).max()))
    return {"final": l, "min_l1": min_l1, "argmin_t": argmin_t, "max_violation": max_viol, "changes": changes, "final_l1": float(l[0])}


def consistency_census(d: int, beta: int, steps: int, completions=("uniform", "hkz13", "hkz"), seeds=(0, 1, 2), delta0: float = 1.0219) -> Dict:
    """All schedules and completion rules from the LLL-like profile of volume 1, against the certified floor."""
    floor = floor_l1_float(d, beta, 0.0, 0.0)["l1_floor"]
    l0 = lll_like_profile(d, 0.0, delta0)
    rows = []
    for completion in completions:
        for name in ("tours", "reverse", "random", "greedy", "tail_first"):
            for seed in (seeds if name == "random" else (0,)):
                sched = make_greedy(d, beta, completion) if name == "greedy" else SCHEDULES[name](d, beta)
                r = run_schedule(l0, beta, sched, steps, completion, seed)
                rows.append({"completion": completion, "schedule": name, "seed": seed, "min_l1_minus_floor": r["min_l1"] - floor,
                             "final_l1_minus_floor": r["final_l1"] - floor, "argmin_t": r["argmin_t"], "max_violation": r["max_violation"],
                             "changes": r["changes"]})
    return {"d": d, "beta": beta, "steps": steps, "floor_l1": floor, "initial_l1": float(l0[0]), "rows": rows}
