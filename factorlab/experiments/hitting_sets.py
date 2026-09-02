"""E20: smoothness hitting sets -- how many fixed group orders cover every prime?

Proposition A of notes_probabilistic.md shows that deterministic subexponential
factoring exists non-uniformly; the natural explicit advice for smoothness
methods is a *separating family*: a fixed list of groups attached to p (the
multiplicative group, the norm-one torus, elliptic curves E_sigma(F_p)) such
that for every pair of distinct primes p, q in a range, some member succeeds
modulo exactly one of them.  Equivalently, all primes have distinct success
signatures.  This condition is stronger than mere coverage and guarantees
that a gcd factors every semiprime pq in the range.  At B1 = p^{1/u}, stage 1
costs ~ p^{1/u} = N^{1/(2u)} per member and succeeds with a constant
probability rho(u) > 0, so a random-signature heuristic predicts O_u(log N)
members suffice to separate all primes up to sqrt N.  This would give
N^{1/(2u) + o(1)} for every fixed u -- below the proven deterministic N^{1/5}
for u > 5/2 -- and L_N[1/2] when u grows optimally.  The proof obstacle is the
separating property of an *explicit* family, not its finite evaluation.

This module measures it: for all primes in [x, 2x) and the fixed family
(p - 1 with base 2, Williams p + 1 with bases 3, 5, 11, Suyama curves
sigma = 6, 7, 8, ...), with B1 = x^{1/u}, it records both the coverage length
and the first length at which every prime has a distinct stage-1 signature,
how the unresolved-pair count decays, the per-prime success distribution on a
sample, and independent-signature predictions.  An optional continuation to
B2 = x^{2/u} tests every prime l in (B1, B2] and detects l times the stage-1
element/point becoming the identity.  This is the exact one-large-prime
predicate used for the population measurement (and a subset of the residual-
order <= B2 predicate exposed by a generic BSGS); the prime loop is not a
simulation of the BSGS operation count.  All arithmetic is vectorised over
the prime array (numpy int64, p < 2^31).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from ..numth import small_primes
from ..algorithms.ecm import stage1_exponents
from .ecm_hitting import primes_in_range, vec_powmod, vec_suyama, vec_ladder, vec_xdbl, vec_xadd


def stage2_primes(B1: int, B2: int) -> list[int]:
    return [l for l in small_primes(int(B2) + 1) if l > B1]


# --------------------------------------------------------------------------
# Vectorised methods: stage 1 on all primes, stage 2 on a subset
# --------------------------------------------------------------------------

def _suyama_den_zero(sigma: int, p: np.ndarray) -> np.ndarray:
    """Primes exposed immediately because Suyama's construction denominator is zero."""
    s = np.full_like(p, sigma) % p
    u = (s * s - 5) % p
    v = (4 * s) % p
    u3 = ((u * u) % p) * u % p
    return ((16 * u3) % p) * v % p == 0


def ecm_success(sigma: int, p: np.ndarray, B1: int, B2: int | None = None,
                mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Stage-1 exposure (including a zero construction denominator) and one-large-prime stage 2."""
    p = p.astype(np.int64)
    immediate = _suyama_den_zero(sigma, p)
    a24, X, Z = vec_suyama(sigma, p)
    for pe in stage1_exponents(int(B1)):
        X, Z = vec_ladder(pe, X, Z, a24, p)
    s1 = immediate | ((Z % p) == 0)
    s2 = np.zeros_like(s1)
    if B2 is None:
        return s1, s2
    sel = ~s1 if mask is None else (~s1 & mask)
    idx = np.nonzero(sel)[0]
    if idx.size:
        pp, Xs, Zs, aa = p[idx], X[idx], Z[idx], a24[idx]
        hit = np.zeros(idx.size, dtype=bool)
        for l in stage2_primes(B1, B2):
            _, Zl = vec_ladder(l, Xs, Zs, aa, pp)
            hit |= (Zl % pp) == 0
        s2[idx] = hit
    return s1, s2


def ecm_plain_success(A: int, p: np.ndarray, B1: int, B2: int | None = None,
                      mask: np.ndarray | None = None, X0: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Plain Montgomery curve B y^2 = x^3 + A x^2 + x with a24 = (A + 2)/4 and the x-only
    point (X0 : 1), which lies on the curve or on its quadratic twist: either way
    Z = 0 after the ladder is a valid one-sided exposure predicate.  The family
    has the 2-torsion point (0, 0) but no forced 3-torsion, unlike Suyama's.
    Primes dividing 4 or A^2 - 4 (singular or undefined) are marked exposed."""
    p = p.astype(np.int64)
    A_ = np.full_like(p, A) % p
    den = np.full_like(p, 4) % p
    degenerate = (den == 0) | (((A_ * A_ - 4) % p) == 0)
    safe_den = np.where(degenerate, 1, den)
    a24 = (((A_ + 2) % p) * vec_powmod(safe_den, p - 2, p)) % p
    X = np.full_like(p, X0) % p
    Z = np.ones_like(p)
    for pe in stage1_exponents(int(B1)):
        X, Z = vec_ladder(pe, X, Z, a24, p)
    s1 = degenerate | ((Z % p) == 0)
    s2 = np.zeros_like(s1)
    if B2 is None:
        return s1, s2
    sel = ~s1 if mask is None else (~s1 & mask)
    idx = np.nonzero(sel)[0]
    if idx.size:
        pp, Xs, Zs, aa = p[idx], X[idx], Z[idx], a24[idx]
        hit = np.zeros(idx.size, dtype=bool)
        for l in stage2_primes(B1, B2):
            _, Zl = vec_ladder(l, Xs, Zs, aa, pp)
            hit |= (Zl % pp) == 0
        s2[idx] = hit
    return s1, s2


def pm1_success(p: np.ndarray, B1: int, B2: int | None = None, base: int = 2,
                mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """p - 1 method with the given base: stage 1 (all primes), stage 2 on the subset."""
    p = p.astype(np.int64)
    x = np.full_like(p, base) % p
    for pe in stage1_exponents(int(B1)):
        x = vec_powmod(x, np.full_like(p, pe), p)
    s1 = x == 1
    s2 = np.zeros_like(s1)
    if B2 is None:
        return s1, s2
    sel = ~s1 if mask is None else (~s1 & mask)
    idx = np.nonzero(sel)[0]
    if idx.size:
        pp, xs = p[idx], x[idx]
        hit = np.zeros(idx.size, dtype=bool)
        for l in stage2_primes(B1, B2):
            hit |= vec_powmod(xs, np.full_like(pp, l), pp) == 1
        s2[idx] = hit
    return s1, s2


def vec_lucas_v(P: np.ndarray, n: int, p: np.ndarray) -> np.ndarray:
    """V_n(P, 1) modulo p, elementwise, by the (V_k, V_{k+1}) ladder."""
    vk = np.full_like(p, 2) % p
    vk1 = P % p
    for bit in bin(int(n))[2:]:
        if bit == "1":
            vk, vk1 = (vk * vk1 - P) % p, (vk1 * vk1 - 2) % p
        else:
            vk, vk1 = (vk * vk - 2) % p, (vk * vk1 - P) % p
    return vk


def williams_success(p: np.ndarray, B1: int, B2: int | None = None, P0: int = 3,
                     mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Williams' method with seed P0: V = 2 detects gamma^L = 1 (order in p - 1 or p + 1)."""
    p = p.astype(np.int64)
    V = np.full_like(p, P0) % p
    for pe in stage1_exponents(int(B1)):
        V = vec_lucas_v(V, pe, p)
    s1 = V == 2
    s2 = np.zeros_like(s1)
    if B2 is None:
        return s1, s2
    sel = ~s1 if mask is None else (~s1 & mask)
    idx = np.nonzero(sel)[0]
    if idx.size:
        pp, Vs = p[idx], V[idx]
        hit = np.zeros(idx.size, dtype=bool)
        for l in stage2_primes(B1, B2):
            hit |= vec_lucas_v(Vs, l, pp) == 2
        s2[idx] = hit
    return s1, s2


# --------------------------------------------------------------------------
# Covering experiment
# --------------------------------------------------------------------------

def family(primes: np.ndarray, B1: int, B2: int | None, deterministic: bool, max_curves: int,
           curves: str = "suyama"):
    """Ordered list of (name, callable(mask) -> (s1, s2)).

    ``curves`` selects the curve family: 'suyama' (sigma = 6, 7, ...), 'plain'
    (Montgomery A = 7, 8, ...; no forced 3-torsion) or 'mixed' (alternating).
    """
    out: list[tuple[str, Callable]] = []
    if deterministic:
        out.append(("pm1_2", lambda m: pm1_success(primes, B1, B2, 2, m)))
        for P0 in (3, 5, 11):
            out.append((f"williams_{P0}", lambda m, P0=P0: williams_success(primes, B1, B2, P0, m)))
    for i in range(max_curves):
        if curves == "suyama" or (curves == "mixed" and i % 2 == 0):
            s = 6 + (i if curves == "suyama" else i // 2)
            out.append((f"ecm_{s}", lambda m, s=s: ecm_success(s, primes, B1, B2, m)))
        elif curves in ("plain", "mixed"):
            A = 7 + (i if curves == "plain" else i // 2)
            out.append((f"plain_{A}", lambda m, A=A: ecm_plain_success(A, primes, B1, B2, m)))
        else:
            raise ValueError(f"unknown curve family {curves!r}")
    return out


def _refine_classes(classes: np.ndarray, bit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Refine exact signature classes by one Boolean coordinate; return compact ids and class sizes."""
    _, inverse, sizes = np.unique(classes * 2 + bit.astype(np.int64), return_inverse=True, return_counts=True)
    return inverse.astype(np.int64), sizes.astype(np.int64)


def cover(primes: np.ndarray, B1: int, B2: int | None, deterministic: bool, max_curves: int,
          curves: str = "suyama") -> dict:
    """Run the fixed family; measure coverage and exact stage-1 separation of every prime pair.

    If B2 is None only stage 1 is used. With a continuation, it is evaluated
    on previously uncovered primes and every member of a non-singleton combined
    signature class; singleton classes need no later coordinates because classes
    only refine, never merge. Combined and stage-1 separation are therefore exact
    for the emitted predicates.
    """
    n = primes.size
    uncovered = np.ones(n, dtype=bool)
    uncovered_s1 = np.ones(n, dtype=bool)
    classes_s1 = np.zeros(n, dtype=np.int64)
    classes = np.zeros(n, dtype=np.int64)
    first_hit = np.full(n, -1, dtype=np.int64)
    history = []
    K_star = None
    K_star_s1 = None
    K_separate = None
    K_separate_s1 = None
    last_pairs: list[tuple[int, int]] | None = None
    last_pairs_method = None
    tail_pairs: list[tuple[int, int]] | None = None
    tail_pairs_method = None
    for i, (name, fn) in enumerate(family(primes, B1, B2, deterministic, max_curves, curves)):
        prior_sizes = np.bincount(classes) if n else np.array([], dtype=np.int64)
        active = prior_sizes[classes] > 1 if n else np.array([], dtype=bool)
        need_continuation = uncovered | active
        s1, s2 = fn(need_continuation)
        bit = s1 | s2
        old_uncovered = uncovered.copy()
        newly = bit & old_uncovered
        first_hit[newly] = i
        uncovered &= ~bit
        uncovered_s1 &= ~s1
        classes, sizes = _refine_classes(classes, bit)
        classes_s1, sizes_s1 = _refine_classes(classes_s1, s1)
        unresolved_pairs = int(np.sum(sizes * (sizes - 1) // 2))
        unresolved_pairs_s1 = int(np.sum(sizes_s1 * (sizes_s1 - 1) // 2))
        if 0 < unresolved_pairs <= 60:
            big = np.nonzero(sizes > 1)[0]
            snap = []
            for c in big:
                members = [int(v) for v in primes[classes == c]]
                snap.extend((members[a], members[b]) for a in range(len(members)) for b in range(a + 1, len(members)))
            last_pairs, last_pairs_method = snap, name
            if tail_pairs is None:
                tail_pairs, tail_pairs_method = snap, name
        history.append({"method": name, "stage1_rate_all": float(s1.mean()),
                        "stage2_rate_on_uncovered": float((s2 & old_uncovered).sum() /
                                                            max(1, int(((~s1) & old_uncovered).sum()))),
                        "uncovered_fraction": float(uncovered.mean()),
                        "uncovered_fraction_stage1_only": float(uncovered_s1.mean()),
                        "signature_classes": int(sizes.size),
                        "unresolved_prime_pairs": unresolved_pairs,
                        "max_signature_class": int(sizes.max()),
                        "stage1_signature_classes": int(sizes_s1.size),
                        "stage1_unresolved_prime_pairs": unresolved_pairs_s1,
                        "stage1_max_signature_class": int(sizes_s1.max())})
        if K_star is None and not uncovered.any():
            K_star = i + 1
        if K_star_s1 is None and not uncovered_s1.any():
            K_star_s1 = i + 1
        if K_separate is None and sizes.size == n:
            K_separate = i + 1
        if K_separate_s1 is None and sizes_s1.size == n:
            K_separate_s1 = i + 1
        if K_star is not None and K_star_s1 is not None and K_separate is not None and K_separate_s1 is not None:
            break
    uncovered_idx = np.nonzero(first_hit < 0)[0]
    covered_idx = np.nonzero(first_hit >= 0)[0]
    covered_idx = covered_idx[np.argsort(-first_hit[covered_idx])]
    ranked = np.concatenate((uncovered_idx, covered_idx))
    # 'hardest_primes' is a coverage statistic: uncovered primes first, then the
    # latest-covered ones; it does not identify the last unresolved pair.
    hard = [(int(primes[j]), None if first_hit[j] < 0 else int(first_hit[j]) + 1) for j in ranked[:5]]
    final_sizes = np.bincount(classes) if n else np.array([], dtype=np.int64)
    final_sizes_s1 = np.bincount(classes_s1) if n else np.array([], dtype=np.int64)
    return {"K_star": K_star, "K_star_stage1_only": K_star_s1,
            "K_separate": K_separate, "K_separate_stage1": K_separate_s1,
            "signature_information_lower_bound": int(math.ceil(math.log2(n))) if n > 1 else 0,
            "n_methods_run": len(history),
            "history": history, "hardest_primes": hard,
            "uncovered_at_end": int(uncovered.sum()),
            "last_unresolved_pairs": last_pairs, "last_unresolved_pairs_after": last_pairs_method,
            "tail_unresolved_pairs": tail_pairs, "tail_unresolved_pairs_after": tail_pairs_method,
            "signature_classes_at_end": int(final_sizes.size),
            "unresolved_prime_pairs_at_end": int(np.sum(final_sizes * (final_sizes - 1) // 2)) if n else 0,
            "max_signature_class_at_end": int(final_sizes.max()) if n else 0,
            "stage1_signature_classes_at_end": int(final_sizes_s1.size),
            "stage1_unresolved_prime_pairs_at_end": int(np.sum(final_sizes_s1 * (final_sizes_s1 - 1) // 2)) if n else 0,
            "stage1_max_signature_class_at_end": int(final_sizes_s1.max()) if n else 0}


def theta_sample(primes: np.ndarray, B1: int, B2: int, n_curves: int = 40, sample: int = 2000, seed: int = 1) -> dict:
    """Per-prime success probability of Suyama curves (stage 1 + 2) on a sample, and the
    independent-trials prediction of the covering length for the full prime count."""
    rng = np.random.default_rng(seed)
    if primes.size == 0:
        return {"sample": 0, "n_curves": n_curves,
                "theta_mean": 0.0, "theta_quantiles": [0.0] * 4,
                "theta_min": 0.0, "fraction_theta_zero": 0.0,
                "predicted_K_star_independent_jeffreys": 0,
                "theta1_mean": 0.0, "theta1_min": 0.0,
                "theta1_fraction_zero": 0.0,
                "predicted_K_star_stage1_jeffreys": 0,
                "predicted_K_separate_independent_jeffreys": 0,
                "predicted_K_separate_stage1_jeffreys": 0,
                "signature_pair_prediction_draws": 0}
    required = min(2, int(primes.size))
    if sample < required:
        raise ValueError(f"sample must be at least {required} for a population of {primes.size} primes")
    idx = np.sort(rng.choice(primes.size, size=min(sample, primes.size), replace=False))
    ps = primes[idx]
    S = np.zeros((n_curves, ps.size), dtype=bool)
    for j, s in enumerate(range(6, 6 + n_curves)):
        s1, s2 = ecm_success(s, ps, B1, B2, None)
        S[j] = s1 | s2
    theta = S.mean(axis=0)
    # A zero among finitely many sampled curves is not evidence of true zero
    # success probability. Jeffreys smoothing is explicit and used only for
    # extrapolation; raw theta and its zero fraction remain the measurements.
    theta_j = (S.sum(axis=0) + 0.5) / (n_curves + 1.0)
    n = primes.size
    t_pred = None
    for t in range(1, 100000):
        if n * float(np.mean((1.0 - theta_j) ** t)) < 1.0:
            t_pred = t
            break
    S1only = np.zeros((n_curves, ps.size), dtype=bool)
    for j, s in enumerate(range(6, 6 + n_curves)):
        s1, _ = ecm_success(s, ps, B1, None, None)
        S1only[j] = s1
    theta1 = S1only.mean(axis=0)
    theta1_j = (S1only.sum(axis=0) + 0.5) / (n_curves + 1.0)
    t_pred1 = None
    for t in range(1, 1000000):
        if n * float(np.mean((1.0 - theta1_j) ** t)) < 1.0:
            t_pred1 = t
            break
    # Under independent Bernoulli signatures, a pair with probabilities a,b
    # remains unresolved in one coordinate with probability ab+(1-a)(1-b).
    # Monte Carlo over sampled distinct prime pairs estimates its heterogeneous
    # average; powers are accumulated to avoid repeated exponentiation.
    if ps.size < 2:
        draws = 0
        t_pred_sep = t_pred_sep_all = 0
    else:
        draws = min(100000, ps.size * (ps.size - 1) // 2)
        ii = rng.integers(0, ps.size, size=draws)
        # Draw uniformly from {0, ..., size-1} \ {ii}: draw one of size-1
        # slots, then skip over ii. This does not double a neighbouring pair.
        jj0 = rng.integers(0, ps.size - 1, size=draws)
        jj = jj0 + (jj0 >= ii)
        total_pairs = n * (n - 1) / 2.0

        def separation_threshold(theta_values):
            same = (theta_values[ii] * theta_values[jj]
                    + (1.0 - theta_values[ii]) * (1.0 - theta_values[jj]))
            powers = np.ones(draws, dtype=np.float64)
            for t in range(1, 1000000):
                powers *= same
                if total_pairs * float(powers.mean()) < 1.0:
                    return t
            return None

        t_pred_sep_all = separation_threshold(theta_j)
        t_pred_sep = separation_threshold(theta1_j)
    return {"sample": int(ps.size), "n_curves": n_curves,
            "theta_mean": float(theta.mean()), "theta_quantiles": [float(q) for q in np.quantile(theta, [0.01, 0.1, 0.5, 0.9])],
            "theta_min": float(theta.min()), "fraction_theta_zero": float((theta == 0).mean()),
            "predicted_K_star_independent_jeffreys": t_pred,
            "theta1_mean": float(theta1.mean()), "theta1_min": float(theta1.min()),
            "theta1_fraction_zero": float((theta1 == 0).mean()),
            "predicted_K_star_stage1_jeffreys": t_pred1,
            "predicted_K_separate_independent_jeffreys": t_pred_sep_all,
            "predicted_K_separate_stage1_jeffreys": t_pred_sep,
            "signature_pair_prediction_draws": int(draws)}


def _suyama_degenerate_labels(sigma: int, p: np.ndarray) -> np.ndarray:
    """0 for a regular construction; -1 if the denominator vanishes; -2 if a24 = 0 (A = -2);
    -3 if a24 = 1 (A = 2).  The three degenerate types are exposed by different gcds, so
    they must carry different labels."""
    s = np.full_like(p, sigma) % p
    u = (s * s - 5) % p
    v = (4 * s) % p
    u3 = ((u * u) % p) * u % p
    den = ((16 * u3) % p) * v % p
    d = (v - u) % p
    d3 = (((d * d) % p) * d) % p
    num = (d3 * ((3 * u + v) % p)) % p
    lab = np.zeros(p.size, dtype=np.int64)
    lab[(den != 0) & (((num - den) % p) == 0)] = -3
    lab[(den != 0) & (num == 0)] = -2
    lab[den == 0] = -1
    return lab


def residual_order_labels(sigma: int, p: np.ndarray, B1: int, B2: int) -> np.ndarray:
    """Exposure label of E_sigma modulo each prime: -1, -2, -3 for the degenerate constructions
    (denominator zero; singular with a24 = 0; singular with a24 = 1); otherwise the order d of
    the stage-1 point if d <= B2 (d = 1 is a stage-1 success); 0 if the order exceeds B2.
    `fixed_list_ecm` factors N = pq on this curve iff the labels of p and q differ
    (Proposition V' of notes_probabilistic.md); equal labels are two-sided.  Orders are
    found by walking jQ, j = 1..B2, by differential additions and recording the first j
    with Z = 0."""
    p = p.astype(np.int64)
    degenerate = _suyama_degenerate_labels(sigma, p)
    immediate = degenerate != 0
    a24, X, Z = vec_suyama(sigma, p)
    for pe in stage1_exponents(int(B1)):
        X, Z = vec_ladder(pe, X, Z, a24, p)
    labels = np.zeros(p.size, dtype=np.int64)
    labels[immediate] = degenerate[immediate]
    s1 = ((Z % p) == 0) & ~immediate
    labels[s1] = 1
    idx = np.nonzero(~immediate & ~s1)[0]
    if idx.size == 0 or int(B2) < 2:
        return labels
    pp, X1, Z1, aa = p[idx], X[idx], Z[idx], a24[idx]
    found = np.zeros(idx.size, dtype=np.int64)
    prev = (X1, Z1)
    cur = vec_xdbl(X1, Z1, aa, pp)
    hit = ((cur[1] % pp) == 0) & (found == 0)
    found[hit] = 2
    for j in range(3, int(B2) + 1):
        nxt = vec_xadd(cur[0], cur[1], X1, Z1, prev[0], prev[1], pp)  # jQ = (j-1)Q + Q, difference (j-2)Q
        prev, cur = cur, nxt
        hit = ((cur[1] % pp) == 0) & (found == 0)
        if hit.any():
            found[hit] = j
    labels[idx] = found
    return labels


def cover_residual(primes: np.ndarray, B1: int, B2: int, max_curves: int, sigma0: int = 6) -> dict:
    """Suyama curves in order; exact coverage and Proposition V' separation (classes refined by
    exposure label).  Labels are evaluated on uncovered primes and on members of
    non-singleton classes; singletons need no further labels."""
    n = primes.size
    uncovered = np.ones(n, dtype=bool)
    classes = np.zeros(n, dtype=np.int64)
    classes_bin = np.zeros(n, dtype=np.int64)
    history = []
    K_cov = None
    K_sep = None
    K_sep_bin = None
    for i in range(max_curves):
        sizes = np.bincount(classes) if n else np.array([], dtype=np.int64)
        sizes_bin = np.bincount(classes_bin) if n else np.array([], dtype=np.int64)
        active = (sizes[classes] > 1) | (sizes_bin[classes_bin] > 1) if n else np.zeros(0, dtype=bool)
        need = uncovered | active
        idx = np.nonzero(need)[0]
        lab = np.zeros(n, dtype=np.int64)
        if idx.size:
            lab[idx] = residual_order_labels(sigma0 + i, primes[idx], B1, B2)
        exposed = lab != 0
        uncovered &= ~exposed
        key = classes * (int(B2) + 4) + (lab + 3)
        _, classes = np.unique(key, return_inverse=True)
        classes = classes.astype(np.int64)
        sizes = np.bincount(classes)
        _, classes_bin = np.unique(classes_bin * 2 + exposed.astype(np.int64), return_inverse=True)
        classes_bin = classes_bin.astype(np.int64)
        sizes_bin = np.bincount(classes_bin)
        # Every member of an unresolved class has the same entire label history,
        # hence the same cumulative exposure state.  Split by that state, not by
        # the current curve's label (an earlier equal exposure may be followed by 0).
        class_uncovered = np.zeros(sizes.size, dtype=bool)
        class_uncovered[classes] = uncovered
        pairs = sizes * (sizes - 1) // 2
        history.append({"curve": i + 1, "sigma": sigma0 + i, "evaluated": int(idx.size),
                        "exposed_rate_evaluated": float(exposed[idx].mean()) if idx.size else None,
                        "uncovered_fraction": float(uncovered.mean()),
                        "classes": int(sizes.size), "unresolved_pairs": int(pairs.sum()),
                        "unresolved_pairs_unexposed": int(pairs[class_uncovered].sum()),
                        "unresolved_pairs_equal_exposure": int(pairs[~class_uncovered].sum()),
                        "classes_binary": int(sizes_bin.size),
                        "unresolved_pairs_binary": int(np.sum(sizes_bin * (sizes_bin - 1) // 2)),
                        "max_class": int(sizes.max())})
        if K_cov is None and not uncovered.any():
            K_cov = i + 1
        if K_sep is None and sizes.size == n:
            K_sep = i + 1
        if K_sep_bin is None and sizes_bin.size == n:
            K_sep_bin = i + 1
        if K_cov is not None and K_sep is not None and K_sep_bin is not None:
            break
    return {"K_cov": K_cov, "K_sep_residual": K_sep, "K_sep_binary_projection": K_sep_bin,
            "n_curves_run": len(history), "history": history,
            "uncovered_at_end": int(uncovered.sum()),
            "unresolved_pairs_at_end": history[-1]["unresolved_pairs"] if history else int(n * (n - 1) // 2)}


def residual_scaling_experiment(log2_xs: Sequence[int] = (14, 16, 18, 20), u: float = 3.0,
                                max_curves: int = 80) -> dict:
    rows = []
    for lx in log2_xs:
        x = 1 << int(lx)
        primes = primes_in_range(x, 2 * x)
        B1 = max(5, int(round(x ** (1.0 / u))))
        B2 = int(round(x ** (2.0 / u)))
        res = cover_residual(primes, B1, B2, max_curves)
        rows.append({"log2_x": int(lx), "n_primes": int(primes.size), "B1": B1, "B2": B2, "u": u, **res})
    return {"u": u, "max_curves": max_curves, "rows": rows}


def label_entropy_profile(log2_x: int = 16,
                          us: Sequence[float] = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
                          max_curves: int = 400) -> dict:
    """Exact-label separation information per unit stage-2 scale, on every prime in [x,2x).

    For each u, R_label is the fraction of prime pairs unresolved by the first
    curve, I_label = -log2 R_label is its collision information, and B1 is the
    per-curve scale.  Ksep*B1 is the observed deterministic-list work proxy.
    The binary projection is reported alongside it.
    """
    x = 1 << int(log2_x)
    primes = primes_in_range(x, 2 * x)
    total_pairs = primes.size * (primes.size - 1) / 2.0
    rows = []
    for u in us:
        B1 = max(5, int(round(x ** (1.0 / u))))
        B2 = int(round(x ** (2.0 / u)))
        res = cover_residual(primes, B1, B2, max_curves)
        h0 = res["history"][0]
        R = h0["unresolved_pairs"] / total_pairs
        Rb = h0["unresolved_pairs_binary"] / total_pairs
        info = -math.log2(max(R, 1.0 / total_pairs))
        info_b = -math.log2(max(Rb, 1.0 / total_pairs))
        rows.append({"u": float(u), "B1": B1, "B2": B2,
                     "first_curve_exposed_rate": h0["exposed_rate_evaluated"],
                     "R_label": R, "information_bits_label": info,
                     "information_per_B1": info / B1,
                     "R_binary": Rb, "information_bits_binary": info_b,
                     "K_cov": res["K_cov"], "K_sep_label": res["K_sep_residual"],
                     "K_sep_binary": res["K_sep_binary_projection"],
                     "separation_work_proxy": (res["K_sep_residual"] * B1
                                                if res["K_sep_residual"] is not None else None),
                     "coverage_work_proxy": (res["K_cov"] * B1 if res["K_cov"] is not None else None)})
    finite = [r for r in rows if r["separation_work_proxy"] is not None]
    best = min(finite, key=lambda r: r["separation_work_proxy"]) if finite else None
    return {"log2_x": int(log2_x), "n_primes": int(primes.size), "max_curves": int(max_curves),
            "best_observed_u": None if best is None else best["u"], "best": best, "rows": rows}


def greedy_label_schedule(log2_x: int = 14,
                          us: Sequence[float] = (2.5, 3.0, 3.5, 4.0, 4.5),
                          curves_per_u: int = 12, sigma0: int = 6) -> dict:
    """Greedy finite separating schedule across u.

    Candidate (u, sigma) labels are computed on every prime in [x,2x).  At
    each step choose the coordinate resolving the most currently-unresolved
    prime pairs per proxy cost B1.  This is non-uniform finite optimisation,
    not an explicit asymptotic construction; it tests whether stacking bounds
    improves constants beyond the best fixed-u prefix.
    """
    x = 1 << int(log2_x)
    primes = primes_in_range(x, 2 * x)
    candidates = []
    for u in us:
        B1 = max(5, int(round(x ** (1.0 / u))))
        B2 = int(round(x ** (2.0 / u)))
        for sigma in range(sigma0, sigma0 + int(curves_per_u)):
            lab = residual_order_labels(sigma, primes, B1, B2)
            candidates.append({"u": float(u), "sigma": sigma, "B1": B1, "B2": B2, "labels": lab})
    # Same-budget fixed-u baselines (same sigmas and proxy cost) for a controlled comparison.
    fixed = []
    for u in us:
        seq = sorted((c for c in candidates if c["u"] == float(u)), key=lambda c: c["sigma"])
        cls = np.zeros(primes.size, dtype=np.int64)
        pairs = int(primes.size * (primes.size - 1) // 2)
        cost = 0
        steps_u = 0
        for c in seq:
            key = cls * (c["B2"] + 4) + (c["labels"] + 3)
            _, cls, sizes = np.unique(key, return_inverse=True, return_counts=True)
            pairs = int(np.sum(sizes * (sizes - 1) // 2))
            cost += c["B1"]
            steps_u += 1
            if pairs == 0:
                break
        fixed.append({"u": float(u), "separated": pairs == 0, "steps": steps_u,
                      "cost_proxy": cost, "unresolved_pairs": pairs,
                      "B1": seq[0]["B1"] if seq else None})
    fixed_finite = [r for r in fixed if r["separated"]]
    best_fixed = min(fixed_finite, key=lambda r: r["cost_proxy"]) if fixed_finite else None
    classes = np.zeros(primes.size, dtype=np.int64)
    current_pairs = int(primes.size * (primes.size - 1) // 2)
    steps = []
    total_cost = 0
    while current_pairs and candidates:
        best = None
        for k, c in enumerate(candidates):
            key = classes * (c["B2"] + 4) + (c["labels"] + 3)
            _, inv, sizes = np.unique(key, return_inverse=True, return_counts=True)
            pairs = int(np.sum(sizes * (sizes - 1) // 2))
            gain = current_pairs - pairs
            score = gain / c["B1"]
            candidate = (score, gain, -c["B1"], -k, inv.astype(np.int64), pairs)
            if best is None or candidate[:4] > best[:4]:
                best = candidate
        score, gain, _, negk, inv, pairs = best
        if gain <= 0:
            break
        k = -negk
        c = candidates.pop(k)
        classes = inv
        total_cost += c["B1"]
        steps.append({"step": len(steps) + 1, "u": c["u"], "sigma": c["sigma"],
                      "B1": c["B1"], "B2": c["B2"], "gain": gain,
                      "gain_per_B1": score, "unresolved_pairs": pairs,
                      "classes": int(np.unique(classes).size), "cumulative_cost": total_cost})
        current_pairs = pairs
    separated = current_pairs == 0
    termination = "separated" if separated else ("no_positive_gain" if candidates else "candidates_exhausted")
    return {"log2_x": int(log2_x), "n_primes": int(primes.size), "us": [float(u) for u in us],
            "curves_per_u": int(curves_per_u), "separated": separated,
            "termination": termination, "n_steps": len(steps), "total_cost_proxy": total_cost,
            "unresolved_pairs": current_pairs, "steps": steps,
            "fixed_u_baselines": fixed, "best_fixed_u": best_fixed,
            "greedy_over_best_fixed_cost": (total_cost / best_fixed["cost_proxy"]
                                             if separated and best_fixed else None)}


def pair_residue_analysis(pairs: Sequence[tuple[int, int]], primes: np.ndarray,
                          moduli: Sequence[int] = (3, 4, 8, 12, 5, 7, 24)) -> dict:
    """For the last unresolved pairs: fraction with p == q (mod m) against the chance
    rate for a uniformly random pair of *distinct* population primes,
    sum_r n_r (n_r - 1) / (n (n - 1)), per modulus."""
    out = {"n_pairs": len(pairs), "per_modulus": {}}
    n = int(primes.size)
    if not pairs or n < 2:
        return out
    P = np.array(pairs, dtype=np.int64)
    for m in moduli:
        counts = np.bincount(primes % m, minlength=m).astype(np.float64)
        chance = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
        same = float(np.mean((P[:, 0] % m) == (P[:, 1] % m)))
        out["per_modulus"][str(m)] = {"same_class_fraction": same, "chance": chance,
                                     "ratio": same / chance if chance > 0 else None}
    return out


def hitting_scaling_experiment(log2_xs: Sequence[int] = (14, 16, 18, 20, 22), u: float = 3.0,
                               max_curves: int = 60, sample: int = 2000,
                               stage2_through_bits: int | None = 18) -> dict:
    rows = []
    for lx in log2_xs:
        x = 1 << int(lx)
        primes = primes_in_range(x, 2 * x)
        B1 = max(5, int(round(x ** (1.0 / u))))
        B2 = int(round(x ** (2.0 / u)))
        det_s1 = cover(primes, B1, None, True, max_curves)
        ecm_s1 = cover(primes, B1, None, False, max_curves)
        if stage2_through_bits is not None and lx <= stage2_through_bits:
            det_s2 = cover(primes, B1, B2, True, max_curves)
            ecm_s2 = cover(primes, B1, B2, False, max_curves)
        else:
            det_s2 = ecm_s2 = None
        th = theta_sample(primes, B1, B2, n_curves=min(40, max_curves), sample=sample)
        rows.append({"log2_x": int(lx), "n_primes": int(primes.size), "B1": B1, "B2": B2, "u": u,
                     "with_deterministic_stage1": det_s1, "ecm_only_stage1": ecm_s1,
                     "with_deterministic_stage2": det_s2, "ecm_only_stage2": ecm_s2,
                     "theta": th, "log_n_primes": math.log(primes.size)})
    return {"u": u, "max_curves": max_curves, "stage2_through_bits": stage2_through_bits, "rows": rows}
