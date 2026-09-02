"""The price of oracle calls in the Poisson block world (docs/notes_lattice_barrier.md, section 7), with the exact tail identities.

The per-basis decomposition  l_1 - S/d = h_{d,beta}(0) - sum_i y_i eps_i(B) + R(B)  makes the y-weighted sub-GH mass  M := sum_i y_i eps_i
the quantity that a reduction algorithm must accumulate to beat the zero-slack floor.  In the Poisson block world every oracle query at
position i returns a block whose ratio eps = log(GH/lambda_1) is an independent draw from the sign-pair Poisson law, eps = -(1/beta_i) log(2X),
X ~ Exp(1); a position queried N_i times keeps the best of its draws, so its statistic is -(1/beta_i) log(2 X_i) with X_i ~ Exp(N_i), and

    E[M] = sum_i (y_i / beta_i) (log(N_i / 2) + gamma).

The logarithmic marginal price of calls is therefore  A_{d,beta} := sum_i y_i / beta_i.

Exact tail identity (`tail_identity_check`).  From the two-term recurrence of the positivity theorem, for 1 <= m <= beta - 1,

    y_{d-m} = (1 / (m beta)) sum_{j=1}^{min(m, d-beta)} j y_{d-beta+1-j} :

each shrinking-tail multiplier is a triangular average of the full-block multipliers feeding it.  Consequently

    A_{d,beta} = (1/beta) [ sum_{k <= d-beta+1} y_k  +  sum_{j=1}^{min(beta-2, d-beta)} (1 - j/(beta-1)) y_{d-beta+1-j} ],
    (d-1)/(beta(beta-1)) <= A_{d,beta} <= 2(d-1)/(beta(beta-1)),        rho := max_i y_i/beta_i <= (d-1)/(beta(beta-1)),

the lower bound strict for beta >= 3 (equality at beta = 2).  `call_price` returns all of these exactly.

Call budget (`expected_mass`, `optimal_fixed_mass`, `adaptive_bound`).  With the normalised weights w_i := (y_i/beta_i)/A (sum 1) and their
Shannon entropy H(w) := -sum_i w_i log w_i, Gibbs' inequality gives, for every allocation with sum N_i = N,
    sum_i w_i log N_i <= log N - H(w),   with equality iff N_i = N w_i,
so E[M] <= A (log(N/2) + gamma - H(w)) for every fixed allocation, attained by the proportional allocation whenever N w_i >= 1 for all i
(the water-filling optimum with no floor binding), and the number of calls that consumes a mass m is  N*(m) = 2 exp(m/A + H(w) - gamma).
For a fully adaptive algorithm with deterministic total N (the position of each query may depend on all earlier answers) with every position
queried at least once pathwise, predictable sampling gives  E sum_t 1[p_t = i](H_t - L)^+ = E[tau_i] E(H - L)^+  (the indicator is fixed before the
answer is drawn; the local count tau_i need not be a stopping time), and with E (H - L)^+ <= e^{-L}, L = log E[tau_i], this yields
E[max_{j <= tau_i} H_ij] <= log E[tau_i] + 1 for standard Gumbel H_j; hence E[M] <= A (log(N/2) + 1 - H(w)): adaptivity is
worth at most a factor e^{1-gamma} = 1.53 in calls, in expectation.  Since every y_i > 0 the mass is monotone in each eps_i, so 'best of the
draws' dominates every selection rule whose output at i is one of the answers received there (a final block changed by later insertions is
outside the model).  Dropping H(w) >= 0 gives the crude bounds A (log(N/2) + gamma) and A (log(N/2) + 1).  The proportional allocation attains
the fixed-allocation bound in the continuous relaxation; integer rounding effects are not bounded here.

Concentration (`concentration`).  M - E[M] = sum_i (y_i/beta_i) G_i with independent centred standard Gumbel G_i for a fixed allocation that keeps
the best answer, so Var M = (pi^2/6) sum_i (y_i/beta_i)^2 <= (pi^2/6) rho A, and  P[M >= E M + t] <= exp(-(t - zeta(2) A / 2) / (2 rho))  (log Gamma(1-a) -
gamma a <= zeta(2) a^2 / (2(1-a)) with a = s y_i/beta_i, s = 1/(2 rho)).

Adaptive high-probability bound (`adaptive_tail_bound`, `simulate_adaptive`).  For ANY adaptive N-query algorithm (every position queried at
least once, output at i one of the answers received there), with the uniform level L = log N:  M <= A (L - log 2) + Z,  Z := sum_t c_{p_t}
(H_t - L)^+, c_i = y_i/beta_i.  Given the past, E[e^{s c (H-L)^+}] <= 1 + s c e^{-L} / (1 - s c) (P[H > L + u] <= e^{-L-u}), so by the tower property
log E e^{sZ} <= N (s rho e^{-L}) / (1 - s rho) = s rho / (1 - s rho) for s < 1/rho, and Markov at s = 1/(2 rho) gives
    P[M >= A log(N/2) + t] <= exp(1 - t / (2 rho)),      rho = y_1 / beta,
while E[Z] <= rho gives E[M] <= A log(N/2) + rho (weaker than the entropy bound in expectation, but the tail bound holds pathwise).

The consistent uniform-completion world (`uniform_world_run`).  A profile l (a bona fide state: any real vector is the log Gram-Schmidt profile
of some basis) evolves by insertions: a query at position i draws eps from the sign-pair law, and if g_i - eps < l_i (g_i := log chat(beta_i) +
avg_{block i} l, the GH value of the block) sets l_i <- g_i - eps and spreads the removed mass uniformly over the other beta_i - 1 positions of the
block (volume preserved).  The profile violation v_i := g_i - l_i then satisfies, pathwise: an insertion at i sets v_i <- eps if eps > v_i (else
nothing changes), and an insertion at any other position j never increases v_i (the proof of the preservation proposition: the raised
positions of block j and the lowered position j change block i's mean by at most the change of l_i when j < i, and lower block i's mean when
j > i).  Hence from an admissible start (v_i(0) <= 0 for all i) v_i(t) <= max(0, max_{queries at i} eps) at every time, and since
l_1 - S/d = h(0) - sum_i y_i v_i identically,
    l_1 - S/d  >=  h(0) - sum_i y_i (max_j eps_ij)^+   at all times, for every schedule,
with no realization assumption: the expected and pathwise call-budget bounds hold for the constructed profile (the adaptive tail bound
verbatim, since (max_j H_ij - log 2)^+ <= (L - log 2) + sum_t 1[p_t = i](H_t - L)^+ for L >= log 2, and unqueried positions have v_i <= 0).
The uniform completion is an abstraction of the physical completion (section 6 shows physical insertions do create violations in neighbouring
blocks), so this closes the realization gap for the model, not for real bases.

The model defines a collection of random block statistics; it does not construct a geometrically consistent basis realising them, and the
sign-pair Poisson law is a random-lattice limit law applied heuristically to projected blocks.  Everything here is exact inside that model.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, Sequence

import numpy as np

from latticelab.profile_floor import block_sizes, dual_certificate as _dual_certificate_uncached, log_chat

EULER_GAMMA = float(np.euler_gamma)
ZETA2 = math.pi ** 2 / 6

_CERT_CACHE: Dict = {}


def dual_certificate(d: int, beta: int):
    """profile_floor.dual_certificate with a process-level cache (the exact recurrence costs seconds at d ~ 2000 and is reused here)."""
    key = (d, beta)
    if key not in _CERT_CACHE:
        _CERT_CACHE[key] = _dual_certificate_uncached(d, beta)
    return _CERT_CACHE[key]


def tail_identity_check(d: int, beta: int) -> bool:
    """Exact check of  y_{d-m} = (1/(m beta)) sum_{j=1}^{min(m, d-beta)} j y_{d-beta+1-j}  for every 1 <= m <= beta - 1 (2 <= beta < d),
    in flint fmpq arithmetic with running prefix sums (O(beta) exact operations)."""
    from flint import fmpq

    if not (2 <= beta < d):
        raise ValueError("need 2 <= beta < d")
    y, _ = dual_certificate(d, beta)
    Y = {k: fmpq(y[k - 1].numerator, y[k - 1].denominator) for k in range(1, d)}
    prefix = fmpq(0)  # sum_{j <= min(m, d-beta)} j y_{d-beta+1-j}
    for m in range(1, beta):
        if m <= d - beta:
            prefix += fmpq(m) * Y[d - beta + 1 - m]
        if Y[d - m] * (m * beta) != prefix:
            return False
    return True


def call_price(d: int, beta: int) -> Dict:
    """A_{d,beta} = sum_i y_i/beta_i exactly (Fraction), its tail decomposition, the bounds of the lemma, rho = max_i y_i/beta_i, and the
    forward discrete pure-GSA slope of the head, |Delta_beta l_1^GSA| = (d-1)[log chat(beta)/(beta-1) - log chat(beta+1)/beta], which
    converts mass into local blocksize units (`blocks_per_e_calls` = A / |Delta|; for beta + 1 <= d)."""
    if not (2 <= beta < d):
        raise ValueError("need 2 <= beta < d")
    y, _ = dual_certificate(d, beta)
    bs = block_sizes(d, beta)
    A = sum((yi / b for yi, b in zip(y, bs)), Fraction(0))
    full = sum((y[k - 1] for k in range(1, d - beta + 2)), Fraction(0))  # positions with beta_k = beta
    tri = sum((Fraction(beta - 1 - j, beta - 1) * y[d - beta + 1 - j - 1] for j in range(1, min(beta - 2, d - beta) + 1)), Fraction(0))
    assert A == (full + tri) / beta, "tail decomposition of A failed"
    lower, upper = Fraction(d - 1, beta * (beta - 1)), Fraction(2 * (d - 1), beta * (beta - 1))
    assert lower <= A <= upper, "bounds on A failed"
    rho = max(yi / b for yi, b in zip(y, bs))
    assert rho <= lower, "bound on rho failed"
    slope = (d - 1) * (log_chat(beta) / (beta - 1) - log_chat(beta + 1) / beta) if beta + 1 <= d else float("nan")
    return {"d": d, "beta": beta, "A": A, "A_float": float(A), "full_block_mass": full, "tail_triangular_sum": tri,
            "lower": lower, "upper": upper, "ratio_to_lower": float(A / lower), "rho": rho, "rho_float": float(rho),
            "sum_y": float(sum(y, Fraction(0))), "gsa_forward_slope_abs": abs(slope),
            "blocks_per_e_calls": float(A) / abs(slope) if slope else float("nan"), "y": y, "block_sizes": bs}


def expected_mass(d: int, beta: int, N_alloc: Sequence[float]) -> float:
    """E[sum_i y_i eps_i] = sum_i (y_i/beta_i)(log(N_i/2) + gamma) for a fixed allocation N_alloc (length d-1, each >= 1)."""
    y, _ = dual_certificate(d, beta)
    bs = block_sizes(d, beta)
    if len(N_alloc) != d - 1 or any(n < 1 for n in N_alloc):
        raise ValueError("allocation must give every position at least one call")
    return float(sum(float(yi) / b * (math.log(n / 2) + EULER_GAMMA) for yi, b, n in zip(y, bs, N_alloc)))


def waterfilling(d: int, beta: int, N: float) -> Dict:
    """The continuous relaxation of the optimal allocation: N_i = max(1, c y_i/beta_i) with c fixed by sum N_i = N (Lagrange, concavity of
    the logarithm).  Returns the allocation, its expected mass, the uniform allocation's mass, and the fixed-allocation bound
    A (log(N/2) + gamma) that neither can exceed."""
    y, _ = dual_certificate(d, beta)
    bs = block_sizes(d, beta)
    w = np.array([float(yi) / b for yi, b in zip(y, bs)])
    if N < d - 1:
        raise ValueError("need at least one call per position")
    A = float(w.sum())
    # bisection on c: sum max(1, c w_i) = N
    lo, hi = 0.0, N / w.min() + 1
    for _ in range(200):
        c = (lo + hi) / 2
        if np.maximum(1.0, c * w).sum() > N:
            hi = c
        else:
            lo = c
    alloc = np.maximum(1.0, lo * w)
    bound = A * (math.log(N / 2) + EULER_GAMMA)
    ex = expected_mass(d, beta, alloc)
    uni = expected_mass(d, beta, [N / (d - 1)] * (d - 1))
    assert ex <= bound + 1e-12 and uni <= ex + 1e-12
    ent = optimal_fixed_mass(d, beta, N)
    assert ex <= ent + 1e-9  # the entropy form is the unconstrained optimum; equal when no floor binds
    return {"N": N, "allocation": alloc, "expected_mass": ex, "uniform_mass": uni, "fixed_allocation_bound": bound, "entropy_optimum": ent,
            "floor_binding": int(np.sum(alloc <= 1.0 + 1e-12)), "A": A}


def weight_entropy(d: int, beta: int) -> Dict:
    """The normalised weights w_i = (y_i/beta_i)/A, their Shannon entropy H(w) = -sum w_i log w_i (nats), the min-entropy -log max w_i,
    and the largest N below which the proportional allocation N w_i has some position under one call (N_floor = 1/min_i w_i)."""
    y, _ = dual_certificate(d, beta)
    bs = block_sizes(d, beta)
    w = np.array([float(yi) / b for yi, b in zip(y, bs)])
    A = float(w.sum())
    w = w / A
    return {"A": A, "w": w, "entropy": float(-(w * np.log(w)).sum()), "min_entropy": float(-math.log(w.max())), "max_entropy": math.log(d - 1),
            "N_floor": float(1 / w.min())}


def optimal_fixed_mass(d: int, beta: int, N: float) -> float:
    """A (log(N/2) + gamma - H(w)): the exact maximum of E[mass] over fixed allocations with sum N_i = N when N >= N_floor (no position under
    one call); for smaller N it is still an upper bound (the constrained optimum is the water-filling value, which is lower)."""
    we = weight_entropy(d, beta)
    return we["A"] * (math.log(N / 2) + EULER_GAMMA - we["entropy"])


def calls_to_consume(d: int, beta: int, mass: float) -> Dict:
    """N*(m) = 2 exp(m/A + H(w) - gamma): the number of calls whose optimal fixed allocation has expected mass m (valid when N* >= N_floor);
    the adaptive analogue divides by e^{1-gamma}."""
    we = weight_entropy(d, beta)
    N = 2 * math.exp(mass / we["A"] + we["entropy"] - EULER_GAMMA)
    return {"mass": mass, "calls": N, "log2_calls": math.log2(N), "log2_calls_adaptive_lower_bound": math.log2(N / math.exp(1 - EULER_GAMMA)),
            "valid": N >= we["N_floor"], "N_floor": we["N_floor"]}


def adaptive_bound(d: int, beta: int, N: float, with_entropy: bool = True) -> float:
    """A_{d,beta} (log(N/2) + 1 - H(w)) (or without the entropy term): the bound on E[mass] for every adaptive algorithm making N queries with
    each position queried at least once (Wald's identity at level L = log n_i, n_i = E[tau_i], then Gibbs over positions)."""
    if N < d - 1:
        raise ValueError("need at least one call per position")
    we = weight_entropy(d, beta)
    return we["A"] * (math.log(N / 2) + 1.0 - (we["entropy"] if with_entropy else 0.0))


def concentration(d: int, beta: int, t: float) -> Dict:
    """Var M = (pi^2/6) sum_i (y_i/beta_i)^2 (exact in the model) and the tail bound P[M >= E M + t] <= exp(-(t - zeta(2) A/2) / (2 rho))."""
    y, _ = dual_certificate(d, beta)
    bs = block_sizes(d, beta)
    w = np.array([float(yi) / b for yi, b in zip(y, bs)])
    A, rho = float(w.sum()), float(w.max())
    var = ZETA2 * float((w ** 2).sum())
    tail = math.exp(-(t - ZETA2 * A / 2) / (2 * rho)) if t > ZETA2 * A / 2 else 1.0
    return {"variance": var, "sd": math.sqrt(var), "variance_bound": ZETA2 * rho * A, "tail_bound": min(1.0, tail), "t": t, "A": A, "rho": rho}


def adaptive_tail_bound(d: int, beta: int, N: int, t: float) -> Dict:
    """For an integer deterministic query count N >= d - 1: the pathwise bound for every adaptive N-query algorithm, P[M >= A log(N/2) + t] <=
    exp(1 - t/(2 rho)), rho = y_1/beta (uniform level L = log N, conditional MGF of the excesses, tower property, Markov at s = 1/(2 rho)).
    Returns the threshold A log(N/2) + t and the bound.  (Evaluating the formula at a non-integer N would only be a numerical interpolation and
    is refused.)"""
    N = _as_count(N, "N", d - 1)
    if t < 0:
        raise ValueError("the tail bound is stated for t >= 0")
    cp = call_price(d, beta)
    A, rho = cp["A_float"], cp["rho_float"]
    return {"threshold": A * math.log(N / 2) + t, "tail_bound": min(1.0, math.exp(1 - t / (2 * rho))), "A_log_N_over_2": A * math.log(N / 2),
            "expectation_bound_uniform_level": A * math.log(N / 2) + rho, "rho": rho, "A": A}


def simulate_adaptive(d: int, beta: int, N: int, trials: int, seed: int = 0, rule: str = "greedy") -> Dict:
    """Monte Carlo of an adaptive strategy in the Poisson block world: after one draw per position, each further query goes to the position
    maximising the marginal expected gain c_i (log((n_i+1)/2) - log(n_i/2)) ('greedy' on the fixed-allocation formula, i.e. water-filling
    online) or, for 'chase', to the position whose current best answer is worst relative to its weight c_i * (current max) (a state-dependent
    rule) -- to the position whose current best answer is worst relative to its weight (argmin of c_i * best_i, a state-dependent rule).
    Returns the empirical mean, sd, max and the 0.5/0.9/0.99 quantiles of M, and the fraction of trials with M >= A log(N/2) + t for a few t,
    against the bound exp(1 - t/(2 rho))."""
    if trials <= 0:
        raise ValueError("need at least one trial")
    if rule not in ("greedy", "chase"):
        raise ValueError(rule)
    rng = np.random.default_rng(seed)
    y, _ = dual_certificate(d, beta)
    bs = block_sizes(d, beta)
    c = np.array([float(yi) / b for yi, b in zip(y, bs)])
    A, rho = float(c.sum()), float(c.max())
    m = d - 1
    if N < m:
        raise ValueError("need at least one call per position")
    Ms = np.empty(trials)
    for tr in range(trials):
        H = -np.log(rng.exponential(size=m))  # one standard Gumbel answer per position
        best = H.copy()
        n = np.ones(m)
        for _ in range(N - m):
            if rule == "greedy":
                i = int(np.argmax(c * np.log((n + 1) / n)))
            elif rule == "chase":
                i = int(np.argmin(c * best))  # the weighted-worst position
            else:
                raise ValueError(rule)
            h = -math.log(rng.exponential())
            n[i] += 1
            if h > best[i]:
                best[i] = h
        Ms[tr] = float(np.dot(c, best - math.log(2)))
    base = A * math.log(N / 2)
    out = {"rule": rule, "N": N, "trials": trials, "mean_M": float(Ms.mean()), "sd_M": float(Ms.std()), "max_M": float(Ms.max()),
           "quantiles": {q: float(np.quantile(Ms, q)) for q in (0.5, 0.9, 0.99)},
           "A_log_N_over_2": base, "expectation_bound_entropy": adaptive_bound(d, beta, N), "exceedances": {}}
    for t in (0.0, 0.5 * rho, 2 * rho, 4 * rho, 8 * rho):
        out["exceedances"][round(t, 6)] = {"empirical": float(np.mean(Ms >= base + t)), "bound": min(1.0, math.exp(1 - t / (2 * rho)))}
    return out


def _as_count(value, name: str, minimum: int) -> int:
    """Validate an integer-valued count exactly (no float round trip): rejects bool, non-finite and fractional values (including exact
    rationals above float precision) and values below `minimum`; returns the int."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a bool")
    try:
        n = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite integer") from exc
    if n != value:
        raise ValueError(f"{name} must be an integer (got {value!r})")
    if n < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return n


def uniform_world_run(d: int, beta: int, N: int, seed: int = 0, schedule: str = "random", delta0: float = 1.0219, check_every: int = 1) -> Dict:
    """One run of the consistent uniform-completion Poisson world from the LLL-like line (admissible for every beta): N queries at positions
    chosen by `schedule` ('random': uniform i.i.d. positions; 'tours': cyclic 0, 1, ..., d-2; 'head': the position of largest current weighted
    violation deficit c_i (v_i^+ - best_i^+), an adaptive rule), each answered by an independent sign-pair draw eps = -(1/beta_i) log(2X).
    Records the final profile, the violations v_i = g_i - l_i, the best draw per position, the coordinatewise pathwise check
    max_t max_i [v_i(t) - max(0, best_i(t))] <= 0 evaluated after EVERY query (claim (a) of the consistent-world theorem), the aggregate
    sum_i y_i v_i(t)^+ - sum_i y_i best_i(t)^+ at the sampled checkpoints (every `check_every` queries and at the end), the identity gap
    l_1 - S/d - (h(0) - sum y_i v_i), and the head against the zero-slack floor."""
    N = _as_count(N, "N", 0)
    check_every = _as_count(check_every, "check_every", 1)
    rng = np.random.default_rng(seed)
    y, _ = dual_certificate(d, beta)
    yf = np.array([float(v) for v in y])
    bs = block_sizes(d, beta)
    c = yf / np.array(bs, dtype=float)
    A = float(c.sum())
    Lc = np.array([log_chat(b) for b in bs])
    l = -2.0 * math.log(delta0) * np.arange(d, dtype=float)
    l -= l.mean()  # sum zero
    h0 = float(sum(float(yi) * log_chat(b) for yi, b in zip(y, bs)))

    def viol(lv):
        return np.array([Lc[i] + lv[i:i + bs[i]].mean() - lv[i] for i in range(d - 1)])

    v0 = viol(l)
    if v0.max() > 1e-12:
        raise ValueError("initial profile is not admissible")
    best = np.full(d - 1, -np.inf)
    worst_gap = float(np.dot(yf, np.maximum(v0, 0)))  # = 0 - 0 before any query (no draws yet)
    worst_coord_gap = float(np.max(v0))  # max_i (v_i - max(0, best_i)) with no draws: max_i v_i(0) <= 0
    for t in range(N):
        if schedule == "random":
            i = int(rng.integers(d - 1))
        elif schedule == "tours":
            i = t % (d - 1)
        elif schedule == "head":
            v = viol(l)
            i = int(np.argmax(c * (np.maximum(v, 0) - np.maximum(best, 0))))  # where the profile lags its own best draw most, weighted
        else:
            raise ValueError(schedule)
        eps = -math.log(2 * rng.exponential()) / bs[i]
        best[i] = max(best[i], eps)
        g = Lc[i] + l[i:i + bs[i]].mean()
        if g - eps < l[i]:
            m = l[i] - (g - eps)
            l[i] = g - eps
            l[i + 1:i + bs[i]] += m / (bs[i] - 1)
        v = viol(l)
        worst_coord_gap = max(worst_coord_gap, float(np.max(v - np.maximum(best, 0.0))))  # claim (a), every query, every coordinate
        if (t + 1) % check_every == 0 or t == N - 1:
            worst_gap = max(worst_gap, float(np.dot(yf, np.maximum(v, 0)) - np.dot(yf, np.maximum(best, 0))))
    v = viol(l)
    head = float(l[0] - l.mean())
    mass_profile = float(np.dot(yf, v))
    mass_profile_pos = float(np.dot(yf, np.maximum(v, 0)))
    mass_best_pos = float(np.dot(yf, np.maximum(best, 0)))
    return {"d": d, "beta": beta, "N": N, "schedule": schedule, "final_profile": l, "violations": v, "best_draws": best,
            "head_minus_mean": head, "floor_h0": h0, "head_minus_floor": head - h0, "mass_profile": mass_profile,
            "mass_profile_positive": mass_profile_pos, "mass_best_positive": mass_best_pos, "pathwise_worst_gap": worst_gap,
            "coordinatewise_worst_gap": worst_coord_gap,
            "identity_gap": head - (h0 - mass_profile), "A_log_N_over_2": A * math.log(N / 2) if N >= 2 else None,
            "queried_positions": int(np.sum(np.isfinite(best))),
            "adaptive_bound_entropy": adaptive_bound(d, beta, N) if N >= d - 1 else None}


def uniform_world_tail_check(d: int, beta: int, N: int, trials: int, seed: int = 0, schedule: str = "head") -> Dict:
    """Monte Carlo of the adaptive tail bound on constructed profiles: the fraction of runs with  h(0) - (l_1 - S/d) >= A log(N/2) + t
    against exp(1 - t/(2 rho)), for a few t, together with the largest coordinatewise gap max_t max_i [v_i - max(0, best_i)] seen over all
    queries of all runs (must be <= 0; claim (a))."""
    if trials <= 0:
        raise ValueError("need at least one trial")
    N = _as_count(N, "N", 2)  # the adaptive tail bound uses the level L = log N >= log 2
    cp = call_price(d, beta)
    A, rho = cp["A_float"], cp["rho_float"]
    deficits, worst = [], -np.inf
    for tr in range(trials):
        r = uniform_world_run(d, beta, N, seed=seed + tr, schedule=schedule, check_every=max(1, N // 50))
        deficits.append(r["floor_h0"] - r["head_minus_mean"])
        worst = max(worst, r["coordinatewise_worst_gap"])
    deficits = np.array(deficits)
    base = A * math.log(N / 2)
    out = {"schedule": schedule, "N": N, "trials": trials, "mean_deficit": float(deficits.mean()), "max_deficit": float(deficits.max()),
           "A_log_N_over_2": base, "pathwise_worst_gap": float(worst), "exceedances": {}}
    for t in (0.0, 2 * rho, 4 * rho, 8 * rho):
        out["exceedances"][round(t, 6)] = {"empirical": float(np.mean(deficits >= base + t)), "bound": min(1.0, math.exp(1 - t / (2 * rho)))}
    return out


def gumbel_stopped_max_check(n: int, trials: int = 20000, seed: int = 0) -> Dict:
    """Monte Carlo illustration of the Wald bound on one position.  For the adaptive rule 'stop at the first draw above L = log n' the
    stopping time tau is geometric with P[H > L] = p = 1 - exp(-1/n), so E[tau] = 1/p, and the stopped maximum is the last draw, whose law
    is H conditioned on H > L (inverse cdf x = -log(-log(F(L) + u p)), F(x) = exp(-e^{-x})).  Returns the empirical mean of the stopped
    maximum against the bound log(E tau) + 1 and the fixed-n formula log n + gamma.  This experiment has a random total stopping time and
    illustrates the expected-budget level bound; it is not a sharpness witness for the deterministic-total adaptive theorem."""
    rng = np.random.default_rng(seed)
    L = math.log(n)
    FL = math.exp(-math.exp(-L))
    p = 1 - FL  # P[H > L]
    taus = rng.geometric(p, size=trials)
    u = rng.random(trials)
    last = -np.log(-np.log(FL + u * p))
    mean_max = float(last.mean())
    return {"n": n, "E_tau": 1 / p, "mean_max": mean_max, "wald_bound": math.log(1 / p) + 1, "fixed_n_formula": math.log(n) + EULER_GAMMA,
            "mean_tau_empirical": float(taus.mean()), "sd_max": float(last.std())}


def main(argv=None):
    """CLI: `python -m latticelab.poisson_world --points 1003,403 1424,625 1885,877 --calls 20 40 --out results/lattice_poisson_world_bounds.json`
    records the exact call price, its bounds, the water-filling optimum and the adaptive bound at 2^calls queries."""
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser(description="call price and call-budget bounds in the Poisson block world")
    ap.add_argument("--points", nargs="+", default=["1003,403", "1424,625", "1885,877"])
    ap.add_argument("--calls", nargs="+", type=float, default=[13.5, 20, 30, 40, 60])
    ap.add_argument("--kappa", nargs="+", type=float, default=[0.0593, 0.0862, 0.1061], help="|kappa| per point, to price its consumption")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = {"note": "exact call price A = sum y_i/beta_i with the tail identity and bounds (d-1)/(beta(beta-1)) <= A <= 2(d-1)/(beta(beta-1)); "
                   "entropy form of the fixed-allocation optimum A(log(N/2)+gamma-H(w)), adaptive bound A(log(N/2)+1-H(w)), water-filling; "
                   "calls needed to consume |kappa|; concentration", "rows": []}
    for idx, pt in enumerate(a.points):
        d, beta = (int(x) for x in pt.split(","))
        cp = call_price(d, beta)
        we = weight_entropy(d, beta)
        assert tail_identity_check(d, beta)
        kap = a.kappa[idx] if idx < len(a.kappa) else None
        row = {"d": d, "beta": beta, "A": cp["A_float"], "A_exact": str(cp["A"]), "lower": float(cp["lower"]), "upper": float(cp["upper"]),
               "ratio_to_lower": cp["ratio_to_lower"], "rho": cp["rho_float"], "rho_is_y1_over_beta": cp["rho"] == cp["y"][0] / beta,
               "sum_y": cp["sum_y"], "gsa_forward_slope_abs": cp["gsa_forward_slope_abs"], "blocks_per_e_calls": cp["blocks_per_e_calls"],
               "entropy": we["entropy"], "min_entropy": we["min_entropy"], "max_entropy": we["max_entropy"], "N_floor": we["N_floor"],
               "sd_mass": concentration(d, beta, 0.0)["sd"], "calls_to_consume_kappa": calls_to_consume(d, beta, kap) if kap else None, "budgets": []}
        for c in a.calls:
            N = 2.0 ** c
            wf = waterfilling(d, beta, N)
            row["budgets"].append({"log2_calls": c, "waterfilling_mass": wf["expected_mass"], "uniform_mass": wf["uniform_mass"],
                                   "floor_binding": wf["floor_binding"],
                                   "entropy_optimum": wf["entropy_optimum"], "crude_fixed_bound": wf["fixed_allocation_bound"],
                                   "adaptive_bound": adaptive_bound(d, beta, N), "adaptive_bound_no_entropy": adaptive_bound(d, beta, N, False),
                                   "adaptive_bound_blocks": adaptive_bound(d, beta, N) / cp["gsa_forward_slope_abs"],
                                   "tail_prob_mass_exceeds_mean_by_0.02": concentration(d, beta, 0.02)["tail_bound"]})
        out["rows"].append(row)
        print(f"({d}, {beta}): A = {cp['A_float']:.6f} in [{float(cp['lower']):.6f}, {float(cp['upper']):.6f}] (ratio to lower {cp['ratio_to_lower']:.4f}); "
              f"rho = {cp['rho_float']:.3e}; H(w) = {we['entropy']:.3f} (min-entropy {we['min_entropy']:.3f}, log(d-1) = {we['max_entropy']:.3f}); "
              f"N_floor = 2^{math.log2(we['N_floor']):.2f}; sd(mass) = {row['sd_mass']:.4f}; blocks per factor e = {cp['blocks_per_e_calls']:.3f}"
              + (f"; calls to consume |kappa| = {kap}: 2^{row['calls_to_consume_kappa']['log2_calls']:.2f} (adaptive >= 2^{row['calls_to_consume_kappa']['log2_calls_adaptive_lower_bound']:.2f})" if kap else ""), flush=True)
        for b in row["budgets"]:
            print(f"   2^{b['log2_calls']:g} calls: water-filling {b['waterfilling_mass']:.4f} (positions at the one-call floor: {b['floor_binding']}), entropy optimum {b['entropy_optimum']:.4f}, uniform {b['uniform_mass']:.4f}, "
                  f"crude fixed bound {b['crude_fixed_bound']:.4f}, adaptive bound {b['adaptive_bound']:.4f} ({b['adaptive_bound_blocks']:.1f} blocksizes)", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
