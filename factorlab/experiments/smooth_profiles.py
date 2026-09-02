"""E6: success probability of factoring methods as a function of the cost exponent.

For a method A and cost budget N^c define s_A(c) = Pr[A factors N within N^c
operations], N drawn from the RSA-style family.  This module computes

* the Dickman function rho(u) (integral equation, trapezoid on a fine grid);
* the Bach-Peralta semismooth probability
      G(alpha, beta) = rho(1/alpha) + int_alpha^beta rho((1 - t)/alpha) dt / t,
  the asymptotic probability that an integer x has largest prime factor
  <= x^beta and second largest (with multiplicity) <= x^alpha;
* a brute-force semismooth count for validating G;
* empirical semismoothness of p-1, p+1, q-1, q+1 for RSA-style primes, with
  random-integer and random-even-integer controls of the same size.  With
  stage 1 at B1 = N^c (cost ~ 1.44 B1 mulmods) and a BSGS stage 2 to
  B2 = B1^2 (cost ~ 2 sqrt(B2) = 2 N^c), the p-1 method succeeds iff p-1 is
  (B1, B2)-semismooth; the heuristic prediction (shifted primes behave like
  random integers of their size) is G(2c, 4c) because p ~ N^{1/2};
* the rho length (tail + cycle) of x -> x^2 + c mod p, i.e. Pollard rho's
  collision time, whose scale sqrt(p) makes success at cost N^c a threshold
  phenomenon at c = 1/4;
* Fermat's exact step count (p + q)/2 - ceil(sqrt N), threshold at c = 1/2.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
from sympy import factorint

from ..gen import make_semiprime, random_prime_in
from ..numth import mpz, isqrt, isqrt_ceil, gcd, powmod, jacobi, small_primes
from ..algorithms.rho import lucas_v

# ---------------------------------------------------------------------------
# Dickman rho and semismooth probabilities
# ---------------------------------------------------------------------------

_RHO_H = 1e-3
_RHO_UMAX = 120.0
_RHO_TABLE: tuple[np.ndarray, np.ndarray] | None = None


def _dickman_table():
    """Tabulate rho on a grid of step _RHO_H via u rho'(u) = -rho(u - 1)."""
    global _RHO_TABLE
    if _RHO_TABLE is not None:
        return _RHO_TABLE
    h = _RHO_H
    m = int(round(1.0 / h))
    n = int(round(_RHO_UMAX / h)) + 1
    t = np.arange(n) * h
    rho = np.empty(n)
    rho[: m + 1] = 1.0
    rho[m + 1: 2 * m + 1] = 1.0 - np.log(t[m + 1: 2 * m + 1])
    for i in range(2 * m + 1, n):
        rho[i] = rho[i - 1] - 0.5 * h * (rho[i - 1 - m] / t[i - 1] + rho[i - m] / t[i])
    _RHO_TABLE = (t, rho)
    return _RHO_TABLE


def dickman_rho(u: float) -> float:
    """Dickman's function: the asymptotic probability that an integer x is x^{1/u}-smooth."""
    if u <= 1.0:
        return 1.0
    if u <= 2.0:
        return 1.0 - math.log(u)
    t, rho = _dickman_table()
    if u >= t[-1]:
        return 0.0
    return float(np.interp(u, t, rho))


def semismooth_G(alpha: float, beta: float, n: int = 4000) -> float:
    """Bach-Peralta G(alpha, beta): P(largest prime factor <= x^beta, second largest <= x^alpha)."""
    if not (0 < alpha <= beta):
        raise ValueError("need 0 < alpha <= beta")
    beta = min(beta, 1.0)
    base = dickman_rho(1.0 / alpha)
    if beta <= alpha:
        return base
    ts = np.linspace(alpha, beta, 2 * n + 1)
    f = np.array([dickman_rho((1.0 - t) / alpha) / t for t in ts])
    h = (beta - alpha) / (2 * n)
    integral = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
    return float(base + integral)


def largest_two_prime_factors_upto(X: int):
    """Arrays (l1, l2) with l1[n] = largest prime factor of n, l2[n] = second
    largest counting multiplicity (1 if n is a prime power of a single prime), n <= X."""
    X = int(X)
    lpf = np.zeros(X + 1, dtype=np.int64)
    for p in range(2, X + 1):
        if lpf[p] == 0:  # p is prime: primes are processed in increasing order
            lpf[p::p] = p  # later (larger) primes overwrite, leaving the largest
    n = np.arange(X + 1, dtype=np.int64)
    l1 = lpf.copy()
    l1[:2] = 1
    cof = np.ones(X + 1, dtype=np.int64)
    cof[2:] = n[2:] // l1[2:]
    l2 = lpf[cof]
    l2[cof <= 1] = 1
    return l1, l2


def brute_force_G(X: int, alpha: float, beta: float) -> float:
    """Fraction of n in [2, X] with l1 <= X^beta and l2 <= X^alpha (finite-size check of G)."""
    l1, l2 = largest_two_prime_factors_upto(X)
    yb = X ** beta
    ya = X ** alpha
    ok = (l1[2:] <= yb) & (l2[2:] <= ya)
    return float(ok.mean())


# ---------------------------------------------------------------------------
# Empirical semismoothness of shifted primes
# ---------------------------------------------------------------------------

def _factor(n: int) -> dict[int, int]:
    return {int(p): int(e) for p, e in factorint(int(n)).items()}


def _top_two_from(fac: dict[int, int]) -> tuple[int, int]:
    """(largest, second largest with multiplicity) prime factors from a factorisation."""
    ps = sorted(fac.items(), reverse=True)
    l1 = ps[0][0]
    if ps[0][1] >= 2:
        l2 = l1
    elif len(ps) >= 2:
        l2 = ps[1][0]
    else:
        l2 = 1
    return l1, l2


def _top_two(n: int) -> tuple[int, int]:
    """(largest, second largest with multiplicity) prime factors of n >= 2."""
    return _top_two_from(_factor(n))


def stage1_exponent(B1: int):
    """L = lcm{l^e <= B1} as an integer (the stage-1 exponent of p-1 / p+1)."""
    L = mpz(1)
    for l in small_primes(int(B1) + 1):
        pe = l
        while pe * l <= B1:
            pe *= l
        L *= pe
    return L


def multiplicative_order(a: int, p: int, fac_pm1: dict[int, int]):
    """ord_p(a) from the factorisation of p - 1."""
    o = mpz(p - 1)
    for l, e in fac_pm1.items():
        for _ in range(e):
            if powmod(a, o // l, p) == 1:
                o //= l
            else:
                break
    return o


def lucas_order(P0: int, p: int, fac_plus: dict[int, int], fac_minus: dict[int, int]):
    """Order of the Lucas sequence V_n(P0, 1) mod p, i.e. of x = P0/2 + sqrt(P0^2/4 - 1).

    Returns (order, symbol) with symbol = (P0^2 - 4 | p): -1 means the order
    divides p + 1 (Williams' intended case), +1 means it divides p - 1.
    """
    d = (P0 * P0 - 4) % p
    s = int(jacobi(d, p))
    if s == 0:
        return None, 0
    n, fac = (p + 1, fac_plus) if s == -1 else (p - 1, fac_minus)
    o = mpz(n)
    for l, e in fac.items():
        for _ in range(e):
            if lucas_v(P0, o // l, p) == 2:
                o //= l
            else:
                break
    return o, s


def exact_success(order, L, B2) -> bool:
    """Stage 1 with exponent L followed by a BSGS stage 2 to B2 finds p iff
    the order of the stage-1 output, order / gcd(order, L), is <= B2."""
    if order is None:
        return False
    o = order // gcd(order, L)
    return o <= B2


def semismooth_profile(nbits: int = 48, count: int = 4000, exponents: Sequence[float] = (1/8, 1/6, 1/5, 2/9, 1/4),
                       seed: int = 5, family: str = "rsa", pp1_bases: Sequence[int] = (3, 5, 11)) -> dict:
    """Success frequencies of p-1 / p+1 style methods at cost N^c.

    Idealised criterion: a shifted prime s in {p-1, p+1, q-1, q+1} is
    (B1, B2)-semismooth, l2(s) <= B1 = N^c and l1(s) <= N^{2c}.
    Exact exposure predicate: for p-1 with base 2, d_p = ord_p(2)/gcd(ord_p(2), L)
    <= B2 where L = lcm{l^e <= B1}; for Williams with base P0, the same for the
    order of gamma = (P0 + sqrt(P0^2 - 4))/2, which divides p+1 iff
    (P0^2-4 | p) = -1 and p-1 otherwise.  The default bases 3, 5, 11 have
    discriminants 5, 21, 117 = 9*13 in three independent square classes (bases
    3 and 7 would share the class of 5).  A BSGS stage 2 exposes p iff d_p <= B2;
    it yields a nontrivial factor unless both primes are exposed with equal
    residual orders d_p = d_q (every collision then vanishes modulo N); the
    frequency of that degeneracy is recorded.  Controls: random integers and
    random even integers drawn from the same interval as the primes.
    """
    rng = random.Random(seed)
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    h = nbits // 2
    lo = isqrt(mpz(2) << (2 * h - 2)) + 1
    hi = mpz(1) << h
    facs = []  # per instance: dict name -> factorisation
    for inst in insts:
        p, q = int(inst.p), int(inst.q)
        facs.append({"p-1": _factor(p - 1), "p+1": _factor(p + 1), "q-1": _factor(q - 1), "q+1": _factor(q + 1)})
    shifted = {k: [_top_two_from(f[k]) for f in facs] for k in ("p-1", "p+1", "q-1", "q+1")}
    ctrl_int = [_top_two(rng.randrange(int(lo), int(hi))) for _ in range(count)]
    ctrl_even = [_top_two(2 * rng.randrange(int(lo) // 2, int(hi) // 2)) for _ in range(count)]
    ord2 = {"p": [multiplicative_order(2, int(i.p), f["p-1"]) for i, f in zip(insts, facs)],
            "q": [multiplicative_order(2, int(i.q), f["q-1"]) for i, f in zip(insts, facs)]}
    lucas = {"p": [[lucas_order(P0, int(i.p), f["p+1"], f["p-1"])[0] for P0 in pp1_bases] for i, f in zip(insts, facs)],
             "q": [[lucas_order(P0, int(i.q), f["q+1"], f["q-1"])[0] for P0 in pp1_bases] for i, f in zip(insts, facs)]}
    symbol_minus_one = float(np.mean([int(jacobi((P0 * P0 - 4) % int(i.p), int(i.p))) == -1
                                      for i in insts for P0 in pp1_bases[:1]]))

    def succ(pairs, c):
        B1 = 2.0 ** (c * nbits)
        B2 = B1 * B1
        return np.array([(l2 <= B1) and (l1 <= B2) for (l1, l2) in pairs], dtype=bool)

    out = {"nbits": nbits, "count": count, "family": family, "pp1_bases": list(pp1_bases),
           "symbol_minus_one_fraction_base3": symbol_minus_one, "rows": []}
    for c in exponents:
        s = {k: succ(v, c) for k, v in shifted.items()}
        B1 = int(2.0 ** (c * nbits))
        B2 = mpz(B1) * B1
        L = stage1_exponent(B1)
        ex_minus = {k: np.array([exact_success(o, L, B2) for o in ord2[k]], dtype=bool) for k in ("p", "q")}
        resid = {k: [o // gcd(o, L) for o in ord2[k]] for k in ("p", "q")}
        both_exposed = ex_minus["p"] & ex_minus["q"]
        equal_order = np.array([dp == dq for dp, dq in zip(resid["p"], resid["q"])], dtype=bool)
        # Miller descent: from alpha^E = 1 mod N (E a multiple of both orders) the
        # sequence alpha^{E/2^j} yields a nontrivial square root of 1 iff the 2-adic
        # valuations of ord_p(alpha) and ord_q(alpha) differ
        v2 = lambda o: (int(o) & -int(o)).bit_length() - 1  # noqa: E731
        recoverable = np.array([v2(op) != v2(oq) for op, oq in zip(ord2["p"], ord2["q"])], dtype=bool)
        degenerate = both_exposed & equal_order
        ex_plus1 = {k: np.array([exact_success(os[0], L, B2) for os in lucas[k]], dtype=bool) for k in ("p", "q")}
        ex_plus = {k: np.array([any(exact_success(o, L, B2) for o in os) for os in lucas[k]], dtype=bool) for k in ("p", "q")}
        row = {
            "c": c,
            "B1_bits": c * nbits,
            "pred_G": semismooth_G(2 * c, min(4 * c, 1.0)) if c < 0.25 else 1.0,
            "pred_rho_stage1_only": dickman_rho(1.0 / (2 * c)),
            "p-1": float(s["p-1"].mean()), "p+1": float(s["p+1"].mean()),
            "q-1": float(s["q-1"].mean()), "q+1": float(s["q+1"].mean()),
            "minus_pooled": float(np.concatenate([s["p-1"], s["q-1"]]).mean()),
            "plus_pooled": float(np.concatenate([s["p+1"], s["q+1"]]).mean()),
            "any_minus": float((s["p-1"] | s["q-1"]).mean()),
            "any_of_four": float((s["p-1"] | s["q-1"] | s["p+1"] | s["q+1"]).mean()),
            "ctrl_int": float(succ(ctrl_int, c).mean()),
            "ctrl_even": float(succ(ctrl_even, c).mean()),
            "stage1_only_minus_pooled": float(np.concatenate([
                np.array([l1 <= 2.0 ** (c * nbits) for (l1, l2) in shifted["p-1"]]),
                np.array([l1 <= 2.0 ** (c * nbits) for (l1, l2) in shifted["q-1"]])]).mean()),
            # exact algorithm predicates
            "exact_minus_pooled_base2": float(np.concatenate([ex_minus["p"], ex_minus["q"]]).mean()),
            "exact_plus_pooled_base3": float(np.concatenate([ex_plus1["p"], ex_plus1["q"]]).mean()),
            "exact_plus_pooled_3bases": float(np.concatenate([ex_plus["p"], ex_plus["q"]]).mean()),
            "exact_any_of_four": float((ex_minus["p"] | ex_minus["q"] | ex_plus["p"] | ex_plus["q"]).mean()),
            "exact_minus_vs_ideal_disagree": float(np.mean(np.concatenate([ex_minus["p"], ex_minus["q"]])
                                                         != np.concatenate([s["p-1"], s["q-1"]]))),
            "exact_minus_both_exposed": float(both_exposed.mean()),
            "exact_minus_both_exposed_equal_order": float(degenerate.mean()),
            "degenerate_recoverable_by_descent": float(recoverable[degenerate].mean()) if degenerate.any() else None,
            "degenerate_count": int(degenerate.sum()),
            "exact_minus_disagree_count": int(np.sum(np.concatenate([ex_minus["p"], ex_minus["q"]])
                                                  != np.concatenate([s["p-1"], s["q-1"]]))),
        }
        # correlation between p-1 and p+1 semismoothness (pooled over p and q)
        a = np.concatenate([s["p-1"], s["q-1"]]).astype(float)
        b = np.concatenate([s["p+1"], s["q+1"]]).astype(float)
        if a.std() > 0 and b.std() > 0:
            row["corr_minus_plus"] = float(np.corrcoef(a, b)[0, 1])
        else:
            row["corr_minus_plus"] = 0.0
        row["pred_any_of_four_indep"] = 1.0 - (1.0 - row["pred_G"]) ** 4
        out["rows"].append(row)
    return out


# ---------------------------------------------------------------------------
# Collision methods: rho length and Fermat steps
# ---------------------------------------------------------------------------

def rho_length(p: int, c: int, x0: int) -> tuple[int, int]:
    """(tail mu, cycle lambda) of x -> x^2 + c mod p from x0 (Brent's algorithm).

    mu + lambda is the number of distinct values before the first repetition,
    i.e. the collision time of Pollard rho modulo p (up to the detector's
    constant).
    """
    p = int(p)
    f = lambda x: (x * x + c) % p  # noqa: E731
    power = lam = 1
    tortoise, hare = x0, f(x0)
    while tortoise != hare:
        if power == lam:
            tortoise = hare
            power *= 2
            lam = 0
        hare = f(hare)
        lam += 1
    tortoise = hare = x0
    for _ in range(lam):
        hare = f(hare)
    mu = 0
    while tortoise != hare:
        tortoise = f(tortoise)
        hare = f(hare)
        mu += 1
    return mu, lam


def collision_profile(nbits: int = 48, count: int = 1000, exponents: Sequence[float] = (1/6, 1/5, 2/9, 1/4, 0.3),
                      seed: int = 11, family: str = "rsa") -> dict:
    """Rho collision times modulo p and q, Fermat step counts, and the implied
    success frequencies at cost N^c."""
    rng = random.Random(seed)
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    T_p, T_q, T_min, fermat_steps = [], [], [], []
    for inst in insts:
        c = rng.randrange(1, 1000)
        x0 = rng.randrange(0, 1000)
        mu, lam = rho_length(int(inst.p), c, x0)
        tp = mu + lam
        mu, lam = rho_length(int(inst.q), c, x0)
        tq = mu + lam
        T_p.append(tp)
        T_q.append(tq)
        T_min.append(min(tp, tq))
        fermat_steps.append(int((inst.p + inst.q) // 2 - isqrt_ceil(inst.N)))
    T_p = np.array(T_p, dtype=float)
    T_q = np.array(T_q, dtype=float)
    T_min = np.array(T_min, dtype=float)
    F = np.array(fermat_steps, dtype=float)
    sqrt_p = np.array([math.sqrt(int(i.p)) for i in insts])
    sqrt_q = np.array([math.sqrt(int(i.q)) for i in insts])
    norm = np.concatenate([T_p / sqrt_p, T_q / sqrt_q])
    out = {
        "nbits": nbits, "count": count, "family": family,
        "rho_length_over_sqrt_p_mean": float(norm.mean()),
        "rho_length_over_sqrt_p_std": float(norm.std()),
        "random_mapping_mean": math.sqrt(math.pi / 2),
        "rho_min_over_sqrt_p_mean": float((T_min / sqrt_p).mean()),
        "fermat_steps_over_sqrtN_median": float(np.median(F / 2.0 ** (nbits / 2))),
        "rows": [],
    }
    for c in exponents:
        budget = 2.0 ** (c * nbits)
        out["rows"].append({
            "c": c,
            "rho_success": float((T_min <= budget).mean()),
            "rho_pred_rayleigh": float(np.mean(1.0 - np.exp(-budget ** 2 / (2 * sqrt_p ** 2)) ** 2)),
            "fermat_success": float((F <= budget).mean()),
        })
    return out


def rho_length_map(f, x0: int) -> tuple[int, int]:
    """(mu, lambda) for an arbitrary map f on integers, Brent's algorithm."""
    power = lam = 1
    tortoise, hare = x0, f(x0)
    while tortoise != hare:
        if power == lam:
            tortoise = hare
            power *= 2
            lam = 0
        hare = f(hare)
        lam += 1
    tortoise = hare = x0
    for _ in range(lam):
        hare = f(hare)
    mu = 0
    while tortoise != hare:
        tortoise = f(tortoise)
        hare = f(hare)
        mu += 1
    return mu, lam


def rho_k_profile(ks: Sequence[int] = (2, 3, 4, 6, 8, 12), pbits: int = 22, count: int = 400, seed: int = 29) -> dict:
    """Mean rho length of x -> x^k + c mod p over primes p = 1 (mod lcm(ks)), against
    the in-degree-variance prediction E[mu + lambda] ~ sqrt(pi p / (2 (k - 1))).

    Heuristic: x^k + c maps Z_p onto a set of (p-1)/k + 1 points; restricted to
    its image the in-degree is Binomial(k, 1/k) (mean 1, variance 1 - 1/k).  For
    random mappings on n points with in-degree variance sigma^2 the expected rho
    length is sqrt(pi n / 2) / sigma (Arney-Bender), giving
    sqrt(pi (p/k) / 2) / sqrt(1 - 1/k) = sqrt(pi p / (2 (k - 1))).  For k = 2 this
    equals the unrestricted random-mapping value sqrt(pi p / 2): the 2-to-1
    structure of x^2 + c gives no gain.  A random mapping (random table) is run
    as a control.
    """
    rng = random.Random(seed)
    L = 1
    for k in ks:
        L = L * k // math.gcd(L, k)
    lo, hi = 1 << (pbits - 1), 1 << pbits
    primes = []
    while len(primes) < count:
        p = random_prime_in(rng, lo, hi)
        if (p - 1) % L == 0:
            primes.append(int(p))
    out = {"pbits": pbits, "count": count, "modulus_condition": f"p = 1 mod {L}", "rows": []}
    ratios_ctrl = []
    for p in primes:
        memo: dict[int, int] = {}

        def random_map(x, memo=memo, p=p):
            # a uniformly random mapping, sampled lazily on first access: the
            # explored orbit has exactly the random-mapping distribution
            y = memo.get(x)
            if y is None:
                y = rng.randrange(p)
                memo[x] = y
            return y

        mu, lam = rho_length_map(random_map, rng.randrange(p))
        ratios_ctrl.append((mu + lam) / math.sqrt(p))
    out["random_mapping_control"] = {"mean": float(np.mean(ratios_ctrl)), "pred": math.sqrt(math.pi / 2),
                                     "stderr": float(np.std(ratios_ctrl) / math.sqrt(len(ratios_ctrl)))}
    for k in ks:
        ratios = []
        for p in primes:
            c = rng.randrange(1, p)
            x0 = rng.randrange(p)
            mu, lam = rho_length_map(lambda x, k=k, c=c, p=p: (pow(x, k, p) + c) % p, x0)
            ratios.append((mu + lam) / math.sqrt(p))
        out["rows"].append({"k": k, "mean": float(np.mean(ratios)),
                            "stderr": float(np.std(ratios) / math.sqrt(len(ratios))),
                            "pred_arney_bender": math.sqrt(math.pi / (2 * (k - 1))),
                            "naive_image_only": math.sqrt(math.pi / (2 * k))})
    return out


def ecm_profile(nbits: int = 48, count: int = 300, exponents: Sequence[float] = (1/8, 1/6, 1/5),
                curves: int = 8, seed: int = 17, family: str = "rsa") -> dict:
    """Empirical ECM stage-1 success at B1 = N^c with a fixed number of curves."""
    from ..registry import get_algorithm
    ecm = get_algorithm("ecm")
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    out = {"nbits": nbits, "count": count, "curves": curves, "rows": []}
    for c in exponents:
        B1 = max(5, int(round(2.0 ** (c * nbits))))
        found = 0
        used = []
        work = []
        for i, inst in enumerate(insts):
            res = ecm(inst.N, B1=B1, curves=curves, seed=seed + i)
            if res.found:
                found += 1
                used.append(res.meta["curve"] + 1)
            work.append(res.primary_work)
        # per-curve success rate from the geometric structure of first-success indices
        total_curves = sum(used) + (count - found) * curves
        out["rows"].append({
            "c": c, "B1": B1,
            "success": found / count,
            "per_curve_success_est": found / total_curves if total_curves else 0.0,
            "pred_rho_per_prime": dickman_rho(math.log(2.0 ** (nbits / 2) / 12) / math.log(B1)),
            "mean_mulmod": float(np.mean(work)),
        })
    return out
