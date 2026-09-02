"""E18: the joint floor of NFS polynomial selection.

A polynomial f of degree d with a root m modulo N gives, over an unskewed
region of side A, an algebraic side of size ~ ||f||_oo A^d and a rational side
~ m A.  The product  P = ||f||_oo * m  is therefore the figure of merit of the
pair (f, m).  For a *fixed* root m the coefficient vectors form the lattice
L_{N,m} of determinant N and minimum ~ N^{1/(d+1)} (Proposition F of
notes_beyond_gnfs.md), and the base-m construction with m ~ N^{1/(d+1)} gives
P ~ N^{2/(d+1)}.  When m is free a counting heuristic gives a smaller floor:

  * the polynomials with top coefficients bounded by h number ~ 2^d h^{d+1}
    (f_d in [1, h], the others in [-h, h]), each has one root modulo N on
    average, and the roots are equidistributed, so the pairs (f, m) with
    ||f||_oo <= H and min(m, N - m) <= M number about 2^{d+1} H^{d+1} M / N;
  * an irreducible f with f(m) = 0 (mod N) has |f(m)| >= N, hence
    (d + 1) ||f||_oo m^d >= N.

Minimising P = H M under both constraints gives the *joint floor*

  P ~ N^{(2d-1)/(d^2+d-1)},  ||f||_oo ~ N^{(d-1)/(d^2+d-1)},  m ~ N^{d/(d^2+d-1)},

i.e. N^{3/5} (d = 2) and N^{5/11} (d = 3) against N^{2/3} and N^{1/2} for the
base-m construction.  The crude constraint gives the crude count

  count_crude(x) = C_d x^{e_d} / N^{g_d},   e_d = (d^2+d-1)/(d-1),  g_d = (2d-1)/(d-1),
  C_d = 2^d (d-1) (d+1)^{(2d-1)/(d-1)} / (d(d^2+d-1))
        (C_2 = 54/5, C_3 = 512/33),

which has the right exponents but overestimates the constant.  (The still
looser count that neither subtracts the forbidden root interval nor quotients
by the mirror symmetry has constants 108 and 512/3; it is not used.) A
polynomial with leading coefficient f_d has f(m) ~ f_d m^d, so its admissible roots are
m >= (N/f_d)^{1/d}, much larger than the crude (N/((d+1)h))^{1/d} when
f_d << h.  The refined count sums, over h = ||f||_oo and over f_d in [1, h],
the number of polynomials with that (h, f_d) times the expected number of
roots with (N/f_d)^{1/d} <= min(m, N-m) <= x/h, i.e. 2 (x/h - (N/f_d)^{1/d})^+ / N
(one root modulo N per polynomial on average, roots equidistributed), and
counts each normalized mirror orbit once: f(x)->f(-x) for even d and
f(x)->-f(-x) for odd d (preserving positive leading coefficient). Fixed
orbits (e.g. f_1 = 0 for d = 2) are lower-dimensional; dividing all counts by
two is a leading-order, not exact finite, orbit count.  For d = 2 the
continuum version is ~ 0.67 x^5 / N^3 instead of 10.8 x^5 / N^3, so the
predicted scale of P_min is larger by about 16.2^{1/5} = 1.75.  Under a
Poisson model Pr[P_min > x] = exp(-count(x)) and E[P_min] follows by
quadrature.

This module computes the *exact* minimum of P over all irreducible polynomials
of degree d with a root modulo N = pq, using the known factorisation to find
roots: square-root tables modulo p and q for d = 2, value tables of the cubic
modulo p and q for d = 3.  Roots m = 0 (mod N), which require N | f_0 and
correspond to the one-sided pairs (x^d - kN, 0) outside this two-sided size
model, are excluded; all bounds below assume min(m, N - m) >= 1.  The search is
over shells h = max(f_d, ..., f_1) with f_0 handled as a vector.  It is
complete because of the necessary condition
    N <= |f(m)| <= f_d m^d + h (m^{d-1} + ... + m) + |f_0|,
which gives m >= m_lo(f_d, h, |f_0|) for every admissible root and hence
P >= max(h, |f_0|) m_lo(f_d, h, |f_0|).  Lower envelopes over |f_0| go through
the continuous root m_c <= m_lo (the integer product is not monotone): shells
stop once h floor(m_c(h, h, h)) exceeds the best product found, tuples of top
coefficients with h floor(m_c(f_d, h, h)) >= best are skipped, and
|f_0| <= (best - 1)/m_lo(f_d, h, F_cap) with F_cap the crude cap.  Controls for
the same N: the base-m construction at
m = ceil(N^{1/(d+1)}), and the leading-coefficient search (a_d = 1..K,
m = round((N/a_d)^{1/d}), balanced digits), which the heuristic says reaches
the joint floor at K ~ N^{(d-1)/(d^2+d-1)}.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from flint import fmpz_poly

from ..gen import make_semiprime
from ..numth import mpz, iroot


# --------------------------------------------------------------------------
# Heuristic prediction
# --------------------------------------------------------------------------

def floor_exponents(d: int) -> dict:
    """Exponents of the joint floor and of the base-m construction."""
    q = d * d + d - 1
    C = 2 ** d * (d - 1) * (d + 1) ** ((2 * d - 1) / (d - 1)) / (d * q)
    return {"product": (2 * d - 1) / q, "coeff": (d - 1) / q, "root": d / q,
            "base_m_product": 2.0 / (d + 1), "e": q / (d - 1), "g": (2 * d - 1) / (d - 1),
            "C": C}


def expected_pairs(x: float, N: int, d: int) -> float:
    """Mirror-consistent crude count using m >= (N/((d+1)h))^{1/d}."""
    ex = floor_exponents(d)
    return ex["C"] * x ** ex["e"] / float(N) ** ex["g"]


def predicted_mean_crude(N: int, d: int) -> float:
    """E[P_min] under the Poisson model with the crude count."""
    ex = floor_exponents(d)
    scale = (float(N) ** ex["g"] / ex["C"]) ** (1.0 / ex["e"])
    return math.gamma(1.0 + 1.0 / ex["e"]) * scale


def refined_pairs(x: float, N: int, d: int) -> float:
    """Refined expected number of mirror-orbits {f, f(-x)} with a root pair of product <= x.

    Sums over h = ||f||_oo and the leading coefficient f_d in [1, h]: the
    number of f with that (h, f_d) is (2h+1)^d if f_d = h and
    (2h+1)^d - (2h-1)^d otherwise; each contributes 2 (x/h - (N/f_d)^{1/d})^+ / N
    expected admissible roots; the total is halved for the normalized mirror
    involution f(x)->f(-x) (even d), f(x)->-f(-x) (odd d). Fixed mirror orbits are lower-dimensional, so this is a leading-order
    orbit count rather than an exact finite orbit enumeration.  Linear in the shell cutoff H = x^{d/(d-1)} N^{-1/(d-1)}: the inner sums
    over f_d use prefix sums of f_d^{-1/d}.
    """
    N = float(N)
    H = int(math.floor(x ** (d / (d - 1.0)) / N ** (1.0 / (d - 1.0))))
    if H < 1:
        return 0.0
    j = np.arange(1, H + 1, dtype=np.float64)
    pref = np.concatenate([[0.0], np.cumsum(j ** (-1.0 / d))])  # pref[k] = sum_{j<=k} j^{-1/d}
    h = j
    M = x / h
    # f_d = h: every lower-coefficient vector, cutoff (N/h)^{1/d} <= M for h <= H
    term_top = (2 * h + 1) ** d * (M - (N / h) ** (1.0 / d))
    # f_d in [f_lo, h-1] with f_lo = ceil(N / M^d); lower coefficients with max exactly h
    f_lo = np.maximum(np.ceil(N / M ** d), 1.0)
    hi = h - 1
    valid = f_lo <= hi
    n_terms = np.where(valid, hi - f_lo + 1, 0.0)
    lo_idx = np.clip(f_lo - 1, 0, H).astype(np.int64)
    hi_idx = np.clip(hi, 0, H).astype(np.int64)
    sum_inv = np.where(valid, pref[hi_idx] - pref[lo_idx], 0.0)
    term_lt = ((2 * h + 1) ** d - (2 * h - 1) ** d) * (n_terms * M - N ** (1.0 / d) * sum_inv)
    total = float(np.sum(term_top + term_lt)) * 2.0 / N
    return 0.5 * total


def predicted_mean(N: int, d: int, tol: float = 1e-9) -> float:
    """E[P_min] = int_0^inf exp(-refined_pairs(x)) dx, tail-adaptive log-grid quadrature."""
    x0 = predicted_mean_crude(N, d)
    x = x0 / 32.0
    xs, surv = [x], [math.exp(-refined_pairs(x, N, d))]
    while surv[-1] > tol:
        x *= 1.02
        xs.append(x)
        surv.append(math.exp(-refined_pairs(x, N, d)))
    xs, surv = np.array(xs), np.array(surv)
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(xs[0] * surv[0] + trap(surv, xs))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def is_irreducible(coeffs: Sequence[int]) -> bool:
    """Irreducible over Q (coefficients low -> high; the content is ignored)."""
    f = fmpz_poly([int(c) for c in coeffs])
    if f.degree() < 1:
        return False
    _, facs = f.factor()
    return len(facs) == 1 and facs[0][1] == 1 and facs[0][0].degree() == f.degree()


def _eval(coeffs: Sequence[int], m: int) -> int:
    v = 0
    for c in reversed(coeffs):
        v = v * m + int(c)
    return v


def _sqrt_table(p: int) -> np.ndarray:
    """tab[a] = a square root of a modulo p, or -1."""
    r = np.arange(p, dtype=np.int64)
    tab = np.full(p, -1, dtype=np.int64)
    tab[(r * r) % p] = r
    return tab


def _lower_bound(h: int, N: int, d: int) -> float:
    """Crude: P >= h^{1-1/d} (N/(d+1))^{1/d} for every irreducible f with ||f||_oo >= h."""
    return h ** (1.0 - 1.0 / d) * (N / (d + 1.0)) ** (1.0 / d)


def _f0_cap(best: int, N: int, d: int) -> int:
    """Exact largest |f_0| that an integer product P < ``best`` can have.

    From (d+1)P^d >= N H^{d-1} and P <= best-1, every improving H satisfies
    H^{d-1} <= floor((d+1)(best-1)^d/N).  The integer root avoids a floating
    certification boundary.
    """
    if not math.isfinite(best):
        raise ValueError("an exact f0 cap requires a finite integer best")
    target = ((d + 1) * (int(best) - 1) ** d) // int(N)
    return int(iroot(mpz(target), d - 1)[0])


def _m_lo(fd: int, h: int, F: int, N: int, d: int) -> int:
    """Least integer m >= 1 with fd m^d + h (m^{d-1} + ... + m) + F >= N.

    For an irreducible f with leading coefficient fd, |f_i| <= h for 0 < i < d,
    |f_0| <= F and a root +-m modulo N, |f(+-m)| >= N forces m >= _m_lo.
    _m_lo is non-increasing in F and in fd.  (The integer product
    max(h, F) _m_lo(fd, h, F) is *not* monotone in F -- N = 7, fd = h = 2 gives
    4 at F = 2 and 3 at F = 3 -- so lower envelopes over F go through the
    continuous root, see _floor_root.)
    """
    def val(m):
        s = 0
        for i in range(1, d):
            s += m ** i
        return fd * m ** d + h * s + F
    lead_target = max(0, (N - F) // fd) if F < N else 0
    m = max(1, int(iroot(mpz(lead_target), d)[0]) - 1) if F < N else 1
    while m > 1 and val(m - 1) >= N:
        m -= 1
    while val(m) < N:
        m += 1
    return m


def _floor_root(fd: int, h: int, F: int, N: int, d: int) -> int:
    """Largest integer m >= 0 with fd m^d + h (m^{d-1} + ... + m) + F <= N, i.e. floor(m_c).

    m_c(fd, h, F) is the real root of fd m^d + h S(m) + F = N, S(m) = m^{d-1} + ... + m.
    Facts used for pruning: (i) _m_lo(fd, h, F) >= m_c(fd, h, F); (ii) m_c
    decreases in F and in fd; (iii) F m_c(fd, h, F) is non-decreasing in F for
    F <= N/2 (its derivative has the sign of d fd m^d + h m S'(m) - F >=
    fd m^d + h S(m) - F = N - 2F >= 0).  Hence for every admissible candidate
    of a tuple (fd, h) with |f_0| = F <= F_cap <= N/2:
        P >= max(h, F) m_c(fd, h, F) >= h m_c(fd, h, h) >= h floor(m_c(fd, h, h)),
    (for F < h because m_c decreases in F, for F >= h by (iii)), and for every
    candidate in a shell h' >= h:
        P >= h' m_c(h', h', h') >= h m_c(h, h, h),
    the last step because h m_c(h, h, h) -- with h(m^d + ... + 1) = N -- is
    increasing in h (derivative sign of m S_full'(m) - S_full(m) = sum (i-1) m^i > 0).
    """
    def val(m):
        s = 0
        for i in range(1, d):
            s += m ** i
        return fd * m ** d + h * s + F
    if F > N:
        return 0
    lead_target = max(0, (N - F) // fd)
    r = int(iroot(mpz(lead_target), d)[0])
    while r >= 1 and val(r) > N:
        r -= 1
    while val(r + 1) <= N:
        r += 1
    return r


def _tuple_bound(fd: int, h: int, N: int, d: int) -> int:
    """h floor(m_c(fd, h, h)): a lower bound on P for every candidate with top coefficients
    of sup-norm h, leading coefficient fd and |f_0| <= N/2 (see _floor_root)."""
    return h * _floor_root(fd, h, h, N, d)


def _shell_bound(h: int, N: int, d: int) -> int:
    """Globally monotone integer lower bound on P in every shell h' >= h.

    From (d+1)P^d >= N H^{d-1} and H >= h, P is at least the ceiling d-th
    root of ceil(N h^{d-1}/(d+1)).  This weaker bound remains valid after the
    continuous shell root drops below one, avoiding a false global
    monotonicity claim for h*m_c(h).
    """
    target = (N * h ** (d - 1) + d) // (d + 1)  # ceil(N h^{d-1}/(d+1))
    r, exact = iroot(mpz(target), d)
    return int(r if exact else r + 1)


# --------------------------------------------------------------------------
# d = 2: exact minimum via square-root tables
# --------------------------------------------------------------------------

def quad_floor(N, p, q, f0_window: int = 1 << 20, seed_K: int | None = None, chunk: int = 1 << 22) -> dict:
    """Exact min of ||f||_oo * min(m, N-m) over irreducible quadratics with a root mod N.

    Shells h = max(f_2, f_1) are enumerated while _shell_bound(h) < best (no
    pair in shell h or beyond can beat ``best`` otherwise); within a shell a
    pair (f_2, f_1) is skipped when _tuple_bound(f_2, h) >= best, and its f_0
    range is |f_0| <= max(h, min(F_cap, (best - 1) // _m_lo(f_2, h, F_cap)))
    with F_cap the crude cap, since F * _m_lo(f_2, h, F) <= P for every
    admissible pair and _m_lo is non-increasing in F.  All rules are necessary
    conditions for |f(m)| >= N, so the search is complete; it verifies that
    ``f0_window`` never truncated a rigorous range and raises otherwise, and
    refuses shells reaching a prime factor.  The best product is seeded with
    the leading-coefficient search (a valid member of the search space).
    """
    N, p, q = int(N), int(p), int(q)
    sp, sq = _sqrt_table(p), _sqrt_table(q)
    pinv_q = pow(p, -1, q)
    inv2p = [0]
    inv2q = [0]
    K = seed_K if seed_K is not None else max(64, int(round(N ** floor_exponents(2)["coeff"])))
    s = leading_coefficient_search(N, 2, K)
    if s is None:
        raise RuntimeError("no seed polynomial found")
    sf = list(s["f"])
    if sf[1] < 0:  # f(-x), preserving positive leading coefficient
        sf[1] = -sf[1]
    best, best_f, best_m = int(s["P"]), tuple(sf), min(s["m"], N - s["m"])
    if 2 * best >= N:
        raise RuntimeError("seed product is not below N/2; F<=N/2 pruning is uncertified")
    h = 0
    elements = 0
    max_window_needed = 0
    while True:
        h += 1
        if _shell_bound(h, N, 2) >= best:
            break
        if h >= min(p, q):
            raise RuntimeError("shell reached a prime factor; degenerate reductions not implemented")
        inv2p.append(pow(2 * h, -1, p))
        inv2q.append(pow(2 * h, -1, q))
        F_cap = min(best - 1, _f0_cap(best, N, 2))
        # new (f2, f1) with max(f2, f1) == h, f1 >= 0 by the symmetry x -> -x
        pairs = [(h, f1) for f1 in range(0, h + 1)] + [(f2, h) for f2 in range(1, h)]
        bufs, size = [], 0
        for f2, f1 in pairs:
            if _tuple_bound(f2, h, N, 2) >= best:
                continue
            W = max(h, min(F_cap, (best - 1) // _m_lo(f2, h, F_cap, N, 2)))
            max_window_needed = max(max_window_needed, W)
            W = min(W, f0_window)
            f0 = np.arange(-W, W + 1, dtype=np.int64)
            bufs.append((np.full(f0.size, f2, dtype=np.int64), np.full(f0.size, f1, dtype=np.int64), f0))
            size += f0.size
            if size >= chunk:
                res = _quad_batch(*[np.concatenate(b) for b in zip(*bufs)], N, p, q, sp, sq, pinv_q, inv2p, inv2q)
                elements += size
                bufs, size = [], 0
                if res is not None and res[0] <= best:
                    best, best_f, best_m = res
        if bufs:
            res = _quad_batch(*[np.concatenate(b) for b in zip(*bufs)], N, p, q, sp, sq, pinv_q, inv2p, inv2q)
            elements += size
            if res is not None and res[0] <= best:
                best, best_f, best_m = res
    assert is_irreducible(best_f)
    assert _eval(best_f, best_m) % N == 0 or _eval(best_f, -best_m) % N == 0
    if max_window_needed > f0_window:
        raise RuntimeError(f"f0 window {f0_window} truncated the certified range {max_window_needed}")
    return {"P": int(best), "f": list(best_f), "m": int(best_m), "Hf": max(abs(c) for c in best_f),
            "shell_stop": h, "f0_cap": _f0_cap(best, N, 2), "elements": int(elements), "certified": True}


def _quad_batch(F2, F1, F0, N, p, q, sp, sq, pinv_q, inv2p, inv2q):
    """Best (P, f, m_min) among the quadratics of a batch, or None."""
    D = F1 * F1 - 4 * F2 * F0
    Dn = np.maximum(D, 0)
    s = np.floor(np.sqrt(Dn.astype(np.float64))).astype(np.int64)
    s = np.where((s + 1) * (s + 1) <= Dn, s + 1, s)
    s = np.where(s * s > Dn, s - 1, s)
    square = (D >= 0) & (s * s == D)  # reducible over Q
    rp = sp[D % p]
    rq = sq[D % q]
    ok = (rp >= 0) & (rq >= 0) & ~square
    if not ok.any():
        return None
    F2, F1, F0, rp, rq = F2[ok], F1[ok], F0[ok], rp[ok], rq[ok]
    ip = np.asarray(inv2p, dtype=np.int64)[F2]
    iq = np.asarray(inv2q, dtype=np.int64)[F2]
    Hf = np.maximum(np.maximum(np.abs(F0), F1), F2)
    mmin = None
    for sgn_p in (1, -1):
        mp = (((-F1 + sgn_p * rp) % p) * ip) % p
        for sgn_q in (1, -1):
            mq = (((-F1 + sgn_q * rq) % q) * iq) % q
            t = (((mq - mp) % q) * pinv_q) % q
            m = mp + p * t
            m = np.minimum(m, N - m)
            m = np.where(m == 0, N, m)  # root 0 mod N is excluded; sentinel N
            mmin = m if mmin is None else np.minimum(mmin, m)
    P = Hf * mmin
    P = np.where(mmin == N, np.iinfo(np.int64).max, P)  # no admissible root
    i = int(np.argmin(P))
    return int(P[i]), (int(F0[i]), int(F1[i]), int(F2[i])), int(mmin[i])


# --------------------------------------------------------------------------
# d = 3: exact minimum via value tables of the cubic modulo p and q
# --------------------------------------------------------------------------

def _crt_min(mps: np.ndarray, mqs: np.ndarray, p: int, q: int, N: int, pinv_q: int) -> int:
    """Smallest min(m, N-m) >= 1 over the CRT combinations of roots mod p and mod q (N if none)."""
    mp = np.repeat(mps, mqs.size)
    mq = np.tile(mqs, mps.size)
    t = (((mq - mp) % q) * pinv_q) % q
    m = mp + p * t
    m = np.minimum(m, N - m)
    m = m[m > 0]
    return int(m.min()) if m.size else N


def cubic_floor(N, p, q, seed_K: int | None = None) -> dict:
    """Exact min of ||f||_oo * min(m, N-m) over irreducible cubics with a root mod N.

    Roots are read off value tables of f_3 m^3 + f_2 m^2 + f_1 m modulo p and q,
    so no invertibility is assumed; f_0 = -g(m) is lifted to the centred range
    [-W, W].  Pruning uses the necessary condition |f(m)| >= N through the
    continuous root m_c (see _floor_root): shells stop when _shell_bound(h) >=
    best, leading coefficients with _tuple_bound(f_3, h) >= best are skipped,
    and W = max(h, min(F_cap, (best - 1) // _m_lo(f_3, h, F_cap))) per leading
    coefficient.  The centred lift is complete while W <= (p-1)/2, (q-1)/2,
    which is verified.  The best product is seeded with the leading-coefficient
    search.
    """
    N, p, q = int(N), int(p), int(q)
    lift_range = (min(p, q) - 1) // 2
    mp_ = np.arange(p, dtype=np.int64)
    P1, P2 = mp_, (mp_ * mp_) % p
    P3 = (P2 * mp_) % p
    mq_ = np.arange(q, dtype=np.int64)
    Q1, Q2 = mq_, (mq_ * mq_) % q
    Q3 = (Q2 * mq_) % q
    pinv_q = pow(p, -1, q)
    K = seed_K if seed_K is not None else max(64, int(round(N ** floor_exponents(3)["coeff"])))
    s = leading_coefficient_search(N, 3, K)
    if s is None:
        raise RuntimeError("no seed polynomial found")
    sf = list(s["f"])
    if sf[2] < 0:  # -f(-x), preserving positive leading coefficient and making f_2 >= 0
        sf = [-sf[0], sf[1], -sf[2], sf[3]]
    best, best_f, best_m = int(s["P"]), tuple(sf), min(s["m"], N - s["m"])
    if 2 * best >= N:
        raise RuntimeError("seed product is not below N/2; F<=N/2 pruning is uncertified")
    h = 0
    triples = 0
    max_window_needed = 0
    while True:
        h += 1
        if _shell_bound(h, N, 3) >= best:
            break
        F_cap = min(best - 1, _f0_cap(best, N, 3))
        # new triples (f3, f2, f1) with max(f3, f2, |f1|) == h; f3 >= 1, f2 >= 0 by symmetry
        for f3 in range(1, h + 1):
            if _tuple_bound(f3, h, N, 3) >= best:
                continue
            W = max(h, min(F_cap, (best - 1) // _m_lo(f3, h, F_cap, N, 3)))
            max_window_needed = max(max_window_needed, W)
            W = min(W, lift_range)
            for f2 in range(0, h + 1):
                if f3 == h or f2 == h:
                    f1s = range(-h, h + 1)
                else:
                    f1s = (-h, h)
                for f1 in f1s:
                    triples += 1
                    gp = (f3 * P3 + f2 * P2 + f1 * P1) % p
                    maskp = (gp <= W) | (gp >= p - W)
                    idx_p = np.nonzero(maskp)[0]
                    if idx_p.size == 0:
                        continue
                    gq = (f3 * Q3 + f2 * Q2 + f1 * Q1) % q
                    maskq = (gq <= W) | (gq >= q - W)
                    idx_q = np.nonzero(maskq)[0]
                    if idx_q.size == 0:
                        continue
                    vp = gp[idx_p]
                    f0p = np.where(vp <= W, -vp, p - vp)  # f0 = -g(m) lifted to [-W, W]
                    vq = gq[idx_q]
                    f0q = np.where(vq <= W, -vq, q - vq)
                    common = np.intersect1d(f0p, f0q)
                    if common.size == 0:
                        continue
                    Hs = np.maximum(np.abs(common), h)
                    order = np.argsort(Hs)
                    for j in order:
                        f0 = int(common[j])
                        Hf = int(Hs[j])
                        if Hf * _m_lo(f3, h, Hf, N, 3) >= best:
                            continue
                        mm = _crt_min(idx_p[f0p == f0], idx_q[f0q == f0], p, q, N, pinv_q)
                        Pv = Hf * mm
                        if Pv > best:
                            continue
                        f = (f0, f1, f2, f3)
                        if _eval(f, mm) == 0 or _eval(f, -mm) == 0 or not is_irreducible(f):
                            continue
                        if Pv < best or tuple(f) <= tuple(best_f):
                            best, best_f, best_m = Pv, f, mm
    if max_window_needed > lift_range:
        raise RuntimeError(f"centred lift range {lift_range} smaller than the certified window {max_window_needed}")
    return {"P": int(best), "f": list(best_f), "m": int(best_m), "Hf": max(abs(c) for c in best_f),
            "shell_stop": h, "f0_cap": _f0_cap(best, N, 3), "triples": int(triples), "certified": True}


# --------------------------------------------------------------------------
# Controls: base-m and the leading-coefficient search
# --------------------------------------------------------------------------

def _balanced_digits(r: int, m: int, count: int) -> tuple[list[int], int]:
    """``count`` digits c_i in (-m/2, m/2] and the leftover t with r = sum c_i m^i + t m^count."""
    out = []
    for _ in range(count):
        c = r % m
        if c > m // 2:
            c -= m
        out.append(int(c))
        r = (r - c) // m
    return out, int(r)


def base_m_construction(N, d: int) -> dict:
    """f from the base-m digits of N with m = ceil(N^{1/(d+1)}): d balanced lower
    digits and the leftover as leading coefficient (which may exceed m/2)."""
    N = int(N)
    m = int(iroot(mpz(N), d + 1)[0]) + 1
    digits, lead = _balanced_digits(N, m, d)
    assert lead != 0
    f = digits + [lead]
    return {"f": f, "m": m, "P": max(abs(c) for c in f) * m, "irreducible": is_irreducible(f)}


def leading_coefficient_search(N, d: int, K: int) -> dict:
    """Best P over a_d = 1..K with m = round((N/a_d)^{1/d}) and balanced lower digits."""
    N = int(N)
    best = None
    for ad in range(1, K + 1):
        m0 = int(iroot(mpz(N // ad), d)[0])
        for m in (m0, m0 + 1):
            if m < 2:
                continue
            r = N - ad * m ** d
            digits, left = _balanced_digits(r, m, d)
            if left != 0:
                continue
            f = digits + [ad]
            P = max(abs(c) for c in f) * m
            if (best is None or P < best["P"]) and is_irreducible(f):
                best = {"f": f, "m": m, "P": P, "a_d": ad}
    return best


def poisson_check(result: dict) -> dict:
    """Distributional test of the Poisson model on a poly_floor_experiment result.

    If Pr[P_min > x] = exp(-count(x)) then u = count(P_min) is Exp(1); returns
    the sample mean of u, the Kolmogorov-Smirnov statistic and p-value against
    Exp(1), and the same for the crude count.
    """
    from scipy import stats
    d = int(result["d"])
    u_ref, u_crude = [], []
    for row in result["rows"]:
        for inst in row["instances"]:
            N, P = int(inst["N"]), float(inst["P"])
            u_ref.append(refined_pairs(P, N, d))
            u_crude.append(expected_pairs(P, N, d))
    u_ref, u_crude = np.array(u_ref), np.array(u_crude)
    ks_r = stats.kstest(u_ref, "expon")
    ks_c = stats.kstest(u_crude, "expon")
    return {"n": int(u_ref.size),
            "refined": {"mean_u": float(u_ref.mean()), "ks": float(ks_r.statistic), "p": float(ks_r.pvalue)},
            "crude": {"mean_u": float(u_crude.mean()), "ks": float(ks_c.statistic), "p": float(ks_c.pvalue)}}


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def poly_floor_experiment(d: int, bits: Sequence[int], counts: Sequence[int], seed: int = 111,
                          family: str = "rsa") -> dict:
    ex = floor_exponents(d)
    rows = []
    for nbits, count in zip(bits, counts):
        per = []
        for i in range(count):
            inst = make_semiprime(nbits, family, seed, i)
            N, p, q = int(inst.N), int(inst.p), int(inst.q)
            res = quad_floor(N, p, q) if d == 2 else cubic_floor(N, p, q)
            Kstar = max(1, int(round(N ** ex["coeff"])))
            base = base_m_construction(N, d)
            s1 = leading_coefficient_search(N, d, Kstar)
            s8 = leading_coefficient_search(N, d, 8 * Kstar)
            per.append({
                "N": str(N), "P": res["P"], "f": res["f"], "m": res["m"], "Hf": res["Hf"],
                "shell_stop": res["shell_stop"],
                "P_over_pred_mean": res["P"] / predicted_mean(N, d),
                "P_over_pred_mean_crude": res["P"] / predicted_mean_crude(N, d),
                "log_P_over_log_N": math.log(res["P"]) / math.log(N),
                "log_Hf_over_log_N": math.log(res["Hf"]) / math.log(N),
                "log_m_over_log_N": math.log(res["m"]) / math.log(N),
                "k": str(abs(_eval(res["f"], res["m"]) if _eval(res["f"], res["m"]) % N == 0
                             else _eval(res["f"], -res["m"])) // N),
                "base_m_P": base["P"], "search_Kstar": Kstar,
                "search_P_Kstar": None if s1 is None else s1["P"],
                "search_P_8Kstar": None if s8 is None else s8["P"],
            })
        Ps = np.array([r["P"] for r in per], dtype=float)
        Ns = np.array([float(r["N"]) for r in per])
        rows.append({
            "nbits": nbits, "count": count,
            "mean_P_over_pred": float(np.mean([r["P_over_pred_mean"] for r in per])),
            "mean_P_over_pred_crude": float(np.mean([r["P_over_pred_mean_crude"] for r in per])),
            "mean_log_P_over_log_N": float(np.mean([r["log_P_over_log_N"] for r in per])),
            "mean_log_Hf_over_log_N": float(np.mean([r["log_Hf_over_log_N"] for r in per])),
            "mean_log_m_over_log_N": float(np.mean([r["log_m_over_log_N"] for r in per])),
            "mean_log2_P": float(np.mean(np.log2(Ps))), "std_log2_P": float(np.std(np.log2(Ps))),
            "mean_log2_base_m_P": float(np.mean([math.log2(r["base_m_P"]) for r in per])),
            "mean_log2_search_P_Kstar": float(np.mean([math.log2(r["search_P_Kstar"]) for r in per if r["search_P_Kstar"]])),
            "mean_log2_search_P_8Kstar": float(np.mean([math.log2(r["search_P_8Kstar"]) for r in per if r["search_P_8Kstar"]])),
            "search_reaches_optimum_fraction": float(np.mean([r["search_P_8Kstar"] == r["P"] for r in per])),
            "mean_log2_N": float(np.mean(np.log2(Ns))),
            "instances": per,
        })
    # exponent fit of log2 P_min against log2 N over all instances
    x = np.array([math.log2(float(r["N"])) for row in rows for r in row["instances"]])
    y = np.array([math.log2(r["P"]) for row in rows for r in row["instances"]])
    yb = np.array([math.log2(r["base_m_P"]) for row in rows for r in row["instances"]])
    (slope, icpt), cov = np.polyfit(x, y, 1, cov=True)
    (slope_b, _), cov_b = np.polyfit(x, yb, 1, cov=True)
    out = {"d": d, "predicted": ex, "rows": rows,
           "fit": {"slope": float(slope), "slope_se": float(math.sqrt(cov[0, 0])), "intercept": float(icpt),
                   "base_m_slope": float(slope_b), "base_m_slope_se": float(math.sqrt(cov_b[0, 0]))}}
    out["poisson_check"] = poisson_check(out)
    return out


def minimiser_structure(result: dict) -> dict:
    """Structure of one selected exact witness per modulus (ties are not enumerated): whether
    |f_d| = ||f||_oo, the root relative to the admissibility cutoff
    (N/f_d)^{1/d}, and the cofactor k = |f(m)|/N."""
    d = int(result["d"])
    lead_at_max, root_ratio, ks = [], [], []
    for row in result["rows"]:
        for inst in row["instances"]:
            N, f, m = int(inst["N"]), inst["f"], int(inst["m"])
            Hf = max(abs(c) for c in f)
            lead_at_max.append(abs(f[-1]) == Hf)
            root_ratio.append(m / (N / abs(f[-1])) ** (1.0 / d))
            ks.append(int(inst["k"]))
    return {"n": len(ks), "lead_at_max_fraction": float(np.mean(lead_at_max)),
            "root_over_cutoff_median": float(np.median(root_ratio)), "root_over_cutoff_max": float(np.max(root_ratio)),
            "k_median": float(np.median(ks)), "k_max": int(np.max(ks))}


if __name__ == "__main__":  # python -m factorlab.experiments.poly_floor results/e18_poly_floor.json
    import json
    import sys

    with open(sys.argv[1]) as fh:
        data = json.load(fh)
    for key, res in sorted(data.items()):
        chk = poisson_check(res)
        st = minimiser_structure(res)
        print(f"d={key}: n={chk['n']}  refined: mean u {chk['refined']['mean_u']:.3f}, KS {chk['refined']['ks']:.3f}, "
              f"p {chk['refined']['p']:.3f} | crude: mean u {chk['crude']['mean_u']:.3f}, KS {chk['crude']['ks']:.3f}, "
              f"p {chk['crude']['p']:.2e}")
        print(f"       minimisers: |f_d| = ||f|| in {st['lead_at_max_fraction']:.2f}; root/cutoff median {st['root_over_cutoff_median']:.2f} "
              f"(max {st['root_over_cutoff_max']:.2f}); k = |f(m)|/N median {st['k_median']:.0f} (max {st['k_max']})")
