"""Experiments around the N^{1/5} barrier and the Frobenius-defect structure.

1. ``frobenius_degree``: the function x -> (x+a)^N - x^N - a on F_p equals
   (x+a)^{d+1} - x^{d+1} - a with d = q - p (since x^p = x on F_p and
   N = pq = p + ... => (x+a)^N = ((x+a)^p)^q = (x+a)^q = (x+a)^{p+d} =
   (x+a)(x+a)^d).  Hence it is a polynomial *function* of degree d on F_p while
   on F_q it has degree p-1.  The (d+1)-th forward difference therefore
   vanishes mod p but (generically) not mod q: gcd(Delta^{d+1} F(0), N) = p.
   This yields an O~(q-p) factoring method -- dominated by Fermat's
   O((q-p)^2/sqrt N) for all balanced N, but it explains the N1 leaks at
   small sizes (folding 2d sparse monomials mod X^r - 1 leaves zero
   coefficients when 2d < r).

2. ``lehman_covering``: numerically measure Sigma_w(P) = total candidate count
   of the Lehman family with parameter r (P = #pairs (a,b), ab <= r) and fit
   Sigma_w ~ sqrt(N) * P^{-theta}; the barrier argument predicts theta = 1/2.

3. ``chirp_hull_complexity``: number of convex-hull vertices of the point set
   {(k, ceil(2 sqrt(kN))) : k <= K}.  Affine length of y = 2 sqrt(Nx) on [1, K]
   is ~ N^{1/6} K^{1/2}; at K = N^{1/3} this is ~ N^{1/3} = K, i.e. the
   sequence has no linear structure at the scale relevant to Lehman.
"""

from __future__ import annotations

import math

from ..numth import mpz, gcd, powmod, isqrt_ceil


# ---------------------------------------------------------------------------
# 1. Frobenius defect as a polynomial function on F_p
# ---------------------------------------------------------------------------

def forward_difference_gcd(N, order: int, a: int = 1, x0: int = 0):
    """gcd(N, Delta^{order} F_a(x0)) with F_a(x) = (x+a)^N - x^N - a over Z_N.

    Delta^m F(x0) = sum_i (-1)^{m-i} C(m, i) F(x0 + i).
    """
    N = mpz(N)
    tot = mpz(0)
    binom = mpz(1)
    for i in range(order + 1):
        x = mpz(x0 + i)
        F = (powmod(x + a, N, N) - powmod(x, N, N) - a) % N
        sign = -1 if (order - i) % 2 else 1
        tot = (tot + sign * binom * F) % N
        binom = binom * (order - i) // (i + 1)
    return gcd(tot, N)


def frobenius_degree_check(N, p, q, a: int = 1) -> dict:
    """Verify: Delta^{d+1} F_a == 0 mod p (d = q - p) and Delta^{d} F_a != 0 mod p."""
    d = int(q - p)
    g_hi = forward_difference_gcd(N, d + 1, a)
    g_lo = forward_difference_gcd(N, d, a)
    return {"d": d, "gcd_order_d_plus_1": int(g_hi), "gcd_order_d": int(g_lo),
            "identity_holds": int(g_hi) % int(p) == 0 and int(g_hi) != int(N),
            "degree_exact": int(g_lo) % int(p) != 0}


def sparse_terms_mod_p(q, p) -> int:
    """Number of nonzero monomials of (X+a)^q - X^q - a over F_p when
    q = p + d, 0 < d < p, by Lucas' theorem: exactly 2d."""
    return 2 * int(q - p)


# ---------------------------------------------------------------------------
# 2. Lehman covering sum
# ---------------------------------------------------------------------------

def lehman_covering(N, r: int) -> dict:
    """P = #{(a,b): ab <= r}; Sigma_w = sum over pairs of width where
    width = N^{1/2} / (4 r sqrt(ab)) (Lemma 3.3 of Harvey 2020).

    Analytic form: sum_{ab<=r} (ab)^{-1/2} = 2 sqrt(r) (log r + 2 gamma - 2 + ...)
    hence Sigma_w / sqrt(N) ~ (log r + c) / (2 sqrt r), and P ~ r (log r + 2 gamma - 1).
    """
    N = mpz(N)
    sqrtN = math.sqrt(float(N))
    P = 0
    sigma = 0.0
    sigma_int = 0
    s_ab = 0.0
    for a in range(1, r + 1):
        for b in range(1, r // a + 1):
            P += 1
            s_ab += 1.0 / math.sqrt(a * b)
            w = sqrtN / (4 * r * math.sqrt(a * b))
            sigma += w
            sigma_int += max(1, math.ceil(w))
    return {"r": r, "P": P, "sigma_w": sigma, "sigma_w_int": sigma_int,
            "sigma_over_sqrtN": sigma / sqrtN,
            "sum_inv_sqrt_ab": s_ab,
            "analytic_sigma_over_sqrtN": s_ab / (4 * r),
            "analytic_leading": math.log(r) / (2 * math.sqrt(r)),
            "predicted_sqrtN_over_sqrtP": sqrtN / math.sqrt(P),
            "ratio_to_prediction": sigma / (sqrtN / math.sqrt(P))}


def cell_coverage_length(N, a: int, b: int, delta0: float, w: float) -> dict:
    """Lebesgue length of {p in [sqrt(N/4), sqrt N] : g(p) in [c_min + delta0, c_min + delta0 + w)}
    for g(p) = aN/p + bp, c_min = min g = 2 sqrt(abN), compared with the convexity
    bound 2 sqrt(2 w / m) = 2 sqrt(w sqrt N / a), where m = min g'' = 2a/sqrt N on p <= sqrt N.

    Preconditions: a, b >= 1, w > 0, delta0 >= 0.

    The level set {g = c_min + delta} has the two roots
        p_hi = (t + s) / (2b),   p_lo = 2aN / (t + s),   t = c_min + delta, s = sqrt(delta (2 c_min + delta)),
    (product of roots = aN/b); both forms are free of cancellation.
    """
    if a < 1 or b < 1:
        raise ValueError("a and b must be positive integers")
    if not w > 0:
        raise ValueError("window width w must be positive")
    if delta0 < 0:
        raise ValueError("delta0 must be >= 0 (levels below the cell minimum are empty)")
    Nf = float(N)
    lo, hi = math.sqrt(Nf / 4), math.sqrt(Nf)
    cmin = 2 * math.sqrt(a * b * Nf)

    def roots(delta):
        s = math.sqrt(delta * (2 * cmin + delta))
        t = cmin + delta
        p_hi = (t + s) / (2 * b)
        p_lo = 2 * a * Nf / (t + s)
        return [p_lo, p_hi]

    def clip(iv):
        x0, x1 = max(iv[0], lo), min(iv[1], hi)
        return max(0.0, x1 - x0)

    length = clip(roots(delta0 + w)) - clip(roots(delta0))
    bound = 2 * math.sqrt(w * math.sqrt(Nf) / a)
    pstar = math.sqrt(a * Nf / b)
    return {"a": a, "b": b, "delta0": delta0, "w": w, "length": length, "bound": bound,
            "ratio": length / bound, "pstar_over_sqrtN": pstar / hi}


# ---------------------------------------------------------------------------
# 3. Hull complexity of the chirp sequence c(k) = ceil(2 sqrt(kN))
# ---------------------------------------------------------------------------

def _upper_hull_vertices(points) -> int:
    """Andrew monotone chain, upper hull vertex count for x-sorted points."""
    h = []
    for x, y in points:
        while len(h) >= 2:
            (ax, ay), (bx, by) = h[-2], h[-1]
            if (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0:
                h.pop()
            else:
                break
        h.append((x, y))
    return len(h)


def _lower_hull_vertices(points) -> int:
    h = []
    for x, y in points:
        while len(h) >= 2:
            (ax, ay), (bx, by) = h[-2], h[-1]
            if (bx - ax) * (y - ay) - (by - ay) * (x - ax) <= 0:
                h.pop()
            else:
                break
        h.append((x, y))
    return len(h)


def chirp_hull_complexity(N, K: int) -> dict:
    N = mpz(N)
    pts = [(k, int(isqrt_ceil(4 * k * N))) for k in range(1, K + 1)]
    up = _upper_hull_vertices(pts)
    lo = _lower_hull_vertices(pts)
    # second differences histogram (structure check)
    c = [y for _, y in pts]
    d2 = [c[i + 2] - 2 * c[i + 1] + c[i] for i in range(len(c) - 2)]
    from collections import Counter
    hist = Counter(d2)
    affine_len = 2 * float(N) ** (1 / 6) * math.sqrt(K)
    return {"K": K, "upper_hull_vertices": up, "lower_hull_vertices": lo,
            "affine_length_estimate": affine_len,
            "second_difference_hist": {int(k): v for k, v in sorted(hist.items())[:8]},
            "N_bits": int(N).bit_length()}
