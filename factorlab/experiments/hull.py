"""D1: the lower convex hull of lattice points in {xy >= N} as a divisor locator.

Facts used
----------
* (d, N/d) is an exposed vertex of the lower hull of E_N = {(x,y) in Z^2_{>0} :
  xy >= N}: the supporting line (N/d) x + d y = 2N touches the hyperbola only
  there and E_N lies weakly above it (AM-GM).
* The hull over a window x in [x0, x1] with x1/x0 bounded has O(N^{1/3})
  vertices (affine length of the arc).

Vertex step (gift wrapping with a Stern-Brocot descent)
--------------------------------------------------------
At a hull vertex V = (x, y), y = ceil(N/x), the next edge direction is the
primitive (a, -b) of *maximal* slope b/a such that V + (a, -b) lies in E_N
("feasible").  In the offset plane the feasible set
    R = {(a, b) : (x + a)(y - b) >= N}
is convex and contains the origin, so:

  (i)  if d_L is feasible, d_R infeasible, det(d_L, d_R) = 1 and the mediant
       M = d_L + d_R is infeasible, then every lattice direction with slope in
       (slope M, slope d_R) is infeasible (they lie in M + cone(M, d_R), which
       is above the tangent of R at the exit point of the ray OM);
  (ii) if in addition slope(d_L) >= N / (x + a_M)^2 (the tangent slope of R's
       boundary at a = a_M) then every direction with slope in
       (slope d_L, slope d_R) is infeasible, so d_L is the edge direction.

Consecutive feasible (d_L += k d_R) and infeasible (d_R += k d_L) mediant
steps are taken in one jump by solving a concave quadratic in k exactly, so a
vertex step costs O(#partial quotients of the edge slope) = O(log N) bigint
operations.

Work counters: ``vertex`` (hull vertices emitted), ``sb_step`` (jumps).
"""

from __future__ import annotations

import time

from ..numth import mpz, isqrt, isqrt_ceil
from ..registry import register
from ..result import Work, success, failure

INF = None  # sentinel for an unbounded interval end


def _ceil_div(a, b):
    return -((-a) // b)


def _intersect(iv, k1=None, k2=None):
    """Intersect interval iv=(a,b) (None = unbounded) with [k1, k2]."""
    if iv is None:
        return None
    a, b = iv
    if k1 is not None:
        a = k1 if a is None else max(a, k1)
    if k2 is not None:
        b = k2 if b is None else min(b, k2)
    if a is not None and b is not None and a > b:
        return None
    return (a, b)


def feasible_interval(A, B, da, db, N):
    """Integer k-interval on which (A + k*da)(B - k*db) >= N AND B - k*db >= 1,
    for da >= 0, db >= 0, N >= 1 and arbitrary integers A, B, k.

    Returns (k1, k2) with k1 in Z or None (-inf) and k2 in Z or None (+inf), or
    None if empty.  On the positive branch B - k*db >= 1 the product is concave
    in k, so the feasible set is an interval.
    """
    A, B, da, db, N = mpz(A), mpz(B), mpz(da), mpz(db), mpz(N)
    # positivity interval P = {k : B - k*db >= 1}
    if db == 0:
        if B < 1:
            return None
        P = (None, None)
    else:
        P = (None, (B - 1) // db)
    a0 = A * B - N

    if da == 0 and db == 0:
        return P if a0 >= 0 else None
    if db == 0:
        # B >= 1 fixed, product increasing in k: a0 + k*B*da >= 0
        return _intersect(P, k1=_ceil_div(-a0, B * da))
    if da == 0:
        # A fixed; on P the product is A*(B - k*db), decreasing in k iff A > 0
        if A > 0:
            # A*(B - k db) >= N  <=>  k <= (A*B - N) / (A*db)
            return _intersect(P, k2=a0 // (A * db))
        # A <= 0: product <= 0 < N on the positive branch
        return None

    def g(k):
        return (A + k * da) * (B - k * db) - N

    c = da * db
    b = B * da - A * db
    disc = b * b + 4 * c * a0
    if disc < 0:
        return None
    # real roots (b -+ sqrt(disc)) / (2c); r <= sqrt(disc) < r + 1, so each of
    # floor(root_hi) and ceil(root_lo) is one of two adjacent integers.
    r = isqrt(disc)
    k2 = (b + r) // (2 * c)
    if g(k2 + 1) >= 0:
        k2 += 1
    k1 = -((r - b) // (2 * c))  # ceil((b - r) / (2c))
    if g(k1 - 1) >= 0:
        k1 -= 1
    # no integer between the roots -> empty (g < 0 on all of Z)
    if k1 > k2 or g(k1) < 0 or g(k2) < 0:
        return None
    return _intersect(P, k1=k1, k2=k2)


def _min_k_at_least(interval, kmin):
    """Smallest k >= kmin in the interval, or None."""
    if interval is None:
        return None
    k1, k2 = interval
    k = kmin if k1 is None else max(k1, kmin)
    if k2 is not None and k > k2:
        return None
    return k


def _smallest_T_with(bL, aL, N):
    """Smallest integer T >= 0 with bL * T^2 >= N * aL."""
    target = N * aL
    T = isqrt(target // bL)
    while bL * T * T < target:
        T += 1
    while T > 0 and bL * (T - 1) * (T - 1) >= target:
        T -= 1
    return T


def next_hull_vertex(x, y, N, w: Work | None = None):
    """Next lower-hull vertex to the right of V = (x, y) (y = ceil(N/x))."""
    x, y, N = mpz(x), mpz(y), mpz(N)
    if y <= 1:
        return x + 1, mpz(1)
    aL, bL = mpz(1), mpz(0)   # feasible (slope 0)
    aR, bR = mpz(0), mpz(1)   # infeasible (slope inf; y is minimal for x)
    jumps = 0
    while True:
        jumps += 1
        # feasible jump: d_L += k d_R, k maximal with V + d_L + k d_R in E_N
        iv = feasible_interval(x + aL, y - bL, aR, bR, N)
        assert iv is not None and (iv[0] is None or iv[0] <= 0), "d_L must be feasible"
        k2 = iv[1]
        assert k2 is not None, "feasible jump unbounded (cannot happen for bR >= 1)"
        if k2 > 0:
            aL += k2 * aR
            bL += k2 * bR
        # now M = d_L + d_R is infeasible
        if bL > 0 and bL * (x + aL + aR) ** 2 >= N * aL:
            break
        # infeasible jump: d_R += (k-1) d_L where k = first k >= 1 making
        # k d_L + d_R feasible, unless the stopping rule fires first
        k_feas = _min_k_at_least(feasible_interval(x + aR, y - bR, aL, bL, N), 1)
        if bL > 0:
            T = _smallest_T_with(bL, aL, N)
            k_stop = max(mpz(1), _ceil_div(T - x - aR, aL))
        else:
            k_stop = None
        if k_feas is None and k_stop is None:
            raise RuntimeError("hull step: no feasible direction and no stop (y too small?)")
        if k_feas is None or (k_stop is not None and k_stop < k_feas):
            # rule fires at M = k_stop d_L + d_R, which is infeasible
            break
        aR += (k_feas - 1) * aL
        bR += (k_feas - 1) * bL
        if jumps > 10_000:
            raise RuntimeError("hull step did not converge")
    if w is not None:
        w.add("sb_step", jumps)
    # walk along d_L as far as possible
    iv = feasible_interval(x, y, aL, bL, N)
    k = iv[1]
    if k is None:  # horizontal forever: only when y == 1
        return x + 1, y
    xn, yn = x + k * aL, y - k * bL
    return xn, yn


def hull_walk(N, x0, x1, w: Work | None = None):
    """Yield lower-hull vertices (x, y) of {P_x = (x, ceil(N/x)) : x >= x0} with x <= x1."""
    N = mpz(N)
    x = mpz(x0)
    y = _ceil_div(N, x)
    while x <= x1:
        if w is not None:
            w.add("vertex")
        yield x, y
        x, y = next_hull_vertex(x, y, N, w)


@register("hull_locator", primary_key="vertex", description="D1: walk lower integer hull of xy>=N over [sqrt(N/C), sqrt N]; divisor = vertex on the curve")
def hull_locator(N, C=4, x0=None, x1=None, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    if N % 2 == 0:
        return success("hull_locator", N, 2, w, "vertex", time.perf_counter() - t0)
    x1 = isqrt(N) if x1 is None else mpz(x1)
    x0 = isqrt_ceil(_ceil_div(N, mpz(C))) if x0 is None else mpz(x0)
    for x, y in hull_walk(N, x0, x1, w):
        if x * y == N and 1 < x < N:
            return success("hull_locator", N, x, w, "vertex", time.perf_counter() - t0,
                           x0=int(x0), x1=int(x1))
    return failure("hull_locator", N, w, "vertex", time.perf_counter() - t0, x0=int(x0), x1=int(x1))


def hull_statistics(N, p, C=4):
    """Per-vertex local features over [sqrt(N/C), sqrt N] and the divisor vertex's
    rank under each.  Features: dx_prev, dx_next (adjacent edge extents),
    dx_min, det (lattice area of the corner).  Rank 0 = most extreme."""
    N = mpz(N)
    x1 = isqrt(N)
    x0 = isqrt_ceil(_ceil_div(N, mpz(C)))
    verts = [(int(x), int(y)) for x, y in hull_walk(N, x0, x1)]
    n = len(verts)
    idx = next((i for i, (x, y) in enumerate(verts) if x == p), None)
    feats = {"dx_next": [], "dx_prev": [], "det": [], "dx_min": []}
    for i, (x, y) in enumerate(verts):
        dxn = verts[i + 1][0] - x if i + 1 < n else None
        dxp = x - verts[i - 1][0] if i > 0 else None
        det = None
        if 0 < i < n - 1:
            ax, ay = x - verts[i - 1][0], y - verts[i - 1][1]
            bx, by = verts[i + 1][0] - x, verts[i + 1][1] - y
            det = abs(ax * by - ay * bx)
        feats["dx_next"].append(dxn)
        feats["dx_prev"].append(dxp)
        feats["det"].append(det)
        cands = [v for v in (dxn, dxp) if v is not None]
        feats["dx_min"].append(min(cands) if cands else None)

    def rank_small(vals, i):
        v = vals[i]
        return None if v is None else sum(1 for u in vals if u is not None and u < v)

    def rank_large(vals, i):
        v = vals[i]
        return None if v is None else sum(1 for u in vals if u is not None and u > v)

    out = {"n_vertices": n, "divisor_index": idx, "N_bits": int(N).bit_length(),
           "mean_dx": (verts[-1][0] - verts[0][0]) / max(1, n - 1)}
    if idx is not None:
        out["divisor_features"] = {k: feats[k][idx] for k in feats}
        out["rank_small"] = {k: rank_small(feats[k], idx) for k in feats}
        out["rank_large"] = {k: rank_large(feats[k], idx) for k in feats}
    return out
