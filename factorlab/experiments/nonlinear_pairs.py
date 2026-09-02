"""E23a: nonlinear polynomial pairs for NFS -- existence floor against constructions.

Two polynomials f, g of degree d with a common root modulo N (NFS with two
algebraic sides) satisfy N | Res(f, g) != 0, hence by Hadamard
    N <= (d+1)^d ||f||_oo^d ||g||_oo^d,
i.e. the product P2 = ||f||_oo ||g||_oo is at least (N/(d+1)^d)^{1/d}
(Proposition K of notes_beyond_gnfs.md).  Counting pairs whose resultant is a
small multiple of N shows that pairs exist at this floor (about N^{1/d} of
them with P2 = O(N^{1/d})), so the floor is the true existence exponent.

Constructions.  Montgomery's two-quadratics method builds a geometric
progression (GP) c = (p', c, (c^2 - N)/p') with common ratio theta = c/p'
modulo N and all entries ~ sqrt N, then reads two quadratics off the rank-2
lattice orthogonal to c: every integer vector f with sum f_i c_i = 0 has
f(theta) = 0 (mod N).  The same idea works for any degree: choose a prime
p' ~ N^{1/d} for which N is a d-th power residue, a d-th root c0 of N modulo
p', and the lift c ~ N^{1/d} of c0; then
    c = (p'^{d-1}, p'^{d-2} c, ..., c^{d-1}, (c^d - N)/p')
is a GP of length d+1 with ratio c/p' and every entry of absolute value at most
2^d N^{1-1/d} when c is the least nonnegative d-th root (c < p' < 2 N^{1/d}).  Its
orthogonal lattice has rank d and determinant ||c||_2 ~ N^{1-1/d}, so its d
successive minima are ~ N^{(d-1)/d^2} each when balanced and the best pair has
product ~ N^{2(d-1)/d^2}: equal to the floor N^{1/d} only for d = 2.  For d = 3 the
construction gives N^{4/9} against the floor N^{1/3}; the literature search found no
polynomial-time construction at the floor for d >= 3
(.maestro/perplexity/nonlinear_nfs_polynomial_pairs_sizes.md).

Exact floor.  With the factorisation of N known, the minimum of P2 over
primitive coprime irreducible pairs of exact degree d is computed exactly
(content coprime to N divides out of a root relation, so primitivity loses
nothing): enumerate f with ||f||_oo <= sqrt(best) (w.l.o.g. ||f|| <= ||g||), find
the roots of f modulo N from its roots modulo p and q, and for each root r take
the shortest admissible g in the root lattice L_{N,r} by certified Fincke-Pohst
enumeration (all vectors below the current bound), rejecting multiples of f and
reducible g.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
from flint import fmpz_poly, nmod_poly

from ..gen import make_semiprime, random_prime_in
from ..numth import mpz, iroot, isqrt, invert, is_prime, sqrt_mod_prime
from .root_lattice import pari, root_lattice_columns


def sup(v) -> int:
    return max(abs(int(x)) for x in v)


def is_primitive(v: Sequence[int]) -> bool:
    return math.gcd(*[abs(int(c)) for c in v]) == 1


def is_irreducible_q(coeffs: Sequence[int]) -> bool:
    f = fmpz_poly([int(c) for c in coeffs])
    if f.degree() < 1:
        return False
    _, facs = f.factor()
    return len(facs) == 1 and facs[0][1] == 1 and facs[0][0].degree() == f.degree()


def poly_eval_mod(coeffs: Sequence[int], x: int, N: int) -> int:
    v = 0
    for c in reversed(coeffs):
        v = (v * x + int(c)) % N
    return v


def resultant_floor(N: int, d: int) -> float:
    """(N/(d+1)^d)^{1/d}: the Hadamard lower bound on ||f||_oo ||g||_oo (Proposition K)."""
    return (float(N) / (d + 1) ** d) ** (1.0 / d)


# --------------------------------------------------------------------------
# Geometric progressions and the orthogonal lattice
# --------------------------------------------------------------------------

def orthogonal_lattice(c: Sequence[int]) -> list[list[int]]:
    """LLL-reduced Z-basis (as row vectors) of {f in Z^n : sum f_i c_i = 0}."""
    P = pari()
    n = len(c)
    M = P.matrix(1, n, [int(x) for x in c])
    K = P.matkerint(M)
    cols = int(P.matsize(K)[1])
    return [[int(K[i, j]) for i in range(n)] for j in range(cols)]


def gp_pair(N: int, d: int, rng: random.Random, tries: int = 200) -> dict | None:
    """Root-lift GP construction of a (d, d) pair; Montgomery's method when d = 2.

    Picks a prime p' in [N^{1/d}, 2 N^{1/d}) with gcd(p', N) = 1 for which N is a
    d-th power residue modulo p' and the least nonnegative d-th root c < p' of N
    modulo p'; the GP (p'^{d-1}, ..., c^{d-1}, (c^d - N)/p') then has every entry of
    absolute value at most 2^d N^{1-1/d}, and the two shortest coprime irreducible
    polynomials among the admissible reduced-basis vectors of its orthogonal lattice
    are returned (basis vectors, not certified successive minima).  For d = 2 the
    lift is taken nearest to sqrt N so that the last entry is ~ sqrt N (Montgomery).
    Randomised: at most ``tries`` primes are sampled; returns None on failure.
    """
    N = int(N)
    P = pari()
    lo = int(iroot(mpz(N), d)[0]) + 1
    for _ in range(tries):
        pp = int(random_prime_in(rng, lo, 2 * lo))
        if N % pp == 0:
            continue
        n_mod = N % pp
        if d == 2:
            s = sqrt_mod_prime(n_mod, pp)
            if s is None:
                continue
            s = int(s)
            root_N = isqrt(mpz(N))
            # lift nearest to sqrt N
            base = int(root_N) - ((int(root_N) - s) % pp)
            c = min((base, base + pp), key=lambda v: abs(v * v - N))
        else:
            try:
                r = P.sqrtn(P.Mod(n_mod, pp), d)
            except Exception:
                continue
            c = int(P.lift(r)) % pp
            if pow(c, d, pp) != n_mod:
                continue
        if (c ** d - N) % pp != 0:
            continue
        gp = [pp ** (d - 1 - i) * c ** i for i in range(d)] + [(c ** d - N) // pp]
        gp_max = max(abs(x) for x in gp)
        theta = (c * int(invert(pp, N))) % N
        basis = orthogonal_lattice(gp)
        cands = []
        for v in basis:
            if v[-1] < 0:
                v = [-x for x in v]
            if v[-1] == 0 or not is_irreducible_q(v):
                continue
            if poly_eval_mod(v, theta, N) != 0:
                continue
            cands.append(v)
        cands.sort(key=sup)
        pair = None
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                f, g = cands[i], cands[j]
                if fmpz_poly(f).gcd(fmpz_poly(g)).degree() == 0:
                    pair = (f, g)
                    break
            if pair:
                break
        if pair is None:
            continue
        f, g = pair
        return {"p_prime": pp, "c": c, "gp": [str(x) for x in gp], "gp_max": gp_max,
                "gp_max_over_N_1_minus_1_over_d": gp_max / float(N) ** (1.0 - 1.0 / d),
                "theta": theta, "f": f, "g": g, "norms": [sup(f), sup(g)],
                "admissible_basis_norms": [sup(v) for v in cands], "P2": sup(f) * sup(g),
                "log_P2_over_log_N": math.log(sup(f) * sup(g)) / math.log(N)}
    return None


# --------------------------------------------------------------------------
# Exact (d, d) floor
# --------------------------------------------------------------------------

def roots_mod_prime(coeffs: Sequence[int], p: int) -> list[int]:
    f = nmod_poly([int(c) % p for c in coeffs], p)
    if f.degree() < 1:
        return []
    return [int(r) for r, _ in f.roots()]


def roots_mod_N(coeffs: Sequence[int], p: int, q: int) -> list[int]:
    N = p * q
    out = []
    rp = roots_mod_prime(coeffs, p)
    if not rp:
        return out
    rq = roots_mod_prime(coeffs, q)
    pinv = pow(p, -1, q)
    for a in rp:
        for b in rq:
            t = ((b - a) * pinv) % q
            out.append(a + p * t)
    return out


def shortest_partner(N: int, r: int, d: int, f: Sequence[int], bound_sup: int) -> tuple[int, list[int]] | None:
    """Shortest g in L_{N,r} (sup-norm < bound_sup) that is irreducible over Q and coprime to f.

    Certified: every lattice vector with squared L2 norm <= (d+1)(bound_sup-1)^2
    is enumerated by Fincke-Pohst (qfminim, flag 2) on the LLL-reduced basis.
    qfminim returns [count, max_norm, V] with count counting +-v separately and
    V holding one column per pair; the column count is iterated, and the
    storage cap is raised until the enumeration is known to be complete
    (fails closed if it cannot be).  Sign representatives suffice because g is
    normalised to a positive leading coefficient.
    """
    if bound_sup <= 1:
        return None
    P = pari()
    cols = root_lattice_columns(N, r, d)
    n = d + 1
    M = P.matrix(n, n, [cols[j][i] for i in range(n) for j in range(n)])
    R = M * P.qflll(M)
    fpoly = fmpz_poly([int(c) for c in f])

    def admissible(g):
        if g[-1] == 0:
            return None
        if g[-1] < 0:
            g = [-x for x in g]
        if not is_primitive(g):
            return None  # content coprime to N divides out (shorter vector found separately); content sharing a factor with N is rejected
        if fmpz_poly(g).gcd(fpoly).degree() > 0 or not is_irreducible_q(g):
            return None
        return g

    # upper bound from the reduced basis: the enumeration radius is then tight
    best = None
    for j in range(n):
        g = admissible([int(R[i, j]) for i in range(n)])
        if g is not None and sup(g) < bound_sup and (best is None or sup(g) < best[0]):
            best = (sup(g), g)
    radius = bound_sup if best is None else best[0]  # every better g has sup-norm <= radius - 1
    if radius <= 1:
        return best
    G = R.mattranspose() * R
    B = (d + 1) * (radius - 1) ** 2
    maxnum = 1 << 16
    while True:
        cnt, _, vecs = P.qfminim(G, B, maxnum, 2)
        ncols = int(P.matsize(vecs)[1])
        if int(cnt) != 2 * ncols:
            raise RuntimeError("unexpected qfminim convention: count is not twice the stored columns")
        if ncols < maxnum:
            break
        if maxnum >= 1 << 26:
            raise RuntimeError("qfminim enumeration exceeded the storage cap; result would not be certified")
        maxnum <<= 2
    for k in range(ncols):
        coeff = [int(vecs[i, k]) for i in range(n)]
        g = [sum(int(R[i, j]) * coeff[j] for j in range(n)) for i in range(n)]
        s = sup(g)
        if s >= radius or (best is not None and s >= best[0]):
            continue
        g = admissible(g)
        if g is None:
            continue
        best = (s, g)
    return best


def exact_pair_floor(N: int, p: int, q: int, d: int, seed_pair: dict | None = None,
                     rng: random.Random | None = None) -> dict:
    """Exact minimum of ||f||_oo ||g||_oo over primitive coprime irreducible pairs of exact
    degree d with a nonzero common root modulo N = pq (roots 0 mod N excluded as in E18).
    Primitivity loses nothing: a content coprime to N divides out of f(r) = 0 (mod N), and a
    content sharing a factor with N would factor N."""
    N, p, q = int(N), int(p), int(q)
    rng = rng or random.Random(1)
    seed = seed_pair or gp_pair(N, d, rng)
    if seed is None:
        best, best_pair, seed_P2 = N, None, None  # certified loop still terminates: ||f||^2 <= P2 <= N
    else:
        best, best_pair, seed_P2 = int(seed["P2"]), (list(seed["f"]), list(seed["g"])), int(seed["P2"])
    count_f = 0
    Hf = 1
    while Hf * Hf < best:  # w.l.o.g. ||f|| <= ||g||, so ||f||^2 <= P2
        # all f with sup-norm exactly Hf, f_d >= 1; f_0 != 0 (else reducible)
        rng_lo = list(range(-Hf, Hf + 1))
        for fd in range(1, Hf + 1):
            for mid in np.ndindex(*([2 * Hf + 1] * (d - 1))):
                coeffs_mid = [x - Hf for x in mid]
                for f0 in rng_lo:
                    if f0 == 0:
                        continue
                    f = [f0] + coeffs_mid + [fd]
                    if sup(f) != Hf:
                        continue
                    if math.gcd(*[abs(c) for c in f]) != 1 or not is_irreducible_q(f):
                        continue
                    count_f += 1
                    for r in roots_mod_N(f, p, q):
                        if r == 0:
                            continue
                        bound = (best + Hf - 1) // Hf  # need Hf * sup(g) <= best  ->  sup(g) <= best // Hf
                        res = shortest_partner(N, r, d, f, bound + 1)
                        if res is None:
                            continue
                        s, g = res
                        if Hf * s <= best and (Hf * s < best or best_pair is None or (f, g) < best_pair):
                            best, best_pair = Hf * s, (f, g)
        Hf += 1
    if best_pair is None:
        raise RuntimeError("no coprime irreducible pair with a nonzero common root and product <= N")
    f, g = best_pair
    # verification
    assert is_primitive(f) and is_primitive(g)
    assert is_irreducible_q(f) and is_irreducible_q(g)
    assert fmpz_poly(f).gcd(fmpz_poly(g)).degree() == 0
    common = [r for r in roots_mod_N(f, p, q) if r != 0 and poly_eval_mod(g, r, N) == 0]
    assert common, "no common root"
    return {"P2": int(best), "f": f, "g": g, "norms": [sup(f), sup(g)], "common_root": common[0],
            "f_enumerated": count_f, "Hf_stop": Hf - 1, "resultant_floor": resultant_floor(N, d),
            "P2_over_floor": best / resultant_floor(N, d), "log_P2_over_log_N": math.log(best) / math.log(N),
            "seed_P2": seed_P2, "certified": True}


def reduced_vectors(N: int, r: int, d: int, radius: int | None = None) -> tuple[int, list[list[int]]]:
    """All nonzero vectors of {g of degree < d : g(r) = 0 mod N} with sup-norm < radius,
    certified by Fincke-Pohst enumeration (L2 radius sqrt(d) (radius - 1)); when radius is
    None the radius is the sup-norm of the best LLL basis vector, so the shortest vector is
    among the results.  Returns (lambda, vectors) with lambda the minimal sup-norm.

    For a monic irreducible f of degree d with f(r) = 0 (mod N) this lattice is the ideal
    p q of Z[alpha] in power-basis coordinates (g <-> g(alpha), N | Norm(g(alpha)) =
    Res(f, g)); it has rank d and determinant N and, unlike the degree-d root lattice,
    contains no copy of Z f, so the enumeration stays small.
    """
    P = pari()
    cols = root_lattice_columns(N, r, d - 1)
    n = d
    M = P.matrix(n, n, [cols[j][i] for i in range(n) for j in range(n)])
    R = M * P.qflll(M)
    lam, vecs_out = None, []
    for j in range(n):
        g = [int(R[i, j]) for i in range(n)]
        if any(g):
            vecs_out.append(g)
            if lam is None or sup(g) < lam:
                lam = sup(g)
    rad = lam + 1 if radius is None else radius
    if rad > 1:
        G = R.mattranspose() * R
        B = n * (rad - 1) ** 2
        maxnum = 1 << 16
        while True:
            cnt, _, vecs = P.qfminim(G, B, maxnum, 2)
            ncols = int(P.matsize(vecs)[1])
            if int(cnt) != 2 * ncols:
                raise RuntimeError("unexpected qfminim convention: count is not twice the stored columns")
            if ncols < maxnum:
                break
            if maxnum >= 1 << 26:
                raise RuntimeError("qfminim enumeration exceeded the storage cap")
            maxnum <<= 2
        vecs_out = []
        for k in range(ncols):
            coeff = [int(vecs[i, k]) for i in range(n)]
            g = [sum(int(R[i, j]) * coeff[j] for j in range(n)) for i in range(n)]
            if any(g) and sup(g) < rad:
                vecs_out.append(g)
                vecs_out.append([-x for x in g])
                lam = min(lam, sup(g))
    return lam, [v for v in vecs_out if sup(v) < rad]


def _phi(gp: Sequence[int], f: Sequence[int], k: int) -> int:
    return max(abs(a + k * b) for a, b in zip(gp, f))


def _convex_argmin(gp: Sequence[int], f: Sequence[int], lo: int, hi: int) -> int:
    """Integer minimiser of the convex function k -> max_i |gp_i + k f_i| on [lo, hi]."""
    a, b = lo, hi
    while b - a > 2:
        m1 = a + (b - a) // 3
        m2 = b - (b - a) // 3
        if _phi(gp, f, m1) <= _phi(gp, f, m2):
            b = m2
        else:
            a = m1
    return min(range(a, b + 1), key=lambda k: _phi(gp, f, k))


def _best_irreducible_lift(gp: Sequence[int], f: Sequence[int], bound: int, max_steps: int = 4096):
    """Among g = gp + k f with k != 0 and sup-norm < bound, the one of least sup-norm that is
    irreducible over Q (or None).  The lifts are visited in nondecreasing sup-norm: on each
    side of zero the function k -> sup(gp + k f) is convex, so walking outward from its
    minimiser on both sides and always advancing the side with the smaller value enumerates
    the candidates in order; the first irreducible one is therefore minimal.  The
    restriction |k| <= bound - 1 loses nothing because f is monic, so the leading
    coefficient of the lift is k and sup(gp + k f) >= |k|.  Returns
    (value, g, complete) where complete is False only if ``max_steps`` reducible lifts were
    seen before the bound was reached (then the result is an upper bound).
    """
    sides = []
    for lo, hi in ((1, bound), (-bound, -1)):
        if lo > hi:
            continue
        k0 = _convex_argmin(gp, f, lo, hi)
        sides.append([k0, k0, k0 - 1, k0 + 1, lo, hi])  # current candidate, centre, next-left, next-right, lo, hi
    # each side yields k in nondecreasing phi order: after the centre, the smaller of phi(left), phi(right)
    def side_next(s):
        k0, centre, left, right, lo, hi = s
        vl = _phi(gp, f, left) if left >= lo else None
        vr = _phi(gp, f, right) if right <= hi else None
        if vl is None and vr is None:
            return None
        if vr is None or (vl is not None and vl <= vr):
            s[2] = left - 1
            return left
        s[3] = right + 1
        return right
    pending = [(_phi(gp, f, s[0]), s[0], s) for s in sides]
    steps = 0
    while pending:
        pending.sort(key=lambda t: t[0])
        v, k, s = pending.pop(0)
        if v >= bound:
            break  # every remaining candidate on every side is at least this large
        g = [a + k * b for a, b in zip(gp, f)]
        if g[-1] < 0:
            g = [-x for x in g]
        if is_primitive(g) and is_irreducible_q(g):
            return v, g, True
        steps += 1
        if steps >= max_steps:
            return None, None, False
        nk = side_next(s)
        if nk is not None:
            pending.append((_phi(gp, f, nk), nk, s))
    return None, None, True


def tiny_partner(N: int, r: int, d: int, f: Sequence[int]) -> dict:
    """For monic irreducible f with f(r) = 0 (mod N): the certified shortest partner of
    degree < d (lambda) and the shortest irreducible partner of degree exactly d.

    Every degree-d partner g satisfies g = g' + k f with g' = g - g_d f of degree < d, k = g_d
    != 0, and ||g'|| <= (1 + ||f||) ||g||; so all degree-d partners with sup-norm <= B are
    obtained from the reduced vectors with sup-norm <= (1 + ||f||) B, and for each g' the
    admissible lifts are scanned in nondecreasing sup-norm (``_best_irreducible_lift``).
    g = g' + k f with g' != 0 is never a multiple of the irreducible f, hence coprime to it.
    The degree-d value is exact when every scan completed (``complete``); the reduced
    minimum lambda always satisfies lambda <= (1 + ||f||) mu for the degree-d minimum mu.
    """
    Hf = sup(f)
    lam, vecs = reduced_vectors(N, r, d)
    best, complete = None, True
    # bootstrap an admissible upper bound: scan lifts of the reduced vectors below a
    # radius that is enlarged until some lift is irreducible; fail closed otherwise
    radius = lam + 1
    for _ in range(8):
        for gp in sorted(vecs, key=sup):
            v, g, ok = _best_irreducible_lift(list(gp) + [0], f, (1 + Hf) * sup(gp) + Hf + 1)
            complete &= ok
            if v is not None and (best is None or v < best[0]):
                best = (v, g)
        if best is not None:
            break
        radius *= 2
        _, vecs = reduced_vectors(N, r, d, radius=radius)
    if best is None:
        return {"lambda": lam, "P_deg_d": None, "complete": False}
    B = best[0]
    _, allv = reduced_vectors(N, r, d, radius=(1 + Hf) * B + 1)
    for gp in allv:
        v, g, ok = _best_irreducible_lift(list(gp) + [0], f, best[0])
        complete &= ok
        if v is not None and v < best[0]:
            best = (v, g)
    return {"lambda": lam, "P_deg_d": best[0], "g": best[1], "complete": complete}


def tiny_f_pairs(N: int, p: int, q: int, d: int, Hf: int = 1) -> dict:
    """The tiny-f route: over all irreducible monic f of degree d with ||f||_oo <= Hf and
    every nonzero root r of f modulo N = pq, the shortest irreducible degree-d partner
    (``tiny_partner``); returns the minimum of ||f||_oo * ||g||_oo together with the
    reduced (degree < d) minimum lambda for the same (f, r).  The minimum is certified
    exactly when ``complete`` is True, i.e. when every lift scan over every (f, r)
    terminated by reaching its bound; then, if the certified (d, d) minimiser has
    ||f||_oo <= Hf, the two agree exactly.  The route scales to large N with one small
    lattice enumeration per root.  Using it requires a root of a small polynomial modulo
    N, which is Exit E1 of notes_beyond_gnfs.md."""
    N, p, q = int(N), int(p), int(q)
    best = None
    n_f = 0
    complete = True
    for mid in np.ndindex(*([2 * Hf + 1] * (d - 1))):
        coeffs_mid = [x - Hf for x in mid]
        for f0 in range(-Hf, Hf + 1):
            if f0 == 0:
                continue
            f = [f0] + coeffs_mid + [1]
            if not is_irreducible_q(f):
                continue
            n_f += 1
            for r in roots_mod_N(f, p, q):
                if r == 0:
                    continue
                t = tiny_partner(N, r, d, f)
                complete &= bool(t["complete"])
                if t["P_deg_d"] is None:
                    continue
                P2 = sup(f) * t["P_deg_d"]
                if best is None or P2 < best[0]:
                    best = (P2, f, t["g"], r, sup(f) * t["lambda"])
    if best is None:
        return {"P2": None, "f_count": n_f, "complete": bool(complete)}
    P2, f, g, r, P2_reduced = best
    return {"P2": int(P2), "P2_reduced": int(P2_reduced), "f": f, "g": g, "norms": [sup(f), sup(g)], "root": r,
            "f_count": n_f, "complete": bool(complete),
            "P2_over_floor": P2 / resultant_floor(N, d), "log_P2_over_log_N": math.log(P2) / math.log(N)}


def tiny_f_scaling(d: int, bits: Sequence[int], count: int, Hf: int = 1, seed: int = 5,
                   family: str = "rsa") -> dict:
    rows = []
    for nbits in bits:
        per = []
        attempts_complete = []
        for i in range(count):
            inst = make_semiprime(nbits, family, seed, i)
            N, p, q = int(inst.N), int(inst.p), int(inst.q)
            t = tiny_f_pairs(N, p, q, d, Hf)
            attempts_complete.append(bool(t["complete"]))
            if t["P2"] is None:
                continue
            per.append({"N": str(N), "P2": t["P2"], "P2_reduced": t["P2_reduced"], "norms": t["norms"], "f": t["f"],
                        "complete": bool(t["complete"]),
                        "P2_over_floor": t["P2_over_floor"], "log_P2_over_log_N": t["log_P2_over_log_N"]})
        rows.append({"nbits": nbits, "count": len(per), "attempted": count,
                     "all_complete": bool(per) and all(attempts_complete),
                     "mean_log_P2_over_log_N": float(np.mean([r["log_P2_over_log_N"] for r in per])) if per else float("nan"),
                     "mean_P2_over_floor": float(np.mean([r["P2_over_floor"] for r in per])) if per else float("nan"),
                     "mean_log2_P2_minus_log2_N_over_d": float(np.mean([math.log2(r["P2"]) - math.log2(float(r["N"])) / d for r in per])) if per else float("nan"),
                     "instances": per})
    x = np.array([math.log2(float(r["N"])) for row in rows for r in row["instances"]])
    y = np.array([math.log2(r["P2"]) for row in rows for r in row["instances"]])
    if x.size > 2:
        (s, b), cov = np.polyfit(x, y, 1, cov=True)
        fit = {"slope": float(s), "slope_se": float(math.sqrt(cov[0, 0])), "n": int(x.size)}
    else:
        fit = {"slope": float("nan"), "slope_se": float("nan"), "n": int(x.size)}
    return {"d": d, "Hf": Hf, "rows": rows, "fit": fit}


def pair_floor_experiment(d: int, bits: Sequence[int], counts: Sequence[int], seed: int = 5,
                          family: str = "rsa") -> dict:
    rng = random.Random(seed)
    rows = []
    for nbits, count in zip(bits, counts):
        per = []
        for i in range(count):
            inst = make_semiprime(nbits, family, seed, i)
            N, p, q = int(inst.N), int(inst.p), int(inst.q)
            gp = gp_pair(N, d, rng)
            ex = exact_pair_floor(N, p, q, d, seed_pair=gp, rng=rng)
            per.append({"N": str(N), "exact_P2": ex["P2"], "exact_norms": ex["norms"], "f": ex["f"], "g": ex["g"],
                        "P2_over_floor": ex["P2_over_floor"], "log_P2_over_log_N": ex["log_P2_over_log_N"],
                        "construction_P2": None if gp is None else gp["P2"],
                        "construction_norms": None if gp is None else gp["norms"],
                        "construction_all_minima": None if gp is None else gp["admissible_basis_norms"],
                        "construction_gp_ratio": None if gp is None else gp["gp_max_over_N_1_minus_1_over_d"],
                        "construction_over_exact": None if gp is None else gp["P2"] / ex["P2"],
                        "f_enumerated": ex["f_enumerated"]})
        rows.append({"nbits": nbits, "count": count,
                     "mean_log_P2_over_log_N": float(np.mean([r["log_P2_over_log_N"] for r in per])),
                     "mean_P2_over_floor": float(np.mean([r["P2_over_floor"] for r in per])),
                     "mean_construction_log_P2_over_log_N": float(np.mean([math.log(r["construction_P2"]) / math.log(float(r["N"])) for r in per if r["construction_P2"]])),
                     "mean_construction_over_exact": float(np.mean([r["construction_over_exact"] for r in per if r["construction_over_exact"]])),
                     "instances": per})
    x = np.array([math.log2(float(r["N"])) for row in rows for r in row["instances"]])
    y = np.array([math.log2(r["exact_P2"]) for row in rows for r in row["instances"]])
    yc = np.array([math.log2(r["construction_P2"]) for row in rows for r in row["instances"] if r["construction_P2"]])
    xc = np.array([math.log2(float(r["N"])) for row in rows for r in row["instances"] if r["construction_P2"]])
    (s, b), cov = np.polyfit(x, y, 1, cov=True) if x.size > 2 else ((float("nan"), float("nan")), np.full((2, 2), float("nan")))
    (sc, bc), covc = np.polyfit(xc, yc, 1, cov=True) if xc.size > 2 else ((float("nan"), float("nan")), np.full((2, 2), float("nan")))
    return {"d": d, "floor_exponent": 1.0 / d, "gp_construction_exponent": 2.0 * (d - 1) / d ** 2,
            "rows": rows,
            "fit": {"exact_slope": float(s), "exact_slope_se": float(math.sqrt(cov[0, 0])),
                    "construction_slope": float(sc), "construction_slope_se": float(math.sqrt(covc[0, 0]))}}
