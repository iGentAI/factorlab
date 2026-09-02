"""Deterministic ECM with a fixed curve list (E21).

Proposition V of notes_probabilistic.md: if a fixed list of one-sided-gcd
methods has pairwise distinct success signatures on a set of primes, running
the list factors every product of two distinct primes of the set.  This module
is the algorithm on N: Suyama curves sigma = 6, 7, 8, ... in order, each with
stage 1 at B1 (lcm exponentiation by every prime power <= B1) and a
baby-step giant-step stage 2 detecting a residual order <= B2, all over Z/N,
with every exposure taken as a gcd.  There is no randomness.  A *two-sided*
event (a gcd equal to N) moves on to the next curve; a one-sided gcd is a
factor.

Parameters follow E20: with a public balance bound C on q/p, the scale
x = sqrt(N/C) puts both primes of N = pq in [x, Cx]; B1 = round(x^{1/u}) and
B2 = round(x^{2/u}) (rounded independently, as in the experiment).  Stage 1
costs ~1.44 B1 ladder steps; stage 2 costs O(sqrt(B2)) ladder steps plus a
product tree and a multipoint evaluation of degree ~sqrt(B2) (python-flint,
which works over the composite modulus), so a curve costs B1^{1+o(1)} and the
algorithm N^{1/(2u)+o(1)} times the list length.  Conjecture E asserts the
list length is O(log N); E20 measures it.

Stage 2.  Let Q be the stage-1 point, m = floor(sqrt(B2)) + 1, T = B2 // m,
R = B2 - T m (0 <= R < m).  The baby steps are jQ (j = 1..m-1) and the giant
steps i(mQ) (i = 1..T), both by differential addition chains.  Every integer
v in [1, B2] is a baby index (v < m), a giant multiple i m, or i m + r with
1 <= r <= m-1 and either i < T or (i = T and r <= R); conversely every cell
tested has both its values i m +- j in [1, B2].  So the residual order d of Q
is <= B2 *iff* some baby or giant is the identity (Z = 0) or some giant and
baby share an x-coordinate (x(P) = x(P') iff P = +-P').  The collisions are
tested by evaluating F(t) = prod_j (t - x_j) at the giants i < T and the
partial product over j <= R at giant T, and taking gcds with N.  Identities
that vanish modulo every prime factor are filtered (they have no affine
x-coordinate) and the collision search continues.  A two-sided aggregate is
backtracked over giants (increasing i) and then baby differences (increasing
j); the scan stops at the first two-sided atom.  This is safe: with N = pq
and exposure labels (Proposition V' of notes_probabilistic.md: construction
failure, residual order d <= B2, or d > B2) that differ, let a = min(d_p, d_q)
= t m + r be the smaller residual order; every atom scanned before the cell
(t, r) -- or the identity at a when r = 0 or a < m -- has both values below
a, hence no relation on either side, and the atom at a is one-sided because
the larger order does not divide a.  So the first atom found is one-sided
whenever the labels differ, and only equal labels end as two-sided.
The curve is first checked for nonsingularity: gcd(a24 (a24 - 1), N) is a
factor if proper; if it equals N, gcd(a24, N) and gcd(a24 - 1, N) are tried
(opposite singular types on the two sides factor), and only then is the
curve skipped.
"""

from __future__ import annotations

import math
import time

import flint

from ..numth import mpz, gcd, invert, isqrt
from ..registry import register
from ..result import Work, success, failure
from .ecm import suyama_curve, xdbl, xadd, ladder, stage1_exponents
from .strassen import product_tree


def fixed_list_parameters(N, u: float = 3.0, C: float = 2.0) -> tuple[int, int]:
    """B1 = round(x^{1/u}), B2 = round(x^{2/u}) with x = sqrt(N/C) (E20 conventions)."""
    ln_x = 0.5 * (math.log(int(N)) - math.log(C))
    B1 = max(5, int(round(math.exp(ln_x / u))))
    B2 = int(round(math.exp(2.0 * ln_x / u)))
    return B1, B2


def _batch_inverse(vals, N):
    """Montgomery's batch inversion; every value must be a unit modulo N."""
    n = len(vals)
    prefix = [mpz(1)] * (n + 1)
    for i, v in enumerate(vals):
        prefix[i + 1] = prefix[i] * v % N
    inv = invert(prefix[n], N)
    out = [None] * n
    for i in range(n - 1, -1, -1):
        out[i] = inv * prefix[i] % N
        inv = inv * vals[i] % N
    return out


def _chain(P, a24, N, count, w: Work):
    """[P, 2P, ..., count*P] when step is P, or [G, 2G, ...] for giants: differential chain."""
    X, Z = P
    out = [P]
    if count >= 2:
        out.append(xdbl(X, Z, a24, N, w))
    for _ in range(3, count + 1):
        Xp, Zp = out[-1]
        Xq, Zq = out[-2]
        out.append(xadd(Xp, Zp, X, Z, Xq, Zq, N, w))  # kP = (k-1)P + P, difference (k-2)P
    return out


def _product_poly_chunked(ctx, roots, w: Work, block: int = 4096):
    """Product polynomial over roots without materialising all linear leaves at once."""
    if not roots:
        return ctx([1])
    level = [product_tree(ctx, roots[i:i + block], w) for i in range(0, len(roots), block)]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(level[i] * level[i + 1])
            w.add("poly_deg", level[i].degree() + level[i + 1].degree())
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def stage2_bsgs(X, Z, a24, N, B2: int, w: Work, poly_block: int = 4096):
    """Residual order of Q = (X : Z) at most B2?  Returns (g, detail): g = 1 (no), a proper
    factor of N, or N (two-sided: every relation found vanishes modulo every prime factor).
    Exact: a relation is tested iff its value lies in [1, B2]."""
    N = mpz(N)
    B2 = int(B2)
    if B2 < 2:
        return mpz(1), None
    m = int(isqrt(mpz(B2))) + 1
    T = B2 // m            # giants i = 1..T
    R = B2 - T * m         # at giant T only babies j <= R keep i m + j <= B2
    babies = _chain((X, Z), a24, N, m - 1, w)  # jQ, j = 1..m-1 (baby m would equal giant 1)
    G = ladder(m, X, Z, a24, N, w)
    giants = _chain(G, a24, N, T, w)
    allZ = [b[1] for b in babies] + [g_[1] for g_ in giants]
    prodZ = mpz(1)
    for z in allZ:
        prodZ = prodZ * z % N
    w.add("mulmod", len(allZ))
    gZ = gcd(prodZ, N)
    w.add("gcd")
    two_sided_identity = False
    unit_mask = [True] * len(allZ)
    if 1 < gZ < N:
        return gZ, "identity"
    if gZ == N:
        # Some sampled point is the identity on at least one CRT side.  Extract
        # a one-sided identity immediately; identities on both sides carry no
        # affine x-coordinate, so filter them and continue the collision search.
        for k, z in enumerate(allZ):
            gz = gcd(mpz(z), N)
            w.add("gcd")
            if 1 < gz < N:
                return gz, "identity"
            if gz == N:
                unit_mask[k] = False
                two_sided_identity = True
    nb = len(babies)
    baby_idx = [k for k in range(nb) if unit_mask[k]]                 # 0-based: baby j = k + 1
    giant_idx = [k for k in range(len(giants)) if unit_mask[nb + k]]  # 0-based: giant i = k + 1
    babies = [babies[k] for k in baby_idx]
    giants = [giants[k] for k in giant_idx]
    if not babies or not giants:
        return (N if two_sided_identity else mpz(1)), ("two_sided_identity" if two_sided_identity else None)
    allZ = [b[1] for b in babies] + [g_[1] for g_ in giants]
    invZ = _batch_inverse(allZ, N)
    w.add("mulmod", 3 * len(allZ))
    xb = [int(b[0] * invZ[k] % N) for k, b in enumerate(babies)]
    xg = [int(g_[0] * invZ[len(babies) + k] % N) for k, g_ in enumerate(giants)]
    w.add("mulmod", len(allZ))
    # rows: full baby set at giants i < T; only babies j <= R at giant T
    full_rows = [k for k, gi in enumerate(giant_idx) if gi + 1 < T]
    last_row = [k for k, gi in enumerate(giant_idx) if gi + 1 == T]
    if full_rows:
        ctx = flint.fmpz_mod_poly_ctx(int(N))
        F = _product_poly_chunked(ctx, xb, w, poly_block)
        # Multipoint evaluation builds its own subproduct tree; block the giant
        # points so its temporary memory is independent of the full degree.
        for start in range(0, len(full_rows), poly_block):
            rows = full_rows[start:start + poly_block]
            vals = [mpz(int(v)) for v in F.multipoint_evaluate([xg[k] for k in rows])]
            w.add("poly_deg", len(rows))
            prod = mpz(1)
            for v in vals:
                prod = prod * v % N
            w.add("mulmod", len(vals))
            g = gcd(prod, N)
            w.add("gcd")
            if 1 < g < N:
                return g, "bsgs"
            if g == N:
                for k, v in zip(rows, vals):
                    gk = gcd(v, N)
                    w.add("gcd")
                    if 1 < gk < N:
                        return gk, "bsgs"
                    if gk == N:
                        for kb in range(len(xb)):
                            gj = gcd(mpz(xg[k] - xb[kb]), N)
                            w.add("gcd")
                            if 1 < gj < N:
                                return gj, "bsgs"
                            if gj == N:
                                return N, "two_sided"
                return N, "two_sided"
    last_babies = [k for k, bj in enumerate(baby_idx) if bj + 1 <= R]
    for k in last_row:
        v = mpz(1)
        for kb in last_babies:
            v = v * (xg[k] - xb[kb]) % N
        w.add("mulmod", len(last_babies))
        g = gcd(v, N)
        w.add("gcd")
        if 1 < g < N:
            return g, "bsgs"
        if g == N:
            for kb in last_babies:
                gj = gcd(mpz(xg[k] - xb[kb]), N)
                w.add("gcd")
                if 1 < gj < N:
                    return gj, "bsgs"
                if gj == N:
                    return N, "two_sided"
            return N, "two_sided"
    return ((N, "two_sided_identity") if two_sided_identity else (mpz(1), None))


@register("fixed_list_ecm", primary_key="mulmod",
          description="deterministic ECM: Suyama sigma = 6, 7, ... at B1 = round(x^{1/u}), exact BSGS stage 2 to B2 = round(x^{2/u}), x = sqrt(N/C), until a one-sided gcd",
          deterministic=True)
def fixed_list_ecm(N, u: float = 3.0, C: float = 2.0, B1=None, B2=None, max_curves: int = 400,
                   sigma0: int = 6, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    if N % 2 == 0:
        return success("fixed_list_ecm", N, 2, w, "mulmod", time.perf_counter() - t0)
    r = isqrt(N)
    if r * r == N:
        return success("fixed_list_ecm", N, r, w, "mulmod", time.perf_counter() - t0, stage="square", curve=0)
    if B1 is None or B2 is None:
        b1, b2 = fixed_list_parameters(N, u, C)
        B1 = int(B1) if B1 is not None else b1
        B2 = int(B2) if B2 is not None else b2
    B1, B2 = int(B1), int(B2)
    exps = stage1_exponents(B1)
    two_sided = 0
    for ci in range(int(max_curves)):
        sigma = sigma0 + ci
        w.add("curve")
        curve, g = suyama_curve(sigma, N)
        if curve is None:
            w.add("gcd")
            if 1 < g < N:
                return success("fixed_list_ecm", N, g, w, "mulmod", time.perf_counter() - t0,
                               curve=ci + 1, sigma=sigma, stage="den", B1=B1, B2=B2, two_sided=two_sided)
            two_sided += 1
            continue
        a24, X, Z = curve
        g = gcd(a24 * (a24 - 1) % N, N)  # singular curve (A = -2 or 2) modulo a factor
        w.add("gcd")
        if g == N:
            # both sides singular: crossed types (a24 = 0 mod p, a24 = 1 mod q) still factor
            for atom in (a24, a24 - 1):
                ga = gcd(atom % N, N)
                w.add("gcd")
                if 1 < ga < N:
                    g = ga
                    break
        if 1 < g < N:
            return success("fixed_list_ecm", N, g, w, "mulmod", time.perf_counter() - t0,
                           curve=ci + 1, sigma=sigma, stage="den", B1=B1, B2=B2, two_sided=two_sided)
        if g == N:
            two_sided += 1
            continue
        for pe in exps:
            X, Z = ladder(pe, X, Z, a24, N, w)
        g = gcd(Z, N)
        w.add("gcd")
        if 1 < g < N:
            return success("fixed_list_ecm", N, g, w, "mulmod", time.perf_counter() - t0,
                           curve=ci + 1, sigma=sigma, stage="1", B1=B1, B2=B2, two_sided=two_sided)
        if g == N:
            two_sided += 1
            continue
        g, detail = stage2_bsgs(X, Z, a24, N, B2, w)
        if 1 < g < N:
            return success("fixed_list_ecm", N, g, w, "mulmod", time.perf_counter() - t0,
                           curve=ci + 1, sigma=sigma, stage="2", detail=detail, B1=B1, B2=B2, two_sided=two_sided)
        if g == N:
            two_sided += 1
    return failure("fixed_list_ecm", N, w, "mulmod", time.perf_counter() - t0,
                   B1=B1, B2=B2, curves=int(max_curves), two_sided=two_sided)
