"""The Schnorr-Lenstra class-group method, on PARI binary quadratic forms.

Let D = -kN (kN = 3 mod 4) or -4kN otherwise, k squarefree and coprime to N,
and Cl(D) the class group of primitive positive definite forms.  Genus theory
gives Cl(D) a 2-rank of omega(D) - 1, and the ambiguous classes (order 2)
correspond to the factorisations |D'| = d e of D' = |D| or |D|/4 into coprime
parts, up to swapping; a reduced ambiguous form (a, b, c) has b = 0, b = a or
a = c and reveals its divisor as a, a or 2a +- b.  Such a divisor is useful
iff it separates p from q.

Algorithm: take a prime form f of discriminant D; stage 1 computes
g = f^{L_odd}, L_odd = lcm{l^e <= B1, l odd}; if the odd part of ord(f) divides
L_odd then g has 2-power order and repeated squaring reaches the identity
through an ambiguous form.  Stage 2 first strips the 2-part, g_odd = g^{2^a}
with a = two_part_bound(D) > log_2 h(D) (from h(D) < sqrt|D| log|D|, so the
2-adic valuation of every element order is below a), and runs BSGS in Cl(D) on g_odd (baby steps g_odd^i, giant steps g_odd^{jm},
m = ceil(sqrt(B2))) to find a multiple e <= B2 of its order, which exists when
the residual odd part is a single prime in (B1, B2]; g^{e} then has 2-power
order and is descended to an ambiguous form.  Failure modes, recorded in the
result meta: the odd part is not semismooth; <f> has trivial 2-part (g =
identity); the ambiguous form reached is useless (its divisor does not split
N); no split prime form below the search limit.  Work is counted in form
compositions (``composition``).
"""

from __future__ import annotations

import math
import random
import time

from ..numth import mpz, gcd, small_primes
from ..registry import register
from ..result import Work, success, failure

_pari = None


def pari():
    global _pari
    if _pari is None:
        import cypari2
        _pari = cypari2.Pari()
        _pari.allocatemem(256 * 1024 * 1024)
    return _pari


def discriminant(k: int, N) -> int:
    m = int(k) * int(N)
    return -m if m % 4 == 3 else -4 * m


def identity_form(D: int):
    P = pari()
    if D % 4 == 0:
        return P.Qfb(1, 0, -D // 4)
    return P.Qfb(1, 1, (1 - D) // 4)


def odd_stage1_exponent(B1: int) -> int:
    L = 1
    for l in small_primes(int(B1) + 1):
        if l == 2:
            continue
        pe = l
        while pe * l <= B1:
            pe *= l
        L *= pe
    return L


def two_part_bound(D: int) -> int:
    """a with 2^a > h(D) >= any element order, from h(D) < sqrt|D| log|D|."""
    aD = abs(int(D))
    return (math.isqrt(aD) * (aD.bit_length() + 2)).bit_length() + 1


def ambiguous_divisor(form) -> list[int]:
    """Candidate divisors of |D'| read off a reduced ambiguous form."""
    a, b, c = int(form[0]), int(form[1]), int(form[2])
    return [a, c, abs(2 * a + b), abs(2 * a - b), abs(b)]


def descend_to_ambiguous(g, ident, w: Work, max_steps: int = 200):
    """Square g until the identity; return the last non-identity element (of
    order 2) or None if g is the identity or the order is not a power of two."""
    x = g
    if x == ident:
        return None, "trivial_2_part"
    for _ in range(max_steps):
        y = x * x
        w.add("composition")
        if y == ident:
            return x, "ambiguous"
        x = y
    return None, "order_not_2_power"


def bsgs_order_multiple(g, ident, B2: int, w: Work):
    """Smallest e in [1, B2] with g^e = identity found by BSGS, or None."""
    m = int(math.isqrt(int(B2))) + 1
    baby = {}
    x = ident
    for i in range(m):
        key = (int(x[0]), int(x[1]), int(x[2]))
        if key not in baby:
            baby[key] = i
        x = x * g
        w.add("composition")
    G = x  # g^m
    y = ident
    best = None
    for j in range(1, m + 1):
        y = y * G
        w.add("composition")
        key = (int(y[0]), int(y[1]), int(y[2]))
        if key in baby:
            e = j * m - baby[key]
            if e > 0 and e <= B2:
                best = e if best is None else min(best, e)
                break
    return best


def factor_from_ambiguous(amb, N):
    N = mpz(N)
    for d in ambiguous_divisor(amb):
        if d:
            gg = gcd(mpz(d), N)
            if 1 < gg < N:
                return gg
    return None


@register("schnorr_lenstra", primary_key="composition",
          description="Schnorr-Lenstra class group method in Cl(-kN): odd stage 1 to B1, BSGS stage 2 on the odd residual to B2, factor from an ambiguous form",
          deterministic=False)
def schnorr_lenstra(N, B1=1000, B2=None, k=1, seed=0, forms=1, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    P = pari()
    k = int(k)
    g0 = gcd(mpz(k), N)
    if 1 < g0 < N:
        return success("schnorr_lenstra", N, g0, w, "composition", time.perf_counter() - t0)
    D = discriminant(k, N)
    ident = identity_form(D)
    L_odd = odd_stage1_exponent(int(B1))
    rng = random.Random(seed)
    reasons = []
    candidates = [l for l in small_primes(2000) if l > 2 and D % l != 0 and int(P.kronecker(D, l)) == 1]
    for attempt in range(int(forms)):
        if not candidates:
            reasons.append("no_split_prime")
            break
        l = rng.choice(candidates[:40])
        f = P.qfbprimeform(D, l)
        g = f ** L_odd
        w.add("composition", int(1.5 * L_odd.bit_length()))
        amb, reason = descend_to_ambiguous(g, ident, w)
        if amb is None and reason == "order_not_2_power" and B2:
            # isolate the odd residual order, then search it by BSGS
            a = two_part_bound(D)
            g_odd = g ** (2 ** a)
            w.add("composition", a)
            e = bsgs_order_multiple(g_odd, ident, int(B2), w)
            if e is not None:
                g2 = g ** e
                w.add("composition", int(1.5 * e.bit_length()))
                amb, reason = descend_to_ambiguous(g2, ident, w)
            else:
                reason = "odd_part_not_semismooth"
        if amb is None:
            reasons.append(reason)
            continue
        d = factor_from_ambiguous(amb, N)
        if d is not None:
            return success("schnorr_lenstra", N, d, w, "composition", time.perf_counter() - t0,
                           k=k, D=str(D), prime_form=l, attempt=attempt, B1=int(B1), B2=int(B2) if B2 else None)
        reasons.append("useless_ambiguous_form")
    return failure("schnorr_lenstra", N, w, "composition", time.perf_counter() - t0,
                   k=k, D=str(D), reasons=reasons, B1=int(B1), B2=int(B2) if B2 else None)
