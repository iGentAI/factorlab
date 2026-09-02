"""Classical exponential-time methods with explicit work counters.

* trial_division      O(N^{1/2})   work key: division
* fermat              O(|q-p|^2/N^{1/2}) work key: sqrt_test
* lehman              O(N^{1/3})   work key: sqrt_test
* hart_olf            heuristic    work key: sqrt_test
* squfof              O(N^{1/4})   work key: candidate (forms examined)
"""

from __future__ import annotations

import time

from ..numth import mpz, isqrt, isqrt_ceil, is_square, gcd, small_primes, iroot
from ..registry import register
from ..result import Work, success, failure


@register("trial_division", primary_key="division", description="divide by odd numbers up to sqrt(N) (or `limit`)")
def trial_division(N, limit=None, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    lim = isqrt(N) if limit is None else mpz(limit)
    if N % 2 == 0:
        w.add("division")
        return success("trial_division", N, 2, w, "division", time.perf_counter() - t0)
    d = mpz(3)
    while d <= lim:
        w.add("division")
        if N % d == 0:
            return success("trial_division", N, d, w, "division", time.perf_counter() - t0)
        d += 2
    return failure("trial_division", N, w, "division", time.perf_counter() - t0)


@register("fermat", primary_key="sqrt_test", description="a^2 - N = b^2 search upward from ceil(sqrt N)")
def fermat(N, max_steps=10**7, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    a = isqrt_ceil(N)
    r = a * a - N
    for _ in range(int(max_steps)):
        w.add("sqrt_test")
        if is_square(r):
            b = isqrt(r)
            d = a - b
            if 1 < d < N:
                return success("fermat", N, d, w, "sqrt_test", time.perf_counter() - t0, a=int(a), b=int(b))
        r += 2 * a + 1
        a += 1
    return failure("fermat", N, w, "sqrt_test", time.perf_counter() - t0)


@register("lehman", primary_key="sqrt_test", description="Lehman 1974: Fermat on 4kN over k <= N^{1/3} after trial division to N^{1/3}")
def lehman(N, **_):
    """Lehman's method (Crandall-Pomerance Alg. 5.1.2 form)."""
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    B = int(iroot(N, 3)[0]) + 1
    for d in small_primes(B + 1):
        w.add("division")
        if N % d == 0:
            return success("lehman", N, d, w, "sqrt_test", time.perf_counter() - t0, stage="trial")
    for k in range(1, B + 1):
        fourkN = 4 * k * N
        a = isqrt_ceil(fourkN)
        # a ranges over ceil(sqrt(4kN)) .. floor(sqrt(4kN) + N^{1/6}/(4 sqrt k))
        amax = isqrt(fourkN) + (isqrt(mpz(B)) // (4 * isqrt(mpz(k)) if k > 1 else 4)) + 1
        # (rigorous bound is N^{1/6}/(4 sqrt k); B^{1/2} ~ N^{1/6})
        while a <= amax:
            w.add("sqrt_test")
            c = a * a - fourkN
            if c >= 0 and is_square(c):
                b = isqrt(c)
                g = gcd(a + b, N)
                if 1 < g < N:
                    return success("lehman", N, g, w, "sqrt_test", time.perf_counter() - t0, k=k)
            a += 1
    return failure("lehman", N, w, "sqrt_test", time.perf_counter() - t0)


@register("hart_olf", primary_key="sqrt_test", description="Hart's one-line factoring: s=ceil(sqrt(iN)), test s^2 mod N square", deterministic=True)
def hart_olf(N, max_iter=10**7, **_):
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    iN = mpz(0)
    for i in range(1, int(max_iter) + 1):
        iN += N
        s = isqrt_ceil(iN)
        m = s * s % N
        w.add("sqrt_test")
        if is_square(m):
            t = isqrt(m)
            g = gcd(s - t, N)
            if 1 < g < N:
                return success("hart_olf", N, g, w, "sqrt_test", time.perf_counter() - t0, i=i)
    return failure("hart_olf", N, w, "sqrt_test", time.perf_counter() - t0)


@register("squfof", primary_key="candidate", description="Shanks SQUFOF with multipliers (forms examined counted)")
def squfof(N, multipliers=(1, 3, 5, 7, 11, 3 * 5, 3 * 7, 3 * 11, 5 * 7, 5 * 11, 7 * 11, 3 * 5 * 7), max_forms=None, **_):
    """Square forms factorisation (Gower-Wagstaff description).

    Forward cycle on the principal form of discriminant 4kN:
        b_i = floor((P_0 + P_{i-1}) / Q_i),  P_i = b_i Q_i - P_{i-1},
        Q_{i+1} = Q_{i-1} + b_i (P_{i-1} - P_i).
    When Q_i is a perfect square for *even* i, take the square root of the form
    and run the reverse cycle until P_i = P_{i-1}; then gcd(N, P_i) is a factor
    (possibly trivial, in which case the forward cycle simply continues).
    """
    N = mpz(N)
    t0 = time.perf_counter()
    w = Work()
    if is_square(N):
        return success("squfof", N, isqrt(N), w, "candidate", time.perf_counter() - t0)
    bound = int(max_forms) if max_forms else int(4 * isqrt(2 * isqrt(N))) + 100
    for k in multipliers:
        D = k * N
        P0 = isqrt(D)
        # state: P = P_{i-1}, Qprev = Q_{i-1}, Qcur = Q_i  (initially i = 1)
        P, Qprev, Qcur = P0, mpz(1), D - P0 * P0
        if Qcur == 0:
            g = gcd(P0, N)
            if 1 < g < N:
                return success("squfof", N, g, w, "candidate", time.perf_counter() - t0, k=k)
            continue
        i = 1
        while i < bound:
            w.add("candidate")
            b = (P0 + P) // Qcur
            Pn = b * Qcur - P
            Qn = Qprev + b * (P - Pn)
            P, Qprev, Qcur = Pn, Qcur, Qn
            i += 1  # Qcur is now Q_i, P is P_{i-1}
            if i % 2 == 0 and is_square(Qcur):
                r = isqrt(Qcur)
                # reverse cycle from the square root of the form
                b = (P0 - P) // r
                Pr = b * r + P
                Qrprev = r
                Qrcur = (D - Pr * Pr) // r
                for _ in range(bound):
                    w.add("candidate")
                    b = (P0 + Pr) // Qrcur
                    Prn = b * Qrcur - Pr
                    if Prn == Pr:
                        g = gcd(N, Pr)
                        if 1 < g < N:
                            return success("squfof", N, g, w, "candidate", time.perf_counter() - t0, k=k, forms=i)
                        break
                    Qrn = Qrprev + b * (Pr - Prn)
                    Pr, Qrprev, Qrcur = Prn, Qrcur, Qrn
                # trivial square: continue forward cycle
    return failure("squfof", N, w, "candidate", time.perf_counter() - t0)
