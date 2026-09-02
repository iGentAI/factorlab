"""N2: state sharing among continued fractions of sqrt(kN), k <= K.

State after j steps of the CF expansion of sqrt(D), D = kN, is (m_j, d_j) with
    m_{j+1} = d_j a_j - m_j,  d_{j+1} = (D - m_{j+1}^2)/d_j,  a_j = floor((a_0 + m_j)/d_j).
SQUFOF looks for d_j a perfect square at even j.

Question: do distinct k share symbolic states, so that a DAG over k-intervals
has K^theta nodes with theta < 1?  We measure two proxies:

1. Partial-quotient prefix sharing: number of distinct prefixes (a_1..a_j)
   across k <= K as a function of j.  If prefixes become unique by j ~ c
   (constant), no compression exists at the level of quotient sequences.
2. Symbolic interval count: for depth j, the number of maximal k-intervals on
   which the sequence (a_1..a_j) is constant.  This is exactly the DAG width
   at depth j for the natural symbolic representation.

Prediction: distinct prefixes ~ min(K, C^j) with C ~ e^{entropy} ~ 10, so
uniqueness by j ~ log K / log 10, i.e. theta = 1 in any useful depth.
"""

from __future__ import annotations

from collections import Counter

from ..numth import mpz, isqrt


def cf_sqrt_quotients(D, depth: int):
    """First ``depth`` partial quotients a_1..a_depth of sqrt(D) (D non-square)."""
    D = mpz(D)
    a0 = isqrt(D)
    m, d, a = mpz(0), mpz(1), a0
    out = []
    for _ in range(depth):
        m = d * a - m
        d = (D - m * m) // d
        if d == 0:
            break
        a = (a0 + m) // d
        out.append(int(a))
    return tuple(out)


def prefix_sharing(N, K: int, depth: int):
    """Distinct quotient prefixes at each depth j <= depth over k = 1..K,
    and the number of maximal constant k-intervals (DAG width)."""
    N = mpz(N)
    seqs = [cf_sqrt_quotients(k * N, depth) for k in range(1, K + 1)]
    distinct, intervals = [], []
    for j in range(1, depth + 1):
        pref = [s[:j] for s in seqs]
        distinct.append(len(set(pref)))
        runs = 1 + sum(1 for i in range(1, K) if pref[i] != pref[i - 1])
        intervals.append(runs)
    return {"K": K, "depth": depth, "distinct_prefixes": distinct, "interval_count": intervals}


def quotient_entropy(N, K: int, depth: int) -> float:
    """Empirical entropy (bits) of the partial quotient a_1 over k <= K, to
    compare with Gauss-Kuzmin (~3.43 bits)."""
    import math
    c = Counter(cf_sqrt_quotients(k * mpz(N), 1)[:1] for k in range(1, K + 1))
    tot = sum(c.values())
    return -sum(v / tot * math.log2(v / tot) for v in c.values())
