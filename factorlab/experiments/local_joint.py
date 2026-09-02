"""E8: exact local theory of the joint small-prime structure of p-1, q-1, p+1, q+1 given N.

Theorem L.  Let l be an odd prime with l not dividing N = pq, let j >= 1 and
n = N mod l^j.  Conditionally on N, p mod l^j is uniform on the units (the
argument of D4: q = N p^{-1} is then determined), so the four events
    A- = [l^j | p-1] = [p = 1],   B- = [l^j | q-1] = [p = n],
    A+ = [l^j | p+1] = [p = -1],  B+ = [l^j | q+1] = [p = -n]
each have probability 1/phi(l^j), and they coincide or are disjoint according
to the coincidences among {1, n, -1, -n} in (Z/l^j)^*:
    n =  1 : A- = B- and A+ = B+ (two disjoint pairs);
    n = -1 : A- = B+ and A+ = B- ;
    otherwise the four events are pairwise disjoint.
Hence corr(1_{A-}, 1_{B-} | n) = 1 if n = 1 and -1/(phi(l^j) - 1) otherwise,
and the expected number of distinct shifted primes divisible by l^j is
2/phi(l^j) when n = +-1 and 4/phi(l^j) otherwise.  The same holds for l = 2,
j >= 2 (p odd, n = N mod 2^j).  The marginal law of each valuation is
N-independent (D4); only the joint law depends on N.

Consequence (heuristic, tested): for the union event "some shifted prime is
semismooth", moduli with N = +-1 modulo many small prime powers concentrate
their small-prime luck on pairs of shifted primes instead of spreading it over
four, so they should have a slightly lower union rate, and p-1, q-1 should be
positively rather than negatively correlated.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from ..gen import make_semiprime
from ..numth import small_primes


def _val(x: int, l: int, cap: int) -> int:
    v = 0
    while v < cap and x % l == 0:
        x //= l
        v += 1
    return v


def joint_valuation_model(l: int, j: int, n: int) -> dict[tuple[int, int, int, int], float]:
    """Exact law of (v(p-1), v(q-1), v(p+1), v(q+1)), valuations capped at j,
    given N = n (mod l^j), for p uniform on the units of Z/l^j."""
    m = l ** j
    if math.gcd(n, l) != 1:
        raise ValueError("l divides N")
    c = Counter()
    units = [u for u in range(1, m) if math.gcd(u, l) == 1]
    for u in units:
        q = (n * pow(u, -1, m)) % m
        c[(_val(u - 1, l, j), _val(q - 1, l, j), _val(u + 1, l, j), _val(q + 1, l, j))] += 1
    return {k: v / len(units) for k, v in c.items()}


def pairing_type(n: int, m: int) -> str:
    """'same' if N = 1, 'cross' if N = -1, else 'disjoint' (mod m)."""
    n %= m
    if n == 1 % m:
        return "same"
    if n == (m - 1) % m:
        return "cross"
    return "disjoint"


def theoretical_corr(l: int, j: int, n: int) -> float:
    """corr(1[l^j | p-1], 1[l^j | q-1] | N = n mod l^j)."""
    phi = (l - 1) * l ** (j - 1)
    if pairing_type(n, l ** j) == "same":
        return 1.0
    # disjoint events of probability a = 1/phi each: corr = -a/(1-a) = -1/(phi-1)
    return -1.0 / (phi - 1)


def local_joint_test(nbits: int = 48, count: int = 6000, seed: int = 5, family: str = "rsa",
                     moduli=((3, 1), (5, 1), (7, 1), (11, 1), (13, 1), (2, 3), (3, 2)), alpha: float = 0.001) -> dict:
    """Chi-square test of the exact joint model, pooled over residue classes of N,
    plus empirical versus theoretical correlations of [l^j | p-1] and [l^j | q-1]."""
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    out = {"nbits": nbits, "count": count, "per_modulus": {}, "pass": True}
    for l, j in moduli:
        m = l ** j
        hist = defaultdict(Counter)
        ind = defaultdict(lambda: ([], []))
        for inst in insts:
            p, q, N = int(inst.p), int(inst.q), int(inst.N)
            n = N % m
            if math.gcd(n, l) != 1:
                continue
            t = (_val(p - 1, l, j), _val(q - 1, l, j), _val(p + 1, l, j), _val(q + 1, l, j))
            hist[n][t] += 1
            a, b = ind[n]
            a.append(int(t[0] >= j))
            b.append(int(t[1] >= j))
        chi2 = 0.0
        dof = 0
        outside = 0
        corr_rows = {}
        for n, c in hist.items():
            model = joint_valuation_model(l, j, n)
            tot = sum(c.values())
            for t, pr in model.items():
                e = tot * pr
                if e >= 1.0:
                    chi2 += (c.get(t, 0) - e) ** 2 / e
                    dof += 1
            dof -= 1
            outside += sum(k for t, k in c.items() if t not in model)
            a, b = np.array(ind[n][0], float), np.array(ind[n][1], float)
            if a.std() > 0 and b.std() > 0:
                corr_rows[str(n)] = {"type": pairing_type(n, m), "count": tot,
                                     "corr_obs": float(np.corrcoef(a, b)[0, 1]), "corr_theory": theoretical_corr(l, j, n)}
        pval = 1.0 - stats.chi2.cdf(chi2, max(dof, 1))
        row = {"chi2": chi2, "dof": dof, "p": pval, "outside_support": outside, "corr_by_residue": corr_rows}
        out["per_modulus"][f"{l}^{j}"] = row
        if pval < alpha or outside:
            out["pass"] = False
    return out


def pairing_score(N: int, L: int = 30) -> float:
    """sum over odd primes l <= L with N = +-1 (mod l) of log(l)/(l-1), plus the
    2-adic term log(2)/2 if N = +-1 (mod 8): a measure of how much small-prime
    divisibility of the four shifted primes is forced to pair up."""
    s = 0.0
    for l in small_primes(L + 1):
        if l == 2:
            if N % 8 in (1, 7):
                s += math.log(2) / 2
        elif N % l in (1, l - 1):
            s += math.log(l) / (l - 1)
    return s


def union_effect(nbits: int = 48, count: int = 6000, c: float = 1 / 6, seed: int = 5, family: str = "rsa",
                 L_: int = 30) -> dict:
    """Does the pairing structure forced by N's residues change the union rate
    'some shifted prime is (N^c, N^{2c})-semismooth' and the sign of
    corr(p-1 semismooth, q-1 semismooth)?  Moduli are split at the median
    pairing score; the rates are compared with standard errors."""
    from .smooth_profiles import _factor, _top_two_from, multiplicative_order, stage1_exponent, exact_success
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    B1 = 2.0 ** (c * nbits)
    B2 = B1 * B1
    L = stage1_exponent(int(B1))

    def ss(fac):
        l1, l2 = _top_two_from(fac)
        return (l2 <= B1) and (l1 <= B2)

    rows = []
    degen = []
    for inst in insts:
        p, q, N = int(inst.p), int(inst.q), int(inst.N)
        fpm, fqm = _factor(p - 1), _factor(q - 1)
        rows.append((pairing_score(N, L_), ss(fpm), ss(fqm), ss(_factor(p + 1)), ss(_factor(q + 1))))
        op = multiplicative_order(2, p, fpm)
        oq = multiplicative_order(2, q, fqm)
        dp, dq = op // math.gcd(int(op), int(L)), oq // math.gcd(int(oq), int(L))
        degen.append(bool(dp <= B2 and dq <= B2 and dp == dq))
    score = np.array([r[0] for r in rows])
    S = np.array([r[1:] for r in rows], dtype=bool)
    degen = np.array(degen, dtype=bool)
    med = float(np.median(score))
    hi, lo = score > med, score <= med
    out = {"nbits": nbits, "count": count, "c": c, "median_score": med, "groups": {}}
    for name, mask in (("high_pairing", hi), ("low_pairing", lo)):
        s = S[mask]
        any4 = s.any(axis=1)
        n = int(mask.sum())
        a, b = s[:, 0].astype(float), s[:, 1].astype(float)
        out["groups"][name] = {
            "count": n,
            "any_of_four": float(any4.mean()), "se": float(math.sqrt(any4.mean() * (1 - any4.mean()) / n)),
            "mean_single": float(s.mean()),
            "corr_pm1_qm1": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0,
            "corr_pm1_pp1": float(np.corrcoef(a, s[:, 2].astype(float))[0, 1]),
            "degenerate_equal_order_rate": float(degen[mask].mean()),
        }
    # finer: N mod 5 in {1,4} (paired at 5) vs {2,3}
    for l in (5, 7, 8):
        m = l
        paired = np.array([int(i.N) % m in (1, m - 1) for i in insts])
        res = {}
        for name, mask in (("paired", paired), ("unpaired", ~paired)):
            s = S[mask]
            any4 = s.any(axis=1)
            a, b = s[:, 0].astype(float), s[:, 1].astype(float)
            res[name] = {"count": int(mask.sum()), "any_of_four": float(any4.mean()),
                         "corr_pm1_qm1": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0,
                         "degenerate_equal_order_rate": float(degen[mask].mean())}
        out[f"by_N_mod_{m}"] = res
    return out
