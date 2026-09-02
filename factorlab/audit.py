"""Statistical audit of the prime generator.

These checks exist so that any "signal" later found in an experiment can be
separated from artefacts of instance generation.  All tests return a dict with
a ``pass`` flag, a p-value (where meaningful) and the raw statistics.

Checks
------
* ``residue_uniformity``      : p mod l uniform over nonzero residues (chi^2).
* ``bit_uniformity``          : every interior bit of p is ~Bernoulli(1/2).
* ``density_profile``         : count of primes in sub-bins of [2^(b-1), 2^b)
                                follows the Li(x) profile (chi^2 against
                                expected proportions), i.e. sampling is uniform
                                over *primes*, not skewed within the interval.
* ``next_prime_gap_bias``     : demonstrates that next_prime sampling is
                                gap-size biased whereas rejection sampling is
                                not (mean preceding gap vs. expectation).
* ``semiprime_residues``      : N mod l distribution for generated semiprimes
                                matches the product-of-uniform model.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

import gmpy2
from scipy import stats

from .gen import random_prime, random_odd, rng_from_seed, make_semiprime
from .numth import mpz, small_primes

__all__ = [
    "residue_uniformity", "bit_uniformity", "density_profile",
    "next_prime_gap_bias", "semiprime_residues", "run_all",
]


def _chi2_uniform(counts: list[int]) -> tuple[float, float]:
    n = sum(counts)
    k = len(counts)
    exp = n / k
    chi2 = sum((c - exp) ** 2 / exp for c in counts)
    pval = 1.0 - stats.chi2.cdf(chi2, k - 1)
    return chi2, pval


def residue_uniformity(nbits: int, n: int = 4000, moduli: Iterable[int] = (3, 5, 7, 11, 13, 17, 19, 23),
                       seed: int = 1, alpha: float = 0.001) -> dict:
    rng = rng_from_seed(seed)
    primes = [random_prime(rng, nbits) for _ in range(n)]
    out = {"nbits": nbits, "n": n, "per_modulus": {}, "pass": True}
    for l in moduli:
        counts = [0] * (l - 1)
        for p in primes:
            r = int(p % l)
            if r == 0:
                continue  # only possible if p == l
            counts[r - 1] += 1
        chi2, pval = _chi2_uniform(counts)
        out["per_modulus"][l] = {"chi2": chi2, "p": pval, "counts": counts}
        if pval < alpha:
            out["pass"] = False
    return out


def bit_uniformity(nbits: int, n: int = 4000, seed: int = 2, alpha: float = 0.001) -> dict:
    """Interior bits 1..nbits-2 of uniformly sampled primes should be fair coins.

    Bit 1 (the 2s bit) is *not* expected to be exactly fair for tiny nbits
    because p mod 4 is equidistributed only asymptotically; we still report it.
    """
    rng = rng_from_seed(seed)
    ones = [0] * nbits
    for _ in range(n):
        p = int(random_prime(rng, nbits))
        for b in range(nbits):
            ones[b] += (p >> b) & 1
    worst_p, worst_bit = 1.0, None
    per_bit = {}
    for b in range(1, nbits - 1):
        # two-sided binomial test
        pval = stats.binomtest(ones[b], n, 0.5).pvalue
        per_bit[b] = {"ones": ones[b], "p": pval}
        if pval < worst_p:
            worst_p, worst_bit = pval, b
    # Bonferroni over nbits-2 interior bits
    return {"nbits": nbits, "n": n, "worst_bit": worst_bit, "worst_p": worst_p,
            "pass": worst_p * max(1, nbits - 2) >= alpha, "per_bit": per_bit}


def density_profile(nbits: int, n: int = 6000, bins: int = 8, seed: int = 3, alpha: float = 0.001) -> dict:
    """Counts per sub-bin should follow int_bin dx/ln x, not be flat.

    For a uniform sample over primes in [2^(b-1), 2^b), the expected fraction in
    a bin [u, v) is (Li(v)-Li(u)) / (Li(2^b)-Li(2^(b-1))).  We test against
    that profile.
    """
    rng = rng_from_seed(seed)
    lo, hi = 2 ** (nbits - 1), 2 ** nbits
    edges = [lo + (hi - lo) * i // bins for i in range(bins + 1)]
    counts = [0] * bins
    for _ in range(n):
        p = int(random_prime(rng, nbits))
        idx = min(bins - 1, (p - lo) * bins // (hi - lo))
        counts[idx] += 1

    weights = [_li(edges[i + 1]) - _li(edges[i]) for i in range(bins)]
    tot = sum(weights)
    expected = [n * w / tot for w in weights]
    chi2 = sum((c - e) ** 2 / e for c, e in zip(counts, expected))
    pval = 1.0 - stats.chi2.cdf(chi2, bins - 1)
    # Also test the (wrong) flat model to show the profile is detectable
    chi2_flat, pval_flat = _chi2_uniform(counts)
    return {"nbits": nbits, "n": n, "counts": counts, "expected": expected,
            "chi2": chi2, "p": pval, "p_flat_model": pval_flat, "pass": pval >= alpha}


def _li(x: float) -> float:
    """Logarithmic integral via scipy (offset irrelevant for differences)."""
    from scipy.special import expi
    return float(expi(math.log(x)))


def next_prime_gap_bias(nbits: int = 40, n: int = 3000, seed: int = 4) -> dict:
    """Mean *preceding* prime gap for next_prime-sampled vs rejection-sampled primes.

    For a size-biased sample the expected preceding gap is E[g^2]/E[g] which
    (under Cramer-type heuristics with exponential gaps of mean ln x) is about
    2 ln x, versus ln x for an unbiased sample.  The ratio of the two means
    should therefore be close to 2; we report it and a Welch t-test p-value
    that the means differ.
    """
    rng = rng_from_seed(seed)
    rej_gaps, np_gaps = [], []
    for _ in range(n):
        p = random_prime(rng, nbits)
        rej_gaps.append(int(p - gmpy2.prev_prime(p)))
        x = random_odd(rng, nbits)
        p2 = gmpy2.next_prime(x)
        np_gaps.append(int(p2 - gmpy2.prev_prime(p2)))
    m_rej = sum(rej_gaps) / n
    m_np = sum(np_gaps) / n
    t = stats.ttest_ind(np_gaps, rej_gaps, equal_var=False)
    ln_x = (nbits - 0.5) * math.log(2)
    rejection_unbiased = abs(m_rej / ln_x - 1) < 0.15
    next_prime_biased = bool(t.pvalue < 1e-6) and m_np > m_rej
    return {"nbits": nbits, "n": n, "mean_gap_rejection": m_rej, "mean_gap_next_prime": m_np,
            "ratio": m_np / m_rej, "ln_x": ln_x, "welch_p": float(t.pvalue),
            "rejection_unbiased": rejection_unbiased,
            "next_prime_biased": next_prime_biased,
            "pass": rejection_unbiased and next_prime_biased}


def semiprime_residues(nbits: int = 64, n: int = 3000, moduli=(3, 5, 7, 11, 13), seed: int = 5,
                       family: str = "balanced", alpha: float = 0.001) -> dict:
    """N = pq mod l: product of two independent uniform nonzero residues is
    uniform on nonzero residues, so N mod l should be uniform on 1..l-1."""
    out = {"nbits": nbits, "n": n, "family": family, "per_modulus": {}, "pass": True}
    Ns = [make_semiprime(nbits, family, seed, i).N for i in range(n)]
    for l in moduli:
        counts = [0] * (l - 1)
        for N in Ns:
            r = int(N % l)
            if r:
                counts[r - 1] += 1
        chi2, pval = _chi2_uniform(counts)
        out["per_modulus"][l] = {"chi2": chi2, "p": pval}
        if pval < alpha:
            out["pass"] = False
    return out


def run_all(nbits: int = 48, verbose: bool = True) -> dict:
    res = {
        "residue_uniformity": residue_uniformity(nbits),
        "bit_uniformity": bit_uniformity(nbits),
        "density_profile": density_profile(nbits),
        "next_prime_gap_bias": next_prime_gap_bias(min(nbits, 40)),
        "semiprime_residues": semiprime_residues(2 * nbits),
    }
    if verbose:
        for k, v in res.items():
            print(f"{k:24s} pass={v['pass']}")
    return res
