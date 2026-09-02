"""Numerical checks behind the hypothesis-thinning floor, the Farey covering families and the head-slack class."""
import math
import random
from fractions import Fraction
from math import gcd

import pytest

from factorlab.experiments.farey_cover import best_single_form, covered_interval, farey_family, single_form_cover
from factorlab.gen import make_semiprime


def test_alignment_lemma_single_class_forces_sublattice():
    """Lemma H1: if a N p^{-1} + b p mod q^2 takes the same value at p' and p' + q (both = sigma mod q, sigma a unit) then
    b = a N sigma^{-2} mod q; and conversely aligned forms give a constant class along the whole progression."""
    rng = random.Random(3)
    for _ in range(200):
        q = rng.choice([5, 7, 9, 11, 13, 15, 21, 25])
        N = rng.randrange(2, 10 ** 6)
        if gcd(N, q) != 1:
            continue
        sigma = rng.choice([s for s in range(1, q) if gcd(s, q) == 1])
        a = rng.randrange(1, 50)
        b = rng.randrange(1, 50)
        gamma = N * pow(sigma, -2, q) % q
        aligned = (b - a * gamma) % q == 0
        t0 = rng.randrange(0, 1000)
        vals = set()
        for t in range(t0, t0 + 6):
            p = sigma + q * t
            vals.add((a * N * pow(p, -1, q * q) + b * p) % (q * q))
        if aligned:
            assert len(vals) == 1
        else:
            assert len(vals) > 1  # two consecutive t already give different classes


def test_sublattice_wide_form_count():
    """Lemma B': for fixed a, the wide forms (a <= b <= ceil(C a)) with b = gamma a mod q number at most (C - 1) a / q + 3."""
    C = 2.0
    for q in (3, 7, 16, 50):
        for gamma in range(q):
            for a in range(1, 300):
                count = sum(1 for b in range(a, math.ceil(C * a) + 1) if (b - gamma * a) % q == 0)
                assert count <= (C - 1) * a / q + 3


@pytest.mark.parametrize("lam", [0.28, 0.34])
def test_farey_family_covers_and_has_the_predicted_scale(lam):
    """Theorem D(c): the Farey family of order Q = round((N/L)^{1/5}) covers I and has cost O(L^{3/5} N^{-1/10})."""
    sp = make_semiprime(44, "rsa", 5, 0)
    N = int(sp.N)
    L = N ** lam
    Q = max(2, round((N / L) ** 0.2))
    rng = random.Random(11)
    P0 = L ** 0.6 * N ** (-0.1)
    for _ in range(4):
        p0 = rng.uniform(math.sqrt(N / 2) + L, math.sqrt(N) - L)
        fam = farey_family(N, p0, L, Q, 2.0)
        assert fam["covers"]
        assert fam["P"] <= 20 * P0 + 5
        assert fam["cost"] <= 20 * P0 + 10
        # independent coverage check with a fine grid
        lo, hi = p0 - L / 2, p0 + L / 2
        ivs = [covered_interval(f["a"], f["b"], N, f["W"]) for f in fam["forms"]]
        for k in range(0, 2001):
            p = lo + (hi - lo) * k / 2000
            assert any(x0 - 1e-9 * L <= p <= x1 + 1e-9 * L for x0, x1 in ivs)


def test_fermat_type_interval_is_covered_by_one_form_at_L_N14():
    """Theorem D(b): an interval within L of sqrt N is covered by the form (1, 1) at cost O(L N^{-1/4})."""
    sp = make_semiprime(44, "rsa", 5, 1)
    N = int(sp.N)
    for lam in (0.28, 0.34):
        L = N ** lam
        p0 = math.sqrt(N) - L
        r = single_form_cover(N, p0, L, 1, 1)
        assert r["covers"]
        assert r["cost"] <= 3 + 6 * L * N ** (-0.25)
        # and the best single near form is at least as cheap as this one only if it is (1, 1) itself or better
        best = best_single_form(N, p0, L, 4, 2.0)
        assert best is not None and best["cost"] <= r["cost"] + 1e-9


def test_head_slack_first_prefix_multiplier_closed_form():
    """Proposition S: w_1^{(m)} = (d - m) beta / (d (beta - 1)) for the prefix certificate."""
    from latticelab.profile_floor import prefix_volume_certificate

    def frac(x):
        if hasattr(x, "p") and hasattr(x, "q"):
            return Fraction(int(x.p), int(x.q))
        return Fraction(x)

    for d, beta, m in [(30, 8, 10), (60, 20, 40), (100, 40, 60), (200, 40, 180)]:
        w, z = prefix_volume_certificate(d, beta, m)
        assert frac(w[0]) == Fraction(d - m) * beta / (d * (beta - 1))
        assert frac(z) == Fraction(m, d)
