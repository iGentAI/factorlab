"""E3: do small-coefficient polynomial Lehman forms have unusually degenerate
local minima, and do such cells cover more than the Hoelder count predicts?

A form u(p, q) = sum a_{ij} p^i q^j (integer coefficients |a_ij| <= A)
restricts to the Laurent polynomial g(p) = u(p, N/p) = sum a_ij N^j p^{i-j}.
At a local minimum p* in J = [sqrt(N/4), sqrt N] a window [g(p*), g(p*) + w)
covers a p-length governed by the first materially non-vanishing even
derivative d:  coverage ~ 2 (d! w / g^{(d)}(p*))^{1/d}.

Hypothesis (H1) of docs/notes_barrier.md (exclusion 2) says that the relevant
derivative keeps its natural scale, i.e. there is no systematic cancellation
among the monomials' contributions.  The honest, degree-independent measure
of cancellation is the *cancellation ratio*

    eta_d(p*) = g^{(d)}(p*) / sum_{monomials} |a_ij N^j (m)_d p*^{m-d}|,   m = i - j,

which is 1 when all contributions have the same sign and ~0 under cancellation.
(The earlier normalisation by A N^{(e-2)/2} was wrong for forms whose
effective degree is below e.)

Log coordinates.  With s = sqrt N and p = s e^t, h(t) := g(s e^t) =
sum a_ij s^{i+j} e^{(i-j) t}; the involution p <-> q is t -> -t, so the
symmetric part of u gives the even part of h and the antisymmetric part the
odd part.  Since h' = p g' and h'' = p^2 g'' + p g', at a critical point
    g''(p*) = h''(t*) / p*^2        exactly,
so cancellation in g'' at a minimum is cancellation in h'' there.  (Away from
critical points the two differ: an antisymmetric form has h''(0) = 0 but
g''(s) = -g'(s)/s != 0.)

Structural source of cancellation found by the survey: a top-degree part that
is a perfect power c (q - p)^k of the antisymmetric linear form, k >= 3.  Then
h = c (-2s)^k sinh^k t + (lower-degree terms) has a k-fold zero at t = 0, and a
lower-degree term b_1 s^{k-1} t displaces the minimum to
    t* ~ (b_1 / (c k 2^k s))^{1/(k-1)},   h''(t*) ~ s^{k - (k-2)/(k-1)}
against the natural scale s^k: a reduction by s^{-(k-2)/(k-1)} (k = 3:
s^{-1/2}).  In p-coordinates g''(p*) ~ s^{(k-2)^2/(k-1)} >> 1/s = g'' of the
Lehman cell (1,1), so such a cell covers less than a Lehman cell at the same
window width by a factor ~ s^{-(1 + (k-2)^2/(k-1))/2} <= N^{-3/8}.  The driver
reports, for the most degenerate minima, whether the top-degree part is such a
perfect power, the well depth (height of the adjacent maximum), and a cubic
reference 2 (6 w / |g'''|)^{1/3} for the regime w > well depth.

All evaluations use 256-bit mpfr (gmpy2): monomials can exceed g by many
orders of magnitude, and doubles cannot resolve windows in that regime.
"""

from __future__ import annotations

import itertools
import math
import random

import gmpy2
import numpy as np
from gmpy2 import mpfr

PREC = 256
gmpy2.get_context().precision = PREC


def monomials(e: int):
    """(i, j) with 1 <= i + j <= e and i != j (monomials with i == j restrict to
    constants N^i and never affect derivatives)."""
    return [(i, j) for i in range(e + 1) for j in range(e + 1) if 1 <= i + j <= e and i != j]


def _ff(m: int, order: int) -> int:
    """Falling factorial (m)_order = m (m-1) ... (m-order+1)."""
    out = 1
    for t in range(order):
        out *= (m - t)
    return out


class Form:
    def __init__(self, coeffs: dict, N):
        self.coeffs = {k: int(v) for k, v in coeffs.items() if v}
        self.N = mpfr(N)
        self._Npow = {}

    def _Nj(self, j: int):
        if j not in self._Npow:
            self._Npow[j] = self.N ** j
        return self._Npow[j]

    def _terms(self, p, order: int):
        p = mpfr(p)
        for (i, j), a in self.coeffs.items():
            m = i - j
            ff = _ff(m, order)
            if ff == 0:
                continue
            yield a * self._Nj(j) * ff * p ** (m - order)

    def deriv(self, p, order: int):
        tot = mpfr(0)
        for t in self._terms(p, order):
            tot += t
        return tot

    def term_scale(self, p, order: int):
        tot = mpfr(0)
        for t in self._terms(p, order):
            tot += abs(t)
        return tot

    def cancellation_ratio(self, p, order: int) -> float:
        s = self.term_scale(p, order)
        return float(self.deriv(p, order) / s) if s > 0 else 0.0

    def value(self, p):
        return self.deriv(p, 0)

    def all_critical_points(self, lo: float, hi: float) -> list:
        """All real roots of g' in [lo, hi], sorted, as mpfr: double-precision
        roots of p^{e+1} g'(p) refined by Newton in 256-bit arithmetic."""
        if not self.coeffs:
            return []
        e = max(i + j for (i, j) in self.coeffs)
        shift = e + 1
        poly = np.zeros(2 * e + 2)
        Nf = float(self.N)
        for (i, j), a in self.coeffs.items():
            m = i - j
            if m == 0:
                continue
            poly[m - 1 + shift] += a * (Nf ** j) * m
        if not np.any(poly):
            return []
        roots = np.roots(poly[::-1])
        out = []
        for z in roots:
            if abs(z.imag) > 1e-6 * max(1.0, abs(z.real)) or z.real <= 0:
                continue
            p = mpfr(float(z.real))
            for _ in range(200):
                d1, d2 = self.deriv(p, 1), self.deriv(p, 2)
                if d2 == 0:
                    break
                step = d1 / d2
                p -= step
                if p <= 0 or abs(step) < mpfr(2) ** (-PREC + 20) * p:
                    break
            if p > 0 and lo <= p <= hi and abs(self.deriv(p, 1)) <= mpfr(2) ** (-PREC + 40) * (self.term_scale(p, 1) + 1):
                out.append(p)
        out.sort()
        merged = []
        for p in out:
            if not merged or abs(p - merged[-1]) > mpfr(1e-20) * p:
                merged.append(p)
        return merged

    def first_material_derivative(self, p, max_order: int = 6):
        """(order, value) of the first derivative of order >= 2 whose cancellation
        ratio exceeds 2^-(PREC-60), i.e. which is not zero to working precision."""
        thr = float(mpfr(2) ** (-PREC + 60))
        for order in range(2, max_order + 1):
            s = self.term_scale(p, order)
            d = self.deriv(p, order)
            if s > 0 and abs(d) > thr * s:
                return order, d
        return None, mpfr(0)

    def local_minima(self, lo: float, hi: float) -> list:
        mins = []
        for p in self.all_critical_points(lo, hi):
            order, d = self.first_material_derivative(p)
            if order is not None and order % 2 == 0 and d > 0:
                mins.append(p)
        return mins

    def coverage(self, pstar, w, lo: float, hi: float):
        """Measure of the connected component of {p in [lo, hi] : g(p) < g(p*) + w}
        containing the local minimum p*.  Between consecutive critical points g
        is monotone, so on each side we walk the critical-point partition: a
        piece whose far endpoint is below the level is traversed entirely;
        otherwise the crossing is bisected inside the piece."""
        pstar = mpfr(pstar)
        level = self.value(pstar) + mpfr(w)
        cps = self.all_critical_points(lo, hi)
        right_pts = [c for c in cps if c > pstar] + [mpfr(hi)]
        left_pts = [c for c in reversed(cps) if c < pstar] + [mpfr(lo)]

        def walk(points, start):
            cur = start
            for nxt in points:
                if self.value(nxt) < level:
                    cur = nxt
                    continue
                a, b = cur, nxt
                for _ in range(200):
                    mid = (a + b) / 2
                    if self.value(mid) < level:
                        a = mid
                    else:
                        b = mid
                return a
            return cur

        right = walk(right_pts, pstar)
        left = walk(left_pts, pstar)
        return float(right - left)

    def well_depth(self, pstar, lo: float, hi: float):
        """Height of the lower of the two adjacent critical values above g(p*)
        (inf if p* has no adjacent critical point in [lo, hi] on either side)."""
        pstar = mpfr(pstar)
        cps = self.all_critical_points(lo, hi)
        g0 = self.value(pstar)
        right = [c for c in cps if c > pstar]
        left = [c for c in cps if c < pstar]
        depths = []
        if right:
            depths.append(float(self.value(right[0]) - g0))
        if left:
            depths.append(float(self.value(left[-1]) - g0))
        return min(depths) if depths else float("inf")

    def symmetry_split(self) -> dict:
        """Norms of the symmetric and antisymmetric parts under p <-> q."""
        sym = asym = 0.0
        seen = set()
        for (i, j), a in self.coeffs.items():
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            b = self.coeffs.get((j, i), 0)
            sym += ((a + b) / 2) ** 2 * (1 if i == j else 2)
            asym += ((a - b) / 2) ** 2 * (1 if i == j else 2)
        return {"sym_norm": math.sqrt(sym), "asym_norm": math.sqrt(asym),
                "asym_fraction": math.sqrt(asym) / (math.sqrt(sym) + math.sqrt(asym)) if sym + asym > 0 else None}

    def h_deriv(self, t, order: int):
        """d^order/dt^order of h(t) = g(sqrt(N) e^t) = sum a_ij s^{i+j} e^{(i-j) t}."""
        s = gmpy2.sqrt(self.N)
        t = mpfr(t)
        tot = mpfr(0)
        for (i, j), a in self.coeffs.items():
            m = i - j
            tot += a * s ** (i + j) * mpfr(m) ** order * gmpy2.exp(m * t)
        return tot

    def top_degree_perfect_power(self):
        """If the top-degree homogeneous part equals c (q - p)^k for an integer
        c != 0 and k >= 3 (coefficient of p^i q^{k-i} is c (-1)^i C(k, i)),
        return (c, k); else None.  k <= 2 is excluded because a linear or
        quadratic top part does not produce the k-fold zero of h at t = 0 that
        drives the cancellation mechanism (for k = 2 the curvature stays at
        natural scale).  Monomials with i == j are constants on pq = N and are
        ignored, so the pattern is checked on the i != j coefficients."""
        if not self.coeffs:
            return None
        k = max(i + j for (i, j) in self.coeffs)
        if k < 3:
            return None
        top = {(i, j): a for (i, j), a in self.coeffs.items() if i + j == k}
        # coefficient of q^k (i = 0) is c
        c = top.get((0, k))
        if not c:
            return None
        for i in range(k + 1):
            j = k - i
            if i == j:
                continue
            expected = c * (-1) ** i * math.comb(k, i)
            if top.get((i, j), 0) != expected:
                return None
        return (c, k)


def hoelder_prediction(form: Form, pstar, w, max_order: int = 6) -> dict:
    """Prediction 2 (d! w / g^{(d)})^{1/d} from the first *materially*
    non-vanishing even derivative at a strict local minimum.

    Rule: for even d = 2, 4, ..., max_order with g^{(d)}(p*) > 0, let
    h_d = (d! w / g^{(d)})^{1/d} be the half-width predicted by that term alone.
    Order d is accepted iff its Taylor term at h_d dominates every other even
    term of order <= max_order at the same h_d.  The smallest accepted d is
    used; if none is accepted the prediction is reported as unavailable.
    """
    pstar = mpfr(pstar)
    w = mpfr(w)
    evens = list(range(2, max_order + 1, 2))
    derivs = {d: form.deriv(pstar, d) for d in evens}
    for d in evens:
        gd = derivs[d]
        if gd <= 0:
            continue
        h = (math.factorial(d) * w / gd) ** (mpfr(1) / d)
        own = gd * h ** d / math.factorial(d)
        if all(abs(derivs[j]) * h ** j / math.factorial(j) <= own for j in evens if j != d):
            return {"derivatives": {k: float(v) for k, v in derivs.items()}, "order_used": d,
                    "prediction": float(2 * h), "available": True}
    return {"derivatives": {k: float(v) for k, v in derivs.items()}, "order_used": None,
            "prediction": None, "available": False}


def cubic_reference(form: Form, pstar, w) -> float:
    """2 (6 w / |g'''|)^{1/3}: the coverage scale of a window once it exceeds
    the well depth of a shallow minimum produced by a cubic term (the
    component then passes the adjacent maximum and the cubic governs).  A
    reference scale, not a bound."""
    g3 = abs(form.deriv(pstar, 3))
    return float(2 * (6 * mpfr(w) / g3) ** (mpfr(1) / 3)) if g3 > 0 else float("inf")


def random_form(rng: random.Random, e: int, A: int, N) -> Form:
    return Form({mono: rng.randrange(-A, A + 1) for mono in monomials(e)}, N)


def survey(N, e: int, A: int, trials: int, seed: int = 0, exhaustive: bool = False) -> dict:
    """Cancellation-ratio statistics at local minima inside J."""
    hi = math.sqrt(float(N))
    lo = hi / 2
    rng = random.Random(seed)
    monos = monomials(e)
    if exhaustive:
        it = (Form(dict(zip(monos, cs)), N) for cs in itertools.product(range(-A, A + 1), repeat=len(monos)))
    else:
        it = (random_form(rng, e, A, N) for _ in range(trials))
    eta2 = []
    best = []
    n_forms = n_min = 0
    for f in it:
        if not f.coeffs:
            continue
        n_forms += 1
        for p in f.local_minima(lo, hi):
            n_min += 1
            r2 = f.cancellation_ratio(p, 2)
            eta2.append(r2)
            best.append((r2, dict(f.coeffs), float(p / hi), f.cancellation_ratio(p, 3),
                         f.symmetry_split()["asym_fraction"], f.top_degree_perfect_power()))
    best.sort(key=lambda t: t[0])
    eta2a = np.array(eta2) if eta2 else np.array([np.nan])

    def frac_below(x):
        return float(np.mean(eta2a < x)) if eta2 else None

    n_pp = sum(1 for b in best if b[5] is not None)
    n_pp_small = sum(1 for b in best if b[5] is not None and b[0] < 1e-2)
    return {
        "e": e, "A": A, "forms": n_forms, "local_minima_in_J": n_min,
        "eta2_min": float(eta2a.min()) if eta2 else None,
        "eta2_median": float(np.median(eta2a)) if eta2 else None,
        "frac_eta2_below_1e-1": frac_below(1e-1),
        "frac_eta2_below_1e-2": frac_below(1e-2), "frac_eta2_below_1e-4": frac_below(1e-4),
        "minima_with_perfect_power_top": n_pp,
        "minima_with_eta2_below_1e-2": int(np.sum(eta2a < 1e-2)) if eta2 else 0,
        "of_which_perfect_power_top": n_pp_small,
        "most_degenerate": [{"eta2": b[0], "coeffs": {f"p{i}q{j}": v for (i, j), v in b[1].items()},
                             "pstar_over_sqrtN": b[2], "eta3": b[3], "asym_fraction": b[4],
                             "perfect_power_top": b[5]} for b in best[:5]],
    }


def coverage_comparison(N, coeffs: dict, pstar_over_sqrtN: float, w_values) -> list[dict]:
    """Exact coverage of the cell's local minimum vs Lehman (1,1), the Hoelder
    prediction, the well depth and the inflection reference."""
    hi = math.sqrt(float(N))
    lo = hi / 2
    form = Form({(int(k[1]), int(k[3])): v for k, v in coeffs.items()}, N)
    p0 = pstar_over_sqrtN * hi
    mins = form.local_minima(lo, hi)
    if not mins:
        return []
    pstar = min(mins, key=lambda p: abs(float(p) - p0))
    lehman = Form({(0, 1): 1, (1, 0): 1}, N)
    depth = form.well_depth(pstar, lo, hi)
    out = []
    for w in w_values:
        cov = form.coverage(pstar, w, lo, hi)
        cov_lehman = lehman.coverage(mpfr(hi), w, lo, hi)
        pred = hoelder_prediction(form, pstar, w)
        out.append({"w": w, "coverage": cov, "lehman11_coverage": cov_lehman,
                    "ratio_to_lehman": cov / cov_lehman if cov_lehman else None,
                    "hoelder_order_used": pred["order_used"], "hoelder_prediction": pred["prediction"],
                    "coverage_over_prediction": (cov / pred["prediction"]) if pred["available"] and pred["prediction"] > 0 else None,
                    "well_depth": depth, "w_exceeds_well_depth": w > depth,
                    "cubic_reference": cubic_reference(form, pstar, w)})
    return out
