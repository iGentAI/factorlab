"""The profile floor (Layer A of docs/lattice_barrier_plan.md): a conditional linear-programming theorem with exact certificates.

Setting.  For a basis of a d-dimensional lattice let l_i = log ||b_i^*||.  For 1 <= i <= d - 1 let beta_i = min(beta, d - i + 1) and let
B_i be the projected block pi_i(L(b_i, ..., b_{i+beta_i-1})), where pi_i is the orthogonal projection onto span(b_1..b_{i-1})^perp (for
i > d - beta + 1 this is the tail projection pi_i(L)).  Since b_i^* = pi_i(b_i) is a nonzero vector of B_i, l_i >= log lambda_1(B_i), and
vol(B_i) = prod_{j=i}^{i+beta_i-1} ||b_j^*||.  The block-GH floor condition  lambda_1(B_i) >= e^{-eps} chat(beta_i) vol(B_i)^{1/beta_i},
chat(n) = V_n^{-1/n} the dimensionless Gaussian-heuristic constant, therefore implies the linear constraints

    (A_i)   (1 - 1/beta_i) l_i - (1/beta_i) sum_{j=i+1}^{i+beta_i-1} l_j  >=  c_i := log chat(beta_i) - eps,     1 <= i <= d - 1,

the (beta, eps)-admissible profiles.  Theorem (conditional profile floor): let (y, z) solve e_1 = sum_i y_i a_i + z 1 with a_i the coefficient
vector of (A_i); then z = 1/d, and if all y_i >= 0 every admissible profile with sum S satisfies l_1 >= sum_i y_i c_i + S/d.  The profile
with every (A_i) tight attains this bound; it is the unique minimiser when every y_i > 0 (for beta = d one has y_1 = 1, y_i = 0 for i >= 2,
and uniqueness fails).  It is a theorem about a fixed basis satisfying the block-GH condition; it says nothing by itself about which
algorithms produce such bases.

Certificates.  `dual_certificate` solves the triangular dual system in exact rational arithmetic and `verify_certificate` re-checks the
identity and y >= 0 independently: exact dual feasibility.  `floor_l1` evaluates the bound as an arb ball (python-flint: rigorous
enclosures of log chat via lgamma, of the dot product with the exact multipliers, and of exp), so a threshold decision made by
`decide_floor_vs_target` -- floor ball and target ball disjoint, with precision doubled until they are -- is rigorous.  `floor_l1_float`
and `dual_float` are double-precision pre-screens, never certificates.

Prefix volumes.  Theorem (prefix-volume floor): for every (beta, eps)-admissible profile l of volume S and every m = 1..d-1, the prefix sum
P_m(l) = l_1 + ... + l_m is at least P_m of the all-tight profile of the same volume, i.e. vol(F_m) >= vol(F_m^tight) for every member of the
flag; m = 1 is the head floor.  Proof (a maximum principle): x := l - l^tight satisfies a_i . x >= 0 for all i and sum x = 0; with the suffix
sums T_j := sum_{i >= j} x_i (T_1 = T_{d+1} = 0) the constraint at i reads T_{i+1} <= (1 - 1/beta_i) T_i + T_{i+beta_i}/beta_i, a convex
combination of an earlier and a later suffix sum; if M := max_j T_j were positive, the least J with T_J = M has J >= 2, T_{J-1} < M and
T_{J-1+beta_{J-1}} <= M, whence T_J < M -- a contradiction; so every suffix sum is <= 0 and every prefix sum of x is >= 0.  By LP duality each
prefix objective 1_{<= m} has a nonnegative certificate (`prefix_volume_certificate`, checked exactly); `linear_certificate` solves the same
triangular system for any right-hand side (multipliers of either sign), which gives rigorous enclosures of every tight entry
(`tight_entry`) and prefix volume (`tight_prefix_volume`).  No single entry other than the head is bounded: min l_k = -inf for k >= 2 and
max l_{d-beta+1} = +inf under the primal and under the two-sided constraints alike (both families only say 'descend at least at the GSA
rate'), and dropping the last full block's own constraint makes even min P_{d-beta} unbounded.  The prefix-volume floor is what the primal
attack's detection condition needs (`spec_chain.detection_chain`): the detection block L/F_{d-b} has log GH = L(b) + (S - P_{d-b})/b <=
l^tight_{d-b+1}.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np


def log_chat(n: int) -> float:
    """log of chat(n) = V_n^{-1/n}, V_n the volume of the unit n-ball (chat(1) = 1/2); double precision."""
    if n < 1:
        raise ValueError("n must be positive")
    log_Vn = (n / 2) * math.log(math.pi) - math.lgamma(n / 2 + 1)
    return -log_Vn / n


def block_sizes(d: int, beta: int) -> List[int]:
    """beta_i = min(beta, d - i + 1) for i = 1..d-1 (1-based), i.e. the sizes of the constrained blocks (all >= 2)."""
    if beta < 2 or d < beta:
        raise ValueError("need 2 <= beta <= d")
    return [min(beta, d - i + 1) for i in range(1, d)]


def dual_certificate(d: int, beta: int) -> Tuple[List[Fraction], Fraction]:
    """Exact rational solution (y_1..y_{d-1}, z) of  e_1 = sum_i y_i a_i + z 1, where a_i has entry (1 - 1/beta_i) at position i and
    -1/beta_i at positions i+1..i+beta_i-1.  Summing the d coordinate equations gives d z = 1; the first d-1 equations are then solved
    forward (each has the positive diagonal coefficient 1 - 1/beta_k) and the d-th holds automatically.  The incoming sum at position k is
    sum_{i = max(1, k-beta+1)}^{k-1} y_i/beta_i: every block starting in that window covers k (a full block covers the next beta - 1 positions,
    a tail block reaches d), so it is a sliding-window sum maintained in O(1) per step and the whole recurrence costs O(d) exact operations
    (`_dual_certificate_windowed` is the original O(d beta) form, kept for cross-checks).  Raises ValueError if some y_i is negative (then no
    certificate of this form exists)."""
    bs = block_sizes(d, beta)
    from flint import fmpq

    z = fmpq(1, d)
    y: List = []
    one = fmpq(1)
    inv_bs = [fmpq(1, b) for b in bs]
    incoming = fmpq(0)  # sum of y_i/beta_i over the window i in [max(1, k-beta+1), k-1]
    for k in range(1, d):
        if k >= 2:
            incoming += y[k - 2] * inv_bs[k - 2]  # position k-1 enters the window
        if k - beta >= 1:
            incoming -= y[k - beta - 1] * inv_bs[k - beta - 1]  # position k-beta leaves it
        rhs = one if k == 1 else fmpq(0)
        yk = (rhs - z + incoming) / (one - inv_bs[k - 1])
        if yk < 0:
            raise ValueError(f"dual multiplier y_{k} = {float(yk):.3e} is negative: no certificate of this form")
        y.append(yk)
    check = z - sum((y[i - 1] * inv_bs[i - 1] for i in range(max(1, d - beta + 1), d) if d <= i + bs[i - 1] - 1), fmpq(0))
    assert check == 0, "certificate inconsistent"
    yF = [Fraction(int(v.p), int(v.q)) for v in y]
    return yF, Fraction(1, d)


def _dual_certificate_windowed(d: int, beta: int) -> Tuple[List[Fraction], Fraction]:
    """The original O(d beta) form of `dual_certificate` (explicit window with the coverage test), for cross-checking the sliding sum."""
    bs = block_sizes(d, beta)
    from flint import fmpq

    z = fmpq(1, d)
    y: List = []
    one = fmpq(1)
    inv_bs = [fmpq(1, b) for b in bs]
    for k in range(1, d):
        incoming = fmpq(0)
        for i in range(max(1, k - beta + 1), k):
            if k <= i + bs[i - 1] - 1:
                incoming += y[i - 1] * inv_bs[i - 1]
        rhs = one if k == 1 else fmpq(0)
        yk = (rhs - z + incoming) / (one - inv_bs[k - 1])
        if yk < 0:
            raise ValueError(f"dual multiplier y_{k} = {float(yk):.3e} is negative: no certificate of this form")
        y.append(yk)
    return [Fraction(int(v.p), int(v.q)) for v in y], Fraction(1, d)


def verify_certificate(d: int, beta: int, y: List[Fraction], z: Fraction) -> bool:
    """Independent exact check of e_1 = sum_i y_i a_i + z 1 and y >= 0: the coordinate k of the right-hand side is z + y_k (1 - 1/beta_k) -
    sum_{i < k, block i covers k} y_i/beta_i (with y_d := 0 at k = d), and block i covers k iff i < k <= i + beta_i - 1, i.e. iff
    max(1, k - beta + 1) <= i <= k - 1 (a full block covers the next beta - 1 positions, a tail block reaches d), a sliding window whose sum is
    maintained in O(1) per coordinate; exact fmpq arithmetic.  `_verify_certificate_windowed` is the original O(d beta) Fraction form."""
    from flint import fmpq

    bs = block_sizes(d, beta)
    if len(y) != d - 1 or any(v < 0 for v in y):
        return False
    yq = [fmpq(v.numerator, v.denominator) for v in y]
    zq = fmpq(z.numerator, z.denominator)
    inv = [fmpq(1, b) for b in bs]
    covering = fmpq(0)  # sum of y_i/beta_i over i in [max(1, k-beta+1), k-1]
    for k in range(1, d + 1):
        if k >= 2:
            covering += yq[k - 2] * inv[k - 2]
        if k - beta >= 1:
            covering -= yq[k - beta - 1] * inv[k - beta - 1]
        own = yq[k - 1] * (1 - inv[k - 1]) if k <= d - 1 else fmpq(0)
        coeff = zq + own - covering
        if coeff != (1 if k == 1 else 0):
            return False
    return True


def _verify_certificate_windowed(d: int, beta: int, y: List[Fraction], z: Fraction) -> bool:
    """The original O(d beta) Fraction form of `verify_certificate` (explicit accumulation of every block's entries), for cross-checks."""
    bs = block_sizes(d, beta)
    if len(y) != d - 1 or any(v < 0 for v in y):
        return False
    coeff = [z] * d
    for i in range(1, d):
        bi = bs[i - 1]
        coeff[i - 1] += y[i - 1] * (1 - Fraction(1, bi))
        for k in range(i + 1, i + bi):
            coeff[k - 1] -= y[i - 1] / bi
    return coeff[0] == 1 and all(c == 0 for c in coeff[1:])


def _arb_log_chat(n: int, prec: int):
    """Rigorous enclosure of log chat(n) at `prec` bits."""
    from flint import arb, ctx, fmpq

    ctx.prec = prec
    if n == 1:
        return -arb(2).log()
    half_n = arb(fmpq(n, 2))
    log_Vn = half_n * arb.pi().log() - (half_n + 1).lgamma()
    return -log_Vn / n


def linear_certificate(d: int, beta: int, c) -> Tuple[List[Fraction], Fraction]:
    """Exact rational (w_1..w_{d-1}, z) with  c = sum_i w_i a_i + z 1  for a rational vector c of length d (entries int/Fraction): z = sum(c)/d,
    then the forward recurrence w_k = (c_k - z + incoming_k)/(1 - 1/beta_k) with the O(1) sliding-window incoming sum of `dual_certificate`;
    the d-th coordinate equation holds automatically and is asserted.  The multipliers may have either sign (for c = e_1 they are the
    nonnegative dual certificate).  Since the all-tight profile is the unique solution of the tight system, any linear functional of it is
    c . l^tight = sum_i w_i (log chat(beta_i) - eps) + z S."""
    from flint import fmpq

    bs = block_sizes(d, beta)
    if len(c) != d:
        raise ValueError("c must have length d")
    cq = [fmpq(Fraction(x).numerator, Fraction(x).denominator) for x in c]
    z = sum(cq, fmpq(0)) / d
    one = fmpq(1)
    inv_bs = [fmpq(1, b) for b in bs]
    w: List = []
    incoming = fmpq(0)
    for k in range(1, d):
        if k >= 2:
            incoming += w[k - 2] * inv_bs[k - 2]
        if k - beta >= 1:
            incoming -= w[k - beta - 1] * inv_bs[k - beta - 1]
        w.append((cq[k - 1] - z + incoming) / (one - inv_bs[k - 1]))
    covering_d = sum((w[i - 1] * inv_bs[i - 1] for i in range(max(1, d - beta + 1), d)), fmpq(0))
    assert z - covering_d == cq[d - 1], "linear certificate inconsistent"
    return [Fraction(int(v.p), int(v.q)) for v in w], Fraction(int(z.p), int(z.q))


def prefix_volume_certificate(d: int, beta: int, m: int) -> Tuple[List[Fraction], Fraction]:
    """The exact nonnegative certificate of the prefix-volume floor: (w, z) with 1_{<= m} = sum_i w_i a_i + z 1 and w >= 0 (the theorem
    guarantees existence by LP duality; nonnegativity is checked exactly and a ValueError raised otherwise).  m = 1 is `dual_certificate`."""
    if not 1 <= m <= d - 1:
        raise ValueError("need 1 <= m <= d - 1")
    w, z = linear_certificate(d, beta, [1 if i < m else 0 for i in range(d)])
    if any(v < 0 for v in w):
        raise ValueError(f"prefix-volume certificate for m={m} has a negative multiplier")
    return w, z


def _tight_functional(d: int, beta: int, w: List[Fraction], z: Fraction, eps, log_vol, prec: int):
    """arb enclosure of sum_i w_i (log chat(beta_i) - eps) + z log_vol at `prec` bits."""
    from flint import arb, ctx, fmpq

    bs = block_sizes(d, beta)
    ctx.prec = prec

    def to_arb(x):
        if isinstance(x, str):
            return arb(x)
        fx = Fraction(x)
        return arb(fmpq(fx.numerator, fx.denominator))

    eps_a, lv_a = to_arb(eps), to_arb(log_vol)
    cache = {}
    total = arb(0)
    for wi, b in zip(w, bs):
        if wi == 0:
            continue
        if b not in cache:
            cache[b] = _arb_log_chat(b, prec) - eps_a
            ctx.prec = prec
        total += arb(fmpq(wi.numerator, wi.denominator)) * cache[b]
    return total + arb(fmpq(z.numerator, z.denominator)) * lv_a


def tight_entry(d: int, beta: int, k: int, eps=0, log_vol=0, prec: int = 256):
    """Rigorous arb enclosure of the all-tight profile's entry l_k (1-based, 1 <= k <= d) at slack eps and the given log volume: the exact
    linear certificate of e_k evaluated with rigorous log chat enclosures.  l^tight_{d-b+1} is the log GH of the detection block of the primal
    attack on the tight profile, hence an upper bound on it over all admissible profiles (prefix-volume floor)."""
    if not 1 <= k <= d:
        raise ValueError("need 1 <= k <= d")
    w, z = linear_certificate(d, beta, [1 if i == k - 1 else 0 for i in range(d)])
    return _tight_functional(d, beta, w, z, eps, log_vol, prec)


def tight_prefix_volume(d: int, beta: int, m: int, eps=0, log_vol=0, prec: int = 256):
    """Rigorous arb enclosure of P_m of the all-tight profile, the certified lower bound on log vol F_m over (beta, eps)-admissible profiles of
    the given volume (nonnegative certificate checked exactly)."""
    w, z = prefix_volume_certificate(d, beta, m)
    return _tight_functional(d, beta, w, z, eps, log_vol, prec)


def tight_entry_float(d: int, beta: int, k: int, eps: float = 0.0, log_vol: float = 0.0) -> float:
    """l^tight_k in double precision by the O(d) recurrence (pre-screen; not a certificate)."""
    if not 1 <= k <= d:
        raise ValueError("need 1 <= k <= d")
    bs = block_sizes(d, beta)
    z = 1.0 / d
    w = np.zeros(d - 1)
    incoming = 0.0
    for j in range(1, d):
        if j >= 2:
            incoming += w[j - 2] / bs[j - 2]
        if j - beta >= 1:
            incoming -= w[j - beta - 1] / bs[j - beta - 1]
        w[j - 1] = ((1.0 if j == k else 0.0) - z + incoming) / (1 - 1 / bs[j - 1])
    return float(np.dot(w, np.array([log_chat(b) - eps for b in bs]))) + z * log_vol


def floor_l1(d: int, beta: int, eps=0, log_vol=0, prec: int = 256) -> Dict:
    """The lower bound on l_1 = log ||b_1|| for (beta, eps)-admissible profiles of a lattice with the given log volume, from the exact
    rational dual certificate, evaluated as a rigorous arb enclosure at `prec` bits.  `eps` and `log_vol` may be int, Fraction, or a
    decimal string (converted exactly); a float is accepted and converted exactly from its binary value.  Returns floats (midpoints) and
    the balls `l1_floor_ball`, `root_hermite_floor_ball`, together with the pure-GSA-line comparison and dual statistics.  Dual
    feasibility is exact; the enclosures are rigorous."""
    from flint import arb, ctx, fmpq

    y, z = dual_certificate(d, beta)
    assert verify_certificate(d, beta, y, z)
    bs = block_sizes(d, beta)
    ctx.prec = prec

    def to_arb(x):
        if isinstance(x, str):
            return arb(x)
        fx = Fraction(x)
        return arb(fmpq(fx.numerator, fx.denominator))

    eps_a, lv_a = to_arb(eps), to_arb(log_vol)
    cache = {}
    total = arb(0)
    for yi, b in zip(y, bs):
        if b not in cache:
            cache[b] = _arb_log_chat(b, prec) - eps_a
            ctx.prec = prec
        total += arb(fmpq(yi.numerator, yi.denominator)) * cache[b]
    bound = total + lv_a / d
    rhf = ((bound - lv_a / d) / (d - 1)).exp()
    first_ratio = (bound - lv_a / d).exp()
    yf = np.array([float(v) for v in y])
    eps_f = float(eps) if isinstance(eps, str) else float(Fraction(eps))
    # kappa is the structural tail correction at eps = 0: g_{d,beta} - h_{d,beta}(0); the eps terms cancel exactly because
    # h(eps) = h(0) - eps sum(y) and the pure-GSA head at the same eps shifts by -(d-1) eps/(beta-1) -- these differ, so report both
    h_zero = float(bound.mid()) - float(lv_a.mid()) / d + eps_f * float(yf.sum())
    gsa_zero = (d - 1) * log_chat(beta) / (beta - 1)
    gsa_head = gsa_zero - (d - 1) * eps_f / (beta - 1) + float(lv_a.mid()) / d  # pure GSA line at the same eps, through the volume
    return {"d": d, "beta": beta, "eps": eps, "prec": prec, "l1_floor": float(bound.mid()), "l1_floor_ball": bound,
            "first_ratio_floor": float(first_ratio.mid()), "root_hermite_floor": float(rhf.mid()), "root_hermite_floor_ball": rhf,
            "gsa_head": gsa_head, "kappa": gsa_zero - h_zero, "gsa_gap_same_eps": gsa_head - float(bound.mid()),
            "dual_sum": float(yf.sum()), "dual_min": float(yf.min()), "y1": float(yf[0])}


def gsa_delta_ball(beta: int, prec: int = 256):
    """Rigorous enclosure of the pure-GSA root-Hermite factor chat(beta)^{1/(beta-1)}."""
    from flint import ctx

    lc = _arb_log_chat(beta, prec)
    ctx.prec = prec
    return (lc / (beta - 1)).exp()


def decide_floor_vs_target(d: int, beta: int, target, eps=0, prec: int = 256, max_prec: int = 4096) -> Tuple[bool, Dict]:
    """Rigorous decision whether the certified root-Hermite floor at (d, beta, eps) is <= the target.  `target` is an arb ball, a
    Fraction/int, or a decimal string.  The floor ball and the target ball are compared with directed bounds; if they overlap the
    precision is doubled up to `max_prec`, after which a ValueError reports an undecidable threshold.  Returns (reaches, record)."""
    from flint import arb, ctx, fmpq

    p = prec
    while True:
        r = floor_l1(d, beta, eps, 0, p)
        ctx.prec = p
        if isinstance(target, str):
            t = arb(target)
        elif hasattr(target, "lower"):
            t = target
        else:
            ft = Fraction(target)
            t = arb(fmpq(ft.numerator, ft.denominator))
        fb = r["root_hermite_floor_ball"]
        if fb.upper() <= t.lower():
            return True, r
        if fb.lower() > t.upper():
            return False, r
        if p >= max_prec:
            raise ValueError(f"threshold undecidable at beta={beta} up to {max_prec} bits: floor {fb} vs target {t}")
        p = min(2 * p, max_prec)


def tight_profile(d: int, beta: int, eps: float = 0.0, log_vol: float = 0.0) -> np.ndarray:
    """The unique solution of the all-tight equations with the given sum.  It is an LP minimiser when the dual multipliers are
    nonnegative, and the unique LP minimiser when they are strictly positive.  It is the fixed point of the idealised GH-equality block
    recurrence; its body is only approximately a GSA line at finite d because of the shrinking tail."""
    bs = block_sizes(d, beta)
    A = np.zeros((d, d))
    b = np.zeros(d)
    for i in range(1, d):
        bi = bs[i - 1]
        A[i - 1, i - 1] = 1 - 1 / bi
        A[i - 1, i:i + bi - 1] = -1 / bi
        b[i - 1] = log_chat(bi) - eps
    A[d - 1, :] = 1.0
    b[d - 1] = log_vol
    return np.linalg.solve(A, b)


def dual_float(d: int, beta: int) -> np.ndarray:
    """The dual multipliers y_1..y_{d-1} in double precision (same recurrence as dual_certificate, sliding-window sum), for pre-screens only."""
    bs = block_sizes(d, beta)
    z = 1.0 / d
    y = np.zeros(d - 1)
    incoming = 0.0
    for k in range(1, d):
        if k >= 2:
            incoming += y[k - 2] / bs[k - 2]
        if k - beta >= 1:
            incoming -= y[k - beta - 1] / bs[k - beta - 1]
        rhs = 1.0 if k == 1 else 0.0
        y[k - 1] = (rhs - z + incoming) / (1 - 1 / bs[k - 1])
    return y


def floor_l1_float(d: int, beta: int, eps: float = 0.0, log_vol: float = 0.0) -> Dict:
    """floor_l1 in double precision (pre-screen; not a certificate)."""
    y = dual_float(d, beta)
    bs = block_sizes(d, beta)
    c = np.array([log_chat(b) - eps for b in bs])
    bound = float(np.dot(y, c)) + log_vol / d
    return {"d": d, "beta": beta, "eps": eps, "l1_floor": bound, "root_hermite_floor": math.exp((bound - log_vol / d) / (d - 1)),
            "dual_min": float(y.min()), "dual_sum": float(y.sum())}


def beta_floor_for_target(d: int, target, eps=0, beta_lo: int = 20, beta_hi: int | None = None, certify: bool = True,
                          exact_all: bool = False, band: float = 1e-5) -> Dict:
    """The least beta in [beta_lo, beta_hi] whose floor on the root-Hermite factor is <= the target -- the smallest blocksize whose
    (beta, eps)-admissible profiles can reach the target, so a basis reaching the target that is (beta, eps)-admissible must have beta at
    least this large.  The floor is not monotone in beta (chat(beta) < 1 for beta <= 12; the pure-GSA factor log chat(beta)/(beta-1)
    has its integer maximum at beta = 36, and the finite-d floor's maximum may depend slightly on d), so the scan never assumes
    monotonicity and the result is relative to the stated range.

    Modes.  exact_all=True: every beta from beta_lo up to the first passing one is decided rigorously (exact rational dual, arb
    enclosures of floor and target, directed comparison; `decide_floor_vs_target`) and the scan stops there; betas above it are not
    examined.  The result is a certificate of leastness within the range: every earlier beta rigorously fails and the returned one
    rigorously passes.  certify=True (default):
    a double-precision pre-screen locates the candidate; betas within `band` of the target are decided rigorously; the returned beta is
    confirmed rigorously to reach and its predecessor rigorously to fail.  The result is labelled as such: leastness over earlier
    betas rests on the double-precision classification, whose discrepancy against the rigorous midpoints at every doubly evaluated
    beta is reported and must stay below band/1000 (else ValueError).  certify=False: float scan only.  `target` may be an arb ball
    (e.g. gsa_delta_ball), a Fraction, or a decimal string; in float modes its midpoint is used."""
    beta_hi = d if beta_hi is None else beta_hi
    if not (2 <= beta_lo <= beta_hi <= d):
        raise ValueError(f"need 2 <= beta_lo <= beta_hi <= d (got {beta_lo}, {beta_hi}, {d})")
    if certify and not exact_all and not (band >= 1e-6):
        raise ValueError(f"band must be at least 1e-6 (got {band})")
    target_f = float(target.mid()) if hasattr(target, "mid") else (float(Fraction(target)) if not isinstance(target, str) else float(target))
    eps_f = float(Fraction(eps)) if not isinstance(eps, str) else float(eps)

    if exact_all:
        for beta in range(beta_lo, beta_hi + 1):
            reaches, r = decide_floor_vs_target(d, beta, target, eps)
            if reaches:
                return {"d": d, "target": target_f, "eps": eps_f, "beta_floor": beta, "root_hermite_floor_at_beta": r["root_hermite_floor"],
                        "dual_exact": True, "dual_min": r["dual_min"], "beta_lo": beta_lo, "beta_hi": beta_hi,
                        "minimality": f"certified: every beta in [{beta_lo}, {beta}] decided rigorously (exact dual, arb enclosures); "
                                      f"least in [{beta_lo}, {beta_hi}]"}
        return {"d": d, "target": target_f, "eps": eps_f, "beta_floor": None,
                "note": f"no beta in [{beta_lo}, {beta_hi}] reaches the target under the axiom (every beta decided rigorously)"}

    exact_checked = 0
    max_disc = 0.0
    for beta in range(beta_lo, beta_hi + 1):
        rf = floor_l1_float(d, beta, eps_f)
        rhf = rf["root_hermite_floor"]
        reaches = rhf <= target_f
        if certify and abs(rhf - target_f) <= band:
            reaches, r = decide_floor_vs_target(d, beta, target, eps)
            exact_checked += 1
            max_disc = max(max_disc, abs(r["root_hermite_floor"] - rhf))
            rhf = r["root_hermite_floor"]
        if not reaches:
            continue
        out = {"d": d, "target": target_f, "eps": eps_f, "beta_floor": beta, "root_hermite_floor_at_beta": rhf,
               "dual_exact": False, "dual_min": rf["dual_min"], "exact_checks_in_band": exact_checked}
        if not certify:
            out["minimality"] = "float scan"
            return out
        reaches_r, r = decide_floor_vs_target(d, beta, target, eps)
        max_disc = max(max_disc, abs(r["root_hermite_floor"] - rf["root_hermite_floor"]))
        if not reaches_r:  # the pre-screen was optimistic beyond the band: keep scanning
            continue
        out.update({"dual_exact": True, "root_hermite_floor_at_beta": r["root_hermite_floor"], "dual_min": r["dual_min"]})
        if beta > beta_lo:
            prev_reaches, prev = decide_floor_vs_target(d, beta - 1, target, eps)
            pf = floor_l1_float(d, beta - 1, eps_f)
            max_disc = max(max_disc, abs(prev["root_hermite_floor"] - pf["root_hermite_floor"]))
            out["root_hermite_floor_at_beta_minus_1"] = prev["root_hermite_floor"]
            out["predecessor_fails_rigorous"] = not prev_reaches
            out["minimality"] = "double-precision scan; beta and beta-1 decided rigorously (exact dual, arb enclosures)"
        else:
            out["minimality"] = "lower boundary of range; beta decided rigorously"
        out["max_float_rigorous_discrepancy"] = max_disc
        if max_disc > band / 1e3:
            raise ValueError(f"float/rigorous discrepancy {max_disc:.2e} exceeds band/1e3; pre-screen not trustworthy")
        return out
    if certify and max_disc > band / 1e3:
        raise ValueError(f"float/rigorous discrepancy {max_disc:.2e} exceeds band/1e3; pre-screen not trustworthy")
    return {"d": d, "target": target_f, "eps": eps_f, "beta_floor": None, "exact_checks_in_band": exact_checked,
            "max_float_rigorous_discrepancy": max_disc if certify else None,
            "note": f"no beta in [{beta_lo}, {beta_hi}] reaches the target under the axiom according to the double-precision scan"
                    + (" (in-band betas decided rigorously)" if certify else "")}
