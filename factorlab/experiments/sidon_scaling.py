"""E26: hypothesis (S) at scale, without a modulus.

For r <= N^{1/5} section 7.6 of notes_barrier.md reduces hypothesis (S) to a statement about
the set of speeds {sqrt k - sqrt k' : k, k' squarefree in (r/2, r]} at the resolution
    rho_r = W/u = 1/(4 sqrt 2 r^{3/2})
(Harvey's real window sqrt N/(4 r sqrt(r/2)) of the cell (1, r/2), divided by u = 2 sqrt N; the
integer window of Lemma D is the ceiling of this width), up to the bounded offset perturbation.
Define
    D*(r) := max over real tau of #{ordered (k, k') : |sqrt k - sqrt k' - tau| < rho_r}.
This needs no N, so r can be taken far beyond what 64-bit moduli allow.  The sweep is exact
for the float64 array of speeds (the numerical margins at r <= 2^15 are many orders above the
floating error, but no separation margin is certified), and it returns one maximising window.

Exact window membership.  For the pair (k, k - delta), sqrt k - sqrt(k - delta) = tau iff
sqrt k + sqrt(k - delta) = delta/tau, hence sqrt k = (delta/tau + tau)/2 and
    k = (delta + tau^2)^2 / (4 tau^2)       (on the branch delta tau > 0, |delta| >= tau^2).
The pair lies in the window (tau - rho, tau + rho) iff k lies strictly between
k(tau + rho, delta) and k(tau - rho, delta), an interval of length about rho delta^2/tau^3;
D(tau) is the number of delta whose interval contains a squarefree k with k - delta squarefree,
both in the shell.  Exact coincidences of many delta at one tau for every shell are the Beatty
chains, which need a square-multiple k and are excluded by squarefreeness.

Controls.  The full shell (all k) shows the sqrt r law of the chains.  The phase-randomised
null keeps every delta-class (the points of one class form a near-arithmetic progression with
spacing >= |delta|/(4 r^{3/2}) > 2 rho for |delta| >= 2, so a class contributes at most one
point per window) but shifts each class by an independent uniform random multiple of that lower
bound on its spacing: inter-class alignment is destroyed while the within-class structure is
kept.  That null is exactly the genericity behind (S); if the real D*(r) tracks it, the chirp
has no Diophantine alignment beyond the chains.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .lehman_cover import squarefree_flags


def rho(r: int) -> float:
    """Harvey's window of the cell (1, r/2) in speed units: W/u = 1/(4 sqrt2 r^{3/2})."""
    return 1.0 / (4.0 * math.sqrt(2.0) * r ** 1.5)


SQUAREFREE_LINK_DENSITY = 0.37  # measured fraction of consecutive chain member pairs with both entries squarefree


def links_per_class_mid_shell(r: int) -> float:
    """Links of one residue class of an e = 1 near-chain inside a 2 rho window placed at the middle of
    the shell (M^2 = 6 j r): rho M^3/(j tau_j) = rho r^{3/2} 6^{3/2} sqrt 2, independent of j."""
    return rho(r) * r ** 1.5 * 6.0 ** 1.5 * math.sqrt(2.0)


def links_per_class_top_shell(r: int) -> float:
    """The same capacity at the top of the shell (M^2 = 8 j r), where the speed drift is slowest and the
    maximising windows are found: rho r^{3/2} 8^{3/2} sqrt 2."""
    return rho(r) * r ** 1.5 * 8.0 ** 1.5 * math.sqrt(2.0)


def predicted_chain_cluster(r: int, j: int) -> float:
    """Saturated cluster of the e = 1 chain at tau_j: squarefree-link density times the top-of-shell
    capacity times the number of residue classes."""
    return SQUAREFREE_LINK_DENSITY * links_per_class_top_shell(r) * 2 ** omega(j)


def shell(r: int, squarefree: bool) -> np.ndarray:
    ks = np.arange(r // 2 + 1, r + 1, dtype=np.int64)
    if squarefree:
        sf = squarefree_flags(r)
        ks = ks[sf[ks]]
    return ks


def speeds_and_deltas(ks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All ordered pairs (k, k'), k != k': speeds sqrt k - sqrt k' and offsets k - k'."""
    s = np.sqrt(ks.astype(np.float64))
    L = (s[:, None] - s[None, :])
    D = (ks[:, None] - ks[None, :])
    mask = ~np.eye(ks.size, dtype=bool)
    return L[mask], D[mask]


def cluster_max(points: np.ndarray, half_width: float) -> tuple[int, float]:
    """max over real tau of #{p : |p - tau| < half_width}, and a maximising tau.

    The maximum is attained with the window's left end just below a point: for sorted points
    p_i count those in [p_i, p_i + 2 half_width).  The returned centre is the midpoint of the
    first and last counted points, whose span is strictly less than 2 half_width, so every
    counted point satisfies the strict inequality |p - tau| < half_width.
    """
    if points.size == 0:
        return 0, 0.0
    pts = np.sort(points)
    hi = np.searchsorted(pts, pts + 2 * half_width, side="left")
    counts = hi - np.arange(pts.size)
    i = int(np.argmax(counts))
    last = int(hi[i]) - 1
    return int(counts[i]), float(0.5 * (pts[i] + pts[last]))


def d_star(r: int, squarefree: bool = True) -> dict:
    ks = shell(r, squarefree)
    L, D = speeds_and_deltas(ks)
    Dm, tau = cluster_max(L, rho(r))
    # the configuration realising the maximum
    sel = np.abs(L - tau) < rho(r)
    return {"r": r, "R": int(ks.size), "pairs": int(L.size), "D_star": Dm, "tau": tau,
            "tau_over_sqrt_r": tau / math.sqrt(r),
            "deltas_at_max": sorted(int(x) for x in D[sel]),
            "k_at_max": sorted(int(k) for k, keep in zip(np.repeat(ks, ks.size - 1), sel) if keep)}


def phase_randomised_null(r: int, rng: np.random.Generator, squarefree: bool = True) -> int:
    """D* of the null in which every delta-class is shifted by an independent uniform random
    multiple (in [0, 1)) of its own minimal spacing |delta|/(4 r^{3/2})."""
    ks = shell(r, squarefree)
    L, D = speeds_and_deltas(ks)
    deltas = np.unique(D)
    shift = rng.uniform(0.0, 1.0, size=deltas.size) * (np.abs(deltas) / (4.0 * r ** 1.5))
    idx = np.searchsorted(deltas, D)
    return cluster_max(L + shift[idx], rho(r))[0]


def window_membership_exact(k: int, delta: int, tau: float, r: int) -> bool:
    """Exact check of |sqrt k - sqrt(k - delta) - tau| < rho(r) through the interval formula."""
    lo_tau, hi_tau = tau - rho(r), tau + rho(r)
    def kk(t):
        return (delta + t * t) ** 2 / (4.0 * t * t)
    # k(t, delta) is decreasing in t on the branch 0 < t, t^2 < delta; elsewhere use the definition
    if not (lo_tau > 0 and hi_tau * hi_tau < delta):
        return abs(math.sqrt(k) - math.sqrt(k - delta) - tau) < rho(r)
    return kk(hi_tau) < k < kk(lo_tau)


def near_square_multiple_mask(ks: np.ndarray, c_max: int, e_max: int) -> np.ndarray:
    """True for k such that c k + e is a perfect square for some 1 <= c <= c_max, 1 <= |e| <= e_max.

    Such k are the members of near-chains: along M = M_0 + s n with c | 2 s^2 and
    c | s (2 M_0 + s), the residue of M^2 modulo c is constant, so k_n = (M_n^2 - e)/c is an
    integer sequence with sqrt(k_{n+1}) - sqrt(k_n) = s/sqrt c + e s/(2 sqrt c M_n M_{n+1}) + O(e^2/M^4):
    a speed drifting only as 1/M^2, which lets a run of the chain sit inside one rho-window
    (e = 0 is the exact Beatty chain, excluded by squarefreeness; e != 0 survives it).
    """
    mask = np.zeros(ks.size, dtype=bool)
    k64 = ks.astype(np.int64)
    for c in range(1, c_max + 1):
        base = c * k64
        for e in range(-e_max, e_max + 1):
            if e == 0:
                continue
            v = base + e
            ok = v >= 0
            root = np.zeros_like(v)
            root[ok] = np.floor(np.sqrt(v[ok].astype(np.float64))).astype(np.int64)
            # correct floating error
            root = np.where((root + 1) ** 2 <= v, root + 1, root)
            root = np.where(root ** 2 > v, root - 1, root)
            mask |= ok & (root * root == v)
    return mask


def d_star_refined(r: int, c_max: int, e_max: int) -> dict:
    """D*(r) of the squarefree shell with near-square-multiple k (c <= c_max, |e| <= e_max) removed."""
    ks = shell(r, True)
    drop = near_square_multiple_mask(ks, c_max, e_max)
    kept = ks[~drop]
    L, D = speeds_and_deltas(kept)
    Dm, tau = cluster_max(L, rho(r))
    sel = np.abs(L - tau) < rho(r)
    return {"r": r, "c_max": c_max, "e_max": e_max, "R_squarefree": int(ks.size), "R_kept": int(kept.size),
            "removed_fraction": float(drop.mean()), "D_star": Dm, "tau": tau, "tau_sq": tau * tau,
            "deltas_at_max": sorted(int(x) for x in D[sel])}


def verify_chain(r: int, c: int, e: int, s: int, M0: int) -> dict:
    """Exact check of a near-chain k_n = (M_n^2 - e)/c along M_n = M0 + s n.

    Requires the congruences that make every term integral: c | M0^2 - e, c | 2 s^2 and
    c | s (2 M0 + s) (then (M0 + s n)^2 = M0^2 + s n (2 M0 + s n) and s n (2 M0 + s n) =
    n s (2 M0 + s) + s^2 n (n - 1) is divisible by c for every n).  Raises otherwise.  For
    each consecutive pair of members inside the shell the baseline speed is
    (M_{n+1} - M_n)/sqrt c and the first-order prediction of the deviation is
    e (M_{n+1} - M_n)/(2 sqrt c M_n M_{n+1}).
    """
    if not (r > 0 and c > 0 and s > 0 and M0 > 0):
        raise ValueError("r, c, s, M0 must be positive")
    if (M0 * M0 - e) % c or (2 * s * s) % c or (s * (2 * M0 + s)) % c:
        raise ValueError("not a near-chain: integrality congruences fail")
    ks, Ms = [], []
    M = M0
    while (M * M - e) <= c * r:
        k = (M * M - e) // c
        if k > r // 2:
            ks.append(k)
            Ms.append(M)
        M += s
    speeds = [pair_speed(a, b) for a, b in zip(ks, ks[1:])]
    gaps = [b - a for a, b in zip(Ms, Ms[1:])]
    tau0 = s / math.sqrt(c)
    return {"members": ks, "M": Ms, "tau0": tau0, "tau0_sq": tau0 * tau0,
            "deviation_over_rho": [(v - g / math.sqrt(c)) / rho(r) for v, g in zip(speeds, gaps)],
            "predicted_over_rho": [e * g / (2 * math.sqrt(c) * Ms[i] * Ms[i + 1]) / rho(r) for i, g in enumerate(gaps)]}


def omega(n: int) -> int:
    cnt, p = 0, 2
    while p * p <= n:
        if n % p == 0:
            cnt += 1
            while n % p == 0:
                n //= p
        p += 1
    return cnt + (1 if n > 1 else 0)


def e1_chain_pairs(r: int, j: int) -> list[tuple[int, int]]:
    """Consecutive links (k, k') of the e = 1 near-chains at tau^2 = j/2 inside the shell: k = (M^2 - 1)/(8j)
    with M odd, M^2 = 1 (mod 8j), k' the member at M + 2j; both k, k' squarefree in (r/2, r]."""
    sf = squarefree_flags(r)
    lo = 8 * j * (r // 2) + 1
    pairs = []
    M = int(math.isqrt(lo))
    while M * M <= 8 * j * r + 1:
        if M % 2 == 1 and (M * M - 1) % (8 * j) == 0:
            k = (M * M - 1) // (8 * j)
            Mp = M + 2 * j
            kp = (Mp * Mp - 1) // (8 * j)
            if r // 2 < k and kp <= r and sf[k] and sf[kp]:
                pairs.append((k, kp))
        M += 1
    return pairs


def j_census(r: int, j_max: int | None = None) -> list[dict]:
    """For every odd squarefree j <= j_max: the number of squarefree e = 1 chain links in the shell,
    the largest number of them inside one rho-window (their speeds lie just above tau_j = sqrt(j/2)),
    and the prediction 0.37 * (links per class at the top of the shell) * 2^omega(j)."""
    if j_max is None:
        j_max = max(3, r // 40)
    sf = squarefree_flags(max(j_max, 3))
    out = []
    for j in range(1, j_max + 1, 2):
        if not sf[j]:
            continue
        pairs = e1_chain_pairs(r, j)
        if len(pairs) < 2:
            continue
        sp = np.array([pair_speed(a, b) for a, b in pairs])
        Dm, _ = cluster_max(sp, rho(r))
        out.append({"j": j, "omega": omega(j), "links": len(pairs), "cluster": Dm, "predicted": predicted_chain_cluster(r, j)})
    return out


def chain_of_maximiser(r: int, tau: float) -> dict | None:
    """If tau^2 is within a few windows of j/2 for an odd squarefree j (the cluster centre sits
    a few rho above tau_j = sqrt(j/2)), return the e = 1 chain parameters."""
    j2 = 2 * tau * tau
    j = int(round(j2))
    if j <= 0 or j % 2 == 0 or abs(j2 - j) > 40 * tau * rho(r) or not squarefree_flags(j)[j]:
        return None
    return {"j": j, "omega": omega(j), "c": 8 * j, "s": 2 * j, "predicted": predicted_chain_cluster(r, j)}


def _crt_roots_of_unity(j: int) -> list[int]:
    """All M mod j with M^2 = 1 (mod j), j odd: the roots modulo each prime power p^a || j are +-1,
    combined by CRT (2^omega(j) roots)."""
    roots = [0]
    mod = 1
    n, p = j, 3
    while n > 1:
        if n % p == 0:
            pa = 1
            while n % p == 0:
                n //= p
                pa *= p
            new = []
            for r0 in roots:
                for s in (1, pa - 1):
                    inv = pow(mod, -1, pa)
                    x = r0 + mod * (((s - r0) * inv) % pa)
                    new.append(x)
            roots, mod = new, mod * pa
        p += 2
        if p * p > n and n > 1:
            p = n
    return sorted(roots)


def e1_chain_classes(j: int) -> list[int]:
    """The odd M_0 mod 2j with M_0^2 = 1 (mod 8j): the odd lift of each root of unity modulo j.
    Requires j odd (raises otherwise); for non-squarefree j the roots modulo p^a are still +-1."""
    if j <= 0 or j % 2 == 0:
        raise ValueError("j must be odd")
    return sorted(x if x % 2 == 1 else x + j for x in _crt_roots_of_unity(j))


def pair_speed(k: int, kp: int) -> float:
    """sqrt(kp) - sqrt(k) computed as (kp - k)/(sqrt kp + sqrt k): relative error ~ 1e-16, against an
    absolute error ~ ulp(sqrt k) ~ 2e-12 for the naive difference at k ~ 2^28, which exceeds rho_r."""
    return (kp - k) / (math.sqrt(kp) + math.sqrt(k))


def e1_chain_links_fast(r: int, j: int, sf: np.ndarray) -> tuple[int, list[float]]:
    """Members of all classes of the e = 1 chain at j inside the shell (r/2, r], and the speeds of the
    consecutive links whose two members are both squarefree; stepping by 2j per class."""
    members = 0
    speeds = []
    lo, hi = 8 * j * (r // 2) + 1, 8 * j * r + 1
    for M0 in e1_chain_classes(j):
        M = M0 + 2 * j * (((math.isqrt(lo) - M0) // (2 * j)) if math.isqrt(lo) > M0 else 0)
        while M * M < lo:
            M += 2 * j
        prev = None
        while M * M <= hi:
            k = (M * M - 1) // (8 * j)
            members += 1
            if prev is not None and sf[prev] and sf[k]:
                speeds.append(pair_speed(prev, k))
            prev = k
            M += 2 * j
    return members, speeds


def chain_census_fast(r: int, j_max: int | None = None) -> dict:
    """Exact clusters of the e = 1 chains for every odd squarefree j <= j_max without the pair sweep:
    a lower bound for D*(r) that scales to r ~ 2^24.  Returns the best chain and the whole census."""
    if j_max is None:
        j_max = max(3, r // 40)
    sf = squarefree_flags(r)
    sfj = squarefree_flags(max(j_max, 3))
    rows = []
    for j in range(1, j_max + 1, 2):
        if not sfj[j]:
            continue
        members, sp = e1_chain_links_fast(r, j, sf)
        if len(sp) < 2:
            continue
        Dm, _ = cluster_max(np.array(sp), rho(r))
        rows.append({"j": j, "omega": omega(j), "members": members, "links": len(sp), "cluster": Dm,
                     "predicted": predicted_chain_cluster(r, j)})
    best = max(rows, key=lambda z: z["cluster"]) if rows else None
    om = max(z["omega"] for z in rows) if rows else 0
    return {"r": r, "j_max": j_max, "best": best, "omega_max": om, "D_chain": best["cluster"] if best else 0,
            "eta_chain": math.log(best["cluster"]) / math.log(r) if best else 0.0, "rows": rows}


def resonance_partition(r: int, margin: float | None = None) -> dict:
    """The cluster maximum of the squarefree shell split by the arithmetic of tau^2 at the window centre:
    within `margin` of a half-integer j/2 (j odd; the near-chain speeds), within `margin` of an integer
    (the Beatty speeds, excluded exactly by squarefreeness), and off both.  The chain deviations move
    tau^2 by at most 1/(8r), so margin = 1/(2r) captures every chain window."""
    if margin is None:
        margin = 0.5 / r
    ks = shell(r, True)
    L, _ = speeds_and_deltas(ks)
    pts = np.sort(L)
    h = rho(r)
    hi = np.searchsorted(pts, pts + 2 * h, side="left")
    counts = hi - np.arange(pts.size)
    last = np.maximum(hi - 1, np.arange(pts.size))
    centres = 0.5 * (pts + pts[last])
    t2 = centres * centres
    frac2 = np.abs(2 * t2 - np.round(2 * t2))          # distance of 2 tau^2 from the nearest integer
    near_half_or_int = frac2 < 2 * margin
    odd_numerator = (np.round(2 * t2).astype(np.int64) % 2 == 1)
    near_half = near_half_or_int & odd_numerator
    near_int = near_half_or_int & ~odd_numerator
    off = ~near_half_or_int
    def best(mask):
        if not mask.any():
            return 0, None
        i = int(np.argmax(np.where(mask, counts, -1)))
        return int(counts[i]), float(t2[i])
    Dh, th = best(near_half)
    Di, ti = best(near_int)
    Do, to = best(off)
    return {"r": r, "margin": margin, "D_half_integer": Dh, "tau_sq_half": th, "D_integer": Di, "tau_sq_int": ti,
            "D_off_resonance": Do, "tau_sq_off": to, "D_star": int(counts.max())}


_DENSITY_PRIMES: np.ndarray | None = None
_DENSITY_BASE: float | None = None
_DENSITY_P = 100_000


def _odd_primes_to(P: int) -> np.ndarray:
    sieve = np.ones(P + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(P ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.nonzero(sieve)[0][1:]


def _bad_residues_mod(j: int, m: int) -> int:
    """Cardinality of {1, -1, 1 - 2j, -1 - 2j} modulo m."""
    return len({1 % m, (-1) % m, (1 - 2 * j) % m, (-1 - 2 * j) % m})


def link_density_model(j: int) -> float:
    """Local (Euler-product) prediction of the fraction of consecutive e = 1 chain member pairs
    (k_M, k_{M+2j}) with both entries squarefree, j odd squarefree.

    For an odd prime p not dividing j: p^2 | k_M iff M = +-1 (mod p^2), so the two members exclude the
    residues {+-1, -2j +- 1} modulo p^2 -- four of them generically, three when j = +-1 (mod p^2)
    (for j = 1 every odd p has three), factor 1 - |U_p|/p^2.  For p | j: M = +-1 (mod p) already and
    p^2 | k_M iff M = +-1 (mod p^3); within the class one bad residue per member, factor 1 - 2/p^2.
    For p = 2: 4 | k_M iff M = +-1 (mod 16); among the eight odd residues modulo 16 the members
    exclude |{+-1, -2j +- 1} mod 16| of them.  Independence across primes is the standard
    squarefree-sieve heuristic.  The generic product over odd p <= 10^5 is cached; per-j corrections
    are applied at p | j and at the finitely many p with p^2 | j -+ 1; the tail beyond 10^5 is
    bounded by exp(-4/10^5).
    """
    global _DENSITY_PRIMES, _DENSITY_BASE
    if _DENSITY_PRIMES is None:
        _DENSITY_PRIMES = _odd_primes_to(_DENSITY_P)
        _DENSITY_BASE = float(np.prod(1.0 - 4.0 / _DENSITY_PRIMES.astype(np.float64) ** 2)) * math.exp(-4.0 / _DENSITY_P)
    prod = _DENSITY_BASE * (1 - _bad_residues_mod(j, 16) / 8)
    # primes dividing j: replace the generic factor by 1 - 2/p^2
    n, p = j, 3
    while n > 1:
        if n % p == 0:
            while n % p == 0:
                n //= p
            prod *= (1 - 2 / p ** 2) / (1 - 4 / p ** 2)
        p += 2
        if p * p > n and n > 1:
            p = n
    # primes p not dividing j with p^2 | j - 1 or p^2 | j + 1: only three bad residues.  For j = 1 every
    # odd prime qualifies (j - 1 = 0), so the generic four-residue product is replaced wholesale.
    if j == 1:
        ratio = float(np.prod((1.0 - 3.0 / _DENSITY_PRIMES.astype(np.float64) ** 2) / (1.0 - 4.0 / _DENSITY_PRIMES.astype(np.float64) ** 2)))
        return prod * ratio * math.exp(1.0 / _DENSITY_P)
    for q in (j - 1, j + 1):
        m, p = q, 3
        while p * p <= m:
            if m % (p * p) == 0 and j % p:
                prod *= (1 - 3 / p ** 2) / (1 - 4 / p ** 2)
            while m % p == 0:
                m //= p
            p += 2
    return prod


def density_check(r: int, j_max: int | None = None) -> dict:
    """Measured squarefree-link fraction of every chain at r against the local model, pooled."""
    cen = chain_census_fast(r, j_max)
    rows = []
    obs_links = obs_pairs = exp_links = 0.0
    for z in cen["rows"]:
        pairs = z["members"] - 2 ** z["omega"]  # consecutive pairs: members minus one per class
        if pairs <= 0:
            continue
        model = link_density_model(z["j"])
        rows.append({"j": z["j"], "omega": z["omega"], "pairs": pairs, "links": z["links"],
                     "observed": z["links"] / pairs, "model": model})
        obs_links += z["links"]
        obs_pairs += pairs
        exp_links += model * pairs
    return {"r": r, "rows": rows, "pooled_observed": obs_links / obs_pairs if obs_pairs else None,
            "pooled_model": exp_links / obs_pairs if obs_pairs else None, "total_pairs": int(obs_pairs),
            "min_model": min(z["model"] for z in rows) if rows else None}


def symmetric_pair_cluster(r: int, j: int, sf: np.ndarray) -> dict:
    """The g = 2 cross-class pairs of the e = 1 chain at odd j: (k_-, k_+) = ((j t^2 - t)/2, (j t^2 + t)/2),
    t >= 1 (M = 2jt -+ 1), both in the shell and squarefree.  Their speed is
        1/sqrt(2j) * (1 + 1/(8 j^2 t^2) + O(t^-4)),
    so a window of width 2 rho holds the t in a range of length 8 sqrt2 rho j^{5/2} t^3, i.e. 5.66 j
    consecutive t at the top of the shell, against 0.414 sqrt(r/j) available t: the two balance at
    j ~ 0.17 r^{1/3}, where the cluster is ~ density * 0.99 r^{1/3}."""
    t_lo = math.isqrt(r // j) + 1          # (j t^2 - t)/2 > r/2  <=  t > sqrt(r/j) roughly; checked exactly below
    t_hi = math.isqrt(2 * r // j) + 1
    speeds, members = [], 0
    for t in range(max(1, t_lo - 2), t_hi + 2):
        km, kp = (j * t * t - t) // 2, (j * t * t + t) // 2
        if (j * t * t - t) % 2 or km <= r // 2 or kp > r:
            continue
        members += 1
        if sf[km] and sf[kp]:
            speeds.append(pair_speed(km, kp))
    Dm = cluster_max(np.array(speeds), rho(r))[0] if len(speeds) >= 1 else 0
    cap = 8 * math.sqrt(2) * rho(r) * j ** 2.5 * (2 * r / j) ** 1.5
    return {"j": j, "members": members, "links": len(speeds), "cluster": Dm, "tau_sq": 1 / (2 * j),
            "capacity_top": cap, "available": 0.414 * math.sqrt(r / j)}


def symmetric_pair_census(r: int, j_max: int | None = None, sf: np.ndarray | None = None) -> dict:
    """Max over odd j <= j_max of the symmetric-pair cluster; j_max defaults to 3 r^{1/3}."""
    if j_max is None:
        j_max = max(3, int(3 * r ** (1 / 3)))
    if sf is None:
        sf = squarefree_flags(r)
    rows = [symmetric_pair_cluster(r, j, sf) for j in range(1, j_max + 1, 2)]
    rows = [z for z in rows if z["links"] >= 1]
    if not rows:
        return {"r": r, "j_max": j_max, "best": None, "D_symmetric": 0, "j_star_predicted": 0.175 * r ** (1 / 3),
                "r_third": r ** (1 / 3), "rows": []}
    best = max(rows, key=lambda z: z["cluster"])
    return {"r": r, "j_max": j_max, "best": best, "D_symmetric": best["cluster"], "j_star_predicted": 0.175 * r ** (1 / 3),
            "r_third": r ** (1 / 3), "rows": rows}


def identify_window_as_class_pairs(r: int, tau: float, j: int, g: int) -> dict:
    """Every ordered squarefree pair (k, k') of the shell with |sqrt k' - sqrt k - tau| < rho(r) is tested
    exactly for membership in the (j, g) class-pair family: 8 j k + 1 = M^2 and 8 j k' + 1 = (M + g)^2.
    Returns the window's pair count and the number of pairs that are (j, g)-links."""
    ks = shell(r, True)
    s = np.sqrt(ks.astype(np.float64))
    blk = s[None, :] - s[:, None]          # blk[i, i'] = sqrt k_{i'} - sqrt k_i
    ii, jj = np.nonzero(np.abs(blk - tau) < rho(r))
    total = links = 0
    for a, b in zip(ii.tolist(), jj.tolist()):
        if a == b:
            continue
        total += 1
        k, kp = int(ks[a]), int(ks[b])
        M = math.isqrt(8 * j * k + 1)
        Mp = math.isqrt(8 * j * kp + 1)
        if M * M == 8 * j * k + 1 and Mp * Mp == 8 * j * kp + 1 and Mp - M == g:
            links += 1
    return {"r": r, "j": j, "g": g, "pairs_in_window": total, "class_pair_links": links, "all_identified": total == links}


def g_link_cluster(r: int, j: int, g: int, sf: np.ndarray) -> dict:
    """Pairs (M, M + g) of members of the e = 1 chain at j (any two classes whose difference is g mod 2j),
    both squarefree in the shell; speed g/sqrt(8j) (1 + 1/(2 M (M+g))); capacity 11.3 j/g per class pair.
    Requires r > 0, g > 0 and j odd (e1_chain_classes raises otherwise)."""
    if r <= 0 or g <= 0:
        raise ValueError("r and g must be positive")
    classes = e1_chain_classes(j)
    cset = set(classes)
    pairs_of_classes = [M0 for M0 in classes if (M0 + g) % (2 * j) in cset]
    lo, hi = 8 * j * (r // 2) + 1, 8 * j * r + 1
    speeds = []
    for M0 in pairs_of_classes:
        M = M0 + 2 * j * max(0, (math.isqrt(lo) - M0) // (2 * j))
        while M * M < lo:
            M += 2 * j
        while (M + g) ** 2 <= hi:
            k, kp = (M * M - 1) // (8 * j), ((M + g) ** 2 - 1) // (8 * j)
            if k > r // 2 and sf[k] and sf[kp]:
                speeds.append(pair_speed(k, kp))
            M += 2 * j
    Dm = cluster_max(np.array(speeds), rho(r))[0] if speeds else 0
    return {"j": j, "g": g, "class_pairs": len(pairs_of_classes), "links": len(speeds), "cluster": Dm,
            "tau_sq": g * g / (8 * j), "capacity_top_per_class_pair": 11.3 * j / g}


def d_star_lean(r: int, squarefree: bool = True, block: int = 2048) -> dict:
    """D*(r) by the same sweep as d_star, materialising the R^2 speeds in row blocks and no offsets or
    masks.  Peak memory is about 8 R^2 bytes for the speeds plus the full-length search arrays of the
    sweep (the shifted copy, the searchsorted indices and the counts, 8 R^2 bytes each), i.e. roughly
    32 R^2 bytes: ~ 13 GB at r = 2^16 (R ~ 2 * 10^4).  The maximising window's ordered pairs are
    reconstructed in a second blocked pass once tau is known."""
    ks = shell(r, squarefree)
    s = np.sqrt(ks.astype(np.float64))
    R = ks.size
    L = np.empty(R * (R - 1), dtype=np.float64)
    pos = 0
    for i0 in range(0, R, block):
        i1 = min(R, i0 + block)
        blk = s[i0:i1, None] - s[None, :]
        rows = np.arange(i0, i1)[:, None]
        keep = np.arange(R)[None, :] != rows
        vals = blk[keep]
        L[pos:pos + vals.size] = vals
        pos += vals.size
    del blk, keep, vals
    L.sort()
    h = rho(r)
    hi = np.searchsorted(L, L + 2 * h, side="left")
    counts = hi - np.arange(L.size)
    i = int(np.argmax(counts))
    last = int(hi[i]) - 1
    tau = float(0.5 * (L[i] + L[last]))
    D = int(counts[i])
    del L, hi, counts
    pairs = []
    for i0 in range(0, R, block):
        i1 = min(R, i0 + block)
        blk = s[i0:i1, None] - s[None, :]
        ii, jj = np.nonzero(np.abs(blk - tau) < h)
        for a, b in zip(ii.tolist(), jj.tolist()):
            if i0 + a != b:
                pairs.append((int(ks[i0 + a]), int(ks[b])))
    assert len(pairs) == D, (len(pairs), D)
    return {"r": r, "R": int(R), "pairs": int(R * (R - 1)), "D_star": D, "tau": tau, "tau_sq": tau * tau,
            "eta_pointwise": math.log(D) / math.log(r), "pairs_at_max": sorted(pairs)}


def _ceil_2sqrt(k: int, N: int) -> int:
    v = 4 * k * N
    s = math.isqrt(v)
    return s if s * s == v else s + 1


def lemma_d_window(N: int, r: int) -> int:
    """Harvey's integer window of the widest a = 1 cell of the shell, k = r//2 + 1: the least W with
    16 r^2 k W^2 >= N."""
    k = r // 2 + 1
    W = math.isqrt(N // (16 * r * r * k))
    while 16 * r * r * k * W * W < N:
        W += 1
    return max(W, 1)


def symmetric_cluster_with_modulus(N: int, r: int, j: int, sf: np.ndarray, W: int | None = None) -> dict:
    """Lemma D's statistic restricted to the symmetric family at j for the cells (1, k): the exact start
    differences d(t) = ceil(2 sqrt(k_+ N)) - ceil(2 sqrt(k_- N)) - t of the squarefree pairs in the shell,
    and the largest number of them inside an integer window of 2W - 1 consecutive integers.  Since
    u (sqrt k_+ - sqrt k_-) - t decreases by more than one per step of t and d(t) is within one of it,
    d is non-increasing and d(t) - d(t + m) > m - 2, so at most 2W consecutive t share a window -- the
    planar cap; for r << N^{1/5} the offset t is negligible against W and the cluster is the
    modulus-free one."""
    if W is None:
        W = lemma_d_window(N, r)
    ds = []
    members = 0
    t_hi = math.isqrt(2 * r // j) + 2
    for t in range(1, t_hi + 1):
        km, kp = (j * t * t - t) // 2, (j * t * t + t) // 2
        if (j * t * t - t) % 2 or km <= r // 2 or kp > r:
            continue
        members += 1
        if sf[km] and sf[kp]:
            ds.append(_ceil_2sqrt(kp, N) - _ceil_2sqrt(km, N) - t)
    monotone = all(b <= a for a, b in zip(ds, ds[1:]))   # in increasing t, before sorting
    ds.sort()
    best = 0
    for i, d in enumerate(ds):
        hi = i
        while hi + 1 < len(ds) and ds[hi + 1] <= d + 2 * W - 2:
            hi += 1
        best = max(best, hi - i + 1)
    return {"N_bits": N.bit_length(), "r": r, "j": j, "W": W, "members": members, "links": len(ds), "cluster": best,
            "planar_cap": 2 * W, "monotone": monotone}


def planar_regime_check(N: int, exponents: Sequence[float] = (0.2, 0.25, 3 / 11, 0.3, 1 / 3), j_cap: float = 3.0) -> list[dict]:
    """For each r = N^e: the best symmetric-family cluster with the modulus (max over odd j <= j_cap r^{1/3}),
    the modulus-free D_sym(r) and the planar cap 2W."""
    out = []
    for e in exponents:
        r = int(round(N ** e))
        sf = squarefree_flags(r)
        W = lemma_d_window(N, r)
        rows = [symmetric_cluster_with_modulus(N, r, j, sf, W) for j in range(1, int(j_cap * r ** (1 / 3)) + 1, 2)]
        rows = [z for z in rows if z["links"] >= 1]
        best = max(rows, key=lambda z: z["cluster"]) if rows else None
        free = symmetric_pair_census(r, sf=sf)
        out.append({"exponent": e, "r": r, "W": W, "planar_cap": 2 * W, "with_modulus": best, "modulus_free_D_sym": free["D_symmetric"],
                    "modulus_free_j": free["best"]["j"] if free["best"] else None, "r_third_law": 0.38 * r ** (1 / 3)})
    return out


def symmetric_pigeonhole_bound(N: int, r: int, x: float, kappa: float = 1.0) -> dict:
    """Theorem X (notes_barrier 7.13): for j = x r^{1/3} the L = |I_j(r)| symmetric pairs of the shell have exact
    start differences d(t) = u s(t) - t + theta(t), |theta| < 1, whose speed term spans at most
    u (1 + 2/(jr))/(16 sqrt2 j^{3/2} r) = W_real (1 + 2/(jr)) r^{1/2}/(4 j^{3/2}) (Lemma S3) and whose offset spans
    L - 1, so all lie among at most V + 1 consecutive integers, V = W_real (1 + 2/(jr)) r^{1/2}/(4 j^{3/2}) + L + 1;
    a window of 2W - 1 consecutive integers therefore holds >= L (2W - 1)/(V + 2W - 1) of them (double counting over
    the window positions meeting the interval).  With kappa = 1 this is the proven finite bound for the full shell
    (L is Lemma S3's lower bound (sqrt2 - 1) sqrt(r/j) - 3 for |I_j(r)|, clamped at 0).  With kappa = kappa_j it is
    the ASYMPTOTIC form of the squarefree bound, kappa_j L (1 - o(1)) (2W - 1)/(V + 2W - 1): Lemma S2's finite error
    terms (3L/z + 4^{pi(z)} + pi(sqrt T2) + 2 pi(sqrt(j T2 + 1))) are not subtracted here, so the kappa-scaled value
    is a first-order quantity at feasible sizes, not a certified count.  Requires W_real >= 1 (the planar regime)."""
    j = max(1, int(round(x * r ** (1 / 3))))
    if j % 2 == 0:
        j += 1
    W = lemma_d_window(N, r)
    Wr = math.sqrt(N) / (2 * math.sqrt(2) * r ** 1.5)
    if Wr < 1:
        raise ValueError("Theorem X's bound is stated for the planar regime W_real >= 1")
    L = max(0.0, (math.sqrt(2) - 1) * math.sqrt(r / j) - 3)          # Lemma S3's lower bound for |I_j(r)|
    V = Wr * (1 + 2 / (j * r)) * math.sqrt(r) / (4 * j ** 1.5) + L + 1
    return {"j": j, "W": W, "W_real": Wr, "L": L, "V": V, "bound": kappa * L * (2 * W - 1) / (V + 2 * W - 1),
            "kappa_is_asymptotic": kappa != 1.0}


def symmetric_density_model(j: int) -> float:
    """kappa_j: the local density of t with both (j t^2 -+ t)/2 squarefree.  For odd p not dividing j the
    bad residues modulo p^2 are t = 0, j^{-1}, -j^{-1} (p^2 | t, p^2 | jt - 1, p^2 | jt + 1; the two factors
    t and jt -+ 1 are coprime), factor 1 - 3/p^2; for p | j only t = 0, factor 1 - 1/p^2; modulo 8 the bad
    residues are t = 0, j, -j (three of eight), factor 5/8.  Independence across primes is exact in the
    Legendre sieve (Lemma S2), so kappa_j is the main-term constant, not a heuristic; it is evaluated here as
    the finite product over p <= 10^5 times exp(-3/10^5), an approximation to the infinite product accurate
    to about 1e-5."""
    global _DENSITY_PRIMES
    if _DENSITY_PRIMES is None:
        _DENSITY_PRIMES = _odd_primes_to(_DENSITY_P)
    P = _DENSITY_PRIMES.astype(np.float64)
    k = 5 / 8 * float(np.prod(1 - 3 / P ** 2)) * math.exp(-3.0 / _DENSITY_P)
    n, p = j, 3
    while n > 1:
        if n % p == 0:
            while n % p == 0:
                n //= p
            k *= (1 - 1 / p ** 2) / (1 - 3 / p ** 2)
        p += 2
        if p * p > n and n > 1:
            p = n
    return k


KAPPA_0 = 5 / 8 * 0.501948  # (5/8) prod_{odd p} (1 - 3/p^2) = 0.313717: the uniform lower bound of kappa_j over odd j


def _pi(x: float) -> int:
    x = int(x)
    if x < 2:
        return 0
    s = np.ones(x + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(x ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return int(s.sum())


def sieve_lower_bound(L: int, T2: int, j: int, z: int) -> float:
    """Lemma S2: for I a set of L consecutive integers with max I <= T2 and j odd,
        #{t in I : (j t^2 - t)/2 and (j t^2 + t)/2 squarefree} >= kappa_j L - 3L/z - 4^pi(z) - pi(sqrt T2) - 2 pi(sqrt(j T2 + 1)).
    The explicit constants are poor (the large-prime term is the trivial count pi(sqrt(j T2))), so the bound is
    informative only for very large T2; the theorem needs only that the error is o(L)."""
    return symmetric_density_model(j) * L - 3 * L / z - 4 ** _pi(z) - _pi(math.isqrt(T2)) - 2 * _pi(math.isqrt(j * T2 + 1))


def symmetric_family_window_fit(r: int, j: int, sf: np.ndarray | None = None) -> dict:
    """Lemma S3 check: the speeds of ALL symmetric pairs at j in the shell (squarefree or not) span an interval;
    reports its length in units of 2 rho_r (<= theta means every pair fits in one window of half-width theta rho_r)
    together with the first-order prediction r^{1/2}/(8 j^{3/2}) and the squarefree count against kappa_j L."""
    speeds = []
    links = 0
    t_hi = math.isqrt(2 * r // j) + 2
    for t in range(1, t_hi + 1):
        km, kp = (j * t * t - t) // 2, (j * t * t + t) // 2
        if (j * t * t - t) % 2 or km <= r // 2 or kp > r:
            continue
        speeds.append(pair_speed(km, kp))
        if sf is not None and sf[km] and sf[kp]:
            links += 1
    L = len(speeds)
    spread = (max(speeds) - min(speeds)) / (2 * rho(r)) if L else 0.0
    return {"r": r, "j": j, "members": L, "spread_over_window": spread, "predicted_spread": math.sqrt(r) / (8 * j ** 1.5),
            "squarefree_links": links, "kappa_L": symmetric_density_model(j) * L, "j_over_r13": j / r ** (1 / 3)}


def theorem_q_prime_check(r: int, theta: float = 1.0, delta: float | None = None, sf: np.ndarray | None = None) -> dict:
    """The theorem's construction at a given r: j = smallest odd integer >= (1 + delta) (8 theta)^{-2/3} r^{1/3} with
    delta = 2/r by default (Lemma S3: the speed spread of all pairs is at most r^{1/2}(1 + 2/(jr))/(8 theta j^{3/2})
    windows, exactly, and (1 + 2/r)^{3/2} >= 1 + 2/(jr)); check that all pairs fit in a window of half-width
    theta rho_r, count the squarefree pairs, and compare with kappa_j L and with the proven constant
    kappa_0 (sqrt2 - 1) (8 theta)^{1/3} r^{1/3}."""
    if sf is None:
        sf = squarefree_flags(r)
    if delta is None:
        delta = 2.0 / r
    j = int(math.ceil((1 + delta) * (8 * theta) ** (-2 / 3) * r ** (1 / 3)))
    if j % 2 == 0:
        j += 1
    z = symmetric_family_window_fit(r, j, sf)
    z.update({"theta": theta, "delta": delta, "fits": z["spread_over_window"] <= theta,
              # Lemma S3's exact bound on (speed span)/(2 theta rho_r), i.e. in units of the theta-window; compare with
              # spread_over_window / theta and with 1
              "exact_fit_bound": math.sqrt(r) * (1 + 2 / (j * r)) / (8 * theta * j ** 1.5),
              "proven_constant_times_r13": KAPPA_0 * (math.sqrt(2) - 1) * (8 * theta) ** (1 / 3) * r ** (1 / 3),
              "links_over_r13": z["squarefree_links"] / r ** (1 / 3)})
    return z


def sidon_scaling_experiment(rs: Sequence[int] = (256, 512, 1024, 2048, 4096, 8192, 16384), null_samples: int = 5, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for r in rs:
        row = d_star(r, squarefree=True)
        row["null_D_star"] = [phase_randomised_null(r, rng, True) for _ in range(null_samples)]
        row["maximiser_chain"] = chain_of_maximiser(r, abs(row["tau"]))
        if r <= 8192:
            full = d_star(r, squarefree=False)
            row["full_shell_D_star"] = full["D_star"]
            row["full_shell_squares_minus_one"] = int(math.isqrt(r) - math.isqrt(r // 2)) - 1
        row["j_census"] = j_census(r)
        rows.append(row)
    x = np.array([math.log(row["r"]) for row in rows])
    y = np.array([row["D_star"] for row in rows], dtype=float)
    (a, b), cov = np.polyfit(x, y, 1, cov=True)
    yl = np.log(y)
    (pw, _), covp = np.polyfit(x, yl, 1, cov=True)
    for row in rows:
        row["eta_pointwise"] = math.log(row["D_star"]) / math.log(row["r"])
    return {"rho_definition": "W/u = 1/(4 sqrt2 r^{3/2})", "links_per_class_mid_shell": links_per_class_mid_shell(rs[0]),
            "links_per_class_top_shell": links_per_class_top_shell(rs[0]),
            "squarefree_link_density": SQUAREFREE_LINK_DENSITY, "rows": rows,
            "fit_linear_in_log_r": {"slope": float(a), "intercept": float(b), "slope_se": float(math.sqrt(cov[0, 0]))},
            "fit_power_law": {"exponent": float(pw), "exponent_se": float(math.sqrt(covp[0, 0]))}}


if __name__ == "__main__":  # python -m factorlab.experiments.sidon_scaling [--quick]
    import json
    import os
    import sys

    from ..bench import RESULTS_DIR

    quick = "--quick" in sys.argv
    res = sidon_scaling_experiment(rs=(256, 512, 1024, 2048) if quick else (256, 512, 1024, 2048, 4096, 8192, 16384, 32768))
    print("== E26: D*(r) of the squarefree shell at resolution W/u = 1/(4 sqrt2 r^{3/2}) (no modulus) ==")
    print(f"  links per class in a window: {res['links_per_class_mid_shell']:.3f} (mid-shell), {res['links_per_class_top_shell']:.3f} (top of shell); "
          f"prediction = {res['squarefree_link_density']} * top-of-shell capacity * 2^omega(j)")
    for row in res["rows"]:
        fs = f"full shell {row['full_shell_D_star']} (squares-1 = {row['full_shell_squares_minus_one']})" if "full_shell_D_star" in row else "full shell --"
        mc = row["maximiser_chain"]
        mcs = f"chain j={mc['j']} (omega {mc['omega']}, c=8j={mc['c']}) predicted {mc['predicted']:.1f}" if mc else "maximiser not an e=1 chain"
        print(f"  r={row['r']:6d}: R={row['R']:5d} pairs={row['pairs']:9d} | D*={row['D_star']:3d} (eta={row['eta_pointwise']:.3f}) at tau^2 = {row['tau']**2:.4f} | {mcs} | "
              f"phase-randomised null {row['null_D_star']} | {fs}")
        best = sorted(row["j_census"], key=lambda z: -z["cluster"])[:4]
        print("           j-census (top): " + "; ".join(f"j={z['j']} omega={z['omega']} links={z['links']} cluster={z['cluster']} pred={z['predicted']:.1f}" for z in best))
    f1, f2 = res["fit_linear_in_log_r"], res["fit_power_law"]
    print(f"  fit D* = a ln r + b: a = {f1['slope']:.3f} +- {f1['slope_se']:.3f}, b = {f1['intercept']:.2f} | power law exponent {f2['exponent']:.3f} +- {f2['exponent_se']:.3f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "e26_sidon_scaling.json"), "w") as fh:
        json.dump(res, fh, indent=1)
