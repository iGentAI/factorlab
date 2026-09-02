"""E9: per-prime ECM success, heterogeneity, and deterministic hitting sets.

Stage 1 of ECM on a Suyama curve E_sigma succeeds modulo a prime p iff the
order of the starting point divides L = lcm{l^e <= B1}.  Running the Montgomery
ladder modulo many primes at once (numpy int64, p < 2^31) gives, for a fixed
list of curves sigma = 6, 7, 8, ..., the success matrix S[sigma, p].  From it:

* the per-prime success probability theta_p and its spread across primes
  (beyond binomial noise), and its dependence on residue classes of p (the
  Galois-theoretic structure of Suyama orders: 12 | #E always, and the higher
  2- and 3-adic valuations depend on p modulo small powers);
* the covering behaviour of the FIXED enumeration: after how many curves is
  every prime in a range hit at least once, compared with a randomly
  permuted order (a random hitting set) and with the coupon-collector
  prediction from the theta_p;
* on the primes p, q of RSA moduli, the correlation between success modulo p
  and modulo q as a function of N modulo small integers (the CRT coupling of
  E8 applied to curve orders).
"""

from __future__ import annotations

import math

import numpy as np

from ..gen import make_semiprime
from ..numth import small_primes
from ..algorithms.ecm import stage1_exponents


def primes_in_range(lo: int, hi: int) -> np.ndarray:
    """All primes in [lo, hi) by a segmented sieve (hi < 2^31)."""
    lo, hi = int(lo), int(hi)
    sieve = np.ones(hi - lo, dtype=bool)
    for l in small_primes(int(math.isqrt(hi)) + 1):
        start = max(l * l, ((lo + l - 1) // l) * l)
        sieve[start - lo::l] = False
    if lo <= 1:
        sieve[: max(0, 2 - lo)] = False
    return (np.nonzero(sieve)[0] + lo).astype(np.int64)


def vec_powmod(base: np.ndarray, exp: np.ndarray, mod: np.ndarray) -> np.ndarray:
    """base^exp mod mod, elementwise, exponents < 2^63."""
    result = np.ones_like(mod)
    b = base % mod
    e = exp.astype(np.int64).copy()
    while np.any(e > 0):
        odd = (e & 1) == 1
        result = np.where(odd, (result * b) % mod, result)
        b = (b * b) % mod
        e >>= 1
    return result


def vec_inv(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    return vec_powmod(a, p - 2, p)


def vec_suyama(sigma: int, p: np.ndarray):
    """a24 = (A+2)/4 and the starting point (X0 : Z0) of Suyama's curve modulo each p."""
    s = np.full_like(p, sigma) % p
    u = (s * s - 5) % p
    v = (4 * s) % p
    u2 = (u * u) % p
    u3 = (u2 * u) % p
    v2 = (v * v) % p
    X0 = u3
    Z0 = (v2 * v) % p
    d = (v - u) % p
    d3 = (((d * d) % p) * d) % p
    num = (d3 * ((3 * u + v) % p)) % p
    den = (((16 * u3) % p) * v) % p
    a24 = (num * vec_inv(den, p)) % p
    return a24, X0, Z0


def vec_xdbl(X, Z, a24, p):
    t1 = (X + Z) % p
    t1 = (t1 * t1) % p
    t2 = (X - Z) % p
    t2 = (t2 * t2) % p
    t3 = (t1 - t2) % p
    X2 = (t1 * t2) % p
    Z2 = (t3 * ((t2 + (a24 * t3) % p) % p)) % p
    return X2, Z2


def vec_xadd(XP, ZP, XQ, ZQ, Xd, Zd, p):
    u = (((XP - ZP) % p) * ((XQ + ZQ) % p)) % p
    v = (((XP + ZP) % p) * ((XQ - ZQ) % p)) % p
    s = (u + v) % p
    d = (u - v) % p
    X = (Zd * ((s * s) % p)) % p
    Z = (Xd * ((d * d) % p)) % p
    return X, Z


def vec_ladder(k: int, X, Z, a24, p):
    if k == 1:
        return X, Z
    R0 = (X, Z)
    R1 = vec_xdbl(X, Z, a24, p)
    for bit in bin(k)[3:]:
        if bit == "1":
            R0 = vec_xadd(R1[0], R1[1], R0[0], R0[1], X, Z, p)
            R1 = vec_xdbl(R1[0], R1[1], a24, p)
        else:
            R1 = vec_xadd(R0[0], R0[1], R1[0], R1[1], X, Z, p)
            R0 = vec_xdbl(R0[0], R0[1], a24, p)
    return R0


def stage1_success(sigma: int, primes: np.ndarray, B1: int) -> np.ndarray:
    """Boolean array: stage 1 of E_sigma with bound B1 succeeds modulo each prime."""
    p = primes.astype(np.int64)
    a24, X, Z = vec_suyama(sigma, p)
    for pe in stage1_exponents(B1):
        X, Z = vec_ladder(pe, X, Z, a24, p)
    return (Z % p) == 0


def success_matrix(primes: np.ndarray, sigmas, B1: int) -> np.ndarray:
    return np.array([stage1_success(int(s), primes, B1) for s in sigmas], dtype=bool)


def hitting_experiment(pbits: int = 18, nsig: int = 200, B1: int = 200, seed: int = 3) -> dict:
    from .smooth_profiles import dickman_rho
    primes = primes_in_range(1 << (pbits - 1), 1 << pbits)
    sigmas = list(range(6, 6 + nsig))
    S = success_matrix(primes, sigmas, B1)
    theta = S.mean(axis=0)
    tbar = float(theta.mean())
    binom_var = tbar * (1 - tbar) / nsig
    rng = np.random.default_rng(seed)
    # first hit in the fixed order
    first_fixed = np.where(S.any(axis=0), S.argmax(axis=0) + 1, nsig + 1)
    # hitting-set control: ONE random order of the curves applied to every prime, repeated
    cover_random = []
    for _ in range(20):
        perm = rng.permutation(nsig)
        Sp_ = S[perm]
        ff = np.where(Sp_.any(axis=0), Sp_.argmax(axis=0) + 1, nsig + 1)
        cover_random.append(int(ff.max()) if (ff <= nsig).all() else nsig + 1)
    covered_fixed = [float((first_fixed <= t).mean()) for t in range(1, nsig + 1)]
    # exact prediction for a common random order without replacement: for a prime with
    # s_p successes among nsig curves, P(no hit in first t) = C(nsig - s_p, t) / C(nsig, t)
    s_counts = S.sum(axis=0)

    def no_hit_prob(s, t):
        if t > nsig - s:
            return 0.0
        return float(np.exp(sum(math.log(nsig - s - i) - math.log(nsig - i) for i in range(t))))

    pred_uncovered = [float(np.mean([no_hit_prob(int(s), t) for s in s_counts])) for t in range(1, 61)]
    pred_uncovered_indep = [float(np.mean((1 - theta) ** t)) for t in range(1, 61)]
    by_class = {}
    for m in (3, 4, 5, 7, 8, 12):
        by_class[f"mod_{m}"] = {str(r): float(theta[primes % m == r].mean()) for r in range(m) if (primes % m == r).any()}
    order = np.argsort(theta)
    p_over_12 = primes / 12.0
    rho_pred = float(np.mean([dickman_rho(math.log(x) / math.log(B1)) for x in p_over_12[:: max(1, len(primes) // 500)]]))
    return {
        "pbits": pbits, "n_primes": int(len(primes)), "n_curves": nsig, "B1": B1,
        "theta_mean": tbar, "theta_std": float(theta.std()), "binomial_std": math.sqrt(binom_var),
        "theta_quantiles": [float(x) for x in np.quantile(theta, [0.01, 0.1, 0.5, 0.9, 0.99])],
        "rho_prediction_p_over_12": rho_pred,
        "by_class": by_class,
        "covering_curves_fixed_order": int(first_fixed.max()) if (first_fixed <= nsig).all() else None,
        "covering_curves_common_random_orders": cover_random,
        "covering_curves_common_random_mean": float(np.mean(cover_random)),
        "uncovered_after_t_fixed": [1 - x for x in covered_fixed[:60]],
        "uncovered_after_t_pred_exact": pred_uncovered,
        "uncovered_after_t_pred_indep": pred_uncovered_indep,
        "hardest_primes": [(int(primes[i]), float(theta[i])) for i in order[:10]],
        "fraction_never_hit": float((first_fixed > nsig).mean()),
    }


def coupling_experiment(nbits: int = 48, count: int = 6000, nsig: int = 60, c: float = 1 / 6, seed: int = 5,
                        family: str = "rsa") -> dict:
    """Correlation of ECM success modulo p and modulo q for RSA moduli, as a
    function of N modulo small integers."""
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    P = np.array([int(i.p) for i in insts], dtype=np.int64)
    Q = np.array([int(i.q) for i in insts], dtype=np.int64)
    Nmod = {m: np.array([int(i.N) % m for i in insts]) for m in (3, 4, 5, 7, 8, 24)}
    B1 = max(5, int(round(2.0 ** (c * nbits))))
    sigmas = list(range(6, 6 + nsig))
    Sp = success_matrix(P, sigmas, B1)
    Sq = success_matrix(Q, sigmas, B1)
    out = {"nbits": nbits, "count": count, "n_curves": nsig, "B1": B1,
           "theta_p": float(Sp.mean()), "theta_q": float(Sq.mean()),
           "overall_corr": float(np.corrcoef(Sp.ravel().astype(float), Sq.ravel().astype(float))[0, 1]),
           "by_class": {}}
    for m, arr in Nmod.items():
        res = {}
        for r in range(m):
            mask = arr == r
            if mask.sum() < 50:
                continue
            a = Sp[:, mask].ravel().astype(float)
            b = Sq[:, mask].ravel().astype(float)
            union = float((Sp[:, mask] | Sq[:, mask]).mean())
            res[str(r)] = {"count": int(mask.sum()), "corr": float(np.corrcoef(a, b)[0, 1]),
                           "union_rate": union, "single_rate": float((a.mean() + b.mean()) / 2)}
        out["by_class"][f"mod_{m}"] = res
    # success within `nsig` curves for the hard-set analysis
    out["per_modulus_success_any_curve"] = (Sp | Sq).any(axis=0).tolist()
    out["_Sp"] = Sp
    out["_Sq"] = Sq
    out["_Nmod"] = Nmod
    return out


def adaptive_selection(Sp: np.ndarray, Sq: np.ndarray, Nres: np.ndarray, k_select: int = 10, seed: int = 9) -> dict:
    """E14: choose curves per residue class of N.  Half the moduli (training)
    rank the curves by their union success rate within each class of N mod m;
    the top k_select per class are evaluated on the other half (test) against
    (i) the first k_select curves of the fixed enumeration and (ii) the
    globally best k_select curves chosen on the training half."""
    U = (Sp | Sq)
    nsig, count = U.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(count)
    train, test = perm[: count // 2], perm[count // 2:]
    k = min(k_select, nsig)
    # baselines
    first_k = float(U[:k][:, test].any(axis=0).mean())
    glob_rank = np.argsort(-U[:, train].mean(axis=1))[:k]
    best_k_global = float(U[glob_rank][:, test].any(axis=0).mean())
    per_curve_global = float(U[glob_rank][:, test].mean())
    # adaptive
    hits = 0
    per_curve_num = 0.0
    per_curve_den = 0
    classes = {}
    for r in np.unique(Nres):
        tr = train[Nres[train] == r]
        te = test[Nres[test] == r]
        if len(tr) < 20 or len(te) == 0:
            sel = glob_rank
        else:
            sel = np.argsort(-U[:, tr].mean(axis=1))[:k]
        hits += int(U[sel][:, te].any(axis=0).sum())
        per_curve_num += float(U[sel][:, te].sum())
        per_curve_den += int(k * len(te))
        classes[str(int(r))] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                                "selected": [int(s) + 6 for s in sel[:5]],
                                "test_union_rate": float(U[sel][:, te].any(axis=0).mean()) if len(te) else None}
    adaptive = hits / len(test)
    return {"k_select": k, "n_test": int(len(test)),
            "first_k_curves": first_k, "best_k_global": best_k_global, "adaptive_per_class": adaptive,
            "per_curve_union_rate_global": per_curve_global, "per_curve_union_rate_adaptive": per_curve_num / per_curve_den,
            "classes": classes}
