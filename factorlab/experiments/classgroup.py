"""E10: class numbers h(-kN) and the Schnorr-Lenstra class-group method.

For D = -kN (kN = 3 mod 4) or D = -4kN (otherwise), k squarefree and coprime
to N, the class group Cl(D) of binary quadratic forms has an element of order
2 for every factorisation of |D| (genus theory: 2-rank = omega(D) - 1).  The
Schnorr-Lenstra method takes a random form f, computes f^{L} with
L = lcm{l^e <= B1} restricted to odd l (plus a BSGS stage 2 to B2), and reads
a factor off an ambiguous form in the 2-Sylow subgroup; it exposes a
factorisation iff the odd part of ord(f) -- generically the odd part of h(D)
-- is (B1, B2)-semismooth.  Class numbers are computed with PARI: qfbclassno
(unconditional for |D| < 2e10) or quadclassunit (GRH) for larger |D|.

Checks: the genus-theory 2-rank; Cohen-Lenstra divisibility of the odd part by
l = 3, 5, 7, 11, 13 (Pr[l | h] = 1 - prod_j (1 - l^-j) against 1/l for random
integers); the semismoothness of h_odd against a Dickman prediction with the
Cohen-Lenstra shift; the exposure profile at cost N^c; and the multiplier
hitting behaviour (how many k until some h_odd(-kN) is semismooth).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
from sympy import factorint

from ..gen import make_semiprime
from ..numth import small_primes

_pari = None


def pari():
    global _pari
    if _pari is None:
        import cypari2
        _pari = cypari2.Pari()
        _pari.allocatemem(256 * 1024 * 1024)
    return _pari


def discriminant(k: int, N: int) -> int:
    m = k * N
    return -m if m % 4 == 3 else -4 * m


def class_number(D: int) -> tuple[int, str]:
    """(h(D), method): qfbclassno below 2e10 (unconditional), else quadclassunit (GRH)."""
    P = pari()
    if abs(D) < 2 * 10 ** 10:
        return int(P.qfbclassno(D)), "qfbclassno"
    return int(P.quadclassunit(D)[0]), "quadclassunit"


def omega(D: int) -> int:
    return len(factorint(abs(D)))


def cohen_lenstra_div_prob(l: int, terms: int = 60) -> float:
    prod = 1.0
    for j in range(1, terms + 1):
        prod *= 1.0 - l ** (-j)
    return 1.0 - prod


def cohen_lenstra_expected_valuation(l: int, jmax: int = 40) -> float:
    """E[v_l(|G|)] under the Cohen-Lenstra measure: Pr[|G_l| = l^j] = eta_inf l^-j / eta_j."""
    eta_inf = 1.0
    for i in range(1, 200):
        eta_inf *= 1.0 - l ** (-i)
    s = 0.0
    eta_j = 1.0
    for j in range(1, jmax + 1):
        eta_j *= 1.0 - l ** (-j)
        s += j * eta_inf * l ** (-j) / eta_j
    return s


def squarefree_multipliers(N: int, kmax_count: int) -> list[int]:
    out = []
    k = 1
    while len(out) < kmax_count:
        f = factorint(k)
        if all(e == 1 for e in f.values()) and math.gcd(k, N) == 1:
            out.append(k)
        k += 1
    return out


def _top_two(fac: dict) -> tuple[int, int]:
    ps = sorted(((int(p), int(e)) for p, e in fac.items()), reverse=True)
    if not ps:
        return 1, 1
    l1 = ps[0][0]
    if ps[0][1] >= 2:
        return l1, l1
    return l1, (ps[1][0] if len(ps) > 1 else 1)


def classgroup_experiment(nbits: int = 40, count: int = 1000, kcount: int = 30,
                          exponents: Sequence[float] = (1 / 8, 1 / 6, 1 / 5), seed: int = 41,
                          family: str = "rsa") -> dict:
    from .smooth_profiles import dickman_rho, semismooth_G
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    records = []  # (i, k, D, h, v2, t, h_odd, l1, l2, method)
    for i, inst in enumerate(insts):
        N = int(inst.N)
        for k in squarefree_multipliers(N, kcount):
            D = discriminant(k, N)
            h, method = class_number(D)
            fac = {int(a): int(b) for a, b in factorint(h).items()}
            v2 = fac.get(2, 0)
            h_odd = h >> v2
            fac_odd = {a: b for a, b in fac.items() if a != 2}
            l1, l2 = _top_two(fac_odd)
            t = omega(D)
            records.append((i, k, D, h, v2, t, h_odd, l1, l2, method, fac_odd))
    v2 = np.array([r[4] for r in records])
    t = np.array([r[5] for r in records])
    h = np.array([r[3] for r in records], dtype=float)
    h_odd = np.array([r[6] for r in records], dtype=float)
    D_abs = np.array([abs(r[2]) for r in records], dtype=float)
    out = {"nbits": nbits, "count": count, "kcount": kcount, "records": len(records),
           "methods": dict(Counter(r[9] for r in records)),
           "genus_2rank_ok": bool(np.all(v2 >= t - 1)),
           "mean_v2_minus_rank": float(np.mean(v2 - (t - 1))),
           "mean_h_over_sqrtD": float(np.mean(h / np.sqrt(D_abs))),
           "odd_divisibility": {}, "semismooth": [], "hitting": {}}
    for l in (3, 5, 7, 11, 13):
        div = np.array([l in r[10] for r in records], dtype=float)
        val = np.array([r[10].get(l, 0) for r in records], dtype=float)
        out["odd_divisibility"][str(l)] = {
            "observed": float(div.mean()), "se": float(div.std() / math.sqrt(len(div))),
            "cohen_lenstra": cohen_lenstra_div_prob(l), "random_integer": 1.0 / l,
            "mean_valuation": float(val.mean()), "cl_expected_valuation": cohen_lenstra_expected_valuation(l),
            "random_expected_valuation": 1.0 / (l - 1)}
    # Cohen-Lenstra shift in effective log-size of the odd part (sum over odd primes < 10^5;
    # the tail beyond is < 1e-4 nats)
    shift = sum(math.log(l) * (cohen_lenstra_expected_valuation(l) - 1.0 / (l - 1)) for l in small_primes(100000) if l > 2)
    out["cohen_lenstra_log_shift"] = shift
    out["cohen_lenstra_log_shift_truncation"] = "odd primes below 1e5"
    mean_log_hodd = float(np.mean(np.log(h_odd)))
    out["mean_log2_h_odd"] = mean_log_hodd / math.log(2)
    out["mean_log2_p"] = nbits / 2
    per_inst = {}
    rng = np.random.default_rng(seed)
    # size-matched control: a random odd integer with the bit length of each h_odd
    ctrl_top = []
    for r in records:
        bl = max(int(r[6]).bit_length(), 2)
        x = int(rng.integers(1 << (bl - 1), 1 << bl)) | 1
        ctrl_top.append(_top_two({int(a): int(b) for a, b in factorint(x).items()}))
    for c in exponents:
        B1 = 2.0 ** (c * nbits)
        B2 = B1 * B1
        exp_flags = np.array([(r[8] <= B1) and (r[7] <= B2) for r in records], dtype=bool)
        ctrl_flags = np.array([(l2 <= B1) and (l1 <= B2) for (l1, l2) in ctrl_top], dtype=bool)
        ks = np.array([r[1] for r in records])
        idx = np.array([r[0] for r in records])
        # prediction: Dickman/G with effective size of h_odd reduced by the CL shift
        alpha = math.log(B1) / (mean_log_hodd - shift)
        pred = semismooth_G(alpha, min(2 * alpha, 1.0)) if alpha < 0.5 else 1.0
        pred_p = semismooth_G(2 * c, min(4 * c, 1.0)) if c < 0.25 else 1.0
        # any k among the first K multipliers
        K_list = [1, 2, 3, 5, 10, 20, kcount]
        any_k = {}
        first_hit = []
        for i in range(count):
            flags = exp_flags[idx == i]
            order = np.argsort(ks[idx == i])
            flags = flags[order]
            for K in K_list:
                any_k.setdefault(K, []).append(bool(flags[:K].any()))
            first_hit.append(int(flags.argmax()) + 1 if flags.any() else kcount + 1)
        per_k = float(exp_flags.mean())
        # independence benchmark from the individual rates of the first t multipliers (in k order)
        k_order = sorted(set(ks.tolist()))
        rate_by_k = {k: float(exp_flags[ks == k].mean()) for k in k_order}
        indep_individual = {}
        for K in K_list:
            prod = 1.0
            for k in k_order[:K]:
                prod *= 1.0 - rate_by_k[k]
            indep_individual[str(K)] = 1.0 - prod
        row = {"c": c, "B1_bits": c * nbits, "per_k_exposure": per_k,
               "any_k_indep_pred_individual": indep_individual,
               "size_matched_random_control": float(ctrl_flags.mean()),
               "pred_G_shifted_h_odd": pred, "pred_G_for_p_minus_1": pred_p,
               "any_k": {str(K): float(np.mean(v)) for K, v in any_k.items()},
               "any_k_indep_pred": {str(K): 1 - (1 - per_k) ** K for K in K_list},
               "first_hit_mean": float(np.mean([x for x in first_hit if x <= kcount])) if any(x <= kcount for x in first_hit) else None,
               "never_hit_fraction": float(np.mean([x > kcount for x in first_hit]))}
        out["semismooth"].append(row)
        per_inst[c] = any_k[10]
    out["per_modulus_exposure_k_le_10"] = {str(c): v for c, v in per_inst.items()}
    # per-multiplier summaries: odd-part size and exposure depend on k through the 2-rank and |D|
    ks_all = sorted(set(r[1] for r in records))
    per_k = []
    for k in ks_all:
        sub = [r for r in records if r[1] == k]
        ho = np.array([r[6] for r in sub], dtype=float)
        row = {"k": k, "omega_k": len(factorint(k)) if k > 1 else 0, "count": len(sub),
               "mean_log2_h_odd": float(np.mean(np.log2(ho))), "mean_v2": float(np.mean([r[4] for r in sub])),
               "mean_2rank": float(np.mean([r[5] - 1 for r in sub])),
               "pr_3_divides_h_odd": float(np.mean([3 in r[10] for r in sub]))}
        for c in exponents:
            B1 = 2.0 ** (c * nbits)
            B2 = B1 * B1
            row[f"exposure_c_{c:.4f}"] = float(np.mean([(r[8] <= B1) and (r[7] <= B2) for r in sub]))
        per_k.append(row)
    out["per_k"] = per_k
    return out


def actual_algorithm_experiment(nbits: int = 40, count: int = 400, ks: Sequence[int] = (1, 2, 3, 5, 6), c: float = 1 / 6,
                                seed: int = 41, family: str = "rsa") -> dict:
    """E10b: run the Schnorr-Lenstra algorithm (one random prime form per (N, k))
    and compare with the exposure predicate; classify failures."""
    from ..registry import get_algorithm
    from ..algorithms.classgroup_factor import discriminant as disc_fn
    sl = get_algorithm("schnorr_lenstra")
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    B1 = int(2.0 ** (c * nbits))
    B2 = B1 * B1
    rows = []
    reasons = Counter()
    useless_by_rank = Counter()
    amb_by_rank = Counter()
    for i, inst in enumerate(insts):
        N = int(inst.N)
        for k in ks:
            if math.gcd(k, N) != 1:
                continue
            D = disc_fn(k, N)
            h, _ = class_number(D)
            fac = {int(a): int(b) for a, b in factorint(h).items()}
            h_odd = h >> fac.get(2, 0)
            l1, l2 = _top_two({a: b for a, b in fac.items() if a != 2})
            predicate = (l2 <= B1) and (l1 <= B2)
            rank = omega(D) - 1
            res = sl(inst.N, B1=B1, B2=B2, k=k, seed=seed + 7 * i + k, forms=1)
            rows.append((k, predicate, bool(res.found), rank))
            if not res.found:
                for r in res.meta.get("reasons", []):
                    reasons[r] += 1
                    if r == "useless_ambiguous_form":
                        useless_by_rank[rank] += 1
                        amb_by_rank[rank] += 1
            else:
                amb_by_rank[rank] += 1
    pred = np.array([r[1] for r in rows])
    found = np.array([r[2] for r in rows])
    ranks = sorted(amb_by_rank)
    return {
        "nbits": nbits, "count": count, "ks": list(ks), "c": c, "B1": B1, "B2": B2, "runs": len(rows),
        "predicate_rate": float(pred.mean()), "actual_success_rate": float(found.mean()),
        "success_given_predicate": float(found[pred].mean()) if pred.any() else None,
        "success_given_not_predicate": float(found[~pred].mean()) if (~pred).any() else None,
        "failure_reasons": dict(reasons),
        "useless_ambiguous_rate_by_2rank": {str(r): {"useless": useless_by_rank[r], "ambiguous_reached": amb_by_rank[r],
                                                      "rate": useless_by_rank[r] / amb_by_rank[r],
                                                      "naive_pred": (2 ** (r - 1) - 1) / (2 ** r - 1)} for r in ranks},
    }
