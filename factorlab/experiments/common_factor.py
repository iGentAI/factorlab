"""The common-factor class: an archived implementation check of the collision search on beta = alpha^{N-1}.

Moduli are constructed as p = 1 + k g, q = 1 + l g with g = 2 * ell (ell prime), gcd(k, l) = 1, 1 < l/k < 2 (so q/p < 2 and the
balance constant is C = 2), and k, l of size about N^{1/4} so that the component orders of beta can be as large as N^{1/4} and
the babystep-giantstep search is genuinely exercised (with k = 1 every non-liar base factors N at the immediate gcd step and
the search never runs).  For each modulus the driver runs `common_factor_attack` and, knowing p and q, recomputes the true least
exponent e = min(ord_p beta, ord_q beta) of the successful base's beta, the number of Fermat liars skipped, the branch that found
the factor, and the ratio e / (sqrt N / g) that the lemma bounds by 1.

Run:  python -m factorlab.experiments.common_factor --count 240 --bits 36 44 --seed 9 --out results/e54_common_factor.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from gmpy2 import gcd, is_prime, mpz, powmod
from sympy import factorint

from factorlab.experiments.harvey_residue import common_factor_attack


def multiplicative_order(x: int, p: int) -> int:
    """Exact order of x modulo the prime p (x a unit), by stripping the prime factors of p - 1."""
    e = p - 1
    for pr, mult in factorint(e).items():
        pr = int(pr)
        for _ in range(mult):
            if powmod(x, e // pr, p) == 1:
                e //= pr
            else:
                break
    return e


def common_factor_moduli(count: int, bits_lo: int, bits_hi: int, seed: int = 9) -> List[Dict]:
    """`count` moduli N = pq with p = 1 + k g, q = 1 + l g, g = 2 ell (ell prime), gcd(k, l) = 1, k < l < 2k, N of bits_lo..bits_hi
    bits, k and l about N^{1/4}.  Rejection sampling from a seeded generator; every returned modulus is checked."""
    rng = np.random.default_rng(seed)
    out: List[Dict] = []
    while len(out) < count:
        bits = int(rng.integers(bits_lo, bits_hi + 1))
        # p ~ 2^{bits/2}: choose k ~ 2^{bits/4}, g ~ 2^{bits/4}
        kb = bits // 4
        k = int(rng.integers(2 ** (kb - 1), 2 ** kb))
        l = int(rng.integers(k + 1, 2 * k))
        if math.gcd(k, l) != 1:
            continue
        gb = bits // 2 - kb
        ell = int(rng.integers(2 ** (gb - 2), 2 ** (gb - 1)))
        if not is_prime(ell):
            continue
        g = 2 * ell
        p, q = 1 + k * g, 1 + l * g
        if not (is_prime(p) and is_prime(q)):
            continue
        N = p * q
        if not (bits_lo <= N.bit_length() <= bits_hi):
            continue
        assert math.gcd(p - 1, q - 1) == g
        out.append({"N": N, "p": p, "q": q, "g": g, "k": k, "l": l, "bits": N.bit_length()})
    return out


def check_modulus(row: Dict) -> Dict:
    """Run the attack and recompute, from the known factors, what the lemma says about the successful base."""
    N, p, q, g = row["N"], row["p"], row["q"], row["g"]
    res = common_factor_attack(N)
    out = dict(row)
    out.update(res)
    out["factor_ok"] = res["factor"] in (p, q)
    alpha = res["alpha"]
    if alpha is not None and res["how"] != "gcd(alpha, N)":
        beta_p = int(powmod(alpha, N - 1, p))
        beta_q = int(powmod(alpha, N - 1, q))
        op = multiplicative_order(beta_p, p) if beta_p != 1 else 1
        oq = multiplicative_order(beta_q, q) if beta_q != 1 else 1
        assert (p - 1) % (op * g) == 0 or op == 1
        assert row["k"] % op == 0 and row["l"] % oq == 0  # the component orders divide k and l
        e = min(op, oq) if (op > 1 and oq > 1) else 1
        out.update({"ord_p_beta": op, "ord_q_beta": oq, "least_exponent": e, "ratio_e_over_sqrtN_g": e / (math.sqrt(N) / g)})
    return out


def experiment(count: int, bits_lo: int, bits_hi: int, seed: int) -> Dict:
    rows = [check_modulus(r) for r in common_factor_moduli(count, bits_lo, bits_hi, seed)]
    hows = {}
    for r in rows:
        hows[r["how"]] = hows.get(r["how"], 0) + 1
    ratios = [r["ratio_e_over_sqrtN_g"] for r in rows if "ratio_e_over_sqrtN_g" in r]
    return {
        "count": len(rows), "bits": [bits_lo, bits_hi], "seed": seed,
        "all_factored": all(r["factor_ok"] for r in rows),
        "how": hows,
        "max_liars": max(r["liars"] for r in rows),
        "liar_histogram": {str(v): sum(1 for r in rows if r["liars"] == v) for v in sorted({r["liars"] for r in rows})},
        "max_ratio": max(ratios) if ratios else None,
        "mean_ratio": sum(ratios) / len(ratios) if ratios else None,
        "max_least_exponent": max(r.get("least_exponent", 0) for r in rows),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=240)
    ap.add_argument("--bits", type=int, nargs=2, default=[36, 44])
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--out", default="results/e54_common_factor.json")
    args = ap.parse_args()
    res = experiment(args.count, args.bits[0], args.bits[1], args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=int)
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
