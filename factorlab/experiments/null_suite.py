"""D4: conditional-residue null tests (calibration of the 'local information' model).

Claim (research_plan.md, section 2, [proven]): for a prime l not dividing N,
conditionally on N mod l = n, the residue p mod l is uniform on the units, so
    s = p + q mod l   is distributed as   t + n t^{-1},  t uniform on (Z/l)^*,
    d = q - p mod l   is distributed as   n t^{-1} - t.
These pushforward distributions are non-uniform on their supports of size
(l +- 1)/2 (values with t = n t^{-1} have half weight).

This module (i) computes the exact model distributions, (ii) generates many
RSA-style moduli with the audited generator, (iii) runs pooled chi-square tests
of the empirical conditional histograms against the model, and (iv) as a
positive control, tests the same data against the *wrong* model 'uniform on
the support', which must be rejected.  A pass means the harness reproduces the
exact local-information picture; it is the baseline against which any claimed
non-local signal would have to be judged.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from scipy import stats

from ..gen import make_semiprime


def model_distribution(l: int, n: int, kind: str) -> dict[int, float]:
    """Exact distribution of s = t + n/t (kind='sum') or d = n/t - t (kind='diff')
    for t uniform on (Z/l)^*."""
    if n % l == 0:
        raise ValueError("l divides N")
    c = Counter()
    for t in range(1, l):
        tinv = pow(t, l - 2, l)
        v = (t + n * tinv) % l if kind == "sum" else (n * tinv - t) % l
        c[v] += 1
    return {v: k / (l - 1) for v, k in c.items()}


def conditional_residue_test(nbits: int = 64, count: int = 20000, moduli=(3, 5, 7, 11, 13, 17, 19, 23),
                             family: str = "rsa", seed: int = 77, alpha: float = 0.001) -> dict:
    insts = [make_semiprime(nbits, family, seed, i) for i in range(count)]
    out = {"nbits": nbits, "count": count, "family": family, "per_modulus": {}, "pass": True,
           "wrong_model_rejected_everywhere": True}
    for l in moduli:
        for kind in ("sum", "diff"):
            # group by n = N mod l
            hist = defaultdict(Counter)
            for inst in insts:
                n = int(inst.N % l)
                if n == 0:
                    continue
                v = int((inst.p + inst.q) % l) if kind == "sum" else int((inst.q - inst.p) % l)
                hist[n][v] += 1
            chi2 = 0.0
            dof = 0
            chi2_wrong = 0.0
            dof_wrong = 0
            outside_support = 0
            for n, c in hist.items():
                model = model_distribution(l, n, kind)
                tot = sum(c.values())
                for v, pr in model.items():
                    e = tot * pr
                    chi2 += (c.get(v, 0) - e) ** 2 / e
                dof += len(model) - 1
                outside_support += sum(k for v, k in c.items() if v not in model)
                # wrong model: uniform on the support
                e_w = tot / len(model)
                for v in model:
                    chi2_wrong += (c.get(v, 0) - e_w) ** 2 / e_w
                dof_wrong += len(model) - 1
            pval = 1.0 - stats.chi2.cdf(chi2, dof)
            pval_wrong = 1.0 - stats.chi2.cdf(chi2_wrong, dof_wrong)
            row = {"chi2": chi2, "dof": dof, "p": pval, "outside_support": outside_support,
                   "p_wrong_uniform_model": pval_wrong}
            out["per_modulus"][f"{l}:{kind}"] = row
            if pval < alpha or outside_support:
                out["pass"] = False
            # the wrong model is distinguishable only when the support has a
            # half-weight point, i.e. n is a QR mod l (support (l+1)/2); at l=3
            # with n=2 (non-residue) the two models coincide.  Require rejection
            # whenever the models differ.
            differs = any(abs(pr - 1.0 / len(model_distribution(l, n, kind))) > 1e-12
                          for n in hist for pr in model_distribution(l, n, kind).values())
            if differs and pval_wrong > alpha:
                out["wrong_model_rejected_everywhere"] = False
    return out
