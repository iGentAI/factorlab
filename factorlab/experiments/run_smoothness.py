"""Driver for the smoothness campaign: E8 local joint structure, E9 ECM
hitting/coupling, E10 class groups, E11 QS yield, E12 residue ladder, E13 hard
sets across methods."""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from ..bench import RESULTS_DIR
from ..gen import make_semiprime
from .local_joint import local_joint_test, union_effect
from .ecm_hitting import hitting_experiment, coupling_experiment, adaptive_selection
from .classgroup import classgroup_experiment, actual_algorithm_experiment
from .qs_yield import yield_experiment, sieve_efficiency
from .residue_ladder import residue_ladder


def _public(obj):
    if isinstance(obj, dict):
        return {k: _public(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_public(v) for v in obj]
    return obj


def dump(name, obj):
    with open(os.path.join(RESULTS_DIR, name), "w") as fh:
        json.dump(_public(obj), fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def hard_sets(nbits: int, count: int, c: float, cg: dict, ecm_cp: dict, seed: int = 41) -> dict:
    """Cross-method hard-set analysis on the moduli of the class-group experiment."""
    from .smooth_profiles import _top_two
    insts = [make_semiprime(nbits, "rsa", seed, i) for i in range(count)]
    B1 = 2.0 ** (c * nbits)
    B2 = B1 * B1

    def ss(x):
        l1, l2 = _top_two(x)
        return (l2 <= B1) and (l1 <= B2)

    pm = np.array([ss(int(i.p) - 1) or ss(int(i.q) - 1) or ss(int(i.p) + 1) or ss(int(i.q) + 1) for i in insts])
    cgx = np.array(cg["per_modulus_exposure_k_le_10"][str(c)], dtype=bool)
    ecmx = np.array(ecm_cp["per_modulus_success_any_curve"], dtype=bool)

    def cond(a, b):
        return float(a[b].mean()) if b.any() else None

    return {
        "nbits": nbits, "count": count, "c": c,
        "rates": {"p_pm_1_any_of_four": float(pm.mean()), "classgroup_k_le_10": float(cgx.mean()),
                  "ecm_" + str(ecm_cp["n_curves"]) + "_curves": float(ecmx.mean())},
        "conditional": {
            "classgroup_given_pm1_fail": cond(cgx, ~pm), "classgroup_given_pm1_success": cond(cgx, pm),
            "ecm_given_pm1_fail": cond(ecmx, ~pm), "ecm_given_pm1_success": cond(ecmx, pm),
            "ecm_given_classgroup_fail": cond(ecmx, ~cgx),
            "all_three_fail": float((~pm & ~cgx & ~ecmx).mean()),
            "indep_pred_all_fail": float((1 - pm.mean()) * (1 - cgx.mean()) * (1 - ecmx.mean())),
        },
        "corr": {"pm1_classgroup": float(np.corrcoef(pm.astype(float), cgx.astype(float))[0, 1]),
                 "pm1_ecm": float(np.corrcoef(pm.astype(float), ecmx.astype(float))[0, 1]),
                 "classgroup_ecm": float(np.corrcoef(cgx.astype(float), ecmx.astype(float))[0, 1])},
    }


def main(quick: bool = False, sections=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n48 = 600 if quick else 6000
    n40 = 200 if quick else 1500
    run = lambda s: sections is None or s in sections  # noqa: E731

    if run("e8"):
        print("== E8: joint small-prime structure of p-1, q-1, p+1, q+1 given N ==")
        lj = local_joint_test(48, n48)
        for k, v in lj["per_modulus"].items():
            cr = "  ".join(f"n={n}:{r['type'][:4]} obs={r['corr_obs']:+.3f} th={r['corr_theory']:+.3f}" for n, r in sorted(v["corr_by_residue"].items(), key=lambda kv: int(kv[0]))[:6])
            print(f"  {k:5s} chi2={v['chi2']:8.2f} dof={v['dof']:3d} p={v['p']:.3f} outside={v['outside_support']}  {cr}")
        print(f"  model pass={lj['pass']}")
        ue = union_effect(48, n48, c=1 / 6)
        for g, v in ue["groups"].items():
            print(f"  {g}: any4={v['any_of_four']:.3f}+-{v['se']:.3f} single={v['mean_single']:.3f} corr(p-1,q-1)={v['corr_pm1_qm1']:+.3f} "
                  f"corr(p-1,p+1)={v['corr_pm1_pp1']:+.3f} degenerate={v['degenerate_equal_order_rate']:.4f} n={v['count']}")
        for m in (5, 7, 8):
            r = ue[f"by_N_mod_{m}"]
            print(f"  N mod {m}: paired any4={r['paired']['any_of_four']:.3f} corr={r['paired']['corr_pm1_qm1']:+.3f} degen={r['paired']['degenerate_equal_order_rate']:.4f} (n={r['paired']['count']}) | "
                  f"unpaired any4={r['unpaired']['any_of_four']:.3f} corr={r['unpaired']['corr_pm1_qm1']:+.3f} degen={r['unpaired']['degenerate_equal_order_rate']:.4f} (n={r['unpaired']['count']})")
        dump("e8_local_joint.json", {"chi2": lj, "union": ue})

    if run("e9"):
        print("== E9: per-prime ECM success, heterogeneity, fixed-enumeration hitting sets ==")
        he = hitting_experiment(pbits=16 if quick else 18, nsig=60 if quick else 200, B1=200)
        print(f"  primes={he['n_primes']} curves={he['n_curves']} B1={he['B1']} theta mean={he['theta_mean']:.4f} std={he['theta_std']:.4f} "
              f"(binomial {he['binomial_std']:.4f}) quantiles={['%.3f' % x for x in he['theta_quantiles']]} rho(p/12)={he['rho_prediction_p_over_12']:.4f}")
        for m, d in he["by_class"].items():
            print(f"   theta by p {m}: " + " ".join(f"{r}:{v:.3f}" for r, v in d.items()))
        print(f"  covering curves: fixed order {he['covering_curves_fixed_order']}; common random orders (20 draws) mean {he['covering_curves_common_random_mean']:.1f} "
              f"range {min(he['covering_curves_common_random_orders'])}-{max(he['covering_curves_common_random_orders'])}; never hit {he['fraction_never_hit']:.4f}")
        print(f"  uncovered after t=1,5,10,20,40 fixed: {[round(he['uncovered_after_t_fixed'][t-1],4) for t in (1,5,10,20,40)]} exact pred: {[round(he['uncovered_after_t_pred_exact'][t-1],4) for t in (1,5,10,20,40)]} indep approx: {[round(he['uncovered_after_t_pred_indep'][t-1],4) for t in (1,5,10,20,40)]}")
        print(f"  hardest primes: {he['hardest_primes'][:5]}")
        cp = coupling_experiment(48, n48, nsig=20 if quick else 60, c=1 / 6)
        print(f"  coupling: theta_p={cp['theta_p']:.3f} overall corr(succ_p, succ_q)={cp['overall_corr']:+.4f}")
        for m, d in cp["by_class"].items():
            print(f"   N {m}: " + " ".join(f"{r}:{v['corr']:+.3f}/{v['union_rate']:.3f}" for r, v in d.items()))
        print("== E14: N-adaptive curve selection (train/test split) ==")
        ad = {}
        for m in (3, 8, 24):
            ad[f"mod_{m}"] = adaptive_selection(cp["_Sp"], cp["_Sq"], cp["_Nmod"][m], k_select=10)
            a = ad[f"mod_{m}"]
            print(f"  classes N mod {m}: first-10 curves {a['first_k_curves']:.4f}  best-10 global {a['best_k_global']:.4f}  adaptive {a['adaptive_per_class']:.4f}  "
                  f"per-curve union global {a['per_curve_union_rate_global']:.4f} adaptive {a['per_curve_union_rate_adaptive']:.4f}")
        dump("e9_ecm_hitting.json", {"hitting": he, "coupling": cp, "adaptive": ad})

    cg = None
    if run("e10") or run("e13"):
        print("== E10: class numbers h(-kN), Cohen-Lenstra, Schnorr-Lenstra exposure ==")
        cg = classgroup_experiment(40, n40, kcount=10 if quick else 30)
        print(f"  records={cg['records']} methods={cg['methods']} genus 2-rank ok={cg['genus_2rank_ok']} mean(v2-rank)={cg['mean_v2_minus_rank']:.3f} "
              f"mean h/sqrt|D|={cg['mean_h_over_sqrtD']:.4f} mean log2 h_odd={cg['mean_log2_h_odd']:.2f} (log2 p={cg['mean_log2_p']}) CL shift={cg['cohen_lenstra_log_shift']:.3f}")
        for l, d in cg["odd_divisibility"].items():
            print(f"   l={l}: Pr[l|h_odd] obs={d['observed']:.4f}+-{d['se']:.4f} CL={d['cohen_lenstra']:.4f} random={d['random_integer']:.4f}  E[v] obs={d['mean_valuation']:.3f} CL={d['cl_expected_valuation']:.3f} rand={d['random_expected_valuation']:.3f}")
        for r in cg["semismooth"]:
            print(f"   c={r['c']:.4f} per-k exposure={r['per_k_exposure']:.3f} size-matched random control={r['size_matched_random_control']:.3f} "
                  f"pred(h_odd,CL shift)={r['pred_G_shifted_h_odd']:.3f} pred(p-1)={r['pred_G_for_p_minus_1']:.3f} "
                  f"any k<=1,3,10,K: {[round(r['any_k'][K],3) for K in ('1','3','10',str(cg['kcount']))]} indep(pooled): {[round(r['any_k_indep_pred'][K],3) for K in ('1','3','10',str(cg['kcount']))]} "
                  f"indep(individual): {[round(r['any_k_indep_pred_individual'][K],3) for K in ('1','3','10',str(cg['kcount']))]} first-hit mean={r['first_hit_mean']}")
        print("   per k: " + "; ".join(f"k={r['k']} rank={r['mean_2rank']:.1f} log2h_odd={r['mean_log2_h_odd']:.2f} exp(1/6)={r['exposure_c_0.1667']:.3f} P3={r['pr_3_divides_h_odd']:.3f}" for r in cg["per_k"][:12]))
        dump("e10_classgroup.json", {k: v for k, v in cg.items() if k != "per_modulus_exposure_k_le_10"})
        print("== E10b: the Schnorr-Lenstra algorithm itself ==")
        aa = actual_algorithm_experiment(40, 100 if quick else 400)
        print(f"  runs={aa['runs']} predicate rate={aa['predicate_rate']:.3f} actual success={aa['actual_success_rate']:.3f} "
              f"success|predicate={aa['success_given_predicate']} success|not={aa['success_given_not_predicate']} reasons={aa['failure_reasons']}")
        for r, d in aa["useless_ambiguous_rate_by_2rank"].items():
            print(f"   2-rank {r}: useless ambiguous {d['useless']}/{d['ambiguous_reached']} = {d['rate']:.3f} (naive {d['naive_pred']:.3f})")
        dump("e10b_schnorr_lenstra.json", aa)

    if run("e13"):
        print("== E13: hard sets across methods (40-bit moduli of E10) ==")
        cp40 = coupling_experiment(40, n40, nsig=20, c=1 / 6, seed=41)
        hs = hard_sets(40, n40, 1 / 6, cg, cp40)
        print(f"  rates={hs['rates']}")
        print(f"  conditional={hs['conditional']}")
        print(f"  corr={hs['corr']}")
        dump("e13_hard_sets.json", hs)

    if run("e11"):
        print("== E11: quadratic sieve yield versus Dickman with the local shift ==")
        ye = yield_experiment(nbits_list=(48, 56, 64) if quick else (48, 56, 64, 72, 80), count=2 if quick else 4)
        for nb, d in ye["by_bits"].items():
            print(f"  {nb} bits: found {d['found']}/{d['count']} full={d['full_total']} obs/pred(shift)={d['ratio_obs_over_pred_shift']:.3f} "
                  f"obs/pred(plain)={d['ratio_obs_over_pred_plain']:.3f} shift={d['mean_shift_nats']:+.2f} nats work={d['mean_work']:.3e} wall={d['mean_wall']:.1f}s")
        print(f"  L[1/2] fit: c={ye['L_half_fit']['slope_c']:.3f}")
        se = sieve_efficiency()
        for nb, d in se["by_bits"].items():
            print(f"  {nb} bits exhaustive: sieve/exhaustive={d['sieve_over_exhaustive']:.3f} exhaustive/pred(shift)={d['exhaustive_over_pred_shift']:.3f} "
                  f"exhaustive/pred(plain)={d['exhaustive_over_pred_plain']:.3f} Q/random-control={d['Q_over_random_control']:.3f} "
                  f"Q/shift-adjusted-control={d['Q_over_shift_adjusted_control']:.3f} "
                  f"random-control/its-pred={d['random_control_over_its_pred']:.3f} shift={d['mean_shift_nats']:+.2f} "
                  f"(smooth Q {d['exhaustive_total']}, sieve {d['sieve_total']}, random {d['random_control_total']})")
        ye["sieve_efficiency"] = se
        dump("e11_qs_yield.json", ye)

    if run("e12"):
        print("== E12: residue ladder (Dixon, Vallee, Lehman range, QS) at 48 bits, B = N^(1/6) ==")
        rl = residue_ladder(48, 30 if quick else 100, 60 if quick else 200)
        for name, d in rl["classes"].items():
            print(f"  {name:16s} log2 r/log2 N={d['mean_log2_r_over_log2_N']:.3f} smooth obs={d['smooth_obs']:.4f} pred={d['smooth_pred_plain']:.4f}/{d['smooth_pred_shift']:.4f} "
                  f"semismooth obs={d['semismooth_obs']:.4f} pred={d['semismooth_pred_plain']:.4f}/{d['semismooth_pred_shift']:.4f} shift={d['mean_shift_nats']:+.2f}")
        dump("e12_residue_ladder.json", rl)


if __name__ == "__main__":
    secs = None
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            secs = set(a[len("--only="):].split(","))
    main(quick="--quick" in sys.argv, sections=secs)
