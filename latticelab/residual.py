"""The residual cap at the forced neighbour (docs/notes_lattice_barrier.md, section 10).

Setting (0-based positions; F_j := L(b_0, ..., b_{j-1}) the flag).  An exact-SVP insertion at kappa with a full block P := F_{kappa+beta}/F_kappa
and answer v (a shortest vector of P, lifted) forces F_{kappa+1}' = F_kappa + Z v and leaves F_{kappa+beta}' = F_{kappa+beta}; with a
block-supported completion every F_j with j <= kappa or j >= kappa+beta is unchanged.  Hence, on the new basis,
    the block [kappa+1, kappa+beta)   (size beta-1)  is  P/v := F_{kappa+beta}/F_{kappa+1}'    -- the HKZ residual of the queried block,
    the block [kappa+1, kappa+beta+1) (size beta)    is  Q/v := F_{kappa+beta+1}/F_{kappa+1}'  -- the forced neighbour, Q := F_{kappa+beta+1}/F_kappa,
both independent of the completion (flag.py), and Q/v is exactly the block a forward tour queries at its next step.

Lemma (residual cap).  P/v is a primitive sublattice of Q/v (P is primitive in Q and v is a primitive vector of P), so
        lambda_1(Q/v) <= lambda_1(P/v).
Writing GH_n(X) := chat(n) vol(X)^{1/n} and L(n) := log chat(n), the forced neighbour's signed sub-GH ratio decomposes EXACTLY as
        eps(Q/v) := log GH_beta(Q/v) - log lambda_1(Q/v)  =  gh_shift + res_ratio + gap,
        gh_shift  := log GH_beta(Q/v) - log GH_{beta-1}(P/v)  =  L(beta) - L(beta-1) + l_{kappa+beta}/beta - log vol(P/v) / (beta (beta-1)),
        res_ratio := log GH_{beta-1}(P/v) - log lambda_1(P/v)     (the residual's own signed ratio: a property of the queried block and of the
                                                                   selected shortest line, of the block alone when that line is unique),
        gap       := log lambda_1(P/v) - log lambda_1(Q/v) >= 0   (the cap's slack: the extension's effect on the minimum; its determinant
                                                                   already enters gh_shift through l_{kappa+beta}),
where l_{kappa+beta} = log ||b_{kappa+beta}^*|| is the input profile's entry (log vol Q - log vol P) and vol(P/v) = vol(P)/||v||; gh_shift is
determined by the post-insertion profile, i.e. by the queried block, the selected shortest vector and the adjacent volume increment.  Consequently
        eps(Q/v) >= gh_shift + res_ratio :
an exact additive transmission identity -- a residual with signed ratio r gives the block queried next a ratio of at least gh_shift + r, and
forces a positive one exactly when gh_shift + r > 0.  On a GSA-tight input profile (l_j = l_kappa - s (j - kappa), s = 2 L(beta)/(beta-1),
l_kappa' = l_kappa; beta >= 3)
        gh_shift = (beta - 2) [f(beta) - f(beta-1)],      f(n) := L(n)/(n-1),
positive for 3 <= beta <= 36 (f increases to its integer maximum at 36) and negative for beta >= 37, of magnitude below 5e-3 for every
integer beta >= 27 (+0.0060 at 25, +0.0052 at 26, +0.0026 at 30, -0.0010 at 40, -0.0027 at 50 and at 403); a steeper (LLL-like) input slope
lowers it by half the slope excess.  In the language of section 7's completion rules, with H := log GH_{beta-1}(P/v) the value the hkz rule
assigns to l_{kappa+1}: H - res_ratio = log lambda_1(P/v), so the true next entry satisfies log lambda_1(Q/v) <= H - res_ratio -- an actual
exact HKZ completion (first residual entry log lambda_1(P/v)) is a literal upper bound on the tour's next entry, the GH rule H is one only up
to the residual's own fluctuation, and no bound of either kind follows for the uniform rule's l_{kappa+1} + m/(beta-1).

`neighbour_terms` evaluates the terms (unpruned enumeration in dimensions beta-1 and beta on a floating-point GSO: numerically exhaustive, not a
formal certificate) on a GSO object; `forced_neighbour_decomposition`
does one insertion and reports the terms before and after; `residual_census` runs strict bounded forward tours step by step, records the
decomposition of the block queried next after every step, and checks that its minimum is what the next step finds; `residual_ratio_random`
measures res_ratio and eps(P) on random beta-dimensional q-ary lattices -- the control for the in-tour residual ratios; `inheritance_stats`
and `control_dependence` are the reproducible post-processing of the archived steps and control rows (correlation of a block's ratio with its
residual's and with the next block's, conditional rates).

The thinned residual (why the residual is long, and why density is inherited).  Let v be a shortest vector of P and x = pi(w) a nonzero point
of P/v with ||x|| = rho.  The lifts of x are w + k v; the one with offset tau := <w + k v, v>/||v|| in (-||v||/2, ||v||/2] has norm^2 = rho^2 + tau^2,
and it is a nonzero vector of P, so minimality of v forces rho^2 + tau^2 >= ||v||^2.  Two consequences.  (1) Deterministic (the classical
Lovasz-type bound): every nonzero x in P/v has ||x||^2 >= (3/4) ||v||^2, i.e. lambda_1(P/v) >= (sqrt 3 / 2) lambda_1(P), so
        res_ratio  <=  log(GH_{beta-1}(P/v) / ||v||) - log(sqrt 3 / 2)  =  A(beta) + beta eps(P)/(beta-1) + 0.1438,
        A(beta) := L(beta-1) - beta L(beta)/(beta-1)  (= log(GH_{beta-1}(P/v)/||v||) when ||v|| = GH_beta(P); -0.031 at 20, -0.024 at 40),
since log GH_{beta-1}(P/v) - log ||v|| = A(beta) + beta eps(P)/(beta-1) exactly (vol(P/v) = vol(P)/||v||).  The same centred lift is
linearly independent of v, so ||w||^2 >= lambda_2(P)^2 and in fact lambda_1(P/v)^2 >= lambda_2(P)^2 - lambda_1(P)^2/4 >= (3/4) lambda_2(P)^2.
(2) Heuristic (the thinned Poisson surrogate): suppose the sign pairs of P/v of norm <= rho ||v|| form a Poisson process of mean N_0(rho) =
(rho/c)^{beta-1}/2, c := GH_{beta-1}(P/v)/||v||, and -- the marked-Poisson surrogate, which deliberately discards the fact that the offsets of a
lattice are the values of a homomorphism P/v -> R/||v||Z and hence dependent across points -- that distinct sign pairs carry iid centred offsets
tau, uniform in (-||v||/2, ||v||/2] and independent of all radii.  Then the points whose lift is shorter than v and the points whose lift is not
are independent Poisson processes (Poisson splitting), and conditioning on v being shortest (no point of the first kind) leaves a Poisson process
of intensity g(rho) dN_0 with g(rho) := P[rho^2 + tau^2 >= 1] = (1 - 2 sqrt(1 - rho^2))^+ for rho < 1 and 1 for rho >= 1: the law of U :=
lambda_1(P/v)/||v|| has P[U > u] = exp(-int_0^u g dN_0), and res_ratio = log c - log U.  The law has no fitted coefficient; it is indexed by beta
and the observed block ratio eps(P).  Its predictions: the bias E[res_ratio] (-0.051, -0.030, -0.021 at beta = 20, 30, 40 against the unthinned
Poisson sign-pair constants -(log 2 - gamma)/(beta-1) = -0.006, -0.004, -0.003, and the q-ary controls' -0.049, -0.028, -0.019), P[res_ratio > 0]
(0.13, 0.16, 0.18 against 0.14, 0.16, 0.21), and the dependence on eps(P): at fixed block covolume a larger eps(P) means a shorter v and a larger
c, so relative to the residual's own GH scale the thinning transition moves inward (g(cz) increases at every fixed z = U/c) and the residual
minimum lands closer to that scale; the regression slope of res_ratio on eps(P) is 0.32, 0.27, 0.23 (controls 0.35, 0.28) where the unthinned
Poisson model, being scale-free in c, gives exactly 0.  The successive-minima ordering correlation of the earlier Poisson proxy is a consequence
of the same constraint (lambda_2(P) is realised by the shortest admissible lift), not a separate ingredient.  `thinned_residual_law`,
`thinned_residual_model` and `compare_control_with_model` evaluate and score this against the archive (moments with standard errors, the
probability-integral transform pooled and by eps-terciles, and its independence from eps: a pooled uniform PIT is necessary, not sufficient).

The extension's undercut (the gap).  Q/v = union of the layers P/v + j w_0 at heights j h_abs, h_abs := ||b_{kappa+beta}^*|| (the last
Gram-Schmidt norm of the block, unchanged by the insertion); write hbar := h_abs/||v|| for the normalised height.  A layer point has norm^2 =
j^2 h_abs^2 + (in-plane norm)^2.  There is no deterministic lift exclusion for the points of Q outside P; as a separate heuristic, the points
of the nonzero layers are modelled as a Poisson process, independent of the residual given (u*, hbar, c), with mean count of sign pairs of norm
<= rho ||v|| equal to  count(rho) = sum_{j >= 1} ((rho^2 - j^2 hbar^2)^+ / c^2)^{(beta-1)/2}  (the layers -j are the negatives of the layers j,
so no factor 1/2), whence P[gap > 0] = 1 - exp(-count(u*)) with u* = lambda_1(P/v)/||v||, and E[gap] = int_hbar^{u*} (1 - exp(-count(rho))) drho/rho.
`undercut_law` and `gap_model_check` evaluate this per step against the census (which records l_kappa = log ||v|| and l_{kappa+beta} for the
purpose; the census steps of one trajectory are dependent, so the standard errors reported there are naive iid ones).

An exact identity ties the three profile quantities together: since vol(Q/v) = vol(P/v) h_abs,
        gh_shift = L(beta) - ((beta-1)/beta) L(beta-1) + (log hbar - log c)/beta          (`gh_shift_from_profile`),
so given the queried block's ratio (hence c) and the normalised height hbar, gh_shift is determined.

The thinned-residual world (`thinned_world_chain`).  Along a forward tour the block queried next is Q/v, so its ratio is
        eps_{k+1} = gh_shift_k + res_ratio(eps_k) + gap_k.
Imposing exogenous forcing by the normalised height hbar (fixed, or drawn iid per step from supplied values) and fresh conditional draws of the
residual and the layers at every step, this becomes a scalar Markov chain on the queried block's ratio: given eps_k, c = c(eps_k), gh_shift from
the identity above, U from the thinned law (so res_ratio = log c - log U and u* = U), the layer minimum rho from the undercut law and gap =
log u* - log min(u*, rho).  The queried-block ratio of a real tour is NOT asserted to be Markov (it retains the profile, the residual lattice,
the extension and the insertion status, and the non-inserting steps -- a retained vector against a re-forced block -- are not modelled).  If
the chain has a stationary law, that law would model the per-query sub-GH statistic of a tour in this surrogate; `world_fixed_point` solves the
mean-map equation e = H(e) + M(e) + G(e) for the fixed point of the conditional means, and the chain's summary also reports the post-hoc
affine balance (gh_shift + E[res | 0] + mean gap)/(1 - slope) under that name.  Everything in it is heuristic except the deterministic
inequalities and the identity it is built on, and it is validated, if at all, only by comparison with the census.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from fpylll import GSO, IntegerMatrix

from latticelab.flag import _insert, _reducer
from latticelab.profile_floor import log_chat


NEXT_QUERY_TOL = 1e-8  # tolerance on |log ||b_{kappa+1}^*|| after the next step - log lambda_1(Q/v) recorded|: floating-point GSO roundoff


def _lambda1(M: GSO.Mat, first: int, last: int, log_vol: float) -> float:
    """Exact lambda_1 of the projected block [first, last) of M by unpruned enumeration (radius grown from 1.05 GH until a vector is found)."""
    from fpylll import Enumeration, EnumerationError

    n = last - first
    radius2 = (1.05 * math.exp(log_chat(n) + log_vol / n)) ** 2
    while True:
        try:
            sols = Enumeration(M).enumerate(first, last, radius2, 0, pruning=None)
            return math.sqrt(min(s[0] for s in sols))
        except EnumerationError:
            radius2 *= 1.5


def neighbour_terms(M: GSO.Mat, kappa: int, beta: int) -> Dict:
    """On the current basis of M: the residual block [kappa+1, kappa+beta) (size beta-1) and the neighbour block [kappa+1, kappa+beta+1) (size
    beta) -- after an insertion at kappa these are P/v and Q/v -- with their exact minima, GH lengths, the three terms and the neighbour's
    signed ratio eps = gh_shift + res_ratio + gap.  Requires a full neighbour block: kappa + beta <= d - 1."""
    d = M.d
    if beta < 3:
        raise ValueError("need beta >= 3 (a residual of dimension >= 2)")
    if not (0 <= kappa and kappa + beta <= d - 1):
        raise ValueError(f"need 0 <= kappa and kappa + beta <= d - 1 for a full neighbour block (kappa={kappa}, beta={beta}, d={d})")
    M.update_gso()
    r = [M.get_r(j, j) for j in range(d)]
    log_vol_res = 0.5 * sum(math.log(r[j]) for j in range(kappa + 1, kappa + beta))
    l_last = 0.5 * math.log(r[kappa + beta])
    log_vol_nb = log_vol_res + l_last
    lam_res = _lambda1(M, kappa + 1, kappa + beta, log_vol_res)
    lam_nb = _lambda1(M, kappa + 1, kappa + beta + 1, log_vol_nb)
    log_gh_res = log_chat(beta - 1) + log_vol_res / (beta - 1)
    log_gh_nb = log_chat(beta) + log_vol_nb / beta
    gh_shift = log_gh_nb - log_gh_res
    res_ratio = log_gh_res - math.log(lam_res)
    gap = math.log(lam_res) - math.log(lam_nb)
    return {"kappa": kappa, "beta": beta, "l_kappa": 0.5 * math.log(r[kappa]), "l_kappa_plus_beta": l_last, "log_vol_residual": log_vol_res,
            "lambda1_residual": lam_res, "lambda1_neighbour": lam_nb, "log_gh_residual": log_gh_res, "log_gh_neighbour": log_gh_nb,
            "gh_shift": gh_shift, "res_ratio": res_ratio, "gap": gap, "eps_neighbour": log_gh_nb - math.log(lam_nb),
            "cap_bound": gh_shift + res_ratio}


def forced_neighbour_decomposition(A: IntegerMatrix, kappa: int, beta: int, variant: str = "bounded_lll") -> Tuple[Dict, IntegerMatrix]:
    """One strict exact-SVP insertion at 0-based kappa (full block and full neighbour: kappa + beta <= d - 1) with the block-supported completion
    `variant`; returns the neighbour terms before (the current neighbour block Q/b_kappa) and after (the forced neighbour Q/v), the removed
    mass, and the created ratio eps_after - eps_before, together with the new basis."""
    d = A.nrows
    if variant == "conveyor":
        raise ValueError("the decomposition needs a block-supported completion (the conveyor changes earlier flag members)")
    if not (0 <= kappa and kappa + beta <= d - 1):
        raise ValueError(f"need 0 <= kappa and kappa + beta <= d - 1 (kappa={kappa}, beta={beta}, d={d})")
    bkz, B, M = _reducer(A)
    before = neighbour_terms(M, kappa, beta)
    _insert(bkz, kappa, beta, variant)
    after = neighbour_terms(M, kappa, beta)
    removed = before["l_kappa"] - after["l_kappa"]
    return ({"kappa": kappa, "beta": beta, "d": d, "variant": variant, "removed_mass": removed, "changed": bool(removed > 1e-9),
             "before": before, "after": after, "created_ratio": after["eps_neighbour"] - before["eps_neighbour"],
             "cap_holds": bool(after["lambda1_neighbour"] <= after["lambda1_residual"] * (1 + 1e-12))}, B)


def _summary(steps: List[Dict]) -> Dict:
    if not steps:
        return {"n": 0}
    g = np.array([s["gh_shift"] for s in steps])
    rr = np.array([s["res_ratio"] for s in steps])
    gp = np.array([s["gap"] for s in steps])
    e = np.array([s["eps_neighbour"] for s in steps])
    cb = g + rr
    out = {"n": len(steps), "mean_gh_shift": float(g.mean()), "mean_res_ratio": float(rr.mean()), "sd_res_ratio": float(rr.std()),
           "mean_gap": float(gp.mean()), "mean_eps_neighbour": float(e.mean()), "frac_eps_neighbour_positive": float((e > 1e-12).mean()),
           "frac_cap_bound_positive": float((cb > 1e-12).mean()), "mean_cap_bound_positive_part": float(np.maximum(cb, 0).mean()),
           "mean_eps_neighbour_positive_part": float(np.maximum(e, 0).mean()), "frac_gap_zero": float((gp < 1e-9).mean()),
           "max_eps_neighbour": float(e.max()), "max_res_ratio": float(rr.max()), "min_cap_slack": float(gp.min())}
    if "created_ratio" in steps[0]:
        cr = np.array([s["created_ratio"] for s in steps])
        ch = np.array([s["changed"] for s in steps])
        out.update({"mean_created_ratio": float(cr.mean()), "frac_changed": float(ch.mean()),
                    "mean_created_ratio_when_changed": float(cr[ch].mean()) if ch.any() else float("nan"),
                    "frac_violation_created": float(np.mean([(s["eps_neighbour"] > 1e-12) and (s["eps_before"] <= 1e-12) for s in steps]))})
    return out


def residual_census(A: IntegerMatrix, beta: int, tours: int, variant: str = "bounded_lll", record_before: bool = True,
                    final_ratios: bool = False) -> Dict:
    """Strict bounded forward tours from A (positions 0..d-2 in order, the insertions of `latticelab.flag`), recording after every step kappa
    with a full neighbour block (kappa + beta <= d - 1) the decomposition of the block the tour queries next, Q_kappa/v_kappa = P_{kappa+1}:
    gh_shift, res_ratio, gap, eps_neighbour, the removed mass, and (if `record_before`) the same block's ratio before the step and the created
    ratio.  Also checks the flag theorem's forced-neighbour claim: the minimum recorded for Q_kappa/v_kappa equals log ||b_{kappa+1}^*|| after
    the next step (which inserts that block's shortest vector, or leaves b_{kappa+1}^* in place when it already is one, up to the tie tolerance);
    the maximal absolute mismatch is reported and compared with NEXT_QUERY_TOL (double-precision GSO roundoff reaches 1e-9 at d = 100).
    With `final_ratios`, the OUTPUT basis's block ratios (block_gh_ratios on the final basis) are compared position by position with the last
    tour's queried-block ratios: the output block at start kappa+1 is F_{kappa+beta+1}^{fin}/F'_{kappa+1} -- the same denominator as the queried
    block Q_kappa/v_kappa = F_{kappa+beta+1}/F'_{kappa+1}, whose numerator the following beta-1 insertions of the tour replace -- so
    delta_kappa := eps(output block) - eps(queried block) is the sub-GH mass created by the later numerator changes.
    Returns the per-step records, per-tour summaries and an overall summary."""
    d = A.nrows
    if variant == "conveyor":
        raise ValueError("the census needs a block-supported completion")
    if beta < 3 or d < beta + 2:
        raise ValueError("need beta >= 3 and d >= beta + 2")
    bkz, B, M = _reducer(A)
    steps: List[Dict] = []
    max_next_mismatch = 0.0
    query_all = None  # the last tour's queried block at EVERY position: size, eps, log lambda_1 (the forced entry), log vol of its numerator
    for T in range(1, tours + 1):
        pending = None  # log lambda_1 recorded for the block queried at the next step
        if T == tours:
            query_all = {"size": [], "eps": [], "log_lambda": [], "log_vol_num": []}
        for kappa in range(d - 1):
            size = min(beta, d - kappa)
            full_nb = kappa + beta <= d - 1
            before = neighbour_terms(M, kappa, beta) if (full_nb and record_before) else None
            M.update_gso()
            l0 = 0.5 * math.log(M.get_r(kappa, kappa))
            _insert(bkz, kappa, size, variant)
            M.update_gso()
            l1 = 0.5 * math.log(M.get_r(kappa, kappa))
            if pending is not None:
                # the previous step's forced neighbour is this step's block: its recorded minimum must be what this step found
                max_next_mismatch = max(max_next_mismatch, abs(l1 - pending))
            pending = None
            if query_all is not None:
                # the block queried at this step, F_{kappa+size}/F'_kappa: its volume is unchanged by the insertion, and in a strict tour the new
                # entry l1 is log lambda_1 of the block (the forced entry of the flag theorem)
                lv_block = 0.5 * sum(math.log(M.get_r(j, j)) for j in range(kappa, kappa + size))
                query_all["size"].append(size); query_all["eps"].append(log_chat(size) + lv_block / size - l1); query_all["log_lambda"].append(l1)
                query_all["log_vol_num"].append(0.5 * sum(math.log(M.get_r(j, j)) for j in range(kappa + size)))
            if full_nb:
                after = neighbour_terms(M, kappa, beta)
                rec = {"tour": T, "kappa": kappa, "removed_mass": l0 - l1, "changed": bool(l0 - l1 > 1e-9),
                       **{k: after[k] for k in ("gh_shift", "res_ratio", "gap", "eps_neighbour", "cap_bound", "lambda1_residual", "lambda1_neighbour",
                                                "l_kappa", "l_kappa_plus_beta")},
                       "log_vol_numerator": 0.5 * sum(math.log(M.get_r(j, j)) for j in range(kappa + beta + 1))}  # log vol F_{kappa+beta+1} now
                if before is not None:
                    rec.update({"eps_before": before["eps_neighbour"], "created_ratio": after["eps_neighbour"] - before["eps_neighbour"]})
                steps.append(rec)
                pending = math.log(after["lambda1_neighbour"])
    per_tour = {T: _summary([s for s in steps if s["tour"] == T]) for T in range(1, tours + 1)}
    out = {"d": d, "beta": beta, "tours": tours, "variant": variant, "steps": steps, "per_tour": per_tour, "summary": _summary(steps),
           "max_next_query_mismatch": max_next_mismatch, "next_query_consistent": bool(max_next_mismatch <= NEXT_QUERY_TOL),
           "final_profile": [0.5 * math.log(M.get_r(j, j)) for j in range(d)]}
    if final_ratios:
        from latticelab.profile import block_gh_ratios

        st = block_gh_ratios(B, beta)
        out_rows = {row["i"] - 1: row for row in st["blocks"]}  # 0-based start -> output block row
        fin = np.array(out["final_profile"])
        last = {s["kappa"]: s for s in steps if s["tour"] == tours}
        q, o, dvol, dmin = [], [], [], []
        for kappa, s in sorted(last.items()):
            if kappa + 1 in out_rows:
                row = out_rows[kappa + 1]
                q.append(s["eps_neighbour"]); o.append(row["log_gh_over_lambda1"])
                # delta = (log vol_out - log vol_q)/beta + (log lambda_1(q) - log lambda_1(out)); the same denominator F'_{kappa+1} cancels
                dvol.append((float(fin[:kappa + beta + 1].sum()) - s["log_vol_numerator"]) / beta)
                dmin.append(math.log(s["lambda1_neighbour"]) - math.log(row["lambda1"]))
        q, o, dvol, dmin = (np.array(x) for x in (q, o, dvol, dmin))
        delta = o - q
        out["output_vs_queried"] = {"n": int(len(q)), "mean_queried_eps": float(q.mean()), "mean_output_eps": float(o.mean()), "mean_delta": float(delta.mean()),
                                    "sd_delta": float(delta.std()), "frac_delta_positive": float((delta > 1e-12).mean()),
                                    "frac_queried_positive": float((q > 1e-12).mean()), "frac_output_positive": float((o > 1e-12).mean()),
                                    "corr_queried_output": float(np.corrcoef(q, o)[0, 1]) if len(q) > 2 and q.std() > 0 and o.std() > 0 else float("nan"),
                                    "mean_delta_volume_part": float(dvol.mean()), "mean_delta_minimum_part": float(dmin.mean()),
                                    "min_delta_minimum_part": float(dmin.min()), "frac_minimum_part_positive": float((dmin > 1e-9).mean()),
                                    "max_decomposition_gap": float(np.max(np.abs(delta - dvol - dmin))),
                                    "queried_eps": q.tolist(), "output_eps": o.tolist(), "output_eps_needed": st["eps_needed"],
                                    "output_frac_tight": st["frac_blocks_with_bstar_shortest"]}
        out["final_basis"] = [[int(B[i, j]) for j in range(d)] for i in range(d)]
        # every position (head block, full blocks and shrinking tail): the forced entry l_k^fin = log lambda_1(Q_k), the profile-level identity
        # nu_k = eps(Q_k) + Delta^vol_k, and the dual-weighted head identity l_1 - S/d = h(0) - sum y_k nu_k = h(0) - sum y_k (eps(Q_k) + Delta^vol_k)
        from latticelab.profile_floor import dual_certificate, floor_l1_float

        sizes = query_all["size"]
        nu = np.array([log_chat(n) + fin[k:k + n].mean() - fin[k] for k, n in enumerate(sizes)])
        eps_q = np.array(query_all["eps"])
        dvol_all = np.array([(float(fin[:k + n].sum()) - lv) / n for k, (n, lv) in enumerate(zip(sizes, query_all["log_vol_num"]))])
        yf = np.array([float(v) for v in dual_certificate(d, beta)[0]])
        h0 = floor_l1_float(d, beta)["l1_floor"]
        head_minus_floor = float(fin[0] - fin.sum() / d - h0)
        out["all_positions"] = {"n": d - 1, "max_forced_entry_gap": float(np.max(np.abs(np.array(query_all["log_lambda"]) - fin[:d - 1]))),
                                "max_nu_identity_gap": float(np.max(np.abs(nu - eps_q - dvol_all))),
                                "head_identity_gap": float(abs(head_minus_floor + float((yf * nu).sum()))), "head_minus_floor": head_minus_floor,
                                "minus_sum_y_eps_query": float(-(yf * eps_q).sum()), "minus_sum_y_dvol": float(-(yf * dvol_all).sum()),
                                "head_block_eps_query": float(eps_q[0]), "mean_eps_query_full": float(eps_q[:d - beta + 1].mean()),
                                "mean_eps_query_tail": float(eps_q[d - beta + 1:].mean()) if d - beta + 1 < d - 1 else float("nan"),
                                "mean_dvol": float(dvol_all.mean()), "eps_query": eps_q.tolist(), "dvol": dvol_all.tolist(), "nu": nu.tolist()}
    return out


def _corr_slope(x: np.ndarray, y: np.ndarray):
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(x, y)[0, 1]), float(np.polyfit(x, y, 1)[0])


def inheritance_stats(steps: List[Dict]) -> Dict:
    """Dependence along a tour, from the archived steps of `residual_census`: for consecutive steps kappa, kappa+1 of the same tour, the block
    queried at kappa+1 has ratio eps(P_{kappa+1}) = eps_neighbour recorded at step kappa, and its residual ratio is res_ratio recorded at step
    kappa+1.  Returns the correlation and regression slope of res_ratio on the block's own ratio, the lag-one correlation of the queried blocks'
    ratios, the conditional frequencies P[next block sub-GH | this block sub-GH] and P[next block sub-GH | this block not], and the mean gap
    in the two cases.  Descriptive statistics over one trajectory (the steps are dependent; use several seeds)."""
    x, y, e, g = [], [], [], []
    for s0, s1 in zip(steps, steps[1:]):
        if s1["tour"] == s0["tour"] and s1["kappa"] == s0["kappa"] + 1:
            x.append(s0["eps_neighbour"]); y.append(s1["res_ratio"]); e.append(s1["eps_neighbour"]); g.append(s1["gap"])
    x, y, e, g = (np.array(v) for v in (x, y, e, g))
    out = {"n_pairs": int(len(x))}
    if len(x) < 3:
        return out
    c1, s1_ = _corr_slope(x, y)
    c2, s2_ = _corr_slope(x, e)
    dense = x > 1e-12
    out.update({"corr_eps_res": c1, "slope_eps_res": s1_, "corr_eps_next": c2, "slope_eps_next": s2_, "n_dense": int(dense.sum()),
                "p_next_dense_given_dense": float((e[dense] > 1e-12).mean()) if dense.any() else float("nan"),
                "p_next_dense_given_not": float((e[~dense] > 1e-12).mean()) if (~dense).any() else float("nan"),
                "mean_gap_given_dense": float(g[dense].mean()) if dense.any() else float("nan"),
                "mean_gap_given_not": float(g[~dense].mean()) if (~dense).any() else float("nan")})
    return out


def control_dependence(rows: List[Dict], beta: int) -> Dict:
    """Dependence of the residual's ratio on the lattice's own ratio in the random control: correlation and slope of res_ratio on eps(P), the
    volume-only slope 1/(beta-1) that ||v|| alone would give (it enters GH_{beta-1}(P/v) through vol(P)/||v||), and the conditional means and
    frequencies of res_ratio on the sign of eps(P)."""
    x = np.array([r["eps_P"] for r in rows])
    y = np.array([r["res_ratio"] for r in rows])
    c, s = _corr_slope(x, y)
    dense = x > 0
    return {"n": int(len(x)), "corr_eps_res": c, "slope_eps_res": s, "volume_only_slope": 1.0 / (beta - 1), "n_dense": int(dense.sum()),
            "mean_res_given_dense": float(y[dense].mean()) if dense.any() else float("nan"),
            "mean_res_given_not": float(y[~dense].mean()) if (~dense).any() else float("nan"),
            "p_res_pos_given_dense": float((y[dense] > 0).mean()) if dense.any() else float("nan"),
            "p_res_pos_given_not": float((y[~dense] > 0).mean()) if (~dense).any() else float("nan")}


def residual_ratio_random(beta: int, n: int, q: int = 2 ** 16 + 1, seed0: int = 1000) -> Dict:
    """Control: on n random beta-dimensional q-ary lattices (qary(beta, beta//2, q, seed), LLL-reduced), insert the shortest vector at position 0
    and measure the residual's signed ratio res_ratio = log GH_{beta-1}(P/v) - log lambda_1(P/v) together with the lattice's own ratio eps(P) =
    log GH_beta(P) - log lambda_1(P).  lambda_1(P) is enumerated independently before the insertion and compared with the inserted vector's norm;
    the per-row `svp_mismatch` (in log) and its maximum are recorded (the insertion is the strict one of `latticelab.flag`, threshold 1 - 1e-12,
    so the mismatch is roundoff; a floating-point GSO enumeration, numerically exhaustive, not a formal certificate)."""
    from latticelab.lattices import lll, qary

    if beta < 3 or n < 1:
        raise ValueError("need beta >= 3 and n >= 1")
    rows = []
    for i in range(n):
        A = lll(qary(beta, beta // 2, q, seed=seed0 + i))
        bkz, B, M = _reducer(A)
        M.update_gso()
        log_vol = 0.5 * sum(math.log(M.get_r(j, j)) for j in range(beta))
        lam_P = _lambda1(M, 0, beta, log_vol)  # independent enumeration of the block minimum
        _insert(bkz, 0, beta, "bounded_lll")
        M.update_gso()
        r = [M.get_r(j, j) for j in range(beta)]
        log_lam = 0.5 * math.log(r[0])
        log_vol_res = log_vol - log_lam
        lam_res = _lambda1(M, 1, beta, log_vol_res)
        rows.append({"seed": seed0 + i, "eps_P": log_chat(beta) + log_vol / beta - log_lam,
                     "res_ratio": log_chat(beta - 1) + log_vol_res / (beta - 1) - math.log(lam_res), "svp_mismatch": abs(log_lam - math.log(lam_P))})
    e = np.array([x["eps_P"] for x in rows])
    rr = np.array([x["res_ratio"] for x in rows])
    return {"beta": beta, "n": n, "q": q, "rows": rows, "mean_eps_P": float(e.mean()), "sd_eps_P": float(e.std()),
            "mean_res_ratio": float(rr.mean()), "sd_res_ratio": float(rr.std()), "frac_res_ratio_positive": float((rr > 0).mean()),
            "se_mean_res_ratio": float(rr.std() / math.sqrt(n)), "max_svp_mismatch": float(max(x["svp_mismatch"] for x in rows))}


def gsa_tight_gh_shift(beta: int) -> float:
    """The deterministic term on a GSA-tight profile, (beta - 2)[f(beta) - f(beta - 1)], f(n) = log chat(n)/(n - 1)."""
    if beta < 3:
        raise ValueError("need beta >= 3")
    f = lambda n: log_chat(n) / (n - 1)
    return (beta - 2) * (f(beta) - f(beta - 1))


def tour_bookkeeping(A: IntegerMatrix, beta: int, tours: int, variant: str = "bounded_lll", change_tol: float = 1e-9, insert_tol: float | None = None,
                     ledger: bool = False) -> Dict:
    """Across tours: strict bounded forward tours from A (insertion whenever the block minimum is below (1 - insert_tol) ||b_k^*||^2 in squared
    norm; the default is the tie tolerance 1e-12 of `latticelab.flag`, the rule of `residual_census`; insert_tol = 0.01 reproduces fpylll's
    conventional delta-based threshold 0.99 ||b_k^*||^2, the rule of `latticelab.dual_census.strict_tours_census`), recording at every tour t and every position k (0-based, sizes min(beta, d-k)) the
    queried block Q_k^(t): its log GH, the forced entry l_k^(t) = log lambda_1(Q_k^(t)) (the flag theorem; the new b_k^*) and its ratio eps_k^(t) =
    log GH - l_k^(t); also whether the step at k itself inserted (|l_after - l_before| > change_tol at k's own query -- a diagnostic only, since
    earlier positions' completions can change b_k^* before k is queried).  Bookkeeping: tau_k := the last tour t >= 1 with |l_k^(t) - l_k^(t-1)| >
    change_tol (0 if the entry never moved); then eps_k^(T) = eps_k^(tau_k) + [log GH_k^(T) - log GH_k^(tau_k)] - [l_k^(T) - l_k^(tau_k)] exactly,
    the last term being the entry's drift since tau_k (zero when change_tol is at roundoff, at most change_tol per tour otherwise -- a coarse
    tolerance such as 0.005, half of fpylll's 0.99 slack in log, treats sub-slack refinements as retention); `identity_gap` reports the roundoff.
    The final ratio is the queried ratio at the tour in which the entry last moved plus the block's GH drift since, minus the entry drift.
    Returns the per-tour arrays, the last-change tours, the three parts of every final ratio and their y-weighted sums (the mass Sigma y eps^(T)
    = Sigma y eps^(tau) + Sigma y GH drift - Sigma y entry drift), the head-floor identity at the end, and the first clean tour (no entry moved
    by more than change_tol in a whole tour).
    With `ledger`, the exact block minima of the basis after every tour (`profile.block_gh_ratios`) give the per-tour mass Sigma y eps_k(B^(t)),
    the created non-tightness R^(t) = Sigma y r_k and, with the recorded queried ratios and profiles, the denominator drift D_k^(t+1) :=
    eps(Q_k^(t+1)) - eps_k(B^(t)) and the window drift Delta^vol_k^(t+1) = (P_{k+n}^(t+1) - P_{k+n}^(t))/n, so that
        mass^(t+1) - mass^(t) = R^(t+1) + Sigma_k y_k (D_k^(t+1) + Delta^vol_k^(t+1))     exactly
    (the within-tour corollary combined with the per-basis identity): the sub-GH mass of the output grows by the non-tightness created in the
    tour plus two drifts, and the converged mass is the input's mass plus the cumulative created non-tightness plus the cumulative drifts."""
    from latticelab.profile_floor import dual_certificate, floor_l1_float

    d = A.nrows
    if variant == "conveyor":
        raise ValueError("the bookkeeping needs a block-supported completion")
    if beta < 3 or d < beta + 2 or tours < 1:
        raise ValueError("need beta >= 3, d >= beta + 2, tours >= 1")
    if insert_tol is not None and not (0 <= insert_tol < 1):
        raise ValueError("need 0 <= insert_tol < 1")
    bkz, B, M = _reducer(A)
    if insert_tol is not None:
        bkz.tie_tol = insert_tol
    sizes = [min(beta, d - k) for k in range(d - 1)]
    log_gh = np.zeros((tours + 1, d - 1))
    ell = np.zeros((tours + 1, d - 1))
    inserted = np.zeros((tours + 1, d - 1), dtype=bool)
    M.update_gso()
    p0 = np.array([0.5 * math.log(M.get_r(j, j)) for j in range(d)])
    ell[0] = p0[:d - 1]
    log_gh[0] = [log_chat(n) + p0[k:k + n].mean() for k, n in enumerate(sizes)]  # the input basis's blocks
    S_total = float(p0.sum())
    yf = np.array([float(v) for v in dual_certificate(d, beta)[0]])
    eps_B, r_B = [], []  # ledger: exact ratios and non-tightness of the basis after each tour (index 0: the input)
    if ledger:
        from latticelab.profile import block_gh_ratios

        def basis_stats():
            st = block_gh_ratios(B, beta)
            return (np.array([row["log_gh_over_lambda1"] for row in st["blocks"]]), np.array([math.log(row["b_star_over_lambda1"]) for row in st["blocks"]]))

        e0, r0 = basis_stats(); eps_B.append(e0); r_B.append(r0)
    for T in range(1, tours + 1):
        for k in range(d - 1):
            n = sizes[k]
            M.update_gso()
            l_before = 0.5 * math.log(M.get_r(k, k))
            _insert(bkz, k, n, variant)
            M.update_gso()
            l_after = 0.5 * math.log(M.get_r(k, k))
            lv_block = 0.5 * sum(math.log(M.get_r(j, j)) for j in range(k, k + n))  # unchanged by the insertion
            log_gh[T, k] = log_chat(n) + lv_block / n
            ell[T, k] = l_after
            inserted[T, k] = abs(l_after - l_before) > change_tol
        if ledger:
            eT, rT = basis_stats(); eps_B.append(eT); r_B.append(rT)
    entry_moved = np.abs(np.diff(ell, axis=0)) > change_tol  # entry_moved[t-1, k]: l_k^(t) != l_k^(t-1)
    clean_tour = next((t for t in range(1, tours + 1) if not entry_moved[t - 1].any()), None)
    eps = log_gh - ell  # eps[t, k] for t >= 1 is the ratio of the block queried at tour t; eps[0] is the input basis's profile-level ratio
    last_change = np.array([max([t for t in range(1, tours + 1) if entry_moved[t - 1, k]], default=0) for k in range(d - 1)])
    eps_at_last = np.array([eps[last_change[k], k] for k in range(d - 1)])
    drift_since = np.array([log_gh[tours, k] - log_gh[last_change[k], k] for k in range(d - 1)])
    entry_drift = np.array([ell[tours, k] - ell[last_change[k], k] for k in range(d - 1)])
    identity_gap = float(np.max(np.abs(eps[tours] - eps_at_last - drift_since + entry_drift)))
    retained_gap = float(np.max(np.abs(entry_drift)))
    M.update_gso()
    fin = np.array([0.5 * math.log(M.get_r(j, j)) for j in range(d)])
    h0 = floor_l1_float(d, beta)["l1_floor"]
    nu_fin = np.array([log_chat(n) + fin[k:k + n].mean() - fin[k] for k, n in enumerate(sizes)])
    out_ledger = None
    if ledger:
        def P(t, j):  # prefix volume after tour t: P_j = sum_{i < j} l_i^(t) for j <= d-1, and P_d = S
            return S_total if j >= d else float(ell[t, :j].sum())

        masses = [float((yf * e).sum()) for e in eps_B]
        Rs = [float((yf * r).sum()) for r in r_B]
        D_terms, V_terms, gaps = [], [], []
        for t in range(1, tours + 1):
            D = eps[t] - eps_B[t - 1]  # eps(Q_k^(t)) - eps_k(B^(t-1))
            V = np.array([(P(t, k + n) - P(t - 1, k + n)) / n for k, n in enumerate(sizes)])
            D_terms.append(float((yf * D).sum())); V_terms.append(float((yf * V).sum()))
            gaps.append(abs(masses[t] - masses[t - 1] - Rs[t] - D_terms[-1] - V_terms[-1]))
        out_ledger = {"mass_per_tour": masses, "R_per_tour": Rs, "sum_yD_per_tour": D_terms, "sum_yDvol_per_tour": V_terms, "max_ledger_gap": float(max(gaps)),
                      "mass_change": masses[-1] - masses[0], "cumulative_R": float(sum(Rs[1:])), "cumulative_sum_yD": float(sum(D_terms)),
                      "cumulative_sum_yDvol": float(sum(V_terms)), "mass_input": masses[0], "mass_final": masses[-1], "R_final": Rs[-1]}
    return {"d": d, "beta": beta, "tours": tours, "sizes": sizes, "insert_tol": bkz.tie_tol, "change_tol": change_tol, "clean_tour": clean_tour, "log_gh": log_gh.tolist(), "ell": ell.tolist(),
            "inserted": inserted.tolist(), "entry_moved": entry_moved.tolist(), "last_change": last_change.tolist(), "eps_final": eps[tours].tolist(),
            "eps_at_last_change": eps_at_last.tolist(),
            "drift_since_last_change": drift_since.tolist(), "entry_drift_since_last_change": entry_drift.tolist(), "identity_gap": identity_gap,
            "max_entry_drift": retained_gap,
            "mass_final": float((yf * eps[tours]).sum()), "mass_at_last_change": float((yf * eps_at_last).sum()), "mass_drift": float((yf * drift_since).sum()),
            "mass_entry_drift": float((yf * entry_drift).sum()),
            "mass_final_positive_part": float((yf * np.maximum(eps[tours], 0)).sum()), "frac_positions_positive_final": float((eps[tours] > 1e-12).mean()),
            "frac_positions_positive_at_last_change": float((eps_at_last > 1e-12).mean()), "mean_eps_at_last_change": float(eps_at_last.mean()),
            "mean_drift_since_last_change": float(drift_since.mean()), "insertions_per_tour": inserted[1:].sum(axis=1).tolist(),
            "entries_moved_per_tour": entry_moved.sum(axis=1).tolist(),
            "head_minus_floor": float(fin[0] - fin.sum() / d - h0), "head_identity_gap": float(abs(fin[0] - fin.sum() / d - h0 + (yf * nu_fin).sum())),
            "nu_final_equals_eps_final_gap": float(np.max(np.abs(nu_fin - eps[tours]))) if clean_tour is not None and clean_tour <= tours else float("nan"),
            "ledger": out_ledger}


SQRT3_2 = math.sqrt(3) / 2


def thinning_factor(u):
    """g(u) = P[a residual point of norm u ||v|| has all its lifts at least as long as v] for a uniform offset tau in (-||v||/2, ||v||/2]:
    0 for u < sqrt(3)/2, 1 - 2 sqrt(1 - u^2) for sqrt(3)/2 <= u < 1, 1 for u >= 1.  Vectorised."""
    u = np.asarray(u, dtype=float)
    return np.where(u >= 1.0, 1.0, np.where(u < SQRT3_2, 0.0, 1.0 - 2.0 * np.sqrt(np.clip(1.0 - u * u, 0.0, None))))


def residual_gh_log_offset(beta: int) -> float:
    """A(beta) = log(GH_{beta-1}(P/v) / ||v||) when ||v|| = GH_beta(P): L(beta-1) - beta L(beta)/(beta-1).  In general
    log GH_{beta-1}(P/v) - log ||v|| = A(beta) + beta eps(P)/(beta-1)."""
    if beta < 3:
        raise ValueError("need beta >= 3")
    return log_chat(beta - 1) - beta * log_chat(beta) / (beta - 1)


def residual_ratio_cap(beta: int, eps_P: float) -> float:
    """The deterministic upper bound on res_ratio from lambda_1(P/v) >= (sqrt 3/2) ||v||: A(beta) + beta eps_P/(beta-1) - log(sqrt 3/2)."""
    return residual_gh_log_offset(beta) + beta * eps_P / (beta - 1) - math.log(SQRT3_2)


def _thinned_survival_grid(beta: int, eps_P: float, grid: int, thin: bool):
    """The grid t = log u and the survival S(t) = P[log U > t] of the thinned (or unthinned) law given eps_P, with log c.  The grid runs from
the lower end of the support (log(sqrt 3/2) when thinned; the point where N_0 = 1e-8 otherwise) to max(the point where N_0 = 60, t_lo + 1):
the upper tail beyond it (survival <= e^{-60}) is dropped."""
    if grid < 2 or not math.isfinite(eps_P):
        raise ValueError("need grid >= 2 and a finite eps_P")
    log_c = residual_gh_log_offset(beta) + beta * eps_P / (beta - 1)
    t_lo = math.log(SQRT3_2) if thin else log_c + math.log(2e-8) / (beta - 1)
    t_hi = max(log_c + math.log(120.0) / (beta - 1), t_lo + 1.0)
    t = np.linspace(t_lo, t_hi, grid)
    u = np.exp(t)
    g = thinning_factor(u) if thin else np.ones_like(u)
    dN0_dt = (beta - 1) * np.exp((beta - 1) * (t - log_c)) / 2.0  # d/dt of (u/c)^{beta-1}/2
    integrand = g * dN0_dt
    Lam = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))])
    return t, np.exp(-Lam), log_c


def thinned_pit(beta: int, eps_P: float, u: float, grid: int = 20000, thin: bool = True) -> float:
    """The probability-integral transform P[U' <= u | eps_P] of an observed U = lambda_1(P/v)/||v|| under the thinned law (grid quadrature) or
    the unthinned one (analytic: 1 - exp(-(u/c)^{beta-1}/2)): uniform on [0, 1] over a sample if the law is right."""
    if not (u > 0 and math.isfinite(u)) or not math.isfinite(eps_P):
        raise ValueError("need a finite u > 0 and a finite eps_P")
    if not thin:
        log_c = residual_gh_log_offset(beta) + beta * eps_P / (beta - 1)
        return 1.0 - math.exp(-0.5 * math.exp((beta - 1) * (math.log(u) - log_c)))
    t, S, _ = _thinned_survival_grid(beta, eps_P, grid, thin)
    lu = math.log(u)
    if lu <= t[0]:
        return 0.0
    if lu >= t[-1]:
        return 1.0
    return 1.0 - float(np.interp(lu, t, S))


def thinned_residual_law(beta: int, eps_P: float = 0.0, grid: int = 40000, thin: bool = True) -> Dict:
    """The law of U = lambda_1(P/v)/||v|| in the thinned Poisson surrogate given eps(P): sign pairs of P/v of norm <= u ||v|| Poisson with mean
    N_0(u) = (u/c)^{beta-1}/2, c = exp(A(beta) + beta eps_P/(beta-1)), thinned by g(u) (thin=True) or not (thin=False: the unthinned Poisson
    sign-pair model, evaluated analytically -- U/c = (2X)^{1/(beta-1)} with X ~ Exp(1), so E[res_ratio] = -(log 2 - gamma)/(beta-1), sd =
    pi/(sqrt 6 (beta-1)) and P[res_ratio > 0] = 1 - e^{-1/2}).  Returns E[res_ratio], its standard deviation, P[res_ratio > 0] = P[U < c],
    E[log U], c and the deterministic cap; res_ratio = log c - log U.  Thinned case: quadrature in t = log u on the grid of
    `_thinned_survival_grid`, using E[phi(log U)] = phi(t_lo) + int phi'(t) P[log U > t] dt (exact on the support [log(sqrt 3/2), inf); the tail
    beyond survival e^{-60} is dropped)."""
    if beta < 3:
        raise ValueError("need beta >= 3")
    if grid < 1000:
        raise ValueError("grid too coarse")
    if not math.isfinite(eps_P):
        raise ValueError("need a finite eps_P")
    m = beta - 1
    log_c = residual_gh_log_offset(beta) + beta * eps_P / (beta - 1)
    c = math.exp(log_c)
    if not thin:
        return {"beta": beta, "eps_P": eps_P, "c": c, "log_c": log_c, "E_res_ratio": -(math.log(2.0) - float(np.euler_gamma)) / m,
                "sd_res_ratio": math.pi / (math.sqrt(6.0) * m), "P_res_ratio_positive": 1.0 - math.exp(-0.5),
                "E_log_U": log_c + (math.log(2.0) - float(np.euler_gamma)) / m, "cap": float("inf"), "thin": False}
    t, S, _ = _thinned_survival_grid(beta, eps_P, grid, True)
    t_lo, t_hi = float(t[0]), float(t[-1])
    e_log = t_lo + float(np.trapezoid(S, t))
    e_log2 = t_lo ** 2 + float(np.trapezoid(2.0 * t * S, t))
    var_log = max(0.0, e_log2 - e_log ** 2)
    if log_c <= t_lo:
        p_pos = 0.0
    elif log_c >= t_hi:
        p_pos = 1.0
    else:
        p_pos = 1.0 - float(np.interp(log_c, t, S))
    return {"beta": beta, "eps_P": eps_P, "c": c, "log_c": log_c, "E_res_ratio": log_c - e_log, "sd_res_ratio": math.sqrt(var_log),
            "P_res_ratio_positive": p_pos, "E_log_U": e_log, "cap": residual_ratio_cap(beta, eps_P), "thin": True}


def _sign_pair_eps_quadrature(beta: int, n_quad: int):
    """Probability-midpoint quadrature of eps(P) = -log(2X)/beta, X ~ Exp(1) (the sign-pair Poisson law of a block's own ratio)."""
    p = (np.arange(n_quad) + 0.5) / n_quad
    return -np.log(2.0 * (-np.log1p(-p))) / beta


def thinned_residual_model(beta: int, eps_values=None, n_quad: int = 200, thin: bool = True, grid: int = 20000) -> Dict:
    """The thinned residual model integrated over eps(P): over the sign-pair law (eps_values=None; n_quad midpoints) or over the given values
    (e.g. the measured eps(P) of a control, equally weighted).  Returns E[res_ratio], P[res_ratio > 0], the population regression slope of
    res_ratio on eps(P) (= Cov(E[res|eps], eps)/Var eps), the correlation (with the within-eps variance included), and the conditional means
    and frequencies on the sign of eps(P); with per-value arrays for scoring rows individually.  A single eps value is accepted (a one-row
    control): the slope and the correlation are then undefined and reported as NaN."""
    es = _sign_pair_eps_quadrature(beta, n_quad) if eps_values is None else np.atleast_1d(np.asarray(eps_values, dtype=float))
    if es.size < 1:
        raise ValueError("need at least one eps value")
    laws = [thinned_residual_law(beta, float(e), grid=grid, thin=thin) for e in es]
    m = np.array([l["E_res_ratio"] for l in laws])
    v = np.array([l["sd_res_ratio"] ** 2 for l in laws])
    pp = np.array([l["P_res_ratio_positive"] for l in laws])
    E_eps, E_res = float(es.mean()), float(m.mean())
    var_eps = float(((es - E_eps) ** 2).mean())
    cov = float(((es - E_eps) * (m - E_res)).mean())
    var_res = float((v + (m - E_res) ** 2).mean())
    dense = es > 0
    out = {"beta": beta, "thin": thin, "n_eps": int(es.size), "E_eps": E_eps, "E_res_ratio": E_res, "sd_res_ratio": math.sqrt(var_res),
           "P_res_ratio_positive": float(pp.mean()), "slope_res_on_eps": cov / var_eps if var_eps > 0 else float("nan"),
           "corr_res_eps": cov / math.sqrt(var_eps * var_res) if var_eps > 0 and var_res > 0 else float("nan"),
           "P_dense": float(dense.mean()), "per_eps": {"eps": es.tolist(), "E_res": m.tolist(), "P_pos": pp.tolist(), "sd_res": np.sqrt(v).tolist()}}
    for name, mask in (("dense", dense), ("sparse", ~dense)):
        out[f"E_res_given_{name}"] = float(m[mask].mean()) if mask.any() else float("nan")
        out[f"P_pos_given_{name}"] = float(pp[mask].mean()) if mask.any() else float("nan")
    return out


def compare_control_with_model(rows: List[Dict], beta: int, grid: int = 20000, n_boot: int = 300, seed: int = 0) -> Dict:
    """Score the thinned surrogate against a q-ary control (rows of `residual_ratio_random`): the model is integrated over the rows' own eps(P)
    values, so the comparison is conditional on the measured block ratios.  Uncertainties: the mean discrepancy d_i = y_i - m(eps_i) with its
    empirical (ddof 1) and null (sum of model conditional variances) standard errors and z; the sign frequency against the Poisson-binomial
    null; row-bootstrap standard errors (n_boot resamples) for the slope, the correlation and the conditional means and frequencies.  The
    deterministic cap is checked on every row.  Distribution level: the probability-integral transform of every row's U = lambda_1(P/v)/||v||
    under its own eps(P) -- pooled Kolmogorov-Smirnov against the uniform law (thinned and unthinned), its Spearman correlation with eps(P)
    (independence), and KS within eps-terciles (conditional calibration); a pooled uniform PIT is necessary, not sufficient."""
    from scipy.stats import kstest, spearmanr

    if not rows:
        raise ValueError("need at least one control row")
    e = np.array([r["eps_P"] for r in rows])
    y = np.array([r["res_ratio"] for r in rows])
    n = len(rows)
    mod = thinned_residual_model(beta, eps_values=e, grid=grid)
    unthinned = thinned_residual_model(beta, eps_values=e, grid=grid, thin=False)
    dp = control_dependence(rows, beta)
    caps = np.array([residual_ratio_cap(beta, float(x)) for x in e])
    m_i = np.array(mod["per_eps"]["E_res"])
    v_i = np.array(mod["per_eps"]["sd_res"]) ** 2
    p_i = np.array(mod["per_eps"]["P_pos"])
    dres = y - m_i
    se_emp = float(dres.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    se_null = float(math.sqrt(v_i.sum()) / n)
    se_sign_null = float(math.sqrt((p_i * (1 - p_i)).sum()) / n)
    # row bootstrap of the dependence statistics
    rng = np.random.default_rng(seed)
    boot = {"slope": [], "corr": [], "E_dense": [], "E_sparse": [], "P_dense": [], "P_sparse": []}
    if n >= 3:
        for _ in range(n_boot):
            idx = rng.integers(n, size=n)
            eb, yb = e[idx], y[idx]
            cb, sb = _corr_slope(eb, yb)
            dense = eb > 0
            boot["slope"].append(sb); boot["corr"].append(cb)
            boot["E_dense"].append(yb[dense].mean() if dense.any() else float("nan")); boot["E_sparse"].append(yb[~dense].mean() if (~dense).any() else float("nan"))
            boot["P_dense"].append((yb[dense] > 0).mean() if dense.any() else float("nan")); boot["P_sparse"].append((yb[~dense] > 0).mean() if (~dense).any() else float("nan"))
    boot_se = {k: float(np.nanstd(np.array(v), ddof=1)) if len(v) > 1 else float("nan") for k, v in boot.items()}
    # U_i = c(eps_i) e^{-res_i}; its PIT under each law
    logU = np.array([residual_gh_log_offset(beta) + beta * ei / (beta - 1) - yi for ei, yi in zip(e, y)])
    pit = np.array([thinned_pit(beta, float(ei), math.exp(float(lu)), grid=grid) for ei, lu in zip(e, logU)])
    pit_un = np.array([thinned_pit(beta, float(ei), math.exp(float(lu)), grid=grid, thin=False) for ei, lu in zip(e, logU)])
    ks = kstest(pit, "uniform") if n >= 2 else None
    ks_un = kstest(pit_un, "uniform") if n >= 2 else None
    sp = spearmanr(pit, e) if n >= 3 else None
    strata = []
    if n >= 9:
        order = np.argsort(e)
        for part in np.array_split(order, 3):
            k = kstest(pit[part], "uniform")
            strata.append({"n": int(part.size), "eps_min": float(e[part].min()), "eps_max": float(e[part].max()), "D": float(k.statistic), "p": float(k.pvalue),
                           "pit_mean": float(pit[part].mean())})
    return {"beta": beta, "n": n,
            "measured": {"E_res_ratio": float(y.mean()), "se_E_res_ratio": float(y.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"), "sd_res_ratio": float(y.std(ddof=1)) if n > 1 else float("nan"),
                         "P_res_ratio_positive": float((y > 0).mean()), "slope_res_on_eps": dp["slope_eps_res"], "corr_res_eps": dp["corr_eps_res"],
                         "E_res_given_dense": dp["mean_res_given_dense"], "E_res_given_sparse": dp["mean_res_given_not"],
                         "P_pos_given_dense": dp["p_res_pos_given_dense"], "P_pos_given_sparse": dp["p_res_pos_given_not"],
                         "bootstrap_se": {"slope_res_on_eps": boot_se["slope"], "corr_res_eps": boot_se["corr"], "E_res_given_dense": boot_se["E_dense"],
                                          "E_res_given_sparse": boot_se["E_sparse"], "P_pos_given_dense": boot_se["P_dense"], "P_pos_given_sparse": boot_se["P_sparse"]}},
            "model": {k: mod[k] for k in ("E_res_ratio", "sd_res_ratio", "P_res_ratio_positive", "slope_res_on_eps", "corr_res_eps", "E_res_given_dense",
                                          "E_res_given_sparse", "P_pos_given_dense", "P_pos_given_sparse")},
            "unthinned": {k: unthinned[k] for k in ("E_res_ratio", "P_res_ratio_positive", "slope_res_on_eps")},
            "mean_discrepancy": float(dres.mean()), "se_discrepancy_empirical": se_emp, "se_discrepancy_null": se_null,
            "bias_z": float(dres.mean() / se_null) if se_null > 0 else float("nan"),
            "sign_z": float(((y > 0).mean() - p_i.mean()) / se_sign_null) if se_sign_null > 0 else float("nan"),
            "cap_holds_all": bool(np.all(y <= caps + 1e-9)), "min_cap_slack": float(np.min(caps - y)),
            "max_svp_mismatch": float(max(r.get("svp_mismatch", float("nan")) for r in rows)),
            "pit_mean": float(pit.mean()), "ks_thinned": {"D": float(ks.statistic), "p": float(ks.pvalue)} if ks is not None else None,
            "ks_unthinned": {"D": float(ks_un.statistic), "p": float(ks_un.pvalue)} if ks_un is not None else None,
            "pit_eps_spearman": {"rho": float(sp.statistic), "p": float(sp.pvalue)} if sp is not None else None,
            "ks_thinned_by_eps_tercile": strata}


def undercut_law(beta: int, u_star: float, h: float, c: float, grid: int = 4000) -> Dict:
    """The extension's undercut of the residual minimum, per step, all lengths in units of ||v||: u_star = lambda_1(P/v)/||v||, h = hbar =
    ||b_{kappa+beta}^*||/||v|| (the NORMALISED layer height), c = GH_{beta-1}(P/v)/||v||.  count(rho) = sum_{j>=1} ((rho^2 - j^2 h^2)^+ / c^2)^{(beta-1)/2}
    is the modelled mean number of sign pairs of Q/v outside P/v with norm <= rho ||v|| (independent-layer Poisson surrogate); P[gap > 0] =
    1 - exp(-count(u_star)) and E[gap] = int_h^{u_star} (1 - exp(-count(rho))) drho/rho."""
    if beta < 3 or not all(math.isfinite(x) for x in (u_star, h, c)) or u_star <= 0 or h <= 0 or c <= 0 or grid < 2:
        raise ValueError("need beta >= 3, finite positive u_star, h, c and grid >= 2")

    def count(rho):
        rho = np.asarray(rho, dtype=float)
        tot = np.zeros_like(rho)
        j = 1
        while j * h < float(np.max(rho)):
            tot += (np.clip(rho ** 2 - (j * h) ** 2, 0.0, None) / c ** 2) ** ((beta - 1) / 2)
            j += 1
        return tot

    p_pos = 1.0 - math.exp(-float(count(u_star)))
    if u_star <= h:
        return {"P_gap_positive": 0.0, "E_gap": 0.0, "count_at_u_star": 0.0}
    rho = np.linspace(h, u_star, grid)
    e_gap = float(np.trapezoid((1.0 - np.exp(-count(rho))) / rho, rho))
    return {"P_gap_positive": p_pos, "E_gap": e_gap, "count_at_u_star": float(count(u_star))}


def gap_model_check(steps: List[Dict], beta: int) -> Dict:
    """Score the undercut law against census steps that record l_kappa and l_kappa_plus_beta: the mean predicted P[gap > 0] and E[gap] against
    the observed frequency and mean, and a calibration by terciles of the predicted probability.  The steps of one trajectory are serially
    dependent, so the standard errors are naive iid ones (labelled so); cluster by seed and tour for an honest error."""
    rows = [s for s in steps if "l_kappa" in s and "l_kappa_plus_beta" in s]
    if not rows:
        return {"n": 0}
    pred_p, pred_g, obs_pos, obs_g = [], [], [], []
    for s in rows:
        v = math.exp(s["l_kappa"])
        u_star = s["lambda1_residual"] / v
        h = math.exp(s["l_kappa_plus_beta"]) / v
        c = u_star * math.exp(s["res_ratio"])  # GH_res = lambda_1(P/v) e^{res_ratio}
        law = undercut_law(beta, u_star, h, c)
        pred_p.append(law["P_gap_positive"]); pred_g.append(law["E_gap"]); obs_pos.append(float(s["gap"] > 1e-9)); obs_g.append(s["gap"])
    pred_p, pred_g, obs_pos, obs_g = (np.array(x) for x in (pred_p, pred_g, obs_pos, obs_g))
    order = np.argsort(pred_p)
    terciles = []
    for part in np.array_split(order, min(3, len(rows))):
        if part.size:
            terciles.append({"n": int(part.size), "mean_predicted_P": float(pred_p[part].mean()), "observed_frequency": float(obs_pos[part].mean())})
    n = len(rows)
    return {"n": n, "mean_predicted_P_gap_positive": float(pred_p.mean()), "observed_frac_gap_positive": float(obs_pos.mean()),
            "se_observed_frac_naive_iid": float(math.sqrt(obs_pos.mean() * (1 - obs_pos.mean()) / n)),
            "mean_predicted_E_gap": float(pred_g.mean()), "observed_mean_gap": float(obs_g.mean()),
            "se_observed_mean_gap_naive_iid": float(obs_g.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"), "calibration_terciles": terciles,
            "mean_u_star": float(np.mean([s["lambda1_residual"] / math.exp(s["l_kappa"]) for s in rows])),
            "mean_h_over_v": float(np.mean([math.exp(s["l_kappa_plus_beta"] - s["l_kappa"]) for s in rows]))}


def _sample_thinned_U(beta: int, eps_P: float, rng, grid: int = 4000) -> float:
    """One draw of U = lambda_1(P/v)/||v|| from the thinned law given eps(P), by inverting the cumulative intensity Lam(t) at an Exp(1) draw
    (P[log U > t] = exp(-Lam(t))); the log-grid is extended until Lam exceeds the draw, so no endpoint atom is created."""
    if grid < 2 or not math.isfinite(eps_P):
        raise ValueError("need grid >= 2 and a finite eps_P")
    log_c = residual_gh_log_offset(beta) + beta * eps_P / (beta - 1)
    t_lo = math.log(SQRT3_2)
    t_hi = max(log_c + math.log(120.0) / (beta - 1), t_lo + 1.0)
    target = rng.exponential()
    for _ in range(8):
        t = np.linspace(t_lo, t_hi, grid)
        integrand = thinning_factor(np.exp(t)) * (beta - 1) * np.exp((beta - 1) * (t - log_c)) / 2.0
        Lam = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))])
        if target < Lam[-1]:
            return math.exp(float(np.interp(target, Lam, t)))
        t_hi += 1.0  # extend the range (Lam grows like e^{(beta-1)t}: one unit suffices in practice)
    return math.exp(t_hi)


def _sample_layer_min(beta: int, u_star: float, h: float, c: float, rng, grid: int = 2000) -> float:
    """One draw of the layer minimum (the shortest point of Q/v outside P/v, in units of ||v||), truncated at u_star: returns u_star when no
    layer point falls below it (a genuine atom of the truncated minimum).  h is the normalised height; the count function is that of
    `undercut_law`."""
    if grid < 2 or not all(math.isfinite(x) for x in (u_star, h, c)) or u_star <= 0 or h <= 0 or c <= 0:
        raise ValueError("need grid >= 2 and finite positive u_star, h, c")
    if u_star <= h:
        return u_star
    rho = np.linspace(h, u_star, grid)
    tot = np.zeros_like(rho)
    j = 1
    while j * h < u_star:
        tot += (np.clip(rho ** 2 - (j * h) ** 2, 0.0, None) / c ** 2) ** ((beta - 1) / 2)
        j += 1
    target = rng.exponential()
    if target >= tot[-1]:
        return u_star
    return float(np.interp(target, tot, rho))


def gh_shift_from_profile(beta: int, h_bar: float, c: float) -> float:
    """The exact identity gh_shift = L(beta) - ((beta-1)/beta) L(beta-1) + (log h_bar - log c)/beta, from vol(Q/v) = vol(P/v) ||b_{kappa+beta}^*||
    with h_bar = ||b_{kappa+beta}^*||/||v|| and c = GH_{beta-1}(P/v)/||v||."""
    if beta < 3 or not (h_bar > 0 and c > 0 and math.isfinite(h_bar) and math.isfinite(c)):
        raise ValueError("need beta >= 3 and finite positive h_bar, c")
    return log_chat(beta) - (beta - 1) / beta * log_chat(beta - 1) + (math.log(h_bar) - math.log(c)) / beta


def census_profile_pairs(steps: List[Dict]) -> np.ndarray:
    """The (h_bar, gh_shift) pairs of census steps that record l_kappa and l_kappa_plus_beta: h_bar = exp(l_{kappa+beta} - l_kappa) and the recorded
    gh_shift, for checking `gh_shift_from_profile` (with c = u* e^{res_ratio}) and for `census_h_values`."""
    rows = [s for s in steps if "l_kappa" in s and "l_kappa_plus_beta" in s]
    return np.array([[math.exp(s["l_kappa_plus_beta"] - s["l_kappa"]), s["gh_shift"]] for s in rows]).reshape(-1, 2)


def census_h_values(steps: List[Dict]) -> np.ndarray:
    """The normalised heights h_bar of census steps, for `thinned_world_chain(h_values=...)`."""
    return census_profile_pairs(steps)[:, 0]


def _expected_gap_given_eps(beta: int, eps: float, h_bar: float, n_quad: int = 40, grid: int = 4000) -> float:
    """E[gap | eps, h_bar] = int E[gap | u*, h_bar, c(eps)] dF_U(u* | eps): probability-midpoint quadrature over the thinned law of U."""
    t, S, log_c = _thinned_survival_grid(beta, eps, grid, True)
    F = 1.0 - S
    p = (np.arange(n_quad) + 0.5) / n_quad
    us = np.exp(np.interp(p, F, t))  # quantiles of log U (F is non-decreasing in t)
    c = math.exp(log_c)
    return float(np.mean([undercut_law(beta, float(u), h_bar, c, grid=1000)["E_gap"] for u in us]))


def world_fixed_point(beta: int, h_bar: float, lo: float = -0.5, hi: float = 0.5, tol: float = 1e-5) -> Dict:
    """The fixed point of the conditional means of the thinned-residual world at a fixed normalised height: solve e = H(e) + M(e) + G(e) with
    H(e) = gh_shift_from_profile(beta, h_bar, c(e)), M(e) = E[res_ratio | e] (thinned law), G(e) = E[gap | e, h_bar] (undercut law over the law of
    U), by bisection on [lo, hi] (the map e -> H + M + G - e is continuous; returns NaN if it does not change sign on the interval)."""
    if beta < 3 or not (h_bar > 0 and math.isfinite(h_bar)) or not lo < hi:
        raise ValueError("need beta >= 3, h_bar > 0 finite, lo < hi")

    def F(e):
        c = math.exp(residual_gh_log_offset(beta) + beta * e / (beta - 1))
        return gh_shift_from_profile(beta, h_bar, c) + thinned_residual_law(beta, e, grid=8000)["E_res_ratio"] + _expected_gap_given_eps(beta, e, h_bar) - e

    flo, fhi = F(lo), F(hi)
    if flo * fhi > 0:
        return {"beta": beta, "h_bar": h_bar, "fixed_point": float("nan"), "F_lo": flo, "F_hi": fhi}
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        fm = F(mid)
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid
    e = 0.5 * (lo + hi)
    c = math.exp(residual_gh_log_offset(beta) + beta * e / (beta - 1))
    return {"beta": beta, "h_bar": h_bar, "fixed_point": e, "gh_shift_at_fixed_point": gh_shift_from_profile(beta, h_bar, c),
            "E_res_at_fixed_point": thinned_residual_law(beta, e, grid=8000)["E_res_ratio"], "E_gap_at_fixed_point": _expected_gap_given_eps(beta, e, h_bar)}


def thinned_world_chain(beta: int, n: int, h_bar: float | None = None, seed: int = 0, burn: int = 200, eps0: float = 0.0, h_values=None,
                        solve_fixed_point: bool = True) -> Dict:
    """The thinned-residual world: iterate eps_{k+1} = gh_shift_k + res_ratio_k + gap_k for n steps after `burn` burn-in steps.  At each step
    c_k = c(eps_k); the normalised height is the fixed h_bar or one value drawn uniformly from `h_values` (e.g. `census_h_values`); gh_shift_k =
    gh_shift_from_profile(beta, h_k, c_k) (the exact identity, so the three profile quantities are consistent); U_k from the thinned law given
    eps_k (res_ratio_k = log c_k - log U_k, u*_k = U_k); rho_k the layer minimum at (u*_k, h_k, c_k); gap_k = log u*_k - log min(u*_k, rho_k).
    Returns the stationary summary (mean and sd of eps, P[eps > 0], mean res_ratio, gap and gh_shift, P[gap > 0], the lag-one correlation of
    eps), the post-hoc affine balance (mean gh_shift + E[res | 0] + mean gap)/(1 - slope) under that name, and, if `solve_fixed_point`, the
    solved fixed point of the conditional means at the mean height (`world_fixed_point`)."""
    if beta < 3 or n < 1 or burn < 0:
        raise ValueError("need beta >= 3, n >= 1, burn >= 0")
    hv = None if h_values is None else np.asarray(h_values, dtype=float).reshape(-1)
    if hv is None:
        if h_bar is None or not (h_bar > 0 and math.isfinite(h_bar)):
            raise ValueError("fixed-height mode needs a finite h_bar > 0 (or supply h_values)")
    elif hv.size == 0 or not np.all(np.isfinite(hv)) or np.any(hv <= 0):
        raise ValueError("h_values must be a nonempty array of finite positive normalised heights")
    rng = np.random.default_rng(seed)
    eps = eps0
    es, rs, gs, hs = [], [], [], []
    for k in range(burn + n):
        h = float(hv[rng.integers(hv.size)]) if hv is not None else h_bar
        log_c = residual_gh_log_offset(beta) + beta * eps / (beta - 1)
        c = math.exp(log_c)
        gh = gh_shift_from_profile(beta, h, c)
        U = _sample_thinned_U(beta, eps, rng)
        res = log_c - math.log(U)
        rho = _sample_layer_min(beta, U, h, c, rng)
        gap = math.log(U) - math.log(rho)
        eps = gh + res + gap
        if k >= burn:
            es.append(eps); rs.append(res); gs.append(gap); hs.append(gh)
    es, rs, gs, hs = np.array(es), np.array(rs), np.array(gs), np.array(hs)
    lag1 = float(np.corrcoef(es[:-1], es[1:])[0, 1]) if n > 2 and es.std() > 0 else float("nan")
    law0 = thinned_residual_model(beta, n_quad=100, grid=8000)
    h_mean = float(hv.mean()) if hv is not None else h_bar
    balance = (float(hs.mean()) + thinned_residual_law(beta, 0.0, grid=8000)["E_res_ratio"] + float(gs.mean())) / (1.0 - law0["slope_res_on_eps"])
    out = {"beta": beta, "h_bar": h_mean, "from_h_values": hv is not None, "n": n, "burn": burn, "mean_eps": float(es.mean()), "sd_eps": float(es.std()),
           "P_eps_positive": float((es > 0).mean()), "mean_res_ratio": float(rs.mean()), "mean_gap": float(gs.mean()), "mean_gh_shift": float(hs.mean()),
           "P_gap_positive": float((gs > 1e-12).mean()), "lag1_corr_eps": lag1, "post_hoc_affine_balance": balance, "max_eps": float(es.max())}
    if solve_fixed_point:
        out["fixed_point"] = world_fixed_point(beta, h_mean)["fixed_point"]
    return out


def main(argv=None):
    """CLI: `python -m latticelab.residual --points 75,30 100,40 --seeds 31 32 33 --tours 8 --random 200 100 --out results/lattice_residual_cap.json`
    archives the in-tour decomposition census and the random-lattice residual control."""
    import argparse
    import json
    import os
    import time

    from latticelab.lattices import lll, qary

    ap = argparse.ArgumentParser(description="the residual cap at the forced neighbour: in-tour decomposition and random-lattice control")
    ap.add_argument("--points", nargs="+", default=["75,30", "100,40"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[31, 32, 33])
    ap.add_argument("--tours", type=int, default=8)
    ap.add_argument("--q", type=int, default=2 ** 16 + 1)
    ap.add_argument("--random", nargs="*", type=int, default=None, help="per point's beta: number of random lattices for the control (one value per point, or one for all)")
    ap.add_argument("--inheritance", action="store_true", help="post-process the archive: inheritance statistics per census row and dependence per control")
    ap.add_argument("--model", action="store_true", help="post-process the archive: score the thinned-residual model against every control and the undercut law against every census row")
    ap.add_argument("--force", action="store_true", help="recompute census rows and controls already in the archive")
    ap.add_argument("--controls-only", action="store_true", help="run only the random-lattice controls of --random (no in-tour census)")
    ap.add_argument("--no-final-ratios", action="store_true", help="skip the output-basis block ratios and their comparison with the last tour's queried-block ratios")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = json.load(open(a.out)) if a.out and os.path.exists(a.out) else {
        "note": "residual cap at the forced neighbour: after every step kappa of a strict bounded forward tour the block queried next, Q/v, has "
                "eps(Q/v) = gh_shift + res_ratio + gap with gap >= 0 (lambda_1(Q/v) <= lambda_1(P/v), P/v the queried block's HKZ residual); "
                "in-tour census per (d, beta, seed, tour) and the random-lattice control of res_ratio", "census": [], "random": []}
    if a.model:
        out["note_model"] = ("thinned-residual surrogate: the residual's sign pairs are thinned by the minimality of the inserted vector (a point of norm "
                             "rho < ||v|| survives iff the absolute centred offset of its lift is at least sqrt(||v||^2 - rho^2); iid uniform offsets across "
                             "sign pairs assumed; thinning factor (1 - 2 sqrt(1 - rho^2))^+), integrated over the control's own eps(P); the undercut law "
                             "treats the layers of Q/v over P/v as an independent Poisson process at normalised heights j ||b_{kappa+beta}^*||/||v||")
        for r in out["random"]:
            r["model"] = cm = compare_control_with_model(r["rows"], r["beta"])
            me, mo, un, bs = cm["measured"], cm["model"], cm["unthinned"], cm["measured"]["bootstrap_se"]
            print(f"model vs control beta={r['beta']} n={r['n']}: E[res] {me['E_res_ratio']:+.4f} (se {me['se_E_res_ratio']:.4f}) vs model {mo['E_res_ratio']:+.4f} "
                  f"(unthinned {un['E_res_ratio']:+.4f}; discrepancy {cm['mean_discrepancy']:+.4f}, z_null = {cm['bias_z']:+.1f}, se_emp {cm['se_discrepancy_empirical']:.4f}); "
                  f"P[res>0] {me['P_res_ratio_positive']:.3f} vs {mo['P_res_ratio_positive']:.3f} (z {cm['sign_z']:+.1f}); "
                  f"slope {me['slope_res_on_eps']:+.3f} (boot se {bs['slope_res_on_eps']:.3f}) vs {mo['slope_res_on_eps']:+.3f} (unthinned {un['slope_res_on_eps']:+.3f}); "
                  f"corr {me['corr_res_eps']:+.3f} (se {bs['corr_res_eps']:.3f}) vs {mo['corr_res_eps']:+.3f}; "
                  f"E[res|dense] {me['E_res_given_dense']:+.4f} (se {bs['E_res_given_dense']:.4f}) vs {mo['E_res_given_dense']:+.4f}; E[res|sparse] {me['E_res_given_sparse']:+.4f} "
                  f"(se {bs['E_res_given_sparse']:.4f}) vs {mo['E_res_given_sparse']:+.4f}; P[res>0|dense] {me['P_pos_given_dense']:.2f} (se {bs['P_pos_given_dense']:.2f}) vs "
                  f"{mo['P_pos_given_dense']:.2f}; P[res>0|sparse] {me['P_pos_given_sparse']:.2f} (se {bs['P_pos_given_sparse']:.2f}) vs {mo['P_pos_given_sparse']:.2f}; "
                  f"cap holds {cm['cap_holds_all']} (min slack {cm['min_cap_slack']:.4f}); svp mismatch max {cm['max_svp_mismatch']:.1e}", flush=True)
            if cm["ks_thinned"]:
                print(f"   PIT mean {cm['pit_mean']:.3f}; KS thinned D={cm['ks_thinned']['D']:.3f} p={cm['ks_thinned']['p']:.3f}, unthinned D={cm['ks_unthinned']['D']:.3f} "
                      f"p={cm['ks_unthinned']['p']:.2e}; PIT-eps Spearman rho={cm['pit_eps_spearman']['rho']:+.3f} p={cm['pit_eps_spearman']['p']:.3f}; by eps-tercile "
                      + " ".join(f"[n={s['n']} D={s['D']:.3f} p={s['p']:.2f} mean {s['pit_mean']:.2f}]" for s in cm["ks_thinned_by_eps_tercile"]), flush=True)
        for c in out["census"]:
            c["gap_model"] = gm = gap_model_check(c["steps"], c["beta"])
            if gm["n"]:
                print(f"undercut law ({c['d']},{c['beta']}) seed {c['seed']}: P[gap>0] observed {gm['observed_frac_gap_positive']:.3f} (naive se {gm['se_observed_frac_naive_iid']:.3f}) vs predicted "
                      f"{gm['mean_predicted_P_gap_positive']:.3f}; E[gap] observed {gm['observed_mean_gap']:.4f} (naive se {gm['se_observed_mean_gap_naive_iid']:.4f}) vs predicted {gm['mean_predicted_E_gap']:.4f}; "
                      f"mean u* {gm['mean_u_star']:.3f}, h/||v|| {gm['mean_h_over_v']:.3f}; terciles "
                      + " ".join(f"[{t['mean_predicted_P']:.2f} -> {t['observed_frequency']:.2f}]" for t in gm["calibration_terciles"]), flush=True)
            else:
                print(f"undercut law ({c['d']},{c['beta']}) seed {c['seed']}: steps lack l_kappa (rerun the census with --force)", flush=True)
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1, default=str)
        print("RESIDUAL_DONE", flush=True)
        return
    if a.inheritance:
        for c in out["census"]:
            c["inheritance"] = st = inheritance_stats(c["steps"])
            print(f"({c['d']},{c['beta']}) seed {c['seed']}: {st['n_pairs']} adjacent pairs; corr(eps_P, res_ratio) {st.get('corr_eps_res', float('nan')):+.3f} "
                  f"(slope {st.get('slope_eps_res', float('nan')):+.3f}); corr(eps_k, eps_k+1) {st.get('corr_eps_next', float('nan')):+.3f}; "
                  f"P[next dense | dense] {st.get('p_next_dense_given_dense', float('nan')):.2f} (n={st.get('n_dense')}) vs {st.get('p_next_dense_given_not', float('nan')):.2f}", flush=True)
        for r in out["random"]:
            r["dependence"] = dp = control_dependence(r["rows"], r["beta"])
            print(f"random beta={r['beta']} n={r['n']}: corr(eps_P, res_ratio) {dp['corr_eps_res']:+.3f} slope {dp['slope_eps_res']:+.3f} "
                  f"(volume-only 1/(beta-1) = {dp['volume_only_slope']:.3f}); res_ratio mean | eps_P>0: {dp['mean_res_given_dense']:+.4f} (n={dp['n_dense']}), "
                  f"| eps_P<=0: {dp['mean_res_given_not']:+.4f}; P[res>0 | eps_P>0] {dp['p_res_pos_given_dense']:.2f}, | eps_P<=0: {dp['p_res_pos_given_not']:.2f}", flush=True)
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1, default=str)
        print("RESIDUAL_DONE", flush=True)
        return
    for pi, pt in enumerate(a.points):
        d, beta = (int(x) for x in pt.split(","))
        for seed in ([] if a.controls_only else a.seeds):
            if not a.force and any(c["d"] == d and c["beta"] == beta and c["seed"] == seed and c.get("q") == a.q and c["tours"] >= a.tours for c in out["census"]):
                continue
            t0 = time.time()
            A = lll(qary(d, d // 2, a.q, seed=seed))
            c = residual_census(A, beta, a.tours, final_ratios=not a.no_final_ratios)
            row = {"d": d, "beta": beta, "seed": seed, "q": a.q, "tours": a.tours, "seconds": time.time() - t0, "gsa_tight_gh_shift": gsa_tight_gh_shift(beta),
                   "summary": c["summary"], "per_tour": c["per_tour"], "next_query_consistent": c["next_query_consistent"],
                   "max_next_query_mismatch": c["max_next_query_mismatch"], "steps": c["steps"]}
            if "output_vs_queried" in c:
                row["output_vs_queried"] = c["output_vs_queried"]
                row["final_basis"] = c["final_basis"]
                row["all_positions"] = {k: v for k, v in c["all_positions"].items() if k not in ("eps_query", "dvol", "nu")}
            out["census"] = [x for x in out["census"] if not (x["d"] == d and x["beta"] == beta and x["seed"] == seed and x.get("q") == a.q)] + [row]
            s = c["summary"]
            print(f"({d},{beta}) seed {seed}, {a.tours} tours, {s['n']} steps: mean gh_shift {s['mean_gh_shift']:+.4f} (GSA-tight {row['gsa_tight_gh_shift']:+.4f}), "
                  f"res_ratio {s['mean_res_ratio']:+.4f} (sd {s['sd_res_ratio']:.4f}, max {s['max_res_ratio']:+.3f}), gap {s['mean_gap']:.4f} (zero in {s['frac_gap_zero']:.2f}); "
                  f"eps_next {s['mean_eps_neighbour']:+.4f}, >0 in {s['frac_eps_neighbour_positive']:.2f}, forced by the cap in {s['frac_cap_bound_positive']:.2f} "
                  f"(cap positive part {s['mean_cap_bound_positive_part']:.4f} of {s['mean_eps_neighbour_positive_part']:.4f}); created {s.get('mean_created_ratio', float('nan')):+.4f}, "
                  f"violations created in {s.get('frac_violation_created', float('nan')):.2f}; next-query consistent {c['next_query_consistent']} "
                  f"(mismatch {c['max_next_query_mismatch']:.1e}) [{row['seconds']:.0f}s]", flush=True)
            for T in sorted(c["per_tour"]):
                p = c["per_tour"][T]
                print(f"   tour {T}: gh_shift {p['mean_gh_shift']:+.4f} res_ratio {p['mean_res_ratio']:+.4f} gap {p['mean_gap']:.4f} eps_next {p['mean_eps_neighbour']:+.4f} "
                      f">0 {p['frac_eps_neighbour_positive']:.2f} cap>0 {p['frac_cap_bound_positive']:.2f} changed {p.get('frac_changed', float('nan')):.2f}", flush=True)
            if "output_vs_queried" in c:
                o = c["output_vs_queried"]
                print(f"   output vs queried (last tour, {o['n']} positions): queried mean eps {o['mean_queried_eps']:+.4f} (>0 {o['frac_queried_positive']:.2f}) -> output "
                      f"{o['mean_output_eps']:+.4f} (>0 {o['frac_output_positive']:.2f}); delta mean {o['mean_delta']:+.4f} (sd {o['sd_delta']:.4f}, >0 {o['frac_delta_positive']:.2f}) = volume part "
                      f"{o['mean_delta_volume_part']:+.4f} + minimum part {o['mean_delta_minimum_part']:+.4f} (min {o['min_delta_minimum_part']:+.1e}, >0 in {o['frac_minimum_part_positive']:.2f}; "
                      f"decomposition gap {o['max_decomposition_gap']:.1e}), corr {o['corr_queried_output']:+.2f}; output eps_needed {o['output_eps_needed']:.3f}, SVP-tight {o['output_frac_tight']:.2f}", flush=True)
                ap_ = c["all_positions"]
                print(f"   all {ap_['n']} positions (last tour): forced-entry gap {ap_['max_forced_entry_gap']:.1e}, nu = eps_q + dvol to {ap_['max_nu_identity_gap']:.1e}, head identity to "
                      f"{ap_['head_identity_gap']:.1e}; head - floor {ap_['head_minus_floor']:+.4f} = -sum y eps_q ({ap_['minus_sum_y_eps_query']:+.4f}) - sum y dvol ({ap_['minus_sum_y_dvol']:+.4f}); "
                      f"eps_q head block {ap_['head_block_eps_query']:+.4f}, full mean {ap_['mean_eps_query_full']:+.4f}, tail mean {ap_['mean_eps_query_tail']:+.4f}, mean dvol {ap_['mean_dvol']:+.4f}", flush=True)
            if a.out:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(out, open(a.out, "w"), indent=1, default=str)
        if a.random:
            n = a.random[pi] if pi < len(a.random) else a.random[-1]
            if n > 0 and (a.force or not any(r["beta"] == beta and r.get("q") == a.q and r["n"] >= n for r in out["random"])):
                t0 = time.time()
                rr = residual_ratio_random(beta, n, a.q)
                rr["seconds"] = time.time() - t0
                out["random"] = [x for x in out["random"] if not (x["beta"] == beta and x.get("q") == a.q)] + [rr]
                print(f"random beta={beta}, n={n}: eps(P) mean {rr['mean_eps_P']:+.4f} (sd {rr['sd_eps_P']:.4f}); res_ratio mean {rr['mean_res_ratio']:+.4f} "
                      f"(sd {rr['sd_res_ratio']:.4f}, se {rr['se_mean_res_ratio']:.4f}), positive in {rr['frac_res_ratio_positive']:.2f} [{rr['seconds']:.0f}s]", flush=True)
                if a.out:
                    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("RESIDUAL_DONE", flush=True)


if __name__ == "__main__":
    main()
