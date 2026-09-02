"""The call-complexity of block reduction as a game (docs/lattice_barrier_plan.md, T1; notes section 10).

The block-oracle model lets the algorithm choose which projected block to query and when.  Tours are one schedule; the model's floor must hold
for every schedule, adaptive in the answers received.  This module measures, at equal numbers of exact-SVP calls, what position-choice buys:

  * `tours`          -- the canonical forward tour (positions 0, 1, ..., d-2 in order; block-supported bounded-LLL completion);
  * `reverse`        -- reverse tours (d-2, ..., 0);
  * `gh_greedy`      -- adaptive on the PROFILE only (no probing): query the position whose entry exceeds its block's GH by most, l_k - log GH(B_k),
                        the block most likely to hold an insertion, ties by position; a position queried without effect is not queried again until
                        the basis changes, and when every position is stale the basis is a fixed point and the run ends early;
  * `random`         -- a uniformly random position per call;
  * `omniscient_phi` -- an adversary that, at every call, PROBES the insertion at every position (probes are free; only the applied call is counted)
                        and applies the one with the largest decrease of the LLL potential Phi = sum_i (d - i) l_i;
  * `omniscient_head`-- the same adversary applying the largest decrease of the head l_1, falling back to the largest Phi decrease when no probe
                        lowers the head.

The omniscient rules are not algorithms (their probes are not counted); they bound what adaptive position choice with full knowledge of the
block minima can achieve per counted call -- an upper bound on the power of scheduling in the model.  For every rule the run records, after
every call, the head l_1 - S/d relative to the zero-slack floor h_{d,beta}(0), the potential, and, at the tour marks (every d - 1 calls), the
y-weighted sub-GH mass sum y_i eps_i(B) and the residual R(B) of the current basis (exact block minima by enumeration; `insertion.weighted_subgh_mass`).
Every insertion is the strict one of `latticelab.flag` (insert whenever the block minimum is shorter than b_k^* by more than the tie tolerance).
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from fpylll import IntegerMatrix

from latticelab.flag import _insert, _reducer
from latticelab.profile_floor import floor_l1_float, log_chat

RULES = ("tours", "reverse", "gh_greedy", "random", "omniscient_phi", "omniscient_head")


def _profile(M) -> np.ndarray:
    M.update_gso()
    return np.array([0.5 * math.log(M.get_r(j, j)) for j in range(M.d)])


def _phi(p: np.ndarray) -> float:
    d = len(p)
    return float(sum((d - i) * p[i] for i in range(d)))


def _gh_excess(p: np.ndarray, beta: int) -> np.ndarray:
    """l_k - log GH(B_k) for every start k (0-based), the profile-level excess of the block's first entry over its GH."""
    d = len(p)
    out = np.zeros(d - 1)
    for k in range(d - 1):
        n = min(beta, d - k)
        out[k] = p[k] - (log_chat(n) + p[k:k + n].mean())
    return out


def _probe(B: IntegerMatrix, k: int, beta: int):
    """The profile after a strict insertion at k on a copy of B (the copy is returned with it)."""
    C = IntegerMatrix.from_matrix(B)
    bkz, C2, M = _reducer(C)
    _insert(bkz, k, min(beta, C2.nrows - k), "bounded_lll")
    return _profile(M), C2


def play(A: IntegerMatrix, beta: int, calls: int, rule: str, seed: int = 0, mass_every: int | None = None) -> Dict:
    """Run `calls` exact-SVP calls on A under the given rule; returns the per-call head (l_1 - S/d - h(0)) and potential, the positions queried,
    the number of calls that changed the basis, and the y-weighted mass and residual at the marks (every `mass_every` calls; default d - 1)."""
    from latticelab.insertion import weighted_subgh_mass

    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    d = A.nrows
    if beta < 2 or d < beta + 2 or calls < 1:
        raise ValueError("need 2 <= beta <= d - 2 and calls >= 1")
    mass_every = d - 1 if mass_every is None else mass_every
    if not (isinstance(mass_every, int) and mass_every >= 1):
        raise ValueError("mass_every must be a positive integer")
    rng = np.random.default_rng(seed)
    h0 = floor_l1_float(d, beta)["l1_floor"]
    B = IntegerMatrix.from_matrix(A)
    bkz, B, M = _reducer(B)
    p = _profile(M)
    S = float(p.sum())
    heads, phis, positions, changed, marks = [], [], [], 0, []
    moved_flags: List[bool] = []
    stale: set = set()  # gh_greedy: positions queried without effect since the last change
    ended_early = None
    for t in range(calls):
        if rule == "tours":
            k = t % (d - 1)
        elif rule == "reverse":
            k = (d - 2) - (t % (d - 1))
        elif rule == "random":
            k = int(rng.integers(d - 1))
        elif rule == "gh_greedy":
            ex = _gh_excess(p, beta)
            live = [kk for kk in range(d - 1) if kk not in stale]
            if not live:
                ended_early = t  # every position is stale: a fixed point of the strict insertion (clean)
                break
            k = max(live, key=lambda kk: (ex[kk], -kk))
        else:
            best, best_state, best_head, best_phi = None, None, None, None
            phi_p = _phi(p)
            for kk in range(d - 1):
                q, C2 = _probe(B, kk, beta)
                dh, dphi = q[0] - p[0], _phi(q) - phi_p
                if rule == "omniscient_head":
                    # stage 1: the largest genuine head decrease; stage 2 (no probe lowers the head): the largest potential decrease
                    key = (0, dh) if dh < -1e-9 else (1, dphi)
                else:
                    key = (0, dphi)
                if best is None or key < (best_head, best_phi):
                    best, best_state, (best_head, best_phi) = kk, C2, key
            k = best
            # apply: replace the working basis by the probed copy (identical to inserting at k on B)
            B = best_state
            bkz, B, M = _reducer(B)
        if rule not in ("omniscient_phi", "omniscient_head"):
            _insert(bkz, k, min(beta, d - k), "bounded_lll")
        q = _profile(M)
        moved = bool(np.max(np.abs(q - p)) > 1e-9)
        changed += int(moved)
        moved_flags.append(moved)
        if rule == "gh_greedy":
            if moved:
                stale = set()
            else:
                stale.add(k)
        p = q
        positions.append(k)
        heads.append(float(p[0] - S / d - h0))
        phis.append(_phi(p))
        if (t + 1) % mass_every == 0 or t + 1 == calls:
            w = weighted_subgh_mass(B, beta)
            marks.append({"calls": t + 1, "head_minus_floor": heads[-1], "mass_signed": w["weighted_eps_signed"], "mass_positive": w["weighted_eps_positive"],
                          "residual": w["residual"], "frac_tight": w["frac_tight"]})
    if ended_early is not None and (not marks or marks[-1]["calls"] != len(heads)):
        w = weighted_subgh_mass(B, beta)
        marks.append({"calls": len(heads), "head_minus_floor": heads[-1], "mass_signed": w["weighted_eps_signed"], "mass_positive": w["weighted_eps_positive"],
                      "residual": w["residual"], "frac_tight": w["frac_tight"]})
    return {"d": d, "beta": beta, "calls": len(heads), "calls_requested": calls, "ended_early_at": ended_early, "rule": rule, "seed": seed, "heads": heads,
            "phis": phis, "positions": positions, "moved": moved_flags, "changed": changed, "marks": marks, "final_head_minus_floor": heads[-1], "final_profile": p.tolist()}


def compare(d: int, beta: int, q: int, seed: int, calls: int, rules=RULES, rng_seed: int = 0) -> Dict:
    """All rules from the same LLL basis of qary(d, d//2, q, seed) for the same number of calls."""
    from latticelab.lattices import lll, qary

    A = lll(qary(d, d // 2, q, seed=seed))
    out = {"d": d, "beta": beta, "q": q, "seed": seed, "calls": calls, "runs": {}}
    for r in rules:
        out["runs"][r] = play(A, beta, calls, r, rng_seed)
    return out


def main(argv=None):
    """CLI: `python -m latticelab.schedule_game --points 40,10 --seeds 31 32 33 --tours 8 --out results/lattice_schedule_game.json`."""
    import argparse
    import json
    import os
    import time

    ap = argparse.ArgumentParser(description="the call-complexity of block reduction as a game: schedules at equal call counts")
    ap.add_argument("--points", nargs="+", default=["40,10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[31, 32, 33])
    ap.add_argument("--tours", type=int, default=8, help="calls = tours * (d - 1)")
    ap.add_argument("--q", type=int, default=2 ** 16 + 1)
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = json.load(open(a.out)) if a.out and os.path.exists(a.out) else {"note": "schedules of exact-SVP calls at equal call counts: tours, reverse tours, profile-only GH-greedy, random, and omniscient adversaries "
                                                                              "(free probes of every position, applying the largest potential or head decrease); head - floor, potential, and the y-weighted mass at the tour marks",
                                                                      "rows": []}
    for pt in a.points:
        d, beta = (int(x) for x in pt.split(","))
        for seed in a.seeds:
            t0 = time.time()
            c = compare(d, beta, a.q, seed, a.tours * (d - 1), a.rules)
            row = {"d": d, "beta": beta, "q": a.q, "seed": seed, "calls": c["calls"], "seconds": time.time() - t0,
                   "runs": {r: {k: v for k, v in run.items() if k not in ("phis", "positions")} for r, run in c["runs"].items()}}
            out["rows"] = [x for x in out["rows"] if not (x["d"] == d and x["beta"] == beta and x["seed"] == seed and x["q"] == a.q)] + [row]
            marks_at = lambda run, T: next((m for m in run["marks"] if m["calls"] == T * (d - 1)), None)
            for r in a.rules:
                run = c["runs"][r]
                summary = " ".join(f"T{T}:{marks_at(run, T)['head_minus_floor']:+.3f}/{marks_at(run, T)['mass_signed']:+.3f}" for T in (1, 2, 4, a.tours) if marks_at(run, T))
                print(f"({d},{beta}) seed {seed} {r:15s}: head-floor/mass at tour marks {summary}; changed {run['changed']}/{run['calls']} [{row['seconds']:.0f}s]", flush=True)
            if a.out:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("GAME_DONE", flush=True)


if __name__ == "__main__":
    main()
