"""E42 -- the additive energy of the Lehman speeds and the energy form of Lemma D.

Proposition E (``docs/notes_barrier.md`` 7.16): with E_theta(r) := #{(P, Q) : |v_P - v_Q| < theta rho_r}
over ordered pairs of pairs of shell cells, Robert--Sargos's four-variable spacing bound
(|sqrt k1 + sqrt k4 - sqrt k2 - sqrt k3| < delta M^{1/2} has << M^{2+eps}(1 + delta M^2) solutions in
[M, 2M]^4) gives E_theta(r) << r^{2+eps} -- the Poissonian order, since the diagonal alone is
~ r^2/8 -- while the maximum cluster is >= c r^{1/3} (Theorem Q''): clustering is a tail phenomenon.
Consequently the number of pairwise disjoint windows of half-width theta rho_r / 2 with >= lambda pairs
is << r^{2+eps} / lambda^2.

(a) ``modfree_energy`` computes E_1(r), the mean and maximum of D(P) := #{Q : |v_Q - v_P| < rho_r} and
    the tail counts #{P : D(P) >= lambda} exactly on the squarefree shell, and the same for a
    phase-randomised null (every d-class shifted by an independent uniform fraction of its local spacing).
(b) ``planar_top_mass`` computes, from the exact window starts, the distribution of D(u) = #{P :
    |s_I - s_I' - u| < W} over all integers u (run-length encoded), and reports D_max against the mean
    of the m largest values -- the quantity the energy form of Lemma D would use in place of D_max.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict

import numpy as np

from factorlab.experiments.planar_census import shell_starts
from factorlab.experiments.sidon_bucketed import rho, squarefree_mask, squarefree_shell
from factorlab.experiments.sidon_scaling import lemma_d_window
from factorlab.gen import make_semiprime


def _pair_speeds(ks: np.ndarray, null: bool = False, seed: int = 0):
    """All speeds (k'-k)/(sqrt k' + sqrt k) for k < k' in ks, with their differences d; optionally
    phase-randomised: each d-class shifted by U_d times its local spacing d/(4 k^{3/2})."""
    ks = np.asarray(ks, dtype=np.int64)
    R = len(ks)
    i, j = np.triu_indices(R, k=1)
    a, b = ks[i], ks[j]
    d = (b - a).astype(np.float64)
    v = d / (np.sqrt(b.astype(np.float64)) + np.sqrt(a.astype(np.float64)))
    if null:
        rng = np.random.default_rng(seed)
        dmax = int(d.max())
        U = rng.random(dmax + 1)
        v = v + U[(b - a)] * d / (4.0 * a.astype(np.float64) ** 1.5)
    return v, (b - a)


def modfree_energy(r: int, theta: float = 1.0, null: bool = False, seed: int = 0) -> Dict:
    """E42(a): exact energy, mean/max cluster and tail counts of D(P) on the squarefree shell."""
    t0 = time.time()
    ks = squarefree_shell(r)
    v, _ = _pair_speeds(ks, null=null, seed=seed)
    vs = np.sort(v)
    w = theta * rho(r)
    D = np.searchsorted(vs, vs + w, side="left") - np.searchsorted(vs, vs - w, side="right")
    # D counts Q with |v_Q - v_P| < w including P itself
    E = int(D.sum())
    npairs = int(len(vs))
    lam_max = int(D.max())
    tail = {int(l): int((D >= l).sum()) for l in range(1, lam_max + 1)}
    return {"r": int(r), "theta": theta, "null": null, "R": int(len(ks)), "pairs": npairs, "energy": E,
            "energy_over_pairs": E / npairs, "energy_over_r2": E / r ** 2, "D_max": lam_max,
            "tail": tail, "time_s": time.time() - t0}


def energy_experiment(log2_rs=(12, 13, 14, 15), theta: float = 1.0, seed: int = 0):
    rows = []
    for e in log2_rs:
        r = 2 ** e
        act = modfree_energy(r, theta)
        nul = modfree_energy(r, theta, null=True, seed=seed + e)
        rows.append({"log2_r": e, "actual": act, "null": nul,
                     "excess_energy": act["energy"] - nul["energy"],
                     "null_D_max": nul["D_max"],
                     "pairs_above_null_max": int(sum(1 for l, c in act["tail"].items() if l > nul["D_max"]) and
                                                 act["tail"].get(nul["D_max"] + 1, 0))})
        print(f"2^{e}: pairs={act['pairs']} E={act['energy']} (E/pairs={act['energy_over_pairs']:.3f}, "
              f"E/r^2={act['energy_over_r2']:.4f}) D_max={act['D_max']} | null E={nul['energy']} "
              f"(E/pairs={nul['energy_over_pairs']:.3f}) D_max={nul['D_max']} | pairs with D > null max: "
              f"{act['tail'].get(nul['D_max'] + 1, 0)}", flush=True)
    return rows


def planar_top_mass(bits: int, r: int, index: int = 0, seed: int = 7, m_values=(), kind: str = "squarefree") -> Dict:
    """E42(b): the distribution of D(u) over all integers u for the exact starts of the shell at r, and
    the mean of the m largest values against D_max, for each m in m_values (default N^{1/5} and 2 N^{1/5})."""
    t0 = time.time()
    sp = make_semiprime(bits, "rsa", seed, index)
    N = int(sp.N)
    ks = squarefree_shell(r) if kind == "squarefree" else np.arange(r // 2 + 1, r + 1, dtype=np.int64)
    mask = np.zeros(r + 1, dtype=bool)
    mask[ks] = True
    _, s = shell_starts(N, r, mask)
    s = np.asarray(s, dtype=np.int64)
    W = int(lemma_d_window(N, r))
    R = len(s)
    i, j = np.triu_indices(R, k=1)
    delta = s[j] - s[i]
    delta = np.concatenate([delta, -delta])          # ordered pairs (I, I')
    # D(u) = #{P : |Delta_P - u| < W} = #{P : Delta_P in (u - W, u + W)}; as u increases it changes at
    # u = Delta_P - W + 1 (P enters) and u = Delta_P + W (P leaves): run-length encode over sorted events
    enter = np.sort(delta - W + 1)
    leave = np.sort(delta + W)
    events = np.concatenate([enter, leave])
    kinds = np.concatenate([np.ones(len(enter), dtype=np.int64), -np.ones(len(leave), dtype=np.int64)])
    order = np.argsort(events, kind="stable")
    events, kinds = events[order], kinds[order]
    # collapse equal event positions
    uniq, idx = np.unique(events, return_index=True)
    net = np.add.reduceat(kinds, idx)
    level = np.cumsum(net)                            # D(u) for u in [uniq[t], uniq[t+1])
    run = np.diff(uniq)                               # lengths of the runs (last run has level 0)
    values, lengths = level[:-1], run
    keep = values > 0
    values, lengths = values[keep], lengths[keep]
    order = np.argsort(-values, kind="stable")
    values, lengths = values[order], lengths[order]
    csum = np.cumsum(lengths)
    total_mass = int((values * lengths).sum())
    D_max = int(values[0]) if len(values) else 0
    if not m_values:
        T = int(round(N ** 0.2))
        m_values = (T, 2 * T, 4 * T)
    top = {}
    for m in m_values:
        # sum of the m largest D(u) values over distinct integers u
        full = np.searchsorted(csum, m, side="left")   # runs fully inside the top m
        mass = int((values[:full] * lengths[:full]).sum())
        used = int(csum[full - 1]) if full > 0 else 0
        if full < len(values) and used < m:
            mass += int(values[full]) * (m - used)
        top[int(m)] = {"mean_top_m": mass / m, "ratio_Dmax_over_mean": D_max / (mass / m) if mass else None}
    n_u_ge = {int(l): int(lengths[values >= l].sum()) for l in sorted(set(values.tolist()))[:40]}
    return {"bits": bits, "index": index, "N": str(N), "r": int(r), "W": W, "R": int(R), "pairs": int(len(delta)),
            "total_mass": total_mass, "D_max": D_max, "top": top, "count_u_with_D_ge": n_u_ge,
            "time_s": time.time() - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=int, nargs="*", default=[], help="log2 radii for the modulus-free energy")
    ap.add_argument("--planar", type=int, nargs="*", default=[], help="bit sizes for the planar top-mass statistic")
    ap.add_argument("--planar-r", type=str, default="third,crossing", help="comma list of radii: third|crossing|quarter")
    ap.add_argument("--out", default="results/e42_energy.json")
    a = ap.parse_args()
    out = {}
    if a.energy:
        out["energy"] = energy_experiment(tuple(a.energy))
    if a.planar:
        rows = []
        for bits in a.planar:
            N = int(make_semiprime(bits, "rsa", 7, 0).N)
            for name in a.planar_r.split(","):
                r = {"third": int(round(N ** (1 / 3))), "crossing": int(round(1.44 * N ** (3 / 11))),
                     "quarter": int(round(N ** 0.25))}[name]
                row = planar_top_mass(bits, r)
                row["radius"] = name
                rows.append(row)
                print(f"{bits} bits r={r} ({name}) W={row['W']} R={row['R']} D_max={row['D_max']} "
                      + " ".join(f"m={m}: mean={t['mean_top_m']:.2f} ratio={t['ratio_Dmax_over_mean']:.2f}"
                                 for m, t in row["top"].items()) + f" ({row['time_s']:.0f}s)", flush=True)
        out["planar"] = rows
    d = os.path.dirname(a.out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
