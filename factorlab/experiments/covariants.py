"""E23c: does the Hessian covariant of a known-root cubic carry a related root modulo N?

A proposal (reviewed in the research log, 2026-08-24) suggested that for a binary
cubic F with F(m, 1) = kN the quadratic covariant H (the Hessian) has a related root
modulo N, which would reduce cubic selection to the degree-2 near-square search of
E19.  The classical syzygy says otherwise.  With
    H = F_xx F_yy - F_xy^2,     G = F_x H_y - F_y H_x,     Delta = disc F,
one has identically (constants determined here by sympy and verified symbolically)
    G^2 = -H^3 - 432 Delta F^2,
so at a root, where F(m, 1) = kN, the only consequence is the congruence
    G(m, 1)^2 = -H(m, 1)^3  (mod N^2),
an exact identity that holds modulo p and q alike (the CRT obstruction of
notes_beyond_gnfs.md, Exit E2); H(m, 1) itself is a nonzero residue.  This module
verifies the syzygy symbolically and evaluates H and G at the root for the E18 cubic
witnesses.
"""

from __future__ import annotations

import json
import sys

import sympy as sp

_x, _y, _a, _b, _c, _d = sp.symbols("x y a b c d")
F = _a * _x ** 3 + _b * _x ** 2 * _y + _c * _x * _y ** 2 + _d * _y ** 3
H = sp.expand(sp.diff(F, _x, 2) * sp.diff(F, _y, 2) - sp.diff(F, _x, _y) ** 2)
G = sp.expand(sp.diff(F, _x) * sp.diff(H, _y) - sp.diff(F, _y) * sp.diff(H, _x))
DELTA = sp.discriminant(F.subs(_y, 1), _x)


def syzygy_constants() -> tuple[int, int]:
    """(l1, l2) with G^2 = l1 H^3 + l2 Delta F^2 identically; raises if no such identity."""
    l1, l2 = sp.symbols("l1 l2")
    pts = [{_a: 2, _b: -3, _c: 5, _d: 7, _x: 3, _y: 1}, {_a: 1, _b: 4, _c: -2, _d: 3, _x: -2, _y: 5}]
    sol = sp.solve([sp.Eq((G ** 2).subs(p), (l1 * H ** 3 + l2 * DELTA * F ** 2).subs(p)) for p in pts], [l1, l2])
    if sp.expand(G ** 2 - sol[l1] * H ** 3 - sol[l2] * DELTA * F ** 2) != 0:
        raise RuntimeError("no syzygy of the assumed form")
    return int(sol[l1]), int(sol[l2])


def covariants_at_root(f: list[int], m: int, N: int) -> dict:
    """H(m,1), G(m,1) for the cubic with coefficients f (low to high) at its root m modulo N."""
    f0, f1, f2, f3 = [int(t) for t in f]
    if (f3 * m ** 3 + f2 * m ** 2 + f1 * m + f0) % N != 0:
        m = -m
    assert (f3 * m ** 3 + f2 * m ** 2 + f1 * m + f0) % N == 0
    sub = {_a: f3, _b: f2, _c: f1, _d: f0, _x: m, _y: 1}
    Hm, Gm = int(H.subs(sub)), int(G.subs(sub))
    l1, _ = syzygy_constants()
    return {"H_mod_N": Hm % N, "H_vanishes": Hm % N == 0,
            "syzygy_congruence_mod_N2": (Gm * Gm - l1 * Hm ** 3) % (N * N) == 0}


def check_e18_witnesses(path: str = "results/e18_poly_floor.json") -> dict:
    data = json.load(open(path))["3"]
    n = vanish = cong = 0
    for row in data["rows"]:
        for inst in row["instances"]:
            r = covariants_at_root(inst["f"], int(inst["m"]), int(inst["N"]))
            n += 1
            vanish += r["H_vanishes"]
            cong += r["syzygy_congruence_mod_N2"]
    return {"witnesses": n, "hessian_vanishes_at_root": vanish, "syzygy_congruence_holds": cong,
            "syzygy_constants": syzygy_constants()}


if __name__ == "__main__":  # python -m factorlab.experiments.covariants [results/e18_poly_floor.json]
    out = check_e18_witnesses(*sys.argv[1:2])
    print(f"syzygy: G^2 = {out['syzygy_constants'][0]} H^3 + {out['syzygy_constants'][1]} Delta F^2")
    print(f"E18 cubic witnesses: {out['witnesses']}; Hessian vanishes at the root mod N: {out['hessian_vanishes_at_root']}; "
          f"G(m)^2 = -H(m)^3 mod N^2: {out['syzygy_congruence_holds']}")
