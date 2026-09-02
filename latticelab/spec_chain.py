"""The Kyber round-3 primal-attack blocksize chain (specification section 5.1.2, read in the primary text) with the GSA root-Hermite factor,
and the same chain with the certified profile floor in its place.

The specification builds, from m LWE samples, the lattice of dimension d = m + kn + 1 and volume q^m containing the unique short vector
(s, e, 1) of norm ~ sigma sqrt(kn + m), models BKZ-b by the GSA with delta(b) = ((pi b)^{1/b} b/(2 pi e))^{1/(2(b-1))} (the Chen-Nguyen form), and
declares the attack successful iff its condition (9) holds:
        sigma sqrt(b)  <=  delta^{2b - d - 1} q^{m/d},
the number of samples m in [0, (k+1) n] being optimised numerically; core-SVP prices the attack at 2^{0.292 b} (classical) and 2^{0.265 b}
(quantum).  Table 4 prints (d, b) = (1003, 403), (1424, 625), (1885, 877) for Kyber512/768/1024 with sigma^2 = eta_1/2 = 3/2, 1, 1.

With the floor.  Inside the profile model (notes section 4) a (b, eps)-admissible output basis has root-Hermite factor at least
delta_0^floor(d, b, eps), so the model's necessary blocksize for the *specification's own* success condition is the least b for which some m
satisfies (9) with delta_0^floor(d, b, eps) in place of delta(b) -- the target moving with b and d exactly as in the specification, unlike the
fixed-target scans of `profile_floor.beta_floor_for_target`.  For 2b - d - 1 < 0 (all cases here) a larger delta makes the right side smaller.

What is decided rigorously and what is not.  `chain` is a double-precision search over the FULL range m in [0, (k+1) n] and b in [b_lo, b_hi]
(the floor via `profile_floor.floor_l1_float`, a pre-screen).  `certify_floor_chain` then makes directed arb decisions with exact rational duals
(`profile_floor.floor_l1`, about 0.5 s per (d, b) at d ~ 1000 with the O(d) recurrence): (i) the found crossing (b*, m*) passes
(upper(floor ball) <= lower(target ball)); (ii) EVERY (b, m) with b in [b_lo, b* - 1] and m in the full valid range fails (lower(floor) >
upper(target)); m with 2b >= d + 1 are outside the condition's domain and are listed separately.  The target delta_req(b, m) := (q^{m/d} / (sigma
sqrt b))^{1/(d + 1 - 2b)} is an arb ball from exact rationals.  When both hold the result is a leastness certificate for the floor chain within
[b_lo, b*] x [0, (k+1) n] -- relative to the stated b-range, as the floor is not monotone in b and the target moves with b and d; blocksizes below
b_lo are excluded by the double-precision screen only.  The result dictionary states exactly this in `certification_scope`.

The detection chain (the correction).  Substituting delta_0^floor into (9) keeps the GSA *shape*: it moves the tail entry by |kappa| (d + 1 -
2b)/(d - 1), about 0.013 at the Kyber points, and gives +2 blocksizes.  The floor-side statement the axiom actually supports is different.  The
attack detects the planted vector when its projection orthogonally to F_{d-b} (norm about sigma sqrt b) is the shortest vector of the last block
L/F_{d-b} -- the only block whose numerator is the whole lattice -- which in the Gaussian-heuristic sense requires  sigma sqrt b <= GH_b(L/F_{d-b}) =
chat(b) (vol L / vol F_{d-b})^{1/b}, a function of the prefix volume vol F_{d-b} alone (condition (9) reads that volume off the GSA line).  By the
prefix-volume floor (profile_floor), for the profile of the lattice's random part -- every block admissible, as the GSA argument also assumes --
vol F_{d-b} >= vol F_{d-b}^tight, so log GH_b(L/F_{d-b}) <= l^tight_{d-b+1}, the tight profile's own entry at the detection position, and detection
needs  log(sigma sqrt b) <= l^tight_{d-b+1}(d, b, eps)  with d = m + kn + 1 and S = m log q, optimised over m.  The tight profile's body runs
parallel to the GSA line but lower by the full |kappa| (its HKZ-like tail rises to hold the volume), so the shift is several times the +2:
`detection_chain` (double precision, O(d) tight entries; the GSA-line control reproduces 406/624/874) and `certify_detection_chain` (rigorous: arb
tight entries against the exact left side, the crossing passing and every earlier (b, m) failing).  Caveat: the block axiom is applied to the
random part at every position including the detection block; without the detection block's own constraint the prefix volume is unbounded
(profile_floor docstring), so the statement is model-relative in the same way as condition (9) itself.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Dict, Optional, Tuple

from latticelab.profile import delta_gsa

KYBER = {"Kyber512": {"k": 2, "eta1": 3, "printed": (1003, 403)}, "Kyber768": {"k": 3, "eta1": 2, "printed": (1424, 625)},
         "Kyber1024": {"k": 4, "eta1": 2, "printed": (1885, 877)}}
N_RING, Q = 256, 3329


def log_delta_req(b: int, m: int, k: int, sigma2: Fraction) -> float:
    """log of the root-Hermite factor the attack needs at (b, m): condition (9) is  log delta <= [ (m/d) log q - log(sigma sqrt b) ] / (d + 1 - 2b)."""
    d = m + k * N_RING + 1
    if d + 1 - 2 * b <= 0:
        raise ValueError("the condition changes direction for 2b >= d + 1")
    return ((m / d) * math.log(Q) - 0.5 * math.log(float(sigma2)) - 0.5 * math.log(b)) / (d + 1 - 2 * b)


def chain(k: int, eta1: int, log_delta: Callable[[int, int], float], b_lo: int = 300, b_hi: int = 1100, m_range: Optional[Tuple[int, int]] = None) -> Dict:
    """The least b in [b_lo, b_hi] for which some m in [0, (k+1) n] (or in m_range) satisfies (9) with the root-Hermite factor exp(log_delta(d, b)).
    Double precision.  Returns b, the admissible m, the one of largest margin, d, and for every b examined the m of largest margin (used by the
    certification)."""
    sigma2 = Fraction(eta1, 2)
    lo_m, hi_m = (0, (k + 1) * N_RING) if m_range is None else m_range
    best_m_per_b = {}
    for b in range(b_lo, b_hi + 1):
        rows = []
        for m in range(lo_m, hi_m + 1):
            d = m + k * N_RING + 1
            if d + 1 - 2 * b <= 0:
                continue
            rows.append((log_delta_req(b, m, k, sigma2) - log_delta(d, b), m, d))
        if not rows:  # every m has 2b >= d + 1: the condition changes direction and (9) is not the right test
            best_m_per_b[b] = None
            continue
        rows.sort(reverse=True)
        best_m_per_b[b] = {"m": rows[0][1], "d": rows[0][2], "margin": rows[0][0]}
        ok = [r for r in rows if r[0] >= 0]
        if ok:
            return {"b": b, "m_best": ok[0][1], "d_best": ok[0][2], "margin_best": ok[0][0], "admissible_m": sorted(x[1] for x in ok),
                    "n_admissible_m": len(ok), "sigma2": sigma2, "k": k, "m_range": [lo_m, hi_m], "best_m_per_b": best_m_per_b}
    return {"b": None, "k": k, "sigma2": sigma2, "m_range": [lo_m, hi_m], "best_m_per_b": best_m_per_b, "note": f"no b in [{b_lo}, {b_hi}] succeeds"}


def gsa_chain(k: int, eta1: int, **kw) -> Dict:
    """The specification's chain: Chen-Nguyen delta(b)."""
    return chain(k, eta1, lambda d, b: math.log(delta_gsa(b)), **kw)


def floor_chain(k: int, eta1: int, eps=0.0, **kw) -> Dict:
    """The same chain with the floor delta_0^floor(d, b, eps) (double-precision dual; the certification is separate)."""
    from latticelab.profile_floor import floor_l1_float

    cache = {}

    def ld(d, b):
        if (d, b) not in cache:
            cache[(d, b)] = math.log(floor_l1_float(d, b, float(eps))["root_hermite_floor"])
        return cache[(d, b)]

    return chain(k, eta1, ld, **kw)


def target_ball(b: int, m: int, k: int, sigma2: Fraction, prec: int = 256):
    """delta_req(b, m) as an arb ball: exp(((m/d) log q - (1/2) log sigma2 - (1/2) log b) / (d + 1 - 2b))."""
    from flint import arb, ctx, fmpq

    d = m + k * N_RING + 1
    ctx.prec = prec
    num = arb(fmpq(m, d)) * arb(Q).log() - arb(fmpq(sigma2.numerator, sigma2.denominator)).log() / 2 - arb(b).log() / 2
    return (num / (d + 1 - 2 * b)).exp()


def detection_entry(d: int, b: int, S: float, model: str, eps: float = 0.0) -> float:
    """log of the detection block's GH at (d, b) for a lattice of log volume S: on the GSA line, (2b - d - 1) L(b)/(b - 1) + S/d (condition (9)'s
    right side); on the tight profile, l^tight_{d-b+1} (double precision, O(d))."""
    from latticelab.profile_floor import log_chat, tight_entry_float

    if model == "gsa":
        return (2 * b - d - 1) * log_chat(b) / (b - 1) + S / d
    if model == "tight":
        return tight_entry_float(d, b, d - b + 1, eps, S)
    raise ValueError("model must be 'gsa' or 'tight'")


def detection_chain(k: int, eta1: int, model: str = "tight", eps: float = 0.0, b_lo: int = 300, b_hi: int = 1100,
                    m_range: Optional[Tuple[int, int]] = None) -> Dict:
    """The least b in [b_lo, b_hi] for which some m in [0, (k+1) n] (or m_range) satisfies the detection condition  log(sigma sqrt b) <=
    detection_entry(d, b, m log q, model)  with d = m + kn + 1 and 2b < d + 1 (the standard domain of (9)).  model='gsa' is the specification's
    chain in this form (it reproduces `gsa_chain`); model='tight' is the prefix-volume floor's form.  Double precision."""
    sigma2 = Fraction(eta1, 2)
    lo_m, hi_m = (0, (k + 1) * N_RING) if m_range is None else m_range
    best_m_per_b = {}
    for b in range(b_lo, b_hi + 1):
        rows = []
        lhs = 0.5 * math.log(float(sigma2) * b)
        for m in range(lo_m, hi_m + 1):
            d = m + k * N_RING + 1
            if d + 1 - 2 * b <= 0:
                continue
            rows.append((detection_entry(d, b, m * math.log(Q), model, eps) - lhs, m, d))
        if not rows:
            best_m_per_b[b] = None
            continue
        rows.sort(reverse=True)
        best_m_per_b[b] = {"m": rows[0][1], "d": rows[0][2], "margin": rows[0][0]}
        ok = [r for r in rows if r[0] >= 0]
        if ok:
            return {"b": b, "m_best": ok[0][1], "d_best": ok[0][2], "margin_best": ok[0][0], "admissible_m": sorted(x[1] for x in ok),
                    "n_admissible_m": len(ok), "sigma2": sigma2, "k": k, "model": model, "eps": eps, "m_range": [lo_m, hi_m], "best_m_per_b": best_m_per_b}
    return {"b": None, "k": k, "sigma2": sigma2, "model": model, "eps": eps, "m_range": [lo_m, hi_m], "best_m_per_b": best_m_per_b,
            "note": f"no b in [{b_lo}, {b_hi}] succeeds"}


def certify_detection_chain(k: int, eta1: int, det_result: Dict, b_lo: int, eps=0, prec: int = 256, log=None, m_stride: int = 1) -> Dict:
    """Rigorous decisions for the detection chain found by `detection_chain(model='tight')`: the crossing (b*, m*) passes (lower(tight entry
    ball) >= upper(left-side ball)), and every (b, m) with b in [b_lo, b* - 1] and m in [0, (k+1) n] with 2b < d + 1 (every m_stride-th m if
    m_stride > 1, reported) fails (upper(entry) < lower(left side)); the left side (1/2) log(sigma2 b) is an arb ball from exact rationals, the
    entry `profile_floor.tight_entry` with the exact linear certificate.  Stops at the first non-failing pair.  A full-range leastness
    certificate (`certified`) is issued only for m_stride == 1; with a larger stride the verdict is `certified_sampled`, whose scope is the
    sampled m only and which claims nothing about the unvisited sample counts."""
    from flint import arb, ctx, fmpq

    from latticelab.profile_floor import tight_entry

    if not (isinstance(m_stride, int) and m_stride >= 1):
        raise ValueError("m_stride must be a positive integer")

    sigma2 = Fraction(eta1, 2)
    b_star, m_star = det_result["b"], det_result["m_best"]
    d_star = m_star + k * N_RING + 1
    m_max = (k + 1) * N_RING

    def decide(b, m):
        d = m + k * N_RING + 1
        ctx.prec = prec
        lhs = (arb(fmpq(sigma2.numerator, sigma2.denominator)) * b).log() / 2
        S = arb(m) * arb(Q).log()
        ent = tight_entry(d, b, d - b + 1, eps, 0, prec) + S / d  # the entry at log volume S: the certificate's z is 1/d
        if ent.lower() >= lhs.upper():
            return "passes", ent, lhs
        if ent.upper() < lhs.lower():
            return "fails", ent, lhs
        return "undecided", ent, lhs

    verdict, ent, lhs = decide(b_star, m_star)
    if log:
        log(f"detection b*={b_star} m*={m_star} d={d_star}: entry {ent} vs log(sigma sqrt b) {lhs}: {verdict}")
    out = {"b_star": b_star, "m_star": m_star, "d_star": d_star, "crossing": verdict, "earlier_b": {}, "prec": prec, "eps": eps, "b_lo": b_lo,
           "m_range": [0, m_max], "m_stride": m_stride}
    all_fail = True
    for b in range(b_lo, b_star):
        n_fail, skipped, first_bad = 0, 0, None
        for m in range(0, m_max + 1, m_stride):
            d = m + k * N_RING + 1
            if d + 1 - 2 * b <= 0:
                skipped += 1
                continue
            v, ent, lhs = decide(b, m)
            if v != "fails":
                first_bad = {"m": m, "verdict": v}
                all_fail = False
                if log:
                    log(f"detection b={b} m={m} d={d}: {v} (entry {ent}, lhs {lhs})")
                break
            n_fail += 1
        out["earlier_b"][b] = {"n_decided_fail": n_fail, "n_skipped_direction": skipped, "first_non_failing": first_bad}
        if log:
            log(f"detection b={b}: {n_fail} values of m rigorously fail ({skipped} outside the condition's domain); first non-failing: {first_bad}")
        if first_bad is not None:
            break
    all_pass_and_fail = verdict == "passes" and all_fail
    certified = all_pass_and_fail and m_stride == 1
    out["certified"] = certified
    out["certified_sampled"] = all_pass_and_fail and m_stride > 1
    out["all_earlier_b_fail_sampled_m"] = all_fail
    if certified:
        out["certification_scope"] = (f"rigorous (exact linear certificate of the tight entry, arb balls, directed comparisons): the crossing (b*={b_star}, m*={m_star}, "
                                      f"d={d_star}) passes and every (b, m) with b in [{b_lo}, {b_star - 1}] and m in [0, {m_max}] (2b < d + 1) fails, so "
                                      f"{b_star} is the least blocksize of the detection chain within [{b_lo}, {b_star}] x [0, {m_max}]. NOT certified: b < {b_lo}.")
    elif out["certified_sampled"]:
        out["certification_scope"] = (f"SAMPLED, not a leastness certificate: the crossing (b*={b_star}, m*={m_star}, d={d_star}) passes rigorously and every (b, m) with b in "
                                      f"[{b_lo}, {b_star - 1}] and m in the sampled set (every {m_stride}-th m of [0, {m_max}], 2b < d + 1) fails rigorously; the unvisited m "
                                      f"were not decided, so nothing is claimed about them.")
    else:
        out["certification_scope"] = (f"NOT certified: crossing verdict '{verdict}', every earlier (b, sampled m) fails: {all_fail}; see earlier_b.")
    return out


def certify_floor_chain(k: int, eta1: int, floor_result: Dict, b_lo: int, eps=0, prec: int = 256, log=None) -> Dict:
    """Rigorous decisions for the floor chain found by `floor_chain`: the crossing (b*, m*) passes, and every (b, m) with b in [b_lo, b* - 1] and m
    in [0, (k+1) n] with 2b < d + 1 fails (directed arb comparisons of the exact-dual floor ball with the target ball).  Stops at the first
    non-failing pair.  Returns the verdicts and a `certification_scope` derived from them."""
    from latticelab.profile_floor import floor_l1

    sigma2 = Fraction(eta1, 2)
    b_star, m_star = floor_result["b"], floor_result["m_best"]
    d_star = m_star + k * N_RING + 1
    m_max = (k + 1) * N_RING

    def decide(b, m):
        d = m + k * N_RING + 1
        fb = floor_l1(d, b, eps, 0, prec)["root_hermite_floor_ball"]
        t = target_ball(b, m, k, sigma2, prec)
        if fb.upper() <= t.lower():
            return "passes", fb, t
        if fb.lower() > t.upper():
            return "fails", fb, t
        return "undecided", fb, t

    verdict, fb, t = decide(b_star, m_star)
    if log:
        log(f"b*={b_star} m*={m_star} d={d_star}: floor {fb} vs target {t}: {verdict}")
    out = {"b_star": b_star, "m_star": m_star, "d_star": d_star, "crossing": verdict, "earlier_b": {}, "prec": prec, "eps": eps, "b_lo": b_lo,
           "m_range": [0, m_max]}
    all_fail = True
    for b in range(b_lo, b_star):
        n_fail, skipped, first_bad = 0, [], None
        for m in range(0, m_max + 1):
            d = m + k * N_RING + 1
            if d + 1 - 2 * b <= 0:
                skipped.append(m)
                continue
            v, fb, t = decide(b, m)
            if v != "fails":
                first_bad = {"m": m, "verdict": v}
                all_fail = False
                if log:
                    log(f"b={b} m={m} d={d}: {v} (floor {fb}, target {t})")
                break
            n_fail += 1
        out["earlier_b"][b] = {"n_decided_fail": n_fail, "n_skipped_direction": len(skipped), "first_non_failing": first_bad}
        if log:
            log(f"b={b}: {n_fail} values of m rigorously fail ({len(skipped)} outside the condition's domain); first non-failing: {first_bad}")
        if first_bad is not None:
            break
    certified = verdict == "passes" and all_fail
    out["certified"] = certified
    out["all_earlier_b_fail_all_m"] = all_fail
    if certified:
        out["certification_scope"] = (f"rigorous (exact rational dual, arb balls, directed comparisons): the crossing (b*={b_star}, m*={m_star}, d={d_star}) "
                                      f"passes and every (b, m) with b in [{b_lo}, {b_star - 1}] and m in [0, {m_max}] (2b < d + 1) fails, so {b_star} is the least "
                                      f"blocksize of the floor chain within [{b_lo}, {b_star}] x [0, {m_max}]. NOT certified: b < {b_lo}, excluded by the double-precision "
                                      f"screen only (the floor is not monotone in b and the target moves with b and d).")
    else:
        out["certification_scope"] = (f"NOT certified: crossing verdict '{verdict}', every earlier (b, m) fails: {all_fail}; see earlier_b. "
                                      f"The double-precision screen stands alone.")
    return out


def main(argv=None):
    """CLI: `python -m latticelab.spec_chain --sets Kyber512 --eps 0 --margin 3 --out results/lattice_spec_chain.json`: the GSA chain and the floor
    chain, both over the full m range in double precision, and the rigorous leastness certificate of the floor chain over [b_gsa - margin, b*] x
    [0, (k+1) n] described in the module docstring."""
    import argparse
    import json
    import os
    import time

    ap = argparse.ArgumentParser(description="Kyber round-3 blocksize chain with the GSA delta and with the certified floor")
    ap.add_argument("--sets", nargs="+", default=list(KYBER))
    ap.add_argument("--eps", default="0")
    ap.add_argument("--margin", type=int, default=3, help="the certified b-range starts this many blocksizes below the GSA chain's b")
    ap.add_argument("--no-certify", action="store_true")
    ap.add_argument("--detection", action="store_true", help="run the detection chain (the prefix-volume floor's form of condition (9)) instead of the delta-substitution chain")
    ap.add_argument("--b-hi-margin", type=int, default=40, help="detection chain: the scan runs up to the GSA chain's b plus this margin")
    ap.add_argument("--m-stride", type=int, default=1, help="detection chain certification: decide every m-th sample count of the earlier blocksizes (1 = all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    eps_f = float(Fraction(a.eps))
    out = json.load(open(a.out)) if a.out and os.path.exists(a.out) else {"note": "Kyber round-3 primal chain (spec 5.1.2, condition (9), m optimised over "
                                                                              "the full range): least blocksize with the Chen-Nguyen delta and with the profile floor in its place; "
                                                                              "see certification_scope for what is rigorous", "rows": []}
    if a.detection:
        out.setdefault("detection_rows", [])
        out["note_detection"] = ("detection chain: the prefix-volume floor's form of the primal attack's condition -- log(sigma sqrt b) <= l^tight_{d-b+1}(d, b, eps), the tight "
                                 "profile's entry at the detection position (an upper bound on the detection block's log GH over all admissible profiles) -- against the "
                                 "GSA-line control in the same form; see certification_scope")
        for name in a.sets:
            p = KYBER[name]
            t0 = time.time()
            g = detection_chain(p["k"], p["eta1"], "gsa")
            t = detection_chain(p["k"], p["eta1"], "tight", eps_f, b_lo=g["b"], b_hi=g["b"] + a.b_hi_margin)
            row = {"set": name, "k": p["k"], "eta1": p["eta1"], "sigma2": str(Fraction(p["eta1"], 2)), "printed_d_beta": p["printed"], "eps": a.eps,
                   "gsa_detection": {"b": g["b"], "m": g["m_best"], "d": g["d_best"], "n_admissible_m": g["n_admissible_m"]},
                   "tight_detection_float": {"b": t["b"], "m": t.get("m_best"), "d": t.get("d_best"), "n_admissible_m": t.get("n_admissible_m"),
                                             "best_m_per_b": {str(b): v for b, v in t["best_m_per_b"].items()}}}
            if t["b"] is not None:
                row["classical_bits_0.292"] = {"gsa": 0.292 * g["b"], "tight": 0.292 * t["b"]}
                print(f"{name}: GSA detection chain b = {g['b']} (m = {g['m_best']}, d = {g['d_best']}); prefix-volume floor detection chain (eps = {a.eps}) b = {t['b']} "
                      f"(m = {t['m_best']}, d = {t['d_best']}, {t['n_admissible_m']} admissible m): +{t['b'] - g['b']} blocksizes, {0.292 * (t['b'] - g['b']):.1f} classical core-SVP bits", flush=True)
                if not a.no_certify:
                    row["certification"] = certify_detection_chain(p["k"], p["eta1"], t, max(2, g["b"] - a.margin), a.eps, log=lambda s: print("   " + s, flush=True), m_stride=a.m_stride)
            else:
                print(f"{name}: GSA detection chain b = {g['b']}; prefix-volume floor detection chain: no b up to {g['b'] + a.b_hi_margin} passes", flush=True)
            row["seconds"] = time.time() - t0
            out["detection_rows"] = [r for r in out["detection_rows"] if not (r["set"] == name and r["eps"] == a.eps)] + [row]
            if a.out:
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(out, open(a.out, "w"), indent=1, default=str)
        print("DETECTION_DONE", flush=True)
        return
    for name in a.sets:
        p = KYBER[name]
        t0 = time.time()
        g = gsa_chain(p["k"], p["eta1"])
        f = floor_chain(p["k"], p["eta1"], eps_f, b_lo=g["b"])  # the floor dominates the GSA on this range, so b_lo = the GSA chain's b
        row = {"set": name, "k": p["k"], "eta1": p["eta1"], "sigma2": str(Fraction(p["eta1"], 2)), "printed_d_beta": p["printed"], "eps": a.eps,
               "gsa_chain": {"b": g["b"], "m": g["m_best"], "d": g["d_best"], "n_admissible_m": g["n_admissible_m"], "m_range": g["m_range"]},
               "floor_chain_float": {"b": f["b"], "m": f["m_best"], "d": f["d_best"], "n_admissible_m": f["n_admissible_m"], "m_range": f["m_range"],
                                     "best_m_per_b": {str(b): v for b, v in f["best_m_per_b"].items()}},
               "classical_bits_0.292": {"gsa": 0.292 * g["b"], "floor": 0.292 * f["b"]}}
        print(f"{name}: printed (d, b) = {p['printed']}; GSA chain b = {g['b']} (m = {g['m_best']}, d = {g['d_best']}, {g['n_admissible_m']} admissible m); "
              f"floor chain (eps = {a.eps}, double precision, full m range) b = {f['b']} (m = {f['m_best']}, d = {f['d_best']}, {f['n_admissible_m']} admissible m): "
              f"+{f['b'] - g['b']} blocksizes, {0.292 * (f['b'] - g['b']):.1f} classical core-SVP bits", flush=True)
        if not a.no_certify:
            row["certification"] = certify_floor_chain(p["k"], p["eta1"], f, max(2, g["b"] - a.margin), a.eps, log=lambda s: print("   " + s, flush=True))
        row["seconds"] = time.time() - t0
        out["rows"] = [r for r in out["rows"] if not (r["set"] == name and r["eps"] == a.eps)] + [row]
        if a.out:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(out, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
