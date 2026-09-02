"""Benchmark driver: run algorithms over reproducible suites, fit scaling exponents.

A *scaling fit* regresses log(work) (or log(wall)) against log(N) over a
range of bit sizes and reports the slope -- the empirical exponent e in
``work ~ N^e`` -- with a standard error.  Because ``N`` spans many orders of
magnitude, these fits are robust even with modest sample counts.

Results are appended as JSON lines to ``results/<experiment>.jsonl`` so every
number in the research log can be regenerated.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .gen import make_semiprime, Semiprime
from .registry import get_algorithm
from .result import FactorResult

RESULTS_DIR = os.environ.get("FACTORLAB_RESULTS", "results")


@dataclass
class Fit:
    exponent: float
    stderr: float
    intercept: float
    n_points: int
    r2: float

    def __str__(self) -> str:
        return f"N^{self.exponent:.4f} (+/- {self.stderr:.4f}, r2={self.r2:.3f}, n={self.n_points})"


def fit_exponent(Ns: Iterable, ys: Iterable[float]) -> Fit:
    """Least squares of log y on log N.  Points with y <= 0 are dropped."""
    x, y = [], []
    for N, v in zip(Ns, ys):
        if v is not None and v > 0:
            x.append(math.log(int(N)))
            y.append(math.log(v))
    if len(x) < 3:
        return Fit(float("nan"), float("nan"), float("nan"), len(x), float("nan"))
    x, y = np.array(x), np.array(y)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coef
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-300
    dof = max(1, len(x) - 2)
    sigma2 = ss_res / dof
    var_slope = sigma2 / float(((x - x.mean()) ** 2).sum() or 1e-300)
    return Fit(float(slope), float(math.sqrt(var_slope)), float(intercept), len(x), 1 - ss_res / ss_tot)


@dataclass
class RunRecord:
    instance: Semiprime
    result: FactorResult

    def to_json(self) -> dict:
        return {"instance": self.instance.to_json(), "result": self.result.to_json()}


def run_suite(algorithm: str, nbits_list: Iterable[int], count: int = 5, family: str = "balanced",
              seed: int = 0, params: Optional[dict] = None, gen_params: Optional[dict] = None,
              experiment: Optional[str] = None, verbose: bool = True,
              timeout_per_instance: Optional[float] = None) -> list[RunRecord]:
    """Run ``algorithm`` on ``count`` instances at each bit size.

    ``timeout_per_instance`` is a soft budget: if the mean wall time at one bit
    size exceeds it, larger sizes are skipped (keeps exploratory sweeps cheap).
    """
    algo = get_algorithm(algorithm)
    params = params or {}
    gen_params = gen_params or {}
    records: list[RunRecord] = []
    out_path = None
    if experiment:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"{experiment}.jsonl")
    for nb in nbits_list:
        walls = []
        for i in range(count):
            inst = make_semiprime(nb, family, seed, i, **gen_params)
            res = algo(inst.N, **params)
            if res.found and not (res.p == inst.p or res.p == inst.q):
                raise AssertionError(f"{algorithm} returned wrong factor for {inst}")
            rec = RunRecord(inst, res)
            records.append(rec)
            walls.append(res.wall)
            if out_path:
                with open(out_path, "a") as fh:
                    fh.write(json.dumps({"algorithm": algorithm, "params": params,
                                         "gen_params": gen_params, **rec.to_json()}) + "\n")
        if verbose:
            found = sum(r.result.found for r in records if r.instance.nbits == nb)
            work = [r.result.primary_work for r in records if r.instance.nbits == nb]
            print(f"{algorithm:18s} {nb:4d}b  found {found}/{count}  "
                  f"wall {statistics.mean(walls):9.4f}s  {algo.primary_key} median {statistics.median(work):.3g}")
        if timeout_per_instance and statistics.mean(walls) > timeout_per_instance:
            if verbose:
                print(f"  stopping sweep: mean wall {statistics.mean(walls):.2f}s > budget")
            break
    return records


def summarize(records: list[RunRecord], use_wall: bool = False) -> dict:
    """Per-bit-size medians and an exponent fit over medians."""
    by_bits: dict[int, list[RunRecord]] = {}
    for r in records:
        by_bits.setdefault(r.instance.nbits, []).append(r)
    rows = []
    for nb in sorted(by_bits):
        rs = by_bits[nb]
        vals = [(r.result.wall if use_wall else r.result.primary_work) for r in rs if r.result.found]
        rows.append({
            "nbits": nb,
            "count": len(rs),
            "found": sum(r.result.found for r in rs),
            "median": statistics.median(vals) if vals else None,
            "mean": statistics.mean(vals) if vals else None,
            "max": max(vals) if vals else None,
            "N_typical": 2 ** nb,
        })
    fit = fit_exponent([row["N_typical"] for row in rows], [row["median"] for row in rows])
    return {"rows": rows, "fit": fit, "metric": "wall" if use_wall else "work"}


def print_summary(summary: dict) -> None:
    print(f"metric={summary['metric']}  fit: {summary['fit']}")
    for row in summary["rows"]:
        med = row["median"]
        print(f"  {row['nbits']:4d}b  found {row['found']}/{row['count']}  median {med if med is None else f'{med:.4g}'}")
