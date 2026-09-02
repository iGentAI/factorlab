# factorlab and latticelab

Code, tests and archived results behind two technical reports by Sean Ward and Maestro (iGent AI, 2026), distributed separately
as PDFs:

- *Floors for deterministic integer factoring: covering models, partial information, the additive structure of the Lehman–Harvey
  start set, and three further floors* — a covering model containing Harvey's exponent-one-fifth algorithm, its floors $N^{1/3}$,
  $N^{1/5}$ and $N^{1/6}$ and their robustness; sharp floors and matching algorithms given partial information about a factor; the
  additive structure of the window starts that any explicit difference cover must respect; a deterministic curve for moduli whose
  $p-1$ and $q-1$ share a large factor; exposure labels for a fixed list of elliptic curves; coefficient floors for number-field-sieve
  polynomial selection.
- *Certified conditional profile optima for block reduction, and the Kyber round-3 primal chain re-evaluated* — a linear programme over
  Gram–Schmidt profiles under the block Gaussian-heuristic inequalities, its exact dual certificates, the prefix-volume floor, the
  round-3 Kyber primal chain re-evaluated with certified crossings, and an audit of the class on real reduced bases.

Every measured or certified statement in either report has a row in `REPRODUCING.md` naming the function that produced it, the
archive under `results/` that holds it, and the command that regenerates it. Nothing here is a lower bound on factoring or on
attacking a deployed scheme; every floor is a theorem inside a stated model, and the reports say so.

## Layout

| path | contents |
|---|---|
| `factorlab/` | the factoring harness: `numth` (gmpy2 primitives), `gen` (seeded semiprime generation by rejection sampling), `result`/`registry`/`audit`/`bench` (work counters, factor validation, registration, benchmarks), `algorithms/` (trial division, Fermat, Lehman, Hart, SQUFOF, Pollard rho, Pollard–Strassen, babystep–giantstep on $p+q$, ECM, a quadratic sieve, Schnorr–Lenstra class groups, a deterministic fixed-list ECM), `experiments/` (one module per experiment of the factoring report; each has a `__main__` or a `run_*` driver) |
| `latticelab/` | the lattice arm: `profile_floor` (the profile floor, its $O(d)$ exact dual certificate, interval-arithmetic decisions), `check_certificate` (an independent checker), `spec_chain` (the round-3 chain and the detection chain with certification), `head_slack`, `uniform_slack`, `qceiling` (the $q$-ary ceiling, the head-clipped extremal profile, the dual route), `simulator_chain` (the detection condition on Chen–Nguyen and Bai–Stehlé–Wen simulated profiles), `certify_audit` and `audit_detection` (exact certification of the audit bases and their weighted deficits), `dual_census`, `residual`, `insertion`, `schedule_game`, `poisson_world`, `lattices`, `sieve`, `profile` |
| `tests/` | the test suite (`python -m pytest -q`; 366 tests, about a minute) |
| `results/` | the archives: JSON and JSONL outputs of every experiment and certification named in the reports, with the logs of the long runs; these are the datasets of the release |
| `REPRODUCING.md` | statement → producing function → archive → command, for both reports |
| `requirements.txt` | the pinned Python stack the archives were produced with |
| `CITATION.cff` | how to cite |

## Installation

Ubuntu 24.04, Python 3.12. System packages for the lattice arm (fpylll):

    sudo apt-get install -y libfplll-dev fplll-tools libgmp-dev libmpfr-dev libqd-dev

Python stack, pinned to the versions the archives were produced with:

    python3 -m pip install --user -r requirements.txt

fpylll's pruned reduction needs its strategies file; if `SVP.shortest_vector` or pruned BKZ reports "Cannot open strategies file",
link the system file into the path fpylll expects:

    sudo mkdir -p /project/local/share/fplll/strategies
    sudo ln -s /usr/share/libfplll8/strategies/default.json /project/local/share/fplll/strategies/default.json

On macOS the binary wheel's strategies path points into its own build tree (`/Users/runner/work/fpylll/...`), so the symlink does not
apply and exact-SVP calls raise `FileNotFoundError` (the suite crashes in `tests/test_schnorr_lattice.py`). Install fplll with
Homebrew (`brew install fplll`) and reinstall fpylll from its source distribution against that library, e.g.
`pip install --no-binary fpylll --force-reinstall fpylll==0.6.4` with Homebrew's include and lib directories on `CFLAGS`/`LDFLAGS` if
the build does not find them; the suite then passes on arm64 with Python 3.12.

Run the tests from the repository root:

    python3 -m pytest -q

## Reproducing

`REPRODUCING.md` gives, for each statement, an executed command. Short computations run in seconds to minutes; the long ones
(exhaustive certification of the detection chains, the strict-tour censuses, the head-slack and uniform-slack certificates) take
hours and write their archives incrementally. All work is counted in machine-independent units (modular multiplications, gcds,
candidates, babies and giants, BKZ tours, oracle calls), never wall time, and every experiment is seeded; seeds are in the
reproduction guide and in the archives.

## Conventions

- *Certified* means an exact rational dual certificate together with interval (ball) arithmetic from exact rational inputs and a
  directed comparison; a double-precision scan is a pre-screen and is labelled as such. A dual-feasible bound certifies a failure,
  never a pass; a pass needs a primal witness.
- Archives record exact values where the computation is exact (integers, rationals, ball endpoints as exact rationals) and say
  where floating point was used.
- Every negative literature claim in the reports names the nearest related work; every priority claim was searched on the statement.

## Authors and citation

Sean Ward and Maestro, iGent AI. Maestro is a research system; the work was produced in an interactive session and every result
was checked by computation and by adversarial review before it entered a report. See `CITATION.cff`.

## Licence

MIT; see `LICENSE`. The licence covers the code and the archived results alike.
