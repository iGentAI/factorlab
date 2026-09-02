# Testing guide

## Layout

The suite is flat: one file per module or experiment, named `test_<module>.py`, all under `tests/`. There is no `conftest.py`; every
test builds its own inputs, and every random choice flows from an explicit seed (`factorlab.gen.make_semiprime(nbits, family, seed,
index)`), so a failure reproduces exactly. Tests exercise the code, not the archives, with four exceptions that cross-check a small
recomputation against the archived value: `test_resonant_census.py` requires `results/e31_planar_census.json` and
`results/e32_resonant_census.json`; `test_theorem_checks.py` and `test_uniform_slack.py` compare against
`results/e56_theorem_checks.json` and `results/lattice_uniform_slack_screen.json` when present; `test_audit_detection_simulator.py`
skips one test when `results/lattice_l6_strict.json` is absent.

| area | files |
|---|---|
| factoring harness (`factorlab/`) | `test_gen.py`, `test_algorithms.py`, `test_audit.py`, `test_bench.py`, `test_cli.py`, `test_smoothness.py`, `test_probabilistic.py` |
| factoring experiments (`factorlab/experiments/`) | `test_experiments.py`, `test_beyond.py`, `test_frontier.py`, `test_separable.py` and one `test_<experiment>.py` per later experiment |
| lattice arm (`latticelab/`) | `test_latticelab.py`, `test_certify_audit.py`, `test_deficits.py`, `test_qceiling.py`, `test_uniform_slack.py`, `test_audit_detection_simulator.py` |

## Running

From the repository root (`pytest.ini` sets `testpaths = tests` and `-q`):

    python3 -m pytest            # the whole suite: 357 tests, about a minute
    python3 -m pytest -m "not slow"   # skip the statistical audits (a few seconds each)
    python3 -m pytest tests/test_latticelab.py -k floor   # one file, one keyword

The only marker is `slow`, declared in `pytest.ini`.

## Environment

The lattice tests need `fpylll`, which needs the system packages named in `README.md`. Pruned BKZ also needs fpylll's strategies
file at the path fpylll was built with; if a test crashes inside the extension (the `cysignals` crash handler starts and attaches
`gdb`) rather than failing, the file is missing. Link the system copy as `README.md` describes:

    sudo mkdir -p /project/local/share/fplll/strategies
    sudo ln -s /usr/share/libfplll8/strategies/default.json /project/local/share/fplll/strategies/default.json

## Conventions

- A test asserts on exact values where the computation is exact (integers, rationals, ball endpoints) and on tolerances only where the
  code itself uses floating point.
- Statistical checks are marked `slow` and use fixed seeds; do not loosen a tolerance to make one pass.
- A new experiment module gets a `tests/test_<module>.py` that exercises its pure functions on small inputs; long runs belong in
  `REPRODUCING.md`, not in the suite.
