# Repository cleanup report - 2026-09-02

**Scope**: whole repository (code, dependencies, documentation, tests). **Safety mode**: safe - nothing was removed without a usage
check, no data under `results/` was touched, and every change was validated by the full test suite. Produced by the `repo-cleanup`
workflow (baseline, per-area references, usage checks, validation, report).

## Summary

| area | finding | action |
|---|---|---|
| build / OS / IDE artefacts | none tracked (no `__pycache__`, `.pytest_cache`, `.DS_Store`, swap or backup files; no empty files) | none needed |
| unused imports | 50 unused imports and 1 redefinition (`ruff --select F401,F811`) across 33 files | removed |
| unused local variables | 7 (`ruff --select F841`) across 5 files | removed |
| dead modules | none: every module is imported by another module or a test, or is a `run_*` / `__main__` driver named in the README | none needed |
| dead functions (`vulture`, confidence >= 80 %) | one hit, `latticelab/flag.py` `tracer` - a parameter required by fpylll's `svp_call` signature (false positive) | kept |
| unimported pinned packages | `numba`, `matplotlib`, `pypdf` are imported by no file under `factorlab/`, `latticelab/` or `tests/` | removed from `requirements.txt` |
| documentation drift | stale test count (338 vs 357 actual), TeX toolchain listed though no TeX source is in the repo, an empty "Building the reports" section claiming PDFs are committed | fixed in `REPRODUCING.md` |
| test smells | no focused tests, no assertion-free files, no duplicate test names, one conditional skip (archive-dependent) | none needed; `tests/README.md` added |
| `.gitignore` | no coverage, tool-cache, virtualenv, editor or OS patterns | extended |
| sprint artefacts | none (no PLAN/SUMMARY/REPORT/TODO documents, no scratch or backup files) | not applicable |

Repository size is 1.9 MB excluding `.git`, of which 1.7 MB is the published `results/` archive; there was no disk bloat to recover.
The pruned Python packages represent roughly 80 MB of installed wheels (`llvmlite` alone is 60 MB) that a fresh environment no longer
fetches.

## Baseline

`python -m pytest -q` from a clean checkout on Ubuntu 24.04 / Python 3.12.3 with the pinned stack: **357 passed in 46 s**.

A first attempt crashed inside the fpylll extension (the `cysignals` crash handler attached `gdb` and hung) because the fpylll
strategies file was not at `/project/local/share/fplll/strategies/default.json`. That is the environment step `README.md` already
documents; after linking the system file the suite ran cleanly. `tests/README.md` now describes the symptom and the fix.

## Actions taken

### Code

Unused imports removed (ruff F401/F811, mechanical, reviewed line by line - no re-export of any removed name exists anywhere in the
codebase):

- `factorlab/`: `audit.py`, `bench.py`, `algorithms/qs.py`, `experiments/{barrier, chirp_blocks, chirp_dynamics, classgroup,
  common_factor, energy_stats, farey_cover, harvey_residue, local_joint, nonlinear_pairs, prime_subfamily, residue_ladder,
  run_barrier, run_d1_n1_n2, run_smoothness, smooth_profiles}.py`
- `latticelab/`: `check_certificate.py`, `deficits.py`, `head_slack.py`, `insertion.py`, `poisson_world.py`, `profile.py`,
  `qceiling.py`
- `tests/`: `test_average_case_pm1.py`, `test_balanced_structure.py`, `test_chirp_dynamics.py`, `test_gen.py`,
  `test_harvey_residue.py` (also a redundant inner `import math`), `test_olf_squares.py`, `test_planar_census.py`

Unused local variables removed (ruff F841):

| file | variable | note |
|---|---|---|
| `factorlab/experiments/arms_e45.py` | `xi_num, xi_den` | aliases of `p, q`, never read |
| `factorlab/experiments/classgroup.py` | `h_odd` (in `actual_algorithm_experiment`) | the recorded `h_odd` in `classgroup_experiment` is a different local and is untouched |
| `factorlab/experiments/frobenius_defect.py` | `ctx` (in `leakage_over_ZN`) | `frobenius_defect_poly` builds its own context |
| `factorlab/experiments/nonlinear_pairs.py` | `N` (in `roots_mod_N`) | computed, never read |
| `tests/test_smoothness.py` | `Xj, Zj` | never read |

Nothing else in the code was changed. `print` calls in `factorlab/__main__.py`, `audit.py` and `bench.py` are CLI output inside
`main()`/`verbose` branches, not debugging residue, and were kept. There are no TODO/FIXME/XXX/HACK markers, no `breakpoint()`/`pdb`,
no `sys.path` manipulation and no commented-out code blocks.

### Dependencies

| package | pinned | imported by | action |
|---|---|---|---|
| `numba` (pulls `llvmlite`) | 0.67.0 | nothing | removed |
| `matplotlib` (pulls `contourpy`, `cycler`, `fonttools`, `kiwisolver`) | 3.11.1 | nothing | removed; figure generation is not part of this repository |
| `pypdf` | 6.12.2 | nothing | removed; belonged to the report build, which is not part of this repository |
| `mpmath` | 1.3.0 | nothing directly; sympy's numerical backend | kept, with a comment saying why it is pinned |

All other pins (`gmpy2`, `python-flint`, `fpylll`, `cypari2`, `numpy`, `scipy`, `sympy`, `pytest`) are imported. Verification: the
three packages and their transitive-only dependencies were uninstalled from the test environment and the full suite re-run.

### Documentation

`REPRODUCING.md`:

- Environment: dropped TeX Live and `poppler-utils`; nothing in the repository uses them.
- Test count corrected from 338 to 357, with a pointer to `tests/README.md`.
- "Building the reports" (an empty command block followed by "The two PDFs are committed beside their sources", which they are not)
  replaced by "The reports": the PDFs are distributed separately and the TeX sources are not in this repository.

`README.md` was already consistent with the code (357 tests, correct install steps) and was not changed.

New: `tests/README.md` (layout, how to run, the `slow` marker, environment prerequisites, which tests read archives, conventions).

### `.gitignore`

Added `.coverage`, `.coverage.*`, `htmlcov/`, `.hypothesis/`, `.mypy_cache/`, `.ruff_cache/`, `.venv/`, `venv/`, `.vscode/`,
`.idea/`, `*.swp`, `*.swo`, `.DS_Store`, `Thumbs.db`, `.cleanup-archive/`. Deliberately **not** added: `*.log` and `*.json` -
`results/*.log` and `results/*.json` are the published archives; the file now says so.

## Validation

| check | result |
|---|---|
| `ruff check --select F factorlab latticelab tests` | all checks passed (58 findings before) |
| `python -m compileall factorlab latticelab tests` | ok |
| import of all 103 modules under `factorlab/`, `latticelab/` | 0 failures |
| `pip install --dry-run -r requirements.txt` | resolves |
| full suite with `numba`, `matplotlib`, `pypdf` uninstalled | **357 passed in 46.34 s** (baseline 357 passed in 46 s) |

## Catalogued, not changed

These are judgement calls for the authors; each is reversible or additive and none affects correctness.

1. **88 of 175 archives under `results/` are not named in `README.md` or `REPRODUCING.md`.** They are data and were not touched. Most
   map by name to a producing module (`e10_classgroup.json` -> `experiments/classgroup.py`, `n1_frobenius_defect.json` ->
   `experiments/frobenius_defect.py`, `lattice_poisson_world_*.json` -> `latticelab/poisson_world.py`, `baseline_*.jsonl` ->
   `bench.py` over the registered algorithms, the 18 `e22_fixed_scale_*.json` -> `run_fixed_scale`, and so on); a few do not
   (`barrier_coverage_lemma.json`, `barrier_frobenius_degree.json`, `d4_conditional_residues.json`, `e13_hard_sets.json`,
   `e28_planar_regime.json`, `fig_information_axes.json`, `lattice_dual_constraints_*.json`, `lattice_l1_l2.json`,
   `lattice_l6_{lll_control,scaling,selfdual,tours}.json`, `lattice_t4_orbit_angles.json`, `lattice_tour_ledger.json`). A short
   `results/README.md` index (archive -> producer -> status: cited / supporting / superseded) would close the gap; producing it
   requires provenance the repository does not record, so it is left to the authors.
2. **Docstrings cite working notes that are not in the repository**: `docs/notes_barrier.md` (12 references), `docs/notes_lattice_barrier.md` (7),
   `docs/lattice_barrier_plan.md` (6, including the `latticelab/__init__.py` package docstring), `notes_probabilistic.md` (5),
   `notes_beyond_gnfs.md` (4). They carry proposition names and were left intact; either ship the notes under `docs/` or reword the
   references to the corresponding report sections.
3. **`tests/test_latticelab.py` is 984 lines / 20 tests.** Above the usual 500-line guideline; splitting by module
   (`profile_floor`, `spec_chain`, `sieve`, ...) is a mechanical follow-up with no functional gain, so it was not done here.
4. **`REPRODUCING.md` carries an "Added 2 September 2026" addendum** whose rows belong to the lattice and factoring tables above it.
   Merging them is an editorial choice for the authors.
5. **Flat test layout.** The skill's unit/integration split was not imposed: the suite is 47 files with one-file-per-module naming,
   documented commands (`python -m pytest -q`) depend on the current layout, and there is no integration tier to separate.

## Prevention

- Run `ruff check --select F` (or `python -m pyflakes`) before committing; the 58 findings here were all of the kind it catches.
- Keep `requirements.txt` to packages something imports; when a package is added for tooling outside the repository, say so in a
  comment or leave it out.
- When the test count in a document changes, change it in both `README.md` and `REPRODUCING.md` (or point one at the other, as
  `REPRODUCING.md` now points at `tests/README.md`).

## Suggested commit message

```
chore: repository cleanup (imports, pins, doc drift, gitignore)

- remove 50 unused imports and 7 unused locals (pyflakes F401/F811/F841); no behaviour change
- requirements.txt: drop numba, matplotlib, pypdf (imported nowhere); keep mpmath as sympy's backend
- REPRODUCING.md: correct test count (357), drop TeX toolchain, replace empty "Building the reports" section
- .gitignore: coverage/tool-cache/virtualenv/editor/OS patterns; document why results/*.{log,json} stay tracked
- add tests/README.md and docs/cleanup-reports/REPO_CLEANUP_2026-09-02.md

Full suite: 357 passed before and after, the latter with the pruned packages uninstalled.
```
