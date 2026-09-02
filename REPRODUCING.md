# Reproducing the manuscripts

This repository contains the code, tests and archived results behind two technical reports, distributed separately as PDFs:

| report | contents |
|---|---|
| *Floors for deterministic integer factoring: covering models, partial information, the additive structure of the Lehman--Harvey start set, and three further floors* (Sean Ward and Maestro, 2026) | Part 1 covering floors; Part 2 partial information; Part 3 the additive structure of the start set; Part 4 the common-factor class; Part 5 fixed-list ECM exposure labels; Part 6 NFS polynomial selection |
| *Certified conditional profile optima for block reduction, and the Kyber round-3 primal chain re-evaluated* (Sean Ward and Maestro, 2026) | the profile class and its linear programme, the prefix-volume floor, certificates, the round-3 chain re-evaluated, the applicability audit |

Every measured or certified statement in either report has a row below naming the function that produced it, the archive under `results/`
that holds it, and the command that regenerates it. Section names refer to the reports' parts and sections.

## Environment

Ubuntu 24.04, Python 3.12. System packages: `libfplll-dev fplll-tools libgmp-dev libmpfr-dev libqd-dev` (for `fpylll`), and TeX Live
(`texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-science`) with `poppler-utils` to build the papers. Python
packages are pinned in `requirements.txt` to the versions the archives were produced with; `cypari2 2.2.4` bundles PARI 2.17.2. If
`fpylll` cannot find its strategy file, link `/usr/share/libfplll8/strategies/default.json` to
`/project/local/share/fplll/strategies/default.json`. The system packages above must be present: without `libfplll` the pruned-BKZ tests
crash inside the extension rather than fail. Run the tests with `python -m pytest -q` (338 tests, about a minute). All commands
below run from the repository root.

All pseudorandom populations come from `factorlab.gen.make_semiprime(nbits, "rsa", seed, index)` (balanced semiprimes with $q/p <
\sqrt2$; `nbits` must be even). Rigorous decisions use exact rational arithmetic (`fmpq`) and ball arithmetic (Arb, through
`python-flint`) at 256 bits with directed comparisons; a statement is certified only when the two balls are disjoint. Double-precision
computations are screens and are labelled as such in the manuscripts.

## The lattice paper (`profile_floor_mlkem`)

The three parameter sets are $(d, \beta_{\rm spec}) = (1003, 403), (1424, 625), (1885, 877)$; the commands are shown for the first and
are repeated with the other two pairs.

| statement | producer | archive | regenerate |
|---|---|---|---|
| exact dual certificate, floor $\delta_0^{\rm floor}$, GSA target and tail correction $\kappa$ at $(1003, 403)$ | `latticelab.profile_floor.floor_l1`, `gsa_delta_ball` | `results/lattice_profile_floor_mlkem.json`, `results/lattice_profile_floor_mlkem.log` | `python -c "from latticelab.profile_floor import floor_l1; r = floor_l1(1003, 403); print(r['root_hermite_floor_ball'], r['kappa'])"` (16 s) |
| every $\beta \in [2, 413]$ decided against $\delta_{\rm GSA}(403)$: passing set $[2, 14] \cup \{413\}$ (and $[2,14]\cup\{645\}$, $[2,13]\cup\{908\}$) | `profile_floor.decide_floor_vs_target` | `results/lattice_profile_floor_full_scan.json`, `results/lattice_profile_floor_full_scan.log` | `python -c "from latticelab.profile_floor import decide_floor_vs_target, gsa_delta_ball; t = gsa_delta_ball(403); print([b for b in range(2, 414) if decide_floor_vs_target(1003, b, t)[0]])"` (144 s; 518 s and 1439 s for the other two) |
| slack crossings 409/640/902 at $\varepsilon = 0.01$ and 401/629/889 at $0.03$ | `decide_floor_vs_target(..., eps)` | `results/lattice_profile_floor_mlkem_exactall.json`, `results/lattice_profile_floor_mlkem_exactall.log` | `python -c "from fractions import Fraction; from latticelab.profile_floor import decide_floor_vs_target, gsa_delta_ball; t = gsa_delta_ball(403); print([b for b in range(395, 414) if decide_floor_vs_target(1003, b, t, eps=Fraction(1, 100))[0]])"` (first element 409; `Fraction(3, 100)` gives 401) |
| independent checker (direct forward solve, exact identity and positivity, arb enclosure, directed decision) | `latticelab.check_certificate` | `results/lattice_checker_transcript.log` | `python -m latticelab.check_certificate --d 1003 --beta 413 --gsa-beta 403` (and `--beta 412`); `python -m latticelab.check_certificate --detect --d 1030 --b 417 --m 517 --k 2 --eta1 3` (and `--b 416`) |
| certificate, verification and decision timings | `dual_certificate`, `verify_certificate`, `decide_floor_vs_target`, `tight_entry` | `results/lattice_certificate_timings.json`, `results/lattice_certificate_timings.log` | script A below |
| the round-3 chain reconstructed (406/624/874); floor substituted into condition (9) (408/626/876); every $b$ from $b_{\rm GSA} - 3$ decided | `latticelab.spec_chain.chain`, `floor_chain`, `certify_floor_chain` | `results/lattice_spec_chain.json`, `results/lattice_spec_chain.log` | `python -m latticelab.spec_chain` (defaults: all three sets, $\varepsilon = 0$, margin 3) |
| double-precision screen of the substituted chain from $b = 300$ | `spec_chain.floor_chain` | `results/lattice_floor_chain_from300.json`, `results/lattice_floor_chain_from300.log` | `python -c "from latticelab.spec_chain import floor_chain; print(floor_chain(2, 3, b_lo=300))"` (Kyber512; `(3, 2)` and `(4, 2)` for 768 and 1024; 48/293/874 s) |
| prefix-volume crossings 417/642/900 and every $b$ below decided at every admissible $m$ | `spec_chain.detection_chain`, `certify_detection_chain` (stride 1) | `results/lattice_spec_chain.json` (detection keys), `results/lattice_spec_chain_detection.log` | `python -m latticelab.spec_chain --detection` (defaults `--m-stride 1 --b-hi-margin 40`; 24 min for Kyber512, about 5 h for Kyber1024) |
| head-slack certificates: every $b$ in $[b_{\rm GSA}-3, b^*-1]$ decided at every admissible $m$ under $\varepsilon_1 = 1$ | `latticelab.head_slack.decide_set` | `results/lattice_head_slack_kyber512.json`, `results/lattice_head_slack_kyber768.json`, `results/lattice_head_slack_kyber1024.json`, `results/lattice_head_slack.log` | `python -m latticelab.head_slack --sets kyber512 kyber768 kyber1024 --eps1 1` (about 4 h) |
| $b = 899$ at Kyber1024: fails at all 508 admissible $m$ under $\varepsilon_1 = 0.3$, passes at 34 ($m = 873$--$906$) under $\varepsilon_1 = 1$ | `profile_floor.tight_entry` plus the exact head-slack term $b\varepsilon_1/(d(b-1))$ -- the per-$(b, m)$ decision of `head_slack` | `results/lattice_head_slack_kyber1024_b899.json` | script B below (11 min) |
| uniform-slack screen 415/639/897 ($\varepsilon = 0.01$) and 411/634/891 ($0.03$), double precision | `latticelab.uniform_slack.screen` | `results/lattice_uniform_slack_screen.json` | `python -m latticelab.uniform_slack --screen` |
| uniform-slack certification at Kyber512 | `uniform_slack.certify` | `results/lattice_uniform_slack_kyber512_eps0p01.json`, `results/lattice_uniform_slack_kyber512_eps0p03.json`, `results/lattice_uniform_slack_kyber512.log` | `python -m latticelab.uniform_slack --certify --sets kyber512 --eps 0.01 0.03` (12 min per slack) |
| two-sided shifts at the printed tuples: full-size $+0.03449/+0.03916/+0.04572$, all-block $+0.17414/+0.69803/+1.20066$ | `latticelab.dual_floor.combined_certificate` | `results/lattice_two_sided_tuples.json`, `results/lattice_two_sided_tuples.log` | `python -c "from latticelab.dual_floor import combined_certificate; print(combined_certificate(1003, 403, dual_mode='full')['shift_ball'])"` (and `dual_mode='all'`; 5--43 s) |
| two-sided crossings 418/654/924 ($\varepsilon = 0$), 414/649/917 ($0.01$), 406/638/903 ($0.03$), first passing in the scanned range | `dual_floor.two_sided_beta_floor` | `results/lattice_two_sided_scan.json`, `results/lattice_two_sided_scan.log`, `results/lattice_two_sided_scan_eps.log` | `python -m latticelab.dual_floor --d 1003 --beta-spec 403 --lo 410 --hi 421 --eps 0 --dual-mode full --out results/lattice_two_sided_scan.json`; scan starts 640 and 905 for the other sets; at slack, `--eps 0.01 --lo 409` (640, 902) and `--eps 0.03 --lo 401` (629, 889) |
| applicability audit: strict exact-SVP tours to a clean tour or 512 tours at $(75,30)$, $(100,40)$, $(150,30)$, $(225,30)$, seeds 31--33 | `latticelab.dual_census.strict_tours_census` (fpylll, unpruned enumeration, LLL threshold 0.99) | `results/lattice_l6_strict.json` (final bases under `bases["strict,d,beta,seed"]`), `results/lattice_l6_strict.log` | `python -m latticelab.dual_census --strict-census --points 75,30 100,40 150,30 225,30 --seeds 31 32 33 --checkpoints 2 8 32 128 512 --out results/lattice_l6_strict.json` (resumable from the archive; about 40 min per seed at $(100,40)$) |
| certified head positions and deficits of the twelve archived bases | `latticelab.certify_audit` | `results/lattice_audit_heads_certified.json`, `results/lattice_audit_heads_certified_large.json` | `python -m latticelab.certify_audit` (the six $(75,30)$, $(100,40)$ bases); `python -m latticelab.certify_audit --keys strict,150,30,31 strict,150,30,32 strict,150,30,33 strict,225,30,31 strict,225,30,32 strict,225,30,33 --out results/lattice_audit_heads_certified_large.json` |
| positional deficits $\nu_k$ and their $y$-weights from a profile alone | `latticelab.deficits.profile_deficits` | recomputed from `results/lattice_l6_strict.json` | `python -m latticelab.deficits --archive results/lattice_l6_strict.json --keys strict,100,40,31 strict,100,40,32 strict,100,40,33` |
| dual-block census on BKZ 2.0 outputs at $(100,40)$, $(125,50)$ | `dual_census.two_sided_mass` | `results/lattice_l6_dual.json`, `results/lattice_l6_dual.log` | `python -m latticelab.dual_census --dual-census --points 50,20 75,30 100,40 125,50 --seeds 31 32 33 --tours 8 --out results/lattice_l6_dual.json` |
| head-only sensitivity $w_1^{(613)} = 0.4058$ at $(1030, 417)$ | `profile_floor.prefix_volume_certificate` | recomputed | `python -c "from fractions import Fraction; from latticelab.profile_floor import prefix_volume_certificate; w, z = prefix_volume_certificate(1030, 417, 613); print(float(Fraction(w[0])))"` |

Script A (certificate timings; the archived values are 0.121/0.155/0.359 s certificate, 0.062/0.090/0.228 s verification,
0.173/0.284/0.684 s decision, 0.08 s entry enclosure, on the machine named in the archive):

```
python - <<'EOF'
import math, time
from latticelab.profile_floor import dual_certificate, verify_certificate, decide_floor_vs_target, gsa_delta_ball, tight_entry
for d, b in ((1003, 403), (1424, 625), (1885, 877)):
    t0 = time.perf_counter(); y, z = dual_certificate(d, b)
    t1 = time.perf_counter(); ok = verify_certificate(d, b, y, z)
    t2 = time.perf_counter(); decide_floor_vs_target(d, b, gsa_delta_ball(b))
    t3 = time.perf_counter(); print(d, b, round(t1 - t0, 3), round(t2 - t1, 3), round(t3 - t2, 3), ok)
t0 = time.perf_counter(); v = tight_entry(1030, 417, 614, 0, 517 * math.log(3329)); print("entry", round(time.perf_counter() - t0, 3), v)
EOF
```

Script B (the $b = 899$ decision at Kyber1024 under head slack $\varepsilon_1 = 3/10$ and $1$; ball arithmetic at 256 bits with directed
endpoints; prints `3/10 0 [] []` and then `1 34 [873, 874] [905, 906]`):

```
python - <<'EOF'
from fractions import Fraction
from flint import arb, ctx, fmpq
from latticelab.profile_floor import tight_entry
ctx.prec = 256
b, k, eta1, q = 899, 4, 2, 3329
lhs = (arb(eta1) / 2).log() / 2 + arb(b).log() / 2
logq = arb(q).log()
for eps1 in (Fraction(3, 10), Fraction(1)):
    non_failing = []
    for m in range(0, (k + 1) * 256 + 1):
        d = m + k * 256 + 1
        if 2 * b >= d + 1:
            continue
        bound = tight_entry(d, b, d - b + 1, 0, 0) + m * logq / d + arb(fmpq(b * eps1.numerator, d * (b - 1) * eps1.denominator))
        if not (lhs.lower() > bound.upper()):
            non_failing.append(m)
    print(eps1, len(non_failing), non_failing[:2], non_failing[-2:])
EOF
```

## The covering paper (`covering_floors`)

| statement | producer | archive | regenerate |
|---|---|---|---|
| moduli of the measurement sections | `factorlab.gen.make_semiprime(bits, "rsa", seed, idx)`, seeds 7 (Section 6.1), 5 (6.2), 21 (7.1) | printed in the paper's tables | `python -c "from factorlab.gen import make_semiprime; print(make_semiprime(22, 'rsa', 7, 0).N)"` |
| explicit covers of the shifted start set at $r = \lfloor N^{1/3}\rfloor$ and $\lfloor N^{1/4}\rfloor$, 12 offsets from 25 candidates | `factorlab.experiments.cover_search.cover_experiment` | `results/e46_cover_search_starts.json`, `results/e46_cover_search_starts.log` | `python -m factorlab.experiments.cover_search --bits 22 24 26 28 30 32 34 36 --regimes third quarter --out results/e46_cover_search_starts.json` |
| the same with 80 offsets from 120 candidates at 22/26/30 bits | `cover_search.cover_experiment` | `results/e46_cover_search_bigbudget_starts.json`, `results/e46_cover_search_bigbudget_starts.log` | `python -m factorlab.experiments.cover_search --bits 22 26 30 --regimes third --top-candidates 120 --max-offsets 80 --out results/e46_cover_search_bigbudget_starts.json` |
| balanced / primitive / unique-product census at nineteen moduli (seed 5, 30--48 bits), own-window statistics, Lemma D values | `factorlab.experiments.balanced_structure.census_point`, `family_stats` | `results/e53_balanced_structure.json` (every modulus listed) | `python -m factorlab.experiments.balanced_structure --bits 30 32 34 36 38 40 42 44 46 48 --count 2 --out results/e53_balanced_structure.json`; any single modulus $N$ of the archive by `census_point(N, C=2, seed=5)` |
| every difference attaining the unique-product family's exact maximum at the nineteen moduli, with its realising pairs: a maximiser on the line $2b = 3a + 1$ at every modulus, the unique one at thirteen | `balanced_structure.unique_product_maximisers` | `results/e53_uniq_maximisers.json` | `python -m factorlab.experiments.balanced_structure --maximisers --moduli 880634351 964728493 3722008519 3601086119 16577631001 14048619007 46551225037 51465965107 197762968751 187452546833 876190896151 680093802697 3890830613443 3072789281321 11427345574369 16247581009873 55862770399391 46686914783221 176552314063291 --out results/e53_uniq_maximisers.json` (2 min) |
| half-offset census | `balanced_structure.half_offset_census` | `results/e53_half_offset.json` | `python -m factorlab.experiments.balanced_structure --half-offset --bits 42 48 --out results/e53_half_offset.json` |
| excision census, $M = 3, 4, 6, 8$ | `balanced_structure.excision_census` | `results/e53_excision.json` | `python -m factorlab.experiments.balanced_structure --excision --bits 42 48 --out results/e53_excision.json` |
| product-modality sample: 80 balanced 34-bit moduli, seed 21, $\alpha = 2$ | `factorlab.experiments.lehman_product.spurious_experiment` | `results/e47_lehman_product_v2.json` | `python -m factorlab.experiments.lehman_product --spurious --bits 34 --count 80 --out results/e47_lehman_product_v2.json` |

## The partial-information paper (`partial_information`)

| statement | producer | archive | regenerate |
|---|---|---|---|
| residue-class variant of Harvey's search: six 40-bit moduli, $M = 1, 4, 16, 64, 256$ | `factorlab.experiments.harvey_residue.experiment` (seed 12) | `results/e51_harvey_residue.json` | `python -m factorlab.experiments.harvey_residue --bits 40 --count 6 --moduli 1 4 16 64 256 --out results/e51_harvey_residue.json` |
| order-element selection: 60 moduli, the 22 adversarial moduli, 40 constructed moduli | `factorlab.experiments.order_selection.experiment` (seed 3) | `results/e50_order_selection.json` | `python -m factorlab.experiments.order_selection` (defaults `--bits 36 44 --count 30`) |
| Farey covering families: 144 cases, coverage verified, cost against $P_0$; the depth-one interval | `factorlab.experiments.farey_cover.landscape_experiment` (seed 17, $C = 2$) | `results/e55_farey_cover.json` | `python -m factorlab.experiments.farey_cover --bits 40 50 60 --lam 0.26 0.30 0.34 0.375 --count 12 --seed 17 --out results/e55_farey_cover.json` |

## The common-factor note

| statement | producer | archive | regenerate |
|---|---|---|---|
| 240 constructed moduli of 36--44 bits, 239 factored by the collision search, $e/(\sqrt N/g) \le 0.9955$ | `factorlab.experiments.common_factor.experiment` (seed 9) | `results/e54_common_factor.json` | `python -m factorlab.experiments.common_factor --count 240 --bits 36 44 --seed 9 --out results/e54_common_factor.json` |

## The ECM note

| statement | producer | archive | regenerate |
|---|---|---|---|
| separating lengths at $u = 3$ on all primes of $[x, 2x)$, $x = 2^{14} \ldots 2^{22}$; one-large-prime lengths; plain Montgomery curves; residue enrichment 195/311 | `factorlab.experiments.hitting_sets.residual_scaling_experiment`, `cover`, `pair_residue_analysis` via the driver | `results/e20_hitting_sets.json`, `results/e20_family_comparison.json`, `results/e20_last_pairs.json`, `results/e38_conjecture_e_2p22.json` | `python -m factorlab.experiments.run_hitting` (the `--quick` flag reduces the sizes) |
| separating lengths at $u = 4$ | `hitting_sets.residual_scaling_experiment(u=4)` via the driver | `results/e49_conjecture_e_u4.json` | `python -m factorlab.experiments.run_hitting` |
| 616-pair certificate test; cost-scaling exponent 0.175 | `factorlab.experiments.fixed_list_check.certificate_test`, `cost_scaling` (seeds 3, 9) | `results/e21_fixed_list.json` | `python -m factorlab.experiments.run_fixed_list` |
| scaling runs to 180 bits; memory fit | `fixed_list_check.scalability_probe` | `results/e22_fixed_scale_*.json`, `results/e22_scalability.json` | `python -m factorlab.experiments.run_fixed_scale 128 0 3 2` (arguments BITS INDEX U C; one run per archived file) |
| greedy schedules | `hitting_sets.greedy_label_schedule` via the driver | `results/e22_scale_strategy.json` | `python -m factorlab.experiments.run_scale_strategy` |
| $p - 1$ sample: 2000 moduli per size at 40--64 bits, seed 11, $\theta = 0.31$ | `factorlab.experiments.average_case_pm1.average_case_experiment` | `results/e40_average_case_pm1.json` | `python -m factorlab.experiments.average_case_pm1 --bits 40 48 56 64 --count 2000 --theta 0.31 --out results/e40_average_case_pm1.json` |

## The NFS note

The three experiments of this note share one driver, `python -m factorlab.experiments.run_beyond` (about 8 minutes), which writes the
three archives below; the `--quick` flag reduces the sample sizes.

| statement | producer | archive | regenerate |
|---|---|---|---|
| fixed-root shortest vectors; base-$m$ ratios | `factorlab.experiments.root_lattice.root_lattice_experiment(count=20, seed=97)` | `results/e15_root_lattice.json` | `python -m factorlab.experiments.run_beyond` |
| exact $d = 2, 3$ minima and slopes; Poisson checks; leading-coefficient search | `factorlab.experiments.poly_floor.poly_floor_experiment` (seed 111) | `results/e18_poly_floor.json`; the deduplicated $d = 3$ slope in `results/e18_poly_floor_dedup.json` | `python -m factorlab.experiments.run_beyond`; the Poisson checks and minimiser structure are printed by `python -m factorlab.experiments.poly_floor results/e18_poly_floor.json`; the deduplicated slope is script C below |
| quadratic bridge statistics | `factorlab.experiments.quadratic_bridge.quadratic_bridge_experiment`, `k_scaling_experiment` (seed 131) | `results/e19_quadratic_bridge.json` | `python -m factorlab.experiments.run_beyond` |
| $(d,d)$ exact floors, tiny-$f$ scaling, geometric-progression slopes | `factorlab.experiments.nonlinear_pairs` (seed 5) via the frontier driver | `results/e23_nonlinear_pairs.json` | `python -m factorlab.experiments.run_frontier` (about 11 minutes) |
| selection frontier | `factorlab.experiments.selection_frontier.frontier_experiment(count=3, T_max=2**18, seed=23)` | `results/e23_selection_frontier.json` | `python -m factorlab.experiments.run_frontier` |

Script C (the deduplicated $d = 3$ slope: ordinary least squares of $\log_2 P_{\min}$ on $\log_2 N$ over the 42 distinct moduli among the
44 archived instances; prints `42 0.4603339565599227 0.013745534719764757 -0.539312757169349`):

```
python - <<'EOF'
import json, math, numpy as np
e = json.load(open("results/e18_poly_floor.json"))["3"]
pts = {}
for row in e["rows"]:
    for inst in row["instances"]:
        pts[str(inst["N"])] = (math.log2(int(inst["N"])), math.log2(int(inst["P"])))
x = np.array([v[0] for v in pts.values()]); y = np.array([v[1] for v in pts.values()])
A = np.vstack([x, np.ones_like(x)]).T
coef = np.linalg.lstsq(A, y, rcond=None)[0]
n = len(x); resid = y - A @ coef; se = math.sqrt((resid @ resid) / (n - 2) / ((x - x.mean()) ** 2).sum())
print(n, coef[0], se, coef[1])
EOF
```

## The additive-structure report

| statement | producer | archive | regenerate |
|---|---|---|---|
| Table 1: $D^*(r)$ at $r = 2^8 \ldots 2^{15}$ with the phase-randomised null; the near-chain census | `factorlab.experiments.sidon_scaling` | `results/e26_sidon_scaling.json` | `python -m factorlab.experiments.sidon_scaling` |
| Table 3: the symmetric family to $r = 2^{32}$; the cross-class pairs; the Theorem Q$'$ construction | `sidon_scaling` (symmetric census, `theorem_q_prime_check`) | `results/e27_cross_class_pairs.json`, `results/e27_big.json`, `results/e29_theorem_q_prime.json` | `python -m factorlab.experiments.sidon_scaling` |
| certified exact $D^*_1$ at $2^{16}, 2^{17}, 2^{18}$ and the prime shell at $2^{19}, 2^{20}$ | `factorlab.experiments.sidon_bucketed.d_star_bucketed` | `results/e37_sidon_bucketed.json` | `python -m factorlab.experiments.sidon_bucketed --r 65536 131072 262144 --prime-r 524288 1048576 --parts 4` |
| Table 5: exact statistic against the census at 56 and 64 bits | `sidon_bucketed.run_planar` | `results/e37_planar_exact.json` | `python -m factorlab.experiments.sidon_bucketed --planar 56 --planar-r 55095 109907 --kinds squarefree full prime` (and `--planar 64` with the radii listed in the archive) |
| Table 2: drift-free census at $r = 2^{16} \ldots 2^{30}$ | `factorlab.experiments.modfree_census` | `results/e39_modfree_census.json` | `python -m factorlab.experiments.modfree_census --r 65536 262144 1048576 4194304 16777216 67108864 268435456 1073741824 --q-max 120 --m-max 64 --out results/e39_modfree_census.json` |
| boundary certification of the exact maxima; pooled maxima; out-of-sample radii; 200-bit rechecks | `factorlab.experiments.pileup_certify.certify_boundary`, `pooled_high_precision`, `out_of_sample`, `recheck_census` | `results/e41_certify_oos.json`, `results/e41_pooled.json`, `results/e41_e27_recheck.json` | `python -m factorlab.experiments.pileup_certify --certify --oos --out results/e41_certify_oos.json`; `--pool --pool-hp --out results/e41_pooled.json`; `--recheck --out results/e41_e27_recheck.json` |
| additive energy against the null; planar energy form | `factorlab.experiments.energy_stats.modfree_energy`, `planar_top_mass` | `results/e42_energy.json` | `python -m factorlab.experiments.energy_stats --energy --planar --out results/e42_energy.json` |
| enlarged census box $q \le 1200$ | `modfree_census` | `results/e42_census_bigq.json` | `python -m factorlab.experiments.modfree_census --r 4194304 16777216 --q-max 1200 --m-max 64 --out results/e42_census_bigq.json` |
| class counts of drift-free families; hit classification against the continued fraction | `factorlab.experiments.arms_e45` | `results/e45_arms.json` | `python -m factorlab.experiments.arms_e45 --classes --hits --out results/e45_arms.json` |
| non-drift-free census | `factorlab.experiments.nondriftfree_census` | `results/e43_nondriftfree_census.json` | `python -m factorlab.experiments.nondriftfree_census` (defaults $r = 2^{16}, 2^{18}, 2^{20}$, $q \le 24$) |
| Table 4: planar census on the 40- and 48-bit moduli; the prime shell to 62 bits | `factorlab.experiments.planar_census` (seed 7) | `results/e31_planar_census.json`, `results/e31_prime_60_62.json` | `python -m factorlab.experiments.planar_census` |
| resonant census to 96 bits | `factorlab.experiments.resonant_census` (seed 7) | `results/e32_resonant_census.json` | `python -m factorlab.experiments.resonant_census` (a single size with `--bits 64`) |
| Table 6: explicit covers | `cover_search` | `results/e46_cover_search_starts.json`, `results/e46_cover_search_bigbudget_starts.json` | `python -m factorlab.experiments.cover_search --bits 22 24 26 28 30 32 34 36 --regimes third quarter --out results/e46_cover_search_starts.json`; `python -m factorlab.experiments.cover_search --bits 22 26 30 --regimes third --top-candidates 120 --max-offsets 80 --out results/e46_cover_search_bigbudget_starts.json` |
| balanced, unique-product, half-offset and excision censuses; the maximisers | `balanced_structure` | `results/e53_balanced_structure.json`, `results/e53_uniq_maximisers.json`, `results/e53_half_offset.json`, `results/e53_excision.json` | `python -m factorlab.experiments.balanced_structure --bits 30 32 34 36 38 40 42 44 46 48 --count 2 --out results/e53_balanced_structure.json`; `python -m factorlab.experiments.balanced_structure --maximisers --moduli 880634351 964728493 3722008519 3601086119 16577631001 14048619007 46551225037 51465965107 197762968751 187452546833 876190896151 680093802697 3890830613443 3072789281321 11427345574369 16247581009873 55862770399391 46686914783221 176552314063291 --out results/e53_uniq_maximisers.json`; `python -m factorlab.experiments.balanced_structure --half-offset --bits 42 48 --out results/e53_half_offset.json`; `python -m factorlab.experiments.balanced_structure --excision --bits 42 48 --out results/e53_excision.json` |
| Theorem W and W$'$ checks; prime onset (seed 7, $\lambda = 0.8$) | `factorlab.experiments.theorem_checks.experiment` | `results/e56_theorem_checks.json` | `python -m factorlab.experiments.theorem_checks --w-bits 48 64 96 128 160 200 --wp-bits 64 96 128 --onset-bits 96 128 160 200 256 --seed 7 --lam 0.8 --out results/e56_theorem_checks.json` |

## Building the reports

```
```

Both build with no errors, no undefined references and no duplicated labels; the assembler refuses to write the report otherwise. The two
PDFs are committed beside their sources.

## Added 2 September 2026: audit deficits at the detection position, simulator re-evaluation, part extracts

| statement | command | archive |
|---|---|---|
| Lattice report §7: prefix-weighted deficits of the twelve audit bases at m = d − β, with head/body/tail decomposition and the exact-identity check | `python -m latticelab.audit_detection --archive results/lattice_l6_strict.json --out results/lattice_audit_detection_deficits.json` (≈1 min) | `results/lattice_audit_detection_deficits.json` |
| Lattice report §6: detection crossings under the Chen–Nguyen (`cn`, converged) and Bai–Stehlé–Wen (`bsw`, 50-tour budget) simulators from a Z-shape start | `python -m latticelab.simulator_chain --sets Kyber512 Kyber768 Kyber1024 --models cn bsw --m-stride 4 --bsw-m-stride 32 --bsw-tours 50 --out results/lattice_simulator_chain.json` (CN ≈ 3 min per set; BSW ≈ 10–40 min per set) | `results/lattice_simulator_chain.json`, `.log` |
| Lattice report §6: the specification's own script | `git clone https://github.com/pq-crystals/security-estimates && cd security-estimates && python3 Kyber.py` (commit 75c26949; ≈6 min; prints "Primal attacks uses block-size 406 and 486 samples; dim d=999", 626/650/1419, 878/860/1885) | the three lines quoted in the command column are the script's output (Kyber512, Kyber768, Kyber1024) |
| Lattice report §6: the q-ary ceiling — extremal head against log q at the certified crossings, the head-clipped (q-aware) extremal chain, and the specification's dual route under the head floor | `python -m latticelab.qceiling --qaware --dual --sets Kyber512 Kyber768 Kyber1024 --m-stride 2 --dual-m-stride 2 --out results/lattice_qceiling.json` (≈1 h for the three sets) | `results/lattice_qceiling.json`, `.log` |
| Factoring report, Part 4, Proposition (curve), numerical paragraph: Routes B and C on constructed moduli with a prescribed common factor | `python -m factorlab.experiments.common_factor_curve --bits 40 48 56 --gammas 0.0833 0.125 0.1667 0.2 --count 4 --out results/e57_common_factor_curve.json` (≈1 min) | `results/e57_common_factor_curve.json` |
| Lattice report §6, the Kyber1024 row of the simulator comparison under BSW18 (the full scan was stopped at b = 871 after 1 h 32 m; the crossing window was rescanned) | `python -m latticelab.simulator_chain --sets Kyber1024 --models bsw --b-lo 896 --b-hi 904 --bsw-m-stride 64 --bsw-tours 50 --out results/lattice_simulator_chain_k1024_bsw.json` (≈10 min), merged into `results/lattice_simulator_chain.json` under the key `Kyber1024,bsw` | `results/lattice_simulator_chain_k1024_bsw.json`, `.log` |
| Lattice report §6: fixed points of the Chen–Nguyen tour (converged output tested against l_k ≤ log GH(B_k) with fpylll's own constants: 58/40 tours, 3/2 entries at log q, none above, strict inequality only there with deficits 1.8e-2, 1.0e-2, 2.5e-3 and 7.4e-3, 1.7e-3, equality below 1e-6 elsewhere; head-clipped tight profile within 4.3e-3 / 2.9e-3 over the first d-45 entries) and the passing sample-count intervals [481, 555], [651, 747], [833, 949] at b = 417, 642, 900 (75/97/117 counts, contiguous; maximum-margin counts 517/698/890; nothing passes at b = 416) | `python -m latticelab.simulator_chain --fixed-point Kyber512:417:520 Kyber768:642:700 --intervals Kyber512:417 Kyber768:642 Kyber1024:900 Kyber512:416 --out results/lattice_cn_fixed_point.json` (about 1 min) | `results/lattice_cn_fixed_point.json`, log `results/lattice_cn_fixed_point.log` |
| Lattice report §6: Kyber768 uniform-slack crossing at ε = 0.01 certified (every b ∈ [621, 638] fails; 639 passes at 58 sample counts) | `python -m latticelab.uniform_slack --certify --sets kyber768 --eps 0.01 --out-prefix results/lattice_uniform_slack_kyber768` (≈1 h) | `results/lattice_uniform_slack_kyber768_kyber768_eps0p01.json` |
| Lattice report §6: the Kyber1024 rows of the q-aware chain and the dual route (the full scan was stopped at b = 881; the crossing window was rescanned) | `python -c "from latticelab.qceiling import qaware_chain, dual_route; import json; o={'Kyber1024,qaware': qaware_chain('Kyber1024', 897, 910, m_stride=2)}; o['Kyber1024,dual']=dual_route('Kyber1024', m_stride=4); json.dump(o, open('results/lattice_qceiling_k1024.json','w'), indent=1)"` (≈20 min), merged into `results/lattice_qceiling.json` | `results/lattice_qceiling_k1024.json`, `.log` |
