import math


from factorlab.experiments.olf_squares import (
    delta_k, olf_hits, lattice_parameter, random_square_parameter, delta_linear_complexity, convergent_denominators,
    tail_diagnostic,
)
from factorlab.experiments.separable import berlekamp_massey
from factorlab.gen import make_semiprime


def test_delta_k_arithmetic():
    for N, k in ((143, 1), (143, 5), (10403, 7), (2 ** 61 - 1, 12345)):
        c, D = delta_k(N, k)
        assert c * c >= 4 * k * N > (c - 1) * (c - 1) and D == c * c - 4 * k * N
        assert 0 <= D < 4 * math.sqrt(k * N) + 1


def test_every_square_is_a_proper_factorisation_and_a_cell():
    # exhaustive over all semiprimes pq with odd primes p < q < 120, N >= 39, and all 1 <= k <= N/16
    primes = [p for p in range(3, 120) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
    hits = 0
    for i, p in enumerate(primes):
        for q in primes[i + 1:]:
            N = p * q
            if N < 39:
                continue
            for h in olf_hits(N, N // 16, p, q):
                hits += 1
                assert h["proper"], (N, h)
                assert h["cell"] is not None and h["cell"][0] * h["cell"][1] == h["k"]
                a, b = h["cell"]
                assert h["c"] == a * q + b * p and h["t"] == abs(a * q - b * p)
                assert (math.sqrt(a * q) - math.sqrt(b * p)) ** 2 < 1
                assert {h["c"] - h["t"], h["c"] + h["t"]} == {2 * b * p, 2 * a * q}
    assert hits > 100
    # the factorisation is unordered: here c - t = 2q (a = 1) and c + t = 4p (b = 2)
    hs = olf_hits(101 * 199, 2, 101, 199)
    h = [z for z in hs if z["k"] == 2][0]
    assert (h["c"], h["t"], h["cell"]) == (401, 3, (1, 2)) and h["c"] - h["t"] == 2 * 199 and h["c"] + h["t"] == 4 * 101


def test_model_parameters():
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    r = int(round(N ** (1 / 3)))
    lam = lattice_parameter(N, r)
    assert abs(lam - 8 / 3) < 0.01                       # r = N^{1/3} exactly gives 8/3
    assert abs(random_square_parameter(N, r) * 4 - lam) < 0.03 * lam   # the random model is a quarter of the lattice count
    # the hits of one modulus are cells and proper
    sp = make_semiprime(40, "rsa", 7, 3)
    hs = olf_hits(int(sp.N), 4 * r, int(sp.p), int(sp.q))
    assert all(h["proper"] and h["cell"] is not None for h in hs)


def test_linear_complexity_controls():
    N = int(make_semiprime(40, "rsa", 7, 0).N)
    z = delta_linear_complexity(N, 128, 1009)
    assert z["control_lfsr_order2"] == 2 and z["linear_complexity"] >= 60
    assert berlekamp_massey([1, 1, 2, 3, 5, 8, 13, 21], 1009) == 2


def test_convergent_denominators():
    # 355/113 = [3; 7, 16] has convergents 3, 22/7, 355/113 (333/106 is an intermediate fraction)
    assert convergent_denominators(355, 113) == [1, 7, 113]
    assert convergent_denominators(8, 13) == [1, 1, 2, 3, 5, 13]   # 8/13 = [0; 1, 1, 1, 1, 1, 2]
    # empty groups are reported as None, not as a crash
    z = tail_diagnostic(32, 4, c=1e-6)      # r = round(1e-6 N^{1/3}) = 0: no hits at all, every modulus in the tail
    assert z["tail_count"] == 4 and z["rest_count"] == 0
    assert z["rest_median_jump"] is None and z["rest_frac_jump_ge_tail_min"] is None and z["tail_min_jump"] is not None
    z = tail_diagnostic(32, 4, c=64.0)      # every modulus hits by 64 N^{1/3}: empty tail
    assert z["tail_count"] == 0 and z["tail_min_jump"] is None and z["rest_frac_jump_ge_tail_min"] is None
