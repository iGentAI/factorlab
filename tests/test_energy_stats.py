from factorlab.experiments.energy_stats import modfree_energy, planar_top_mass
from factorlab.experiments.sidon_bucketed import d_star_bucketed, squarefree_shell
from factorlab.gen import make_semiprime


def test_modfree_energy_basic_and_null():
    r = 2 ** 10
    act = modfree_energy(r)
    nul = modfree_energy(r, null=True, seed=1)
    # the diagonal alone contributes one per pair; the energy is near-minimal (Proposition E)
    assert act["energy"] >= act["pairs"] and act["energy_over_pairs"] < 2.0
    # the phase-randomised null has the same pair count and an energy within a few per cent
    assert nul["pairs"] == act["pairs"]
    assert abs(nul["energy"] - act["energy"]) < 0.05 * act["energy"]
    # a speed-centred window of half-width rho_r never holds more than the window maximum D*_1(r) (E26: 4 at 2^10)
    ks = squarefree_shell(r)
    dstar = d_star_bucketed(ks, r, theta=1.0)
    assert dstar["exact"] and act["D_max"] <= dstar["D_star"] == 4
    assert act["tail"][1] == act["pairs"]


def test_planar_top_mass_identities():
    bits = 32
    N = int(make_semiprime(bits, "rsa", 7, 0).N)
    r = int(round(N ** (1 / 3)))
    row = planar_top_mass(bits, r, m_values=(10, 50))
    # sum over all integers h of D(h) counts every ordered pair once per h within W of its start difference
    assert row["total_mass"] == row["pairs"] * (2 * row["W"] - 1)
    # the top-m means are non-increasing in m and bounded by D_max
    m10, m50 = row["top"][10]["mean_top_m"], row["top"][50]["mean_top_m"]
    assert row["D_max"] >= m10 >= m50 > 0
    assert row["count_u_with_D_ge"][row["D_max"]] >= 1
