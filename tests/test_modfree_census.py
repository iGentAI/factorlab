"""E39: the modulus-free family census reproduces the exact 2^18 maximiser and is consistent with the symmetric baseline."""
from fractions import Fraction

from factorlab.experiments.modfree_census import family_cluster, census, balance_A


def test_exact_2p18_maximiser_family():
    # E37's certified exact D*_1(2^18) = 31 is the family A = 7/15, C = 8/15 (alpha = 105, gamma = 120, q = 15)
    res = family_cluster(2 ** 18, 105, 120, 15)
    assert res["n_classes"] == 4 and res["M"] == 1
    assert res["members"] == 82 and res["cluster_all"] == 82  # every member fits one window
    assert res["members_sf"] == 31 and res["cluster_sf"] == 31
    assert Fraction(res["alpha"], 15 ** 2) == Fraction(7, 15) and Fraction(res["gamma"], 15 ** 2) == Fraction(8, 15)


def test_census_contains_symmetric_baseline_and_is_consistent():
    r = 2 ** 12
    out = census(r, q_max=20, m_max=8)
    assert out["sym_sf"] is not None and out["max_sf"] >= out["sym_sf"] >= 1
    for fam in out["top"]:
        # M is recomputed from (alpha, gamma, q) and every family in the window has members
        assert fam["M"] == (fam["q"] ** 4 - 4 * fam["alpha"] * fam["gamma"]) // fam["q"] ** 2
        assert fam["members"] >= 2 and fam["cluster_sf"] <= fam["members_sf"] <= fam["members"]
    # the symmetric family reported is C = 0 with q = 1 and A near the balance point
    sym = out["sym_top"]
    assert sym["gamma"] == 0 and sym["q"] == 1
    assert balance_A(r, 1, 1) / 6 <= sym["alpha"] <= 6 * balance_A(r, 1, 1)
