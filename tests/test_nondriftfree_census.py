from factorlab.experiments.nondriftfree_census import (
    admissible_residues, census, driftfree_baseline, family_cluster, family_members,
)
from factorlab.experiments.resonant_census import integral_classes


def test_admissible_residues_reduce_to_integral_classes_when_B_is_zero():
    # the drift-free family A = 7/15, C = 8/15 (alpha = 105, gamma = 120, q = 15): with beta = 0 the residues
    # mod 2q^2 project onto exactly the integral classes mod q
    q, alpha, gamma = 15, 105, 120
    res = admissible_residues(q, alpha, 0, gamma)
    assert set(int(x) % q for x in res) == set(int(c) % q for c in integral_classes(alpha, gamma, q))


def test_family_members_and_cluster_consistency():
    r = 2 ** 14
    mem = family_members(r, 1, 2, -1, 218)          # A = 2, B = -1, C = 218: a family with a stationary speed
    assert mem is not None
    km, kp = mem
    assert (km > r // 2).all() and (kp <= r).all() and (kp > km).all()
    # k_- must be an integer of the family: 2 q^2 k_- = alpha d^2 + (beta q - q^2) d + gamma with q = 1
    d = kp - km
    assert ((2 * km) == 2 * d * d + (-1 - 1) * d + 218).all()
    res = family_cluster(r, 1, 2, -1, 218)
    assert res["members"] == len(km) and 1 <= res["cluster_sf"] <= res["cluster_all"] <= res["members"]


def test_small_census_runs_and_reports_baseline():
    out = census(2 ** 12, q_max=3, A_grid=6, betas=(-1, 1), verbose=False)
    assert out["families_tried"] > 0 and out["max_sf"] <= out["max_all"]
    assert "ratio_sf_over_driftfree" in out and "driftfree_baseline" in out
    # the E39 file has no row at 2^12, so the baseline is None and the ratio None
    assert driftfree_baseline(2 ** 12) is None and out["ratio_sf_over_driftfree"] is None
