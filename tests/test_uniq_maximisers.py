"""The unique-product maximisers: every difference attaining the family's exact maximum, against census_point and the archive."""
from factorlab.experiments.balanced_structure import census_point, unique_product_maximisers
from factorlab.gen import make_semiprime


def test_unique_product_maximisers_match_census_and_archive():
    N = int(make_semiprime(30, "rsa", 5, 0).N)
    assert N == 880634351  # the first modulus of the archived census
    res = unique_product_maximisers(N)
    cp = census_point(N, C=2, seed=5, controls=0)["families"]["unique_product"]
    assert res["D"] == cp["D_exact"] == 3
    assert res["cells"] == cp["cells"]
    # every maximiser is realised by exactly D pairs, and the differences are distinct
    assert all(len(m["pairs"]) == res["D"] for m in res["maximisers"])
    assert len({m["t"] for m in res["maximisers"]}) == res["n_maximisers"] >= 2
    # the archived line maximiser t = 24229 is among them and is the one on the line 2b = 3a + 1
    by_t = {m["t"]: m for m in res["maximisers"]}
    assert 24229 in by_t and by_t[24229]["on_line"]
    assert by_t[24229]["pairs"] == [((21, 32), (21, 31)), ((23, 35), (23, 34)), ((25, 38), (25, 37))]
    assert res["line_maximiser_exists"] and res["n_on_line"] == 1
    # the on-line test is exactly the consecutive-cell condition
    for m in res["maximisers"]:
        expected = all(a1 == a2 and b2 == b1 - 1 and 2 * b1 == 3 * a1 + 1 for (a1, b1), (a2, b2) in m["pairs"])
        assert m["on_line"] == expected


def test_unique_product_maximisers_refuses_moduli_outside_the_exact_range():
    import pytest

    with pytest.raises(ValueError):
        unique_product_maximisers(1 << 50)
    with pytest.raises(ValueError):
        unique_product_maximisers(7)
