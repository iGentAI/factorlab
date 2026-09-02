import json
import subprocess
import sys
from fractions import Fraction

import pytest

from factorlab.experiments.modfree_census import in_census_box, pooled_cluster, census
from factorlab.experiments.pileup_certify import (
    certify_boundary, high_precision_cluster, out_of_sample, parse_log2r_j, pooled_high_precision,
    recheck_e27_symmetric, speed_error_bound, symmetric_family_recheck,
)


def test_pooled_high_precision_small_r_and_empty():
    hp = pooled_high_precision(2 ** 12, q_max=30, m_max=16)
    p = pooled_cluster(2 ** 12, q_max=30, m_max=16)
    # at 2^12 the float bracket is tight, so float and 200-bit values coincide
    assert hp["pooled_sf_hp"] == p["pooled_sf"] and hp["pooled_all_hp"] == p["pooled_all"]
    assert hp["max_family_sf_hp"] == p["max_family_sf"]
    e = pooled_high_precision(2 ** 12, q_max=0, m_max=0)
    assert e["pairs_pooled"] == 0 and e["pooled_sf_hp"] == 0 and e["pileup_sf"] is False


def test_high_precision_cluster_matches_e27():
    # E27: the symmetric family j = 15 has cluster 25 at r = 2^18
    hp = high_precision_cluster(2 ** 18, 15, 0, 1)
    assert hp["cluster_sf"] == 25


def test_speed_error_bound_scale():
    # 4u times the largest shell speed 0.29 sqrt r: 6.6e-14 at 2^18, 4.3e-12 at 2^30
    assert 5e-14 < speed_error_bound(2 ** 18) < 8e-14
    assert 3e-12 < speed_error_bound(2 ** 30) < 5e-12


def test_certify_boundary_small_r_matches_e26():
    # E26: D*_1(2^12) = 9 on the squarefree shell; both tolerance sweeps must return it
    c = certify_boundary(2 ** 12)
    assert c["certified"] and c["D_lo"] == c["D_hi"] == 9
    assert c["eps"] >= c["speed_error_bound"]


def test_pooled_cluster_consistency():
    p = pooled_cluster(2 ** 12, q_max=30, m_max=16)
    # the pooled maximum can never be below the best single family, and pairs are distinct
    assert p["pooled_sf"] >= p["max_family_sf"] and p["pooled_all"] >= p["max_family_all"]
    assert p["pairs_pooled_sf"] <= p["pairs_pooled"]
    # the census's maximum over the same box is the max_family value
    c = census(2 ** 12, q_max=30, m_max=16)
    assert c["max_sf"] == p["max_family_sf"]


def test_pooled_cluster_empty_box():
    p = pooled_cluster(2 ** 12, q_max=0, m_max=0)   # no family enumerated at all
    assert p["families"] == 0 and p["pairs_pooled"] == 0
    assert p["pooled_sf"] == 0 and p["pooled_all"] == 0
    assert p["pooled_sf_bracket"] == [0, 0] and p["pooled_all_bracket"] == [0, 0] and p["speed_eps"] == 0.0
    assert p["pileup_sf"] is False and p["pileup_all"] is False


def test_pooled_cluster_symmetric_only_box():
    p = pooled_cluster(2 ** 12, q_max=1, m_max=0)   # only the symmetric families
    assert p["pooled_sf"] >= p["max_family_sf"] >= 1
    assert p["pooled_sf_bracket"][0] <= p["pooled_sf"] <= p["pooled_sf_bracket"][1]
    assert p["pooled_all_bracket"][0] <= p["pooled_all"] <= p["pooled_all_bracket"][1]


def test_in_census_box_known_families():
    # the exact 2^18 maximiser (q = 15, M = 1) is enumerated; the period-255 chain of 2^16 is not
    assert in_census_box(Fraction(7, 15), Fraction(8, 15), 2 ** 18)
    assert not in_census_box(Fraction(1, 255), Fraction(16256, 255), 2 ** 16)
    # the symmetric family j = 15 (A = 15, C = 0) sits inside the A-window at 2^18
    assert in_census_box(15, 0, 2 ** 18)
    # C = 0 with q > 1 is not enumerated by convention
    assert not in_census_box(Fraction(15, 4), 0, 2 ** 18)


def test_out_of_sample_small_r():
    o = out_of_sample(3000, q_max=30, m_max=16)
    assert o["exact"] and o["exact_D_star"] >= o["census_max_sf"]
    assert o["agree"] == (o["exact_D_star"] == o["census_max_sf"])


def test_symmetric_family_recheck_matches_archived_e27_row():
    # the first row of results/e41_e27_recheck.json: r = 2^14, j = 9 -> float 8, 200-bit 8 (squarefree) / 18 (all), 18 members
    row = symmetric_family_recheck(2 ** 14, 9)
    assert row == {"log2_r": 14, "j": 9, "float_D_sym": 8, "mpfr_sf": 8, "mpfr_all": 18, "members": 18}
    assert recheck_e27_symmetric([parse_log2r_j("14:9")], verbose=False) == [row]
    assert parse_log2r_j("16:15") == (2 ** 16, 15)
    for bad in ("14", "a:b", "0:9", "14:0"):
        with pytest.raises(ValueError):
            parse_log2r_j(bad)


def test_cli_rejects_bare_list_flags_and_nothing_to_do(tmp_path):
    out = tmp_path / "e41.json"
    for args in (["--certify"], [], ["--recheck-e27", "14:9", "--recheck"], ["--recheck-e27", "14"]):
        r = subprocess.run([sys.executable, "-m", "factorlab.experiments.pileup_certify", *args, "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 2, (args, r.stderr)
    assert not out.exists()
    r = subprocess.run([sys.executable, "-m", "factorlab.experiments.pileup_certify", "--recheck-e27", "14:9", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text()) == [{"log2_r": 14, "j": 9, "float_D_sym": 8, "mpfr_sf": 8, "mpfr_all": 18, "members": 18}]
