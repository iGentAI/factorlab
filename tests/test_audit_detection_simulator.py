"""Tests for latticelab.audit_detection and latticelab.simulator_chain."""
import json
import math
import os

import pytest

from latticelab.audit_detection import audit_archive, log_norms_from_rows, weighted_deficits
from latticelab.profile_floor import tight_profile
from latticelab.simulator_chain import detection_readings, fixed_point_check, passing_interval, point, zshape_profile

ARCHIVE = os.path.join(os.path.dirname(__file__), "..", "results", "lattice_l6_strict.json")


def test_weighted_deficit_vanishes_on_the_tight_profile_and_identity_holds_off_it():
    d, beta = 60, 20
    ell = list(tight_profile(d, beta, 0.0, 12.5))
    r = weighted_deficits(ell, beta, d - beta)
    assert abs(r["W"]) < 1e-9 and abs(r["identity_gap"]) < 1e-9
    # perturb the profile keeping the volume: the identity P_m(l) - P_m(l_tight) + W = 0 must still hold exactly
    ell2 = ell[:]
    ell2[3] += 0.2
    ell2[40] -= 0.2
    r2 = weighted_deficits(ell2, beta, d - beta)
    assert abs(r2["identity_gap"]) < 1e-9 and r2["W"] != 0.0
    assert r2["n_nonzero_w"] >= 1 and r2["z"] == pytest.approx((d - beta) / d)


def test_weighted_deficit_rejects_bad_arguments():
    ell = [0.0] * 30
    with pytest.raises(ValueError):
        weighted_deficits(ell, 10, 0)
    with pytest.raises(ValueError):
        weighted_deficits(ell, 30, 5)


@pytest.mark.skipif(not os.path.exists(ARCHIVE), reason="strict-census archive not present")
def test_archived_bases_have_log_norms_summing_to_the_log_volume_and_a_closed_identity():
    arch = json.load(open(ARCHIVE))
    key = "strict,75,30,31"
    ell = log_norms_from_rows(arch["bases"][key])
    assert len(ell) == 75
    res = audit_archive(ARCHIVE, [key])
    row = res["rows"][key]
    assert row["m"] == 45 and abs(row["identity_gap"]) < 1e-7
    assert row["head_term"] == pytest.approx(row["w1"] * row["nu1"])
    assert row["last_block_shift"] == pytest.approx(row["W"] / 30)


def test_zshape_profile_has_the_prescribed_volume_and_shape():
    d, log_q = 300, math.log(3329)
    m = 150
    ell = zshape_profile(d, m * log_q, log_q)
    assert len(ell) == d and abs(sum(ell) - m * log_q) < 1e-6
    assert all(0.0 <= x <= log_q + 1e-12 for x in ell)
    assert all(ell[i] >= ell[i + 1] - 1e-12 for i in range(d - 1))


def test_detection_readings_agree_on_a_flat_profile():
    d, b = 100, 20
    ell = [0.3] * d
    r = detection_readings(ell, b, 1.0)
    # flat profile: the entry is 0.3 and the last-block GH is log c_hat_b + 0.3
    assert r["entry"] == pytest.approx(0.3)
    assert r["log_gh_last"] - 0.3 == pytest.approx(r["log_gh_last"] - r["entry"])
    assert r["lhs"] == pytest.approx(0.5 * math.log(b))


def test_simulator_point_runs_at_a_small_kyber_like_size():
    # a small instance keeps the test fast: k=2 gives d = m + 513; choose m = 40 and b = 60
    p = point(2, 3, 60, 40, "cn", max_tours=50)
    assert p["d"] == 553 and p["tours"] >= 1
    assert math.isfinite(p["margin_gh"]) and math.isfinite(p["margin_entry"])
    with pytest.raises(ValueError):
        point(2, 3, 400, 40, "cn")


def test_converged_cn_profile_is_a_fixed_point_pinned_only_at_the_clipped_head():
    r = fixed_point_check("Kyber512", 417, 520)
    assert r["d"] == 1033 and r["stopped_by_criterion"]
    # the fixed-point inequality holds everywhere (to rounding) and no entry exceeds log q
    assert r["deficit_min"] > -1e-9 and r["max_excess_over_log_q"] <= 1e-12
    # strict inequality exactly at the head entries pinned at log q, whose number is the q-aware clip depth
    assert r["n_entries_at_log_q"] == r["clip_depth"] == 3
    assert r["positions_deficit_above_tol"] == [0, 1, 2]
    # the head-clipped tight profile has the same structure and lies within 5e-3 of the simulator's fixed point outside the tail
    assert r["max_abs_diff_to_clipped_tight_first_d_minus_45"] < 5e-3
    assert all(x > 1e-3 for x in r["clipped_tight_deficits_at_clipped_positions"])
    assert abs(r["clipped_tight_deficit_max_elsewhere_first_d_minus_45"]) < 1e-9 and r["clipped_tight_deficit_min_first_d_minus_45"] > -1e-9
    with pytest.raises(ValueError):
        fixed_point_check("Kyber512", 800, 100)


def test_passing_interval_at_the_kyber512_crossing():
    r = passing_interval("Kyber512", 417)
    assert (r["m_lo"], r["m_hi"], r["count"], r["contiguous"]) == (481, 555, 75, True)
    assert r["passing"] == list(range(481, 556))
    assert r["m_max_margin"] == 517 and r["d_max_margin"] == 1030 and r["margin_max"] > 0
    # one blocksize below the certified crossing nothing passes, and the result keeps the same shape with a negative best margin
    e = passing_interval("Kyber512", 416)
    assert e["count"] == 0 and e["passing"] == [] and e["m_lo"] is None and e["m_hi"] is None and e["contiguous"] is True
    assert set(e) == set(r) and e["margin_max"] < 0 and e["m_max_margin"] is not None
