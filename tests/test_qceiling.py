"""Tests for latticelab.qceiling."""
import math

import pytest

from latticelab.qceiling import LOG_Q, clipped_tight_profile, detection_margin, dual_cost_bits, n_above_ceiling, tight_head
from latticelab.profile_floor import tight_profile


def test_clipped_profile_has_no_entry_above_log_q_and_keeps_the_volume():
    d, b = 300, 60
    S = 250 * LOG_Q                       # a volume so large that the unclipped head exceeds log q
    assert tight_head(d, b, S) > LOG_Q
    k, prof = clipped_tight_profile(d, b, S)
    assert k >= 1 and len(prof) == d
    assert abs(float(sum(prof)) - S) < 1e-6
    assert max(prof) <= LOG_Q + 1e-9
    # the suffix is the tight profile of the remaining volume and its head is at most log q, while one fewer q-entry would exceed it
    assert tight_head(d - k, b, S - k * LOG_Q) <= LOG_Q
    assert tight_head(d - (k - 1), b, S - (k - 1) * LOG_Q) > LOG_Q


def test_clipping_is_the_identity_when_the_head_is_below_log_q():
    d, b = 200, 40
    S = 60 * LOG_Q
    assert tight_head(d, b, S) <= LOG_Q
    k, prof = clipped_tight_profile(d, b, S)
    assert k == 0
    assert max(abs(prof - tight_profile(d, b, 0.0, S))) < 1e-12


def test_full_volume_gives_the_all_q_profile_and_more_is_infeasible():
    d, b = 50, 20                       # b > 12: the tight head lies above the mean, so a near-full volume forces clipping
    S = d * LOG_Q
    k, prof = clipped_tight_profile(d, b, S)
    assert k == d and len(prof) == d and abs(float(sum(prof)) - S) < 1e-9 and max(abs(prof - LOG_Q)) < 1e-12
    with pytest.raises(ValueError):
        clipped_tight_profile(d, b, S + 1.0)
    with pytest.raises(ValueError):
        clipped_tight_profile(d, b, S + 1e-6)          # a material excess, far beyond machine roundoff, is rejected
    # just below the boundary: the volume is preserved exactly, not replaced by d log q
    k2, prof2 = clipped_tight_profile(d, b, S - 1e-6)
    assert abs(float(sum(prof2)) - (S - 1e-6)) < 1e-9 and k2 >= 1


def test_near_full_volume_leaves_a_suffix_shorter_than_the_blocksize():
    d, b = 50, 20
    S = d * LOG_Q - 3.0                 # only a few entries' worth of volume is missing: the suffix must be shorter than b
    k, prof = clipped_tight_profile(d, b, S)
    assert d - k < b and k >= 1
    assert abs(float(sum(prof)) - S) < 1e-6
    # the construction is head-clipped: the suffix head is at most log q, while its rising tail may exceed the ceiling and is reported
    assert prof[k] <= LOG_Q + 1e-9
    assert n_above_ceiling(prof) == sum(1 for x in prof if x > LOG_Q + 1e-9)


def test_two_entry_suffix_always_suffices():
    # a suffix of dimension 2 has head = mean + log c_hat_2 < mean <= log q, so the clipping depth never exceeds d - 2
    d, b = 30, 20
    S = d * LOG_Q - 0.05
    k, prof = clipped_tight_profile(d, b, S)
    assert 1 <= k <= d - 2 and prof[k] <= LOG_Q + 1e-9 and abs(float(sum(prof)) - S) < 1e-9


def test_clipping_raises_the_detection_margin():
    # moving head volume above log q into the suffix raises the tail and hence the last block's Gaussian heuristic
    d, b = 300, 60
    S = 250 * LOG_Q
    k, prof = clipped_tight_profile(d, b, S)
    assert detection_margin(prof, b, 1.0) > detection_margin(tight_profile(d, b, 0.0, S), b, 1.0)


def test_dual_cost_matches_the_script_at_kybers_printed_dual_optimum():
    # Kyber512 dual: the script prints l = 3294.02, log2(epsilon) = -41.82, log2 nvector per run 83.63 at b = 403, m = 512 (d = 1024)
    c = dual_cost_bits(math.log(3294.02), 403, 1.5)
    assert c["log2_eps"] == pytest.approx(-41.82, abs=0.01)
    assert c["log2_R"] == pytest.approx(0.0, abs=0.02)
    assert c["cost_bits"] == pytest.approx(0.292 * 403, abs=0.3)
    # a longer first vector costs more
    assert dual_cost_bits(math.log(3600.0), 403, 1.5)["cost_bits"] > c["cost_bits"]
