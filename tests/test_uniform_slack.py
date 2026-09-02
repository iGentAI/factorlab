"""Tests for latticelab.uniform_slack: the double-precision screen reproduces the certified zero-slack crossing and the archived
0.01 crossing at Kyber512, and the float and ball left-hand sides agree."""
import json
import math
import os

from latticelab.profile_floor import tight_entry_float
from latticelab.uniform_slack import Q, lhs_ball, lhs_float, screen


def test_zero_slack_screen_reproduces_certified_crossing():
    r = screen("kyber512", 0.0, 414, 418)
    assert r["crossing"] == 417
    # the archived crossing (417, m = 517, d = 1030) also passes in double precision
    m, d = 517, 1030
    bound = tight_entry_float(d, 417, d - 417 + 1, 0.0, m * math.log(Q))
    assert lhs_float(417, 3) <= bound


def test_slack_screen_matches_archive():
    r = screen("kyber512", 0.01, 413, 416)
    assert r["crossing"] == 415
    arch = "results/lattice_uniform_slack_screen.json"
    if os.path.exists(arch):
        rows = [x for x in json.load(open(arch)) if x["set"] == "kyber512" and abs(x["eps"] - 0.01) < 1e-12]
        assert rows and rows[0]["crossing"] == 415 and rows[0]["m"] == r["m"]


def test_lhs_float_and_ball_agree():
    for b, eta1 in ((417, 3), (642, 2), (900, 2)):
        ball = lhs_ball(b, eta1, 128)
        assert math.isclose(float(ball.mid()), lhs_float(b, eta1), rel_tol=1e-14)
        assert float(ball.rad()) < 1e-30
