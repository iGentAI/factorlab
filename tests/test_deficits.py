"""Tests for latticelab.deficits: nu_i = log GH(B_i) - ell_i from the profile, and the pairing identity with the head multipliers."""
import math

import numpy as np

from latticelab.deficits import profile_deficits
from latticelab.profile_floor import floor_l1_float, tight_profile


def test_tight_profile_has_zero_deficits():
    d, beta = 40, 10
    ell = tight_profile(d, beta, 0.0, 0.0)
    r = profile_deficits(ell, beta)
    assert max(abs(x) for x in r["nu"]) < 1e-9
    assert abs(r["weighted_deficit"]) < 1e-9


def test_pairing_identity_on_random_profiles():
    rng = np.random.default_rng(3)
    for d, beta in [(30, 8), (50, 12), (64, 20)]:
        ell = rng.normal(size=d) * 0.3 + np.linspace(2.0, -2.0, d)
        ell -= ell.mean()  # volume 0
        r = profile_deficits(ell, beta)
        h0 = floor_l1_float(d, beta, 0.0, 0.0)["l1_floor"]
        # ell_1 - S/d = h(0) - sum_i y_i nu_i, with S = 0
        assert math.isclose(ell[0], h0 - r["weighted_deficit"], rel_tol=0, abs_tol=1e-9)
        assert math.isclose(r["weighted_deficit"], r["y1_nu1"] + r["rest"], abs_tol=1e-12)
        assert len(r["nu"]) == d - 1 and len(r["y"]) == d - 1
