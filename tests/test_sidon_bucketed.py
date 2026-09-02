"""E37: the bucketed exact statistics agree with brute force and with the materialised sweeps of E26/E30/E31."""
import numpy as np

from factorlab.experiments.sidon_bucketed import (d_star_bucketed, dmax_bucketed, squarefree_shell,
                                                  prime_shell, rho, planar_exact_point)


def _brute(ks, r, theta=1.0):
    ks = np.asarray(ks, dtype=np.int64)
    sq = np.sqrt(ks.astype(float))
    v = []
    for i in range(len(ks) - 1):
        v.append((ks[i + 1:] - ks[i]) / (sq[i + 1:] + sq[i]))
    v = np.sort(np.concatenate(v))
    w = 2 * theta * rho(r)
    ends = np.searchsorted(v, v + w, side="left")
    return int((ends - np.arange(len(v))).max())


def _brute_int(s, W):
    s = np.asarray(s, dtype=np.int64)
    d = np.sort(np.concatenate([s[i + 1:] - s[i] for i in range(len(s) - 1)]))
    ends = np.searchsorted(d, d + 2 * W - 2, side="right")
    return int((ends - np.arange(len(d))).max())


def test_default_interface_matches_brute_force():
    for r in (512, 1024, 2048):
        ks = squarefree_shell(r)
        res = d_star_bucketed(ks, r, nbins_target=2e5)
        assert res["exact"] and res["upper"] == res["D_star"] == _brute(ks, r)


def test_e26_values_default_interface():
    # E26, Table: squarefree shell at Harvey's resolution, theta = 1.
    for r, expected in ((2 ** 10, 4), (2 ** 12, 9)):
        res = d_star_bucketed(squarefree_shell(r), r, nbins_target=1e6)
        assert res["exact"] and res["D_star"] == expected


def test_e30_prime_shell_small_maximum():
    # a small true maximum: the per-bin descent stalls, the collective refinement certifies it
    ks = prime_shell(2 ** 14)
    truth = _brute(ks, 2 ** 14)
    assert truth == 3
    res = d_star_bucketed(ks, 2 ** 14, nbins_target=1e6, max_refine=5)
    assert res["exact"] and res["D_star"] == truth == res["upper"]
    assert res["T_collect"] is not None and res["collected_pairs"] > 0
    # with the collective step disabled the same call returns honest bounds only
    res2 = d_star_bucketed(ks, 2 ** 14, nbins_target=1e6, max_refine=5, collect_budget=0)
    assert res2["D_star"] <= truth <= res2["upper"]


def test_budget_exhaustion_gives_certified_bounds():
    ks = squarefree_shell(2 ** 11)
    truth = _brute(ks, 2 ** 11)
    res = d_star_bucketed(ks, 2 ** 11, nbins_target=2e5, max_refine=1, collect_budget=0)
    assert res["D_star"] <= truth <= res["upper"]
    res = d_star_bucketed(ks, 2 ** 11, nbins_target=2e5, max_refine=1)
    assert res["exact"] and res["D_star"] == truth


def test_budget_reached_on_last_candidate_still_certifies():
    ks = squarefree_shell(2 ** 11)
    full = d_star_bucketed(ks, 2 ** 11, nbins_target=2e5, collect_budget=0)
    assert full["exact"]
    res = d_star_bucketed(ks, 2 ** 11, nbins_target=2e5, max_refine=full["refined"], collect_budget=0)
    assert res["exact"] and res["D_star"] == full["D_star"] == _brute(ks, 2 ** 11)


def test_integer_statistic_matches_brute_force_and_e31():
    from factorlab.gen import make_semiprime
    from factorlab.experiments.planar_census import shell_starts, exact_dmax
    from factorlab.experiments.sidon_bucketed import squarefree_mask
    N = int(make_semiprime(36, "rsa", 7, 0).N)
    r = round(N ** (1 / 3))
    ks, s = shell_starts(N, r, squarefree_mask(r))
    for W in (1, 2, 3):
        res = dmax_bucketed(ks, s, W, nbins_target=2e6)
        assert res["exact"] and res["D_max"] == _brute_int(s, W) == exact_dmax(ks, s, W)["D_max"]
        # the reported centre is an integer whose Lemma D window recounts exactly D_max pairs
        t = res["t"]
        assert isinstance(t, int)
        d = np.concatenate([s[i + 1:] - s[i] for i in range(len(s) - 1)])
        assert int(np.sum(np.abs(d - t) < W)) == res["D_max"]
        assert all(abs((s[list(ks).index(b)] - s[list(ks).index(a)]) - t) < W for a, b in res["pairs"])
    # the driver reproduces the same value at Lemma D's window
    pt = planar_exact_point(N, r, "squarefree", nbins_target=2e6)
    assert pt["exact"] and pt["D_max"] == exact_dmax(ks, s, pt["W"])["D_max"]


def test_range_splitting_matches_unsplit():
    ks = squarefree_shell(2 ** 12)
    one = d_star_bucketed(ks, 2 ** 12, nbins_target=1e6)
    for parts in (2, 5):
        res = d_star_bucketed(ks, 2 ** 12, nbins_target=2e5, parts=parts)
        assert res["exact"] and res["D_star"] == one["D_star"] == 9 and len(res["parts"]) == parts
    from factorlab.gen import make_semiprime
    from factorlab.experiments.planar_census import shell_starts
    from factorlab.experiments.sidon_bucketed import squarefree_mask
    N = int(make_semiprime(36, "rsa", 7, 0).N)
    r = round(N ** (1 / 3))
    ks, s = shell_starts(N, r, squarefree_mask(r))
    for W in (1, 3):
        res = dmax_bucketed(ks, s, W, nbins_target=5e5, parts=3)
        assert res["exact"] and res["D_max"] == _brute_int(s, W)


def test_parts_broadcast_and_validation():
    import pytest
    from factorlab.experiments.sidon_bucketed import _parts_for, _split_max
    assert _parts_for(3, 4) == [3, 3, 3, 3]
    assert _parts_for([2], 3) == [2, 2, 2]
    assert _parts_for([1, 4, 8], 3) == [1, 4, 8]
    with pytest.raises(ValueError):
        _parts_for([1, 2], 3)
    # coinciding merged bounds certify the maximum even when a part was only bracketed
    fake = iter([{"vrange": None, "pairs_in_range": 0, "mean_occupancy": 0, "refined": 0, "T_collect": None,
                  "collected_pairs": 0, "D": 5, "exact": True, "upper": 5},
                 {"vrange": None, "pairs_in_range": 0, "mean_occupancy": 0, "refined": 0, "T_collect": None,
                  "collected_pairs": 0, "D": 3, "exact": False, "upper": 5}])
    merged = _split_max(lambda lo, hi: next(fake), 0.0, 10.0, 0.1, 2, False, False, "t")
    assert merged["D"] == 5 and merged["upper"] == 5 and merged["exact"]


def test_shells():
    ks = squarefree_shell(1000)
    assert ks.min() > 500 and ks.max() <= 1000 and 504 not in ks and 507 not in ks and 505 in ks
    ps = prime_shell(100)
    assert list(ps) == [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
