"""Tests for E46 (difference-cover search)."""
from factorlab.experiments.cover_search import (
    cover_cost,
    greedy_G,
    lehman_starts,
    lemma_d_lower_bound,
    local_search,
    verify_cover,
)


def test_trivial_and_greedy_covers_are_valid():
    S = [10, 13, 20, 23, 30, 33, 41]
    G = greedy_G(S, [0])
    assert verify_cover(S, [0], G) and len(G) == len(S)
    G2 = greedy_G(S, [0, 3])
    assert verify_cover(S, [0, 3], G2) and len(G2) <= 4  # {10, 20, 30} cover 6 elements with offsets {0, 3}, plus 41


def test_local_search_never_worse_than_trivial_and_finds_structure():
    S = [100 + 7 * i for i in range(20)]  # an arithmetic progression is cheap to cover
    B, G, cost = local_search(S, 1, candidates=[7, 14, 21, 28, 35])
    assert verify_cover(S, B, G)
    assert cost <= cover_cost([0], S, 1)
    assert cost < 12  # {0,7,14,21,28} + 4 giants = 9


def test_lemma_d_bound_and_starts():
    assert lemma_d_lower_bound(100, 1, 1) == min(50, 2 * (10000 / 4) ** (1 / 3))
    S = lehman_starts(10007 * 10009, 100)
    assert len(S) == 50 and S == sorted(S)


def test_random_control_has_exact_cardinality_in_both_branches():
    import numpy as np

    from factorlab.experiments.cover_search import random_control

    rng = np.random.default_rng(1)
    small = random_control(0, 1000, 300, rng)
    assert len(small) == 300 == len(set(small)) and all(0 <= x <= 1000 for x in small)
    big = random_control(0, 10 ** 9, 500, rng)  # large-range branch
    assert len(big) == 500 == len(set(big)) and all(0 <= x <= 10 ** 9 for x in big)
