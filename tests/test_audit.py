import pytest

from factorlab import audit


@pytest.mark.slow
def test_residue_uniformity():
    r = audit.residue_uniformity(40, n=2000)
    assert r["pass"], r


@pytest.mark.slow
def test_bit_uniformity():
    r = audit.bit_uniformity(40, n=2000)
    assert r["pass"], (r["worst_bit"], r["worst_p"])


@pytest.mark.slow
def test_density_profile():
    r = audit.density_profile(36, n=4000)
    assert r["pass"], r


@pytest.mark.slow
def test_next_prime_gap_bias_detected():
    r = audit.next_prime_gap_bias(40, n=1500)
    assert r["rejection_unbiased"], r
    assert r["next_prime_biased"], r
    assert 1.6 < r["ratio"] < 2.6, r


@pytest.mark.slow
def test_semiprime_residues():
    r = audit.semiprime_residues(64, n=1500)
    assert r["pass"], r
