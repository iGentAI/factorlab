from factorlab.bench import fit_exponent, run_suite, summarize


def test_fit_exponent_recovers_slope():
    Ns = [2 ** b for b in range(20, 60, 4)]
    ys = [3.0 * float(N) ** 0.25 for N in Ns]
    fit = fit_exponent(Ns, ys)
    assert abs(fit.exponent - 0.25) < 1e-9
    assert fit.r2 > 0.999999


def test_run_suite_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("factorlab.bench.RESULTS_DIR", str(tmp_path))
    recs = run_suite("trial_division", [20, 24, 28], count=3, family="balanced", seed=1,
                     experiment="t", verbose=False)
    assert len(recs) == 9 and all(r.result.found for r in recs)
    s = summarize(recs)
    # trial division on balanced semiprimes: work ~ p ~ N^{1/2}
    assert 0.3 < s["fit"].exponent < 0.7
    assert (tmp_path / "t.jsonl").exists()
