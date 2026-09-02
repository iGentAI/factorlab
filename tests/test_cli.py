"""CLI smoke tests in a fresh interpreter so registry side effects from other
test modules cannot mask import-order problems."""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args, env_extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, "-m", "factorlab", *args], capture_output=True,
                          text=True, env=env, cwd=ROOT, timeout=300)


def test_cli_list():
    r = _run("list")
    assert r.returncode == 0, r.stderr
    assert "pollard_rho" in r.stdout and "squfof" in r.stdout


def test_cli_bench_fresh_process(tmp_path):
    r = _run("bench", "trial_division", "--bits", "20,24", "--count", "2",
             env_extra={"FACTORLAB_RESULTS": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    assert "found 2/2" in r.stdout


def test_cli_gen():
    r = _run("gen", "--bits", "32", "--count", "2", "--family", "rsa")
    assert r.returncode == 0, r.stderr
    rows = [json.loads(line) for line in r.stdout.strip().splitlines()]
    assert len(rows) == 2 and all(int(x["p"]) * int(x["q"]) == int(x["N"]) for x in rows)
