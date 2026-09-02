import json
import subprocess
import sys

from factorlab.experiments.energy_stats import modfree_energy, planar_top_mass, update_archive
from factorlab.experiments.sidon_bucketed import d_star_bucketed, squarefree_shell
from factorlab.gen import make_semiprime


def test_modfree_energy_basic_and_null():
    r = 2 ** 10
    act = modfree_energy(r)
    nul = modfree_energy(r, null=True, seed=1)
    # the diagonal alone contributes one per pair; the energy is near-minimal (Proposition E)
    assert act["energy"] >= act["pairs"] and act["energy_over_pairs"] < 2.0
    # the phase-randomised null has the same pair count and an energy within a few per cent
    assert nul["pairs"] == act["pairs"]
    assert abs(nul["energy"] - act["energy"]) < 0.05 * act["energy"]
    # a speed-centred window of half-width rho_r never holds more than the window maximum D*_1(r) (E26: 4 at 2^10)
    ks = squarefree_shell(r)
    dstar = d_star_bucketed(ks, r, theta=1.0)
    assert dstar["exact"] and act["D_max"] <= dstar["D_star"] == 4
    assert act["tail"][1] == act["pairs"]


def test_planar_top_mass_identities():
    bits = 32
    N = int(make_semiprime(bits, "rsa", 7, 0).N)
    r = int(round(N ** (1 / 3)))
    row = planar_top_mass(bits, r, m_values=(10, 50))
    # sum over all integers h of D(h) counts every ordered pair once per h within W of its start difference
    assert row["total_mass"] == row["pairs"] * (2 * row["W"] - 1)
    # the top-m means are non-increasing in m and bounded by D_max
    m10, m50 = row["top"][10]["mean_top_m"], row["top"][50]["mean_top_m"]
    assert row["D_max"] >= m10 >= m50 > 0
    assert row["count_u_with_D_ge"][row["D_max"]] >= 1


def test_update_archive_merges_by_key_and_keeps_other_keys(tmp_path):
    path = tmp_path / "e42.json"
    path.write_text(json.dumps({"other": {"kept": True},
                                "energy": [{"log2_r": 12, "v": "old"}],
                                "planar": [{"bits": 40, "radius": "third", "v": 1}]}))
    data = update_archive(str(path), energy_rows=[{"log2_r": 12, "v": "new"}, {"log2_r": 13, "v": "x"}],
                          planar_rows=[{"bits": 48, "radius": "quarter", "v": 2}])
    # same-key rows are replaced in place, new rows appended, unrelated keys untouched, and the file holds the result
    assert data["other"] == {"kept": True}
    assert [(z["log2_r"], z["v"]) for z in data["energy"]] == [(12, "new"), (13, "x")]
    assert [(z["bits"], z["radius"]) for z in data["planar"]] == [(40, "third"), (48, "quarter")]
    assert json.loads(path.read_text()) == data
    # a run selecting only one statistic leaves the other key alone; a fresh archive is created when absent
    data2 = update_archive(str(path), planar_rows=[{"bits": 40, "radius": "third", "v": 3}])
    assert data2["energy"] == data["energy"] and [z["v"] for z in data2["planar"]] == [3, 2]
    fresh = update_archive(str(tmp_path / "sub" / "new.json"), energy_rows=[{"log2_r": 8}])
    assert fresh == {"energy": [{"log2_r": 8}]}


def test_cli_rejects_bare_flags_and_nothing_to_do(tmp_path):
    out = tmp_path / "e42.json"
    for args in (["--energy"], ["--planar"], []):
        r = subprocess.run([sys.executable, "-m", "factorlab.experiments.energy_stats", *args, "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 2, (args, r.stderr)
    assert not out.exists()
