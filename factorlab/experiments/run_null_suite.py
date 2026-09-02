"""Driver for the D4 null suite."""

from __future__ import annotations

import json
import os

from ..bench import RESULTS_DIR
from .null_suite import conditional_residue_test


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("== D4: conditional residue distributions of p+q and q-p given N mod l ==")
    res = conditional_residue_test(nbits=64, count=20000)
    for k, v in sorted(res["per_modulus"].items(), key=lambda kv: (int(kv[0].split(':')[0]), kv[0])):
        print(f"  l,kind={k:8s} chi2={v['chi2']:8.2f} dof={v['dof']:3d} p={v['p']:.4f}  "
              f"outside_support={v['outside_support']}  p(wrong uniform model)={v['p_wrong_uniform_model']:.2e}")
    print(f"  model pass={res['pass']}   wrong model rejected wherever it differs={res['wrong_model_rejected_everywhere']}")
    with open(os.path.join(RESULTS_DIR, "d4_conditional_residues.json"), "w") as fh:
        json.dump(res, fh, indent=1)
