"""Fresh-process scalability probe for deterministic fixed-list ECM.

Usage: python -m factorlab.experiments.run_fixed_scale BITS [INDEX] [U] [C]
Writes results/e22_fixed_scale_uU_BITS_INDEX.json (legacy u=3,C=2 filenames
remain unchanged).
"""

from __future__ import annotations

import json
import os
import sys

from ..bench import RESULTS_DIR
from .fixed_list_check import scalability_probe


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m factorlab.experiments.run_fixed_scale BITS [INDEX] [U] [C]")
    bits = int(sys.argv[1])
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    u = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    C = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    os.makedirs(RESULTS_DIR, exist_ok=True)
    row = scalability_probe(bits, index, u=u, C=C)
    if u == 3.0 and C == 2.0:
        filename = f"e22_fixed_scale_{bits}_{index}.json"
    else:
        us = str(u).replace(".", "p")
        cs = str(C).replace(".", "p")
        filename = f"e22_fixed_scale_u{us}_C{cs}_{bits}_{index}.json"
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as fh:
        json.dump(row, fh, indent=1)
    print(f"E22 bits={bits} index={index} u={u:g} C={C:g}: B1={row['B1']} B2={row['B2']} degree={row['stage2_degree']} "
          f"curve={row['curve']} stage={row['stage']} mulmod={row['mulmod']} poly_deg={row['poly_deg']} "
          f"wall={row['wall']:.3f}s peak_rss={row['peak_rss_kb']/1024:.1f}MB")
