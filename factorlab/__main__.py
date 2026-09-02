"""Command-line entry point.

    python -m factorlab audit [--bits 48]
    python -m factorlab list
    python -m factorlab bench ALGO --bits 40,48,56 --count 5 --family balanced [--param k=v ...]
    python -m factorlab gen --bits 64 --family rsa --count 3
"""

from __future__ import annotations

import argparse
import json
import sys


def _parse_params(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="factorlab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--bits", type=int, default=48)

    sub.add_parser("list")

    b = sub.add_parser("bench")
    b.add_argument("algorithm")
    b.add_argument("--bits", default="32,40,48")
    b.add_argument("--count", type=int, default=5)
    b.add_argument("--family", default="balanced")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--param", action="append")
    b.add_argument("--gen-param", action="append")
    b.add_argument("--experiment")
    b.add_argument("--budget", type=float, default=None, help="soft per-instance wall budget (s)")
    b.add_argument("--wall", action="store_true", help="fit wall time instead of work")

    g = sub.add_parser("gen")
    g.add_argument("--bits", type=int, default=64)
    g.add_argument("--family", default="balanced")
    g.add_argument("--count", type=int, default=3)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--gen-param", action="append")

    args = ap.parse_args(argv)

    if args.cmd == "audit":
        from .audit import run_all
        res = run_all(args.bits)
        ok = all(v["pass"] for v in res.values())
        print("ALL PASS" if ok else "SOME FAILURES")
        return 0 if ok else 1

    if args.cmd == "list":
        from .registry import ALGORITHMS
        from . import algorithms  # noqa: F401
        from . import experiments  # noqa: F401
        for name, info in sorted(ALGORITHMS.items()):
            print(f"{name:20s} [{info.primary_key:10s}] {info.description}")
        return 0

    if args.cmd == "bench":
        from .bench import run_suite, summarize, print_summary
        bits = [int(x) for x in args.bits.split(",")]
        recs = run_suite(args.algorithm, bits, args.count, args.family, args.seed,
                         _parse_params(args.param), _parse_params(args.gen_param),
                         args.experiment, True, args.budget)
        print_summary(summarize(recs, use_wall=args.wall))
        return 0

    if args.cmd == "gen":
        from .gen import semiprime_suite
        for s in semiprime_suite(args.bits, args.count, args.family, args.seed, **_parse_params(args.gen_param)):
            print(json.dumps(s.to_json()))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
