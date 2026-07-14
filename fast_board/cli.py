#!/usr/bin/env python3
"""CLI: generate FENs or run benchmark / perft checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python -m fast_board.cli` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_board import benchmark, generate_fens, perft, version  # noqa: E402
from fast_board.generator import validate_fens  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Fast random legal chess board generator")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="Generate FENs")
    g.add_argument("-n", type=int, default=10)
    g.add_argument("--min-ply", type=int, default=8)
    g.add_argument("--max-ply", type=int, default=60)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("-o", "--output", type=str, default=None)

    b = sub.add_parser("bench", help="Throughput benchmark")
    b.add_argument("-n", type=int, default=50_000)
    b.add_argument("--min-ply", type=int, default=8)
    b.add_argument("--max-ply", type=int, default=40)
    b.add_argument("--no-compare", action="store_true")

    t = sub.add_parser("perft", help="Run perft from startpos")
    t.add_argument("--depth", type=int, default=5)

    v = sub.add_parser("validate", help="Generate + validate with python-chess")
    v.add_argument("-n", type=int, default=5000)
    v.add_argument("--min-ply", type=int, default=1)
    v.add_argument("--max-ply", type=int, default=80)
    v.add_argument("--seed", type=int, default=1)

    sub.add_parser("version")

    args = p.parse_args()

    if args.cmd == "version":
        print(version())
        return

    if args.cmd == "perft":
        # Expected: d1=20 d2=400 d3=8902 d4=197281 d5=4865609
        expected = {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609, 6: 119060324}
        for d in range(1, args.depth + 1):
            n = perft(d)
            ok = expected.get(d)
            status = "OK" if ok is None or n == ok else f"FAIL expected {ok}"
            print(f"perft({d}) = {n}  {status}")
        return

    if args.cmd == "bench":
        stats = benchmark(
            n=args.n,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            compare_python_chess=not args.no_compare,
        )
        print(json.dumps(stats, indent=2))
        return

    if args.cmd == "validate":
        fens = generate_fens(args.n, args.min_ply, args.max_ply, seed=args.seed)
        validate_fens(fens, sample=min(1000, len(fens)))
        print(f"OK: {len(fens)} FENs generated, sample validated ({version()})")
        return

    if args.cmd == "gen":
        fens = generate_fens(args.n, args.min_ply, args.max_ply, seed=args.seed)
        text = "\n".join(fens) + ("\n" if fens else "")
        if args.output:
            Path(args.output).write_text(text)
            print(f"wrote {len(fens)} FENs -> {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return


if __name__ == "__main__":
    main()
