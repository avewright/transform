#!/usr/bin/env python3
"""Legacy entrypoint — delegates to harness.elo.

Historical default was book+Syzygy ON. For pure-policy champion Elo use:
  python -m harness.elo --ckpt PATH
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    # Preserve legacy positional CLI: elo_eval_latest.py CKPT [PREFIX] [--no-syzygy] ...
    argv = list(sys.argv[1:])
    # Default legacy aids ON unless caller opts out
    if "--no-book" not in argv and "--book" not in argv:
        argv.append("--book")
    if "--no-syzygy" not in argv and "--syzygy" not in argv:
        argv.append("--syzygy")
    # Normalize --no-syzygy: harness treats --syzygy as enable; strip conflicting
    if "--no-syzygy" in argv:
        argv = [a for a in argv if a not in ("--syzygy", "--no-syzygy")]
    from harness.elo import main as harness_main

    return harness_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
