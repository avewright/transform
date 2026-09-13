#!/usr/bin/env python3
"""One-hot best-PV extract from Lichess/chess-position-evaluations, n < 14 pieces.

Keeps the deepest / highest-knodes / best-score first move per FEN.
Does not reconstruct MultiPV.

  MOVE_VOCAB_VERSION=compact python3 -u scripts/build_lichess_endgame_bestline.py --go
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from upload_exp201_hf import load_hf_token  # noqa: E402


def main() -> None:
    load_hf_token()
    sys.argv = [
        sys.argv[0],
        "--download",
        "--one-hot",
        "--min-pieces", "2",
        "--max-pieces", "13",
        "--min-depth", "0",
        "--min-knodes", "0",
        "--target", "0",
        "--flush-every", "100000",
        "--inbox", "outputs/lichess_endgame_bestline/inbox",
        "--output", "outputs/lichess_endgame_bestline/soft_cache.pt",
        *sys.argv[1:],
    ]
    # --go is accepted and ignored so launch scripts can pass it
    if "--go" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--go"]
    from build_lichess_evals_soft_cache import main as build_main

    build_main()


if __name__ == "__main__":
    main()
