#!/usr/bin/env python3
"""Build or inspect the versioned chess_master dataset. No training, no upload."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

from chess_master.cli import main

if __name__ == "__main__":
    main()
