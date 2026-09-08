#!/usr/bin/env python3
"""Build outputs/hf_elo_mix/deep_cache.pt from avewright/chess-soft-syzygy."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_hf_elo_mix import build_syzygy, log  # noqa: E402
import torch


def main() -> None:
    env = ROOT / ".env"
    if env.exists() and not os.environ.get("HF_TOKEN"):
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'").strip('"')
                break
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/hf_elo_mix")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400_000
    out.mkdir(parents=True, exist_ok=True)
    deep = build_syzygy(n)
    path = out / "deep_cache.pt"
    torch.save(deep, path)
    log(f"wrote {path} n={int(deep['board_array'].shape[0]):,}")


if __name__ == "__main__":
    main()
