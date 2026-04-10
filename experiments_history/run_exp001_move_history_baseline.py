"""History-track baseline launcher.

This keeps the move-history research line in its own folder while reusing the
existing exp160 implementation as the baseline.

Usage:
  python experiments_history/run_exp001_move_history_baseline.py \
    --train-pgn outputs/lichess_sf_games.pgn \
    --output-path outputs/exp_history_001/best.pt
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "experiments" / "exp160_move_history_transformer.py"


def load_exp160_main():
    spec = importlib.util.spec_from_file_location("exp160_move_history_transformer", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experiment source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing baseline experiment file: {SOURCE}")
    exp160_main = load_exp160_main()
    exp160_main()


if __name__ == "__main__":
    main()
