"""Max-Elo train/eval harness.

Champion metric: greedy policy Elo (no book, no Syzygy), pinned Stockfish.
Train screening: top1 only. Soft_loss never promotes.
"""

__version__ = "0.1.0"
