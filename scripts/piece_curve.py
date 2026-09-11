"""Piece-count target: one truncated normal over n ∈ {2…32}.

Default N(17, 6) — peak in the middle of the board, openings and bare
endgames as tails. The harvest plays full games and keeps a row only when
that n is still under its share of the curve.
"""
from __future__ import annotations

import numpy as np

MIN_N = 2
MAX_N = 32
DEFAULT_MU = 17.0
DEFAULT_SIGMA = 6.0
STREAMS = ("opening", "middlegame", "endgame")
LABEL_WHEN_PIECES_LE = {
    "opening": 32,
    "middlegame": 23,
    "endgame": 12,
}


def target_pmf(mu: float = DEFAULT_MU, sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """p[n] for n=0..32. p[0]=p[1]=0."""
    p = np.zeros(MAX_N + 1, dtype=np.float64)
    xs = np.arange(MIN_N, MAX_N + 1, dtype=np.float64)
    q = np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    p[MIN_N : MAX_N + 1] = q / q.sum()
    return p


def n_pieces_from_row(row: dict) -> int:
    arr = np.asarray(row["board_array"]).reshape(-1)
    return int(np.count_nonzero(arr))


def should_keep(n: int, have: np.ndarray, total: int, pmf: np.ndarray, *, slack: float = 0.15) -> bool:
    if n < MIN_N or n > MAX_N:
        return False
    want = (total + 1) * float(pmf[n]) * (1.0 + slack)
    return int(have[n]) < want


def label_stride(n_pieces: int, *, default: int = 2) -> int:
    del n_pieces
    return max(1, default)


def empty_counts() -> np.ndarray:
    return np.zeros(MAX_N + 1, dtype=np.int64)


def recount_inbox_pieces(inbox) -> np.ndarray:
    """Sum n_pieces from READY shards. Used on harvest resume."""
    from pathlib import Path

    import torch

    have = empty_counts()
    root = Path(inbox)
    for cache in sorted(root.glob("shard_*/soft_cache.pt")):
        if not (cache.parent / "READY").exists():
            continue
        data = torch.load(cache, map_location="cpu", weights_only=False)
        pcs = (data["board_array"] != 0).sum(dim=1).cpu().numpy().astype(int)
        for n in pcs:
            if MIN_N <= int(n) <= MAX_N:
                have[int(n)] += 1
    return have


def most_deficit_n(have: np.ndarray, pmf: np.ndarray) -> int | None:
    """Piece count furthest below its share of the bell. None if the curve is filled."""
    total = int(have[MIN_N : MAX_N + 1].sum())
    best_n = None
    best_d = 0.5
    for n in range(MIN_N, MAX_N + 1):
        d = (total + 1) * float(pmf[n]) - float(have[n])
        if d > best_d:
            best_d = d
            best_n = n
    return best_n


def local_copy_paths(root) -> list:
    """Already-SF19 MultiPV rows we can copy without a new search."""
    from pathlib import Path

    root = Path(root)
    files: list[Path] = []
    for p in (
        root / "outputs/organized_chess_v1/sf19_train.pt",
        root / "outputs/organized_chess_resume/soft_cache.pt",
    ):
        if p.is_file():
            files.append(p)
    for inbox in (
        root / "outputs/sf19_soft/eco_1m/inbox",
        root / "outputs/sf19_soft/expand50m/inbox",
        root / "outputs/sf19_soft/gold_eco_1m/inbox",
        root / "outputs/sf19_soft/prod/inbox",
    ):
        if not inbox.is_dir():
            continue
        for cache in sorted(inbox.glob("shard_*/soft_cache.pt")):
            if (cache.parent / "READY").exists():
                files.append(cache)
    return files


def local_fen_paths(root) -> list:
    """Other local boards — FEN seeds for relabel / play, not copy-as-teacher."""
    from pathlib import Path

    root = Path(root)
    files = []
    for p in (
        root / "outputs/organized_chess_v1/lichess_train.pt",
        root / "outputs/organized_chess_v1/syzygy_train.pt",
        root / "outputs/organized_chess_v1/puzzles_train.pt",
        root / "outputs/hf_elo_mix/soft_cache.pt",
        root / "outputs/hf_elo_mix/deep_cache.pt",
        root / "outputs/hf100m_lapse_ft/soft_cache.pt",
        root / "outputs/exp193_tactical/soft_cache.pt",
        root / "outputs/mac_correction_v1/bucket/variant/soft_cache.pt",
        root / "outputs/hf_soft_mix_5m/soft_cache.pt",
    ):
        if p.is_file():
            files.append(p)
    return files


def hist_fracs(have: np.ndarray) -> dict[str, float]:
    tot = float(have[MIN_N : MAX_N + 1].sum()) or 1.0
    return {str(n): float(have[n]) / tot for n in range(MIN_N, MAX_N + 1)}


def curve_summary(have: np.ndarray) -> dict:
    sl = have[MIN_N : MAX_N + 1].astype(np.float64)
    tot = float(sl.sum()) or 1.0
    xs = np.arange(MIN_N, MAX_N + 1, dtype=np.float64)
    mean = float((xs * sl).sum() / tot)
    var = float(((xs - mean) ** 2 * sl).sum() / tot)
    return {
        "n": int(sl.sum()),
        "mean": round(mean, 2),
        "std": round(var ** 0.5, 2),
        "peak": int(xs[int(sl.argmax())]),
    }
