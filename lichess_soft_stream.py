#!/usr/bin/env python3
"""Stream Lichess/chess-position-evaluations → soft MultiPV microbatches.

Denormalized parquet rows are grouped online into FEN soft targets (same
softmax/τ recipe as build_lichess_evals_soft_cache.py). Used to keep the GPU
busy while full RAM caches materialize.
"""
from __future__ import annotations

import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from data_loader import _fast_parse_fen, board_array_to_fused, compute_wdl, ep_square_to_file  # noqa: E402
from move_vocab import UCI_TO_IDX  # noqa: E402
from scripts.build_lichess_evals_soft_cache import (  # noqa: E402
    find_parquet_files,
    phase_from_board,
    white_score_to_stm,
)

SOFT_K = 8


def _fen_to_row(fen: str, moves: dict, tau: float) -> dict | None:
    if not moves:
        return None
    arr = np.zeros(64, dtype=np.int8)
    try:
        t, c, e = _fast_parse_fen(fen if fen.count(" ") >= 3 else fen + " 0 1", arr)
    except Exception:
        return None
    items = sorted(
        ((uci, trip[2], trip[0]) for uci, trip in moves.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:SOFT_K]
    scores = [s for _, s, _ in items]
    mx = max(scores)
    exps = [math.exp((s - mx) / tau) for s in scores]
    z = sum(exps) or 1.0
    probs = [x / z for x in exps]
    soft_i = np.full(SOFT_K, -1, dtype=np.int64)
    soft_p = np.zeros(SOFT_K, dtype=np.float32)
    for j, (uci, _s, _d) in enumerate(items):
        soft_i[j] = UCI_TO_IDX[uci]
        soft_p[j] = probs[j]
    best = items[0][0]
    # STM-centric cp proxy from best score (mate already mapped)
    best_score = items[0][1]
    cp = int(np.clip(best_score, -10000, 10000)) if abs(best_score) < 50000 else 0
    mate = 0
    if abs(best_score) >= 50000:
        mate = 1 if best_score > 0 else -1
        cp = 0
    return {
        "board_array": arr.copy(),
        "turn": np.int8(t),
        "castling": np.int8(c),
        "ep_square": np.int8(e),
        "move_idx": np.int64(UCI_TO_IDX[best]),
        "cp": np.int32(cp),
        "mate": np.int32(mate),
        "soft_indices": soft_i,
        "soft_probs": soft_p,
        "label_depth": np.int16(max(trip[0] for trip in moves.values())),
        "phase": np.int8(phase_from_board(arr)),
    }


def _stack_rows(rows: list[dict]) -> dict:
    return {
        "board_array": torch.from_numpy(np.stack([r["board_array"] for r in rows])),
        "turn": torch.from_numpy(np.stack([r["turn"] for r in rows])),
        "castling": torch.from_numpy(np.stack([r["castling"] for r in rows])),
        "ep_square": torch.from_numpy(np.stack([r["ep_square"] for r in rows])),
        "move_idx": torch.from_numpy(np.stack([r["move_idx"] for r in rows])),
        "cp": torch.from_numpy(np.stack([r["cp"] for r in rows])),
        "mate": torch.from_numpy(np.stack([r["mate"] for r in rows])),
        "soft_indices": torch.from_numpy(np.stack([r["soft_indices"] for r in rows])),
        "soft_probs": torch.from_numpy(np.stack([r["soft_probs"] for r in rows])),
    }


def iter_lichess_soft_batches(
    *,
    batch_size: int = 256,
    min_depth: int = 22,
    min_knodes: int = 5000,
    tau: float = 120.0,
    buffer_fens: int = 8192,
    batch_rows: int = 100_000,
    shards: list[Path] | None = None,
    shard_start: int = 4,
    seed: int = 0,
    infinite: bool = True,
) -> Iterator[dict]:
    """Yield soft-cache-shaped CPU batches forever (reshuffles shard order)."""
    files = shards or find_parquet_files(None)
    # Prefer virgin-ish shards by default (4..end); FT3/8M/12M chewed 0–3.
    if shards is None and shard_start > 0:
        files = files[shard_start:] or files
    rng = random.Random(seed)
    arr_buf = np.zeros(64, dtype=np.int8)  # noqa: F841 — kept for parity

    while True:
        order = list(files)
        rng.shuffle(order)
        # fen -> {uci: (depth, knodes, stm_score)} ; OrderedDict for eviction
        acc: OrderedDict[str, dict] = OrderedDict()
        ready: list[dict] = []

        def flush_oldest(n: int = 1) -> None:
            for _ in range(n):
                if not acc:
                    break
                fen, moves = acc.popitem(last=False)
                row = _fen_to_row(fen, moves, tau)
                if row is not None:
                    ready.append(row)

        for path in order:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=batch_rows,
                columns=["fen", "line", "depth", "knodes", "cp", "mate"],
            ):
                depth = batch.column("depth").to_numpy()
                kn = batch.column("knodes").to_numpy()
                mask = (depth >= min_depth) & (kn >= min_knodes)
                idxs = np.nonzero(mask)[0]
                if idxs.size == 0:
                    continue
                fens = batch.column("fen").to_pylist()
                lines = batch.column("line").to_pylist()
                cps = batch.column("cp").to_pylist()
                mates = batch.column("mate").to_pylist()
                for j in idxs:
                    fen = fens[j]
                    line = lines[j]
                    if not fen or not line:
                        continue
                    mv = line.split(" ", 1)[0]
                    if mv not in UCI_TO_IDX:
                        continue
                    parts = fen.split(" ")
                    if len(parts) < 2:
                        continue
                    turn_black = parts[1] == "b"
                    score = white_score_to_stm(cps[j], mates[j], turn_black)
                    if score is None:
                        continue
                    d, k = int(depth[j]), int(kn[j])
                    if fen in acc:
                        acc.move_to_end(fen)
                        old = acc[fen].get(mv)
                        if old is None or (d, k) > (old[0], old[1]):
                            acc[fen][mv] = (d, k, score)
                    else:
                        acc[fen] = {mv: (d, k, score)}
                        if len(acc) > buffer_fens:
                            flush_oldest(max(1, len(acc) - buffer_fens))

                while len(ready) >= batch_size:
                    chunk = ready[:batch_size]
                    del ready[:batch_size]
                    yield _stack_rows(chunk)

            # end of shard: flush some so we don't hold forever
            if len(acc) > buffer_fens // 2:
                flush_oldest(len(acc) - buffer_fens // 2)
            while len(ready) >= batch_size:
                chunk = ready[:batch_size]
                del ready[:batch_size]
                yield _stack_rows(chunk)

        # epoch end: flush rest
        flush_oldest(len(acc))
        while len(ready) >= batch_size:
            chunk = ready[:batch_size]
            del ready[:batch_size]
            yield _stack_rows(chunk)
        if not infinite:
            if ready:
                # pad last partial by repeating
                while len(ready) < batch_size:
                    ready.append(ready[len(ready) % max(len(ready), 1)])
                yield _stack_rows(ready[:batch_size])
            break


def batch_dict_to_train_tensors(batch: dict, device: torch.device):
    """Match prepare_soft_batch return signature (no hflip)."""
    ba = batch["board_array"]
    board_input = {
        "fused_ids": board_array_to_fused(ba).to(device, non_blocking=True),
        "turn": batch["turn"].long().to(device, non_blocking=True),
        "castling": batch["castling"].long().to(device, non_blocking=True),
        "ep_file": ep_square_to_file(batch["ep_square"]).long().to(device, non_blocking=True),
    }
    wdl = compute_wdl(batch["cp"], batch["mate"]).to(device, non_blocking=True)
    return (
        board_input,
        batch["move_idx"].long().to(device, non_blocking=True),
        wdl,
        batch["soft_indices"].to(device, non_blocking=True),
        batch["soft_probs"].to(device, non_blocking=True),
    )


if __name__ == "__main__":
    it = iter_lichess_soft_batches(batch_size=64, buffer_fens=2048, batch_rows=50_000, infinite=False)
    b = next(it)
    print({k: tuple(v.shape) for k, v in b.items()})
    print("soft_width", float((b["soft_indices"] >= 0).sum(1).float().mean()))
