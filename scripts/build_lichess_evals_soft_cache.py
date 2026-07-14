#!/usr/bin/env python3
"""Build exp191-compatible soft_cache from Lichess/chess-position-evaluations.

Streams parquet with PyArrow (no datasets.map). Groups denormalized PV rows by
FEN, keeps the deepest / highest-knodes snapshot, reconstructs MultiPV soft
targets from first-moves via softmax(score/τ).

Lichess cp/mate are White-relative; we convert to STM scores for soft labels.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import PIECE_MAP, _fast_parse_fen  # noqa: E402
from move_vocab import UCI_TO_IDX  # noqa: E402

SOFT_K = 8
DEFAULT_REPO = "Lichess/chess-position-evaluations"


def log(msg: str) -> None:
    print(msg, flush=True)


def white_score_to_stm(cp: int | None, mate: int | None, turn_black: bool) -> int | None:
    """Map White-relative eval → STM-centric score (higher = better for side to move)."""
    if mate is not None:
        m = int(mate)
        # White mate>0 means White mates; for Black STM that is bad.
        sign = 1 if m > 0 else -1
        white_s = sign * (100_000 - min(abs(m), 1000))
    elif cp is None:
        return None
    else:
        white_s = int(cp)
    return -white_s if turn_black else white_s


def phase_from_board(arr: np.ndarray) -> int:
    """0=opening 1=middlegame 2=endgame (piece-count heuristic)."""
    n = int(np.count_nonzero(arr))
    if n >= 26:
        return 0
    if n >= 14:
        return 1
    return 2


def find_parquet_files(explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    hub = Path.home() / ".cache/huggingface/hub/datasets--Lichess--chess-position-evaluations"
    snaps = sorted(hub.glob("snapshots/*/data/data_*.parquet"))
    if snaps:
        return snaps
    raise FileNotFoundError(
        "No local Lichess eval parquets. Download with:\n"
        "  huggingface-cli download Lichess/chess-position-evaluations --repo-type dataset"
    )


def accumulate_shard(
    path: Path,
    acc: dict,
    *,
    min_depth: int,
    min_knodes: int,
    target: int,
    batch_rows: int,
) -> tuple[int, int]:
    """Update acc[fen] = {uci: (depth, knodes, stm_score)}.

    Keeps the best (depth, knodes) score per first-move across snapshots so soft
    width is not wiped when a deeper single-PV row arrives.
    """
    pf = pq.ParquetFile(path)
    rows = kept = 0
    for batch in pf.iter_batches(
        batch_size=batch_rows,
        columns=["fen", "line", "depth", "knodes", "cp", "mate"],
    ):
        depth = batch.column("depth").to_numpy()
        kn = batch.column("knodes").to_numpy()
        mask = (depth >= min_depth) & (kn >= min_knodes)
        idxs = np.nonzero(mask)[0]
        rows += len(depth)
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
            d = int(depth[j])
            k = int(kn[j])
            kept += 1
            moves = acc.get(fen)
            if moves is None:
                if len(acc) >= target:
                    continue
                acc[fen] = {mv: (d, k, score)}
                continue
            old = moves.get(mv)
            if old is None or (d, k) > (old[0], old[1]):
                moves[mv] = (d, k, score)
        if len(acc) >= target:
            # Cap fill: stop scanning this shard once target unique FENs hit.
            return rows, kept
        if rows and rows % (batch_rows * 2) < batch_rows:
            log(f"    … rows={rows:,} kept={kept:,} unique={len(acc):,}")
    return rows, kept


def materialize(acc: dict, tau: float) -> dict:
    n = len(acc)
    board = np.zeros((n, 64), dtype=np.int8)
    turn = np.zeros(n, dtype=np.int8)
    castling = np.zeros(n, dtype=np.int8)
    ep = np.zeros(n, dtype=np.int8)
    move_idx = np.zeros(n, dtype=np.int64)
    cp_out = np.zeros(n, dtype=np.int32)
    mate_out = np.zeros(n, dtype=np.int32)
    soft_i = np.full((n, SOFT_K), -1, dtype=np.int64)
    soft_p = np.zeros((n, SOFT_K), dtype=np.float32)
    label_depth = np.zeros(n, dtype=np.int16)
    phase = np.zeros(n, dtype=np.int8)

    arr_buf = np.zeros(64, dtype=np.int8)
    skip = 0
    out_i = 0
    for fen, moves in acc.items():
        if not moves:
            skip += 1
            continue
        try:
            t, c, e = _fast_parse_fen(fen if fen.count(" ") >= 3 else fen + " 0 1", arr_buf)
        except Exception:
            skip += 1
            continue
        # ranked by STM score desc; label_depth = max depth among kept moves
        items = sorted(((uci, trip[2], trip[0]) for uci, trip in moves.items()),
                       key=lambda x: x[1], reverse=True)[:SOFT_K]
        scores = [s for _, s, _ in items]
        mx = max(scores)
        exps = [math.exp((s - mx) / tau) for s in scores]
        z = sum(exps) or 1.0
        probs = [x / z for x in exps]
        best_uci = items[0][0]
        board[out_i] = arr_buf
        turn[out_i] = t
        castling[out_i] = c
        ep[out_i] = e
        move_idx[out_i] = UCI_TO_IDX[best_uci]
        best_stm = items[0][1]
        cp_out[out_i] = -best_stm if t == 1 else best_stm
        mate_out[out_i] = 0
        for k, (uci, _, _) in enumerate(items):
            soft_i[out_i, k] = UCI_TO_IDX[uci]
            soft_p[out_i, k] = probs[k]
        label_depth[out_i] = max(d for _, _, d in items)
        phase[out_i] = phase_from_board(arr_buf)
        out_i += 1

    board = board[:out_i]
    return {
        "board_array": torch.from_numpy(board.copy()),
        "turn": torch.from_numpy(turn[:out_i].copy()),
        "castling": torch.from_numpy(castling[:out_i].copy()),
        "ep_square": torch.from_numpy(ep[:out_i].copy()),
        "move_idx": torch.from_numpy(move_idx[:out_i].copy()),
        "cp": torch.from_numpy(cp_out[:out_i].copy()),
        "mate": torch.from_numpy(mate_out[:out_i].copy()),
        "soft_indices": torch.from_numpy(soft_i[:out_i].copy()),
        "soft_probs": torch.from_numpy(soft_p[:out_i].copy()),
        "label_depth": torch.from_numpy(label_depth[:out_i].copy()),
        "phase": torch.from_numpy(phase[:out_i].copy()),
        "_meta_skip": skip,
        "_meta_n": out_i,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/lichess_evals_soft/soft_cache.pt")
    ap.add_argument("--parquet", nargs="*", default=None)
    ap.add_argument("--min-depth", type=int, default=22)
    ap.add_argument("--min-knodes", type=int, default=5000)
    ap.add_argument("--target", type=int, default=3_000_000)
    ap.add_argument("--tau", type=float, default=120.0)
    ap.add_argument("--batch-rows", type=int, default=250_000)
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all available")
    ap.add_argument("--download", action="store_true", help="Ensure shards via hub download")
    args = ap.parse_args()

    if args.download:
        from huggingface_hub import snapshot_download

        log(f"downloading {DEFAULT_REPO}…")
        snapshot_download(DEFAULT_REPO, repo_type="dataset")

    files = find_parquet_files(args.parquet)
    if args.max_shards > 0:
        files = files[: args.max_shards]
    log(f"shards={len(files)} min_depth={args.min_depth} min_knodes={args.min_knodes} target={args.target:,}")

    acc: dict = {}
    t0 = time.time()
    total_rows = total_kept = 0
    for i, path in enumerate(files):
        if len(acc) >= args.target:
            log(f"target reached — stopping before {path.name}")
            break
        log(f"[{i+1}/{len(files)}] {path.name} acc={len(acc):,}")
        rows, kept = accumulate_shard(
            path,
            acc,
            min_depth=args.min_depth,
            min_knodes=args.min_knodes,
            target=args.target,
            batch_rows=args.batch_rows,
        )
        total_rows += rows
        total_kept += kept
        rate = len(acc) / max(time.time() - t0, 1e-6)
        log(
            f"  rows={rows:,} kept={kept:,} unique={len(acc):,} "
            f"({rate:.0f} fen/s wall)"
        )

    log(f"materialize unique={len(acc):,} tau={args.tau}")
    out = materialize(acc, args.tau)
    n = out.pop("_meta_n")
    skip = out.pop("_meta_skip")
    widths = (out["soft_indices"] >= 0).sum(dim=1).float()
    log(
        f"saved_rows={n:,} skip={skip} soft_width mean={widths.mean():.2f} "
        f"depth_mean={out['label_depth'].float().mean():.1f}"
    )
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(".pt.tmp")
    torch.save(out, tmp)
    os.replace(tmp, outp)
    mb = outp.stat().st_size / 1e6
    log(f"wrote {outp} ({mb:.1f} MB) in {time.time()-t0:.1f}s rows_scanned={total_rows:,}")


if __name__ == "__main__":
    main()
