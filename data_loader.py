"""Fast data loading for chess experiments.

Provides a unified loader that:
1. Checks for local .pt cache (loads in ~2s)
2. Falls back to HF streaming dataset
3. Falls back to raw parquet (builds cache for next time)

All experiments should use this instead of hand-rolling data loading.

Usage:
    from data_loader import load_training_data

    train, eval_data, eval_tensors = load_training_data(
        n_train=500_000,
        n_eval=2500,
        encoder_type="fused",  # or "baseline" or "both"
        seed=42,
    )
    # train["fused_ids"] is (N, 64) tensor ready for FusedBoardEncoder
    # train["move_idx"] is (N,) target tensor
    # eval_data is list of dicts with "board" key for legal_move_mask
"""

import gc
import math
import os
import random
import time
from pathlib import Path

import chess
import numpy as np
import torch

from chess_features import (
    NUM_PIECE_TYPES, NUM_COLORS, NUM_FUSED_TOKENS,
)
from move_vocab import UCI_TO_IDX, IDX_TO_UCI

CACHE_DIR = Path(__file__).resolve().parent / "outputs" / "data_cache"
PARQUET_GLOB = "outputs/lichess_cache/datasets--Lichess--chess-position-evaluations/snapshots/*/data/train-00000-of-00017.parquet"
HF_DATASET = "avewright/chess-positions-lichess-sf"


# ── Encoding-agnostic → specific encoder tensors ──

def board_array_to_fused(board_array):
    """board_array (N, 64) int → fused_ids (N, 64) long. Already the same encoding."""
    return board_array.long()


def board_array_to_baseline(board_array):
    """board_array (N, 64) int → piece_ids (N, 64), color_ids (N, 64).

    board_array: 0=empty, 1-6=white P..K, 7-12=black P..K
    piece_ids:   0=empty, 1-6=P..K (same for both colors)
    color_ids:   0=empty, 1=white, 2=black
    """
    ba = board_array.long()
    # piece_ids: for white (1-6) keep as-is, for black (7-12) subtract 6
    piece_ids = torch.where(ba <= 6, ba, ba - 6)
    # color_ids: 0 if empty, 1 if white (1-6), 2 if black (7-12)
    color_ids = torch.zeros_like(ba)
    color_ids[ba >= 1] = 1  # white
    color_ids[ba >= 7] = 2  # black
    return piece_ids, color_ids


def ep_square_to_file(ep_squares):
    """Convert ep_square (-1 or 0-63) to file encoding (0=none, 1-8=a-h)."""
    result = torch.zeros_like(ep_squares)
    mask = ep_squares >= 0
    result[mask] = (ep_squares[mask] % 8) + 1
    return result


# ── Cache I/O ──

def _cache_path(n_total, min_depth, seed):
    return CACHE_DIR / f"lichess_d{min_depth}_n{n_total}_s{seed}.pt"


def _save_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved cache: {path.name} ({size_mb:.1f} MB)")


def _load_cache(path):
    print(f"  Loading cache: {path.name}...")
    t0 = time.time()
    data = torch.load(path, weights_only=True)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return data


# ── Build from parquet ──

def _build_from_parquet(n_total, min_depth, seed):
    """Build encoding-agnostic cache from local parquet."""
    import glob as globmod
    import pandas as pd

    parquet_files = globmod.glob(str(Path(__file__).resolve().parent / PARQUET_GLOB))
    if not parquet_files:
        return None

    parquet_path = parquet_files[0]
    print(f"  Building cache from parquet: {Path(parquet_path).name}")
    t0 = time.time()

    df = pd.read_parquet(parquet_path, columns=["fen", "line", "depth", "cp", "mate"])
    print(f"  Read {len(df):,} rows in {time.time()-t0:.1f}s")

    df = df[df["depth"].notna() & (df["depth"] >= min_depth)]
    df = df[df["line"].notna() & (df["line"].str.len() > 0)]
    print(f"  After filter: {len(df):,}")

    rng = np.random.RandomState(seed)
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    board_arrays = []
    turns = []
    castlings = []
    ep_squares = []
    move_idxs = []
    cps = []
    mates = []
    depths = []
    fens = []

    t1 = time.time()
    for row_idx in range(min(len(df), n_total * 2)):  # overshoot for filtering
        if len(board_arrays) >= n_total:
            break
        try:
            row = df.iloc[row_idx]
            line = row["line"]
            best_uci = line.split()[0]
            if best_uci not in UCI_TO_IDX:
                continue

            fen = row["fen"]
            board = chess.Board(fen)
            move = chess.Move.from_uci(best_uci)
            if move not in board.legal_moves:
                continue

            # Board array (encoding-agnostic)
            arr = [0] * 64
            for sq, piece in board.piece_map().items():
                arr[sq] = piece.piece_type if piece.color else piece.piece_type + 6
            board_arrays.append(arr)

            turns.append(0 if board.turn == chess.WHITE else 1)
            castlings.append(
                (8 if board.has_kingside_castling_rights(chess.WHITE) else 0)
                | (4 if board.has_queenside_castling_rights(chess.WHITE) else 0)
                | (2 if board.has_kingside_castling_rights(chess.BLACK) else 0)
                | (1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
            )
            ep_squares.append(board.ep_square if board.ep_square is not None else -1)
            move_idxs.append(UCI_TO_IDX[best_uci])

            cp_val = int(row["cp"]) if pd.notna(row["cp"]) else 0
            mate_val = int(row["mate"]) if pd.notna(row["mate"]) else 0
            cps.append(cp_val)
            mates.append(mate_val)
            depths.append(int(row["depth"]))
            fens.append(fen)

            n = len(board_arrays)
            if n % 100000 == 0:
                elapsed = time.time() - t1
                print(f"    {n:,} ({n/max(elapsed,0.1):.0f} pos/s)...", flush=True)
        except Exception:
            continue

    del df
    gc.collect()

    tok_time = time.time() - t1
    n = len(board_arrays)
    print(f"  Built {n:,} positions in {tok_time:.1f}s ({n/max(tok_time,0.1):.0f} pos/s)")

    data = {
        "board_array": torch.tensor(board_arrays, dtype=torch.int8),
        "turn": torch.tensor(turns, dtype=torch.int8),
        "castling": torch.tensor(castlings, dtype=torch.int8),
        "ep_square": torch.tensor(ep_squares, dtype=torch.int8),
        "move_idx": torch.tensor(move_idxs, dtype=torch.int32),
        "cp": torch.tensor(cps, dtype=torch.int32),
        "mate": torch.tensor(mates, dtype=torch.int32),
        "depth": torch.tensor(depths, dtype=torch.int16),
        "fen": fens,
    }

    return data


# ── Build from HF streaming ──

def _build_from_hf(n_total, seed):
    """Build from HF streaming dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    print(f"  Streaming from HF: {HF_DATASET}")
    t0 = time.time()

    try:
        ds = load_dataset(HF_DATASET, split="train", streaming=True)
    except Exception as e:
        print(f"  HF dataset not available: {e}")
        return None

    ds = ds.shuffle(seed=seed, buffer_size=10000)

    board_arrays = []
    turns = []
    castlings = []
    ep_squares = []
    move_idxs = []
    cps = []
    mates = []
    depths = []
    fens = []

    for row in ds:
        if len(board_arrays) >= n_total:
            break
        board_arrays.append(row["board_array"])
        turns.append(row["turn"])
        castlings.append(row["castling"])
        ep_squares.append(row["ep_square"])
        move_idxs.append(row["move_idx"])
        cps.append(row["cp"])
        mates.append(row["mate"])
        depths.append(row["depth"])
        fens.append(row["fen"])

        if len(board_arrays) % 100000 == 0:
            elapsed = time.time() - t0
            print(f"    {len(board_arrays):,} ({len(board_arrays)/max(elapsed,0.1):.0f} pos/s)...",
                  flush=True)

    n = len(board_arrays)
    elapsed = time.time() - t0
    print(f"  Streamed {n:,} in {elapsed:.1f}s ({n/max(elapsed,0.1):.0f} pos/s)")

    data = {
        "board_array": torch.tensor(board_arrays, dtype=torch.int8),
        "turn": torch.tensor(turns, dtype=torch.int8),
        "castling": torch.tensor(castlings, dtype=torch.int8),
        "ep_square": torch.tensor(ep_squares, dtype=torch.int8),
        "move_idx": torch.tensor(move_idxs, dtype=torch.int32),
        "cp": torch.tensor(cps, dtype=torch.int32),
        "mate": torch.tensor(mates, dtype=torch.int32),
        "depth": torch.tensor(depths, dtype=torch.int16),
        "fen": fens,
    }
    return data


# ── WDL conversion ──

def compute_wdl(cp, mate):
    """Vectorized WDL from cp/mate tensors. Returns (N, 3) float tensor."""
    N = cp.shape[0]
    wdl = torch.zeros(N, 3)

    # Mate positions
    mate_pos = mate > 0
    mate_neg = mate < 0
    wdl[mate_pos, 0] = 1.0  # win
    wdl[mate_neg, 2] = 1.0  # loss

    # CP positions (no mate)
    no_mate = mate == 0
    k = 1.0 / 111.7
    cp_float = cp[no_mate].float()
    win = 1.0 / (1.0 + torch.exp(-k * cp_float))
    loss = 1.0 - win
    draw = torch.clamp(0.5 - torch.abs(win - 0.5), min=0.0) * 2
    total = win + draw + loss
    wdl[no_mate, 0] = win / total
    wdl[no_mate, 1] = draw / total
    wdl[no_mate, 2] = loss / total

    return wdl


def compute_phase(fens):
    """Compute game phase from FEN strings."""
    phases = []
    for fen in fens:
        board_part = fen.split()[0]
        n = sum(1 for c in board_part if c.isalpha() and c.lower() != 'k')
        if n >= 14:
            phases.append("opening")
        elif n >= 6:
            phases.append("middlegame")
        else:
            phases.append("endgame")
    return phases


# ── Main API ──

def load_training_data(
    n_train=500_000,
    n_eval=2500,
    encoder_type="fused",  # "fused", "baseline", or "both"
    min_depth=15,
    seed=42,
    include_weights=False,
):
    """Load pre-cached training data with instant tensor conversion.

    Returns:
        train_tensors: dict of tensors keyed by encoder input names + targets
        eval_data: list of dicts with "board", "move", "wdl", "phase"
        eval_tensors: dict of tensors for fast eval batching
    """
    n_total = n_train + n_eval
    cache_path = _cache_path(n_total, min_depth, seed)

    # 1. Try local cache
    if cache_path.exists():
        raw = _load_cache(cache_path)
    else:
        # 2. Try HF streaming
        raw = _build_from_hf(n_total, seed)
        if raw is None:
            # 3. Fall back to parquet
            raw = _build_from_parquet(n_total, min_depth, seed)
        if raw is None:
            raise RuntimeError("No data source available (no cache, no HF, no parquet)")

        # Save cache for next time
        _save_cache(cache_path, raw)

    n_available = raw["board_array"].shape[0]
    n_eval_actual = min(n_eval, n_available)
    n_train_actual = min(n_train, n_available - n_eval_actual)

    print(f"  Splitting: {n_train_actual:,} train, {n_eval_actual:,} eval")

    # Split: first n_eval → eval (need boards for legal_move_mask), rest → train
    eval_fens = raw["fen"][:n_eval_actual]
    eval_move_idxs = raw["move_idx"][:n_eval_actual]

    # Build eval_data with Board objects (only for eval, small count)
    t0 = time.time()
    eval_data = []
    eval_wdl = compute_wdl(raw["cp"][:n_eval_actual], raw["mate"][:n_eval_actual])
    eval_phases = compute_phase(eval_fens)

    for i in range(n_eval_actual):
        try:
            board = chess.Board(eval_fens[i])
            uci = IDX_TO_UCI[eval_move_idxs[i].item()]
            move = chess.Move.from_uci(uci)
            eval_data.append({
                "board": board,
                "move": move,
                "wdl": (eval_wdl[i, 0].item(), eval_wdl[i, 1].item(), eval_wdl[i, 2].item()),
                "phase": eval_phases[i],
            })
        except Exception:
            continue
    print(f"  Built {len(eval_data)} eval boards in {time.time()-t0:.1f}s")

    # Build tensors from encoding-agnostic board_array
    t1 = time.time()
    train_ba = raw["board_array"][n_eval_actual:n_eval_actual + n_train_actual]
    eval_ba = raw["board_array"][:len(eval_data)]

    train_turn = raw["turn"][n_eval_actual:n_eval_actual + n_train_actual].long()
    train_castling = raw["castling"][n_eval_actual:n_eval_actual + n_train_actual].long()
    train_ep = ep_square_to_file(raw["ep_square"][n_eval_actual:n_eval_actual + n_train_actual].long())

    eval_turn = raw["turn"][:len(eval_data)].long()
    eval_castling = raw["castling"][:len(eval_data)].long()
    eval_ep = ep_square_to_file(raw["ep_square"][:len(eval_data)].long())

    train_tensors = {
        "turn": train_turn,
        "castling": train_castling,
        "ep_file": train_ep,
        "move_idx": raw["move_idx"][n_eval_actual:n_eval_actual + n_train_actual].long(),
        "wdl": compute_wdl(
            raw["cp"][n_eval_actual:n_eval_actual + n_train_actual],
            raw["mate"][n_eval_actual:n_eval_actual + n_train_actual],
        ),
    }

    eval_tensors = {
        "turn": eval_turn,
        "castling": eval_castling,
        "ep_file": eval_ep,
    }

    # Add encoder-specific tensors
    if encoder_type in ("fused", "both"):
        train_tensors["fused_ids"] = board_array_to_fused(train_ba)
        eval_tensors["fused_ids"] = board_array_to_fused(eval_ba)

    if encoder_type in ("baseline", "both"):
        tp, tc = board_array_to_baseline(train_ba)
        train_tensors["baseline_piece_ids"] = tp
        train_tensors["baseline_color_ids"] = tc
        ep, ec = board_array_to_baseline(eval_ba)
        eval_tensors["baseline_piece_ids"] = ep
        eval_tensors["baseline_color_ids"] = ec

    if include_weights:
        # Compute quality weights from cp/depth
        train_cp = raw["cp"][n_eval_actual:n_eval_actual + n_train_actual]
        train_tensors["weight"] = _compute_weights(train_ba, train_cp)

    print(f"  Tensorized in {time.time()-t1:.1f}s")

    # Cleanup
    del raw
    gc.collect()

    print(f"  Ready: {n_train_actual:,} train, {len(eval_data):,} eval")
    return train_tensors, eval_data, eval_tensors


def _compute_weights(board_array, cp_tensor):
    """Compute sample weights from position features."""
    N = board_array.shape[0]
    weights = torch.ones(N, dtype=torch.float32)

    # Forced moves (only 1 reasonable response in obvious positions)
    abs_cp = cp_tensor.abs().float()
    weights[abs_cp > 500] = 0.6  # obvious
    weights[abs_cp > 200] = 1.0  # clear but interesting (override above)
    weights[abs_cp <= 200] = 0.9  # contested

    # Count non-empty squares to detect trivial endgame positions
    n_pieces = (board_array != 0).sum(dim=1).float()
    weights[n_pieces <= 4] = 0.3  # near-trivial endgame (K+piece vs K+piece)

    return weights


def get_batch_input(train_tensors, indices, encoder_type, device):
    """Extract a batch for the given encoder type."""
    result = {"turn": train_tensors["turn"][indices].to(device),
              "castling": train_tensors["castling"][indices].to(device),
              "ep_file": train_tensors["ep_file"][indices].to(device)}
    if encoder_type == "baseline":
        result["piece_ids"] = train_tensors["baseline_piece_ids"][indices].to(device)
        result["color_ids"] = train_tensors["baseline_color_ids"][indices].to(device)
    else:
        result["fused_ids"] = train_tensors["fused_ids"][indices].to(device)
    return result


def get_eval_batch_input(eval_tensors, idx_slice, encoder_type, device):
    """Extract eval batch."""
    result = {"turn": eval_tensors["turn"][idx_slice].to(device),
              "castling": eval_tensors["castling"][idx_slice].to(device),
              "ep_file": eval_tensors["ep_file"][idx_slice].to(device)}
    if encoder_type == "baseline":
        result["piece_ids"] = eval_tensors["baseline_piece_ids"][idx_slice].to(device)
        result["color_ids"] = eval_tensors["baseline_color_ids"][idx_slice].to(device)
    else:
        result["fused_ids"] = eval_tensors["fused_ids"][idx_slice].to(device)
    return result


# ── CLI: Build cache manually ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build data cache")
    parser.add_argument("--n-total", type=int, default=502500,
                        help="Total positions (train+eval)")
    parser.add_argument("--min-depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_path = _cache_path(args.n_total, args.min_depth, args.seed)
    if cache_path.exists():
        print(f"Cache already exists: {cache_path}")
        data = _load_cache(cache_path)
        print(f"  {data['board_array'].shape[0]:,} positions")
    else:
        print("Building cache from parquet...")
        data = _build_from_parquet(args.n_total, args.min_depth, args.seed)
        if data:
            _save_cache(cache_path, data)
            print(f"  {data['board_array'].shape[0]:,} positions cached")
        else:
            print("ERROR: No parquet file found")
