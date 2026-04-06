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
PARQUET_GLOB = "outputs/lichess_cache/datasets--Lichess--chess-position-evaluations/snapshots/*/data/train-*-of-*.parquet"
HF_DATASET = "avewright/chess-positions-lichess-sf"


# ── Horizontal flip augmentation tables ──
# Chess has mirror symmetry: reflecting the board left-right (a↔h files)
# produces an equivalent position with equivalent best moves.

def _build_mirror_tables():
    """Build lookup tables for horizontal board flipping."""
    # Square mirror: flip file within each rank
    sq_table = torch.zeros(64, dtype=torch.long)
    for sq in range(64):
        rank, file = sq // 8, sq % 8
        sq_table[sq] = rank * 8 + (7 - file)

    # Move index mirror: remap from_sq and to_sq through square mirror
    move_table = torch.zeros(len(IDX_TO_UCI), dtype=torch.long)
    for idx, uci in enumerate(IDX_TO_UCI):
        fs = chess.parse_square(uci[:2])
        ts = chess.parse_square(uci[2:4])
        promo = uci[4:] if len(uci) > 4 else ""
        m_fs = (fs // 8) * 8 + (7 - fs % 8)
        m_ts = (ts // 8) * 8 + (7 - ts % 8)
        m_uci = chess.square_name(m_fs) + chess.square_name(m_ts) + promo
        move_table[idx] = UCI_TO_IDX[m_uci]

    # Castling mirror: K(8)↔Q(4), k(2)↔q(1)
    castling_table = torch.zeros(16, dtype=torch.long)
    for c in range(16):
        K, Q, k, q = bool(c & 8), bool(c & 4), bool(c & 2), bool(c & 1)
        castling_table[c] = (Q << 3) | (K << 2) | (q << 1) | int(k)

    return sq_table, move_table, castling_table


MIRROR_SQ, MIRROR_MOVE, MIRROR_CASTLING = _build_mirror_tables()


def hflip_board_array(board_array):
    """Horizontally flip board_array: (N, 64) or (64,)."""
    return board_array[..., MIRROR_SQ]


def hflip_move_idx(move_idx):
    """Mirror move indices through horizontal flip."""
    return MIRROR_MOVE[move_idx.long()]


def hflip_castling(castling):
    """Mirror castling rights: K↔Q, k↔q."""
    return MIRROR_CASTLING[castling.long()]


def hflip_ep_square(ep_square):
    """Mirror en passant square. -1 stays -1, otherwise flip file."""
    result = ep_square.clone().long()
    valid = result >= 0
    if valid.any():
        result[valid] = MIRROR_SQ[result[valid]]
    return result


def _maybe_load_hf_token_from_env():
    """Populate HF_TOKEN from a local .env file if the process does not have it."""
    if os.environ.get("HF_TOKEN"):
        return

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != "HF_TOKEN":
                continue
            value = value.strip().strip("'").strip('"')
            if value:
                os.environ["HF_TOKEN"] = value
                return
    except OSError:
        return


def _hf_token():
    _maybe_load_hf_token_from_env()
    return os.environ.get("HF_TOKEN")


def get_hf_dataset_layout(repo_id):
    """Return a canonical view of the parquet layout for a HF dataset repo."""
    from huggingface_hub import HfApi, list_repo_files

    token = _hf_token()
    api = HfApi(token=token)
    info = api.dataset_info(repo_id)
    files = list_repo_files(repo_id, repo_type="dataset", token=token)

    train_main = sorted([
        f for f in files
        if f.startswith("data/train-") and "of-" in f and f.endswith(".parquet")
    ])
    train_src = sorted([
        f for f in files
        if f.startswith("data/train-src") and f.endswith(".parquet")
    ])
    test_main = sorted([
        f for f in files
        if f.startswith("data/test-") and "of-" in f and f.endswith(".parquet")
    ])
    test_src = sorted([
        f for f in files
        if f.startswith("data/test-src") and f.endswith(".parquet")
    ])

    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "train_main": train_main,
        "train_src": train_src,
        "test_main": test_main,
        "test_src": test_src,
        "all_files": files,
    }


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


# ── Fast FEN parser (no chess.Board) ──

PIECE_MAP = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
}
CASTLING_MAP = {'K': 8, 'Q': 4, 'k': 2, 'q': 1}


def _fast_parse_fen(fen_str, arr_out):
    """Parse FEN string into board_array (written to arr_out[64]), turn, castling, ep_square.
    Pure string parsing — no chess.Board(). ~10× faster."""
    parts = fen_str.split(' ')
    board_part = parts[0]
    arr_out[:] = 0
    rank = 7
    file_idx = 0
    for ch in board_part:
        if ch == '/':
            rank -= 1
            file_idx = 0
        elif '1' <= ch <= '8':
            file_idx += int(ch)
        else:
            arr_out[rank * 8 + file_idx] = PIECE_MAP[ch]
            file_idx += 1
    turn = 0 if parts[1] == 'w' else 1
    castling = 0
    if parts[2] != '-':
        for ch in parts[2]:
            castling |= CASTLING_MAP.get(ch, 0)
    ep = -1
    if len(parts) > 3 and parts[3] != '-':
        ep = (int(parts[3][1]) - 1) * 8 + (ord(parts[3][0]) - ord('a'))
    return turn, castling, ep


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

    parquet_files = sorted(globmod.glob(str(Path(__file__).resolve().parent / PARQUET_GLOB)))
    if not parquet_files:
        return None

    print(f"  Building cache from {len(parquet_files)} parquet shard(s)...")
    t0 = time.time()

    dfs = [pd.read_parquet(p, columns=["fen", "line", "depth", "cp", "mate"]) for p in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"  Read {len(df):,} rows from {len(parquet_files)} shards in {time.time()-t0:.1f}s")

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
    """Build from HF streaming dataset.

    Handles both pre-encoded schema (board_array, move_idx, ...) and
    the raw lichess-sf schema (fen, best_move, eval_type, eval_value, ...).
    """
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

    ds = ds.shuffle(seed=seed, buffer_size=100000)

    # Peek at first row to detect schema
    ds_iter = iter(ds)
    first_row = next(ds_iter)
    pre_encoded = "board_array" in first_row

    board_arrays = []
    turns = []
    castlings = []
    ep_squares = []
    move_idxs = []
    cps = []
    mates = []
    depths = []
    fens = []

    import itertools
    all_rows = itertools.chain([first_row], ds_iter)

    for row in all_rows:
        if len(board_arrays) >= n_total:
            break

        if pre_encoded:
            # Pre-encoded schema: fields are ready to use
            board_arrays.append(row["board_array"])
            turns.append(row["turn"])
            castlings.append(row["castling"])
            ep_squares.append(row["ep_square"])
            move_idxs.append(row["move_idx"])
            cps.append(row["cp"])
            mates.append(row["mate"])
            depths.append(row["depth"])
            fens.append(row["fen"])
        else:
            # Raw lichess-sf schema: parse from fen/best_move/eval_*
            try:
                best_uci = row["best_move"]
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

                eval_type = row.get("eval_type", "cp")
                eval_value = int(row.get("eval_value", 0))
                if eval_type == "mate":
                    cps.append(0)
                    mates.append(eval_value)
                else:
                    cps.append(eval_value)
                    mates.append(0)

                depths.append(int(row["depth"]))
                fens.append(fen)
            except Exception:
                continue

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
    eval_surviving = []  # track which indices successfully built
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
            eval_surviving.append(i)
        except Exception:
            continue
    print(f"  Built {len(eval_data)} eval boards in {time.time()-t0:.1f}s")

    # Build tensors — use surviving indices to keep alignment
    t1 = time.time()
    train_ba = raw["board_array"][n_eval_actual:n_eval_actual + n_train_actual]
    eval_idx = torch.tensor(eval_surviving, dtype=torch.long)
    eval_ba = raw["board_array"][eval_idx]

    train_turn = raw["turn"][n_eval_actual:n_eval_actual + n_train_actual].long()
    train_castling = raw["castling"][n_eval_actual:n_eval_actual + n_train_actual].long()
    train_ep = ep_square_to_file(raw["ep_square"][n_eval_actual:n_eval_actual + n_train_actual].long())

    eval_turn = raw["turn"][eval_idx].long()
    eval_castling = raw["castling"][eval_idx].long()
    eval_ep = ep_square_to_file(raw["ep_square"][eval_idx].long())

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


# ── Sharded data pipeline for mass training ──
#
# For datasets that don't fit comfortably as one .pt cache (>10M rows),
# pretokenize parquet into compact shard files, then iterate with
# ShardedChessLoader. Each shard is loaded one at a time — full dataset
# never materialized in RAM.
#
# Usage:
#   pretokenize_parquet_to_shards(parquet_files, shard_dir, n_eval=5000)
#   eval_data, eval_tensors = build_eval_from_pretokenized(shard_dir / "eval.pt")
#   loader = ShardedChessLoader(shard_dir, batch_size=256, device="cuda")
#   for batch_input, move_targets, wdl_targets in loader:
#       result = model(batch_input)


def _flush_shard(output_dir, shard_idx, ba, turns, castlings,
                 eps, midxs, cp_arr, mate_arr, depth_arr, n):
    """Write a pretokenized shard to disk atomically (tmp + rename)."""
    shard_path = output_dir / f"shard_{shard_idx:05d}.pt"
    tmp_path = output_dir / f"shard_{shard_idx:05d}.pt.tmp"
    torch.save({
        "board_array": torch.from_numpy(ba[:n].copy()),
        "turn": torch.from_numpy(turns[:n].copy()),
        "castling": torch.from_numpy(castlings[:n].copy()),
        "ep_square": torch.from_numpy(eps[:n].copy()),
        "move_idx": torch.from_numpy(midxs[:n].copy()),
        "cp": torch.from_numpy(cp_arr[:n].copy()),
        "mate": torch.from_numpy(mate_arr[:n].copy()),
        "depth": torch.from_numpy(depth_arr[:n].copy()),
    }, tmp_path)
    os.rename(tmp_path, shard_path)


def _flush_eval(output_dir, eval_ba, eval_turns, eval_castlings,
                eval_eps, eval_midxs, eval_cps, eval_mates, eval_fens, n):
    """Write eval.pt to disk atomically."""
    output_dir = Path(output_dir)
    eval_path = output_dir / "eval.pt"
    tmp_path = output_dir / "eval.pt.tmp"
    torch.save({
        "board_array": torch.from_numpy(eval_ba[:n].copy()),
        "turn": torch.from_numpy(eval_turns[:n].copy()),
        "castling": torch.from_numpy(eval_castlings[:n].copy()),
        "ep_square": torch.from_numpy(eval_eps[:n].copy()),
        "move_idx": torch.from_numpy(eval_midxs[:n].copy()),
        "cp": torch.from_numpy(eval_cps[:n].copy()),
        "mate": torch.from_numpy(eval_mates[:n].copy()),
        "fen": eval_fens[:n],
    }, tmp_path)
    os.rename(tmp_path, eval_path)
    print(f"  Eval: {n:,} positions saved to {eval_path.name}")


def pretokenize_parquet_to_shards(parquet_files, output_dir, n_eval=5000,
                                   rows_per_shard=3_000_000):
    """Convert parquet files to compact pretokenized .pt shard files.

    Handles both avewright/chess-positions-lichess-sf schema
    (fen, best_move, eval_type, eval_value) and original Lichess schema
    (fen, line, cp, mate, depth).

    First n_eval valid rows are saved to eval.pt with FEN strings.
    Remaining rows are split into shard_XXXXX.pt files.

    Returns:
        (n_shards, total_train_positions)
    """
    import pyarrow.parquet as pq

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arr_buf = np.zeros(64, dtype=np.int8)
    shard_idx = 0

    # Preallocate accumulator arrays
    ba = np.zeros((rows_per_shard, 64), dtype=np.int8)
    turns_a = np.zeros(rows_per_shard, dtype=np.int8)
    castlings_a = np.zeros(rows_per_shard, dtype=np.int8)
    eps_a = np.zeros(rows_per_shard, dtype=np.int8)
    midxs_a = np.zeros(rows_per_shard, dtype=np.int32)
    cp_a = np.zeros(rows_per_shard, dtype=np.int32)
    mate_a = np.zeros(rows_per_shard, dtype=np.int32)
    depth_a = np.zeros(rows_per_shard, dtype=np.int16)
    pos = 0

    # Eval accumulators
    eval_fens = []
    eval_ba = np.zeros((n_eval, 64), dtype=np.int8)
    eval_turns = np.zeros(n_eval, dtype=np.int8)
    eval_castlings = np.zeros(n_eval, dtype=np.int8)
    eval_eps = np.zeros(n_eval, dtype=np.int8)
    eval_midxs = np.zeros(n_eval, dtype=np.int32)
    eval_cps = np.zeros(n_eval, dtype=np.int32)
    eval_mates = np.zeros(n_eval, dtype=np.int32)
    eval_pos = 0

    total_written = 0
    total_skipped = 0
    eval_written = False
    t0 = time.time()

    for pq_path in parquet_files:
        pq_path = Path(pq_path)
        print(f"  Reading {pq_path.name}...", flush=True)

        table = pq.read_table(pq_path)
        col_names = set(table.column_names)
        is_lichess_sf = "best_move" in col_names

        fen_col = table.column("fen").to_pylist()
        n_rows = len(fen_col)

        if is_lichess_sf:
            move_col = table.column("best_move").to_pylist()
            et_col = table.column("eval_type").to_pylist() if "eval_type" in col_names else None
            ev_col = table.column("eval_value").to_pylist() if "eval_value" in col_names else None
            depth_col = table.column("depth").to_pylist() if "depth" in col_names else None
        else:
            line_col = table.column("line").to_pylist()
            cp_col = table.column("cp").to_pylist() if "cp" in col_names else None
            mate_col_data = table.column("mate").to_pylist() if "mate" in col_names else None
            depth_col = table.column("depth").to_pylist() if "depth" in col_names else None

        del table
        gc.collect()

        for i in range(n_rows):
            if is_lichess_sf:
                best_uci = move_col[i]
            else:
                line = line_col[i]
                if not line:
                    total_skipped += 1
                    continue
                best_uci = line.split()[0]

            if best_uci not in UCI_TO_IDX:
                total_skipped += 1
                continue

            fen = fen_col[i]
            try:
                turn, castling, ep = _fast_parse_fen(fen, arr_buf)
            except Exception:
                total_skipped += 1
                continue

            if is_lichess_sf:
                et = et_col[i] if et_col else "cp"
                ev = int(ev_col[i]) if ev_col and ev_col[i] is not None else 0
                cp_val = 0 if et == "mate" else ev
                mate_val = ev if et == "mate" else 0
            else:
                cp_val = int(cp_col[i]) if cp_col and cp_col[i] is not None else 0
                mate_val = int(mate_col_data[i]) if mate_col_data and mate_col_data[i] is not None else 0

            depth_val = int(depth_col[i]) if depth_col and depth_col[i] is not None else 0

            # Fill eval first
            if eval_pos < n_eval:
                eval_fens.append(fen)
                eval_ba[eval_pos] = arr_buf
                eval_turns[eval_pos] = turn
                eval_castlings[eval_pos] = castling
                eval_eps[eval_pos] = ep
                eval_midxs[eval_pos] = UCI_TO_IDX[best_uci]
                eval_cps[eval_pos] = cp_val
                eval_mates[eval_pos] = mate_val
                eval_pos += 1
                # Write eval.pt as soon as we have enough (enables concurrent training)
                if eval_pos >= n_eval and not eval_written:
                    _flush_eval(output_dir, eval_ba, eval_turns, eval_castlings,
                               eval_eps, eval_midxs, eval_cps, eval_mates,
                               eval_fens, eval_pos)
                    eval_written = True
                continue

            # Training row
            ba[pos] = arr_buf
            turns_a[pos] = turn
            castlings_a[pos] = castling
            eps_a[pos] = ep
            midxs_a[pos] = UCI_TO_IDX[best_uci]
            cp_a[pos] = cp_val
            mate_a[pos] = mate_val
            depth_a[pos] = depth_val
            pos += 1

            if pos >= rows_per_shard:
                _flush_shard(output_dir, shard_idx, ba, turns_a, castlings_a,
                            eps_a, midxs_a, cp_a, mate_a, depth_a, pos)
                total_written += pos
                shard_idx += 1
                pos = 0
                elapsed = time.time() - t0
                print(f"    Shard {shard_idx-1}: {total_written:,} total "
                      f"({total_written/elapsed:,.0f} pos/s)", flush=True)

        # Free column lists
        del fen_col
        if is_lichess_sf:
            del move_col
        gc.collect()

    # Flush remaining
    if pos > 0:
        _flush_shard(output_dir, shard_idx, ba, turns_a, castlings_a,
                    eps_a, midxs_a, cp_a, mate_a, depth_a, pos)
        total_written += pos
        shard_idx += 1

    # Write eval shard if not already written (small datasets or early exit)
    if not eval_written:
        _flush_eval(output_dir, eval_ba, eval_turns, eval_castlings,
                   eval_eps, eval_midxs, eval_cps, eval_mates,
                   eval_fens, eval_pos)

    elapsed = time.time() - t0
    print(f"  Pretokenized {total_written:,} train + {eval_pos:,} eval "
          f"into {shard_idx} shards in {elapsed:.0f}s "
          f"({(total_written + eval_pos)/elapsed:,.0f} pos/s, "
          f"skipped={total_skipped:,})")

    return shard_idx, total_written


def build_eval_from_pretokenized(eval_path, encoder_type="fused"):
    """Build eval_data and eval_tensors from pretokenized eval.pt file.

    Returns:
        eval_data: list of dicts with "board", "move", "wdl", "phase"
        eval_tensors: dict of tensors for fast eval batching
    """
    print(f"  Loading eval data from {eval_path}...")
    raw = torch.load(eval_path, map_location="cpu", weights_only=False)

    n = raw["board_array"].shape[0]
    fens = raw["fen"]
    move_idxs = raw["move_idx"]
    wdl = compute_wdl(raw["cp"], raw["mate"])
    phases = compute_phase(fens)

    eval_data = []
    surviving = []
    for i in range(n):
        try:
            board = chess.Board(fens[i])
            uci = IDX_TO_UCI[move_idxs[i].item()]
            move = chess.Move.from_uci(uci)
            eval_data.append({
                "board": board,
                "move": move,
                "wdl": (wdl[i, 0].item(), wdl[i, 1].item(), wdl[i, 2].item()),
                "phase": phases[i],
            })
            surviving.append(i)
        except Exception:
            continue

    idx = torch.tensor(surviving, dtype=torch.long)
    eval_ba = raw["board_array"][idx]

    eval_tensors = {
        "turn": raw["turn"][idx].long(),
        "castling": raw["castling"][idx].long(),
        "ep_file": ep_square_to_file(raw["ep_square"][idx].long()),
    }

    if encoder_type in ("fused", "both"):
        eval_tensors["fused_ids"] = board_array_to_fused(eval_ba)
    if encoder_type in ("baseline", "both"):
        p, c = board_array_to_baseline(eval_ba)
        eval_tensors["baseline_piece_ids"] = p
        eval_tensors["baseline_color_ids"] = c

    print(f"  Eval: {len(eval_data)} positions ready")
    return eval_data, eval_tensors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Streaming HF loader — streams parquets one at a time from HuggingFace
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tokenize_parquet(pq_path):
    """Read a single parquet file → dict of numpy arrays ready for tensor conversion.

    Returns dict with keys: board_array, turn, castling, ep_square, move_idx, cp, mate
    or None if the file has no valid rows.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(pq_path)
    col_names = set(table.column_names)
    is_lichess_sf = "best_move" in col_names

    fen_col = table.column("fen").to_pylist()
    n_rows = len(fen_col)

    if is_lichess_sf:
        move_col = table.column("best_move").to_pylist()
        et_col = table.column("eval_type").to_pylist() if "eval_type" in col_names else None
        ev_col = table.column("eval_value").to_pylist() if "eval_value" in col_names else None
    else:
        line_col = table.column("line").to_pylist()
        cp_col = table.column("cp").to_pylist() if "cp" in col_names else None
        mate_col_data = table.column("mate").to_pylist() if "mate" in col_names else None

    del table

    # Preallocate
    ba = np.zeros((n_rows, 64), dtype=np.int8)
    turns = np.zeros(n_rows, dtype=np.int8)
    castlings = np.zeros(n_rows, dtype=np.int8)
    eps = np.zeros(n_rows, dtype=np.int8)
    midxs = np.zeros(n_rows, dtype=np.int32)
    cps = np.zeros(n_rows, dtype=np.int32)
    mates = np.zeros(n_rows, dtype=np.int32)
    arr_buf = np.zeros(64, dtype=np.int8)

    pos = 0
    for i in range(n_rows):
        if is_lichess_sf:
            best_uci = move_col[i]
        else:
            line = line_col[i]
            if not line:
                continue
            best_uci = line.split()[0]

        if best_uci not in UCI_TO_IDX:
            continue

        try:
            turn, castling, ep = _fast_parse_fen(fen_col[i], arr_buf)
        except Exception:
            continue

        if is_lichess_sf:
            et = et_col[i] if et_col else "cp"
            ev = int(ev_col[i]) if ev_col and ev_col[i] is not None else 0
            cp_val = 0 if et == "mate" else ev
            mate_val = ev if et == "mate" else 0
        else:
            cp_val = int(cp_col[i]) if cp_col and cp_col[i] is not None else 0
            mate_val = int(mate_col_data[i]) if mate_col_data and mate_col_data[i] is not None else 0

        ba[pos] = arr_buf
        turns[pos] = turn
        castlings[pos] = castling
        eps[pos] = ep
        midxs[pos] = UCI_TO_IDX[best_uci]
        cps[pos] = cp_val
        mates[pos] = mate_val
        pos += 1

    if pos == 0:
        return None

    return {
        "board_array": ba[:pos],
        "turn": turns[:pos],
        "castling": castlings[:pos],
        "ep_square": eps[:pos],
        "move_idx": midxs[:pos],
        "cp": cps[:pos],
        "mate": mates[:pos],
    }


def build_eval_from_hf(repo_id, n_eval=5000, encoder_type="fused"):
    """Download the test split from HF and build eval_data + eval_tensors.

    Falls back to first train parquet if no test split exists.
    """
    from huggingface_hub import hf_hub_download

    layout = get_hf_dataset_layout(repo_id)
    token = _hf_token()

    if layout["test_main"]:
        target = layout["test_main"][0]
    elif layout["test_src"]:
        target = layout["test_src"][0]
    else:
        train_candidates = layout["train_main"] or layout["train_src"]
        target = train_candidates[0]

    print(f"  Downloading eval data: {target}...")
    local_path = hf_hub_download(
        repo_id,
        target,
        repo_type="dataset",
        token=token,
        revision=layout["revision"],
    )

    raw = _tokenize_parquet(local_path)
    if raw is None:
        raise RuntimeError(f"No valid eval positions in {target}")

    n = min(n_eval, raw["board_array"].shape[0])
    print(f"  Tokenized {raw['board_array'].shape[0]:,} eval candidates, using {n:,}")

    # Build eval_data (needs chess.Board for legal_move_mask)
    fens_for_phase = []
    arr_buf = np.zeros(64, dtype=np.int8)
    eval_data = []
    surviving = []
    wdl = compute_wdl(torch.tensor(raw["cp"][:n]), torch.tensor(raw["mate"][:n]))

    for i in range(n):
        try:
            # Reconstruct FEN from board_array for chess.Board
            board_arr = raw["board_array"][i]
            turn = raw["turn"][i]
            castling = raw["castling"][i]
            ep = raw["ep_square"][i]
            fen = _reconstruct_fen(board_arr, turn, castling, ep)
            board = chess.Board(fen)
            uci = IDX_TO_UCI[raw["move_idx"][i]]
            move = chess.Move.from_uci(uci)
            fens_for_phase.append(fen)
            eval_data.append({
                "board": board,
                "move": move,
                "wdl": (wdl[i, 0].item(), wdl[i, 1].item(), wdl[i, 2].item()),
                "cp": int(raw["cp"][i]),
                "mate": int(raw["mate"][i]),
                "phase": None,  # filled below
            })
            surviving.append(i)
        except Exception:
            continue

    phases = compute_phase(fens_for_phase)
    for j, ed in enumerate(eval_data):
        ed["phase"] = phases[j]

    idx = torch.tensor(surviving, dtype=torch.long)
    ba_t = torch.from_numpy(raw["board_array"][:n])[idx]
    turn_t = torch.from_numpy(raw["turn"][:n])[idx].long()
    cast_t = torch.from_numpy(raw["castling"][:n])[idx].long()
    ep_t = ep_square_to_file(torch.from_numpy(raw["ep_square"][:n])[idx].long())

    eval_tensors = {"turn": turn_t, "castling": cast_t, "ep_file": ep_t}
    if encoder_type in ("fused", "both"):
        eval_tensors["fused_ids"] = board_array_to_fused(ba_t)

    print(f"  Eval ready: {len(eval_data)} positions")
    return eval_data, eval_tensors


# Reverse board_array → FEN (for eval only — not performance-critical)
_INV_PIECE = {v: k for k, v in PIECE_MAP.items()}
_INV_CASTLING = {8: 'K', 4: 'Q', 2: 'k', 1: 'q'}

def _reconstruct_fen(board_arr, turn, castling, ep):
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            p = board_arr[rank * 8 + file]
            if p == 0:
                empty += 1
            else:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                row += _INV_PIECE[p]
        if empty > 0:
            row += str(empty)
        rows.append(row)
    fen = "/".join(rows)
    fen += " w " if turn == 0 else " b "
    c = ""
    for bit, ch in sorted(_INV_CASTLING.items(), key=lambda x: -x[0]):
        if castling & bit:
            c += ch
    fen += c if c else "-"
    if ep >= 0:
        fen += " " + chr(ord('a') + ep % 8) + str(ep // 8 + 1)
    else:
        fen += " -"
    fen += " 0 1"
    return fen


class StreamingHFChessLoader:
    """Stream training data from HuggingFace parquet files, one file at a time.

    Downloads each parquet via hf_hub_download (HF handles caching),
    tokenizes in RAM (~20MB per source parquet), yields minibatches,
    then drops it and loads the next file.

    Peak RAM: ~50MB (1 tokenized parquet + GPU batch tensors).
    Disk: only HF's download cache (~4MB per file, auto-managed).

    Cursor-based resume: call get_cursor() at any point to snapshot the
    iteration state. Pass that dict back as resume_cursor= to skip ahead
    to exactly where you left off (same file, same batch offset).

    Args:
        repo_id: HuggingFace dataset repo (e.g. "avewright/chess-positions-lichess-sf")
        batch_size: minibatch size
        encoder_type: "fused" or "baseline"
        device: target device for tensors
        seed: shuffle seed (determines file order)
        drop_last: drop last incomplete batch
        file_pattern: glob pattern for selecting parquet files (default: source shards)
        start_file: skip this many files from the *sorted* list before shuffling
        max_files: limit number of files to use (None = all)
        cache_dir: optional HF cache directory (None = default ~/.cache/huggingface)
        resume_cursor: dict from get_cursor() to resume mid-stream
    """

    def __init__(self, repo_id, batch_size, encoder_type="fused",
                 device="cpu", seed=42, drop_last=True,
                 file_pattern="src", start_file=0, max_files=None,
                 cache_dir=None, resume_cursor=None):
        self.repo_id = repo_id
        self.batch_size = batch_size
        self.encoder_type = encoder_type
        self.device = device
        self.seed = seed
        self.drop_last = drop_last
        self.cache_dir = cache_dir
        self.layout = get_hf_dataset_layout(repo_id)
        self.revision = self.layout["revision"]
        self.token = _hf_token()

        if file_pattern == "src":
            self.dataset_family = "train_src"
            self.parquet_files = list(self.layout["train_src"])
        elif file_pattern == "main":
            self.dataset_family = "train_main"
            self.parquet_files = list(self.layout["train_main"])
        else:
            self.dataset_family = f"custom:{file_pattern}"
            all_files = self.layout["all_files"]
            self.parquet_files = sorted([
                f for f in all_files
                if f.startswith("data/train") and f.endswith(".parquet")
                and file_pattern in f
            ])

        # Apply start_file and max_files
        self.parquet_files = self.parquet_files[start_file:]
        if max_files is not None:
            self.parquet_files = self.parquet_files[:max_files]

        # Deterministic file order (same seed → same permutation)
        rng = torch.Generator().manual_seed(self.seed)
        self._file_order = torch.randperm(len(self.parquet_files), generator=rng).tolist()

        # Estimate total positions (~254K per source parquet, ~3M per main)
        if file_pattern == "src":
            self._est_per_file = 254_000
        else:
            self._est_per_file = 3_000_000
        self._est_total = len(self.parquet_files) * self._est_per_file
        self._actual_total = 0  # updated as we iterate

        # Iteration cursor (updated during __iter__)
        self._files_completed = 0
        self._batches_in_current_file = 0
        self._positions_yielded = 0

        # Resume support
        self._resume_cursor = resume_cursor

        print(f"  StreamingHFChessLoader: {len(self.parquet_files)} parquet files"
              f" ({file_pattern}), est ~{self._est_total/1e6:.0f}M positions"
              f", batch={batch_size}, device={device}, rev={self.revision[:8]}")
        if resume_cursor:
            print(f"    Resuming from cursor: file_seq={resume_cursor['files_completed']}/"
                  f"{len(self.parquet_files)}, "
                  f"batch_offset={resume_cursor['batches_in_current_file']}, "
                  f"positions={resume_cursor['positions_yielded']:,}")

    @property
    def file_order(self):
        """The deterministic shuffled file index order."""
        return list(self._file_order)

    @property
    def fingerprint(self):
        """Stable hash of (sorted file list, seed). Use to verify checkpoint
        compatibility — if the fingerprint changes, the cursor is invalid."""
        import hashlib
        h = hashlib.sha256()
        h.update(str(self.seed).encode())
        for f in self.parquet_files:  # already sorted
            h.update(f.encode())
        return h.hexdigest()[:16]

    def get_cursor(self):
        """Return the current iteration state as a serializable dict.

        Save this in your checkpoint. Pass it back as resume_cursor= to
        restart iteration from exactly this point.
        """
        return {
            "files_completed": self._files_completed,
            "batches_in_current_file": self._batches_in_current_file,
            "positions_yielded": self._positions_yielded,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "dataset_revision": self.revision,
            "dataset_family": self.dataset_family,
        }

    @property
    def total_positions(self):
        """Actual total if we've iterated, else estimate."""
        return self._actual_total if self._actual_total > 0 else self._est_total

    @property
    def num_files(self):
        return len(self.parquet_files)

    def __iter__(self):
        from huggingface_hub import hf_hub_download
        import threading
        import queue

        # Use the pre-computed deterministic file order
        file_perm = self._file_order
        positions_yielded = 0
        files_done = 0

        # Resume: figure out where to skip to
        skip_files = 0
        skip_batches_in_file = 0
        if self._resume_cursor is not None:
            skip_files = self._resume_cursor["files_completed"]
            skip_batches_in_file = self._resume_cursor["batches_in_current_file"]
            positions_yielded = self._resume_cursor["positions_yielded"]
            if skip_files > 0:
                print(f"    [stream] Skipping {skip_files} completed files...",
                      flush=True)
            self._resume_cursor = None  # only apply cursor once

        # Determine which files to actually process
        remaining_perm = file_perm[skip_files:]

        # Prefetch buffer: download + tokenize next file in background thread
        prefetch_q = queue.Queue(maxsize=2)

        def _prefetch_worker(file_indices):
            """Background thread: download & tokenize parquets, put tensors in queue."""
            for fi in file_indices:
                pq_name = self.parquet_files[fi]
                try:
                    local_path = hf_hub_download(
                        self.repo_id, pq_name, repo_type="dataset",
                        cache_dir=self.cache_dir,
                        token=self.token,
                        revision=self.revision,
                    )
                    raw = _tokenize_parquet(local_path)
                except Exception as e:
                    print(f"    [stream] Error loading {pq_name}: {e}", flush=True)
                    raw = None
                prefetch_q.put((fi, raw))
            prefetch_q.put(None)  # sentinel

        worker = threading.Thread(
            target=_prefetch_worker, args=(remaining_perm,), daemon=True
        )
        worker.start()

        is_first_file_after_resume = (skip_batches_in_file > 0)

        while True:
            item = prefetch_q.get()
            if item is None:
                break  # sentinel from worker
            fi, raw = item

            if raw is None:
                files_done += 1
                self._files_completed = skip_files + files_done
                self._batches_in_current_file = 0
                continue

            n = raw["board_array"].shape[0]

            # Convert to tensors
            fused = board_array_to_fused(torch.from_numpy(raw["board_array"]))
            turn = torch.from_numpy(raw["turn"]).long()
            castling = torch.from_numpy(raw["castling"]).long()
            ep_file = ep_square_to_file(torch.from_numpy(raw["ep_square"]).long())
            move_idx = torch.from_numpy(raw["move_idx"]).long()
            wdl = compute_wdl(torch.from_numpy(raw["cp"]), torch.from_numpy(raw["mate"]))
            del raw

            # Shuffle within file (deterministic per file index)
            row_rng = torch.Generator().manual_seed(self.seed + fi * 31)
            perm = torch.randperm(n, generator=row_rng)

            batch_idx_in_file = 0
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                if self.drop_last and (end - start) < self.batch_size:
                    break

                # Skip batches we already yielded before the crash
                if is_first_file_after_resume and batch_idx_in_file < skip_batches_in_file:
                    batch_idx_in_file += 1
                    continue
                is_first_file_after_resume = False

                idx = perm[start:end]
                batch_input = {
                    "turn": turn[idx].to(self.device),
                    "castling": castling[idx].to(self.device),
                    "ep_file": ep_file[idx].to(self.device),
                }
                if self.encoder_type in ("fused", "both"):
                    batch_input["fused_ids"] = fused[idx].to(self.device)

                positions_yielded += idx.shape[0]
                batch_idx_in_file += 1

                # Update cursor state before yielding
                self._files_completed = skip_files + files_done
                self._batches_in_current_file = batch_idx_in_file
                self._positions_yielded = positions_yielded

                yield (batch_input,
                       move_idx[idx].to(self.device),
                       wdl[idx].float().to(self.device))

            del fused, turn, castling, ep_file, move_idx, wdl
            files_done += 1
            self._files_completed = skip_files + files_done
            self._batches_in_current_file = 0

            if files_done % 50 == 0:
                print(f"    [stream] {skip_files + files_done}/{len(self.parquet_files)} files, "
                      f"{positions_yielded:,} positions", flush=True)

        self._actual_total = positions_yielded


class ShardedChessLoader:
    """Streaming minibatch loader over pretokenized shard files.

    Reads one shard at a time, shuffles within shard, yields minibatches.
    Shard order reshuffled each epoch. Never materializes full dataset.

    Supports concurrent pretokenization: if expected_shards is set, the loader
    will wait for shard files to appear on disk before loading them.

    Yields (batch_input, move_targets, wdl_targets) tuples where:
      - batch_input: dict ready for model forward (fused_ids, turn, castling, ep_file)
      - move_targets: (B,) long tensor
      - wdl_targets: (B, 3) float tensor
    """

    def __init__(self, shard_dir, batch_size, encoder_type="fused",
                 device="cpu", seed=42, drop_last=True, skip_positions=0,
                 expected_shards=None, start_shard=0, hflip=False,
                 include_cp=False):
        self.shard_dir = Path(shard_dir)
        self.batch_size = batch_size
        self.encoder_type = encoder_type
        self.device = device
        self.seed = seed
        self.hflip = hflip
        self.include_cp = include_cp
        self.drop_last = drop_last
        self.skip_positions = skip_positions
        self.epoch = 0
        self.expected_shards = expected_shards

        def _data_shards(d):
            return sorted(f for f in d.glob("shard_*.pt")
                          if "_soft" not in f.stem)

        if expected_shards is not None:
            # Dynamic mode: wait for first shard, scan sizes lazily
            self._wait_for_shard(0)
            self.shard_files = _data_shards(self.shard_dir)
        else:
            self.shard_files = _data_shards(self.shard_dir)
        if not self.shard_files and expected_shards is None:
            raise FileNotFoundError(f"No shard_*.pt in {shard_dir}")

        # Drop leading shards (e.g. already trained on)
        if start_shard > 0:
            skipped = self.shard_files[:start_shard]
            self.shard_files = self.shard_files[start_shard:]
            print(f"  start_shard={start_shard}: skipping {len(skipped)} shard(s)")

        # Count rows per shard (only for already-existing shards)
        self._shard_sizes = []
        self._total = 0
        for sf in self.shard_files:
            data = torch.load(sf, map_location="cpu", weights_only=True)
            n = data["board_array"].shape[0]
            self._shard_sizes.append(n)
            self._total += n
            del data

        print(f"  ShardedChessLoader: {len(self.shard_files)} shards ready, "
              f"{self._total:,} positions, batch={batch_size}"
              f"{f', expecting {expected_shards} total' if expected_shards else ''}")

    def _wait_for_shard(self, shard_idx, timeout=600):
        """Block until shard file appears on disk (atomic rename ensures completeness)."""
        shard_path = self.shard_dir / f"shard_{shard_idx:05d}.pt"
        if shard_path.exists():
            return shard_path
        t0 = time.time()
        while not shard_path.exists():
            if time.time() - t0 > timeout:
                raise TimeoutError(f"Shard {shard_idx} not ready after {timeout}s")
            time.sleep(0.5)
        return shard_path

    @property
    def total_positions(self):
        return self._total

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        # In dynamic mode, this is an approximation until all shards are loaded
        effective = self._total - self.skip_positions
        if self.drop_last:
            return effective // self.batch_size
        return math.ceil(effective / self.batch_size)

    def update_total(self):
        """Refresh total_positions from shard sizes (call after all shards ready)."""
        self.shard_files = sorted(f for f in self.shard_dir.glob("shard_*.pt")
                                   if "_soft" not in f.stem)
        # Scan any new shards not yet counted
        for i in range(len(self._shard_sizes), len(self.shard_files)):
            data = torch.load(self.shard_files[i], map_location="cpu", weights_only=True)
            self._shard_sizes.append(data["board_array"].shape[0])
            self._total += data["board_array"].shape[0]
            del data

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch * 7919)

        if self.expected_shards is not None:
            # Dynamic mode: iterate sequentially, waiting for each shard
            n_shards = self.expected_shards
            shard_order = list(range(n_shards))
        else:
            n_shards = len(self.shard_files)
            shard_perm = torch.randperm(n_shards, generator=rng)
            shard_order = shard_perm.tolist()

        skip_remaining = self.skip_positions

        for si in shard_order:
            # In dynamic mode, wait for shard and register if new
            if self.expected_shards is not None:
                shard_path = self._wait_for_shard(si)
                if si >= len(self.shard_files):
                    self.shard_files = sorted(f for f in self.shard_dir.glob("shard_*.pt")
                                               if "_soft" not in f.stem)
                if si >= len(self._shard_sizes):
                    data = torch.load(shard_path, map_location="cpu", weights_only=True)
                    n = data["board_array"].shape[0]
                    self._shard_sizes.append(n)
                    self._total += n
                    del data

            shard_size = self._shard_sizes[si]

            # Skip entire shards for resume
            if skip_remaining >= shard_size:
                skip_remaining -= shard_size
                continue

            shard = torch.load(self.shard_files[si], map_location="cpu",
                              weights_only=True)
            n = shard["board_array"].shape[0]

            # Horizontal flip augmentation: flip a random 50% of positions
            if self.hflip:
                hflip_rng = torch.Generator().manual_seed(
                    self.seed + self.epoch * 3571 + si * 17)
                flip_mask = torch.rand(n, generator=hflip_rng) < 0.5
                if flip_mask.any():
                    ba = shard["board_array"].clone()
                    ba[flip_mask] = hflip_board_array(ba[flip_mask])
                    shard_ba = ba

                    mi = shard["move_idx"].clone()
                    mi[flip_mask] = hflip_move_idx(mi[flip_mask]).to(mi.dtype)
                    shard_mi = mi

                    ca = shard["castling"].clone()
                    # Zero castling rights for flipped positions: after hflip
                    # the king is on d-file, not e-file, so castling flags are
                    # inconsistent. Zeroing is safe — the flipped position just
                    # looks like a post-castling game.
                    ca[flip_mask] = 0
                    shard_ca = ca

                    ep = shard["ep_square"].clone()
                    ep[flip_mask] = hflip_ep_square(ep[flip_mask]).to(ep.dtype)
                    shard_ep = ep
                else:
                    shard_ba = shard["board_array"]
                    shard_mi = shard["move_idx"]
                    shard_ca = shard["castling"]
                    shard_ep = shard["ep_square"]
            else:
                shard_ba = shard["board_array"]
                shard_mi = shard["move_idx"]
                shard_ca = shard["castling"]
                shard_ep = shard["ep_square"]

            # Precompute derived fields for full shard
            fused = board_array_to_fused(shard_ba)
            turn = shard["turn"].long()
            castling_t = shard_ca.long()
            ep_file = ep_square_to_file(shard_ep.long())
            move_idx = shard_mi.long()
            wdl = compute_wdl(shard["cp"], shard["mate"])
            cp_vals = shard["cp"].float() if self.include_cp else None
            del shard, shard_ba, shard_mi, shard_ca, shard_ep

            # Shuffle within shard
            row_rng = torch.Generator().manual_seed(
                self.seed + self.epoch * 7919 + si * 31)
            perm = torch.randperm(n, generator=row_rng)

            # Skip batches within this shard for resume
            start_offset = 0
            if skip_remaining > 0:
                skip_batches = skip_remaining // self.batch_size
                start_offset = skip_batches * self.batch_size
                skip_remaining = 0

            for start in range(start_offset, n, self.batch_size):
                end = min(start + self.batch_size, n)
                if self.drop_last and (end - start) < self.batch_size:
                    break

                idx = perm[start:end]

                batch_input = {
                    "turn": turn[idx].to(self.device),
                    "castling": castling_t[idx].to(self.device),
                    "ep_file": ep_file[idx].to(self.device),
                }
                if self.encoder_type in ("fused", "both"):
                    batch_input["fused_ids"] = fused[idx].to(self.device)
                if self.encoder_type in ("baseline", "both"):
                    pass  # extend when needed
                if cp_vals is not None:
                    batch_input["cp"] = cp_vals[idx].to(self.device)

                yield (batch_input,
                       move_idx[idx].to(self.device),
                       wdl[idx].float().to(self.device))

            del fused, turn, castling_t, ep_file, move_idx, wdl, cp_vals


# ── CLI: Build cache manually ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build data cache or pretokenize")
    sub = parser.add_subparsers(dest="command")

    # Legacy cache builder
    cache_p = sub.add_parser("cache", help="Build .pt cache (legacy sampler)")
    cache_p.add_argument("--n-total", type=int, default=502500,
                        help="Total positions (train+eval)")
    cache_p.add_argument("--min-depth", type=int, default=15)
    cache_p.add_argument("--seed", type=int, default=42)

    # Pretokenize parquet shards
    pre_p = sub.add_parser("pretokenize", help="Pretokenize parquet shards")
    pre_p.add_argument("parquet_dir", help="Directory containing parquet files")
    pre_p.add_argument("output_dir", help="Output directory for shard .pt files")
    pre_p.add_argument("--n-eval", type=int, default=5000)
    pre_p.add_argument("--rows-per-shard", type=int, default=3_000_000)
    pre_p.add_argument("--glob", default="train-*-of-*.parquet",
                       help="Glob pattern for parquet files within parquet_dir")

    args = parser.parse_args()

    if args.command == "pretokenize":
        import glob as globmod
        parquet_files = sorted(globmod.glob(str(Path(args.parquet_dir) / args.glob)))
        if not parquet_files:
            print(f"ERROR: No files matching {args.glob} in {args.parquet_dir}")
        else:
            print(f"Found {len(parquet_files)} parquet files")
            pretokenize_parquet_to_shards(
                parquet_files, args.output_dir,
                n_eval=args.n_eval, rows_per_shard=args.rows_per_shard,
            )
    else:
        # Default/cache command
        n_total = getattr(args, 'n_total', 502500)
        min_depth = getattr(args, 'min_depth', 15)
        seed = getattr(args, 'seed', 42)
        cache_path = _cache_path(n_total, min_depth, seed)
        if cache_path.exists():
            print(f"Cache already exists: {cache_path}")
            data = _load_cache(cache_path)
            print(f"  {data['board_array'].shape[0]:,} positions")
        else:
            print("Building cache from parquet...")
            data = _build_from_parquet(n_total, min_depth, seed)
            if data:
                _save_cache(cache_path, data)
                print(f"  {data['board_array'].shape[0]:,} positions cached")
            else:
                print("ERROR: No parquet file found")
