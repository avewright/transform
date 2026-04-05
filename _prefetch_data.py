"""Prefetch data for exp136 by downloading and tokenizing parquets directly.

This avoids HF datasets streaming (which times out) and instead uses
hf_hub_download to get individual parquet files, then tokenizes them
into a local .pt cache file.
"""
import sys, os, time, gc
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, ".")

import numpy as np
import torch
from data_loader import (
    get_hf_dataset_layout, _tokenize_parquet, _fast_parse_fen,
    board_array_to_fused, ep_square_to_file, compute_wdl,
    _hf_token, CACHE_DIR, UCI_TO_IDX,
)

N_TRAIN = 500_000
N_EVAL = 2_500
N_TOTAL = N_TRAIN + N_EVAL
MIN_DEPTH = 15
SEED = 42
REPO = "avewright/chess-positions-lichess-sf"

cache_path = CACHE_DIR / f"lichess_d{MIN_DEPTH}_n{N_TOTAL}_s{SEED}.pt"
if cache_path.exists():
    print(f"Cache already exists: {cache_path}")
    data = torch.load(cache_path, weights_only=True)
    print(f"  {data['board_array'].shape[0]:,} positions")
    sys.exit(0)

print(f"Building cache: {N_TOTAL:,} positions from {REPO}")
print(f"  Target: {cache_path}")

from huggingface_hub import hf_hub_download

# Get file list
layout = get_hf_dataset_layout(REPO)
token = _hf_token()

# Use src parquets (small, fast to download)
src_files = layout["train_src"]
print(f"  Available: {len(src_files)} src parquets")

# Accumulate positions
all_ba = []
all_turn = []
all_castling = []
all_ep = []
all_midx = []
all_cp = []
all_mate = []
all_depth = []
all_fen = []

t0 = time.time()
files_used = 0

total_pos = 0
for fi, pq_name in enumerate(src_files):
    if total_pos >= N_TOTAL:
        break
    
    try:
        local_path = hf_hub_download(
            REPO, pq_name, repo_type="dataset",
            token=token, revision=layout["revision"],
        )
    except Exception as e:
        print(f"  Skip {pq_name}: {e}")
        continue
    
    # Tokenize
    raw = _tokenize_parquet(local_path)
    if raw is None:
        continue
    
    n = raw["board_array"].shape[0]
    all_ba.append(raw["board_array"])
    all_turn.append(raw["turn"])
    all_castling.append(raw["castling"])
    all_ep.append(raw["ep_square"])
    all_midx.append(raw["move_idx"])
    all_cp.append(raw["cp"])
    all_mate.append(raw["mate"])
    
    # We need FENs for eval — reconstruct from board_array for a subset
    # (only needed for the first N_EVAL positions)
    
    files_used += 1
    total_pos = sum(a.shape[0] for a in all_ba)
    elapsed = time.time() - t0
    print(f"  File {fi+1}: +{n:,} = {total_pos:,} total ({elapsed:.0f}s, "
          f"{total_pos/max(elapsed,1):.0f} pos/s)", flush=True)

# Concatenate
print(f"\nConcatenating {files_used} files...")
ba = np.concatenate(all_ba)[:N_TOTAL]
turn = np.concatenate(all_turn)[:N_TOTAL]
castling = np.concatenate(all_castling)[:N_TOTAL]
ep = np.concatenate(all_ep)[:N_TOTAL]
midx = np.concatenate(all_midx)[:N_TOTAL]
cp = np.concatenate(all_cp)[:N_TOTAL]
mate = np.concatenate(all_mate)[:N_TOTAL]

# Shuffle deterministically
rng = np.random.RandomState(SEED)
perm = rng.permutation(ba.shape[0])
ba = ba[perm]
turn = turn[perm]
castling = castling[perm]
ep = ep[perm]
midx = midx[perm]
cp = cp[perm]
mate = mate[perm]

# We need FENs for the eval split (to build chess.Board for legal_move_mask)
# Reconstruct FEN from board_array for eval positions
print("Reconstructing FENs for eval split...")
PIECE_CHARS = ".PNBRQKpnbrqk"

def board_array_to_fen(ba_row, turn_val, castling_val, ep_val):
    """Reconstruct FEN from encoded board array."""
    fen_rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            sq = rank * 8 + file
            p = ba_row[sq]
            if p == 0:
                empty += 1
            else:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                row += PIECE_CHARS[p]
        if empty > 0:
            row += str(empty)
        fen_rows.append(row)
    
    board_str = "/".join(fen_rows)
    turn_str = "w" if turn_val == 0 else "b"
    
    castle_str = ""
    if castling_val & 8: castle_str += "K"
    if castling_val & 4: castle_str += "Q"
    if castling_val & 2: castle_str += "k"
    if castling_val & 1: castle_str += "q"
    if not castle_str: castle_str = "-"
    
    if ep_val >= 0:
        ep_file = chr(ord('a') + ep_val % 8)
        ep_rank = str(ep_val // 8 + 1)
        ep_str = ep_file + ep_rank
    else:
        ep_str = "-"
    
    return f"{board_str} {turn_str} {castle_str} {ep_str} 0 1"

fens = []
for i in range(N_TOTAL):
    fens.append(board_array_to_fen(ba[i], turn[i], castling[i], ep[i]))
    if (i + 1) % 100000 == 0:
        print(f"  FENs: {i+1:,}/{N_TOTAL:,}")

# Save cache
data = {
    "board_array": torch.from_numpy(ba),
    "turn": torch.from_numpy(turn),
    "castling": torch.from_numpy(castling),
    "ep_square": torch.from_numpy(ep),
    "move_idx": torch.from_numpy(midx),
    "cp": torch.from_numpy(cp),
    "mate": torch.from_numpy(mate),
    "depth": torch.zeros(N_TOTAL, dtype=torch.int16),  # not tracked per-file
    "fen": fens,
}

cache_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(data, cache_path)
sz = cache_path.stat().st_size / 1e6
elapsed = time.time() - t0
print(f"\nSaved: {cache_path} ({sz:.1f} MB)")
print(f"Total: {ba.shape[0]:,} positions in {elapsed:.0f}s")
