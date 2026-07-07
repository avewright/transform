"""Offline Stockfish labeling pipeline — the "dataset factory".

Generates a cached corpus of chess positions with:
  - FEN (normalized, deduplicated)
  - Stockfish eval for ALL legal moves (action-values)
  - Best move UCI
  - WDL estimation from eval
  - Game phase bucket (opening/middlegame/endgame)
  - Source metadata (origin, game_id for leakage-free splits)
  - Top-k move data with centipawn deltas

Usage:
  # Random position generation + labeling (legacy):
  python label_positions.py --num 5000 --depth 8 --output data/sf_labels_5k_d8.jsonl

  # Lichess JSONL relabeling with game-level split:
  python label_positions.py --source data/lichess_games.jsonl --depth 8 \
      --output data/lichess_sf.jsonl --split-by-game

  # Build reproducible shards:
  python label_positions.py --source data/lichess_games.jsonl --depth 8 \
      --output data/lichess_sf.jsonl --split-by-game --write-splits

Output format (JSONL, one JSON object per line):
  {
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "phase": "opening",
    "best_uci": "e7e5",
    "best_cp": -48,
    "move_values": [
      {"uci": "e7e5", "cp": -48, "type": "cp"},
      {"uci": "d7d5", "cp": -52, "type": "cp"},
      ...
    ],
    "wdl": [0.43, 0.14, 0.43],
    "num_legal": 20,
    "source": "lichess",
    "game_id": "abc123"
  }
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import chess


def _find_stockfish() -> str:
    """Auto-detect Stockfish binary (Windows or Linux)."""
    candidates = [
        "stockfish/stockfish/stockfish-windows-x86-64-avx2.exe",
        "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
        "stockfish",
    ]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c))
    # Fall back to PATH
    return "stockfish"


STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH") or _find_stockfish()


def classify_phase(board: chess.Board) -> str:
    """Classify position as opening/middlegame/endgame by material + move number."""
    material = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
            material += vals.get(piece.piece_type, 0)
    if board.fullmove_number <= 10 and material >= 60:
        return "opening"
    elif material <= 20:
        return "endgame"
    return "middlegame"


def normalize_fen(board: chess.Board) -> str:
    """Normalize FEN for deduplication (board + turn + castling + ep)."""
    return board.board_fen() + (" w " if board.turn else " b ") + board.castling_xfen()


def generate_positions(n: int, seed: int = 42) -> list[chess.Board]:
    """Generate diverse, deduplicated positions via random play with stratified ply depths."""
    random.seed(seed)
    seen = set()
    positions = []

    ply_ranges = [
        (4, 15, 0.25),    # opening
        (16, 40, 0.45),   # middlegame
        (41, 80, 0.20),   # endgame
        (10, 30, 0.10),   # extra middlegame diversity
    ]

    for min_ply, max_ply, fraction in ply_ranges:
        target = int(n * fraction)
        collected = 0
        attempts = 0
        while collected < target and attempts < target * 20:
            board = chess.Board()
            ply = random.randint(min_ply, max_ply)
            for _ in range(ply):
                if board.is_game_over():
                    break
                board.push(random.choice(list(board.legal_moves)))
            if not board.is_game_over() and list(board.legal_moves):
                key = normalize_fen(board)
                if key not in seen:
                    seen.add(key)
                    positions.append(board.copy())
                    collected += 1
            attempts += 1

    random.shuffle(positions)
    return positions[:n]


def cp_to_wdl(cp: int, eval_type: str = "cp") -> list[float]:
    """Convert centipawn eval to [win, draw, loss] probabilities.

    Uses sigmoid scaling consistent with LC0/Lichess WDL model.
    """
    if eval_type == "mate":
        if cp > 0:
            return [1.0, 0.0, 0.0]
        elif cp < 0:
            return [0.0, 0.0, 1.0]
        return [0.0, 1.0, 0.0]

    k = 1.0 / 111.7
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    # Draw model: peaks near 0 cp
    draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
    total = win + draw + loss
    return [round(win / total, 4), round(draw / total, 4), round(loss / total, 4)]


def label_position(sf, board: chess.Board,
                   source: str = "random_play", game_id: str | None = None) -> dict | None:
    """Label one position with Stockfish evals for all legal moves.

    Returns dict with fen, phase, best move, all move values, WDL, source metadata.
    """
    fen = board.fen()
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    move_values = []
    best_cp = None
    best_uci = None
    best_sort = -float("inf")

    for move in legal_moves:
        board.push(move)
        child_fen = board.fen()
        board.pop()

        try:
            sf.set_fen_position(child_fen)
            ev = sf.get_evaluation()
        except Exception:
            continue

        eval_type = ev.get("type", "cp")
        eval_value = ev.get("value", 0)

        # Negate: child position is from opponent's perspective
        cp_from_mover = -eval_value

        move_values.append({
            "uci": move.uci(),
            "cp": cp_from_mover,
            "type": eval_type,
        })

        if eval_type == "mate":
            sort_val = (100000 - abs(cp_from_mover)) if cp_from_mover > 0 else (-100000 + abs(cp_from_mover))
        else:
            sort_val = cp_from_mover

        if sort_val > best_sort:
            best_sort = sort_val
            best_uci = move.uci()
            best_cp = cp_from_mover

    if not move_values or best_uci is None:
        return None

    # Sort descending by eval (best first)
    def sort_key(mv):
        if mv["type"] == "mate":
            return (100000 - abs(mv["cp"])) if mv["cp"] > 0 else (-100000 + abs(mv["cp"]))
        return mv["cp"]
    move_values.sort(key=sort_key, reverse=True)

    return {
        "fen": fen,
        "phase": classify_phase(board),
        "best_uci": best_uci,
        "best_cp": best_cp,
        "move_values": move_values,
        "wdl": cp_to_wdl(best_cp, move_values[0]["type"]),
        "num_legal": len(legal_moves),
        "source": source,
        **({"game_id": game_id} if game_id else {}),
    }


def label_all(positions: list[chess.Board], depth: int, threads: int,
              output_path: Path, resume: bool = True) -> list[dict]:
    """Label all positions and write incrementally to JSONL."""
    from stockfish import Stockfish

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=depth,
        parameters={"Threads": threads, "Hash": 256},
    )

    # Resume support: count existing lines
    start_idx = 0
    if resume and output_path.exists():
        with open(output_path) as f:
            start_idx = sum(1 for _ in f)
        print(f"  Resuming from position {start_idx}")

    labeled = []
    t0 = time.time()
    mode = "a" if (resume and start_idx > 0) else "w"

    with open(output_path, mode) as f:
        for i in range(start_idx, len(positions)):
            entry = label_position(sf, positions[i])
            if entry:
                f.write(json.dumps(entry) + "\n")
                labeled.append(entry)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                done = i + 1 - start_idx
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(positions) - i - 1) / rate if rate > 0 else 0
                phases = {}
                for e in labeled[-100:]:
                    phases[e["phase"]] = phases.get(e["phase"], 0) + 1
                print(f"  {i+1}/{len(positions)} | {rate:.1f} pos/s | "
                      f"ETA {eta/60:.1f}m | phases: {phases}")

    return labeled


def compute_stats(output_path: Path) -> dict:
    """Compute summary statistics for the labeled dataset."""
    phases = {}
    sources = {}
    total_moves = 0
    cp_values = []
    n = 0
    seen_fens = set()
    dup_count = 0
    game_ids = set()

    with open(output_path) as f:
        for line in f:
            entry = json.loads(line)
            n += 1
            phases[entry["phase"]] = phases.get(entry["phase"], 0) + 1
            total_moves += entry["num_legal"]
            if entry["move_values"][0]["type"] == "cp":
                cp_values.append(entry["best_cp"])
            src = entry.get("source", "random_play")
            sources[src] = sources.get(src, 0) + 1
            fen_key = entry["fen"].rsplit(" ", 2)[0]  # strip halfmove/fullmove
            if fen_key in seen_fens:
                dup_count += 1
            seen_fens.add(fen_key)
            gid = entry.get("game_id")
            if gid:
                game_ids.add(gid)

    avg_moves = total_moves / max(n, 1)
    cp_mean = sum(cp_values) / max(len(cp_values), 1) if cp_values else 0
    cp_std = (sum((c - cp_mean)**2 for c in cp_values) / max(len(cp_values), 1)) ** 0.5 if cp_values else 0

    return {
        "total_positions": n,
        "unique_fens": len(seen_fens),
        "duplicate_fens": dup_count,
        "total_move_evals": total_moves,
        "avg_legal_moves": round(avg_moves, 1),
        "phase_distribution": phases,
        "source_distribution": sources,
        "unique_games": len(game_ids) if game_ids else None,
        "cp_mean": round(cp_mean, 1),
        "cp_std": round(cp_std, 1),
    }


def split_by_game(output_path: Path, seed: int = 42,
                  train_frac: float = 0.85, val_frac: float = 0.10) -> dict:
    """Split labeled data by game_id to prevent position leakage.

    Positions from the same game always land in the same split.
    Falls back to position-level split if no game_id metadata is present.

    Returns dict with split file paths and counts.
    """
    rng = random.Random(seed)

    # Group entries by game_id
    game_entries: dict[str, list[str]] = {}
    no_game_entries: list[str] = []

    with open(output_path) as f:
        for line in f:
            entry = json.loads(line)
            gid = entry.get("game_id")
            if gid:
                game_entries.setdefault(gid, []).append(line)
            else:
                no_game_entries.append(line)

    # Shuffle game IDs
    game_ids = list(game_entries.keys())
    rng.shuffle(game_ids)

    n_games = len(game_ids)
    n_train = int(n_games * train_frac)
    n_val = int(n_games * val_frac)

    train_ids = set(game_ids[:n_train])
    val_ids = set(game_ids[n_train:n_train + n_val])
    test_ids = set(game_ids[n_train + n_val:])

    # Also split no-game entries by position (fallback)
    rng.shuffle(no_game_entries)
    n_ng = len(no_game_entries)
    ng_train = int(n_ng * train_frac)
    ng_val = int(n_ng * val_frac)

    # Write split files
    base = output_path.with_suffix("")
    paths = {
        "train": Path(f"{base}_train.jsonl"),
        "val": Path(f"{base}_val.jsonl"),
        "test": Path(f"{base}_test.jsonl"),
    }
    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        with open(paths[split_name], "w") as f:
            for gid in split_ids:
                for line in game_entries[gid]:
                    f.write(line)
                    counts[split_name] += 1

    # Append no-game entries to their respective splits
    split_slices = [
        ("train", no_game_entries[:ng_train]),
        ("val", no_game_entries[ng_train:ng_train + ng_val]),
        ("test", no_game_entries[ng_train + ng_val:]),
    ]
    for split_name, lines in split_slices:
        with open(paths[split_name], "a") as f:
            for line in lines:
                f.write(line)
                counts[split_name] += 1

    # Save split metadata
    meta = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": round(1.0 - train_frac - val_frac, 4),
        "total_games": n_games,
        "total_positions": sum(counts.values()),
        "counts": counts,
        "paths": {k: str(v) for k, v in paths.items()},
        "no_game_id_positions": len(no_game_entries),
    }
    meta_path = output_path.with_suffix(".splits.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Split by game ({n_games} games):")
    for k, v in counts.items():
        print(f"    {k}: {v} positions")
    print(f"  Metadata: {meta_path}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Stockfish labeling pipeline")
    parser.add_argument("--num", type=int, default=5000, help="Number of positions to generate")
    parser.add_argument("--depth", type=int, default=8, help="Stockfish search depth per move")
    parser.add_argument("--threads", type=int, default=2, help="Stockfish threads")
    parser.add_argument("--output", type=str, default="data/sf_labels.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, don't resume")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to source JSONL (Lichess, etc.) — skips random generation")
    parser.add_argument("--split-by-game", action="store_true",
                        help="After labeling, split train/val/test by game_id")
    parser.add_argument("--write-splits", action="store_true",
                        help="Write separate train/val/test JSONL files")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Stockfish Labeling Pipeline ===")
    print(f"  Depth: {args.depth}")
    print(f"  Threads: {args.threads}")
    print(f"  Output: {output_path}")
    print(f"  Seed: {args.seed}")
    if args.source:
        print(f"  Source: {args.source}")

    t0 = time.time()

    if args.source:
        # Load from external source JSONL (e.g. Lichess games)
        source_path = Path(args.source)
        print(f"\n[1/3] Loading positions from {source_path}...")
        positions = []
        seen = set()
        with open(source_path) as f:
            for line in f:
                entry = json.loads(line)
                try:
                    board = chess.Board(entry["fen"])
                    key = normalize_fen(board)
                    if key not in seen and not board.is_game_over() and list(board.legal_moves):
                        seen.add(key)
                        positions.append(board.copy())
                except Exception:
                    continue
        if args.num and len(positions) > args.num:
            random.seed(args.seed)
            random.shuffle(positions)
            positions = positions[:args.num]
        print(f"  Loaded {len(positions)} unique positions (deduped from source)")
    else:
        # Generate random positions
        print(f"\n[1/3] Generating {args.num} diverse positions...")
        positions = generate_positions(args.num, seed=args.seed)
        print(f"  Generated {len(positions)} unique positions in {time.time()-t0:.1f}s")

    # Label
    print(f"\n[2/3] Labeling with Stockfish depth={args.depth}...")
    labeled = label_all(positions, depth=args.depth, threads=args.threads,
                        output_path=output_path, resume=not args.no_resume)

    # Stats
    print(f"\n[3/3] Summary:")
    stats = compute_stats(output_path)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save stats alongside
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # Optional game-level split
    if args.split_by_game or args.write_splits:
        print(f"\n[Split] Writing train/val/test splits by game...")
        split_meta = split_by_game(output_path, seed=args.seed)

    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
