#!/usr/bin/env python3
"""Massively parallel data generation + SF labeling pipeline.

Uses multiprocessing to run many independent workers, each with its own
Stockfish instance (1 thread each). Optimized for high core count machines.

Each worker: generate small batch → label with SF → write JSONL.
Main process: monitor progress, merge batches, trigger HF uploads.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 -u generate_massive.py \
        --workers 80 --batch 2000 --total 2000000 --depth 8
"""

import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool, Value, Lock
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from move_vocab import UCI_TO_IDX

SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
OUTPUT_DIR = Path("outputs/massive_data")

# ─── Position generation (inlined for pickle-ability) ───

OPENING_BOOK = [
    "e2e4", "d2d4", "c2c4", "g1f3", "g2g3",  # 1st moves
    "e2e4 e7e5", "e2e4 c7c5", "e2e4 e7e6", "e2e4 c7c6", "e2e4 d7d5",
    "d2d4 d7d5", "d2d4 g8f6", "d2d4 e7e6", "d2d4 f7f5",
    "c2c4 e7e5", "c2c4 g8f6", "c2c4 c7c5",
    "g1f3 d7d5", "g1f3 g8f6", "g1f3 c7c5",
    "e2e4 e7e5 g1f3 b8c6", "e2e4 e7e5 g1f3 g8f6",
    "e2e4 c7c5 g1f3 d7d6", "e2e4 c7c5 g1f3 b8c6", "e2e4 c7c5 g1f3 e7e6",
    "d2d4 d7d5 c2c4", "d2d4 d7d5 g1f3", "d2d4 g8f6 c2c4",
    "d2d4 g8f6 c2c4 g7g6", "d2d4 g8f6 c2c4 e7e6",
    "e2e4 e7e5 g1f3 b8c6 f1b5",  # Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 d2d4",  # Scotch
    "e2e4 e7e5 g1f3 b8c6 f1c4",  # Italian
    "e2e4 c7c5 g1f3 d7d6 d2d4",  # Open Sicilian
    "d2d4 d7d5 c2c4 e7e6",       # QGD
    "d2d4 d7d5 c2c4 c7c6",       # Slav
    "d2d4 g8f6 c2c4 g7g6 b8c6",  # Grunfeld-ish
    "e2e4 e7e6 d2d4 d7d5",       # French
    "e2e4 c7c6 d2d4 d7d5",       # Caro-Kann
    "e2e4 d7d5 e4d5 d8d5",       # Scandinavian
    "d2d4 f7f5",                  # Dutch
    "e2e4 g7g6",                  # Modern
    "e2e4 d7d6",                  # Pirc
    "g1f3 d7d5 g2g3",             # Reti
    "c2c4 e7e5 b8c3",             # English
    "e2e4 e7e5 f2f4",             # King's Gambit
    "d2d4 d7d5 c2c4 d5c4",       # QGA
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",  # Morphy Defense
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",  # QID
    "d2d4 g8f6 c2c4 e7e6 g1f3 f8b4",  # Nimzo-Indian via transposition
]


def classify_phase(board):
    material = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.piece_type != chess.KING:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                    chess.ROOK: 5, chess.QUEEN: 9}
            material += vals.get(p.piece_type, 0)
    if material >= 50 and board.fullmove_number <= 12:
        return "opening"
    elif material <= 26:
        return "endgame"
    return "middlegame"


def gen_opening_book(rng, count):
    """Generate positions from opening book lines + random continuation."""
    positions = []
    for _ in range(count * 3):
        if len(positions) >= count:
            break
        line = rng.choice(OPENING_BOOK)
        board = chess.Board()
        try:
            for uci in line.split():
                board.push_uci(uci)
        except Exception:
            continue
        # Play 0-8 more random moves
        for _ in range(rng.randint(0, 8)):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        if not board.is_game_over() and list(board.legal_moves):
            positions.append((board.copy(), "opening_book"))
    return positions[:count]


def gen_weighted_play(rng, count):
    """Generate positions from games with realistic move selection."""
    positions = []
    for _ in range(count * 5):
        if len(positions) >= count:
            break
        board = chess.Board()
        game_len = rng.randint(10, 80)
        for ply in range(game_len):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            # Weight moves: captures, center control, development
            weights = []
            for m in moves:
                w = 1.0
                if board.is_capture(m):
                    w *= 3.0
                to_sq = m.to_square
                to_file = chess.square_file(to_sq)
                to_rank = chess.square_rank(to_sq)
                # Center bonus
                if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
                    w *= 1.5
                # Development in early game
                if ply < 20:
                    piece = board.piece_at(m.from_square)
                    if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                        w *= 1.5
                weights.append(w)
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            # Weighted random choice
            r = rng.random()
            cum = 0
            chosen = moves[-1]
            for m, p in zip(moves, probs):
                cum += p
                if r <= cum:
                    chosen = m
                    break
            board.push(chosen)
        # Sample a position from the game
        if board.move_stack and not board.is_game_over():
            # Rewind to a random position
            n_moves = len(board.move_stack)
            target = rng.randint(max(1, n_moves // 4), n_moves - 1)
            replay = chess.Board()
            for i, move in enumerate(board.move_stack[:target]):
                replay.push(move)
            if list(replay.legal_moves) and not replay.is_game_over():
                positions.append((replay.copy(), "weighted_play"))
    return positions[:count]


def gen_aggressive_play(rng, count):
    """Generate tactical positions (capture-heavy games)."""
    positions = []
    for _ in range(count * 5):
        if len(positions) >= count:
            break
        board = chess.Board()
        for _ in range(rng.randint(10, 60)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            captures = [m for m in moves if board.is_capture(m)]
            checks = [m for m in moves if board.gives_check(m)]
            tactical = list(set(captures + checks))
            if tactical and rng.random() < 0.7:
                board.push(rng.choice(tactical))
            else:
                board.push(rng.choice(moves))
        if not board.is_game_over() and list(board.legal_moves):
            positions.append((board.copy(), "aggressive_play"))
    return positions[:count]


def gen_endgame(rng, count):
    """Generate endgame positions (synthetic + tradedown)."""
    positions = []
    LIGHT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK]
    
    # Synthetic: kings + 1-4 pieces
    synth_target = int(count * 0.6)
    for _ in range(synth_target * 30):
        if len(positions) >= synth_target:
            break
        board = chess.Board.empty()
        squares = list(chess.SQUARES)
        rng.shuffle(squares)
        wk, bk_pool = squares[0], squares[1:]
        bk_cands = [s for s in bk_pool if chess.square_distance(s, wk) > 1]
        if not bk_cands:
            continue
        bk = rng.choice(bk_cands)
        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        occupied = {wk, bk}
        remaining = [s for s in squares if s not in occupied]
        for _ in range(rng.randint(1, 4)):
            if not remaining:
                break
            sq = remaining.pop()
            color = rng.choice([chess.WHITE, chess.BLACK])
            ptype = rng.choice(LIGHT)
            rank = chess.square_rank(sq)
            if ptype == chess.PAWN and rank in (0, 7):
                ptype = rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK])
            board.set_piece_at(sq, chess.Piece(ptype, color))
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.castling_rights = 0
        board.ep_square = None
        board.halfmove_clock = rng.randint(0, 30)
        board.fullmove_number = rng.randint(30, 60)
        if board.is_valid() and list(board.legal_moves):
            positions.append((board.copy(), "endgame_synth"))

    # Tradedown: play aggressive, keep endgame positions
    tradedown_target = count - len(positions)
    for _ in range(tradedown_target * 20):
        if len(positions) >= count:
            break
        board = chess.Board()
        for _ in range(rng.randint(30, 90)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            captures = [m for m in moves if board.is_capture(m)]
            if captures and rng.random() < 0.7:
                board.push(rng.choice(captures))
            else:
                board.push(rng.choice(moves))
        if classify_phase(board) == "endgame" and not board.is_game_over():
            if list(board.legal_moves):
                positions.append((board.copy(), "endgame_tradedown"))
    return positions[:count]


def gen_perturbed(rng, count, templates):
    """Mutate existing positions for material imbalance variety."""
    if not templates:
        return []
    positions = []
    for _ in range(count * 10):
        if len(positions) >= count:
            break
        board = rng.choice(templates)[0].copy()
        non_king = [sq for sq in chess.SQUARES
                    if board.piece_at(sq) and board.piece_at(sq).piece_type != chess.KING]
        if not non_king:
            continue
        if rng.random() < 0.67:  # remove
            board.remove_piece_at(rng.choice(non_king))
        else:  # swap
            sq = rng.choice(non_king)
            p = board.piece_at(sq)
            new_type = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
            rank = chess.square_rank(sq)
            if new_type == chess.PAWN and rank in (0, 7):
                new_type = chess.KNIGHT
            board.set_piece_at(sq, chess.Piece(new_type, p.color))
        if board.is_valid() and list(board.legal_moves) and not board.is_game_over():
            positions.append((board.copy(), "perturbed"))
    return positions[:count]


def generate_batch(seed, batch_size):
    """Generate a diverse batch of positions. No dedup across workers (by design)."""
    rng = random.Random(seed)
    # Allocation: 15% opening, 30% weighted, 15% aggressive, 20% endgame, 20% perturbed
    n_open = int(batch_size * 0.15)
    n_weight = int(batch_size * 0.30)
    n_aggr = int(batch_size * 0.15)
    n_end = int(batch_size * 0.20)

    positions = []
    positions.extend(gen_opening_book(rng, n_open))
    positions.extend(gen_weighted_play(rng, n_weight))
    positions.extend(gen_aggressive_play(rng, n_aggr))
    positions.extend(gen_endgame(rng, n_end))

    # Perturbed uses existing as templates
    n_pert = batch_size - len(positions)
    if n_pert > 0:
        positions.extend(gen_perturbed(rng, n_pert, positions))

    rng.shuffle(positions)
    return positions[:batch_size]


# ─── SF labeling ───

def label_position(board, source, sf):
    """Label a single position with SF. Returns dict or None."""
    try:
        fen = board.fen()
        sf.set_fen_position(fen)
        top_moves = sf.get_top_moves(5)
        if not top_moves:
            return None

        best = top_moves[0]
        best_move = best["Move"]
        if best_move not in UCI_TO_IDX:
            return None
        move_obj = chess.Move.from_uci(best_move)
        if move_obj not in board.legal_moves:
            return None

        eval_type = "mate" if best.get("Mate") is not None else "cp"
        eval_value = best["Mate"] if eval_type == "mate" else best.get("Centipawn", 0)

        if eval_type == "mate":
            wdl = [1.0, 0.0, 0.0] if eval_value > 0 else [0.0, 0.0, 1.0]
        else:
            k = 1.0 / 111.7
            win = 1.0 / (1.0 + math.exp(-k * eval_value))
            loss_p = 1.0 - win
            draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
            total = win + draw + loss_p
            wdl = [win / total, draw / total, loss_p / total]

        top_moves_data = []
        for m in top_moves:
            entry = {"uci": m["Move"]}
            if m.get("Mate") is not None:
                entry["mate"] = m["Mate"]
            else:
                entry["cp"] = m.get("Centipawn", 0)
            top_moves_data.append(entry)

        return {
            "fen": fen,
            "best_move": best_move,
            "eval_type": eval_type,
            "eval_value": eval_value,
            "wdl": wdl,
            "phase": classify_phase(board),
            "source": source,
            "top_moves": top_moves_data,
            "num_legal": len(list(board.legal_moves)),
            "ply": board.ply(),
        }
    except Exception:
        return None


def worker_fn(args):
    """Worker: generate batch → label with SF → write JSONL → return stats."""
    worker_id, seed, batch_size, depth, output_dir = args
    from stockfish import Stockfish

    t0 = time.time()

    # Generate positions
    positions = generate_batch(seed, batch_size)
    gen_time = time.time() - t0

    # Create SF instance (1 thread for maximum per-core throughput)
    sf = Stockfish(path=SF_PATH, depth=depth,
                   parameters={"Threads": 1, "Hash": 64})

    # Label
    results = []
    sf_crashes = 0
    for board, source in positions:
        result = label_position(board, source, sf)
        if result is not None:
            results.append(result)
        else:
            # Try to recover SF if it crashed
            sf_crashes += 1
            if sf_crashes % 10 == 0:
                try:
                    sf = Stockfish(path=SF_PATH, depth=depth,
                                   parameters={"Threads": 1, "Hash": 64})
                except Exception:
                    break

    label_time = time.time() - t0 - gen_time

    # Write results
    out_path = Path(output_dir) / f"worker_{worker_id:04d}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total_time = time.time() - t0
    rate = len(results) / max(label_time, 0.1)

    return {
        "worker_id": worker_id,
        "generated": len(positions),
        "labeled": len(results),
        "gen_time": round(gen_time, 1),
        "label_time": round(label_time, 1),
        "total_time": round(total_time, 1),
        "rate": round(rate, 1),
        "file": str(out_path),
    }


def merge_to_megabatch(output_dir, mega_id):
    """Merge all worker_*.jsonl files into a megabatch and clear originals."""
    worker_files = sorted(Path(output_dir).glob("worker_*.jsonl"))
    if not worker_files:
        return None, 0

    mega_path = Path(output_dir) / f"megabatch_{mega_id:03d}.jsonl"
    count = 0
    seen_fens = set()

    with open(mega_path, "w") as out:
        for wf in worker_files:
            with open(wf) as f:
                for line in f:
                    data = json.loads(line)
                    fen = data["fen"]
                    if fen not in seen_fens:
                        seen_fens.add(fen)
                        out.write(line)
                        count += 1

    # Remove worker files
    for wf in worker_files:
        wf.unlink()

    return mega_path, count


def upload_to_hf(output_dir, repo_id, token):
    """Upload all megabatch files to HF, deduplicating against existing."""
    from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

    mega_files = sorted(Path(output_dir).glob("megabatch_*.jsonl"))
    if not mega_files:
        print("  No megabatch files to upload")
        return

    # Load all local records
    records = []
    for mf in mega_files:
        with open(mf) as f:
            for line in f:
                records.append(json.loads(line))

    if not records:
        return

    local_fens = {r["fen"] for r in records}
    print(f"  Local: {len(records)} records ({len(local_fens)} unique FENs)")

    # Try to load existing dataset
    try:
        existing = load_dataset(repo_id, split="train", token=token)
        print(f"  Existing HF dataset: {len(existing)} rows")
        existing_fens = set(existing["fen"])
        new_records = [r for r in records if r["fen"] not in existing_fens]
        print(f"  New unique: {len(new_records)} (filtered {len(records) - len(new_records)} dupes)")
        if not new_records:
            print("  Nothing new to upload")
            return
        new_ds = Dataset.from_list(new_records)
        combined = concatenate_datasets([existing, new_ds])
    except Exception as e:
        print(f"  No existing dataset ({e}), creating fresh")
        combined = Dataset.from_list(records)

    # Train/test split
    split = combined.train_test_split(test_size=0.05, seed=42)
    ds_dict = DatasetDict({"train": split["train"], "test": split["test"]})

    print(f"  Uploading {len(combined)} total (train={len(split['train'])}, test={len(split['test'])})")
    ds_dict.push_to_hub(
        repo_id, token=token,
        commit_message=f"Add positions (total: {len(combined)})",
    )
    print(f"  Uploaded to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Massively parallel chess data generation")
    parser.add_argument("--workers", type=int, default=80,
                        help="Number of parallel workers (each gets 1 SF thread)")
    parser.add_argument("--batch", type=int, default=2000,
                        help="Positions per worker per round")
    parser.add_argument("--total", type=int, default=2000000,
                        help="Total positions to generate")
    parser.add_argument("--depth", type=int, default=8,
                        help="Stockfish search depth")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--upload-every", type=int, default=100000,
                        help="Upload to HF after this many positions")
    parser.add_argument("--repo", type=str, default="avewright/chess-positions-sf-labeled",
                        help="HF dataset repo ID")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HF uploads")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load HF token
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip()
                    break

    print(f"{'='*60}")
    print(f" MASSIVE PARALLEL DATA GENERATION")
    print(f"{'='*60}")
    print(f"  Workers:       {args.workers}")
    print(f"  Batch/worker:  {args.batch}")
    print(f"  Positions/round: {args.workers * args.batch:,}")
    print(f"  Total target:  {args.total:,}")
    print(f"  SF depth:      {args.depth}")
    print(f"  Upload every:  {args.upload_every:,}")
    print(f"  Repo:          {args.repo}")
    print(f"  Output:        {OUTPUT_DIR}/")
    print(f"{'='*60}")
    print()

    total_labeled = 0
    mega_id = 0
    round_num = 0
    since_upload = 0
    t_global = time.time()

    # Check for existing megabatch files to continue from
    existing_megas = sorted(OUTPUT_DIR.glob("megabatch_*.jsonl"))
    if existing_megas:
        for mf in existing_megas:
            with open(mf) as f:
                n = sum(1 for _ in f)
                total_labeled += n
        mega_id = len(existing_megas)
        print(f"  Resuming: found {len(existing_megas)} megabatches, {total_labeled:,} existing positions")
        print()

    while total_labeled < args.total:
        round_num += 1
        remaining = args.total - total_labeled
        # How many workers this round
        positions_per_round = args.workers * args.batch
        if remaining < positions_per_round:
            # Reduce workers for last round
            n_workers = max(1, remaining // args.batch)
        else:
            n_workers = args.workers

        print(f"Round {round_num}: {n_workers} workers × {args.batch} = {n_workers * args.batch:,} positions")

        # Prepare worker arguments (each gets unique seed)
        worker_args = []
        for w in range(n_workers):
            seed = args.seed + round_num * 10000 + w
            worker_args.append((
                w, seed, args.batch, args.depth, str(OUTPUT_DIR)
            ))

        # Launch parallel workers
        t_round = time.time()
        with Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, worker_args)

        round_time = time.time() - t_round
        round_labeled = sum(r["labeled"] for r in results)
        round_generated = sum(r["generated"] for r in results)
        round_rate = round_labeled / max(round_time, 0.1)

        # Phase stats
        phase_counts = {"opening": 0, "middlegame": 0, "endgame": 0}
        source_counts = {}
        for r_path in [Path(r["file"]) for r in results]:
            if r_path.exists():
                with open(r_path) as f:
                    for line in f:
                        d = json.loads(line)
                        phase_counts[d.get("phase", "unknown")] = phase_counts.get(d.get("phase", "unknown"), 0) + 1
                        src = d.get("source", "unknown")
                        source_counts[src] = source_counts.get(src, 0) + 1

        # Merge worker files into megabatch
        mega_path, mega_count = merge_to_megabatch(OUTPUT_DIR, mega_id)
        if mega_path:
            mega_id += 1

        total_labeled += round_labeled
        since_upload += round_labeled
        elapsed = time.time() - t_global
        overall_rate = total_labeled / max(elapsed, 0.1)

        print(f"  Generated: {round_generated:,} | Labeled: {round_labeled:,} ({round_labeled/max(round_generated,1)*100:.0f}%)")
        print(f"  Rate: {round_rate:.0f} pos/s | Time: {round_time:.0f}s")
        print(f"  Phases: O={phase_counts.get('opening',0)} M={phase_counts.get('middlegame',0)} E={phase_counts.get('endgame',0)}")
        src_str = " ".join(f"{k}={v}" for k, v in sorted(source_counts.items()))
        print(f"  Sources: {src_str}")
        print(f"  TOTAL: {total_labeled:,}/{args.total:,} ({total_labeled/args.total*100:.1f}%) | Overall: {overall_rate:.0f} pos/s | Elapsed: {elapsed/60:.1f}m")
        if mega_path:
            print(f"  Saved: {mega_path.name} ({mega_count:,} unique positions)")

        # Upload to HF periodically
        if not args.no_upload and hf_token and since_upload >= args.upload_every:
            print(f"\n  >>> Uploading to HuggingFace ({since_upload:,} new since last upload)...")
            try:
                upload_to_hf(str(OUTPUT_DIR), args.repo, hf_token)
                since_upload = 0
                print(f"  >>> Upload complete\n")
            except Exception as e:
                print(f"  >>> Upload failed: {e}\n")

        print()

    # Final upload
    if not args.no_upload and hf_token and since_upload > 0:
        print(f"Final upload ({since_upload:,} remaining)...")
        try:
            upload_to_hf(str(OUTPUT_DIR), args.repo, hf_token)
        except Exception as e:
            print(f"Upload failed: {e}")

    total_time = time.time() - t_global
    print(f"\n{'='*60}")
    print(f" COMPLETE: {total_labeled:,} positions in {total_time/60:.1f}m")
    print(f" Rate: {total_labeled/max(total_time,0.1):.0f} pos/s overall")
    print(f" Output: {OUTPUT_DIR}/ ({mega_id} megabatch files)")
    print(f"{'='*60}")

    # Write manifest
    manifest = {
        "total_labeled": total_labeled,
        "sf_depth": args.depth,
        "workers": args.workers,
        "batch_per_worker": args.batch,
        "seed": args.seed,
        "total_time_s": round(total_time),
        "overall_rate": round(total_labeled / max(total_time, 0.1)),
        "megabatches": mega_id,
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
