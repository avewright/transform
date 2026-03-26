#!/usr/bin/env python3
"""Generate custom chess positions with Stockfish and upload to HF in lichess-sf format.

This script:
1. Generates diverse positions using multiprocessing + Stockfish
2. Converts to parquet shards in the avewright/chess-positions-lichess-sf schema
3. Uploads shards incrementally to HF

Schema matches prepare_hf_dataset.py output:
  fen, best_move, eval_type, eval_value, wdl_win, wdl_draw, wdl_loss,
  phase, num_legal, source, game_id, top_moves, ply, depth

Usage:
    export HF_TOKEN=...
    python3 -u generate_and_upload.py --workers 48 --total 5000000 --depth 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from move_vocab import UCI_TO_IDX

SF_PATH = "/usr/games/stockfish"
OUTPUT_DIR = Path("outputs/custom_generated")
TARGET_REPO = "avewright/chess-positions-lichess-sf"

# ─── Opening book for position diversity ───

OPENING_BOOK = [
    "e2e4", "d2d4", "c2c4", "g1f3", "g2g3",
    "e2e4 e7e5", "e2e4 c7c5", "e2e4 e7e6", "e2e4 c7c6", "e2e4 d7d5",
    "d2d4 d7d5", "d2d4 g8f6", "d2d4 e7e6", "d2d4 f7f5",
    "c2c4 e7e5", "c2c4 g8f6", "c2c4 c7c5",
    "g1f3 d7d5", "g1f3 g8f6", "g1f3 c7c5",
    "e2e4 e7e5 g1f3 b8c6", "e2e4 e7e5 g1f3 g8f6",
    "e2e4 c7c5 g1f3 d7d6", "e2e4 c7c5 g1f3 b8c6",
    "d2d4 d7d5 c2c4", "d2d4 d7d5 g1f3", "d2d4 g8f6 c2c4",
    "d2d4 g8f6 c2c4 g7g6", "d2d4 g8f6 c2c4 e7e6",
    "e2e4 e7e5 g1f3 b8c6 f1b5",
    "e2e4 e7e5 g1f3 b8c6 d2d4",
    "e2e4 e7e5 g1f3 b8c6 f1c4",
    "e2e4 c7c5 g1f3 d7d6 d2d4",
    "d2d4 d7d5 c2c4 e7e6",
    "d2d4 d7d5 c2c4 c7c6",
    "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5",
    "e2e4 d7d5 e4d5 d8d5",
    "d2d4 f7f5",
    "e2e4 g7g6",
    "e2e4 d7d6",
    "g1f3 d7d5 g2g3",
    "c2c4 e7e5 b1c3",
    "e2e4 e7e5 f2f4",
    "d2d4 d7d5 c2c4 d5c4",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
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


# ─── Position generation strategies ───

def gen_opening_book(rng):
    line = rng.choice(OPENING_BOOK)
    board = chess.Board()
    for uci in line.split():
        try:
            board.push_uci(uci)
        except Exception:
            return board, "opening_book"
    # Play a few more random moves
    for _ in range(rng.randint(0, 6)):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            break
        board.push(rng.choice(legal))
    return board, "opening_book"


def gen_weighted_play(rng):
    board = chess.Board()
    line = rng.choice(OPENING_BOOK)
    for uci in line.split():
        try:
            board.push_uci(uci)
        except Exception:
            break
    # Weighted random play (prefer captures, center moves, development)
    for _ in range(rng.randint(5, 40)):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            break
        weights = []
        for m in legal:
            w = 1.0
            if board.is_capture(m):
                w += 3.0
            to_sq = m.to_square
            if 27 <= to_sq <= 36 or 19 <= to_sq <= 44:
                w += 1.0
            if board.piece_at(m.from_square) and board.piece_at(m.from_square).piece_type in (chess.KNIGHT, chess.BISHOP):
                if m.from_square in (chess.B1, chess.G1, chess.C1, chess.F1, chess.B8, chess.G8, chess.C8, chess.F8):
                    w += 2.0
            weights.append(w)
        total_w = sum(weights)
        r = rng.random() * total_w
        cumulative = 0
        chosen = legal[-1]
        for m, w in zip(legal, weights):
            cumulative += w
            if cumulative >= r:
                chosen = m
                break
        board.push(chosen)
    return board, "weighted_play"


def gen_aggressive_play(rng):
    board = chess.Board()
    for _ in range(rng.randint(3, 50)):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            break
        captures = [m for m in legal if board.is_capture(m)]
        checks = [m for m in legal if board.gives_check(m)]
        if captures and rng.random() < 0.6:
            board.push(rng.choice(captures))
        elif checks and rng.random() < 0.4:
            board.push(rng.choice(checks))
        else:
            board.push(rng.choice(legal))
    return board, "aggressive_play"


def gen_endgame_synth(rng):
    board = chess.Board(fen=None)
    board.clear()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    wk_sq = rng.choice(chess.SQUARES)
    while wk_sq == chess.E1:
        wk_sq = rng.choice(chess.SQUARES)
    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.BLACK))
    n_pieces = rng.randint(2, 8)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    for _ in range(n_pieces):
        sq = rng.choice(chess.SQUARES)
        if board.piece_at(sq):
            continue
        pt = rng.choice(piece_types)
        color = rng.choice([chess.WHITE, chess.BLACK])
        if pt == chess.PAWN and (chess.square_rank(sq) in (0, 7)):
            continue
        board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    if not board.is_valid() or board.is_game_over() or not list(board.legal_moves):
        return gen_weighted_play(rng)
    return board, "endgame_synth"


def gen_endgame_tradedown(rng):
    board, _ = gen_weighted_play(rng)
    for _ in range(rng.randint(10, 30)):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over():
            break
        captures = [m for m in legal if board.is_capture(m)]
        if captures and rng.random() < 0.7:
            board.push(rng.choice(captures))
        else:
            board.push(rng.choice(legal))
    return board, "endgame_tradedown"


def gen_perturbed(rng):
    board, _ = gen_weighted_play(rng)
    if rng.random() < 0.3:
        pieces = list(board.piece_map().items())
        if len(pieces) > 4:
            sq, _ = rng.choice(pieces)
            if board.piece_at(sq).piece_type != chess.KING:
                board.remove_piece_at(sq)
    if rng.random() < 0.2:
        empty_sqs = [sq for sq in chess.SQUARES if not board.piece_at(sq)]
        if empty_sqs:
            sq = rng.choice(empty_sqs)
            pt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK])
            color = rng.choice([chess.WHITE, chess.BLACK])
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                pt = chess.KNIGHT
            board.set_piece_at(sq, chess.Piece(pt, color))
    if not board.is_valid() or board.is_game_over() or not list(board.legal_moves):
        return gen_weighted_play(rng)
    return board, "perturbed"


GENERATORS = [
    (gen_opening_book, 0.15),
    (gen_weighted_play, 0.30),
    (gen_aggressive_play, 0.15),
    (gen_endgame_synth, 0.10),
    (gen_endgame_tradedown, 0.10),
    (gen_perturbed, 0.20),
]


def generate_batch(seed, batch_size):
    rng = random.Random(seed)
    positions = []
    gen_funcs, gen_weights = zip(*GENERATORS)
    for _ in range(batch_size):
        fn = rng.choices(gen_funcs, weights=gen_weights, k=1)[0]
        try:
            board, source = fn(rng)
            if board.is_valid() and not board.is_game_over() and list(board.legal_moves):
                positions.append((board, source))
        except Exception:
            continue
    return positions


# ─── SF labeling ───

def label_position(board, source, sf, depth):
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
            wdl_win = 1.0 if eval_value > 0 else 0.0
            wdl_loss = 0.0 if eval_value > 0 else 1.0
            wdl_draw = 0.0
        else:
            k = 1.0 / 111.7
            win = 1.0 / (1.0 + math.exp(-k * eval_value))
            loss_p = 1.0 - win
            draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
            total = win + draw + loss_p
            wdl_win = win / total
            wdl_draw = draw / total
            wdl_loss = loss_p / total

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
            "eval_value": int(eval_value) if eval_value is not None else 0,
            "wdl_win": round(wdl_win, 6),
            "wdl_draw": round(wdl_draw, 6),
            "wdl_loss": round(wdl_loss, 6),
            "phase": classify_phase(board),
            "num_legal": len(list(board.legal_moves)),
            "source": f"generated_{source}",
            "game_id": "",
            "top_moves": json.dumps(top_moves_data),
            "ply": board.ply(),
            "depth": depth,
        }
    except Exception:
        return None


def worker_fn(args):
    """Worker: generate batch → label with SF → return results."""
    worker_id, seed, batch_size, depth = args
    from stockfish import Stockfish

    t0 = time.time()
    positions = generate_batch(seed, batch_size)

    try:
        sf = Stockfish(path=SF_PATH, depth=depth,
                       parameters={"Threads": 1, "Hash": 64})
    except Exception as e:
        return {"worker_id": worker_id, "results": [], "error": str(e)}

    results = []
    sf_crashes = 0
    for board, source in positions:
        result = label_position(board, source, sf, depth)
        if result is not None:
            results.append(result)
        else:
            sf_crashes += 1
            if sf_crashes % 10 == 0:
                try:
                    sf = Stockfish(path=SF_PATH, depth=depth,
                                   parameters={"Threads": 1, "Hash": 64})
                except Exception:
                    break

    elapsed = time.time() - t0
    rate = len(results) / max(elapsed, 0.1)

    return {
        "worker_id": worker_id,
        "results": results,
        "generated": len(positions),
        "labeled": len(results),
        "rate": round(rate, 1),
        "elapsed": round(elapsed, 1),
    }


# ─── Parquet shard writing + upload ───

SCHEMA = pa.schema([
    ("fen", pa.string()),
    ("best_move", pa.string()),
    ("eval_type", pa.string()),
    ("eval_value", pa.int32()),
    ("wdl_win", pa.float32()),
    ("wdl_draw", pa.float32()),
    ("wdl_loss", pa.float32()),
    ("phase", pa.string()),
    ("num_legal", pa.int32()),
    ("source", pa.string()),
    ("game_id", pa.string()),
    ("top_moves", pa.string()),
    ("ply", pa.int32()),
    ("depth", pa.int32()),
])


def write_shard(records, shard_path):
    """Write a list of dicts as a parquet shard."""
    arrays = {col: [] for col in SCHEMA.names}
    for r in records:
        for col in SCHEMA.names:
            arrays[col].append(r.get(col))

    table = pa.table(arrays, schema=SCHEMA)
    pq.write_table(table, shard_path, compression="snappy")
    return len(records)


def upload_shards(shard_dir, repo_id, token, prefix="custom"):
    """Upload all parquet shards from shard_dir to HF repo."""
    from huggingface_hub import HfApi, CommitOperationAdd

    api = HfApi(token=token)
    shard_files = sorted(Path(shard_dir).glob("*.parquet"))
    if not shard_files:
        print("  No shards to upload")
        return

    operations = []
    for sf in shard_files:
        # Use a unique prefix to avoid collisions with lichess shards
        remote_name = f"data/train-{prefix}-{sf.name}"
        operations.append(CommitOperationAdd(
            path_in_repo=remote_name,
            path_or_fileobj=str(sf),
        ))

    print(f"  Uploading {len(operations)} shards to {repo_id}...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add {len(operations)} generated position shards ({prefix})",
    )
    print(f"  Uploaded {len(operations)} shards")


def load_hf_token():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
    return token


def main():
    parser = argparse.ArgumentParser(description="Generate chess positions and upload to HF")
    parser.add_argument("--workers", type=int, default=48, help="Parallel workers")
    parser.add_argument("--batch", type=int, default=1000, help="Positions per worker per round")
    parser.add_argument("--total", type=int, default=5_000_000, help="Total positions to generate")
    parser.add_argument("--depth", type=int, default=10, help="Stockfish search depth")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--upload-every", type=int, default=250_000, help="Upload to HF after N positions")
    parser.add_argument("--shard-size", type=int, default=250_000, help="Rows per parquet shard")
    parser.add_argument("--repo", type=str, default=TARGET_REPO, help="HF dataset repo")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploads")
    parser.add_argument("--prefix", type=str, default="gen", help="Shard name prefix")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_dir = OUTPUT_DIR / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    hf_token = load_hf_token()
    if not hf_token and not args.no_upload:
        print("WARNING: No HF_TOKEN found, uploads will be skipped")
        args.no_upload = True

    print(f"{'='*70}")
    print(f" CUSTOM POSITION GENERATION + HF UPLOAD")
    print(f"{'='*70}")
    print(f"  Workers: {args.workers}")
    print(f"  Batch size: {args.batch}")
    print(f"  Total target: {args.total:,}")
    print(f"  SF depth: {args.depth}")
    print(f"  Upload every: {args.upload_every:,}")
    print(f"  Shard size: {args.shard_size:,}")
    print(f"  Repo: {args.repo}")
    print(f"  SF path: {SF_PATH}")
    print()

    total_generated = 0
    total_labeled = 0
    pending_records = []
    shard_count = 0
    upload_count = 0
    round_num = 0
    since_last_upload = 0
    t_start = time.time()

    # Check for existing progress
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        total_labeled = manifest.get("total_labeled", 0)
        shard_count = manifest.get("shard_count", 0)
        upload_count = manifest.get("upload_count", 0)
        print(f"  Resuming: {total_labeled:,} previously labeled, {shard_count} shards, {upload_count} uploads")

    remaining = args.total - total_labeled
    if remaining <= 0:
        print(f"  Already generated {total_labeled:,} >= {args.total:,} target. Done.")
        return

    print(f"  Generating {remaining:,} more positions...")
    print()

    while total_labeled < args.total:
        round_num += 1
        # Build worker args for this round
        n_needed = min(args.total - total_labeled, args.workers * args.batch)
        n_workers = min(args.workers, math.ceil(n_needed / args.batch))
        worker_args = [
            (i, args.seed + round_num * 10000 + i, args.batch, args.depth)
            for i in range(n_workers)
        ]

        t_round = time.time()
        with Pool(n_workers) as pool:
            results = pool.map(worker_fn, worker_args)

        round_generated = 0
        round_labeled = 0
        for r in results:
            if "error" in r and r.get("results") is not None and len(r["results"]) == 0:
                print(f"  Worker {r['worker_id']} error: {r['error']}")
                continue
            records = r.get("results", [])
            pending_records.extend(records)
            round_generated += r.get("generated", 0)
            round_labeled += len(records)

        total_generated += round_generated
        total_labeled += round_labeled
        since_last_upload += round_labeled

        elapsed = time.time() - t_start
        overall_rate = total_labeled / max(elapsed, 0.1)
        round_time = time.time() - t_round
        round_rate = round_labeled / max(round_time, 0.1)

        print(f"  Round {round_num}: +{round_labeled:,} labeled ({round_rate:.0f} pos/s) "
              f"| Total: {total_labeled:,}/{args.total:,} ({overall_rate:.0f} pos/s overall) "
              f"| Pending: {len(pending_records):,}",
              flush=True)

        # Write shards when buffer is large enough
        while len(pending_records) >= args.shard_size:
            shard_records = pending_records[:args.shard_size]
            pending_records = pending_records[args.shard_size:]
            shard_path = shard_dir / f"shard_{shard_count:05d}.parquet"
            write_shard(shard_records, shard_path)
            shard_count += 1
            print(f"    Wrote shard {shard_count}: {len(shard_records):,} rows ({shard_path.name})")

        # Upload periodically
        if since_last_upload >= args.upload_every and not args.no_upload:
            # Write any remaining pending as a partial shard
            if pending_records:
                shard_path = shard_dir / f"shard_{shard_count:05d}.parquet"
                write_shard(pending_records, shard_path)
                shard_count += 1
                print(f"    Wrote partial shard {shard_count}: {len(pending_records):,} rows")
                pending_records = []

            upload_shards(shard_dir, args.repo, hf_token, prefix=args.prefix)
            upload_count += 1
            since_last_upload = 0

            # Clean uploaded shards
            for sf in shard_dir.glob("*.parquet"):
                sf.unlink()

            # Save manifest
            manifest = {
                "total_generated": total_generated,
                "total_labeled": total_labeled,
                "shard_count": shard_count,
                "upload_count": upload_count,
                "depth": args.depth,
                "workers": args.workers,
                "seed": args.seed,
                "elapsed_s": round(elapsed),
                "overall_rate": round(overall_rate, 1),
                "last_upload_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"    Upload #{upload_count} complete. Manifest saved.")

    # Final flush
    if pending_records:
        shard_path = shard_dir / f"shard_{shard_count:05d}.parquet"
        write_shard(pending_records, shard_path)
        shard_count += 1
        pending_records = []

    if not args.no_upload:
        upload_shards(shard_dir, args.repo, hf_token, prefix=args.prefix)
        upload_count += 1
        for sf in shard_dir.glob("*.parquet"):
            sf.unlink()

    total_time = time.time() - t_start
    final_rate = total_labeled / max(total_time, 0.1)

    print(f"\n{'='*70}")
    print(f" GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total labeled: {total_labeled:,}")
    print(f"  Total shards: {shard_count}")
    print(f"  Total uploads: {upload_count}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Overall rate: {final_rate:.0f} pos/s")
    print(f"  Repo: {args.repo}")

    manifest = {
        "total_generated": total_generated,
        "total_labeled": total_labeled,
        "shard_count": shard_count,
        "upload_count": upload_count,
        "depth": args.depth,
        "workers": args.workers,
        "seed": args.seed,
        "total_time_s": round(total_time),
        "overall_rate": round(final_rate, 1),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
