"""exp110_puzzle_harvest: Convert Lichess puzzles to training data.

Each puzzle has a known correct solution. We convert this into our
training format with the puzzle's best move as the hard label and
optional Stockfish multi-PV as soft targets.

Key: Puzzles are positions where there's ONE clearly best move (or sequence).
This provides very clean supervision — the opposite of the noisy cp_gap<50
positions that make up 68% of our current data.

Runs on CPU only — no GPU needed.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.polyglot
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from move_vocab import UCI_TO_IDX

OUTPUT_DIR = Path("outputs/exp110_puzzle_harvest")
DATASET_DIR = OUTPUT_DIR / "dataset"
LOG_PATH = OUTPUT_DIR / "puzzle_harvest.log"

SHARD_SIZE = 5000
LABEL_TAU = 120.0

STOP_REQUESTED = False


def log(msg: str):
    stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(stamped + "\n")
    except Exception:
        pass


def detect_stockfish():
    import shutil
    candidates = [
        os.environ.get("STOCKFISH_PATH", ""),
        shutil.which("stockfish") or "",
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        str(Path(__file__).resolve().parent.parent / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    raise FileNotFoundError("Stockfish not found")


def phase_name(board: chess.Board) -> str:
    pieces = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and board.piece_type_at(sq) != chess.KING)
    if pieces >= 20:
        return "opening"
    if pieces >= 10:
        return "middlegame"
    return "endgame"


def cp_to_wdl_class(cp: int) -> int:
    if cp > 100:
        return 2  # win
    if cp < -100:
        return 0  # loss
    return 1  # draw


def softmax_probs(cps: list[int], tau: float) -> list[float]:
    t = torch.tensor(cps, dtype=torch.float32)
    return F.softmax(t / tau, dim=0).tolist()


def analyze_position(engine, board, depth=8, multipv=5):
    """Multi-PV analysis of a position."""
    try:
        n_legal = board.legal_moves.count()
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=min(multipv, n_legal))
        if isinstance(infos, dict):
            infos = [infos]
        
        results = []
        seen = set()
        for info in infos:
            pv = info.get("pv", [])
            score = info.get("score")
            if not pv or not score:
                continue
            uci = pv[0].uci()
            if uci in seen:
                continue
            seen.add(uci)
            cp = score.relative.score(mate_score=10000)
            if cp is not None:
                results.append({
                    "uci": uci,
                    "cp": cp,
                    "pv": [m.uci() for m in pv[:5]],
                })
        return results
    except Exception:
        return []


def process_puzzle(puzzle: dict, engine=None, depth=8, multipv=5) -> dict | None:
    """Convert a Lichess puzzle to our training format."""
    fen = puzzle.get("FEN")
    moves_str = puzzle.get("Moves", "")
    rating = puzzle.get("Rating", 0)
    themes = puzzle.get("Themes", "")
    
    if not fen or not moves_str:
        return None
    
    moves = moves_str.split()
    if len(moves) < 2:
        return None  # Need at least opponent's move + our response
    
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    
    # The puzzle format: first move is the opponent's last move,
    # second move is our correct response
    # We want to create training data for OUR move (the puzzle solution)
    
    # Apply opponent's last move
    try:
        opp_move = chess.Move.from_uci(moves[0])
        if opp_move not in board.legal_moves:
            return None
        board.push(opp_move)
    except Exception:
        return None
    
    # Now it's our turn — moves[1] is the correct response
    best_uci = moves[1]
    try:
        best_move = chess.Move.from_uci(best_uci)
        if best_move not in board.legal_moves:
            return None
    except Exception:
        return None
    
    # Check if move is in vocab
    if best_uci not in UCI_TO_IDX:
        return None
    
    # Get Stockfish analysis if engine available
    soft_targets = []
    best_cp = 0
    if engine:
        analysis = analyze_position(engine, board, depth=depth, multipv=multipv)
        if analysis:
            cps = [r["cp"] for r in analysis]
            probs = softmax_probs(cps, LABEL_TAU)
            for r, p in zip(analysis, probs):
                soft_targets.append({
                    "uci": r["uci"],
                    "prob": float(p),
                    "cp": r["cp"],
                    "eval_type": "stockfish",
                    "rank": len(soft_targets) + 1,
                    "pv": r["pv"],
                })
            best_cp = analysis[0]["cp"]
    
    if not soft_targets:
        # Fallback: use the puzzle's known best move as a hard target
        soft_targets = [{
            "uci": best_uci,
            "prob": 1.0,
            "cp": 500,  # assume winning since it's a puzzle solution
            "eval_type": "puzzle",
            "rank": 1,
            "pv": [best_uci],
        }]
        best_cp = 500
    
    cp_gap = soft_targets[0]["cp"] - soft_targets[1]["cp"] if len(soft_targets) > 1 else 500
    
    pos_key = f"{chess.polyglot.zobrist_hash(board):016x}"
    
    return {
        "source": "exp110_puzzle",
        "fen": board.fen(),
        "position_key": pos_key,
        "phase": phase_name(board),
        "ply": board.ply(),
        "best_move": best_uci,
        "best_cp": best_cp,
        "value_target": cp_to_wdl_class(best_cp),
        "label_depth": depth if engine else 0,
        "label_multipv": len(soft_targets),
        "label_tau": LABEL_TAU,
        "soft_targets": soft_targets,
        "num_legal": board.legal_moves.count(),
        "num_labeled": len(soft_targets),
        "unlabeled_legal": board.legal_moves.count() - len(soft_targets),
        "cp_gap_top1_top2": cp_gap,
        "puzzle_rating": rating,
        "puzzle_themes": themes if isinstance(themes, str) else ",".join(themes) if themes else "",
    }


def main():
    global STOP_REQUESTED
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-puzzles", type=int, default=50000)
    parser.add_argument("--min-rating", type=int, default=1200)
    parser.add_argument("--max-rating", type=int, default=2500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4, help="Stockfish workers for multi-PV analysis")
    parser.add_argument("--no-stockfish", action="store_true", help="Skip Stockfish analysis, use puzzle answers only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    def sig_handler(sig, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        log("Shutdown requested...")
    
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    random.seed(args.seed)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    log("exp110_puzzle_harvest: Converting Lichess puzzles to training data")
    log(f"  Target: {args.max_puzzles}, Rating: {args.min_rating}-{args.max_rating}")
    log(f"  Stockfish: {'disabled' if args.no_stockfish else f'depth {args.depth}, multipv {args.multipv}'}")
    
    # Stream puzzles from HuggingFace
    from datasets import load_dataset
    log("Loading Lichess/chess-puzzles from HuggingFace...")
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    
    # Start Stockfish engine if needed
    engine = None
    if not args.no_stockfish:
        try:
            sf_path = detect_stockfish()
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            engine.configure({"Threads": 1, "Hash": 32})
            log(f"Stockfish engine started: {sf_path}")
        except Exception as e:
            log(f"WARNING: Could not start Stockfish ({e}), using puzzle answers only")
    
    t0 = time.time()
    written = 0
    skipped = 0
    phase_counts = Counter()
    theme_counts = Counter()
    shard_idx = 1
    shard_file = open(DATASET_DIR / f"positions_{shard_idx:06d}.jsonl", "w")
    records_in_shard = 0
    
    for puzzle in ds:
        if STOP_REQUESTED or written >= args.max_puzzles:
            break
        
        rating = puzzle.get("Rating", 0)
        if rating < args.min_rating or rating > args.max_rating:
            skipped += 1
            continue
        
        record = process_puzzle(puzzle, engine=engine, depth=args.depth, multipv=args.multipv)
        if record is None:
            skipped += 1
            continue
        
        shard_file.write(json.dumps(record) + "\n")
        written += 1
        records_in_shard += 1
        phase_counts[record["phase"]] += 1
        
        themes = record.get("puzzle_themes", "")
        if themes:
            for t in themes.split(","):
                t = t.strip()
                if t:
                    theme_counts[t] += 1
        
        if records_in_shard >= SHARD_SIZE:
            shard_file.close()
            shard_idx += 1
            shard_file = open(DATASET_DIR / f"positions_{shard_idx:06d}.jsonl", "w")
            records_in_shard = 0
        
        if written % 1000 == 0:
            elapsed = (time.time() - t0) / 60
            rate = written / elapsed if elapsed > 0 else 0
            log(f"  Written: {written}/{args.max_puzzles} ({rate:.0f}/min) "
                f"phases: {dict(phase_counts)} "
                f"skipped: {skipped}")
    
    shard_file.close()
    
    if engine:
        engine.quit()
    
    elapsed = (time.time() - t0) / 60
    
    log(f"\n=== PUZZLE HARVEST COMPLETE ===")
    log(f"  Written: {written}, Skipped: {skipped}")
    log(f"  Time: {elapsed:.1f}m ({written/elapsed:.0f}/min)" if elapsed > 0 else "  Time: 0m")
    log(f"  Phases: {dict(phase_counts)}")
    log(f"  Top themes: {dict(list(sorted(theme_counts.items(), key=lambda x: -x[1]))[:10])}")
    log(f"  Output: {DATASET_DIR}")


if __name__ == "__main__":
    main()
