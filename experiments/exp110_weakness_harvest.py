"""exp110_weakness_harvest: Generate training data from model's own mistakes.

Strategy: Play the model against Stockfish, identify positions where the model
blunders (large eval drop), and generate multi-PV soft targets for those
high-value positions. This is hard example mining — every position is one
the model actually struggles with.

Much more efficient than random HF sampling because positions are targeted
at the model's actual weaknesses.
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

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import UCI_TO_IDX, IDX_TO_UCI, VOCAB_SIZE, legal_move_mask, index_to_move
from play import ChessTransformer200M, load_model

OUTPUT_DIR = Path("outputs/exp110_weakness_harvest")
DATASET_DIR = OUTPUT_DIR / "dataset"
LOG_PATH = OUTPUT_DIR / "exp110_weakness.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STOCKFISH_PATH = None  # auto-detect
SHARD_SIZE = 5000
LABEL_TAU = 120.0

# Positions where model drops >75cp from Stockfish's eval are "blunders"
BLUNDER_THRESHOLD_CP = 75
# Also collect positions with moderate error (25-75cp)
INACCURACY_THRESHOLD_CP = 25

STOP_REQUESTED = False

OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e4", "c7c6"],
    ["d2d4", "d7d5", "c2c4"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["e2e4", "c7c5", "g1f3"],
    ["d2d4", "g8f6", "c2c4", "g7g6"],
]


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


@torch.no_grad()
def get_model_move(model, board, device):
    """Get the model's top move and full policy distribution."""
    board_input = batch_boards_to_fused_token_ids([board], device)
    result = model(board_input)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[mask == False] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    move_idx = logits.argmax().item()
    move = index_to_move(move_idx)

    # Get WDL
    wdl_logits = result["value_logits"][0].float()
    wdl_probs = F.softmax(wdl_logits, dim=-1).tolist()

    return move, probs, wdl_probs


def classify_phase(board):
    """Simple game phase classification."""
    piece_count = sum(1 for _ in board.piece_map()) - 2  # exclude kings
    if piece_count <= 6:
        return "endgame"
    ply = len(board.move_stack)
    if ply <= 20:
        return "opening"
    return "middlegame"


def analyze_position(engine, board, depth=8, multipv=5):
    """Multi-PV analysis of a position."""
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        results = []
        for info in infos:
            score = info.get("score")
            pv = info.get("pv", [])
            if score and pv:
                cp = score.relative.score(mate_score=10000)
                if cp is not None:
                    results.append({
                        "uci": pv[0].uci(),
                        "cp": cp,
                        "pv": [m.uci() for m in pv[:5]],
                    })
        return results
    except Exception:
        return []


def build_soft_targets(analysis_results, tau=LABEL_TAU):
    """Convert multi-PV analysis to soft targets."""
    if not analysis_results:
        return []

    cps = torch.tensor([r["cp"] for r in analysis_results], dtype=torch.float32)
    probs = F.softmax(cps / tau, dim=0).tolist()

    targets = []
    for r, p in zip(analysis_results, probs):
        targets.append({
            "uci": r["uci"],
            "prob": float(p),
            "cp": r["cp"],
            "eval_type": "stockfish",
            "rank": len(targets) + 1,
            "pv": r["pv"],
        })
    return targets


def play_and_harvest(model, engine, sf_elo, opening, model_color, depth=8, multipv=5):
    """Play one game, return list of error positions with analysis."""
    board = chess.Board()
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    positions = []
    prev_eval = None

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 200:
        if STOP_REQUESTED:
            break

        is_model_turn = board.turn == model_color

        if is_model_turn:
            # Get Stockfish eval of current position BEFORE model moves
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                sf_eval = info["score"].relative.score(mate_score=10000)
            except Exception:
                sf_eval = None

            # Get model move
            model_move, model_probs, model_wdl = get_model_move(model, board, DEVICE)

            if model_move not in board.legal_moves:
                model_move = next(iter(board.legal_moves))

            # Push model move and check what Stockfish thinks of the resulting position
            board.push(model_move)

            try:
                info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
                eval_after = info_after["score"].relative.score(mate_score=10000)
                # Flip perspective (opponent's eval)
                if eval_after is not None:
                    eval_after = -eval_after
            except Exception:
                eval_after = None

            # Compute error: how much eval dropped after model's move
            if sf_eval is not None and eval_after is not None:
                eval_drop = sf_eval - eval_after
                board.pop()  # go back to analyze the position

                if eval_drop >= INACCURACY_THRESHOLD_CP:
                    # This is an error position! Analyze with multi-PV
                    analysis = analyze_position(engine, board, depth=depth, multipv=multipv)
                    if analysis and len(analysis) >= 2:
                        soft_targets = build_soft_targets(analysis)
                        best = analysis[0]

                        # Value target
                        if best["cp"] > 100:
                            value_target = 2
                        elif best["cp"] < -100:
                            value_target = 0
                        else:
                            value_target = 1

                        cp_gap = analysis[0]["cp"] - analysis[1]["cp"] if len(analysis) > 1 else 0
                        phase = classify_phase(board)
                        pos_key = f"{chess.polyglot.zobrist_hash(board):016x}"

                        error_type = "blunder" if eval_drop >= BLUNDER_THRESHOLD_CP else "inaccuracy"

                        positions.append({
                            "source": "exp110_weakness",
                            "fen": board.fen(),
                            "position_key": pos_key,
                            "phase": phase,
                            "ply": len(board.move_stack),
                            "best_move": best["uci"],
                            "best_cp": best["cp"],
                            "model_move": model_move.uci(),
                            "model_eval": eval_after,
                            "eval_drop": eval_drop,
                            "error_type": error_type,
                            "value_target": value_target,
                            "label_depth": depth,
                            "label_multipv": len(soft_targets),
                            "label_tau": LABEL_TAU,
                            "soft_targets": soft_targets,
                            "num_legal": board.legal_moves.count(),
                            "num_labeled": len(soft_targets),
                            "unlabeled_legal": board.legal_moves.count() - len(soft_targets),
                            "cp_gap_top1_top2": cp_gap,
                            "sf_elo": sf_elo,
                        })

                    board.push(model_move)  # re-push to continue game
                else:
                    board.push(model_move)  # re-push to continue game
            # If eval was None, just continue
        else:
            # Stockfish's turn
            try:
                result = engine.play(board, chess.engine.Limit(time=0.05))
                board.push(result.move)
            except Exception:
                break

    outcome = board.outcome(claim_draw=True)
    result_str = "draw"
    if outcome and outcome.winner is not None:
        result_str = "win" if outcome.winner == model_color else "loss"

    return positions, result_str


def main():
    global STOP_REQUESTED

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/hf_checkpoint/best_model.pt")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--sf-elos", type=int, nargs="+", default=[1600, 1750, 1900])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    def sig_handler(sig, frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
        log("Shutdown requested...")

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    rng = random.Random(args.seed)

    log(f"exp110_weakness_harvest: Mining model mistakes for training data")
    log(f"  Checkpoint: {args.checkpoint}")
    log(f"  Games: {args.games}, Depth: {args.depth}, SF Elos: {args.sf_elos}")

    # Load model
    log("Loading model...")
    model = load_model(args.checkpoint, DEVICE)
    model.eval()
    log("Model loaded")

    # Start Stockfish
    sf_path = detect_stockfish()
    log(f"Stockfish: {sf_path}")

    t0 = time.time()
    all_positions = []
    game_results = Counter()
    error_types = Counter()
    phase_counts = Counter()

    shard_idx = 1
    shard_file = open(DATASET_DIR / f"positions_{shard_idx:06d}.jsonl", "w")
    records_in_shard = 0
    total_written = 0

    for game_num in range(args.games):
        if STOP_REQUESTED:
            break

        sf_elo = rng.choice(args.sf_elos)
        opening = rng.choice(OPENINGS)
        model_color = rng.choice([chess.WHITE, chess.BLACK])

        try:
            engine = chess.engine.SimpleEngine.popen_uci(sf_path)
            engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": sf_elo,
                "Threads": 1,
                "Hash": 16,
            })

            positions, result = play_and_harvest(
                model, engine, sf_elo, opening, model_color,
                depth=args.depth, multipv=args.multipv,
            )
            engine.quit()

            game_results[result] += 1

            for pos in positions:
                shard_file.write(json.dumps(pos) + "\n")
                total_written += 1
                records_in_shard += 1
                error_types[pos["error_type"]] += 1
                phase_counts[pos["phase"]] += 1

                if records_in_shard >= SHARD_SIZE:
                    shard_file.close()
                    shard_idx += 1
                    shard_file = open(DATASET_DIR / f"positions_{shard_idx:06d}.jsonl", "w")
                    records_in_shard = 0

            if (game_num + 1) % 10 == 0:
                elapsed = (time.time() - t0) / 60
                rate = total_written / elapsed if elapsed > 0 else 0
                log(f"Game {game_num+1}/{args.games}: positions={total_written} ({rate:.1f}/min) "
                    f"results={dict(game_results)} errors={dict(error_types)}")

        except Exception as e:
            log(f"  Game {game_num+1} error: {e}")
            continue

    shard_file.close()
    elapsed = (time.time() - t0) / 60

    log(f"\n=== WEAKNESS HARVEST COMPLETE ===")
    log(f"  Games: {game_num + 1}, Positions: {total_written}")
    log(f"  Time: {elapsed:.1f}m ({total_written/elapsed:.1f}/min)")
    log(f"  Results: {dict(game_results)}")
    log(f"  Error types: {dict(error_types)}")
    log(f"  Phases: {dict(phase_counts)}")
    log(f"  Output: {DATASET_DIR}")

    status = {
        "completed": True,
        "games": game_num + 1,
        "written": total_written,
        "results": dict(game_results),
        "error_types": dict(error_types),
        "phases": dict(phase_counts),
        "elapsed_min": round(elapsed, 1),
    }
    (OUTPUT_DIR / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
