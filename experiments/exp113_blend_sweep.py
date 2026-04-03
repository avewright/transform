"""exp113: Sweep blend strategies to maximize ELO.

From exp112: blend_k10 (w=0.3) scored ~1690 ELO vs greedy ~1600.
This experiment sweeps:
  - value_weight: 0.1, 0.2, 0.3, 0.5
  - top_k: 5, 10
  - anti-repetition penalty
  - batched child evaluation (much faster)

Uses 32 games per config at SF1750 (the bracket boundary) for faster signal.
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

import chess
import chess.engine
import chess.syzygy
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
SF = ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2"
SYZYGY_DIR = ROOT / "syzygy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from play import ChessTransformer200M
from chess_features import batch_boards_to_fused_token_ids
from move_vocab import UCI_TO_IDX, IDX_TO_UCI, VOCAB_SIZE

SYZYGY_TB = None


def init_syzygy():
    global SYZYGY_TB
    if SYZYGY_DIR.exists() and any(SYZYGY_DIR.glob("*.rtbw")):
        try:
            SYZYGY_TB = chess.syzygy.open_tablebase(str(SYZYGY_DIR))
            print(f"Syzygy tablebases loaded from {SYZYGY_DIR}")
        except Exception:
            SYZYGY_TB = None


def get_syzygy_move(board: chess.Board) -> chess.Move | None:
    if SYZYGY_TB is None or len(board.piece_map()) > 5:
        return None
    try:
        best_move = None
        best_wdl = -3
        best_dtz = 0
        for move in board.legal_moves:
            board.push(move)
            try:
                wdl = -SYZYGY_TB.probe_wdl(board)
                dtz = -SYZYGY_TB.probe_dtz(board)
                if wdl > best_wdl or (wdl == best_wdl and (
                    (wdl > 0 and dtz < best_dtz) or
                    (wdl < 0 and dtz > best_dtz) or
                    (wdl == 0 and abs(dtz) < abs(best_dtz))
                )):
                    best_move = move
                    best_wdl = wdl
                    best_dtz = dtz
            except Exception:
                pass
            board.pop()
        return best_move
    except Exception:
        return None


LOG_FILE = None


def log(msg):
    stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(stamped + "\n")


def load_model(checkpoint_path: Path) -> ChessTransformer200M:
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    return model


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in UCI_TO_IDX:
            mask[UCI_TO_IDX[uci]] = True
    return mask


def index_to_move(idx: int) -> chess.Move:
    return chess.Move.from_uci(IDX_TO_UCI[idx])


# ── Batched value evaluation (much faster than 1-by-1) ──
@torch.no_grad()
def batch_board_values(model, boards, device) -> list[float]:
    """Get value for multiple boards at once. Returns White-absolute P(W wins) - P(W loses)."""
    if not boards:
        return []
    inp = batch_boards_to_fused_token_ids(boards, device)
    result = model(inp)
    wdl = F.softmax(result["value_logits"].float(), dim=-1)  # (N, 3)
    # White-absolute: idx0=P(W wins), idx2=P(W loses)
    white_values = (wdl[:, 0] - wdl[:, 2]).tolist()
    return white_values


def stm_value(white_value: float, turn: chess.Color) -> float:
    """Convert White-absolute value to side-to-move perspective."""
    return white_value if turn == chess.WHITE else -white_value


# ── Strategy: Greedy policy ──
@torch.no_grad()
def strategy_greedy(model, board, device, **kwargs):
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    move_idx = logits.argmax().item()
    return index_to_move(move_idx)


# ── Strategy: Batched blend with anti-repetition ──
@torch.no_grad()
def strategy_blend_batched(model, board, device, top_k=10, value_weight=0.3,
                           repetition_penalty=0.15, **kwargs):
    """Blend policy + batched 1-ply value. Penalize moves leading to repeated positions."""
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    k = min(top_k, int(mask.sum().item()))
    if k == 0:
        return next(iter(board.legal_moves))
    topk = torch.topk(probs, k)

    # Collect child boards
    candidate_moves = []
    candidate_probs = []
    candidate_boards = []
    game_over_results = {}  # idx -> val

    parent_turn = board.turn

    for i, (idx, policy_prob) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue
        board.push(move)
        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
                # We just moved and opponent is in checkmate = we win
                game_over_results[len(candidate_moves)] = 1.0
            else:
                game_over_results[len(candidate_moves)] = 0.0
        else:
            candidate_boards.append(board.copy())
        board.pop()
        candidate_moves.append(move)
        candidate_probs.append(policy_prob)

    if not candidate_moves:
        return next(iter(board.legal_moves))

    # Batch evaluate non-terminal positions
    if candidate_boards:
        white_values = batch_board_values(model, candidate_boards, device)
    else:
        white_values = []

    # Score each candidate
    best_score = float("-inf")
    best_move = None
    board_idx = 0

    for i, (move, policy_prob) in enumerate(zip(candidate_moves, candidate_probs)):
        if i in game_over_results:
            val_stm = game_over_results[i]
        else:
            white_val = white_values[board_idx]
            # Convert to STM perspective then negate (child is opponent's turn)
            val_stm = -stm_value(white_val, chess.WHITE if parent_turn == chess.BLACK else chess.BLACK)
            board_idx += 1

        # Repetition penalty
        rep_pen = 0.0
        if repetition_penalty > 0:
            board.push(move)
            if board.is_repetition(2):
                rep_pen = -repetition_penalty
            board.pop()

        # Blend: (1-w)*policy + w*value_norm + rep_penalty
        val_norm = (val_stm + 1.0) / 2.0  # [-1,1] -> [0,1]
        score = (1.0 - value_weight) * policy_prob + value_weight * val_norm + rep_pen

        if score > best_score:
            best_score = score
            best_move = move

    return best_move or next(iter(board.legal_moves))


# ── Game playing ──
DEFAULT_OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
]


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return 0.0, 1.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def play_one(engine, model, strategy_fn, sf_elo, model_color, opening, movetime, ply_cap, use_syzygy=True):
    board = chess.Board()
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            tb_move = get_syzygy_move(board) if use_syzygy else None
            if tb_move is not None:
                move = tb_move
            else:
                move = strategy_fn(model, board, DEVICE)
        else:
            move = engine.play(board, chess.engine.Limit(time=movetime)).move
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    elif outcome.winner == model_color:
        score = 1.0
    else:
        score = 0.0

    termination = "PLY_CAP"
    if outcome:
        termination = outcome.termination.name

    return {
        "sf_elo": sf_elo,
        "color": "white" if model_color == chess.WHITE else "black",
        "opening": " ".join(opening) if opening else "startpos",
        "result": board.result(claim_draw=True),
        "score": score,
        "plies": len(board.move_stack),
        "termination": termination,
    }


def run_strategy_eval(model, strategy_fn, strategy_name, sf_elos, openings,
                      games_per_opening_per_color=2, movetime=0.05, ply_cap=200,
                      use_syzygy=True):
    log(f"\n{'='*60}")
    log(f"Strategy: {strategy_name}")
    log(f"{'='*60}")

    summaries = []
    all_games = []

    for sf_elo in sf_elos:
        log(f"begin sf_elo={sf_elo}")
        engine = chess.engine.SimpleEngine.popen_uci(str(SF))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1, "Hash": 32})
        results = []

        try:
            for opening in openings:
                for color in [chess.WHITE, chess.BLACK]:
                    for repeat in range(games_per_opening_per_color):
                        result = play_one(engine, model, strategy_fn, sf_elo, color,
                                         opening, movetime, ply_cap, use_syzygy)
                        result["repeat_idx"] = repeat
                        results.append(result)
                        log(f"game {json.dumps(result)}")
        finally:
            engine.quit()

        games = len(results)
        total_score = sum(r["score"] for r in results)
        score = total_score / games if games else 0.0
        ci_low, ci_high = wilson_interval(total_score, games)

        w = sum(1 for r in results if r["score"] == 1.0)
        d = sum(1 for r in results if r["score"] == 0.5)
        l = sum(1 for r in results if r["score"] == 0.0)

        # Count termination types
        terms = Counter(r["termination"] for r in results)

        summary = {
            "sf_elo": sf_elo, "games": games, "score": score,
            "score_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "w": w, "d": d, "l": l,
            "avg_plies": round(sum(r["plies"] for r in results) / games, 1) if games else 0.0,
            "terminations": dict(terms),
        }
        summaries.append(summary)
        all_games.extend(results)
        log(f"summary {json.dumps(summary)}")

    # Estimate ELO
    ordered = sorted(summaries, key=lambda s: s["sf_elo"])
    above = [s for s in ordered if s["score"] >= 0.5]
    below = [s for s in ordered if s["score"] < 0.5]
    lb = max((s["sf_elo"] for s in above), default=None)
    ub = min((s["sf_elo"] for s in below), default=None)

    if lb is None:
        est = ordered[0]["sf_elo"] - 100
    elif ub is None:
        est = ordered[-1]["sf_elo"] + 100
    else:
        ls = next(s for s in ordered if s["sf_elo"] == lb)
        us = next(s for s in ordered if s["sf_elo"] == ub)
        if ls["score"] == us["score"]:
            est = lb
        else:
            frac = (0.5 - ls["score"]) / (us["score"] - ls["score"])
            est = round(lb + frac * (ub - lb))

    elo_est = {"estimated_elo": est, "lower_bound": lb, "upper_bound": ub}
    log(f"estimate {json.dumps(elo_est)}")

    return {
        "strategy": strategy_name,
        "elo_estimate": elo_est,
        "summaries": summaries,
        "games": all_games,
    }


def main():
    global LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-tag", default="exp113_blend_sweep")
    parser.add_argument("--sf-elos", type=int, nargs="+", default=[1750, 1900])
    parser.add_argument("--games-per-opening-per-color", type=int, default=2)
    parser.add_argument("--no-syzygy", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    LOG_FILE = output_dir / f"elo_eval_{args.output_tag}.log"
    json_out = output_dir / f"elo_eval_{args.output_tag}.json"

    init_syzygy()

    log(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Model loaded ({param_count/1e6:.0f}M params) on {DEVICE}")

    # Strategies to sweep — cheapest signal first
    strategies = [
        ("greedy", strategy_greedy, {}),
        ("blend_k10_w15", strategy_blend_batched, {"top_k": 10, "value_weight": 0.15, "repetition_penalty": 0.0}),
        ("blend_k10_w30", strategy_blend_batched, {"top_k": 10, "value_weight": 0.30, "repetition_penalty": 0.0}),
        ("blend_k10_w15_antirep", strategy_blend_batched, {"top_k": 10, "value_weight": 0.15, "repetition_penalty": 0.15}),
        ("blend_k10_w30_antirep", strategy_blend_batched, {"top_k": 10, "value_weight": 0.30, "repetition_penalty": 0.15}),
        ("blend_k5_w15_antirep", strategy_blend_batched, {"top_k": 5, "value_weight": 0.15, "repetition_penalty": 0.15}),
    ]

    results = {}
    for strategy_name, fn, kwargs in strategies:
        def make_fn(f, kw):
            return lambda model, board, device: f(model, board, device, **kw)
        wrapped = make_fn(fn, kwargs)

        result = run_strategy_eval(
            model, wrapped, strategy_name, args.sf_elos, DEFAULT_OPENINGS,
            games_per_opening_per_color=args.games_per_opening_per_color,
            use_syzygy=not args.no_syzygy,
        )
        results[strategy_name] = result

        with open(json_out, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint),
                "strategies": results,
            }, f, indent=2)

    log(f"\n{'='*60}")
    log("FINAL COMPARISON")
    log(f"{'='*60}")

    # Sort by estimated ELO
    ranked = sorted(results.items(), key=lambda x: x[1]["elo_estimate"]["estimated_elo"], reverse=True)
    for name, r in ranked:
        est = r["elo_estimate"]
        scores = " | ".join(f"SF{s['sf_elo']}:{s['score']:.3f}" for s in r["summaries"])
        log(f"  {name:30s} ELO~{est['estimated_elo']:4d}  {scores}")


if __name__ == "__main__":
    main()
