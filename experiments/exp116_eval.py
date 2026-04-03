"""exp116_eval: Compare baseline vs exp116 (correct value targets) with blend strategy.

Tests both checkpoints with:
  - greedy (policy only)
  - blend_k10_w30 (our best strategy)

At SF1900 and SF2050 (32 games each).
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from collections import Counter
from functools import partial

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


def get_syzygy_move(board):
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


def load_model(checkpoint_path):
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def legal_move_mask(board):
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        uci = move.uci()
        if uci in UCI_TO_IDX:
            mask[UCI_TO_IDX[uci]] = True
    return mask


def index_to_move(idx):
    return chess.Move.from_uci(IDX_TO_UCI[idx])


@torch.no_grad()
def batch_board_values(model, boards, device):
    if not boards:
        return []
    inp = batch_boards_to_fused_token_ids(boards, device)
    result = model(inp)
    wdl = F.softmax(result["value_logits"].float(), dim=-1)
    white_values = (wdl[:, 0] - wdl[:, 2]).tolist()
    return white_values


@torch.no_grad()
def strategy_greedy(model, board, device, **kwargs):
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    return index_to_move(logits.argmax().item())


@torch.no_grad()
def strategy_blend_batched(model, board, device, top_k=10, value_weight=0.3, **kwargs):
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

    candidate_moves = []
    candidate_probs = []
    candidate_boards = []
    game_over_results = {}
    parent_turn = board.turn

    for idx, policy_prob in zip(topk.indices.tolist(), topk.values.tolist()):
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue
        board.push(move)
        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
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

    if candidate_boards:
        white_values = batch_board_values(model, candidate_boards, device)
    else:
        white_values = []

    best_score = float("-inf")
    best_move = None
    board_idx = 0

    for i, (move, policy_prob) in enumerate(zip(candidate_moves, candidate_probs)):
        if i in game_over_results:
            val_stm = game_over_results[i]
        else:
            white_val = white_values[board_idx]
            val_stm = -(-white_val if parent_turn == chess.WHITE else white_val)
            board_idx += 1

        val_norm = (val_stm + 1.0) / 2.0
        score = (1.0 - value_weight) * policy_prob + value_weight * val_norm

        if score > best_score:
            best_score = score
            best_move = move

    return best_move or next(iter(board.legal_moves))


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


def play_one(engine, model, strategy_fn, sf_elo, model_color, opening, movetime=0.05, ply_cap=200):
    board = chess.Board()
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            tb_move = get_syzygy_move(board)
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


def run_eval(model, strategy_fn, strategy_name, sf_elos, openings, games_per=2):
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
                    for _ in range(games_per):
                        r = play_one(engine, model, strategy_fn, sf_elo, color, opening)
                        results.append(r)
                        log(f"game {json.dumps(r)}")
        finally:
            engine.quit()

        games = len(results)
        total_score = sum(r["score"] for r in results)
        score = total_score / games if games else 0.0
        ci_low, ci_high = wilson_interval(total_score, games)

        w = sum(1 for r in results if r["score"] == 1.0)
        d = sum(1 for r in results if r["score"] == 0.5)
        l = sum(1 for r in results if r["score"] == 0.0)

        summary = {
            "sf_elo": sf_elo, "games": games, "score": score,
            "score_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "w": w, "d": d, "l": l,
            "avg_plies": round(sum(r["plies"] for r in results) / games, 1) if games else 0.0,
        }
        summaries.append(summary)
        all_games.extend(results)
        log(f"summary {json.dumps(summary)}")

    return {"strategy": strategy_name, "summaries": summaries, "games": all_games}


def main():
    global LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--sf-elos", type=int, nargs="+", default=[1900, 2050])
    parser.add_argument("--games-per", type=int, default=2)
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    LOG_FILE = output_dir / "elo_eval_exp116b.log"
    json_out = output_dir / "elo_eval_exp116b.json"

    init_syzygy()

    # Checkpoints to compare
    baseline_ckpt = ROOT / "outputs" / "hf_checkpoint" / "best_model.pt"
    # Prefer exp116b (low-LR, better results) over original exp116
    exp116b_best = ROOT / "outputs" / "exp116b_low_lr" / "best_model.pt"
    exp116_best = ROOT / "outputs" / "exp116_correct_value_finetune" / "best_model.pt"
    exp116_latest = ROOT / "outputs" / "exp116_correct_value_finetune" / "latest_model.pt"

    exp116_ckpt = exp116b_best if exp116b_best.exists() else (exp116_best if exp116_best.exists() else exp116_latest)

    checkpoints = {"baseline": baseline_ckpt}
    if exp116_ckpt.exists():
        checkpoints["exp116"] = exp116_ckpt
    else:
        log(f"WARNING: exp116 checkpoint not found, evaluating baseline only")

    strategies = [
        ("greedy", strategy_greedy, {}),
        ("blend_k10_w30", strategy_blend_batched, {"top_k": 10, "value_weight": 0.30}),
    ]

    all_results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        log(f"\nLoading {ckpt_name} from {ckpt_path}")
        model = load_model(ckpt_path)
        log(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

        for strat_name, strat_fn, strat_kwargs in strategies:
            full_name = f"{ckpt_name}_{strat_name}"

            def make_fn(f, kw):
                return lambda model, board, device: f(model, board, device, **kw)
            wrapped = make_fn(strat_fn, strat_kwargs)

            result = run_eval(
                model, wrapped, full_name, args.sf_elos, DEFAULT_OPENINGS,
                games_per=args.games_per,
            )
            all_results[full_name] = result

            with open(json_out, "w") as f:
                json.dump(all_results, f, indent=2)

    # Final comparison
    log(f"\n{'='*60}")
    log("FINAL COMPARISON")
    log(f"{'='*60}")

    for name, result in all_results.items():
        scores = " | ".join(
            f"SF{s['sf_elo']}:{s['score']:.3f}({s['w']}W-{s['d']}D-{s['l']}L)"
            for s in result["summaries"]
        )
        log(f"  {name:35s} {scores}")


if __name__ == "__main__":
    main()
