"""exp115: Depth-2 blend strategy — look deeper with opponent policy response.

From exp113/114: blend_k10_w30 gives ~1900 ELO (+250 over greedy).
This experiment tests deeper search:
  - depth1_blend: same as blend_k10_w30 (1-ply value after our move)
  - depth2_blend: for each candidate, look at opponent's top response,
    then evaluate the position 2 plies deeper
  - minimax_blend: full minimax over top-k × top-j combinations

All use the BASELINE checkpoint (pre-training value head > retrained).
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
LOG_FILE = None


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


@torch.no_grad()
def batch_eval(model, boards, device):
    """Get policy probs and white-absolute value for multiple boards."""
    if not boards:
        return [], []
    inp = batch_boards_to_fused_token_ids(boards, device)
    result = model(inp)
    wdl = F.softmax(result["value_logits"].float(), dim=-1)
    white_values = (wdl[:, 0] - wdl[:, 2]).tolist()
    return result["policy_logits"].float(), white_values


def stm_value(white_value: float, turn: chess.Color) -> float:
    return white_value if turn == chess.WHITE else -white_value


# ── Strategy: Greedy ──
@torch.no_grad()
def strategy_greedy(model, board, device):
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    return index_to_move(logits.argmax().item())


# ── Strategy: Depth-1 blend (same as exp113 blend_k10_w30) ──
@torch.no_grad()
def strategy_depth1_blend(model, board, device, top_k=10, value_weight=0.30):
    """1-ply: policy + value after our move."""
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

    parent_turn = board.turn
    candidate_moves, candidate_probs, child_boards = [], [], []
    terminal_values = {}

    for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue
        board.push(move)
        if board.is_game_over():
            outcome = board.outcome()
            terminal_values[len(candidate_moves)] = 1.0 if (outcome and outcome.winner is not None) else 0.0
        else:
            child_boards.append(board.copy())
        board.pop()
        candidate_moves.append(move)
        candidate_probs.append(prob)

    if not candidate_moves:
        return next(iter(board.legal_moves))

    _, child_white_vals = batch_eval(model, child_boards, device)

    best_score, best_move = float("-inf"), None
    child_idx = 0
    for i, (move, prob) in enumerate(zip(candidate_moves, candidate_probs)):
        if i in terminal_values:
            val_stm = terminal_values[i]
        else:
            wv = child_white_vals[child_idx]
            child_idx += 1
            # Child position is opponent's turn — negate to get our perspective
            val_stm = stm_value(wv, parent_turn)

        val_norm = (val_stm + 1.0) / 2.0
        score = (1.0 - value_weight) * prob + value_weight * val_norm
        if score > best_score:
            best_score, best_move = score, move

    return best_move or next(iter(board.legal_moves))


# ── Strategy: Depth-2 blend ──
@torch.no_grad()
def strategy_depth2_blend(model, board, device, top_k=10, opp_k=3, value_weight=0.30):
    """2-ply: for each of our top-k moves, opponent responds with their top-j,
    we evaluate the resulting position. Use the worst-case (minimax) opponent response."""

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

    parent_turn = board.turn

    # Collect all level-1 child boards (non-terminal)
    our_moves = []
    our_probs = []
    child_boards_L1 = []  # boards after our move (opponent to play)
    terminal_L1 = {}  # our_move_idx -> val

    for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue
        board.push(move)
        if board.is_game_over():
            outcome = board.outcome()
            terminal_L1[len(our_moves)] = 1.0 if (outcome and outcome.winner is not None) else 0.0
        else:
            child_boards_L1.append(board.copy())
        board.pop()
        our_moves.append(move)
        our_probs.append(prob)

    if not our_moves:
        return next(iter(board.legal_moves))

    # Get opponent policy for all L1 children (to find their best responses)
    if child_boards_L1:
        opp_policy_logits_batch, l1_white_vals = batch_eval(model, child_boards_L1, device)
    else:
        opp_policy_logits_batch, l1_white_vals = torch.tensor([]), []

    # For each L1 child, generate opponent's top-opp_k responses -> L2 boards
    l2_boards = []
    l2_map = []  # (our_move_idx, opp_move_idx_in_topk) -> index into l2_boards
    l2_terminal = {}  # index into l2_boards conceptually -> val

    board_child_idx = 0
    for i, move in enumerate(our_moves):
        if i in terminal_L1:
            continue  # terminal at L1, no need for L2

        opp_logits = opp_policy_logits_batch[board_child_idx].clone()
        child_board = child_boards_L1[board_child_idx]
        board_child_idx += 1

        # Mask illegal opponent moves
        opp_mask = legal_move_mask(child_board).to(device)
        opp_logits[~opp_mask] = float("-inf")
        opp_k_actual = min(opp_k, int(opp_mask.sum().item()))
        if opp_k_actual == 0:
            # Opponent has no moves — stalemate or checkmate
            # stm_value for opponent position
            l2_terminal[len(l2_map)] = 0.0  # draw (stalemate)
            l2_map.append(i)
            continue

        opp_topk = torch.topk(F.softmax(opp_logits, dim=-1), opp_k_actual)

        for j, (opp_idx, opp_prob) in enumerate(zip(opp_topk.indices.tolist(), opp_topk.values.tolist())):
            opp_move = index_to_move(opp_idx)
            if opp_move not in child_board.legal_moves:
                continue
            child_board.push(opp_move)
            if child_board.is_game_over():
                outcome = child_board.outcome()
                l2_idx = len(l2_map)
                if outcome and outcome.winner is not None:
                    # Opponent moved and we're checkmated = we lose
                    l2_terminal[l2_idx] = -1.0
                else:
                    l2_terminal[l2_idx] = 0.0
                l2_map.append(i)
            else:
                l2_boards.append(child_board.copy())
                l2_map.append(i)
            child_board.pop()

    # Batch evaluate all L2 boards
    if l2_boards:
        _, l2_white_vals = batch_eval(model, l2_boards, device)
    else:
        l2_white_vals = []

    # Aggregate: for each of our moves, take the MINIMUM value across opponent responses (minimax)
    minimax_vals = {}  # our_move_idx -> worst-case value for us
    l2_board_idx = 0
    for map_idx, our_idx in enumerate(l2_map):
        if map_idx in l2_terminal:
            val_stm = l2_terminal[map_idx]
        else:
            wv = l2_white_vals[l2_board_idx]
            l2_board_idx += 1
            # L2 position is our turn again — stm_value gives our perspective
            val_stm = stm_value(wv, parent_turn)

        if our_idx not in minimax_vals:
            minimax_vals[our_idx] = val_stm
        else:
            minimax_vals[our_idx] = min(minimax_vals[our_idx], val_stm)

    # Score each move
    best_score, best_move = float("-inf"), None
    l1_board_idx = 0
    for i, (move, prob) in enumerate(zip(our_moves, our_probs)):
        if i in terminal_L1:
            val_stm = terminal_L1[i]
        elif i in minimax_vals:
            val_stm = minimax_vals[i]
        else:
            # Fallback to L1 value (shouldn't happen normally)
            wv = l1_white_vals[l1_board_idx]
            val_stm = stm_value(wv, parent_turn)
            l1_board_idx += 1

        val_norm = (val_stm + 1.0) / 2.0
        score = (1.0 - value_weight) * prob + value_weight * val_norm
        if score > best_score:
            best_score, best_move = score, move

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
                        result = play_one(engine, model, strategy_fn, sf_elo, model_color=color,
                                         opening=opening, movetime=movetime, ply_cap=ply_cap,
                                         use_syzygy=use_syzygy)
                        result["repeat_idx"] = repeat
                        results.append(result)
                        log(f"game {json.dumps(result)}")
        finally:
            engine.quit()

        total_score = sum(r["score"] for r in results)
        games = len(results)
        score = total_score / games if games else 0.0
        ci_low, ci_high = wilson_interval(total_score, games)

        w = sum(1 for r in results if r["score"] == 1.0)
        d = sum(1 for r in results if r["score"] == 0.5)
        l = sum(1 for r in results if r["score"] == 0.0)
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
        frac = (0.5 - ls["score"]) / (us["score"] - ls["score"]) if ls["score"] != us["score"] else 0.0
        est = round(lb + frac * (ub - lb))

    elo_est = {"estimated_elo": est, "lower_bound": lb, "upper_bound": ub}
    log(f"estimate {json.dumps(elo_est)}")

    return {"strategy": strategy_name, "elo_estimate": elo_est,
            "summaries": summaries, "games": all_games}


def main():
    global LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs" / "hf_checkpoint" / "best_model.pt")
    parser.add_argument("--sf-elos", type=int, nargs="+", default=[1900, 2050])
    parser.add_argument("--output-tag", default="exp115_depth2")
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    LOG_FILE = output_dir / f"elo_eval_{args.output_tag}.log"
    json_out = output_dir / f"elo_eval_{args.output_tag}.json"

    init_syzygy()
    model = load_model(args.checkpoint)
    log(f"Model loaded on {DEVICE}")

    strategies = [
        ("depth1_blend_k10_w30", lambda m, b, d:
            strategy_depth1_blend(m, b, d, top_k=10, value_weight=0.30)),
        ("depth2_blend_k10_opp3_w30", lambda m, b, d:
            strategy_depth2_blend(m, b, d, top_k=10, opp_k=3, value_weight=0.30)),
        ("depth2_blend_k10_opp5_w30", lambda m, b, d:
            strategy_depth2_blend(m, b, d, top_k=10, opp_k=5, value_weight=0.30)),
        ("depth2_blend_k5_opp3_w30", lambda m, b, d:
            strategy_depth2_blend(m, b, d, top_k=5, opp_k=3, value_weight=0.30)),
    ]

    results = {}
    for name, fn in strategies:
        result = run_strategy_eval(
            model, fn, name, args.sf_elos, DEFAULT_OPENINGS,
            games_per_opening_per_color=2,
        )
        results[name] = result
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)

    log(f"\n{'='*60}")
    log("FINAL COMPARISON")
    log(f"{'='*60}")

    ranked = sorted(results.items(),
                    key=lambda x: x[1]["elo_estimate"]["estimated_elo"], reverse=True)
    for name, r in ranked:
        est = r["elo_estimate"]
        scores = " | ".join(f"SF{s['sf_elo']}:{s['score']:.3f}" for s in r["summaries"])
        log(f"  {name:40s} ELO~{est['estimated_elo']:4d}  {scores}")


if __name__ == "__main__":
    main()
