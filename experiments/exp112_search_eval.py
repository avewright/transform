"""exp112: Search-enhanced ELO evaluation.

Test whether 1-ply value reranking improves ELO over pure greedy policy.
Run on BASELINE checkpoint (the strongest known model) to measure
search gains without conflating training quality issues.

Strategies tested:
  - greedy: pure policy argmax (current default)
  - rerank_k5: top-5 policy → pick best by value head
  - rerank_k10: top-10 policy → pick best by value head  
  - depth2_k5: 2-ply minimax with value head, top-5 moves
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

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
SYZYGY_MAX_PIECES = 5


def init_syzygy():
    global SYZYGY_TB
    if SYZYGY_DIR.exists() and any(SYZYGY_DIR.glob("*.rtbw")):
        try:
            SYZYGY_TB = chess.syzygy.open_tablebase(str(SYZYGY_DIR))
            print(f"Syzygy tablebases loaded from {SYZYGY_DIR}")
        except Exception:
            SYZYGY_TB = None


def get_syzygy_move(board: chess.Board) -> chess.Move | None:
    if SYZYGY_TB is None or len(board.piece_map()) > SYZYGY_MAX_PIECES:
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


@torch.no_grad()
def board_value(model, board, device) -> float:
    """Value from side-to-move's perspective: P(win) - P(loss).
    
    Model WDL convention (from pre-training on Lichess data):
      idx0 = P(White wins), idx1 = P(draw), idx2 = P(White loses)
    This is White-absolute, NOT side-to-move relative.
    We convert to side-to-move perspective here.
    """
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    wdl = F.softmax(result["value_logits"][0].float(), dim=-1)
    # White-absolute: P(W wins) - P(W loses)
    white_value = (wdl[0] - wdl[2]).item()
    # Flip sign for Black to move
    return white_value if board.turn == chess.WHITE else -white_value


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


# ── Strategy: 1-ply value reranking ──
@torch.no_grad()
def strategy_rerank(model, board, device, top_k=5, **kwargs):
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    
    k = min(top_k, int(mask.sum().item()))
    topk = torch.topk(logits, k)
    
    best_value = float("-inf")
    best_move = None
    
    for idx in topk.indices.tolist():
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue
        board.push(move)
        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
                child_value = 1.0
            else:
                child_value = 0.0
        else:
            child_value = -board_value(model, board, device)
        board.pop()
        
        if child_value > best_value:
            best_value = child_value
            best_move = move
    
    return best_move or next(iter(board.legal_moves))


# ── Strategy: 2-ply minimax ──
@torch.no_grad()
def strategy_depth2(model, board, device, top_k=5, **kwargs):
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    
    k = min(top_k, int(mask.sum().item()))
    topk = torch.topk(logits, k)
    
    best_value = float("-inf")
    best_move = None
    
    for idx in topk.indices.tolist():
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue
        
        board.push(move)
        
        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
                child_value = 1.0
            else:
                child_value = 0.0
        else:
            # Opponent's best reply (minimax)
            opp_inp = batch_boards_to_fused_token_ids([board], device)
            opp_result = model(opp_inp)
            opp_logits = opp_result["policy_logits"][0].float()
            opp_mask = legal_move_mask(board).to(device)
            opp_logits[~opp_mask] = float("-inf")
            opp_k = min(top_k, int(opp_mask.sum().item()))
            opp_topk = torch.topk(opp_logits, opp_k)
            
            worst_for_us = float("inf")
            for opp_idx in opp_topk.indices.tolist():
                opp_move = index_to_move(opp_idx)
                if opp_move not in board.legal_moves:
                    continue
                board.push(opp_move)
                if board.is_game_over():
                    outcome = board.outcome()
                    if outcome and outcome.winner is not None:
                        leaf_value = -1.0  # opponent won
                    else:
                        leaf_value = 0.0
                else:
                    leaf_value = board_value(model, board, device)
                board.pop()
                worst_for_us = min(worst_for_us, leaf_value)
            
            child_value = worst_for_us if worst_for_us != float("inf") else 0.0
        
        board.pop()
        
        if child_value > best_value:
            best_value = child_value
            best_move = move
    
    return best_move or next(iter(board.legal_moves))


# ── Strategy: Policy + value blend ──
@torch.no_grad()
def strategy_blend(model, board, device, top_k=10, value_weight=0.3, **kwargs):
    """Blend policy probability with 1-ply value to pick move."""
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    
    probs = F.softmax(logits, dim=-1)
    k = min(top_k, int(mask.sum().item()))
    topk = torch.topk(probs, k)
    
    best_score = float("-inf")
    best_move = None
    
    for idx, policy_prob in zip(topk.indices.tolist(), topk.values.tolist()):
        move = index_to_move(idx)
        if move not in board.legal_moves:
            continue
        board.push(move)
        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner is not None:
                val = 1.0
            else:
                val = 0.0
        else:
            val = -board_value(model, board, device)
        board.pop()
        
        # Blend: (1-w)*policy_prob + w*value
        # Normalize value from [-1,1] to [0,1]
        val_norm = (val + 1.0) / 2.0
        score = (1.0 - value_weight) * policy_prob + value_weight * val_norm
        
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
            # Try Syzygy first
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
                      games_per_opening_per_color=1, movetime=0.05, ply_cap=160,
                      stop_after_bracket=True, use_syzygy=True):
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
        
        summary = {
            "sf_elo": sf_elo, "games": games, "score": score,
            "score_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "w": w, "d": d, "l": l,
            "avg_plies": round(sum(r["plies"] for r in results) / games, 1) if games else 0.0,
        }
        summaries.append(summary)
        all_games.extend(results)
        log(f"summary {json.dumps(summary)}")
        
        if stop_after_bracket:
            above = [s for s in summaries if s["score"] >= 0.5]
            below = [s for s in summaries if s["score"] < 0.5]
            if above and below:
                log("bracketing complete")
                break
    
    # Estimate ELO
    ordered = sorted(summaries, key=lambda s: s["sf_elo"])
    above = [s for s in ordered if s["score"] >= 0.5]
    below = [s for s in ordered if s["score"] < 0.5]
    lb = max((s["sf_elo"] for s in above), default=None)
    ub = min((s["sf_elo"] for s in below), default=None)
    
    if lb is None:
        est = ordered[0]["sf_elo"]
    elif ub is None:
        est = ordered[-1]["sf_elo"]
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
    parser.add_argument("--output-tag", default="exp112_search")
    parser.add_argument("--elos", type=int, nargs="+", default=[1600, 1750, 1900, 2050])
    parser.add_argument("--games-per-opening-per-color", type=int, default=1)
    parser.add_argument("--strategies", nargs="+", 
                       default=["greedy", "rerank_k5", "rerank_k10", "blend_k10"],
                       choices=["greedy", "rerank_k5", "rerank_k10", "depth2_k5", "blend_k10"])
    parser.add_argument("--stop-after-bracket", action="store_true", default=True)
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
    
    strategy_map = {
        "greedy": (strategy_greedy, {}),
        "rerank_k5": (strategy_rerank, {"top_k": 5}),
        "rerank_k10": (strategy_rerank, {"top_k": 10}),
        "depth2_k5": (strategy_depth2, {"top_k": 5}),
        "blend_k10": (strategy_blend, {"top_k": 10, "value_weight": 0.3}),
    }
    
    results = {}
    for strategy_name in args.strategies:
        fn, kwargs = strategy_map[strategy_name]
        # Create a wrapper with the kwargs baked in
        def make_fn(f, kw):
            return lambda model, board, device: f(model, board, device, **kw)
        wrapped = make_fn(fn, kwargs)
        
        result = run_strategy_eval(
            model, wrapped, strategy_name, args.elos, DEFAULT_OPENINGS,
            games_per_opening_per_color=args.games_per_opening_per_color,
            stop_after_bracket=args.stop_after_bracket,
            use_syzygy=not args.no_syzygy,
        )
        results[strategy_name] = result
        
        # Save after each strategy
        with open(json_out, "w") as f:
            json.dump({
                "checkpoint": str(args.checkpoint),
                "strategies": results,
            }, f, indent=2)
    
    log(f"\n{'='*60}")
    log("FINAL SUMMARY")
    log(f"{'='*60}")
    for name, r in results.items():
        est = r["elo_estimate"]
        log(f"  {name}: ELO ~{est['estimated_elo']} [{est.get('lower_bound', '?')}-{est.get('upper_bound', '?')}]")
    
    log("Done.")


if __name__ == "__main__":
    main()
