"""exp117: Inference-time optimization sweep (no training needed).

Tests various inference-time tricks on the BASELINE checkpoint:
1. Policy temperature (sharper/softer softmax)
2. Higher value weights in blend  
3. Dynamic value weight (based on policy entropy)
4. Top-k variations

Goal: Find the best inference-time config to maximize ELO without any training.
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


def load_model(cp):
    model = ChessTransformer200M()
    state = torch.load(cp, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def legal_move_mask(board):
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
    for m in board.legal_moves:
        uci = m.uci()
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
    return (wdl[:, 0] - wdl[:, 2]).tolist()


# ── Strategy: Greedy with temperature ──
@torch.no_grad()
def strategy_greedy(model, board, device, temperature=1.0, **kw):
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    if temperature != 1.0:
        logits = logits / temperature
    return index_to_move(logits.argmax().item())


# ── Strategy: Blend with temperature and configurable params ──
@torch.no_grad()
def strategy_blend(model, board, device, top_k=10, value_weight=0.3,
                   temperature=1.0, adaptive_vw=False, **kw):
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    # Apply temperature before softmax
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)
    k = min(top_k, int(mask.sum().item()))
    if k == 0:
        return next(iter(board.legal_moves))
    topk = torch.topk(probs, k)

    # Adaptive value weight: increase when policy is uncertain
    if adaptive_vw:
        # Compute entropy of top-k probs
        topk_probs = topk.values
        entropy = -(topk_probs * (topk_probs + 1e-10).log()).sum().item()
        max_entropy = math.log(k)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        # Scale value_weight: low entropy → low vw, high entropy → high vw
        vw = value_weight * (0.5 + norm_entropy)  # range: [vw*0.5, vw*1.5]
    else:
        vw = value_weight

    candidate_moves = []
    candidate_probs = []
    candidate_boards = []
    game_over_results = {}
    parent_turn = board.turn

    for idx, pp in zip(topk.indices.tolist(), topk.values.tolist()):
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
        candidate_probs.append(pp)

    if not candidate_moves:
        return next(iter(board.legal_moves))

    if candidate_boards:
        white_values = batch_board_values(model, candidate_boards, device)
    else:
        white_values = []

    best_score = float("-inf")
    best_move = None
    board_idx = 0

    for i, (move, pp) in enumerate(zip(candidate_moves, candidate_probs)):
        if i in game_over_results:
            vs = game_over_results[i]
        else:
            wv = white_values[board_idx]
            vs = wv if parent_turn == chess.WHITE else -wv
            board_idx += 1

        vn = (vs + 1.0) / 2.0
        score = (1.0 - vw) * pp + vw * vn

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


def wilson_interval(s, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    p = s / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0.0, c - m), min(1.0, c + m)


def play_one(engine, model, strategy_fn, sf_elo, color, opening, movetime=0.05, ply_cap=200):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == color:
            tb = get_syzygy_move(board)
            move = tb if tb else strategy_fn(model, board, DEVICE)
        else:
            move = engine.play(board, chess.engine.Limit(time=movetime)).move
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    elif outcome.winner == color:
        score = 1.0
    else:
        score = 0.0

    return {
        "sf_elo": sf_elo, "color": "w" if color == chess.WHITE else "b",
        "score": score, "plies": len(board.move_stack),
        "term": outcome.termination.name if outcome else "PLY_CAP",
    }


def run_eval(model, strategy_fn, name, sf_elos, openings, games_per=2):
    log(f"\n{'='*60}")
    log(f"Strategy: {name}")
    log(f"{'='*60}")

    summaries = []
    all_games = []

    for sf_elo in sf_elos:
        log(f"  sf_elo={sf_elo}")
        engine = chess.engine.SimpleEngine.popen_uci(str(SF))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1, "Hash": 32})
        results = []
        try:
            for opening in openings:
                for color in [chess.WHITE, chess.BLACK]:
                    for _ in range(games_per):
                        r = play_one(engine, model, strategy_fn, sf_elo, color, opening)
                        results.append(r)
        finally:
            engine.quit()

        games = len(results)
        ts = sum(r["score"] for r in results)
        sc = ts / games if games else 0.0
        ci = wilson_interval(ts, games)
        w = sum(1 for r in results if r["score"] == 1.0)
        d = sum(1 for r in results if r["score"] == 0.5)
        l = sum(1 for r in results if r["score"] == 0.0)

        summary = {"sf_elo": sf_elo, "games": games, "score": sc, "ci95": list(ci), "w": w, "d": d, "l": l}
        summaries.append(summary)
        all_games.extend(results)
        log(f"  → {sc:.3f} ({w}W-{d}D-{l}L)")

    return {"strategy": name, "summaries": summaries, "games": all_games}


def main():
    global LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--sf-elos", type=int, nargs="+", default=[1900])
    parser.add_argument("--games-per", type=int, default=2)
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    LOG_FILE = output_dir / "elo_eval_exp117_inference_opt.log"
    json_out = output_dir / "elo_eval_exp117_inference_opt.json"

    init_syzygy()

    ckpt = ROOT / "outputs" / "hf_checkpoint" / "best_model.pt"
    log(f"Loading baseline from {ckpt}")
    model = load_model(ckpt)
    log(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    # Strategies to test
    strategies = [
        # Baseline
        ("greedy_t1.0", strategy_greedy, {"temperature": 1.0}),
        # Temperature variations on greedy
        ("greedy_t0.5", strategy_greedy, {"temperature": 0.5}),
        ("greedy_t0.7", strategy_greedy, {"temperature": 0.7}),
        # Blend with default temp
        ("blend_k10_w30_t1.0", strategy_blend, {"top_k": 10, "value_weight": 0.30}),
        # Blend with lower temperature (sharper policy)
        ("blend_k10_w30_t0.5", strategy_blend, {"top_k": 10, "value_weight": 0.30, "temperature": 0.5}),
        ("blend_k10_w30_t0.7", strategy_blend, {"top_k": 10, "value_weight": 0.30, "temperature": 0.7}),
        # Higher value weight
        ("blend_k10_w50_t1.0", strategy_blend, {"top_k": 10, "value_weight": 0.50}),
        ("blend_k10_w50_t0.5", strategy_blend, {"top_k": 10, "value_weight": 0.50, "temperature": 0.5}),
        # Wider k
        ("blend_k20_w30_t1.0", strategy_blend, {"top_k": 20, "value_weight": 0.30}),
        # Adaptive value weight
        ("blend_k10_w30_adaptive", strategy_blend, {"top_k": 10, "value_weight": 0.30, "adaptive_vw": True}),
    ]

    results = {}
    for name, fn, kwargs in strategies:
        def make_fn(f, kw):
            return lambda model, board, device: f(model, board, device, **kw)
        wrapped = make_fn(fn, kwargs)

        result = run_eval(model, wrapped, name, args.sf_elos, DEFAULT_OPENINGS, args.games_per)
        results[name] = result

        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)

    log(f"\n{'='*60}")
    log("FINAL RANKING (by score)")
    log(f"{'='*60}")

    # Rank by average score across SF levels
    ranked = sorted(results.items(),
                    key=lambda x: sum(s["score"] for s in x[1]["summaries"]) / len(x[1]["summaries"]),
                    reverse=True)
    for name, r in ranked:
        scores = " | ".join(f"SF{s['sf_elo']}:{s['score']:.3f}({s['w']}W-{s['d']}D-{s['l']}L)" for s in r["summaries"])
        log(f"  {name:35s} {scores}")


if __name__ == "__main__":
    main()
