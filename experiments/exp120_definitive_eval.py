"""exp120: High-sample-count definitive ELO evaluation.

Previous evals used 32 games → ±150 ELO noise. This runs 128 games
per config to get reliable measurements.

Tests ONLY the configs that have shown promise:
  - baseline greedy 
  - baseline blend_k10_w30
  - exp116b blend_k10_w30 (corrected value head)

At SF1900 only (our bracket boundary).
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


def init_syzygy():
    global SYZYGY_TB
    if SYZYGY_DIR.exists() and any(SYZYGY_DIR.glob("*.rtbw")):
        try:
            SYZYGY_TB = chess.syzygy.open_tablebase(str(SYZYGY_DIR))
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


@torch.no_grad()
def strategy_greedy(model, board, device):
    inp = batch_boards_to_fused_token_ids([board], device)
    result = model(inp)
    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    return index_to_move(logits.argmax().item())


@torch.no_grad()
def strategy_blend(model, board, device, top_k=10, value_weight=0.3):
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
        score = (1.0 - value_weight) * pp + value_weight * vn

        if score > best_score:
            best_score = score
            best_move = move

    return best_move or next(iter(board.legal_moves))


OPENINGS = [
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


def elo_from_score(score):
    """Convert win-rate to ELO difference."""
    if score <= 0:
        return -400
    if score >= 1:
        return 400
    return -400 * math.log10(1 / score - 1)


def play_one(engine, model, strategy_fn, color, opening, movetime=0.05, ply_cap=200):
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
        return 0.5
    elif outcome.winner == color:
        return 1.0
    else:
        return 0.0


def run_eval(model, strategy_fn, name, sf_elo, n_games):
    log(f"\n{'='*60}")
    log(f"Config: {name} vs SF{sf_elo} ({n_games} games)")
    log(f"{'='*60}")

    engine = chess.engine.SimpleEngine.popen_uci(str(SF))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1, "Hash": 32})

    scores = []
    w = d = l = 0
    try:
        game_idx = 0
        while game_idx < n_games:
            opening = OPENINGS[game_idx % len(OPENINGS)]
            color = chess.WHITE if (game_idx // len(OPENINGS)) % 2 == 0 else chess.BLACK
            s = play_one(engine, model, strategy_fn, color, opening)
            scores.append(s)
            if s == 1.0:
                w += 1
            elif s == 0.5:
                d += 1
            else:
                l += 1
            game_idx += 1

            if game_idx % 16 == 0:
                total = sum(scores)
                avg = total / len(scores)
                ci = wilson_interval(total, len(scores))
                elo_diff = elo_from_score(avg)
                log(f"  [{game_idx}/{n_games}] score={avg:.3f} ({w}W-{d}D-{l}L) "
                    f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO_diff={elo_diff:+.0f} "
                    f"est_ELO={sf_elo + elo_diff:.0f}")
    finally:
        engine.quit()

    total = sum(scores)
    avg = total / len(scores)
    ci = wilson_interval(total, len(scores))
    elo_diff = elo_from_score(avg)

    result = {
        "name": name, "sf_elo": sf_elo, "games": len(scores),
        "score": avg, "ci95": list(ci), "w": w, "d": d, "l": l,
        "elo_diff": round(elo_diff), "est_elo": round(sf_elo + elo_diff),
    }
    log(f"FINAL: {name} score={avg:.3f} ({w}W-{d}D-{l}L) "
        f"CI=[{ci[0]:.3f},{ci[1]:.3f}] ELO≈{sf_elo + elo_diff:.0f}")
    return result


def main():
    global LOG_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--sf-elo", type=int, default=1900)
    parser.add_argument("--games", type=int, default=128)
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    LOG_FILE = output_dir / "elo_eval_exp120_definitive.log"
    json_out = output_dir / "elo_eval_exp120_definitive.json"

    init_syzygy()

    baseline_ckpt = ROOT / "outputs" / "hf_checkpoint" / "best_model.pt"
    exp116b_ckpt = ROOT / "outputs" / "exp116b_low_lr" / "best_model.pt"

    results = []

    # 1. Baseline greedy
    log(f"\nLoading baseline from {baseline_ckpt}")
    model = load_model(baseline_ckpt)

    r = run_eval(model, strategy_greedy, "baseline_greedy", args.sf_elo, args.games)
    results.append(r)
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    # 2. Baseline blend
    def blend_fn(model, board, device):
        return strategy_blend(model, board, device, top_k=10, value_weight=0.3)

    r = run_eval(model, blend_fn, "baseline_blend_k10_w30", args.sf_elo, args.games)
    results.append(r)
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)

    # 3. exp116b blend (if checkpoint exists)
    if exp116b_ckpt.exists():
        log(f"\nLoading exp116b from {exp116b_ckpt}")
        model = load_model(exp116b_ckpt)

        r = run_eval(model, blend_fn, "exp116b_blend_k10_w30", args.sf_elo, args.games)
        results.append(r)
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    log(f"\n{'='*60}")
    log("DEFINITIVE RESULTS")
    log(f"{'='*60}")
    for r in results:
        log(f"  {r['name']:30s} {r['score']:.3f} ({r['w']}W-{r['d']}D-{r['l']}L) "
            f"CI=[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] ELO≈{r['est_elo']}")


if __name__ == "__main__":
    main()
