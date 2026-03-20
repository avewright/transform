"""exp_av_real_games.py — Action-value on REAL game positions.

Hypothesis: Action-value Q(s,a) training will outperform policy-only CE
when applied to real game positions (HF dataset), where position quality
is no longer the bottleneck.

Previous finding: On random-play positions, AV and policy CE tie at ~8-9%.
On HF game-play positions, policy CE reaches ~22%. The question is whether
AV's ~30x denser gradient signal helps when positions are already good.

Method:
  1. Sample 3K positions from HF dataset (real games)
  2. Label each with Stockfish depth=8 all-move evals (~3 min)
  3. Train two models: policy-only vs AV on same data
  4. Same forward path, same eval (hardened comparison)

Time budget: ~10 min (3 min labeling + 2×3 min training).
Primary metric: top-1 accuracy on 500 held-out HF positions.
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/exp_av_real_games")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_POSITIONS = 3000
SF_DEPTH = 8
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
EVAL_SIZE = 500


def cp_to_q(cp, eval_type="cp"):
    if eval_type == "mate":
        return (1.0 - 0.001 * min(abs(cp), 50)) if cp > 0 else (0.001 * min(abs(cp), 50)) if cp < 0 else 0.5
    return 1.0 / (1.0 + math.exp(-cp / 111.7))


def cp_to_wdl(cp, eval_type="cp"):
    if eval_type == "mate":
        return [1.0, 0.0, 0.0] if cp > 0 else [0.0, 0.0, 1.0] if cp < 0 else [0.0, 1.0, 0.0]
    k = 1.0 / 111.7
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
    total = win + draw + loss
    return [win/total, draw/total, loss/total]


class ActionValueHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, vocab_size))
    def forward(self, x):
        return torch.sigmoid(self.head(x))


# === HF dataset → Board ===

_PIECE2CHESS = {
    1: (chess.PAWN, chess.WHITE), 2: (chess.KNIGHT, chess.WHITE),
    3: (chess.BISHOP, chess.WHITE), 4: (chess.ROOK, chess.WHITE),
    5: (chess.QUEEN, chess.WHITE), 6: (chess.KING, chess.WHITE),
    7: (chess.PAWN, chess.BLACK), 8: (chess.KNIGHT, chess.BLACK),
    9: (chess.BISHOP, chess.BLACK), 10: (chess.ROOK, chess.BLACK),
    11: (chess.QUEEN, chess.BLACK), 12: (chess.KING, chess.BLACK),
}

def hf_sample_to_board(board_flat, turn_raw):
    board = chess.Board(fen=None)
    board.clear()
    for idx in range(64):
        piece_val = board_flat[idx]
        if piece_val > 0:
            row = idx // 8; col = idx % 8
            rank = 7 - row; sq = rank * 8 + col
            pt, color = _PIECE2CHESS[piece_val]
            board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = bool(turn_raw)
    return board


def extract_hf_positions(n, seed=42):
    """Extract n valid positions from HF dataset."""
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    
    positions = []
    for i in indices:
        if len(positions) >= n:
            break
        s = ds[i]
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            if not board.is_game_over() and list(board.legal_moves):
                positions.append(board)
        except Exception:
            continue
    return positions


def label_positions_with_sf(positions, depth=SF_DEPTH):
    """Label each position with Stockfish evals for all legal moves."""
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=depth, parameters={"Threads": 2, "Hash": 128})
    
    labeled = []
    t0 = time.time()
    for i, board in enumerate(positions):
        fen = board.fen()
        legal_moves = list(board.legal_moves)
        move_values = []
        best_cp = None; best_uci = None; best_sort = -float("inf")
        
        for move in legal_moves:
            board.push(move)
            try:
                sf.set_fen_position(board.fen())
                ev = sf.get_evaluation()
                eval_type = ev.get("type", "cp")
                cp_from_mover = -ev.get("value", 0)
                move_values.append({"uci": move.uci(), "cp": cp_from_mover, "type": eval_type})
                sort_val = (100000 - abs(cp_from_mover)) if (eval_type == "mate" and cp_from_mover > 0) else (-100000 + abs(cp_from_mover)) if eval_type == "mate" else cp_from_mover
                if sort_val > best_sort:
                    best_sort = sort_val; best_uci = move.uci(); best_cp = cp_from_mover
            except Exception:
                pass
            board.pop()
        
        if move_values and best_uci and best_uci in UCI_TO_IDX:
            move_values.sort(key=lambda m: m["cp"], reverse=True)
            labeled.append({
                "fen": fen, "best_uci": best_uci, "best_cp": best_cp,
                "move_values": move_values,
                "wdl": cp_to_wdl(best_cp, move_values[0]["type"]),
                "num_legal": len(legal_moves),
            })
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Labeled {i+1}/{len(positions)} ({rate:.1f} pos/s, ETA {(len(positions)-i-1)/rate:.0f}s)")
    
    return labeled


# === Shared forward, batching, eval ===

def forward_pass(model, batch_input):
    tokens = model.encoder(batch_input)
    embeds = model.input_proj(tokens)
    embeds = embeds.to(next(model.backbone.parameters()).dtype)
    outputs = model.backbone(inputs_embeds=embeds, use_cache=False)
    hidden = outputs.last_hidden_state.float()
    g = hidden[:, 0, :]
    return g, model.policy_head(g), model.value_head(g)


def make_batches(data, batch_size, device, include_av=False):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i+batch_size]
        boards = [chess.Board(d["fen"]) for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        best_idx = torch.tensor([UCI_TO_IDX[d["best_uci"]] for d in chunk], dtype=torch.long, device=device)
        wdl_targets = torch.tensor([max(range(3), key=lambda x: d["wdl"][x]) for d in chunk], dtype=torch.long, device=device)
        batch = {"input": batch_input, "best_idx": best_idx, "wdl": wdl_targets}
        if include_av:
            q_targets = torch.zeros(len(chunk), VOCAB_SIZE, device=device)
            q_mask = torch.zeros(len(chunk), VOCAB_SIZE, dtype=torch.bool, device=device)
            for j, d in enumerate(chunk):
                for mv in d["move_values"]:
                    if mv["uci"] in UCI_TO_IDX:
                        idx = UCI_TO_IDX[mv["uci"]]
                        q_targets[j, idx] = cp_to_q(mv["cp"], mv["type"])
                        q_mask[j, idx] = True
            batch["q_targets"] = q_targets; batch["q_mask"] = q_mask
        batches.append(batch)
    return batches


def evaluate(model, eval_data, device, batch_size=64):
    model.eval()
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i+batch_size]
            boards = [chess.Board(d["fen"]) for d in chunk]
            targets = [UCI_TO_IDX.get(d["best_uci"], 0) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            _, logits, _ = forward_pass(model, batch_input)
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()
            for j, t in enumerate(targets):
                total += 1
                if preds[j] == t: correct += 1
                if t in top3s[j]: top3_correct += 1
    return {"accuracy": correct/max(total,1), "top3_accuracy": top3_correct/max(total,1), "total": total}


def train_variant(train_data, eval_data, device, full_model, use_av=False, label=""):
    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    av_head = ActionValueHead(model.hidden_size, VOCAB_SIZE).to(device) if use_av else None
    
    params = [p for p in model.parameters() if p.requires_grad]
    if av_head:
        params += list(av_head.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        if av_head: av_head.train()
        batches = make_batches(train_data, BATCH_SIZE, device, include_av=use_av)
        ep_loss = steps = 0
        for batch in batches:
            g, policy_logits, value_logits = forward_pass(model, batch["input"])
            p_loss = F.cross_entropy(policy_logits, batch["best_idx"])
            v_loss = F.cross_entropy(value_logits, batch["wdl"])
            loss = p_loss + 0.5 * v_loss
            if use_av and av_head:
                q_pred = av_head(g)
                mask_sum = batch["q_mask"].float().sum().clamp(min=1.0)
                av_loss = ((q_pred - batch["q_targets"])**2 * batch["q_mask"].float()).sum() / mask_sum
                loss = loss + av_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step(); scheduler.step()
            ep_loss += loss.item(); steps += 1
        
        ev = evaluate(model, eval_data, device)
        elapsed = time.time() - t0
        history.append({**ev, "loss": ep_loss/max(steps,1), "epoch": epoch+1})
        print(f"  [{label}] Ep {epoch+1}/{EPOCHS}: loss={ep_loss/max(steps,1):.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")
    
    return model, history


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    config = {
        "experiment": "exp_av_real_games",
        "hypothesis": "AV training helps on real game positions where data quality is not the bottleneck",
        "primary_metric": "top-1 accuracy on SF-labeled HF game positions",
        "command": " ".join(sys.argv),
        "seed": SEED, "num_positions": NUM_POSITIONS, "sf_depth": SF_DEPTH,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "eval_size": EVAL_SIZE,
    }
    print(f"Config: {json.dumps(config, indent=2)}")

    # Step 1: Extract real game positions from HF
    print(f"\n[1/4] Extracting {NUM_POSITIONS + EVAL_SIZE} positions from HF game dataset...")
    all_positions = extract_hf_positions(NUM_POSITIONS + EVAL_SIZE, seed=SEED)
    print(f"  Got {len(all_positions)} positions")

    # Step 2: Label with Stockfish
    print(f"\n[2/4] Labeling with Stockfish depth={SF_DEPTH} (all legal moves)...")
    all_labeled = label_positions_with_sf(all_positions, depth=SF_DEPTH)
    print(f"  Labeled {len(all_labeled)} positions")

    # Split
    rng = random.Random(SEED)
    rng.shuffle(all_labeled)
    eval_data = all_labeled[:EVAL_SIZE]
    train_data = all_labeled[EVAL_SIZE:]
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")
    avg_moves = sum(d["num_legal"] for d in all_labeled) / max(len(all_labeled), 1)
    print(f"  Avg legal moves: {avg_moves:.1f}")

    # Step 3: Load model & train both variants
    print(f"\n[3/4] Loading backbone...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    print(f"\n{'='*60}")
    print(f" A: Policy CE + Value CE (real game positions)")
    print(f"{'='*60}")
    model_a, hist_a = train_variant(train_data, eval_data, device, full_model, use_av=False, label="Policy")
    final_a = evaluate(model_a, eval_data, device)
    best_a = max(h["accuracy"] for h in hist_a)
    print(f"  Final: acc={final_a['accuracy']:.1%} top3={final_a['top3_accuracy']:.1%} (best={best_a:.1%})")

    del model_a; torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f" B: AV MSE + Policy CE + Value CE (real game positions)")
    print(f"{'='*60}")
    model_b, hist_b = train_variant(train_data, eval_data, device, full_model, use_av=True, label="AV")
    final_b = evaluate(model_b, eval_data, device)
    best_b = max(h["accuracy"] for h in hist_b)
    print(f"  Final: acc={final_b['accuracy']:.1%} top3={final_b['top3_accuracy']:.1%} (best={best_b:.1%})")

    # Step 4: Compare
    diff = best_b - best_a
    winner = "B:ACTION-VALUE" if diff > 0.005 else "A:POLICY-ONLY" if diff < -0.005 else "TIE"
    total_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f" COMPARISON ({len(eval_data)} eval positions from real games)")
    print(f"{'='*60}")
    print(f"  A (policy+value): best={best_a:.1%} final={final_a['accuracy']:.1%} top3={final_a['top3_accuracy']:.1%}")
    print(f"  B (AV+p+v):       best={best_b:.1%} final={final_b['accuracy']:.1%} top3={final_b['top3_accuracy']:.1%}")
    print(f"  Winner: {winner} (delta: {diff:+.1%})")
    print(f"  Time: {total_time:.0f}s")

    results = {
        "config": config,
        "data": {"train": len(train_data), "eval": len(eval_data), "avg_legal": round(avg_moves, 1),
                 "source": "HF game-play + SF all-move labels"},
        "variant_a": {"best": best_a, "final": final_a, "history": hist_a},
        "variant_b": {"best": best_b, "final": final_b, "history": hist_b},
        "comparison": {"winner": winner, "delta": round(diff, 4)},
        "total_time_s": round(total_time, 1),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
