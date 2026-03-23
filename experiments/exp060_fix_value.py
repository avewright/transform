"""exp060: Fix value head for exp059 checkpoint.

Hypothesis: The exp059 model has excellent policy (47.2%) but broken value-reranking
(12.5% vs old 37.5%) because the value head was trained on noisy synthetic WDL targets.
Fine-tuning ONLY the value head on HF data (real game-outcome WDL) while freezing
the policy/encoder should restore strong reranking without hurting policy accuracy.

Plan:
  1. Load exp059 best checkpoint (47.2% policy)
  2. Freeze everything except value head
  3. Fine-tune value head on 47.5K HF positions (real WDL labels)
  4. Evaluate policy accuracy (should stay ~47.2%) + gameplay

Experiment contract:
  - Primary metric: gameplay score at SF d1-d3 with value_rerank_k5
  - Baseline: exp059 value_rerank_k5 = 12.5% at SF d1
  - Target: restore to exp055-level reranking (37.5%) with exp059-level policy
  - Runtime: ~5 minutes (small data, frozen encoder)
"""

import json
import math
import os
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
from chess_model import LearnedBoardEncoder
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)

OUTPUT_DIR = Path("outputs/exp060_fix_value")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
CHECKPOINT = Path("outputs/exp059_data_scaling/best_checkpoint.pt")

# Model config (must match exp059)
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1
HEAD_DIM = 256

# Training config
VALUE_EPOCHS = 10
VALUE_LR = 5e-4
VALUE_BATCH_SIZE = 128
SEED = 42

# Game config
SF_GAME_DEPTHS = [1, 2, 3]
GAMES_PER_DEPTH = 8
OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4"],
    ["e2e4", "e7e6"],
]


# ── Model (same as exp059) ──

def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    for i in range(VOCAB_SIZE):
        uci = IDX_TO_UCI[i]
        from_sqs.append(chess.parse_square(uci[:2]))
        to_sqs.append(chess.parse_square(uci[2:4]))
        promo_types.append(promo_map.get(uci[4:5], 0))
    return (
        torch.tensor(from_sqs, dtype=torch.long),
        torch.tensor(to_sqs, dtype=torch.long),
        torch.tensor(promo_types, dtype=torch.long),
    )


class SpatialPolicyHead(nn.Module):
    def __init__(self, hidden_size, n_ctx_tokens=4, head_dim=256):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)

    def forward(self, hidden_states, cls_hidden):
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(cls_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


class ChessTransformerV2(nn.Module):
    def __init__(self, encoder_dim=256, hidden_dim=512,
                 num_layers=8, num_heads=8, dropout=0.1, head_dim=256):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = SpatialPolicyHead(
            hidden_dim, n_ctx_tokens=4, head_dim=head_dim,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 3),
        )
        self.hidden_dim = hidden_dim

    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        B = hidden.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]
        policy_logits = self.policy_head(hidden, cls_hidden)
        value_logits = self.value_head(cls_hidden)
        return {
            "policy_logits": policy_logits,
            "value_logits": value_logits,
            "cls_hidden": cls_hidden,
        }

    @torch.no_grad()
    def get_policy_topk(self, board, device, k=10):
        self.eval()
        board_input = batch_boards_to_token_ids([board], device)
        result = self(board_input)
        logits = result["policy_logits"][0]
        mask = legal_move_mask(board).to(device)
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        topk = probs.topk(k)
        moves = []
        for idx, p in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()):
            m = index_to_move(idx)
            if m is not None and m in board.legal_moves:
                moves.append((m, p))
        return moves

    @torch.no_grad()
    def get_values_batch(self, boards, device):
        self.eval()
        board_input = batch_boards_to_token_ids(boards, device)
        result = self(board_input)
        wdl = F.softmax(result["value_logits"], dim=-1)
        return (wdl[:, 0] - wdl[:, 2]).cpu().tolist()


# ── Strategies ──

def strategy_policy_argmax(model, board, device, **kw):
    model.eval()
    with torch.no_grad():
        board_input = batch_boards_to_token_ids([board], device)
        result = model(board_input)
        logits = result["policy_logits"][0]
        mask = legal_move_mask(board).to(device)
        logits[~mask] = float("-inf")
        idx = logits.argmax().item()
    return IDX_TO_UCI[idx]


def strategy_value_rerank_k5(model, board, device, **kw):
    candidates = model.get_policy_topk(board, device, k=5)
    if not candidates:
        return strategy_policy_argmax(model, board, device)
    child_boards = []
    moves = []
    for m, p in candidates:
        child = board.copy()
        child.push(m)
        child_boards.append(child)
        moves.append(m)
    child_values = model.get_values_batch(child_boards, device)
    our_values = [-v for v in child_values]
    best_idx = max(range(len(our_values)), key=lambda i: our_values[i])
    return moves[best_idx].uci()


def strategy_value_rerank_k10(model, board, device, **kw):
    candidates = model.get_policy_topk(board, device, k=10)
    if not candidates:
        return strategy_policy_argmax(model, board, device)
    child_boards = []
    moves = []
    for m, p in candidates:
        child = board.copy()
        child.push(m)
        child_boards.append(child)
        moves.append(m)
    child_values = model.get_values_batch(child_boards, device)
    our_values = [-v for v in child_values]
    best_idx = max(range(len(our_values)), key=lambda i: our_values[i])
    return moves[best_idx].uci()


# ── Game playing ──

def play_game(model, device, strategy_fn, sf_depth, opening_moves):
    from stockfish import Stockfish
    sf = Stockfish(SF_PATH, depth=sf_depth)
    results = []
    for model_color in [chess.WHITE, chess.BLACK]:
        board = chess.Board()
        move_list = []
        for uci in opening_moves:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)
                move_list.append(uci)
        while not board.is_game_over() and len(move_list) < 200:
            if board.turn == model_color:
                move_uci = strategy_fn(model, board, device)
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    move = list(board.legal_moves)[0]
                    move_uci = move.uci()
            else:
                sf.set_fen_position(board.fen())
                move_uci = sf.get_best_move()
                move = chess.Move.from_uci(move_uci)
            board.push(move)
            move_list.append(move_uci)
        result = board.result()
        if result == "1-0":
            outcome = 1.0 if model_color == chess.WHITE else 0.0
        elif result == "0-1":
            outcome = 0.0 if model_color == chess.WHITE else 1.0
        elif result == "1/2-1/2":
            outcome = 0.5
        else:
            outcome = 0.5
        results.append({
            "model_color": "white" if model_color == chess.WHITE else "black",
            "outcome": outcome, "result": result,
            "num_moves": len(move_list),
        })
    return results


def run_games(model, device):
    strategies = {
        "policy_argmax": strategy_policy_argmax,
        "value_rerank_k5": strategy_value_rerank_k5,
        "value_rerank_k10": strategy_value_rerank_k10,
    }
    all_results = {}
    for sname, sfn in strategies.items():
        print(f"    {sname}:")
        strat_results = {}
        for sf_depth in SF_GAME_DEPTHS:
            wins = draws = losses = total_moves = 0
            for g in range(GAMES_PER_DEPTH // 2):
                opening = OPENINGS[g % len(OPENINGS)]
                game_results = play_game(model, device, sfn, sf_depth, opening)
                for gr in game_results:
                    total_moves += gr["num_moves"]
                    if gr["outcome"] == 1.0: wins += 1
                    elif gr["outcome"] == 0.5: draws += 1
                    else: losses += 1
            n_games = max(wins + draws + losses, 1)
            score = (wins + 0.5 * draws) / n_games
            strat_results[f"d{sf_depth}"] = {
                "wins": wins, "draws": draws, "losses": losses,
                "score": round(score, 3),
                "avg_moves": round(total_moves / n_games),
            }
            print(f"      d{sf_depth}: W{wins}/D{draws}/L{losses} "
                  f"({score:.1%}, avg {total_moves//n_games}mv)")
        all_results[sname] = strat_results
    return all_results


# ── Evaluation ──

def evaluate_policy(model, eval_data, device, batch_size=128):
    model.eval()
    correct = top3_correct = total = 0
    val_correct = val_total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()
            # Value accuracy
            wdl_pred = F.softmax(result["value_logits"], dim=-1)
            for j, d in enumerate(chunk):
                wdl = d.get("wdl", None)
                if wdl:
                    val_total += 1
                    pred_class = wdl_pred[j].argmax().item()
                    true_class = max(range(3), key=lambda k: wdl[k])
                    if pred_class == true_class:
                        val_correct += 1
            for j, t in enumerate(targets):
                total += 1
                if preds[j] == t: correct += 1
                if t in top3s[j]: top3_correct += 1
    return {
        "accuracy": correct / max(total, 1),
        "top3": top3_correct / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
        "n": total,
    }


# ── Main ──

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {device}")
    print(f"Experiment: exp060_fix_value")
    print(f"Hypothesis: Fine-tune value head on real WDL → restore reranking")
    print()

    # Load model
    print("[1/4] Loading exp059 checkpoint...")
    model = ChessTransformerV2(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        dropout=DROPOUT, head_dim=HEAD_DIM,
    ).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {CHECKPOINT} ({n_params:,} params)")

    # Load HF data
    print("\n[2/4] Loading HF data...")
    from hf_data import load_training_set, load_eval_set
    hf_train = load_training_set()
    hf_eval = load_eval_set(n=2500)
    train_data = [{
        "board": d["board"], "move": d["move"],
        "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
    } for d in hf_train]
    eval_data = [{
        "board": d["board"], "move": d["move"],
        "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
    } for d in hf_eval]
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Baseline evaluation
    print("\n  Baseline (exp059 checkpoint):")
    baseline = evaluate_policy(model, eval_data, device)
    print(f"    Policy acc: {baseline['accuracy']:.1%}, "
          f"Top-3: {baseline['top3']:.1%}, "
          f"Value acc: {baseline['value_accuracy']:.1%}")

    # Freeze everything except value head
    print("\n[3/4] Fine-tuning value head only...")
    for name, param in model.named_parameters():
        if "value_head" not in name:
            param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,} (value head only)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=VALUE_LR, weight_decay=0.01,
    )

    best_val_acc = baseline["value_accuracy"]
    best_state = None

    for epoch in range(VALUE_EPOCHS):
        t0 = time.time()
        model.train()
        # But keep encoder/transformer in eval mode for batch norm etc
        model.encoder.eval()
        model.transformer.eval()
        random.shuffle(train_data)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), VALUE_BATCH_SIZE):
            chunk = train_data[i:i + VALUE_BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            wdl_targets = torch.tensor(
                [d["wdl"] for d in chunk], device=device, dtype=torch.float32,
            )
            batch_input = batch_boards_to_token_ids(boards, device)

            with torch.no_grad():
                tokens = model.encoder(batch_input)
                hidden = model.input_proj(tokens)
                B = hidden.shape[0]
                cls = model.cls_token.expand(B, -1, -1)
                hidden = torch.cat([cls, hidden], dim=1)
                hidden = hidden + model.pos_embed
                hidden = model.transformer(hidden)
                hidden = model.norm(hidden)
                cls_hidden = hidden[:, 0, :]

            # Only value head gets gradients
            value_logits = model.value_head(cls_hidden)
            value_log_probs = F.log_softmax(value_logits, dim=-1)
            loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ev = evaluate_policy(model, eval_data, device)
        ep_time = time.time() - t0

        marker = " *" if ev["value_accuracy"] > best_val_acc else ""
        print(f"  Ep{epoch+1}: vl={avg_loss:.4f} "
              f"policy={ev['accuracy']:.1%} top3={ev['top3']:.1%} "
              f"val_acc={ev['value_accuracy']:.1%} [{ep_time:.0f}s]{marker}")

        if ev["value_accuracy"] > best_val_acc:
            best_val_acc = ev["value_accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        ckpt_path = OUTPUT_DIR / "best_checkpoint.pt"
        torch.save(best_state, ckpt_path)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Saved: {ckpt_path} (best value acc: {best_val_acc:.1%})")

    # Unfreeze for inference
    for param in model.parameters():
        param.requires_grad = False

    # Play games
    print(f"\n[4/4] Playing games vs Stockfish...")
    game_results = run_games(model, device)

    total_time = time.time() - t_start

    # Results
    results = {
        "experiment": "exp060_fix_value",
        "hypothesis": "Fine-tune value head on real WDL restores reranking",
        "baseline_checkpoint": str(CHECKPOINT),
        "baseline_policy_acc": baseline["accuracy"],
        "baseline_value_acc": baseline["value_accuracy"],
        "final_value_acc": best_val_acc,
        "games": game_results,
        "total_time_s": round(total_time),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" RESULTS: exp060_fix_value")
    print(f"{'='*60}")
    print(f"  Policy acc: {baseline['accuracy']:.1%} (unchanged)")
    print(f"  Value acc: {baseline['value_accuracy']:.1%} → {best_val_acc:.1%}")
    print(f"\n  Gameplay vs Stockfish:")
    for sname, sdata in game_results.items():
        print(f"    {sname}:")
        for dk, dv in sdata.items():
            print(f"      {dk}: W{dv['wins']}/D{dv['draws']}/L{dv['losses']} "
                  f"({dv['score']:.1%})")
    print(f"\n  Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
