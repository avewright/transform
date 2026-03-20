"""train_action_value.py — Main action-value trainer.

Consumes cached Stockfish-labeled JSONL from label_positions.py.
Trains on Q(s,a) for ALL legal moves (not just best-move cross-entropy).

Loss = action_value_MSE + 0.3 * policy_CE + 0.5 * value_WDL_CE

Usage:
  python train_action_value.py --data data/sf_labels_5k_d8.jsonl
  python train_action_value.py --data data/sf_labels_50k_d8.jsonl --epochs 5 --batch 64

References:
  Ruoss et al., "Grandmaster-Level Chess Without Search" (2024)
  — action-value prediction as core training signal to reach 2895 Elo WITHOUT search
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, index_to_move, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/train_action_value")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Defaults
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
ENCODER_DIM = 256
SEED = 42
EVAL_FRAC = 0.05  # 5% held out for eval
NUM_GAMES = 4
GAME_SF_DEPTH = 3


# === Action-Value Head ===

class ActionValueHead(nn.Module):
    """Predicts Q(s,a) ∈ [0,1] for each move in the vocabulary."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


# === Data Loading ===

def cp_to_q(cp: int, eval_type: str = "cp") -> float:
    """Convert centipawn to win probability Q ∈ [0,1] via sigmoid."""
    if eval_type == "mate":
        if cp > 0:
            return 1.0 - 0.001 * min(abs(cp), 50)
        elif cp < 0:
            return 0.001 * min(abs(cp), 50)
        return 0.5
    k = 1.0 / 111.7
    return 1.0 / (1.0 + math.exp(-k * cp))


def load_labeled_data(path: Path) -> list[dict]:
    """Load JSONL from label_positions.py."""
    data = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            # Only keep entries where best_uci is in our move vocab
            if entry["best_uci"] in UCI_TO_IDX:
                data.append(entry)
    return data


def split_data(data: list[dict], eval_frac: float, seed: int) -> tuple[list, list]:
    """Split by game source diversity — shuffle then split."""
    rng = random.Random(seed)
    rng.shuffle(data)
    n_eval = max(int(len(data) * eval_frac), 50)
    return data[n_eval:], data[:n_eval]


# === Batch Construction ===

def make_av_batches(data: list[dict], batch_size: int, device: torch.device):
    """Build batches with Q-targets for all legal moves + policy targets + value targets."""
    random.shuffle(data)
    batches = []

    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [chess.Board(d["fen"]) for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)

        q_targets = torch.zeros(len(chunk), VOCAB_SIZE, device=device)
        q_mask = torch.zeros(len(chunk), VOCAB_SIZE, dtype=torch.bool, device=device)
        best_move_idx = torch.zeros(len(chunk), dtype=torch.long, device=device)
        value_targets = torch.zeros(len(chunk), dtype=torch.long, device=device)

        for j, d in enumerate(chunk):
            for mv in d["move_values"]:
                uci = mv["uci"]
                if uci in UCI_TO_IDX:
                    idx = UCI_TO_IDX[uci]
                    q_targets[j, idx] = cp_to_q(mv["cp"], mv["type"])
                    q_mask[j, idx] = True

            if d["best_uci"] in UCI_TO_IDX:
                best_move_idx[j] = UCI_TO_IDX[d["best_uci"]]

            # Value target from WDL: argmax → 0=win, 1=draw, 2=loss
            wdl = d.get("wdl", [0.33, 0.34, 0.33])
            value_targets[j] = max(range(3), key=lambda x: wdl[x])

        batches.append((batch_input, q_targets, q_mask, best_move_idx, value_targets))

    return batches


# === Evaluation ===

def evaluate_accuracy(chess_model, eval_data, device, n=None, batch_size=64):
    """Batched policy accuracy evaluation."""
    chess_model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = top3_correct = total = 0

    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            chunk = subset[i:i + batch_size]
            boards = [chess.Board(d["fen"]) for d in chunk]
            targets = [UCI_TO_IDX.get(d["best_uci"], 0) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = chess_model(batch_input)
            logits = result["policy_logits"]

            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")

            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()

            for j, target_idx in enumerate(targets):
                total += 1
                if preds[j] == target_idx:
                    correct += 1
                if target_idx in top3s[j]:
                    top3_correct += 1

    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def play_game_vs_stockfish(chess_model, sf_depth, model_color, device, max_moves=100):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth, parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    chess_model.eval()
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            pred, _ = chess_model.predict_move(board)
            if pred not in board.legal_moves:
                pred = random.choice(list(board.legal_moves))
            board.push(pred)
        else:
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None:
                break
            board.push(chess.Move.from_uci(sf_uci))
    result = board.result()
    winner = "white" if result == "1-0" else "black" if result == "0-1" else "draw"
    model_result = (
        "win" if (winner == "white" and model_color == chess.WHITE) or
                 (winner == "black" and model_color == chess.BLACK)
        else "loss" if winner != "draw" else "draw"
    )
    term = board.outcome().termination.name if board.outcome() else "max_moves"
    return {
        "model_color": "white" if model_color else "black",
        "result": result, "model_result": model_result,
        "moves": board.fullmove_number, "termination": term,
    }


# === Training ===

def train(chess_model, av_head, train_data, eval_data, device, epochs, batch_size, lr):
    """Train with combined action-value + policy + value loss."""

    # Param groups: encoder/heads at LR, AV head at LR
    all_params = (
        list(chess_model.encoder.parameters()) +
        list(chess_model.input_proj.parameters()) +
        list(chess_model.policy_head.parameters()) +
        list(chess_model.value_head.parameters()) +
        list(av_head.parameters())
    )
    optimizer = AdamW(all_params, lr=lr, weight_decay=0.01)
    total_steps = epochs * (len(train_data) // batch_size + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    t0 = time.time()

    for epoch in range(epochs):
        chess_model.train()
        av_head.train()
        batches = make_av_batches(train_data, batch_size, device)
        ep_av_loss = ep_p_loss = ep_v_loss = 0.0
        steps = 0

        for batch_input, q_targets, q_mask, best_move_idx, value_targets in batches:
            # Forward through encoder → backbone
            tokens = chess_model.encoder(batch_input)
            embeds = chess_model.input_proj(tokens)
            backbone_dtype = next(chess_model.backbone.parameters()).dtype
            embeds = embeds.to(backbone_dtype)
            outputs = chess_model.backbone(inputs_embeds=embeds, use_cache=False)
            hidden = outputs.last_hidden_state.float()
            global_hidden = hidden[:, 0, :]

            # Policy logits
            policy_logits = chess_model.policy_head(global_hidden)

            # Value logits
            value_logits = chess_model.value_head(global_hidden)

            # Action-value Q(s,a)
            q_pred = av_head(global_hidden)

            # Losses
            av_loss = ((q_pred - q_targets) ** 2 * q_mask.float()).sum() / q_mask.float().sum()
            policy_loss = F.cross_entropy(policy_logits, best_move_idx)
            value_loss = F.cross_entropy(value_logits, value_targets)

            loss = av_loss + 0.3 * policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            ep_av_loss += av_loss.item()
            ep_p_loss += policy_loss.item()
            ep_v_loss += value_loss.item()
            steps += 1

        avg_av = ep_av_loss / max(steps, 1)
        avg_p = ep_p_loss / max(steps, 1)
        avg_v = ep_v_loss / max(steps, 1)

        ev = evaluate_accuracy(chess_model, eval_data, device, n=min(300, len(eval_data)))
        elapsed = time.time() - t0
        history.append({
            **ev, "av_loss": avg_av, "p_loss": avg_p, "v_loss": avg_v,
            "epoch": epoch + 1,
        })
        print(f"  Epoch {epoch+1}/{epochs}: av={avg_av:.4f} p={avg_p:.4f} v={avg_v:.4f} "
              f"| acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")

    return history


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Action-value trainer")
    parser.add_argument("--data", type=str, required=True, help="Path to labeled JSONL")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--games", type=int, default=NUM_GAMES, help="Games vs Stockfish")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print(f"\n[1/5] Loading labeled data from {args.data}...")
    all_data = load_labeled_data(Path(args.data))
    train_data, eval_data = split_data(all_data, EVAL_FRAC, args.seed)

    avg_moves = sum(d["num_legal"] for d in all_data) / max(len(all_data), 1)
    total_evals = sum(d["num_legal"] for d in all_data)
    phases = {}
    for d in all_data:
        phases[d["phase"]] = phases.get(d["phase"], 0) + 1
    print(f"  Total: {len(all_data)} positions, train: {len(train_data)}, eval: {len(eval_data)}")
    print(f"  Avg legal moves: {avg_moves:.1f}, total move evals: {total_evals:,}")
    print(f"  Phases: {phases}")

    # --- Load model ---
    print(f"\n[2/5] Loading Qwen3-0.6B backbone...")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    av_head = ActionValueHead(chess_model.hidden_size, VOCAB_SIZE).to(device)

    trainable = chess_model.trainable_params() + sum(p.numel() for p in av_head.parameters())
    print(f"  Trainable: {trainable:,} (model) + AV head")

    # Pre-train eval
    pre_eval = evaluate_accuracy(chess_model, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre_eval['accuracy']:.1%} top3={pre_eval['top3_accuracy']:.1%}")

    # --- Train ---
    print(f"\n[3/5] Training ({len(train_data)} positions, {args.epochs} epochs, batch={args.batch})...")
    history = train(chess_model, av_head, train_data, eval_data, device,
                    epochs=args.epochs, batch_size=args.batch, lr=args.lr)

    # --- Final eval ---
    print(f"\n[4/5] Final evaluation...")
    final_eval = evaluate_accuracy(chess_model, eval_data, device)
    print(f"  Final: acc={final_eval['accuracy']:.1%} top3={final_eval['top3_accuracy']:.1%}")
    best_acc = max(h["accuracy"] for h in history)
    best_top3 = max(h["top3_accuracy"] for h in history)
    print(f"  Best:  acc={best_acc:.1%} top3={best_top3:.1%}")

    # --- Play games ---
    if args.games > 0:
        print(f"\n[5/5] Playing {args.games} games vs Stockfish depth={GAME_SF_DEPTH}...")
        game_results = []
        for g in range(args.games):
            color = chess.WHITE if g % 2 == 0 else chess.BLACK
            r = play_game_vs_stockfish(chess_model, GAME_SF_DEPTH, color, device)
            game_results.append(r)
            sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
            print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']} moves ({r['termination']})")

        wins = sum(1 for r in game_results if r["model_result"] == "win")
        draws = sum(1 for r in game_results if r["model_result"] == "draw")
        losses = sum(1 for r in game_results if r["model_result"] == "loss")
        score = wins + 0.5 * draws
        print(f"  Score: {score}/{args.games} (W{wins}/D{draws}/L{losses})")
    else:
        game_results = []
        wins = draws = losses = 0
        score = 0

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "data_path": args.data,
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "avg_legal_moves": round(avg_moves, 1),
        "total_move_evals": total_evals,
        "phases": phases,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "trainable_params": trainable,
        "pre_eval": pre_eval,
        "final_eval": final_eval,
        "best_accuracy": best_acc,
        "best_top3": best_top3,
        "history": history,
        "games": game_results,
        "game_score": {"wins": wins, "draws": draws, "losses": losses, "score": score},
        "total_time_s": round(total_time, 1),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
