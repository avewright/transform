"""exp061: Soft policy targets from SF top-5 move rankings.

Hypothesis: Training with soft targets (CP-weighted distribution over SF top moves)
instead of hard one-hot best-move targets will improve policy quality and generalization.

Intuition: SF's ranking of top-5 moves contains richer information than just "best move."
A position where the best move is +200cp and second is +195cp is different from one
where best is +200cp and second is -100cp. Soft targets capture this.

Plan:
  1. Load existing 200K generated JSONL (has top_moves with cp scores)
  2. Build soft targets: softmax(cp_scores / temperature) over legal top moves
  3. Train with KL-divergence loss instead of cross-entropy
  4. Compare accuracy + gameplay vs exp059 (hard targets)

Experiment contract:
  - Primary metric: top-1 accuracy on HF eval set
  - Baseline: exp059 hard targets = 47.2% accuracy
  - Runtime: ~2.5 hours (6 epochs on 247.5K, same as exp059)
  - Evaluation: same 2500 HF test positions
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

OUTPUT_DIR = Path("outputs/exp061_soft_targets")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
GENERATED_DATA = Path("outputs/exp059_data_scaling/generated_200k.jsonl")

# Model config
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1
HEAD_DIM = 256

# Training config
EPOCHS = 6
BATCH_SIZE = 128
LR = 2e-4
WARMUP_FRAC = 0.05
VALUE_WEIGHT = 0.5
SOFT_TEMP = 100.0  # Temperature for CP→prob conversion (in centipawns)
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


# ── Model ──

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


# ── Data loading with soft targets ──

def load_generated_data_soft(jsonl_path, temperature=SOFT_TEMP):
    """Load generated data with soft policy targets from SF top moves."""
    data = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            try:
                board = chess.Board(d["fen"])
                best_move = chess.Move.from_uci(d["best_move"])
                if best_move not in board.legal_moves:
                    continue
                if d["best_move"] not in UCI_TO_IDX:
                    continue

                # Build soft target distribution from top moves
                soft_target = torch.zeros(VOCAB_SIZE)
                valid_moves = []
                cp_scores = []

                for tm in d.get("top_moves", []):
                    uci = tm["uci"]
                    if uci not in UCI_TO_IDX:
                        continue
                    mv = chess.Move.from_uci(uci)
                    if mv not in board.legal_moves:
                        continue
                    # Handle mate scores
                    if "mate" in tm:
                        cp = 10000 if tm["mate"] > 0 else -10000
                    else:
                        cp = tm.get("cp", 0)
                    valid_moves.append(UCI_TO_IDX[uci])
                    cp_scores.append(cp)

                if not valid_moves:
                    continue

                # Softmax over centipawn scores / temperature
                cp_tensor = torch.tensor(cp_scores, dtype=torch.float32)
                probs = F.softmax(cp_tensor / temperature, dim=0)
                for idx, p in zip(valid_moves, probs):
                    soft_target[idx] = p.item()

                data.append({
                    "board": board,
                    "move": best_move,
                    "best_move_idx": UCI_TO_IDX[d["best_move"]],
                    "soft_target": soft_target,
                    "wdl": tuple(d["wdl"]),
                    "phase": d.get("phase", "unknown"),
                })
            except Exception:
                continue
    return data


# ── Training ──

def train_model(train_data, eval_data, device):
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = ChessTransformerV2(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        dropout=DROPOUT, head_dim=HEAD_DIM,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_data) // BATCH_SIZE) * EPOCHS
    warmup_steps = max(int(total_steps * WARMUP_FRAC), 1)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        random.shuffle(train_data)
        total_policy_loss = total_value_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]

            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)

            # Soft policy loss: KL divergence between model output and SF soft targets
            soft_targets = torch.stack([d["soft_target"] for d in chunk]).to(device)
            log_probs = F.log_softmax(result["policy_logits"], dim=-1)
            policy_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean")

            # Value loss (same as exp059)
            wdl_targets = torch.tensor(
                [d["wdl"] for d in chunk], device=device, dtype=torch.float32,
            )
            value_log_probs = F.log_softmax(result["value_logits"], dim=-1)
            value_loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")

            loss = policy_loss + VALUE_WEIGHT * value_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        avg_pl = total_policy_loss / max(n_batches, 1)
        avg_vl = total_value_loss / max(n_batches, 1)
        ev = evaluate(model, eval_data, device)
        ep_time = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "policy_loss": round(avg_pl, 4),
            "value_loss": round(avg_vl, 4),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()},
            "time_s": round(ep_time),
        })

        marker = " *" if ev["accuracy"] > best_acc else ""
        print(f"  Ep{epoch+1}: pl={avg_pl:.3f} vl={avg_vl:.3f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} "
              f"val_acc={ev.get('value_accuracy', 0):.1%} "
              f"sf_rank={ev['mean_sf_rank']:.1f} [{ep_time:.0f}s]{marker}")

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        ckpt_path = OUTPUT_DIR / "best_checkpoint.pt"
        torch.save(best_state, ckpt_path)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Saved: {ckpt_path} (best acc: {best_acc:.1%})")

    return model, history, best_acc


def evaluate(model, eval_data, device, batch_size=128):
    model.eval()
    correct = top3_correct = total = 0
    entropy_sum = sf_rank_sum = 0.0
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
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()

            if "value_logits" in result:
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
                p = probs[j]
                legal_p = p[p > 0]
                entropy_sum += -(legal_p * legal_p.log()).sum().item()
                sorted_idx = logits[j].argsort(descending=True).cpu().tolist()
                sf_rank = sorted_idx.index(t) + 1 if t in sorted_idx else len(sorted_idx)
                sf_rank_sum += sf_rank

    result = {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "n_eval": total,
    }
    if val_total > 0:
        result["value_accuracy"] = val_correct / val_total
    return result


# ── Strategies + games (same as exp059/060) ──

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
    child_boards, moves = [], []
    for m, p in candidates:
        child = board.copy()
        child.push(m)
        child_boards.append(child)
        moves.append(m)
    child_values = model.get_values_batch(child_boards, device)
    our_values = [-v for v in child_values]
    best_idx = max(range(len(our_values)), key=lambda i: our_values[i])
    return moves[best_idx].uci()


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
        else:
            outcome = 0.5
        results.append({"outcome": outcome, "num_moves": len(move_list)})
    return results


def run_games(model, device):
    strategies = {
        "policy_argmax": strategy_policy_argmax,
        "value_rerank_k5": strategy_value_rerank_k5,
    }
    all_results = {}
    for sname, sfn in strategies.items():
        print(f"    {sname}:")
        strat_results = {}
        for sf_depth in SF_GAME_DEPTHS:
            wins = draws = losses = total_moves = 0
            for g in range(GAMES_PER_DEPTH // 2):
                opening = OPENINGS[g % len(OPENINGS)]
                for gr in play_game(model, device, sfn, sf_depth, opening):
                    total_moves += gr["num_moves"]
                    if gr["outcome"] == 1.0: wins += 1
                    elif gr["outcome"] == 0.5: draws += 1
                    else: losses += 1
            n_games = max(wins + draws + losses, 1)
            score = (wins + 0.5 * draws) / n_games
            strat_results[f"d{sf_depth}"] = {
                "wins": wins, "draws": draws, "losses": losses,
                "score": round(score, 3),
            }
            print(f"      d{sf_depth}: W{wins}/D{draws}/L{losses} ({score:.1%})")
        all_results[sname] = strat_results
    return all_results


# ── Main ──

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {device}")
    print(f"Experiment: exp061_soft_targets")
    print(f"Hypothesis: Soft SF top-5 targets > hard best-move targets")
    print(f"Temperature: {SOFT_TEMP}")
    print()

    # Load generated data with soft targets
    print("[1/4] Loading generated data with soft targets...")
    t0 = time.time()
    gen_data = load_generated_data_soft(GENERATED_DATA, temperature=SOFT_TEMP)
    print(f"  Generated positions (soft): {len(gen_data):,} ({time.time()-t0:.0f}s)")

    # Load HF data (hard targets for these — no top_moves available)
    print("[2/4] Loading HF data...")
    from hf_data import load_training_set, load_eval_set
    hf_train = load_training_set()
    hf_eval = load_eval_set(n=2500)

    # HF data gets hard targets (one-hot soft target)
    hf_data = []
    for d in hf_train:
        move_idx = move_to_index(d["move"])
        soft_target = torch.zeros(VOCAB_SIZE)
        soft_target[move_idx] = 1.0
        hf_data.append({
            "board": d["board"],
            "move": d["move"],
            "best_move_idx": move_idx,
            "soft_target": soft_target,
            "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
            "phase": d.get("phase", "unknown"),
        })

    train_data = gen_data + hf_data
    print(f"  Combined train: {len(train_data):,} "
          f"({len(gen_data):,} soft + {len(hf_data):,} hard)")

    eval_data = [{
        "board": d["board"], "move": d["move"],
        "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
    } for d in hf_eval]

    # Train
    print(f"\n[3/4] Training with soft targets (temp={SOFT_TEMP})...")
    model, history, best_acc = train_model(train_data, eval_data, device)

    # Play games
    print(f"\n[4/4] Playing games vs Stockfish...")
    game_results = run_games(model, device)

    total_time = time.time() - t_start

    # Save & print
    results = {
        "experiment": "exp061_soft_targets",
        "hypothesis": "Soft SF top-5 targets > hard best-move targets",
        "baseline": "exp059 hard targets: 47.2% acc, 31.2% argmax SF d1",
        "temperature": SOFT_TEMP,
        "data": {"generated_soft": len(gen_data), "hf_hard": len(hf_data),
                 "total": len(train_data)},
        "training": {"best_accuracy": round(best_acc, 4), "history": history},
        "games": game_results,
        "total_time_s": round(total_time),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" RESULTS: exp061_soft_targets")
    print(f"{'='*60}")
    print(f"  Best accuracy: {best_acc:.1%} (baseline: 47.2%)")
    print(f"  Delta: {(best_acc - 0.472)*100:+.1f}pp")
    for sname, sdata in game_results.items():
        print(f"  {sname}:")
        for dk, dv in sdata.items():
            print(f"    {dk}: W{dv['wins']}/D{dv['draws']}/L{dv['losses']} ({dv['score']:.1%})")
    print(f"  Total: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
