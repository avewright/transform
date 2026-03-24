"""exp065_quick: Fast baseline — 8L Medium on 237.5K combined HF data.

Hypothesis: Training 8L/512d on combined chess-positions (47.5K) + chess-positions-sf-200k (190K)
= 237.5K total will establish a strong baseline on this 24GB GPU.

Experiment contract:
  - Primary metric: top-1 accuracy on HF chess-positions test (2500 positions)
  - Seed: 42
  - Epochs: 3 (with 237K data, 3 epochs = ~711K gradient steps at bs=256)
  - Batch: 256 (fits 24GB easily)
  - Target: >45% accuracy in <10 minutes
  - Device: RTX A5000 (24GB)
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
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)

OUTPUT_DIR = Path("outputs/exp065_quick")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Model config — Medium
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1
HEAD_DIM = 256

# Training config — optimized for 24GB
EPOCHS = 3
BATCH_SIZE = 256
LR = 3e-4
WARMUP_FRAC = 0.05
VALUE_WEIGHT = 0.5
SOFT_TEMP = 100.0
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


# ── Model (same as exp062) ──

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
    def __init__(self, encoder_dim=256, hidden_dim=512, num_layers=8,
                 num_heads=8, dropout=0.1, head_dim=256):
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = SpatialPolicyHead(hidden_dim, n_ctx_tokens=4, head_dim=head_dim)
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
        return {"policy_logits": policy_logits, "value_logits": value_logits}

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


# ── Data ──

def build_soft_target(top_moves_json, board, temperature=100.0):
    """Build soft target from SF top-k CP scores."""
    import json as _json
    target = torch.zeros(VOCAB_SIZE)
    try:
        top_moves = _json.loads(top_moves_json) if isinstance(top_moves_json, str) else top_moves_json
    except Exception:
        return None
    valid_moves = []
    for m in top_moves:
        uci = m.get("uci", m.get("Move", ""))
        if uci not in UCI_TO_IDX:
            continue
        move_obj = chess.Move.from_uci(uci)
        if move_obj not in board.legal_moves:
            continue
        cp = m.get("cp", 0)
        if m.get("mate") is not None:
            cp = 10000 if m["mate"] > 0 else -10000
        valid_moves.append((UCI_TO_IDX[uci], cp))
    if not valid_moves:
        return None
    cps = torch.tensor([cp for _, cp in valid_moves], dtype=torch.float32)
    probs = F.softmax(cps / temperature, dim=0)
    for (idx, _), p in zip(valid_moves, probs):
        target[idx] = p.item()
    return target


def load_combined_data():
    """Load chess-positions + chess-positions-sf-200k from HF, combine, dedup."""
    from datasets import load_dataset

    print("  Loading HF datasets...")
    t0 = time.time()

    # Dataset 1: original chess-positions (47.5K train, 2.5K test)
    ds1_train = load_dataset("avewright/chess-positions", split="train")
    ds1_test = load_dataset("avewright/chess-positions", split="test")

    # Dataset 2: SF-labeled 200K (190K train, 10K test)
    ds2_train = load_dataset("avewright/chess-positions-sf-200k", split="train")

    print(f"  HF loaded: ds1={len(ds1_train)}+{len(ds1_test)}, ds2={len(ds2_train)} ({time.time()-t0:.1f}s)")

    seen = set()
    train_data = []
    eval_data = []

    # Process ds1 train
    for row in ds1_train:
        try:
            fen = row["fen"]
            if fen in seen:
                continue
            seen.add(fen)
            board = chess.Board(fen)
            move = chess.Move.from_uci(row["best_move"])
            if row["best_move"] not in UCI_TO_IDX or move not in board.legal_moves:
                continue
            soft = None
            if row.get("top_moves"):
                soft = build_soft_target(row["top_moves"], board, SOFT_TEMP)
            train_data.append({
                "board": board, "move": move,
                "wdl": (row["wdl_win"], row["wdl_draw"], row["wdl_loss"]),
                "phase": row.get("phase", "unknown"),
                "soft_target": soft,
            })
        except Exception:
            continue

    # Process ds2 train
    for row in ds2_train:
        try:
            fen = row["fen"]
            if fen in seen:
                continue
            seen.add(fen)
            board = chess.Board(fen)
            move = chess.Move.from_uci(row["best_move"])
            if row["best_move"] not in UCI_TO_IDX or move not in board.legal_moves:
                continue
            soft = None
            if row.get("top_moves_json"):
                soft = build_soft_target(row["top_moves_json"], board, SOFT_TEMP)
            train_data.append({
                "board": board, "move": move,
                "wdl": (row["wdl_win"], row["wdl_draw"], row["wdl_loss"]),
                "phase": row.get("phase", "unknown"),
                "soft_target": soft,
            })
        except Exception:
            continue

    # Eval from ds1 test
    for row in ds1_test:
        try:
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(row["best_move"])
            if row["best_move"] not in UCI_TO_IDX or move not in board.legal_moves:
                continue
            eval_data.append({
                "board": board, "move": move,
                "wdl": (row["wdl_win"], row["wdl_draw"], row["wdl_loss"]),
                "phase": row.get("phase", "unknown"),
            })
        except Exception:
            continue

    n_soft = sum(1 for d in train_data if d["soft_target"] is not None)
    print(f"  Combined: {len(train_data):,} train, {len(eval_data):,} eval")
    print(f"  Soft targets: {n_soft:,}/{len(train_data):,} ({100*n_soft/max(len(train_data),1):.0f}%)")
    return train_data, eval_data


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
    print(f"  Model: {NUM_LAYERS}L/{HIDDEN_DIM}d/{NUM_HEADS}h, {n_params:,} params")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = len(train_data) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(int(total_steps * WARMUP_FRAC), 1)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler()

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        random.shuffle(train_data)
        total_pl = total_vl = 0.0
        n_batches = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            if len(chunk) < 2:
                continue

            boards = [d["board"] for d in chunk]
            hard_targets = torch.tensor([move_to_index(d["move"]) for d in chunk], device=device)
            wdl_targets = torch.tensor([d["wdl"] for d in chunk], device=device, dtype=torch.float32)

            has_soft = [d["soft_target"] is not None for d in chunk]
            soft_targets = None
            if any(has_soft):
                soft_list = []
                for d in chunk:
                    if d["soft_target"] is not None:
                        soft_list.append(d["soft_target"])
                    else:
                        onehot = torch.zeros(VOCAB_SIZE)
                        onehot[move_to_index(d["move"])] = 1.0
                        soft_list.append(onehot)
                soft_targets = torch.stack(soft_list).to(device)

            batch_input = batch_boards_to_token_ids(boards, device)

            with autocast(dtype=torch.float16):
                result = model(batch_input)
                if soft_targets is not None:
                    log_probs = F.log_softmax(result["policy_logits"], dim=-1)
                    policy_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean")
                else:
                    policy_loss = F.cross_entropy(result["policy_logits"], hard_targets)
                value_log_probs = F.log_softmax(result["value_logits"], dim=-1)
                value_loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")
                loss = policy_loss + VALUE_WEIGHT * value_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            total_pl += policy_loss.item()
            total_vl += value_loss.item()
            n_batches += 1

        avg_pl = total_pl / max(n_batches, 1)
        avg_vl = total_vl / max(n_batches, 1)
        ev = evaluate(model, eval_data, device)
        ep_time = time.time() - t0

        history.append({
            "epoch": epoch + 1, "policy_loss": round(avg_pl, 4),
            "value_loss": round(avg_vl, 4),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()},
            "time_s": round(ep_time),
        })

        marker = " *" if ev["accuracy"] > best_acc else ""
        print(f"  Ep{epoch+1}: pl={avg_pl:.3f} vl={avg_vl:.3f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} "
              f"val_acc={ev.get('value_accuracy',0):.1%} "
              f"sf_rank={ev['mean_sf_rank']:.1f} [{ep_time:.0f}s]{marker}", flush=True)

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        ckpt_path = OUTPUT_DIR / "best_checkpoint.pt"
        torch.save(best_state, ckpt_path)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Saved: {ckpt_path} (best acc: {best_acc:.1%})")

    return model, history, best_acc


def evaluate(model, eval_data, device, batch_size=256):
    model.eval()
    correct = top3_correct = total = 0
    entropy_sum = sf_rank_sum = 0.0
    val_correct = val_total = 0
    phase_stats = {}

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]

            for j, d in enumerate(chunk):
                l = logits[j].clone()
                mask = legal_move_mask(d["board"]).to(device)
                l[~mask] = float("-inf")
                probs = F.softmax(l, dim=-1)
                pred_idx = l.argmax().item()
                true_idx = move_to_index(d["move"])
                if pred_idx == true_idx:
                    correct += 1
                if true_idx in l.topk(3).indices.tolist():
                    top3_correct += 1
                p = probs[probs > 0]
                entropy_sum += -(p * p.log()).sum().item()
                sorted_i = l.argsort(descending=True).tolist()
                sf_rank_sum += sorted_i.index(true_idx) + 1 if true_idx in sorted_i else len(sorted_i)
                total += 1
                phase = d.get("phase", "unknown")
                if phase not in phase_stats:
                    phase_stats[phase] = {"correct": 0, "total": 0}
                phase_stats[phase]["total"] += 1
                if pred_idx == true_idx:
                    phase_stats[phase]["correct"] += 1

            wdl_logits = result["value_logits"]
            for j, d in enumerate(chunk):
                if "wdl" not in d:
                    continue
                pred_class = wdl_logits[j].argmax().item()
                true_class = max(range(3), key=lambda k: d["wdl"][k])
                if pred_class == true_class:
                    val_correct += 1
                val_total += 1

    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
        "phase_accuracy": {p: round(s["correct"]/max(s["total"],1), 4) for p, s in phase_stats.items()},
        "n_eval": total,
    }


# ── Game play ──

def strategy_policy_argmax(model, board, device):
    moves = model.get_policy_topk(board, device, k=1)
    return moves[0][0] if moves else random.choice(list(board.legal_moves))

def strategy_value_rerank_k5(model, board, device):
    top_moves = model.get_policy_topk(board, device, k=5)
    if not top_moves:
        return random.choice(list(board.legal_moves))
    boards2 = []
    for m, _ in top_moves:
        b = board.copy(); b.push(m); boards2.append(b)
    values = model.get_values_batch(boards2, device)
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    scored = sorted([(m, sign*v) for (m,_), v in zip(top_moves, values)], key=lambda x: -x[1])
    return scored[0][0]

def play_game(model, device, strategy_fn, sf_depth, opening_moves=None):
    from stockfish import Stockfish
    sf = Stockfish(path=SF_PATH, depth=sf_depth, parameters={"Threads": 1, "Hash": 16})
    results = []
    for model_color in [chess.WHITE, chess.BLACK]:
        board = chess.Board()
        move_list = []
        if opening_moves:
            for uci in opening_moves:
                board.push(chess.Move.from_uci(uci))
                move_list.append(uci)
        while not board.is_game_over() and len(move_list) < 200:
            if board.turn == model_color:
                move = strategy_fn(model, board, device)
            else:
                sf.set_fen_position(board.fen())
                move = chess.Move.from_uci(sf.get_best_move())
            board.push(move)
            move_list.append(move.uci())
        result = board.result()
        if result == "1-0":
            outcome = 1.0 if model_color == chess.WHITE else 0.0
        elif result == "0-1":
            outcome = 0.0 if model_color == chess.WHITE else 1.0
        else:
            outcome = 0.5
        results.append({
            "model_color": "white" if model_color == chess.WHITE else "black",
            "outcome": outcome, "result": result, "num_moves": len(move_list),
            "termination": board.outcome().termination.name if board.outcome() else "max_moves",
        })
    return results

def run_games(model, device):
    strategies = {"policy_argmax": strategy_policy_argmax, "value_rerank_k5": strategy_value_rerank_k5}
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
            n = max(wins+draws+losses, 1)
            score = (wins + 0.5*draws) / n
            strat_results[f"d{sf_depth}"] = {"wins": wins, "draws": draws, "losses": losses,
                                              "score": round(score, 3), "avg_moves": round(total_moves/n)}
            print(f"      d{sf_depth}: W{wins}/D{draws}/L{losses} ({score:.1%}, {total_moves//n}mv)")
        all_results[sname] = strat_results
    return all_results


# ── Main ──

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); torch.manual_seed(SEED)

    print(f"{'='*60}")
    print(f"exp065_quick: Fast baseline — 8L Medium on 237K HF data")
    print(f"{'='*60}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Model: {NUM_LAYERS}L/{HIDDEN_DIM}d/{NUM_HEADS}h, Batch: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}, LR: {LR}, Soft temp: {SOFT_TEMP}")
    print()

    # Load data
    print("[1/3] Loading data...")
    train_data, eval_data = load_combined_data()

    # Train
    print(f"\n[2/3] Training ({EPOCHS} epochs, {len(train_data):,} positions)...")
    model, history, best_acc = train_model(train_data, eval_data, device)

    # Games
    print(f"\n[3/3] Playing games vs Stockfish...")
    game_results = run_games(model, device)

    total_time = time.time() - t_start

    results = {
        "experiment": "exp065_quick",
        "hypothesis": "8L Medium on 237K combined HF data, fast baseline on 24GB GPU",
        "seed": SEED,
        "config": {"encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
                   "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
                   "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                   "value_weight": VALUE_WEIGHT, "soft_temp": SOFT_TEMP},
        "data": {"combined_train": len(train_data), "eval": len(eval_data)},
        "training": {"best_accuracy": round(best_acc, 4), "history": history},
        "games": game_results,
        "timing": {"total_s": round(total_time), "total_min": round(total_time/60, 1)},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"RESULTS: exp065_quick")
    print(f"  Best accuracy: {best_acc:.1%}")
    print(f"  Data: {len(train_data):,} | Time: {total_time/60:.1f}min")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")
    for sn, sr in game_results.items():
        d1 = sr.get("d1", {})
        print(f"  {sn} d1: W{d1.get('wins',0)}/D{d1.get('draws',0)}/L{d1.get('losses',0)} ({d1.get('score',0):.1%})")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
