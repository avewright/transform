"""exp062: Massive data scaling — train on 700K+ combined positions.

Hypothesis: Training the Medium spatial model (26M params) on ~750K combined
positions will significantly beat the 247K exp059 baseline (47.2% accuracy).

Data sources:
  1. 500K CPU-generated positions (SF depth 8, seed 123) — outputs/generated_data/batch_*.jsonl
  2. 200K exp059-generated positions (SF depth 6, seed 42) — outputs/exp059_data_scaling/generated_200k.jsonl
  3. 47.5K HF real game positions — cached avewright/chess-positions

Combined: ~747.5K training positions (deduped by FEN)

Key design choices:
  - Hard cross-entropy targets (proven in exp059, simpler than soft targets)
  - 4 epochs (more data → fewer epochs needed)
  - Same architecture as exp059 (Medium 26M params)

Experiment contract:
  - Primary metric: top-1 accuracy on HF eval split (2500 positions)
  - Baseline: exp059 = 47.2%, exp024 (old arch) = 48.7%, exp031 = 51.2%
  - Target: beat 51.2% (old arch best with extended training on 460K)
  - Runtime: ~4-5 hours on RTX 2000 Ada (16GB)
"""

import glob
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

OUTPUT_DIR = Path("outputs/exp062_massive_data")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Data paths
GENERATED_BATCHES = "outputs/generated_data/batch_*.jsonl"
EXP059_DATA = Path("outputs/exp059_data_scaling/generated_200k.jsonl")

# Model config (Medium, same as exp059)
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1
HEAD_DIM = 256

# Training config
EPOCHS = 4
BATCH_SIZE = 128
LR = 2e-4
WARMUP_FRAC = 0.05
VALUE_WEIGHT = 0.5
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
        combined = from_proj + to_proj + global_proj + promo_feats
        return self.score_proj(torch.tanh(combined)).squeeze(-1)


class ChessTransformerV2(nn.Module):
    def __init__(self, encoder_dim=256, hidden_dim=512, num_layers=8,
                 num_heads=8, dropout=0.1, head_dim=256):
        super().__init__()
        self.encoder = LearnedBoardEncoder(output_dim=encoder_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_dim) * 0.02)
        self.turn_embed = nn.Embedding(2, encoder_dim)
        self.castling_embed = nn.Embedding(16, encoder_dim)
        self.phase_embed = nn.Embedding(3, encoder_dim)
        n_tokens = 1 + 3 + 64
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, encoder_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=num_heads,
            dim_feedforward=hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(encoder_dim)
        self.policy_head = SpatialPolicyHead(encoder_dim, n_ctx_tokens=4, head_dim=head_dim)
        self.value_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, batch_input):
        sq = self.encoder(batch_input)
        bs = sq.size(0)
        cls = self.cls_token.expand(bs, -1, -1)
        boards = batch_input["board_tensor"]
        turn = self.turn_embed(batch_input["turn"].long()).unsqueeze(1)
        castling = self.castling_embed(batch_input["castling"].long()).unsqueeze(1)
        phase_raw = batch_input["phase"].long().clamp(0, 2)
        phase = self.phase_embed(phase_raw).unsqueeze(1)
        hidden = torch.cat([cls, turn, castling, phase, sq], dim=1)
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


# ── Data loading ──

def load_jsonl_data(path):
    """Load JSONL data file, return list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_training_data(all_jsonl_data, hf_data):
    """Combine JSONL data + HF data into unified training format, dedup by FEN."""
    seen_fens = set()
    combined = []

    # HF data first (highest quality — real games)
    for d in hf_data:
        fen = d["board"].fen()
        if fen in seen_fens:
            continue
        seen_fens.add(fen)
        combined.append({
            "board": d["board"],
            "move": d["move"],
            "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
            "phase": d.get("phase", "unknown"),
        })

    # Generated data
    for d in all_jsonl_data:
        try:
            fen = d["fen"]
            if fen in seen_fens:
                continue
            seen_fens.add(fen)
            board = chess.Board(fen)
            move = chess.Move.from_uci(d["best_move"])
            if move not in board.legal_moves:
                continue
            if d["best_move"] not in UCI_TO_IDX:
                continue
            combined.append({
                "board": board,
                "move": move,
                "wdl": tuple(d["wdl"]),
                "phase": d.get("phase", "unknown"),
            })
        except Exception:
            continue

    return combined


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
            targets = torch.tensor(
                [move_to_index(d["move"]) for d in chunk], device=device,
            )
            wdl_targets = torch.tensor(
                [d["wdl"] for d in chunk], device=device, dtype=torch.float32,
            )

            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)

            policy_loss = F.cross_entropy(result["policy_logits"], targets)
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
              f"sf_rank={ev['mean_sf_rank']:.1f} [{ep_time:.0f}s]{marker}",
              flush=True)

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
            true_moves = [d["move"] for d in chunk]

            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]

            for j, (board, true_move) in enumerate(zip(boards, true_moves)):
                l = logits[j].clone()
                mask = legal_move_mask(board).to(device)
                l[~mask] = float("-inf")
                probs = F.softmax(l, dim=-1)

                pred_idx = l.argmax().item()
                true_idx = move_to_index(true_move)

                if pred_idx == true_idx:
                    correct += 1
                topk = l.topk(3).indices.tolist()
                if true_idx in topk:
                    top3_correct += 1

                p = probs[probs > 0]
                entropy_sum += -(p * p.log()).sum().item()

                sorted_indices = l.argsort(descending=True).tolist()
                rank = sorted_indices.index(true_idx) + 1 if true_idx in sorted_indices else len(sorted_indices)
                sf_rank_sum += rank
                total += 1

            # Value accuracy
            if any("wdl" in d for d in chunk):
                wdl_logits = result["value_logits"]
                for j, d in enumerate(chunk):
                    if "wdl" not in d:
                        continue
                    pred_class = wdl_logits[j].argmax().item()
                    true_wdl = d["wdl"]
                    true_class = max(range(3), key=lambda k: true_wdl[k])
                    if pred_class == true_class:
                        val_correct += 1
                    val_total += 1

    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
    }


# ── Game play ──

def strategy_policy_argmax(model, board, device):
    moves = model.get_policy_topk(board, device, k=1)
    return moves[0][0] if moves else random.choice(list(board.legal_moves))


def strategy_value_rerank_k5(model, board, device):
    top_moves = model.get_policy_topk(board, device, k=5)
    if not top_moves:
        return random.choice(list(board.legal_moves))
    candidate_boards = []
    for m, _ in top_moves:
        b2 = board.copy()
        b2.push(m)
        candidate_boards.append(b2)
    values = model.get_values_batch(candidate_boards, device)
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    scored = [(m, sign * v) for (m, _), v in zip(top_moves, values)]
    scored.sort(key=lambda x: -x[1])
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
        max_moves = 150
        while not board.is_game_over() and len(move_list) < max_moves:
            if board.turn == model_color:
                move = strategy_fn(model, board, device)
            else:
                sf.set_fen_position(board.fen())
                move_uci = sf.get_best_move()
                move = chess.Move.from_uci(move_uci)
            board.push(move)
            move_list.append(move.uci())
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
            "termination": board.outcome().termination.name if board.outcome() else "max_moves",
        })
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


# ── Main ──

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {device}")
    print(f"Experiment: exp062_massive_data")
    print(f"Hypothesis: 700K+ combined data >> 247K (exp059)")
    print(f"Baseline: exp059 = 47.2%, exp031 (old arch) = 51.2%")
    print()

    # Phase 1: Load all generated data
    print("[1/4] Loading generated data...")
    t_load = time.time()

    all_jsonl = []

    # Load 500K CPU-generated batches
    batch_files = sorted(glob.glob(GENERATED_BATCHES))
    for bf in batch_files:
        data = load_jsonl_data(bf)
        print(f"  {bf}: {len(data):,} positions")
        all_jsonl.extend(data)

    # Load exp059's 200K
    if EXP059_DATA.exists():
        data = load_jsonl_data(str(EXP059_DATA))
        print(f"  {EXP059_DATA}: {len(data):,} positions")
        all_jsonl.extend(data)

    print(f"  Total JSONL loaded: {len(all_jsonl):,} ({time.time() - t_load:.0f}s)")

    # Phase 2: Load HF data and combine
    print("\n[2/4] Loading HF data and combining...")
    from hf_data import load_training_set, load_eval_set
    hf_train = load_training_set()
    hf_eval = load_eval_set(n=2500)
    print(f"  HF: {len(hf_train)} train, {len(hf_eval)} eval")

    train_data = prepare_training_data(all_jsonl, hf_train)
    eval_data = [{
        "board": d["board"], "move": d["move"],
        "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
        "phase": d.get("phase", "unknown"),
    } for d in hf_eval]
    print(f"  Combined train: {len(train_data):,} (deduped)")

    # Phase 3: Train
    print(f"\n[3/4] Training Medium model ({EPOCHS} epochs, bs={BATCH_SIZE})...")
    model, history, best_acc = train_model(train_data, eval_data, device)

    # Phase 4: Play games
    print(f"\n[4/4] Playing games vs Stockfish...")
    game_results = run_games(model, device)

    total_time = time.time() - t_start

    # Save results
    results = {
        "experiment": "exp062_massive_data",
        "hypothesis": "700K+ combined data >> 247K exp059",
        "baseline": "exp059 = 47.2%, exp031 (old arch) = 51.2%",
        "seed": SEED,
        "config": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "value_weight": VALUE_WEIGHT,
        },
        "data": {
            "jsonl_total": len(all_jsonl),
            "hf_train": len(hf_train),
            "combined_deduped": len(train_data),
            "eval": len(eval_data),
            "batch_files": batch_files,
        },
        "training": {
            "best_accuracy": round(best_acc, 4),
            "history": history,
        },
        "games": game_results,
        "timing": {
            "total_s": round(total_time),
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f" RESULTS: exp062_massive_data")
    print(f"{'='*70}")
    print(f"  Data: {len(all_jsonl):,} generated + {len(hf_train):,} HF "
          f"= {len(train_data):,} combined (deduped)")
    print(f"  Best accuracy: {best_acc:.1%}")
    if history:
        best_ep = max(history, key=lambda h: h["accuracy"])
        print(f"  Best epoch: {best_ep['epoch']} "
              f"(top3={best_ep['top3_accuracy']:.1%}, "
              f"sf_rank={best_ep['mean_sf_rank']:.1f})")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
