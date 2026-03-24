"""exp063: Search-policy with soft multi-PV targets at scale.

Hypothesis: Combining exp062's 722K data scale with exp061's soft target approach
will achieve both higher accuracy AND better move distribution quality, enabling
the model to serve as a strong "search prior" that beats node-limited Stockfish.

Key innovations over previous experiments:
  1. 722K data (3x exp061's 247K) with soft CP-weighted targets
  2. Deep-labeled subset (10K positions at SF d12, 10 PVs) mixed in
  3. Fixed-node SF evaluation: compare model's 0-node policy against SF at
     100/1K/10K/100K nodes — the fair compute-matched regime
  4. KL-divergence loss on full move distribution (search-policy training)

Experiment contract:
  - Primary metric: top-1 accuracy on HF eval (2500 pos)
  - Secondary: agreement rate with SF at various node budgets
  - Baseline: exp062 (hard targets, 722K), exp061 (soft targets, 247K)
  - Target: >50% accuracy, >60% agreement with SF at 1K nodes
  - Runtime: ~5 hours on RTX 2000 Ada (16GB)
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

OUTPUT_DIR = Path("outputs/exp063_search_policy")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Data paths
GENERATED_BATCHES = "outputs/generated_data/batch_*.jsonl"
EXP059_DATA = Path("outputs/exp059_data_scaling/generated_200k.jsonl")
DEEP_DATA = Path("outputs/deep_labeled/deep_d12_pv10.jsonl")

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
SOFT_TEMP = 100.0  # Temperature for CP→prob conversion
SEED = 42

# Fixed-node evaluation config
EVAL_NODE_BUDGETS = [100, 1000, 10000, 100000]

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


# ── Model (same as exp059/062) ──

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


# ── Data loading ──

def build_soft_target(top_moves, board, temperature):
    """Build soft target distribution from SF top moves."""
    target = torch.zeros(VOCAB_SIZE)
    valid_idxs = []
    cp_scores = []

    for tm in top_moves:
        uci = tm["uci"]
        if uci not in UCI_TO_IDX:
            continue
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            continue
        if "mate" in tm:
            cp = 10000 if tm["mate"] > 0 else -10000
        else:
            cp = tm.get("cp", 0)
        valid_idxs.append(UCI_TO_IDX[uci])
        cp_scores.append(cp)

    if not valid_idxs:
        return None

    cp_tensor = torch.tensor(cp_scores, dtype=torch.float32)
    probs = F.softmax(cp_tensor / temperature, dim=0)
    for idx, p in zip(valid_idxs, probs):
        target[idx] = p.item()

    return target


def load_jsonl_data(path):
    """Load JSONL data file, return list of dicts."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_training_data(all_jsonl_data, hf_data, temperature):
    """Combine JSONL data + HF data with soft targets where available."""
    seen_fens = set()
    combined = []

    # HF data first (highest quality — real games, hard targets only)
    for d in hf_data:
        fen = d["board"].fen()
        if fen in seen_fens:
            continue
        seen_fens.add(fen)
        move_idx = move_to_index(d["move"])
        # Hard target as one-hot soft target
        hard_target = torch.zeros(VOCAB_SIZE)
        hard_target[move_idx] = 1.0
        combined.append({
            "board": d["board"],
            "move": d["move"],
            "soft_target": hard_target,
            "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
            "phase": d.get("phase", "unknown"),
            "has_soft": False,
        })

    # Generated data (has top_moves for soft targets)
    n_soft = 0
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

            # Build soft target from top_moves
            top_moves = d.get("top_moves", [])
            soft_target = build_soft_target(top_moves, board, temperature)
            if soft_target is None:
                # Fallback to hard target
                move_idx = UCI_TO_IDX[d["best_move"]]
                soft_target = torch.zeros(VOCAB_SIZE)
                soft_target[move_idx] = 1.0
                has_soft = False
            else:
                has_soft = True
                n_soft += 1

            combined.append({
                "board": board,
                "move": move,
                "soft_target": soft_target,
                "wdl": tuple(d["wdl"]),
                "phase": d.get("phase", "unknown"),
                "has_soft": has_soft,
            })
        except Exception:
            continue

    print(f"  Soft targets: {n_soft:,}/{len(combined):,} "
          f"({100*n_soft/max(len(combined),1):.1f}%)")
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

            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)

            # Soft policy loss: KL divergence with soft targets
            soft_targets = torch.stack([d["soft_target"] for d in chunk]).to(device)
            log_probs = F.log_softmax(result["policy_logits"], dim=-1)
            policy_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean")

            # Value loss
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


# ── Fixed-node SF evaluation ──

def evaluate_vs_sf_nodes(model, eval_data, device, node_budgets, n_positions=500):
    """Compare model's top-1 move against SF at various node budgets."""
    from stockfish import Stockfish

    model.eval()
    # Use a random subset for speed
    random.seed(SEED)
    subset = random.sample(eval_data, min(n_positions, len(eval_data)))

    results = {}
    for budget in node_budgets:
        sf = Stockfish(path=SF_PATH, parameters={"Threads": 1, "Hash": 16})
        agree = 0
        total = 0

        for d in subset:
            board = d["board"]
            fen = board.fen()

            # Model's top move
            moves = model.get_policy_topk(board, device, k=1)
            if not moves:
                continue
            model_move = moves[0][0].uci()

            # SF top move at this node budget
            sf.set_fen_position(fen)
            sf_moves = sf.get_top_moves(1, num_nodes=budget)
            if not sf_moves:
                continue
            sf_move = sf_moves[0]["Move"]

            if model_move == sf_move:
                agree += 1
            total += 1

        rate = agree / max(total, 1)
        results[f"nodes_{budget}"] = {
            "agreement": round(rate, 4),
            "agree": agree,
            "total": total,
        }
        print(f"    SF {budget:>7} nodes: {rate:.1%} agreement ({agree}/{total})",
              flush=True)

    return results


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
    print(f"Experiment: exp063_search_policy")
    print(f"Hypothesis: Soft multi-PV targets at 722K scale → stronger search prior")
    print(f"Baseline: exp062 (hard targets), exp061 (soft, 247K)")
    print()

    # Phase 1: Load all generated data
    print("[1/5] Loading generated data...")
    t_load = time.time()

    all_jsonl = []

    # Load CPU-generated batches
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

    # Load deep-labeled data (higher priority — will override shallow labels)
    n_deep = 0
    if DEEP_DATA.exists():
        deep_data = load_jsonl_data(str(DEEP_DATA))
        print(f"  {DEEP_DATA}: {len(deep_data):,} positions (deep d12)")
        # Replace existing entries for same FENs with deep versions
        deep_fens = {d["fen"] for d in deep_data}
        all_jsonl = [d for d in all_jsonl if d["fen"] not in deep_fens]
        all_jsonl.extend(deep_data)
        n_deep = len(deep_data)

    print(f"  Total JSONL loaded: {len(all_jsonl):,} ({n_deep:,} deep) "
          f"({time.time() - t_load:.0f}s)")

    # Phase 2: Load HF data and combine
    print("\n[2/5] Loading HF data and combining...")
    from hf_data import load_training_set, load_eval_set
    hf_train = load_training_set()
    hf_eval = load_eval_set(n=2500)
    print(f"  HF: {len(hf_train)} train, {len(hf_eval)} eval")

    train_data = prepare_training_data(all_jsonl, hf_train, SOFT_TEMP)
    eval_data = [{
        "board": d["board"], "move": d["move"],
        "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
        "phase": d.get("phase", "unknown"),
    } for d in hf_eval]
    print(f"  Combined train: {len(train_data):,} (deduped)")

    # Phase 3: Train
    print(f"\n[3/5] Training with soft targets ({EPOCHS} epochs, bs={BATCH_SIZE})...")
    model, history, best_acc = train_model(train_data, eval_data, device)

    # Phase 4: Fixed-node SF evaluation
    print(f"\n[4/5] Fixed-node SF evaluation (model 0-node vs SF)...")
    node_results = evaluate_vs_sf_nodes(model, eval_data, device, EVAL_NODE_BUDGETS)

    # Phase 5: Play games
    print(f"\n[5/5] Playing games vs Stockfish...")
    game_results = run_games(model, device)

    total_time = time.time() - t_start

    # Save results
    results = {
        "experiment": "exp063_search_policy",
        "hypothesis": "Soft multi-PV at 722K scale → stronger search prior",
        "baseline": "exp062 (hard, 722K), exp061 (soft, 247K)",
        "seed": SEED,
        "config": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "value_weight": VALUE_WEIGHT, "soft_temp": SOFT_TEMP,
        },
        "data": {
            "jsonl_total": len(all_jsonl),
            "deep_labeled": n_deep,
            "hf_train": len(hf_train),
            "combined_deduped": len(train_data),
            "eval": len(eval_data),
        },
        "training": {
            "best_accuracy": round(best_acc, 4),
            "history": history,
        },
        "fixed_node_eval": node_results,
        "games": game_results,
        "timing": {
            "total_s": round(total_time),
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f" RESULTS: exp063_search_policy")
    print(f"{'='*70}")
    print(f"  Data: {len(all_jsonl):,} JSONL + {len(hf_train):,} HF "
          f"= {len(train_data):,} combined")
    print(f"  Best accuracy: {best_acc:.1%}")
    if history:
        best_ep = max(history, key=lambda h: h["accuracy"])
        print(f"  Best epoch: {best_ep['epoch']} "
              f"(top3={best_ep['top3_accuracy']:.1%}, "
              f"sf_rank={best_ep['mean_sf_rank']:.1f})")
    print(f"  Fixed-node agreement:")
    for k, v in node_results.items():
        print(f"    {k}: {v['agreement']:.1%}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
