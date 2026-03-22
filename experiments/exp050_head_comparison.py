"""exp050: Controlled head comparison — flat vs spatial policy head.

Hypothesis: The factorized SpatialPolicyHead outperforms the flat policy head
on the same data, same model body, same training budget, because it can
leverage per-square hidden states rather than compressing everything into
one pooled vector.

This is the first "fewer, better-controlled" experiment per the Codex roadmap.
It follows the experiment contract:

  - One hypothesis: spatial head > flat head on same data
  - Primary metric: top-1 SF-accuracy on held-out eval set
  - Fixed evaluation set: game-level split, no position leakage
  - Seeds: 42, 123, 314
  - Data: cached Lichess+SF labeled data from exp049, or falls back to
    random-generated SF-labeled data if Lichess cache unavailable
  - Runtime target: <10 minutes total

Design:
  Variant A — ChessTransformer with FLAT policy head (Linear -> VOCAB_SIZE)
  Variant B — ChessTransformer with SPATIAL policy head (from×to factored)
  Both share: same LearnedBoardEncoder, same body (8-layer encoder-only
  transformer, hidden_dim=512), same optimizer, same schedule, same data
  split, same evaluation.
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
from chess_model import LearnedBoardEncoder
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move

OUTPUT_DIR = Path("outputs/exp050_head_comparison")

# Data sources (try Lichess+SF first, fall back to random SF)
LICHESS_SF_DATA = Path("outputs/exp049_lichess_sf/lichess_sf_labeled.jsonl")
RANDOM_SF_DATA = Path("data/sf_labels_50k_d8.jsonl")
SMALL_SF_DATA = Path("data/sf_labels_5k_d8.jsonl")

# Hyperparams — matched across variants
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1
EPOCHS = 5
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 3e-4
WARMUP_FRAC = 0.05
SEEDS = [42, 123, 314]
NUM_EVAL = 2000


# === Architecture components ===

def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
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
    """Factorized policy: from_sq × to_sq × promotion, using per-square hiddens."""
    def __init__(self, hidden_size, head_dim=256):
        super().__init__()
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer('from_sqs', from_sqs)
        self.register_buffer('to_sqs', to_sqs)
        self.register_buffer('promo_types', promo_types)

    def forward(self, hidden_states):
        """hidden_states: (B, 67, D) — [turn, castling, ep, sq0..sq63]."""
        sq_hidden = hidden_states[:, 3:67, :]  # 64 square tokens
        global_hidden = hidden_states[:, 0, :]  # turn token as global
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(global_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


class FlatPolicyHead(nn.Module):
    """Standard flat policy head: pool -> Linear -> VOCAB_SIZE."""
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, VOCAB_SIZE),
        )

    def forward(self, hidden_states):
        """hidden_states: (B, 67, D) — uses token 0 as global pooled state."""
        return self.head(hidden_states[:, 0, :])


class ChessTransformer(nn.Module):
    """Shared body — head is injected."""
    def __init__(self, policy_head_cls, encoder_dim=256, hidden_dim=512,
                 num_layers=8, num_heads=8, dropout=0.1, **head_kwargs):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 67, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = policy_head_cls(hidden_dim, **head_kwargs)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 3),
        )
        self.hidden_dim = hidden_dim

    def forward(self, board_input, **kw):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens) + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)
        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)
        return {"policy_logits": policy_logits, "value_logits": value_logits}

    @torch.no_grad()
    def predict_move(self, board):
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return index_to_move(probs.argmax().item()), probs


# === Data ===

def load_data_auto():
    """Load the best available labeled data."""
    for path in [LICHESS_SF_DATA, RANDOM_SF_DATA, SMALL_SF_DATA]:
        if path.exists():
            print(f"  Loading data from {path}...")
            data = []
            with open(path) as f:
                for line in f:
                    entry = json.loads(line)
                    try:
                        board = chess.Board(entry["fen"])
                        # Try SF-labeled format first, then generic
                        move_uci = entry.get("sf_move") or entry.get("best_uci") or entry.get("best_move")
                        if not move_uci:
                            continue
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves and move_uci in UCI_TO_IDX:
                            game_id = entry.get("game_id", entry.get("fen", "")[:20])
                            data.append({"board": board, "move": move, "game_id": game_id})
                    except Exception:
                        continue
            print(f"  Loaded {len(data)} positions from {path.name}")
            return data, path.name
    raise FileNotFoundError("No labeled data found. Run label_positions.py or exp049 first.")


def split_by_game(data, seed, eval_n):
    """Split data by game_id to prevent leakage from correlated positions."""
    rng = random.Random(seed)
    # Group by game_id
    games = {}
    for d in data:
        gid = d.get("game_id", id(d))
        games.setdefault(gid, []).append(d)
    game_ids = list(games.keys())
    rng.shuffle(game_ids)

    # Greedily assign games to eval until we have enough
    eval_data, train_data = [], []
    for gid in game_ids:
        if len(eval_data) < eval_n:
            eval_data.extend(games[gid])
        else:
            train_data.extend(games[gid])

    rng.shuffle(train_data)
    rng.shuffle(eval_data)
    eval_data = eval_data[:eval_n]
    return train_data, eval_data


# === Training ===

def train_one_seed(variant_name, model, train_data, eval_data, device, seed):
    """Train a single variant for one seed. Returns history and best accuracy."""
    torch.manual_seed(seed)
    random.seed(seed)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_data) // (BATCH_SIZE * ACCUM_STEPS)) * EPOCHS
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
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor([move_to_index(d["move"]) for d in chunk], device=device)
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            loss = F.cross_entropy(result["policy_logits"], targets) / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS
            n_batches += 1

            if n_batches % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if n_batches % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(n_batches, 1)
        ev = evaluate_accuracy(model, eval_data, device)
        ep_time = time.time() - t0
        history.append({
            "epoch": epoch + 1, "loss": round(avg_loss, 4),
            "accuracy": ev["accuracy"], "top3_accuracy": ev["top3_accuracy"],
            "time_s": round(ep_time),
        })
        print(f"    [{variant_name}/s{seed}] Ep{epoch+1}: loss={avg_loss:.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{ep_time:.0f}s]")

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return history, best_acc


def evaluate_accuracy(model, eval_data, device, batch_size=128):
    model.eval()
    correct = top3_correct = total = 0
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
            for j, t in enumerate(targets):
                total += 1
                if preds[j] == t:
                    correct += 1
                if t in top3s[j]:
                    top3_correct += 1
    return {
        "accuracy": round(correct / max(total, 1), 4),
        "top3_accuracy": round(top3_correct / max(total, 1), 4),
    }


# === Main ===

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp050_head_comparison")
    print(f"Hypothesis: SpatialPolicyHead > FlatPolicyHead on same data/body/budget")

    # Load data
    print(f"\n[1/3] Loading data...")
    all_data, data_source = load_data_auto()

    # Collect results per variant, per seed
    results = {
        "experiment": "exp050_head_comparison",
        "hypothesis": "Spatial head > flat head on same data, body, and budget",
        "data_source": data_source,
        "seeds": SEEDS,
        "config": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        },
        "variants": {},
    }

    variants = {
        "flat": {"cls": FlatPolicyHead, "kwargs": {}},
        "spatial": {"cls": SpatialPolicyHead, "kwargs": {"head_dim": 256}},
    }

    for vname, vspec in variants.items():
        print(f"\n{'='*60}")
        print(f"  Variant: {vname}")
        print(f"{'='*60}")
        seed_results = []

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            # Split data (same split for both variants at same seed)
            train_data, eval_data = split_by_game(all_data, seed=seed, eval_n=NUM_EVAL)
            print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

            # Build model
            model = ChessTransformer(
                policy_head_cls=vspec["cls"],
                encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                dropout=DROPOUT, **vspec["kwargs"],
            ).to(device)

            n_params = sum(p.numel() for p in model.parameters())
            n_head = sum(p.numel() for p in model.policy_head.parameters())
            print(f"  Params: {n_params:,} total, {n_head:,} in policy head")

            history, best_acc = train_one_seed(vname, model, train_data, eval_data, device, seed)
            seed_results.append({
                "seed": seed,
                "best_accuracy": best_acc,
                "history": history,
                "params_total": n_params,
                "params_head": n_head,
                "train_size": len(train_data),
                "eval_size": len(eval_data),
            })

        # Summary for variant
        accs = [r["best_accuracy"] for r in seed_results]
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc)**2 for a in accs) / len(accs)) ** 0.5
        results["variants"][vname] = {
            "seeds": seed_results,
            "mean_accuracy": round(mean_acc, 4),
            "std_accuracy": round(std_acc, 4),
        }
        print(f"\n  [{vname}] Mean acc: {mean_acc:.1%} ± {std_acc:.1%}")

    # === Final comparison ===
    total_time = time.time() - t0
    results["timing"] = {"total_s": round(total_time)}

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" RESULTS: exp050_head_comparison")
    print(f" Data: {data_source}")
    for vname, vdata in results["variants"].items():
        per_seed = ", ".join(f"s{r['seed']}={r['best_accuracy']:.1%}" for r in vdata["seeds"])
        print(f"  {vname:>8}: {vdata['mean_accuracy']:.1%} ± {vdata['std_accuracy']:.1%}  ({per_seed})")
    flat_mean = results["variants"]["flat"]["mean_accuracy"]
    spatial_mean = results["variants"]["spatial"]["mean_accuracy"]
    delta = spatial_mean - flat_mean
    winner = "spatial" if delta > 0 else "flat"
    print(f"\n  Delta (spatial - flat): {delta:+.1%}")
    print(f"  Winner: {winner}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
