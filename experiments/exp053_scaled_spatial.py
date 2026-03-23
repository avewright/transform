"""exp053: Scale spatial head to Medium model config.

Hypothesis: Scaling from Small (256d, 6L, 4.7M) to Medium (512d, 8L, ~17M)
with the spatial head improves accuracy, because exp052 showed the model was
still learning at epoch 3 (loss still dropping). More capacity + more epochs
should help.

Experiment contract:
  - Hypothesis: Medium model > Small model with spatial head
  - Primary metric: top-1 SF-accuracy on HF test split
  - Secondary: top-3, per-phase accuracy, entropy, SF-move rank
  - Evaluation set: HF avewright/chess-positions test (2500)
  - Seeds: 42, 123 (2 seeds — Medium is ~3x slower)
  - Training data: HF train split (~47.5K)
  - Model: Medium (512d, 8L, 8H, ~17M params)
  - Epochs: 5 (vs 3 in exp052)
  - Runtime target: <10 min total
  - Device: CUDA (RTX 2000 Ada, 16GB)
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
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)

OUTPUT_DIR = Path("outputs/exp053_scaled_spatial")

# ── Medium model config ──
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1

# ── Training ──
EPOCHS = 5
BATCH_SIZE = 128   # smaller than exp052 due to larger model
LR = 2e-4          # slightly lower for larger model
WARMUP_FRAC = 0.05
SEEDS = [42, 123]


# =====================================================================
# Architecture (reuse from exp052)
# =====================================================================

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
    """Factorized policy: from_sq * to_sq * promotion via per-square hiddens."""

    def __init__(self, hidden_size, n_ctx_tokens, head_dim=256):
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
        sq_hidden = hidden_states[:, self.n_ctx : self.n_ctx + 64, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(cls_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


class ChessTransformerV2(nn.Module):
    """Chess-native encoder-only transformer with learned CLS token.
    Token layout: [CLS] [turn] [castling] [ep] [sq0..sq63] = 68 tokens
    """

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
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = SpatialPolicyHead(
            hidden_dim, n_ctx_tokens=4, head_dim=head_dim
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 3),
        )
        self.hidden_dim = hidden_dim

    def forward(self, board_input, **kw):
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


# =====================================================================
# Data loading
# =====================================================================

def load_hf_data():
    from hf_data import load_training_set, load_eval_set, dataset_info
    info = dataset_info()
    print(f"  HF: {info['train']['num_rows']} train, "
          f"{info['test']['num_rows']} test")

    def _enrich(raw_list):
        return [
            {
                "board": d["board"],
                "move": d["move"],
                "phase": d.get("phase", "unknown"),
                "eval_type": d.get("eval_type", ""),
                "eval_value": d.get("eval_value", 0),
            }
            for d in raw_list
        ]

    train_data = _enrich(load_training_set())
    eval_data = _enrich(load_eval_set(n=2500))
    print(f"  Loaded: {len(train_data)} train, {len(eval_data)} eval")
    return train_data, eval_data


# =====================================================================
# Training
# =====================================================================

def train_one_seed(model, train_data, eval_data, device, seed, save_dir):
    torch.manual_seed(seed)
    random.seed(seed)

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
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i : i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor(
                [move_to_index(d["move"]) for d in chunk], device=device
            )
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            loss = F.cross_entropy(result["policy_logits"], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ev = evaluate_rich(model, eval_data, device)
        ep_time = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 4),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in ev.items()},
            "time_s": round(ep_time),
        })
        marker = " *" if ev["accuracy"] > best_acc else ""
        print(
            f"  [s{seed}] Ep{epoch+1}: "
            f"loss={avg_loss:.4f} acc={ev['accuracy']:.1%} "
            f"top3={ev['top3_accuracy']:.1%} "
            f"entropy={ev['mean_entropy']:.2f} "
            f"sf_rank={ev['mean_sf_rank']:.1f} [{ep_time:.0f}s]{marker}"
        )

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        ckpt_path = save_dir / f"spatial_medium_s{seed}.pt"
        torch.save(best_state, ckpt_path)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Saved: {ckpt_path.name}")

    return history, best_acc


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_rich(model, eval_data, device, batch_size=128):
    model.eval()
    correct = top3_correct = total = 0
    entropy_sum = sf_rank_sum = 0.0
    phase_stats = {}

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i : i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            phases = [d["phase"] for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]

            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()

            for j, t in enumerate(targets):
                total += 1
                is_correct = preds[j] == t
                if is_correct:
                    correct += 1
                if t in top3s[j]:
                    top3_correct += 1
                p = probs[j]
                legal_p = p[p > 0]
                entropy_sum += -(legal_p * legal_p.log()).sum().item()
                sorted_idx = logits[j].argsort(descending=True).cpu().tolist()
                sf_rank = sorted_idx.index(t) + 1 if t in sorted_idx else len(sorted_idx)
                sf_rank_sum += sf_rank
                ph = phases[j]
                if ph not in phase_stats:
                    phase_stats[ph] = {"correct": 0, "total": 0}
                phase_stats[ph]["total"] += 1
                if is_correct:
                    phase_stats[ph]["correct"] += 1

    result = {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "n_eval": total,
    }
    for ph, ps in sorted(phase_stats.items()):
        result[f"acc_{ph}"] = ps["correct"] / max(ps["total"], 1)
        result[f"n_{ph}"] = ps["total"]
    return result


# =====================================================================
# Main
# =====================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp053_scaled_spatial")
    print(f"Hypothesis: Medium (512d, 8L) > Small (256d, 6L) with spatial head")
    print(f"Config: {HIDDEN_DIM}d, {NUM_LAYERS}L, {NUM_HEADS}H, "
          f"bs={BATCH_SIZE}, lr={LR}, epochs={EPOCHS}")

    print(f"\n[1/2] Loading data...")
    train_data, eval_data = load_hf_data()

    results = {
        "experiment": "exp053_scaled_spatial",
        "hypothesis": "Medium > Small with spatial head + more epochs",
        "data_source": "hf:avewright/chess-positions",
        "baseline": "exp052 spatial Small: 30.3% +/- 0.2%",
        "seeds": SEEDS,
        "config": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        },
        "seeds_data": [],
    }

    print(f"\n[2/2] Training Medium spatial model...")
    seed_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        model = ChessTransformerV2(
            encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
            dropout=DROPOUT, head_dim=256,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        history, best_acc = train_one_seed(
            model, train_data, eval_data, device, seed, OUTPUT_DIR
        )
        final_eval = evaluate_rich(model, eval_data, device)

        seed_results.append({
            "seed": seed,
            "best_accuracy": best_acc,
            "final_eval": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in final_eval.items()},
            "history": history,
            "params": n_params,
        })
        del model
        torch.cuda.empty_cache()

    accs = [r["best_accuracy"] for r in seed_results]
    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
    results["seeds_data"] = seed_results
    results["mean_accuracy"] = round(mean_acc, 4)
    results["std_accuracy"] = round(std_acc, 4)
    results["timing"] = {"total_s": round(time.time() - t_start)}

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" RESULTS: exp053_scaled_spatial")
    print(f" Config: {HIDDEN_DIM}d, {NUM_LAYERS}L, epochs={EPOCHS}")
    for r in seed_results:
        print(f"  s{r['seed']}: {r['best_accuracy']:.1%}")
        fe = r["final_eval"]
        phases = [k for k in fe if k.startswith("acc_")]
        if phases:
            print(f"    phases: " + "  ".join(
                f"{k.replace('acc_','')}={fe[k]:.1%}" for k in sorted(phases)))
        print(f"    top3={fe['top3_accuracy']:.1%} "
              f"entropy={fe['mean_entropy']:.2f} sf_rank={fe['mean_sf_rank']:.1f}")
    print(f"\n  Mean: {mean_acc:.1%} +/- {std_acc:.1%}")
    print(f"  Baseline (exp052 Small spatial): 30.3% +/- 0.2%")
    delta = mean_acc - 0.303
    print(f"  Delta vs baseline: {delta:+.1%}")
    print(f"  Total time: {time.time() - t_start:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
