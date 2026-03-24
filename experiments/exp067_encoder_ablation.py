"""exp067: Encoding ablation — baseline vs fused 12-piece tokens.

Hypothesis: A fused 13-token embedding (empty + 12 piece-color types) matches
or beats the factored piece_embed + color_proj encoder on policy accuracy,
with fewer encoder params and simpler forward pass.

Controlled variables:
  - Same model body (512d, 8L, 8H, ffn_ratio=4, bidirectional)
  - Same data (500K Lichess positions, depth >= 15, same seed split)
  - Same schedule (1 epoch, AdamW, cosine LR, warmup 5%)
  - Same policy + value head
  - Same evaluation (2500 positions)

Independent variable: encoder type (baseline vs fused)

Seeds: 42, 123, 314 — per instructions, <2pp requires 3 seeds.

Metrics:
  - top-1 accuracy (primary)
  - top-3 accuracy
  - mean SF rank
  - value accuracy
  - phase breakdown (opening/middlegame/endgame)
  - throughput (pos/s)
  - encoder param count
"""

import gc
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
from torch.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_model import LearnedBoardEncoder, FusedBoardEncoder
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)
from data_loader import (
    load_training_data, get_batch_input, get_eval_batch_input,
)

OUTPUT_DIR = Path("outputs/exp067_encoder_ablation")
SEED = 42  # data split seed (fixed); training seeds vary

# ── Fixed training config ──
TRAIN_POSITIONS = 500_000
EVAL_POSITIONS = 2500
MIN_DEPTH = 15
BATCH_SIZE = 256
LR = 3e-4
WARMUP_FRAC = 0.05
VALUE_WEIGHT = 0.5
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
FFN_RATIO = 4
ENCODER_DIM = 256
HEAD_DIM = 256

TRAINING_SEEDS = [42, 123, 314]


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


class ChessTransformerAblation(nn.Module):
    """Configurable model for A/B encoder comparison.

    encoder_type: "baseline" or "fused"
    """

    def __init__(self, encoder_type="baseline"):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "baseline":
            self.encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
        elif encoder_type == "fused":
            self.encoder = FusedBoardEncoder(embed_dim=ENCODER_DIM)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.input_proj = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        # 68 = 1 CLS + 3 context (turn, castling, ep) + 64 squares
        self.pos_embed = nn.Parameter(torch.randn(1, 68, HIDDEN_DIM) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * FFN_RATIO, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_LAYERS,
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.policy_head = SpatialPolicyHead(HIDDEN_DIM, n_ctx_tokens=4, head_dim=HEAD_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 256), nn.ReLU(), nn.Linear(256, 3),
        )

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
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }


# ── Training ──

def train_one(encoder_type, train_tensors, eval_data, eval_tensors, device, seed):
    """Train one config for one seed. Returns metrics dict."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    random.seed(seed)

    model = ChessTransformerAblation(encoder_type=encoder_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())

    print(f"\n  [{encoder_type} seed={seed}] Params: {n_params:,} (encoder: {enc_params:,})")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_train = train_tensors["move_idx"].shape[0]
    steps = n_train // BATCH_SIZE
    warmup_steps = max(int(steps * WARMUP_FRAC), 1)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler('cuda')

    # Shuffle training indices
    perm = torch.randperm(n_train)

    model.train()
    total_pl = total_vl = 0.0
    n_batches = 0
    t0 = time.time()

    for i in range(0, n_train, BATCH_SIZE):
        indices = perm[i:i + BATCH_SIZE]
        if len(indices) < 2:
            continue

        batch_input = get_batch_input(train_tensors, indices, encoder_type, device)
        targets = train_tensors["move_idx"][indices].to(device)
        wdl_targets = train_tensors["wdl"][indices].float().to(device)

        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)
            policy_loss = F.cross_entropy(result["policy_logits"], targets)
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

        if n_batches % 400 == 0:
            elapsed = time.time() - t0
            tp = (n_batches * BATCH_SIZE) / elapsed
            print(f"    step {n_batches}/{steps} | pl={total_pl/n_batches:.3f} | "
                  f"vl={total_vl/n_batches:.3f} | {tp:.0f} pos/s", flush=True)

    train_time = time.time() - t0
    throughput = (n_batches * BATCH_SIZE) / train_time

    # Evaluate
    ev = evaluate(model, eval_data, eval_tensors, encoder_type, device)

    result = {
        "encoder_type": encoder_type,
        "seed": seed,
        "params": n_params,
        "encoder_params": enc_params,
        "policy_loss": round(total_pl / max(n_batches, 1), 4),
        "value_loss": round(total_vl / max(n_batches, 1), 4),
        "accuracy": round(ev["accuracy"], 4),
        "top3_accuracy": round(ev["top3_accuracy"], 4),
        "mean_sf_rank": round(ev["mean_sf_rank"], 2),
        "value_accuracy": round(ev["value_accuracy"], 4),
        "phase_accuracy": ev["phase_accuracy"],
        "train_time_s": round(train_time),
        "throughput_pos_s": round(throughput),
    }

    print(f"    RESULT: acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} "
          f"sf_rank={ev['mean_sf_rank']:.1f} val={ev['value_accuracy']:.1%} "
          f"{throughput:.0f} pos/s ({train_time:.0f}s)")

    # Save checkpoint
    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{encoder_type}_s{seed}.pt")

    del model, optimizer, scaler, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ── Evaluation ──

def evaluate(model, eval_data, eval_tensors, encoder_type, device, batch_size=256):
    model.eval()
    correct = top3_correct = total = 0
    sf_rank_sum = 0.0
    val_correct = val_total = 0
    phase_stats = {}

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            n = len(chunk)
            idx = slice(i, i + n)

            batch_input = get_eval_batch_input(eval_tensors, idx, encoder_type, device)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"].float()

            for j, d in enumerate(chunk):
                board = d["board"]
                true_move = d["move"]
                phase = d.get("phase", "unknown")

                l = logits[j].clone()
                mask = legal_move_mask(board).to(device)
                l[~mask] = float("-inf")

                pred_idx = l.argmax().item()
                true_idx = move_to_index(true_move)
                hit = pred_idx == true_idx
                if hit:
                    correct += 1
                topk = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk:
                    top3_correct += 1

                sorted_indices = l.argsort(descending=True).tolist()
                rank = sorted_indices.index(true_idx) + 1 if true_idx in sorted_indices else len(sorted_indices)
                sf_rank_sum += rank
                total += 1

                if phase not in phase_stats:
                    phase_stats[phase] = {"correct": 0, "total": 0}
                phase_stats[phase]["total"] += 1
                if hit:
                    phase_stats[phase]["correct"] += 1

            # Value accuracy
            wdl_logits = result["value_logits"].float()
            for j, d in enumerate(chunk):
                pred_class = wdl_logits[j].argmax().item()
                true_wdl = d["wdl"]
                true_class = max(range(3), key=lambda k: true_wdl[k])
                if pred_class == true_class:
                    val_correct += 1
                val_total += 1

    phase_accuracy = {}
    for p, s in phase_stats.items():
        phase_accuracy[p] = round(s["correct"] / max(s["total"], 1), 4)

    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
        "phase_accuracy": phase_accuracy,
        "n_eval": total,
    }


# ── Main ──

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*70}")
    print(f" EXP067: ENCODER ABLATION — baseline vs fused")
    print(f"{'='*70}")
    print(f"  Device: {device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'})")
    print(f"  Model: {HIDDEN_DIM}d, {NUM_LAYERS}L, {NUM_HEADS}H, FFN={FFN_RATIO}x")
    print(f"  Train: {TRAIN_POSITIONS:,}, Eval: {EVAL_POSITIONS:,}")
    print(f"  Seeds: {TRAINING_SEEDS}")
    print(f"  Batch: {BATCH_SIZE}, LR: {LR}")
    print()

    # Load data once — shared across all runs (instant from .pt cache)
    train_tensors, eval_data, eval_tensors = load_training_data(
        n_train=TRAIN_POSITIONS, n_eval=EVAL_POSITIONS,
        encoder_type="both", min_depth=MIN_DEPTH, seed=SEED,
    )

    results = []
    t_total = time.time()

    for encoder_type in ["baseline", "fused"]:
        print(f"\n{'='*70}")
        print(f"  ENCODER: {encoder_type}")
        print(f"{'='*70}")

        for seed in TRAINING_SEEDS:
            r = train_one(encoder_type, train_tensors, eval_data, eval_tensors, device, seed)
            results.append(r)

            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(results, f, indent=2)

    total_time = time.time() - t_total

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Encoder':<12} {'Seed':>5} {'Params':>10} {'EncP':>8} {'Acc':>7} {'Top3':>7} "
          f"{'SFRank':>7} {'ValAcc':>7} {'Pos/s':>7}")
    print("-" * 80)

    for r in results:
        print(f"{r['encoder_type']:<12} {r['seed']:>5} {r['params']:>10,} {r['encoder_params']:>8,} "
              f"{r['accuracy']:>6.1%} {r['top3_accuracy']:>6.1%} "
              f"{r['mean_sf_rank']:>7.1f} {r['value_accuracy']:>6.1%} "
              f"{r['throughput_pos_s']:>6}")

    # Aggregate by encoder type
    print(f"\n{'='*80}")
    print(f" AGGREGATED (mean ± std across seeds)")
    print(f"{'='*80}")

    for enc_type in ["baseline", "fused"]:
        runs = [r for r in results if r["encoder_type"] == enc_type]
        if not runs:
            continue
        accs = [r["accuracy"] for r in runs]
        top3s = [r["top3_accuracy"] for r in runs]
        ranks = [r["mean_sf_rank"] for r in runs]
        vals = [r["value_accuracy"] for r in runs]
        tps = [r["throughput_pos_s"] for r in runs]

        import statistics
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0

        print(f"  {enc_type:<12}: acc={mean_acc:.1%}±{std_acc:.1%}  "
              f"top3={statistics.mean(top3s):.1%}  "
              f"sf_rank={statistics.mean(ranks):.1f}  "
              f"val={statistics.mean(vals):.1%}  "
              f"tp={statistics.mean(tps):.0f} pos/s  "
              f"enc_params={runs[0]['encoder_params']:,}")

        # Phase breakdown
        phases = set()
        for r in runs:
            phases.update(r.get("phase_accuracy", {}).keys())
        if phases:
            for phase in sorted(phases):
                phase_accs = [r["phase_accuracy"].get(phase, 0) for r in runs]
                print(f"    {phase:<12}: {statistics.mean(phase_accs):.1%}")

    # Delta
    baseline_runs = [r for r in results if r["encoder_type"] == "baseline"]
    fused_runs = [r for r in results if r["encoder_type"] == "fused"]
    if baseline_runs and fused_runs:
        import statistics
        b_mean = statistics.mean([r["accuracy"] for r in baseline_runs])
        f_mean = statistics.mean([r["accuracy"] for r in fused_runs])
        delta = f_mean - b_mean
        print(f"\n  DELTA (fused - baseline): {delta:+.1%}")
        if abs(delta) < 0.02:
            print("  ** Delta < 2pp — PROVISIONAL. Need more seeds to claim significance. **")
        elif delta > 0:
            print("  ** Fused encoder WINS. **")
        else:
            print("  ** Baseline encoder WINS. **")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")

    # Save summary
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "experiment": "exp067_encoder_ablation",
            "hypothesis": "Fused 13-token encoder matches or beats factored piece+color encoder",
            "primary_metric": "top-1 accuracy",
            "total_time_s": round(total_time),
            "train_positions": TRAIN_POSITIONS,
            "eval_positions": EVAL_POSITIONS,
            "min_depth": MIN_DEPTH,
            "seeds": TRAINING_SEEDS,
            "model_config": {
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "ffn_ratio": FFN_RATIO,
                "encoder_dim": ENCODER_DIM,
            },
            "command": f"python3 experiments/exp067_encoder_ablation.py",
        }, f, indent=2)

    print(f"Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
