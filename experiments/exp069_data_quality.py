"""exp069: Data quality features for training sample weighting.

Hypothesis: Using confidence-weighted sampling (based on cp_gap, forced moves,
agreement flags) improves policy accuracy compared to uniform sampling, without
changing the model architecture.

This experiment extends the Lichess parquet data with quality features:
  - cp_gap_1_2: centipawn gap between 1st and 2nd best move (larger = clearer)
  - is_forced: only 1 legal move (trivial example, down-weight)
  - n_legal_moves: number of legal moves (context for difficulty)
  - sample_weight: composite weight balancing confidence and difficulty

Weighting scheme (priority 3 from instructions):
  - High confidence (large cp_gap): weight 1.0
  - Medium confidence: weight 0.7
  - Forced moves: weight 0.2 (near-trivial)
  - No PV data: weight 0.5

This uses the same model architecture as exp067/068 baseline.
The ONLY change is the sampling distribution during training.

Seeds: 42, 123, 314
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

from chess_model import FusedBoardEncoder
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask
from data_loader import load_training_data, get_batch_input, get_eval_batch_input

OUTPUT_DIR = Path("outputs/exp069_data_quality")
SEED = 42

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


# ── Model (same as exp067 fused variant) ──

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


class ChessTransformerFused(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FusedBoardEncoder(embed_dim=ENCODER_DIM)
        self.input_proj = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
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

def train_one(use_weights, train_tensors, eval_data, eval_tensors, device, seed):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    random.seed(seed)

    label = "weighted" if use_weights else "uniform"
    model = ChessTransformerFused().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n  [{label} seed={seed}] Params: {n_params:,}")

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
    perm = torch.randperm(n_train)

    model.train()
    total_pl = total_vl = 0.0
    n_batches = 0
    t0 = time.time()

    for i in range(0, n_train, BATCH_SIZE):
        indices = perm[i:i + BATCH_SIZE]
        if len(indices) < 2:
            continue

        batch_input = get_batch_input(train_tensors, indices, "fused", device)
        targets = train_tensors["move_idx"][indices].to(device)
        wdl_targets = train_tensors["wdl"][indices].float().to(device)

        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)

            if use_weights:
                # Weighted cross-entropy: per-sample weighting
                weights = train_tensors["weight"][indices].to(device)
                per_sample_loss = F.cross_entropy(
                    result["policy_logits"], targets, reduction="none"
                )
                policy_loss = (per_sample_loss * weights).mean()
            else:
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

    ev = evaluate(model, eval_data, eval_tensors, device)

    result = {
        "variant": label,
        "seed": seed,
        "params": n_params,
        "use_weights": use_weights,
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

    ckpt_dir = OUTPUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{label}_s{seed}.pt")

    del model, optimizer, scaler, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ── Evaluation (same as exp067/068) ──

def evaluate(model, eval_data, eval_tensors, device, batch_size=256):
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

            batch_input = get_eval_batch_input(eval_tensors, idx, "fused", device)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"].float()

            for j, d in enumerate(chunk):
                board, true_move = d["board"], d["move"]
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

            wdl_logits = result["value_logits"].float()
            for j, d in enumerate(chunk):
                pred_class = wdl_logits[j].argmax().item()
                true_wdl = d["wdl"]
                true_class = max(range(3), key=lambda k: true_wdl[k])
                if pred_class == true_class:
                    val_correct += 1
                val_total += 1

    phase_accuracy = {p: round(s["correct"] / max(s["total"], 1), 4) for p, s in phase_stats.items()}
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
    print(f" EXP069: DATA QUALITY WEIGHTED SAMPLING")
    print(f"{'='*70}")
    print(f"  Device: {device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'})")
    print(f"  Model: fused encoder, {HIDDEN_DIM}d, {NUM_LAYERS}L, {NUM_HEADS}H")
    print(f"  Train: {TRAIN_POSITIONS:,}, Eval: {EVAL_POSITIONS:,}")
    print(f"  Seeds: {TRAINING_SEEDS}")
    print()

    train_tensors, eval_data, eval_tensors = load_training_data(
        n_train=TRAIN_POSITIONS, n_eval=EVAL_POSITIONS,
        encoder_type="fused", min_depth=MIN_DEPTH, seed=SEED,
        include_weights=True,
    )

    results = []
    t_total = time.time()

    for use_weights in [False, True]:
        label = "weighted" if use_weights else "uniform"
        print(f"\n{'='*70}")
        print(f"  VARIANT: {label}")
        print(f"{'='*70}")

        for seed in TRAINING_SEEDS:
            r = train_one(use_weights, train_tensors, eval_data, eval_tensors, device, seed)
            results.append(r)

            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(results, f, indent=2)

    total_time = time.time() - t_total

    # Summary
    import statistics
    print(f"\n{'='*80}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*80}")
    for r in results:
        print(f"  {r['variant']:<10} s{r['seed']:>3}: acc={r['accuracy']:.1%} "
              f"top3={r['top3_accuracy']:.1%} sf={r['mean_sf_rank']:.1f} "
              f"val={r['value_accuracy']:.1%}")

    print(f"\n  AGGREGATED:")
    for variant in ["uniform", "weighted"]:
        runs = [r for r in results if r["variant"] == variant]
        if not runs:
            continue
        accs = [r["accuracy"] for r in runs]
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0
        print(f"    {variant:<10}: acc={mean_acc:.1%}±{std_acc:.1%}  "
              f"top3={statistics.mean([r['top3_accuracy'] for r in runs]):.1%}  "
              f"sf={statistics.mean([r['mean_sf_rank'] for r in runs]):.1f}")

        phases = set()
        for r in runs:
            phases.update(r.get("phase_accuracy", {}).keys())
        for phase in sorted(phases):
            phase_accs = [r["phase_accuracy"].get(phase, 0) for r in runs]
            print(f"      {phase}: {statistics.mean(phase_accs):.1%}")

    u = [r for r in results if r["variant"] == "uniform"]
    w = [r for r in results if r["variant"] == "weighted"]
    if u and w:
        delta = statistics.mean([r["accuracy"] for r in w]) - statistics.mean([r["accuracy"] for r in u])
        print(f"\n  DELTA (weighted - uniform): {delta:+.1%}")
        if abs(delta) < 0.02:
            print("  ** Delta < 2pp — PROVISIONAL **")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "experiment": "exp069_data_quality",
            "hypothesis": "Confidence-weighted sampling improves policy accuracy",
            "primary_metric": "top-1 accuracy",
            "total_time_s": round(total_time),
            "train_positions": TRAIN_POSITIONS,
            "eval_positions": EVAL_POSITIONS,
            "seeds": TRAINING_SEEDS,
            "command": "python3 experiments/exp069_data_quality.py",
        }, f, indent=2)

    print(f"Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
