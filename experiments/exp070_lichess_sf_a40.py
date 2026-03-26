"""exp070: Large-scale training on avewright/chess-positions-lichess-sf (A40).

Hypothesis: Training a 12-layer/512d FusedBoardEncoder model on 2M+ Lichess
positions (SF-labeled) with the A40's 46GB VRAM will push accuracy well beyond
prior baselines by leveraging both deeper architecture and massive data.

Hardware: NVIDIA A40 (46GB VRAM), 9 vCPUs, 50GB RAM
Data: avewright/chess-positions-lichess-sf (streamed via data_loader.py)
Model: FusedBoardEncoder (256d) + 12L/512d transformer + SpatialPolicyHead + WDL

Key settings for A40:
  - batch_size=512 (fits ~46GB), accum_steps=2 → effective batch 1024
  - 12 layers (38.7M params) — prior exp065 showed depth helps with enough data
  - 2M training positions, 5000 eval positions
  - 2 epochs over the dataset
  - Mixed precision (FP16) for throughput

Baselines:
  - exp069 (8L/500K): ~35-37% accuracy (estimated)
  - exp065 (12L/1M LearnedEncoder): ~40%+ (estimated)
  - This experiment targets >42% accuracy with larger data

Seed: 42
"""

import gc
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

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

OUTPUT_DIR = Path("outputs/exp070_lichess_sf_a40")

# ── Config ──
TRAIN_POSITIONS = 2_000_000
EVAL_POSITIONS = 5_000
MIN_DEPTH = 10           # lower threshold to get more data
BATCH_SIZE = 512         # A40 46GB handles this fine
ACCUM_STEPS = 2          # effective batch = 1024
EPOCHS = 2
LR = 3e-4
WARMUP_FRAC = 0.05
VALUE_WEIGHT = 0.5
GRAD_CLIP = 1.0
SEED = 42

# Model dims
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 12          # deeper for more data
NUM_HEADS = 8
FFN_RATIO = 4
DROPOUT = 0.1
HEAD_DIM = 256

LOG_INTERVAL = 200
EVAL_INTERVAL = 2000     # eval every N steps
SAVE_INTERVAL = 5000     # checkpoint every N steps


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


class ChessTransformerFused(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FusedBoardEncoder(embed_dim=ENCODER_DIM)
        self.input_proj = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, HIDDEN_DIM) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * FFN_RATIO, dropout=DROPOUT,
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


# ── Evaluation ──

def evaluate(model, eval_data, eval_tensors, device, batch_size=128):
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

    model.train()
    phase_accuracy = {p: round(s["correct"] / max(s["total"], 1), 4) for p, s in phase_stats.items()}
    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
        "phase_accuracy": phase_accuracy,
        "n_eval": total,
    }


# ── Training ──

def train(train_tensors, eval_data, eval_tensors, device):
    model = ChessTransformerFused().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_train = train_tensors["move_idx"].shape[0]
    steps_per_epoch = n_train // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(int(total_steps * WARMUP_FRAC), 1)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler('cuda')

    # Initial eval
    print("\n  Initial evaluation...")
    ev0 = evaluate(model, eval_data, eval_tensors, device)
    print(f"  Random baseline: acc={ev0['accuracy']:.1%} top3={ev0['top3_accuracy']:.1%}")

    best_acc = 0.0
    best_state = None
    global_step = 0
    results_log = []

    model.train()
    t_start = time.time()

    for epoch in range(EPOCHS):
        perm = torch.randperm(n_train)
        epoch_pl = epoch_vl = 0.0
        epoch_batches = 0

        print(f"\n  Epoch {epoch + 1}/{EPOCHS} ({steps_per_epoch} steps)")

        for i in range(0, n_train, BATCH_SIZE):
            indices = perm[i:i + BATCH_SIZE]
            if len(indices) < 2:
                continue

            batch_input = get_batch_input(train_tensors, indices, "fused", device)
            targets = train_tensors["move_idx"][indices].to(device)
            wdl_targets = train_tensors["wdl"][indices].float().to(device)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                policy_loss = F.cross_entropy(result["policy_logits"], targets)
                value_log_probs = F.log_softmax(result["value_logits"], dim=-1)
                value_loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")
                loss = (policy_loss + VALUE_WEIGHT * value_loss) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (global_step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_pl += policy_loss.item()
            epoch_vl += value_loss.item()
            epoch_batches += 1
            global_step += 1

            # Logging
            if epoch_batches % LOG_INTERVAL == 0:
                elapsed = time.time() - t_start
                tp = (global_step * BATCH_SIZE) / elapsed
                lr_now = scheduler.get_last_lr()[0]
                print(f"    [{epoch+1}] step {epoch_batches}/{steps_per_epoch} "
                      f"| pl={epoch_pl/epoch_batches:.4f} "
                      f"| vl={epoch_vl/epoch_batches:.4f} "
                      f"| lr={lr_now:.2e} | {tp:.0f} pos/s", flush=True)

            # Periodic eval
            if global_step % EVAL_INTERVAL == 0:
                ev = evaluate(model, eval_data, eval_tensors, device)
                print(f"    ** EVAL step {global_step}: acc={ev['accuracy']:.1%} "
                      f"top3={ev['top3_accuracy']:.1%} sf_rank={ev['mean_sf_rank']:.1f} "
                      f"val={ev['value_accuracy']:.1%}")
                for phase, acc in sorted(ev['phase_accuracy'].items()):
                    print(f"       {phase}: {acc:.1%}")

                results_log.append({
                    "step": global_step,
                    "epoch": epoch + 1,
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()},
                })

                if ev["accuracy"] > best_acc:
                    best_acc = ev["accuracy"]
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"    ** New best: {best_acc:.1%}")

                # Save intermediate results
                with open(OUTPUT_DIR / "training_log.json", "w") as f:
                    json.dump(results_log, f, indent=2)

                model.train()

            # Checkpoint
            if global_step % SAVE_INTERVAL == 0:
                ckpt_path = OUTPUT_DIR / "checkpoints" / f"step_{global_step}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                print(f"    Saved checkpoint: {ckpt_path.name}")

        # End of epoch eval
        ev = evaluate(model, eval_data, eval_tensors, device)
        elapsed = time.time() - t_start
        tp = (global_step * BATCH_SIZE) / elapsed
        print(f"\n  Epoch {epoch+1} done: acc={ev['accuracy']:.1%} "
              f"top3={ev['top3_accuracy']:.1%} sf_rank={ev['mean_sf_rank']:.1f} "
              f"val={ev['value_accuracy']:.1%} | {tp:.0f} pos/s ({elapsed:.0f}s)")
        for phase, acc in sorted(ev['phase_accuracy'].items()):
            print(f"    {phase}: {acc:.1%}")

        results_log.append({
            "step": global_step,
            "epoch": epoch + 1,
            "epoch_end": True,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()},
        })

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.train()

    total_time = time.time() - t_start

    # Save best model
    if best_state is not None:
        best_path = OUTPUT_DIR / "best_model.pt"
        torch.save(best_state, best_path)
        print(f"\n  Best model saved ({best_acc:.1%}): {best_path}")

    # Save final model
    torch.save(model.state_dict(), OUTPUT_DIR / "final_model.pt")

    return results_log, total_time, n_params, best_acc


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*70}")
    print(f" EXP070: LARGE-SCALE LICHESS-SF TRAINING (A40)")
    print(f"{'='*70}")
    print(f"  Device: {device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'})")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print(f"  Model: FusedBoardEncoder {ENCODER_DIM}d → {HIDDEN_DIM}d, {NUM_LAYERS}L, {NUM_HEADS}H")
    print(f"  Data: {TRAIN_POSITIONS:,} train, {EVAL_POSITIONS:,} eval (min_depth={MIN_DEPTH})")
    print(f"  Training: {EPOCHS} epochs, batch={BATCH_SIZE}, accum={ACCUM_STEPS}, lr={LR}")
    print(f"  Effective batch: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"  Seed: {SEED}")
    print()

    torch.manual_seed(SEED)
    random.seed(SEED)

    # Load data (streams from HF, caches locally)
    print("Loading data...")
    t0 = time.time()
    train_tensors, eval_data, eval_tensors = load_training_data(
        n_train=TRAIN_POSITIONS, n_eval=EVAL_POSITIONS,
        encoder_type="fused", min_depth=MIN_DEPTH, seed=SEED,
    )
    data_time = time.time() - t0
    print(f"  Data loaded in {data_time:.0f}s")
    print(f"  Train: {train_tensors['move_idx'].shape[0]:,} positions")
    print(f"  Eval: {len(eval_data):,} positions")

    # Train
    results_log, total_time, n_params, best_acc = train(
        train_tensors, eval_data, eval_tensors, device
    )

    # Final summary
    print(f"\n{'='*70}")
    print(f" FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Model params: {n_params:,}")
    print(f"  Best accuracy: {best_acc:.1%}")
    print(f"  Total training time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Throughput: {(TRAIN_POSITIONS * EPOCHS) / total_time:.0f} pos/s")

    if results_log:
        final = results_log[-1]
        print(f"  Final: acc={final['accuracy']:.1%} top3={final['top3_accuracy']:.1%} "
              f"sf_rank={final['mean_sf_rank']:.1f} val={final['value_accuracy']:.1%}")

    # Save summary
    summary = {
        "experiment": "exp070_lichess_sf_a40",
        "hypothesis": "12L/512d FusedEncoder on 2M lichess-sf positions",
        "dataset": "avewright/chess-positions-lichess-sf",
        "hardware": "NVIDIA A40 46GB",
        "model_params": n_params,
        "train_positions": TRAIN_POSITIONS,
        "eval_positions": EVAL_POSITIONS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "lr": LR,
        "num_layers": NUM_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "best_accuracy": round(best_acc, 4),
        "total_time_s": round(total_time),
        "seed": SEED,
        "results_log": results_log,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
