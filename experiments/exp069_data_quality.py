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

from chess_features import board_to_fused_token_ids
from chess_model import FusedBoardEncoder
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)

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

PARQUET_GLOB = "outputs/lichess_cache/datasets--Lichess--chess-position-evaluations/snapshots/*/data/train-00000-of-00017.parquet"


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


# ── Data quality features ──

def compute_sample_weight(cp, n_legal, line):
    """Compute sample weight from data quality features.

    Returns (weight, features_dict).
    """
    is_forced = n_legal <= 1

    # cp_gap: approximate from PV line if it has multiple moves
    # The parquet `line` field is the PV (sequence of best moves), not top-k.
    # We only have the best move's CP, not the 2nd best.
    # Use heuristic: positions with extreme cp (|cp| > 300) are often obvious.
    cp_gap_heuristic = 0
    if cp is not None:
        abs_cp = abs(int(cp))
        if abs_cp > 500:
            cp_gap_heuristic = 2  # likely forced/obvious
        elif abs_cp > 200:
            cp_gap_heuristic = 1  # probably clear
        else:
            cp_gap_heuristic = 0  # contested

    # Weight assignment
    if is_forced:
        weight = 0.2  # trivial
    elif cp_gap_heuristic == 2:
        weight = 0.6  # obvious position
    elif cp_gap_heuristic == 1:
        weight = 1.0  # clear but interesting
    else:
        weight = 0.9  # contested — moderate up-weight vs uniform

    features = {
        "is_forced": is_forced,
        "n_legal": n_legal,
        "cp_gap_heuristic": cp_gap_heuristic,
        "weight": weight,
    }
    return weight, features


def cp_to_wdl(cp, mate=None):
    if mate is not None:
        return (1.0, 0.0, 0.0) if mate > 0 else (0.0, 0.0, 1.0)
    if cp is None:
        return (0.33, 0.34, 0.33)
    k = 1.0 / 111.7
    win = 1.0 / (1.0 + math.exp(-k * cp))
    loss = 1.0 - win
    draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
    total = win + draw + loss
    return (win / total, draw / total, loss / total)


def fen_to_phase(fen):
    board_part = fen.split()[0]
    n = sum(1 for c in board_part if c.isalpha() and c.lower() != 'k')
    if n >= 14:
        return "opening"
    elif n >= 6:
        return "middlegame"
    return "endgame"


def load_data(n_train, n_eval, min_depth=15, seed=42):
    """Load data with quality weights."""
    import glob as globmod
    import pandas as pd
    import numpy as np

    parquet_files = globmod.glob(str(Path(__file__).resolve().parent.parent / PARQUET_GLOB))
    if not parquet_files:
        raise FileNotFoundError("No parquet file found.")

    parquet_path = parquet_files[0]
    total_needed = n_train + n_eval

    print(f"  Loading from local parquet: {Path(parquet_path).name}")
    t0 = time.time()

    df = pd.read_parquet(parquet_path, columns=["fen", "line", "depth", "cp", "mate"])
    print(f"  Read {len(df):,} rows in {time.time()-t0:.1f}s")
    if "depth" in df.columns:
        df = df[df["depth"].notna() & (df["depth"] >= min_depth)]
    df = df[df["line"].notna() & (df["line"].str.len() > 0)]
    print(f"  After depth/line filter: {len(df):,}")
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)

    fused_ids_list = []
    turn_list = []
    castling_list = []
    ep_list = []
    move_idx_list = []
    wdl_list = []
    weight_list = []
    phase_list = []
    eval_boards = []
    eval_moves = []

    n_collected = 0
    t1 = time.time()
    for row_idx in range(len(df)):
        if n_collected >= total_needed:
            break
        try:
            row = df.iloc[row_idx]
            line = row["line"]
            best_move_uci = line.split()[0]
            if best_move_uci not in UCI_TO_IDX:
                continue

            fen = row["fen"]
            board = chess.Board(fen)
            move = chess.Move.from_uci(best_move_uci)
            if move not in board.legal_moves:
                continue

            cp_val = row["cp"]
            mate_val = row["mate"]
            n_legal = len(list(board.legal_moves))
            weight, _ = compute_sample_weight(
                int(cp_val) if pd.notna(cp_val) else None,
                n_legal, line,
            )

            ft = board_to_fused_token_ids(board)
            fused_ids_list.append(ft["fused_ids"])
            turn_list.append(ft["turn"])
            castling_list.append(ft["castling"])
            ep_list.append(ft["ep_file"])
            move_idx_list.append(UCI_TO_IDX[best_move_uci])
            wdl_list.append(cp_to_wdl(
                int(cp_val) if pd.notna(cp_val) else None,
                int(mate_val) if pd.notna(mate_val) else None,
            ))
            weight_list.append(weight)
            phase_list.append(fen_to_phase(fen))

            if n_collected < n_eval:
                eval_boards.append(board)
                eval_moves.append(move)

            n_collected += 1
            if n_collected % 50000 == 0:
                elapsed = time.time() - t1
                rate = n_collected / max(elapsed, 0.1)
                print(f"    {n_collected:,} loaded ({rate:.0f} pos/s)...", flush=True)
        except Exception:
            continue

    tok_time = time.time() - t1
    print(f"  Tokenized {n_collected:,} in {tok_time:.1f}s ({n_collected/max(tok_time,0.1):.0f} pos/s)")

    del df
    gc.collect()

    # Split: first n_eval -> eval, rest -> train
    n_eval_actual = min(n_eval, len(eval_boards))
    n_train_actual = min(n_train, n_collected - n_eval_actual)

    # Weight statistics
    import statistics as stat_mod
    print(f"  Weight stats: mean={stat_mod.mean(weight_list):.3f} "
          f"min={min(weight_list):.2f} max={max(weight_list):.2f}")
    w_counts = {}
    for w in weight_list:
        w_counts[w] = w_counts.get(w, 0) + 1
    for w in sorted(w_counts):
        pct = w_counts[w] / n_collected * 100
        print(f"    weight={w:.1f}: {w_counts[w]:,} ({pct:.1f}%)")

    eval_data = []
    for i in range(n_eval_actual):
        eval_data.append({
            "board": eval_boards[i], "move": eval_moves[i],
            "wdl": wdl_list[i], "phase": phase_list[i],
        })

    train_tensors = {
        "fused_ids": torch.stack(fused_ids_list[n_eval_actual:n_eval_actual + n_train_actual]),
        "turn": torch.stack(turn_list[n_eval_actual:n_eval_actual + n_train_actual]),
        "castling": torch.stack(castling_list[n_eval_actual:n_eval_actual + n_train_actual]),
        "ep_file": torch.stack(ep_list[n_eval_actual:n_eval_actual + n_train_actual]),
        "move_idx": torch.tensor(move_idx_list[n_eval_actual:n_eval_actual + n_train_actual], dtype=torch.long),
        "wdl": torch.tensor(wdl_list[n_eval_actual:n_eval_actual + n_train_actual], dtype=torch.float32),
        "weight": torch.tensor(weight_list[n_eval_actual:n_eval_actual + n_train_actual], dtype=torch.float32),
    }

    eval_tensors = {
        "fused_ids": torch.stack(fused_ids_list[:n_eval_actual]),
        "turn": torch.stack(turn_list[:n_eval_actual]),
        "castling": torch.stack(castling_list[:n_eval_actual]),
        "ep_file": torch.stack(ep_list[:n_eval_actual]),
    }

    del fused_ids_list, turn_list, castling_list, ep_list
    del move_idx_list, wdl_list, weight_list, eval_boards, eval_moves
    gc.collect()

    print(f"  Train: {n_train_actual:,}, Eval: {n_eval_actual:,}")
    return train_tensors, eval_data, eval_tensors


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
    n_train = train_tensors["fused_ids"].shape[0]
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

        batch_input = {
            "fused_ids": train_tensors["fused_ids"][indices].to(device),
            "turn": train_tensors["turn"][indices].to(device),
            "castling": train_tensors["castling"][indices].to(device),
            "ep_file": train_tensors["ep_file"][indices].to(device),
        }
        targets = train_tensors["move_idx"][indices].to(device)
        wdl_targets = train_tensors["wdl"][indices].to(device)

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

            batch_input = {
                "fused_ids": eval_tensors["fused_ids"][idx].to(device),
                "turn": eval_tensors["turn"][idx].to(device),
                "castling": eval_tensors["castling"][idx].to(device),
                "ep_file": eval_tensors["ep_file"][idx].to(device),
            }

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

    train_tensors, eval_data, eval_tensors = load_data(
        TRAIN_POSITIONS, EVAL_POSITIONS, min_depth=MIN_DEPTH, seed=SEED,
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
