"""exp066: Architecture scaling ablation on Lichess data.

Goal: Find the ideal architecture BEFORE committing to full 845M-row training.
Test multiple model configs on a 500K Lichess subset (1 epoch each) to find
which architecture scales best. Results guide the multi-GPU training investment.

Ablation axes:
  A) Width: 256d vs 512d vs 768d (fixed 8L)
  B) Depth: 8L vs 12L vs 16L vs 24L (fixed 512d)
  C) Attention: bidirectional vs causal
  D) FFN ratio: 4x vs 6x (wider MLP)

Each config trains 1 epoch on 500K Lichess positions (depth >= 15, streamed).
~5-8 min per config. Total: ~1 hour for all configs.

Data: Lichess/chess-position-evaluations — 845M positions with SF depth 20+ evals.
  Best move = first move of PV line. WDL from centipawn eval.

Experiment contract:
  - Primary metric: top-1 accuracy on 2500 eval positions
  - Secondary: top-3 accuracy, SF rank, value accuracy, throughput (pos/s)
  - Constraint: must fit in 24GB VRAM with AMP
  - Seed: 42
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

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)

OUTPUT_DIR = Path("outputs/exp066_arch_ablation")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# ── Fixed training config ──
TRAIN_POSITIONS = 500_000
EVAL_POSITIONS = 2500
MIN_DEPTH = 15          # Lichess positions with SF depth >= 15
BATCH_SIZE = 256
LR = 3e-4
WARMUP_FRAC = 0.05
VALUE_WEIGHT = 0.5
SEED = 42

# ── Architecture configs to ablate ──
CONFIGS = {
    # Width ablation (fixed 8L)
    "small_256d_8L": dict(encoder_dim=256, hidden_dim=256, num_layers=8, num_heads=8, ffn_ratio=4, causal=False),
    "medium_512d_8L": dict(encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8, ffn_ratio=4, causal=False),
    "large_768d_8L": dict(encoder_dim=256, hidden_dim=768, num_layers=8, num_heads=8, ffn_ratio=4, causal=False),

    # Depth ablation (fixed 512d)
    "deep_512d_12L": dict(encoder_dim=256, hidden_dim=512, num_layers=12, num_heads=8, ffn_ratio=4, causal=False),
    "deep_512d_16L": dict(encoder_dim=256, hidden_dim=512, num_layers=16, num_heads=8, ffn_ratio=4, causal=False),
    "deep_512d_24L": dict(encoder_dim=256, hidden_dim=512, num_layers=24, num_heads=8, ffn_ratio=4, causal=False),

    # Causal attention (for comparison)
    "causal_512d_8L": dict(encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8, ffn_ratio=4, causal=True),

    # Wider FFN
    "wide_ffn_512d_8L": dict(encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8, ffn_ratio=6, causal=False),
}


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
    """Architecture with configurable width, depth, FFN ratio, and attention type."""

    def __init__(self, encoder_dim=256, hidden_dim=512, num_layers=8,
                 num_heads=8, dropout=0.1, head_dim=256, ffn_ratio=4,
                 causal=False):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        # 68 = 1 CLS + 3 context (turn, castling, ep) + 64 squares
        self.pos_embed = nn.Parameter(torch.randn(1, 68, hidden_dim) * 0.02)
        self.causal = causal

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_ratio, dropout=dropout,
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

    def _get_causal_mask(self, seq_len, device):
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        B = hidden.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embed

        if self.causal:
            mask = self._get_causal_mask(hidden.shape[1], hidden.device)
            hidden = self.transformer(hidden, mask=mask)
        else:
            hidden = self.transformer(hidden)

        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]
        policy_logits = self.policy_head(hidden, cls_hidden)
        value_logits = self.value_head(cls_hidden)
        return {
            "policy_logits": policy_logits,
            "value_logits": value_logits,
        }


# ── Lichess data loading from local parquet ──

PARQUET_GLOB = "outputs/lichess_cache/datasets--Lichess--chess-position-evaluations/snapshots/*/data/train-00000-of-00017.parquet"


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


def load_data(n_train, n_eval, min_depth=15, seed=42):
    """Load and pre-tensorize data from local Lichess parquet shard.

    Pre-converts all boards to token tensors to avoid keeping chess.Board objects
    in memory (which causes Python GC pressure and massive throughput drops).
    
    Returns:
        train_tensors: dict of pre-batched tensors
        eval_data: list of dicts with Board objects (eval needs legal_move_mask)
    """
    import glob as globmod
    import pyarrow.parquet as pq
    import numpy as np
    from chess_features import board_to_token_ids

    parquet_files = globmod.glob(str(Path(__file__).resolve().parent.parent / PARQUET_GLOB))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet file found. Download first:\n"
            f"  python3 -c \"from huggingface_hub import hf_hub_download; "
            f"hf_hub_download('Lichess/chess-position-evaluations', "
            f"'data/train-00000-of-00017.parquet', repo_type='dataset', "
            f"cache_dir='outputs/lichess_cache')\""
        )

    parquet_path = parquet_files[0]
    total_needed = n_train + n_eval

    print(f"  Loading from local parquet: {Path(parquet_path).name}")
    t0 = time.time()

    pf = pq.ParquetFile(parquet_path)
    n_row_groups = pf.metadata.num_row_groups
    rng = np.random.RandomState(seed)
    rg_order = rng.permutation(n_row_groups)

    # Collect raw data: tensorized inputs + move indices + WDL
    piece_ids_list = []
    color_ids_list = []
    turn_list = []
    castling_list = []
    ep_list = []
    move_idx_list = []
    wdl_list = []
    # Keep Board objects only for eval set (need legal_move_mask)
    eval_boards = []
    eval_moves = []

    for rg_idx in rg_order:
        if len(piece_ids_list) >= total_needed:
            break

        table = pf.read_row_group(rg_idx, columns=["fen", "line", "depth", "cp", "mate"])
        fens = table.column("fen").to_pylist()
        lines = table.column("line").to_pylist()
        depths = table.column("depth").to_pylist()
        cps = table.column("cp").to_pylist()
        mates = table.column("mate").to_pylist()

        indices = list(range(len(fens)))
        rng.shuffle(indices)

        for idx in indices:
            if len(piece_ids_list) >= total_needed:
                break

            try:
                d = depths[idx]
                if d is not None and int(d) < min_depth:
                    continue

                line = lines[idx]
                if not line:
                    continue
                best_move_uci = line.split()[0]
                if best_move_uci not in UCI_TO_IDX:
                    continue

                board = chess.Board(fens[idx])
                move = chess.Move.from_uci(best_move_uci)
                if move not in board.legal_moves:
                    continue

                wdl = cp_to_wdl(cps[idx], mates[idx])
                move_idx = UCI_TO_IDX[best_move_uci]

                # Pre-tensorize board
                tokens = board_to_token_ids(board)
                piece_ids_list.append(tokens["piece_ids"])
                color_ids_list.append(tokens["color_ids"])
                turn_list.append(tokens["turn"])
                castling_list.append(tokens["castling"])
                ep_list.append(tokens["ep_file"])
                move_idx_list.append(move_idx)
                wdl_list.append(wdl)

                # Keep Board+move for eval positions only
                n = len(piece_ids_list)
                if n <= n_eval:
                    eval_boards.append(board)
                    eval_moves.append(move)
            except Exception:
                continue

        if len(piece_ids_list) % 50000 < len(fens):
            elapsed = time.time() - t0
            rate = len(piece_ids_list) / max(elapsed, 0.1)
            print(f"    {len(piece_ids_list):,} loaded ({rate:.0f} pos/s)...", flush=True)

    elapsed = time.time() - t0
    n_loaded = len(piece_ids_list)
    print(f"  Loaded {n_loaded:,} in {elapsed:.1f}s ({n_loaded/max(elapsed,0.1):.0f} pos/s)")

    # Shuffle indices
    all_indices = list(range(n_loaded))
    random.Random(seed).shuffle(all_indices)

    # Build eval data (keep Board objects for legal_move_mask)
    eval_indices = all_indices[:n_eval]
    eval_data = []
    for i in eval_indices:
        if i < len(eval_boards):
            eval_data.append({
                "board": eval_boards[i],
                "move": eval_moves[i],
                "wdl": wdl_list[i],
                "piece_ids": piece_ids_list[i],
                "color_ids": color_ids_list[i],
                "turn": turn_list[i],
                "castling": castling_list[i],
                "ep_file": ep_list[i],
            })

    # Build train tensors (stacked, no Board objects needed)
    train_indices = all_indices[n_eval:n_eval + n_train]
    train_tensors = {
        "piece_ids": torch.stack([piece_ids_list[i] for i in train_indices]),
        "color_ids": torch.stack([color_ids_list[i] for i in train_indices]),
        "turn": torch.stack([turn_list[i] for i in train_indices]),
        "castling": torch.stack([castling_list[i] for i in train_indices]),
        "ep_file": torch.stack([ep_list[i] for i in train_indices]),
        "move_idx": torch.tensor([move_idx_list[i] for i in train_indices], dtype=torch.long),
        "wdl": torch.tensor([wdl_list[i] for i in train_indices], dtype=torch.float32),
    }

    # Free the per-item lists to release memory
    del piece_ids_list, color_ids_list, turn_list, castling_list, ep_list
    del move_idx_list, wdl_list, eval_boards, eval_moves
    gc.collect()

    n_train_actual = train_tensors["piece_ids"].shape[0]
    print(f"  Train: {n_train_actual:,} (tensors), Eval: {len(eval_data):,} (with boards)")
    return train_tensors, eval_data


# ── Training ──

def train_one_config(config_name, config, train_data, eval_data, device):
    """Train a single architecture config for 1 epoch. Returns metrics."""
    # Clean up before each config
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(SEED)
    random.seed(SEED)

    vram_free = torch.cuda.mem_get_info()[0] / 1e6
    print(f"\n{'='*60}")
    print(f"  Config: {config_name}")

    model = ChessTransformerV2(
        encoder_dim=config["encoder_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_ratio=config["ffn_ratio"],
        causal=config["causal"],
        dropout=0.1,
        head_dim=256,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    print(f"  Hidden: {config['hidden_dim']}d, Layers: {config['num_layers']}, "
          f"FFN: {config['ffn_ratio']}x, Causal: {config['causal']}")
    print(f"  VRAM free: {vram_free:.0f} MB")
    print(f"{'='*60}")

    # Determine batch size — progressive fallback on OOM
    accum_steps = 1
    effective_bs = BATCH_SIZE
    if n_params > 50_000_000:
        effective_bs = 64
        accum_steps = 4
    elif n_params > 25_000_000:
        effective_bs = 128
        accum_steps = 2

    # Try training with progressive batch size reduction on OOM
    for attempt in range(4):
        try:
            result = _train_loop(config_name, config, model, n_params,
                                 train_data, eval_data, device,
                                 effective_bs, accum_steps)
            return result
        except RuntimeError as e:
            if "out of memory" in str(e) and attempt < 3:
                # Halve batch size and double accum
                old_bs = effective_bs
                effective_bs = max(effective_bs // 2, 16)
                accum_steps = max(BATCH_SIZE // effective_bs, 1)
                print(f"  OOM with bs={old_bs}, retrying with bs={effective_bs}×{accum_steps}...",
                      flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                # Recreate model (optimizer state may be corrupted)
                model = ChessTransformerV2(
                    encoder_dim=config["encoder_dim"],
                    hidden_dim=config["hidden_dim"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    ffn_ratio=config["ffn_ratio"],
                    causal=config["causal"],
                    dropout=0.1,
                    head_dim=256,
                ).to(device)
            else:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  OOM — model too large even with bs={effective_bs}")
                return {"config": config_name, "status": "OOM", "params": n_params}

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"config": config_name, "status": "OOM", "params": n_params}


def _train_loop(config_name, config, model, n_params, train_data, eval_data,
                device, effective_bs, accum_steps):
    """Inner training loop — raises RuntimeError on OOM for retry."""
    print(f"  Effective batch: {effective_bs} × {accum_steps} = {effective_bs * accum_steps}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(train_data) // (effective_bs * accum_steps)
    warmup_steps = max(int(steps * WARMUP_FRAC), 1)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler('cuda')

    # Train 1 epoch
    model.train()
    random.shuffle(train_data)
    total_policy_loss = total_value_loss = 0.0
    n_batches = 0
    n_accum = 0
    t0 = time.time()

    for i in range(0, len(train_data), effective_bs):
        chunk = train_data[i:i + effective_bs]
        if len(chunk) < 2:
            continue

        boards = [d["board"] for d in chunk]
        targets = torch.tensor(
            [move_to_index(d["move"]) for d in chunk], device=device,
        )
        wdl_targets = torch.tensor(
            [d["wdl"] for d in chunk], device=device, dtype=torch.float32,
        )
        batch_input = batch_boards_to_token_ids(boards, device)

        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)
            policy_loss = F.cross_entropy(result["policy_logits"], targets)
            value_log_probs = F.log_softmax(result["value_logits"], dim=-1)
            value_loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")
            loss = (policy_loss + VALUE_WEIGHT * value_loss) / accum_steps

        scaler.scale(loss).backward()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1
        n_accum += 1

        if n_accum >= accum_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            n_accum = 0

        if n_batches % 200 == 0:
            elapsed = time.time() - t0
            throughput = (n_batches * effective_bs) / elapsed
            print(f"    step {n_batches}/{steps} | "
                  f"pl={total_policy_loss/n_batches:.3f} | "
                  f"vl={total_value_loss/n_batches:.3f} | "
                  f"{throughput:.0f} pos/s", flush=True)

    # Flush remaining gradients
    if n_accum > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    train_time = time.time() - t0
    avg_pl = total_policy_loss / max(n_batches, 1)
    avg_vl = total_value_loss / max(n_batches, 1)
    throughput = (n_batches * effective_bs) / train_time

    # Evaluate
    ev = evaluate(model, eval_data, device)

    result = {
        "config": config_name,
        "status": "OK",
        "params": n_params,
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "ffn_ratio": config["ffn_ratio"],
        "causal": config["causal"],
        "policy_loss": round(avg_pl, 4),
        "value_loss": round(avg_vl, 4),
        "accuracy": round(ev["accuracy"], 4),
        "top3_accuracy": round(ev["top3_accuracy"], 4),
        "mean_sf_rank": round(ev["mean_sf_rank"], 2),
        "value_accuracy": round(ev["value_accuracy"], 4),
        "phase_accuracy": ev["phase_accuracy"],
        "train_time_s": round(train_time),
        "throughput_pos_s": round(throughput),
    }

    print(f"  Result: acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} "
          f"sf_rank={ev['mean_sf_rank']:.1f} val_acc={ev['value_accuracy']:.1%}")
    print(f"  Time: {train_time:.0f}s ({throughput:.0f} pos/s) | "
          f"pl={avg_pl:.3f} vl={avg_vl:.3f}")

    # Save checkpoint
    ckpt_path = OUTPUT_DIR / f"{config_name}.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Free memory
    del model, optimizer, scaler, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return result


def evaluate(model, eval_data, device, batch_size=256):
    model.eval()
    correct = top3_correct = total = 0
    sf_rank_sum = 0.0
    val_correct = val_total = 0
    phase_stats = {}

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            true_moves = [d["move"] for d in chunk]

            batch_input = batch_boards_to_token_ids(boards, device)
            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"].float()

            for j, (board, true_move) in enumerate(zip(boards, true_moves)):
                l = logits[j].clone()
                mask = legal_move_mask(board).to(device)
                l[~mask] = float("-inf")

                pred_idx = l.argmax().item()
                true_idx = move_to_index(true_move)
                if pred_idx == true_idx:
                    correct += 1
                topk = l.topk(3).indices.tolist()
                if true_idx in topk:
                    top3_correct += 1

                sorted_indices = l.argsort(descending=True).tolist()
                rank = sorted_indices.index(true_idx) + 1 if true_idx in sorted_indices else len(sorted_indices)
                sf_rank_sum += rank
                total += 1

            # Value accuracy
            wdl_logits = result["value_logits"].float()
            for j, d in enumerate(chunk):
                pred_class = wdl_logits[j].argmax().item()
                true_wdl = d["wdl"]
                true_class = max(range(3), key=lambda k: true_wdl[k])
                if pred_class == true_class:
                    val_correct += 1
                val_total += 1

    return {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
        "phase_accuracy": {},
        "n_eval": total,
    }


# ── Main ──

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f" EXP066: ARCHITECTURE SCALING ABLATION")
    print(f"{'='*60}")
    print(f"  Device: {device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'})")
    print(f"  Train: {TRAIN_POSITIONS:,} positions, Eval: {EVAL_POSITIONS:,}")
    print(f"  Min depth: {MIN_DEPTH}")
    print(f"  Configs: {len(CONFIGS)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print()

    # Load data once, share across all configs
    train_data, eval_data = load_data(
        TRAIN_POSITIONS, EVAL_POSITIONS, min_depth=MIN_DEPTH, seed=SEED,
    )

    # Run each config
    results = []
    t_total = time.time()

    for config_name, config in CONFIGS.items():
        gc.collect()
        torch.cuda.empty_cache()
        result = train_one_config(config_name, config, train_data, eval_data, device)
        results.append(result)

        # Save intermediate results
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - t_total

    # Print summary table
    print(f"\n{'='*80}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Params':>10} {'Acc':>7} {'Top3':>7} {'SFRank':>7} "
          f"{'ValAcc':>7} {'Time':>6} {'Pos/s':>7} {'Status':>6}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True):
        if r["status"] == "OOM":
            print(f"{r['config']:<25} {r['params']:>10,} {'—':>7} {'—':>7} {'—':>7} "
                  f"{'—':>7} {'—':>6} {'—':>7} {'OOM':>6}")
        else:
            print(f"{r['config']:<25} {r['params']:>10,} "
                  f"{r['accuracy']:>6.1%} {r['top3_accuracy']:>6.1%} "
                  f"{r['mean_sf_rank']:>7.1f} {r['value_accuracy']:>6.1%} "
                  f"{r['train_time_s']:>5}s {r['throughput_pos_s']:>6} {'OK':>6}")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")

    # Analysis
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        best = max(ok_results, key=lambda x: x["accuracy"])
        print(f"\nBEST: {best['config']} — {best['accuracy']:.1%} acc, "
              f"{best['params']:,} params, {best['throughput_pos_s']} pos/s")

        # Efficiency metric: accuracy per M params
        print(f"\nEfficiency (acc per M params):")
        for r in sorted(ok_results, key=lambda x: x["accuracy"] / max(x["params"], 1), reverse=True):
            eff = r["accuracy"] / (r["params"] / 1e6)
            print(f"  {r['config']:<25} {eff:.2f}%/M ({r['accuracy']:.1%} / {r['params']/1e6:.1f}M)")

        # Scaling analysis
        print(f"\nWidth scaling (fixed 8L):")
        for name in ["small_256d_8L", "medium_512d_8L", "large_768d_8L"]:
            r = next((x for x in ok_results if x["config"] == name), None)
            if r:
                print(f"  {name}: {r['accuracy']:.1%} ({r['params']/1e6:.1f}M, {r['throughput_pos_s']} pos/s)")

        print(f"\nDepth scaling (fixed 512d):")
        for name in ["medium_512d_8L", "deep_512d_12L", "deep_512d_16L", "deep_512d_24L"]:
            r = next((x for x in ok_results if x["config"] == name), None)
            if r:
                print(f"  {name}: {r['accuracy']:.1%} ({r['params']/1e6:.1f}M, {r['throughput_pos_s']} pos/s)")

    # Save final results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump({
            "best_config": best["config"] if ok_results else None,
            "best_accuracy": best["accuracy"] if ok_results else 0,
            "total_time_s": round(total_time),
            "configs_tested": len(results),
            "configs_ok": len(ok_results),
            "train_positions": TRAIN_POSITIONS,
            "eval_positions": EVAL_POSITIONS,
            "min_depth": MIN_DEPTH,
        }, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
