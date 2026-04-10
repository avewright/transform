"""exp170: Medium-scale confirmation that SwiGLU+RelBias gains hold at scale.

exp169 showed SwiGLU+RelBias (variant C) was the dominant architecture:
  - +2pp top-1, +5pp top-3 over baseline at 3.3M params
  - C@step1000 already beats A@step3000 (3x sample efficiency)

This experiment tests whether those gains persist at medium scale (~25M params).
Two variants only — we're confirming, not exploring:
  F. BASELINE_MED  — Vanilla transformer (GELU, abs pos embed, SpatialPolicyHead)
  G. RELBIAS_MED   — SwiGLU + Chess Relative Bias (the exp169 winner)

Model: 8L / 512d / 8H = ~25M params
Training: 5000 steps on 2 shards (2M positions), eval on shard 2

Usage:
  python experiments/exp170_medium_scale_confirm.py
  python experiments/exp170_medium_scale_confirm.py --variant G --steps 3000
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MOVE_VOCAB_VERSION'] = 'compact'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import SpatialPolicyHead, _build_move_square_indices
from chess_model import FusedBoardEncoder
from move_vocab import VOCAB_SIZE, LEGACY_UCI_TO_IDX, legacy_to_compact_map
from data_loader import board_array_to_fused, ep_square_to_file, compute_wdl

ROOT = Path(__file__).resolve().parent.parent
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Compact vocab remap ──
def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t

REMAP = build_remap_tensor()


# ── SwiGLU FFN ──
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()
        inner = int(ffn_dim * 2 / 3)
        self.w_gate = nn.Linear(d_model, inner)
        self.w_up = nn.Linear(d_model, inner)
        self.w_down = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ── Chess Relative Bias ──
class ChessRelBias(nn.Module):
    def __init__(self, num_heads, n_ctx=4):
        super().__init__()
        self.num_heads = num_heads
        self.n_ctx = n_ctx
        seq_len = n_ctx + 64

        rank_dist = torch.zeros(64, 64, dtype=torch.long)
        file_dist = torch.zeros(64, 64, dtype=torch.long)
        same_diag = torch.zeros(64, 64, dtype=torch.long)
        knight_rel = torch.zeros(64, 64, dtype=torch.long)

        for i in range(64):
            ri, fi = i // 8, i % 8
            for j in range(64):
                rj, fj = j // 8, j % 8
                dr, df = abs(ri - rj), abs(fi - fj)
                rank_dist[i, j] = dr
                file_dist[i, j] = df
                same_diag[i, j] = int(dr == df and dr > 0)
                knight_rel[i, j] = int(sorted([dr, df]) == [1, 2])

        self.rank_bias = nn.Embedding(8, num_heads)
        self.file_bias = nn.Embedding(8, num_heads)
        self.diag_bias = nn.Embedding(2, num_heads)
        self.knight_bias = nn.Embedding(2, num_heads)
        self.ctx_bias = nn.Parameter(torch.zeros(num_heads, seq_len, seq_len))

        self.register_buffer("rank_dist", rank_dist)
        self.register_buffer("file_dist", file_dist)
        self.register_buffer("same_diag", same_diag)
        self.register_buffer("knight_rel", knight_rel)

    def forward(self, seq_len):
        nc = self.n_ctx
        bias = self.ctx_bias[:, :seq_len, :seq_len].clone()
        rb = self.rank_bias(self.rank_dist).permute(2, 0, 1)
        fb = self.file_bias(self.file_dist).permute(2, 0, 1)
        db = self.diag_bias(self.same_diag).permute(2, 0, 1)
        kb = self.knight_bias(self.knight_rel).permute(2, 0, 1)
        bias[:, nc:nc+64, nc:nc+64] = rb + fb + db + kb
        return bias


# ── SwiGLU Transformer Layer ──
class SwiGLUTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_bias)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


# ── Medium-Scale Chess Transformer ──
class MediumChessTransformer(nn.Module):
    def __init__(self, d_model=512, n_layers=8, n_heads=8, ffn_ratio=4,
                 dropout=0.05, policy_head_dim=256,
                 use_swiglu=False, use_rel_bias=False):
        super().__init__()
        self.d_model = d_model
        self.n_ctx = 4

        # Encoder outputs 256d, project up to d_model if needed
        encoder_dim = 256
        self.encoder = FusedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, d_model) if encoder_dim != d_model else nn.Identity()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, d_model) * 0.02)

        self.rel_bias = ChessRelBias(n_heads, n_ctx=4) if use_rel_bias else None

        ffn_dim = d_model * ffn_ratio
        if use_swiglu:
            self.layers = nn.ModuleList([
                SwiGLUTransformerLayer(d_model, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ])
            self._custom_layers = True
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=ffn_dim, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self._custom_layers = False

        self.norm = nn.LayerNorm(d_model)

        self.policy_head = SpatialPolicyHead(
            d_model, n_ctx_tokens=4, head_dim=policy_head_dim,
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, board_input):
        hidden = self.encoder(board_input)
        hidden = self.input_proj(hidden)
        B = hidden.shape[0]

        hidden = torch.cat([self.cls_token.expand(B, -1, -1), hidden], dim=1)
        hidden = hidden + self.pos_embed

        if self._custom_layers:
            bias = self.rel_bias(hidden.shape[1]) if self.rel_bias else None
            if bias is not None:
                bias = bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(
                    B * self.layers[0].attn.num_heads, hidden.shape[1], hidden.shape[1])
            for layer in self.layers:
                hidden = layer(hidden, attn_bias=bias)
        else:
            hidden = self.transformer(hidden)

        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]

        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }


# ── Data ──
def load_shard(shard_idx=0):
    path = SHARD_DIR / f"shard_{shard_idx:05d}.pt"
    return torch.load(path, weights_only=True, map_location="cpu")


def prepare_batch(data, indices, device):
    ba = data["board_array"][indices]
    turn = data["turn"][indices]
    castling = data["castling"][indices]
    ep = data["ep_square"][indices]
    move_idx = data["move_idx"][indices]
    cp = data["cp"][indices]
    mate = data["mate"][indices]

    fused_ids = board_array_to_fused(ba)
    ep_file = ep_square_to_file(ep)
    move_compact = REMAP[move_idx.long()]
    valid = move_compact >= 0
    wdl = compute_wdl(cp.float(), mate.float())

    board_input = {
        "fused_ids": fused_ids.to(device),
        "turn": turn.long().to(device),
        "castling": castling.long().to(device),
        "ep_file": ep_file.long().to(device),
    }
    return board_input, move_compact[valid].to(device), wdl[valid].to(device), valid


@torch.no_grad()
def evaluate(model, eval_data, device, n_eval=5000):
    model.eval()
    ba = eval_data["board_array"][:n_eval]
    turn = eval_data["turn"][:n_eval]
    castling = eval_data["castling"][:n_eval]
    ep = eval_data["ep_square"][:n_eval]
    move_idx = eval_data["move_idx"][:n_eval]

    fused_ids = board_array_to_fused(ba)
    ep_file = ep_square_to_file(ep)
    move_compact = REMAP[move_idx.long()]
    valid = move_compact >= 0

    total_correct_1 = total_correct_3 = total_n = 0
    total_loss = 0.0

    bs = 256
    for start in range(0, valid.sum().item(), bs):
        end = min(start + bs, valid.sum().item())
        valid_indices = valid.nonzero(as_tuple=True)[0][start:end]
        bi = {
            "fused_ids": fused_ids[valid_indices].to(device),
            "turn": turn[valid_indices].long().to(device),
            "castling": castling[valid_indices].long().to(device),
            "ep_file": ep_file[valid_indices].long().to(device),
        }
        targets = move_compact[valid_indices].to(device)

        with autocast("cuda", dtype=torch.float16):
            out = model(bi)
        logits = out["policy_logits"].float()

        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        _, top3 = logits.topk(3, dim=-1)
        total_correct_1 += (top3[:, 0] == targets).sum().item()
        total_correct_3 += (top3 == targets.unsqueeze(1)).any(dim=1).sum().item()
        total_n += targets.shape[0]

    model.train()
    return {
        "loss": total_loss / max(total_n, 1),
        "top1": total_correct_1 / max(total_n, 1),
        "top3": total_correct_3 / max(total_n, 1),
        "n": total_n,
    }


# ── Variants ──
VARIANTS = {
    "F": {"name": "BASELINE_MED", "use_swiglu": False, "use_rel_bias": False},
    "G": {"name": "RELBIAS_MED",  "use_swiglu": True,  "use_rel_bias": True},
}


def train_variant(variant_key, max_steps=5000, eval_every=500, batch_size=64,
                  lr=2e-4, seed=42, n_shards=2):
    cfg = VARIANTS[variant_key]
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"  ABLATION {variant_key}: {name}")
    print(f"  swiglu={cfg['use_swiglu']} rel_bias={cfg['use_rel_bias']}")
    print(f"  steps={max_steps} bs={batch_size} lr={lr} shards={n_shards}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)

    model = MediumChessTransformer(
        d_model=512, n_layers=8, n_heads=8, ffn_ratio=4,
        dropout=0.05, policy_head_dim=256,
        use_swiglu=cfg["use_swiglu"],
        use_rel_bias=cfg["use_rel_bias"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()

    # Load training data (multiple shards)
    print(f"  Loading {n_shards} training shards...")
    all_shards = []
    for i in range(n_shards):
        all_shards.append(load_shard(i))
    # Concatenate all shard data
    train_data = {}
    for key in all_shards[0].keys():
        train_data[key] = torch.cat([s[key] for s in all_shards], dim=0)
    del all_shards
    n_train = train_data["board_array"].shape[0]
    print(f"  Training positions: {n_train:,}")

    print("  Loading eval data (shard 2)...")
    eval_data = load_shard(2)

    results = {"variant": variant_key, "name": name, "params": n_params, "evals": []}
    step = 0
    t0 = time.time()

    while step < max_steps:
        perm = torch.randperm(n_train)
        for i in range(0, n_train - batch_size, batch_size):
            if step >= max_steps:
                break

            indices = perm[i:i+batch_size]
            board_input, targets, wdl, valid = prepare_batch(train_data, indices, DEVICE)
            if targets.shape[0] < 2:
                continue

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.float16):
                out = model(board_input)
                policy_logits = out["policy_logits"][valid]
                p_loss = F.cross_entropy(policy_logits, targets)
                v_loss = F.cross_entropy(out["value_logits"][valid], wdl)
                loss = p_loss + 0.5 * v_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1

            if step % 100 == 0:
                elapsed = time.time() - t0
                pos_per_sec = step * batch_size / elapsed
                print(f"  step={step:5d} p_loss={p_loss.item():.3f} "
                      f"v_loss={v_loss.item():.3f} "
                      f"{pos_per_sec:.0f} pos/s")

            if step % eval_every == 0 or step == max_steps:
                ev = evaluate(model, eval_data, DEVICE, n_eval=5000)
                results["evals"].append({"step": step, **ev})
                print(f"  >>> EVAL step={step}: top1={ev['top1']:.4f} "
                      f"top3={ev['top3']:.4f} loss={ev['loss']:.3f}")

    elapsed = time.time() - t0
    results["total_time_s"] = elapsed
    results["final_pos_per_s"] = max_steps * batch_size / elapsed
    print(f"\n  Completed {name} in {elapsed:.1f}s ({results['final_pos_per_s']:.0f} pos/s)")

    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default=None, help="F or G")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Compact vocab size: {VOCAB_SIZE}")

    variants = [args.variant] if args.variant else list(VARIANTS.keys())
    all_results = {}

    for v in variants:
        res = train_variant(v, max_steps=args.steps, eval_every=args.eval_every,
                           batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                           n_shards=args.shards)
        all_results[v] = res

    # Summary
    print(f"\n{'='*80}")
    print(f"  MEDIUM-SCALE CONFIRMATION (step {args.steps})")
    print(f"{'='*80}")
    print(f"{'Variant':<12} {'Name':<16} {'Params':>10} {'Top-1':>8} {'Top-3':>8} "
          f"{'Loss':>8} {'Time(s)':>8} {'pos/s':>8}")
    print("-" * 88)
    for v, res in all_results.items():
        final = res["evals"][-1]
        print(f"{v:<12} {res['name']:<16} {res['params']:>10,} {final['top1']:>7.2%} "
              f"{final['top3']:>7.2%} {final['loss']:>8.3f} "
              f"{res['total_time_s']:>7.1f}s {res['final_pos_per_s']:>8.0f}")

    # Save
    out_path = ROOT / "outputs" / "exp170_medium_scale_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
