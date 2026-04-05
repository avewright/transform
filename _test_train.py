"""Minimal training test with streaming loader."""
import sys, time
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

from chess_transformer_factory import build_model, ChessTransformerConfig

config = ChessTransformerConfig(
    encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
    ffn_ratio=4, dropout=0.1, policy_head_dim=256, value_hidden=256,
)
model = build_model(config)
n = sum(p.numel() for p in model.parameters())
print(f"Model: {n/1e6:.1f}M params")
model.to("cuda")
model.train()

optimizer = AdamW(model.parameters(), lr=2e-4)
scaler = GradScaler("cuda")

from data_loader import StreamingHFChessLoader

loader = StreamingHFChessLoader(
    repo_id="avewright/chess-positions-lichess-sf",
    batch_size=256, encoder_type="fused", device="cuda",
    seed=42, drop_last=True, file_pattern="src", max_files=1,
)

print("Starting training loop...")
t0 = time.time()
step = 0
accum = 0
ACCUM_STEPS = 4

for batch_input, move_targets, wdl_targets in loader:
    with autocast("cuda", dtype=torch.float16):
        result = model(batch_input)
        ce = F.cross_entropy(result["policy_logits"], move_targets)
        val = F.cross_entropy(result["value_logits"], wdl_targets)
        loss = (ce + 0.5 * val) / ACCUM_STEPS

    scaler.scale(loss).backward()
    accum += 1

    if accum >= ACCUM_STEPS:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        accum = 0
        step += 1

        if step % 10 == 0:
            elapsed = time.time() - t0
            pos = step * 256 * ACCUM_STEPS
            print(f"  step={step}, pos={pos:,}, ce={ce.item():.4f}, val={val.item():.4f}, "
                  f"{pos/elapsed:.0f} pos/s, elapsed={elapsed:.1f}s")

        if step >= 50:
            break

elapsed = time.time() - t0
pos = step * 256 * ACCUM_STEPS
print(f"\nDone: {step} steps, {pos:,} positions in {elapsed:.1f}s ({pos/elapsed:.0f} pos/s)")
