"""Minimal training test - simplified."""
import sys, time, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from chess_transformer_factory import build_model, ChessTransformerConfig
from data_loader import StreamingHFChessLoader

DEVICE = "cuda"

# Build model
print("Building model...", flush=True)
config = ChessTransformerConfig(
    encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
    ffn_ratio=4, dropout=0.1, policy_head_dim=256, value_hidden=256,
)
model = build_model(config)
n = sum(p.numel() for p in model.parameters())
print(f"Model: {n/1e6:.1f}M params", flush=True)
model.to(DEVICE).train()
print("Model on CUDA", flush=True)

optimizer = AdamW(model.parameters(), lr=2e-4)
scaler = GradScaler("cuda")
print("Optimizer ready", flush=True)

# Create loader
print("Creating loader...", flush=True)
loader = StreamingHFChessLoader(
    repo_id="avewright/chess-positions-lichess-sf",
    batch_size=256, encoder_type="fused", device=DEVICE,
    seed=42, drop_last=True, file_pattern="src", max_files=1,
)
print("Loader ready", flush=True)

# Training
print("Starting loop...", flush=True)
t0 = time.time()
step = 0
accum = 0
ACCUM_STEPS = 4
optimizer.zero_grad()

for batch_input, move_targets, wdl_targets in loader:
    print(f"  Got batch {accum+1}/{ACCUM_STEPS}, shapes: {move_targets.shape}", flush=True)

    with autocast("cuda", dtype=torch.float16):
        result = model(batch_input)
        ce = F.cross_entropy(result["policy_logits"], move_targets)
        val = F.cross_entropy(result["value_logits"], wdl_targets)
        loss = (ce + 0.5 * val) / ACCUM_STEPS

    print(f"    Loss: {loss.item():.4f}", flush=True)
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
        print(f"  Step {step} complete, ce={ce.item():.4f}", flush=True)

        if step >= 3:
            break

print(f"\nDone after {step} steps in {time.time()-t0:.1f}s", flush=True)
