"""Profile: is the bottleneck model forward/backward or data loading?"""
import os, sys, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

from chess_transformer_factory import build_model, ChessTransformerConfig

DEVICE = torch.device("cuda")

config = ChessTransformerConfig(
    encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
    ffn_ratio=4, dropout=0.1, policy_head_dim=256, value_hidden=256,
)

print("Building model...", flush=True)
model = build_model(config)
model.to(DEVICE).train()

optimizer = AdamW(model.parameters(), lr=2e-4)
scaler = GradScaler('cuda')
bs = 256

# Synthetic data on GPU — no data loading overhead
fused_ids = torch.randint(0, 13, (bs, 64), device=DEVICE)
turn = torch.randint(0, 2, (bs,), device=DEVICE)
castling = torch.randint(0, 16, (bs,), device=DEVICE)
ep_file = torch.randint(0, 9, (bs,), device=DEVICE) # 0-8, 8=no ep
move_targets = torch.randint(0, 1968, (bs,), device=DEVICE)
wdl_targets = torch.randn(bs, 3, device=DEVICE).softmax(-1)

batch_input = {"fused_ids": fused_ids, "turn": turn, "castling": castling, "ep_file": ep_file}

# Warmup
for _ in range(3):
    with autocast('cuda', dtype=torch.float16):
        result = model(batch_input)
        loss = F.cross_entropy(result["policy_logits"], move_targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

torch.cuda.synchronize()
print("Warmup done", flush=True)

# Timed: 50 forward+backward steps
t0 = time.time()
N_STEPS = 50
for i in range(N_STEPS):
    with autocast('cuda', dtype=torch.float16):
        result = model(batch_input)
        ce = F.cross_entropy(result["policy_logits"], move_targets)
        val = F.cross_entropy(result["value_logits"], wdl_targets)
        loss = (ce + 0.5 * val) / 4
    scaler.scale(loss).backward()
    if (i + 1) % 4 == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

torch.cuda.synchronize()
elapsed = time.time() - t0
pos = N_STEPS * bs
print(f"\n{N_STEPS} steps x {bs} batch = {pos:,} positions in {elapsed:.2f}s")
print(f"  {pos/elapsed:.0f} pos/s")
print(f"  {elapsed/N_STEPS*1000:.1f} ms/step")
print(f"  VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
