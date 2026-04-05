"""Per-step timing: isolate forward, backward, optimizer."""
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

model = build_model(config)
model.to(DEVICE).train()
optimizer = AdamW(model.parameters(), lr=2e-4)
scaler = GradScaler('cuda')

# Try multiple batch sizes
for bs in [16, 64, 128, 256]:
    fused_ids = torch.randint(0, 13, (bs, 64), device=DEVICE)
    turn = torch.randint(0, 2, (bs,), device=DEVICE)
    castling = torch.randint(0, 16, (bs,), device=DEVICE)
    ep_file = torch.randint(0, 9, (bs,), device=DEVICE)
    move_targets = torch.randint(0, 1968, (bs,), device=DEVICE)
    wdl_targets = torch.randn(bs, 3, device=DEVICE).softmax(-1)
    batch_input = {"fused_ids": fused_ids, "turn": turn, "castling": castling, "ep_file": ep_file}

    # 1 warmup
    optimizer.zero_grad()
    with autocast('cuda', dtype=torch.float16):
        r = model(batch_input)
        loss = F.cross_entropy(r["policy_logits"], move_targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed: 3 steps with sync
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.time()
        
        with autocast('cuda', dtype=torch.float16):
            r = model(batch_input)
            ce = F.cross_entropy(r["policy_logits"], move_targets)
            val = F.cross_entropy(r["value_logits"], wdl_targets)
            loss = ce + 0.5 * val
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    
    avg = sum(times) / len(times)
    pos_s = bs / avg
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"bs={bs:4d}: {avg*1000:.0f}ms/step, {pos_s:.0f} pos/s, VRAM={vram:.2f}GB  times={[f'{t*1000:.0f}ms' for t in times]}", flush=True)

print("\nDone!", flush=True)
