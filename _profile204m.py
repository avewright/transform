"""Profile 204M default model training speed on RTX 4060."""
import os, sys, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from chess_transformer_factory import build_model, count_parameters

DEVICE = torch.device("cuda")

# Default 204M config
model = build_model(None)  # uses DEFAULT_200M_CONFIG
n_params = count_parameters(model)
print(f"204M Model: {n_params:,} params ({n_params/1e6:.1f}M)", flush=True)
model.to(DEVICE).train()
optimizer = AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler('cuda')

for bs in [8, 16, 32]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    fused_ids = torch.randint(0, 13, (bs, 64), device=DEVICE)
    turn = torch.randint(0, 2, (bs,), device=DEVICE)
    castling = torch.randint(0, 16, (bs,), device=DEVICE)
    ep_file = torch.randint(0, 9, (bs,), device=DEVICE)
    move_targets = torch.randint(0, 1968, (bs,), device=DEVICE)
    wdl_targets = torch.randn(bs, 3, device=DEVICE).softmax(-1)
    batch_input = {"fused_ids": fused_ids, "turn": turn, "castling": castling, "ep_file": ep_file}

    # Warmup
    try:
        optimizer.zero_grad()
        with autocast('cuda', dtype=torch.float16):
            r = model(batch_input)
            loss = F.cross_entropy(r["policy_logits"], move_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  bs={bs}: OOM", flush=True)
            torch.cuda.empty_cache()
            continue
        raise

    # Timed: 3 steps
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
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"  bs={bs:3d}: {avg*1000:.0f}ms/step, {bs/avg:.0f} pos/s, VRAM={vram:.2f}GB", flush=True)

print("Done!", flush=True)
