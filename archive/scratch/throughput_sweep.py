"""Quick throughput sweep to find optimal batch size for A40 46GB."""
import gc
import os
import sys
import time
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.exp101_hf_scale_training import ChessTransformer200M, load_model

DEVICE = "cuda"
CHECKPOINT = "outputs/hf_checkpoint/best_model.pt"

def bench_batch_size(model, batch_size, n_iters=10):
    """Benchmark forward+backward at a given batch size. Returns pos/s or None if OOM."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = GradScaler('cuda')
    
    # Create dummy batch
    batch_input = {
        "turn": torch.zeros(batch_size, dtype=torch.long, device=DEVICE),
        "castling": torch.zeros(batch_size, dtype=torch.long, device=DEVICE),
        "ep_file": torch.zeros(batch_size, dtype=torch.long, device=DEVICE),
        "fused_ids": torch.randint(0, 13, (batch_size, 64), dtype=torch.long, device=DEVICE),
    }
    targets = torch.randint(0, 1968, (batch_size,), device=DEVICE)
    wdl = torch.randn(batch_size, 3, device=DEVICE).softmax(dim=-1)
    
    try:
        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.float16):
                out = model(batch_input)
                loss = F.cross_entropy(out["policy_logits"], targets) + 0.25 * F.cross_entropy(out["value_logits"], wdl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.float16):
                out = model(batch_input)
                loss = F.cross_entropy(out["policy_logits"], targets) + 0.25 * F.cross_entropy(out["value_logits"], wdl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        
        pos_per_s = (batch_size * n_iters) / elapsed
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        return pos_per_s, mem_gb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None, None

def main():
    print("Loading model...")
    model = load_model(CHECKPOINT, DEVICE)
    model.train()
    
    batch_sizes = [64, 128, 192, 256, 384, 512, 640, 768]
    
    print(f"\n{'Batch':>6} | {'pos/s':>8} | {'VRAM (GB)':>10} | {'Status':>10}")
    print("-" * 50)
    
    best_bs = 64
    best_speed = 0
    
    for bs in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        speed, mem = bench_batch_size(model, bs)
        if speed is None:
            print(f"{bs:>6} | {'OOM':>8} | {'---':>10} | {'FAIL':>10}")
            break
        else:
            status = "BEST" if speed > best_speed else "ok"
            if speed > best_speed:
                best_speed = speed
                best_bs = bs
            print(f"{bs:>6} | {speed:>8.0f} | {mem:>10.1f} | {status:>10}")
    
    print(f"\nBest: batch_size={best_bs}, {best_speed:.0f} pos/s")
    
    # Now test with accum steps to find effective batch = 512
    print(f"\nEffective batch 512 with best micro-batch={best_bs}:")
    accum = max(1, 512 // best_bs)
    print(f"  accum_steps={accum}, effective_batch={best_bs * accum}")

if __name__ == "__main__":
    main()
