"""Speed benchmark: 204M model training with optimized policy head.

Previous measurement: 5 pos/s at bs=16 (21 days for 10M positions).
Goal: measure with optimized SpatialPolicyHead + gradient checkpointing.
"""
import os, sys, time
from pathlib import Path
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import build_model, ChessTransformerConfig, DEFAULT_200M_CONFIG
from move_vocab import VOCAB_SIZE
from data_loader import board_array_to_fused, ep_square_to_file, compute_wdl

DEVICE = torch.device("cuda")
ROOT = Path(__file__).resolve().parent.parent
SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"


def bench_config(bs, grad_ckpt=False, n_steps=20):
    """Benchmark training throughput for 204M model."""
    print(f"\n--- bs={bs}, grad_ckpt={grad_ckpt} ---")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = build_model(DEFAULT_200M_CONFIG)
    
    if grad_ckpt:
        # Enable gradient checkpointing on transformer layers
        for layer in model.transformer.layers:
            layer.self_attn._qkv_same_embed_dim = True  # Ensure compatibility
        model.transformer.layers = nn.ModuleList([
            torch.utils.checkpoint.checkpoint_wrapper(layer) if hasattr(torch.utils.checkpoint, 'checkpoint_wrapper') else layer 
            for layer in model.transformer.layers
        ])

    model.to(DEVICE)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = GradScaler('cuda')
    
    # Load one shard for data
    shard = torch.load(SHARD_DIR / "shard_00000.pt", map_location="cpu", weights_only=True)
    fused = board_array_to_fused(shard["board_array"][:bs * n_steps])
    turn = shard["turn"][:bs * n_steps].long()
    castling = shard["castling"][:bs * n_steps].long()
    ep_file = ep_square_to_file(shard["ep_square"][:bs * n_steps].long())
    move_idx = shard["move_idx"][:bs * n_steps].long()
    wdl = compute_wdl(shard["cp"][:bs * n_steps], shard["mate"][:bs * n_steps])
    del shard
    
    mem_before = torch.cuda.memory_allocated() / 1e6
    print(f"  Model loaded: {mem_before:.0f} MB GPU")
    
    # Warmup
    try:
        batch_input = {
            "fused_ids": fused[:bs].to(DEVICE),
            "turn": turn[:bs].to(DEVICE),
            "castling": castling[:bs].to(DEVICE),
            "ep_file": ep_file[:bs].to(DEVICE),
        }
        targets = move_idx[:bs].to(DEVICE)
        wdl_t = wdl[:bs].to(DEVICE)
        
        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)
            loss = F.cross_entropy(result["policy_logits"], targets) + \
                   0.5 * F.cross_entropy(result["value_logits"], wdl_t)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak memory: {peak:.0f} MB / 8188 MB")
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at bs={bs}")
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        return None
    
    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    n_pos = 0
    
    for step in range(n_steps):
        i = step * bs
        batch_input = {
            "fused_ids": fused[i:i+bs].to(DEVICE),
            "turn": turn[i:i+bs].to(DEVICE),
            "castling": castling[i:i+bs].to(DEVICE),
            "ep_file": ep_file[i:i+bs].to(DEVICE),
        }
        targets = move_idx[i:i+bs].to(DEVICE)
        wdl_t = wdl[i:i+bs].to(DEVICE)
        
        with autocast('cuda', dtype=torch.float16):
            result = model(batch_input)
            loss = F.cross_entropy(result["policy_logits"], targets) + \
                   0.5 * F.cross_entropy(result["value_logits"], wdl_t)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        n_pos += bs
    
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    pos_per_sec = n_pos / elapsed
    peak = torch.cuda.max_memory_allocated() / 1e6
    
    print(f"  Speed: {pos_per_sec:.1f} pos/s ({n_pos} positions in {elapsed:.1f}s)")
    print(f"  Peak: {peak:.0f} MB")
    
    eta_1epoch = 10_000_000 / pos_per_sec / 3600
    print(f"  ETA 1 epoch (10M): {eta_1epoch:.1f} hours")
    
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return pos_per_sec


if __name__ == "__main__":
    print("204M Training Speed Benchmark (optimized policy head)")
    print("=" * 60)
    
    # Test different batch sizes
    for bs in [8, 16, 24, 32]:
        result = bench_config(bs, grad_ckpt=False, n_steps=10)
        if result is None:
            break
    
    print("\nWith gradient checkpointing:")
    for bs in [16, 32, 48]:
        result = bench_config(bs, grad_ckpt=True, n_steps=10)
        if result is None:
            break
