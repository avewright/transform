"""Minimal debug script: replicate exp136 training loop to find hang point."""
import os, sys, time
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

from chess_transformer_factory import build_model, ChessTransformerConfig, count_parameters
from data_loader import load_training_data, get_batch_input
from move_vocab import VOCAB_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = ChessTransformerConfig(
    encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
    ffn_ratio=4, dropout=0.1, policy_head_dim=256, value_hidden=256,
)

print(f"Device: {DEVICE}", flush=True)
print("Loading data...", flush=True)
train_tensors, eval_data, eval_tensors = load_training_data(
    n_train=500_000, n_eval=2500, encoder_type="fused", min_depth=15, seed=42)
n_train = train_tensors["move_idx"].shape[0]
print(f"Data loaded: {n_train:,}", flush=True)

print("Building model...", flush=True)
model = build_model(config)
print(f"Params: {count_parameters(model):,}", flush=True)
model.to(DEVICE).train()
print("Model on CUDA", flush=True)

optimizer = AdamW(model.parameters(), lr=2e-4)
scaler = GradScaler('cuda')
bs = 256
accum = 4

print("Creating permutation...", flush=True)
perm = torch.randperm(n_train)
print(f"Perm shape: {perm.shape}", flush=True)

optimizer.zero_grad()
accum_count = 0

for step_i in range(20):  # 20 micro-batches = 5 optimizer steps
    batch_start = step_i * bs
    batch_end = batch_start + bs
    idx = perm[batch_start:batch_end]
    
    print(f"  micro-batch {step_i}: idx[0]={idx[0].item()}, getting batch...", flush=True)
    batch_input = get_batch_input(train_tensors, idx, "fused", DEVICE)
    move_targets = train_tensors["move_idx"][idx].to(DEVICE)
    wdl_targets = train_tensors["wdl"][idx].to(DEVICE)
    
    print(f"  micro-batch {step_i}: forward...", flush=True)
    with autocast('cuda', dtype=torch.float16):
        result = model(batch_input)
        ce_loss = F.cross_entropy(result["policy_logits"], move_targets)
        value_loss = F.cross_entropy(result["value_logits"], wdl_targets)
        total_loss = ce_loss + 0.5 * value_loss
        scaled_loss = total_loss / accum
    
    print(f"  micro-batch {step_i}: ce={ce_loss.item():.4f}, backward...", flush=True)
    scaler.scale(scaled_loss).backward()
    
    accum_count += 1
    if accum_count >= accum:
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        accum_count = 0
        opt_step = (step_i + 1) // accum
        print(f"  OPTIMIZER STEP {opt_step}: gnorm={gnorm:.4f}", flush=True)

print("\nDone! Training loop works.", flush=True)
