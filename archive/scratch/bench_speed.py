"""Quick benchmark: measure pos/s for different configs on the actual model."""
import sys, time, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Enable TF32 for A40 (Ampere)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from experiments.exp073_200m_full_epoch import ChessTransformer200M

device = 'cuda'

def bench(model, B, tag, N=30, warmup=5):
    inp = {
        'fused_ids': torch.randint(0, 13, (B, 64), device=device),
        'turn': torch.randint(0, 2, (B,), device=device),
        'castling': torch.randint(0, 16, (B,), device=device),
        'ep_file': torch.randint(0, 9, (B,), device=device),
    }
    tgt = torch.randint(0, 5504, (B,), device=device)
    wdl = torch.randn(B, 3, device=device).softmax(-1)

    model.train()
    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(warmup):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(inp)
            pl = F.cross_entropy(out['policy_logits'], tgt)
            vl = F.kl_div(F.log_softmax(out['value_logits'], -1), wdl, reduction='batchmean')
            loss = (pl + 0.5 * vl) / 4
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(N):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(inp)
            pl = F.cross_entropy(out['policy_logits'], tgt)
            vl = F.kl_div(F.log_softmax(out['value_logits'], -1), wdl, reduction='batchmean')
            loss = (pl + 0.5 * vl) / 4
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    pos_s = B * N / elapsed
    ms = elapsed / N * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {tag:40s} | B={B:>4} | {ms:6.1f} ms/step | {pos_s:>7.0f} pos/s | {mem:.1f} GB VRAM")
    torch.cuda.reset_peak_memory_stats()
    return pos_s, mem

print("=" * 90)
print("BENCHMARK: ChessTransformer200M on A40")
print("=" * 90)

# 1) Baseline (no TF32 compile)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = ChessTransformer200M().to(device)
bench(model, 256, "baseline (fp16 only, B=256)")
del model; torch.cuda.empty_cache()

# 2) TF32 enabled
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = ChessTransformer200M().to(device)
bench(model, 256, "TF32 enabled, B=256")

# 3) TF32 + larger batch
for B in [512, 768, 1024]:
    try:
        torch.cuda.reset_peak_memory_stats()
        bench(model, B, f"TF32, B={B}")
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at B={B}")
        torch.cuda.empty_cache()
        break
del model; torch.cuda.empty_cache()

# 4) torch.compile + TF32
model = ChessTransformer200M().to(device)
model = torch.compile(model)
bench(model, 256, "torch.compile + TF32, B=256", warmup=10)
bench(model, 512, "torch.compile + TF32, B=512", warmup=3)
try:
    bench(model, 768, "torch.compile + TF32, B=768", warmup=3)
except torch.cuda.OutOfMemoryError:
    print("  OOM at B=768 compiled")
    torch.cuda.empty_cache()
try:
    bench(model, 1024, "torch.compile + TF32, B=1024", warmup=3)
except torch.cuda.OutOfMemoryError:
    print("  OOM at B=1024 compiled")
    torch.cuda.empty_cache()
del model; torch.cuda.empty_cache()

# 5) torch.compile + gradient checkpointing + TF32 (bigger batch)
from torch.utils.checkpoint import checkpoint

class ChessTransformer200M_GC(ChessTransformer200M):
    """Same model but with gradient checkpointing on transformer layers."""
    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        B = hidden.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embed
        # Gradient checkpoint each transformer layer
        for layer in self.transformer.layers:
            hidden = checkpoint(layer, hidden, use_reentrant=False)
        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }

model = ChessTransformer200M_GC().to(device)
model = torch.compile(model)
for B in [512, 1024, 1536, 2048]:
    try:
        bench(model, B, f"compile+gradckpt+TF32, B={B}", warmup=8)
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at B={B} with grad checkpointing")
        torch.cuda.empty_cache()
        break

print("\n✓ Benchmark complete")
