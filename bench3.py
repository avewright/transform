"""Minimal benchmark: just the configs we need."""
import sys, time, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.checkpoint import checkpoint as ckpt_fn
sys.path.insert(0, str(Path(__file__).resolve().parent))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from experiments.exp073_200m_full_epoch import ChessTransformer200M, SpatialPolicyHead
from experiments.exp073_200m_full_epoch import (
    ENCODER_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, FFN_RATIO,
    DROPOUT, POLICY_HEAD_DIM, VALUE_HIDDEN
)
from chess_model import FusedBoardEncoder
import torch.nn as nn

device = 'cuda'
LOG = open(Path(__file__).parent / "outputs" / "bench_results.txt", "w")

def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()

class FastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FusedBoardEncoder(embed_dim=ENCODER_DIM)
        self.input_proj = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, HIDDEN_DIM) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * FFN_RATIO, dropout=DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.policy_head = SpatialPolicyHead(HIDDEN_DIM, n_ctx_tokens=4, head_dim=POLICY_HEAD_DIM)
        self.value_head = nn.Sequential(nn.Linear(HIDDEN_DIM, VALUE_HIDDEN), nn.ReLU(), nn.Linear(VALUE_HIDDEN, 3))

    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        B = hidden.shape[0]
        hidden = torch.cat([self.cls_token.expand(B, -1, -1), hidden], dim=1)
        hidden = hidden + self.pos_embed
        for layer in self.transformer.layers:
            hidden = ckpt_fn(layer, hidden, use_reentrant=False)
        hidden = self.norm(hidden)
        cls_hidden = hidden[:, 0, :]
        return {"policy_logits": self.policy_head(hidden, cls_hidden), "value_logits": self.value_head(cls_hidden)}

def bench(model, B, tag, N=20, warmup=5):
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
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(warmup):
        opt.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(inp)
            loss = (F.cross_entropy(out['policy_logits'], tgt) + 0.5 * F.kl_div(F.log_softmax(out['value_logits'],-1), wdl, reduction='batchmean')) / 4
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(N):
        opt.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(inp)
            loss = (F.cross_entropy(out['policy_logits'], tgt) + 0.5 * F.kl_div(F.log_softmax(out['value_logits'],-1), wdl, reduction='batchmean')) / 4
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    pos_s = B * N / elapsed
    mem = torch.cuda.max_memory_allocated() / 1e9
    log(f"  {tag:45s} | B={B:>4} | {pos_s:>7.0f} pos/s | {mem:.1f} GB")
    torch.cuda.reset_peak_memory_stats()
    return pos_s, mem

log("=" * 85)
log("BENCHMARK: grad checkpointing sweep (no compile first, then with compile)")
log("=" * 85)

# --- Without compile ---
log("\n--- Gradient checkpointing, NO compile ---")
model = FastModel().to(device)
results = []
for B in [256, 512, 768, 1024]:
    try:
        ps, mem = bench(model, B, f"gradckpt, B={B}")
        results.append((B, ps, mem))
    except torch.cuda.OutOfMemoryError:
        log(f"  OOM at B={B}")
        torch.cuda.empty_cache()
        break
del model; torch.cuda.empty_cache()

# --- With compile ---
log("\n--- Gradient checkpointing + torch.compile ---")
model = FastModel().to(device)
model = torch.compile(model)
for B in [256, 512, 768, 1024]:
    try:
        ps, mem = bench(model, B, f"gradckpt+compile, B={B}", warmup=8)
        results.append((B, ps, mem))
    except torch.cuda.OutOfMemoryError:
        log(f"  OOM at B={B} compiled")
        torch.cuda.empty_cache()
        break
del model; torch.cuda.empty_cache()

log("\nDone!")
LOG.close()
