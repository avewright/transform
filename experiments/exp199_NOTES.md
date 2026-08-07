# exp199 — 2500 Push: Stockfish WDL+CP + AlphaZero Hybrid Root Search

## Why 1950 stalls (audit)

| Component | Current state | Root cause |
|-----------|---------------|------------|
| `chess_model.py:539-547` | `value_head: 1024→256→3` hard CE, weight 0.5, White-absolute | Collapses to argmax bucket; cp +20 and cp+300 both supervise as "win". Draw mass invisible. Head under-trained → search hurts Elo (exp094/097/098). |
| `chess_transformer_factory.py:1510-1521` | `cls` or `pool` 3-class, or `SearchValueHead:1332` fused prior | Hard CE only; soft WDL / HL-Gauss path exists for 128-bin (factory:34) but autoresearch never uses it. `SearchValueHead` tanh backup is unstable with noisy WDL labels. |
| `exp195_meta_latent_search.py:53-79` | Uses **non-existent** `use_latent_search/latent_topk` keys (factory has `use_search_policy_head/policy_topk`) — config silently drops them, so the "latent search" run was actually a plain spatial head. Explains why exp195 did not beat the 28L/256d baseline. | Config-protocol mismatch. |
| `exp198_hybrid_blend.py` | Tested S0..S5: flat blend is wash, S3_gate/S4_valuegate win only when `gate≈30cp` and `tf_conf≥0.4` | Signal is good only on SF near-ties gated by TF value agreement. Requires a calibrated value head to veto, which we don't have. |

## Proposed factory edits (do not apply yet — exp199 monkey-patches for A/B)

### 1. `chess_transformer_factory.py` — new head
Add after `PooledValueHead:1151` (~line 1172):

```python
class StockfishWDLCPHead(nn.Module):
    """Stockfish-style dual head: soft WDL (3) + cp scalar (1).

    Input: CLS (hidden_dim) + mean-pooled squares. Shared MLP stem,
    then split: WDL logits via KL against cp-derived soft WDL,
    cp via tanh Huber against clip(cp)/1500. Returned dict has
    value_logits (B,3) and cp_scalar (B,) in [-1,1] White-absolute.
    Fused scalar = 0.6*wdl_scalar + 0.4*cp_scalar for PUCT backup.
    """
    def __init__(self, hidden_dim, value_hidden=512, n_ctx_tokens=4):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.shared = nn.Sequential(nn.Linear(hidden_dim*2, value_hidden), nn.ReLU())
        self.wdl = nn.Linear(value_hidden, 3)
        self.cp  = nn.Sequential(nn.Linear(value_hidden, value_hidden//2), nn.ReLU(),
                                 nn.Linear(value_hidden//2, 1))
    def forward(self, hidden, cls_hidden):
        pool = hidden[:, self.n_ctx:self.n_ctx+64].mean(1)
        h = self.shared(torch.cat([cls_hidden, pool], -1))
        return {"value_logits": self.wdl(h), "cp_scalar": torch.tanh(self.cp(h).squeeze(-1))}
```

Add config flag:
```python
# ChessTransformerConfig
use_wdl_cp_head: bool = False
wdl_cp_weight: float = 0.3  # Huber weight inside value loss
```

Wire in `ChessTransformer.__init__` alongside the `value_head` branches (after line 1509):
```python
elif config.use_wdl_cp_head:
    self.value_head = StockfishWDLCPHead(...)
    self._pool_value = True  # needs hidden
```

### 2. `data_loader.py` — soft targets
No code change needed if using `prepare_batch` in exp199 (cp→soft-WDL computed on the fly via `cp_to_soft_wdl` with `k(ply)=111.7+0.2*max(0,50-ply)`). For a permanent fix, expose `cp_to_soft_wdl` next to `compute_wdl:44`.

### 3. `experiments/exp198_hybrid_blend.py` — promote gate to PUCT
Replace flat `pick()` blend with AlphaZero root PUCT (keep S3/S4 gate as fallback):
```python
# root PUCT: Q = fused_value (WDL+CP), P = TF policy, visits ∝ P * sqrt(N)/ (1+n)
# Gate: only run PUCT when cp_spread < 35 and max(tf_policy) > 0.35 and sign(cp) == sign(v)
# else fall back to SF argmax. Adds ~5ms/pos, no SF deep search.
```

## Experiment file

`experiments/exp199_wdl_cp_hybrid_search.py` — **already drafted** (26555 bytes).

- Fixes exp195 config keys: `use_search_policy_head=True, policy_topk=16, policy_search_steps=3` (factory-correct).
- Policy: `0.35*hard + 0.65*soft + 0.25*base_aux` (exp195 recipe, keeps spatial prior honest).
- Value: `KL(soft-WDL) + 0.3*Huber(cp)` with `value_weight=0.5` (5× baseline). Supports 128-bin HL-Gauss if `n_value_classes=128` (imports `chess_qwen_factory.hl_gauss_loss`).
- Optim: NorMuon 0.02/3e-4, warmup 200, cosine 0.05, clip 1.0, hflip 0.5 (matches `search_space.json` baseline).
- Two presets: `DEFAULT_25M_CFG` (20L/256d, smoke-safe) and `DEFAULT_A100_CFG` (32L/768d, 120M, ckpt, meta+shaw+dual).
- Elo via `autoresearch_8gb/elo_trial.py` (same SPRT as exp194/exp195); optional `--hybrid-eval` runs exp198 sweep post-training.
- Graceful fallback: if factory not yet patched, monkey-patches `value_head` with `_WDLCPWrapper` so the script runs unmodified.

## Command to run on A100 / RunPod

```bash
# RunPod A100 80GB — PyTorch 2.4 / CUDA 12.4 template, 1× GPU
# 1) deps
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q python-chess datasets huggingface_hub
# optional NorMuon (if available): pip install -q /workspace/normuon

# 2) env
export MOVE_VOCAB_VERSION=compact
export HF_TOKEN=...            # for hf cache streaming fallback
export PYTHONUNBUFFERED=1

# 3) train — full 8k-step run (~90 min, 25M) or A100 32L/768d
MOVE_VOCAB_VERSION=compact python experiments/exp199_wdl_cp_hybrid_search.py --go \
  --soft-cache outputs/autoresearch_8gb/soft_cache_200k.pt \
  --deep-cache outputs/autoresearch_8gb/puzzle_syzygy_mix.pt \
  --max-steps 8000 --batch-size 96 --accum 2 \
  --value-weight 0.5 --cp-weight 0.3 --hybrid-eval

# A100 deep-narrow (120M, needs checkpointing):
MOVE_VOCAB_VERSION=compact python experiments/exp199_wdl_cp_hybrid_search.py --go --a100 \
  --soft-cache outputs/autoresearch_8gb/soft_cache_200k.pt \
  --max-steps 8000 --batch-size 64 --accum 2 --hybrid-eval

# 4) smoke sanity (20 steps, MPS/CPU)
MOVE_VOCAB_VERSION=compact python experiments/exp199_wdl_cp_hybrid_search.py --go --smoke

# 5) Elo-only from ckpt
MOVE_VOCAB_VERSION=compact python experiments/exp199_wdl_cp_hybrid_search.py --go --elo-only \
  --checkpoint outputs/exp199_wdl_cp_hybrid/latest.pt

# 6) isolated hybrid sweep (needs Stockfish binary at stockfish/stockfish-native-arm64)
python experiments/exp198_hybrid_blend.py --checkpoint outputs/exp199_wdl_cp_hybrid/latest.pt --multipv 8
python hybrid_uci.py --stockfish stockfish/stockfish-native-arm64 \
  --checkpoint outputs/exp199_wdl_cp_hybrid/latest.pt --multipv 8 --policy-weight 0.35 --temp 0.9
```

Expected uplift: value acc >62% (vs 54% hard), gated hybrid +60–120 Elo on near-tie subset, latent policy refinement +20–40 Elo on policy argmax (exp064 pattern). Combined target 2050–2100 on the 25M board, scaling to 700M as deep-narrow.
