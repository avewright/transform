# exp191 400M Meta-Attention Implementation Plan

> **For agentic workers:** implement/verify against `docs/superpowers/specs/2026-07-11-meta-attention-400m-design.md`

**Goal:** Ship ≥400M factored content×position attention + Shaw-on-ss, train script, remove handcrafted rel-bias/full-dim/seq PE.

## Done in this session

1. Spec written  
2. `forward_streams` on fused/strengthened encoders  
3. `MetaFactoredAttention` + `MetaFactoredEncoderLayer`  
4. `DEFAULT_400M_META_CONFIG` (~437M)  
5. `experiments/exp191_400m_meta_attention.py`  

## Verify

```bash
MOVE_VOCAB_VERSION=compact python -c "from chess_transformer_factory import *; m=build_model(DEFAULT_400M_META_CONFIG); print(count_parameters(m)/1e6)"
MOVE_VOCAB_VERSION=compact python experiments/exp191_400m_meta_attention.py --go --smoke
```

## Full train

```bash
MOVE_VOCAB_VERSION=compact python experiments/exp191_400m_meta_attention.py --go \
  --soft-cache outputs/exp186_sf_multipv/soft_cache.pt \
  --deep-soft-cache outputs/exp190_phase_deep/soft_cache.pt
```
