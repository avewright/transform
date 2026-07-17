# Chessformer CF-240M → 8GB autoresearch transfer

## Papers

| Paper | Size | Link |
|-------|------|------|
| **Mastering Chess with a Transformer Model** (Monroe & Chalmers) | **CF-240M** (~243M) | [arXiv:2409.12272](https://arxiv.org/abs/2409.12272) — local `research.html` |
| Grandmaster-Level Chess Without Search (Ruoss et al.) | GC-270M / GC-136M | [arXiv:2402.04494](https://arxiv.org/abs/2402.04494) (related; distillation / value@1) |
| Chessformer: Unified Architecture (later GAB work) | varies | [arXiv:2605.19091](https://arxiv.org/html/2605.19091v1) (Geometric Attention Bias; not yet in search space) |

Primary target for Wave 3: **Monroe & Chalmers Chessformer / CF-240M**.

## Ideas already in our stack

- Square tokens + STM-normalized board
- Shaw-style relative position on attention (`use_shaw_on_pos` / meta factored attn)
- Relative bias ablations (`use_rel_bias`)
- From–to spatial policy head (`SpatialPolicyHead`) ≈ Chessformer source–destination attention policy
- Soft multipv policy mix (`soft_alpha`)

## Wave-3 experiments (new)

| Trial id | Chessformer idea | Implementation |
|----------|------------------|----------------|
| `cf_soft_temp` | Soft policy head, T=4 | Same policy head + aux CE on π^(1/T); `soft_temp=4`, `soft_temp_weight=0.5` |
| `cf_soft_temp_heavy` | High soft weight (paper c_softpol=8, muted by T) | `soft_alpha=0.7` + stronger T=4 aux |
| `cf_swa` | SWA final checkpoint | Avg params from `swa_start_frac=0.75` |
| `cf_shaw_recipe` | Shaw + soft T=4 + SWA, drop handcrafted rel_bias | Inherits `meta_shaw_no_relbias` |
| `cf_value_heavy` | Stronger value term | `value_weight=1.0` |

## Deferred (need data / larger GPU)

- 7-ply history piece planes in input tokens
- Separate soft-policy head parameters (we share the main head)
- DeepNorm + Mish FFN
- Full CF-240M scale (15L / 1024d) — use A40 / exp191 path
- Geometric Attention Bias (2026 Chessformer follow-up)

## Run

After wave-1 (and optionally wave-2) frees the GPU:

```bash
bash scripts/run_autoresearch_wave3_chessformer.sh
```

Promotion rule unchanged: **policy Elo only** (±100 noise); soft_loss diagnostic.
