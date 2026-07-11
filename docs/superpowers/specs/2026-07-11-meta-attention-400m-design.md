# Design: 400M Meta-Factored Attention Chess Transformer

**Date:** 2026-07-11  
**Status:** Approved via "wire up the next best experiment"  
**Experiment:** `exp191_400m_meta_attention.py`

## Problem

The ~200M compact-soft model reached ~1700 Elo (policy-only) then plateaued. Absolute `piece + square` embeddings + vanilla attention glue geometry into tokens too early. Handcrafted `ChessRelativeBias` (diag/knight flags) names patterns instead of letting the model learn them.

## Goal

Train a **≥400M** model whose attention can form conjunctions like “geometry × piece content” without handcrafted chess-move features, using the soft MultiPV recipe that worked for the 200M.

## Architecture

### Meta-factored attention (4 terms)

Per square keep separate streams:

- **content** `c` — piece/empty (+ local conv features)
- **position** `p` — square identity (absolute grid)

Attention logits (added; softmax multiplies in probability space):

```
cc = q_c(i) · k_c(j)     # piece ↔ piece
ss = q_p(i) · k_p(j)     # square ↔ square (+ Shaw)
cs = q_c(i) · k_p(j)     # piece → square
sc = q_p(i) · k_c(j)     # square → piece
score = (cc + ss + cs + sc) / √d_h
```

Values come from the **content** stream. Residual + FFN update content. Position stream is an absolute anchor (not deep-residual), so geometry stays available every layer.

### Shaw only on `ss` (not handcrafted)

Learned vectors keyed by raw `(Δfile, Δrank)` ∈ {−7…7}² (15×15 buckets). No diagonal/knight flags. Optional `a^V` omitted in v1 (scores-only Shaw) to control params/VRAM.

### Removed (irrelevant / anti-goal)

| Removed | Why |
|---------|-----|
| `use_rel_bias` / `ChessRelativeBias` | Handcrafted move geometry |
| `use_pos_embed` sequence PE | Redundant with position stream |
| `full_dim_attention` | Expensive scale trick; not relational |
| 128-class value | 200M win used 3-class WDL aux |

### Kept from 200M success

- Compact vocab (1968)
- `SpatialPolicyHead`
- Soft MultiPV CE + hard CE mix + light WDL aux
- STM-normalized strengthened encoder
- SwiGLU FFN
- H-flip aug on soft batches

## Model config (`DEFAULT_400M_META_CONFIG`)

| Knob | Value |
|------|-------|
| encoder | strengthened, dim 512, STM on, 2 conv blocks |
| hidden / layers / heads | 1280 / 18 / 20 |
| ffn_ratio | 4 + SwiGLU |
| meta attention | on |
| Shaw on ss | on |
| rel_bias / pos_embed / full_dim | off |
| value | WDL 3-class |
| grad checkpoint | on (A40/A100 friendly) |

Target: **≥400M params** (verify at build). Train **from scratch** (architecture break vs 200M).

## Training (`exp191`)

- Soft cache: exp186 MultiPV (+ optional exp190 deep mix)
- Hard ballast: HF lichess-sf `min_depth≥15`
- Defaults mirror exp189 (soft_frac≈0.72, soft_alpha≈0.35–0.40) with lower LR for larger model (~3e-4 warmup cosine or 1e-4)
- Smoke: `--go --smoke`

## Success criteria

1. Smoke forward/backward completes; params ≥ 400M  
2. Soft holdout top-1 trends up over first 2–5k steps (not collapsed)  
3. Elo eval vs SF limited later — beat ~1700 wall if data+scale allow  

## Non-goals (v1)

- Loading 200M weights  
- MoE / RoPE / from–to policy rewrite  
- Online RL / MCTS in the train loop  
