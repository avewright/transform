# Current Architecture

This document describes the architecture used in the active experiment series
(`exp067`–`exp069`) and shared via `data_loader.py` + `chess_model.py`.

It supersedes the earlier exp050-era description.
The aspirational V1 design is in [`ARCHITECTURE_V1.md`](ARCHITECTURE_V1.md).

## Overview

A chess-native encoder–transformer with:

- **FusedBoardEncoder** (primary) or **LearnedBoardEncoder** (baseline comparison)
- A learned **[CLS]** token prepended to the sequence
- **SpatialPolicyHead** for move prediction
- **WDL Value Head** (win/draw/loss) trained jointly

All output logits over the fixed 5 504-move vocabulary from
[`move_vocab.py`](move_vocab.py).

## End-to-End Diagram

```text
chess.Board  (or board_array from cache)
   |
   v
Encoder input dict
  FusedBoardEncoder:
    fused_ids: (B, 64)   # 0–12: empty + 6 white + 6 black
    turn:      (B,)
    castling:  (B,)
    ep_file:   (B,)
  LearnedBoardEncoder (baseline):
    piece_ids: (B, 64)
    color_ids: (B, 64)
    turn, castling, ep_file as above
   |
   v
Encoder  (embed_dim = 256)
   |
   v
Tokens: (B, 67, 256)
[turn] [castling] [ep] [sq0] … [sq63]
   |
   v
Linear input projection  256 → 512
   |
   v
Prepend learned [CLS] token: (B, 68, 512)
   |
   v
+ learned positional embedding: (1, 68, 512)
   |
   v
TransformerEncoder
  8 layers · 8 heads · FFN 4× · GELU · dropout 0.1 · norm_first
   |
   v
LayerNorm → hidden states: (B, 68, 512)
   |
   +------------------------------+
   |                              |
   v                              v
SpatialPolicyHead              Value Head
  cls_hidden + sq tokens         cls_hidden → MLP
   |                              512 → 256 → 3
   v                              v
Policy logits: (B, 5504)       Value logits: (B, 3)  [win/draw/loss]
   |
   v
Mask illegal moves (legal_move_mask)
   |
   v
Softmax → argmax → predicted best move
```

## Token Layout

```text
Position 0:  [CLS]      (learned, not tied to any board feature)
Position 1:  [turn]
Position 2:  [castling]
Position 3:  [ep]
Positions 4–67:  [sq0] … [sq63]
```

The **[CLS]** token at position 0 serves as the global readout for:
- the value head
- the global context inside `SpatialPolicyHead`

This replaces the earlier design where the turn token doubled as the
global readout.

## Encoder Variants

### FusedBoardEncoder (primary)

Single embedding table of 13 tokens (empty + 6 white + 6 black pieces).
Input `fused_ids` maps each square to one of these 13 values.

```text
fused_embed(fused_ids) + square_embed(0..63)
```

Then 3 context tokens are prepended: turn, castling, ep.

Output: `(B, 67, 256)`

### LearnedBoardEncoder (baseline)

Factored representation: separate `piece_embed` and per-color linear
projections.

```text
color_proj[color](piece_embed[piece]) + square_embed(0..63)
```

Same 3 context tokens prepended. Same output shape.

## Transformer Body

- Input projection: `256 → 512`
- Learned [CLS] token prepended: sequence length 67 → 68
- Learnable positional embedding: `(1, 68, 512)`
- `nn.TransformerEncoder` with `norm_first=True`
- 8 layers, 8 heads, FFN = 2048, dropout = 0.1
- Final `LayerNorm`

Output: `(B, 68, 512)`

## SpatialPolicyHead

Reads:
- square tokens `hidden[:, 4:68, :]` (64 squares, after CLS + 3 context)
- `cls_hidden = hidden[:, 0, :]` as global context

For each of the 5 504 moves:

```text
from_feat = from_proj(sq[from_sq])
to_feat   = to_proj(sq[to_sq])
global    = global_proj(cls_hidden)
promo     = promo_embed(promo_type)

combined = from_feat * to_feat + global + promo
logit    = score_proj(ReLU(combined))
```

Output: `policy_logits (B, 5504)`

## Value Head

```text
cls_hidden → Linear(512, 256) → ReLU → Linear(256, 3)
```

Output: `value_logits (B, 3)` — win / draw / loss

Trained jointly with policy via:
```text
loss = policy_CE + 0.5 * KL_div(log_softmax(value_logits), wdl_targets)
```

## Training Setup (exp067–069 defaults)

| Parameter | Value |
|-----------|-------|
| Train positions | 500 000 |
| Eval positions | 2 500 |
| Batch size | 256 |
| Learning rate | 3e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Schedule | Linear warmup 5% → cosine decay |
| Precision | AMP float16 |
| Seeds | 42, 123, 314 |
| Min depth | 15 |

## Data Pipeline

Data loads instantly from a `.pt` cache built by `data_loader.py`:

1. **Local cache** (`outputs/data_cache/*.pt`) — ~2s load
2. **HF streaming** (`avewright/chess-positions-lichess-sf`) — fallback
3. **Raw parquet** (Lichess/Stockfish 49.7M rows) — last resort

The cache stores encoding-agnostic `board_array[64]` with values 0–12,
plus metadata (turn, castling, ep_square, move_idx, cp, mate, depth, fen).
Both fused and baseline tokenizations are derived at load time.

## Current Shapes Summary

| Stage | Shape |
|-------|-------|
| encoder input | dict of (B,64) + (B,) features |
| encoder output | (B, 67, 256) |
| + CLS + projection | (B, 68, 512) |
| transformer output | (B, 68, 512) |
| policy logits | (B, 5504) |
| value logits | (B, 3) |

## Notes

- Move legality is enforced by post-hoc masking, not built into the head.
- The model is encoder-only and chess-native (no language backbone).
- exp067 compares LearnedBoardEncoder vs FusedBoardEncoder.
- exp068 tests relative geometry bias in attention.
- exp069 tests confidence-weighted sampling vs uniform.
- The current implementation uses the `turn` token as the de facto global token.
