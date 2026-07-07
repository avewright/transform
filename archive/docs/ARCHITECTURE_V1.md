# Architecture V1 Specification

This document defines the target architecture for the chess model family.
It replaces the ad-hoc experiment-specific model definitions with a single
standardized design that all future experiments and training runs should use.

## Design Principles

1. **Chess-native encoder-only transformer** — bidirectional attention, not
   a frozen causal LLM backbone. Every token sees every other token.
2. **Factorized spatial policy head** — from-square × to-square × promotion,
   not a flat 5504-way linear layer.
3. **Dedicated global token** — a learnable `[CLS]`-style token prepended to
   the sequence, used consistently for value and any global readout.
4. **Relative board biases** — encode square-pair distance in attention,
   capturing chess geometry (knight jumps, diagonals, files, ranks).
5. **Multi-target value head** — WDL, centipawn bucket, optional mate-distance.

## Token Layout

```
[CLS] [turn] [castling] [ep] [sq_a1] [sq_b1] ... [sq_h8]
  0      1       2        3     4       5           67
```

Total: 68 tokens (1 CLS + 3 context + 64 squares).

CLS is a learnable parameter (not derived from the board), used as the
aggregation point for global predictions (value head, any scalar outputs).
Context tokens encode side-to-move, castling rights, and en-passant file.
Square tokens encode piece identity via `piece_embed[type] × color_proj[color] + square_embed[sq]`.

## Encoder: `LearnedBoardEncoderV2`

Inherits the structure from `LearnedBoardEncoder` with two changes:

1. **Prepend a learnable CLS token** instead of relying on first context
   token (turn) as the global readout.
2. **LayerNorm** on output (already present).

Parameters: ~250K at `embed_dim=256`.

## Transformer Body: `ChessTransformerV1`

| Hyperparameter | Small | Medium | Large |
|----------------|-------|--------|-------|
| `hidden_dim`   | 256   | 512    | 768   |
| `num_layers`   | 6     | 8      | 12    |
| `num_heads`    | 8     | 8      | 12    |
| `ff_mult`      | 4     | 4      | 4     |
| `dropout`      | 0.1   | 0.1    | 0.05  |
| Total params   | ~4M   | ~17M   | ~55M  |

- **Encoder-only** `nn.TransformerEncoder` with `norm_first=True`, GELU activation.
- **Positional encoding**: Learnable `nn.Parameter` of shape `(1, 68, hidden_dim)`.
- **Relative board bias** (optional): Additive attention bias matrix `(68, 68)` or
  per-head `(num_heads, 68, 68)` derived from precomputed square-pair features
  (Chebyshev distance, same file, same rank, same diagonal, knight-move).
  Allows the model to learn that diagonally adjacent squares are related even
  without seeing examples of every piece on every diagonal.

## Policy Head: `SpatialPolicyHead`

Factorized scoring over the move vocabulary:

```
from_feat = from_proj(sq_hidden[from_sq])   # (B, V, D)
to_feat   = to_proj(sq_hidden[to_sq])       # (B, V, D)
global_feat = global_proj(cls_hidden)        # (B, 1, D)
promo_feat  = promo_embed(promo_type)        # (V, D)

combined = from_feat * to_feat + global_feat + promo_feat
logits   = score_proj(ReLU(combined))        # (B, V)
```

This head uses per-square hidden states (not just a single pooled vector),
so it can distinguish the spatial character of different moves. The CLS
token provides global context (material balance, king safety, etc.).

At inference, illegal moves are masked to `-inf` before softmax.

## Value Head: `MultiTargetValueHead`

Reads from the CLS token hidden state:

```
cls -> Linear(hidden, 256) -> ReLU -> {
    wdl:    Linear(256, 3)   # softmax cross-entropy vs WDL target
    cp_bucket: Linear(256, 32)  # 32 centipawn buckets for calibration
}
```

CP buckets: `[-inf, -500, -400, ..., -50, 0, 50, ..., 400, 500, +inf]` (32 bins).
Trained with cross-entropy against the bucket containing the Stockfish eval.

## Qwen Backbone Mode (Transfer Baseline)

For comparison, keep the existing `ChessModel` (causal Qwen backbone) as a
baseline, but fix the pooling to always use `hidden[:, -1, :]` (last token
in causal sequence has full context). This is the **transfer baseline** — it
should not be the primary training target going forward.

## Move Vocabulary

Unchanged: 5504 UCI moves from `move_vocab.py`. The factorized head still
indexes into this vocabulary but scores moves via spatial decomposition
rather than a flat projection.

## Standard Evaluation Protocol

Every training run must report:

| Metric | Description |
|--------|-------------|
| `sf_acc` | Top-1 accuracy on Stockfish best-move eval set |
| `sf_top3` | Top-3 accuracy on Stockfish best-move eval set |
| `human_acc` | Top-1 accuracy on human-move eval set |
| `legal_rate` | Fraction of predictions that are legal (should be 100% with masking) |
| `games_w/d/l` | W/D/L vs Stockfish depth 3 over 8+ games |
| `value_cal` | WDL calibration (optional, when value head is trained) |

Eval sets must be **split by game** (not by shuffled position) to prevent
leakage from correlated positions within the same game.

## File Mapping

| Component | File |
|-----------|------|
| `LearnedBoardEncoderV2` | `chess_model.py` |
| `ChessTransformerV1` | `chess_model.py` |
| `SpatialPolicyHead` | `chess_model.py` |
| `MultiTargetValueHead` | `chess_model.py` |
| Dataset factory | `label_positions.py` |
| Move vocabulary | `move_vocab.py` |
| Training loop | `train_action_value.py` or new `train_v1.py` |
