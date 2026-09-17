# Experiment 287: ChessBot architecture at 99M, ChessFENS 732M

Follow-up to [exp286](exp286_explicit_board_encoding.md): instead of a residual
rule adapter on squares64, train ChessBot's own architecture at our 99M size
and depth on the same LCZero data ChessBot used. Goal: beat
[Maxlegrec/ChessBot](https://huggingface.co/Maxlegrec/ChessBot) (34.7M, 10×512)
search-free.

Fresh init. Do not copy squares64 weights or write the 99M incumbent.

## What is kept from ChessBot

- 19-plane encoding: 12 pieces, side to move, EP square, four castling flags,
  50-move clock. Rank-flipped so token 0 is a8, matching ChessBot, not our
  a1=0 squares64 order.
- MaGating, sinusoidal PE, Transformer-XL relative attention, GELU FFN.
- Global 64×64 QK policy, then a bias-free map onto the 1929-move ChessBot
  vocab. First 1858 slots are [Maxlegrec/ChessFENS](https://huggingface.co/datasets/Maxlegrec/ChessFENS)
  LC0 moves.
- Two WDL heads in ChessBot order `[black_win, draw, white_win]`.

Published ChessBot card says FFN 736; the actual `config.json` is `d_ff=1024`.
This experiment uses the checkpoint config, not the card typo.

## What comes from the 99M

- 736d, 8 heads, dropout 0.05.
- 4 prefix + 7 recurrent × 3 + 4 suffix = 15 unique / 29 effective layers.
- `d_ff=2112`, SwiGLU, QK-norm, zero-init out projections: **99,352,334**.
- Polar-NorMuon (2D) + AdamW (norms/heads), 99M pretrain LRs 0.02 / 3e-4.
- Recurrent gradient averaging after backward.

## Data

Stream `Maxlegrec/ChessFENS` (~732M rows, 133 GB). Do not download the whole
set unless you mean to. Rows are FEN + STM WDL + 1858-way soft policy
(illegal = -1). Convert WDL to ChessBot order. Rank-flip half the batch so
Black-to-move is actually trained; ChessBot's card notes the raw set is
white-perspective.

Loss: `(1-0.85)` hard policy CE + `0.85` soft policy CE + `0.15` ×
(hard WDL CE + soft WDL CE on the Q head). Same legal support on every arm.

## Commands

```bash
python experiments/exp287_chessbot_99m.py prepare
python experiments/exp287_chessbot_99m.py smoke
python experiments/exp287_chessbot_99m.py train
python experiments/exp287_chessbot_99m.py train --full-epoch   # one pass of 732M
python experiments/exp287_chessbot_99m.py eval --vs-chessbot
python experiments/exp287_chessbot_99m.py eval --sf
```

`configs/exp287_chessbot_99m.json` is a 100k-update pilot (6.4M effective
positions at batch 16 × accum 4). `--full-epoch` is the ChessBot-matched
data budget. This file does not rent hardware.

## Evaluation

Greedy legal policy, T=0, no book, no tablebases. Promote only after:

1. Direct match vs pinned ChessBot, same openings / both colors.
2. Shared SF 1900/2050/2200 screen vs that same ChessBot dump.

Teacher CE on ChessFENS is not Elo. A 100k-step pilot is not a claim.

The recommended next scientific experiment is not a longer scratch 99M run.
See [exp288](exp288_chessbot_recurrent.md): wrap published ChessBot as
2+6N+2, identity-check N=1, then conservative A/B continuation.
