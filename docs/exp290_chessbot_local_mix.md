# Experiment 290: published ChessBot on dedicated phase packs

Continue [Maxlegrec/ChessBot](https://huggingface.co/Maxlegrec/ChessBot)
(34.7M, 10×512) on unused local/HF teachers. ChessFENS is banned.
Do not stop exp287. Do not write the 99M squares64 incumbent.

## Architecture

2/6/2 wrap of the published layers. N=1 is identity. Train shuffled
unrolls `{2, 3}` so effective depth is 16 or 22. Frozen published KL
on the N=1 student. Slow linear warmup (8,000 updates) to AdamW 1e-5,
then cosine to 10%.

The bank is repeated directly. There is no identity-preserving residual
gate, so N>1 changes behavior immediately. There is also no matched
one-pass training control: evaluating the same weights at N=1 is useful
but does not isolate recurrence from ordinary fine-tuning.

## Even mix (20% each)

| Bucket | Source |
| --- | --- |
| opening | `avewright/lichess-opening-bestline` |
| middlegame | `avewright/lichess-middlegame-bestline` |
| endgame | round-robin `lichess-endgame-bestline` + `endgame-dataset` + Syzygy |
| puzzles | `Lichess/chess-puzzles` |
| soft | `avewright/chess-soft-sf19` |

These are dedicated packs, not a phase filter on MultiPV-Lichess.
Value CE only on SF19 / endgame-dataset rows. Compact-1968 maps to
ChessBot 1929 (knight promos drop `n`; king-to-rook castle → e1g1).

The endgame bucket must rotate across all three sources. A sequential
`for source in sources` over an infinite parquet reader never leaves
the first Lichess endgame pack.

## Permanent holdout

Each source reserves first-shard rows `[8192, 16384)`. Validation is
drawn only from that window. Training skips it on every epoch, including
wraparound. The old v1 cache (`skip 8192, then take`) was not excluded
from train and is not evidence of generalization.

Val reports N=1, N=2, and N=3.

## Status

Contaminated run stopped after step 4000. Frozen copies:

- `outputs/exp290_chessbot_local_mix/preserved/step4000.pt`
- `outputs/exp290_chessbot_local_mix/preserved/eval_step3400.pt`

Do not continue to 5M presentations until this checkpoint plays paired
greedy games vs published ChessBot.

## Promote

Greedy T=0 games vs published ChessBot, then the SF ladder. Loss is
not Elo.
