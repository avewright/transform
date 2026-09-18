# ChessBot RL evaluation contract

Three questions, three kinds of evidence. PPO loss is never a strength signal.

| Question | Evidence |
|---|---|
| **Is it stronger?** | Greedy paired match vs frozen one-pass ChessBot and vs a fixed incumbent |
| **Is RL responsible?** | Same candidate vs the matched control (anchor distill, no PPO) |
| **Does recurrence help?** | The **same checkpoint** at one pass vs two passes |

Live development uses the existing 32 opening pairs. A larger confirmation set is required before replacing the incumbent. The running job is not modified by this contract; new code applies on the next launch.

## Verdicts

Every match is labeled exactly one of:

- `not_evaluated` — no completed color-swapped pairs
- `inconclusive` — 95% paired interval contains 0.5
- `stronger` — interval lies entirely above 0.5
- `weaker` — interval lies entirely below 0.5

`score_bounds` only describe unfinished games (pessimistic / optimistic). They are **not** confidence intervals.

Statistical uncertainty treats each color-swapped opening as one unit (Fishtest paired-game practice). Pair game-score is the mean of the two sides, each scored 1 / 0.5 / 0. The 95% interval is a Student-t interval over those pair scores. Pentanomial counts (0, 0.5, 1, 1.5, 2 pair points) are logged. This is not a sequential SPRT and not an Elo claim.

## Opponents

- `original` — untouched one-pass published ChessBot, loaded separately
- `incumbent` — initialized to published ChessBot (actor-shaped weights at identity init). Replaced **only** after a development `stronger` plus a separate confirmation `stronger`
- `control` — frozen copy of the matched no-PPO net
- `previous` — last league snapshot **excluding** the checkpoint just written. Adding the current file to the league before selecting previous is a self-match and is forbidden
- `self_n1` — candidate weights forced to one pass

Development screens run after every completed iteration. Confirmation uses `confirm_pairs` (default 96) openings disjoint from the development set.

## Recurrence diagnostics

Logged each iteration, independent of match verdicts:

- gate `tanh(alpha)` and `dL/d alpha`
- RMS of `bank(h) − h` after the first bank pass
- legal-argmax disagreements between one and two passes, plus those UCIs
- optional fixed-node Stockfish preference on disagreements
- added move latency (ms)

## Board inspector

For a small FEN list (start + validation), store top-five legal moves and Q-values from:

published ChessBot · incumbent · candidate one-pass · candidate two-pass

Highlight N=2 changes vs published and vs N=1. Improvements/regressions are match + SF judgements, not policy-loss signs.

## Dashboard

Must show the three questions, the four verdict words, paired CI (not `score_bounds`), gate, and the inspector. Falling PPO loss must not flip a verdict to `stronger`.
