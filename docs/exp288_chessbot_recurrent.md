# Experiment 288: recurrent depth on pretrained ChessBot

Status: design only; no training launched. Do not stop or redirect the live
exp287 scratch run for this. This is not a 99M hybrid, not a ChessBot
reproduction from scratch, and not an invitation to add SwiGLU, QK-norm,
Polar-NorMuon, or zero-init on existing trained projections.

## Question

Does extra recurrent computation improve [Maxlegrec/ChessBot](https://huggingface.co/Maxlegrec/ChessBot)
(published 34.7M, 10×512, `d_ff=1024`) after conservative continuation, or does
repeating layers that were trained to run once just hurt?

Separate three things: the published checkpoint, new-teacher continuation at
the original depth, and extra passes through a shared middle bank.

## Why this, not another 99M scratch run

exp287 tests a new hybrid from random weights on the full 732M stream. That is
a long, confounded pretrain. This experiment starts from demonstrated strength
and changes one structural fact: middle-layer reuse. Parameter count stays the
published 34.7M. Extra passes cost latency, not parameters.

## Architecture

Load the public ChessBot checkpoint with the existing card-compatible loader
(`scripts/elo_eval_chessbot.py`: config + safetensors, not broken
`from_pretrained` finalize). Keep 19-plane encoding, rank-flipped a8=0 order,
MaGating, sinusoidal PE fed to every layer, GELU FFN, 1929-move QK policy, and
both WDL heads. Do not alter layer internals.

Split the ten unique encoder layers as:

    prefix  layers 0–1   run once
    bank    layers 2–7   six modules, unrolled N times
    suffix  layers 8–9   run once

Effective depth = 2 + 6N + 2. N=1 is 10 layers and must be the published net.

| N | Effective layers | Parameters |
| --- | --- | --- |
| 1 | 10 | published 34.7M |
| 2 | 16 | unchanged |
| 3 | 22 | unchanged |

2/6/2 is the registered split, not an optimized one. 3/4/3, 1/8/1, and 4/2/4
are later ablations; do not run them in this budget. The bank is six distinct
modules, each reused across unrolls. It is not one block repeated six times.

A runtime `recurrent_unrolls` override changes N without changing checkpoint
keys. Default N=1. Average bank gradients by the actual N after backward, as in
exp283. N=1 averaging is a no-op.

## Identity gate (blocks all later work)

On CPU, then on the training device, N=1 must match the untouched published
model on real boards including startpos, EP, castling, promotions, and
Black-to-move.

1. Policy and both value heads agree within device-appropriate tolerance
   (CPU: tight; CUDA FP32: record the actual max abs error, fail if it is
   large enough to change legal argmax on a 256-position smoke set).
2. State-dict load is strict after remapping `layers.{0..9}` →
   `prefix.0/1`, `bank.0..5`, `suffix.0/1`. No missing or unexpected trained
   tensors.
3. Save/load of the wrapped model still matches the published model at N=1.
4. The published checkpoint still loads through the old path, unchanged.
5. Record the safetensors SHA256 and the wrap-time parameter count
   (expected ~34.7M; measure, do not assume the card).

If identity fails, stop. Do not train a broken wrap.

## Phase 0: frozen sweep, no training

Evaluate the wrapped published weights at N=1,2,3,4 on one frozen holdout
(see Data). Metrics: legal top-1, hard CE, soft CE, WDL CE, paired
rescues/regressions versus N=1, batched positions/sec, and greedy move
latency. This is the damage/help curve of extra passes before any
continuation. It is not Elo.

If N>1 already wins games against N=1 here, still run the training arms;
untrained extra passes are not a trained result.

## Arms

Every trained arm starts from the same published bytes after a successful
identity wrap. Fresh AdamW. No Polar-NorMuon.

| Arm | Train N | What it tests |
| --- | --- | --- |
| A | always 1 | New-teacher continuation at the original depth |
| B | shuffled 1/2, equal counts | Whether a second bank pass can be made useful |

Both arms see the same rows in the same order, same seeds, same update count,
same loss coefficients. B does extra FLOP on half its updates. Report wall
time, peak memory, and ms/move. Do not claim FLOP matching.

Loss, identical reductions and masks:

    CE = 0.45 hard policy + 0.55 new-teacher soft policy + 0.15 eligible WDL
    L  = CE(N_train) + 0.20 KL(student_{N=1} || frozen_published)

The KL is the only ChessFENS-era signal: a frozen copy of the published
checkpoint, one-pass student only. It is an anti-forgetting prior, not a
training set. A and B both pay it, so distillation is not credited to
recurrence. Stop the teacher graph. Temperature 1. No KD on N=2 logits.

WDL CE is masked unless the row is value-eligible (SF19 White-absolute only).
Convert that WDL to ChessBot order `[black, draw, white]`. Lichess, puzzles,
and Syzygy stay policy-only. Do not invent CP/WDL.

Do not add QK-norm, SwiGLU, zero-init of existing out-projections, mixed
precision, compile, or geometry. Those are later single-factor tests after a
playing-strength win, each with its own identity check.

If B fails the screen, stop. If B wins, a follow-up C may add shuffled 1/2/3
at a newly fixed budget. Do not sneak C into this run.

## Data

ChessBot was trained on [Maxlegrec/ChessFENS](https://huggingface.co/datasets/Maxlegrec/ChessFENS)
(~732M LCZero policies). Another epoch of that set is not new supervision.
**ChessFENS is banned from the training pool.** It may appear only as a 2,048-row
forgetting probe, never in the optimizer.

Train on teachers ChessBot has not published as its dataset: our Stockfish /
Lichess / puzzle / tablebase packs. Same four sources as
`chess_master/recipes/pilot_45_35_15_5.json`.

| Share | Source | Label | Value |
| --- | --- | --- | --- |
| 45% | `avewright/chess-soft-sf19` @ `68cef6c9…` | SF19 MultiPV soft policy | White-absolute WDL, eligible |
| 35% | `avewright/chess-soft-multipv-lichess` @ `291acea2…` | engine soft policy | off |
| 15% | `Lichess/chess-puzzles` @ `479ea9bc…` | one-hot solver move | off |
| 5% | `avewright/chess-soft-syzygy` @ `3889985c…` | tablebase policy | off |

Quality gates stay `quality_relaxation: never` (SF19 min depth 12 / 100k nodes,
Lichess depth 22–127, puzzles 600–3500, Syzygy WDL in ±2). Overlap policy is
`exclusive_source_bucket`. Prefer the frozen `organized_chess_v1` membership
as the audited 1M core, then **add unused rows from those same HF revisions**
until the pack hits 5,000,000 unique boards or the sources are exhausted.
Do not refill a shortfall by relaxing gates or by opening ChessFENS.

If the assembled unique-row count is U:

- Hold out 8,192 rows, stratified by source, then drop train overlap by FEN
  and by board state (pieces, turn, castling, EP).
- Training exposure is one pass through the remaining rows, rounded down to a
  multiple of 64. Updates = exposure / 64. A and B get that same count.
- Minimum shipable pack: the 1M organized_v1 mix (≈15,625 updates). Below
  that, stop and fix sourcing. Above 1M, use the extra unique rows; do not
  repeat the 1M just to look bigger.
- Persist source revision, row manifest, SHA256s, and the actual U. Same
  manifest for A, B, and seed 1288.

Map compact-1968 / UCI teachers onto ChessBot’s 1929 vocab by UCI. Knight
promotions drop the trailing `n`, matching ChessBot’s legal lookup. Rows that
cannot map a positive-mass teacher move are dropped from both arms, not
rescued with a different teacher.

Reconstruct FEN from the stored board (pieces, turn, castling, EP). Clocks
are almost all missing in these packs: encode halfmove as 0 and do not claim
clock supervision. Both colors stay as stored; no rank-flip augmentation.

A 2,048-row ChessFENS probe (frozen, hashed, disjoint from nothing we train)
is diagnostic only: if new-teacher CE falls while probe CE explodes, the arm
forgot LCZero behavior. That cannot promote an arm and cannot justify adding
ChessFENS to train.

Report source counts, unique boards, white-to-move rate, EP rate, mapped-move
coverage, and value-eligible fraction. Empty slices are not evidence.

## Training

Batch 64, AdamW, LR 1e-5, weight decay .01, 500-update warmup, cosine to 10%
of peak, clip 1, dropout 0 (ChessBot’s trained attention dropout is 0; do not
introduce 0.05). Recurrent gradient averaging uses the N of that update.
Update count is determined by the frozen pack (see Data), not a second
ChessFENS epoch.

Precompute sample indices for the seed. Reset RNG after wrapping each arm so
module construction does not shift data order. Seed 288 first; repeat A and
the selected candidate with 1288 only if the first seed’s game screen is
promising.

Save optimizer, scheduler, RNG, and sampler position. The comparison
checkpoint is the final step. Diagnostics at 25/50/100% do not authorize
picking a lucky intermediate.

First: wrap, identity, frozen sweep, 100-update smoke on both arms. Then
benchmark 100 updates on the real GPU and estimate hours for the full pack.
This document does not rent hardware and does not kill exp287.

## Validation and evaluation

Diagnostics on the frozen holdout at N=1 and N=2 (and N=3/4 for the frozen
sweep only): legal top-1, hard/soft CE, WDL CE, rescues vs regressions
versus published N=1. Not Elo.

Greedy legal policy, T=0, no book, no tablebases. Pin Stockfish 19,
threads=1, hash=32MB, 50ms/move, 400-ply cap. Reevaluate the published
checkpoint under that protocol in the same session. Report ms/move at each N.

Screen at the end of seed 288, 96 games per evaluated snapshot (eight
openings, both colors, two repeats) at SF 1900/2050/2200:

- published ChessBot (N=1)
- A at N=1
- B at N=1 and N=2

Choose at most one candidate. Prefer B@N=2 if it beats both published and A
on aggregate game score. Exact ties go to fewer passes, then to A. If nobody
beats A, stop.

Confirmation, only if the screen selected B: 200 held-out opening pairs
(400 games) against A, then 400 games against published ChessBot. Repeat
with seed 1288 using the same openings. Bootstrap opening clusters. Report
capped games separately; extend those to 800 plies as a sensitivity check.
If caps change the call, the result is inconclusive.

Practical target: ≥30 relative Elo versus A **and** versus published ChessBot,
paired 95% interval above zero, same direction in both seeds, and no material
SF-ladder regression. 400 games can still be inconclusive. Loss is not
promotion.

## Interpretation

- B@2 beats A and published: extra trained passes help; keep the wrap and
  replicate. Then consider C (1/2/3) or a single optimization (AMP or LR).
- B@2 beats A but not published: continuation recovered poorly; inspect
  forgetting before blaming or praising recurrence.
- B@1 beats published, B@2 does not: extra passes are harmful; do not ship N=2.
- Only A beats published: new-teacher continuation helped; recurrence did not.
- Nobody beats published: this budget does not support altering ChessBot.
- Probe CE explodes while train CE falls: the arm forgot LCZero; do not
  “fix” that by training on ChessFENS. Shrink LR or raise KL and rerun A/B
  as a new budget if we still care.
- Frozen N>1 already stronger, training then worse: the wrap is usable at
  inference-only extra depth; say that, and do not sell the trained arm as
  the reason.

This cannot show that 2/6/2 is optimal, that 99M recurrence was the right
idea, or that SF19 is a better teacher than LCZero. It asks whether their
trained 10-layer net can use a second look on labels it was not published as
having trained on.

## Implementation deliverables

When executed: ChessBot recurrent wrap with strict identity tests, frozen
depth sweep, compact1968→ChessBot-1929 adapter, new-teacher pack builder
(no ChessFENS train rows), matched A/B trainer with one-pass KD, Elo hooks
through the existing ChessBot move path, and a results manifest. No runner
or result is implied by this file.

The live exp287 hybrid remains a separate scratch pilot. Do not write its
weights into this experiment.

PPO, search-free winner-move CE, and any other outcome RL are exp289. They
are unused labels, but they confound recurrence. Do not fold them in here.
