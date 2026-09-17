# Experiment 289: PPO on published ChessBot

Status: design only; no training launched. Do not stop or redirect the live
exp287 scratch run. Do not fold this into exp288. This is not a recurrent wrap,
not a 99M hybrid, and not another ChessFENS epoch.

## Question

Does on-policy PPO, anchored by a frozen published ChessBot and unused
teacher replay, improve [Maxlegrec/ChessBot](https://huggingface.co/Maxlegrec/ChessBot),
or does search-free outcome noise just wreck a strong supervised net?

The unused signal is the game result. ChessFENS already spent LCZero policy
and WDL on this checkpoint. Outcomes from new games are not that dataset.

## Why this, not RL inside 288

exp288 asks whether a second bank pass helps after new-teacher SL. PPO asks
whether unused outcomes help the published net. Mixing them credits neither.

`rl_selfplay/ppo.py` is the loop we already trust on the 99M, and it cannot
be pointed at ChessBot as-is: it requires `Squares64RecurrentTransformer`,
compact-1968, and White-absolute `P(White)-P(White loses)` on `value_logits`.
ChessBot is 1929-move QK policy, 19-plane encoding, and WDL
`[black, draw, white]`. Copy the algorithm, not the module.

Do not use exp281 winner-move CE. That recipe only trains on student wins
and throws the rest away. Do not use evolutionary `selfplay.py`. Do not add
MCTS. We do not have a ChessBot search stack, and this budget does not build
one.

## Why not RL instead of the SF19 pack

Outcome credit over a 60-move game is high variance. The 45/35/15/5 pack is
cheap unused supervision. Keep it as **replay**, not as the only experiment
and not as a reason to skip PPO. Replay every update. Do not run a separate
SL warmup arm in this budget; the 0.25 replay weight is the warmup.

If we want a clean SL-only comparison, that is exp288 arm A without the wrap.
Do not re-run it here.

## Architecture

Load the public ChessBot checkpoint with the existing card-compatible loader.
N=1 only. No wrap, no extra unrolls, no SwiGLU, QK-norm, Polar-NorMuon, AMP,
or compile.

Use the inference value head (`value_logits_q` in `chess_chessbot.py`). Leave
`value_head` in the graph so load stays strict, but GAE and the value loss
read the q-head only. Convert `[black, draw, white]` to White-absolute scalar
with `P(white) - P(black)`, then flip into the actor frame exactly as
`actor_value` does today. Do not reuse `wdl_value()` from `rl_selfplay/ppo.py`;
that function assumes class 0 is White.

Policy sampling is legal-masked 1929, temperature 0.8, knight promotions
without a trailing `n`. Dropout off for collection and updates.

## Identity and value-frame gates (block all later work)

On CPU, then on the training device:

1. Frozen published N=1 still matches the untouched card loader on startpos,
   EP, castling, promotions, and Black-to-move. Policy argmax and both value
   heads stay within the same tolerances as exp288.
2. Startpos White-absolute value is positive. After `e4`, still White-positive.
   After a one-ply mate in 1 for White, White-absolute value is above 0.5.
   After the same mate in 1 for Black, it is below −0.5. If any of these fail,
   the advantage sign is wrong; stop.
3. A 4-game smoke (both colors, ply cap 16) produces only legal UCI, records
   one actor transition per actor ply, and terminates or truncates without
   mutating the frozen reference.
4. Record the safetensors SHA256 and parameter count (~34.7M; measure).

If any gate fails, stop. Do not train a sign-flipped critic.

## Loop

Reuse the exp285 iteration, with ChessBot encoders and 1929 legal masks.

Each iteration:

1. Collect paired-color games against either the frozen published net (50%)
   or a recent league snapshot. Same opening for both colors. Actor
   temperature 0.8; opponent 0.8. Ply cap 400. Intermediate rewards 0;
   terminal is win/draw/loss from the actor. Truncation bootstraps the
   actor's next decision.
2. PPO-clip the behavior log-prob, clip the q-head value, add
   `KL(current || frozen_published)` over the same legal 1929 support, and
   mix new-teacher replay. No KL against league snapshots. No KD on
   ChessFENS rows.
3. Snapshot the actor into a league of 4. Optimizer and RNG live only in
   `latest.pt`.

`STOP` is honored at an iteration boundary. Resume requires an identical
config, including the replay SHA256. Output directories must be new or empty
unless `--resume`.

Registered pilot: 20 iterations × 32 paired games, 2 epochs, minibatch 32,
LR 2e-6, clip 0.15, value clip 0.2, value weight 0.5, reference KL 0.02,
entropy 0.001, target KL 0.02, GAE λ 0.95, γ 1.0, replay weight 0.25. Same
numbers as `configs/exp285_ppo_99m.json` except `depths` is absent (always
one published pass).

If the 20-iter smoke does not collapse legal play and does not explode
reference KL, a second budget may extend to 200 iterations on the same
replay SHA256. That extension is a new run, not a silent continue of a
failed 20.

## Data

**Replay (optimizer sees these rows):** the exp288 new-teacher pack.
ChessFENS is banned. Map compact-1968 UCI → ChessBot 1929. Value CE only on
SF19 White-absolute rows, converted to `[black, draw, white]`. Soft policy
CE uses the teacher distribution on mapped legal slots.

If the 5M pack is not assembled yet, replay is the frozen
`organized_chess_v1` mix (1M). Do not refill with ChessFENS. Persist SHA256.

**Rollouts (the unused labels):** games the actor plays in this run. They
are not a stored teacher dataset. Do not mix published ChessFENS FENs into
openings just to “use more data.” Openings are a frozen 8-position book plus
both colors, same as the Elo screen.

**Probe (not trained):** 2,048 ChessFENS rows, diagnostic only. If replay CE
and outcome improve while probe CE explodes, the actor forgot LCZero. Raise
KL or cut LR in a new budget. Do not add ChessFENS to replay to “fix” it.

## Evaluation

The loop does not play Stockfish and does not claim Elo.

Diagnostics every 5 iterations: legal move rate, mean entropy, reference KL,
replay hard/soft CE, probe CE, actor win/draw/loss vs frozen published and
vs league, ply-cap fraction. Not promotion.

Greedy legal policy, T=0, no book, no tablebases. Pin Stockfish 19,
threads=1, hash=32MB, 50ms/move, 400-ply cap. Reevaluate published ChessBot
in the same session.

Screen after the 20-iter pilot, 96 games (eight openings, both colors, two
repeats) at SF 1900/2050/2200:

- published ChessBot
- PPO actor

Choose the actor only if it beats published on aggregate game score. Exact
ties stay with published. If the actor fails, stop; do not launch 200
iterations.

Confirmation, only if the screen selected the actor: 200 held-out opening
pairs (400 games) against published. Repeat with seed 1289 using the same
openings. Bootstrap opening clusters. Report capped games separately.

Practical target: ≥30 relative Elo versus published, paired 95% interval
above zero, same direction in both seeds, and no material SF-ladder
regression. 400 games can still be inconclusive. A self-play win rate
against old league snapshots is not promotion.

## Interpretation

- Actor beats published on the confirmation screen: unused outcomes helped;
  keep the PPO loop and consider the 200-iter budget or, separately, exp288
  recurrence on this snapshot.
- Actor beats league but not published: it exploited a weak copy of itself.
  Not a win.
- Legal play collapses or reference KL hits the target every minibatch:
  the prior died. Stop. Do not “rescue” with ChessFENS replay.
- Replay CE falls, games do not: it is imitating SF19/Lichess and not using
  the outcome. Lower replay weight in a new budget; do not call that this
  result.
- Probe CE explodes: it forgot LCZero. Same rule as exp288; do not train on
  ChessFENS.

This cannot show that PPO beats MCTS-RL, that 2e-6 is the right LR, or that
the 99M would have done better. It asks whether search-free outcomes can
move a published 34.7M ChessBot that already saw 732M LCZero labels.

## Implementation deliverables

When executed: ChessBot-native collect/PPO (1929 mask, q-head value frame),
strict identity and sign tests, new-teacher replay loader, league snapshots,
Elo hooks through the existing ChessBot move path, and a results manifest.
Do not import `check_model` from `rl_selfplay/ppo.py`. No runner or result
is implied by this file.

The live exp287 hybrid remains a separate scratch pilot. exp288 remains the
recurrence-plus-SL design. Do not write either run's weights into this
experiment unless a later budget names that checkpoint as the start.
