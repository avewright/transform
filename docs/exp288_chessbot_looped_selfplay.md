# Experiment 288: pretrained ChessBot, looped transformer, league self-play

Status: experimental design only. No training launched, rental changed, or
exp287 interrupted. Start from published Maxlegrec/ChessBot weights (~34.7M),
not the fresh 99M hybrid. Pin repository revision and file SHA256s before running.

## Objective

Improve greedy, search-free playing strength with outcome-based reinforcement
learning, and test whether shared extra transformer passes make RL more useful.
PPO is the initial algorithm; no MCTS is used in this pilot. Recurrence performs
additional computation on the current board, not explicit future-board search.
Extra passes do not carry hidden state between moves.

## Preserve the trained model

Keep the original 19-plane encoding, square order, move vocabulary, attention,
GELU FFNs, post-LayerNorm blocks, policy head, and both WDL heads. Do not insert
QK normalization, replace GELU with SwiGLU, or reset pretrained projections.
Load weights strictly and verify outputs against the published implementation.
Audit every promotion and castling mapping; do not inherit a fallback that turns
an ambiguous promotion into a queen by default.

Split the ten blocks: prefix 1–2, shared bank 3–8, suffix 9–10.

    h = prefix(encode(board))
    h = bank(h)                         # original path
    repeat N-1 times:
        h = h + tanh(alpha) * (bank(h) - h)
    output = original_heads(suffix(h))

N=1/2/3 executes 10/16/22 transformer blocks. Reuse bank weights and the original
positional encoding on each pass. One shared scalar alpha starts at zero; this
preserves original outputs for every supported depth, up to numerical tolerance.
The added gate is outside the complete post-normalized bank; placing a new norm
after this blend would break identity preservation. At alpha=0 its gradient can
be nonzero although the additional branch's weight gradients are zero. Existing
bank weights still learn through the first, ungated pass. No second zero gate.

This is damped recurrence, not six newly initialized layers. Record alpha,
tanh(alpha), refinement norm relative to h, and per-depth predictions. A gate
remaining near zero means recurrence was not learned. Negative gates are allowed
and must be reported; do not interpret all learned corrections as deeper search.
Parameter increase: one scalar. Inference cost increases with N.

## Stage 0: integration and calibration

Build a local wrapper, with old checkpoints still loadable. Read the existing
exp285 PPO collector/update implementation as reusable infrastructure, but adapt
its vocabulary, encoding and value frame explicitly. Do not assume drop-in support.

Required tests: N=1 original-output equivalence; alpha=0 N=2/3 equivalence; nonzero
gate gradient; strict save/load; train/inference feature parity; board/color flips;
all promotion types; terminal mates/draws; correct signed rewards for both colors;
PPO likelihood ratio exactly 1 before updating on saved rollout decisions.

Use full python-chess Board histories in the environment, even though the model
observes FEN only. Repetition is not reconstructible from a FEN. Define automatic
claimable draws identically in collection and evaluation; document this convention.

Reserve development and confirmation openings plus ChessFENS replay/validation
splits before training. Audit policy-index ordering, illegal=-1 handling, WDL frame,
and Chess960 rows. Exclude Chess960 or ambiguous castling rows from standard-chess
replay; document counts. Group replay/validation splits by board state, ignoring
clock, and by source game when available. Historical overlap with published
pretraining data is unknown; do not call this unseen-pretraining evaluation.

Measure original model's greedy strength on our pinned protocol and benchmark
N=1/2 on the actual GPU. Do not extrapolate the 99M run's training speed to RL.
Measure games/hour, actor decisions/sec, update time, peak VRAM, and move latency.
Start with 32 concurrent environments and batch by model/depth; increase only
after measuring. BF16/compilation are optional after numerical checks. Keep logits,
legal softmax, log probabilities, advantage and loss arithmetic in FP32.

## Stage 1: conservative recurrence preparation

Create two branches from the same original weights:

- Fixed: N=1 continuation.
- Looped: sample N=1/2 equally, train alpha and shared weights.

Budget 250,000 ChessFENS position presentations per branch, identical row order
and replay labels. AdamW LR 1e-6 for pretrained weights, 1e-4 for alpha, clip 0.5,
100-update warmup, effective batch 128, no dropout. Retain original module dropout
settings in metadata but disable stochastic dropout during this experiment.
Use .85 soft policy CE + .15 argmax-policy CE + .15 soft WDL CE on Q head,
plus .1 KL(current one-pass || frozen original) on a matched anchor batch.
Loss reductions must be batch means, not sums over varying legal move counts.
Never use argmax of probabilistic WDL as a substitute for actual game outcomes.

Validate before/after and screen actual games. A looped branch clearly regressing
against its starting point does not automatically advance to an expensive RL run.
This budget is adaptation, not a claim that recurrence has converged.

## Stage 2: controlled PPO self-play

Four arms separate RL improvement from additional replay and recurrence:

| Arm | Stage-1 source | Continuation |
| --- | --- | --- |
| F-replay | Fixed | Matched supervised replay updates only |
| F-RL | Fixed | PPO + reference KL + same replay |
| R-replay | Looped | Matched supervised replay updates only |
| R-RL | Looped | PPO + reference KL + same replay |

Replay-only controls receive exactly the same replay rows, optimizer update count,
LR and replay coefficient as their RL counterpart, but no PPO, RL value, entropy,
or rollout-reference losses. They need not generate games. This is an ablation of
the outcome-learning package, not equal total GPU time. Report total GPU-hours.
Also compare R-RL against its frozen stage-1 checkpoint at each inference depth.
F-RL versus R-RL is matched by actor decision count; also plot strength/GPU-hour
and strength/move-latency. Extra recurrent compute is not free.

One iteration collects exactly 32,768 actor decision transitions before updates,
with a frozen behavior checkpoint throughout collection. Freeze opponent pool
for that iteration. Pilot: 20 iterations = 655,360 decisions per RL arm. Any
unfinished environments continue across collection segments; never relabel a
segment boundary as a draw. A changed actor between segments is recorded.

League per paired opening: 40% untouched original ChessBot, 40% uniform among
up to four historical actor snapshots, 20% latest frozen actor snapshot. Empty
historical slots use the original. Store snapshots every five iterations, with
one opponent identity/depth fixed for both games in a color-swapped opening pair.
Only learner decisions enter PPO; frozen opponents' moves never do.

F-RL uses N=1. R-RL samples N=1/2 equally once per actor game and retains it for
the game, recording N with every transition; do not resample depth during PPO.
An opponent snapshot's depth is likewise sampled/fixed. No adaptive halting yet.

Learner sampling: legal-masked softmax at T=0.8 throughout. Frozen league players
also use T=0.8 during collection. PPO likelihoods use exactly this temperature,
mask, depth and action mapping in both old and current policies. Evaluation uses
greedy argmax. Do not add unrecorded epsilon-greedy or post-sampling move repairs.

Reward: +1 win, 0 draw, -1 loss from learner color. No material, engine-evaluation,
move-length, or win-only imitation bonuses. A transition spans the actor move and
opponent reply, so consecutive states share the same learner perspective. If the
actor move ends the game, terminate immediately. Convert Q-head output from
[black win, draw, white win] to actor-frame scalar V=P(actor wins)-P(actor loses).

GAE gamma=1.0, lambda=.95. True terminal: bootstrap zero. Segment/training-time
truncation: bootstrap from the next actor decision with matching behavior model
and depth; stop GAE propagation across the boundary. At an 800-ply safety limit,
discard/reset the game only after marking it truncated, never as a draw. Log rate.
Persistent heavy truncation is a diagnostic failure requiring review.

Loss = PPO-clip + .5 critic MSE + beta*KL(current || original)
       - .001 legal-policy entropy + .25 supervised replay loss.

PPO clip=.1, two epochs, minibatch 256 actor decisions (microbatch/accumulate to
fit). Normalize advantages over the rollout, recording win/draw/loss and depth
subgroups. Use Q-head scalar as critic; retain the other WDL head via replay.
Replay soft-policy and WDL losses use audited ChessFENS rows. Never mix replay
actions into on-policy PPO ratios. Update all pretrained weights at LR 1e-6;
gate LR 1e-4; AdamW weight decay 0.0, grad clip .5. Do not divide recurrent bank
gradients by N in this pilot: use the actual unrolled objective's gradient.

KL references are distinct: original-checkpoint KL resists forgetting; old-behavior
KL measures PPO update size. Both use the same legal support and T=.8. Start
beta=.02; after each iteration double beta if reference KL>.04 nats, halve if
<.01, bounded [.005,.5], target .02. Stop further PPO epochs when measured
old/current empirical KL exceeds .01. Log actual KL, clip fraction, entropy,
policy/value/replay losses, explained variance, gradient norms, and legal failures.
Thresholds are pilot hypotheses, not established ChessBot optima.

Freeze pilot hyperparameters before comparing arms; numerical failure or invalid
probabilities triggers rollback and diagnosis. Lower loss or beating stale league
members never automatically promotes a model. Require both colors to improve.

## Evaluation, budget gates and stopping

Save versioned milestones at iterations 0/5/10/20 plus atomic resumable latest.
Checkpoint optimizer, RNGs, sampler/replay cursor, gate, league identities, counters,
and complete serialized environment histories if mid-game resume is supported.
Otherwise explicitly restart fresh games and label resume as non-exact.

Development screens use fresh paired-color games versus original ChessBot and
the same pinned Stockfish 19 at UCI_Elo 2400/2600/2800, 50ms, threads=1, hash=32MB.
Record binary hash, openings, hardware and settings. Treat UCI_Elo as a benchmark
setting, not a human rating. No search/book/tablebase help for the model.
Use 800-ply cap and report unresolved games, never silently score them as draws.

Pre-register iteration 20 for selection; interim screens are collapse diagnostics.
Select at most one recurrent depth from N=1/2 using development results; N=3 is
an extrapolation diagnostic only until explicitly trained in a later experiment.

Confirmation: 200 fresh opening pairs (400 games) versus untouched ChessBot and
400 versus the selected candidate's matched replay control; repeat the relevant
training arms with a second seed. Also compare fixed versus looped finalists at
their declared inference budgets. Freeze choices before opening confirmation data.
Bootstrap complete opening pairs, clustering shared openings across seeds.

Target >=30 relative Elo over original and improvement over replay-only control,
with paired 95% CI above zero and consistent direction across seeds. For unresolved
games, report best/worst-case score bounds; if they change the decision, inconclusive.
Small samples may not resolve 30 Elo. Do not repeatedly extend until significant;
a larger confirmation requires a new fixed budget. Report gains versus both
inference latency and total training GPU-hours, not parameter count alone.

Execution gates: (1) correctness smoke; (2) 256-game throughput benchmark;
(3) five-iteration operational check; (4) finish the predeclared 20-iteration pilot
only if stable; (5) second seed and independent confirmation if promising.
Compute hours/cost from measured completed games and update timing. Current A5000
is a plausible single-GPU test platform, but no runtime/cost promise before profiling.
Do not run concurrent training over exp287 or stop it without a separate instruction.

## If outcome-only PPO stalls

First check exploration, value calibration, truncations, color handling and KL
constraints. Do not immediately enlarge the network. A separately budgeted next
experiment can generate batched MCTS self-play (initially 32/64 simulations), train
on root visit distributions and game outcomes, and evaluate greedy policy afterward.
This is search-assisted expert iteration, not the same PPO arm. Recurrence and tree
search multiply compute; benchmark both. No promise that search-free PPO or extra
loops will surpass the original model.

## References and implementation targets

- ChessBot model/config: https://huggingface.co/Maxlegrec/ChessBot
- PPO: https://arxiv.org/abs/1707.06347
- AlphaZero self-play/search alternative: https://arxiv.org/abs/1712.01815
- Reuse concepts from docs/exp285_ppo_league.md and experiments/exp285_ppo_league.py.

To implement: pretrained-compatible recurrent wrapper; audited ChessBot adapter
for the collector; replay preparation; fixed/looped PPO and replay-only runners;
stage-1 adaptation; correctness tests; pinned evaluation and paired reporting.
This document does not imply those executable components already exist.
