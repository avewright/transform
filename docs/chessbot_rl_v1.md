# ChessBot RL v1: conservative PPO from the published checkpoint

Status: design, not an executable runner. Supersedes the recurrence-first scope
of exp288 for the next RL pilot. Does not stop or alter any remote experiment.

## Question and fixed scope

Can game-outcome reinforcement learning improve the original ~34.7M ChessBot's
greedy playing strength without losing its pretrained competence?

Start from a pinned revision of Maxlegrec/ChessBot, never exp290 or the fresh 99M.
Keep original ten layers, single pass, encoding, policy vocabulary, normalization,
feed-forward blocks, and both value heads. No recurrence, new heads, search,
teacher engine labels, or winner-only imitation. All weights remain trainable.
No preliminary supervised fine-tuning. This isolates RL from the changes that
previously hurt Elo.

Use PPO with a frozen-reference anchor and a historical opponent league. These
are pilot choices, not evidence that PPO will improve this already strong model.

## Three comparisons

| Arm | Initialization | Updates |
| --- | --- | --- |
| Original | Published weights | None; permanent benchmark |
| Anchor control | Same weights | Frozen-reference distillation only |
| PPO | Same weights | Outcome PPO + critic + reference anchor |

Anchor control uses the same update count, minibatch indices, anchor positions,
optimizer settings, and anchor losses as PPO, but no reward/value/entropy losses.
Rollout positions supplied to its anchor term are PPO's collected observations;
there is no separate off-policy PPO update. Control is near a no-op initially,
which is intentional: it detects pipeline/optimizer drift. These comparisons test
outcome learning beyond preserving the reference, not additional SF supervision.

## Data and reference protection

Create a frozen 100k-position anchor bank from existing source FENs, balanced by
phase. Ignore their old best-move/value labels: targets come solely from the
original ChessBot. No ChessFENS download is required. Preserve real clocks, EP,
turn and castling; do not reconstruct clocks as zero silently. Filter invalid,
terminal, unsupported and Chess960 positions. Freeze hashes and source counts.

Reserve a separate 10k-position diagnostic set permanently excluded by board-state
hash, ignoring halfmove clock so clock variants stay in one split. Group by source
game when available. Training must skip these hashes on every pass. Never implement
holdout as a temporary offset into a stream. Deduplicate across all sources.
These splits are held out from this continuation, not proven unseen in pretraining.

At each optimizer step compute:

- current-to-original policy KL on rollout states;
- the same KL on 128 sampled anchor states;
- Q-head soft WDL distillation on the anchor states.

Reference is always immutable original ChessBot, evaluated at the same policy
temperature and legal support as the student. Its predictions can be cached for
anchor states. Keep FP32 log probabilities; renormalize strictly over legal actions.
The unused non-Q WDL head remains frozen; the shared trunk can still change it,
so do not claim that head's outputs are preserved.

## Self-play collection

Start with 32 concurrent environments, increasing only after benchmarking.
Each iteration freezes the learner behavior weights and opponent pool, then
collects 32,768 learner decision transitions. No asynchronous policy lag in v1.

Opponents, sampled per opening pair:

- 40% untouched original;
- 40% uniform over up to four recent actor snapshots;
- 20% frozen current actor.

Empty historical slots use original. Add a snapshot every five iterations.
Each opening is played with learner as White and Black against the same opponent
snapshot. Opponent weights never update within a game. Train only on learner moves.
Use a fixed training opening bank of 2–8 plies; keep separate development and final
confirmation banks with no duplicate starting positions or shared opening IDs.

All collection policies sample the full legal softmax at T=.8. No top-k truncation,
untracked epsilon exploration, heuristic fallback, or engine assistance. Greedy T=0
is reserved for evaluation. Disable dropout in collection AND gradient updates
(eval mode does not disable autograd). Record behavior log-prob and critic value.

Environment stores full move history; the model still observes only the current
FEN. Automatically accept claimable draws consistently in all arms and evaluation.
This is a documented environment convention, not a learned draw-claim action.

Reward is +1 win, 0 draw, -1 loss in learner-color frame. No engine score, material,
length, or repetition penalty. Consecutive learner states span its move and the
opponent reply, keeping perspective constant. Terminal on the learner move ends
the transition immediately. After an opponent terminal reply, assign reward to
the learner's preceding action; never drop the loss.

Rollout boundaries bootstrap and retain environment histories. On the next
iteration an unfinished game's learner uses the new frozen behavior snapshot;
opponent stays fixed. Store behavior identity per transition. At an 800-ply safety
cap, bootstrap at the next learner decision, mark truncated and reset. Never turn
caps into draws. If caps exceed 5% of completed/reset games, pause to diagnose.

## PPO update: initial settings

| Setting | Value |
| --- | --- |
| Iterations | 20 |
| Learner decisions / iteration | 32,768 |
| Total decision budget | 655,360 |
| PPO epochs / iteration | 2 maximum |
| Effective minibatch | 256 rollout decisions |
| Microbatch | 32; accumulate consistently |
| Anchor minibatch | 128, sampled once per optimizer update |
| Optimizer | AdamW, LR 1e-6, weight decay 0 |
| PPO ratio clip | .10 |
| Gamma / GAE lambda | 1.0 / .95 |
| Critic coefficient | .5 |
| Reference policy KL coefficient | .05 on each of rollout and anchor means |
| Anchor Q-WDL coefficient | .10 |
| Entropy coefficient | .001 |
| Global gradient clip | .5 |
| Behavior-policy KL stop | .01 nats |

Critic = P(learner win)-P(learner loss), using original Q-head probabilities.
Original class order is [black win, draw, white win]; explicitly convert for color.
Check this against the pinned source and test both colors before running.
Use critic MSE against lambda-return targets, without target clipping to [-1,1].
Targets outside the critic range are logged as a diagnostic. Normalize advantages
over the rollout. gamma=1 avoids an artificial preference for shorter wins.

Minimize:

    L_PPOclip + .5 L_value
    + .05 KL_rollout(current || original)
    + .05 KL_anchor(current || original)
    + .10 CE_anchor_Q(original_WDL, current_WDL)
    - .001 H(current legal policy)

All terms are batch means, not sums over legal moves. Policy ratios recompute
the sampled-action probability at T=.8 using the identical mask. At a true
terminal bootstrap 0; at a truncation bootstrap the matching behavior critic and
cut GAE recursion. Policy/critic targets and advantages are detached.

After each optimizer update estimate old/current behavior-policy KL on a fixed
2,048-state rollout audit subset; stop remaining PPO updates for the iteration
if mean KL exceeds .01. Record exact control update count. Check reference KL
at T=1 on diagnostics separately; pause if mean exceeds .10 nats for two successive
iterations. These are conservative operational guards, not Elo guarantees.
Do not silently change LR or coefficients mid-run. A changed recipe gets a new ID.

## Required correctness gates

1. Strict checkpoint load and policy/Q-value agreement with original implementation.
2. One-to-one legal UCI action mapping, including queen and all underpromotions,
   both castles, en passant, and positions with one legal move.
3. PPO ratio=1 before updates; finite logits/log-probs; zero illegal sampling.
4. White/Black terminal rewards and critic conversion, mate on either ply,
   repetition/50-move draws, rollout boundary and safety-cap bootstrapping.
5. Replaying stored transitions reconstructs identical masks and probabilities.
6. Anchor/diagnostic exclusion persists across epochs and source restarts.
7. Resume reproduces next sampled batch/action when exact resume is claimed.

No randomly initialized model smoke test substitutes for real-checkpoint parity.
Use FP32 first. BF16 or compilation requires separate parity and sampled-action
checks; keep probability/loss arithmetic FP32. Do not use the existing 99M PPO
collector unchanged without verifying its feature and vocabulary assumptions.

## Operational stages and budget

Stage A: integration tests plus 64 complete games to measure collection/update
time, moves/game, truncation rate and peak memory. Do not estimate RL throughput
from exp287/290 supervised positions/sec. Benchmark on the current A5000 only
when it is available; don't compete with active runs or stop them implicitly.

Stage B: a two-iteration engineering smoke. Resolve bugs, then reset both arms to
original before the scientific pilot. Freeze hyperparameters and opening banks.

Stage C: run the fixed 20-iteration pilot. Save immutable iteration 0/5/10/20
checkpoints and atomic latest. Diagnostic game screens at 5/10/20 use development
openings only. Interim screens can stop clear collapse, not select a lucky winner.
Decision point is iteration 20; further training needs a separately fixed budget.

Log wins/draws/losses by opponent and learner color, uncertainty, completed games,
decisions, entropy, explained variance, policy/value/anchor losses, KL, clipping,
gradient norms, illegal actions, clock/repetition terminations, and capped games.
Self-play win rate near 50% against its own snapshot is expected, not failure.

Store full run config and data/model hashes, optimizer, RNG, anchor sampler,
league revisions, rollout counters and environment histories. Persist behavior
models needed for unfinished data or restart only at an explicit clean boundary.
Weights-only reload is not exact resume. No uploads or automatic promotion.

## Strength evaluation and interpretation

Before training, baseline original on the same environment convention and engine
binary used afterward. Record both policy latency and actual game strength.
Development: paired games against original plus fixed Stockfish 19 UCI_Elo
2400/2600/2800, 50ms/move, one thread, 32MB hash, pinned binary hash. No model search,
book or tablebases. UCI_Elo is a benchmark setting, not FIDE/Lichess Elo.

Freeze final iteration-20 candidate before confirmation. Play 200 fresh opening
pairs (400 games) versus original and 200 pairs versus anchor control. Repeat PPO
and control from original with a second training seed, retaining the recipe.
Report paired-opening bootstrap 95% CIs, clustering shared openings across seeds.
Do not pool repeated identical deterministic games as independent evidence.

Predeclared practical target: >=30 relative Elo point estimate versus original,
95% CI above zero, improvement over anchor control, consistent direction across
seeds, and no meaningful regression against the common SF ladder. Forty-eight
games cannot establish a small gain; even 400 may remain inconclusive.
No sequential extension until significance. A larger test is a new fixed budget.

Caps in evaluation remain unknown. Show best/worst-case bounds, or finish games
under a predeclared larger cap. If unknowns change the conclusion, inconclusive.
Also report decisive fraction and both colors; drawing more losing games can be
a real improvement, but arbitrary caps must not manufacture it.

- PPO beats original and control: replicate, then consider more self-play.
- Reward rises but external strength falls: opponent overfitting/forgetting; reject.
- Anchor agreement improves without game gains: no success claim.
- PPO unchanged with tiny policy movement: KL/step budget may constrain learning;
  change one setting in a follow-up, not multiple architecture pieces.
- Both arms regress: debug parity, encoding and anchor path before more RL.
- Stable PPO stalls: separately test MCTS-generated visit targets/outcomes and
  evaluate greedy policy. Search-assisted expert iteration is not PPO evidence.

Recurrence is a later controlled experiment only after this pipeline preserves
baseline strength and ideally produces a repeatable gain.

## Implementation deliverables and sources

Needed: original-model adapter for rl_selfplay; history-aware vectorized collector;
PPO/control runner; immutable anchor builder and exclusion manifests; checkpoint
and league serialization; focused tests; paired evaluation report. Reuse exp285
infrastructure after auditing, not exp290's temporary holdout mechanism.

- Model: https://huggingface.co/Maxlegrec/ChessBot
- PPO algorithm: https://arxiv.org/abs/1707.06347
- Search-assisted alternative: https://arxiv.org/abs/1712.01815

This recipe intentionally uses original-model distillation instead of changed
engine-label supervision. It is not a reproduction of published ChessBot training
or a claim of expected Elo gain.
