# Running ChessBot RL on this Mac

Implementation: experiments/chessbot_rl_mac.py and rl_selfplay/chessbot_ppo.py.
Configuration: configs/chessbot_rl_mac.json. Original model sources/weights live
in outputs/chessbot_rl_source; their SHA256s are recorded in each manifest.
Original 34,654,982 parameters, one pass, no recurrent layers, no remote writes.

## Current pilot

20 iterations, at least 32,768 learner decisions per iteration, temperature .8
for both collection players, paired colors, 16 concurrent environments, 800-ply
safety cap. Whole game batches finish before updates, so actual exposure can
exceed the 655,360-decision target slightly. This intentional simplification avoids
carrying partially completed games across policy updates and permits clean resume
at iteration boundaries. Capped games bootstrap and are never rewarded as draws.

Two PPO epochs maximum, minibatch 256, microbatch 8, LR 1e-6. Frozen-original KL
on rollout/anchor states, plus Q-WDL distillation. A separate anchor-only control
receives matched anchor updates. Audit 2,048 rollout actions after every update;
stop updates for the iteration at approximate behavior KL >.01. Reference drift
>.1 on two iterations or >5% game truncations stops the run for review.

This is an operational adaptation of docs/chessbot_rl_v1.md, not a claim that its
full scientific protocol has completed. Specific differences:

- Anchor bank has 10,483 complete-FEN positions from existing exp193 tactical
  development files, plus 384 permanently disjoint diagnostic positions. It is
  tactical-source biased and smaller than the proposed 100k/10k broad bank.
  Old labels are ignored. Original ChessBot supplies all anchor targets.
- Deduplication/exclusion uses first four FEN fields. No source-game IDs were
  available; game-level exclusion is not claimed. Original pretraining overlap
  remains unknown. Full clocks are required rather than imputed from packed data.
- Rollouts finish paired game batches; decision budgets are minimum thresholds.
- Development evaluation every five iterations: 32 distinct opening pairs against
  original and control, greedy. These are small screens, not confirmation or Elo
  evidence. Openings are generated from the original policy with a separate seed
  and exclude training starting states. No SF ladder or second seed launched yet.
- The non-Q value head's own parameters are frozen; its shared trunk can change.
- Checkpoints serialize models, optimizers, league, CPU/MPS and sampling RNGs.
  Interruptions during an iteration resume from its beginning, discarding that
  iteration's unsaved progress. No partial on-policy trajectories are resumed.

## Evidence before launch

Six tests passed: reward/color handling, GAE terminal and truncation returns,
legal masking, published policy/value parity, promotions/castling/EP mappings,
behavior ratio and save/load parity, and collector bootstrap perspective.

Actual MPS benchmark: 64 complete games, 3,767 learner decisions, 16.07 seconds,
234.5 learner decisions/sec, zero truncations. This measures collection only;
updates, reference/control forwards and KL audits add substantial runtime.

Separate two-iteration engineering smoke completed 1,003 decisions and 16 PPO
updates. All losses finite; final diagnostic reference KL approximately 9.8e-6.
Control/reference agreement and learner/reference agreement were both 100% on
the 384-position diagnostic at the final smoke checkpoint. This is not strength
evidence. Resume loaded successfully. Main run starts again from original weights.

## Inspect and control

Live log: outputs/chessbot_rl_mac.log
Structured events: outputs/chessbot_rl_mac/events.jsonl
Current stage: outputs/chessbot_rl_mac/status.json
Models/optimizers: outputs/chessbot_rl_mac/latest.pt
Actor/control milestones: outputs/chessbot_rl_mac/actor_005.pt etc.
Games and development matches: outputs/chessbot_rl_mac/games_*.json and eval_*.json
Process information: outputs/chessbot_rl_mac_process.json

Training is a detached process. A process-scoped caffeinate assertion prevents
idle system sleep while it runs; it does not guarantee training through lid closure.

Graceful stop: create outputs/chessbot_rl_mac/STOP; honored at the next iteration
boundary. Remove STOP before resuming. Resume with the same source/data/config:

    .venv/bin/python -u experiments/chessbot_rl_mac.py train --resume

Do not start a second process targeting the same output directory. None of these
artifacts replace the published model or the user's incumbent checkpoint. No
automatic promotion, uploads, or remote training changes occur.
