# Experiment 285: PPO + frozen reference + historical league

Hypothesis: on-policy PPO from the 99M search-free incumbent can improve game
outcomes if the policy stays near the supervised prior. A frozen reference KL,
supervised replay, and a short historical league are the anchors. Do not RL
from scratch. Do not promote on train loss, teacher agreement, or smoke output.

This is a continuation of the 99M trunk after exp283/284. Encoding, vocabulary,
and WDL class order stay as they are. The value head is White-absolute
(`P(White wins) - P(White loses)`). Collection and GAE convert that into the
actor's frame so Black is not trained on inverted advantages.

## What the loop does

Each iteration:

1. Collect paired-color games against either the frozen reference or a recent
   league snapshot. Same opening and recurrent depth for both colors.
2. Record only actor decisions. An actor transition is the actor move plus the
   opponent reply. Intermediate rewards are 0; the terminal is win/draw/loss
   from the actor. Truncation bootstraps the actor's next decision; a game may
   exceed `ply_cap` by one ply when the cap lands on the opponent.
3. PPO-clip the behavior log-prob, clip the value, add `KL(current || frozen
   reference)` over the same legal support, and mix a supervised soft-cache
   replay loss. Dropout is off for collection and updates.
4. Snapshot the actor into the league. Keep `league_size` snapshots. Optimizer
   and RNG state live only in `latest.pt`.

`STOP` is honored at an iteration boundary. Resume requires an identical run
config, including the replay SHA256. Output directories must be new or empty
unless `--resume`. Nothing is uploaded or promoted automatically.

## Commands

From the repository root:

```bash
export MOVE_VOCAB_VERSION=compact
# Pipeline check only. Random tiny GAB model. Not Elo evidence.
.venv/bin/python experiments/exp285_ppo_league.py --smoke --out outputs/exp285_smoke

.venv/bin/python experiments/exp285_ppo_league.py --config configs/exp285_ppo_99m.json

.venv/bin/python experiments/exp285_ppo_league.py --resume --out outputs/exp285_ppo_99m

# Greedy policy vs full-strength Stockfish at a pinned node budget.
# Unfinished games stay unknown; they are not scored as draws.
.venv/bin/python scripts/eval_stockfish_full.py \
  --ckpt outputs/exp285_ppo_99m/latest.pt \
  --protocol configs/stockfish_full_policy.json \
  --stockfish "${STOCKFISH_PATH:?pin the engine}" \
  --out outputs/exp285_sf_screen
```

`configs/exp285_ppo_99m.json` is a short local pilot: 20 iterations, 32 paired
games, depth 3, LR 2e-6, replay weight 0.25 against
`outputs/organized_chess_v1/soft_cache.pt`. Change `checkpoint` if the
incumbent is not `outputs/hf_100m_squares64/latest.pt`. Replay rows keep
`split==0` and `policy_mask`.

## Evaluation

The training loop does not play Stockfish and does not claim Elo. Screen with
`scripts/eval_stockfish_full.py` using the same pinned engine, protocol, and
openings for the incumbent and the PPO snapshot. Report unfinished / ply-cap
games separately. The paired bootstrap in the summary is descriptive, not a
sequential promotion test.

Promote only if fresh paired-opening games beat the same 99M checkpoint under
that protocol. A drop in replay CE or a self-play win rate against old league
snapshots is not enough.
