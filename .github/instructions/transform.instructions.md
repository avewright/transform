---
description: Always use these instructions
---
The goal is to get the highest elo possible. Unless specifically told otherwise by the user, design experiments to get the highest elo possible. Check the hardware available and utilize it. 
Fall into an autonmous loop of consistently experimenting and logging. Check the alphazero folder and stockfish_md for inspiration of very strong engines. You also have a md file describing the huggingface datasets you have access to. 
# Chess-Transformer Agent Instructions

This repository is for autonomous chess research around a chess-native transformer stack with strong data and evaluation discipline:

- supervised policy/value training on large Stockfish-labeled position sets
- controlled architecture ablations (encoder, policy head, search-aware heads)
- evaluation on move quality, gameplay, and search-time utility

The agent should behave like an autonomous research loop operator, not a generic coding assistant.

## First Steps For Every Session

1. Read [README.md](../../README.md) for the current workflow and CLI.
2. Read [codex_ideas.md](../../codex_ideas.md) if it exists. Use it as the working log for research feedback, follow-up ideas, and notable observations.
3. Inspect the active experiment path before proposing changes:
   - [experiments/](../../experiments/) (latest exp0xx scripts first)
   - [hf_data.py](../../hf_data.py)
   - [generate_data_cpu.py](../../generate_data_cpu.py)
   - [relabel_deep.py](../../relabel_deep.py)
   - [chess_model.py](../../chess_model.py)
4. Check the current workspace state before editing:
   - existing outputs under `outputs/` if present
   - any local notes, logs, or experiment artifacts already created
5. Prefer understanding the active training path before changing abstractions.
6. When you have feedback, hypotheses, architecture ideas, or experiment suggestions, record them in [codex_ideas.md](../../codex_ideas.md) so future sessions can build on them.

## Repository Purpose

The goal is to improve chess-playing behavior of the base model through repeated experimental loops.

LOOP FOREVER:
    - check current state and best results
    - hypothesize changes that would improve performance:
        - how can the encoder learn and provide as much information and context without loss?
        - what model architecture/pipeline would enhance the chess model the most?
        - what training structure would enhance the model the most?
        - to do this, feel free to reference research (arXiv)
    - create an experiment file
    - start with a cheap falsification run before large compute
    - with a standard metric, test your hypothesis.
    - regardless of the result, log the result with durable artifacts
    - Repeat

Default research direction should favor:
1. data and supervision quality improvements
2. controlled architecture ablations
3. search-time evaluation only after policy prior quality improves

## Autonomous Research Loop

When asked to do research or iterate autonomously, follow this loop:

1. Establish baseline behavior from the current code and config.
2. Choose one concrete hypothesis.
3. Implement the smallest code or config change needed to test it.
4. Run a bounded experiment.
5. Record the outcome in a durable artifact if the repo has or needs one.
6. Keep the change only if it improves the target metric or meaningfully improves the research harness.
7. Continue with the next hypothesis.

Do not make multiple experimental changes at once unless the variables are tightly coupled.

## Current Priority Order

Use this order unless the user explicitly redirects:

1. Data quality and labeling confidence:
   - larger diverse generated sets
   - deep relabel subsets
   - confidence/margin metadata
2. Controlled encoder/head ablations:
   - baseline vs fused 12-piece tokenization
   - relative geometry bias
   - policy readout improvements
3. Scale successful variants to larger data
4. Search complexity and internal search heads after policy gains are proven

## Success Metrics

Prefer metrics that reflect chess usefulness, not just code execution:

- legal move rate
- move accuracy on labeled positions
- top-3 accuracy
- mean SF rank of target move
- phase-sliced accuracy (opening/middlegame/endgame)
- gameplay score vs fixed Stockfish depths
- robustness across seeds
- evaluation throughput per unit compute

If a new metric is introduced, document exactly how it is computed and where it is reported.

## Experiment Rules

1. One hypothesis per experiment.
2. Keep quick tests cheap before running larger jobs.
3. Preserve reproducibility:
   - set seeds
   - log command lines
   - save configs with outputs
4. Prefer additive changes over destructive rewrites.
5. Do not claim improvement from anecdotal game samples alone.
6. If an experiment changes training behavior, also verify evaluation still works.
7. For architecture comparisons, keep data, steps, optimizer, and eval fixed.
8. Treat <2pp top-1 gains as provisional unless replicated.

## Experiment Contract

Every new experiment must record:

- one-sentence hypothesis
- primary metric
- fixed evaluation set or exact split procedure
- seed or seeds used
- train sample count and eval sample count
- runtime and device
- exact command used
- whether the result is preliminary (single seed) or replicated

Every experiment should define the primary metric before running.

Treat improvements smaller than 1 to 2 percentage points as provisional unless they are replicated across multiple seeds.

For fair-comparison experiments, keep model size, training steps, evaluation procedure, and optimizer schedule matched unless one of those differences is the variable being tested.

Prefer fixed validation sets stored on disk when possible. If not, document the exact split procedure and seed.

Outputs should include failure cases, not just aggregate metrics, when practical.

If static-label metrics improve, verify whether that improvement survives search-time or gameplay evaluation before prioritizing large follow-up work.

When estimating next steps, default to the cheapest experiment that can falsify the current hypothesis.

For architecture ablations, include:

- explicit baseline reference script/checkpoint
- parameter count delta
- throughput delta (samples/sec or epoch time)
- exact variable under test

## Coding Expectations

1. Keep changes aligned with the current repo structure.
2. Prefer explicit configs over hard-coded constants.
3. Add small, useful comments only where the logic is non-obvious.
4. Avoid introducing new dependencies unless clearly justified.
5. Do not silently break existing experiment scripts.
6. If changing data schema, provide backward-compatible reads or migration notes.
7. If changing CLI behavior in [train.py](../../train.py), update [README.md](../../README.md) too.

## Data Engineering Guidance

Data quality is a primary research surface. Favor:

- dedup by FEN and prevent split leakage
- preserving `source`, `phase`, and provenance metadata
- recording label confidence features (e.g., cp gap, shallow/deep agreement)
- keeping top-k move distributions when available for soft-target training
- writing manifests with generation settings, seed, depth, and counts

When compute is limited, prefer:

- broad shallow labeling for coverage
- smaller deep-labeled subset for high-confidence supervision

## Self-Play Guidance

When working on self-play:

- preserve alternating-color evaluation
- avoid changes that hide color bias
- treat plateau detection, noise annealing, and challenger selection as tunable research surfaces
- prefer checkpointable loops with enough metadata to compare generations later
- watch for degenerate behavior such as repetitive draws, trivial legal-move play, or collapse into random-looking action

## Evaluation Guidance

Evaluation should be comparable across runs.

Prefer:

- fixed validation subsets when comparing experiments
- explicit reporting of sample size
- separate reporting for legality and quality
- evaluation against prior checkpoints or fixed baselines when possible

If an experiment improves speed, state whether quality stayed flat, improved, or regressed.

## Expected Agent Behavior

The agent is expected to operate with initiative:

- identify the next sensible experiment
- implement it
- run it when feasible
- summarize result, risk, and next step

Do not stop at brainstorming if the task clearly asks for execution.
Do not invent external results that were not run locally.
If compute, model weights, or dependencies block a run, state the blocker precisely and leave the repo in a runnable state.

## Useful Entry Points

- [experiments/](../../experiments/): primary research scripts and latest results
- [hf_data.py](../../hf_data.py): HF dataset loader
- [generate_data_cpu.py](../../generate_data_cpu.py): bulk CPU Stockfish labeling
- [relabel_deep.py](../../relabel_deep.py): deeper relabel pass for quality subset
- [chess_model.py](../../chess_model.py): board encoder implementations
- [move_vocab.py](../../move_vocab.py): move space and legality masks
- [train.py](../../train.py): main CLI for `selfplay` and `randopt`
- [selfplay.py](../../selfplay.py): game loop, match logic, evolutionary update
- [evaluate.py](../../evaluate.py): model and ensemble evaluation
- [randopt.py](../../randopt.py): perturbation search and selection
- [model.py](../../model.py): model load/save and AttnRes wrapping
- [data.py](../../data.py): chess position formatting and data pipeline
- [constrained.py](../../constrained.py): constrained decoding utilities
- [attnres.py](../../attnres.py): Attention Residual implementation

## Definition Of A Good Change

A change is good if it does at least one of the following:

- improves a chess metric
- improves experimental rigor
- reduces compute waste
- makes training/evaluation more reproducible
- makes future research loops faster and safer

A change is not good if it only adds complexity without improving measurement, training stability, or chess behavior.
