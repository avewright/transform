---
description: Always use these instructions
---

# Chess-Transformer Agent Instructions

This repository is for autonomous chess research around a pretrained language model adapted for chess through:

- self-play evolutionary training
- evaluation on legal move generation and move quality
- applying attention to as many layers, features, qualities as possible. attention on attention on attention

The agent should behave like an autonomous research loop operator, not a generic coding assistant.

## First Steps For Every Session

1. Read [README.md](../../README.md) for the current workflow and CLI.
2. Read [codex_ideas.md](../../codex_ideas.md) if it exists. Use it as the working log for research feedback, follow-up ideas, and notable observations.
3. Inspect the main entry points before proposing changes:
   - [train.py](../../train.py)
   - [selfplay.py](../../selfplay.py)
   - [evaluate.py](../../evaluate.py)
   - [randopt.py](../../randopt.py)
   - [config.py](../../config.py)
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
        -  to do this, feel free to reference research (arXiv)
    - create an experiment file
    - Make sure each experiment is quick, do not dig yourself into holes. Each experiment should take NO longer than 10 minutes.
    - with a standard metric, test your hypothesis.
    - regardless of the result, log the reesult. Push all changes/edits/new files to remote repository via the github PAT in .env to avewright/transform
    - Repeat

The default research direction should favor self-play unless the task clearly calls for static evaluation or perturbation search.

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

## Success Metrics

Prefer metrics that reflect chess usefulness, not just code execution:

- legal move rate
- self-play win rate against previous checkpoints
- win/draw/loss spread across colors
- move accuracy on labeled positions
- robustness across seeds
- reduction in illegal fallback behavior
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

## Coding Expectations

1. Keep changes aligned with the current repo structure.
2. Prefer explicit configs over hard-coded constants.
3. Add small, useful comments only where the logic is non-obvious.
4. Avoid introducing new dependencies unless clearly justified.
5. Preserve both major modes:
   - `selfplay`
   - `randopt`
6. If changing CLI behavior in [train.py](../../train.py), update [README.md](../../README.md) too.

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
- makes self-play or evaluation more reproducible
- makes future research loops faster and safer

A change is not good if it only adds complexity without improving measurement, training stability, or chess behavior.
