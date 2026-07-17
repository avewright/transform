# Design: 8GB Elo Autoresearch Lab

**Date:** 2026-07-15  
**Status:** Approved (architecture + train + data + efficiency; Elo is the objective)  
**Constraint:** RTX-class **8GB VRAM** lab (no 400M local runs)  
**North star:** Maximize **policy Elo** vs Stockfish LimitStrength (`elo_eval_latest.py`)

## Problem

The ~400M meta-attention model peaked around **~1850 policy Elo**. Further soft-target training improved holdout metrics but **hurt Elo**. Overnight FT chains exist, but there is no systematic search that treats Elo as the objective and explores architecture, data, and train efficiency together under a tight compute budget.

## Goal

Run an **autoresearch loop on 8GB** that discovers recipes (model + data mix + optimizer/pipeline) which maximize Elo, with train throughput as a secondary Pareto axis so winners are cheap to scale on A40 later.

## Non-goals (v1)

- Training or fine-tuning the 400M locally  
- Replacing MCTS/search Elo as the primary metric (policy Elo first; MCTS is a later transfer test)  
- Building a full Bayesian optimizer UI — start with a simple controller + trial journal  

## Lab model class

Default baseline: **~25M** deep-small family (`DEFAULT_A40_DEEP_SMALL`-class: strengthened encoder, ~28L/256d, SwiGLU, compact vocab).

- Fits 8GB with headroom for batch / compile experiments  
- Mutate architecture **around this size class** (same order of magnitude params)  
- Do **not** grow toward 400M in the lab loop  

When A40 returns: re-run **champion recipes** at 400M / larger scale as a separate promote stage (out of v1 automation, documented as a checklist).

## Outer loop

```text
sample or queue a trial config
  → prepare data view (subset / mix weights) if needed
  → train under fixed budget (wall-clock OR steps; default wall-clock for rental/laptop fairness)
  → log pos/s, VRAM peak, soft holdout (diagnostic only)
  → run elo_eval_latest with fixed protocol
  → record (elo_estimate, elo_ci/noise, pos/s, config, ckpt)
  → update Pareto front + champion if Elo beats noise threshold
```

Controller features:

- Resumable (`trials.jsonl`, stage markers)  
- Skip completed trial ids  
- Hard fail on OOM → mark trial failed, shrink batch hint for next sample  
- Optional watchdog restart (reuse `watchdog_maxelo.sh` pattern)  

## Elo protocol (sole promotion metric)

Reuse `elo_eval_latest.py`:

| Knob | v1 default |
|------|------------|
| Move | Policy legal-mask argmax (temperature 0) |
| SF | `UCI_LimitStrength`, movetime **0.05s** |
| Games | 8 openings × 2 colors × **1** = **16** per level |
| Levels | Bracket around expected range (e.g. 1320–1900) |
| Stop | `--stop-after-bracket` |
| Noise | Treat **±100 Elo** as tie; new champion needs clear win **or** Elo within noise **and** ≥20% higher pos/s |

Soft top1 / soft_loss are **logged only** — never used to pick champions (that was the 1850 failure mode).

## Search space (all fair game)

### 1. Model architecture
- Meta-factored attention on/off; Shaw on/off  
- Handcrafted `rel_bias` on/off (ablate vs learned Shaw)  
- SwiGLU vs GELU FFN; STM normalize; fused vs strengthened encoder  
- Depth↔width swaps at ~constant param count  
- Policy head variants (spatial only in v1; from–to optional later)  
- Value head weight / WDL aux strength  

### 2. Training pipeline
- Steps vs wall-clock budgets  
- Soft/hard schedule; warmup; cosine floor  
- Checkpoint selection: **must be Elo-gated**, not best soft_loss  
- `torch.compile`, grad checkpoint, TF32/bf16  
- Early stop if mid-train Elo probe drops vs init  

### 3. Dataset engineering
- Soft MultiPV caches (shallow / deep / phase-balanced / puzzle / edge)  
- Mix weights: `soft_frac`, `soft_alpha`, `deep_mix_frac`, phase quotas  
- H-flip aug on/off  
- Hard ballast depth filters  
- **Held-out soft caches for diagnostics only** — Elo remains the judge  
- Forbidden: crowning a run because soft_loss improved while Elo fell  

### 4. Training efficiency (optimizer, etc.)
- NorMuon (+ AdamW aux) vs AdamW vs Polar-NorMuon if present  
- Muon/Adam LR pairs, weight decay, grad clip  
- Batch size × accum (effective batch) under 8GB  
- Throughput logging: pos/s, step time, VRAM  

## Trial budget (laptop-sane defaults)

| Item | Default |
|------|---------|
| Train budget | **45 min** wall-clock **or** 3k steps (whichever first) |
| Data | Fixed frozen soft+hard slices (same paths for all trials unless the trial *is* a data-mix experiment) |
| Elo | Full fast bracket after train (~20–40 min) |
| Parallelism | 1 GPU trial at a time on 8GB |

Optional **mid-train Elo probe** (single SF level, 8 games) every N steps for early kill — off by default in v1 to keep the harness simple.

## Artifacts

```text
outputs/autoresearch_8gb/
  trials.jsonl          # one JSON object per trial
  pareto.json           # non-dominated (elo, pos/s)
  champion.json         # best Elo config + ckpt path
  champion.pt           # symlink or copy of best weights
  trials/<id>/
    config.json
    train.log
    elo_*.json
    best.pt / latest.pt
```

## Success criteria

1. Harness runs unattended overnight without crashing the machine  
2. ≥1 champion that **beats the 25M baseline Elo** under the fixed protocol  
3. Pareto list includes at least one **faster** recipe within Elo noise of champ  
4. Written “scale-up card”: how to replay champion settings on A40 / 400M when VRAM returns  

## Implementation sketch (for the plan)

1. `scripts/autoresearch_8gb/` or `experiments/exp194_autoresearch_8gb.py` — controller  
2. Thin wrappers: `train_trial(config) → metrics`, `elo_trial(ckpt) → estimate`  
3. Seed baseline trial (deep-small + current best soft mix) to set champion floor  
4. Queued + random search over a YAML/JSON search space  
5. Docs: this spec + short README in outputs  

## Risks

| Risk | Mitigation |
|------|------------|
| 16-game Elo noise | Noise band; optional confirm-re-eval for champ |
| Soft metrics misleading | Elo-only promotion |
| OOM / slow compile | Fail trial; prefer non-compile until stable |
| Data path missing on laptop | Require soft cache path; fail fast with clear error |
| Overfitting search to SF LimitStrength quirks | Later: holdout SF Elo levels + human/puzzle pack (v2) |
