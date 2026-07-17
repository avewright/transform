# 8GB Elo Autoresearch Implementation Plan

> **For agentic workers:** implement task-by-task against `docs/superpowers/specs/2026-07-15-elo-autoresearch-8gb-design.md`

**Goal:** Build a resumable 8GB autoresearch loop that trains ~25M variants and promotes champions by policy Elo only.

**Architecture:** JSON search space → `train_trial` (soft/hard mix, NorMuon/AdamW) → `elo_eval_latest` subprocess → Pareto/champion journal under `outputs/autoresearch_8gb/`.

**Tech Stack:** PyTorch, existing `chess_transformer_factory`, `elo_eval_latest.py`, optional NorMuon.

## Global Constraints

- 8GB VRAM lab; no 400M local training
- Compact move vocab (`MOVE_VOCAB_VERSION=compact`)
- Elo is sole promotion metric (±100 noise band)
- Soft metrics diagnostic only

---

### Task 1: Search space + Pareto helpers

**Files:**
- Create: `scripts/autoresearch_8gb/search_space.json`
- Create: `scripts/autoresearch_8gb/pareto.py`
- Create: `scripts/autoresearch_8gb/__init__.py`

- [x] JSON dims for arch/train/data/opt
- [x] `update_pareto`, `should_promote_champion`

### Task 2: Train trial runner

**Files:**
- Create: `scripts/autoresearch_8gb/train_trial.py`

- [x] Build model from trial config (~25M class)
- [x] Soft+hard mix training with wall-clock/step budget
- [x] Return pos/s, VRAM, ckpt path

### Task 3: Elo trial + controller

**Files:**
- Create: `scripts/autoresearch_8gb/elo_trial.py`
- Create: `experiments/exp194_autoresearch_8gb.py`
- Create: `scripts/run_autoresearch_8gb.sh`
- Create: `docs/superpowers/specs/2026-07-15-elo-autoresearch-scaleup.md`

- [x] Wrap `elo_eval_latest.py`
- [x] Loop: sample → train → elo → journal → champion
- [x] `--smoke` path

### Spec coverage

| Spec item | Task |
|-----------|------|
| Outer loop + artifacts | 3 |
| Elo protocol | 3 |
| Search space all four areas | 1–2 |
| Pareto + noise | 1, 3 |
| Scale-up card | 3 |
| 8GB ~25M baseline | 2 |
