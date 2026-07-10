# Research notes — 2026-07-09 A40 session

## Pivot: exp185 deep-small (user redirect)

Stopped exp184 (~449M wide, ~2.4k/6k steps, holdout ~15% top-1) to prioritize a model that can be **fully trained + RL'd in 3–5h**.

- **Architecture:** 28L / 256d / 8H, ~**24.8M**, strengthened encoder 256d, SwiGLU, chess rel-bias, standard MHA (not full-dim), pre-norm residuals every layer, no grad ckpt.
- **Throughput:** ~990–1000 pos/s @ bs=1024 on A40 (~38GB).
- **Pretrain budget:** 12k steps × 1024 ≈ **12.3M positions** (~3.4h, ~34 tok/param).
- **Then:** exp183 SF expert-iter RL (full strength, shallow depth) for remaining time.
- Soft MultiPV cache reused from exp184 (`outputs/exp184_a40_wide_soft/soft_cache.pt`).

## exp184 (paused / superseded)

- 8L/1152d full-dim attn, 449M, NorMuon, soft+hard mix.
- Checkpoints kept under `outputs/exp184_a40_wide_soft/` (best around step 2k).
- Lesson: too big for a short rental if the goal is full train + RL.
