# Research notes — 2026-07-23 M5 Pro (24GB unified)

**Frontier experiment card:** [docs/superpowers/specs/2026-07-23-frontier-sota-experiments.md](docs/superpowers/specs/2026-07-23-frontier-sota-experiments.md) (F0–F7 phases to SOTA).

**Executing now (2026-07-28):**
- Pulled HF soft: `avewright/chess-soft-multipv-lichess` (91M rows; 2M local) + `chess-soft-syzygy` (498k)
- Built `outputs/hf_soft_mix/soft_cache.pt` (1.5M, depth p50=29) via `scripts/build_hf_soft_mix.py`
- **Running:** `exp197` wider_shallower @ 16k steps / 4h on HF mix → `outputs/exp197_wider_shallower/`
- Next after Elo: same mix with `meta_shaw_elo`; then larger random subsample of the 91M


## Hardware / harness

- Machine: Apple **M5 Pro**, **24GB unified**, Stockfish 18 via Homebrew.
- Autoresearch path: `experiments/exp194_autoresearch_8gb.py` + `scripts/run_autoresearch_mac.sh`.
- Soft labels: local MultiPV harvest (`scripts/harvest_local_multipv.py`) — HF soft repos private / unauthenticated.
- Device: **MPS** (not CUDA). AMP disabled on MPS; CUDA path unchanged (bf16).

## Stockfish-style: linear meta attention + search-transformer value head (2026-08-06)

Two new factory features wired into the autoreearch search space (`search_space.json`):

1. **`use_linear_meta_attention`** — `LinearMetaFactoredAttention`: the 4-term
   content×position meta scores (cc/ss/cs/sc) computed with a positive
   Performer feature map, each term normalized by its own row-sum denominator
   (the classic `q·KV / q·Ks` linear-attention form). O(1) per position so the
   meta geometry can go deep/cheap. Only two KV moment matrices needed (content
   and position keys, both against content values); optional per-head absolute
   square bias for cheap geometry.Shaw deltas are skipped (inherently quadratic).
   Note: per-term normalization is required — sharing one denominator across the
   two query streams cancels the q-direction gradient algebraically at init.

2. **`use_search_value_head`** — `SearchValueHead`: a Stockfish-style neural
   one-ply eval. Coarse spatial prior → top-k latent children → refinement
   "search transformer" (candidates cross-attend to the parent board, then
   [candidates; children] jointly self-attend) → per-child value → soft-max
   backed-up best-child scalar → fused with the CLS prior into the 3-class WDL.
   Returns `value_logits` + `base_value` / `searched_value` for a backup aux loss
   (target = WDL scalar) added in `train_trial.py`.

New trials: `linear_meta_trunk` (O(N) linear meta trunk), `search_value_head`
(quadratic meta trunk + search value head), `linear_meta_search_value` (both),
and the stale `meta_latent_search` trial now maps to `use_search_value_head`
(its old `use_latent_search`/`latent_topk`/`latent_search_steps` were never
fields in the factory and were silently dropped).


## Bugfix (required for MPS train)

- `fused_ids_to_planes` returned a permute-strided tensor; MPS `Conv2d`/`BatchNorm2d` backward crashed with `view size is not compatible...`.
- Fix: `.contiguous()` on planes + strengthened-encoder conv→token permute.
- Fused encoder already worked; strengthened (baseline) did not until this fix.

## Active autoresearch wave (small ~25–35M) — resumed 2026-07-23

Goal: maximize verified Elo; explore arch / data / train for performance + efficiency + speed.

Done so far (short budget, both hit Elo floor):
- `baseline_deep_small` Elo≈1320 @ ~187 pos/s
- `wider_shallower` Elo≈1320 @ ~203 pos/s (faster Pareto)

New wave (`scripts/run_autoresearch_elo_wave.sh`, 45 min / 4k steps):
- Arch: `meta_shaw_elo`, `meta_latent_search`, `qk_norm_zero_init`, `gab`, `stack_ultimate`, `sota_stack_v1`, `fused_encoder`, `arch_deep_thin`, `infer_speed_recipe`
- Data: `soft_heavy_mix`, `elo_safe_mix`, `data_soft_peak`, `cf_soft_temp`, `soft_temp_t2`
- Train/opt: `polar_normuon`, `adamw_only`, `muon_hot`, `cf_swa`, `eff_mps_fat`, `compile_speed`
- Speed: `speed_micro_hot`, `micro_qk_swa_soft`

Artifacts: `outputs/autoresearch_8gb/{trials.jsonl,pareto.json,champion.json,run_mac.log}`

## Literature / optimization bets (for SOTA)

**Architecture (already in search space)**
- Consensus stack: pre-norm + SwiGLU + QK-Norm + zero-init residual outs (modded-nanogpt / 2025 transformer crystallization).
- Chess geometry: GAB and/or Shaw meta-attention vs handcrafted `rel_bias` (ablate, don’t stack blindly).
- Depth>width at ~25M (`baseline_deep_small` 28L/256d) historically preferred here; `wider_shallower` is the control.

**Data pipeline**
- Soft MultiPV distillation > hard-only; Elo failed when soft-loss was the crowning metric (~1850 wall).
- Mix knobs: `soft_alpha`, `deep_mix_frac`, Chessformer soft-temp aux (T≈2–4), puzzle/Syzygy deep packs when available.
- Local harvest now; once HF auth available, pull `chess-soft-syzygy` + MultiPV Lichess packs.

**Training**
- NorMuon (+ AdamW aux) default; Polar-NorMuon and hotter Muon LR in wave.
- SWA last 25%; label smoothing; longer warmup; large effective batch on 24GB unified (mps cap ~192).

**Inference (post-champion)**
- Policy legal-mask argmax first (Elo protocol).
- Then: temperature/value-blend (`exp117`), Syzygy probe, PUCT/MCTS only after policy floor rises.
- Prefer recipes without meta/GAB if pos/s ≫ and Elo within noise (`infer_speed_recipe`).

## Next after champion

1. Confirm-re-eval champion (±100 Elo noise).
2. Scale-up card → A40 / 400M (`docs/superpowers/specs/2026-07-15-elo-autoresearch-scaleup.md`).
3. Expert-iter RL (`exp183`) only after supervised Elo improves.
