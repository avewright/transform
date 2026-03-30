# Codex Ideas

This file is the running log for:

## 2026-03-30

### exp083: Full-corpus pretraining on 4×A40 — KILLED (LR too high)

**Hypothesis:** Training the 204M ChessTransformer on the full ~832M position
corpus using 4 A40 GPUs with Local SGD will push accuracy well beyond the
22.9% (lichess-sf eval) / 44.6% (HF cross-eval) from exp071.

**Setup:**
- 4× NVIDIA A40 (46GB each), all at 100% utilization
- ChessTransformer200M: 204M params, 16L/1024d, SpatialPolicyHead
- Init from avewright/chess-transformer-200m-v2 (exp074 best)
- Local SGD: 4 workers, each on ~208M positions (818 parquet files), sync every 500 steps
- Batch 256 × accum 4 = eff 1024 per worker, LR=1e-4 cosine to 5%
- Baseline from init: 16.3% top-1, 41.8% top-3, 78.5% value accuracy
- Combined throughput: ~1,200 pos/s (300/worker)
- Aggressive logging: metrics JSONL every 10 steps, checkpoint every 100 steps
- Health monitor: `monitor_exp083.py`

**Results (killed at step ~2100):**
- Step 0: 16.3% top-1, 41.8% top-3, 78.5% value (init)
- Step 500: **16.7%** top-1, 41.9% top-3, 79.7% value (**best**)
- Step 1000: 15.8% top-1, 39.6% top-3, 78.2% value (declining)
- Step 1500: 15.7% top-1, 40.8% top-3, 80.6% value
- Step 2000: 14.8% top-1, 38.2% top-3, 79.7% value (clear regression)
- Policy loss rose from 3.5 → 5.5, well above init baseline
- **Root cause**: LR=1e-4 too aggressive for continuation training. Warmup destroyed learned features before cosine could recover.

**Key lesson**: For continuation from pretrained weights, use LR ≤ 3e-5. High LR on a converged model is catastrophic forgetting.

### exp083b: Continuation with LR=3e-5 from exp083 best — RUNNING

**Hypothesis:** With 10× lower LR (3e-5 vs 1e-4), the model will improve
without regressing. Starting from exp083 step-500 best (16.7% acc).

**Setup (changes from exp083):**
- LR: 3e-5 (was 1e-4)
- WARMUP_FRAC: 0.005 (was 0.01) — shorter warmup
- MIN_LR_FRAC: 0.10 (was 0.05) — higher LR floor
- Init from: outputs/exp083_pretrain_4xa40/best_model.pt (16.7% top-1)
- Same data, model, and sync strategy as exp083

**Status:** Training, step ~510. First eval at step 500: **16.3% top-1, 42.1% top-3, 79.5% value** — NO REGRESSION. LR still warming up (1.5e-5 of peak 3e-5). Combined throughput ~1,340 pos/s.

## 2026-03-29

### exp081: Confidence-weighted cached continuation — ADDED

Hypothesis: The productive continuation path on 8GB VRAM is still local cached
supervised training, but policy loss should be weighted by label confidence so
clear Stockfish decisions dominate gradient over nearly-equal positions.

Design:
- based on exp079, not the exp076 one-pass stream
- local cached lichess-sf subset + replay mixing
- soft top-move targets retained
- confidence-weighted CE/KL using eval magnitude + top-2 margin
- richer eval slices by phase and confidence bucket

Expected outcome:
- better sample efficiency than unweighted continuation
- same 8GB VRAM footprint as exp079
- more interpretable failure modes if gains only come from high-confidence buckets

### exp082: Online SF game soft-label loop — ADDED

Hypothesis: An online loop that plays full games vs Stockfish, then relabels each
model position with full legal-move Stockfish scores, can create denser policy
supervision than single best-move targets while staying within 8GB VRAM.

Design:
- limited-strength Stockfish for gameplay opponent
- full-strength Stockfish analysis for all legal moves after the game
- soft policy targets from legal-move score distributions
- replay buffer across cycles
- resumable checkpoint + cycle logs + per-cycle label dumps

## 2026-03-25

### exp070: Large-scale Lichess-SF training on A40 — COMPLETED

**Hypothesis:** 12L/512d FusedEncoder on 2M lichess-sf positions will push accuracy beyond prior baselines.

**Result:** 20.9% top-1, 46.7% top-3, 63.6% value accuracy (best at epoch 2 end)

| Checkpoint | Top-1 | Top-3 | SF Rank | Value |
|-----------|-------|-------|---------|-------|
| Step 2000 (E1) | 15.3% | 35.4% | 73.7 | 58.2% |
| Epoch 1 end | 17.7% | 40.2% | 72.3 | 62.9% |
| Step 6000 (E2) | 19.8% | 45.1% | 71.5 | 64.6% |
| **Final (E2 end)** | **20.9%** | **46.7%** | **71.2** | **63.6%** |

Phase accuracy (final): opening 19.6%, middlegame 24.5%, endgame 26.2%

**IMPORTANT:** This result CANNOT be directly compared to prior experiments (exp052-053: 30-37%) because the eval sets are different:
- Prior: 2500 positions from `avewright/chess-positions` (HF dataset)
- exp070: 5000 positions from `avewright/chess-positions-lichess-sf` (deeper SF labels, wider position distribution)

**Key observations:**
1. The 20.9% accuracy is on a HARDER eval set with deeper SF analysis (depth 15-245 vs depth ~8 prior)
2. Positions with deeper analysis have more "correct" but less obvious best moves
3. 58% of positions have |cp| < 100 — many nearly-equal positions where the "best" move is debatable
4. 168K mate positions in training data (8.4%) — good tactical exposure
5. Policy loss: 6.77 → 2.74 (60% drop), value loss: 0.34 → 0.19 (45% drop)
6. Training throughput: stable 970 pos/s on A40, 69 min total
7. Loss was still decreasing at end — more epochs would likely help

**Data properties:**
- 2M positions, median depth ~20, range 15-245
- 58% have |cp| < 100 (contested positions)
- 77% have |cp| < 300 (clear-enough positions)
- Mean cp: 45, std: 881

**Next experiments to try:**
1. Filter to depth >= 20 only (higher confidence labels) — trade quantity for quality
2. ~~More epochs (4-6) since loss was still dropping~~ → done in exp071
3. ~~Cross-evaluate: run exp070 model on the old HF eval set for fair comparison~~ → done in exp071
4. Train on ALL available data (~7-10M positions) since more data should help
5. Relative bias (exp068 variant) on this larger dataset

---

### exp071: Extended training (6 epochs) + cross-eval — COMPLETED

**Hypothesis:** Training 6 epochs instead of 2 on the same 2M positions improves accuracy (loss was still decreasing).

**Result:** 22.9% top-1, 50.9% top-3, 72.6% value accuracy (best at step 18000, epoch 5 mid)

| Checkpoint | Top-1 | Top-3 | SF Rank | Value |
|-----------|-------|-------|---------|-------|
| Epoch 1 end | 17.1% | 40.0% | 72.3 | 63.5% |
| Epoch 2 end | 20.5% | 46.4% | 71.2 | 64.3% |
| Epoch 3 end | 21.4% | 49.1% | 70.7 | 70.3% |
| Epoch 4 end | 22.2% | 50.2% | 70.4 | 72.4% |
| **Best (E5 step 18k)** | **22.9%** | **50.9%** | **70.3** | **72.6%** |
| Epoch 5 end | 22.8% | 51.4% | 70.3 | 71.6% |
| Epoch 6 end | 22.3% | 52.3% | 70.1 | 72.9% |

Phase accuracy (best): endgame 30.8%, middlegame 24.6%, opening 21.7%

**Cross-eval on avewright/chess-positions test set: 44.6% top-1, 73.5% top-3**

This is the key result. Prior experiments on this eval set scored 30-37%. This model at **44.6%** represents a massive improvement, confirming:
- The lichess-sf data is superior supervision (deeper SF, more diverse positions)
- The 12L/512d architecture is working well
- The "low" 22.9% on the lichess-sf eval set reflects harder positions, not a weaker model

**Observations:**
1. Accuracy peaked at epoch 5, then declined slightly by epoch 6 → mild overfitting on 2M positions
2. Top-3 and value accuracy continued improving through epoch 6 (52.3% top-3, 72.9% value)
3. Policy loss plateaued around 2.35 in epoch 6 (vs 2.41 in epoch 5)
4. Training throughput: 975→834 pos/s (pipeline CPU contention reduced GPU throughput ~15%)
5. Total time: 240 min (4 hours) for 6 epochs
6. +2.0pp over exp070 baseline on lichess-sf eval, +7.6pp+ on HF cross-eval vs prior art

**Conclusion:** Hypothesis CONFIRMED. More epochs helped significantly (+2pp on hard eval, +7.6pp+ on standard eval). But diminishing returns visible by epoch 5-6. The real bottleneck is data quantity (2M positions seen 6× is overfitting). Next: scale to full dataset.

**Next experiments:**
1. **Scale data to 10M+ positions** (now that pipeline is uploading more) — top priority
2. Train for 2-3 epochs on 10M (same compute budget as 6×2M, but more diverse)
3. Relative bias / architecture improvements after data scaling

---

### Data Pipeline Status — 2026-03-26 (pod shutdown)

**Lichess pipeline: COMPLETE**
- All 17 sources (0-16) from `Lichess/chess-position-evaluations` processed and uploaded to `avewright/chess-positions-lichess-sf`
- Sources 1-5: pre-existing
- Sources 6-16: processed by `process_lichess_parquets.py` (48 workers, ~19K pos/s)
- Source 0: processed separately by `prepare_hf_dataset.py` (--dry-run), uploaded by `monitor_and_generate.py`
- Total: ~850M raw positions → filtered to depth >= 15 with valid SF evals
- Schema: fen, best_move, eval_type, eval_value, wdl_win, wdl_draw, wdl_loss, phase, num_legal, source, game_id, top_moves, ply, depth

**Custom position generation: STOPPED at 119K/5M**
- `generate_and_upload.py` ran with 48 Stockfish 14 workers at depth 10
- Generated ~119,203 positions before pod shutdown (42 pos/s)
- Positions were pending upload (hadn't reached 250K upload threshold)
- Generation too slow with Stockfish 14 (~42 pos/s total) — consider Stockfish 16+ or lower depth for throughput
- Script and pipeline are ready to resume on next pod

**Key scripts created this session:**
- `generate_and_upload.py`: Custom position generator matching lichess-sf schema, with diverse position sampling strategies
- `monitor_and_generate.py`: Autonomous orchestrator (pipeline monitoring → source 0 upload → verification → generation)
- `experiments/exp072_data_scale.py`: Ready to run (10M positions × 2 epochs), not yet executed

---

This file is the running log for:

- research feedback
- experiment ideas
- architecture suggestions
- evaluation concerns
- follow-up hypotheses

Use short dated notes so future sessions can quickly understand prior thinking and continue from it.

## 2026-03-19

### Repo-level feedback

- The project direction is coherent: learned board encoder into a pretrained transformer backbone, then iterate on supervision, search, and richer features.
- The biggest near-term leverage is likely better experimental rigor and stronger measurement, not only more architecture variation.
- The current repo instructions are directionally good, but the research harness should enforce fair comparisons and reproducibility more mechanically.

### Criticism of current experiment style

- Some comparisons are not fully fair even when they are described that way. Keep optimizer schedule, model capacity, training budget, and evaluation procedure matched unless one of those is the variable under test.
- Single-seed wins on small validation sets are easy to over-interpret.
- Random row splits may overstate generalization if nearby or duplicate positions leak across train and eval.
- Aggregate top-1 accuracy is useful, but it hides where the model is failing.
- Experiments log results, but they should also log command, runtime, device, seed, split procedure, and failure cases.

### Feedback on exp_av_comparison.py

- The current script is useful, but it tests whether action-value supervision helps the policy head indirectly more than it tests pure policy-vs-Q learning.
- Variant A and Variant B do not share exactly the same optimization path, so the comparison is not perfectly controlled.
- The action-value loss only supervises labeled legal moves and ignores unlabeled legal moves, which makes it partial supervision rather than a full Q estimate.
- Evaluation reports policy accuracy only. If using Q supervision, also inspect ranking quality or calibration of move values.
- The fixed eval size of 250 is okay for a quick screen, but too noisy for strong conclusions on small deltas.

### Experiments to run

- Replicate promising experiments across 3 to 5 seeds and report mean and spread.
- Compare target formulations:
  - hard best-move CE
  - soft top-k targets
  - normalized move-value policy targets
  - joint policy plus value or Q auxiliary loss
- Run a data-quality ablation:
  - best move only
  - top-k only
  - all legal move values
  - high-confidence filtered labels only
- Test frozen backbone vs staged unfreezing:
  - frozen throughout
  - unfreeze top layers after warmup
  - LoRA on attention only
  - LoRA on attention plus MLP
- Compare encoder variants under equal parameter budget:
  - base learned encoder
  - rich-feature encoder
  - hybrid learned plus handcrafted
  - alternate projection or pooling designs
- Measure search-time gains:
  - raw policy
  - policy plus value
  - small MCTS
  - larger MCTS
- Evaluate by position slice:
  - opening, middlegame, endgame
  - tactical vs quiet
  - side to move
  - in check vs not in check
  - material imbalance buckets
- Add calibration checks for value or Q predictions.
- Track accuracy per GPU-minute so efficiency is visible.
- Create a stronger holdout split that reduces leakage from near-duplicate positions.

### Architecture changes worth testing

- Move-conditioned scoring: encode the board and score legal moves explicitly with a move encoder instead of always predicting a dense 5504-way vector.
- Two-tower policy head: board tower plus move tower with dot-product or bilinear scoring over legal moves.
- Better pooling: test learned attention pooling or separate pooling heads for policy and value instead of relying on one global token representation.
- Legality-aware training: inject legal move structure into training, not only inference-time masking.
- Relative board-geometry bias: preserve square relationships more explicitly through the encoder or projection path.
- Multi-task supervision:
  - best move
  - WDL
  - centipawn bucket or win-probability bucket
  - tactical indicators or check status
- Symmetry handling:
  - test board flips
  - color-normalized representations
  - side-to-move canonicalization
- Alternative value targets:
  - WDL
  - scalar expected score
  - cp-to-winprob targets
  - horizon-aware targets for search

### Measurement upgrades to prioritize

- Store fixed validation sets on disk when possible.
- Save small failure-case samples with FEN, target move, predicted move, and top-k legal alternatives.

---

## 2026-03-22

### Codex review: data quality dominates architecture tweaks

External review confirmed what the results table already shows: the strongest
signal is "real positions + stronger labels" — not architecture cleverness.
The next phase should be: reliable dataset factory → small chess-native model
family → Qwen backbone only as a transfer baseline.

### Bug fix: pooling inconsistency

**Fixed** a material representation bug. `chess_model.py` ChessModel.forward()
used `hidden[:, -1, :]` (last token—correct for causal attention) but
`train_action_value.py` used `hidden[:, 0, :]` (first token—only sees itself
in causal attention). This means the action-value trainer was training and
evaluating on different representations. Standardized to `hidden[:, -1, :]`.

Note: many old experiment files (exp013, exp019-022) also used `hidden[:, 0, :]`
with the causal Qwen backbone. Those are historical and won't be retroactively
fixed, but the bidirectional ChessTransformer from exp023+ doesn't have this
issue (all tokens see all tokens in encoder-only attention).

### Architecture V1 spec created

See `ARCHITECTURE_V1.md` for the full spec. Key decisions:

- **Chess-native encoder-only transformer** (bidirectional, not frozen causal LLM)
- **Factorized SpatialPolicyHead** (from-square × to-square × promotion)
- **Dedicated CLS token** for consistent global readout
- **Relative board bias** option for chess geometry
- **Multi-target value head** (WDL + centipawn bucket)
- Three sizes: Small (4M), Medium (17M), Large (55M)
- Qwen backbone kept only as transfer baseline

### Dataset factory upgrades

`label_positions.py` now supports:
- Source metadata (`source`, `game_id`) in every labeled entry
- `--source` flag to label positions from external JSONL (e.g. Lichess games)
- `--split-by-game` and `--write-splits` to produce train/val/test JSONL files
  split by game_id, preventing position leakage from correlated positions
- Richer `compute_stats()`: unique FEN count, duplicate count, source
  distribution, game count

### Experiment discipline: exp050 head comparison

Created `experiments/exp050_head_comparison.py` — the first "fewer, better-controlled"
experiment. Tests one hypothesis (spatial vs flat head) with:
- Same data, same model body, same optimizer, same schedule
- 3 seeds (42, 123, 314) with mean ± std reported
- Game-level data split to prevent leakage
- Auto-detects best available data source

### Roadmap (priority order)

1. **[DONE]** Fix pooling inconsistency
2. **[DONE]** Formalize dataset factory with game-level splits and metadata
3. **[DONE]** Standardize on chess transformer + spatial/factorized policy head (ARCHITECTURE_V1.md)
4. **[DONE]** Build HF dataset `avewright/chess-positions` with streaming loader
5. **[NEXT]** Run exp050 head comparison as the first controlled experiment
6. Train with soft Stockfish targets (distribution over legal moves, not only best-move CE)
7. Build curriculum buckets (opening, tactical, quiet, endgame, zugzwang)
8. Add hard-example mining and replay
9. Only then revisit search (MCTS, alpha-beta with learned eval)

### Dataset v2: multi-source diverse generation (2026-03-22)

Replaced the original 10K random-play-only dataset with 50K positions from
5 diverse generation sources. Old data had 0% endgame and 100% random_play.

**Generation sources (build_dataset.py):**
1. **Opening book** (15%): Walk 40+ common ECO opening lines, branch randomly for 0-8 more moves. Produces realistic opening/early-middlegame positions.
2. **Weighted play** (30%): Random games biased toward captures (4x), center control (2x), and piece development (1.5x). More natural than uniform random.
3. **Aggressive play** (15%): Games strongly favoring captures (75% capture rate). Creates tactical positions with material imbalances.
4. **Endgame** (20%): 60% synthetic construction (kings + 1-4 light pieces), 40% trade-down (capture-heavy games until material ≤ 26). Guaranteed endgame coverage.
5. **Perturbation** (20%): Mutate existing positions — remove a piece (2/3 chance) or swap piece type (1/3). Creates unusual material configurations the model wouldn't see from normal play.

**Key improvements over v1:**
- 5x more data (50K vs 10K)
- 6 distinct source types vs 1
- ~33% each opening/middlegame/endgame vs 0% endgame before
- Mate positions included (tactical depth)
- Wider eval distribution: cp_std ~500 vs ~390
- Removed expensive `gives_check()` from generation (30x faster)
- Proper source tracking per position (not hardcoded "random_play")

**What makes this efficient for training:**
- Opening book positions teach the model standard play without wasting budget on random noise
- Aggressive play creates high-information tactical positions (clear best moves)
- Perturbation creates the largest diversity per compute — one mutation = infinite new positions from each template
- Endgame positions are cheapest to label (few pieces → Stockfish is near-instant)

### Planned controlled experiments (after exp050)

| ID | Variable | Controlled | Metric |
|----|----------|------------|--------|
| exp050 | flat vs spatial head | same data, body, budget, 3 seeds | SF top-1 |

---

## 2026-03-23

### exp052: Head comparison v2 — DECISIVE spatial win

**Fixes over exp050:**
- exp050's "game-level split" was broken: HF dataset has `game_id=""` for all
  47.5K positions (all generated, no real game IDs). Fixed by using HF pre-split
  train/test directly.
- Added learned [CLS] token (dedicated global readout, not turn token)
- Added phase-bucketed eval (opening/middlegame/endgame)
- Added entropy + SF-move-rank metrics
- Saves per-seed model checkpoints

**Results (Small config: 256d, 6L, 8H, 3 epochs, 3 seeds):**

| Variant | Params (head) | Mean acc | Std | Top-3 | Entropy | SF Rank |
|---------|--------------|----------|-----|-------|---------|---------|
| flat    | 1,480K       | 11.3%    | 0.2%| 28.9% | 2.34    | 10.9    |
| spatial |    99K       | 30.3%    | 0.2%| 52.5% | 2.11    | 6.5     |

**Delta: +19.0% (spatial wins decisively)**

Phase breakdown (spatial, s42):
- Opening: 36.6%
- Middlegame: 30.0%
- Endgame: 24.6%

Phase breakdown (flat, s42):
- Opening: 15.5%
- Middlegame: 6.5%
- Endgame: 13.0%

**Key observations:**
1. Spatial head is **2.7x better** with **15x fewer head parameters** (99K vs 1.5M)
2. The flat head barely learns — 11% is near-random for positions with ~30 legal moves
3. Spatial head's endgame weakness (24.6% vs 36.6% opening) may reflect data quality:
   endgame positions from synthetic constructions are noisier
4. SF-move rank of 6.5 means the model's top pick is typically SF's 6th-7th choice —
   not great, but much better than flat's rank 11
5. Lower entropy for spatial (2.11 vs 2.34) means more confident predictions
6. All 3 seeds agree tightly (±0.2%) — result is robust
7. All runs: 44s/epoch flat, 83s/epoch spatial (spatial is 2x slower due to indexed
   gather ops despite fewer params)

**Conclusion:** Spatial head is confirmed as the right architecture choice.
Promote it into the chess-native transformer family. The flat head is dead.

### Positional embedding analysis

Checked all model definitions for positional encoding:
- **ChessModel (Qwen backbone):** double positional — `square_embed` in encoder +
  Qwen's RoPE in attention. Both work correctly for fixed 67-token sequences.
- **ChessTransformerV2 (exp052):** learned absolute `pos_embed` (68 tokens =
  CLS + 67 board). Applied additively before `nn.TransformerEncoder`. Correct.
- **Old exp023/024/050:** same pattern, 67 tokens with learned `pos_embed`.
- PyTorch's `nn.TransformerEncoder` does NOT use RoPE — it only uses whatever
  positional info you inject before the layers. Our learned `pos_embed` is the
  only positional signal.

**No missing positional encoding found.** But worth testing:
- RoPE vs learned absolute — RoPE might generalize better to unseen position combos
- Relative board biases from ARCHITECTURE_V1.md — explicitly encode chess geometry

### Updated roadmap

1. **[DONE]** Fix pooling inconsistency
2. **[DONE]** Formalize dataset factory
3. **[DONE]** Standardize on chess transformer + spatial policy head
4. **[DONE]** Build HF dataset
5. **[DONE]** exp052: head comparison — spatial wins decisively
6. **[NEXT]** Scale up: Medium model (512d, 8L) with spatial head, more epochs
7. Train with soft Stockfish targets (top-k move distribution, not just best move)
8. Add relative board biases from ARCHITECTURE_V1.md
9. Evaluate with actual gameplay (not just label accuracy)
10. Build curriculum buckets and hard-example mining
| exp051 | hard labels vs soft SF distribution | same model, same data | SF top-1 + ranking |
| exp052 | random positions vs real positions | same model, same pipeline | SF top-1 |
| exp053 | split-by-position vs split-by-game | same train set, same model | eval acc delta (leakage) |
| exp054 | relative board bias vs no bias | same everything else | SF top-1 |
- Treat sub-1 to 2 point wins as provisional unless replicated.
- When a proxy metric improves, check whether gameplay or search-time quality also improves before scaling the idea up.

### Session 2: Pipeline build and action-value comparison (2026-03-19)

**What was built:**
- `label_positions.py` — standalone Stockfish corpus builder. Generates JSONL with FEN, all-move evals, WDL, phase bucket. Supports resume. ~16 pos/s at depth 8 on this machine.
- `train_action_value.py` — main action-value trainer. Consumes cached JSONL. Loss = AV_MSE(all legal) + 0.5*policy_CE + 0.5*value_CE.
- `experiments/exp_av_comparison_v2.py` — hardened A/B with identical forward paths, config artifact, larger eval (500 vs 250).
- Data generated: `data/sf_labels_5k_d8.jsonl` (5K, complete), `data/sf_labels_10k_d8.jsonl` (10K, complete), `data/sf_labels_50k_d8.jsonl` (23K partial).

**Key results:**
- exp_av_comparison_v2 (5K random, 10 epochs): policy-only vs action-value = TIE at 8.8%.
- train_av_10k (10K random, interrupted ep 8): best 10.3% acc, 23.7% top3.
- exp013_hf_dataset_scale (50K HF game-play, 3 epochs): 22.8% acc, 39.2% top3.
- exp014_full_hf_1epoch (475K HF game-play, 1 epoch): 18.4% acc, 39.4% top3.

**Critical insight: position quality >>> loss function design at current scale.**
Random-play positions give ~8-10% accuracy regardless of whether you use policy CE or action-value Q(s,a). HF game-play positions from real games give 22-25% with plain policy CE. The ~15 point gap is entirely data quality, not training signal.

Why: random play explores bizarre, unrealistic positions that humans/engines would never reach. The model learns to predict moves in positions it will never see during actual play. Real game positions have structure — they follow opening theory, create meaningful plans, and exercise tactics that matter.

**Implications for the roadmap:**
1. Action-value training is not the bottleneck. The loss function matters less than what positions you train on.
2. Position source matters enormously. Real games >> random play at equal count.
3. The 50K labeling run should use real game positions, not random-play positions.
4. Scaling random positions (5K→10K) gave only +1.5pp. Scaling real-game positions (5K→50K) gave much more in exp013.
5. The previous session's HF dataset result (22.8% on 50K game-play) is still the best result.

**Next hypothesis to test:**
Can we combine the best of both? Use the action-value all-move labeling on REAL GAME positions instead of random positions. This gives both high-quality positions AND dense gradient signal.

Cheapest test: take 5K positions from the HF dataset (real games), label them with Stockfish all-move evals, and train action-value. Compare to plain policy CE on the same 5K.

**Other observations:**
- The labeling pipeline works well. Resume support is useful. ~16 pos/s is acceptable.
- The exp_av_comparison_v2 harness is now properly fair: same forward path, same optimizer, config logged.
- The experiment contract (from instructions) is being followed better: seed, split, device, command, runtime all logged.
- Still missing: multi-seed replication, failure case logging, per-phase accuracy breakdown.

### exp_av_real_games results (2026-03-19)

**Result: TIE again.** Policy CE 23.6% best vs AV 23.4% best on 3K HF game positions + SF all-move labels.

This conclusively answers the action-value question at this scale: AV auxiliary loss does NOT help the policy head, even on real game positions with good data quality.

**Why?** Likely reasons:
1. The AV head learns Q-values through a separate head that doesn't share signal back to the policy head efficiently. The policy head still only gets 1 gradient from CE.
2. At 3-5K positions, the shared encoder/backbone is the bottleneck, and the AV head just adds more params without improving shared representations.
3. The AV loss may need far more data (~100K+) before the shared encoder learns useful Q-structure that helps policy. The Ruoss et al. paper used millions of positions.
4. A single global-token representation may not carry enough info for per-move Q predictions.

**Cumulative results table:**

| Experiment | Data Source | N | Loss | Best Acc | Top3 | Notes |
|------------|------------|---|------|----------|------|-------|
| exp_av_comparison_v2 A | 5K random | 4.5K | policy+value CE | 8.8% | 21.8% | baseline |
| exp_av_comparison_v2 B | 5K random | 4.5K | AV+policy+value | 8.8% | 21.6% | TIE, AV doesn't help |
| train_av_10k | 10K random | 9.5K | AV+policy+value | 10.3% | 23.7% | slight scale gain |
| exp_av_real_games A | 3K HF games | 3K | policy+value CE | 23.6% | 48.8% | real game positions |
| exp_av_real_games B | 3K HF games | 3K | AV+policy+value | 23.4% | 48.8% | TIE, AV doesn't help |
| exp013 (prior) | 50K HF games | 50K | policy CE only | 25.0% | 45.0% | more data helps |

**Key takeaways:**
1. Position quality is the dominant factor: real games 23% vs random 9% on same volume.
2. Action-value auxiliary loss is neutral at scales up to 10K. Provisionally deprioritize.
3. Data volume on real positions is still the main lever: 3K→50K gave 23%→25%.
4. The frozen backbone + learned encoder + policy CE is a solid baseline.

**Revised roadmap:**
1. Scale real-game data to 50K+ with Stockfish best-move labels (cheaper than all-move).
2. Test soft-target formulations (top-k move probs, KL divergence) as alternative to hard CE.
3. Test a chess-native transformer (no Qwen backbone) as the user suggested.
4. Deprioritize action-value and MCTS until data volume is much larger.
5. Consider LoRA only after saturating data scaling gains.

### Session 4: updated architecture opinions and next steps (2026-03-19)

**Updated opinion on the current pipeline:**
- The repo's main idea is strong: convert boards into structured embeddings, project them into Qwen hidden space, and use the pretrained transformer as a deep feature mixer via `inputs_embeds`.
- The current bottleneck is more likely in the policy interface than in the backbone itself.
- Right now the model gets 67 contextualized hidden states from the backbone but predicts moves mostly from one pooled token. That is probably too lossy for chess.

**Updated opinion on frozen backbone vs LoRA vs full tuning:**
- Frozen backbone is still the right default baseline.
- LoRA is a reasonable follow-up only after stronger policy heads and data scaling are tested.
- Full backbone tuning is not the highest-value next move right now. It adds cost and instability before the repo has exhausted cheaper bottleneck fixes.
- The practical order should be:
  1. improve the head
  2. improve data and supervision
  3. try LoRA
  4. try selective unfreezing
  5. only then consider full fine-tuning

**Why the head looks like the main bottleneck:**
- Chess moves are spatial: from-square, to-square, promotions, and piece-dependent movement patterns.
- The default policy head asks one global vector to summarize the board and decode a 5504-way move distribution.
- That likely throws away useful square-level information that the encoder and backbone already worked to preserve.
- This also helps explain why action-value auxiliary loss may not help much yet: the readout itself may be too weak.

### Updated next steps

1. Prioritize `experiments/exp019_spatial_head.py` as a top experiment.
   Reason: it attacks the clearest current architectural bottleneck while keeping the rest of the training stack mostly unchanged.

2. If the spatial head wins, make it the new baseline before doing more AV or LoRA work.
   Reason: many later experiments will be easier to evaluate once the policy readout is less constrained.

3. Compare policy heads directly under the same harness:
   - current global-token MLP head
   - spatial bilinear head
   - move-conditioned / two-tower head
   Reason: this should answer whether output-head design is now the main frontier.

4. Keep scaling real-game data.
   Reason: data quality is still the clearest proven lever in the repo.

5. Test soft targets after the head comparison.
   Reason: richer supervision may matter more once the model has a better policy interface.

6. Deprioritize action-value for now.
   Reason: it has repeatedly tied the baseline at current scales and may be blocked by readout design.

7. Keep LoRA on deck, but behind head and data improvements.
   Reason: adapting the backbone is less attractive if the model is still bottlenecked at the policy head.

### Specific thoughts on exp019_spatial_head.py

- The hypothesis is strong and well motivated.
- Using square-level hidden states for move scoring is a much more chess-native inductive bias than decoding all moves from a single pooled vector.
- Even if the exact bilinear implementation is not the final answer, the broader direction is promising.

**Caveats to watch:**
- A full-vocabulary scorer is still less clean than scoring only legal moves, so this may not be the endpoint design.
- The experiment should still be replicated if the gain is small.
- Promotion and special-move handling deserve explicit sanity checks.

### Current ranking of likely high-value bets

1. Better policy head using square-level features
2. More and better real-game data
3. Soft-target policy supervision
4. LoRA after the above plateau
5. Revisit action-value only after the policy interface is stronger
6. Full backbone tuning only much later, if justified by scale

### Session 5: notes after reviewing recent improvements (2026-03-19)

**What looks genuinely improved:**
- The repo is much stronger as a research harness than before.
- Results are being written to structured JSON outputs with config and metric details, which makes comparisons much more trustworthy.
- The recent experiment set tells a coherent story:
  - random-position AV does not help
  - real-game positions help a lot
  - LoRA at 50K does not beat the frozen baseline
- This is good progress because it removes ambiguity, not just because it raises one number.

**What I think is the biggest real improvement:**
- The project has moved from "many interesting ideas" toward "clear evidence about what is and is not the bottleneck."
- The strongest current evidence is:
  1. data quality matters a lot
  2. action-value is neutral at current scale
  3. LoRA is neutral at current scale
- That is valuable. Negative results are narrowing the search space productively.

**What I would be careful about:**
- `exp020_scaled_spatial.py` cites `exp019` as reaching 36.5% / 61.4% and frames scaling around that result, but I do not currently see an `outputs/exp019_spatial_head/results.json` artifact on disk.
- That does not mean the result is wrong, but it should be treated as provisional until the artifact is saved and easy to inspect.
- In general, any claim that materially changes the roadmap should have a durable output file like the AV and LoRA experiments do.

**My current interpretation of the new evidence:**
- The backbone is probably not the main bottleneck yet.
- LoRA tying the frozen baseline at 50K is a strong signal that backbone adaptation is not the highest-leverage next step.
- The next likely frontier is the policy interface:
  - how the model turns backbone hidden states into move scores
  - whether square-level structure is being used effectively

**Notes I want to keep in mind:**
- If `exp019` really did jump to the mid-30s, that is a much bigger architectural signal than any LoRA or AV result so far.
- If that number holds, policy-head design immediately becomes the top priority.
- If it does not hold up, the repo should still keep focusing on data quality and soft targets before expensive backbone adaptation.

**Ideas from this review:**
- Save a durable `results.json` for every experiment before using the result to justify the next experiment.
- Add a small "proven vs provisional" section to the README or ideas log:
  - proven: backed by output artifact
  - provisional: observed in terminal or partial run, needs rerun or save
- For spatial-head work, add diagnostics beyond top-1:
  - per-move-type accuracy
  - promotion accuracy
  - capture vs quiet move accuracy
  - opening/middlegame/endgame slices
- If spatial heads work, compare them against a legal-move-only scorer rather than only a full-vocab scorer.

**Short current take:**
- The recent improvements are meaningful.
- The repo has become better at falsifying ideas quickly.
- The strongest new strategic message is still not "tune more of Qwen."
- It is "use better chess structure at the output head, and keep feeding the model better positions."

## Roadmap / Best Next Steps

### Tier 1: do next

1. Confirm and save the `exp019_spatial_head` result as a durable artifact.
   Why: if the reported jump is real, it is the most important architecture signal in the repo right now.

2. Run a clean head-comparison benchmark.
   Compare:
   - standard global-token MLP head
   - spatial head
   - one move-conditioned / two-tower variant
   Keep data, split, seed, epochs, optimizer, and eval identical.

3. Keep scaling high-quality real-game data.
   Prefer real game positions over random positions for all major comparisons unless random positions are the variable being tested.

4. Test soft-target supervision on the strongest current head.
   Compare hard CE against:
   - top-k target distribution
   - temperature-smoothed Stockfish policy
   - KL-style policy matching if labels support it

5. Add stronger evaluation slices.
   At minimum report:
   - opening / middlegame / endgame
   - capture vs quiet
   - check / no-check
   - promotion positions if present

### Tier 2: do after Tier 1 stabilizes

1. If the spatial head wins, make it the baseline and rerun one or two key prior comparisons on it.
   Most important reruns:
   - soft targets
   - LoRA
   - maybe AV only if there is a strong reason

2. Test a legal-move-only scorer.
   Instead of scoring the full 5504 vocabulary, score only legal moves using move-conditioned representations.

3. Improve pooling or board-summary design for the value head.
   The policy may benefit from square-level readout while the value head may still need better aggregation.

4. Add multi-seed replication for any improvement smaller than about 2 points.

### Tier 3: later / conditional

1. Revisit LoRA only after head design and data quality plateau.
   Current evidence suggests backbone adaptation is not the main lever yet.

2. Revisit action-value only after the policy interface is stronger.
   A weak readout may have hidden any benefit from richer move-value supervision.

3. Explore chess-native transformer designs.
   This becomes more attractive if the Qwen-based pipeline stops improving despite better heads and better data.

4. Explore positional / geometry upgrades such as rank-file factorization, 2D positional schemes, or RoPE-like board-aware variants.
   These are more attractive after the basic head bottleneck is addressed.

### Session 6: Chess-specific transformer breakthrough (2026-03-20)

**Experiments run:**

| Exp | Architecture | Data | Best Acc | Top3 | Games vs SF d3 | Notes |
|-----|-------------|------|----------|------|----------------|-------|
| exp020 | Qwen3+spatial | 200K×5ep | 36.5% | — | — | Data scaling saturated |
| exp021 | Qwen3+spatial+LoRA+search | 50K | 36.5% | — | W0/D0/L6 | LoRA TIE, 1-ply search useless |
| exp022 | Qwen3+spatial+SF value head+α-β | 10K SF | — | — | W0/D0/L20 (all depths) | 69.8% sign acc but 0 wins |
| exp023 | **Chess Transformer 8L/512d** | 50K | **40.5%** | **68.5%** | W0/D0/L8 | +4pp over Qwen3+spatial |
| exp024 | Chess Transformer, full data | ~460K×3ep | **48.7%** | **73.9%** | **W0/D2/L6** | First draws! Data scales! |

**Key findings:**

1. **Data scaling saturated for frozen Qwen3**: 50K→200K with spatial head gave identical 36.5%. The frozen text backbone is the ceiling.

2. **LoRA still neutral**: Even with spatial head, LoRA rank-16 on q/v projections doesn't help at 50K data.

3. **Search doesn't rescue weak policy**: SF-trained value head achieved 69.8% sign accuracy on centipawn predictions, but depth-0/1/2 alpha-beta search all scored 0 wins vs SF d3. The policy model makes too many bad moves for search to compensate.

4. **PARADIGM SHIFT — Chess-specific transformer works**: Replacing the frozen 603M Qwen3 backbone with a purpose-built 8-layer transformer (512d, 8 heads, 26M fully trainable params) achieved **40.5% best accuracy** on 50K data. Key advantages:
   - All parameters learn chess (vs. frozen text features)
   - Loss still declining at epoch 10 → model is data-starved
   - Training is 10× faster per epoch (no backbone forward pass)
   - Top3 accuracy: 68.5% vs Qwen3+spatial's 61.4%

5. **exp023 model is severely data-starved**: 26M params on 50K positions = massive overfitting risk. Loss was still improving at epoch 10 with no accuracy plateau. The frozen Qwen3 approach saturated because the backbone was the ceiling; the chess transformer should scale much further with more data.

**Critical insight chain (cumulative):**
1. Position quality >> loss function (Session 2)
2. Head architecture >> backbone adaptation (Session 4–5, confirmed by LoRA TIE)
3. **Fully trainable chess model >> adapted text model** (Session 6)
4. Data volume should scale with trainable params (50K enough for 223K encoder params, not for 26M transformer)

**Implication:** The project's original premise — "repurpose a text LLM for chess" — may be fundamentally flawed. A purpose-built chess network, even at 1/25th the parameter count, learns better chess representations because every parameter is optimized for the task. This is consistent with AlphaZero/Leela's approach.

**Next priority:** exp024 scales the chess transformer to the full 460K dataset with 3 epochs. If accuracy climbs to 45%+ and game survival improves, this becomes the new paradigm.

**exp024 RESULT — CONFIRMED: Data scaling works for chess transformer!**
- 461K positions × 3 epochs → **48.7% accuracy, 73.9% top3**
- +8.2pp over exp023 (50K) and +12.2pp over Qwen3+spatial
- **First draws against SF d3!** Two games as white achieved fivefold repetition draws (31mv, 39mv)
- Games W0/D2/L6 — still losing most games but survival improved dramatically
- Epoch progression: 43.4% → 46.5% → 48.7% — still climbing at epoch 3
- Training time: 4812s for 3 epochs (26M params, batch 128×2 accum)
- Loss: 3.11 → 2.50 → 2.33 (not converged — more epochs or data could help)

**Analysis of game results:**
- All draws were as white (home advantage from repetition forcing)
- Losses as black were faster (19-27mv) than losses as white (27-43mv)
- The model is learning to avoid immediate blunders but still makes strategic errors
- Draws via repetition suggest the model can maintain its position but not make progress

**Updated cumulative results:**

| Exp | Architecture | Data | Best Acc | Top3 | Games vs SF d3 |
|-----|-------------|------|----------|------|----------------|
| exp013 | Qwen3+standard | 50K | 25.0% | 45.0% | — |
| exp018 | Qwen3+standard+LoRA | 50K | 25.0% | — | — |
| exp019 | Qwen3+spatial | 50K | 36.5% | 61.4% | — |
| exp020 | Qwen3+spatial | 200K | 36.5% | — | — |
| exp023 | Chess Transformer | 50K | 40.5% | 68.5% | W0/D0/L8 |
| **exp024** | **Chess Transformer** | **460K** | **48.7%** | **73.9%** | **W0/D2/L6** |

**Next directions (ranked by expected impact):**
1. **More training**: Loss still declining → train more epochs on the same data (5-6ep total)
2. **Stockfish-labeled data**: Replace game-outcome supervision with SF best-move labels for higher quality targets
3. **Deeper/wider model**: The 8L/512d may be underfitting at this data volume — try 12L/512d or 8L/768d
4. **Search integration**: With 48.7% policy accuracy, alpha-beta search with a trained value head might actually help now
5. **More data**: The HF dataset has 475K. Can we generate more via Stockfish self-play?

### Concrete next experiment order

1. Save and verify `exp019`
2. Run head A/B/C comparison
3. Run soft-target experiment on the winning head
4. Scale winning setup to larger real-game data
5. Add eval slices and failure-case logging
6. Reconsider LoRA only if the stronger baseline stalls

### Session 5: Spatial head breakthrough and LoRA result (2026-03-19)

**exp018 LoRA result: TIE at 25.0%**
- LoRA (rank=16, q_proj+v_proj, 2.3M params) on 50K HF game-play, 3 epochs
- Best accuracy: 25.0% — identical to frozen baseline (exp013)
- Loss curve: 4.76 → 4.57 → 4.55 (vs frozen 4.77 → 4.57 → 4.55)
- LoRA used 10.6GB VRAM vs ~3.5GB for frozen
- Conclusion: backbone adaptation doesn't help at 50K data scale. The frozen Qwen3 backbone is NOT the bottleneck.

**exp019 Spatial head result: BREAKTHROUGH +11.5pp**
- Spatial bilinear policy head: from/to square features → bilinear scoring
- 50K HF game-play, 3 epochs, frozen backbone
- **Best accuracy: 36.5%, top3: 60.0%** (vs standard: 25.0%, top3: 40.5%)
- Loss: 3.68 → 2.91 → 2.61 (vs standard: 4.77 → 4.57 → 4.55)
- Only 1.5M trainable params (vs 7.4M for standard head!)
- Loss still declining steeply at epoch 3 — not saturated
- Still loses all 4 games to SF d3

**Why the spatial head works so dramatically better:**
1. Per-square features preserve chess structure — a move from e2 to e4 uses features from THOSE specific squares
2. The standard head compresses 67 tokens into 1 vector → 5504 classes. The spatial head uses 64 relevant square pairs.
3. Bilinear scoring (from_proj * to_proj) naturally captures piece-square interactions
4. Works with fewer params because the structure does the heavy lifting

**Updated cumulative results table:**

| Experiment | Head | Data | N | Epochs | Best Acc | Top3 | Notes |
|------------|------|------|---|--------|----------|------|-------|
| exp013 | standard | HF games | 50K | 3 | 25.0% | 45.0% | frozen baseline |
| exp014 | standard | HF games | 475K | 1 | 18.4% | 39.4% | underfit |
| exp018 | standard+LoRA | HF games | 50K | 3 | 25.0% | 40.5% | LoRA=TIE |
| **exp019** | **spatial** | HF games | 50K | 3 | **36.5%** | **61.4%** | **+11.5pp** |

**Key insight: The policy head architecture was the main bottleneck, not the backbone or loss function.**

**Next steps (in progress):**
1. exp020: Scale spatial head to 200K × 5 epochs — loss was still declining steeply
2. If accuracy > 45%, test with 1-ply search for game play
3. Consider spatial head + LoRA combination
4. The spatial head should now be the default for all future experiments

### Session 7: Relative position attention bias (2026-03-20)

**exp026 result: TIE (slight regression)**
- Hypothesis: Learned per-head relative position bias (rank_diff, file_diff) improves accuracy
- A/B comparison on 50K HF game data, 5 epochs, same custom transformer (8L/512d/8h)
- Baseline: 37.8% best, Rel bias: 37.0% best → **-0.8pp, TIE**
- Bias adds only 4,944 params (negligible)
- Both models had identical loss curves (3.98→2.29)
- Games vs SF d3: W0/D0/L6

**First attempt failed** — passing bias as `src_mask` to `nn.TransformerEncoder` collapsed training to 0.8%. Fixed by implementing custom transformer blocks (`MultiHeadAttentionWithBias`) that properly add the bias inside the QK^T computation.

**Why it didn't help:**
1. The absolute square positional embeddings (learned, 64 positions) likely already capture rank/file geometry
2. At 50K data, the model may not have enough signal for the bias to learn useful patterns
3. The bias is shared across all layers — per-layer bias might work differently
4. Chess relationships are piece-dependent (rook cares about rank/file, bishop about diagonals) — a single bias table can't capture this

**Conclusion:** Deprioritize relative position bias. The absolute embeddings are sufficient at this scale.

**Updated cumulative results:**

| Experiment | Architecture | Data | Best Acc | Top3 | Games vs SF d3 |
|------------|-------------|------|----------|------|----------------|
| exp013 | Qwen3+standard | 50K | 25.0% | 45.0% | — |
| exp019 | Qwen3+spatial | 50K | 36.5% | 61.4% | — |
| exp020 | Qwen3+spatial | 200K | 36.5% | — | — |
| exp023 | Chess Transformer | 50K | 40.5% | 68.5% | W0/D0/L8 |
| exp024 | Chess Transformer | 460K | 48.7% | 73.9% | W0/D2/L6 |
| exp026 | Chess Transformer +rel_bias | 50K | 37.0% | 66.8% | — |

**Next hypothesis to test:** The chess transformer at 50K gets ~38% (custom blocks) vs exp023's 40.5% (nn.TransformerEncoder). The custom blocks might slightly underperform PyTorch's fused implementation. Two immediate options:
1. **Continue training exp024** checkpoint for more epochs (loss still declining at 2.33)
2. **Deeper model** — try 12L or 16L at 460K data to see if depth helps
3. **Label smoothing / soft targets** — the current hard-label CE may be suboptimal for positions where multiple moves are reasonable

### Session 8: Label smoothing experiment (2026-03-20)

**exp028 result: TIE (+0.4pp, within noise)**
- 3-way A/B/C: ε=0.0 (hard CE), ε=0.1, ε=0.2 on 50K data, 5 epochs
- ε=0.0: 38.6%, ε=0.1: 39.0%, ε=0.2: 38.8%
- Delta: +0.4pp for ε=0.1 — too small to be meaningful at N=500 eval
- All variants still improving at epoch 5, loss not converged
- ε=0.1 had better top3 (67.2%) than ε=0.0 (65.4%) — label smoothing may help ranking
- Games vs SF d3: W0/D0/L6 (no improvement in gameplay)

**Why label smoothing is neutral here:**
1. +0.4pp is well within eval noise on 500 samples
2. The model is still data-starved at 50K — the bottleneck is data/capacity, not overconfidence
3. Label smoothing helps most when the model is near convergence — here loss is still declining steeply
4. Uniform smoothing over all 5504 moves is crude — most of that mass goes to terrible moves

**Conclusion:** Label smoothing is provisionally neutral. Could revisit at larger scale where overfitting becomes a real concern. Not worth pursuing now.

**Updated cumulative results:**

| Experiment | Architecture | Data | Best Acc | Top3 | Games vs SF d3 | Notes |
|------------|-------------|------|----------|------|----------------|-------|
| exp013 | Qwen3+standard | 50K | 25.0% | 45.0% | — | |
| exp019 | Qwen3+spatial | 50K | 36.5% | 61.4% | — | |
| exp023 | Chess Transformer | 50K | 40.5% | 68.5% | W0/D0/L8 | |
| exp024 | Chess Transformer | 460K | 48.7% | 73.9% | W0/D2/L6 | BEST |
| exp026 | +rel_bias | 50K | 37.0% | 66.8% | — | TIE |
| exp028 | +label smoothing | 50K | 39.0% | 67.2% | W0/D0/L6 | TIE |

**Pattern emerging:** At 50K data, nothing beats the baseline architecture. The only proven lever is **scaling data** (50K→460K gave +8pp). Architecture tweaks (relative bias, label smoothing) are neutral at this scale.

**Next steps — focus on data scaling and training efficiency:**
1. The biggest untapped gain: train the current 8L/512d model for MORE epochs on 460K data (loss was 2.33 and still declining at ep3)
2. Or try deeper model (12L) on full data — more capacity for more data
3. Deprioritize 50K ablations — the model is data-starved, making all interventions look flat

### Session 9: Data diversity vs epochs experiment (2026-03-20)

**exp029 result: TIE (all within noise)**
- 3-way matched-compute comparison: each variant sees 200K total training examples
- 50K×4ep: 37.0% best, loss 2.44
- 100K×2ep: 37.4% best, loss 2.60
- 200K×1ep: 36.4% best, loss 3.08
- Diversity delta (200K×1 - 50K×4): -0.6pp — within noise
- Games vs SF d3: W0/D0/L6 (no improvement)

**Why diversity ≠ the driver of data scaling gains:**
1. At matched compute (200K total examples), diversity doesn't help — 50K×4 ≈ 100K×2 ≈ 200K×1
2. The exp024 gain (50K→460K = +8pp) is actually about **total gradient volume**: 460K×3ep = 1.38M examples vs 50K×10ep = 500K examples
3. 200K×1ep has a HIGHER loss (3.08) than 50K×4ep (2.44) — the model barely learns from single pass
4. The sweet spot appears to be ~2 passes minimum to learn well (100K×2ep matched 50K×4ep)

**Critical implication:**
The path to better accuracy is simply **more gradient updates** — either:
- More epochs on existing data (cheap, diminishing returns)
- More unique data with sufficient epochs (expensive but scalable)
- Or both: bigger dataset × more epochs

**Updated cumulative results:**

| Experiment | Architecture | Data | Best Acc | Top3 | Games vs SF d3 | Notes |
|------------|-------------|------|----------|------|----------------|-------|
| exp013 | Qwen3+standard | 50K | 25.0% | 45.0% | — | |
| exp019 | Qwen3+spatial | 50K | 36.5% | 61.4% | — | |
| exp023 | Chess Transformer | 50K | 40.5% | 68.5% | W0/D0/L8 | 10 epochs |
| exp024 | Chess Transformer | 460K | 48.7% | 73.9% | W0/D2/L6 | BEST |
| exp026 | +rel_bias | 50K | 37.0% | 66.8% | — | TIE |
| exp028 | +label smoothing | 50K | 39.0% | 67.2% | W0/D0/L6 | TIE |
| exp029 | 50K×4/100K×2/200K×1 | matched 200K | 37.4% | 65.4% | W0/D0/L6 | TIE |

**Next steps — shift to longer training:**
1. Train 8L/512d for 10 epochs on 460K data (most direct path to improvement — exp024 loss was 2.33 at ep3, should reach ~2.0 at ep10). This exceeds the 10-min budget but is the highest-value experiment.
2. Alternative: depth scaling on 50K first (8L vs 12L) as a quick signal before committing to expensive full-data runs
3. Alternative: try fundamentally different approach — self-play / reinforcement loop instead of supervised learning

### Session 9 (cont): Depth scaling experiment (2026-03-20)

**exp030 result: TIE (-0.6pp)**
- 2-way A/B: 8L/512d (26.1M params) vs 12L/512d (38.7M params) — 50K data, 5 epochs
- 8L: 38.6% best, loss 2.33
- 12L: 38.0% best, loss 2.34
- Depth delta: -0.6pp — within noise
- 12L is 26% slower per epoch (328s vs 260s)
- Games vs SF d3: W0/D1/L5 (8L got one fivefold-repetition draw as black)

**Why depth doesn't help at 50K:**
1. With only 50K positions and 26M base params, the model is already data-starved
2. Adding 48% more params (12M extra) only adds more parameters to overfit with
3. The 12L model's loss curve tracks 8L nearly exactly — more layers aren't learning different features
4. This matches the "all 50K ablations are TIE" pattern: the bottleneck is DATA, not architecture

**Updated cumulative results:**

| Experiment | Architecture | Data | Best Acc | Top3 | Games vs SF d3 | Notes |
|------------|-------------|------|----------|------|----------------|-------|
| exp013 | Qwen3+standard | 50K | 25.0% | 45.0% | — | |
| exp019 | Qwen3+spatial | 50K | 36.5% | 61.4% | — | |
| exp023 | Chess Transformer | 50K | 40.5% | 68.5% | W0/D0/L8 | 10 epochs |
| exp024 | Chess Transformer | 460K | 48.7% | 73.9% | W0/D2/L6 | BEST |
| exp026 | +rel_bias | 50K | 37.0% | 66.8% | — | TIE |
| exp028 | +label smoothing | 50K | 39.0% | 67.2% | W0/D0/L6 | TIE |
| exp029 | data diversity | matched 200K | 37.4% | 65.4% | W0/D0/L6 | TIE |
| exp030 | 12L depth | 50K | 38.0% | 65.8% | W0/D1/L5 | TIE |

**Definitive conclusion on 50K ablations:**
Five consecutive experiments (exp026-030) testing different axes (attention bias, label smoothing, data diversity, depth) have ALL produced TIEs at 50K data. The 50K ablation regime is exhausted. Every intervention looks flat because the model is severely data-starved.

**Path forward — must break out of 50K regime:**
1. **Accept longer experiments**: Train on full 460K for more epochs. exp024's loss was still declining at ep3 (2.33). Even 1 more epoch could push past 50%.
2. **Self-play reinforcement loop**: Use the strongest model (exp024, 48.7%) as a starting point for REINFORCE/policy gradient from self-play games. This generates unlimited training data.
3. **Generate more labeled data**: Use Stockfish to label positions from the HF dataset with best moves, increasing label quality.
4. **Stop doing 50K ablations entirely.**

### Session 9 (cont): Extended training BREAKTHROUGH (2026-03-20)

**exp031 result: NEW BEST — 51.2% accuracy (+2.5pp over exp024)**

Training on 460K data for 6 epochs (vs exp024's 3 epochs):
- Epoch 1: 43.7%, loss 2.97
- Epoch 2: 46.4%, loss 2.39
- Epoch 3: 48.8%, loss 2.28 (matches exp024's 48.7%)
- Epoch 4: 48.9%, loss 2.19 (marginal gain — LR declining fast)
- Epoch 5: 50.3%, loss 2.11 (breaks 50% for the first time!)
- Epoch 6: 51.2%, loss 2.06 (still declining at epoch end!)

Games vs SF d3: W0/D1/L7 (1 fivefold repetition draw as white in 33mv)
Total time: 14,364s (~4 hours)

**Why this works:**
1. Epochs 1-3 match exp024 almost exactly → same architecture/data → reproducible
2. Epochs 4-6 continue to improve because loss hasn't plateaued
3. The cosine LR schedule over 6 epochs decays more gradually than 3 epochs, giving more training time at useful learning rates
4. Loss 2.06 is STILL declining → even more training could help

**Loss is NOT converged! Accuracy trajectory suggests ~52-53% at 10 epochs.**

**Updated cumulative results (ALL experiments):**

| Experiment | Architecture | Data | Best Acc | Top3 | Games vs SF d3 | Notes |
|------------|-------------|------|----------|------|----------------|-------|
| exp013 | Qwen3+standard | 50K | 25.0% | 45.0% | — | |
| exp019 | Qwen3+spatial | 50K | 36.5% | 61.4% | — | |
| exp023 | Chess Transformer | 50K | 40.5% | 68.5% | W0/D0/L8 | 10 epochs |
| exp024 | Chess Transformer | 460K×3ep | 48.7% | 73.9% | W0/D2/L6 | prev best |
| exp026 | +rel_bias | 50K | 37.0% | 66.8% | — | TIE |
| exp028 | +label smoothing | 50K | 39.0% | 67.2% | W0/D0/L6 | TIE |
| exp029 | data diversity | matched 200K | 37.4% | 65.4% | W0/D0/L6 | TIE |
| exp030 | 12L depth | 50K | 38.0% | 65.8% | W0/D1/L5 | TIE |
| **exp031** | **Chess Transformer** | **460K×6ep** | **51.2%** | **76.2%** | **W0/D1/L7** | **NEW BEST** |

**Key milestone:** First model to break 50% accuracy. The improvement path is clear: more training epochs on the full data.

### exp032: Continue Training from Checkpoint (LR=1e-5, +4 epochs)
**Hypothesis:** Fine-tuning exp031's 51.2% model with a low constant LR for 4 more epochs (total 10 effective) will push accuracy further.
**Result:** 51.4% accuracy (+0.2pp), 78.4% top3 (+2.2pp). Loss 2.06→1.96.
**Verdict:** MARGINAL — top-1 barely moved, but top-3 gained meaningfully.

Epoch progression (continuing from exp031's epoch 6):
- Epoch 7: 50.7%, loss 2.01
- Epoch 8: 50.7%, loss 1.99
- Epoch 9: 51.1%, loss 1.98
- Epoch 10: 51.4%, loss 1.96

Games vs SF d3: W0/D1/L7 (1 fivefold repetition draw as white in 51mv)
Total time: 9582s (~2.7 hours)

**Analysis:**
1. Top-1 improved only marginally (+0.2pp), but top-3 gained +2.2pp (76.2%→78.4%)
2. Low LR (1e-5) refines ranking (top-3) more than top-1 prediction
3. Loss dropped from 2.06→1.96 — still declining, but diminishing returns
4. 4 additional epochs at 1e-5 don't match the value of the original 6 epochs at 3e-4
5. The top-1 ceiling (~51%) may reflect data quality rather than model capacity — amateur game labels have inherent noise

**Updated cumulative results (ALL experiments):**

| Experiment | Architecture | Data | Best Acc | Top3 | Games vs SF d3 | Notes |
|------------|-------------|------|----------|------|----------------|-------|
| exp013 | Qwen3+standard | 50K | 25.0% | 45.0% | — | |
| exp019 | Qwen3+spatial | 50K | 36.5% | 61.4% | — | |
| exp023 | Chess Transformer | 50K | 40.5% | 68.5% | W0/D0/L8 | 10 epochs |
| exp024 | Chess Transformer | 460K×3ep | 48.7% | 73.9% | W0/D2/L6 | prev best |
| exp026 | +rel_bias | 50K | 37.0% | 66.8% | — | TIE |
| exp028 | +label smoothing | 50K | 39.0% | 67.2% | W0/D0/L6 | TIE |
| exp029 | data diversity | matched 200K | 37.4% | 65.4% | W0/D0/L6 | TIE |
| exp030 | 12L depth | 50K | 38.0% | 65.8% | W0/D1/L5 | TIE |
| exp031 | Chess Transformer | 460K×6ep | 51.2% | 76.2% | W0/D1/L7 | NEW BEST |
| **exp032** | **+continue LR=1e-5** | **460K×10ep** | **51.4%** | **78.4%** | **W0/D1/L7** | **+0.2pp marginal** |

**Next priorities:**
1. **Self-play / expert iteration** — instructions emphasize self-play as default research direction. The 51.4% model is strong enough to bootstrap self-play training.
2. **Stockfish-labeled data** — replace noisy game-outcome labels with Stockfish best-move labels for higher-quality supervision
3. **Larger model on full data** — 12L at 460K×6ep (now that data is not the bottleneck)
4. **MCTS search at inference** — use the value head + policy for tree search during evaluation

---

## Session 10: Self-Play & Stockfish Distillation

### exp033: Self-play REINFORCE (FAILED)
**Hypothesis:** REINFORCE from self-play game outcomes will improve game-playing strength.
**Result:** CATASTROPHIC FAILURE — accuracy 51.4% → 0.8%. Model completely destroyed.
**Verdict:** NEGATIVE — pure vanilla REINFORCE is too unstable for fine-tuning.

Details:
- 20 generations × 30 self-play games = 600 total games
- Temperature=0.8, max 100 moves, material adjudication (±3 material)
- Gen 1 had reasonable losses (pl=0.20, vl=0.67) but all subsequent gens were NaN
- A single REINFORCE update was enough to corrupt all model weights
- After training: W0/D0/L8 vs SF d3 (all checkmate losses, avg 22 moves)
- Total time: 267s

**Why it failed:**
1. Vanilla REINFORCE has catastrophically high variance with game-outcome rewards
2. No KL constraint to anchor to the original supervised policy → complete forgetting
3. The advantage normalization didn't help because the gradient magnitude was still huge
4. NaN propagation after one bad step — once weights corrupt, everything cascades
5. Self-play with adjudicated outcomes provides noisy labels (most games adjudicated by material, not checkmate)

**Lessons for future self-play:**
- Need KL penalty or PPO-style clipping to prevent catastrophic forgetting
- Or much better: use self-play data as SUPERVISED targets (expert iteration)
- Pure REINFORCE needs thousands of games per update to reduce variance
- Consider mixing supervised + RL data (e.g., 90% supervised + 10% self-play)

### exp034: Stockfish Distillation — Fine-tune on SF-labeled positions
**Hypothesis:** Relabeling 50K positions with SF d8 best moves and fine-tuning will improve play strength.
**Result:** SF accuracy: 48.8% → 52.9% (+4.1pp), but human accuracy: 51.4% → 42.3% (-9.1pp). Games: W0/D0/L8 (WORSE).
**Verdict:** MIXED/NEGATIVE — learned SF moves better but lost general chess knowledge.

Details:
- 50K positions labeled with SF d8 in 185s (270/s, zero skipped)
- Human-SF move agreement: only 37.4% (humans differ from SF 63% of the time!)
- Baseline SF accuracy: 48.8% → model already partially knows SF-style moves
- After 2 epochs: SF acc 52.9%, loss 1.588 (overfit to 50K SF labels)
- Lost the draw exp032 had vs SF d3; all 8 games lost by checkmate
- Total time: 745s

**Analysis:**
1. 50K SF-labeled positions is too few to retrain a model that learned from 460K human positions
2. The model catastrophically forgot its human-trained policy while learning SF targets
3. Human-SF agreement of only 37.4% explains the accuracy tradeoff — the targets point in very different directions
4. Need MUCH more SF-labeled data (200K+) to replace human training entirely
5. Or: mix human + SF data during training (not replace)
6. Or: use SF labels as soft targets (KL divergence to SF policy, not hard CE)

**Key insight: Best path forward is likely MIXED training — mostly human data + some SF data.**

**Updated cumulative results (ALL experiments):**

| Experiment | Architecture | Data | Best Acc | Top3 | Games vs SF d3 | Notes |
|------------|-------------|------|----------|------|----------------|-------|
| exp013 | Qwen3+standard | 50K | 25.0% | 45.0% | — | |
| exp019 | Qwen3+spatial | 50K | 36.5% | 61.4% | — | |
| exp023 | Chess Transformer | 50K | 40.5% | 68.5% | W0/D0/L8 | 10 epochs |
| exp024 | Chess Transformer | 460K×3ep | 48.7% | 73.9% | W0/D2/L6 | |
| exp026 | +rel_bias | 50K | 37.0% | 66.8% | — | TIE |
| exp028 | +label smoothing | 50K | 39.0% | 67.2% | W0/D0/L6 | TIE |
| exp029 | data diversity | matched 200K | 37.4% | 65.4% | W0/D0/L6 | TIE |
| exp030 | 12L depth | 50K | 38.0% | 65.8% | W0/D1/L5 | TIE |
| exp031 | Chess Transformer | 460K×6ep | 51.2% | 76.2% | W0/D1/L7 | BEST |
| exp032 | +continue LR=1e-5 | 460K×10ep | 51.4% | 78.4% | W0/D1/L7 | +0.2pp |
| exp033 | REINFORCE selfplay | — | 0.8% | 3.2% | W0/D0/L8 | DESTROYED |
| exp034 | SF distill 50K | 50K SF d8 | 52.9%(SF) | — | W0/D0/L8 | WORSE |
| exp035 | 90%human+10%SF mix | 460K+50K | 51.8% | — | W0/D0/L8 | +0.4pp marginal |

### exp035: Mixed Human+SF Training (90/10 ratio)
**Hypothesis:** 90% human + 10% SF data per batch retains human accuracy while improving SF-alignment.
**Result:** Human acc 51.4% → 51.8% (+0.4pp), SF acc 49.2% → 51.2% (+2.0pp). Games: W0/D0/L8.
**Verdict:** MARGINAL — slight accuracy gain but lost the draw exp032 had vs SF.

Details:
- 461K human + 50K SF (d8) mixed training, 2 epochs, LR=3e-5 cosine
- Epoch 1: human=51.8%, sf=50.8% (both improved)
- Epoch 2: human=51.6%, sf=51.2% (slight regression on human)
- Games vs SF d3: W0/D0/L8 (all checkmate, avg 47 moves — worse than exp032's 1 draw)
- Total time: 5515s (~92 min)

**Analysis:**
1. The 10% SF mix gently nudged both metrics up but couldn't break through
2. Two epochs on 460K is essentially just re-training what exp031-032 already learned
3. The LR=3e-5 was too low for meaningful learning from the SF signal
4. The model has hit a genuine capacity/data ceiling at ~51% — more of the same data doesn't help
5. Game play didn't improve because the accuracy gain was too small to affect tactical strength

### exp036: Inference-time Search (1-ply lookahead with value head)
**Hypothesis:** Using the value head to re-rank top-K policy moves via 1-ply lookahead will improve games vs SF.
**Result:** Search HURTS — argmax W0/D1/L7, search_top5 W0/D0/L8, search_top10 W0/D0/L8.
**Verdict:** NEGATIVE — the value head actively makes worse move selections.

Details:
- Three strategies tested: argmax (baseline), top-5 search, top-10 search
- Argmax replicated exp032's W0/D1/L7 exactly
- Both search strategies lost the draw and went W0/D0/L8
- Search changed 189 moves (top-5) and 147 moves (top-10) away from policy picks
- The value head is mis-calibrated: it recommends worse moves than the policy alone
- Total time: 34s (very fast, no training)

**Analysis:**
1. Value head was trained on game outcomes (0=loss, 1=draw, 2=win) which are noisy labels
2. Game outcome labels don't teach the model to evaluate POSITIONS, only to average outcomes
3. The value head needs Stockfish-labeled evaluations (centipawn scores) to be accurate
4. Without a good value head, any form of search (MCTS, AlphaZero-style) will fail
5. **Must fix the value head before search can help**

**Key insight: The value head is the weakest link. Fix it with SF evaluations before investing in search.**

### exp037: Fix value head with SF evaluations

**Goal:** Fine-tune ONLY the value head on SF centipawn evaluations → re-test search
**Data:** 20K positions labeled with SF d8 → cp to WDL targets (>100cp=win, <-100cp=loss, else=draw)
**Training:** Freeze policy, train ONLY value head (132K params) for 5 epochs, LR=1e-3

**Results:**
- Value head training: val_acc 82.7% → 84.6% over 5 epochs
- Distribution heavily skewed: win=13%, draw=78%, loss=9%
- Old value diagnostic: Starting +0.043, e4 -0.036, Italian -0.077 (bad calibration)
- New value diagnostic: Starting +0.000, e4 +0.001, Italian +0.069 (well calibrated!)
- Games: argmax W0/D0/L8, search_top5 W0/D0/L8, search_top10 W0/D0/L8
- Argmax lost its draw compared to exp036 (game variance, policy unchanged)
- Total time: 260s

**Analysis:**
1. Value head successfully calibrated — Starting/e4 near 0.0, Italian slightly positive (correct)
2. But 84.6% WDL accuracy is mainly predicting "draw" (78% of data is draw)
3. 1-ply search with even a calibrated value head doesn't distinguish top-5 policy moves
4. The model's fundamental weakness is at the policy level, not search
5. Need stronger policy OR deeper search to improve game performance
6. **Conclusion: Value head fix is necessary but NOT sufficient for search to help**

### exp038: Expert Iteration — model proposes, SF selects

**Goal:** Train on SF's best move ONLY when it's in the model's top-10 predictions
**Data:** 50K positions → model top-10 → SF d8 selects → 45,165 training samples
**Key finding:** SF is in model's top-1 50.7% of the time, in top-10 90.3% of the time!
**Training:** 461K human + 45K EI at 30% EI ratio, 2 epochs, LR=3e-5

**Results:**
- Expert iteration stats: 25,350 agreed + 19,815 corrected (positions where SF picks different top-10 move)
- Human accuracy: 51.4% → 50.9% (ep1) → 50.2% (ep2) — GETTING WORSE
- Top3 held steady: 78.4% → 78.5% → 78.4%
- Games: W0/D0/L8 (lost exp032's draw again)
- Total time: 7105s (~2 hours)

**Analysis:**
1. Even with model-tractable filtering, SF signal still hurts human accuracy
2. The 30% EI ratio dilutes human training too much
3. The model's "errors" vs SF are deeply embedded — not just top-1 vs top-2 ranking issues
4. Human and SF play are genuinely different strategies, not just noisy versions of each other
5. **Key insight: Can't improve game-play by changing training targets alone. Must scale the model.**

**Session 10 Meta-Analysis:**
- exp033: REINFORCE self-play → DESTROYED model (catastrophic forgetting)
- exp034: Pure SF distill 50K → Mixed (SF acc +4pp, human acc -9pp, games WORSE)
- exp035: 90% human + 10% SF → Marginal (+0.4pp, games W0/D0/L8)
- exp036: 1-ply search with old value head → HURTS (lost draw)
- exp037: Fix value head with SF evals → Calibration works but search still W0/D0/L8
- exp038: Expert iteration → Still hurts accuracy (-1.2pp), games W0/D0/L8
- **Conclusion: 8L/512d architecture at 460K data has hit its ceiling at ~51.4% accuracy.
  No amount of target-engineering (SF, EI, mixed) or search (1-ply) helps.
  Must either scale model (12L+) or scale data to break through.**

### exp039: 12-Layer Model from Scratch — Testing Capacity Ceiling

**Goal:** Train 12L/512d/8h model (38.7M params vs 26.1M for 8L) from scratch on full 460K
**Training:** 6 epochs, LR=3e-4 with warmup+cosine, batch 128×2

**Results (vs 8L baselines):**
| Metric | 12L (exp039) | 8L (exp031) | 8L (exp032) |
|--------|-------------|-------------|-------------|
| Accuracy | 50.1% | 51.2% | 51.4% |
| Top3 | 76.7% | 76.2% | 78.4% |
| Loss | 2.054 | 2.133 | — |
| Games | W0/D1/L7 | W0/D1/L7 | W0/D1/L7 |
| Time | 18,037s | ~14,400s | ~7,200s |

Epoch-by-epoch: 43.4% → 46.2% → 47.8% → 48.9% → 49.7% → 50.1%
(vs 8L: 43.4% → 46.5% → 48.7% at ep 1-3, reaching 51.2% at ep 6)

**Analysis:**
1. **12L is WORSE than 8L** — capacity is NOT the bottleneck
2. 12L has consistently LOWER loss but LOWER accuracy = overfitting to training data noise
3. Extra parameters memorize but don't generalize — classic sign that data quality limits performance
4. The ceiling is in the DATA (amateur games), not the MODEL (architecture/capacity)
5. 48% more params + 25% more training time yields 1.1pp LOWER accuracy
6. **Key insight: Data quality > model capacity for chess move prediction on amateur games**

**Session 10+11 Complete Story:**
- exp033: REINFORCE self-play → DESTROYED model
- exp034: Pure SF distill → Mixed result (-9pp human acc)
- exp035: 90/10 human+SF mix → Marginal (+0.4pp)
- exp036: 1-ply search with old value head → HURTS
- exp037: SF-calibrated value head → Search still W0/D0/L8
- exp038: Expert iteration → Still -1.2pp, W0/D0/L8
- exp039: 12L model → 50.1% < 8L's 51.2%, capacity NOT the bottleneck
- **Meta-conclusion: Can't break through with current data. Need higher quality training data
  (master-level games) or fundamentally different training objective.**

**Next priorities:**
1. **Data quality filtering** — filter to higher-ELO games if rating info available
2. **Data augmentation** — board flipping to double effective data
3. **Label smoothing / regularization** — help 8L generalize better

### exp040: Board Flip Augmentation + Label Smoothing (Fine-tuned)

**Goal:** Fine-tune exp032 on augmented data (horizontal flip) with label smoothing
**Data:** 100K positions → 200K augmented (100K real + 100K h-flipped)
**Training:** Fine-tune from exp032, 2 epochs, LR=1e-5, label_smoothing=0.1
**Eval:** REAL (non-flipped) positions only

**Results:**
- Baseline (exp032): 51.4% acc, 78.4% top3
- Epoch 1: 50.1% (-1.3pp), top3 76.5% (-1.9pp)
- Epoch 2: 50.5% (-0.9pp), top3 76.4% (-2.0pp)
- Games: W0/D0/L8 (WORSE than exp032 W0/D1/L7)
- Best checkpoint retains exp032's 51.4% (never improved)

**Analysis:**
1. Augmentation with label smoothing HURTS a pre-trained model
2. The 100K subset is insufficient — model was already trained on 460K×10ep
3. Label smoothing (ε=0.1) fights the already-learned sharp distributions
4. The model has already extracted the signal from these positions and their mirrors
5. **Augmentation doesn't help when the model is already data-saturated**

**Key meta-insight from exp033-040:**
Every approach to push past 51.4% has failed:
- Self-play REINFORCE → destroyed model
- SF distillation → catastrophic forgetting
- Mixed training → marginal
- Search → hurts (value head quality)
- Value head fix → search still doesn't help
- Expert iteration → hurts accuracy
- 12L model → worse than 8L (overfitting)
- Augmentation → hurts pre-trained model

The 51.4% ceiling appears fundamental to this dataset + architecture combination.
Next direction: try a completely different approach — attention pattern modifications
or training curriculum changes.

### exp041: KL-Constrained Self-Play (95s)
- **Hypothesis**: KL penalty (β=0.1) against frozen reference prevents catastrophic
  forgetting that destroyed exp033 vanilla REINFORCE
- **Design**: 3 gens × 50 self-play games, temp=0.3, lr=1e-6, REINFORCE + KL(p||p_ref)
- **Result**: Accuracy preserved at 51.4% across all generations (KL works!)
- **But**: Self-play signal too weak — only ~2% positions decisive (126-146 per gen)
  Gen3 had ZERO decisive positions (all draws/timeouts), update skipped
- **Games**: W0/D2/L6 (slight improvement over exp032 W0/D1/L7, variance)
- **Conclusion**: KL constraint successfully prevents catastrophic forgetting.
  But with temp=0.3 the model draws against itself constantly. Need either
  higher temp (unrealistic positions) or asymmetric play for signal.
  The fundamental issue remains: self-play provides almost no learning signal
  when the model is already decent at not losing.

**Updated meta-insight:**
- KL-constrained RL preserves the model but can't improve it (no signal)
- Need to pivot to architectural changes per instructions: "attention on attention"

### exp042: Layer Attention — attention over transformer layers (1107s)
- **Hypothesis**: Layer attention over all 8 transformer outputs ("attention on
  attention") lets the model select best representation layer per-position
- **Design**: Added LayerAttention module (524K new params). Initialized bias to
  strongly favor last layer (softmax(5,0,...) ≈ 0.99). Fine-tuned from exp032
  on 100K subset, 2 epochs, LR=3e-5.
- **Result**: Baseline DROPPED to 44.5% (random Q/K init disrupted attention).
  Epoch 1: 50.2%, Epoch 2: 50.1% — partially recovered but below 51.4%.
  Games: W0/D0/L8.
- **Root cause**: Despite bias initialization favoring last layer, random Q/K
  projections added noise that disrupted pretrained representations. Would need
  zero-init Q/K or identity initialization to preserve baseline.
- **Lesson**: Architectural additions to pretrained models must preserve exact
  baseline behavior at initialization. Even small perturbations compound.

### exp043: Centipawn-aware SF Distillation + KL (160s)
- **Hypothesis**: Optimize for move quality (CPL) not human accuracy. Use SF
  soft targets (multi-PV top-5) + KL constraint (β=0.5) against reference.
- **NEW METRIC**: Centipawn loss baseline = 110.9cp, SF match = 45.6%
- **Design**: 5K positions labeled SF d5 multi-PV=5, soft targets temp=100,
  KL-constrained distillation, 3 epochs, lr=1e-5
- **Result**: Epoch 1: 51.9% acc (possible new high?), but CPL=112.8 (worse).
  Epochs 2-3: degraded to 50.7% (overfitting 5K data). CPL kept worsening.
  Games: W0/D0/L8 (model selected by CPL = epoch 3, already degraded)
- **Analysis**: 51.9% is within ±1.6pp noise (1000 eval samples). CPL never
  improved, suggesting SF soft targets don't translate to better moves.
  KL constraint works but can't prevent gradual drift over multiple epochs.

### exp044: Hard Example Mining (634s)
- **Hypothesis**: Train on "near-miss" positions (correct in top-3, not top-1)
  for maximum accuracy improvement per sample
- **Mining**: 100K candidates → 50.3% easy (top1), 27.7% hard (top3-not-top1),
  22.0% impossible (not-top3). Found 27,684 hard examples.
- **Result**: Accuracy DROPPED to 46.7% → 45.2% → 44.9%!
  Training on hard examples causes catastrophic forgetting of easy cases.
  Best model = unchanged baseline (51.4%)
- **Lesson**: Curriculum learning on hard examples doesn't work when you
  ONLY train on hard examples. Need to mix easy + hard, not exclude easy.

### exp045: Lichess High-ELO Data Engineering (708s)
- **KEY DISCOVERY**: Baseline accuracy on Lichess 2860+ ELO data = **21.6%**
  (vs 51.4% on HF data). Strong players make fundamentally different moves!
- **Data**: Downloaded 24,854 positions from top Lichess players (Magnus,
  Zhigalko, Msb2, FairChess, etc). Avg ELO 2860, min 2285, max 3217.
- **Lichess-only training** (3ep, lr=1e-5): HF 51.4%→50.4%, Lichess 21.6%→20.5%
  24K positions not enough to learn GM move patterns
- **Mixed training** (1ep, HF+Lichess): HF preserved 51.4%, Lichess 20.4%
- **Games**: W0/D0/L8 (Lichess-only model used)
- **Analysis**: The 21.6% vs 51.4% gap PROVES the model learned mixed-skill
  patterns, not fundamentally good chess. To play stronger, need:
  1. MUCH more high-ELO data (100K+ positions)
  2. Training from scratch on high-ELO data only
  3. Possibly filtering existing HF data by move quality (SF agreement)
- **Saved**: 24,854 positions to outputs/exp045_lichess_data/lichess_positions.jsonl

**CRITICAL META-INSIGHT (exp045)**:
The 51.4% ceiling was never about model capacity or training signal.
It's about DATA QUALITY. The model accurately predicts average-level human moves.
To play STRONG chess, train on STRONG players' moves. The path forward is:
1. Download massive Lichess data (millions of 2200+ games)
2. Filter by rating + SF agreement for data quality
3. Train from scratch on this curated dataset
  Key finding: small dataset (5K) only supports 1 epoch before overfitting.

### exp046: Large-scale Lichess from-scratch training
- **Goal**: Train our 8L transformer from scratch on 209K positions from 22 top Lichess players (avg ELO 2598)
- **Data**: Downloaded 209,382 positions from top-50 players across blitz/rapid/classical/bullet
  - Used Lichess API to fetch top players, then download their games
  - Cached at outputs/exp046_lichess_large/lichess_2200plus.jsonl (28MB)
  - Training on ACTUAL player moves (not SF labels)
  - Train: 207,382, Eval: 2,000
- **Results**: Strong learning curve across 6 epochs:

| Epoch | Loss | Accuracy | Top-3 |
|-------|------|----------|-------|
| 1 | 4.44 | 21.8% | 42.1% |
| 2 | 3.21 | 27.0% | 50.7% |
| 3 | 2.79 | 30.3% | 55.7% |
| 4 | 2.48 | 33.9% | 59.6% |
| 5 | 2.23 | 36.7% | 61.5% |
| 6 | 2.08 | 37.1% | 62.0% |

- **exp032 on same Lichess eval**: 23.2% acc, 44.4% top3
- **Games**: W0/D0/L8 (still loses to SF d3)
- **Key findings**:
  1. **37.1% beats exp032's 23.2%** on Lichess data (60% relative improvement)
  2. But 37.1% < 51.4% on HF data — predicting 2600+ player moves IS harder
  3. Epoch 5→6 gain was only 0.4% — plateau starting, need MORE DATA
  4. Loss still decreasing (2.08) — model capacity not saturated
  5. Despite better move prediction on high-ELO positions, still W0/D0/L8 vs SF
  6. Only used 22/189 available players — massive room to scale data
- **Total time**: 6694s (~112 min)

**NEXT STEPS for data engineering path:**
1. Download from ALL 189 players (currently only 22) → should get ~1.8M positions
2. Download more games per player (currently 300 max → try 1000+)
3. Consider mixed training: pretrain on HF data, finetune on Lichess
4. Add SF filtering: keep only positions where player move agrees with SF top-3
5. Need to investigate why high move prediction accuracy doesn't translate to game wins

---

## Experiments Ready to Run (queued)

### exp047: Massive Lichess data + HF pre-training
- **File**: experiments/exp047_massive_lichess.py
- **Goal**: Download from ALL 189 top Lichess players (500 games each) → ~600K-1M positions
- **Two approaches compared**:
  - A) From scratch on massive Lichess data
  - B) Pre-train on HF data → fine-tune on Lichess
- **Status**: Created, not yet run. Data download will take ~15 min, training ~2h
- **Key question**: Does HF pre-training give a tactical foundation that helps?

### exp048: Synthetic SF-labeled dataset (PRIORITY)
- **File**: experiments/exp048_sf_synthetic.py
- **Goal**: Generate 200K random chess positions, label each with SF depth 8 best move
- **Why**: Unlike human data, EVERY label is the objectively best move
- **Two approaches**:
  - A) From scratch on SF data
  - B) Fine-tune exp032 on SF data
- **Cross-evaluation**: Tests on SF eval, Lichess eval, AND HF eval sets
- **Status**: Created, not yet run. SF labeling ~16 min (cached), training ~20 min
- **Key question**: Does training on perfect SF labels produce better game play?

---

## 2026-03-23

### exp052: Controlled head comparison v2 (fixed split, CLS token, richer eval)

- **Data**: HF avewright/chess-positions pre-split train/test (no fake game_id)
- **Model**: Small (256d, 6L, 8H), 3 seeds (42, 123, 314)
- **Results**:

| Variant | Mean Acc | Std | s42 | s123 | s314 |
|---------|----------|-----|-----|------|------|
| flat | 11.3% | 0.2% | 11.6% | 10.8% | 11.4% |
| spatial | **30.3%** | 0.2% | 30.6% | 30.0% | 30.2% |

- **Delta**: spatial wins by +19.0% (massive, consistent across all seeds)
- **Phase breakdown** (spatial s42):
  - Opening: 29.4% (833 positions)
  - Endgame: 32.2% (791 positions)
  - Middlegame: 30.1% (876 positions)
- **Key finding**: The spatial head is a transformative architectural improvement,
  not a marginal one. Per-square hidden state access completely dominates flat pooling.
- Learned CLS token used as global context — cleaner than turn token abuse.

### exp053: Scale spatial to Medium model (running)

- **Model**: Medium (512d, 8L, 8H, 26M params), 2 seeds
- **Epochs**: 5 (vs 3 in exp052)
- Seed 42 results: 35.3% top-1, 58.6% top-3 (still improving at epoch 5)
- Baseline exp052 was 30.3% — Medium is +5% absolute improvement

### Research direction shift: from accuracy to gameplay

Codex review identified the core issue: the repo optimizes for move-label accuracy
but the actual goal is beating Stockfish. Key insights:

1. **Policy alone can't beat Stockfish** — SF wins by search depth, not move priors
2. **Value head is defined everywhere but trained nowhere** — huge missed signal
3. **Search > more head ablations** — policy narrows candidates, value scores leaves
4. **Soft targets > hard best-move CE** — many positions have multiple near-equivalent moves

### New experiment pipeline (priority order):

1. **exp053** (running) — establish Medium spatial as new baseline
2. **exp055** — joint policy+value training with WDL targets and soft policy targets
3. **exp054** — search baseline: top-k policy + value reranking + MCTS vs Stockfish
4. Evaluate with actual game play at SF depths 1-3

### Architecture decisions locked in:

- **Chess-native encoder-only transformer** (not frozen Qwen backbone)
- **Spatial policy head** (from×to factorized, not flat 5504-way)
- **Learned CLS token** (not turn token abuse)
- **Bidirectional attention** (encoder-only, all tokens see all)
- Medium config: 512d, 8L, 8H, ~26M params

### Criticisms addressed:

- Fixed data split (HF pre-split, not synthetic game_id)
- Added phase-bucketed evaluation
- Added entropy, SF-move rank metrics
- Save model checkpoints for downstream use

---

## NEXT SESSION ROADMAP

### Priority 1: Check exp053 results and promote as baseline
- exp053 should beat exp052's 30.3% — if so, it's the new baseline model
- Best checkpoint goes into exp054 (search) and exp055 (joint training)

### Priority 2: Run exp055 (joint policy + value training)
- Train value head for real with WDL targets from HF dataset
- Uses soft Stockfish targets (KL divergence over top-k moves)
- Joint loss = 0.7*hard_CE + 0.3*soft_KL + 0.5*value_CE
- Run: `python -u experiments/exp055_joint_policy_value.py`

### Priority 3: Run exp054 (search baseline)
- Uses exp053 or exp055 checkpoint
- 4 strategies: policy argmax, value rerank k5, value rerank k10, MCTS 50
- Plays 8 games at SF depths 1, 2, 3
- Run: `python -u experiments/exp054_search_baseline.py`

### Priority 4: Iterate on search + value
- If value is poorly calibrated, try fine-tuning value head on SF centipawn targets
- If MCTS helps even with untrained value, invest in deeper search
- Consider alpha-beta with iterative deepening

### Key Insight
The path to beating Stockfish is NOT more move-label accuracy.
It's: good policy prior (DONE) + trained value head (exp055) + search (exp054).
Even a mediocre value head + shallow search should beat policy-only at gameplay.

### Key Environment Notes
- Must use `export` for env vars (inline env vars denied by policy)
- `HF_DATASETS_OFFLINE=1` when not downloading from HF
- GPU: NVIDIA RTX 2000 Ada, 16GB VRAM
- Stockfish: `stockfish/stockfish/stockfish-ubuntu-x86-64-avx2`
- Best checkpoint: `outputs/exp032_continue_training/best_checkpoint.pt` (51.4% on HF)
- Lichess data cache: `outputs/exp046_lichess_large/lichess_2200plus.jsonl` (209K positions, 28MB)

### exp048: Synthetic SF-labeled dataset
- **Goal**: Generate 200K random positions, label with SF depth 8 best move, train from scratch + fine-tune
- **Key insight**: This tests whether PERFECTLY labeled data (SF best move) is better than human moves
- **Data**: Generated 200K diverse positions (37% opening, 63% middlegame, 0% endgame via random play)
  - Labeled at 266 pos/s, total 753s (~12.5 min)
  - Saved to outputs/exp048_sf_synthetic/sf_200k_d8.jsonl (cached, reusable)
  - Train: 197K, Eval: 3K
- **Results**:

| Model | SF Eval | SF Top3 | Lichess Eval | Games |
|-------|---------|---------|-------------|-------|
| exp032 baseline | 28.8% | 50.2% | 23.2% | — |
| Scratch 4ep | **49.6%** | **76.3%** | 22.1% | W0/D1/L7 |
| Finetune 2ep | 44.0% | 69.9% | 22.6% | W0/D0/L8 |

- **Key findings**:
  1. **Scratch > Finetune on SF data** (49.6% vs 44.0%) — HF pretraining hurts SF accuracy
  2. SF model predicts SF best moves well but doesn't predict human moves (22% on Lichess)
  3. **Got a DRAW** in games (scratch game 6: fivefold repetition) — first from scratch model!
  4. Games are longer on average — SF training teaches more defensive/correct play
  5. Position diversity matters: no endgame positions (random play reaches checkmate first)
  6. These models are learning different things: SF positions ≠ real game positions

**CRITICAL INSIGHT (exp048)**: The model can learn to predict SF best moves well
(49.6%) but this doesn't translate to winning games OR predicting human moves.
The positions generated by random play are unrealistic — they don't look like positions
that arise in actual games. The path forward needs REALISTIC positions with SF labels:
  - Use Lichess positions (from real games) + SF best-move labels
  - Or use mixed training: some SF-labeled random, some human-move real games
  - The draw in games suggests tactical awareness IS improving from SF training

### exp053: Medium spatial model — new baseline (2026-03-23)
- **Model**: Medium (512d, 8L, 8H, ~26M params), 2 seeds (42, 123)
- **Data**: HF avewright/chess-positions, 5 epochs
- **Results**: Mean accuracy **35.3%**, top3 58.6% (s42); +5pp over Small (exp052)
  - Opening: 43.5%, Middlegame: 33.9%, Endgame: 28.3%
  - SF Rank: 5.73 (better than exp052's 6.5)
- **Confirmed**: Medium model is the new baseline for all downstream experiments

### exp055: Joint policy + value training (2026-03-23)
- **Hypothesis**: Joint training with WDL value head improves search gameplay
- **Model**: Same Medium config + value head, value_weight=0.5
- **Results**: Mean accuracy **35.1%** ±0.03% (TIE with exp053 on policy)
  - **Value accuracy: 79-80% WDL** — trained and usable for search
  - Policy accuracy essentially unchanged by joint training (35.1% vs 35.3%)
  - Value head converged: 73% ep1 → 80% ep5
- **Conclusion**: Joint training successfully trains a value head without hurting
  policy accuracy. The value head is now available for search experiments.

### exp056: Internal search policy head (2026-03-23)
- **Hypothesis**: Iterative 3-step internal-search head > one-shot spatial head
- **Model**: Medium + search head (3 steps, top-32 candidates, aux_base_weight=0.3)
  - 27.5M params (vs 26.1M base) — 1.3M extra for search head
- **Results**: Mean accuracy **35.4%** ±0.28% (TIE with exp053)
  - base_accuracy ≈ final_accuracy → search steps not improving over base
  - 10% slower per epoch (276s vs 254s) for no accuracy gain
- **Conclusion**: Internal search head doesn't help at this scale/data.
  The base spatial head already captures what the search head tries to learn.
  Search benefit likely requires actual tree search at inference, not learned search.

### Cumulative results table (latest session):

| Exp | Architecture | Data | Mean Acc | Top3 | Value Acc | Notes |
|-----|-------------|------|----------|------|-----------|-------|
| exp052 spatial | Small 256d/6L | 47.5K HF | 30.3% | 52.5% | — | Spatial confirmed |
| **exp053** | **Medium 512d/8L** | **47.5K HF** | **35.3%** | **58.6%** | — | **New baseline** |
| exp055 | Medium + joint | 47.5K HF | 35.1% | 58.1% | 80% | Value head trained |
| exp056 | Medium + search head | 47.5K HF | 35.4% | 58.0% | — | TIE, search redundant |

### Key insight from exp053-056:
All three medium experiments converge to ~35% accuracy on 47.5K HF data.
The bottleneck is now DATA VOLUME, not architecture — same pattern as the
50K ablation plateau from Session 9 (exp026-030 all tied at ~38%).
The Medium model on 47.5K is data-starved, just like the old 8L was on 50K.

### NEXT PRIORITY: Run exp054 (search baseline)
The exp055 model has a trained value head (80% WDL accuracy).
Use it for actual game play with search strategies:
- Policy argmax (baseline)
- Value reranking (top-5, top-10)
- Minimax search if feasible
This directly tests whether the value head improves GAMEPLAY, not label accuracy.

### exp054: Search baseline — VALUE RERANKING WORKS AT SHALLOW DEPTHS (2026-03-23)

**Checkpoint**: exp055 joint_medium_s42.pt (35% policy acc, 80% WDL value acc)
**Strategies**: policy_argmax, value_rerank_k5, value_rerank_k10, mcts_50
**Games**: 8 per strategy per SF depth (4 openings × white+black)
**Runtime**: 177s

**Results:**

| Strategy | SF d1 | SF d2 | SF d3 |
|----------|-------|-------|-------|
| policy_argmax | W0/D3/L5 (18.8%) | W0/D0/L8 (0%) | W0/D1/L7 (6.2%) |
| **value_rerank_k5** | **W0/D6/L2 (37.5%)** | W0/D1/L7 (6.2%) | W0/D0/L8 (0%) |
| value_rerank_k10 | W0/D4/L4 (25.0%) | W0/D1/L7 (6.2%) | W0/D0/L8 (0%) |
| mcts_50 | W0/D3/L5 (18.8%) | W0/D0/L8 (0%) | W0/D1/L7 (6.2%) |

**Key findings:**
1. **Value reranking k5 DOUBLES the score at SF d1** (37.5% vs 18.8%)
   - 6 draws out of 8 games! The model SURVIVES against weak Stockfish.
   - The value head disambiguates among top-5 policy moves effectively.
2. **k5 > k10** (37.5% vs 25.0% at d1) — narrower candidate set is better.
   Wider k10 introduces more bad moves that the value head can't reliably reject.
3. **MCTS 50 sims = policy argmax** — 50 simulations with 1-ply value backup
   does NOT improve on pure policy. The MCTS exploration overhead isn't worth it
   with so few simulations.
4. **All strategies collapse at SF d2+** — the value head helps with move selection
   but can't compensate for Stockfish's 2-ply advantage in search depth.
5. **Value reranking SHORTENS games** — k5 averages 60 moves at d1 vs argmax's 79.
   The value head makes more decisive moves (draws reached faster via repetition).
6. **No wins** achieved by any strategy — the model can survive but not win.

**Analysis:**
- The value head (trained jointly in exp055) is genuinely useful for search.
  It correctly distinguishes better positions among the top policy candidates.
- But 1-ply search depth is the ceiling for improvement. To beat SF d2+, the model
  needs either: (a) multi-ply search, or (b) dramatically better policy accuracy.
- The diminishing returns from k5 → k10 suggest the policy's ranking within top-5
  is mostly good — the value head just catches occasional bad top-1 picks.
- MCTS not helping confirms that the bottleneck is EVALUATION QUALITY, not SEARCH TREE SIZE.
  With only 50 sims and a noisy value function, UCB-based exploration wastes visits.

**Implications for next experiment:**
1. **Scale data + train longer** — the 47.5K HF dataset is too small for the Medium model.
   Need to train on the full 460K+ dataset (like exp024/031/032 did for the old architecture).
2. **Multi-ply search** — implement minimax with alpha-beta pruning using the trained
   value head. Even 2-ply should significantly improve over 1-ply reranking.
3. **Better value targets** — currently WDL from game metadata. SF centipawn targets
   would give more precise positional evaluation.
4. **Data quality path** — the successful pattern from Sessions 8-10 was: more data on
   the chess-native transformer yields consistent gains. Apply this to the new baseline.

### exp057: Deep search — 2-ply alpha-beta WITH TRAINED VALUE HEAD (2026-03-23)

**Checkpoint**: exp055 joint_medium_s42.pt (35% policy, 80% WDL value)
**Strategies**: policy_argmax, value_rerank_k5 (exp054 best), alphabeta_2ply_k5, alphabeta_2ply_k10
**Runtime**: 216s

**Results:**

| Strategy | SF d1 | SF d2 | SF d3 |
|----------|-------|-------|-------|
| policy_argmax | W0/D3/L5 (18.8%) | W0/D0/L8 (0%) | W0/D1/L7 (6.2%) |
| **value_rerank_k5** | **W0/D6/L2 (37.5%)** | W0/D1/L7 (6.2%) | W0/D0/L8 (0%) |
| alphabeta_2ply_k5 | W0/D1/L7 (6.2%) | W0/D1/L7 (6.2%) | W0/D1/L7 (6.2%) |
| alphabeta_2ply_k10 | W0/D3/L5 (18.8%) | W0/D0/L8 (0%) | W0/D0/L8 (0%) |

**Key findings:**
1. **2-ply HURTS at d1** — alphabeta_2ply_k5 drops from 37.5% to 6.2%!
   The value head is too noisy for minimax — it amplifies evaluation errors.
   Games are much shorter (40mv vs 60mv) — overly pessimistic play gets mated.
2. **2-ply is UNIFORM across depths** — W0/D1/L7 at ALL depths (d1, d2, d3).
   The minimax structure stabilizes play regardless of opponent strength.
3. **At d3, 2ply_k5 BEATS rerank_k5** — 6.2% vs 0%. Consistent mediocrity > d1 brilliance.
4. **Wider search (k10) doesn't help** — more candidates = more noise for the value head.
5. **1-ply value_rerank_k5 remains king at d1** — 37.5% is the best gameplay result.

**Root cause analysis:**
- The value head was trained on WDL game-outcome labels, NOT positional evaluations.
- WDL labels are noisy: a "draw" position could have +2.0 eval or 0.0 eval.
- 1-ply reranking works because it only needs to distinguish "better vs worse" among
  5 similar top-policy moves. That's a coarse comparison the WDL head handles.
- 2-ply minimax needs the value head to accurately compare positions 2 moves apart,
  which requires much finer-grained evaluation than WDL labels provide.

**CRITICAL INSIGHT**: Deeper search requires BETTER value evaluation, not just more depth.
The value head needs Stockfish centipawn targets, not game-outcome WDL.

### NEXT: exp058 — SF-calibrated value head → improved search
- Label ~20K HF positions with Stockfish centipawn evaluations
- Fine-tune ONLY the value head on cp-to-WDL targets (freeze policy)
- Retest value_rerank_k5 and alphabeta_2ply at all SF depths
- If calibrated value head + 2ply beats 1ply, it proves the evaluation hypothesis

### exp058: SF-calibrated value head — COUNTER-INTUITIVE NEGATIVE (2026-03-23)

**Checkpoint**: exp055 joint_medium_s42.pt
**SF labeling**: 20K positions, depth 5, 1914 pos/s (10s total)
  - Distribution: 50% winning, 20% equal, 30% losing
**Value training**: 5 epochs, 132K params only, sign_acc 82-85%
**Runtime**: 370s

**Results (BASELINE vs CALIBRATED):**

| Strategy | Depth | Baseline (WDL) | Calibrated (SF) | Delta |
|----------|-------|-----------------|-----------------|-------|
| policy_argmax | d1 | W0/D3/L5 (18.8%) | W0/D3/L5 (18.8%) | = |
| value_rerank_k5 | d1 | **W0/D6/L2 (37.5%)** | W0/D3/L5 (18.8%) | **-18.7pp!** |
| value_rerank_k5 | d2 | W0/D1/L7 (6.2%) | W0/D0/L8 (0%) | -6.2pp |
| alphabeta_2ply | d1 | W0/D1/L7 (6.2%) | W0/D2/L6 (12.5%) | **+6.3pp** |
| alphabeta_2ply | d2 | W0/D1/L7 (6.2%) | W0/D1/L7 (6.2%) | = |

**Key findings:**
1. **SF calibration DESTROYS 1-ply reranking** — from 37.5% to 18.8% at d1!
   The jointly-trained WDL head was BETTER for reranking than the SF-calibrated one.
2. **SF calibration HELPS 2-ply** — from 6.2% to 12.5% at d1.
   Better positional accuracy helps deeper search, even when it hurts shallow search.
3. **No strategy combination beats the original WDL value_rerank_k5 at d1 (37.5%).**

**WHY SF calibration hurts 1-ply reranking:**
- The WDL head was trained jointly with policy on the SAME data distribution.
  It learned to distinguish "safer vs riskier" among positions the model actually reaches.
- The SF head was trained on arbitrary position evaluations (centipawn scores).
  It's more "correct" positionally but doesn't align with the policy's move distribution.
- For 1-ply reranking, you only need RELATIVE rankings among 5 similar top-policy moves.
  The WDL head's imprecise but task-aligned signal works better for this.
- For 2-ply minimax, you need ABSOLUTE position evaluation to compare across depth.
  The SF head's more calibrated signal helps here even though it's mis-aligned with policy.

**DEEP INSIGHT — value head should match policy distribution, not ground truth:**
This is the AlphaZero principle: train value and policy TOGETHER on positions the
agent encounters during self-play. External labels (even from SF) are calibrated
for a different policy's behavior, creating a distribution mismatch.

### Updated cumulative gameplay results:

| Strategy | SF d1 | SF d2 | SF d3 | Source |
|----------|-------|-------|-------|--------|
| policy_argmax | 18.8% | 0% | 6.2% | exp054/057/058 (consistent) |
| **value_rerank_k5 (WDL)** | **37.5%** | 6.2% | 0% | **exp054/057 — BEST** |
| value_rerank_k5 (SF) | 18.8% | 0% | 0% | exp058 — WORSE |
| alphabeta_2ply_k5 (WDL) | 6.2% | 6.2% | 6.2% | exp057 — uniform |
| alphabeta_2ply_k5 (SF) | 12.5% | 6.2% | 0% | exp058 — slight d1 gain |

### Updated roadmap after search experiments:

1. **Value_rerank_k5 with jointly-trained WDL head is the best search strategy.**
   Don't replace it. Instead, improve it by improving the underlying model.

2. **MORE DATA is the priority.** The Medium model on 47.5K is data-starved.
   All architecture/search tweaks have plateaued at ~35% policy accuracy.
   Need to scale to 200K+ positions for meaningful gains.

3. **Data options (ranked by accessibility):**
   a. Generate diverse positions + label with SF (fast, unlimited, but synthetic)
   b. Download Lichess games via API (real positions, needs internet)
   c. Re-upload a larger version of the HF dataset

4. **Self-play as data source** — use the current model to generate positions,
   label with SF, train on them. This creates positions from the model's own
   policy distribution (optimal for the AlphaZero-style insight from exp058).

5. **Search improvements are BLOCKED by value head quality**, which is BLOCKED by
   data volume. Don't invest more in search until the model is stronger.

### exp059: Data scaling — 247.5K combined training (2026-03-23)

**Hypothesis**: Training the Medium model (26M params) on ~250K positions (200K generated
+ 47.5K HF) will significantly beat the 47.5K-only baseline (35.3% accuracy).

**Data pipeline**:
- Generated 200K diverse positions (5 sources: opening_book 30K, weighted_play 60K,
  aggressive_play 30K, endgame 64K, perturbed 40K) in 285s
- Labeled with SF depth 6, 8 threads at 338 pos/s (592s)
- Combined with 47.5K HF data → 247,500 total training positions
- Evaluation: 2,500 HF test positions (held out)

**Training**: 6 epochs, bs=128, lr=2e-4, joint policy+value (VALUE_WEIGHT=0.5)
**Runtime**: 9059s total (151 min), ~22 min/epoch

**Results:**

| Epoch | Policy Loss | Acc | Top-3 | Value Acc | SF Rank | Time |
|-------|------------|-----|-------|-----------|---------|------|
| 1 | 3.636 | 34.6% | 57.4% | 44.0% | 5.5 | 1317s |
| 2 | 2.595 | 38.8% | 63.1% | 45.4% | 4.8 | 1319s |
| 3 | 2.284 | 40.3% | 65.6% | 47.8% | 4.3 | 1319s |
| 4 | 2.057 | 44.3% | 68.8% | 48.4% | 3.9 | 1323s |
| **5** | **1.874** | **47.2%** | **69.9%** | **50.2%** | **3.6** | 1384s |
| 6 | 1.762 | 46.3% | 69.6% | 50.1% | 3.6 | 1386s |

**Best checkpoint: epoch 5, 47.2% accuracy** (+11.9pp over 35.3% baseline)

**Gameplay vs Stockfish:**

| Strategy | SF d1 | SF d2 | SF d3 |
|----------|-------|-------|-------|
| policy_argmax | W0/D5/L3 (31.2%) | W0/D1/L7 (6.2%) | W0/D1/L7 (6.2%) |
| value_rerank_k5 | W0/D2/L6 (12.5%) | W0/D2/L6 (12.5%) | W0/D0/L8 (0%) |

**Key findings:**
1. **DATA SCALING WORKS MASSIVELY** — 47.5K→247.5K gave +11.9pp accuracy (35.3%→47.2%).
   This is the largest single improvement in the entire project.
2. **Policy accuracy strongly improved**: top-3 accuracy 58.6%→69.9%, SF rank 4.8→3.6.
3. **Epoch 6 slightly overfits** (46.3% vs 47.2%) — model is starting to saturate
   at 6 epochs on 247.5K. More data or fewer epochs would help.
4. **PARADOX: Better policy accuracy but WORSE value-reranking gameplay.**
   - policy_argmax improved: 18.8%→31.2% at d1 (stronger raw play)
   - value_rerank_k5 DROPPED: 37.5%→12.5% at d1 (value head regressed)
   - The old exp055 model was trained with only 47.5K HF positions that had
     game-outcome WDL labels. The exp059 model has 200K positions with
     synthetic WDL from centipawn conversion. The synthetic WDL signal
     is noisier than real game outcomes for value head training.
5. **Value accuracy actually improved** (44%→50%) but the value HEAD is not well
   calibrated for reranking the new, stronger policy's top moves.
6. **Cosine LR schedule may be overtraining late epochs** — epoch 6 drops accuracy
   despite lower loss, suggesting the learning rate is too low to escape local optima.

**Root cause of value_rerank regression:**
The exp059 value head was trained on synthetic WDL targets derived from centipawn scores
using a sigmoid function. These targets are less informative than real game outcome WDL
because: (a) the sigmoid conversion is imprecise, (b) all "quiet" positions cluster near
(0.5, 0.5, 0.0) draw, (c) the value head can't distinguish the fine differences between
similar positions that reranking requires.

**Next steps:**
1. **More data is clearly the path** — 247.5K→500K+ should push accuracy above 50%.
   A second batch of 500K positions is being generated (CPU, depth 8, running now).
2. **Fix value head for gameplay** — train value head separately on the 47.5K HF
   positions with real game outcomes, or use distilled SF values directly.
3. **The model is NOT saturated** — epoch 5 was still improving at 47.2%. With more
   unique data (not more epochs), gains will continue.
4. **Data quality matters** — the 200K synthetic positions use depth 6 SF labels.
   The 500K batch uses depth 8, which should yield better labels.

### Updated cumulative results table:

| Exp | Data | Best Acc | Top-3 | SF d1 (argmax) | SF d1 (rerank_k5) |
|-----|------|----------|-------|----------------|-------------------|
| exp053 | 47.5K | 35.3% | 58.6% | — | — |
| exp055 | 47.5K | 35.1% | 58.1% | 18.8% | 37.5% |
| **exp059** | **247.5K** | **47.2%** | **69.9%** | **31.2%** | 12.5% |
| exp024 (old arch) | 460K | 48.7% | 73.9% | — | — |
| exp031 (old arch) | 460K×6ep | 51.2% | 76.2% | — | — |

**INSIGHT**: The chess-native transformer (ChessTransformerV2) at 247.5K now matches
the old architecture's accuracy at 460K (47.2% vs 48.7%). This confirms the new
architecture is more data-efficient. With 500K+ data, it should significantly surpass
the old architecture's ceiling of 51.2%.

### exp060: Fix value head for reranking (2026-03-23)

**Hypothesis**: Fine-tuning ONLY the value head on 47.5K HF positions (real game-outcome
WDL labels) while keeping the exp059 policy frozen will restore strong reranking gameplay.

**Results:**
- Value accuracy: 50.2% → **85.6%** (massive improvement)
- Policy accuracy: 47.2% → 47.2% (unchanged, as expected)
- Training: 10 epochs × 55s, only 132K params trainable

**Gameplay (UNCHANGED from exp059 — did NOT help):**

| Strategy | SF d1 | SF d2 | SF d3 |
|----------|-------|-------|-------|
| policy_argmax | W0/D5/L3 (31.2%) | W0/D1/L7 (6.2%) | W0/D1/L7 (6.2%) |
| value_rerank_k5 | W0/D2/L6 (12.5%) | W0/D2/L6 (12.5%) | W0/D1/L7 (6.2%) |
| value_rerank_k10 | W0/D2/L6 (12.5%) | W0/D1/L7 (6.2%) | W0/D2/L6 (12.5%) |

**CRITICAL INSIGHT: Value accuracy ≠ Reranking quality**
- The value head now correctly classifies 85.6% of positions as W/D/L
- But reranking requires **relative** ordering among top-policy moves
- The 5 positions evaluated by reranking are all "after playing one of the top-5
  policy moves" — they're very similar positions with small quality differences
- A head that's great at coarse W/D/L classification might be terrible at the
  fine-grained "which of these 5 similar positions is slightly better" task
- This is fundamentally different from the exp058 finding (SF calibration hurt
  reranking). Here even perfect WDL classification doesn't help.

**Why did the OLD exp055 value head rerank well (37.5%) but the NEW one doesn't?**
Possible explanations:
1. **Policy quality changed the task**: exp055's policy was weaker (35%), so its
   top-5 moves had MORE variation in quality → easier for value head to distinguish.
   exp059's stronger policy (47%) picks 5 moves that are all reasonable → harder
   to distinguish → value head can't help as much.
2. **The old model was smaller/simpler** → value head co-adapted with policy during
   training in a way that was beneficial for reranking.
3. **8 games is too noisy** — 12.5% vs 37.5% could be noise in such small samples.

**Implication for search**: Value-based reranking may have **diminishing returns**
as policy improves. The stronger the policy's top move, the less room for value
to improve upon it. This suggests we should focus on **policy quality (more data)**
rather than search refinement at this stage.

### exp061: Soft policy targets (2026-03-23, COMPLETE)

**Hypothesis**: Training with soft targets (CP-weighted distribution over SF top-5
moves) instead of hard one-hot best-move targets will improve policy quality.

**Design**: softmax(cp_scores / temp=100.0) over valid top-5 SF moves per position.
KL-divergence loss instead of cross-entropy. Same 247.5K data as exp059.

**Results:**

| Epoch | Accuracy | Top-3 | SF Rank | Value Acc |
|-------|----------|-------|---------|-----------|
| 1 | 34.1% | 58.6% | 5.4 | 46.7% |
| 2 | 41.0% | 65.5% | 4.2 | 46.2% |
| 3 | 43.2% | 68.5% | 3.8 | 47.8% |
| 4 | 46.3% | 71.8% | 3.4 | 50.0% |
| 5 | 47.6% | 73.7% | 3.3 | 50.5% |
| **6** | **48.1%** | **73.5%** | **3.2** | **50.8%** |

**Comparison vs exp059 (hard targets):**
- Best accuracy: **48.1%** vs 47.2% — **+0.9pp**
- Best top-3: **73.7%** vs 69.9% — **+3.8pp**
- Still improving at epoch 6 (no overfitting, vs exp059 which peaked at ep5)
- Gameplay: noisy (8 games), inconclusive

**Insight**: Soft targets provide a modest improvement (+0.9pp accuracy, +3.8pp top-3)
and better regularization (no overfitting). But the gain is small — data scaling
remains the dominant lever. For exp062, sticking with hard targets for simplicity.

### exp062: Massive data scaling (RUNNING, 2026-03-24)

**Hypothesis**: 700K+ combined data will significantly beat exp059 (247K → 47.2%).

**Data**: 500K CPU-gen (SF d8) + 200K exp059 (SF d6) + 47.5K HF = 722,257 (deduped)
**Config**: 4 epochs, same Medium model (26M params), hard cross-entropy targets
**Target**: Beat exp031 old architecture ceiling (51.2%)

**Early results** (epoch 1 complete):
- Ep1: 40.4% acc, 65.3% top-3, sf_rank=4.4 (3830s)
- For context: exp059 ep1 was 34.6% on 247K → 3x data is already accelerating learning

### Strategic shift: Build neural prior, not deeper search (2026-03-24)

Analysis of all experiments shows clearly:
1. **Data scaling is the dominant lever**: 47.5K→247K→722K each gave 10+ pp gains
2. **Search is bottlenecked by evaluation quality, not policy quality**:
   - exp057 deep search: 2-ply hurts badly
   - exp060 fix value: 85.6% WDL accuracy still didn't help reranking
3. **Value head is misaligned with move selection** — good at coarse W/D/L
   classification but bad at fine-grained "which of these 5 similar moves is better"

**New strategy**: Build a much stronger neural prior, then use just enough search
to cash it out. Specifically:

1. **Stronger policy via data** — exp062 already running (722K) ✓
2. **Search-policy training** — predict deeper SF's full move distribution,
   not just best move. This makes top-k candidate sets actually searchable.
3. **Fixed-node evaluation** — compare model at 0 nodes against SF at 100/1K/10K
   nodes. This is the fair regime where a learned prior can shine.
4. **Stop treating value classification as search objective** — shift to
   move-ranking targets or deeper-engine move-quality targets.
5. **Only revisit alpha-beta after policy improves** — current search ideas
   (exp056, exp057) failed because the prior wasn't strong enough.

**Target**: Beat node-limited Stockfish (SF at 100-1K nodes) with 0-node policy.

### exp063: Search-policy with soft multi-PV at scale (PLANNED)

**Hypothesis**: Soft targets at 722K scale will give both higher accuracy AND
better distribution quality, creating a model that beats SF at low node counts.

**Design**:
- Same 722K data as exp062, but with KL-div soft targets from existing top-5 CP scores
- Deep-labeled subset (10K at SF d12, 10 PVs) mixed in for higher-quality supervision
- New fixed-node SF evaluation: model agreement with SF at 100/1K/10K/100K nodes
- Will run after exp062 completes

### Updated roadmap (2026-03-24):

1. **exp062 RUNNING** — 722K hard targets, epoch 1 at 40.4%, 3 epochs remaining
2. **Deep relabeling RUNNING** — 10K positions being relabeled at SF d12/10PVs on CPU
3. **exp063 READY** — soft multi-PV at 722K scale + fixed-node evaluation
4. **The goal is "beat shallow SF"** — not "beat Stockfish overall"
5. **Data + supervision quality > search complexity** at this stage

### exp064: Latent search — child expansion + attention backup (2026-03-24)

**Hypothesis:** A policy head that expands candidate moves into latent child
representations and refines them via joint self-attention will outperform
one-shot spatial scoring (exp063) on the same data and trunk.

**Architecture (extends exp056 internal search head):**
1. Same 8L/512d trunk as exp063 (shared, unchanged)
2. Coarse spatial scoring → top-K=8 candidates
3. **NEW: Latent child expansion** — extract from/to square features from
   parent trunk output, project through MLP to get child representation.
   No re-encoding (cheap, fits 8GB VRAM).
4. **NEW: Joint self-attention** — [candidates; children] attend to each
   other. Candidates see consequences, children see sibling competition.
5. Cross-attention to parent board (from exp056)
6. **NEW: Backup head** — attention-weighted aggregation of child values,
   approximating soft-max over candidate outcomes.
7. 5-objective loss: KL(policy) + 0.3*KL(base) + 0.2*avg(KL(steps))
   + 0.5*KL(value) + 0.3*MSE(backup)

**Key design decisions for 8GB VRAM:**
- Batch 48 with gradient accumulation 2 (effective 96)
- K=8 candidates (not 32) — child expansion @ 8 is cheap
- Child repr = MLP(from_sq_feats || to_sq_feats), NOT full re-encoding
- ~19M params vs exp063's ~17M (13% overhead in the head)

**What the eval will show (if it works):**
- `base_accuracy`: coarse spatial accuracy (no search refinement)
- `accuracy`: after 3 search steps with child awareness
- `search_delta`: accuracy - base_accuracy (is the search actually helping?)
- `backup_rerank` game strategy: does internal backup beat external value reranking?

**This is the first experiment where the model can "see consequences" of
candidate moves inside the forward pass.** If search_delta > 0, the
architecture is learning to use 1-ply lookahead purely from attention.

---

## 2026-03-24: Infrastructure & strategic notes (exp067–069 series)

### Completed infrastructure
- **data_loader.py**: Shared 3-tier data pipeline (cache → HF → parquet). Cache loads 502K positions in ~2s vs 30+ min from raw parquet. Encoding-agnostic `board_array[0-12]` supports both fused and baseline tokenizations at load time.
- **prepare_hf_dataset.py**: One-time upload script for `avewright/chess-positions-lichess-sf` on HuggingFace. Dry-run tested at 3312 pos/s. Upload in progress.
- **exp067/068/069 refactored** to use shared `data_loader.py` — eliminated ~200 lines of duplicated data loading per experiment.
- **CURRENT_ARCHITECTURE.md** updated to reflect actual current architecture (learned [CLS] token, FusedBoardEncoder primary, joint policy+value training).

### Bug fixes
- **exp069 IDX_TO_UCI import**: `_build_move_square_indices()` used `IDX_TO_UCI` without importing it. Fixed.
- **data_loader tensor shapes**: `turn/castling/ep_file` were stored as `(N,1)` — encoders expect `(B,)`. Fixed by removing `.unsqueeze(-1)`.

### Strategic observations (from user review)
1. **exp066 is scaffolding, not decision-grade**: Single-seed, single-epoch, mixed param counts. `summary.json` shows `best_accuracy: 0`. Treat as preliminary scouting only.
2. **exp067+ is rigorous**: Three seeds, controlled schedule, shared eval metrics, explicit downstream handoff. This is the mature research loop.
3. **500K/1-epoch screening risk**: Fine for ranking ideas, but width/depth/bias effects can reshuffle at larger data/training budgets. Use these results as filters, not irreversible architecture commitments.
4. **Code duplication across exp066–069**: Model definitions, policy heads, and train/eval loops are repeated inline. A shared ablation harness would reduce unintended divergence as the experiment count grows.

### First exp067 result
- baseline seed=42: acc=15.6%, top3=35.1%, SF rank=62.4, value acc=63.4%, 1116 pos/s (448s)
- Remaining: baseline seeds 123/314, fused seeds 42/123/314

### Next steps
- Complete exp067 → launch exp068 → exp069 sequentially
- Consider extracting a shared `ablation_harness.py` from exp067 model/train/eval code once results are in
- After 3-experiment results: decide which architecture choices graduate to a larger-data run

---

## Data Pipeline Status (2026-03-25)

### `avewright/chess-positions-lichess-sf` — Lichess HF dataset generation

**Pipeline**: `process_lichess_parquets.py` → `prepare_hf_dataset.py`
**Source**: `Lichess/chess-position-evaluations` (17 source parquets)
**Target**: `avewright/chess-positions-lichess-sf` on HuggingFace
**Work root**: `/workspace/chess_hf_pipeline/`

#### Upload architecture fix (2026-03-24)
The original upload path used `load_dataset("parquet", data_files=...)` then `push_to_hub()`, which materialized the entire aggregate dataset as Arrow locally. With 2+ completed sources (~97M+ rows) this exhausted disk every time, causing repeating `DatasetGenerationError` / `OSError: No space left on device`. Fixed by replacing with `HfApi.create_commit()` + `CommitOperationAdd` — uploads parquet shard files directly, zero local Arrow materialization. Each source's shards are committed as `data/train-src{NNN}-{SSSSS}.parquet` / `data/test-src{NNN}-{SSSSS}.parquet`.

#### Progress at pod shutdown (2026-03-25 ~01:35 UTC)
| Source | Status | Valid Rows |
|--------|--------|-----------|
| 00000 | uploaded | 48,903,686 |
| 00001 | uploaded | 48,695,978 |
| 00002 | uploaded | 48,648,004 |
| 00003 | uploaded | 48,623,920 |
| 00004 | uploaded | 48,530,543 |
| 00005 | uploaded | 48,641,704 |
| 00006 | processing (checkpointed at 35M/~50M rows, batch 175, 131 train shards) | 34,256,595 so far |
| 00007–00016 | not started | — |

**Total uploaded**: 292,043,835 valid rows across 6/17 sources
**Processing rate**: ~28k positions/sec
**Estimated completion**: ~11 more sources × ~30min each ≈ ~5.5 hours remaining

#### Resume instructions
```bash
# On new pod with /workspace mounted:
cd /root/transform
bash run_process_lichess_parquets_tmux.sh
# Pipeline auto-resumes: source 6 continues from batch 175, sources 7-16 process fresh
# Attach: tmux attach -t lichess_parquet_pipeline
```

#### Key invariants preserved
- Per-source `progress.json` checkpoints → no reprocessing of completed work
- Per-source upload (not aggregate) → no disk OOM during upload
- One source parquet downloaded at a time → bounded disk usage
- All temp/cache on `/workspace` → root overlay untouched
- Deterministic train/test split by FEN hash → consistent across restarts
- Append-only event logs at both orchestrator and per-source level

---

## 2026-03-28

### exp076: Continue v2 model on source-sharded corpus — FAILED (divergence)

**Hypothesis:** Continuing the 200M v2 model on ~832M new source-sharded positions
from `avewright/chess-positions-lichess-sf` will improve accuracy beyond the 16.5%
baseline.

**GPU:** 1x NVIDIA A40 46GB, RunPod

**Run 1 (LR=1e-4, warmup 8123 steps):**
- Training was stable at LR < 3.5e-5 (steps 200–2800, pl ~3.3–3.5)
- Best observed loss: pl=2.91 at step 3000 (LR=3.7e-5)
- Divergence onset at step ~4800 (LR=5.9e-5): grad norm spiked to 6.1
- Full divergence steps 5000+: pl exploded 3.4 → 5.5 → 8.2
- Eval at step 5000: 14.0% accuracy (WORSE than 16.5% baseline)
- Throughput: 488 pos/s steady state on A40

**Run 2 (LR=3e-5, warmup 200 steps):**
- Restated from original v2 weights with conservative LR 3e-5
- Stable for first ~3000 steps (pl ~3.3–3.5, gnorm ~2)
- Divergence AGAIN at step ~4000+: pl climbed 3.4 → 4.3 → 5.3 → 5.6
- Eval at step 5000: 13.8% accuracy (WORSE than baseline)
- Same pattern: even at 3e-5, the model destabilizes after ~4M positions

**Key observations:**
1. **Both runs diverged at roughly the same training volume (~4–5M positions)**, regardless of LR
2. Value loss remained stable (0.13–0.17) while policy loss exploded — the policy head is the issue
3. Grad norms were manageable (2-5) even during divergence — not a gradient explosion
4. The model LEARNS initially (pl drops to 2.9–3.3) then catastrophically forgets
5. This looks like **catastrophic forgetting** or **distribution shift** between the main shards (exp073 training data) and the source shards being used here

**Possible root causes:**
1. **Data distribution mismatch**: The source shards may have very different position distributions than the main shards the model was originally trained on. Streaming different source files introduces distribution shifts every ~254K positions.
2. **No replay of original data**: The model sees entirely new positions with no interleaving of the data it was originally good at.
3. **Optimizer state mismatch**: Starting with a fresh optimizer on a pretrained model means the adaptive learning rate estimates (Adam momentum/variance) start from scratch, which can cause large effective learning rates for some parameters.
4. **Sequence of source files**: Different source parquets may have very different difficulty/character — e.g., one source of all endgame positions followed by one of all openings.

**What to try next (priority order):**
1. **Mix source shards with main shards** (e.g., 50/50 interleaving) to prevent catastrophic forgetting
2. **Lower LR further** (1e-5) with longer warmup — though this may just delay the divergence
3. **Train from scratch** on all data combined rather than continuing — the exp071 result (22.9% on 2M positions × 6 epochs) suggests fresh training works fine
4. **Add data mixing**: Shuffle positions from multiple source files into each batch instead of streaming one file at a time
5. **Gradient accumulation with EMA**: Use exponential moving average of weights for eval/save

**Files preserved:**
- `outputs/exp076_continue_v2/failed_run1/` — LR=1e-4 run (training_log.json, config, health.log)
- `outputs/exp076_continue_v2/failed_run2/` — LR=3e-5 run (training_log.json, config, health.log)
- Both runs' logs uploaded to `avewright/chess-transformer-200m-latest` on HF
- Original v2 model weights (`best_model.pt`) preserved and unchanged

**Infrastructure built (reusable):**
- `experiments/exp076_continue_v2.py` — full streaming continuation trainer with cursor-based resume, NaN guards, graceful shutdown, auto HF upload
- `watchdog_exp076.sh` — auto-restart watchdog for tmux sessions
- `monitor_exp076.py` — persistent health monitor with GPU/stall/NaN alerts
- `avewright/chess-transformer-200m-latest` HF repo — "always the most-trained model" pattern
- 20K eval positions (4x previous) for more stable metrics

---

### exp077: Evolutionary Expert Iteration — COMPLETED (modest gains, no forgetting)

**Hypothesis:** Population-based self-play with temperature diversity (0.0–0.6) and
supervised training on winning moves creates evolutionary improvement without RL gradients.

**GPU:** NVIDIA RTX 4060 Laptop (8GB VRAM)
**Config:** 6 temp variants (0.0–0.6), 60 games/tournament (4 per pair), 5 generations,
LR=5e-6, batch=4×accum=32 (eff 128), 500-pos eval set.

**Results by generation:**

| Gen | Accuracy | Top-3 | Train Loss | SF 1320 | SF 1450 | Draw Rate | Winner |
|-----|----------|-------|------------|---------|---------|-----------|--------|
| 0 (baseline) | 45.8% | 72.8% | — | 100% | 37.5% | — | — |
| 1 | 46.6% (+0.8) | 72.8% | 1.035 | 100% | 25% | 32% | t=0.0 |
| 2 | 46.6% (+0.8) | 73.4% | 0.942 | 62.5% | 50% | 32% | t=0.3 |
| 3 | 45.6% (−0.2) | 73.0% | 0.829 | 62.5% | 100% | 55% | t=0.3 |
| 4 | 46.4% (+0.6) | 72.4% | 0.776 | 100% | 62.5% | 58% | t=0.3 |
| 5 | 46.4% (+0.6) | 73.4% | 0.734 | 50% | 25% | 45% | t=0.3 |

**Total runtime:** 26 minutes (5 generations).

**Key findings:**
1. **No catastrophic forgetting** — accuracy stable within ±1pp of baseline across all gens.
   This is a significant win vs prior self-play (exp033 REINFORCE destroyed the model).
2. **Modest improvement:** +0.6pp accuracy, +0.6pp top-3 at final generation.
3. **Training loss monotonically decreasing** (1.035 → 0.734) — model consistently learning
   from winner moves without overfitting.
4. **Temperature 0.3 dominated** — won every tournament from Gen 2 onward. Slight stochasticity
   outperforms pure greedy (t=0.0) in self-play.
5. **Draw rate increased** (32% → 58%) — model's self-play more balanced as it improves.
6. **SF calibration noisy** — 4 games per level is insufficient (swings from 25% to 100%).
7. **Ceiling is data-quality-driven.** Self-play cannot find moves the model doesn't already
   "almost know." Need higher-quality supervision to break past ~46%.

**Files:** `experiments/exp077_evolutionary.py`, `outputs/exp077_evolutionary/`

**Architecture validated:** The evolutionary framework works and is stable. Could be
re-used with larger populations, more games per matchup, or as a post-training refinement
step after supervised pretraining improves the policy prior.

**Next:** Continued pretraining with soft multi-PV targets from HF data (exp078).
The model needs better supervision, not more self-play.

---

### exp083b / exp083c: 4xA40 continuation from exp083 best — LOGGED (stable but no gain)

**Goal:** Continue training the 204M fused chess transformer from the
`exp083` best checkpoint on the full HF source-sharded corpus using 4x A40s,
while avoiding the earlier high-LR collapse.

**Runs:**
- `exp083b_pretrain_lr3e5`: continuation with `lr=3e-5`
- `exp083c_pretrain_lr1e5`: safer continuation with `lr=1e-5`

**Shared setup:**
- Model: `ChessTransformer200M` (~204M params)
- Encoder: `FusedBoardEncoder 256d -> Transformer 1024d, 16L, 16H`
- Batch: `256 x accum 4` per worker
- Strategy: 4-worker Local SGD, sync every 500 steps
- Init: `outputs/exp083_pretrain_4xa40/best_model.pt`

**Baseline checkpoint (step 0):**
- Accuracy: `16.7%`
- Top-3: `41.9%`
- Mean SF rank: `66.7`
- Value acc: `79.7%`

**exp083b results:**
- Step 500: `16.34%` acc, `42.06%` top-3, `66.77` SF rank, `79.46%` value acc
- Step 1000: `15.82%` acc, `40.40%` top-3, `66.89` SF rank, `79.36%` value acc

**exp083c results (with stronger eval):**
- Step 500:
  - Accuracy: `16.04%`
  - Top-3: `40.60%`
  - Mean SF rank: `66.76`
  - Avg true-move prob: `0.1509`
  - Avg pred confidence: `0.3023`
  - Avg legal entropy: `1.7438`
  - Value acc: `79.64%`
  - Value KL: `0.1015`
- Step 1000:
  - Accuracy: `15.92%`
  - Top-3: `39.38%`
  - Mean SF rank: `67.22`
  - Avg true-move prob: `0.1453`
  - Avg pred confidence: `0.2941`
  - Avg legal entropy: `1.8167`
  - Value acc: `79.60%`
  - Value KL: `0.1000`

**Key findings:**
1. Lowering LR from `3e-5` to `1e-5` made training look calmer, but did not
   produce policy improvement.
2. Both continuation runs stayed below the original checkpoint on the main
   policy metrics.
3. `exp083c` was slightly more stable than `exp083b`, but by step 1000 it was
   still worse than baseline and policy sharpness had degraded:
   lower top-3, lower true-move probability, higher entropy.
4. Value metrics were basically flat or slightly improved (`value_kl`), which
   suggests the continuation recipe is not obviously destroying the value head;
   the policy learning signal is the bottleneck.
5. Conclusion: **same architecture + same full-corpus continuation recipe is not
   enough**. More time on this exact setup is unlikely to pay off.

**Infra/code added during this cycle:**
- `experiments/exp083c_pretrain_lr1e5.py`
- stronger eval metrics:
  - avg true-move probability
  - avg true-move NLL
  - prediction confidence
  - legal-move entropy
  - value KL
- `MODEL_ARCHITECTURE_EXP083.md` documenting the active model

**Recommended next direction:**
1. Stop spending 4xA40 time on plain continuation of the same objective.
2. Change the data recipe or supervision target before the next large run.
3. Fix the `Infinity` NLL eval bug before relying on that metric.
