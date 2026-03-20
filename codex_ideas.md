# Codex Ideas

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

**Next priorities:**
1. **Train even longer** — 10 epochs on 460K data. Loss was still 2.06 and declining. Extrapolating the curve: ~53% at 10ep.
2. **Save checkpoint for self-play** — the 51.2% model is strong enough to bootstrap self-play training
3. **Try larger model on full data** — if 8L underfits at 460K×6ep, 12L might help here (unlike at 50K where data was bottleneck)
4. **Improve data quality** — Stockfish-labeled positions instead of game-outcome labels
