# Roadmap to 3000+ Elo

## Current State (April 8, 2026)

| Metric | Value |
|--------|-------|
| **Best verified Elo** | ~2077 (exp100, 800 MCTS sims, pure policy eval) |
| **Best policy accuracy** | 17.76% top-1 (exp159, 5K eval set) |
| **Best model** | 204M params, 16L/1024d/16H, FusedBoardEncoder |
| **Training data** | 10.2M SF-labeled positions (1 epoch partial) |
| **Soft targets** | 647K multi-PV positions (8 PVs each, cached) |
| **Value head** | 128-bin distributional HL-Gauss (exp159+) / 3-class WDL (exp100) |
| **Hardware (local)** | RTX 4060 Laptop 8GB — ~105 pos/s for 204M |
| **Hardware (target)** | A40 48GB — expected ~800-1200 pos/s for 204M |

### Critical Bottlenecks (Ranked)

1. **Policy quality** — 17.76% top-1 is massively undertrained vs Ruoss's ~40-50% at convergence
2. **Training throughput** — RTX 4060 can't finish a full epoch in <24h
3. **Data diversity** — 79% opening positions, weak mid/endgame coverage
4. **No soft-target training** — behavioral cloning on 1-hot targets wastes 30x information
5. **No board flip augmentation** — model must learn W and B patterns separately

---

## Gap Analysis: Where Are the Missing Elo?

```
Current:        ~2077 Elo (pure policy @ 800 sims)
Target:         3000+ Elo
Gap:            ~923+ Elo

Ruoss 2024:     2895 Elo (270M, 10M games, action-value, NO search)
ChessFormer:    ~3200+ Elo (240M, 500M self-play games)
```

### Elo Budget Breakdown (Estimated Gains)

| Improvement | Expected Gain | Evidence | Status |
|-------------|--------------|----------|--------|
| **Full training (3 epochs, 10M)** | +200-400 | exp149 trend: still climbing at 47K/319K steps | NOT DONE |
| **Soft-target policy (multi-PV)** | +100-200 | Ruoss: AV >> BC (~30× more info/pos) | DATA READY |
| **Board flip augmentation** | +50-100 | ChessFormer: 2× effective data, halves state space | CODED |
| **Compact vocab (1968)** | +30-50 | Less noise in label smoothing, smaller policy head | CODED |
| **Phase-balanced sampling** | +30-60 | Ruoss: uniform >> natural sampling | CODED |
| **Attention policy head** | +20-50 | ChessFormer: more efficient from/to attention | CODED |
| **Auxiliary losses (material, phase)** | +30-50 | Czech 2023: +100 Elo from richer features | CODED |
| **SWA (Stochastic Weight Avg)** | +20-50 | ChessFormer: free post-training improvement | CODED |
| **Data scaling (50M+ positions)** | +100-200 | More diverse positions = better coverage | NEEDS COMPUTE |
| **Value head to action-value** | +100-300 | Ruoss: AV >> SV >> BC | REQUIRES REDESIGN |
| **Self-play fine-tuning** | +200-500 | AlphaZero/KataGo: self-play data corrects BC bias | FUTURE |
| **TOTAL POTENTIAL** | **+880-2010** | | |

### Realistic Target with Available Improvements: **2800-3200 Elo**

---

## Phase 1: Foundation Training (A40, ~24-48h GPU)

**Goal: 2500+ Elo from supervised training alone**

### 1A. Full From-Scratch Training with Best Architecture

```
Architecture:
  - 204M params (16L, 1024d, 16H)
  - FusedBoardEncoder (13-token, 256d)
  - Compact vocab (1968 moves)
  - 128-bin distributional value (HL-Gauss)
  - Board flip to side-to-move perspective
  - SpatialPolicyHead (project-then-gather)

Data:
  - 10.2M Stockfish-labeled positions × 3 epochs
  - Phase-balanced sampling (downsample openings, upsample mid/endgame)
  - Horizontal flip augmentation (50% random)
  - Label smoothing ε=0.05 (only over 1968 compact vocab)

Training:
  - LR=2e-4, warmup=2000 steps, cosine decay to 5e-7
  - Batch: 256 (A40 can fit), gradient accumulation as needed
  - Dropout 0.1
  - Mixed precision (bf16 on A40)  
  - Save checkpoints every 5K steps for SWA
  - Eval every 2K steps on 5K eval set

Expected: ~35-40% top-1 accuracy → ~2200-2400 Elo with MCTS
Compute: ~12-18h on A40
```

### 1B. Soft-Target Fine-Tuning (Post Phase 1A)

```
Starting from: Phase 1A best checkpoint
Data: 647K soft-target positions (multi-PV, 8 moves per position)
Loss: (1-α) * hard_CE + α * soft_CE + value_weight * HL_Gauss
  - Run α sweep: {0.3, 0.5, 0.7, 1.0} at 5K steps each
  - Promote best α, then full fine-tune for 50K steps
LR: 5e-5 (fine-tuning), cosine decay
Expected: +2-5% top-1 accuracy → +100-200 Elo
Compute: ~4-6h on A40

CRITICAL: Generate additional soft targets during Phase 1A training:
  - Run SF multi-PV (depth 12+) on 1M more positions
  - Target: 2M+ soft-target positions for fine-tuning
  - Can use CPU workers while GPU trains
```

### 1C. SWA + Post-Training (Free)

```
Average last 5-10 step checkpoints from Phase 1A/1B
Expected: +20-50 Elo for zero compute cost
```

### Phase 1 Expected Result: **2400-2600 Elo**

---

## Phase 2: Architecture Ablations (A40, ~24h GPU)

**Goal: Find the best architecture variant, then retrain**

### 2A. Attention Policy Head (Priority 1)

```
Replace SpatialPolicyHead with scaled dot-product attention:
  score(from→to) = Σ_h gate_h * (Q_from^h · K_to^h) / √d_head + promo_bias
  - 8 attention heads, 128d each
  - Global gate from CLS token
  - 1.06M params (vs 1.58M spatial → 33% reduction)

Ablation: 10K steps from scratch, compare vs spatial head
  If +1% top-1: adopt permanently
Compute: ~3h on A40

Status: CODED (exp163_attention_policy.py)
```

### 2B. Auxiliary Losses (Priority 2)

```
Add lightweight aux heads from CLS token:
  - Material balance: Huber loss (piece values from fused_ids)
  - Game phase: 3-class CE (opening/mid/end from piece count)
  - Combined aux_weight = 0.1

Ablation: 10K steps, compare trunk representational quality
  If value MAE improves AND policy doesn't regress: keep
Compute: ~3h on A40

Status: CODED (exp164_aux_losses.py)
```

### 2C. Shaw Relative Position Encoding (Priority 3)

```
Replace learned absolute position embeddings with Shaw-style relative:
  a_ij = learned bias per (from_sq, to_sq) pair added to attention scores
  - Encodes board topology (diagonals, ranks, files, knight L-shapes)
  - ChessFormer: "substantially outperforms both relative bias and absolute"

Effort: Medium — requires modifying transformer attention layers
  Need custom attention with relative bias addition
  ~400 additional lines of code
Compute: ~6h for full comparison on A40

Status: NOT YET CODED
```

### 2D. Combine Winners

```
Take all ablation winners → single architecture → retrain from scratch
Compute: ~18h on A40
Expected improvement over Phase 1: +100-200 Elo
```

### Phase 2 Expected Result: **2600-2800 Elo**

---

## Phase 3: Data Scaling + Action-Value Training (A40, ~48-72h GPU)

**Goal: Bridge the gap to 3000+ Elo through data and supervision**

### 3A. Action-Value Training (Ruoss Approach)

```
Instead of predicting JUST the best move (behavioral cloning):
  - For each position, compute Q(s,a) for ALL legal moves
  - Train: predict win% for each legal move independently
  - This provides ~30× more training signal per position

Data generation:
  - For each position, run SF multi-PV with enough PVs to cover all legal moves
  - Or: run SF at depth 10+ for each legal move individually
  - Target: 1M positions × 30 legal moves avg = 30M (state, action, value) tuples

Architecture change:
  - Separate action-value head that takes (from_sq, to_sq) and predicts win%
  - OR: use the spatial policy scores as action-value estimates directly

Compute: 
  - Data generation: ~100h CPU (parallelizable across machines)
  - Training: ~24h on A40

Expected: +200-400 Elo (Ruoss's biggest finding)
```

### 3B. Extended Dataset (50M+ Positions)

```
Current: 10.2M positions from lichess-sf
Target: 50M+ diverse positions

Sources:
  - avewright/chess-positions-lichess-sf: 832M positions (subsample 20M)
  - Generate endgame positions from Syzygy tablebases
  - Generate tactical puzzles from Lichess puzzle database
  - Generate opening-specific positions (book openings → middlegame)

Phase balance target:
  - 33% opening, 33% middlegame, 34% endgame (uniform)
  - Or learn optimal weights via Ruoss-style curriculum

Compute: ~48h on A40 for full training
```

### Phase 3 Expected Result: **2800-3100 Elo**

---

## Phase 4: Self-Play + Search (A40, ongoing)

**Goal: Break through 3000 definitively**

### 4A. Expert Iteration (Supervised → Self-Play Loop)

```
Loop:
  1. Use best model to play 100K self-play games with MCTS (800 sims)
  2. Label each position with MCTS visit distribution (soft policy) + game outcome (value)
  3. Fine-tune model on self-play data + SF data mix
  4. Repeat

Key: Self-play data corrects biases in SF-only supervision:
  - BC model never learns from its own mistakes
  - Self-play games reveal where the model is weak
  - Value signals from self-play are calibrated to MODEL strength, not SF strength

Expected: +200-500 Elo (AlphaZero's core innovation)
Compute: ~100h GPU (iterative, can checkpoint)
```

### 4B. Gumbel MCTS (Better Search)

```
Replace PUCT at root with Gumbel-top-k + Sequential Halving:
  - Near-optimal play at 1-100 simulations
  - No c_puct tuning needed
  - Guarantees policy improvement with ANY number of sims

Expected: +50-100 Elo at 100 sims (more at lower sim budgets)
Compute: Pure code change, no retraining
Status: NOT CODED (complex algorithm)
```

### 4C. NNUE Distillation (Fast Search)

```
Distill 204M transformer → <1M NNUE model:
  - NNUE runs at 60M+ pos/sec on CPU
  - Enables depth-30+ alpha-beta search
  - Use 204M as teacher: generate 100M (position, value) pairs
  - Train NNUE to match teacher evaluations

Expected: Competitive with Stockfish at equivalent search depth
Compute: ~12h for distillation on A40
```

### Phase 4 Expected Result: **3000-3200+ Elo**

---

## Compute Budget Summary (A40 Rental)

| Phase | GPU Hours | Priority | Expected Elo |
|-------|-----------|----------|-------------|
| **1A**: Full from-scratch training | 12-18h | MUST DO | 2200-2400 |
| **1B**: Soft-target fine-tuning | 4-6h | MUST DO | 2400-2600 |
| **1C**: SWA + eval | 1h | FREE | +20-50 |
| **2A-D**: Architecture ablations | 24h | HIGH | 2600-2800 |
| **3A**: Action-value training | 24-48h | HIGH | 2800-3000 |
| **3B**: Data scaling | 48h | MEDIUM | +100-200 |
| **4A**: Self-play loop | 100h+ | FUTURE | 3000+ |
| **TOTAL minimum** | **~48h** | | **~2600 Elo** |
| **TOTAL comprehensive** | **~200h** | | **~3000+ Elo** |

---

## Quick Wins Available Right Now (RTX 4060, < 60 min each)

1. **Resume exp161 training** — compact vocab + dist value from scratch is at step 10K. Even 5K more steps would give useful signal.
2. **Soft-target fine-tune exp149** — convert compact soft targets to legacy indices, run 5K steps.
3. **Architecture smoke tests** — verify attention head, aux losses work end-to-end on small data.
4. **Elo gauntlet comparisons** — compare exp100 vs exp149 vs exp159 with MCTS search.
5. **Generate more soft targets** — CPU SF labeling while GPU is idle.

---

## Key Architectural Decisions for A40

### What to Keep
- 204M params (16L, 1024d, 16H) — proven model depth amplifies through MCTS
- FusedBoardEncoder (13-token) — compact and effective
- Pre-norm transformer — stable training
- SpatialPolicyHead with project-then-gather — 11× speedup

### What to Change
- **Compact vocab (1968)** — eliminates ~3500 impossible moves
- **128-bin HL-Gauss value** — proven superior to 3-class WDL
- **Board flip** — halves effective state space
- **Phase-balanced sampling** — correct 79% opening bias
- **Label smoothing only over legal moves** — fix 98.5% waste
- **Soft-target training** — use multi-PV SF distributions

### What to Test
- **Attention policy head** — more elegant and parameter-efficient
- **Auxiliary losses** — regularize trunk with cheap supervision
- **Shaw relative encoding** — ChessFormer's key architectural advantage
- **Deeper value neck** — separate trunk features for policy vs value

---

## Research References

| Paper | Year | Key Finding | Elo | Relevance |
|-------|------|-------------|-----|-----------|
| Ruoss et al. | 2024 | 270M decoder, action-value, 128-bin HL-Gauss | 2895 | AV training, value discretization |
| Monroe et al. (ChessFormer) | 2024 | Shaw encoding, attention head, soft policy | ~3200+ | Architecture design |
| Czech et al. | 2023 | Input features + WDLP value → +180 Elo | +180 | Auxiliary targets |
| Farebrother et al. | 2024 | HL-Gauss >> regression for value | +70% | HL-Gauss validation |
| Wu (KataGo) | 2019 | Dynamic cPUCT, subtree bias correction | +75-150 | Search improvements |
| Danihelka et al. (Gumbel MuZero) | 2022 | Optimal play at low sim budgets | N/A | Low-resource search |
