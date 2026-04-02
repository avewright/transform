# Stockfish — Modern Improvements Reference

## Overview

Stockfish is already heavily optimized — it benefits from decades of incremental
refinement. Improvements fall into two categories: (1) replacing or augmenting
classical heuristics with learned components, and (2) applying modern ML and systems
techniques to the NNUE pipeline. This document covers both, with concrete
implementation notes.

---

## 1. NNUE: Larger and Deeper Networks

### Current state
Stockfish's HalfKAv2_hm uses a very shallow post-accumulator network:
`2048 → 16 → 32 → 1`. The accumulator is large (1024 per side) but the reasoning
layers are tiny by modern standards.

### Improvement: Wider hidden layers

```python
# Current
nn.Sequential(
    nn.Linear(2048, 16),  nn.ClippedReLU(),
    nn.Linear(16,   32),  nn.ClippedReLU(),
    nn.Linear(32,    1),
)

# Improved — still fast enough for inference with INT8 + SIMD
nn.Sequential(
    nn.Linear(2048, 128), nn.ClippedReLU(),
    nn.Linear(128,   64), nn.ClippedReLU(),
    nn.Linear(64,     1),
)
```

Lc0 experiments show that widening the post-accumulator layers from 16→128 gives
~30–50 Elo with manageable inference overhead. The bottleneck is the accumulator
update, not the dense layers, so wider dense layers are relatively cheap.

### Improvement: Multiple NNUE networks (bucket system)

Stockfish already uses a "king bucket" system where different accumulator weights
are selected based on king position. Extend this to the post-accumulator layers:

```python
NUM_PHASE_BUCKETS = 8  # e.g., indexed by piece count

class BucketedNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.accumulator_weights = nn.Embedding(64, 45056 * 1024)  # per king square
        # Separate output networks per game phase bucket
        self.output_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 128), ClippedReLU(),
                nn.Linear(128, 32),   ClippedReLU(),
                nn.Linear(32, 1)
            ) for _ in range(NUM_PHASE_BUCKETS)
        ])

    def forward(self, accumulator, phase_bucket):
        return self.output_nets[phase_bucket](accumulator)
```

This lets the network specialize: opening/middlegame positions see different weights
than endgame positions, without paying the cost of a larger single network.

---

## 2. NNUE: Transformer Accumulator

### Problem with the current accumulator
The HalfKAv2 accumulator is a single linear layer: a large lookup table summed over
active features. It has no interaction between pieces — piece A's contribution to the
accumulator is identical regardless of where piece B is.

### Improvement: Attention over piece embeddings

Replace the linear accumulator with a small transformer that reasons over piece
interactions before producing the accumulator vector.

```python
class PieceAttentionAccumulator(nn.Module):
    """
    Each piece on the board becomes a token.
    Self-attention captures piece interactions.
    Output pooled to a fixed-size accumulator vector.
    """
    def __init__(self, d_piece=64, d_model=256, nhead=4, nlayers=2):
        super().__init__()
        # Embed each piece by (type, square, king_relative_square)
        self.piece_embed  = nn.Embedding(10 * 64, d_piece)   # piece_type * square
        self.king_rel_embed = nn.Embedding(64, d_piece)       # relative to king
        self.proj = nn.Linear(d_piece * 2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.Linear(d_model, 1024)  # compress to accumulator size

    def forward(self, piece_types, piece_squares, king_sq):
        # piece_types, piece_squares: (B, num_pieces)
        rel_squares = piece_squares - king_sq.unsqueeze(1)  # king-relative
        x = torch.cat([
            self.piece_embed(piece_types * 64 + piece_squares),
            self.king_rel_embed(rel_squares % 64)
        ], dim=-1)
        x = self.proj(x)                       # (B, num_pieces, d_model)
        x = self.transformer(x)                # (B, num_pieces, d_model)
        x = x.mean(dim=1)                      # pool over pieces
        return self.pool(x)                    # (B, 1024) — drop-in accumulator
```

Trade-off: loses incremental update property of the linear accumulator. Must recompute
from scratch on every node. Only viable if the network is small enough that full
recomputation is faster than the search overhead it saves via better eval accuracy.

---

## 3. NNUE: WDL Output Head

### Problem
Stockfish's NNUE outputs a single centipawn score. Draws are implicitly near zero
but indistinguishable from genuinely unclear positions.

### Improvement: 3-class WDL output

```python
class WDLHead(nn.Module):
    def __init__(self, in_features=32):
        super().__init__()
        self.fc = nn.Linear(in_features, 3)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)  # [P(win), P(draw), P(loss)]

# Expected centipawn value for search:
# cp = (wdl[0] - wdl[2]) * SCALE_FACTOR

# Loss: cross-entropy against outcome label + blend with engine eval
def wdl_loss(pred_wdl, game_result, engine_cp, lambda_=0.7):
    # Engine WDL from centipawn score
    def cp_to_wdl(cp, k=400):
        p_win  = torch.sigmoid(torch.tensor(cp / k))
        p_loss = torch.sigmoid(torch.tensor(-cp / k))
        p_draw = 1 - p_win - p_loss
        return torch.stack([p_win, p_draw, p_loss])

    target_engine = cp_to_wdl(engine_cp)
    target_result = F.one_hot(torch.tensor(game_result + 1), 3).float()
    target = lambda_ * target_engine + (1 - lambda_) * target_result
    return F.cross_entropy(pred_wdl, target)
```

Stockfish already partially implements WDL scoring in its output UCI display. Making
it the primary training target improves draw detection and reduces overconfidence in
near-equal positions.

---

## 4. NNUE: Uncertainty Estimation

### Problem
The network has no way to signal "I'm not confident in this evaluation." This matters
in novel positions, unbalanced material configurations, and long forcing sequences
where the engine's depth is the limiting factor, not the evaluation.

### Improvement: Monte Carlo Dropout uncertainty

```python
class UncertainNNUE(nn.Module):
    def __init__(self, dropout_p=0.05):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        # ... rest of NNUE layers

    def forward(self, x, training=False):
        # Enable dropout even at inference for uncertainty estimation
        if training or self.uncertainty_mode:
            x = self.dropout(x)
        return self.output_head(x)

def estimate_uncertainty(model, pos, n_samples=8):
    """Sample n_samples forward passes, return mean and variance."""
    model.uncertainty_mode = True
    scores = [model(pos) for _ in range(n_samples)]
    model.uncertainty_mode = False
    scores = torch.stack(scores)
    return scores.mean().item(), scores.var().item()
```

**Usage in search:** High uncertainty → increase search depth at that node.
Low uncertainty → allow more aggressive pruning (LMR, futility).

```python
def lmr_reduction_with_uncertainty(depth, move_count, uncertainty):
    base_r = LMR_TABLE[depth][move_count]
    # Reduce less aggressively in uncertain positions
    uncertainty_penalty = min(2, int(uncertainty / UNCERTAINTY_SCALE))
    return max(0, base_r - uncertainty_penalty)
```

---

## 5. Search: Learning Heuristic Parameters

### Problem
Stockfish's search heuristics have dozens of hand-tuned constants: `c_puct` equivalents
for LMR, futility margins per depth, null move reduction formulas, history bonus
formulas. These are tuned by SPSA (gradient-free optimization) over thousands of
games. The process is slow and finds only local optima.

### Improvement: Gradient-based tuning via differentiable search

Replace SPSA with a differentiable proxy:

```python
class LearnedLMRTable(nn.Module):
    """Learn LMR reductions as a function of (depth, move_count, features)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Softplus()  # positive output only
        )

    def forward(self, depth, move_count, improving, is_pv, history_score, see_score, threat, complexity):
        x = torch.tensor([depth, move_count, improving, is_pv,
                          history_score, see_score, threat, complexity], dtype=torch.float)
        return self.net(x)
```

Train on engine game outcomes: positions where LMR caused search to miss the best
move get negative reward; correct reductions get positive reward. This is a
contextual bandit problem — tractable with a small policy network.

---

## 6. Search: Neural Network Move Ordering

### Problem
Stockfish's move ordering (TT → MVV-LVA → killers → history) is fast but uses no
positional context. The history heuristic is a frequency count, not a pattern recognizer.

### Improvement: Lightweight policy network for move ordering

```python
class MoveOrderingNet(nn.Module):
    """
    Small network that scores moves for ordering.
    Runs BEFORE full NNUE evaluation — must be very fast.
    Input: (board features, move features) → scalar score
    """
    def __init__(self, board_dim=64, move_dim=16, hidden=64):
        super().__init__()
        # Board: compressed piece-square table (not full NNUE)
        self.board_embed = nn.Linear(board_dim, hidden)
        # Move: from_sq, to_sq, piece_type, captured_type, promotion_type, etc.
        self.move_embed  = nn.Linear(move_dim, hidden)
        self.output      = nn.Linear(hidden * 2, 1)

    def forward(self, board_feats, move_feats):
        b = F.relu(self.board_embed(board_feats))
        m = F.relu(self.move_embed(move_feats))
        return self.output(torch.cat([b, m], dim=-1))

# Training: supervised on positions where we know the best move from deep search
# Loss: pairwise ranking loss (best move should score higher than all others)
def ranking_loss(scores, best_move_idx):
    best = scores[best_move_idx]
    others = torch.cat([scores[:best_move_idx], scores[best_move_idx+1:]])
    return F.relu(1.0 - best + others).mean()
```

This is essentially what AlphaZero's policy head does, but used only for ordering —
the actual evaluation still comes from alpha-beta + NNUE. Inference must be <1µs
per position to not bottleneck search.

---

## 7. Search: Learned Contempt

### Problem
Stockfish has a contempt parameter that biases the engine toward playing for a win
vs. accepting a draw. It is set statically. The engine doesn't adapt contempt to the
opponent's strength, game situation, or time pressure.

### Improvement: Dynamic contempt from a contextual model

```python
class ContemptModel(nn.Module):
    """
    Predict optimal contempt based on game context.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Tanh()  # output in [-1, 1], scaled to centipawns
        )

    def forward(self, features):
        # features: [material_imbalance, eval, move_number, time_remaining_ratio,
        #            opponent_elo_estimate, draw_rate_for_position_type,
        #            phase, game_result_so_far]
        return self.net(features) * 50  # scale to centipawns

# Usage: add contempt to the value returned at terminal nodes
def evaluate_with_contempt(pos, contempt):
    base = nnue_evaluate(pos)
    if is_draw(pos):
        return -contempt  # drawing is bad by contempt amount
    return base
```

---

## 8. Search: MCTS Hybrid (MCTSfish-style)

### Problem
Alpha-beta is brittle to evaluation errors at the frontier — a single miscalculated
leaf can corrupt an entire branch. MCTS degrades gracefully because errors average
out across many simulations.

### Improvement: Monte Carlo Tree Search with NNUE rollouts

Use alpha-beta at shallow depth per node, and MCTS to allocate the search budget:

```python
class HybridNode:
    def __init__(self, pos):
        self.pos = pos
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}
        # Policy from a lightweight net (not full NNUE)
        self.P = move_policy_net(pos)

    def select(self):
        """UCB over children."""
        best_score = -float('inf')
        best_move = None
        for move, child in self.children.items():
            u = self.P[move] * math.sqrt(self.N) / (1 + child.N)
            score = child.Q + 1.5 * u
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def evaluate(self):
        """Run shallow alpha-beta (depth 4) + NNUE as leaf evaluator."""
        return alpha_beta(self.pos, depth=4, alpha=-INF, beta=INF, ply=0)
```

This is the architecture of Leela Chess Zero's direct competitor to Stockfish in
TCEC matches — a hybrid that combines classical search efficiency with MCTS robustness.

---

## 9. NNUE Training: Reinforcement Learning from Self-Play

### Problem
NNUE is trained on positions labeled by Stockfish itself at fixed depth. This creates
a circular dependency — the network can only be as good as the engine that generated
the labels. Systematic errors in classical evaluation (e.g., certain pawn structures)
are baked into training data.

### Improvement: Policy gradient fine-tuning on game outcomes

After supervised pretraining on engine evaluations, fine-tune with RL:

```python
def rl_finetune_step(model, optimizer, game_trajectory):
    """
    game_trajectory: list of (position, nnue_score, game_result)
    Fine-tune using REINFORCE with baseline.
    """
    baseline = sum(t[1] for t in game_trajectory) / len(game_trajectory)

    total_loss = 0
    for pos, score, result in game_trajectory:
        advantage = result - baseline
        pred = model(pos)  # NNUE score
        # Policy gradient: push score toward result when advantage is positive
        loss = -advantage * F.logsigmoid(torch.tensor(result * pred / 400.0))
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

This is how KataGo bootstrapped beyond human-level Go play — supervised pretraining
followed by RL fine-tuning on self-play outcomes.

---

## 10. Systems: Batch NNUE Inference

### Problem
Stockfish evaluates positions one at a time during search. Each evaluation is a
fast CPU operation (SIMD INT8), but GPU utilization is zero.

### Improvement: Batched GPU inference for leaf nodes

```python
class BatchedNNUEInference:
    """
    Accumulate leaf positions, batch-evaluate on GPU, return results.
    Introduces latency but dramatically increases throughput for deep searches.
    """
    def __init__(self, model, batch_size=256, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.pending = []
        self.callbacks = []

    def queue(self, accumulator, callback):
        self.pending.append(accumulator)
        self.callbacks.append(callback)
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        batch = torch.stack(self.pending).to(self.device)
        with torch.no_grad():
            scores = self.model(batch)
        for score, cb in zip(scores, self.callbacks):
            cb(score.item())
        self.pending.clear()
        self.callbacks.clear()
```

This approach is most useful in analysis mode (fixed depth, no time pressure) where
latency is less critical than throughput. In time-controlled games, the batching
latency may exceed the benefit.

---

## 11. Systems: Quantization Beyond INT8

### Current state
Stockfish uses INT8 quantization for NNUE inference with AVX2/VNNI SIMD. This is
already highly optimized.

### Improvement: INT4 / block-float quantization for the accumulator

The accumulator (45056 → 1024 per side) dominates memory bandwidth. INT4 halves it:

```python
import torch.ao.quantization as quant

# Post-training static quantization to INT4
def quantize_accumulator_int4(model, calibration_data):
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)

    # Calibrate on representative positions
    with torch.no_grad():
        for pos in calibration_data:
            model(pos)

    quant.convert(model, inplace=True)
    return model
```

Expected: ~2x memory bandwidth reduction for accumulator lookup, ~5–10% throughput
improvement. Elo cost typically <5 Elo for INT4 vs INT8 in practice.

### Improvement: Sparse accumulator updates with threshold

```python
def sparse_update(accumulator, delta_features_add, delta_features_remove, threshold=0.001):
    """
    Skip accumulator updates for features whose weight magnitude is below threshold.
    Reduces the number of vector additions per move.
    """
    significant_add = [f for f in delta_features_add
                       if abs(weights[f]).max() > threshold]
    significant_remove = [f for f in delta_features_remove
                          if abs(weights[f]).max() > threshold]
    for f in significant_add:
        accumulator += weights[f]
    for f in significant_remove:
        accumulator -= weights[f]
```

---

## 12. Search: Learned Endgame Corrections

### Problem
Stockfish's NNUE is weakest in endgames with unusual material configurations —
K+R vs K+B, K+2B vs K+N, etc. These positions are rare in training data.

### Improvement: Endgame-specific correction network

```python
class EndgameCorrection(nn.Module):
    """
    Small correction network activated only in low-piece-count positions.
    Adds a correction to the base NNUE score.
    """
    def __init__(self):
        super().__init__()
        # Triggered when total non-king pieces <= 6
        self.net = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Input: material counts, pawn structure features, king distance, tempo

    def should_activate(self, pos):
        return pos.total_non_king_pieces() <= 6

    def forward(self, endgame_features):
        return self.net(endgame_features)

# At evaluation:
def evaluate(pos):
    base = nnue_evaluate(pos)
    if endgame_correction.should_activate(pos):
        correction = endgame_correction(extract_endgame_features(pos))
        return base + correction
    return base
```

Training data: tablebases (Syzygy) provide perfect evaluations for all 7-piece
positions. Use these as ground truth for the correction network.

---

## 13. Full Modern Stack Summary

| Component | Current Stockfish | Modern Replacement |
|---|---|---|
| NNUE accumulator | Linear feature sum, INT8 | Attention over piece tokens (optional) |
| NNUE hidden layers | 2048 → 16 → 32 → 1 | 2048 → 128 → 64 → 1, phase-bucketed |
| Value output | Single centipawn scalar | WDL 3-class softmax |
| Eval confidence | None | MC Dropout uncertainty |
| Search heuristics | Hand-tuned SPSA | Gradient-based learned parameters |
| Move ordering | MVV-LVA + history table | Lightweight policy network |
| Search algorithm | Pure alpha-beta | Hybrid alpha-beta + MCTS |
| NNUE training | Supervised (engine labels) | Supervised + RL fine-tuning |
| Inference hardware | CPU SIMD (AVX2/VNNI) | Batched GPU for analysis mode |
| Quantization | INT8 | INT4 accumulator + sparse updates |
| Endgame handling | NNUE + tablebases | Tablebase-trained correction network |
| Contempt | Static parameter | Contextual learned contempt |

---

## 14. References

- Stockfish NNUE paper: Nasu, "Efficiently updatable neural-network-based evaluation functions" (2018)
- HalfKAv2 feature set: Stockfish GitHub, `src/nnue/features/`
- KataGo: Wu, "Accelerating self-play learning in Go" (2019) — RL fine-tuning approach
- Lc0 transformer experiments: lczero.org/blog, "Testing transformers"
- Gumbel search: Danihelka et al. (2022)
- SPSA tuning: Stockfish wiki, "Stockfish Tuning"
- Syzygy tablebases: github.com/syzygy1/tb
- Prioritized replay: Schaul et al. (2016)
- GradNorm: Chen et al. (2018)
- MCTSfish: Various TCEC/CCC analysis threads, computer-chess.org
```