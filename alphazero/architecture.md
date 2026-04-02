# AlphaZero Architecture — Implementation Reference

## Overview

AlphaZero has three components that operate in a loop: a neural network, Monte Carlo Tree Search (MCTS), and a training pipeline. The network guides search; search produces training targets; training improves the network.

---

## 1. Neural Network

### Input
- **Shape:** `(N, 119, 8, 8)` — batch × planes × rank × file
- **119 planes:**
  - 8 planes × 8 history steps = 64 planes for current player's pieces (P1: K, Q, R, B, N, pawn × 8 timesteps)
  - 8 planes × 8 history steps = 64 planes for opponent's pieces
  - 7 scalar planes broadcast to 8×8: castling rights ×4, side to move ×1, move count ×1, no-progress count ×1 (50-move rule)
- Planes are binary (0/1) except scalar planes

### Body — Residual Network
```
Input (119, 8, 8)
→ Conv2d(119, 256, kernel=3, padding=1)
→ BatchNorm2d(256)
→ ReLU
→ [ResidualBlock × 19]  ← original paper uses 19 or 39 blocks
→ trunk output: (256, 8, 8)
```

**ResidualBlock:**
```
x → Conv2d(256, 256, 3, padding=1) → BN → ReLU
  → Conv2d(256, 256, 3, padding=1) → BN
  → add(x) → ReLU
```

### Policy Head
```
trunk
→ Conv2d(256, 73, kernel=1)   ← 73 move planes (see below)
→ BatchNorm2d(73)
→ ReLU
→ Flatten → Linear(73*64, 4672)
→ output: logits over 4672 possible moves
```

**Move encoding (4672 = 73 × 64):**
- 56 planes: queen-type moves (7 distances × 8 directions)
- 8 planes: knight moves
- 9 planes: underpromotions (3 piece types × 3 directions)
- Each plane is indexed by the source square (64 squares)
- Illegal moves are masked to −∞ before softmax

### Value Head
```
trunk
→ Conv2d(256, 1, kernel=1)
→ BatchNorm2d(1)
→ ReLU
→ Flatten → Linear(64, 256) → ReLU
→ Linear(256, 1) → Tanh
→ output: scalar in [−1, 1]
  (+1 = current player wins, −1 = current player loses, 0 = draw)
```

---

## 2. MCTS

Each node stores:
```
N(s, a)   — visit count
W(s, a)   — total value (sum of backed-up V)
Q(s, a)   — mean value = W / N
P(s, a)   — prior probability from policy head
```

### Selection
Traverse from root, at each node pick action:
```
a* = argmax[ Q(s,a) + U(s,a) ]

U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

- `c_puct` ≈ 1.25 (exploration constant)
- `N(s)` = total visits to parent node
- First visit to any child: Q=0, so exploration term dominates → tries all children before revisiting

### Expansion
When a leaf node is reached:
1. Call the neural network: `p, v = NN(state)`
2. Initialize child nodes for all legal moves with `P(s,a) = p[a]`, `N=W=Q=0`
3. Return `v` to backpropagation

### Backpropagation
Walk back up the path, at each node:
```
N(s, a) += 1
W(s, a) += v
Q(s, a) = W(s, a) / N(s, a)
v = -v   ← flip sign because perspective alternates
```

### At move selection
After 800 simulations, compute policy target:
```
π(a|s) = N(s,a)^(1/τ) / Σ N(s,a')^(1/τ)

τ = 1.0   for first 30 moves (exploration)
τ → 0     after move 30 (greedy/deterministic)
```

Select move by sampling from π (training) or argmax (evaluation).

---

## 3. Training Pipeline

### Self-Play Data Generation
- Run MCTS at every position, store `(s, π, z)` tuples
- `z` is assigned retroactively after game ends: +1 / −1 / 0 from the perspective of the player to move at each position
- Buffer stores ~500k most recent positions (sliding window)

### Loss Function
```
L = (z - v)²  +  (-π · log p)  +  λ||θ||²

MSE on value head  +  cross-entropy on policy head  +  L2 regularization
```

- `λ` = 1e-4
- No weighting between value and policy terms — equal contribution
- Gradients flow through both heads into the shared ResNet body

### Optimizer
```
SGD with momentum 0.9
Learning rate schedule:
  0–200k steps:   lr = 0.2
  200k–400k:      lr = 0.02
  400k–600k:      lr = 0.002
Weight decay via L2 in loss (not optimizer)
```

### Compute setup (original paper)
- 5,000 TPUs for self-play data generation
- 16 TPUs for training
- Training ran for ~700k steps (~9 hours for chess)
- Network evaluated every 1k steps; new weights used for self-play only if win rate vs previous checkpoint > 55%

---

## 4. Key Implementation Details

**Input normalization:** Move count and no-progress count planes are normalized to [0,1] before input.

**Dirichlet noise at root:** During self-play (not evaluation), add noise to root priors to force exploration:
```
P(s,a) = (1 - ε) * p(a) + ε * η(a)
η ~ Dirichlet(α)
ε = 0.25,  α = 0.3 (chess), 0.15 (Go), 0.03 (shogi)
```

**Virtual loss:** To enable parallel MCTS across threads, increment N and decrement W immediately on selection before the NN call returns. Corrected after backprop.

**Resignation:** If the value head returns < −0.95 for the last 10 consecutive moves, the game is resigned. 10% of games are played to completion regardless to generate endgame data.

**No transposition table:** AlphaZero does not deduplicate positions in the tree. Each occurrence of a position is treated as a separate node. Simpler implementation; works because 800 sims is far too few to hit many transpositions.

---

## 5. Minimal PyTorch Skeleton
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, in_planes=119, channels=256, num_blocks=19, policy_size=4672):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 73, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(73)
        self.policy_fc   = nn.Linear(73 * 64, policy_size)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(64, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        x = self.body(self.stem(x))

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.flatten(1))  # logits, apply softmax + legal mask outside

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def alphazero_loss(pred_v, target_z, pred_p_logits, target_pi, legal_mask, lam=1e-4, params=None):
    value_loss  = F.mse_loss(pred_v.squeeze(-1), target_z)
    # Mask illegal moves before log_softmax
    masked_logits = pred_p_logits.masked_fill(~legal_mask, float('-inf'))
    log_probs   = F.log_softmax(masked_logits, dim=-1)
    policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()
    l2_loss     = sum(p.pow(2).sum() for p in params) * lam if params else 0
    return value_loss + policy_loss + l2_loss
```

---

## 6. Data Flow Summary
```
Board state (119×8×8)
    ↓
AlphaZeroNet → (policy logits [4672], value scalar)
    ↓
MCTS (800 simulations using p, v at each leaf)
    ↓
Visit counts → π [4672]  +  move selected
    ↓
Game played to completion → z ∈ {−1, 0, +1}
    ↓
Replay buffer: (state, π, z)
    ↓
Mini-batch SGD: minimize (v−z)² + (−π·log p) + λ||θ||²
    ↓
Updated weights → back to top
```