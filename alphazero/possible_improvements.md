# AlphaZero — Modern Improvements Reference

## Overview

AlphaZero (2017) predates several architectural and algorithmic advances that would
materially improve its performance, efficiency, or generality. This document is organized
by component — architecture, search, training, and systems — with concrete implementation
notes for each improvement.

---

## 1. Architecture: Replace ResNet with a Transformer Backbone

### Problem with the original
Conv layers have a fixed receptive field. A 19-block ResNet with 3×3 kernels needs
~10 layers before information from a8 can reach h1. Long-range piece interactions
(pins, discovered attacks, pawn structure symmetry) are underrepresented in early layers.

### Modern replacement: Vision Transformer (ViT) or hybrid

```python
import torch
import torch.nn as nn

class ChessViT(nn.Module):
    """
    Treat each square as a token. 64 tokens, each with a learned embedding
    from the 119-plane input at that square.
    """
    def __init__(self, in_planes=119, d_model=256, nhead=8, num_layers=12):
        super().__init__()
        # Project each square's 119-plane stack to d_model
        self.patch_embed = nn.Linear(in_planes, d_model)
        # Learned positional encoding for 64 squares
        self.pos_embed = nn.Parameter(torch.randn(64, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024,
            dropout=0.0, batch_first=True, norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Policy and value heads attach here (same as original)

    def forward(self, x):
        # x: (B, 119, 8, 8)
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(B, 64, 119)  # (B, 64, 119)
        x = self.patch_embed(x) + self.pos_embed         # (B, 64, d_model)
        x = self.transformer(x)                          # (B, 64, d_model)
        return x  # feed into policy/value heads
```

### Hybrid option (conv stem + transformer body)
Keep a 2–3 layer conv stem for local pattern extraction (piece clusters, pawn chains),
then feed into transformer layers for global reasoning. This matches what works in
vision tasks where local features are cheap to extract convolutionally.

```python
self.stem = nn.Sequential(
    nn.Conv2d(119, 256, 3, padding=1, bias=False),
    nn.BatchNorm2d(256), nn.GELU(),
    nn.Conv2d(256, 256, 3, padding=1, bias=False),
    nn.BatchNorm2d(256), nn.GELU(),
)
# Then reshape to (B, 64, 256) and pass to transformer
```

### Evidence
Leela Chess Zero (Lc0) has experimented with attention bodies. The "BT4" and "transformer"
nets in Lc0 testing consistently outperform equivalent-FLOP ResNets on tactical puzzles
and long-range endgame positions.

---

## 2. Architecture: Decouple Policy and Value Heads

### Problem with the original
The policy head (what move to play) and value head (who is winning) share all 19 ResNet
blocks. The gradient from both losses flows through the same weights. These tasks have
partially conflicting feature requirements — tactical sharpness vs. positional evaluation.

### Modern replacement: Task-specific neck layers

```python
class DecoupledHeads(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # Shared trunk output feeds into separate necks
        self.policy_neck = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.value_neck = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.policy_head = nn.Linear(d_model, 4672)
        self.value_head  = nn.Sequential(nn.Linear(d_model * 64, 256), nn.GELU(),
                                          nn.Linear(256, 1), nn.Tanh())

    def forward(self, trunk_out):
        # trunk_out: (B, 64, d_model)
        p = self.policy_neck(trunk_out)         # per-square policy features
        v = self.value_neck(trunk_out)          # per-square value features
        policy_logits = self.policy_head(p).reshape(trunk_out.shape[0], -1)
        value = self.value_head(v.flatten(1))
        return policy_logits, value
```

### Why it helps
Gradient conflict between the two heads is a documented problem in multi-task learning.
Separate neck layers let each head learn task-specific representations without corrupting
the shared trunk. PCGrad or GradNorm can also be applied to the shared trunk gradients
if conflict is measurable.

---

## 3. Architecture: WDL Value Head (Win / Draw / Loss)

### Problem with the original
The value head outputs a single scalar in [−1, 1]. Draws are represented as 0, which
is identical to "uncertain position." The network cannot distinguish "this is a forced
draw" from "I have no idea who's winning."

### Modern replacement: 3-class softmax

```python
class WDLHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.fc1 = nn.Linear(d_model * 64, 256)
        self.fc2 = nn.Linear(256, 3)  # [win, draw, loss]

    def forward(self, x):
        x = torch.relu(self.fc1(x.flatten(1)))
        return torch.softmax(self.fc2(x), dim=-1)  # (B, 3)

# Expected value for MCTS backup:
# v = wdl[0] * 1.0 + wdl[1] * 0.0 + wdl[2] * -1.0
# Loss: cross-entropy against one-hot {z=+1: [1,0,0], z=0: [0,1,0], z=-1: [0,0,1]}
```

This is used in Stockfish NNUE, Lc0, and KataGo. It materially improves draw detection
in endgames and reduces value head overconfidence.

---

## 4. Search: MuZero-Style Learned Dynamics (Latent MCTS)

### Problem with the original
AlphaZero requires a perfect game simulator to generate next states during MCTS. This
means the system is hard-coded to know the rules of chess. It cannot generalize to
environments without a known simulator.

### Modern replacement: Learned dynamics model (MuZero)

Three networks replace the single AlphaZero network:

```
h_θ : state → latent root        (representation network)
g_θ : latent + action → latent'  (dynamics network, also outputs reward)
f_θ : latent → (policy, value)   (prediction network)
```

MCTS now runs entirely in latent space:
```
root = h_θ(board_state)
for each simulation:
    traverse tree using f_θ for policy/value at leaves
    expand using g_θ to produce next latent state
    backprop value as before
```

Training adds a reward prediction loss and requires unrolling k steps:
```
L = Σ_{t=1}^{k} [ (z_t - v_t)² + (-π_t · log p_t) + (r_t - reward_t)² ]
```

**When to use:** If you need the system to learn from pixels or any environment without
a simulator. For chess specifically, MuZero adds complexity without a clear benefit since
the rules are known.

---

## 5. Search: Adaptive Simulation Budget

### Problem with the original
800 simulations per move regardless of position complexity. A forced recapture sequence
needs 20 sims; a complex positional choice might need 2000.

### Modern replacement: Uncertainty-gated simulation budget

```python
def should_continue_search(root, min_sims=100, max_sims=1600, threshold=0.05):
    """
    Stop early if the top move's visit share is stable.
    Continue if value uncertainty is high.
    """
    visits = torch.tensor([child.N for child in root.children], dtype=torch.float)
    pi = visits / visits.sum()
    top_share = pi.max().item()

    # Value variance across children as uncertainty proxy
    q_values = torch.tensor([child.Q for child in root.children])
    value_spread = q_values.std().item()

    n_total = visits.sum().item()
    if n_total < min_sims:
        return True
    if n_total >= max_sims:
        return False
    # Continue if top move uncertain or value spread is large
    return top_share < (1 - threshold) or value_spread > 0.15
```

KataGo uses a variant of this with explicit uncertainty estimates from the value head.

---

## 6. Search: Replace UCB with a Learned Exploration Policy

### Problem with the original
`c_puct` is a fixed hand-tuned scalar (1.25). The exploration-exploitation tradeoff
is the same everywhere in the tree regardless of depth, game phase, or position type.

### Modern replacement: Gumbel MuZero / Sequential Halving

Gumbel AlphaZero (DeepMind, 2022) replaces UCB with a principled sampling scheme:

```
1. Sample k actions without replacement using Gumbel noise + log policy:
   g_a = log p(a) + Gumbel(0,1)
   Select top-k by g_a

2. Allocate the simulation budget across these k actions using
   Sequential Halving: run n/2 sims on k actions, keep top k/2, repeat.

3. Final action selected by argmax of completed Q estimates.
```

Benefits:
- No `c_puct` hyperparameter to tune
- Guaranteed to find the best action given enough budget
- Works well with very small simulation budgets (as few as 1 sim per move in some settings)

Reference: "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)

---

## 7. Training: Prioritized Experience Replay

### Problem with the original
Uniform sampling from the replay buffer. Positions where the network is already accurate
are sampled as often as positions where it is badly wrong.

### Modern replacement: Prioritized replay (PER)

```python
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity=500_000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha   # priority exponent
        self.beta  = beta    # IS correction exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        probs = self.priorities[:n] / self.priorities[:n].sum()
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return [self.buffer[i] for i in indices], weights, indices

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = (abs(err) + 1e-6) ** self.alpha
```

**td_error** for AlphaZero = `|z - v|` (value head residual). Policy loss can also
contribute to priority.

---

## 8. Training: Loss Weighting and Gradient Surgery

### Problem with the original
Equal weighting of value loss and policy loss. In early training, value loss dominates
gradients. In late training, policy loss dominates. Neither is ideal.

### Modern replacement: GradNorm or dynamic loss weighting

```python
class GradNorm:
    """
    Dynamically reweight policy and value losses so their gradient norms
    stay proportional to their initial values. Prevents one head from
    dominating the shared trunk.
    """
    def __init__(self, alpha=1.5):
        self.alpha = alpha
        self.initial_losses = None

    def weights(self, losses):
        # losses: [value_loss, policy_loss]
        losses = torch.stack(losses)
        if self.initial_losses is None:
            self.initial_losses = losses.detach()
        # Relative inverse training rate
        r = losses.detach() / self.initial_losses
        r_mean = r.mean()
        targets = r_mean ** self.alpha
        return (targets / r).detach()

# Usage:
# w = gradnorm.weights([value_loss, policy_loss])
# total_loss = w[0] * value_loss + w[1] * policy_loss
```

---

## 9. Training: Auxiliary Losses

### Problem with the original
The only training signal is the game outcome (z) and the MCTS policy (π).
Both are very sparse — z only arrives at game end, and π is only computed for the
move actually played.

### Modern auxiliary tasks

```python
# 1. Next-state value consistency (from MuZero)
# Predict the value of the position k steps ahead
# Forces the value head to be temporally consistent

# 2. Ownership map (from KataGo — Go-specific but adaptable)
# For each square, predict whether it will belong to White or Black at game end
# Provides dense per-square supervision

# 3. Move sequence prediction
# Given a position, predict the next 3 moves as a sequence
# Similar to language model next-token prediction; improves policy head calibration

# 4. Material count prediction
# Predict the current material balance from the latent representation
# Forces the value-head neck to encode basic positional facts

class AuxiliaryLosses(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.material_head = nn.Linear(d_model * 64, 1)   # predict centipawn balance
        self.phase_head    = nn.Linear(d_model * 64, 3)   # opening / middlegame / endgame

    def forward(self, trunk, material_target, phase_target):
        flat = trunk.flatten(1)
        mat_loss   = F.mse_loss(self.material_head(flat).squeeze(-1), material_target)
        phase_loss = F.cross_entropy(self.phase_head(flat), phase_target)
        return mat_loss + 0.1 * phase_loss
```

---

## 10. Training: Distributed Self-Play with Population

### Problem with the original
Single agent plays itself. Known failure mode: the policy can overfit to its own style,
developing blind spots against strategies it rarely encounters in self-play.

### Modern replacement: League training (AlphaStar-style)

```
Maintain a pool of agents:
  - Current agent (gets gradient updates)
  - Historical snapshots (checkpoints every N steps, frozen)
  - Exploiter agents (trained specifically to beat the current agent)

Matchmaking:
  - Current agent plays ~50% self-play, ~35% vs historical, ~15% vs exploiters
  - Win rate against historical pool is the primary progress metric
  - Any agent that is exploitable by the exploiter triggers a policy update
```

Simpler version: just maintain a ring buffer of 20 past checkpoints and mix 30%
of games against past selves. This alone closes many stylistic blind spots.

---

## 11. Systems: Inference Optimization for MCTS

### Problem with the original
MCTS is bottlenecked by NN inference latency — each leaf expansion requires a forward
pass, and simulations are sequential if you do not batch them.

### Modern solutions

**Batched leaf evaluation:**
```python
# Instead of evaluating one leaf at a time, accumulate K leaves
# then batch-evaluate them in a single GPU forward pass.
# K=8 to K=32 is typical; tradeoff: stale values vs. throughput.

class BatchedMCTS:
    def __init__(self, net, batch_size=16):
        self.net = net
        self.batch_size = batch_size
        self.pending_leaves = []

    def evaluate_leaves(self):
        states = torch.stack([leaf.state for leaf in self.pending_leaves])
        with torch.no_grad():
            policies, values = self.net(states)
        for leaf, p, v in zip(self.pending_leaves, policies, values):
            leaf.expand(p.cpu(), v.item())
        self.pending_leaves.clear()
```

**Quantization:**
```python
# INT8 quantization of the ResNet/ViT body reduces inference latency by ~2-3x
# on modern GPUs with negligible Elo loss (~5-10 Elo).
model_int8 = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)
```

**torch.compile:**
```python
# PyTorch 2.0+ — compiles the network graph for ~30% throughput improvement
net = torch.compile(AlphaZeroNet(), mode='reduce-overhead')
```

---

## 12. Full Modern Stack Summary

| Component | AlphaZero (2017) | Modern Replacement |
|---|---|---|
| Backbone | 19-block ResNet | Hybrid conv-stem + 12-layer Transformer |
| Value head | Single scalar, tanh | WDL 3-class softmax |
| Policy/value coupling | Fully shared trunk | Shared trunk + decoupled neck layers |
| MCTS policy | UCB with fixed c_puct | Gumbel sampling + Sequential Halving |
| Simulation budget | Fixed 800 | Adaptive, uncertainty-gated |
| Replay sampling | Uniform | Prioritized (PER) |
| Loss weighting | Fixed equal | GradNorm dynamic weighting |
| Training signal | z + π only | z + π + auxiliary (material, phase, consistency) |
| Self-play | Single agent | League (current + historical + exploiter) |
| Inference | Sequential leaf eval | Batched + INT8 + torch.compile |
| Rules dependency | Hard-coded simulator | MuZero latent dynamics (optional) |

---

## References

- AlphaZero: Silver et al., "A general reinforcement learning algorithm..." (2018)
- MuZero: Schrittwieser et al., "Mastering Atari, Go, chess and shogi..." (2020)
- Gumbel AlphaZero: Danihelka et al., "Policy improvement by planning with Gumbel" (2022)
- KataGo: Wu, "Accelerating self-play learning in Go" (2019)
- AlphaStar: Vinyals et al., "Grandmaster level in StarCraft II..." (2019)
- Lc0 architecture experiments: lczero.org/blog
- GradNorm: Chen et al., "GradNorm: Gradient normalization for adaptive loss balancing" (2018)
- Prioritized Experience Replay: Schaul et al. (2016)
```