# Stockfish Architecture — Implementation Reference

## Overview

Stockfish is a classical alpha-beta search engine augmented with a learned evaluation
function (NNUE). It has no self-play loop — it uses handcrafted search heuristics
refined over decades, combined with a neural network trained on engine-generated
positions. The two components are largely independent: search drives the tree traversal,
NNUE scores the leaves.

---

## 1. Search: Alpha-Beta with Iterative Deepening

### Core algorithm

```python
def alpha_beta(pos, depth, alpha, beta, ply):
    """
    Negamax alpha-beta. Returns score from the perspective of the side to move.
    alpha: lower bound (best score current player can guarantee)
    beta:  upper bound (best score opponent will allow)
    """
    if depth <= 0:
        return quiescence_search(pos, alpha, beta, ply)

    # Transposition table lookup
    tt_entry = tt.probe(pos.hash)
    if tt_entry and tt_entry.depth >= depth:
        if tt_entry.flag == EXACT:
            return tt_entry.score
        elif tt_entry.flag == LOWER:
            alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == UPPER:
            beta = min(beta, tt_entry.score)
        if alpha >= beta:
            return tt_entry.score

    best_score = -INF
    best_move = NONE

    for move in generate_moves(pos):  # ordered — see move ordering below
        pos.make_move(move)
        score = -alpha_beta(pos, depth - 1, -beta, -alpha, ply + 1)
        pos.undo_move(move)

        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # beta cutoff — opponent won't allow this line

    # Store in transposition table
    flag = EXACT if alpha < beta else LOWER
    tt.store(pos.hash, depth, best_score, best_move, flag)
    return best_score
```

### Iterative deepening

```python
def iterative_deepening(pos, time_limit):
    best_move = NONE
    for depth in range(1, MAX_DEPTH + 1):
        score, move = aspiration_window_search(pos, depth)
        best_move = move
        if time_expired(time_limit):
            break
    return best_move
```

Iterative deepening is not wasteful — the transposition table from shallower iterations
populates move ordering for deeper iterations, making each increment fast.

### Aspiration windows

```python
def aspiration_window_search(pos, depth, prev_score=0):
    delta = 25  # centipawns
    alpha = prev_score - delta
    beta  = prev_score + delta
    while True:
        score = alpha_beta(pos, depth, alpha, beta, ply=0)
        if score <= alpha:
            alpha -= delta * 2
            delta *= 2
        elif score >= beta:
            beta += delta * 2
            delta *= 2
        else:
            return score
```

Narrow initial window causes more cutoffs. On fail-low/fail-high, widen and re-search.
Delta doubles on each failure to guarantee eventual completion.

---

## 2. Quiescence Search

Prevents the horizon effect — positions that look quiet at depth 0 but have a hanging
piece on depth 1.

```python
def quiescence_search(pos, alpha, beta, ply):
    """
    Search captures and checks until a quiet position is reached.
    """
    stand_pat = evaluate(pos)  # NNUE evaluation
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)

    for move in generate_captures(pos):  # only captures + promotions
        # Delta pruning: skip if even capturing the best possible piece
        # can't raise alpha
        if stand_pat + PIECE_VALUES[captured_piece(move)] + 200 < alpha:
            continue

        pos.make_move(move)
        score = -quiescence_search(pos, -beta, -alpha, ply + 1)
        pos.undo_move(move)

        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha
```

---

## 3. Search Pruning Heuristics

These are the core of Stockfish's speed advantage. They prune branches that are
almost certainly not the best move without fully searching them.

### Null move pruning

```python
def null_move_pruning(pos, depth, beta, ply):
    """
    If we skip our move entirely and the position is still above beta,
    our position is so good that we can prune this branch.
    Only valid in non-zugzwang positions (not pure pawn endgames).
    """
    if (depth >= 3
        and not pos.in_check()
        and pos.has_non_pawn_material()
        and evaluate(pos) >= beta):

        R = 3 + depth // 4  # reduction factor
        pos.make_null_move()
        score = -alpha_beta(pos, depth - R - 1, -beta, -beta + 1, ply + 1)
        pos.undo_null_move()

        if score >= beta:
            return beta  # null move cutoff
    return None  # no cutoff
```

### Late move reductions (LMR)

The most impactful pruning heuristic. Later moves in the ordered list are searched
at reduced depth — if they fail to beat alpha anyway, we avoid a full re-search.

```python
def lmr_reduction(depth, move_count, improving, is_pv_node):
    """
    Stockfish uses a precomputed table. This approximates the formula.
    """
    if depth < 3 or move_count < 2:
        return 0
    # Base reduction from log formula
    r = LMR_TABLE[min(depth, 63)][min(move_count, 63)]
    # Adjust
    if not improving:
        r += 1
    if is_pv_node:
        r -= 1
    return max(0, r)

# Precompute:
LMR_TABLE = [[0] * 64 for _ in range(64)]
for d in range(1, 64):
    for m in range(1, 64):
        LMR_TABLE[d][m] = int(0.75 + math.log(d) * math.log(m) / 2.25)
```

### Futility pruning

```python
def futility_pruning(depth, alpha, static_eval):
    """
    At low depths, if the static eval is far below alpha even after adding
    a large margin, the node is hopeless — skip it.
    """
    FUTILITY_MARGINS = [0, 100, 200, 300, 400]  # per depth (centipawns)
    if (depth < 5
        and not pos.in_check()
        and static_eval + FUTILITY_MARGINS[depth] <= alpha):
        return True
    return False
```

### Razoring

```python
def razor_pruning(depth, alpha, static_eval):
    """
    At depth 1, if even qsearch can't save us, skip the node.
    """
    if depth <= 1 and static_eval + 300 < alpha:
        q = quiescence_search(pos, alpha - 1, alpha, ply)
        if q < alpha:
            return True
    return False
```

### SEE (Static Exchange Evaluation) pruning

Used to avoid searching clearly losing captures.

```python
def see(pos, move):
    """
    Simulate the sequence of captures on the target square.
    Returns the material gain/loss for the moving side.
    """
    gain = [0] * 32
    d = 0
    target_sq = move_to(move)
    gain[d] = PIECE_VALUES[pos.piece_on(target_sq)]

    attackers = pos.attackers_to(target_sq)
    side = pos.side_to_move()

    while True:
        d += 1
        gain[d] = PIECE_VALUES[smallest_attacker(attackers, side)] - gain[d-1]
        if max(-gain[d-1], gain[d]) < 0:
            break
        attackers = update_attackers(attackers, target_sq)
        side ^= 1
        if not has_attacker(attackers, side):
            break

    while d > 1:
        d -= 1
        gain[d-1] = -max(-gain[d-1], gain[d])
    return gain[0]
```

---

## 4. Move Ordering

Alpha-beta efficiency depends entirely on searching the best move first. Stockfish
applies a priority stack:

```python
def order_moves(pos, moves, tt_move, ply):
    """
    Score each move; sort descending before searching.
    """
    scores = []
    for move in moves:
        score = 0

        # 1. TT move (previously found best) — search first
        if move == tt_move:
            score += 10_000_000

        # 2. Winning captures by MVV-LVA (Most Valuable Victim, Least Valuable Attacker)
        elif is_capture(move):
            victim   = PIECE_VALUES[captured_piece(move)]
            attacker = PIECE_VALUES[moving_piece(move)]
            see_val  = see(pos, move)
            if see_val >= 0:
                score += 8_000_000 + victim - attacker  # winning/equal capture
            else:
                score -= 1_000_000 + attacker           # losing capture — try last

        # 3. Killer moves (non-captures that caused beta cutoffs at this ply)
        elif move in killer_moves[ply]:
            score += 6_000_000

        # 4. Counter move (move that refuted the opponent's last move historically)
        elif move == counter_move[pos.last_move()]:
            score += 4_000_000

        # 5. History heuristic (accumulated score from past beta cutoffs)
        else:
            score += history_table[pos.side_to_move()][move_from(move)][move_to(move)]

        scores.append(score)

    return sorted(zip(scores, moves), reverse=True)
```

### History heuristic update

```python
def update_history(move, depth, side):
    # Bonus proportional to depth squared — deeper cutoffs are more valuable
    bonus = depth * depth
    # Decay old values toward zero before adding (prevents overflow)
    history_table[side][move_from(move)][move_to(move)] += bonus - \
        history_table[side][move_from(move)][move_to(move)] * abs(bonus) // 16384
```

---

## 5. Transposition Table

```python
class TTEntry:
    __slots__ = ['key', 'move', 'score', 'depth', 'flag', 'age']
    # flag: EXACT=0, LOWER=1 (lower bound / beta cutoff), UPPER=2 (upper bound / alpha)

class TranspositionTable:
    def __init__(self, size_mb=256):
        # Each entry is ~16 bytes; fill with power-of-2 number of entries
        n_entries = (size_mb * 1024 * 1024) // 16
        self.table = [None] * n_entries
        self.mask  = n_entries - 1
        self.age   = 0

    def probe(self, key):
        entry = self.table[key & self.mask]
        if entry and entry.key == key:
            return entry
        return None

    def store(self, key, depth, score, move, flag):
        idx = key & self.mask
        existing = self.table[idx]
        # Replacement scheme: prefer deeper entries from same search generation
        if (not existing
            or existing.age != self.age
            or depth >= existing.depth - 4
            or flag == EXACT):
            self.table[idx] = TTEntry(key, move, score, depth, flag, self.age)
```

**Score adjustment for mate distances:**
```python
# Mate scores must be adjusted relative to the root, not the TT storage ply
def tt_score_to_search(score, ply):
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score

def search_score_to_tt(score, ply):
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score
```

---

## 6. NNUE Evaluation

NNUE (Efficiently Updatable Neural Network) is a shallow neural network trained on
positions evaluated by the engine itself. Its key property: it can be incrementally
updated as pieces move rather than recomputed from scratch each call.

### Architecture (current HalfKAv2_hm variant)

```
Input:  ~45,000 binary features (see below)
        Mirrored: one set from White's king perspective, one from Black's
→ Linear(45056, 1024) × 2 (one per side)  — "accumulator"
→ ClippedReLU (clamp to [0, 127] in INT8)
→ Concatenate both sides: (2048,)
→ Linear(2048, 16) → ClippedReLU
→ Linear(16, 32)   → ClippedReLU
→ Linear(32, 1)    → output in centipawns (× scaling factor)
```

### Input features (HalfKAv2)

For each side (us/them), features are indexed by:
```
feature = king_square * 64 * 11 + piece_square * 11 + piece_type
```

- `king_square`: position of our king (0–63, possibly mirrored)
- `piece_square`: position of any piece on the board (0–63)
- `piece_type`: 10 types (P, N, B, R, Q for each color) + 1 padding = 11

Total: 64 × 64 × 11 = 45,056 features per perspective.
Input is a sparse binary vector — typically ~30 active features per position.

```python
def compute_active_features(pos, perspective):
    """
    Returns list of active feature indices for the given side's perspective.
    """
    features = []
    king_sq = pos.king_square(perspective)
    if perspective == BLACK:
        king_sq = mirror(king_sq)  # flip vertically

    for piece_type in range(10):
        color = WHITE if piece_type < 5 else BLACK
        ptype = piece_type % 5
        for sq in pos.pieces(color, ptype):
            if perspective == BLACK:
                sq = mirror(sq)
            feat = king_sq * 64 * 11 + sq * 11 + piece_type
            features.append(feat)
    return features
```

### Incremental accumulator update

This is NNUE's core efficiency property. Instead of recomputing the first layer
from scratch, update only the changed features:

```python
class Accumulator:
    def __init__(self, weights):  # weights: (45056, 1024)
        self.weights = weights
        self.values = [np.zeros(1024), np.zeros(1024)]  # [white_perspective, black_perspective]

    def refresh(self, pos):
        """Full recompute — done on king moves."""
        for perspective in [WHITE, BLACK]:
            self.values[perspective] = NNUE_BIAS.copy()
            for feat in compute_active_features(pos, perspective):
                self.values[perspective] += self.weights[feat]

    def update(self, pos, move, added_features, removed_features, perspective):
        """Incremental update — done on most moves."""
        acc = self.values[perspective].copy()
        for feat in added_features:
            acc += self.weights[feat]
        for feat in removed_features:
            acc -= self.weights[feat]
        self.values[perspective] = acc
```

### Forward pass (INT8 quantized, C-style)

```python
def nnue_evaluate(accumulator, side_to_move):
    """
    Simplified Python version. Real implementation uses AVX2/VNNI SIMD.
    """
    # Concatenate: [our_perspective | their_perspective]
    if side_to_move == WHITE:
        x = np.concatenate([accumulator.values[WHITE], accumulator.values[BLACK]])
    else:
        x = np.concatenate([accumulator.values[BLACK], accumulator.values[WHITE]])

    x = np.clip(x, 0, 127).astype(np.int16)  # ClippedReLU

    # Remaining layers are tiny (2048→16→32→1), run as dense matmul
    x = clipped_relu(layer1_weights @ x + layer1_bias)
    x = clipped_relu(layer2_weights @ x + layer2_bias)
    score = layer3_weights @ x + layer3_bias

    # Scale to centipawns
    return int(score * SCALE_FACTOR)
```

### NNUE training

```python
# Loss: blend of engine WDL and material truth
# Training data: self-play positions from Stockfish at depth 8
# Label: (nnue_score, game_outcome)

def nnue_loss(pred_score, target_score, game_result, lambda_=0.7):
    """
    lambda_: blend between eval target and game result
    """
    # Sigmoid-scaled targets
    def sigmoid(x, k=400): return 1 / (1 + 10 ** (-x / k))

    target_eval   = sigmoid(target_score)
    target_result = (game_result + 1) / 2  # map {-1,0,1} → {0, 0.5, 1}
    target = lambda_ * target_eval + (1 - lambda_) * target_result

    pred = sigmoid(pred_score)
    return F.binary_cross_entropy(pred, target)

# Optimizer: Adam, lr=0.001, weight decay=1e-6
# Training: ~1 billion positions, ~800 epochs
# Hardware: distributed across many CPUs (no GPU required for inference)
```

---

## 7. Time Management

```python
class TimeManager:
    def __init__(self, wtime, btime, winc, binc, movestogo):
        self.wtime = wtime
        self.btime = btime
        self.winc  = winc
        self.binc  = binc
        self.movestogo = movestogo or 50  # assume 50 moves remaining if not given

    def allocated_time(self, side):
        time  = self.wtime if side == WHITE else self.btime
        inc   = self.winc  if side == WHITE else self.binc
        # Base allocation
        base = time / self.movestogo + inc * 0.75
        # Hard limit: never use more than 80% of remaining time
        hard_limit = time * 0.8
        return min(base, hard_limit)

    def should_stop(self, elapsed, nodes, best_move_changes):
        """
        Stop early if:
        - Elapsed > soft limit and best move hasn't changed recently
        - Best move changed many times (unstable — use more time)
        """
        soft = self.allocated_time(side) * 0.6
        if elapsed > soft and best_move_changes < 3:
            return True
        return elapsed > self.allocated_time(side)
```

---

## 8. Key Data Structures

### Bitboards

```python
# Board represented as 12 bitboards (one per piece type per color)
# Each is a 64-bit integer; bit i = 1 if that piece occupies square i

class Position:
    def __init__(self):
        # 12 bitboards
        self.pieces = [[0] * 6 for _ in range(2)]  # [color][piece_type]
        self.occupied = [0, 0]   # all pieces per color
        self.all_pieces = 0      # all pieces

    def piece_on(self, sq):
        mask = 1 << sq
        for color in range(2):
            for ptype in range(6):
                if self.pieces[color][ptype] & mask:
                    return (color, ptype)
        return None

    def attackers_to(self, sq):
        """Return bitboard of all pieces attacking the given square."""
        return (
            (pawn_attacks(WHITE, sq) & self.pieces[BLACK][PAWN]) |
            (pawn_attacks(BLACK, sq) & self.pieces[WHITE][PAWN]) |
            (knight_attacks(sq) & (self.pieces[WHITE][KNIGHT] | self.pieces[BLACK][KNIGHT])) |
            (bishop_attacks(sq, self.all_pieces) &
                (self.pieces[WHITE][BISHOP] | self.pieces[BLACK][BISHOP] |
                 self.pieces[WHITE][QUEEN]  | self.pieces[BLACK][QUEEN])) |
            (rook_attacks(sq, self.all_pieces) &
                (self.pieces[WHITE][ROOK]  | self.pieces[BLACK][ROOK] |
                 self.pieces[WHITE][QUEEN] | self.pieces[BLACK][QUEEN]))
        )
```

### Magic bitboards (sliding piece attacks)

```python
# Precomputed attack tables for bishops and rooks using "magic" multipliers
# O(1) lookup for any piece on any square with any blocker configuration

ROOK_MAGIC   = [...]  # 64 magic numbers (precomputed or found by trial)
ROOK_SHIFTS  = [...]  # 64 shift amounts
ROOK_ATTACKS = [[0] * 4096 for _ in range(64)]  # [square][blocker_index]

def rook_attacks(sq, occupied):
    # Mask relevant occupancy bits, multiply by magic, shift to index
    blockers = occupied & ROOK_MASKS[sq]
    idx = (blockers * ROOK_MAGIC[sq]) >> ROOK_SHIFTS[sq]
    return ROOK_ATTACKS[sq][idx]
```

---

## 9. Full Search Stack Summary

```
iterative_deepening(pos, time_limit)
    └── aspiration_window_search(pos, depth)
            └── alpha_beta(pos, depth, alpha, beta, ply)
                    ├── TT lookup → possible early return
                    ├── Null move pruning → possible cutoff
                    ├── Razoring → possible qsearch shortcut
                    ├── Futility pruning → skip hopeless nodes
                    ├── for each move (ordered by TT/MVV-LVA/killers/history):
                    │       ├── LMR: reduce depth for late moves
                    │       ├── SEE pruning: skip losing captures
                    │       └── recursive alpha_beta call
                    └── quiescence_search(pos, alpha, beta, ply)
                            └── NNUE evaluate(pos) — leaf node scoring
```

---

## 10. Approximate Complexity / Scale

| Component | Detail |
|---|---|
| NNUE input features | ~45,056 per perspective |
| NNUE first layer | 1024 neurons per side, INT8 |
| NNUE total parameters | ~4M |
| Search depth (typical) | 20–30 ply in middlegame |
| Nodes per second | ~50–200M (single thread, modern CPU) |
| TT size (default) | 16 MB (configurable up to GBs) |
| Training positions | ~1 billion self-play positions |
| Quantization | INT8 throughout inference, SIMD (AVX2/VNNI) |
```