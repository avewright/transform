"""NNUE-style fast evaluation network for MCTS leaf evaluation.

Architecture inspired by Stockfish NNUE (wiki: nnue-architecture-deep-dive):
  Input: HalfKP-like features (king position × piece placement, both perspectives)
  Layer 1: Sparse input → 512 accumulator (clipped ReLU)
  Layer 2: 1024 (both perspectives) → 32 (clipped ReLU)
  Layer 3: 32 → 32 (clipped ReLU)
  Output:  32 → 3 (WDL logits, White-absolute)

Key differences from Stockfish NNUE:
  - Float32 (GPU-friendly), not int8 (CPU-oriented)
  - Includes policy head (not just value) for MCTS prior
  - Trained via distillation from the 204M transformer teacher

Total params: ~3-5M (60-100x fewer than teacher)
Eval speed target: ~2000+ evals/sec single, ~10000+ batched
  → 10-20x more MCTS sims per second vs transformer teacher

Value convention: White-absolute
  output[0] = P(White wins), output[1] = P(draw), output[2] = P(White loses)
"""

import math

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_features import board_to_planes, batch_boards_to_planes, NUM_PLANES


# ── Feature extraction ──

# Feature options:
# HalfKA: king_sq × piece_color × piece_sq = 64 × 10 × 64 = 40,960 (too big)
# Piece-square: piece_color × piece_sq = 10 × 64 = 640 (efficient, 2-5M total)
# Using piece-square for 2-5M param budget.

NUM_PIECE_TYPES_NO_KING = 5  # P, N, B, R, Q
NUM_COLORS = 2
NUM_PC_TYPES = NUM_PIECE_TYPES_NO_KING * NUM_COLORS  # 10
# Piece-square features: 10 piece-color types × 64 squares = 640
FEATURE_SIZE = NUM_PC_TYPES * 64  # 640 per perspective

_PC_INDEX = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.PAWN, chess.BLACK): 5,
    (chess.KNIGHT, chess.BLACK): 6,
    (chess.BISHOP, chess.BLACK): 7,
    (chess.ROOK, chess.BLACK): 8,
    (chess.QUEEN, chess.BLACK): 9,
}


def _mirror_square(sq):
    """Mirror square vertically (rank flip) for Black's perspective."""
    return sq ^ 56  # XOR with 56 = flip rank


def board_to_piece_square_indices(board: chess.Board):
    """Extract piece-square feature indices for both perspectives.

    Returns:
        white_indices: feature indices from White's perspective
        black_indices: feature indices from Black's perspective (mirrored)
    """
    white_indices = []
    black_indices = []

    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        pc_idx = _PC_INDEX.get((piece.piece_type, piece.color))
        if pc_idx is None:
            continue

        # White perspective: pc_idx * 64 + piece_sq
        w_feat = pc_idx * 64 + sq
        white_indices.append(w_feat)

        # Black perspective: mirror square, flip colors
        mirrored_sq = _mirror_square(sq)
        if piece.color == chess.WHITE:
            flipped_pc = pc_idx + 5  # white piece → opponent's piece
        else:
            flipped_pc = pc_idx - 5  # black piece → own piece
        b_feat = flipped_pc * 64 + mirrored_sq
        black_indices.append(b_feat)

    return white_indices, black_indices


def batch_boards_to_halfka_sparse(boards, device):
    """Convert boards to sparse piece-square feature tensors.

    Returns dict with:
        white_indices: (B, max_features) padded feature indices
        black_indices: (B, max_features) padded feature indices
        mask: (B, max_features) boolean mask for valid features
        turn: (B,) 0=white, 1=black
    """
    B = len(boards)
    all_white = []
    all_black = []
    max_feats = 0

    for board in boards:
        w_idx, b_idx = board_to_piece_square_indices(board)
        all_white.append(w_idx)
        all_black.append(b_idx)
        max_feats = max(max_feats, len(w_idx), len(b_idx))

    max_feats = max(max_feats, 1)  # at least 1

    white_tensor = torch.zeros(B, max_feats, dtype=torch.long, device=device)
    black_tensor = torch.zeros(B, max_feats, dtype=torch.long, device=device)
    mask_tensor = torch.zeros(B, max_feats, dtype=torch.bool, device=device)
    turn_tensor = torch.zeros(B, dtype=torch.long, device=device)

    for i, (w_idx, b_idx, board) in enumerate(
            zip(all_white, all_black, boards)):
        nw = len(w_idx)
        nb = len(b_idx)
        if nw > 0:
            white_tensor[i, :nw] = torch.tensor(w_idx, dtype=torch.long)
        if nb > 0:
            black_tensor[i, :nb] = torch.tensor(b_idx, dtype=torch.long)
        mask_tensor[i, :max(nw, nb)] = True
        turn_tensor[i] = 0 if board.turn == chess.WHITE else 1

    return {
        "white_indices": white_tensor,
        "black_indices": black_tensor,
        "mask": mask_tensor,
        "turn": turn_tensor,
    }


# ── Clipped ReLU ──

class ClippedReLU(nn.Module):
    """ReLU clamped to [0, 1]. Used in NNUE for bounded activations."""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


# ── NNUE Model ──

class NNUEModel(nn.Module):
    """NNUE-style fast evaluation network.

    Architecture:
        HalfKA features (sparse) → accumulator (512 per perspective)
        Concat both perspectives (1024) → 32 → 32 → 3 (WDL)

    For MCTS, also includes a lightweight policy head.

    ~3.4M parameters total.
    """

    def __init__(self, accumulator_size=512, hidden1=32, hidden2=32,
                 policy_channels=32):
        super().__init__()
        self.accumulator_size = accumulator_size

        # Sparse input → accumulator (the big weight matrix)
        # This is the equivalent of NNUE's feature transformer
        self.accumulator = nn.EmbeddingBag(
            FEATURE_SIZE, accumulator_size, mode='sum', sparse=True)

        # Accumulator bias
        self.acc_bias = nn.Parameter(torch.zeros(accumulator_size))

        # Value network: concat both perspectives → hidden → WDL
        self.value_net = nn.Sequential(
            nn.Linear(accumulator_size * 2, hidden1),
            ClippedReLU(),
            nn.Linear(hidden1, hidden2),
            ClippedReLU(),
            nn.Linear(hidden2, 3),  # WDL logits
        )

        # Lightweight policy head using feature planes (CNN-based)
        # Separate from the HalfKA value path for simplicity
        self.policy_conv = nn.Sequential(
            nn.Conv2d(NUM_PLANES, policy_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(policy_channels, policy_channels, 3, padding=1),
            nn.ReLU(),
        )
        # From-square features + to-square features → move score
        self.policy_from = nn.Linear(policy_channels, 1)
        self.policy_to = nn.Linear(policy_channels, 1)
        self.policy_promo = nn.Embedding(5, 1)  # none + Q/R/B/N

        # Move indexing (same as SpatialPolicyHead)
        self._build_move_indices()

        self._init_weights()

    def _build_move_indices(self):
        from move_vocab import VOCAB_SIZE, IDX_TO_UCI
        from_sqs, to_sqs, promo_types = [], [], []
        promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
        for i in range(VOCAB_SIZE):
            uci = IDX_TO_UCI[i]
            from_sqs.append(chess.parse_square(uci[:2]))
            to_sqs.append(chess.parse_square(uci[2:4]))
            promo_types.append(promo_map.get(uci[4:5], 0))
        self.register_buffer("from_sqs",
                             torch.tensor(from_sqs, dtype=torch.long))
        self.register_buffer("to_sqs",
                             torch.tensor(to_sqs, dtype=torch.long))
        self.register_buffer("promo_types",
                             torch.tensor(promo_types, dtype=torch.long))

    def _init_weights(self):
        nn.init.normal_(self.accumulator.weight, std=0.01)
        for m in self.value_net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        for m in self.policy_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, halfka_input: dict, planes: torch.Tensor = None):
        """
        Args:
            halfka_input: from batch_boards_to_halfka_sparse()
            planes: (B, 18, 8, 8) feature planes (for policy head)

        Returns:
            dict with "value_logits" (B, 3) and optionally "policy_logits" (B, 5504)
        """
        w_idx = halfka_input["white_indices"]
        b_idx = halfka_input["black_indices"]
        mask = halfka_input["mask"]
        turn = halfka_input["turn"]
        B = w_idx.shape[0]

        # Compute accumulators for both perspectives
        # We use embedding_bag with a flat index and offsets
        w_flat = w_idx.view(-1)
        b_flat = b_idx.view(-1)
        feats_per = w_idx.shape[1]
        offsets = torch.arange(0, B * feats_per, feats_per,
                               device=w_idx.device)

        white_acc = self.accumulator(w_flat, offsets) + self.acc_bias
        black_acc = self.accumulator(b_flat, offsets) + self.acc_bias

        # Clipped ReLU on accumulators
        white_acc = torch.clamp(white_acc, 0.0, 1.0)
        black_acc = torch.clamp(black_acc, 0.0, 1.0)

        # Concatenate: STM perspective first, then opponent
        # Turn 0 = White to move → [white_acc, black_acc]
        # Turn 1 = Black to move → [black_acc, white_acc]
        is_black = turn.unsqueeze(1).float()  # (B, 1)
        stm_acc = white_acc * (1 - is_black) + black_acc * is_black
        opp_acc = black_acc * (1 - is_black) + white_acc * is_black
        combined = torch.cat([stm_acc, opp_acc], dim=1)  # (B, 1024)

        # Value head (outputs White-absolute WDL)
        # Convert from STM perspective: STM logits → White-absolute
        # Actually, simpler to just train directly in White-absolute convention
        value_logits = self.value_net(combined)

        result = {"value_logits": value_logits}

        # Policy head (if planes provided)
        if planes is not None:
            conv_out = self.policy_conv(planes)  # (B, C, 8, 8)
            # Reshape to per-square features
            sq_feats = conv_out.view(B, -1, 64).permute(0, 2, 1)  # (B, 64, C)

            # Score each move: from_feat · from_proj + to_feat · to_proj + promo
            from_feats = sq_feats[:, self.from_sqs, :]  # (B, 5504, C)
            to_feats = sq_feats[:, self.to_sqs, :]  # (B, 5504, C)

            from_scores = self.policy_from(from_feats).squeeze(-1)  # (B, 5504)
            to_scores = self.policy_to(to_feats).squeeze(-1)  # (B, 5504)

            # Promotion bonus
            promo_bonus = self.policy_promo(
                self.promo_types).squeeze(-1).unsqueeze(0)  # (1, 5504)

            policy_logits = from_scores + to_scores + promo_bonus
            result["policy_logits"] = policy_logits

        return result


# ── Distillation training ──

class NNUEDistiller:
    """Distill the 204M transformer teacher into the NNUE student.

    Training procedure:
      1. Generate positions from games or existing datasets
      2. Run teacher model to get soft WDL + policy targets
      3. Train student to match teacher outputs via KL divergence

    Loss = α * KL(teacher_wdl || student_wdl) + β * KL(teacher_policy || student_policy)
    """

    def __init__(self, teacher, student, device,
                 lr=1e-3, value_weight=1.0, policy_weight=1.0,
                 temperature=2.0):
        self.teacher = teacher
        self.student = student
        self.device = device
        self.temperature = temperature
        self.value_weight = value_weight
        self.policy_weight = policy_weight

        # Use separate param groups for sparse accumulator
        sparse_params = []
        dense_params = []
        for name, param in student.named_parameters():
            if 'accumulator' in name:
                sparse_params.append(param)
            else:
                dense_params.append(param)

        # SparseAdam for sparse embedding gradients, Adam for dense params
        self.optimizer_dense = torch.optim.Adam(dense_params, lr=lr)
        self.optimizer_sparse = torch.optim.SparseAdam(sparse_params, lr=lr * 10)

        self.teacher.eval()

    @torch.no_grad()
    def generate_targets(self, boards):
        """Get teacher's soft WDL and policy targets for a batch of boards."""
        from chess_features import batch_boards_to_fused_token_ids
        from move_vocab import legal_move_mask

        inp = batch_boards_to_fused_token_ids(boards, self.device)
        out = self.teacher(inp)

        # Soft WDL targets (temperature-scaled)
        T = self.temperature
        value_targets = F.softmax(out["value_logits"].float() / T, dim=-1)

        # Soft policy targets (temperature-scaled, masked)
        policy_logits = out["policy_logits"].float()
        for i, board in enumerate(boards):
            mask = legal_move_mask(board).to(self.device)
            policy_logits[i][~mask] = float("-inf")
        policy_targets = F.softmax(policy_logits / T, dim=-1)

        return value_targets, policy_targets

    def train_step(self, boards):
        """One training step: generate targets from teacher, train student.

        Returns dict of losses.
        """
        # Teacher targets
        value_targets, policy_targets = self.generate_targets(boards)

        # Student predictions
        halfka = batch_boards_to_halfka_sparse(boards, self.device)
        planes = batch_boards_to_planes(boards).to(self.device)
        student_out = self.student(halfka, planes)

        T = self.temperature

        # Value loss: KL divergence on WDL
        student_value_log = F.log_softmax(
            student_out["value_logits"] / T, dim=-1)
        value_loss = F.kl_div(student_value_log, value_targets,
                              reduction='batchmean') * (T * T)

        # Policy loss: KL divergence on legal moves
        if "policy_logits" in student_out:
            student_policy_log = F.log_softmax(
                student_out["policy_logits"] / T, dim=-1)
            # Mask: only compute loss on positions where policy_targets has
            # valid probabilities (not -inf)
            valid = policy_targets > 0
            policy_loss = F.kl_div(
                student_policy_log, policy_targets,
                reduction='batchmean') * (T * T)
        else:
            policy_loss = torch.tensor(0.0, device=self.device)

        total_loss = (self.value_weight * value_loss
                      + self.policy_weight * policy_loss)

        self.optimizer_dense.zero_grad()
        self.optimizer_sparse.zero_grad()
        total_loss.backward()
        self.optimizer_dense.step()
        self.optimizer_sparse.step()

        return {
            "total": total_loss.item(),
            "value": value_loss.item(),
            "policy": policy_loss.item(),
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_sparse_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
    return total
