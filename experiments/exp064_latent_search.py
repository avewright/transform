"""exp064: Latent search — child expansion + attention backup.

Hypothesis: A policy head that expands candidate moves into latent child
representations (by extracting from/to square features from the parent board)
and refines candidates via joint self-attention over [candidates; children]
will outperform the one-shot spatial head (exp063) on the same data.

This is the attention-native analogue of 1-ply search: instead of externally
pushing each move and re-evaluating with Stockfish, the model "imagines"
consequences inside the forward pass by reading what sits on the from/to
squares and attending to those representations during refinement.

Key architectural innovations over exp056/063:
  1. Latent child expansion: for each top-K candidate, extract the from-square
     and to-square features from the parent trunk output — cheap proxy for
     "what does the board look like after this move?"
  2. Joint self-attention: candidates AND their child representations attend
     to each other, so candidate A can see that candidate B leads to trouble.
  3. Backup head: attention-weighted aggregation of child values produces a
     "searched" root value — the neural analogue of minimax backup.

Memory constraints (8 GB VRAM):
  - Batch size 48 with gradient accumulation 2 (effective batch 96)
  - K=8 candidates (not 32) — keeps child expansion cheap
  - Child representations are feature vectors, NOT full re-encodings
  - Total model ~19M params (vs exp063's ~17M)

Experiment contract:
  - Hypothesis: latent child expansion > one-shot spatial head
  - Primary metric: top-1 accuracy on HF eval (2500 pos)
  - Secondary: base_accuracy (coarse, no search), top-3, SF-move rank
  - Baseline: exp063 (same data, same trunk, one-shot spatial head)
  - Training data: same as exp063 (722K soft + HF + deep-labeled)
  - Seed: 42
  - Runtime: ~3-5 hours on 8 GB GPU
"""

import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_token_ids
from chess_model import LearnedBoardEncoder
from move_vocab import (
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)

OUTPUT_DIR = Path("outputs/exp064_latent_search")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Data paths (same as exp063)
GENERATED_BATCHES = "outputs/generated_data/batch_*.jsonl"
EXP059_DATA = Path("outputs/exp059_data_scaling/generated_200k.jsonl")
DEEP_DATA = Path("outputs/deep_labeled/deep_d12_pv10.jsonl")

# Model config (same trunk as exp063)
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1
HEAD_DIM = 256

# Latent search head config
SEARCH_STEPS = 3
CANDIDATE_TOPK = 8

# Training config (adjusted for 8 GB VRAM)
EPOCHS = 4
BATCH_SIZE = 48
GRAD_ACCUM = 2  # effective batch = 96
LR = 2e-4
WARMUP_FRAC = 0.05
VALUE_WEIGHT = 0.5
BACKUP_WEIGHT = 0.3
BASE_AUX_WEIGHT = 0.3
STEP_AUX_WEIGHT = 0.2
SOFT_TEMP = 100.0
SEED = 42

# Game config
SF_GAME_DEPTHS = [1, 2, 3]
GAMES_PER_DEPTH = 8
OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4"],
    ["e2e4", "e7e6"],
]


# ── Model ──

def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    for i in range(VOCAB_SIZE):
        uci = IDX_TO_UCI[i]
        from_sqs.append(chess.parse_square(uci[:2]))
        to_sqs.append(chess.parse_square(uci[2:4]))
        promo_types.append(promo_map.get(uci[4:5], 0))
    return (
        torch.tensor(from_sqs, dtype=torch.long),
        torch.tensor(to_sqs, dtype=torch.long),
        torch.tensor(promo_types, dtype=torch.long),
    )


class LatentSearchPolicyHead(nn.Module):
    """Spatial prior + latent child expansion + iterative refinement + backup.

    Stage 1 — Coarse spatial scoring:
      Score all 5504 moves with factorized from × to × promo.

    Stage 2 — Latent child expansion:
      For each top-K candidate, extract the from-square and to-square hidden
      states from the parent trunk output. A small MLP produces a "child
      representation" — what the board looks like after this move, without
      running the full encoder again.

    Stage 3 — Iterative refinement (×search_steps):
      - Cross-attention: candidates attend to parent board tokens
      - Joint self-attention: [candidates; children] attend to each other
        (candidates see consequences, children see sibling competition)
      - MLP refinement on candidate states
      - Delta score per step

    Stage 4 — Backup head:
      Soft-max weighted aggregation of child values → backed-up root value.
      Differentiable approximation of "value of best candidate."
    """

    def __init__(
        self,
        hidden_size,
        n_ctx_tokens=4,
        head_dim=256,
        candidate_topk=8,
        search_steps=3,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.head_dim = head_dim
        self.candidate_topk = candidate_topk
        self.search_steps = search_steps

        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)

        # ── Stage 1: Coarse spatial prior ──
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.base_score = nn.Linear(head_dim, 1)

        # ── Stage 2: Child expansion ──
        # from_sq features (H) + to_sq features (H) → child representation (D)
        self.child_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim),
        )

        # ── Stage 3: Iterative refinement ──
        self.board_k = nn.Linear(hidden_size, head_dim)
        self.board_v = nn.Linear(hidden_size, head_dim)

        # Cross-attention: candidates → parent board tokens
        self.cross_attn = nn.MultiheadAttention(
            head_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        # Joint self-attention: [candidates; children] → [candidates; children]
        self.joint_self_attn = nn.MultiheadAttention(
            head_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.refine_mlp = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim * 4),
            nn.GELU(),
            nn.Linear(head_dim * 4, head_dim),
        )
        self.delta_score = nn.Linear(head_dim, 1)
        self.step_gate = nn.Parameter(torch.zeros(search_steps))

        # ── Stage 4: Backup head ──
        # Predicts scalar value per child from (candidate, child) pair
        self.child_value_head = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        self.backup_temp = nn.Parameter(torch.tensor(1.0))

    def _base_components(self, hidden_states, cls_hidden):
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(cls_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types).unsqueeze(0)
        move_states = from_proj * to_proj + global_proj + promo_feats
        base_logits = self.base_score(F.relu(move_states)).squeeze(-1)
        return move_states, base_logits

    def forward(self, hidden_states, cls_hidden):
        B = hidden_states.shape[0]
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]  # (B, 64, H)
        board_ctx = hidden_states  # full sequence for cross-attention

        # Stage 1: Coarse scoring over all 5504 moves
        move_states, base_logits = self._base_components(hidden_states, cls_hidden)

        # Select top-K candidates
        topk = min(self.candidate_topk, base_logits.shape[-1])
        cand_idx = base_logits.topk(topk, dim=-1).indices  # (B, K)
        gather_idx = cand_idx.unsqueeze(-1).expand(-1, -1, self.head_dim)
        cand_states = torch.gather(move_states, 1, gather_idx)  # (B, K, D)

        # Stage 2: Latent child expansion
        cand_from_sq = self.from_sqs[cand_idx]  # (B, K)
        cand_to_sq = self.to_sqs[cand_idx]  # (B, K)

        # Extract from/to square features from parent trunk
        batch_idx = torch.arange(B, device=hidden_states.device).unsqueeze(1)
        from_sq_feats = sq_hidden[batch_idx, cand_from_sq]  # (B, K, H)
        to_sq_feats = sq_hidden[batch_idx, cand_to_sq]  # (B, K, H)

        # Child representation: "what happens after this move?"
        child_repr = self.child_proj(
            torch.cat([from_sq_feats, to_sq_feats], dim=-1)
        )  # (B, K, D)

        # Stage 3: Iterative refinement
        board_kv = (self.board_k(board_ctx), self.board_v(board_ctx))
        step_logits = []

        for step in range(self.search_steps):
            # Cross-attention: candidates attend to parent board tokens
            cross_out, _ = self.cross_attn(
                cand_states, board_kv[0], board_kv[1], need_weights=False,
            )
            cand_states = cand_states + cross_out

            # Joint self-attention: [candidates; children]
            combined = torch.cat([cand_states, child_repr], dim=1)  # (B, 2K, D)
            combined_out, _ = self.joint_self_attn(
                combined, combined, combined, need_weights=False,
            )
            combined = combined + combined_out
            cand_states = combined[:, :topk]
            child_repr = combined[:, topk:]

            # MLP refinement on candidate states
            cand_states = cand_states + self.refine_mlp(cand_states)

            # Delta score for this step
            delta = self.delta_score(F.gelu(cand_states)).squeeze(-1)
            gated_delta = torch.tanh(self.step_gate[step]) * delta
            step_logits.append(
                base_logits.scatter_add(1, cand_idx, gated_delta)
            )

        # Stage 4: Backup value
        child_val = self.child_value_head(
            torch.cat([cand_states, child_repr], dim=-1)
        ).squeeze(-1)  # (B, K)

        temp = self.backup_temp.abs() + 0.1
        backup_weights = F.softmax(child_val / temp, dim=-1)
        backed_up_value = torch.tanh(
            (backup_weights * child_val).sum(dim=-1)
        )  # (B,)

        final_logits = step_logits[-1] if step_logits else base_logits

        return {
            "policy_logits": final_logits,
            "base_policy_logits": base_logits,
            "candidate_indices": cand_idx,
            "step_policy_logits": step_logits,
            "child_values": child_val,
            "backed_up_value": backed_up_value,
        }


class ChessTransformerLatentSearch(nn.Module):
    """Chess-native transformer with latent search policy head.

    Same trunk as exp063's ChessTransformerV2, but the policy head performs
    latent child expansion + iterative refinement + backup instead of
    one-shot spatial scoring.
    """

    def __init__(
        self,
        encoder_dim=256,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        head_dim=256,
        search_steps=3,
        candidate_topk=8,
    ):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = LatentSearchPolicyHead(
            hidden_dim, n_ctx_tokens=4, head_dim=head_dim,
            candidate_topk=candidate_topk, search_steps=search_steps,
            num_heads=num_heads, dropout=dropout,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 3),
        )
        self.hidden_dim = hidden_dim

    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        B = hidden.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)

        cls_hidden = hidden[:, 0, :]
        policy_result = self.policy_head(hidden, cls_hidden)
        value_logits = self.value_head(cls_hidden)

        return {
            **policy_result,
            "value_logits": value_logits,
            "cls_hidden": cls_hidden,
        }

    @torch.no_grad()
    def get_policy_topk(self, board, device, k=10):
        self.eval()
        board_input = batch_boards_to_token_ids([board], device)
        result = self(board_input)
        logits = result["policy_logits"][0]
        mask = legal_move_mask(board).to(device)
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        topk = probs.topk(k)
        moves = []
        for idx, p in zip(topk.indices.cpu().tolist(), topk.values.cpu().tolist()):
            m = index_to_move(idx)
            if m is not None and m in board.legal_moves:
                moves.append((m, p))
        return moves

    @torch.no_grad()
    def get_values_batch(self, boards, device):
        self.eval()
        board_input = batch_boards_to_token_ids(boards, device)
        result = self(board_input)
        wdl = F.softmax(result["value_logits"], dim=-1)
        return (wdl[:, 0] - wdl[:, 2]).cpu().tolist()

    @torch.no_grad()
    def get_backup_values_batch(self, boards, device):
        """Get backed-up values from latent search (search-informed eval)."""
        self.eval()
        board_input = batch_boards_to_token_ids(boards, device)
        result = self(board_input)
        return result["backed_up_value"].cpu().tolist()


# ── Data loading (same as exp063) ──

def build_soft_target(top_moves, board, temperature):
    target = torch.zeros(VOCAB_SIZE)
    valid_idxs = []
    cp_scores = []
    for tm in top_moves:
        uci = tm["uci"]
        if uci not in UCI_TO_IDX:
            continue
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            continue
        if "mate" in tm:
            cp = 10000 if tm["mate"] > 0 else -10000
        else:
            cp = tm.get("cp", 0)
        valid_idxs.append(UCI_TO_IDX[uci])
        cp_scores.append(cp)
    if not valid_idxs:
        return None
    cp_tensor = torch.tensor(cp_scores, dtype=torch.float32)
    probs = F.softmax(cp_tensor / temperature, dim=0)
    for idx, p in zip(valid_idxs, probs):
        target[idx] = p.item()
    return target


def load_jsonl_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_training_data(all_jsonl_data, hf_data, temperature):
    seen_fens = set()
    combined = []

    for d in hf_data:
        fen = d["board"].fen()
        if fen in seen_fens:
            continue
        seen_fens.add(fen)
        move_idx = move_to_index(d["move"])
        hard_target = torch.zeros(VOCAB_SIZE)
        hard_target[move_idx] = 1.0
        combined.append({
            "board": d["board"],
            "move": d["move"],
            "soft_target": hard_target,
            "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
            "phase": d.get("phase", "unknown"),
            "has_soft": False,
        })

    n_soft = 0
    for d in all_jsonl_data:
        try:
            fen = d["fen"]
            if fen in seen_fens:
                continue
            seen_fens.add(fen)
            board = chess.Board(fen)
            move = chess.Move.from_uci(d["best_move"])
            if move not in board.legal_moves:
                continue
            if d["best_move"] not in UCI_TO_IDX:
                continue
            top_moves = d.get("top_moves", [])
            soft_target = build_soft_target(top_moves, board, temperature)
            if soft_target is None:
                move_idx = UCI_TO_IDX[d["best_move"]]
                soft_target = torch.zeros(VOCAB_SIZE)
                soft_target[move_idx] = 1.0
                has_soft = False
            else:
                has_soft = True
                n_soft += 1
            combined.append({
                "board": board,
                "move": move,
                "soft_target": soft_target,
                "wdl": tuple(d["wdl"]),
                "phase": d.get("phase", "unknown"),
                "has_soft": has_soft,
            })
        except Exception:
            continue

    print(f"  Soft targets: {n_soft:,}/{len(combined):,} "
          f"({100 * n_soft / max(len(combined), 1):.1f}%)")
    return combined


# ── Training ──

def compute_loss(result, soft_targets, wdl_targets):
    """Multi-objective loss for latent search model.

    Returns total_loss and dict of component losses for logging.
    """
    # Main policy loss: KL divergence on refined logits
    log_probs = F.log_softmax(result["policy_logits"], dim=-1)
    policy_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean")

    # Base policy auxiliary: KL on coarse logits (before search)
    base_log_probs = F.log_softmax(result["base_policy_logits"], dim=-1)
    base_loss = F.kl_div(base_log_probs, soft_targets, reduction="batchmean")

    # Step-wise auxiliary: average KL across refinement steps
    step_losses = []
    for step_logits in result["step_policy_logits"]:
        step_lp = F.log_softmax(step_logits, dim=-1)
        step_losses.append(F.kl_div(step_lp, soft_targets, reduction="batchmean"))
    step_loss = (
        torch.stack(step_losses).mean()
        if step_losses
        else policy_loss.new_tensor(0.0)
    )

    # Value loss: KL on WDL
    value_log_probs = F.log_softmax(result["value_logits"], dim=-1)
    value_loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")

    # Backup loss: MSE on backed-up value vs WDL-derived scalar
    # scalar_target: P(W) - P(L) in [-1, 1]
    scalar_target = wdl_targets[:, 0] - wdl_targets[:, 2]
    backup_loss = F.mse_loss(result["backed_up_value"], scalar_target)

    total = (
        policy_loss
        + BASE_AUX_WEIGHT * base_loss
        + STEP_AUX_WEIGHT * step_loss
        + VALUE_WEIGHT * value_loss
        + BACKUP_WEIGHT * backup_loss
    )

    metrics = {
        "policy_loss": policy_loss.item(),
        "base_loss": base_loss.item(),
        "step_loss": step_loss.item(),
        "value_loss": value_loss.item(),
        "backup_loss": backup_loss.item(),
    }
    return total, metrics


def train_model(train_data, eval_data, device):
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = ChessTransformerLatentSearch(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        dropout=DROPOUT, head_dim=HEAD_DIM,
        search_steps=SEARCH_STEPS, candidate_topk=CANDIDATE_TOPK,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    head_params = sum(
        p.numel() for p in model.policy_head.parameters()
    )
    print(f"  Model params: {n_params:,} (head: {head_params:,})")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = len(train_data) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS // GRAD_ACCUM
    warmup_steps = max(int(total_steps * WARMUP_FRAC), 1)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        random.shuffle(train_data)

        accum = {
            "policy_loss": 0.0, "base_loss": 0.0,
            "step_loss": 0.0, "value_loss": 0.0, "backup_loss": 0.0,
        }
        n_accum = 0
        optimizer.zero_grad()

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)

            soft_targets = torch.stack(
                [d["soft_target"] for d in chunk]
            ).to(device)
            wdl_targets = torch.tensor(
                [d["wdl"] for d in chunk], device=device, dtype=torch.float32,
            )

            loss, metrics = compute_loss(result, soft_targets, wdl_targets)
            loss = loss / GRAD_ACCUM
            loss.backward()

            for k, v in metrics.items():
                accum[k] += v
            n_accum += 1

            if (i // BATCH_SIZE + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Handle leftover accumulation
        if (steps_per_epoch % GRAD_ACCUM) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avgs = {k: v / max(n_accum, 1) for k, v in accum.items()}
        ev = evaluate(model, eval_data, device)
        ep_time = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            **{k: round(v, 4) for k, v in avgs.items()},
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in ev.items()},
            "time_s": round(ep_time),
        })

        marker = " *" if ev["accuracy"] > best_acc else ""
        print(
            f"  Ep{epoch + 1}: "
            f"pl={avgs['policy_loss']:.3f} "
            f"bl={avgs['base_loss']:.3f} "
            f"sl={avgs['step_loss']:.3f} "
            f"vl={avgs['value_loss']:.3f} "
            f"bk={avgs['backup_loss']:.3f} | "
            f"acc={ev['accuracy']:.1%} "
            f"base={ev['base_accuracy']:.1%} "
            f"top3={ev['top3_accuracy']:.1%} "
            f"rank={ev['mean_sf_rank']:.1f} "
            f"[{ep_time:.0f}s]{marker}",
            flush=True,
        )

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state:
        ckpt_path = OUTPUT_DIR / "best_checkpoint.pt"
        torch.save(best_state, ckpt_path)
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()}
        )
        print(f"  Saved: {ckpt_path} (best acc: {best_acc:.1%})")

    return model, history, best_acc


def evaluate(model, eval_data, device, batch_size=64):
    """Evaluate accuracy, base accuracy (no search), and ranking metrics."""
    model.eval()
    correct = base_correct = top3_correct = total = 0
    entropy_sum = sf_rank_sum = 0.0
    val_correct = val_total = 0

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            true_moves = [d["move"] for d in chunk]

            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)

            logits = result["policy_logits"]
            base_logits = result["base_policy_logits"]

            for j, (board, true_move) in enumerate(zip(boards, true_moves)):
                true_idx = move_to_index(true_move)
                mask = legal_move_mask(board).to(device)

                # Final (searched) accuracy
                l = logits[j].clone()
                l[~mask] = float("-inf")
                pred_idx = l.argmax().item()
                if pred_idx == true_idx:
                    correct += 1
                topk = l.topk(3).indices.tolist()
                if true_idx in topk:
                    top3_correct += 1

                # Base (coarse, no search) accuracy
                bl = base_logits[j].clone()
                bl[~mask] = float("-inf")
                base_pred = bl.argmax().item()
                if base_pred == true_idx:
                    base_correct += 1

                # Entropy and SF rank on final logits
                probs = F.softmax(l, dim=-1)
                p = probs[probs > 0]
                entropy_sum += -(p * p.log()).sum().item()
                sorted_idx = l.argsort(descending=True).tolist()
                rank = (
                    sorted_idx.index(true_idx) + 1
                    if true_idx in sorted_idx
                    else len(sorted_idx)
                )
                sf_rank_sum += rank
                total += 1

            # Value accuracy
            if any("wdl" in d for d in chunk):
                wdl_logits = result["value_logits"]
                for j, d in enumerate(chunk):
                    if "wdl" not in d:
                        continue
                    pred_class = wdl_logits[j].argmax().item()
                    true_wdl = d["wdl"]
                    true_class = max(range(3), key=lambda k: true_wdl[k])
                    if pred_class == true_class:
                        val_correct += 1
                    val_total += 1

    return {
        "accuracy": correct / max(total, 1),
        "base_accuracy": base_correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "search_delta": (correct - base_correct) / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "value_accuracy": val_correct / max(val_total, 1),
    }


# ── Game play ──

def strategy_policy_argmax(model, board, device):
    moves = model.get_policy_topk(board, device, k=1)
    return moves[0][0] if moves else random.choice(list(board.legal_moves))


def strategy_value_rerank_k5(model, board, device):
    top_moves = model.get_policy_topk(board, device, k=5)
    if not top_moves:
        return random.choice(list(board.legal_moves))
    candidate_boards = []
    for m, _ in top_moves:
        b2 = board.copy()
        b2.push(m)
        candidate_boards.append(b2)
    values = model.get_values_batch(candidate_boards, device)
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    scored = [(m, sign * v) for (m, _), v in zip(top_moves, values)]
    scored.sort(key=lambda x: -x[1])
    return scored[0][0]


def strategy_backup_rerank(model, board, device):
    """Use the latent search backup value to rerank top candidates.

    This reranks the top-5 policy moves using the internal backed-up
    value instead of the root value head — testing whether the latent
    child expansion produces better move selection at inference.
    """
    top_moves = model.get_policy_topk(board, device, k=5)
    if not top_moves:
        return random.choice(list(board.legal_moves))
    candidate_boards = []
    for m, _ in top_moves:
        b2 = board.copy()
        b2.push(m)
        candidate_boards.append(b2)
    values = model.get_backup_values_batch(candidate_boards, device)
    sign = 1.0 if board.turn == chess.WHITE else -1.0
    scored = [(m, sign * v) for (m, _), v in zip(top_moves, values)]
    scored.sort(key=lambda x: -x[1])
    return scored[0][0]


def play_game(model, device, strategy_fn, sf_depth, opening_moves=None):
    from stockfish import Stockfish
    sf = Stockfish(
        path=SF_PATH, depth=sf_depth,
        parameters={"Threads": 1, "Hash": 16},
    )
    results = []
    for model_color in [chess.WHITE, chess.BLACK]:
        board = chess.Board()
        move_list = []
        if opening_moves:
            for uci in opening_moves:
                board.push(chess.Move.from_uci(uci))
                move_list.append(uci)
        while not board.is_game_over() and len(move_list) < 150:
            if board.turn == model_color:
                move = strategy_fn(model, board, device)
            else:
                sf.set_fen_position(board.fen())
                move_uci = sf.get_best_move()
                move = chess.Move.from_uci(move_uci)
            board.push(move)
            move_list.append(move.uci())
        result = board.result()
        if result == "1-0":
            outcome = 1.0 if model_color == chess.WHITE else 0.0
        elif result == "0-1":
            outcome = 0.0 if model_color == chess.WHITE else 1.0
        elif result == "1/2-1/2":
            outcome = 0.5
        else:
            outcome = 0.5
        results.append({
            "model_color": "white" if model_color == chess.WHITE else "black",
            "outcome": outcome,
            "result": result,
            "num_moves": len(move_list),
            "termination": (
                board.outcome().termination.name
                if board.outcome() else "max_moves"
            ),
        })
    return results


def run_games(model, device):
    strategies = {
        "policy_argmax": strategy_policy_argmax,
        "value_rerank_k5": strategy_value_rerank_k5,
        "backup_rerank": strategy_backup_rerank,
    }
    all_results = {}
    for sname, sfn in strategies.items():
        print(f"    {sname}:")
        strat_results = {}
        for sf_depth in SF_GAME_DEPTHS:
            wins = draws = losses = total_moves = 0
            for g in range(GAMES_PER_DEPTH // 2):
                opening = OPENINGS[g % len(OPENINGS)]
                game_results = play_game(
                    model, device, sfn, sf_depth, opening,
                )
                for gr in game_results:
                    total_moves += gr["num_moves"]
                    if gr["outcome"] == 1.0:
                        wins += 1
                    elif gr["outcome"] == 0.5:
                        draws += 1
                    else:
                        losses += 1
            n_games = max(wins + draws + losses, 1)
            score = (wins + 0.5 * draws) / n_games
            strat_results[f"d{sf_depth}"] = {
                "wins": wins, "draws": draws, "losses": losses,
                "score": round(score, 3),
                "avg_moves": round(total_moves / n_games),
            }
            print(
                f"      d{sf_depth}: W{wins}/D{draws}/L{losses} "
                f"({score:.1%}, avg {total_moves // n_games}mv)"
            )
        all_results[sname] = strat_results
    return all_results


# ── Main ──

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {device}")
    print(f"Experiment: exp064_latent_search")
    print(f"Hypothesis: Latent child expansion + attention backup > one-shot spatial")
    print(f"Baseline: exp063 (same data, same trunk, one-shot head)")
    print(f"VRAM target: 8 GB (batch={BATCH_SIZE}, accum={GRAD_ACCUM}, K={CANDIDATE_TOPK})")
    print()

    # Phase 1: Load generated data
    print("[1/5] Loading generated data...")
    t_load = time.time()
    all_jsonl = []

    batch_files = sorted(glob.glob(GENERATED_BATCHES))
    for bf in batch_files:
        data = load_jsonl_data(bf)
        print(f"  {bf}: {len(data):,} positions")
        all_jsonl.extend(data)

    if EXP059_DATA.exists():
        data = load_jsonl_data(str(EXP059_DATA))
        print(f"  {EXP059_DATA}: {len(data):,} positions")
        all_jsonl.extend(data)

    n_deep = 0
    if DEEP_DATA.exists():
        deep_data = load_jsonl_data(str(DEEP_DATA))
        print(f"  {DEEP_DATA}: {len(deep_data):,} positions (deep d12)")
        deep_fens = {d["fen"] for d in deep_data}
        all_jsonl = [d for d in all_jsonl if d["fen"] not in deep_fens]
        all_jsonl.extend(deep_data)
        n_deep = len(deep_data)

    print(
        f"  Total JSONL loaded: {len(all_jsonl):,} ({n_deep:,} deep) "
        f"({time.time() - t_load:.0f}s)"
    )

    # Phase 2: Load HF data and combine
    print("\n[2/5] Loading HF data and combining...")
    from hf_data import load_training_set, load_eval_set

    hf_train = load_training_set()
    hf_eval = load_eval_set(n=2500)
    print(f"  HF: {len(hf_train)} train, {len(hf_eval)} eval")

    train_data = prepare_training_data(all_jsonl, hf_train, SOFT_TEMP)
    eval_data = [
        {
            "board": d["board"],
            "move": d["move"],
            "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
            "phase": d.get("phase", "unknown"),
        }
        for d in hf_eval
    ]
    print(f"  Combined train: {len(train_data):,} (deduped)")

    # Phase 3: Train
    print(
        f"\n[3/5] Training latent search model "
        f"({EPOCHS} epochs, bs={BATCH_SIZE}×{GRAD_ACCUM}, K={CANDIDATE_TOPK})..."
    )
    model, history, best_acc = train_model(train_data, eval_data, device)

    # Phase 4: Play games
    print("\n[4/5] Playing games vs Stockfish...")
    try:
        game_results = run_games(model, device)
    except Exception as e:
        print(f"  Games skipped: {e}")
        game_results = {}

    total_time = time.time() - t_start

    # Phase 5: Save results
    print("\n[5/5] Saving results...")
    n_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.policy_head.parameters())

    results = {
        "experiment": "exp064_latent_search",
        "hypothesis": "Latent child expansion + attention backup > one-shot spatial",
        "baseline": "exp063 (same data, same trunk, one-shot head)",
        "seed": SEED,
        "config": {
            "encoder_dim": ENCODER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "search_steps": SEARCH_STEPS,
            "candidate_topk": CANDIDATE_TOPK,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "lr": LR,
            "value_weight": VALUE_WEIGHT,
            "backup_weight": BACKUP_WEIGHT,
            "base_aux_weight": BASE_AUX_WEIGHT,
            "step_aux_weight": STEP_AUX_WEIGHT,
            "soft_temp": SOFT_TEMP,
        },
        "model": {
            "total_params": n_params,
            "head_params": head_params,
        },
        "data": {
            "jsonl_total": len(all_jsonl),
            "deep_labeled": n_deep,
            "hf_train": len(hf_train),
            "combined_deduped": len(train_data),
            "eval": len(eval_data),
        },
        "training": {
            "best_accuracy": round(best_acc, 4),
            "history": history,
        },
        "games": game_results,
        "timing": {"total_s": round(total_time)},
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f" RESULTS: exp064_latent_search")
    print(f"{'=' * 70}")
    print(f"  Model: {n_params:,} params (head: {head_params:,})")
    print(
        f"  Data: {len(all_jsonl):,} JSONL + {len(hf_train):,} HF "
        f"= {len(train_data):,} combined"
    )
    print(f"  Best accuracy: {best_acc:.1%}")
    if history:
        best_ep = max(history, key=lambda h: h["accuracy"])
        print(
            f"  Best epoch: {best_ep['epoch']} "
            f"(base={best_ep['base_accuracy']:.1%}, "
            f"searched={best_ep['accuracy']:.1%}, "
            f"delta={best_ep['search_delta']:+.1%}, "
            f"top3={best_ep['top3_accuracy']:.1%}, "
            f"rank={best_ep['mean_sf_rank']:.1f})"
        )
        print(f"  Search delta (searched - base): {best_ep['search_delta']:+.1%}")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
