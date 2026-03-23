"""exp056: Internal-search policy head via iterative attention refinement.

Hypothesis:
  A policy head that performs a small amount of learned "internal search"
  over candidate moves can outperform a one-shot spatial policy head on the
  same data and model body. The head first scores all moves coarsely, then
  repeatedly refines a top-k candidate set with attention over board tokens
  and over the candidate set itself.

This is not alpha-beta or MCTS outside the model. The extra computation
happens inside the forward pass, so the model learns to spend more compute
on a small set of candidate continuations before committing to a move.

Experiment contract:
  - Hypothesis: iterative candidate refinement > one-shot spatial head
  - Primary metric: top-1 SF-accuracy on HF test split
  - Secondary: top-3, per-phase accuracy, entropy, SF-move rank
  - Evaluation set: HF avewright/chess-positions test (2500)
  - Seeds: 42, 123
  - Training data: HF train split (~47.5K)
  - Model: Medium body (512d, 8L, 8H) + internal-search head
  - Epochs: 5
"""

import json
import math
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
    VOCAB_SIZE,
    IDX_TO_UCI,
    move_to_index,
    legal_move_mask,
)

OUTPUT_DIR = Path("outputs/exp056_internal_search_policy")

# Model config
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1

# Internal-search head config
HEAD_DIM = 256
SEARCH_STEPS = 3
CANDIDATE_TOPK = 32
AUX_BASE_WEIGHT = 0.3

# Training
EPOCHS = 5
BATCH_SIZE = 96
LR = 2e-4
WARMUP_FRAC = 0.05
SEEDS = [42, 123]


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


class InternalSearchPolicyHead(nn.Module):
    """Spatial prior + iterative candidate refinement via attention.

    Stage 1:
      Score all 5504 moves with a spatial factorized head.

    Stage 2:
      Select top-k candidates from the coarse prior and refine those move
      states for several steps with:
        - cross-attention to board hidden states
        - self-attention among candidate moves
        - residual MLP updates

      Each step predicts a delta score for candidate moves; deltas are added
      back into the full move-logit vector.
    """

    def __init__(
        self,
        hidden_size,
        n_ctx_tokens=4,
        head_dim=256,
        candidate_topk=32,
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

        # Coarse spatial prior over the full vocabulary.
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.base_score = nn.Linear(head_dim, 1)

        # Refinement stack over top-k candidate moves.
        self.board_k = nn.Linear(hidden_size, head_dim)
        self.board_v = nn.Linear(hidden_size, head_dim)
        self.cross_attn = nn.MultiheadAttention(
            head_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            head_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.refine_mlp = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim * 4),
            nn.GELU(),
            nn.Linear(head_dim * 4, head_dim),
        )
        self.delta_score = nn.Linear(head_dim, 1)
        self.step_gate = nn.Parameter(torch.zeros(search_steps))

    def _base_components(self, hidden_states, cls_hidden):
        sq_hidden = hidden_states[:, self.n_ctx : self.n_ctx + 64, :]
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
        board_ctx = hidden_states
        move_states, base_logits = self._base_components(hidden_states, cls_hidden)

        topk = min(self.candidate_topk, base_logits.shape[-1])
        cand_idx = base_logits.topk(topk, dim=-1).indices
        gather_idx = cand_idx.unsqueeze(-1).expand(-1, -1, self.head_dim)
        cand_states = torch.gather(move_states, 1, gather_idx)

        board_kv = self.board_k(board_ctx), self.board_v(board_ctx)
        step_logits = []

        for step in range(self.search_steps):
            cross_out, _ = self.cross_attn(
                cand_states,
                board_kv[0],
                board_kv[1],
                need_weights=False,
            )
            cand_states = cand_states + cross_out

            self_out, _ = self.self_attn(
                cand_states,
                cand_states,
                cand_states,
                need_weights=False,
            )
            cand_states = cand_states + self_out
            cand_states = cand_states + self.refine_mlp(cand_states)

            delta = self.delta_score(F.gelu(cand_states)).squeeze(-1)
            gated_delta = torch.tanh(self.step_gate[step]) * delta
            step_logits.append(base_logits.scatter_add(1, cand_idx, gated_delta))

        final_logits = step_logits[-1] if step_logits else base_logits
        return {
            "policy_logits": final_logits,
            "base_policy_logits": base_logits,
            "candidate_indices": cand_idx,
            "step_policy_logits": step_logits,
        }


class ChessTransformerInternalSearch(nn.Module):
    """Chess-native transformer with iterative internal-search policy head."""

    def __init__(
        self,
        encoder_dim=256,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        head_dim=256,
        search_steps=3,
        candidate_topk=32,
    ):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = InternalSearchPolicyHead(
            hidden_dim,
            n_ctx_tokens=4,
            head_dim=head_dim,
            candidate_topk=candidate_topk,
            search_steps=search_steps,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        batch_size = hidden.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)

        cls_hidden = hidden[:, 0, :]
        policy = self.policy_head(hidden, cls_hidden)
        value_logits = self.value_head(cls_hidden)

        return {
            **policy,
            "value_logits": value_logits,
            "cls_hidden": cls_hidden,
        }


def load_hf_data():
    from hf_data import load_training_set, load_eval_set, dataset_info

    info = dataset_info()
    print(f"  HF: {info['train']['num_rows']} train, {info['test']['num_rows']} test")

    def _enrich(raw_list):
        return [
            {
                "board": d["board"],
                "move": d["move"],
                "phase": d.get("phase", "unknown"),
            }
            for d in raw_list
        ]

    train_data = _enrich(load_training_set())
    eval_data = _enrich(load_eval_set(n=2500))
    print(f"  Loaded: {len(train_data)} train, {len(eval_data)} eval")
    return train_data, eval_data


def compute_policy_loss(result, targets):
    final_loss = F.cross_entropy(result["policy_logits"], targets)
    base_loss = F.cross_entropy(result["base_policy_logits"], targets)

    step_losses = []
    for step_logits in result["step_policy_logits"]:
        step_losses.append(F.cross_entropy(step_logits, targets))
    step_loss = torch.stack(step_losses).mean() if step_losses else final_loss.new_tensor(0.0)

    loss = final_loss + AUX_BASE_WEIGHT * base_loss + 0.2 * step_loss
    metrics = {
        "final_policy_loss": final_loss.item(),
        "base_policy_loss": base_loss.item(),
        "step_policy_loss": step_loss.item(),
    }
    return loss, metrics


def train_one_seed(model, train_data, eval_data, device, seed, save_dir):
    torch.manual_seed(seed)
    random.seed(seed)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_data) // BATCH_SIZE) * EPOCHS
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
        total_final = 0.0
        total_base = 0.0
        total_step = 0.0
        n_batches = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i : i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor([move_to_index(d["move"]) for d in chunk], device=device)
            batch_input = batch_boards_to_token_ids(boards, device)

            result = model(batch_input)
            loss, loss_metrics = compute_policy_loss(result, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_final += loss_metrics["final_policy_loss"]
            total_base += loss_metrics["base_policy_loss"]
            total_step += loss_metrics["step_policy_loss"]
            n_batches += 1

        avg_final = total_final / max(n_batches, 1)
        avg_base = total_base / max(n_batches, 1)
        avg_step = total_step / max(n_batches, 1)
        ev = evaluate_rich(model, eval_data, device)
        ep_time = time.time() - t0
        marker = " *" if ev["accuracy"] > best_acc else ""
        history.append({
            "epoch": epoch + 1,
            "final_policy_loss": round(avg_final, 4),
            "base_policy_loss": round(avg_base, 4),
            "step_policy_loss": round(avg_step, 4),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()},
            "time_s": round(ep_time),
        })
        print(
            f"  [s{seed}] Ep{epoch+1}: "
            f"final={avg_final:.4f} base={avg_base:.4f} step={avg_step:.4f} "
            f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} "
            f"base_acc={ev['base_accuracy']:.1%} "
            f"[{ep_time:.0f}s]{marker}",
            flush=True,
        )

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        ckpt_path = save_dir / f"internal_search_s{seed}.pt"
        torch.save(best_state, ckpt_path)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Saved: {ckpt_path.name}")

    return history, best_acc


def evaluate_rich(model, eval_data, device, batch_size=128):
    model.eval()
    correct = base_correct = top3_correct = total = 0
    entropy_sum = sf_rank_sum = 0.0
    phase_stats = {}

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i : i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            phases = [d["phase"] for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]
            base_logits = result["base_policy_logits"]

            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
                base_logits[j, ~mask] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().tolist()
            base_preds = base_logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()

            for j, target in enumerate(targets):
                total += 1
                is_correct = preds[j] == target
                if is_correct:
                    correct += 1
                if base_preds[j] == target:
                    base_correct += 1
                if target in top3s[j]:
                    top3_correct += 1

                p = probs[j]
                legal_p = p[p > 0]
                entropy_sum += -(legal_p * legal_p.log()).sum().item()
                sorted_idx = logits[j].argsort(descending=True).cpu().tolist()
                sf_rank_sum += sorted_idx.index(target) + 1 if target in sorted_idx else len(sorted_idx)

                phase = phases[j]
                if phase not in phase_stats:
                    phase_stats[phase] = {"correct": 0, "total": 0}
                phase_stats[phase]["total"] += 1
                if is_correct:
                    phase_stats[phase]["correct"] += 1

    metrics = {
        "accuracy": correct / max(total, 1),
        "base_accuracy": base_correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "n_eval": total,
    }
    for phase, stats in sorted(phase_stats.items()):
        metrics[f"acc_{phase}"] = stats["correct"] / max(stats["total"], 1)
        metrics[f"n_{phase}"] = stats["total"]
    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Experiment: exp056_internal_search_policy")
    print(
        "Hypothesis: iterative attention over top-k candidate moves "
        "beats one-shot spatial scoring"
    )
    print(
        f"Config: {HIDDEN_DIM}d, {NUM_LAYERS}L, {NUM_HEADS}H, "
        f"steps={SEARCH_STEPS}, topk={CANDIDATE_TOPK}, "
        f"bs={BATCH_SIZE}, lr={LR}, epochs={EPOCHS}"
    )

    print("\n[1/2] Loading data...")
    train_data, eval_data = load_hf_data()

    results = {
        "experiment": "exp056_internal_search_policy",
        "hypothesis": "iterative internal-search head > one-shot spatial head",
        "data_source": "hf:avewright/chess-positions",
        "baseline": "exp053 medium spatial, exp055 joint medium",
        "seeds": SEEDS,
        "config": {
            "encoder_dim": ENCODER_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "search_steps": SEARCH_STEPS,
            "candidate_topk": CANDIDATE_TOPK,
            "aux_base_weight": AUX_BASE_WEIGHT,
        },
        "seeds_data": [],
    }

    print("\n[2/2] Training internal-search model...")
    seed_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        model = ChessTransformerInternalSearch(
            encoder_dim=ENCODER_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            head_dim=HEAD_DIM,
            search_steps=SEARCH_STEPS,
            candidate_topk=CANDIDATE_TOPK,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        history, best_acc = train_one_seed(model, train_data, eval_data, device, seed, OUTPUT_DIR)
        final_eval = evaluate_rich(model, eval_data, device)
        seed_results.append({
            "seed": seed,
            "best_accuracy": best_acc,
            "final_eval": {k: round(v, 4) if isinstance(v, float) else v for k, v in final_eval.items()},
            "history": history,
            "params": n_params,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accs = [r["best_accuracy"] for r in seed_results]
    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
    results["seeds_data"] = seed_results
    results["mean_accuracy"] = round(mean_acc, 4)
    results["std_accuracy"] = round(std_acc, 4)
    results["timing"] = {"total_s": round(time.time() - t_start)}

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(" RESULTS: exp056_internal_search_policy")
    for r in seed_results:
        fe = r["final_eval"]
        print(
            f"  s{r['seed']}: acc={r['best_accuracy']:.1%} "
            f"base={fe['base_accuracy']:.1%} "
            f"top3={fe['top3_accuracy']:.1%}"
        )
    print(f"\n  Mean: {mean_acc:.1%} +/- {std_acc:.1%}")
    print(f"  Total time: {time.time() - t_start:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
