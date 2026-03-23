"""exp055: Joint policy + value training with soft Stockfish targets.

Hypothesis: Training the value head jointly with the policy head, using
WDL targets from Stockfish, will produce a model that plays better games
(via search) than a policy-only model at the same accuracy level.

Key changes from exp052/053:
  1. Value head is trained (WDL cross-entropy), not just defined
  2. Policy uses soft targets from Stockfish top-k moves (KL divergence)
  3. Joint loss = policy_loss + value_weight * value_loss
  4. Saves checkpoints compatible with exp054 search baseline

Experiment contract:
  - Hypothesis: joint policy+value > policy-only for gameplay (via search)
  - Primary metric: top-1 SF-accuracy (should be >= exp053 baseline)
  - Secondary: value head calibration, game performance via exp054
  - Evaluation set: HF test split (2500)
  - Seeds: 42, 123
  - Training data: HF train split with WDL + top_moves
  - Model: Medium (512d, 8L) spatial head — same as exp053
  - Epochs: 5
  - Runtime target: <30 min total
  - Device: CUDA (RTX 2000 Ada, 16GB)
"""

import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

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

OUTPUT_DIR = Path("outputs/exp055_joint_policy_value")

# ── Medium model config (same as exp053) ──
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1

# ── Training ──
EPOCHS = 5
BATCH_SIZE = 128
LR = 2e-4
WARMUP_FRAC = 0.05
SEEDS = [42, 123]
VALUE_WEIGHT = 0.5       # weight of value loss in joint loss
SOFT_TEMP = 1.0          # temperature for soft policy targets


# =====================================================================
# Architecture (same as exp053 ChessTransformerV2 + trained value head)
# =====================================================================

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


class SpatialPolicyHead(nn.Module):
    def __init__(self, hidden_size, n_ctx_tokens=4, head_dim=256):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)

    def forward(self, hidden_states, cls_hidden):
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(cls_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


class ChessTransformerJoint(nn.Module):
    """Chess-native transformer with CLS, spatial policy, and WDL value head."""

    def __init__(self, encoder_dim=256, hidden_dim=512,
                 num_layers=8, num_heads=8, dropout=0.1, head_dim=256):
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
        self.policy_head = SpatialPolicyHead(
            hidden_dim, n_ctx_tokens=4, head_dim=head_dim,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),  # WDL
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
        policy_logits = self.policy_head(hidden, cls_hidden)
        value_logits = self.value_head(cls_hidden)
        return {
            "policy_logits": policy_logits,
            "value_logits": value_logits,
            "cls_hidden": cls_hidden,
        }


# =====================================================================
# Data loading — includes WDL targets and top-k move soft targets
# =====================================================================

def load_hf_data_rich():
    """Load HF data with WDL and top_moves for joint training."""
    from hf_data import dataset_info
    from datasets import load_dataset

    info = dataset_info()
    print(f"  HF: {info['train']['num_rows']} train, "
          f"{info['test']['num_rows']} test")

    ds_train = load_dataset("avewright/chess-positions", split="train")
    ds_test = load_dataset("avewright/chess-positions", split="test")

    def _process(ds, max_n=None):
        data = []
        for row in ds:
            try:
                board = chess.Board(row["fen"])
                move_uci = row["best_move"]
                move = chess.Move.from_uci(move_uci)
                if move_uci not in UCI_TO_IDX or move not in board.legal_moves:
                    continue

                # WDL target (normalize to distribution)
                wdl = [
                    max(row.get("wdl_win", 0), 0),
                    max(row.get("wdl_draw", 0), 0),
                    max(row.get("wdl_loss", 0), 0),
                ]
                wdl_sum = sum(wdl)
                if wdl_sum > 0:
                    wdl = [w / wdl_sum for w in wdl]
                else:
                    wdl = [0.33, 0.34, 0.33]

                # Parse top_moves for soft targets
                top_moves_raw = row.get("top_moves", "")
                soft_targets = {}
                if top_moves_raw:
                    try:
                        top_moves = json.loads(top_moves_raw) if isinstance(top_moves_raw, str) else top_moves_raw
                        for tm in top_moves:
                            uci = tm.get("uci", tm.get("move", tm.get("Move", "")))
                            cp = tm.get("cp", tm.get("Centipawn", None))
                            if uci in UCI_TO_IDX and cp is not None:
                                soft_targets[UCI_TO_IDX[uci]] = cp
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Precompute soft target tensor (CPU, moved to GPU in training)
                soft_t = precompute_soft_target(soft_targets)

                data.append({
                    "board": board,
                    "move": move,
                    "wdl": wdl,
                    "soft_target": soft_t,  # tensor or None
                    "phase": row.get("phase", "unknown"),
                    "eval_type": row.get("eval_type", ""),
                    "eval_value": row.get("eval_value", 0),
                })
                if max_n and len(data) >= max_n:
                    break
            except Exception:
                continue
        return data

    train_data = _process(ds_train)
    eval_data = _process(ds_test, max_n=2500)
    print(f"  Loaded: {len(train_data)} train, {len(eval_data)} eval")

    # Stats on soft targets
    n_soft = sum(1 for d in train_data if d["soft_target"] is not None)
    print(f"  Positions with top-k soft targets: {n_soft}/{len(train_data)}")

    return train_data, eval_data


def precompute_soft_target(soft_targets, temp=SOFT_TEMP):
    """Precompute a soft target distribution over the move vocabulary.
    Returns a (VOCAB_SIZE,) tensor or None if <2 moves."""
    if len(soft_targets) < 2:
        return None
    target = torch.zeros(VOCAB_SIZE)
    indices = list(soft_targets.keys())
    cps = [soft_targets[i] / (100.0 * temp) for i in indices]
    probs = F.softmax(torch.tensor(cps), dim=0)
    for i, idx in enumerate(indices):
        target[idx] = probs[i]
    return target


# =====================================================================
# Training
# =====================================================================

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
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_soft_loss = 0.0
        n_batches = 0
        n_soft_batches = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor(
                [move_to_index(d["move"]) for d in chunk], device=device
            )
            wdl_targets = torch.tensor(
                [d["wdl"] for d in chunk], dtype=torch.float, device=device
            )

            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)

            # Policy loss — mix of hard CE and soft KL
            policy_logits = result["policy_logits"]
            hard_loss = F.cross_entropy(policy_logits, targets)

            # Soft policy targets (batched KL divergence)
            soft_targets_list = [d["soft_target"] for d in chunk]
            soft_mask = [s is not None for s in soft_targets_list]
            n_soft = sum(soft_mask)

            if n_soft > 0:
                soft_batch = torch.stack(
                    [s for s in soft_targets_list if s is not None]
                ).to(device)
                soft_logits = policy_logits[torch.tensor(soft_mask)]
                log_probs = F.log_softmax(soft_logits, dim=-1)
                soft_loss = F.kl_div(log_probs, soft_batch, reduction="batchmean")
                policy_loss = 0.7 * hard_loss + 0.3 * soft_loss
                total_soft_loss += soft_loss.item()
                n_soft_batches += 1
            else:
                policy_loss = hard_loss

            # Value loss — WDL cross-entropy
            value_logits = result["value_logits"]
            value_loss = F.cross_entropy(value_logits, wdl_targets)

            # Joint loss
            loss = policy_loss + VALUE_WEIGHT * value_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_policy_loss += hard_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        avg_pol = total_policy_loss / max(n_batches, 1)
        avg_val = total_value_loss / max(n_batches, 1)
        avg_soft = total_soft_loss / max(n_soft_batches, 1) if n_soft_batches else 0
        ev = evaluate_rich(model, eval_data, device)
        ep_time = time.time() - t0
        history.append({
            "epoch": epoch + 1,
            "policy_loss": round(avg_pol, 4),
            "value_loss": round(avg_val, 4),
            "soft_loss": round(avg_soft, 4),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in ev.items()},
            "time_s": round(ep_time),
        })
        marker = " *" if ev["accuracy"] > best_acc else ""
        print(
            f"  [s{seed}] Ep{epoch+1}: "
            f"pol={avg_pol:.4f} val={avg_val:.4f} soft={avg_soft:.4f} "
            f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} "
            f"val_acc={ev.get('value_accuracy', 0):.1%} "
            f"[{ep_time:.0f}s]{marker}",
            flush=True,
        )

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        ckpt_path = save_dir / f"joint_medium_s{seed}.pt"
        torch.save(best_state, ckpt_path)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Saved: {ckpt_path.name}")

    return history, best_acc


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_rich(model, eval_data, device, batch_size=128):
    model.eval()
    correct = top3_correct = total = 0
    entropy_sum = sf_rank_sum = 0.0
    value_correct = value_total = 0
    phase_stats = {}

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            phases = [d["phase"] for d in chunk]
            wdl_targets = [d["wdl"] for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]

            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()

            # Value head accuracy (most-probable WDL class matches target)
            value_preds = result["value_logits"].argmax(dim=-1).cpu().tolist()

            for j, t in enumerate(targets):
                total += 1
                is_correct = preds[j] == t
                if is_correct:
                    correct += 1
                if t in top3s[j]:
                    top3_correct += 1
                p = probs[j]
                legal_p = p[p > 0]
                entropy_sum += -(legal_p * legal_p.log()).sum().item()
                sorted_idx = logits[j].argsort(descending=True).cpu().tolist()
                sf_rank = sorted_idx.index(t) + 1 if t in sorted_idx else len(sorted_idx)
                sf_rank_sum += sf_rank

                # Value accuracy
                wdl_t = wdl_targets[j]
                true_class = max(range(3), key=lambda k: wdl_t[k])
                value_total += 1
                if value_preds[j] == true_class:
                    value_correct += 1

                ph = phases[j]
                if ph not in phase_stats:
                    phase_stats[ph] = {"correct": 0, "total": 0}
                phase_stats[ph]["total"] += 1
                if is_correct:
                    phase_stats[ph]["correct"] += 1

    result = {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "value_accuracy": value_correct / max(value_total, 1),
        "n_eval": total,
    }
    for ph, ps in sorted(phase_stats.items()):
        result[f"acc_{ph}"] = ps["correct"] / max(ps["total"], 1)
        result[f"n_{ph}"] = ps["total"]
    return result


# =====================================================================
# Main
# =====================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp055_joint_policy_value")
    print(f"Hypothesis: Joint policy+value training improves gameplay via search")
    print(f"Config: {HIDDEN_DIM}d, {NUM_LAYERS}L, {NUM_HEADS}H, "
          f"bs={BATCH_SIZE}, lr={LR}, epochs={EPOCHS}, "
          f"value_weight={VALUE_WEIGHT}")

    print(f"\n[1/2] Loading data with WDL + top-k targets...")
    train_data, eval_data = load_hf_data_rich()

    results = {
        "experiment": "exp055_joint_policy_value",
        "hypothesis": "Joint policy+value > policy-only for gameplay (via search)",
        "data_source": "hf:avewright/chess-positions (WDL + top_moves)",
        "baseline": "exp053 spatial Medium: ~35% (policy-only)",
        "seeds": SEEDS,
        "config": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "value_weight": VALUE_WEIGHT, "soft_temp": SOFT_TEMP,
        },
        "seeds_data": [],
    }

    print(f"\n[2/2] Training joint model...")
    seed_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        model = ChessTransformerJoint(
            encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
            dropout=DROPOUT, head_dim=256,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        history, best_acc = train_one_seed(
            model, train_data, eval_data, device, seed, OUTPUT_DIR,
        )
        final_eval = evaluate_rich(model, eval_data, device)

        seed_results.append({
            "seed": seed,
            "best_accuracy": best_acc,
            "final_eval": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in final_eval.items()},
            "history": history,
            "params": n_params,
        })
        del model
        torch.cuda.empty_cache()

    accs = [r["best_accuracy"] for r in seed_results]
    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc)**2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
    results["seeds_data"] = seed_results
    results["mean_accuracy"] = round(mean_acc, 4)
    results["std_accuracy"] = round(std_acc, 4)
    results["timing"] = {"total_s": round(time.time() - t_start)}

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" RESULTS: exp055_joint_policy_value")
    print(f" Config: {HIDDEN_DIM}d, {NUM_LAYERS}L, epochs={EPOCHS}")
    for r in seed_results:
        fe = r["final_eval"]
        print(f"  s{r['seed']}: acc={r['best_accuracy']:.1%} "
              f"val_acc={fe.get('value_accuracy', 0):.1%}")
    print(f"\n  Mean policy acc: {mean_acc:.1%} +/- {std_acc:.1%}")
    val_accs = [r["final_eval"].get("value_accuracy", 0) for r in seed_results]
    print(f"  Mean value acc: {sum(val_accs)/len(val_accs):.1%}")
    print(f"  Total time: {results['timing']['total_s']}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
