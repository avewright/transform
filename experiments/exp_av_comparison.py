"""exp_av_comparison.py — Fair A/B: policy-only CE vs action-value Q(s,a)

Same 5K Stockfish-labeled data, same model, same epochs.
Tests whether action-value signal yields better policy accuracy.
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
from chess_model import LearnedBoardEncoder, ChessModel
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, move_to_index, index_to_move, legal_move_mask
from config import Config

OUTPUT_DIR = Path("outputs/exp_av_comparison")
DATA_PATH = Path("data/sf_labels_5k_d8.jsonl")

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
ENCODER_DIM = 256
SEED = 42


def cp_to_q(cp, eval_type="cp"):
    if eval_type == "mate":
        return 1.0 - 0.001 * min(abs(cp), 50) if cp > 0 else 0.001 * min(abs(cp), 50) if cp < 0 else 0.5
    return 1.0 / (1.0 + math.exp(-cp / 111.7))


class ActionValueHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, vocab_size))
    def forward(self, x):
        return torch.sigmoid(self.head(x))


def load_data():
    data = []
    with open(DATA_PATH) as f:
        for line in f:
            e = json.loads(line)
            if e["best_uci"] in UCI_TO_IDX:
                data.append(e)
    rng = random.Random(SEED)
    rng.shuffle(data)
    n_eval = 250
    return data[n_eval:], data[:n_eval]


def make_policy_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i+batch_size]
        boards = [chess.Board(d["fen"]) for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        targets = torch.tensor([UCI_TO_IDX[d["best_uci"]] for d in chunk], dtype=torch.long, device=device)
        wdl_targets = torch.tensor(
            [max(range(3), key=lambda x: d.get("wdl", [0.33, 0.34, 0.33])[x]) for d in chunk],
            dtype=torch.long, device=device)
        batches.append((batch_input, targets, wdl_targets))
    return batches


def make_av_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i+batch_size]
        boards = [chess.Board(d["fen"]) for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        q_targets = torch.zeros(len(chunk), VOCAB_SIZE, device=device)
        q_mask = torch.zeros(len(chunk), VOCAB_SIZE, dtype=torch.bool, device=device)
        best_idx = torch.tensor([UCI_TO_IDX[d["best_uci"]] for d in chunk], dtype=torch.long, device=device)
        wdl_targets = torch.tensor(
            [max(range(3), key=lambda x: d.get("wdl", [0.33, 0.34, 0.33])[x]) for d in chunk],
            dtype=torch.long, device=device)
        for j, d in enumerate(chunk):
            for mv in d["move_values"]:
                if mv["uci"] in UCI_TO_IDX:
                    idx = UCI_TO_IDX[mv["uci"]]
                    q_targets[j, idx] = cp_to_q(mv["cp"], mv["type"])
                    q_mask[j, idx] = True
        batches.append((batch_input, q_targets, q_mask, best_idx, wdl_targets))
    return batches


def evaluate(chess_model, eval_data, device, batch_size=64):
    chess_model.eval()
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i+batch_size]
            boards = [chess.Board(d["fen"]) for d in chunk]
            targets = [UCI_TO_IDX.get(d["best_uci"], 0) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = chess_model(batch_input)
            logits = result["policy_logits"]
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()
            for j, t in enumerate(targets):
                total += 1
                if preds[j] == t: correct += 1
                if t in top3s[j]: top3_correct += 1
    return {"accuracy": correct/max(total,1), "top3_accuracy": top3_correct/max(total,1), "total": total}


def train_policy_only(train_data, eval_data, device, full_model):
    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        batches = make_policy_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0
        for batch_input, targets, wdl_targets in batches:
            optimizer.zero_grad()
            result = model(batch_input, move_targets=targets, value_targets=wdl_targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += result["loss"].item(); steps += 1

        ev = evaluate(model, eval_data, device)
        elapsed = time.time() - t0
        history.append({**ev, "loss": ep_loss/max(steps,1), "epoch": epoch+1})
        print(f"  [Policy] Ep {epoch+1}/{EPOCHS}: loss={ep_loss/max(steps,1):.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")
    return model, history


def train_action_value(train_data, eval_data, device, full_model):
    torch.manual_seed(SEED)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    model = ChessModel(full_model, encoder=encoder, freeze_backbone=True).to(device)
    av_head = ActionValueHead(model.hidden_size, VOCAB_SIZE).to(device)

    all_params = (list(model.encoder.parameters()) + list(model.input_proj.parameters()) +
                  list(model.policy_head.parameters()) + list(model.value_head.parameters()) +
                  list(av_head.parameters()))
    optimizer = AdamW(all_params, lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train(); av_head.train()
        batches = make_av_batches(train_data, BATCH_SIZE, device)
        ep_av = ep_p = ep_v = steps = 0
        for batch_input, q_targets, q_mask, best_idx, wdl_targets in batches:
            tokens = model.encoder(batch_input)
            embeds = model.input_proj(tokens)
            embeds = embeds.to(next(model.backbone.parameters()).dtype)
            outputs = model.backbone(inputs_embeds=embeds, use_cache=False)
            hidden = outputs.last_hidden_state.float()
            g = hidden[:, 0, :]
            policy_logits = model.policy_head(g)
            value_logits = model.value_head(g)
            q_pred = av_head(g)

            av_loss = ((q_pred - q_targets)**2 * q_mask.float()).sum() / q_mask.float().sum()
            p_loss = F.cross_entropy(policy_logits, best_idx)
            v_loss = F.cross_entropy(value_logits, wdl_targets)
            loss = av_loss + 0.5 * p_loss + 0.5 * v_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
            ep_av += av_loss.item(); ep_p += p_loss.item(); ep_v += v_loss.item(); steps += 1

        ev = evaluate(model, eval_data, device)
        elapsed = time.time() - t0
        history.append({**ev, "av_loss": ep_av/max(steps,1), "p_loss": ep_p/max(steps,1),
                       "v_loss": ep_v/max(steps,1), "epoch": epoch+1})
        print(f"  [AV] Ep {epoch+1}/{EPOCHS}: av={ep_av/max(steps,1):.4f} p={ep_p/max(steps,1):.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")
    return model, av_head, history


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data, eval_data = load_data()
    print(f"Data: {len(train_data)} train, {len(eval_data)} eval")

    cfg = Config()
    full_model, _ = load_base_model(cfg)
    full_model = full_model.to(device)

    # A: Policy-only
    print(f"\n{'='*60}\n VARIANT A: Policy-Only CE ({EPOCHS} epochs)\n{'='*60}")
    model_a, hist_a = train_policy_only(train_data, eval_data, device, full_model)
    final_a = evaluate(model_a, eval_data, device)
    best_a = max(h["accuracy"] for h in hist_a)
    print(f"  Final: acc={final_a['accuracy']:.1%} top3={final_a['top3_accuracy']:.1%} (best={best_a:.1%})")

    del model_a; torch.cuda.empty_cache()

    # B: Action-Value
    print(f"\n{'='*60}\n VARIANT B: Action-Value Q(s,a) ({EPOCHS} epochs)\n{'='*60}")
    model_b, av_head, hist_b = train_action_value(train_data, eval_data, device, full_model)
    final_b = evaluate(model_b, eval_data, device)
    best_b = max(h["accuracy"] for h in hist_b)
    print(f"  Final: acc={final_b['accuracy']:.1%} top3={final_b['top3_accuracy']:.1%} (best={best_b:.1%})")

    # Compare
    diff = best_b - best_a
    winner = "ACTION-VALUE" if diff > 0.005 else "POLICY-ONLY" if diff < -0.005 else "TIE"
    print(f"\n{'='*60}\n COMPARISON\n{'='*60}")
    print(f"  Policy-only:  best={best_a:.1%}  final={final_a['accuracy']:.1%} / top3={final_a['top3_accuracy']:.1%}")
    print(f"  Action-value: best={best_b:.1%}  final={final_b['accuracy']:.1%} / top3={final_b['top3_accuracy']:.1%}")
    print(f"  Winner: {winner} (delta: {diff:+.1%})")

    results = {
        "experiment": "av_comparison",
        "data": str(DATA_PATH), "train": len(train_data), "eval": len(eval_data),
        "epochs": EPOCHS,
        "policy_only": {"best": best_a, "final": final_a, "history": hist_a},
        "action_value": {"best": best_b, "final": final_b, "history": hist_b},
        "winner": winner, "delta": diff,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
