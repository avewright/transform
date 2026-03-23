"""exp058: SF-calibrated value head → improved search gameplay.

Hypothesis: Fine-tuning the value head on Stockfish centipawn evaluations
(instead of noisy WDL game outcomes) will improve search-based gameplay.
exp057 showed 2-ply search HURTS because the WDL-trained value head is
too noisy for minimax. A SF-calibrated value head should fix this.

Plan:
  Phase 1: Label ~20K HF positions with Stockfish cp evaluations (depth 5)
  Phase 2: Convert cp to WDL targets using sigmoid mapping
  Phase 3: Fine-tune ONLY the value head (freeze policy + encoder)
  Phase 4: Retest search strategies at SF d1, d2, d3

Experiment contract:
  - Hypothesis: SF-calibrated value head + search > WDL value head + search
  - Primary metric: W/D/L vs Stockfish d1, d2, d3
  - Secondary: value head accuracy, calibration quality
  - Data: 20K HF positions labeled with SF depth 5
  - Seeds: 1 (fixed — deterministic labeling + same checkpoint)
  - Runtime target: <10 minutes total (labeling ~3min + training ~1min + games ~3min)
  - Device: CUDA (RTX 2000 Ada, 16GB)
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
    VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI,
    move_to_index, legal_move_mask, index_to_move,
)

OUTPUT_DIR = Path("outputs/exp058_sf_value_head")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

CHECKPOINT = Path("outputs/exp055_joint_policy_value/joint_medium_s42.pt")

# Labeling config
N_LABEL = 20000     # positions to label with SF
SF_DEPTH = 5        # depth 5 is fast and reasonably accurate
SF_THREADS = 4

# Value training config
VALUE_EPOCHS = 5
VALUE_LR = 1e-3     # higher LR since only training small head
VALUE_BATCH = 256

# Search game config
SF_GAME_DEPTHS = [1, 2, 3]
GAMES_PER_DEPTH = 8

# Fixed opening book (same as exp054/057)
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


# =====================================================================
# Model architecture (same as exp055/057)
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


class ChessTransformerV2(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=8, num_heads=8,
                 encoder_dim=256, head_dim=256, dropout=0.0):
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
        policy_logits = self.policy_head(hidden, cls_hidden)
        value_logits = self.value_head(cls_hidden)
        return {
            "policy_logits": policy_logits,
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
    def get_value(self, board, device):
        self.eval()
        board_input = batch_boards_to_token_ids([board], device)
        result = self(board_input)
        wdl = F.softmax(result["value_logits"][0], dim=-1)
        return (wdl[0] - wdl[2]).item()

    @torch.no_grad()
    def get_values_batch(self, boards, device):
        self.eval()
        board_input = batch_boards_to_token_ids(boards, device)
        result = self(board_input)
        wdl = F.softmax(result["value_logits"], dim=-1)
        return (wdl[:, 0] - wdl[:, 2]).cpu().tolist()


# =====================================================================
# Phase 1: SF labeling
# =====================================================================

def cp_to_wdl(cp, k=0.004):
    """Convert centipawn evaluation to (win, draw, loss) using Lichess model.

    Uses a sigmoid mapping calibrated to approximate real WDL distributions.
    k=0.004 maps: cp=0→(25%, 50%, 25%), cp=100→(42%, 41%, 17%), cp=300→(82%, 15%, 3%)
    """
    # Win probability from side-to-move perspective
    win_prob = 1.0 / (1.0 + math.exp(-k * cp))
    loss_prob = 1.0 - win_prob
    # Draw probability peaks near 0, decays outward
    draw_prob = max(0.0, 1.0 - abs(2 * win_prob - 1) ** 0.5)
    total = win_prob + draw_prob + loss_prob
    return (win_prob / total, draw_prob / total, loss_prob / total)


def label_positions_with_sf(positions, depth=SF_DEPTH):
    """Label positions with Stockfish centipawn evaluations.

    Returns list of (board, cp_value, wdl_target) tuples.
    """
    from stockfish import Stockfish

    sf = Stockfish(SF_PATH, depth=depth, parameters={"Threads": SF_THREADS})
    labeled = []
    t0 = time.time()

    for i, board in enumerate(positions):
        try:
            sf.set_fen_position(board.fen())
            ev = sf.get_evaluation()
            if ev["type"] == "cp":
                cp = ev["value"]
            elif ev["type"] == "mate":
                cp = 10000 if ev["value"] > 0 else -10000
            else:
                continue

            wdl = cp_to_wdl(cp)
            labeled.append({"board": board, "cp": cp, "wdl": wdl})

        except Exception:
            continue

        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"    Labeled {i+1}/{len(positions)} ({rate:.0f}/s)")

    elapsed = time.time() - t0
    print(f"    Done: {len(labeled)} positions in {elapsed:.0f}s "
          f"({len(labeled)/elapsed:.0f}/s)")
    return labeled


# =====================================================================
# Phase 2: Value head fine-tuning
# =====================================================================

def train_value_head(model, labeled_data, device):
    """Fine-tune ONLY the value head on SF-calibrated WDL targets."""

    # Freeze everything except value_head
    for name, param in model.named_parameters():
        param.requires_grad = "value_head" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=VALUE_LR, weight_decay=0.01,
    )

    # Split into train/val (90/10)
    random.shuffle(labeled_data)
    split = int(0.9 * len(labeled_data))
    train_data = labeled_data[:split]
    val_data = labeled_data[split:]

    for epoch in range(VALUE_EPOCHS):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), VALUE_BATCH):
            chunk = train_data[i:i + VALUE_BATCH]
            boards = [d["board"] for d in chunk]
            # WDL targets: (win, draw, loss)
            wdl_targets = torch.tensor(
                [d["wdl"] for d in chunk], device=device, dtype=torch.float32,
            )

            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            value_logits = result["value_logits"]

            # KL divergence between predicted WDL and target WDL
            log_probs = F.log_softmax(value_logits, dim=-1)
            loss = F.kl_div(log_probs, wdl_targets, reduction="batchmean")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_data), VALUE_BATCH):
                chunk = val_data[i:i + VALUE_BATCH]
                boards = [d["board"] for d in chunk]
                wdl_targets = torch.tensor(
                    [d["wdl"] for d in chunk], device=device, dtype=torch.float32,
                )
                cp_values = [d["cp"] for d in chunk]

                batch_input = batch_boards_to_token_ids(boards, device)
                result = model(batch_input)
                value_logits = result["value_logits"]
                log_probs = F.log_softmax(value_logits, dim=-1)
                val_loss += F.kl_div(log_probs, wdl_targets, reduction="batchmean").item()

                # Check sign accuracy (does predicted W>L agree with cp>0?)
                wdl_pred = F.softmax(value_logits, dim=-1)
                pred_sign = (wdl_pred[:, 0] - wdl_pred[:, 2]).cpu()
                for j, cp in enumerate(cp_values):
                    val_total += 1
                    if (cp > 50 and pred_sign[j] > 0.05) or \
                       (cp < -50 and pred_sign[j] < -0.05) or \
                       (abs(cp) <= 50 and abs(pred_sign[j]) <= 0.1):
                        val_correct += 1

        avg_train = total_loss / max(n_batches, 1)
        avg_val = val_loss / max(len(val_data) // VALUE_BATCH, 1)
        sign_acc = val_correct / max(val_total, 1)
        print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f} "
              f"val_loss={avg_val:.4f} sign_acc={sign_acc:.1%}")

    # Unfreeze for downstream
    for param in model.parameters():
        param.requires_grad = True

    return model


# =====================================================================
# Phase 3: Search strategies (same as exp057)
# =====================================================================

def strategy_policy_argmax(model, board, device, **kw):
    probs, _ = model.get_policy_topk(board, device, k=1), None
    board_input = batch_boards_to_token_ids([board], device)
    result = model(board_input)
    logits = result["policy_logits"][0]
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")
    idx = logits.argmax().item()
    return IDX_TO_UCI[idx]


def strategy_value_rerank_k5(model, board, device, **kw):
    candidates = model.get_policy_topk(board, device, k=5)
    if not candidates:
        return strategy_policy_argmax(model, board, device)

    child_boards = []
    moves = []
    for m, p in candidates:
        child = board.copy()
        child.push(m)
        child_boards.append(child)
        moves.append(m)

    child_values = model.get_values_batch(child_boards, device)
    our_values = [-v for v in child_values]
    best_idx = max(range(len(our_values)), key=lambda i: our_values[i])
    return moves[best_idx].uci()


def strategy_alphabeta_2ply(model, board, device, top_k=5, **kw):
    our_candidates = model.get_policy_topk(board, device, k=top_k)
    if not our_candidates:
        return strategy_policy_argmax(model, board, device)

    best_move = None
    best_value = -2.0

    for our_move, _ in our_candidates:
        child = board.copy()
        child.push(our_move)

        if child.is_game_over():
            result = child.result()
            if result == "1-0":
                v = 1.0 if board.turn == chess.WHITE else -1.0
            elif result == "0-1":
                v = -1.0 if board.turn == chess.WHITE else 1.0
            else:
                v = 0.0
            if v > best_value:
                best_value = v
                best_move = our_move
            continue

        opp_candidates = model.get_policy_topk(child, device, k=top_k)
        if not opp_candidates:
            v = -model.get_value(child, device)
            if v > best_value:
                best_value = v
                best_move = our_move
            continue

        grandchild_boards = []
        for opp_move, _ in opp_candidates:
            gc = child.copy()
            gc.push(opp_move)
            grandchild_boards.append(gc)

        gc_values = model.get_values_batch(grandchild_boards, device)
        for i, gc in enumerate(grandchild_boards):
            if gc.is_game_over():
                result = gc.result()
                if result == "1-0":
                    gc_values[i] = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":
                    gc_values[i] = -1.0 if board.turn == chess.WHITE else 1.0
                else:
                    gc_values[i] = 0.0

        min_value = min(gc_values)
        if min_value > best_value:
            best_value = min_value
            best_move = our_move

    return best_move.uci() if best_move else strategy_policy_argmax(model, board, device)


# =====================================================================
# Game playing
# =====================================================================

def play_game(model, device, strategy_fn, sf_depth, opening_moves,
              strategy_kw=None):
    from stockfish import Stockfish
    sf = Stockfish(SF_PATH, depth=sf_depth)
    strategy_kw = strategy_kw or {}

    results = []
    for model_color in [chess.WHITE, chess.BLACK]:
        board = chess.Board()
        move_list = []

        for uci in opening_moves:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)
                move_list.append(uci)

        while not board.is_game_over() and len(move_list) < 200:
            if board.turn == model_color:
                move_uci = strategy_fn(model, board, device, **strategy_kw)
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    move = list(board.legal_moves)[0]
                    move_uci = move.uci()
            else:
                sf.set_fen_position(board.fen())
                move_uci = sf.get_best_move()
                move = chess.Move.from_uci(move_uci)

            board.push(move)
            move_list.append(move_uci)

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
            "termination": board.outcome().termination.name if board.outcome() else "max_moves",
        })

    return results


def run_search_games(model, device):
    """Run search games and return structured results."""
    strategies = {
        "policy_argmax": {"fn": strategy_policy_argmax, "kw": {}},
        "value_rerank_k5": {"fn": strategy_value_rerank_k5, "kw": {}},
        "alphabeta_2ply_k5": {"fn": strategy_alphabeta_2ply, "kw": {"top_k": 5}},
    }

    all_results = {}
    for sname, sspec in strategies.items():
        print(f"\n    Strategy: {sname}")
        strat_results = {}
        for sf_depth in SF_GAME_DEPTHS:
            wins = draws = losses = 0
            total_moves = 0
            depth_results = []

            for g in range(GAMES_PER_DEPTH // 2):
                opening = OPENINGS[g % len(OPENINGS)]
                game_results = play_game(
                    model, device, sspec["fn"], sf_depth, opening,
                    strategy_kw=sspec["kw"],
                )
                for gr in game_results:
                    depth_results.append(gr)
                    total_moves += gr["num_moves"]
                    if gr["outcome"] == 1.0:
                        wins += 1
                    elif gr["outcome"] == 0.5:
                        draws += 1
                    else:
                        losses += 1

            score = (wins + 0.5 * draws) / max(len(depth_results), 1)
            avg_moves = total_moves / max(len(depth_results), 1)
            strat_results[f"d{sf_depth}"] = {
                "wins": wins, "draws": draws, "losses": losses,
                "score": round(score, 3),
                "avg_moves": round(avg_moves),
            }
            print(f"      d{sf_depth}: W{wins}/D{draws}/L{losses} "
                  f"(score={score:.1%}, avg={avg_moves:.0f}mv)")

        all_results[sname] = strat_results
    return all_results


# =====================================================================
# Main
# =====================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)

    print(f"Device: {device}")
    print(f"Experiment: exp058_sf_value_head")
    print(f"Hypothesis: SF-calibrated value head improves search gameplay")

    # Load model
    print(f"\n[1/4] Loading model from {CHECKPOINT}")
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    hidden_dim = state["cls_token"].shape[-1]
    layer_keys = [k for k in state if "transformer.layers" in k
                  and "self_attn.in_proj_weight" in k]
    num_layers = len(layer_keys)

    model = ChessTransformerV2(
        hidden_dim=hidden_dim, num_layers=num_layers, num_heads=8,
    ).to(device)
    model.load_state_dict(state)
    print(f"  Loaded: {hidden_dim}d, {num_layers}L")

    # Phase 1: Run games BEFORE calibration (baseline)
    print(f"\n[2/4] Baseline games (WDL value head)...")
    baseline_games = run_search_games(model, device)

    # Phase 2: Label positions with Stockfish
    print(f"\n[3/4] Labeling {N_LABEL} positions with SF depth {SF_DEPTH}...")
    from hf_data import load_training_set
    raw_data = load_training_set(n=N_LABEL, split="train")
    positions = [d["board"] for d in raw_data]
    labeled = label_positions_with_sf(positions)

    # Diagnostics
    cp_values = [d["cp"] for d in labeled]
    n_winning = sum(1 for cp in cp_values if cp > 100)
    n_losing = sum(1 for cp in cp_values if cp < -100)
    n_equal = sum(1 for cp in cp_values if abs(cp) <= 100)
    print(f"  Distribution: winning={n_winning} ({100*n_winning/len(cp_values):.0f}%), "
          f"equal={n_equal} ({100*n_equal/len(cp_values):.0f}%), "
          f"losing={n_losing} ({100*n_losing/len(cp_values):.0f}%)")

    # Phase 3: Fine-tune value head
    print(f"\n[3b/4] Fine-tuning value head on SF targets...")
    model = train_value_head(model, labeled, device)

    # Save calibrated checkpoint
    ckpt_path = OUTPUT_DIR / "sf_calibrated_checkpoint.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # Phase 4: Run games AFTER calibration
    print(f"\n[4/4] Calibrated games (SF value head)...")
    calibrated_games = run_search_games(model, device)

    total_time = time.time() - t_start

    # Save results
    results = {
        "experiment": "exp058_sf_value_head",
        "hypothesis": "SF-calibrated value head improves search gameplay",
        "checkpoint": str(CHECKPOINT),
        "n_labeled": len(labeled),
        "sf_depth": SF_DEPTH,
        "cp_distribution": {
            "winning": n_winning, "equal": n_equal, "losing": n_losing,
        },
        "baseline_games": baseline_games,
        "calibrated_games": calibrated_games,
        "timing": {"total_s": round(total_time)},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print comparison
    print(f"\n{'='*70}")
    print(f" RESULTS: exp058_sf_value_head")
    print(f"{'='*70}")
    print(f"\n  {'':25} {'BASELINE (WDL)':>20} {'CALIBRATED (SF)':>20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    for sname in baseline_games:
        print(f"  {sname}:")
        for dk in ["d1", "d2", "d3"]:
            b = baseline_games[sname].get(dk, {})
            c = calibrated_games[sname].get(dk, {})
            b_str = f"W{b.get('wins',0)}/D{b.get('draws',0)}/L{b.get('losses',0)}"
            c_str = f"W{c.get('wins',0)}/D{c.get('draws',0)}/L{c.get('losses',0)}"
            b_score = b.get('score', 0)
            c_score = c.get('score', 0)
            delta = c_score - b_score
            marker = " +" if delta > 0 else (" -" if delta < 0 else "  ")
            print(f"    {dk}: {b_str:>15} → {c_str:>15} "
                  f"({b_score:.1%} → {c_score:.1%}){marker}")

    print(f"\n  Total time: {total_time:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
