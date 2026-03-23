"""exp059: Data scaling — generate 200K positions, train Medium joint model.

Hypothesis: Training the Medium spatial model (26M params) on ~250K combined
positions (200K generated + 47.5K HF) with joint policy+value will significantly
beat the 47.5K-only baseline (35.3% accuracy, 37.5% score at SF d1).

Evidence: Sessions 8-11 showed data scaling is the dominant lever for the
chess-native transformer (50K→460K gave ~13pp gain). The Medium model on
47.5K is severely data-starved.

Plan:
  Phase 1: Generate 200K diverse positions using build_dataset generators (~60s)
  Phase 2: Label in parallel with 12 SF workers × 4 threads each (~2 min)
  Phase 3: Combine with cached HF dataset (47.5K) → ~245K total
  Phase 4: Train Medium model with joint policy+value loss (6 epochs, ~25 min)
  Phase 5: Evaluate + play search games vs Stockfish

Experiment contract:
  - Hypothesis: ~250K data > 47.5K data for Medium model
  - Primary metric: top-1 accuracy, gameplay score at SF d1-d3
  - Evaluation set: HF test split (2500 positions, held out)
  - Seeds: 42
  - Runtime target: ~30 minutes on RTX 2000 Ada (16GB)
  - Device: CUDA + 48 CPU threads for SF labeling
"""

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

OUTPUT_DIR = Path("outputs/exp059_data_scaling")
SF_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# ── Data generation config ──
N_GENERATE = 200000
SF_LABEL_DEPTH = 6      # depth 6: good accuracy + fast (~1000 pos/s)
SF_THREADS = 8           # single Stockfish instance, 8 threads

# ── Model config (Medium, same as exp053/055) ──
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
DROPOUT = 0.1
HEAD_DIM = 256

# ── Training config ──
EPOCHS = 6
BATCH_SIZE = 128
LR = 2e-4
WARMUP_FRAC = 0.05
VALUE_WEIGHT = 0.5
SEED = 42

# ── Search game config ──
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


# =====================================================================
# Phase 1: Position generation (reuse build_dataset.py generators)
# =====================================================================

def generate_diverse_positions(n, seed=42):
    """Generate n diverse positions from 5 sources. Returns list of (Board, source)."""
    from build_dataset import generate_positions
    return generate_positions(n, seed=seed)


# =====================================================================
# Phase 2: SF labeling (single process, reliable)
# =====================================================================

def label_positions(positions, depth=SF_LABEL_DEPTH, threads=SF_THREADS):
    """Label positions with Stockfish best move + eval. Single process, robust."""
    from stockfish import Stockfish

    sf = Stockfish(
        path=SF_PATH,
        depth=depth,
        parameters={"Threads": threads, "Hash": 256},
    )

    results = []
    t0 = time.time()

    for i, (board, source) in enumerate(positions):
        try:
            fen = board.fen()
            sf.set_fen_position(fen)
            top_moves = sf.get_top_moves(5)
            if not top_moves:
                continue

            best = top_moves[0]
            best_move = best["Move"]

            if best_move not in UCI_TO_IDX:
                continue
            move_obj = chess.Move.from_uci(best_move)
            if move_obj not in board.legal_moves:
                continue

            eval_type = "mate" if best.get("Mate") is not None else "cp"
            eval_value = best["Mate"] if eval_type == "mate" else best.get("Centipawn", 0)

            if eval_type == "mate":
                wdl = (1.0, 0.0, 0.0) if eval_value > 0 else (0.0, 0.0, 1.0)
            else:
                k = 1.0 / 111.7
                win = 1.0 / (1.0 + math.exp(-k * eval_value))
                loss_p = 1.0 - win
                draw = max(0.0, 0.5 - abs(win - 0.5)) * 2
                total = win + draw + loss_p
                wdl = (win / total, draw / total, loss_p / total)

            # Phase classification
            material = 0
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.piece_type != chess.KING:
                    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                            chess.ROOK: 5, chess.QUEEN: 9}
                    material += vals.get(piece.piece_type, 0)
            if material >= 50 and board.fullmove_number <= 12:
                phase = "opening"
            elif material <= 26:
                phase = "endgame"
            else:
                phase = "middlegame"

            top_moves_data = []
            for m in top_moves:
                entry = {"uci": m["Move"]}
                if m.get("Mate") is not None:
                    entry["mate"] = m["Mate"]
                else:
                    entry["cp"] = m.get("Centipawn", 0)
                top_moves_data.append(entry)

            results.append({
                "fen": fen,
                "best_move": best_move,
                "eval_type": eval_type,
                "eval_value": eval_value,
                "wdl": wdl,
                "phase": phase,
                "source": source,
                "top_moves": top_moves_data,
            })

        except Exception:
            # Recreate SF on crash
            try:
                sf = Stockfish(path=SF_PATH, depth=depth,
                               parameters={"Threads": threads, "Hash": 256})
            except Exception:
                pass
            continue

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(positions) - i - 1) / rate
            print(f"    {i+1:,}/{len(positions):,} | "
                  f"{len(results):,} labeled | {rate:.0f}/s | ETA {eta/60:.1f}m")

    elapsed = time.time() - t0
    print(f"  Done: {len(results):,} / {len(positions):,} in {elapsed:.0f}s "
          f"({len(results)/max(elapsed,1):.0f}/s)")
    return results


# =====================================================================
# Model architecture (ChessTransformerV2 with value head)
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
    def get_values_batch(self, boards, device):
        self.eval()
        board_input = batch_boards_to_token_ids(boards, device)
        result = self(board_input)
        wdl = F.softmax(result["value_logits"], dim=-1)
        return (wdl[:, 0] - wdl[:, 2]).cpu().tolist()


# =====================================================================
# Training
# =====================================================================

def prepare_training_data(generated_labeled, hf_data):
    """Combine generated SF-labeled data with HF data into unified format."""
    combined = []

    # HF data (has board + move objects)
    for d in hf_data:
        combined.append({
            "board": d["board"],
            "move": d["move"],
            "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
            "phase": d.get("phase", "unknown"),
        })

    # Generated data (has FEN + UCI strings)
    for d in generated_labeled:
        try:
            board = chess.Board(d["fen"])
            move = chess.Move.from_uci(d["best_move"])
            if move not in board.legal_moves:
                continue
            if d["best_move"] not in UCI_TO_IDX:
                continue
            combined.append({
                "board": board,
                "move": move,
                "wdl": d["wdl"],
                "phase": d["phase"],
            })
        except Exception:
            continue

    return combined


def train_model(train_data, eval_data, device):
    """Train Medium model with joint policy + value loss."""
    torch.manual_seed(SEED)
    random.seed(SEED)

    model = ChessTransformerV2(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        dropout=DROPOUT, head_dim=HEAD_DIM,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

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
        total_policy_loss = total_value_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor(
                [move_to_index(d["move"]) for d in chunk], device=device,
            )
            wdl_targets = torch.tensor(
                [d["wdl"] for d in chunk], device=device, dtype=torch.float32,
            )

            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)

            policy_loss = F.cross_entropy(result["policy_logits"], targets)
            value_log_probs = F.log_softmax(result["value_logits"], dim=-1)
            value_loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")

            loss = policy_loss + VALUE_WEIGHT * value_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        avg_pl = total_policy_loss / max(n_batches, 1)
        avg_vl = total_value_loss / max(n_batches, 1)
        ev = evaluate_rich(model, eval_data, device)
        ep_time = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "policy_loss": round(avg_pl, 4),
            "value_loss": round(avg_vl, 4),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in ev.items()},
            "time_s": round(ep_time),
        })

        marker = " *" if ev["accuracy"] > best_acc else ""
        print(f"  Ep{epoch+1}: pl={avg_pl:.3f} vl={avg_vl:.3f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} "
              f"val_acc={ev.get('value_accuracy', 0):.1%} "
              f"sf_rank={ev['mean_sf_rank']:.1f} [{ep_time:.0f}s]{marker}")

        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best checkpoint
    if best_state:
        ckpt_path = OUTPUT_DIR / "best_checkpoint.pt"
        torch.save(best_state, ckpt_path)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  Saved: {ckpt_path} (best acc: {best_acc:.1%})")

    return model, history, best_acc


def evaluate_rich(model, eval_data, device, batch_size=128):
    """Rich evaluation with phase breakdown and value accuracy."""
    model.eval()
    correct = top3_correct = total = 0
    entropy_sum = sf_rank_sum = 0.0
    phase_stats = {}
    val_correct = val_total = 0

    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            phases = [d.get("phase", "unknown") for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]

            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()

            # Value accuracy
            if "value_logits" in result:
                wdl_pred = F.softmax(result["value_logits"], dim=-1)
                for j, d in enumerate(chunk):
                    wdl = d.get("wdl", None)
                    if wdl:
                        val_total += 1
                        pred_class = wdl_pred[j].argmax().item()
                        true_class = max(range(3), key=lambda k: wdl[k])
                        if pred_class == true_class:
                            val_correct += 1

            for j, t in enumerate(targets):
                total += 1
                if preds[j] == t:
                    correct += 1
                if t in top3s[j]:
                    top3_correct += 1
                p = probs[j]
                legal_p = p[p > 0]
                entropy_sum += -(legal_p * legal_p.log()).sum().item()
                sorted_idx = logits[j].argsort(descending=True).cpu().tolist()
                sf_rank = sorted_idx.index(t) + 1 if t in sorted_idx else len(sorted_idx)
                sf_rank_sum += sf_rank
                ph = phases[j]
                if ph not in phase_stats:
                    phase_stats[ph] = {"correct": 0, "total": 0}
                phase_stats[ph]["total"] += 1
                if preds[j] == t:
                    phase_stats[ph]["correct"] += 1

    result = {
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "mean_entropy": entropy_sum / max(total, 1),
        "mean_sf_rank": sf_rank_sum / max(total, 1),
        "n_eval": total,
    }
    if val_total > 0:
        result["value_accuracy"] = val_correct / val_total
    for ph, ps in sorted(phase_stats.items()):
        result[f"acc_{ph}"] = ps["correct"] / max(ps["total"], 1)
        result[f"n_{ph}"] = ps["total"]
    return result


# =====================================================================
# Search game play
# =====================================================================

def strategy_policy_argmax(model, board, device, **kw):
    model.eval()
    with torch.no_grad():
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


def play_game(model, device, strategy_fn, sf_depth, opening_moves):
    from stockfish import Stockfish
    sf = Stockfish(SF_PATH, depth=sf_depth)
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
                move_uci = strategy_fn(model, board, device)
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
            "outcome": outcome, "result": result,
            "num_moves": len(move_list),
            "termination": board.outcome().termination.name if board.outcome() else "max_moves",
        })
    return results


def run_games(model, device):
    strategies = {
        "policy_argmax": strategy_policy_argmax,
        "value_rerank_k5": strategy_value_rerank_k5,
    }
    all_results = {}
    for sname, sfn in strategies.items():
        print(f"    {sname}:")
        strat_results = {}
        for sf_depth in SF_GAME_DEPTHS:
            wins = draws = losses = total_moves = 0
            for g in range(GAMES_PER_DEPTH // 2):
                opening = OPENINGS[g % len(OPENINGS)]
                game_results = play_game(model, device, sfn, sf_depth, opening)
                for gr in game_results:
                    total_moves += gr["num_moves"]
                    if gr["outcome"] == 1.0: wins += 1
                    elif gr["outcome"] == 0.5: draws += 1
                    else: losses += 1
            n_games = max(wins + draws + losses, 1)
            score = (wins + 0.5 * draws) / n_games
            strat_results[f"d{sf_depth}"] = {
                "wins": wins, "draws": draws, "losses": losses,
                "score": round(score, 3),
                "avg_moves": round(total_moves / n_games),
            }
            print(f"      d{sf_depth}: W{wins}/D{draws}/L{losses} "
                  f"({score:.1%}, avg {total_moves//n_games}mv)")
        all_results[sname] = strat_results
    return all_results


# =====================================================================
# Main
# =====================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {device}")
    print(f"Experiment: exp059_data_scaling")
    print(f"Hypothesis: ~250K combined data >> 47.5K on Medium model")
    print(f"Target: beat exp053 35.3% acc, exp054 37.5% score at SF d1")
    print()

    # Phase 1: Generate positions
    print(f"[1/5] Generating {N_GENERATE:,} diverse positions...")
    t_gen = time.time()
    positions = generate_diverse_positions(N_GENERATE, seed=SEED)
    gen_time = time.time() - t_gen
    print(f"  Generated {len(positions):,} in {gen_time:.0f}s")

    # Phase 2: Label with Stockfish (sequential, single process)
    print(f"\n[2/5] Labeling with SF depth {SF_LABEL_DEPTH} "
          f"({SF_THREADS} threads)...")
    t_label = time.time()
    generated_labeled = label_positions(positions, depth=SF_LABEL_DEPTH,
                                        threads=SF_THREADS)
    label_time = time.time() - t_label

    # Save labeled data for reuse
    labeled_path = OUTPUT_DIR / "generated_200k.jsonl"
    with open(labeled_path, "w") as f:
        for d in generated_labeled:
            row = {k: v for k, v in d.items()}
            row["wdl"] = list(row["wdl"])
            f.write(json.dumps(row) + "\n")
    print(f"  Saved: {labeled_path} ({len(generated_labeled):,} positions)")

    # Phase stats
    phase_counts = {}
    source_counts = {}
    for d in generated_labeled:
        phase_counts[d["phase"]] = phase_counts.get(d["phase"], 0) + 1
        source_counts[d["source"]] = source_counts.get(d["source"], 0) + 1
    print(f"  Phases: {dict(sorted(phase_counts.items()))}")
    print(f"  Sources: {dict(sorted(source_counts.items()))}")

    # Phase 3: Load HF data and combine
    print(f"\n[3/5] Loading HF data and combining...")
    from hf_data import load_training_set, load_eval_set
    hf_train = load_training_set()
    hf_eval = load_eval_set(n=2500)
    print(f"  HF: {len(hf_train)} train, {len(hf_eval)} eval")

    train_data = prepare_training_data(generated_labeled, hf_train)
    eval_data = [{
        "board": d["board"], "move": d["move"],
        "wdl": d.get("wdl", (0.5, 0.5, 0.0)),
        "phase": d.get("phase", "unknown"),
    } for d in hf_eval]
    print(f"  Combined train: {len(train_data):,}")

    # Phase 4: Train
    print(f"\n[4/5] Training Medium model ({EPOCHS} epochs, bs={BATCH_SIZE})...")
    model, history, best_acc = train_model(train_data, eval_data, device)

    # Phase 5: Play games
    print(f"\n[5/5] Playing games vs Stockfish...")
    game_results = run_games(model, device)

    total_time = time.time() - t_start

    # Save results
    results = {
        "experiment": "exp059_data_scaling",
        "hypothesis": "~250K combined data >> 47.5K for Medium model",
        "baseline": "exp053: 35.3% acc, exp054 rerank_k5: W0/D6/L2 at SF d1",
        "seed": SEED,
        "config": {
            "encoder_dim": ENCODER_DIM, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "value_weight": VALUE_WEIGHT,
        },
        "data": {
            "generated": len(generated_labeled),
            "hf_train": len(hf_train),
            "combined": len(train_data),
            "eval": len(eval_data),
            "phase_distribution": phase_counts,
            "source_distribution": source_counts,
        },
        "training": {
            "best_accuracy": round(best_acc, 4),
            "history": history,
        },
        "games": game_results,
        "timing": {
            "generation_s": round(gen_time),
            "labeling_s": round(label_time),
            "total_s": round(total_time),
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*70}")
    print(f" RESULTS: exp059_data_scaling")
    print(f"{'='*70}")
    print(f"  Data: {len(generated_labeled):,} generated + {len(hf_train):,} HF "
          f"= {len(train_data):,} total")
    print(f"  Best accuracy: {best_acc:.1%} (baseline: 35.3%)")
    print(f"  Delta: {(best_acc - 0.353)*100:+.1f}pp")
    print(f"\n  Gameplay vs Stockfish:")
    for sname, sdata in game_results.items():
        print(f"    {sname}:")
        for dk, dv in sdata.items():
            print(f"      {dk}: W{dv['wins']}/D{dv['draws']}/L{dv['losses']} "
                  f"({dv['score']:.1%})")
    print(f"\n  Generation: {gen_time:.0f}s | Labeling: {label_time:.0f}s | "
          f"Total: {total_time:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
