"""exp043: Centipawn-aware SF distillation with KL constraint.

Hypothesis: Human move accuracy (51.4%) has a ceiling because the dataset
includes moves from all skill levels. Instead of predicting human moves,
optimize for move QUALITY measured by centipawn loss. Use SF multi-PV soft
targets + KL constraint to improve move quality without catastrophic forgetting.

Key insight from prior experiments:
  - exp034 SF distillation: 52.9% on SF-labeled data, but catastrophic forgetting
  - exp041 KL constraint: preserves model perfectly
  - Combining them: SF soft targets + KL should improve quality AND preserve model

Design:
  1. Evaluate exp032 baseline centipawn loss (new metric!)
  2. Label 20K positions with SF multi-PV (top-5 moves + evals)
  3. Create soft policy targets from SF evals
  4. KL-constrained distillation: loss = CE(sf_target) + β*KL(p||p_ref)
  5. Measure both human accuracy AND centipawn loss

Primary metric: centipawn loss of model's chosen move (NEW)
Secondary metric: human accuracy (for comparison)
Time budget: ~10 min
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
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move

OUTPUT_DIR = Path("outputs/exp043_cp_distill")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Config
NUM_LABEL = 5000     # Positions to label with SF (keep small for speed)
NUM_EVAL_CP = 500    # Positions for centipawn evaluation
NUM_EVAL_ACC = 1000  # Positions for accuracy evaluation
EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-5
KL_COEFF = 0.5      # Strong KL constraint
SF_DEPTH = 5         # Fast enough, decent quality
SF_MULTI_PV = 5      # Top-5 moves from SF
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 8
GAME_SF_DEPTH = 3


# === Architecture (same as exp032) ===

def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
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
    def __init__(self, hidden_size, head_dim=256):
        super().__init__()
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer('from_sqs', from_sqs)
        self.register_buffer('to_sqs', to_sqs)
        self.register_buffer('promo_types', promo_types)

    def forward(self, hidden_states):
        sq_hidden = hidden_states[:, 3:67, :]
        global_hidden = hidden_states[:, 0, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        from_proj = self.from_proj(from_feats)
        to_proj = self.to_proj(to_feats)
        global_proj = self.global_proj(global_hidden).unsqueeze(1)
        promo_feats = self.promo_embed(self.promo_types)
        combined = from_proj * to_proj + global_proj + promo_feats.unsqueeze(0)
        return self.score_proj(F.relu(combined)).squeeze(-1)


class ChessTransformer(nn.Module):
    def __init__(self, encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.encoder = LearnedBoardEncoder(embed_dim=encoder_dim)
        self.input_proj = nn.Linear(encoder_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 67, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = SpatialPolicyHead(hidden_dim, head_dim=256)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 3),
        )
        self.hidden_dim = hidden_dim

    def forward(self, board_input, **kw):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens) + self.pos_embed
        hidden = self.transformer(hidden)
        hidden = self.norm(hidden)
        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)
        return {"policy_logits": policy_logits, "value_logits": value_logits}

    @torch.no_grad()
    def predict_move(self, board):
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return index_to_move(probs.argmax().item()), probs


# === Data Utilities ===

def build_old_move_mapping():
    PROMO_TYPES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    def inside(idx, f_from): return 0 <= idx < 64 and abs((idx % 8) - f_from) <= 2
    ucis = set()
    for f in range(64):
        ff = f % 8
        for d in (+8, -8, +1, -1):
            t = f + d
            while inside(t, ff): ucis.add(chess.Move(f, t).uci()); t += d
        for d in (+9, -9, +7, -7):
            t = f + d
            while inside(t, ff): ucis.add(chess.Move(f, t).uci()); t += d
        for off in (+17, +15, +10, +6, -6, -10, -15, -17):
            t = f + off
            if inside(t, ff): ucis.add(chess.Move(f, t).uci())
    for f in range(64):
        file_, rank = f % 8, f // 8
        if rank == 6:
            for df in (-9, -8, -7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES: ucis.add(chess.Move(f, t, promotion=p).uci())
        if rank == 1:
            for df in (+9, +8, +7):
                t = f + df
                if 0 <= t < 64 and abs((t % 8) - file_) <= 1:
                    for p in PROMO_TYPES: ucis.add(chess.Move(f, t, promotion=p).uci())
    return sorted(ucis)


_PIECE2CHESS = {
    1: (chess.PAWN, chess.WHITE), 2: (chess.KNIGHT, chess.WHITE),
    3: (chess.BISHOP, chess.WHITE), 4: (chess.ROOK, chess.WHITE),
    5: (chess.QUEEN, chess.WHITE), 6: (chess.KING, chess.WHITE),
    7: (chess.PAWN, chess.BLACK), 8: (chess.KNIGHT, chess.BLACK),
    9: (chess.BISHOP, chess.BLACK), 10: (chess.ROOK, chess.BLACK),
    11: (chess.QUEEN, chess.BLACK), 12: (chess.KING, chess.BLACK),
}


def hf_sample_to_board(board_flat, turn_raw):
    board = chess.Board(fen=None)
    board.clear()
    for idx in range(64):
        piece_val = board_flat[idx]
        if piece_val > 0:
            row = idx // 8; col = idx % 8
            rank = 7 - row; sq = rank * 8 + col
            pt, color = _PIECE2CHESS[piece_val]
            board.set_piece_at(sq, chess.Piece(pt, color))
    board.turn = bool(turn_raw)
    return board


def prepare_hf_data(dataset, old_sorted_uci, n, offset=0):
    data, skipped = [], 0
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(data) >= n: break
        s = dataset[i]
        old_uci = old_sorted_uci[s["move_id"]]
        if old_uci not in UCI_TO_IDX: skipped += 1; continue
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            move = chess.Move.from_uci(old_uci)
        except ValueError: skipped += 1; continue
        if move not in board.legal_moves: skipped += 1; continue
        data.append({"board": board, "move": move})
    return data


# === SF Labeling ===

def label_positions_with_sf(boards, sf_depth, multi_pv):
    """Label each position with SF top-K moves and evaluations.
    
    Returns list of dicts with:
        board: chess.Board
        sf_moves: list of (uci, centipawn_eval) tuples
    """
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 128, "MultiPV": multi_pv})
    labeled = []
    for i, board in enumerate(boards):
        try:
            sf.set_fen_position(board.fen())
            top_moves = sf.get_top_moves(multi_pv)
            if not top_moves:
                continue
            moves_with_eval = []
            for m in top_moves:
                uci = m["Move"]
                if uci not in UCI_TO_IDX:
                    continue
                # Get centipawn evaluation
                if m.get("Mate") is not None:
                    cp = 10000 if m["Mate"] > 0 else -10000
                else:
                    cp = m.get("Centipawn", 0)
                moves_with_eval.append((uci, cp))
            if moves_with_eval:
                labeled.append({
                    "board": board,
                    "sf_moves": moves_with_eval,
                    "sf_best_cp": moves_with_eval[0][1],
                })
        except Exception:
            continue
        if (i + 1) % 500 == 0:
            print(f"      Labeled {i+1}/{len(boards)} positions")
    return labeled


def sf_moves_to_soft_target(sf_moves, temperature=100.0):
    """Convert SF move evaluations to a soft probability distribution.
    
    Uses softmax over centipawn values rescaled by temperature.
    Higher temperature = more uniform spread.
    """
    target = torch.zeros(VOCAB_SIZE)
    if not sf_moves:
        return target
    
    cps = torch.tensor([cp for _, cp in sf_moves], dtype=torch.float32)
    # Normalize centipawns to reasonable scale
    logits = cps / temperature
    probs = F.softmax(logits, dim=-1)
    
    for j, (uci, _) in enumerate(sf_moves):
        if uci in UCI_TO_IDX:
            target[UCI_TO_IDX[uci]] = probs[j].item()
    
    # Renormalize in case some moves weren't in vocab
    if target.sum() > 0:
        target = target / target.sum()
    return target


# === Centipawn Loss Evaluation ===

def evaluate_centipawn_loss(model, positions, device, sf_depth=5):
    """Evaluate average centipawn loss of model's chosen moves.
    
    For each position:
    1. Get model's top move
    2. Get SF evaluation of that move AND SF's best move
    3. CPL = SF_best_cp - SF_model_move_cp
    """
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 128, "MultiPV": 1})

    model.eval()
    total_cpl = 0.0
    count = 0
    sf_match = 0

    for i, pos in enumerate(positions):
        board = pos["board"]
        try:
            # Get model's move
            pred_move, _ = model.predict_move(board)
            pred_uci = pred_move.uci()

            # Get SF best move and its eval
            sf.set_fen_position(board.fen())
            sf_best = sf.get_best_move()
            sf_eval_best = sf.get_evaluation()
            best_cp = sf_eval_best.get("value", 0) if sf_eval_best["type"] == "cp" else (10000 if sf_eval_best.get("value", 0) > 0 else -10000)

            # Get SF eval after model's move
            board_after = board.copy()
            board_after.push(pred_move)
            sf.set_fen_position(board_after.fen())
            sf_eval_after = sf.get_evaluation()
            model_cp = sf_eval_after.get("value", 0) if sf_eval_after["type"] == "cp" else (10000 if sf_eval_after.get("value", 0) > 0 else -10000)
            # After model's move, eval is from opponent's POV
            model_cp = -model_cp

            cpl = max(0, best_cp - model_cp)
            total_cpl += cpl
            count += 1

            if pred_uci == sf_best:
                sf_match += 1

        except Exception:
            continue

        if (i + 1) % 100 == 0:
            avg = total_cpl / max(count, 1)
            print(f"      CPL eval: {i+1}/{len(positions)}, avg={avg:.1f}cp")

    return {
        "avg_centipawn_loss": round(total_cpl / max(count, 1), 1),
        "sf_match_rate": round(sf_match / max(count, 1), 4),
        "positions_evaluated": count,
    }


# === Accuracy Evaluation ===

def evaluate_accuracy(model, eval_data, device, batch_size=128):
    model.eval()
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            boards = [d["board"] for d in chunk]
            targets = [move_to_index(d["move"]) for d in chunk]
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]
            for j, board in enumerate(boards):
                mask = legal_move_mask(board).to(device)
                logits[j, ~mask] = float("-inf")
            preds = logits.argmax(dim=-1).cpu().tolist()
            top3s = logits.topk(3, dim=-1).indices.cpu().tolist()
            for j, target_idx in enumerate(targets):
                total += 1
                if preds[j] == target_idx: correct += 1
                if target_idx in top3s[j]: top3_correct += 1
    return {
        "accuracy": round(correct / max(total, 1), 4),
        "top3_accuracy": round(top3_correct / max(total, 1), 4),
    }


def play_game_vs_stockfish(model, sf_depth, model_color, device, max_moves=100):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    model.eval()
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            pred, _ = model.predict_move(board)
            if pred not in board.legal_moves:
                pred = random.choice(list(board.legal_moves))
            board.push(pred)
        else:
            sf.set_fen_position(board.fen())
            sf_uci = sf.get_best_move()
            if sf_uci is None: break
            board.push(chess.Move.from_uci(sf_uci))
    result = board.result()
    winner = "white" if result == "1-0" else "black" if result == "0-1" else "draw"
    model_result = (
        "win" if (winner == "white" and model_color == chess.WHITE) or
                 (winner == "black" and model_color == chess.BLACK)
        else "loss" if winner != "draw" else "draw"
    )
    term = board.outcome().termination.name if board.outcome() else "max_moves"
    return {
        "model_color": "white" if model_color else "black",
        "result": result, "model_result": model_result,
        "moves": board.fullmove_number, "termination": term,
    }


# === Main ===

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp043_cp_distill")

    # --- Load models ---
    print("\n[1/6] Loading exp032 checkpoint...")
    def make_model():
        return ChessTransformer(
            encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
        ).to(device)

    model = make_model()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    # Frozen reference for KL
    ref_model = make_model()
    ref_model.load_state_dict(ckpt["model_state"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")

    # --- Load eval data ---
    print("\n[2/6] Loading data...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()

    # Eval data from end of dataset (real positions only)
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL_ACC,
                                offset=len(ds) - NUM_EVAL_ACC * 3)
    cp_eval_data = eval_data[:NUM_EVAL_CP]  # Subset for CPL eval

    # Training positions from beginning
    train_positions = prepare_hf_data(ds, old_sorted_uci, NUM_LABEL, offset=0)
    print(f"  Got {len(eval_data)} eval, {len(cp_eval_data)} CPL eval, "
          f"{len(train_positions)} train positions")

    # --- Baseline ---
    print("\n[3/6] Baseline evaluation...")
    baseline_acc = evaluate_accuracy(model, eval_data, device)
    print(f"  Human accuracy: {baseline_acc['accuracy']:.1%} "
          f"top3={baseline_acc['top3_accuracy']:.1%}")

    print("  Evaluating baseline centipawn loss...")
    baseline_cpl = evaluate_centipawn_loss(model, cp_eval_data, device, sf_depth=SF_DEPTH)
    print(f"  Avg CPL: {baseline_cpl['avg_centipawn_loss']}cp, "
          f"SF match: {baseline_cpl['sf_match_rate']:.1%}")

    # --- Label training data with SF ---
    print(f"\n[4/6] Labeling {len(train_positions)} positions with SF d{SF_DEPTH} "
          f"(multi-PV={SF_MULTI_PV})...")
    label_t0 = time.time()
    boards_for_labeling = [d["board"] for d in train_positions]
    labeled = label_positions_with_sf(boards_for_labeling, SF_DEPTH, SF_MULTI_PV)
    label_time = time.time() - label_t0
    print(f"  Labeled {len(labeled)} positions in {label_time:.0f}s "
          f"({len(labeled)/label_time:.0f} pos/s)")

    # Create soft targets
    print("  Creating soft targets...")
    train_samples = []
    for item in labeled:
        soft_target = sf_moves_to_soft_target(item["sf_moves"], temperature=100.0)
        if soft_target.sum() > 0:
            train_samples.append({
                "board": item["board"],
                "soft_target": soft_target,
            })
    print(f"  Created {len(train_samples)} training samples with soft SF targets")

    # --- KL-Constrained Distillation ---
    print(f"\n[5/6] KL-Constrained SF Distillation: "
          f"{EPOCHS} epochs, lr={LR}, β={KL_COEFF}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_cpl = baseline_cpl['avg_centipawn_loss']
    best_state = None
    epoch_history = []

    for epoch in range(EPOCHS):
        ep_t0 = time.time()
        model.train()
        random.shuffle(train_samples)

        total_distill_loss = 0.0
        total_kl_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_samples), BATCH_SIZE):
            chunk = train_samples[i:i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            targets = torch.stack([d["soft_target"] for d in chunk]).to(device)

            batch_input = batch_boards_to_token_ids(boards, device)

            # Current policy
            result = model(batch_input)
            curr_logits = result["policy_logits"]

            # Reference policy
            with torch.no_grad():
                ref_result = ref_model(batch_input)
                ref_logits = ref_result["policy_logits"]

            # Apply legal move masking
            batch_distill = torch.tensor(0.0, device=device)
            batch_kl = torch.tensor(0.0, device=device)
            valid = 0

            for j in range(len(boards)):
                mask = legal_move_mask(boards[j]).to(device)
                curr_legal_logits = curr_logits[j].clone()
                curr_legal_logits[~mask] = float("-inf")
                ref_legal_logits = ref_logits[j].clone()
                ref_legal_logits[~mask] = float("-inf")

                curr_log_probs = F.log_softmax(curr_legal_logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_legal_logits, dim=-1)
                curr_probs = curr_log_probs.exp()

                # Distillation: KL(target || current) over legal moves
                target_legal = targets[j].clone()
                target_legal[~mask] = 0.0
                if target_legal.sum() > 0:
                    target_legal = target_legal / target_legal.sum()
                    # Only compute where target > 0
                    nonzero = target_legal > 0
                    distill = -(target_legal[nonzero] * curr_log_probs[nonzero]).sum()
                    batch_distill = batch_distill + distill

                # KL(current || reference) over legal moves
                kl = (curr_probs[mask] * (curr_log_probs[mask] - ref_log_probs[mask])).sum()
                batch_kl = batch_kl + kl
                valid += 1

            if valid == 0:
                continue

            loss = (batch_distill + KL_COEFF * batch_kl) / valid
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_distill_loss += (batch_distill / valid).item()
            total_kl_loss += (batch_kl / valid).item()
            n_batches += 1

        avg_distill = total_distill_loss / max(n_batches, 1)
        avg_kl = total_kl_loss / max(n_batches, 1)

        # Evaluate
        eval_acc = evaluate_accuracy(model, eval_data, device)
        eval_cpl = evaluate_centipawn_loss(model, cp_eval_data, device, sf_depth=SF_DEPTH)
        ep_time = time.time() - ep_t0

        ep_info = {
            "epoch": epoch + 1,
            "distill_loss": round(avg_distill, 4),
            "kl_loss": round(avg_kl, 6),
            "accuracy": eval_acc["accuracy"],
            "top3_accuracy": eval_acc["top3_accuracy"],
            "avg_cpl": eval_cpl["avg_centipawn_loss"],
            "sf_match": eval_cpl["sf_match_rate"],
            "time_s": round(ep_time),
        }
        epoch_history.append(ep_info)

        print(f"  Epoch {epoch+1}: distill={avg_distill:.4f} kl={avg_kl:.6f}")
        print(f"    Acc: {eval_acc['accuracy']:.1%} "
              f"CPL: {eval_cpl['avg_centipawn_loss']}cp "
              f"SF match: {eval_cpl['sf_match_rate']:.1%} [{ep_time:.0f}s]")

        if eval_cpl['avg_centipawn_loss'] < best_cpl:
            best_cpl = eval_cpl['avg_centipawn_loss']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best by CPL (not accuracy!)
    if best_state:
        model.load_state_dict(best_state)
        print(f"  Restored best model by CPL: {best_cpl}cp")

    # --- Games ---
    print(f"\n[6/6] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}...")
    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(model, GAME_SF_DEPTH, color, device)
        game_results.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv "
              f"({r['termination']})")

    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")
    print(f"  Score: W{wins}/D{draws}/L{losses}")

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp043_cp_distill",
        "hypothesis": "Centipawn-aware SF distillation with KL constraint",
        "seed": SEED,
        "config": {
            "num_label": NUM_LABEL, "sf_depth": SF_DEPTH, "multi_pv": SF_MULTI_PV,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "kl_coeff": KL_COEFF,
        },
        "baseline": {
            "accuracy": baseline_acc,
            "centipawn_loss": baseline_cpl,
        },
        "training": epoch_history,
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "timing": {"total_s": total_time, "labeling_s": label_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp043_cp_distill")
    print(f" Baseline: {baseline_acc['accuracy']:.1%} acc, "
          f"{baseline_cpl['avg_centipawn_loss']}cp CPL, "
          f"SF match={baseline_cpl['sf_match_rate']:.1%}")
    for h in epoch_history:
        print(f"   Ep{h['epoch']}: acc={h['accuracy']:.1%} "
              f"CPL={h['avg_cpl']}cp SF={h['sf_match']:.1%}")
    print(f" Best CPL: {best_cpl}cp")
    print(f" Games: W{wins}/D{draws}/L{losses}")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
