"""exp037: Fix value head with SF evaluations, then test search.

Hypothesis: The value head (trained on noisy game outcomes) is mis-calibrated,
causing search to hurt game play (exp036). Fine-tuning the value head on
Stockfish centipawn evaluations will calibrate it properly, enabling
1-ply search to outperform pure policy argmax.

Background:
  - exp036: search HURTS — argmax W0/D1/L7, search W0/D0/L8
  - Value head trained on {loss=0, draw=1, win=2} from game outcomes
  - Game outcomes are noisy (amateur games, side-to-move doesn't imply quality)
  - SF centipawn evaluations are positionally accurate

Design:
  Phase 1: Fine-tune value head on SF evals
    - Take 20K positions from training data
    - Get SF centipawn evaluation for each
    - Convert to WDL target: cp>100→win, cp<-100→loss, else→draw
    - Freeze policy, train ONLY value head for 5 epochs
  Phase 2: Test search with calibrated value head
    - argmax (baseline) vs top-5 search vs top-10 search
    - 8 games each vs SF d3

Primary metric: games vs SF d3 with search
Time budget: ~10 min (SF evals ~1 min, value training ~1 min, games ~5 min)
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

OUTPUT_DIR = Path("outputs/exp037_fix_value_head")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_VALUE_TRAIN = 20_000
SF_DEPTH = 8
VALUE_EPOCHS = 5
VALUE_BATCH = 256
VALUE_LR = 1e-3  # Higher LR since we're only training a small head
NUM_GAMES = 8
GAME_SF_DEPTH = 3
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42


# === Architecture ===

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
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.policy_head = SpatialPolicyHead(hidden_dim, head_dim=256)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
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

        return {"policy_logits": policy_logits, "value_logits": value_logits,
                "hidden": hidden}

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

    @torch.no_grad()
    def evaluate_positions_batch(self, boards):
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_batch(boards, device)
        result = self.forward(board_input)
        value_logits = result["value_logits"]
        probs = F.softmax(value_logits, dim=-1)
        return (probs[:, 2] - probs[:, 0]).cpu().tolist()

    @torch.no_grad()
    def predict_with_search(self, board, top_k=5):
        self.eval()
        device = next(self.parameters()).device
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)

        topk_vals, topk_idxs = probs.topk(min(top_k, mask.sum().item()), dim=-1)
        candidate_moves = []
        resulting_boards = []

        for idx in topk_idxs.cpu().tolist():
            move = index_to_move(idx)
            if move in board.legal_moves:
                candidate_moves.append(move)
                b_copy = board.copy()
                b_copy.push(move)
                resulting_boards.append(b_copy)

        if not candidate_moves:
            return index_to_move(probs.argmax().item()), []

        opp_values = self.evaluate_positions_batch(resulting_boards)
        move_info = []
        for i, (move, opp_val) in enumerate(zip(candidate_moves, opp_values)):
            move_info.append({"move": move, "our_value": -opp_val})

        move_info.sort(key=lambda x: x["our_value"], reverse=True)
        return move_info[0]["move"], move_info

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === Data ===

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


def prepare_hf_boards(dataset, n, offset=0):
    boards = []
    for i in range(offset, min(offset + n * 2, len(dataset))):
        if len(boards) >= n: break
        s = dataset[i]
        try:
            board = hf_sample_to_board(s["board"], s["turn"])
            if list(board.legal_moves):
                boards.append(board)
        except (ValueError, IndexError):
            continue
    return boards


def label_with_sf_eval(boards, sf_depth=8, batch_print=5000):
    """Label positions with SF centipawn evals → WDL targets."""
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 128})
    data = []
    t0 = time.time()
    for i, board in enumerate(boards):
        sf.set_fen_position(board.fen())
        eval_info = sf.get_evaluation()
        if eval_info["type"] == "cp":
            cp = eval_info["value"]
        else:  # mate
            cp = 10000 if eval_info["value"] > 0 else -10000

        # WDL from centipawn (relative to side to move)
        if cp > 100:
            vt = 2  # winning
        elif cp < -100:
            vt = 0  # losing
        else:
            vt = 1  # drawn

        data.append({"board": board, "value_target": vt, "cp": cp})

        if (i + 1) % batch_print == 0:
            print(f"    {i+1}/{len(boards)} [{time.time()-t0:.0f}s]")

    print(f"  SF eval labeling: {len(data)} positions [{time.time()-t0:.0f}s]")

    # Distribution
    win = sum(1 for d in data if d["value_target"] == 2)
    draw = sum(1 for d in data if d["value_target"] == 1)
    loss = sum(1 for d in data if d["value_target"] == 0)
    print(f"  Distribution: win={win} ({win/len(data):.0%}) "
          f"draw={draw} ({draw/len(data):.0%}) "
          f"loss={loss} ({loss/len(data):.0%})")
    return data


# === Game playing ===

def play_game_vs_stockfish(model, sf_depth, model_color, device, max_moves=100,
                           search_mode="argmax", top_k=5):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth,
                   parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    model.eval()
    search_changed = 0

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            if search_mode == "argmax":
                pred, _ = model.predict_move(board)
            else:
                pred, info = model.predict_with_search(board, top_k=top_k)
                policy_move, _ = model.predict_move(board)
                if pred != policy_move:
                    search_changed += 1
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
        "search_changed": search_changed,
    }


# === Main ===

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp037_fix_value_head")

    # --- Load checkpoint ---
    print("\n[1/4] Loading exp032 checkpoint...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")

    # --- Evaluate old value head ---
    print("\n  Old value head diagnostic:")
    for fen, desc in [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "1.e4"),
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian"),
    ]:
        board = chess.Board(fen)
        vals = model.evaluate_positions_batch([board])
        print(f"    {desc}: {vals[0]:+.3f}")

    # --- SF eval labeling ---
    print(f"\n[2/4] Labeling {NUM_VALUE_TRAIN} positions with SF d{SF_DEPTH} evals...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    boards = prepare_hf_boards(ds, NUM_VALUE_TRAIN, offset=200_000)
    value_data = label_with_sf_eval(boards, SF_DEPTH)

    # Split 90/10 train/val
    random.shuffle(value_data)
    val_size = len(value_data) // 10
    val_data = value_data[:val_size]
    train_data = value_data[val_size:]

    # --- Phase 1: Fine-tune value head only ---
    print(f"\n[3/4] Fine-tuning value head ({VALUE_EPOCHS} epochs, "
          f"{len(train_data)} train, {len(val_data)} val)")

    # Freeze everything except value head
    for name, param in model.named_parameters():
        if "value_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_p:,} (value head only)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=VALUE_LR, weight_decay=0.01
    )

    value_history = []
    for epoch in range(VALUE_EPOCHS):
        model.train()
        random.shuffle(train_data)
        ep_loss = steps = correct = total = 0

        for i in range(0, len(train_data), VALUE_BATCH):
            chunk = train_data[i:i + VALUE_BATCH]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor([d["value_target"] for d in chunk],
                                   dtype=torch.long, device=device)

            board_input = batch_boards_to_token_ids(boards, device)
            result = model(board_input)
            value_logits = result["value_logits"]
            loss = F.cross_entropy(value_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            steps += 1
            preds = value_logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        # Validation
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_data), VALUE_BATCH):
                chunk = val_data[i:i + VALUE_BATCH]
                boards = [d["board"] for d in chunk]
                targets = torch.tensor([d["value_target"] for d in chunk],
                                       dtype=torch.long, device=device)
                board_input = batch_boards_to_token_ids(boards, device)
                result = model(board_input)
                preds = result["value_logits"].argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += len(targets)

        train_acc = correct / max(total, 1)
        val_acc = val_correct / max(val_total, 1)
        value_history.append({
            "epoch": epoch + 1,
            "loss": round(ep_loss / max(steps, 1), 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        })
        print(f"  Epoch {epoch+1}: loss={ep_loss/steps:.4f} "
              f"train_acc={train_acc:.1%} val_acc={val_acc:.1%}")

    # --- New value head diagnostic ---
    print("\n  New value head diagnostic:")
    # Unfreeze all for inference
    for param in model.parameters():
        param.requires_grad = True

    for fen, desc in [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "1.e4"),
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian"),
    ]:
        board = chess.Board(fen)
        vals = model.evaluate_positions_batch([board])
        print(f"    {desc}: {vals[0]:+.3f}")

    # --- Phase 2: Test search strategies ---
    strategies = [
        ("argmax", {"search_mode": "argmax", "top_k": 0}),
        ("search_top5", {"search_mode": "search", "top_k": 5}),
        ("search_top10", {"search_mode": "search", "top_k": 10}),
    ]

    all_results = {}
    print(f"\n[4/4] Playing {NUM_GAMES} games per strategy vs SF d{GAME_SF_DEPTH}")

    for strat_name, strat_kwargs in strategies:
        print(f"\n  Strategy: {strat_name}")
        game_results = []
        total_changed = 0

        for g in range(NUM_GAMES):
            color = chess.WHITE if g % 2 == 0 else chess.BLACK
            r = play_game_vs_stockfish(model, GAME_SF_DEPTH, color, device,
                                       **strat_kwargs)
            game_results.append(r)
            total_changed += r.get("search_changed", 0)
            sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
            chg = f" (changed {r['search_changed']})" if r.get("search_changed", 0) else ""
            print(f"    Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv "
                  f"({r['termination']}){chg}")

        wins = sum(1 for r in game_results if r["model_result"] == "win")
        draws = sum(1 for r in game_results if r["model_result"] == "draw")
        losses = sum(1 for r in game_results if r["model_result"] == "loss")
        print(f"  Score: W{wins}/D{draws}/L{losses}")

        all_results[strat_name] = {
            "games": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
            "total_search_changed": total_changed,
        }

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp037_fix_value_head",
        "hypothesis": "SF-calibrated value head enables search to help",
        "primary_metric": "games vs SF d3 with search",
        "seed": SEED,
        "value_training": {
            "data": len(value_data),
            "sf_depth": SF_DEPTH,
            "epochs": VALUE_EPOCHS,
            "lr": VALUE_LR,
            "history": value_history,
        },
        "strategies": all_results,
        "comparison": {
            "exp032_argmax": "W0/D1/L7",
            "exp036_search_top5": "W0/D0/L8",
        },
        "timing": {"total_s": total_time},
        "device": str(device),
        "command": "python experiments/exp037_fix_value_head.py",
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp037_fix_value_head")
    print(f" Phase 1: Value head fine-tuning on SF evals")
    for h in value_history:
        print(f"   Ep{h['epoch']}: loss={h['loss']:.4f} val_acc={h['val_acc']:.1%}")
    print(f" Phase 2: Search comparison")
    for name, r in all_results.items():
        s = r["score"]
        print(f"   {name:12s}: W{s['wins']}/D{s['draws']}/L{s['losses']}")
    print(f" Baselines: argmax=W0/D1/L7, search(old)=W0/D0/L8")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
