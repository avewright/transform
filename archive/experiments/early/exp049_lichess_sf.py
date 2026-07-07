"""exp049: Lichess positions relabeled with Stockfish best moves.

Hypothesis: exp046 showed 37.1% on Lichess human moves, exp048 showed 49.6%
on SF moves but from random positions. The critical gap: random positions
aren't realistic. By taking REAL game positions from Lichess and labeling
them with SF best moves, we get the best of both worlds:
  - Realistic positions (from 2200+ rated games)
  - Perfect labels (SF depth 8 best move)

This should outperform both exp046 (noisy human labels) and exp048 (unrealistic positions).

Plan:
  1. Load 209K Lichess positions from exp046 cache
  2. Relabel each with SF depth 8 best move (~13 min at 266 pos/s)
  3. Train from scratch + fine-tune exp032
  4. Eval on SF eval (from exp048), Lichess human eval, and games

Time budget: ~13 min labeling (cached), ~30 min training, ~5 min games
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

OUTPUT_DIR = Path("outputs/exp049_lichess_sf")
SF_LABELED_FILE = Path("outputs/exp049_lichess_sf/lichess_sf_labeled.jsonl")
LICHESS_DATA = Path("outputs/exp046_lichess_large/lichess_2200plus.jsonl")
SF_EVAL_DATA = Path("outputs/exp048_sf_synthetic/sf_200k_d8.jsonl")
HF_CHECKPOINT = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Config
SF_DEPTH = 8
SF_THREADS = 4
EPOCHS = 5
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 3e-4
LR_FT = 5e-5
WARMUP_STEPS = 500
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 8
GAME_SF_DEPTH = 3
NUM_EVAL = 3000


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


# === Data ===

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def relabel_with_sf(lichess_entries, output_path):
    """Take Lichess positions and relabel with SF best move."""
    from stockfish import Stockfish

    # Resume support
    start_idx = 0
    if output_path.exists():
        with open(output_path) as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= len(lichess_entries):
            print(f"  Already relabeled {start_idx} positions, loading cache")
            return load_jsonl(output_path)
        print(f"  Resuming from position {start_idx}/{len(lichess_entries)}")

    sf = Stockfish(
        path=STOCKFISH_PATH,
        depth=SF_DEPTH,
        parameters={"Threads": SF_THREADS, "Hash": 256},
    )

    labeled = []
    if start_idx > 0:
        labeled = load_jsonl(output_path)

    t0 = time.time()
    mode = "a" if start_idx > 0 else "w"
    agreement_count = 0

    with open(output_path, mode) as f:
        for i in range(start_idx, len(lichess_entries)):
            entry = lichess_entries[i]
            try:
                board = chess.Board(entry["fen"])
                sf.set_fen_position(entry["fen"])
                sf_best = sf.get_best_move()

                if sf_best and sf_best in UCI_TO_IDX:
                    ev = sf.get_evaluation()
                    eval_cp = ev.get("value", 0) if ev.get("type") == "cp" else (
                        10000 if ev.get("value", 0) > 0 else -10000
                    )

                    agrees = (sf_best == entry.get("move", ""))

                    new_entry = {
                        "fen": entry["fen"],
                        "sf_move": sf_best,
                        "human_move": entry.get("move", ""),
                        "eval_cp": eval_cp,
                        "agrees": agrees,
                        "white_elo": entry.get("white_elo", 0),
                        "black_elo": entry.get("black_elo", 0),
                    }
                    f.write(json.dumps(new_entry) + "\n")
                    labeled.append(new_entry)
                    if agrees:
                        agreement_count += 1

            except Exception:
                continue

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - start_idx) / elapsed
                remaining = (len(lichess_entries) - i - 1) / rate if rate > 0 else 0
                agree_pct = agreement_count / max(len(labeled), 1)
                print(f"    Relabeled {i+1}/{len(lichess_entries)} "
                      f"({rate:.0f} pos/s, ~{remaining/60:.1f} min left, "
                      f"SF-human agree: {agree_pct:.1%})")

    total_time = time.time() - t0
    agree_pct = agreement_count / max(len(labeled), 1)
    print(f"  Relabeled {len(labeled)} positions in {total_time:.0f}s")
    print(f"  SF-human agreement: {agree_pct:.1%} ({agreement_count}/{len(labeled)})")

    return labeled


def entries_to_training(entries, label_key="sf_move"):
    """Convert entries to training format using specified move label."""
    data = []
    for entry in entries:
        try:
            board = chess.Board(entry["fen"])
            move_uci = entry[label_key]
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves and move_uci in UCI_TO_IDX:
                data.append({"board": board, "move": move})
        except Exception:
            continue
    return data


# === Training ===

def train_model(model, train_data, eval_data, device, epochs, lr, warmup, label=""):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_data) // (BATCH_SIZE * ACCUM_STEPS)) * epochs
    warmup_steps = min(warmup, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        ep_t0 = time.time()
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for i in range(0, len(train_data), BATCH_SIZE):
            chunk = train_data[i:i + BATCH_SIZE]
            boards = [d["board"] for d in chunk]
            targets = torch.tensor([move_to_index(d["move"]) for d in chunk],
                                   device=device)
            batch_input = batch_boards_to_token_ids(boards, device)
            result = model(batch_input)
            logits = result["policy_logits"]
            loss = F.cross_entropy(logits, targets) / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS

            if (i // BATCH_SIZE + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 200 == 0:
                    avg = total_loss / (i // BATCH_SIZE + 1)
                    print(f"    [{label}] Step {step_count}: loss={avg:.4f}")

        if (len(train_data) // BATCH_SIZE) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        ep_loss = total_loss / max(len(train_data) // BATCH_SIZE, 1)
        eval_result = evaluate_accuracy(model, eval_data, device)
        ep_time = time.time() - ep_t0

        ep_info = {
            "epoch": epoch + 1,
            "loss": round(ep_loss, 4),
            "accuracy": eval_result["accuracy"],
            "top3_accuracy": eval_result["top3_accuracy"],
            "time_s": round(ep_time),
        }
        history.append(ep_info)
        print(f"  [{label}] Epoch {epoch+1}: loss={ep_loss:.4f} "
              f"acc={eval_result['accuracy']:.1%} "
              f"top3={eval_result['top3_accuracy']:.1%} [{ep_time:.0f}s]")

        if eval_result['accuracy'] > best_acc:
            best_acc = eval_result['accuracy']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    ** New best: {best_acc:.1%} **")

    if best_state:
        model.load_state_dict(best_state)
    return history, best_acc


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


def play_games(model, device, label=""):
    print(f"\n  [{label}] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}...")
    game_results = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(model, GAME_SF_DEPTH, color, device)
        game_results.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"    Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv "
              f"({r['termination']})")
    wins = sum(1 for r in game_results if r["model_result"] == "win")
    draws = sum(1 for r in game_results if r["model_result"] == "draw")
    losses = sum(1 for r in game_results if r["model_result"] == "loss")
    print(f"    Score: W{wins}/D{draws}/L{losses}")
    return game_results, {"wins": wins, "draws": draws, "losses": losses}


# === Main ===

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: exp049_lichess_sf")

    # === Step 1: Load Lichess positions and relabel with SF ===
    print(f"\n[1/6] Loading Lichess positions and relabeling with SF d{SF_DEPTH}...")

    if SF_LABELED_FILE.exists():
        print(f"  Loading cached relabeled data...")
        sf_entries = load_jsonl(SF_LABELED_FILE)
        print(f"  Loaded {len(sf_entries)} cached entries")
    else:
        if not LICHESS_DATA.exists():
            print(f"  ERROR: Lichess data not found at {LICHESS_DATA}")
            print(f"  Run exp046 first to generate Lichess data")
            return
        lichess_raw = load_jsonl(LICHESS_DATA)
        print(f"  Loaded {len(lichess_raw)} Lichess positions")
        print(f"  Relabeling with SF depth {SF_DEPTH}...")
        SF_LABELED_FILE.parent.mkdir(parents=True, exist_ok=True)
        sf_entries = relabel_with_sf(lichess_raw, SF_LABELED_FILE)

    # Stats
    n_agree = sum(1 for e in sf_entries if e.get("agrees", False))
    agree_pct = n_agree / max(len(sf_entries), 1)
    print(f"  Total: {len(sf_entries)} positions")
    print(f"  SF-human agreement: {agree_pct:.1%} ({n_agree}/{len(sf_entries)})")

    # Convert to training data (using SF labels)
    all_sf_data = entries_to_training(sf_entries, label_key="sf_move")
    random.shuffle(all_sf_data)
    sf_eval = all_sf_data[:NUM_EVAL]
    sf_train = all_sf_data[NUM_EVAL:]
    print(f"  SF-labeled Train: {len(sf_train)}, Eval: {len(sf_eval)}")

    # Also prepare human-move eval for comparison
    all_human_data = entries_to_training(sf_entries, label_key="human_move")
    random.seed(SEED)
    random.shuffle(all_human_data)
    human_eval = all_human_data[:NUM_EVAL]
    print(f"  Human-move eval: {len(human_eval)} positions")

    # Also load random-SF eval from exp048 for cross-comparison
    random_sf_eval = None
    if SF_EVAL_DATA.exists():
        random_sf_raw = load_jsonl(SF_EVAL_DATA)
        random_sf_all = []
        for e in random_sf_raw:
            try:
                board = chess.Board(e["fen"])
                move = chess.Move.from_uci(e["best_move"])
                if move in board.legal_moves and e["best_move"] in UCI_TO_IDX:
                    random_sf_all.append({"board": board, "move": move})
            except Exception:
                continue
        random.seed(SEED)
        random.shuffle(random_sf_all)
        random_sf_eval = random_sf_all[:NUM_EVAL]
        print(f"  Random-SF eval (exp048): {len(random_sf_eval)} positions")

    # === Step 2: Train from scratch ===
    print(f"\n[2/6] From scratch on {len(sf_train)} Lichess+SF positions...")
    model = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    history, best_acc = train_model(
        model, sf_train, sf_eval, device,
        epochs=EPOCHS, lr=LR, warmup=WARMUP_STEPS, label="scratch",
    )

    # Cross-eval
    print(f"\n[3/6] Cross-evaluation...")
    human_result = evaluate_accuracy(model, human_eval, device)
    print(f"  [scratch] On human-move eval: acc={human_result['accuracy']:.1%} "
          f"top3={human_result['top3_accuracy']:.1%}")

    if random_sf_eval:
        rsf_result = evaluate_accuracy(model, random_sf_eval, device)
        print(f"  [scratch] On random-SF eval: acc={rsf_result['accuracy']:.1%} "
              f"top3={rsf_result['top3_accuracy']:.1%}")
    else:
        rsf_result = None

    torch.save({
        "model_state": model.state_dict(),
        "accuracy": best_acc,
    }, OUTPUT_DIR / "scratch_checkpoint.pt")

    # === Step 3: Fine-tune exp032 ===
    print(f"\n[4/6] Fine-tuning exp032 on Lichess+SF data...")
    model_ft = ChessTransformer(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)

    hf_sf_baseline = {"accuracy": 0, "top3_accuracy": 0}
    hf_human_baseline = {"accuracy": 0, "top3_accuracy": 0}
    if HF_CHECKPOINT.exists():
        ckpt = torch.load(HF_CHECKPOINT, map_location=device, weights_only=True)
        model_ft.load_state_dict(ckpt["model_state"])
        hf_sf_baseline = evaluate_accuracy(model_ft, sf_eval, device)
        hf_human_baseline = evaluate_accuracy(model_ft, human_eval, device)
        print(f"  exp032 on Lichess-SF eval: {hf_sf_baseline['accuracy']:.1%}")
        print(f"  exp032 on Lichess-human eval: {hf_human_baseline['accuracy']:.1%}")

    history_ft, best_ft = train_model(
        model_ft, sf_train, sf_eval, device,
        epochs=3, lr=LR_FT, warmup=200, label="finetune",
    )

    ft_human = evaluate_accuracy(model_ft, human_eval, device)
    print(f"  [finetune] On human-move eval: acc={ft_human['accuracy']:.1%}")

    ft_rsf = None
    if random_sf_eval:
        ft_rsf = evaluate_accuracy(model_ft, random_sf_eval, device)
        print(f"  [finetune] On random-SF eval: acc={ft_rsf['accuracy']:.1%}")

    torch.save({
        "model_state": model_ft.state_dict(),
        "accuracy": best_ft,
    }, OUTPUT_DIR / "finetune_checkpoint.pt")

    # === Step 5: Games ===
    print(f"\n[5/6] Games...")
    games_s, score_s = play_games(model, device, "scratch")
    games_ft, score_ft = play_games(model_ft, device, "finetune")

    # === Step 6: Save ===
    total_time = time.time() - t0
    results = {
        "experiment": "exp049_lichess_sf",
        "hypothesis": "Lichess positions + SF labels = best of both worlds",
        "seed": SEED,
        "data": {
            "total_positions": len(sf_entries),
            "sf_human_agreement": round(agree_pct, 4),
            "train": len(sf_train),
            "eval": len(sf_eval),
        },
        "baselines": {
            "exp032_on_lichess_sf_eval": hf_sf_baseline,
            "exp032_on_lichess_human_eval": hf_human_baseline,
        },
        "scratch": {
            "training": history,
            "best_sf_accuracy": best_acc,
            "human_eval": {"accuracy": human_result["accuracy"],
                          "top3": human_result["top3_accuracy"]},
            "random_sf_eval": rsf_result,
            "games": {"results": games_s, "score": score_s},
        },
        "finetune": {
            "training": history_ft,
            "best_sf_accuracy": best_ft,
            "human_eval": {"accuracy": ft_human["accuracy"],
                          "top3": ft_human["top3_accuracy"]},
            "random_sf_eval": ft_rsf,
            "games": {"results": games_ft, "score": score_ft},
        },
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp049_lichess_sf")
    print(f" Data: {len(sf_train)} train ({len(sf_entries)} Lichess pos + SF labels)")
    print(f" SF-human agreement: {agree_pct:.1%}")
    print(f"\n Baselines:")
    print(f"   exp032 on Lichess-SF eval: {hf_sf_baseline['accuracy']:.1%}")
    print(f"   exp032 on Lichess-human eval: {hf_human_baseline['accuracy']:.1%}")
    print(f"\n Scratch ({EPOCHS} epochs):")
    for h in history:
        print(f"   Ep{h['epoch']}: loss={h['loss']:.4f} "
              f"acc={h['accuracy']:.1%} top3={h['top3_accuracy']:.1%}")
    print(f"   Best SF acc: {best_acc:.1%}")
    print(f"   Human-move acc: {human_result['accuracy']:.1%}")
    if rsf_result:
        print(f"   Random-SF acc: {rsf_result['accuracy']:.1%}")
    print(f"   Games: W{score_s['wins']}/D{score_s['draws']}/L{score_s['losses']}")
    print(f"\n Finetune (3 epochs):")
    for h in history_ft:
        print(f"   Ep{h['epoch']}: loss={h['loss']:.4f} "
              f"acc={h['accuracy']:.1%} top3={h['top3_accuracy']:.1%}")
    print(f"   Best SF acc: {best_ft:.1%}")
    print(f"   Human-move acc: {ft_human['accuracy']:.1%}")
    print(f"   Games: W{score_ft['wins']}/D{score_ft['draws']}/L{score_ft['losses']}")
    print(f"\n Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
