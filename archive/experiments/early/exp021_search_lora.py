"""exp021: Search + Spatial LoRA — Combine best gains.

Tests TWO independent hypotheses in one experiment:
  A) Spatial head + LoRA backbone (addresses backbone bottleneck)
  B) 1-ply search: evaluate each legal move's position with value head
     (addresses gameplay without better accuracy)

Plan:
  1. Train spatial + LoRA model on 50K, 3 epochs (quick)
  2. Evaluate accuracy with spatial-only vs spatial+LoRA
  3. Play games with:
     a) Best model, policy-only (greedy)
     b) Best model + 1-ply value search
     c) Best model + 1-ply value search + deeper SF comparison

Primary metric: game results vs SF d3
Time budget: ~15 min (training) + ~5 min (games)
"""

import json
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
from model import load_base_model
from move_vocab import VOCAB_SIZE, UCI_TO_IDX, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move
from config import Config

OUTPUT_DIR = Path("outputs/exp021_search_lora")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TRAIN = 50_000
NUM_EVAL = 500
EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-3
LR_LORA = 5e-5
ENCODER_DIM = 256
SEED = 42
NUM_GAMES = 6
GAME_SF_DEPTH = 3

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "v_proj"]


# === LoRA ===

class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, original.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        result = self.original(x)
        x_f = self.lora_dropout(x).float()
        lora_out = (x_f @ self.lora_A.T @ self.lora_B.T * self.scaling)
        return result + lora_out.to(result.dtype)


def apply_lora(model, rank, alpha, dropout, targets):
    lora_params = 0
    modules_dict = dict(model.named_modules())
    for name, module in list(model.named_modules()):
        for target in targets:
            if target in name and isinstance(module, nn.Linear):
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = modules_dict[parts[0]]
                    attr_name = parts[1]
                else:
                    parent = model
                    attr_name = name
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr_name, lora_layer)
                lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
    return lora_params


# === Spatial Model ===

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


class SpatialChessModel(nn.Module):
    def __init__(self, qwen_model, encoder, encoder_dim=256, freeze_backbone=True):
        super().__init__()
        if hasattr(qwen_model, 'model') and hasattr(qwen_model, 'lm_head'):
            base_model = qwen_model.model
        else:
            base_model = qwen_model
        self.hidden_size = getattr(base_model.config, 'hidden_size', 1024)
        self.encoder = encoder
        self.input_proj = nn.Linear(encoder_dim, self.hidden_size)
        self.backbone = base_model
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.policy_head = SpatialPolicyHead(self.hidden_size, head_dim=256)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.ReLU(), nn.Linear(256, 3),
        )

    def forward(self, board_input, move_targets=None, value_targets=None, **kw):
        tokens = self.encoder(board_input)
        embeds = self.input_proj(tokens)
        backbone_dtype = next(self.backbone.parameters()).dtype
        embeds = embeds.to(backbone_dtype)
        outputs = self.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = outputs.last_hidden_state.float()
        policy_logits = self.policy_head(hidden)
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)
        result = {"policy_logits": policy_logits, "value_logits": value_logits}
        device = board_input["piece_ids"].device if isinstance(board_input, dict) else board_input.device
        total_loss = torch.tensor(0.0, device=device)
        if move_targets is not None:
            total_loss = total_loss + F.cross_entropy(policy_logits, move_targets)
            result["policy_loss"] = total_loss.item()
        if value_targets is not None:
            vl = F.cross_entropy(value_logits, value_targets)
            total_loss = total_loss + 0.5 * vl
            result["value_loss"] = vl.item()
        result["loss"] = total_loss
        return result

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
    def predict_move_with_search(self, board, top_k=5):
        """1-ply search: evaluate top-K candidate moves with value head.

        1. Get policy logits → pick top-K legal moves
        2. For each candidate, make the move, evaluate resulting position
        3. Pick the move whose resulting position has the best value for us
        """
        self.eval()
        device = next(self.parameters()).device

        # Get policy top-K
        board_input = self.encoder.prepare_input(board, device)
        mask = legal_move_mask(board).to(device)
        result = self.forward(board_input)
        logits = result["policy_logits"][0]
        logits[~mask] = float("-inf")
        probs = F.softmax(logits, dim=-1)

        # Get top-K legal moves
        k = min(top_k, mask.sum().item())
        top_idxs = probs.topk(k).indices.tolist()
        candidates = [index_to_move(idx) for idx in top_idxs]

        if len(candidates) <= 1:
            return candidates[0] if candidates else list(board.legal_moves)[0], probs

        # Evaluate each candidate's resulting position
        boards_after = []
        valid_moves = []
        for move in candidates:
            if move in board.legal_moves:
                b = board.copy()
                b.push(move)
                boards_after.append(b)
                valid_moves.append(move)

        if not boards_after:
            return candidates[0], probs

        # Batch evaluate resulting positions
        batch_input = self.encoder.prepare_batch(boards_after, device)
        tokens = self.encoder(batch_input)
        embeds = self.input_proj(tokens)
        backbone_dtype = next(self.backbone.parameters()).dtype
        embeds = embeds.to(backbone_dtype)
        outputs = self.backbone(inputs_embeds=embeds, use_cache=False)
        hidden = outputs.last_hidden_state.float()
        global_hidden = hidden[:, 0, :]
        value_logits = self.value_head(global_hidden)
        value_probs = F.softmax(value_logits, dim=-1)  # (K, 3) = win/draw/loss

        # Score: from OPPONENT's perspective (we just moved, it's their turn)
        # value_probs[:, 0] = win for side-to-move = LOSS for us
        # value_probs[:, 2] = loss for side-to-move = WIN for us
        # So our score = value_probs[:, 2] - value_probs[:, 0] (higher = better for us)
        our_score = value_probs[:, 2] - value_probs[:, 0]

        best_idx = our_score.argmax().item()
        return valid_moves[best_idx], probs

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# === HF Dataset ===

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
        winner = s["winner"]
        if winner == 0: vt = 1
        elif (winner == 1 and board.turn == chess.WHITE) or \
             (winner == 2 and board.turn == chess.BLACK): vt = 2
        else: vt = 0
        data.append({"board": board, "move": move, "value_target": vt})
    print(f"  Prepared {len(data)} samples (skipped {skipped})")
    return data


# === Training ===

def make_batches(data, batch_size, device):
    random.shuffle(data)
    batches = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        boards = [d["board"] for d in chunk]
        batch_input = batch_boards_to_token_ids(boards, device)
        move_targets = torch.tensor([move_to_index(d["move"]) for d in chunk], dtype=torch.long, device=device)
        value_targets = torch.tensor([d["value_target"] for d in chunk], dtype=torch.long, device=device)
        batches.append((batch_input, move_targets, value_targets))
    return batches


def evaluate_accuracy(model, eval_data, device, n=None, batch_size=64):
    model.eval()
    subset = eval_data[:n] if n else eval_data
    correct = top3_correct = total = 0
    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            chunk = subset[i:i + batch_size]
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
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
        "total": total,
    }


def play_game_vs_stockfish(model, sf_depth, model_color, device, use_search=False,
                           search_top_k=5, max_moves=100):
    from stockfish import Stockfish
    sf = Stockfish(path=STOCKFISH_PATH, depth=sf_depth, parameters={"Threads": 2, "Hash": 64})
    board = chess.Board()
    model.eval()
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        if board.turn == model_color:
            if use_search:
                pred, _ = model.predict_move_with_search(board, top_k=search_top_k)
            else:
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("\n[1/5] Loading HuggingFace dataset...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()
    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL, offset=len(ds) - NUM_EVAL * 3)
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)

    # --- Build Spatial + LoRA model ---
    print(f"\n[2/5] Building Spatial + LoRA model")
    cfg = Config()
    full_model, _ = load_base_model(cfg)
    encoder = LearnedBoardEncoder(embed_dim=ENCODER_DIM)
    chess_model = SpatialChessModel(full_model, encoder=encoder, freeze_backbone=True)

    lora_count = apply_lora(chess_model.backbone, rank=LORA_RANK, alpha=LORA_ALPHA,
                            dropout=LORA_DROPOUT, targets=LORA_TARGETS)
    chess_model = chess_model.to(device)

    trainable = chess_model.trainable_params()
    print(f"  LoRA params: {lora_count:,}")
    print(f"  Total trainable: {trainable:,}")

    pre = evaluate_accuracy(chess_model, eval_data, device, n=200)
    print(f"  Pre-train: acc={pre['accuracy']:.1%}")

    # Separate LR for LoRA vs encoder/heads
    lora_params, other_params = [], []
    for name, p in chess_model.named_parameters():
        if p.requires_grad:
            if "lora_" in name:
                lora_params.append(p)
            else:
                other_params.append(p)

    param_groups = [
        {"params": other_params, "lr": LR},
        {"params": lora_params, "lr": LR_LORA},
    ]
    optimizer = AdamW(param_groups, weight_decay=0.01)
    total_steps = EPOCHS * (len(train_data) // BATCH_SIZE + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- Train ---
    print(f"\n[3/5] Training Spatial+LoRA on {len(train_data)} positions, {EPOCHS} epochs")
    history = []
    best_acc = 0
    for epoch in range(EPOCHS):
        chess_model.train()
        batches = make_batches(train_data, BATCH_SIZE, device)
        ep_loss = steps = 0
        for batch_input, move_targets, value_targets in batches:
            optimizer.zero_grad()
            result = chess_model(batch_input, move_targets=move_targets, value_targets=value_targets)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0
            )
            optimizer.step()
            scheduler.step()
            ep_loss += result["loss"].item()
            steps += 1
        avg_loss = ep_loss / max(steps, 1)
        ev = evaluate_accuracy(chess_model, eval_data, device, n=200)
        elapsed = time.time() - t0
        history.append({**ev, "loss": avg_loss, "epoch": epoch + 1})
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} "
              f"acc={ev['accuracy']:.1%} top3={ev['top3_accuracy']:.1%} [{elapsed:.0f}s]")
        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]

    final = evaluate_accuracy(chess_model, eval_data, device)
    print(f"\n  Best: {best_acc:.1%}  Final: {final['accuracy']:.1%} / top3={final['top3_accuracy']:.1%}")
    print(f"  vs frozen spatial baseline: 36.5% (exp019/exp020)")

    # --- Play games: policy-only ---
    print(f"\n[4/5] Games: policy-only vs SF d{GAME_SF_DEPTH}")
    policy_games = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(chess_model, GAME_SF_DEPTH, color, device, use_search=False)
        policy_games.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv ({r['termination']})")

    pw = sum(1 for r in policy_games if r["model_result"] == "win")
    pd = sum(1 for r in policy_games if r["model_result"] == "draw")
    pl = sum(1 for r in policy_games if r["model_result"] == "loss")
    print(f"  Policy-only: W{pw}/D{pd}/L{pl}")

    # --- Play games: 1-ply search ---
    print(f"\n[5/5] Games: 1-ply search (top-5) vs SF d{GAME_SF_DEPTH}")
    search_games = []
    for g in range(NUM_GAMES):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        r = play_game_vs_stockfish(chess_model, GAME_SF_DEPTH, color, device,
                                   use_search=True, search_top_k=5)
        search_games.append(r)
        sym = {"win": "W", "loss": "L", "draw": "D"}[r["model_result"]]
        print(f"  Game {g+1}: {r['model_color']} {sym} in {r['moves']}mv ({r['termination']})")

    sw = sum(1 for r in search_games if r["model_result"] == "win")
    sd = sum(1 for r in search_games if r["model_result"] == "draw")
    sl = sum(1 for r in search_games if r["model_result"] == "loss")
    print(f"  1-ply search: W{sw}/D{sd}/L{sl}")

    # --- Save ---
    total_time = time.time() - t0
    results = {
        "experiment": "exp021_search_lora",
        "hypothesis": "Spatial+LoRA + 1-ply search improves gameplay vs SF",
        "seed": SEED,
        "data": {"train": len(train_data), "eval": len(eval_data)},
        "model": {
            "lora_rank": LORA_RANK, "lora_params": lora_count,
            "trainable": trainable, "head": "spatial",
        },
        "results": {
            "best_acc": best_acc, "final": final, "history": history,
            "spatial_baseline": 0.365,
        },
        "policy_games": policy_games,
        "policy_score": {"wins": pw, "draws": pd, "losses": pl},
        "search_games": search_games,
        "search_score": {"wins": sw, "draws": sd, "losses": sl},
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Spatial+LoRA (50K, {EPOCHS}ep): {best_acc:.1%} best / {final['accuracy']:.1%} final")
    print(f"  vs spatial-only:            36.5%")
    print(f"  Policy-only vs SF d{GAME_SF_DEPTH}: W{pw}/D{pd}/L{pl}")
    print(f"  1-ply search vs SF d{GAME_SF_DEPTH}: W{sw}/D{sd}/L{sl}")
    print(f"  Time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
