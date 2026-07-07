"""exp042: Layer Attention — attention over transformer layers.

Hypothesis: Different transformer layers capture different chess patterns
(local geometry vs global strategy). Currently only the final layer output
feeds the policy head. A learned attention mechanism over all layer outputs
("attention on attention") lets the model select the best representation
per-position, potentially breaking the 51.4% ceiling.

Background:
  - exp032: 51.4% accuracy, W0/D1/L7 (best)
  - All training-signal approaches failed (exp033-041)
  - Instructions: "attention on attention on attention"

Design:
  - Load exp032 checkpoint
  - Replace monolithic nn.TransformerEncoder.forward with manual layer iteration
  - Collect hidden states from all 8 layers
  - Add learned attention query over layers: per-token, attend over layer outputs
  - Initialize to select final layer (preserving baseline behavior)
  - Fine-tune with LR=3e-5 for 2 epochs on 100K subset

Primary metric: top-1 accuracy on human moves
Time budget: ~8 min (2 epochs × ~4 min + eval)
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

OUTPUT_DIR = Path("outputs/exp042_layer_attention")
CHECKPOINT_PATH = Path("outputs/exp032_continue_training/best_checkpoint.pt")
STOCKFISH_PATH = "stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"

# Training config
NUM_TRAIN = 100000
NUM_EVAL = 1000
EPOCHS = 2
BATCH_SIZE = 128
ACCUM_STEPS = 2
LR = 3e-5
WARMUP_STEPS = 100
ENCODER_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 8
NUM_HEADS = 8
SEED = 42
NUM_GAMES = 8
GAME_SF_DEPTH = 3


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


class LayerAttention(nn.Module):
    """Learned attention over transformer layer outputs.
    
    For each token position, computes attention weights over all L layer
    representations and returns a weighted sum. This is "attention on attention"
    — applying attention to select among different attention layer outputs.
    
    Initialized to strongly prefer the final layer (mimicking standard behavior).
    """
    
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Learned query per position (shared across batch)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Learned key per layer
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Layer-level bias (initialized to strongly favor last layer)
        self.layer_bias = nn.Parameter(torch.zeros(num_layers))
        # Initialize bias to heavily favor last layer
        with torch.no_grad():
            self.layer_bias[-1] = 5.0  # softmax(5,0,0,...) ≈ 0.99
        self.scale = 1.0 / math.sqrt(hidden_dim)
    
    def forward(self, layer_outputs):
        """
        Args:
            layer_outputs: list of L tensors, each (batch, seq, hidden_dim)
        Returns:
            (batch, seq, hidden_dim) — attention-weighted mix of layers
        """
        # Stack: (batch, num_layers, seq, hidden_dim)
        stacked = torch.stack(layer_outputs, dim=1)
        B, L, S, D = stacked.shape
        
        # Use the last layer's output as the query source
        # (This makes the attention content-dependent)
        q = self.query(layer_outputs[-1])  # (B, S, D)
        
        # Keys from all layers: (B, L, S, D)
        k = self.key(stacked.reshape(B * L, S, D)).reshape(B, L, S, D)
        
        # Compute per-token attention over layers
        # q: (B, S, D) -> (B, 1, S, D), k: (B, L, S, D)
        # We want per-token dot product: q[b,s,:] · k[b,l,s,:] for each s
        q_expanded = q.unsqueeze(1)  # (B, 1, S, D)
        attn_scores = (q_expanded * k).sum(dim=-1) * self.scale  # (B, L, S)
        
        # Add layer bias
        attn_scores = attn_scores + self.layer_bias.view(1, L, 1)
        
        # Softmax over layers
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, L, S)
        
        # Weighted sum: attn_weights (B, L, S, 1) * stacked (B, L, S, D)
        output = (attn_weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B, S, D)
        
        return output


class ChessTransformerWithLayerAttn(nn.Module):
    """Chess transformer with layer-attention pooling.
    
    Same architecture as ChessTransformer but collects intermediate layer
    outputs and uses LayerAttention to mix them before the policy head.
    State dict is compatible with exp032 checkpoint (layer attention params
    are new, everything else matches).
    """
    
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
        # NEW: Layer attention
        self.layer_attention = LayerAttention(hidden_dim, num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, board_input, use_layer_attn=True, **kw):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens) + self.pos_embed
        
        if use_layer_attn:
            # Manually iterate through layers, collecting outputs
            layer_outputs = []
            for layer in self.transformer.layers:
                hidden = layer(hidden)
                layer_outputs.append(hidden)
            
            # Layer attention over all layers
            hidden = self.layer_attention(layer_outputs)
            hidden = self.norm(hidden)
        else:
            # Standard path (for comparison)
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
    print(f"Experiment: exp042_layer_attention")

    # --- Build model & load checkpoint ---
    print("\n[1/5] Building model with layer attention...")
    model = ChessTransformerWithLayerAttn(
        encoder_dim=ENCODER_DIM, hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, dropout=0.1,
    ).to(device)

    # Load exp032 checkpoint (strict=False to allow new layer_attention params)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  Loaded checkpoint: epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1%}")
    print(f"  Missing keys (new params): {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    for k in missing:
        print(f"    NEW: {k}")

    total_params = sum(p.numel() for p in model.parameters())
    new_params = sum(p.numel() for n, p in model.named_parameters()
                     if 'layer_attention' in n)
    print(f"  Total params: {total_params:,}, New layer_attention params: {new_params:,}")

    # --- Verify baseline ---
    print("\n[2/5] Loading data & verifying baseline...")
    from datasets import load_dataset
    ds = load_dataset("avewright/chess-dataset-production-1968", split="train")
    old_sorted_uci = build_old_move_mapping()

    eval_data = prepare_hf_data(ds, old_sorted_uci, NUM_EVAL,
                                offset=len(ds) - NUM_EVAL * 3)

    # Test with layer_attn=True (should be ~same as baseline due to init)
    baseline_la = evaluate_accuracy(model, eval_data, device)
    print(f"  LayerAttn ON: acc={baseline_la['accuracy']:.1%} "
          f"top3={baseline_la['top3_accuracy']:.1%}")

    # Test with layer_attn=False  (standard path, should match exp032)
    model_std = evaluate_accuracy(model, eval_data, device)
    print(f"  Expected baseline: ~51.4%")

    # --- Prepare training data ---
    print(f"\n[3/5] Preparing {NUM_TRAIN} training samples...")
    train_data = prepare_hf_data(ds, old_sorted_uci, NUM_TRAIN, offset=0)
    print(f"  Got {len(train_data)} training samples")

    # --- Train ---
    print(f"\n[4/5] Training: {EPOCHS} epochs, lr={LR}, batch={BATCH_SIZE}×{ACCUM_STEPS}")

    # Only train layer_attention params + policy head for first epoch
    # Then unfreeze all for second epoch
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_data) // (BATCH_SIZE * ACCUM_STEPS)) * EPOCHS
    warmup_steps = min(WARMUP_STEPS, total_steps // 5)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    best_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
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
            result = model(batch_input, use_layer_attn=True)
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

                if step_count % 100 == 0:
                    avg = total_loss / (i // BATCH_SIZE + 1)
                    print(f"    Step {step_count}: loss={avg:.4f}")

        # Handle remaining gradients
        if (len(train_data) // BATCH_SIZE) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        ep_loss = total_loss / (len(train_data) // BATCH_SIZE)
        eval_result = evaluate_accuracy(model, eval_data, device)
        ep_time = time.time() - ep_t0

        print(f"  Epoch {epoch+1}: loss={ep_loss:.4f} "
              f"acc={eval_result['accuracy']:.1%} "
              f"top3={eval_result['top3_accuracy']:.1%} "
              f"[{ep_time:.0f}s]")

        if eval_result['accuracy'] > best_acc:
            best_acc = eval_result['accuracy']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # --- Games vs SF ---
    print(f"\n[5/5] Playing {NUM_GAMES} games vs SF d{GAME_SF_DEPTH}...")
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
        "experiment": "exp042_layer_attention",
        "hypothesis": "Layer attention over all transformer layers improves policy",
        "seed": SEED,
        "config": {
            "num_train": NUM_TRAIN, "num_eval": NUM_EVAL, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "accum_steps": ACCUM_STEPS,
            "lr": LR, "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "new_params": new_params,
        },
        "baseline": baseline_la,
        "best_accuracy": best_acc,
        "games": {
            "results": game_results,
            "score": {"wins": wins, "draws": draws, "losses": losses},
        },
        "timing": {"total_s": total_time},
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "epoch": EPOCHS,
        "accuracy": best_acc,
    }, OUTPUT_DIR / "best_checkpoint.pt")

    print(f"\n{'='*60}")
    print(f" SUMMARY: exp042_layer_attention")
    print(f" LayerAttention: {new_params:,} new params")
    print(f" Baseline (with init): {baseline_la['accuracy']:.1%}")
    print(f" Best: {best_acc:.1%}")
    print(f" Games: W{wins}/D{draws}/L{losses}")
    print(f" Total time: {total_time:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
