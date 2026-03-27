#!/usr/bin/env python3
"""Play against the ChessTransformer200M interactively in the terminal."""

import sys
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from move_vocab import VOCAB_SIZE, IDX_TO_UCI, index_to_move, legal_move_mask

# ---- Model definition (mirrors exp073) ----

from chess_model import FusedBoardEncoder
import torch.nn as nn

ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1  # must match training config for weight loading
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512


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
    def __init__(self, hidden_size, n_ctx_tokens=4, head_dim=512):
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


class ChessTransformer200M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FusedBoardEncoder(embed_dim=ENCODER_DIM)
        self.input_proj = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, HIDDEN_DIM) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * FFN_RATIO, dropout=DROPOUT,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.policy_head = SpatialPolicyHead(HIDDEN_DIM, n_ctx_tokens=4, head_dim=POLICY_HEAD_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 3),
        )

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
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }


# ---- Inference helpers ----

def encode_board(board: chess.Board, device: torch.device) -> dict[str, torch.Tensor]:
    """Encode a single board for model input (batched with B=1)."""
    from chess_features import batch_boards_to_fused_token_ids
    return batch_boards_to_fused_token_ids([board], device)


@torch.no_grad()
def get_model_move(model, board: chess.Board, device: torch.device,
                   temperature: float = 0.0) -> tuple[chess.Move, dict]:
    """Get the model's move for a position.

    Returns (move, info_dict) where info has top moves, WDL, etc.
    """
    board_input = encode_board(board, device)
    result = model(board_input)

    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    if temperature <= 0:
        # Greedy
        move_idx = logits.argmax().item()
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        move_idx = torch.multinomial(probs, 1).item()

    move = index_to_move(move_idx)

    # Top 5 moves for display
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(5, mask.sum().item()))
    top_moves = []
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        top_moves.append((IDX_TO_UCI[idx], f"{p*100:.1f}%"))

    # WDL evaluation
    wdl_logits = result["value_logits"][0].float()
    wdl_probs = F.softmax(wdl_logits, dim=-1).tolist()

    return move, {
        "top_moves": top_moves,
        "wdl": {"loss": wdl_probs[0], "draw": wdl_probs[1], "win": wdl_probs[2]},
    }


# ---- Display ----

PIECE_SYMBOLS = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def print_board(board: chess.Board, perspective_white: bool = True):
    """Print the board with Unicode pieces."""
    ranks = range(7, -1, -1) if perspective_white else range(8)
    files = range(8) if perspective_white else range(7, -1, -1)

    print()
    for rank in ranks:
        row = f"  {rank + 1}  "
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece:
                row += f" {PIECE_SYMBOLS.get(piece.symbol(), piece.symbol())} "
            else:
                is_dark = (rank + file) % 2 == 0
                row += " · " if is_dark else " . "
        print(row)

    file_labels = "     " + "  ".join(
        chr(ord("a") + f) for f in (range(8) if perspective_white else range(7, -1, -1))
    )
    print(file_labels)
    print()


def print_wdl(wdl: dict, model_is_white: bool):
    """Display WDL bar from the model's perspective."""
    # WDL is from side-to-move perspective in the model
    w, d, l = wdl["win"], wdl["draw"], wdl["loss"]
    bar_len = 30
    w_chars = round(w * bar_len)
    d_chars = round(d * bar_len)
    l_chars = bar_len - w_chars - d_chars
    bar = "█" * w_chars + "▒" * d_chars + "░" * l_chars
    print(f"  Model eval: [{bar}] W:{w:.0%} D:{d:.0%} L:{l:.0%}")


# ---- Main game loop ----

def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load the ChessTransformer200M from a checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = ChessTransformer200M()

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Handle both raw state_dict and wrapped checkpoint formats
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    # Strip torch.compile _orig_mod. prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params) on {device}")
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Play chess against ChessTransformer200M")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="outputs/exp073_200m_full_epoch/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", "-t", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--play-as", choices=["white", "black", "w", "b"], default="white",
                        help="Which side you play")
    parser.add_argument("--fen", type=str, default=None,
                        help="Start from a custom FEN position")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    board = chess.Board(args.fen) if args.fen else chess.Board()
    human_is_white = args.play_as in ("white", "w")

    print("\n" + "=" * 50)
    print("  CHESS — You vs ChessTransformer200M (204M)")
    print("=" * 50)
    print(f"  You play: {'White' if human_is_white else 'Black'}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Commands: UCI moves (e2e4), 'undo', 'fen', 'quit'")
    print("=" * 50)

    print_board(board, perspective_white=human_is_white)

    while not board.is_game_over():
        is_human_turn = (board.turn == chess.WHITE) == human_is_white

        if is_human_turn:
            while True:
                try:
                    user_input = input(f"  {'White' if board.turn == chess.WHITE else 'Black'} to move > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    return

                if user_input in ("quit", "exit", "q"):
                    print("Goodbye!")
                    return
                if user_input == "fen":
                    print(f"  {board.fen()}")
                    continue
                if user_input == "undo":
                    if len(board.move_stack) >= 2:
                        board.pop()
                        board.pop()
                        print_board(board, perspective_white=human_is_white)
                    else:
                        print("  Nothing to undo.")
                    continue

                try:
                    move = board.parse_uci(user_input)
                except ValueError:
                    try:
                        move = board.parse_san(user_input)
                    except ValueError:
                        print(f"  Invalid move: '{user_input}'. Try UCI (e2e4) or SAN (Nf3).")
                        continue

                if move not in board.legal_moves:
                    print(f"  Illegal move: {user_input}")
                    continue

                board.push(move)
                print(f"\n  You played: {move.uci()} ({board.peek().uci()})")
                print_board(board, perspective_white=human_is_white)
                break
        else:
            print(f"  {'White' if board.turn == chess.WHITE else 'Black'} (model) thinking...")
            move, info = get_model_move(model, board, device, temperature=args.temperature)
            board.push(move)

            print(f"\n  Model plays: {move.uci()}")
            top_str = "  Top moves: " + ", ".join(f"{m} ({p})" for m, p in info["top_moves"])
            print(top_str)
            print_wdl(info["wdl"], model_is_white=not human_is_white)
            print_board(board, perspective_white=human_is_white)

    # Game over
    print("=" * 50)
    result = board.result()
    outcome = board.outcome()
    if outcome:
        if outcome.winner is None:
            print(f"  Game over: Draw ({outcome.termination.name})")
        elif outcome.winner == chess.WHITE:
            print(f"  Game over: White wins ({outcome.termination.name})")
        else:
            print(f"  Game over: Black wins ({outcome.termination.name})")
    print(f"  Result: {result}")
    print(f"  PGN moves: {board.variation_san(board.move_stack)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
