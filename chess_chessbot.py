"""Hybrid: ChessBot representation + 99M trunk/training pieces.

ChessBot: 19-plane board (rank-flipped, token 0 = a8), MaGating, Transformer-XL
relative attention, 64x64 QK policy onto the 1929-move vocab, dual WDL heads
in [black, draw, white] order. Data: Maxlegrec/ChessFENS.

99M: 736d / 8H / 4+7x3+4 recurrence, QK-norm, SwiGLU, zero-init out projections,
dropout 0.05, Polar-NorMuon. Fresh init — do not copy squares64 weights.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
_POLICY_PATH = ROOT / "chess_chessbot_policy.json"
CHESSBOT_POLICY: tuple[str, ...] = tuple(json.loads(_POLICY_PATH.read_text()))
CHESSBOT_VOCAB_SIZE = len(CHESSBOT_POLICY)
CHESSFENS_POLICY_SIZE = 1858
CHESSBOT_UCI_TO_IDX: dict[str, int] = {u: i for i, u in enumerate(CHESSBOT_POLICY)}
PAD_INDEX = CHESSBOT_VOCAB_SIZE - 1
NUM_PLANES = 19
EXPECTED_99M_PARAMS = 99_352_334

_PIECE_PLANE = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}


def _require_vocab() -> None:
    if CHESSBOT_VOCAB_SIZE != 1929:
        raise RuntimeError(f"ChessBot vocab must be 1929, got {CHESSBOT_VOCAB_SIZE}")
    if CHESSBOT_POLICY[0] != "a1b1" or CHESSBOT_POLICY[1857] != "h7h8b":
        raise RuntimeError("ChessBot / ChessFENS policy prefix mismatch")
    if CHESSBOT_POLICY[1858] != "a2a1q" or CHESSBOT_POLICY[-1] != "padding_token":
        raise RuntimeError("ChessBot policy suffix mismatch")


_require_vocab()


@dataclass(frozen=True)
class ChessBot99Config:
    d_model: int = 736
    d_ff: int = 2112
    num_heads: int = 8
    dropout: float = 0.05
    prefix_layers: int = 4
    recurrent_layers: int = 7
    recurrent_unrolls: int = 3
    suffix_layers: int = 4
    vocab_size: int = CHESSBOT_VOCAB_SIZE
    n_value_classes: int = 3
    value_hidden: int = 128
    gradient_checkpointing: bool = False
    use_swiglu: bool = True
    use_qk_norm: bool = True
    zero_init_out_proj: bool = True

    @property
    def unique_layers(self) -> int:
        return self.prefix_layers + self.recurrent_layers + self.suffix_layers

    @property
    def effective_depth(self) -> int:
        return (
            self.prefix_layers
            + self.recurrent_layers * self.recurrent_unrolls
            + self.suffix_layers
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.d_model % 8 != 0 or self.d_ff % 8 != 0:
            raise ValueError("d_model and d_ff must be divisible by 8")
        if min(self.prefix_layers, self.recurrent_layers, self.suffix_layers) < 1:
            raise ValueError("Need at least one prefix, bank, and suffix layer")
        if self.recurrent_unrolls < 1:
            raise ValueError("recurrent_unrolls must be >= 1")
        if self.vocab_size != CHESSBOT_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be {CHESSBOT_VOCAB_SIZE}")

    @classmethod
    def from_dict(cls, data: dict) -> "ChessBot99Config":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.validate()
        return cfg


DEFAULT_99M_CHESSBOT_CONFIG = ChessBot99Config()


def chessbot_token_index(square: int) -> int:
    """python-chess a1=0 square → ChessBot token (a8=0, rank-flipped)."""
    rank, file = divmod(int(square), 8)
    return (7 - rank) * 8 + file


def _parse_fen_planes(fen: str, out: np.ndarray) -> None:
    """Write one ChessBot (8, 8, 19) plane tensor. FEN placement is rank 8 first."""
    parts = fen.split()
    placement = parts[0]
    turn = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    rank = 0
    file = 0
    for ch in placement:
        if ch == "/":
            rank += 1
            file = 0
        elif ch.isdigit():
            file += ord(ch) - 48
        else:
            out[rank, file, _PIECE_PLANE[ch]] = 1.0
            file += 1
    if turn == "w":
        out[:, :, 12] = 1.0
    if ep not in {"-", ""}:
        out[8 - int(ep[1]), ord(ep[0]) - 97, 13] = 1.0
    if "K" in castling:
        out[:, :, 14] = 1.0
    if "Q" in castling:
        out[:, :, 15] = 1.0
    if "k" in castling:
        out[:, :, 16] = 1.0
    if "q" in castling:
        out[:, :, 17] = 1.0
    out[:, :, 18] = min(halfmove / 100.0, 1.0)


def fens_to_planes(fens: list[str], device: torch.device | None = None) -> torch.Tensor:
    batch = np.zeros((len(fens), 8, 8, NUM_PLANES), dtype=np.float32)
    for i, fen in enumerate(fens):
        _parse_fen_planes(fen, batch[i])
    planes = torch.from_numpy(batch.reshape(len(fens), 64, NUM_PLANES))
    return planes if device is None else planes.to(device)


def boards_to_planes(
    boards: list[chess.Board],
    device: torch.device | None = None,
) -> torch.Tensor:
    return fens_to_planes([board.fen() for board in boards], device)


def fen_to_planes(fen: str, device: torch.device | None = None) -> torch.Tensor:
    return fens_to_planes([fen], device)


def flip_planes_opposite_color(planes: torch.Tensor) -> torch.Tensor:
    """Rank-flip + color-swap on (B, 64, 19) or (64, 19) ChessBot planes."""
    spatial = planes.reshape(*planes.shape[:-2], 8, 8, NUM_PLANES).flip(-3)
    out = spatial.clone()
    out[..., 0:6] = spatial[..., 6:12]
    out[..., 6:12] = spatial[..., 0:6]
    out[..., 12] = 1.0 - spatial[..., 12]
    out[..., 13] = spatial[..., 13]
    out[..., 14] = spatial[..., 16]
    out[..., 15] = spatial[..., 17]
    out[..., 16] = spatial[..., 14]
    out[..., 17] = spatial[..., 15]
    out[..., 18] = spatial[..., 18]
    return out.reshape(planes.shape)


def legal_policy_mask(board: chess.Board, device: torch.device | None = None) -> torch.Tensor:
    """ChessBot legal mask: knight promotions drop the trailing 'n'."""
    mask = torch.zeros(CHESSBOT_VOCAB_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        uci = move.uci()
        if uci.endswith("n"):
            uci = uci[:-1]
        idx = CHESSBOT_UCI_TO_IDX.get(uci)
        if idx is not None:
            mask[idx] = True
    return mask if device is None else mask.to(device)


def move_to_policy_index(move: chess.Move) -> int:
    uci = move.uci()
    if uci.endswith("n"):
        uci = uci[:-1]
    if uci not in CHESSBOT_UCI_TO_IDX:
        raise KeyError(f"Move {move.uci()} is outside the ChessBot vocab")
    return CHESSBOT_UCI_TO_IDX[uci]


def policy_index_to_move(idx: int, board: chess.Board) -> chess.Move:
    uci = CHESSBOT_POLICY[int(idx)]
    move = chess.Move.from_uci(uci)
    if move in board.legal_moves:
        return move
    if len(uci) == 4:
        for promo in (chess.KNIGHT, chess.QUEEN, chess.ROOK, chess.BISHOP):
            alt = chess.Move.from_uci(uci + chess.piece_symbol(promo))
            if alt in board.legal_moves:
                return alt
    raise ValueError(f"Policy index {idx} ({uci}) is illegal on {board.fen()}")


def stm_wdl_to_chessbot(wdl_stm: torch.Tensor, turn_white: torch.Tensor) -> torch.Tensor:
    """ChessFENS [stm_win, draw, stm_lose] → ChessBot [black, draw, white]."""
    wdl = wdl_stm.to(dtype=torch.float32)
    swapped = wdl[:, [2, 1, 0]]
    return torch.where(turn_white.reshape(-1, 1).bool(), swapped, wdl)


def flip_rank_square(square: int) -> int:
    rank, file = divmod(int(square), 8)
    return (7 - rank) * 8 + file


def flip_rank_uci(uci: str) -> str:
    if uci in {" ", "end_variation", "end", "padding_token"} or not uci:
        return uci
    return f"{uci[0]}{chr(ord('1') + 7 - (ord(uci[1]) - ord('1')))}{uci[2]}{chr(ord('1') + 7 - (ord(uci[3]) - ord('1')))}{uci[4:]}"


def _build_flip_table() -> torch.Tensor:
    table = torch.arange(CHESSBOT_VOCAB_SIZE, dtype=torch.long)
    for i, uci in enumerate(CHESSBOT_POLICY):
        flipped = flip_rank_uci(uci)
        if flipped in CHESSBOT_UCI_TO_IDX:
            table[i] = CHESSBOT_UCI_TO_IDX[flipped]
    return table


FLIP_POLICY_TABLE = _build_flip_table()


def flip_board_opposite_color(board: chess.Board) -> chess.Board:
    """Rank-flip + color-swap. STM becomes the opposite color."""
    out = chess.Board(None)
    out.turn = not board.turn
    for square, piece in board.piece_map().items():
        out.set_piece_at(
            flip_rank_square(square),
            chess.Piece(piece.piece_type, not piece.color),
        )
    rights = ""
    if board.has_kingside_castling_rights(chess.BLACK):
        rights += "K"
    if board.has_queenside_castling_rights(chess.BLACK):
        rights += "Q"
    if board.has_kingside_castling_rights(chess.WHITE):
        rights += "k"
    if board.has_queenside_castling_rights(chess.WHITE):
        rights += "q"
    out.set_castling_fen(rights or "-")
    if board.ep_square is not None:
        out.ep_square = flip_rank_square(board.ep_square)
    out.halfmove_clock = board.halfmove_clock
    out.fullmove_number = board.fullmove_number
    return out


def flip_policy_vector(policy: torch.Tensor) -> torch.Tensor:
    """Permute a 1858 or 1929 policy vector by rank flip."""
    n = policy.shape[-1]
    table = FLIP_POLICY_TABLE.to(device=policy.device)
    if n == CHESSFENS_POLICY_SIZE:
        out = torch.full_like(policy, -1)
        src = policy
        dest = table[:CHESSFENS_POLICY_SIZE]
        valid = dest < CHESSFENS_POLICY_SIZE
        out[..., dest[valid]] = src[..., valid]
        return out
    if n == CHESSBOT_VOCAB_SIZE:
        return policy[..., table]
    raise ValueError(f"policy last dim must be 1858 or 1929, got {n}")


class RelativeMultiHeadAttention(nn.Module):
    """ChessBot Transformer-XL relative attention, plus 99M QK-norm."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, *,
                 use_qk_norm: bool = True, zero_init_out_proj: bool = True):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model % num_heads must be 0")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(d_model)
        self.use_qk_norm = use_qk_norm
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.u_bias = nn.Parameter(torch.empty(num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.empty(num_heads, self.d_head))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.d_head)
            self.k_norm = nn.RMSNorm(self.d_head)
        if zero_init_out_proj:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        q = self.query_proj(x).view(bsz, -1, self.num_heads, self.d_head)
        k = self.key_proj(x).view(bsz, -1, self.num_heads, self.d_head)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        k = k.permute(0, 2, 1, 3)
        v = self.value_proj(x).view(bsz, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        p = self.pos_proj(pos).view(bsz, -1, self.num_heads, self.d_head)
        content = torch.matmul((q + self.u_bias).transpose(1, 2), k.transpose(2, 3))
        rel = torch.matmul((q + self.v_bias).transpose(1, 2), p.permute(0, 2, 3, 1))
        rel = self._shift(rel)
        attn = self.dropout(torch.softmax((content + rel) / self.scale, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        return self.out_proj(ctx)

    @staticmethod
    def _shift(pos_score: torch.Tensor) -> torch.Tensor:
        bsz, heads, q_len, k_len = pos_score.size()
        padded = torch.cat([pos_score.new_zeros(bsz, heads, q_len, 1), pos_score], dim=-1)
        return padded.view(bsz, heads, k_len + 1, q_len)[:, :, 1:].reshape_as(pos_score)


class MaGating(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(64, d_model))
        self.b = nn.Parameter(torch.ones(64, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.exp(self.a) + self.b


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        position = torch.arange(64).unsqueeze(1).float()
        encoding = torch.zeros(1, 64, d_model)
        freqs = torch.arange(0, d_model, step=2).float()
        encoding[:, :, 0::2] = torch.sin(position / (10000 ** (freqs / d_model)))
        encoding[:, :, 1::2] = torch.cos(position / (10000 ** (freqs / d_model)))
        self.register_buffer("positional_encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding.expand(x.size(0), -1, -1)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float, *,
                 use_swiglu: bool = True, use_qk_norm: bool = True,
                 zero_init_out_proj: bool = True):
        super().__init__()
        self.attention = RelativeMultiHeadAttention(
            d_model, num_heads, dropout,
            use_qk_norm=use_qk_norm, zero_init_out_proj=zero_init_out_proj,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if use_swiglu:
            from chess_transformer_factory import SwiGLUFFN
            self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
            if zero_init_out_proj:
                nn.init.zeros_(self.ffn.w_down.weight)
                nn.init.zeros_(self.ffn.w_down.bias)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout),
            )
            if zero_init_out_proj:
                nn.init.zeros_(self.ffn[3].weight)
                nn.init.zeros_(self.ffn[3].bias)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.attention(x, pos) + x)
        return self.norm2(self.ffn(x) + x)


class ValueHead(nn.Module):
    def __init__(self, d_model: int, hidden: int, n_classes: int):
        super().__init__()
        self.dense1 = nn.Linear(d_model, hidden)
        self.dense2 = nn.Linear(hidden * 64, hidden)
        self.dense3 = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.dense1(x)).reshape(x.size(0), -1)
        return self.dense3(F.gelu(self.dense2(h)))


class ChessBot99(nn.Module):
    """99M recurrent ChessBot. Forward input is (B, 64, 19) planes."""

    def __init__(self, config: ChessBot99Config = DEFAULT_99M_CHESSBOT_CONFIG):
        config.validate()
        super().__init__()
        self.config = config
        self.linear1 = nn.Linear(NUM_PLANES, config.d_model)
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.ma_gating = MaGating(config.d_model)
        self.positional = AbsolutePositionalEncoder(config.d_model)

        def _layer() -> EncoderLayer:
            return EncoderLayer(
                config.d_model, config.d_ff, config.num_heads, config.dropout,
                use_swiglu=config.use_swiglu, use_qk_norm=config.use_qk_norm,
                zero_init_out_proj=config.zero_init_out_proj,
            )

        self.prefix = nn.ModuleList([_layer() for _ in range(config.prefix_layers)])
        self.bank = nn.ModuleList([_layer() for _ in range(config.recurrent_layers)])
        self.suffix = nn.ModuleList([_layer() for _ in range(config.suffix_layers)])
        self.policy_tokens_lin = nn.Linear(config.d_model, config.d_model)
        self.queries_pol = nn.Linear(config.d_model, config.d_model)
        self.keys_pol = nn.Linear(config.d_model, config.d_model)
        self.policy_head = nn.Linear(64 * 64, config.vocab_size, bias=False)
        self.value_head = ValueHead(config.d_model, config.value_hidden, config.n_value_classes)
        self.value_head_q = ValueHead(config.d_model, config.value_hidden, config.n_value_classes)

    def _run(self, layer: EncoderLayer, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if self.config.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(layer, h, pos, use_reentrant=False)
        return layer(h, pos)

    def encode(
        self,
        planes: torch.Tensor,
        *,
        recurrent_unrolls: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        unrolls = self.config.recurrent_unrolls if recurrent_unrolls is None else recurrent_unrolls
        if isinstance(unrolls, bool) or not isinstance(unrolls, int) or unrolls < 1:
            raise ValueError("recurrent_unrolls must be a positive integer")
        if planes.dim() == 5:
            planes = planes.view(planes.size(0), 64, NUM_PLANES)
        elif planes.dim() == 4:
            planes = planes.view(planes.size(0), 64, NUM_PLANES)
        h = self.ma_gating(self.layernorm1(F.gelu(self.linear1(planes))))
        pos = self.positional(h)
        for layer in self.prefix:
            h = self._run(layer, h, pos)
        for _ in range(unrolls):
            for layer in self.bank:
                h = self._run(layer, h, pos)
        for layer in self.suffix:
            h = self._run(layer, h, pos)
        return h, pos

    def forward(
        self,
        planes: torch.Tensor,
        *,
        recurrent_unrolls: int | None = None,
    ) -> dict[str, torch.Tensor]:
        h, pos = self.encode(planes, recurrent_unrolls=recurrent_unrolls)
        tokens = F.gelu(self.policy_tokens_lin(h)) + pos
        scores = torch.matmul(self.queries_pol(tokens), self.keys_pol(tokens).transpose(-2, -1))
        scores = scores / math.sqrt(self.config.d_model)
        policy = self.policy_head(scores.reshape(h.size(0), 64 * 64))
        return {
            "policy_logits": policy,
            "value_logits": self.value_head(h),
            "value_logits_q": self.value_head_q(h),
            "square_hidden": h,
        }

    def recurrent_parameters(self):
        return self.bank.parameters()

    def prepare_batch(self, boards: list[chess.Board], device: torch.device) -> torch.Tensor:
        return boards_to_planes(boards, device)

    def prepare_input(self, board: chess.Board, device: torch.device) -> torch.Tensor:
        return boards_to_planes([board], device)

    @torch.no_grad()
    def select_move(
        self,
        board: chess.Board,
        device: torch.device,
        temperature: float = 0.0,
    ) -> tuple[chess.Move, dict]:
        out = self(self.prepare_input(board, device))
        logits = out["policy_logits"][0].float()
        mask = legal_policy_mask(board, device)
        if not bool(mask.any()):
            raise ValueError(f"No ChessBot-vocab legal moves on {board.fen()}")
        logits = logits.masked_fill(~mask, -1e9)
        if temperature <= 0:
            idx = int(logits.argmax())
        else:
            idx = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        move = policy_index_to_move(idx, board)
        probs = F.softmax(logits, dim=-1)
        topk = torch.topk(probs, min(5, int(mask.sum())))
        wdl = F.softmax(out["value_logits_q"][0].float(), dim=-1)
        return move, {
            "source": "chessbot99_policy",
            "temperature": float(temperature),
            "top_moves": [
                (CHESSBOT_POLICY[i], f"{p * 100:.1f}%")
                for i, p in zip(topk.indices.tolist(), topk.values.tolist())
            ],
            "wdl": {
                "black": wdl[0].item(),
                "draw": wdl[1].item(),
                "white": wdl[2].item(),
            },
        }


def average_recurrent_grads(model: ChessBot99, unrolls: int | None = None) -> None:
    n = unrolls if unrolls is not None else model.config.recurrent_unrolls
    if n <= 1:
        return
    scale = 1.0 / float(n)
    for p in model.recurrent_parameters():
        if p.grad is not None:
            p.grad.mul_(scale)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_chessbot99(config: ChessBot99Config | dict | None = None) -> ChessBot99:
    if config is None:
        cfg = DEFAULT_99M_CHESSBOT_CONFIG
    elif isinstance(config, ChessBot99Config):
        cfg = config
    else:
        cfg = ChessBot99Config.from_dict(config)
    cfg.validate()
    return ChessBot99(cfg)


def is_chessbot99_ckpt(ckpt: dict) -> bool:
    if ckpt.get("arch") in {"chessbot99", "chessbot_recurrent"}:
        return True
    config = ckpt.get("config")
    return isinstance(config, dict) and config.get("vocab_size") == CHESSBOT_VOCAB_SIZE and "d_ff" in config
