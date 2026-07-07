"""Load ChessTransformer checkpoints and run move/value inference."""

from __future__ import annotations

from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import (
    ChessTransformer,
    ChessTransformerConfig,
    DEFAULT_8GB_CONFIG,
    build_model,
)
from move_vocab import IDX_TO_UCI, index_to_move, legal_move_mask

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "outputs" / "exp182_pretrain_8gb" / "latest.pt"


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT
    raise FileNotFoundError(
        f"No checkpoint at {DEFAULT_CHECKPOINT}. Pass --checkpoint PATH."
    )


def load_checkpoint(
    path: str | Path | None = None,
    device: torch.device | str | None = None,
) -> ChessTransformer:
    """Load a trained ChessTransformer for inference."""
    ckpt_path = resolve_checkpoint(path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config_data = ckpt.get("config")
    if config_data is None:
        config = DEFAULT_8GB_CONFIG
    elif isinstance(config_data, ChessTransformerConfig):
        config = config_data
    else:
        config = ChessTransformerConfig(**config_data)

    model = build_model(config)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(dev)
    model.eval()
    return model


def _value_from_logits(val_logits: torch.Tensor) -> dict[str, float]:
    """Convert value head output to win/draw/loss display probs."""
    if val_logits.shape[-1] == 3:
        wdl = F.softmax(val_logits, dim=-1)
        return {"win": wdl[0].item(), "draw": wdl[1].item(), "loss": wdl[2].item()}

    n_bins = val_logits.shape[-1]
    centers = torch.linspace(
        0.5 / n_bins, 1 - 0.5 / n_bins, n_bins, device=val_logits.device,
    )
    probs = F.softmax(val_logits, dim=-1)
    win_pct = (probs * centers).sum().item()
    # HL-Gauss: show win% + implied loss; draw mass is fuzzy
    return {"win": win_pct, "draw": 0.0, "loss": 1.0 - win_pct, "win_pct": win_pct}


@torch.no_grad()
def get_model_move(
    model: ChessTransformer,
    board: chess.Board,
    device: torch.device,
    temperature: float = 0.0,
) -> tuple[chess.Move, dict]:
    """Pick a legal move from the model policy head."""
    board_input = batch_boards_to_fused_token_ids([board], device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        result = model(board_input)

    logits = result["policy_logits"][0].float()
    mask = legal_move_mask(board).to(device)
    logits[~mask] = float("-inf")

    if temperature <= 0:
        move_idx = logits.argmax().item()
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        move_idx = torch.multinomial(probs, 1).item()

    move = index_to_move(move_idx)

    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, min(5, int(mask.sum().item())))
    top_moves = [
        (IDX_TO_UCI[idx], f"{p * 100:.1f}%")
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist())
    ]

    wdl = _value_from_logits(result["value_logits"][0].float())
    return move, {"top_moves": top_moves, "wdl": wdl}
