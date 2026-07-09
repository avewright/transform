"""Train on soft MCTS visit distributions + hard Stockfish moves."""

from __future__ import annotations

import random
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import VOCAB_SIZE, legal_move_mask
from rl_selfplay.config import SelfPlayConfig
from rl_selfplay.utils import q_to_wdl

SIGMA_HL_GAUSS = 0.04


def hl_gauss_target(win_pct: torch.Tensor, n_bins: int) -> torch.Tensor:
    bin_centers = torch.linspace(
        0.5 / n_bins, 1 - 0.5 / n_bins, n_bins, device=win_pct.device,
    )
    diff = bin_centers.unsqueeze(0) - win_pct.unsqueeze(1)
    return F.softmax(-0.5 * (diff / SIGMA_HL_GAUSS) ** 2, dim=-1)


def _value_targets(root_qs: torch.Tensor, n_bins: int) -> torch.Tensor:
    if n_bins == 3:
        return torch.stack([
            torch.tensor(q_to_wdl(q), dtype=torch.float32) for q in root_qs.tolist()
        ])
    win_pct = (root_qs + 1.0) / 2.0
    return hl_gauss_target(win_pct, n_bins)


def policy_kl_loss(logits: torch.Tensor, visit_targets: torch.Tensor,
                     boards: list[chess.Board], device: torch.device) -> torch.Tensor:
    for i, board in enumerate(boards):
        mask = legal_move_mask(board).to(device)
        logits[i][~mask] = -1e9
    log_probs = F.log_softmax(logits, dim=-1)
    nonzero = visit_targets > 0
    ce = torch.where(nonzero, visit_targets * log_probs, torch.zeros_like(log_probs))
    return -ce.sum(dim=-1).mean()


def value_loss(logits: torch.Tensor, targets: torch.Tensor, n_bins: int) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    if n_bins == 3:
        return F.kl_div(log_probs, targets, reduction="batchmean")
    return -(targets * log_probs).sum(dim=-1).mean()


def _build_visit_tensor(positions: list[dict]) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    fens = [p["fen"] for p in positions]
    visit = torch.zeros(len(positions), VOCAB_SIZE)
    root_q = torch.zeros(len(positions))
    for i, pos in enumerate(positions):
        for idx, prob in pos.get("visit_dist", {}).items():
            visit[i, int(idx)] = prob
        root_q[i] = float(pos.get("root_q", 0.0))
    return fens, visit, root_q


def _maybe_sf_batch(shard_dir: Path | None, batch_size: int, device: torch.device):
    if shard_dir is None or not shard_dir.exists():
        return None
    from data_loader import ShardedChessLoader
    loader = ShardedChessLoader(
        shard_dir, batch_size=batch_size, encoder_type="fused", device=device, seed=42,
    )
    try:
        return next(iter(loader))
    except StopIteration:
        return None


def _split_sources(positions: list[dict]) -> tuple[list[int], list[int]]:
    mcts_idx, sf_idx = [], []
    for i, p in enumerate(positions):
        src = p.get("source", "mcts")
        if src == "sf":
            sf_idx.append(i)
        else:
            mcts_idx.append(i)
    return mcts_idx, sf_idx


def train_on_positions(
    model: nn.Module,
    positions: list[dict],
    device: torch.device,
    cfg: SelfPlayConfig,
    n_value_classes: int,
    log_fn=print,
) -> dict[str, float]:
    if not positions:
        log_fn("  no positions to train on")
        return {"loss": 0.0, "policy": 0.0, "value": 0.0}

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train_lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    amp_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16

    fens, visit_targets, root_qs = _build_visit_tensor(positions)
    value_targets = _value_targets(root_qs, n_value_classes)
    mcts_idx, sf_idx = _split_sources(positions)
    shard_dir = Path(cfg.sf_shard_dir) if cfg.sf_shard_dir else None

    log_fn(
        f"  train mix: {len(mcts_idx)} soft-MCTS + {len(sf_idx)} SF-hard "
        f"(sf_move_frac={cfg.mix_sf_move_frac})"
    )

    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "batches": 0, "sf_batches": 0, "mcts_batches": 0}

    # Prefer the larger pool for epoch length so we see most data once.
    primary = mcts_idx if len(mcts_idx) >= len(sf_idx) else (sf_idx or mcts_idx)
    if not primary:
        primary = list(range(len(positions)))

    for epoch in range(cfg.train_epochs):
        random.shuffle(primary)
        epoch_batches = 0
        for start in range(0, len(primary), cfg.train_batch_size):
            # Choose soft MCTS vs hard SF for this batch.
            use_recorded_sf = (
                bool(sf_idx)
                and cfg.mix_sf_move_frac > 0
                and random.random() < cfg.mix_sf_move_frac
            )
            use_external_sf = (
                not use_recorded_sf
                and cfg.mix_sf_frac > 0
                and random.random() < cfg.mix_sf_frac
                and shard_dir is not None
            )

            if use_recorded_sf:
                batch_idx = random.sample(sf_idx, min(cfg.train_batch_size, len(sf_idx)))
                batch = [positions[i] for i in batch_idx]
                boards = [chess.Board(p["fen"]) for p in batch]
                batch_input = batch_boards_to_fused_token_ids(boards, device)
                move_targets = torch.tensor(
                    [int(p["chosen_move"]) for p in batch], device=device, dtype=torch.long,
                )
                value_batch = value_targets[batch_idx].to(device)
                mode = "sf_hard"
            elif use_external_sf:
                sf_batch = _maybe_sf_batch(shard_dir, cfg.train_batch_size, device)
                if sf_batch is None:
                    use_external_sf = False
                    mode = "mcts"
                else:
                    batch_input, move_targets, wdl_targets = sf_batch
                    boards = None
                    mode = "sf_shard"
            else:
                use_external_sf = False

            if not use_recorded_sf and not use_external_sf:
                pool = mcts_idx if mcts_idx else primary
                batch_idx = random.sample(pool, min(cfg.train_batch_size, len(pool)))
                batch = [positions[i] for i in batch_idx]
                boards = [chess.Board(p["fen"]) for p in batch]
                batch_input = batch_boards_to_fused_token_ids(boards, device)
                visit_batch = visit_targets[batch_idx].to(device)
                value_batch = value_targets[batch_idx].to(device)
                mode = "mcts"

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                out = model(batch_input)
                if mode == "sf_hard":
                    p_loss = F.cross_entropy(out["policy_logits"], move_targets)
                    v_loss = value_loss(out["value_logits"], value_batch, n_value_classes)
                elif mode == "sf_shard":
                    p_loss = F.cross_entropy(out["policy_logits"], move_targets)
                    if n_value_classes == 3:
                        v_loss = F.cross_entropy(
                            out["value_logits"], wdl_targets.argmax(dim=-1),
                        )
                    else:
                        win_pct = wdl_targets[:, 0] + 0.5 * wdl_targets[:, 1]
                        targets = hl_gauss_target(win_pct, n_value_classes)
                        v_loss = value_loss(out["value_logits"], targets, n_value_classes)
                else:
                    p_loss = policy_kl_loss(
                        out["policy_logits"], visit_batch, boards, device,
                    )
                    v_loss = value_loss(out["value_logits"], value_batch, n_value_classes)
                loss = p_loss + cfg.value_weight * v_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += loss.item()
            totals["policy"] += p_loss.item()
            totals["value"] += v_loss.item()
            totals["batches"] += 1
            if mode.startswith("sf"):
                totals["sf_batches"] += 1
            else:
                totals["mcts_batches"] += 1
            epoch_batches += 1

        if epoch_batches:
            log_fn(
                f"  epoch {epoch + 1}/{cfg.train_epochs}: "
                f"loss={totals['loss']/totals['batches']:.4f} "
                f"p={totals['policy']/totals['batches']:.4f} "
                f"v={totals['value']/totals['batches']:.4f} "
                f"(mcts_batches={totals['mcts_batches']} sf_batches={totals['sf_batches']})"
            )

    model.eval()
    n = max(1, totals["batches"])
    return {k: totals[k] / n for k in ("loss", "policy", "value")}
