"""exp185: A40 deep-small residual transformer + soft targets + NorMuon.

Hypothesis: A smaller deep residual stack (28L/256d/8H, ~25M) with standard
multi-head attention, SwiGLU, chess rel-bias, and soft MultiPV can be trained
much more fully in a 3-5h A40 window than a 449M wide model — then hand off
to full-strength Stockfish expert-iteration RL.

Architecture: many pre-norm residual layers (attention + SwiGLU), strengthened
encoder (256d piece embeds), no grad checkpoint, large batches (~1024).

Data mix:
  - Soft: avewright/exp085-parallel-multipv-harvest (~224K, 8-PV soft targets)
  - Hard scale: avewright/chess-positions-lichess-sf streaming (depth>=12)

Usage:
  python experiments/exp185_a40_deep_small.py --go --smoke
  python experiments/exp185_a40_deep_small.py --go
  python experiments/exp185_a40_deep_small.py --go --resume
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from board_flip import build_flip_move_table, flip_move_targets
from chess_transformer_factory import (
    DEFAULT_A40_DEEP_SMALL_CONFIG,
    ChessTransformerConfig,
    build_model,
    count_parameters,
)
from data_loader import board_array_to_fused, compute_wdl, ep_square_to_file, stream_hf_batches
from move_vocab import UCI_TO_IDX, VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH: Path | None = None
SHUTDOWN = False
FLIP_TABLE = build_flip_move_table()

SOFT_DATASET = "avewright/exp085-parallel-multipv-harvest"
ADAM_NAME_HINTS = (
    "embed", "policy_head", "value_head", "cls_token", "pos_embed",
    "norm", "bn", "rel_bias",
)
SOFT_K = 8


def _signal_handler(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    log("SHUTDOWN requested — will save after current step")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def soft_policy_loss(logits: torch.Tensor, soft_indices: torch.Tensor, soft_probs: torch.Tensor):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    valid = (soft_indices >= 0) & (soft_probs > 0)
    safe = soft_indices.clamp(min=0).long()
    gathered = log_probs.gather(1, safe) * valid.float()
    return -(soft_probs.float() * gathered).sum(dim=-1).mean()


def build_normuon_optimizer(model: nn.Module, muon_lr: float, adam_lr: float, weight_decay: float):
    from normuon import SingleDeviceNorMuonWithAuxAdam

    muon_params, adam_params = [], []
    muon_n = adam_n = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if any(h in name for h in ADAM_NAME_HINTS) or param.ndim < 2:
            adam_params.append(param)
            adam_n += n
        else:
            muon_params.append(param)
            muon_n += n

    opt = SingleDeviceNorMuonWithAuxAdam([
        dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay,
             momentum=0.95, beta2=0.95),
        dict(params=adam_params, use_muon=False, lr=adam_lr, betas=(0.9, 0.95),
             weight_decay=weight_decay),
    ])
    return opt, muon_n, adam_n


def fen_to_board_tensors(fen: str):
    board = chess.Board(fen)
    arr = [0] * 64
    for sq, piece in board.piece_map().items():
        arr[sq] = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6
    ba = torch.tensor([arr], dtype=torch.int8)
    turn = torch.tensor([0 if board.turn == chess.WHITE else 1], dtype=torch.int8)
    castling = torch.tensor([0], dtype=torch.int8)
    if board.has_kingside_castling_rights(chess.WHITE):
        castling[0] |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling[0] |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling[0] |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling[0] |= 8
    ep_sq = torch.tensor(
        [board.ep_square if board.ep_square is not None else 0], dtype=torch.int8,
    )
    return ba, turn, castling, ep_sq, board


def parse_soft_targets(row) -> tuple[list[int], list[float]] | None:
    soft = row.get("soft_targets")
    if soft is None:
        return None
    if isinstance(soft, str):
        try:
            soft = json.loads(soft)
        except json.JSONDecodeError:
            return None
    if not soft:
        return None
    indices, probs = [], []
    for item in soft[:SOFT_K]:
        uci = item.get("uci") if isinstance(item, dict) else None
        if not uci or uci not in UCI_TO_IDX:
            continue
        indices.append(UCI_TO_IDX[uci])
        probs.append(float(item.get("prob", 0.0)))
    if not indices:
        return None
    # Renormalize in case of dropped illegal/unknown UCIs
    s = sum(probs) or 1.0
    probs = [p / s for p in probs]
    while len(indices) < SOFT_K:
        indices.append(-1)
        probs.append(0.0)
    return indices, probs


def iter_soft_jsonl_rows():
    """Stream MultiPV rows from HF jsonl shards (skip non-position files)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f for f in list_repo_files(SOFT_DATASET, repo_type="dataset")
        if f.startswith("dataset/positions_") and f.endswith(".jsonl")
    ]
    files = sorted(files)
    log(f"  soft jsonl shards: {len(files)}")
    for rel in files:
        path = hf_hub_download(SOFT_DATASET, rel, repo_type="dataset")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def cache_soft_dataset(cache_path: Path, max_rows: int | None = None) -> dict:
    if cache_path.exists():
        log(f"Loading soft cache: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        log(f"  soft positions: {data['board_array'].shape[0]:,}")
        return data

    log(f"Building soft cache from {SOFT_DATASET} → {cache_path}")
    boards, turns, castles, eps, moves, cps, mates = [], [], [], [], [], [], []
    soft_idx, soft_pr = [], []
    skipped = seen = 0
    for row in iter_soft_jsonl_rows():
        if max_rows is not None and len(boards) >= max_rows:
            break
        seen += 1
        fen = row.get("fen") or row.get("position_fen")
        best = row.get("best_move")
        if not fen or not best or best not in UCI_TO_IDX:
            skipped += 1
            continue
        parsed = parse_soft_targets(row)
        if parsed is None:
            skipped += 1
            continue
        try:
            ba, turn, castling, ep_sq, board = fen_to_board_tensors(fen)
        except Exception:
            skipped += 1
            continue
        move = chess.Move.from_uci(best)
        if move not in board.legal_moves:
            skipped += 1
            continue
        idx, pr = parsed
        boards.append(ba)
        turns.append(turn)
        castles.append(castling)
        eps.append(ep_sq)
        moves.append(torch.tensor([UCI_TO_IDX[best]], dtype=torch.long))
        cp_val = int(row.get("best_cp", 0) or 0)
        cps.append(torch.tensor([cp_val], dtype=torch.int32))
        mates.append(torch.tensor([0], dtype=torch.int32))
        soft_idx.append(torch.tensor(idx, dtype=torch.long))
        soft_pr.append(torch.tensor(pr, dtype=torch.float32))
        if len(boards) % 25000 == 0:
            log(f"  soft cache progress kept={len(boards):,} seen={seen:,} skip={skipped:,}")

    if not boards:
        raise RuntimeError("No soft positions cached — check HF access / dataset")

    data = {
        "board_array": torch.cat(boards, dim=0),
        "turn": torch.cat(turns, dim=0),
        "castling": torch.cat(castles, dim=0),
        "ep_square": torch.cat(eps, dim=0),
        "move_idx": torch.cat(moves, dim=0),
        "cp": torch.cat(cps, dim=0),
        "mate": torch.cat(mates, dim=0),
        "soft_indices": torch.stack(soft_idx, dim=0),
        "soft_probs": torch.stack(soft_pr, dim=0),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    log(f"  soft cache saved: {data['board_array'].shape[0]:,} positions (skipped {skipped:,})")
    return data


def prepare_soft_batch(data: dict, indices: torch.Tensor, device):
    ba = data["board_array"][indices]
    turn = data["turn"][indices]
    castling = data["castling"][indices]
    ep = data["ep_square"][indices]
    move_idx = data["move_idx"][indices].long()
    cp = data["cp"][indices]
    mate = data["mate"][indices]
    soft_indices = data["soft_indices"][indices]
    soft_probs = data["soft_probs"][indices]

    fused = board_array_to_fused(ba)
    ep_file = ep_square_to_file(ep.long())
    wdl = compute_wdl(cp, mate, turn)

    board_input = {
        "fused_ids": fused.to(device),
        "turn": turn.long().to(device),
        "castling": castling.long().to(device),
        "ep_file": ep_file.long().to(device),
    }
    return (
        board_input,
        move_idx.to(device),
        wdl.float().to(device),
        soft_indices.to(device),
        soft_probs.to(device),
    )


def save_checkpoint(model, optimizer, scaler, step, best_metric, path: Path, config, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "best_metric": best_metric,
        "args": vars(args),
    }, tmp)
    os.replace(str(tmp), str(path))


@torch.no_grad()
def eval_soft_top1(model, soft_data, device, n: int = 4096, bs: int = 256) -> dict:
    model.eval()
    N = min(n, soft_data["board_array"].shape[0])
    # Use a fixed slice from the end as a cheap held-out proxy
    start0 = max(0, soft_data["board_array"].shape[0] - N)
    correct = total = 0
    loss_sum = 0.0
    batches = 0
    for start in range(start0, start0 + N, bs):
        end = min(start + bs, start0 + N)
        idx = torch.arange(start, end)
        board_input, move_idx, wdl, soft_i, soft_p = prepare_soft_batch(soft_data, idx, device)
        with autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(board_input)
            logits = out["policy_logits"]
            loss = soft_policy_loss(logits, soft_i, soft_p)
        preds = logits.argmax(dim=-1)
        correct += (preds == move_idx).sum().item()
        total += move_idx.numel()
        loss_sum += loss.item()
        batches += 1
    model.train()
    return {"top1": correct / max(total, 1), "soft_loss": loss_sum / max(batches, 1), "n": total}


def flip_soft_targets(soft_indices: torch.Tensor, turn: torch.Tensor, flip_table: torch.Tensor):
    """Remap soft move indices for Black-to-move rows (STM board flip)."""
    black = turn == 1
    if not black.any():
        return soft_indices
    out = soft_indices.clone()
    table = flip_table.to(soft_indices.device)
    rows = out[black]
    valid = rows >= 0
    remapped = table[rows.clamp(min=0)]
    rows = torch.where(valid, remapped, rows)
    out[black] = rows
    return out


def train_step_soft(model, batch, args, scaler):
    board_input, move_idx, wdl, soft_i, soft_p = batch
    move_idx = flip_move_targets(move_idx, board_input["turn"], FLIP_TABLE)
    soft_i = flip_soft_targets(soft_i, board_input["turn"], FLIP_TABLE)

    with autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        out = model(board_input)
        logits = out["policy_logits"]
        hard = F.cross_entropy(logits, move_idx, label_smoothing=args.label_smoothing)
        soft = soft_policy_loss(logits, soft_i, soft_p)
        p_loss = (1.0 - args.soft_alpha) * hard + args.soft_alpha * soft
        v_loss = F.cross_entropy(out["value_logits"], wdl.argmax(dim=-1))
        loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
    scaler.scale(loss).backward()
    return p_loss.item(), v_loss.item(), soft.item(), hard.item()


def train_step_hard(model, batch_input, move_targets, wdl_targets, args, scaler):
    move_targets = flip_move_targets(move_targets, batch_input["turn"], FLIP_TABLE)
    with autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        out = model(batch_input)
        p_loss = F.cross_entropy(
            out["policy_logits"], move_targets, label_smoothing=args.label_smoothing,
        )
        v_loss = F.cross_entropy(out["value_logits"], wdl_targets.argmax(dim=-1))
        loss = (p_loss + args.value_weight * v_loss) / args.accum_steps
    scaler.scale(loss).backward()
    return p_loss.item(), v_loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--soft-frac", type=float, default=0.55,
                        help="Fraction of optimizer steps that use soft MultiPV batches")
    parser.add_argument("--soft-alpha", type=float, default=0.7,
                        help="Within soft batches: weight on soft CE vs hard CE")
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--value-weight", type=float, default=0.15,
                        help="Low: policy is primary; WDL is cheap aux")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--shuffle-buffer", type=int, default=4096)
    parser.add_argument("--min-depth", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default="outputs/exp185_a40_deep_small")
    parser.add_argument("--soft-cache", type=str,
                        default="outputs/exp184_a40_wide_soft/soft_cache.pt")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Load model weights only (fresh optimizer/step) for finetune")
    parser.add_argument("--max-soft-rows", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--encoder-dim", type=int, default=None)
    args = parser.parse_args()

    if not args.go:
        print("DRY RUN. Pass --go to train.")
        print("  Smoke: python experiments/exp185_a40_deep_small.py --go --smoke")
        print("  Full:  python experiments/exp185_a40_deep_small.py --go")
        return

    if args.smoke:
        args.steps = 60
        args.log_interval = 5
        args.save_interval = 30
        args.eval_interval = 30
        args.warmup = 10
        args.batch_size = min(args.batch_size, 512)
        args.max_soft_rows = args.max_soft_rows or 8000

    model_config = DEFAULT_A40_DEEP_SMALL_CONFIG
    overrides = {}
    if args.hidden_dim is not None:
        overrides["hidden_dim"] = args.hidden_dim
    if args.num_layers is not None:
        overrides["num_layers"] = args.num_layers
    if args.num_heads is not None:
        overrides["num_heads"] = args.num_heads
    if args.encoder_dim is not None:
        overrides["encoder_dim"] = args.encoder_dim
    if overrides:
        model_config = ChessTransformerConfig(**{**model_config.to_dict(), **overrides})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    LOG_PATH = output_dir / "training.log"

    log("=" * 64)
    log("exp185: A40 deep-small (28L/256d) + soft MultiPV + NorMuon")
    log(f"  device={DEVICE} vocab={VOCAB_SIZE}")
    log(f"  config={model_config}")
    log(f"  soft_frac={args.soft_frac} soft_alpha={args.soft_alpha} value_w={args.value_weight}")
    log("  note: value head is light WDL aux only — primary objective is next-move policy")

    soft_data = cache_soft_dataset(Path(args.soft_cache), max_rows=args.max_soft_rows)
    n_soft = soft_data["board_array"].shape[0]
    # Hold out last 5k for eval when possible
    eval_holdout = min(5000, max(1024, n_soft // 20))
    train_soft_n = max(1, n_soft - eval_holdout)
    log(f"  soft train={train_soft_n:,} eval_holdout={eval_holdout:,}")

    model = build_model(model_config).to(DEVICE)
    n_params = count_parameters(model)
    log(f"  params={n_params/1e6:.1f}M full_dim_attn={model_config.full_dim_attention}")

    try:
        optimizer, muon_n, adam_n = build_normuon_optimizer(
            model, args.muon_lr, args.adam_lr, args.weight_decay,
        )
        log(f"  optimizer=NorMuon ({muon_n/1e6:.1f}M) + AdamW aux ({adam_n/1e6:.1f}M)")
    except ImportError:
        log("ERROR: pip install git+https://github.com/zichongli5/NorMuon.git")
        return

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    # bf16 on Ampere — GradScaler not needed but keep for safety with mixed ops
    scaler = GradScaler("cuda", enabled=False)

    step = 0
    best_metric = float("inf")
    resume_path = output_dir / "latest.pt"
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            log(f"  optimizer resume skipped: {e}")
        step = int(ckpt.get("step", 0))
        best_metric = float(ckpt.get("best_metric", best_metric))
        log(f"  resumed step={step}")
    elif args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        log(f"  init weights from {args.init_checkpoint} (fresh opt/step)")

    def lr_scale(s: int) -> float:
        if s < args.warmup:
            return (s + 1) / max(args.warmup, 1)
        progress = (s - args.warmup) / max(args.steps - args.warmup, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return args.min_lr_frac + (1.0 - args.min_lr_frac) * cosine

    def set_lrs(s: int) -> None:
        scale = lr_scale(s)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale

    hard_iter = iter(stream_hf_batches(
        batch_size=args.batch_size, device=DEVICE, seed=42,
        shuffle_buffer=args.shuffle_buffer, min_depth=args.min_depth,
    ))

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_config.to_dict(), "training": vars(args),
                   "n_params": n_params, "n_soft": n_soft}, f, indent=2)

    eff_bs = args.batch_size * args.accum_steps
    log(f"  batch={args.batch_size} accum={args.accum_steps} eff_bs={eff_bs}")
    log(f"  steps={args.steps:,} muon_lr={args.muon_lr} adam_lr={args.adam_lr}")
    log("=" * 64)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_p = accum_v = accum_soft = accum_hard = 0.0
    accum_n = 0
    soft_steps = hard_steps = 0
    t0 = time.time()
    positions = step * eff_bs
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42 + step)

    # VRAM probe
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        log(f"  vram allocated={torch.cuda.memory_allocated()/1e9:.2f}GB")

    while step < args.steps:
        if SHUTDOWN:
            save_checkpoint(model, optimizer, scaler, step, best_metric, resume_path, model_config, args)
            log(f"Saved on shutdown at step {step}")
            return

        use_soft = (torch.rand(1, generator=rng).item() < args.soft_frac)

        for _ in range(args.accum_steps):
            if use_soft:
                idx = torch.randint(0, train_soft_n, (args.batch_size,), generator=rng)
                batch = prepare_soft_batch(soft_data, idx, DEVICE)
                p, v, s, h = train_step_soft(model, batch, args, scaler)
                accum_soft += s
                accum_hard += h
                soft_steps += 1
            else:
                try:
                    batch_input, move_targets, wdl_targets = next(hard_iter)
                except StopIteration:
                    hard_iter = iter(stream_hf_batches(
                        batch_size=args.batch_size, device=DEVICE, seed=43 + step,
                        shuffle_buffer=args.shuffle_buffer, min_depth=args.min_depth,
                    ))
                    batch_input, move_targets, wdl_targets = next(hard_iter)
                p, v = train_step_hard(model, batch_input, move_targets, wdl_targets, args, scaler)
                hard_steps += 1
            accum_p += p
            accum_v += v
            accum_n += 1
            positions += args.batch_size

        gn = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        set_lrs(step)

        if step % args.log_interval == 0:
            elapsed = max(time.time() - t0, 1e-6)
            pos_s = positions / elapsed
            vram = ""
            if DEVICE.type == "cuda":
                vram = f" | vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
            log(
                f"step {step:,}/{args.steps:,} | "
                f"p={accum_p/accum_n:.4f} v={accum_v/accum_n:.4f} "
                f"soft={accum_soft/max(soft_steps,1):.4f} hardCE={accum_hard/max(soft_steps,1):.4f} | "
                f"mix soft_steps={soft_steps} hard_steps={hard_steps} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} gn={float(gn):.2f} | "
                f"{pos_s:.0f} pos/s{vram}"
            )
            accum_p = accum_v = accum_soft = accum_hard = 0.0
            accum_n = soft_steps = hard_steps = 0

        if step % args.eval_interval == 0:
            metrics = eval_soft_top1(model, soft_data, DEVICE, n=eval_holdout)
            log(f"  eval holdout top1={metrics['top1']*100:.2f}% soft_loss={metrics['soft_loss']:.4f} n={metrics['n']}")
            if metrics["soft_loss"] < best_metric:
                best_metric = metrics["soft_loss"]
                save_checkpoint(
                    model, optimizer, scaler, step, best_metric,
                    output_dir / "best.pt", model_config, args,
                )
                log(f"  new best soft_loss={best_metric:.4f}")

        if step % args.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, step, best_metric, resume_path, model_config, args)
            save_checkpoint(
                model, optimizer, scaler, step, best_metric,
                output_dir / f"step_{step:06d}.pt", model_config, args,
            )
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    save_checkpoint(model, optimizer, scaler, step, best_metric, resume_path, model_config, args)
    metrics = eval_soft_top1(model, soft_data, DEVICE, n=eval_holdout)
    log(f"Done. step={step:,} top1={metrics['top1']*100:.2f}% soft_loss={metrics['soft_loss']:.4f}")


if __name__ == "__main__":
    main()
