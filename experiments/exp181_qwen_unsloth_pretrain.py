"""Pretrain ChessQwen (board tower + Unsloth LoRA Qwen3-0.6B) on SF/Lichess shards.

RAM-conscious: uses ShardedChessLoader (one shard in RAM at a time).
VRAM: 4-bit Qwen + batch on GPU when available.

Requires CUDA + Unsloth for full training:
  pip install -e ".[unsloth]"

Usage:
  python experiments/exp181_qwen_unsloth_pretrain.py --smoke          # local trial, minimal RAM
  python experiments/exp181_qwen_unsloth_pretrain.py --steps 5000       # GPU + shards
  python experiments/exp181_qwen_unsloth_pretrain.py --shard-dir PATH
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_qwen_factory import (
    ChessQwenConfig,
    ChessQwenModel,
    build_chess_qwen,
    configure_backbone,
    count_parameters,
    get_optimizer_param_groups,
    load_chess_qwen_checkpoint,
    prepare_for_inference,
    prepare_for_training,
    _disable_backbone_gradient_checkpointing,
)
from chess_transformer_factory import ChessTransformerEncoderLayer
from data_loader import ShardedChessLoader, board_array_to_fused, compute_wdl, ep_square_to_file, load_training_data, get_batch_input, get_eval_batch_input
from move_vocab import LEGACY_UCI_TO_IDX, VOCAB_SIZE, legacy_to_compact_map, move_to_index, legal_move_mask

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHARD_DIR = ROOT / "outputs" / "exp139_massive_train" / "shards"
OUT_DIR = ROOT / "outputs" / "exp181_qwen_unsloth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_hint() -> str:
    if torch.cuda.is_available():
        return ""
    ver = torch.__version__
    if "+cpu" in ver or "cpu" in ver.split("+")[-1]:
        return (
            f"\nPyTorch is CPU-only ({ver}). Your GPU exists but this Python can't use it.\n"
            "Fix (Python 3.12 recommended):\n"
            "  py -3.12 -m pip install torch==2.8.0+cu129 --index-url https://download.pytorch.org/whl/cu129\n"
            "Then run: .\\scripts\\run_gpu.ps1 experiments/exp181_qwen_unsloth_pretrain.py --smoke\n"
        )
    return "\nCUDA unavailable — check drivers or close other GPU processes.\n"


def build_remap_tensor():
    remap = legacy_to_compact_map()
    legacy_size = max(LEGACY_UCI_TO_IDX.values()) + 1
    t = torch.full((legacy_size,), -1, dtype=torch.long)
    for old_idx, new_idx in remap.items():
        t[old_idx] = new_idx
    return t


REMAP = build_remap_tensor()


def remap_moves(move_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    compact = REMAP[move_idx.long()]
    valid = compact >= 0
    return compact.clamp(min=0), valid


def win_pct_from_shard(cp: torch.Tensor, mate: torch.Tensor) -> torch.Tensor:
    wdl = compute_wdl(cp, mate)
    return wdl[:, 0] + 0.5 * wdl[:, 1]


def prepare_batch_from_tensors(
    board_array, turn, castling, ep_square, move_idx, cp, mate, device,
):
    compact, valid = remap_moves(move_idx)
    win_pct = win_pct_from_shard(cp, mate)
    board_input = {
        "fused_ids": board_array_to_fused(board_array).to(device),
        "turn": turn.long().to(device),
        "castling": castling.long().to(device),
        "ep_file": ep_square_to_file(ep_square).long().to(device),
    }
    return board_input, compact.to(device), win_pct.to(device), valid.to(device)


def cosine_schedule(optimizer, warmup, total, peak_lr, min_lr):
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr / peak_lr, cosine)

    return LambdaLR(optimizer, lr_lambda)


def _advance_scheduler(sched, start_step: int) -> None:
    for _ in range(start_step):
        sched.step()


def _dummy_backbone(hidden_size: int = 1024, n_layers: int = 2):
    layer = ChessTransformerEncoderLayer(hidden_size, 8, hidden_size * 4, use_swiglu=True)

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([layer for _ in range(n_layers)])

        def forward(self, inputs_embeds=None, use_cache=False, **kwargs):
            x = inputs_embeds
            for block in self.layers:
                x = block(x)
            return type("Out", (), {"last_hidden_state": x})()

    return DummyBackbone()


def build_model(config: ChessQwenConfig, smoke: bool) -> ChessQwenModel:
    if smoke and DEVICE.type != "cuda":
        print("  Smoke mode (CPU): dummy Qwen backbone, no download")
        cfg = ChessQwenConfig(
            **{**config.to_dict(), "backbone_mode": "lora", "tower_layers": min(config.tower_layers, 3)}
        )
        backbone = _dummy_backbone().to(DEVICE)
        configure_backbone(backbone, cfg)
        return ChessQwenModel(backbone, hidden_size=1024, config=cfg).to(DEVICE)

    if config.backbone_mode == "unsloth" and DEVICE.type != "cuda":
        raise SystemExit("Unsloth requires CUDA. Use --smoke for a local CPU trial.")

    return build_chess_qwen(config, device=DEVICE)


def synthetic_batch(batch_size: int, device: torch.device):
    """Generate one batch in RAM — no dataset materialized."""
    board_input = {
        "fused_ids": torch.randint(0, 13, (batch_size, 64), device=device),
        "turn": torch.randint(0, 2, (batch_size,), device=device),
        "castling": torch.randint(0, 16, (batch_size,), device=device),
        "ep_file": torch.randint(0, 9, (batch_size,), device=device),
    }
    move_targets = torch.randint(0, VOCAB_SIZE, (batch_size,), device=device)
    win_pct = torch.rand(batch_size, device=device)
    valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    return board_input, move_targets, win_pct, valid


@torch.no_grad()
def evaluate_smoke(model, device, n_batches=20, batch_size=8):
    model.eval()
    if model.config.backbone_mode == "unsloth":
        prepare_for_inference(model)
    correct1 = total = 0
    for _ in range(n_batches):
        board_input, target_move, _, valid = synthetic_batch(batch_size, device)
        logits = model(board_input)["policy_logits"][valid.bool()]
        targets = target_move[valid.bool()]
        correct1 += (logits.argmax(dim=-1) == targets).sum().item()
        total += valid.sum().item()
    if model.config.backbone_mode == "unsloth":
        prepare_for_training(model)
    model.train()
    return {"top1": correct1 / max(total, 1)}


@torch.no_grad()
def evaluate_sharded(model, loader: ShardedChessLoader, device, max_batches=30):
    model.eval()
    if model.config.backbone_mode == "unsloth":
        prepare_for_inference(model)
    correct1 = correct3 = total = 0
    seen = 0
    for batch_input, move_targets, wdl_targets in loader:
        compact, valid = remap_moves(move_targets.cpu())
        win_pct = wdl_targets[:, 0] + 0.5 * wdl_targets[:, 1]
        board_input = {k: v.to(device) for k, v in batch_input.items()}
        v = valid.to(device)
        if v.sum() == 0:
            continue
        board_f = {k: val[v] for k, val in board_input.items()}
        out = model(board_f)
        logits = out["policy_logits"]
        targets = compact.to(device)[v]
        preds = logits.topk(3, dim=-1).indices
        correct1 += (preds[:, 0] == targets).sum().item()
        correct3 += (preds == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += v.sum().item()
        seen += 1
        if seen >= max_batches:
            break
    if model.config.backbone_mode == "unsloth":
        prepare_for_training(model)
    model.train()
    gc.collect()
    return {"top1": correct1 / max(total, 1), "top3": correct3 / max(total, 1)}


def train_step(model, board_input, target_move, win_pct, valid, optimizer):
    v = valid.bool()
    if v.sum() == 0:
        return None
    board_f = {k: val[v] for k, val in board_input.items()}
    out = model(board_f, move_targets=target_move[v], value_targets=win_pct[v])
    loss = out["loss"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return out


def run_smoke(model, config, args, stats):
    print(f"  Smoke training: {args.steps} steps, batch={args.batch_size} (synthetic, ~0 dataset RAM)")
    opt = AdamW(get_optimizer_param_groups(model), weight_decay=0.01)
    sched = cosine_schedule(opt, warmup=min(20, args.steps // 10), total=args.steps,
                            peak_lr=args.peak_lr, min_lr=1e-5)
    model.train()
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        board_input, target_move, win_pct, valid = synthetic_batch(args.batch_size, DEVICE)
        out = train_step(model, board_input, target_move, win_pct, valid, opt)
        if out is None:
            continue
        sched.step()

        if step % max(1, args.steps // 10) == 0 or step == 1:
            print(
                f"step {step}/{args.steps} loss={out['loss'].item():.4f} "
                f"pl={out['policy_loss'].item():.4f} "
                f"vl={out.get('value_loss', torch.tensor(0)).item():.4f}"
            )

        if step % args.eval_every == 0:
            metrics = evaluate_smoke(model, DEVICE)
            print(f"  eval top1={metrics['top1']:.2%} (random baseline ~{100/VOCAB_SIZE:.2f}%)")
            history.append({"step": step, **metrics})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DEVICE.type == "cuda":
        model.save_checkpoint(OUT_DIR / "smoke_final")
    else:
        torch.save({"config": config.to_dict(), "tower_state": {
            k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")
        }}, OUT_DIR / "smoke_tower.pt")

    _write_summary(config, args, stats, history, time.time() - t0)


def _parse_resume_step(resume_dir: Path) -> int:
    name = resume_dir.name
    if name.startswith("step_"):
        return int(name.split("_", 1)[1])
    raise ValueError(f"Cannot infer step from checkpoint dir: {resume_dir}")


def run_hf(model, config, args, stats, start_step=0):
    """Train on avewright/chess-positions-lichess-sf via data_loader (streams + caches)."""
    print(f"  Loading {args.n_train:,} train + {args.n_eval:,} eval from HF ({args.hf_dataset})...")
    train_tensors, eval_data, _eval_tensors = load_training_data(
        n_train=args.n_train,
        n_eval=args.n_eval,
        encoder_type="fused",
        seed=args.seed,
    )
    n_train = train_tensors["fused_ids"].shape[0]
    win_pct_all = train_tensors["wdl"][:, 0] + 0.5 * train_tensors["wdl"][:, 1]
    print(f"  Ready: {n_train:,} training positions")
    if start_step:
        print(f"  Resuming from step {start_step}/{args.steps}")

    opt = AdamW(get_optimizer_param_groups(model), weight_decay=0.01)
    sched = cosine_schedule(
        opt,
        warmup=min(500, args.steps // 10),
        total=args.steps,
        peak_lr=args.peak_lr,
        min_lr=1e-5,
    )
    if start_step:
        _advance_scheduler(sched, start_step)
    model.train()
    if start_step:
        _disable_backbone_gradient_checkpointing(model.backbone)
    else:
        prepare_for_training(model)

    history = []
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        idx = torch.randint(0, n_train, (args.batch_size,))
        board_input = get_batch_input(train_tensors, idx, "fused", DEVICE)
        compact, valid = remap_moves(train_tensors["move_idx"][idx])
        win_pct = win_pct_all[idx].to(DEVICE)

        out = train_step(model, board_input, compact.to(DEVICE), win_pct, valid.to(DEVICE), opt)
        if out is None:
            continue
        sched.step()

        if step % 100 == 0 or step == 1:
            print(
                f"step {step}/{args.steps} loss={out['loss'].item():.4f} "
                f"pl={out['policy_loss'].item():.4f} "
                f"vl={out.get('value_loss', torch.tensor(0)).item():.4f} "
                f"lr={sched.get_last_lr()[0]:.2e}",
                flush=True,
            )

        if step % args.eval_every == 0:
            metrics = evaluate_eval_data(model, eval_data, DEVICE)
            print(f"  eval top1={metrics['top1']:.2%} top3={metrics['top3']:.2%}", flush=True)
            history.append({"step": step, **metrics})
            ckpt = OUT_DIR / f"step_{step:06d}"
            model.save_checkpoint(ckpt)
            print(f"  saved {ckpt}", flush=True)
            gc.collect()

    model.save_checkpoint(OUT_DIR / "final")
    _write_summary(config, args, stats, history, time.time() - t0)


@torch.no_grad()
def evaluate_eval_data(model, eval_data, device, max_samples=2000):
    model.eval()
    prepare_for_inference(model)
    correct1 = correct3 = total = 0
    n = min(max_samples, len(eval_data))
    for i in range(n):
        item = eval_data[i]
        board = item["board"]
        target = move_to_index(item["move"])
        if target < 0:
            continue
        board_input = model.encoder.prepare_input(board, device)
        logits = model(board_input)["policy_logits"][0]
        mask = legal_move_mask(board).to(device)
        logits = logits.masked_fill(~mask, float("-inf"))
        preds = logits.topk(3, dim=-1).indices
        if preds[0].item() == target:
            correct1 += 1
        if (preds == target).any():
            correct3 += 1
        total += 1
    prepare_for_training(model)
    model.train()
    return {"top1": correct1 / max(total, 1), "top3": correct3 / max(total, 1)}


def run_sharded(model, config, args, stats, shard_dir: Path):
    train_loader = ShardedChessLoader(
        shard_dir, batch_size=args.batch_size, device="cpu", drop_last=True,
    )
    eval_loader = ShardedChessLoader(
        shard_dir, batch_size=args.batch_size, device="cpu", drop_last=False,
    )

    opt = AdamW(get_optimizer_param_groups(model), weight_decay=0.01)
    sched = cosine_schedule(opt, warmup=200, total=args.steps, peak_lr=args.peak_lr, min_lr=1e-5)
    model.train()
    prepare_for_training(model)

    history = []
    t0 = time.time()
    step = 0
    loader_iter = iter(train_loader)

    while step < args.steps:
        try:
            batch_input, move_targets, wdl_targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            gc.collect()
            continue

        win_pct = wdl_targets[:, 0] + 0.5 * wdl_targets[:, 1]
        compact, valid = remap_moves(move_targets.cpu())
        board_input = {k: v.to(DEVICE) for k, v in batch_input.items()}

        out = train_step(model, board_input, compact.to(DEVICE), win_pct.to(DEVICE), valid.to(DEVICE), opt)
        if out is None:
            continue
        step += 1
        sched.step()

        if step % 100 == 0 or step == 1:
            print(
                f"step {step}/{args.steps} loss={out['loss'].item():.4f} "
                f"pl={out['policy_loss'].item():.4f} "
                f"vl={out.get('value_loss', torch.tensor(0)).item():.4f} "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )

        if step % args.eval_every == 0:
            metrics = evaluate_sharded(model, eval_loader, DEVICE, max_batches=30)
            print(f"  eval top1={metrics['top1']:.2%} top3={metrics['top3']:.2%}")
            history.append({"step": step, **metrics})
            ckpt = OUT_DIR / f"step_{step:06d}"
            model.save_checkpoint(ckpt)
            print(f"  saved {ckpt}")
            gc.collect()

    model.save_checkpoint(OUT_DIR / "final")
    _write_summary(config, args, stats, history, time.time() - t0)


def _write_summary(config, args, stats, history, elapsed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": config.to_dict(),
        "steps": args.steps,
        "smoke": args.smoke,
        "hf": getattr(args, "hf", False),
        "n_train": getattr(args, "n_train", None),
        "params": stats,
        "history": history,
        "elapsed_s": elapsed,
        "vocab_size": VOCAB_SIZE,
        "device": str(DEVICE),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Done in {elapsed / 60:.1f} min -> {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Minimal RAM trial with synthetic data")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--tower-layers", type=int, default=4)
    parser.add_argument("--peak-lr", type=float, default=2e-4)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--hf", action="store_true", help="Stream from avewright/chess-positions-lichess-sf")
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-eval", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-dataset", default="avewright/chess-positions-lichess-sf")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint dir (e.g. outputs/.../step_004000)")
    args = parser.parse_args()

    if not args.smoke and DEVICE.type == "cuda" and args.batch_size > 4:
        args.batch_size = min(args.batch_size, 4)  # 8GB VRAM default cap

    if args.resume and not args.hf:
        args.hf = True

    if args.smoke:
        args.steps = min(args.steps, 200)
        args.tower_layers = min(args.tower_layers, 3)
        args.eval_every = min(args.eval_every, 50)

    config = ChessQwenConfig(
        qwen_name_or_path=args.model,
        backbone_mode="unsloth" if not args.smoke or DEVICE.type == "cuda" else "lora",
        lora_rank=args.lora_rank,
        tower_layers=args.tower_layers,
        load_in_4bit=not args.no_4bit,
        max_seq_length=128,
    )

    print(f"Device: {DEVICE}{_cuda_hint()}")
    if not args.smoke and DEVICE.type != "cuda":
        raise SystemExit("Full training requires CUDA." + _cuda_hint())
    start_step = 0
    if args.resume:
        print(f"Loading checkpoint {args.resume}...")
        model = load_chess_qwen_checkpoint(args.resume, device=DEVICE, for_training=True)
        start_step = _parse_resume_step(args.resume)
    else:
        print(f"Building ChessQwen (mode={config.backbone_mode}, tower={config.tower_layers}L)...")
        model = build_model(config, smoke=args.smoke)
    stats = count_parameters(model)
    print(f"Params: total={stats['total']:,} trainable={stats['trainable']:,}")

    if args.smoke:
        run_smoke(model, config, args, stats)
        return

    if args.hf:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        run_hf(model, config, args, stats, start_step=start_step)
        return

    if args.shard_dir.exists() and list(args.shard_dir.glob("shard_*.pt")):
        run_sharded(model, config, args, stats, args.shard_dir)
        return

    print("No local shards found — defaulting to HF dataset.")
    args.hf = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_hf(model, config, args, stats)


if __name__ == "__main__":
    main()
