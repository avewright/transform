#!/usr/bin/env python3
"""Fine-tune gated N=3 ChessBot on avewright/chessbot-sf19-policy with Polar-NorMuon."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

import chess

from chess_chessbot import (
    CHESSBOT_VOCAB_SIZE,
    expand_chessbot_policy,
    fens_to_planes,
    legal_policy_mask,
    policy_index_to_move,
)
from chess_chessbot_recurrent import (
    ARCH,
    BANK,
    PREFIX,
    SUFFIX,
    RecurrentChessBot,
    RecurrentSplit,
    average_recurrent_grads,
    count_parameters,
    depth_identity_errors,
    wrap_published,
)
from experiments.value99_pretrain import atomic, build_optimizer, group_lrs
from polar_normuon import unwrap_compiled


REPO = "avewright/chessbot-sf19-policy"
VALUE_SOURCES = (1, 3)
COLS = (
    "fen", "policy_idx", "policy_p", "hard_idx", "wdl",
    "wdl_source", "split", "bucket",
)


def emit(out: Path, row: dict) -> None:
    row = dict(time=time.time(), **row)
    print(json.dumps(row), flush=True)
    with (out / "events.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    (out / "status.json").write_text(json.dumps(row, indent=2))


def lr_factor(step: int, total: int, warmup: int, cosine: bool, floor: float) -> float:
    if warmup > 0 and step < warmup:
        return step / float(warmup)
    if not cosine:
        return 1.0
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def expand_batch(idxs, probs) -> torch.Tensor:
    return torch.stack([
        torch.from_numpy(expand_chessbot_policy(ix, pr)) for ix, pr in zip(idxs, probs)
    ])


def policy_losses(logits: torch.Tensor, policy: torch.Tensor, hard_idx: torch.Tensor, soft_alpha: float):
    mass = policy.clamp(min=0)
    denom = mass.sum(dim=-1, keepdim=True)
    valid = (denom.squeeze(-1) > 0) & (hard_idx >= 0) & (hard_idx < logits.size(-1))
    if not bool(valid.any()):
        zero = logits.sum() * 0
        return zero, zero, zero, valid
    mass = mass / denom.clamp(min=1e-8)
    logp = F.log_softmax(logits, dim=-1)
    soft = -(mass * logp).sum(dim=-1)[valid].mean()
    hard = F.cross_entropy(logits[valid], hard_idx[valid])
    return (1.0 - soft_alpha) * hard + soft_alpha * soft, hard, soft, valid


def value_losses(result: dict, wdl: torch.Tensor, valid: torch.Tensor, weight: float):
    if not bool(valid.any()):
        zero = result["value_logits"].sum() * 0
        return zero, zero, zero
    v = result["value_logits"][valid]
    q = result["value_logits_q"][valid]
    t = wdl[valid]
    hard = F.cross_entropy(v, t.argmax(dim=-1))
    soft = -(t * F.log_softmax(q, dim=-1)).sum(dim=-1).mean()
    return weight * (hard + soft), hard, soft


def keep_row(row: dict, split: int) -> bool:
    return bool(row.get("fen")) and int(row.get("split", -1)) == int(split)


def collate(rows: list[dict], value_sources=VALUE_SOURCES):
    planes = fens_to_planes([r["fen"] for r in rows])
    policy = expand_batch([r["policy_idx"] for r in rows], [r["policy_p"] for r in rows])
    hard = torch.tensor([int(r["hard_idx"]) for r in rows], dtype=torch.long)
    wdl = torch.stack([torch.as_tensor(r["wdl"], dtype=torch.float32) for r in rows])
    valid = torch.tensor([int(r.get("wdl_source", 0)) in value_sources for r in rows], dtype=torch.bool)
    return planes, policy, hard, wdl, valid


def parquet_files(repo: str) -> list[str]:
    files = sorted(
        x for x in HfApi().list_repo_files(repo, repo_type="dataset") if x.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"No parquet files in {repo}")
    return files


def stream_split(repo: str, split: int, seed: int, files: list[str] | None = None):
    names = list(files or parquet_files(repo))
    random.Random(seed).shuffle(names)
    fs = HfFileSystem()
    for name in names:
        uri = f"datasets/{repo}/{name}"
        with fs.open(uri, "rb", block_size=1 << 20) as handle:
            pf = pq.ParquetFile(handle)
            have = set(pf.schema_arrow.names)
            cols = [c for c in COLS if c in have]
            if "fen" not in have or "policy_idx" not in have:
                continue
            for batch in pf.iter_batches(batch_size=256, columns=cols):
                for row in batch.to_pylist():
                    if keep_row(row, split):
                        yield row


def take_rows(stream, n: int) -> list[dict]:
    rows = []
    for row in stream:
        rows.append(row)
        if len(rows) >= n:
            break
    return rows


def iter_train(repo: str, batch: int, seed: int):
    epoch = 0
    files = parquet_files(repo)
    while True:
        stream = stream_split(repo, 0, seed + epoch, files)
        buf: list[dict] = []
        for row in stream:
            buf.append(row)
            if len(buf) == batch:
                yield collate(buf)
                buf = []
        epoch += 1


def synthetic_rows(n: int) -> list[dict]:
    from chess_chessbot import CHESSBOT_UCI_TO_IDX
    e2e4 = CHESSBOT_UCI_TO_IDX["e2e4"]
    d2d4 = CHESSBOT_UCI_TO_IDX["d2d4"]
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "policy_idx": [e2e4, d2d4, -1, -1, -1, -1, -1, -1],
        "policy_p": [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "hard_idx": e2e4,
        "wdl": [0.12, 0.56, 0.32],
        "wdl_source": 1,
        "split": 0,
        "bucket": "opening",
    }
    return [dict(row) for _ in range(n)]


def save(model, optimizer, out: Path, step: int, seen: int, extra: dict) -> None:
    core = unwrap_compiled(model)
    payload = dict(
        arch=ARCH,
        split=[PREFIX, BANK, SUFFIX],
        default_unrolls=int(core.default_unrolls),
        gate_extra=bool(core.gate_extra),
        model={k: v.detach().cpu() for k, v in core.state_dict().items()},
        optimizer=optimizer.state_dict(),
        step=step,
        seen=seen,
        **extra,
    )
    atomic(payload, out / "latest.pt")
    if extra.get("snapshot"):
        atomic(
            dict(
                arch=ARCH,
                split=[PREFIX, BANK, SUFFIX],
                default_unrolls=int(core.default_unrolls),
                model={k: v.detach().cpu() for k, v in core.state_dict().items()},
                step=step,
            ),
            out / f"step_{step:06d}.pt",
        )


@torch.no_grad()
def run_val(model, rows: list[dict], device: torch.device, unrolls: int, batch: int, cfg: dict) -> dict:
    was_train = model.training
    model.eval()
    hard_sum = soft_sum = top_sum = val_sum = 0.0
    n = n_val = 0
    for start in range(0, len(rows), batch):
        planes, policy, hard, wdl, valid = collate(rows[start:start + batch], tuple(cfg.get("value_sources", VALUE_SOURCES)))
        planes = planes.to(device)
        policy = policy.to(device)
        hard = hard.to(device)
        wdl = wdl.to(device)
        valid = valid.to(device)
        out = model(planes, recurrent_unrolls=unrolls)
        pol, h, s, ok = policy_losses(out["policy_logits"], policy, hard, float(cfg.get("soft_alpha", 0.85)))
        if bool(ok.any()):
            pred = out["policy_logits"].argmax(-1)
            top_sum += float((pred == hard)[ok].float().sum())
            hard_sum += float(h) * int(ok.sum())
            soft_sum += float(s) * int(ok.sum())
            n += int(ok.sum())
        if bool(valid.any()):
            _, vh, _ = value_losses(out, wdl, valid, 1.0)
            val_sum += float(vh) * int(valid.sum())
            n_val += int(valid.sum())
    if was_train:
        model.train()
    denom = max(n, 1)
    return {
        "n": n,
        "hard": hard_sum / denom,
        "soft": soft_sum / denom,
        "top1": top_sum / denom,
        "value_hard": val_sum / max(n_val, 1),
        "value_n": n_val,
        "gate": float(torch.tanh(unwrap_compiled(model).alpha)),
        "unrolls": unrolls,
        "depth": unwrap_compiled(model).effective_depth(unrolls),
    }


DEFAULT_OPENINGS = ROOT / "outputs/chessbot_rl_n2/development_openings.json"


def load_openings(path: Path | None, n: int) -> list[list[str]]:
    src = path if path and path.exists() else DEFAULT_OPENINGS
    if not src.exists():
        raise FileNotFoundError(f"no openings at {src}")
    rows = json.loads(src.read_text())
    openings = [list(row) for row in rows if row]
    if n > 0:
        openings = openings[:n]
    if not openings:
        raise ValueError("openings file is empty")
    return openings


def load_student(path: Path, device: torch.device) -> tuple[RecurrentChessBot, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unrolls = int(ckpt.get("default_unrolls") or 3)
    model = wrap_published(
        device, default_unrolls=unrolls, gate_extra=bool(ckpt.get("gate_extra", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def game_reward(board: chess.Board, we_white: bool, ply_cap: int):
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0
        won = (outcome.winner == chess.WHITE) == we_white
        return 1 if won else -1
    if board.ply() >= ply_cap:
        return None
    return None


@torch.no_grad()
def pick_moves(model, boards: list[chess.Board], device: torch.device, unrolls: int):
    if not boards:
        return []
    planes = fens_to_planes([b.fen() for b in boards], device)
    logits = model(planes, recurrent_unrolls=unrolls)["policy_logits"].float()
    moves = []
    for i, board in enumerate(boards):
        mask = legal_policy_mask(board, device)
        idx = int(logits[i].masked_fill(~mask, -1e9).argmax())
        moves.append(policy_index_to_move(idx, board))
    return moves


def freeze_trunk(model: RecurrentChessBot) -> dict:
    """Keep published weights fixed. Only the extra-pass gate can move."""
    trainable, frozen = [], 0
    for name, p in model.named_parameters():
        if name == "alpha" or name.endswith(".alpha"):
            p.requires_grad_(True)
            trainable.append(name)
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    if not trainable:
        raise ValueError("freeze_trunk left no trainable parameters")
    return dict(trainable=trainable, frozen=frozen, trainable_n=sum(
        p.numel() for n, p in model.named_parameters() if n in trainable
    ))


@torch.no_grad()
def greedy_match(student, teacher, openings, device, student_unrolls=3, teacher_unrolls=1, ply_cap=400, chunk=8):
    """Paired-color greedy games. Reward is from the student side."""
    games = []
    for start in range(0, len(openings), chunk):
        boards, colors, origins = [], [], []
        for opening in openings[start:start + chunk]:
            for we_white in (True, False):
                board = chess.Board()
                for uci in opening:
                    board.push_uci(uci)
                boards.append(board)
                colors.append(we_white)
                origins.append(list(opening))
        live = list(range(len(boards)))
        while live:
            finished = []
            for i in live:
                reward = game_reward(boards[i], colors[i], ply_cap)
                if reward is not None or boards[i].ply() >= ply_cap:
                    games.append(dict(
                        opening=origins[i], color=colors[i], reward=reward,
                        truncated=reward is None,
                        moves=[m.uci() for m in boards[i].move_stack],
                    ))
                    finished.append(i)
            live = [i for i in live if i not in finished]
            ours = [i for i in live if boards[i].turn == colors[i]]
            theirs = [i for i in live if i not in ours]
            for model, ids, unrolls in (
                (student, ours, student_unrolls),
                (teacher, theirs, teacher_unrolls),
            ):
                if not ids:
                    continue
                for i, move in zip(ids, pick_moves(model, [boards[i] for i in ids], device, unrolls)):
                    if move not in boards[i].legal_moves:
                        raise RuntimeError(f"illegal {move} on {boards[i].fen()}")
                    boards[i].push(move)
    w = sum(g["reward"] == 1 for g in games)
    d = sum(g["reward"] == 0 for g in games)
    l = sum(g["reward"] == -1 for g in games)
    u = sum(g["reward"] is None for g in games)
    n = max(len(games), 1)
    return dict(
        wins=w, draws=d, losses=l, unknown=u, n=len(games),
        score_bounds=[(w + 0.5 * d) / n, (w + 0.5 * d + u) / n],
        games=games,
    )


def evaluate_pair(
    student,
    teacher,
    openings,
    device: torch.device,
    *,
    student_unrolls: int,
    teacher_unrolls: int = 1,
    opponent: str = "original",
    question: str = "stronger",
    ply_cap: int = 400,
    step: int = 0,
    tag: str | None = None,
    extra: dict | None = None,
) -> dict:
    from rl_selfplay.chessbot_eval import paired_eval

    student.eval()
    teacher.eval()
    t0 = time.monotonic()
    match = greedy_match(
        student, teacher, openings, device,
        student_unrolls=student_unrolls, teacher_unrolls=teacher_unrolls, ply_cap=ply_cap,
    )
    summary = paired_eval(match, kind="development", question=question, opponent=opponent)
    gate = float(torch.tanh(student.alpha)) if hasattr(student, "alpha") else None
    summary.update(
        step=int(step),
        tag=tag or opponent,
        unrolls=student_unrolls,
        teacher_unrolls=teacher_unrolls,
        depth=student.effective_depth(student_unrolls) if hasattr(student, "effective_depth") else None,
        gate=gate,
        pairs=len(openings),
        elapsed_s=time.monotonic() - t0,
        score_bounds=match.get("score_bounds"),
        **(extra or {}),
    )
    return dict(match=match, summary=summary)


def evaluate_vs_chessbot(
    ckpt: Path,
    out: Path,
    device: torch.device,
    *,
    openings: list[list[str]],
    unrolls: int = 3,
    ply_cap: int = 400,
) -> dict:
    student, meta = load_student(ckpt, device)
    teacher = wrap_published(device, default_unrolls=1, gate_extra=True).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    result = evaluate_pair(
        student, teacher, openings, device,
        student_unrolls=unrolls, opponent="original",
        ply_cap=ply_cap, step=int(meta.get("step") or 0),
        extra=dict(checkpoint=str(Path(ckpt).resolve())),
    )
    summary = result["summary"]
    tag = f"{int(summary['step']):06d}"
    (out / f"eval_{tag}_original.json").write_text(json.dumps({**result["match"], **summary}, indent=2))
    del student, teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def identity_check(model: RecurrentChessBot, device: torch.device, unrolls: int) -> dict:
    planes = fens_to_planes([
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ], device)
    n1 = depth_identity_errors(model, planes, 1)
    depth = depth_identity_errors(model, planes, unrolls)
    return {
        "n1_max": max(n1.values()),
        "train_depth_max": max(depth.values()),
        "gate": float(torch.tanh(model.alpha)),
        "alpha": float(model.alpha.detach()),
        "effective_depth": model.effective_depth(unrolls),
    }


def train(cfg: dict, out: Path, device: torch.device, resume: bool, synthetic: bool) -> None:
    if out.exists() and any(out.iterdir()) and not resume:
        raise SystemExit(f"output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.get("seed", 296)))
    if device.type == "cuda":
        torch._dynamo.config.cache_size_limit = max(
            int(getattr(torch._dynamo.config, "cache_size_limit", 8)), 128,
        )
    unrolls = int(cfg.get("unrolls", 3))
    if synthetic:
        from chess_chessbot_recurrent import build_empty_chessbot
        model = RecurrentChessBot(
            build_empty_chessbot(num_layers=4, d_model=32, d_ff=64, num_heads=4),
            default_unrolls=unrolls,
            split=RecurrentSplit(1, 2, 1),
            gate_extra=bool(cfg.get("gate_extra", True)),
        ).to(device)
    else:
        model = wrap_published(
            device, default_unrolls=unrolls, gate_extra=bool(cfg.get("gate_extra", True)),
        ).to(device)
    n = count_parameters(model)
    identity = identity_check(model, device, unrolls)
    if identity["train_depth_max"] > 1e-4:
        raise SystemExit(f"gated N={unrolls} is not identity at init: {identity}")
    frozen = None
    train_cfg = dict(cfg)
    if train_cfg.get("freeze_trunk"):
        frozen = freeze_trunk(unwrap_compiled(model))
        train_cfg["optimizer"] = "adamw"
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(train_cfg.get("lr", train_cfg.get("adam_lr", 1e-3))),
            weight_decay=0.0,
        )
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        opt_info = dict(name="adamw", trainable=frozen["trainable"], frozen=frozen["frozen"])
    else:
        opt, opt_info = build_optimizer(model, train_cfg)
    step0 = 0
    seen = 0
    if resume:
        ckpt = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        unwrap_compiled(model).load_state_dict(ckpt["model"], strict=True)
        opt.load_state_dict(ckpt["optimizer"])
        step0 = int(ckpt["step"])
        seen = int(ckpt.get("seen") or 0)
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "chessbot_sf19_n3",
        "arch": ARCH,
        "dataset": cfg.get("dataset", REPO),
        "unrolls": unrolls,
        "effective_depth": model.effective_depth(unrolls),
        "parameters": n,
        "optimizer": opt_info,
        "config": cfg,
        "identity": identity,
        "freeze_trunk": frozen,
        "note": "wdl is already ChessBot [black, draw, white]. Do not remap.",
    }, indent=2))
    emit(out, dict(
        stage="loaded", parameters=n, device=str(device), unrolls=unrolls,
        depth=model.effective_depth(unrolls), optimizer=opt_info,
        freeze_trunk=frozen, **identity,
    ))
    value_sources = tuple(cfg.get("value_sources", VALUE_SOURCES))
    if synthetic:
        def batches():
            while True:
                yield collate(synthetic_rows(int(cfg["batch"])), value_sources)
        train_batches = batches()
        val_rows = synthetic_rows(min(8, int(cfg.get("val_size", 8))))
    else:
        train_batches = iter_train(cfg.get("dataset", REPO), int(cfg["batch"]), int(cfg.get("seed", 296)))
        val_rows = take_rows(stream_split(cfg.get("dataset", REPO), 1, int(cfg.get("seed", 296))), int(cfg.get("val_size", 1024)))
        if len(val_rows) < 32:
            raise SystemExit(f"holdout too small: {len(val_rows)}")
    emit(out, dict(stage="val_ready", val_rows=len(val_rows), train_split=0, holdout_split=1))
    if train_cfg.get("freeze_trunk"):
        model.eval()
    else:
        model.train()
    started = time.monotonic()
    total = int(train_cfg["steps"])
    use_bf16 = str(train_cfg.get("precision", "bf16")) == "bf16" and device.type == "cuda"
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    for step in range(step0 + 1, total + 1):
        if (out / "STOP").exists():
            save(model, opt, out, step - 1, seen, {})
            emit(out, dict(stage="stopped", step=step - 1, examples=seen))
            return
        factor = lr_factor(
            step, total, int(train_cfg.get("warmup", 0)),
            bool(train_cfg.get("cosine_decay", False)),
            float(train_cfg.get("min_lr_fraction", 0.1)),
        )
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * factor
        planes, policy, hard, wdl, valid = next(train_batches)
        planes = planes.to(device, non_blocking=True)
        policy = policy.to(device, non_blocking=True)
        hard = hard.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with amp:
            out_m = model(planes, recurrent_unrolls=unrolls)
            pol, hce, sce, ok = policy_losses(out_m["policy_logits"], policy, hard, float(cfg["soft_alpha"]))
            val, vh, vs = value_losses(out_m, wdl, valid, float(cfg["value_weight"]))
            loss = pol + val
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        average_recurrent_grads(model, unrolls)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.get("grad_clip", 1.0)), error_if_nonfinite=True,
        )
        opt.step()
        seen += int(planes.size(0))
        if step == 1 or step % int(cfg.get("log_every", 20)) == 0 or step == total:
            top1 = float((out_m["policy_logits"].argmax(-1) == hard)[ok].float().mean()) if bool(ok.any()) else 0.0
            emit(out, dict(
                stage="train", step=step, examples=seen, loss=float(loss.detach()),
                policy=float(pol.detach()), hard=float(hce.detach()), soft=float(sce.detach()),
                value=float(val.detach()), value_hard=float(vh.detach()),
                top1=top1, value_frac=float(valid.float().mean()),
                gate=float(torch.tanh(unwrap_compiled(model).alpha)),
                grad_norm=float(norm), lr=group_lrs(opt),
                pos_per_s=seen / max(time.monotonic() - started, 1e-6),
                peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
            ))
        if step % int(cfg["save_every"]) == 0 or step == total:
            save(model, opt, out, step, seen, dict(snapshot=True))
            emit(out, dict(stage="checkpoint", step=step, examples=seen))
        if step == 1 or step % int(cfg.get("val_every", 0) or 10**9) == 0 or step == total:
            metrics = run_val(model, val_rows, device, unrolls, int(cfg.get("val_batch", 64)), cfg)
            emit(out, dict(stage="val", step=step, examples=seen, **metrics))
    emit(out, dict(stage="complete", steps=total, examples=seen))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/chessbot_sf19_n3.json")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/chessbot_sf19_n3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    train(cfg, a.out, device, a.resume, a.synthetic)


if __name__ == "__main__":
    main()
