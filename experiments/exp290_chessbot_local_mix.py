#!/usr/bin/env python3
"""exp290: continue published 34.7M ChessBot on an even local mix, with recurrence.

Published Maxlegrec/ChessBot, 2/6/2 wrap, train at N=2/3 (depth 16/22).
Even 20% mix: puzzles / endgame / middlegame / opening / SF19 soft.
Slow linear warmup. ChessFENS is not a train source. Does not stop exp287
and does not write the 99M squares64 incumbent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import threading
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache"))

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import chess
import torch
import torch.nn.functional as F

from chess_chessbot import (
    CHESSBOT_UCI_TO_IDX,
    CHESSBOT_VOCAB_SIZE,
    fens_to_planes,
)
from chess_chessbot_recurrent import (
    ARCH,
    BANK,
    PREFIX,
    PUBLISHED_LAYERS,
    REPO,
    SUFFIX,
    RecurrentChessBot,
    average_recurrent_grads,
    count_parameters,
    identity_errors,
    wrap_published,
)
from move_vocab import IDX_TO_UCI, _CASTLE_960_TO_STD
from build_organized_chess_mix import pack_puzzle, reconstruct_board


BUCKETS = ("puzzles", "endgame", "middlegame", "opening", "soft")
ENDGAME_SOURCE_KEYS = ("endgame", "endgame_sf19", "endgame_syzygy")
HOLDOUT_START = 8192
HOLDOUT_COUNT = 8192
VAL_CACHE_VERSION = 2
REVISIONS = {
    "soft": ("avewright/chess-soft-sf19", "68cef6c9ba62c62f904f25305f1a7489dab825e0"),
    "opening": ("avewright/lichess-opening-bestline", "9e2780a689ea3fbda4a33572469294df491486d0"),
    "middlegame": ("avewright/lichess-middlegame-bestline", "0b4ef23b57089d30a9da906f3c7ea62c8481f47b"),
    "endgame": ("avewright/lichess-endgame-bestline", "83c12b11daacd374089663146fcd9e26bf0d4661"),
    "endgame_sf19": ("avewright/endgame-dataset", "9277353b607609c2debb26123a74d8f0a3142a78"),
    "endgame_syzygy": ("avewright/chess-soft-syzygy", "3889985c482837aa3082e182d8f123e005e2c0d4"),
    "puzzles": ("Lichess/chess-puzzles", "479ea9bc9f681385f5adb23fa27a96c2dc8ae599"),
}
SF19_MIN_BUDGET = 100_000
SF19_MIN_DEPTH = 12


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compact_to_chessbot_idx(idx: int) -> int:
    uci = IDX_TO_UCI[int(idx)]
    if uci in _CASTLE_960_TO_STD:
        uci = _CASTLE_960_TO_STD[uci]
    if uci.endswith("n"):
        uci = uci[:-1]
    return CHESSBOT_UCI_TO_IDX.get(uci, -1)


def white_wdl_to_chessbot(wdl) -> torch.Tensor:
    """SF19 [P(White), P(draw), P(Black)] → ChessBot [black, draw, white]."""
    t = torch.as_tensor(wdl, dtype=torch.float32).reshape(-1)[:3]
    return t[[2, 1, 0]]


def soft_to_policy(move_idx, soft_indices, soft_probs) -> torch.Tensor | None:
    mass: dict[int, float] = {}
    for i, p in zip(list(soft_indices), list(soft_probs)):
        if p is None or float(p) <= 0 or int(i) < 0:
            continue
        j = compact_to_chessbot_idx(int(i))
        if j >= 0:
            mass[j] = mass.get(j, 0.0) + float(p)
    hard = compact_to_chessbot_idx(int(move_idx))
    if hard >= 0 and hard not in mass:
        mass[hard] = mass.get(hard, 0.0)
    if not mass:
        return None
    total = sum(mass.values())
    pol = torch.full((CHESSBOT_VOCAB_SIZE,), -1.0)
    if total <= 0:
        if hard < 0:
            return None
        pol[hard] = 1.0
        return pol
    for j, p in mass.items():
        pol[j] = p / total
    return pol


def board_fen(d: dict, i: int) -> str | None:
    board = reconstruct_board(d, i)
    if board is None:
        return None
    return board.fen()


def engine_row(rec: dict, *, value_valid: bool) -> dict | None:
    policy = soft_to_policy(rec["move_idx"], rec["soft_indices"], rec["soft_probs"])
    if policy is None:
        return None
    arr = torch.as_tensor(rec["board_array"], dtype=torch.int8).reshape(64)
    packed = {
        "board_array": arr.unsqueeze(0),
        "turn": torch.tensor([int(rec["turn"])], dtype=torch.int8),
        "castling": torch.tensor([int(rec["castling"])], dtype=torch.int8),
        "ep_square": torch.tensor([int(rec["ep_square"])], dtype=torch.int8),
    }
    fen = board_fen(packed, 0)
    if fen is None:
        return None
    if value_valid and rec.get("wdl") is not None:
        wdl = white_wdl_to_chessbot(rec["wdl"])
        if (not torch.isfinite(wdl).all()) or float(wdl.clamp(min=0).sum()) <= 0:
            value_valid = False
            wdl = torch.tensor([0.0, 1.0, 0.0])
        else:
            wdl = wdl.clamp(min=0)
            wdl = wdl / wdl.sum()
    else:
        wdl = torch.tensor([0.0, 1.0, 0.0])
        value_valid = False
    return {"fen": fen, "policy": policy, "wdl": wdl, "value_valid": bool(value_valid)}


def sf19_ok(rec: dict) -> bool:
    if int(rec.get("split", 0)) != 0:
        return False
    if int(rec.get("policy_mask", 1)) != 1:
        return False
    if int(rec.get("nodes_budget", 0)) < SF19_MIN_BUDGET:
        return False
    if int(rec.get("label_depth", 0)) < SF19_MIN_DEPTH:
        return False
    return True


def syzygy_ok(rec: dict) -> bool:
    try:
        w = int(rec.get("wdl", 0))
        z = int(rec.get("dtz", 0))
    except (TypeError, ValueError):
        return False
    return w in (-2, -1, 0, 1, 2) and abs(z) <= 1000


def parquet_files(repo: str, revision: str) -> list[str]:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(repo, revision=revision)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    if repo.endswith("chess-soft-sf19"):
        files = [f for f in files if not f.endswith("shard_000000.parquet")]
    return files


def holdout_bounds() -> tuple[int, int]:
    return HOLDOUT_START, HOLDOUT_START + HOLDOUT_COUNT


def row_in_holdout(file_index: int, row_index: int) -> bool:
    start, end = holdout_bounds()
    return file_index == 0 and start <= row_index < end


def keep_parquet_row(file_index: int, row_index: int, *, holdout: bool) -> bool:
    inside = row_in_holdout(file_index, row_index)
    return inside if holdout else not inside


def iter_round_robin(gens):
    its = [iter(g) for g in gens]
    while its:
        alive = []
        for it in its:
            try:
                yield next(it)
                alive.append(it)
            except StopIteration:
                pass
        its = alive


def endgame_source_spec(key: str):
    if key == "endgame":
        return None, False
    if key == "endgame_sf19":
        return sf19_ok, True
    if key == "endgame_syzygy":
        return syzygy_ok, False
    raise ValueError(f"unknown endgame source {key}")


def iter_endgame_source(key: str, rows_fn):
    gate, value = endgame_source_spec(key)
    repo, rev = REVISIONS[key]
    for rec in rows_fn(repo, rev):
        if gate is not None and not gate(rec):
            continue
        row = engine_row(rec, value_valid=value)
        if row:
            row["bucket"] = "endgame"
            row["endgame_source"] = key
            yield row


def iter_holdout_parquet_rows(repo: str, revision: str, skip: int | None = None):
    """Reserved first-shard window. `skip` is accepted only if it matches HOLDOUT_START."""
    if skip is not None and int(skip) != HOLDOUT_START:
        raise ValueError(f"holdout skip must be {HOLDOUT_START}, got {skip}")
    yield from iter_parquet_rows(repo, revision, holdout=True)


def take_bucket_rows(name: str, n: int, *, holdout: bool = False) -> list[dict]:
    stream = iter_bucket_from(name, holdout=holdout)
    rows = []
    for row in stream:
        rows.append(row)
        if len(rows) >= n:
            break
    return rows


def iter_bucket_from(name: str, *, holdout: bool = False):
    def rows_fn(repo, revision):
        return iter_parquet_rows(repo, revision, holdout=holdout)
    yield from iter_bucket(name, rows_fn=rows_fn)


def iter_parquet_rows(repo: str, revision: str, *, holdout: bool = False):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    files = parquet_files(repo, revision)
    if not files:
        return
    file_loop = files[:1] if holdout else files
    while True:
        for file_index, fn in enumerate(file_loop):
            path = hf_hub_download(repo, fn, repo_type="dataset", revision=revision)
            row_index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                cols = batch.to_pydict()
                n = len(next(iter(cols.values())))
                for i in range(n):
                    if keep_parquet_row(file_index, row_index, holdout=holdout):
                        yield {k: cols[k][i] for k in cols}
                    row_index += 1
        if holdout:
            return


def iter_bucket(name: str, rows_fn=None):
    rows_fn = rows_fn or iter_parquet_rows
    if name == "puzzles":
        repo, rev = REVISIONS["puzzles"]
        for rec in rows_fn(repo, rev):
            row, _ = pack_puzzle(rec)
            if row is None:
                continue
            packed = {
                "board_array": row["board_array"].reshape(1, 64),
                "turn": row["turn"].reshape(1),
                "castling": row["castling"].reshape(1),
                "ep_square": row["ep_square"].reshape(1),
            }
            fen = board_fen(packed, 0)
            if fen is None:
                continue
            policy = soft_to_policy(
                int(row["move_idx"]), row["soft_indices"].tolist(), row["soft_probs"].tolist(),
            )
            if policy is None:
                continue
            yield {
                "fen": fen, "policy": policy, "wdl": torch.tensor([0.0, 1.0, 0.0]),
                "value_valid": False, "bucket": "puzzles",
            }
        return

    if name == "soft":
        repo, rev = REVISIONS["soft"]
        for rec in rows_fn(repo, rev):
            if not sf19_ok(rec):
                continue
            row = engine_row(rec, value_valid=True)
            if row:
                row["bucket"] = "soft"
                yield row
        return

    if name in {"opening", "middlegame"}:
        repo, rev = REVISIONS[name]
        for rec in rows_fn(repo, rev):
            row = engine_row(rec, value_valid=False)
            if row:
                row["bucket"] = name
                yield row
        return

    if name == "endgame":
        yield from iter_round_robin(
            iter_endgame_source(key, rows_fn) for key in ENDGAME_SOURCE_KEYS
        )
        return

    raise ValueError(f"unknown bucket {name}")


def _prefetch(gen, depth: int = 256):
    q: queue.Queue = queue.Queue(maxsize=depth)

    def worker():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


def bucket_take(batch_size: int, seed: int) -> dict[str, int]:
    if batch_size < len(BUCKETS):
        raise ValueError("batch_size must cover one row per bucket")
    rng = random.Random(seed)
    base = batch_size // len(BUCKETS)
    extra = batch_size % len(BUCKETS)
    take = {name: base for name in BUCKETS}
    for name in rng.sample(list(BUCKETS), extra):
        take[name] += 1
    return take


def iter_even_mix(batch_size: int, seed: int, prefetch: bool = True):
    rng = random.Random(seed)
    raw = {name: iter_bucket(name) for name in BUCKETS}
    streams = {name: _prefetch(gen) if prefetch else gen for name, gen in raw.items()}
    while True:
        take = bucket_take(batch_size, rng.randrange(1 << 30))
        rows = []
        for name in BUCKETS:
            for _ in range(take[name]):
                rows.append(next(streams[name]))
        rng.shuffle(rows)
        planes = fens_to_planes([r["fen"] for r in rows])
        policy = torch.stack([r["policy"] for r in rows])
        wdl = torch.stack([r["wdl"] for r in rows])
        valid = torch.tensor([r["value_valid"] for r in rows], dtype=torch.bool)
        yield planes, policy, wdl, valid


def synthetic_batch(batch_size: int):
    board = chess.Board()
    from chess_chessbot import move_to_policy_index
    pol = torch.full((CHESSBOT_VOCAB_SIZE,), -1.0)
    pol[move_to_policy_index(chess.Move.from_uci("e2e4"))] = 0.7
    pol[move_to_policy_index(chess.Move.from_uci("d2d4"))] = 0.3
    row = {"fen": board.fen(), "policy": pol, "wdl": torch.tensor([0.2, 0.5, 0.3]), "value_valid": True}
    rows = [row] * batch_size
    return (
        fens_to_planes([r["fen"] for r in rows]),
        torch.stack([r["policy"] for r in rows]),
        torch.stack([r["wdl"] for r in rows]),
        torch.tensor([True] * batch_size),
    )


def iter_synthetic(batch_size: int):
    batch = synthetic_batch(batch_size)
    while True:
        yield batch


def depth_schedule(steps: int, depths: list[int], seed: int) -> list[int]:
    if not depths or any(d < 1 for d in depths):
        raise ValueError("depths must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    rng = random.Random(seed)
    out = []
    while len(out) < steps:
        cycle = list(depths)
        rng.shuffle(cycle)
        out.extend(cycle)
    return out[:steps]


def lr_factor(step: int, total: int, warmup: int, cosine: bool, floor: float) -> float:
    if warmup <= 0:
        scale = 1.0
    elif step <= 0:
        scale = 0.0
    elif step < warmup:
        scale = step / float(warmup)
    else:
        scale = 1.0
    if not cosine or step < warmup:
        return scale
    fraction = (step - warmup) / max(total - warmup, 1)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * fraction))


def policy_losses(logits: torch.Tensor, target: torch.Tensor, soft_alpha: float):
    mass = target.clamp(min=0)
    denom = mass.sum(dim=-1, keepdim=True)
    valid = denom.squeeze(-1) > 0
    if not bool(valid.any()):
        zero = logits.sum() * 0
        return zero, zero, valid
    mass = mass / denom.clamp(min=1e-8)
    logp = F.log_softmax(logits, dim=-1)
    soft = -(mass * logp).sum(dim=-1)[valid].mean()
    hard = F.cross_entropy(logits[valid], mass[valid].argmax(dim=-1))
    return (1.0 - soft_alpha) * hard + soft_alpha * soft, hard, valid


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


def policy_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    log_s = F.log_softmax(student.float(), dim=-1)
    log_t = F.log_softmax(teacher.float(), dim=-1)
    return (log_s.exp() * (log_s - log_t)).sum(-1).mean()


def stack_val_rows(rows: list[dict]) -> dict:
    return {
        "fen": [r["fen"] for r in rows],
        "policy": torch.stack([r["policy"] for r in rows]),
        "wdl": torch.stack([r["wdl"] for r in rows]),
        "valid": torch.tensor([r["value_valid"] for r in rows], dtype=torch.bool),
        "bucket": [r["bucket"] for r in rows],
    }


def val_cache_ok(cache: dict, n_per_bucket: int) -> bool:
    hold = cache.get("holdout") or {}
    return (
        int(cache.get("n_per_bucket") or 0) == n_per_bucket
        and list(cache.get("buckets") or []) == list(BUCKETS)
        and int(hold.get("start") or -1) == HOLDOUT_START
        and int(hold.get("count") or -1) == HOLDOUT_COUNT
        and int(hold.get("version") or 0) == VAL_CACHE_VERSION
    )


def ensure_val_cache(path: Path, n_per_bucket: int = 48) -> dict:
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if val_cache_ok(cache, n_per_bucket):
            return cache
        stale = path.with_name(path.stem + "_stale.pt")
        path.replace(stale)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in BUCKETS:
        got = take_bucket_rows(name, n_per_bucket, holdout=True)
        if len(got) < n_per_bucket:
            raise SystemExit(f"val cache short on {name}: {len(got)} < {n_per_bucket}")
        rows.extend(got)
    cache = stack_val_rows(rows)
    cache["n_per_bucket"] = n_per_bucket
    cache["buckets"] = list(BUCKETS)
    cache["holdout"] = {
        "start": HOLDOUT_START,
        "count": HOLDOUT_COUNT,
        "version": VAL_CACHE_VERSION,
        "endgame_sources": list(ENDGAME_SOURCE_KEYS),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(cache, tmp)
    tmp.replace(path)
    return cache


@torch.no_grad()
def run_val(model, teacher, cache: dict, device: torch.device, unrolls: list[int], batch_size: int = 16) -> dict:
    was_train = model.training
    model.eval()
    fens = cache["fen"]
    n = len(fens)
    out = {"n": n, "by_unrolls": {}}
    for u in unrolls:
        hard_sum = soft_sum = kl_sum = top_sum = 0.0
        count = 0
        for start in range(0, n, batch_size):
            sl = slice(start, start + batch_size)
            planes = fens_to_planes(fens[sl], device)
            policy = cache["policy"][sl].to(device)
            student = model(planes, recurrent_unrolls=u)["policy_logits"]
            teacher_logits = teacher(planes, recurrent_unrolls=1)["policy_logits"]
            pol, hard, valid = policy_losses(student, policy, 0.55)
            if not bool(valid.any()):
                continue
            mass = policy.clamp(min=0)
            pred = student.argmax(-1)
            tgt = mass.argmax(-1)
            top_sum += float((pred == tgt)[valid].float().sum())
            hard_sum += float(hard) * int(valid.sum())
            soft_sum += float(pol) * int(valid.sum())
            kl_sum += float(policy_kl(student, teacher_logits)) * int(valid.sum())
            count += int(valid.sum())
        denom = max(count, 1)
        out["by_unrolls"][str(u)] = {
            "hard": hard_sum / denom,
            "soft": soft_sum / denom,
            "kl": kl_sum / denom,
            "top1": top_sum / denom,
            "n": count,
            "depth": model.effective_depth(u),
        }
    if was_train:
        model.train()
    return out


def save_model(model: RecurrentChessBot, path: Path, metadata: dict, optimizer=None) -> None:
    payload = {
        "arch": ARCH,
        "repo": REPO,
        "split": [PREFIX, BANK, SUFFIX],
        "default_unrolls": model.default_unrolls,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "parameters": count_parameters(model),
        "experiment": metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["step"] = metadata.get("step", 0)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_model(path: Path, device: torch.device) -> RecurrentChessBot:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("arch") != ARCH:
        raise ValueError(f"expected {ARCH}, got {ckpt.get('arch')}")
    model = wrap_published(torch.device("cpu"), default_unrolls=int(ckpt.get("default_unrolls", 2)))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(device)


def train_loop(model, teacher, batches, train_cfg, out: Path, metadata: dict, start_step: int = 0):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    if metadata.get("optimizer_state_dict"):
        opt.load_state_dict(metadata.pop("optimizer_state_dict"))
    for group in opt.param_groups:
        group["initial_lr"] = float(train_cfg["lr"])
    model.train()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    steps = int(train_cfg["steps"])
    depths = depth_schedule(steps, list(train_cfg["unrolls"]), int(train_cfg["seed"]))
    warmup = int(train_cfg["warmup"])
    t0 = time.monotonic()
    seen = int(metadata.get("examples", 0))
    latest = {}
    batch_q: queue.Queue = queue.Queue(maxsize=4)

    def feeder():
        try:
            for item in batches:
                batch_q.put(item)
        finally:
            batch_q.put(None)

    threading.Thread(target=feeder, daemon=True).start()
    val_every = int(train_cfg.get("val_every", 0))
    val_cache = metadata.get("val_cache")
    print(json.dumps({"optimizer": "adamw", "device": str(device), "warmup": warmup,
                      "unrolls": train_cfg["unrolls"], "val_every": val_every}), flush=True)
    for step in range(start_step + 1, steps + 1):
        n = depths[step - 1]
        factor = lr_factor(
            step, steps, warmup, bool(train_cfg.get("cosine_decay", True)),
            float(train_cfg.get("min_lr_fraction", 0.1)),
        )
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * factor
        item = batch_q.get()
        if item is None:
            raise SystemExit("data stream ended before the requested step count")
        planes, policy, wdl, valid = item
        planes = planes.to(device, non_blocking=True)
        policy = policy.to(device, non_blocking=True)
        wdl = wdl.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out_n = model(planes, recurrent_unrolls=n)
        pol, hard, _ = policy_losses(out_n["policy_logits"], policy, float(train_cfg["soft_alpha"]))
        val, *_ = value_losses(out_n, wdl, valid, float(train_cfg["value_weight"]))
        if n == 1:
            student_1 = out_n["policy_logits"]
        else:
            student_1 = model(planes, recurrent_unrolls=1)["policy_logits"]
        with torch.no_grad():
            teacher_logits = teacher(planes, recurrent_unrolls=1)["policy_logits"]
        kl = float(train_cfg["kl_weight"]) * policy_kl(student_1, teacher_logits)
        loss = pol + val + kl
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        average_recurrent_grads(model, n)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(train_cfg.get("grad_clip", 1.0)), error_if_nonfinite=True,
        )
        opt.step()
        seen += int(planes.size(0))
        elapsed = time.monotonic() - t0
        latest = dict(
            step=step, loss=float(loss.detach()), policy=float(pol.detach()),
            value=float(val.detach()), kl=float(kl.detach()), hard=float(hard.detach()),
            grad_norm=float(norm), unrolls=n, depth=model.effective_depth(n),
            lr=opt.param_groups[0]["lr"], examples=seen,
            pos_per_s=seen / max(elapsed, 1e-6), elapsed_s=elapsed,
            value_frac=float(valid.float().mean()),
        )
        if step == 1 or step % int(train_cfg.get("log_every", 25)) == 0 or step == steps:
            with (out / "train.jsonl").open("a") as f:
                f.write(json.dumps(latest) + "\n")
            print(json.dumps(latest), flush=True)
        if step % int(train_cfg["save_every"]) == 0 or step == steps:
            save_meta = {k: v for k, v in metadata.items() if k not in {"optimizer_state_dict", "val_cache"}}
            save_model(model, out / "latest.pt", {**save_meta, **latest}, opt)
        if val_cache is not None and val_every and (step == 1 or step % val_every == 0 or step == steps):
            metrics = run_val(
                model, teacher, val_cache, device,
                list(train_cfg.get("val_unrolls", [1, 2, 3])),
                batch_size=int(train_cfg.get("val_batch_size", 16)),
            )
            row = {"step": step, "examples": seen, **metrics}
            with (out / "val.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"val": row}), flush=True)
    return latest


def play_pair(white, black, white_fn, black_fn, device, opening: list[str], ply_cap: int = 400):
    board = chess.Board()
    for uci in opening:
        board.push_uci(uci)
    while not board.is_game_over(claim_draw=True) and board.ply() < ply_cap:
        fn = white_fn if board.turn == chess.WHITE else black_fn
        move, _ = fn(white if board.turn == chess.WHITE else black, board, device, 0.0)
        if move not in board.legal_moves:
            raise RuntimeError(f"illegal {move} on {board.fen()}")
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "cap", 0.5
    if outcome.winner is None:
        return "draw", 0.5
    return ("white" if outcome.winner == chess.WHITE else "black"), (1.0 if outcome.winner == chess.WHITE else 0.0)


def eval_vs_chessbot(
    ckpt: Path,
    device: torch.device,
    out: Path,
    *,
    unrolls: int = 2,
    repeats: int = 2,
    ply_cap: int = 400,
) -> dict:
    from elo_eval_chessbot import chessbot_move, load_chessbot
    from harness.common import load_protocol

    ours = load_model(ckpt, device)
    ours.eval()
    theirs = load_chessbot(REPO, device)

    def our_move(model, board, dev, temperature=0.0):
        return model.select_move(board, dev, temperature=temperature, unrolls=unrolls)

    proto = load_protocol()
    openings = [list(o) for o in proto["openings"]]
    games = []
    our_score = 0.0
    for opening in openings:
        for we_white in (True, False):
            for repeat_idx in range(repeats):
                if we_white:
                    term, score = play_pair(
                        ours, theirs, our_move, chessbot_move, device, opening, ply_cap,
                    )
                    ours_pts = score
                    color = "white"
                else:
                    term, score = play_pair(
                        theirs, ours, chessbot_move, our_move, device, opening, ply_cap,
                    )
                    ours_pts = 1.0 - score
                    color = "black"
                our_score += ours_pts
                games.append({
                    "opening": opening or ["startpos"],
                    "our_color": color,
                    "repeat_idx": repeat_idx,
                    "termination": term,
                    "our_score": ours_pts,
                    "unrolls": unrolls,
                })
                print(json.dumps(games[-1]), flush=True)
    n = max(len(games), 1)
    wins = sum(1 for g in games if g["our_score"] == 1.0)
    draws = sum(1 for g in games if g["our_score"] == 0.5)
    losses = sum(1 for g in games if g["our_score"] == 0.0)
    report = {
        "checkpoint": str(ckpt.resolve()),
        "opponent": REPO,
        "unrolls": unrolls,
        "depth": ours.effective_depth(unrolls),
        "games": len(games),
        "score": our_score / n,
        "w": wins,
        "d": draws,
        "l": losses,
        "results": games,
        "note": "Greedy T=0 paired colors vs published ChessBot. Not a promotion decision.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "vs_chessbot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("games", "score", "w", "d", "l", "note")}), flush=True)
    return report


def prepare_identity(device: torch.device, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model = wrap_published(device, default_unrolls=2)
    start = chess.Board()
    planes = fens_to_planes([start.fen()], device)
    err = identity_errors(model, planes)
    n = count_parameters(model)
    report = {
        "arch": ARCH,
        "repo": REPO,
        "split": [PREFIX, BANK, SUFFIX],
        "layers": PUBLISHED_LAYERS,
        "parameters": n,
        "effective_depth_n1": model.effective_depth(1),
        "effective_depth_n2": model.effective_depth(2),
        "effective_depth_n3": model.effective_depth(3),
        "identity_max_abs": err,
        "ok": max(err.values()) < 1e-5,
    }
    (out / "identity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["ok"]:
        raise SystemExit(f"N=1 identity failed: {err}")
    return report


def smoke(cfg: dict, out: Path, device: torch.device) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model = wrap_published(device, default_unrolls=2)
    teacher = wrap_published(device, default_unrolls=1)
    train_cfg = dict(cfg["train"])
    train_cfg.update(steps=2, batch_size=2, warmup=2, save_every=2, log_every=1, unrolls=[1, 2])
    result = train_loop(model, teacher, iter_synthetic(2), train_cfg, out, {"mode": "smoke"})
    (out / "smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def train(cfg: dict, out: Path, device: torch.device, synthetic: bool, resume: Path | None):
    if out.exists() and any(out.iterdir()) and resume is None:
        raise SystemExit(f"output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["train"]["seed"])
    ident = prepare_identity(device, out)
    model = wrap_published(device, default_unrolls=2)
    teacher = wrap_published(device, default_unrolls=1)
    start_step = 0
    seen = 0
    opt_state = None
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_step = int(ckpt.get("step") or ckpt.get("experiment", {}).get("step") or 0)
        seen = int((ckpt.get("experiment") or {}).get("examples") or 0)
        opt_state = ckpt.get("optimizer_state_dict")
    train_cfg = dict(cfg["train"])
    batches = iter_synthetic(train_cfg["batch_size"]) if synthetic else iter_even_mix(
        train_cfg["batch_size"], train_cfg["seed"],
    )
    val_path = out.parent / "val_cache.pt"
    val_cache = None
    if not synthetic:
        val_cache = ensure_val_cache(val_path, int(train_cfg.get("val_per_bucket", 48)))
        print(json.dumps({"val_cache": str(val_path), "n": len(val_cache["fen"])}), flush=True)
    metadata = {
        "mode": "train",
        "identity": ident,
        "buckets": list(BUCKETS),
        "revisions": {k: list(v) for k, v in REVISIONS.items()},
        "examples": seen,
        "val_cache_path": str(val_path) if val_cache is not None else None,
        "val_n": len(val_cache["fen"]) if val_cache is not None else 0,
        "holdout": {"start": HOLDOUT_START, "count": HOLDOUT_COUNT, "version": VAL_CACHE_VERSION},
        "endgame_sources": list(ENDGAME_SOURCE_KEYS),
        "note": "Even 20% mix. ChessFENS banned. 99M incumbent untouched. CE is not Elo. Holdout window excluded from train. Endgame is round-robin across three sources.",
    }
    if opt_state is not None:
        metadata["optimizer_state_dict"] = opt_state
    (out / "train_manifest.json").write_text(json.dumps(
        {k: v for k, v in metadata.items() if k != "optimizer_state_dict"}, indent=2,
    ))
    metadata["val_cache"] = val_cache
    result = train_loop(model, teacher, batches, train_cfg, out, metadata, start_step=start_step)
    (out / "train_summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["identity", "smoke", "train", "eval"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/exp290_chessbot_local_mix.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--unrolls", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--ply-cap", type=int, default=400)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    out = a.out or ROOT / cfg["output"] / a.mode
    device = torch.device(a.device)
    print(
        f"exp290 published ChessBot wrap {PREFIX}/{BANK}/{SUFFIX} "
        f"train unrolls={cfg['train']['unrolls']} warmup={cfg['train']['warmup']} "
        f"even mix {BUCKETS}",
        flush=True,
    )
    if a.mode == "identity":
        prepare_identity(device, out)
    elif a.mode == "smoke":
        smoke(cfg, out, device)
    elif a.mode == "eval":
        ckpt = a.ckpt or ROOT / "outputs/exp290_chessbot_local_mix/preserved/step4000.pt"
        eval_vs_chessbot(
            ckpt, device, out, unrolls=a.unrolls, repeats=a.repeats, ply_cap=a.ply_cap,
        )
    else:
        train(cfg, out, device, synthetic=a.synthetic, resume=a.resume)


if __name__ == "__main__":
    main()
