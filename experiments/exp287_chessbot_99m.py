#!/usr/bin/env python3
"""exp287: ChessBot architecture at 99M, trained on ChessFENS.

Fresh init. Do not copy squares64 weights or write the 99M incumbent.
See docs/exp287_chessbot_99m.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import chess
import torch
import torch.nn.functional as F

from chess_chessbot import (
    CHESSBOT_VOCAB_SIZE,
    CHESSFENS_POLICY_SIZE,
    DEFAULT_99M_CHESSBOT_CONFIG,
    EXPECTED_99M_PARAMS,
    ChessBot99Config,
    average_recurrent_grads,
    build_chessbot99,
    count_parameters,
    fens_to_planes,
    flip_planes_opposite_color,
    flip_policy_vector,
    stm_wdl_to_chessbot,
)


CHESSBOT_REPO = "Maxlegrec/ChessBot"
CHESSFENS_REPO = "Maxlegrec/ChessFENS"
CHESSFENS_ROWS = 731_601_781


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def model_config(cfg: dict) -> ChessBot99Config:
    return ChessBot99Config.from_dict(cfg.get("model", {}))


def save_model(model, path: Path, metadata: dict, optimizer=None) -> None:
    raw = getattr(model, "_orig_mod", model)
    payload = {
        "arch": "chessbot99",
        "config": raw.config.to_dict(),
        "model_state_dict": {k: v.detach().cpu() for k, v in raw.state_dict().items()},
        "experiment": metadata,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["step"] = metadata.get("step", 0)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def build_optimizer(model, train_cfg: dict, device: torch.device):
    name = str(train_cfg.get("optimizer", "adamw"))
    wd = float(train_cfg.get("weight_decay", 0.01))
    if name == "polar_normuon" and device.type == "cuda":
        from autoresearch_8gb.train_trial import build_polar_normuon_optimizer
        opt, muon_n, adam_n = build_polar_normuon_optimizer(
            model,
            float(train_cfg.get("muon_lr", 0.02)),
            float(train_cfg.get("adam_lr", 3e-4)),
            wd,
            compile_polar=True,
        )
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        return opt, {"muon": muon_n, "adam": adam_n, "name": name}
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("adam_lr", train_cfg.get("lr", 3e-4))),
        weight_decay=wd,
    )
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    return opt, {"name": "adamw"}


def lr_factor(step: int, total: int, warmup: int, cosine: bool, floor: float) -> float:
    if warmup and step < warmup:
        return step / warmup
    if not cosine:
        return 1.0
    fraction = (step - warmup) / max(total - warmup, 1)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * fraction))


def pad_policy(policy: torch.Tensor) -> torch.Tensor:
    if policy.size(-1) == CHESSBOT_VOCAB_SIZE:
        return policy
    if policy.size(-1) != CHESSFENS_POLICY_SIZE:
        raise ValueError(f"policy dim {policy.size(-1)} is not 1858 or 1929")
    extra = policy.new_full(policy.shape[:-1] + (CHESSBOT_VOCAB_SIZE - CHESSFENS_POLICY_SIZE,), -1)
    return torch.cat([policy, extra], dim=-1)


def policy_losses(logits: torch.Tensor, policy: torch.Tensor, soft_alpha: float):
    target = pad_policy(policy).to(dtype=logits.dtype)
    mass = target.clamp(min=0)
    denom = mass.sum(dim=-1, keepdim=True)
    valid = denom.squeeze(-1) > 0
    if not bool(valid.any()):
        zero = logits.sum() * 0
        return zero, zero, valid
    mass = mass / denom.clamp(min=1e-8)
    logp = F.log_softmax(logits, dim=-1)
    soft = -(mass * logp).sum(dim=-1)
    hard = F.cross_entropy(logits, mass.argmax(dim=-1), reduction="none")
    soft = soft[valid].mean()
    hard = hard[valid].mean()
    return (1.0 - soft_alpha) * hard + soft_alpha * soft, hard, valid


def value_losses(result: dict, wdl: torch.Tensor, weight: float):
    hard = F.cross_entropy(result["value_logits"], wdl.argmax(dim=-1))
    logq = F.log_softmax(result["value_logits_q"], dim=-1)
    soft = -(wdl * logq).sum(dim=-1).mean()
    return weight * (hard + soft), hard, soft


def collate_fens(
    fens: list[str],
    policies,
    wdls,
    hflip_p: float,
    generator: torch.Generator | None = None,
):
    planes = fens_to_planes(fens)
    policy = torch.stack([pad_policy(torch.as_tensor(p, dtype=torch.float32)) for p in policies])
    wdl_stm = torch.stack([torch.as_tensor(w, dtype=torch.float32) for w in wdls])
    flip = torch.rand(len(fens), generator=generator) < hflip_p if hflip_p > 0 else torch.zeros(len(fens), dtype=torch.bool)
    if bool(flip.any()):
        planes = planes.clone()
        policy = policy.clone()
        planes[flip] = flip_planes_opposite_color(planes[flip])
        policy[flip] = flip_policy_vector(policy[flip])
    turn = planes[:, 0, 12] > 0.5
    return planes, policy, stm_wdl_to_chessbot(wdl_stm, turn), turn, flip.tolist()


def collate_rows(rows: list[dict], hflip_p: float, generator: torch.Generator | None = None):
    return collate_fens(
        [r["fen"] for r in rows],
        [r["policy"] for r in rows],
        [r["wdl"] for r in rows],
        hflip_p,
        generator,
    )


def synthetic_rows(n: int) -> list[dict]:
    board = chess.Board()
    policy = [-1.0] * CHESSFENS_POLICY_SIZE
    from chess_chessbot import move_to_policy_index
    policy[move_to_policy_index(chess.Move.from_uci("e2e4"))] = 0.7
    policy[move_to_policy_index(chess.Move.from_uci("d2d4"))] = 0.3
    row = {"fen": board.fen(), "wdl": [0.30, 0.46, 0.24], "policy": policy}
    return [dict(row) for _ in range(n)]


def stream_chessfens(repo: str):
    from datasets import load_dataset
    return load_dataset(repo, split="train", streaming=True)


def take_rows(stream, n: int) -> list[dict]:
    rows = []
    for item in stream:
        rows.append({"fen": item["fen"], "wdl": item["wdl"], "policy": item["policy"]})
        if len(rows) >= n:
            break
    return rows


def prefetch(batches, depth: int = 4):
    import queue
    import threading
    q: queue.Queue = queue.Queue(maxsize=depth)

    def worker():
        try:
            for item in batches:
                q.put(item)
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


def train_loop(model, batches, train_cfg, out: Path, metadata: dict, start_step: int = 0) -> dict:
    device = next(model.parameters()).device
    opt, opt_info = build_optimizer(model, train_cfg, device)
    if metadata.get("optimizer_state_dict"):
        opt.load_state_dict(metadata.pop("optimizer_state_dict"))
    model.train()
    steps = int(train_cfg["steps"])
    accum = max(int(train_cfg.get("accum_steps", 1)), 1)
    deadline = None
    if train_cfg.get("max_minutes"):
        deadline = time.monotonic() + float(train_cfg["max_minutes"]) * 60
    t0 = time.monotonic()
    seen = int(metadata.get("examples", 0))
    latest = {}
    batch_iter = prefetch(batches)
    print(json.dumps({"optimizer": opt_info, "device": str(device)}), flush=True)
    for step in range(start_step + 1, steps + 1):
        if deadline is not None and time.monotonic() >= deadline:
            save_model(model, out / "latest.pt", {**metadata, **latest, "stopped": "time"}, opt)
            break
        factor = lr_factor(
            step, steps, int(train_cfg.get("warmup", 0)),
            bool(train_cfg.get("cosine_decay", False)),
            float(train_cfg.get("min_lr_fraction", 1.0)),
        )
        for group in opt.param_groups:
            group["lr"] = group.get("initial_lr", group["lr"]) * factor
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            try:
                planes, policy, wdl, *_ = next(batch_iter)
            except StopIteration:
                raise SystemExit("data stream ended before the requested step count")
            planes = planes.to(device, non_blocking=True)
            policy = policy.to(device, non_blocking=True)
            wdl = wdl.to(device, non_blocking=True)
            out_m = model(planes)
            pol, _, _ = policy_losses(out_m["policy_logits"], policy, train_cfg["soft_alpha"])
            val, *_ = value_losses(out_m, wdl, train_cfg["value_weight"])
            loss = (pol + val) / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            step_loss += float(loss.detach())
            seen += int(planes.size(0))
        average_recurrent_grads(model)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(train_cfg.get("grad_clip", 1.0)), error_if_nonfinite=True,
        )
        opt.step()
        elapsed = time.monotonic() - t0
        latest = dict(
            step=step, loss=step_loss, grad_norm=float(norm), examples=seen,
            lr=[g["lr"] for g in opt.param_groups],
            pos_per_s=seen / max(elapsed, 1e-6),
            elapsed_s=elapsed,
        )
        if step == 1 or step % int(train_cfg.get("log_every", 25)) == 0 or step == steps:
            with (out / "train.jsonl").open("a") as f:
                f.write(json.dumps(latest) + "\n")
            print(json.dumps(latest), flush=True)
        if step % int(train_cfg["save_every"]) == 0 or step == steps:
            save_model(model, out / "latest.pt", {**metadata, **latest}, opt)
    return latest


def iter_synthetic(batch_size: int, hflip_p: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    while True:
        yield collate_rows(synthetic_rows(batch_size), hflip_p, g)[:3]


def iter_stream(repo: str, batch_size: int, hflip_p: float, seed: int, skip_rows: int = 0):
    g = torch.Generator().manual_seed(seed)
    fens, policies, wdls = [], [], []
    skipped = 0
    for item in stream_chessfens(repo):
        if skipped < skip_rows:
            skipped += 1
            continue
        fens.append(item["fen"])
        policies.append(item["policy"])
        wdls.append(item["wdl"])
        if len(fens) == batch_size:
            yield collate_fens(fens, policies, wdls, hflip_p, g)[:3]
            fens, policies, wdls = [], [], []


def prepare(cfg: dict, out: Path, device: torch.device) -> dict:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {out}")
    torch.manual_seed(cfg["train"]["seed"])
    model = build_chessbot99(model_config(cfg)).to(device)
    n = count_parameters(model)
    if n != EXPECTED_99M_PARAMS:
        raise SystemExit(f"params {n} != {EXPECTED_99M_PARAMS}")
    out.mkdir(parents=True, exist_ok=True)
    start = chess.Board()
    with torch.no_grad():
        pred = model(model.prepare_input(start, device))
    manifest = {
        "experiment": "exp287_chessbot_99m",
        "arch": "chessbot99",
        "dataset": cfg.get("dataset", CHESSFENS_REPO),
        "parameters": n,
        "unique_layers": model.config.unique_layers,
        "effective_depth": model.config.effective_depth,
        "config": cfg,
        "device": str(device),
        "startpos_policy_norm": float(pred["policy_logits"][0].float().norm()),
        "policy_vocab_sha256": fingerprint(ROOT / "chess_chessbot_policy.json"),
        "note": "Fresh init. 99M incumbent untouched. CE is not Elo.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save_model(model, out / "init.pt", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def smoke(cfg: dict, out: Path, device: torch.device, tiny: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if tiny:
        model = build_chessbot99(ChessBot99Config(
            d_model=32, d_ff=64, num_heads=4, dropout=0,
            prefix_layers=1, recurrent_layers=1, suffix_layers=1, recurrent_unrolls=2,
        )).to(device)
    else:
        model = build_chessbot99(model_config(cfg)).to(device)
        n = count_parameters(model)
        if n != EXPECTED_99M_PARAMS:
            raise SystemExit(f"params {n} != {EXPECTED_99M_PARAMS}")
    train_cfg = dict(cfg["train"])
    train_cfg.update(
        steps=2, batch_size=2, accum_steps=1, warmup=1, save_every=2, log_every=1,
        optimizer="adamw", torch_compile=False,
    )
    result = train_loop(model, iter_synthetic(2, 0.5, train_cfg["seed"]), train_cfg, out, {"mode": "smoke"})
    (out / "smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def train(cfg: dict, out: Path, device: torch.device, full_epoch: bool, synthetic: bool,
          resume: Path | None = None, max_minutes: float | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["train"]["seed"])
    train_cfg = dict(cfg["train"])
    if max_minutes is not None:
        train_cfg["max_minutes"] = max_minutes
    if full_epoch:
        eff = int(train_cfg["batch_size"]) * max(int(train_cfg.get("accum_steps", 1)), 1)
        train_cfg["steps"] = math.ceil(CHESSFENS_ROWS / eff)
    start_step = 0
    seen = 0
    opt_state = None
    if resume is not None:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model = build_chessbot99(ckpt.get("config") or model_config(cfg)).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        start_step = int(ckpt.get("step") or ckpt.get("experiment", {}).get("step") or 0)
        seen = int((ckpt.get("experiment") or {}).get("examples") or 0)
        opt_state = ckpt.get("optimizer_state_dict")
    else:
        model = build_chessbot99(model_config(cfg)).to(device)
    n = count_parameters(model)
    if n != EXPECTED_99M_PARAMS:
        raise SystemExit(f"params {n} != {EXPECTED_99M_PARAMS}")
    if train_cfg.get("torch_compile") and device.type == "cuda":
        model = torch.compile(model)
    skip = seen
    if synthetic:
        batches = iter_synthetic(train_cfg["batch_size"], train_cfg["hflip_p"], train_cfg["seed"])
        source = "synthetic"
    else:
        batches = iter_stream(
            cfg.get("dataset", CHESSFENS_REPO), train_cfg["batch_size"],
            train_cfg["hflip_p"], train_cfg["seed"], skip_rows=skip,
        )
        source = cfg.get("dataset", CHESSFENS_REPO)
    metadata = {
        "mode": "train",
        "source": source,
        "parameters": n,
        "hybrid": {"qk_norm": True, "swiglu": True, "polar_normuon": True},
        "full_epoch": full_epoch,
        "steps": train_cfg["steps"],
        "examples": seen,
    }
    if opt_state is not None:
        metadata["optimizer_state_dict"] = opt_state
    (out / "train_manifest.json").write_text(json.dumps(
        {k: v for k, v in metadata.items() if k != "optimizer_state_dict"}, indent=2,
    ))
    result = train_loop(model, batches, train_cfg, out, metadata, start_step=start_step)
    (out / "train_summary.json").write_text(json.dumps(result, indent=2))
    return result


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


def eval_vs_chessbot(cfg: dict, ckpt: Path, device: torch.device, out: Path) -> dict:
    from chess_inference import load_checkpoint
    from elo_eval_chessbot import chessbot_move, load_chessbot
    from harness.common import load_protocol

    ours = load_checkpoint(ckpt, device=device)
    theirs = load_chessbot(cfg.get("chessbot_repo", CHESSBOT_REPO), device)
    proto = load_protocol()
    openings = [list(o) for o in proto["openings"]]
    games = []
    our_score = 0.0
    for opening in openings:
        for we_white in (True, False):
            if we_white:
                term, score = play_pair(ours, theirs, ours.select_move, chessbot_move, device, opening)
                our_score += score
                color = "white"
            else:
                term, score = play_pair(theirs, ours, chessbot_move, ours.select_move, device, opening)
                our_score += 1.0 - score
                color = "black"
            games.append({
                "opening": opening, "our_color": color, "termination": term,
                "our_score": score if we_white else 1.0 - score,
            })
    report = {
        "checkpoint": str(ckpt),
        "opponent": cfg.get("chessbot_repo", CHESSBOT_REPO),
        "games": len(games),
        "score": our_score / max(len(games), 1),
        "results": games,
        "note": "Greedy T=0. Not a promotion decision by itself.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "vs_chessbot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("games", "score", "note")}, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "smoke", "train", "eval"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/exp287_chessbot_99m.json")
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tiny", action="store_true", help="smoke: small layers, not the 99M")
    p.add_argument("--synthetic", action="store_true", help="train on startpos only")
    p.add_argument("--full-epoch", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--vs-chessbot", action="store_true")
    p.add_argument("--sf", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    out = a.out or ROOT / cfg["output"] / a.mode
    device = torch.device(a.device)
    print(
        f"exp287 hybrid99 {DEFAULT_99M_CHESSBOT_CONFIG.d_model}d/"
        f"{DEFAULT_99M_CHESSBOT_CONFIG.d_ff}ff "
        f"unique={DEFAULT_99M_CHESSBOT_CONFIG.unique_layers} "
        f"effective={DEFAULT_99M_CHESSBOT_CONFIG.effective_depth} "
        f"params={EXPECTED_99M_PARAMS:,} swiglu+qknorm+polar dataset={CHESSFENS_REPO}",
        flush=True,
    )
    if a.mode == "prepare":
        prepare(cfg, out, device)
    elif a.mode == "smoke":
        smoke(cfg, out, device, tiny=a.tiny or device.type == "cpu")
    elif a.mode == "train":
        train(cfg, out, device, full_epoch=a.full_epoch, synthetic=a.synthetic,
              resume=a.resume, max_minutes=a.minutes)
    else:
        ckpt = a.ckpt or ROOT / cfg["output"] / "train" / "latest.pt"
        if a.sf:
            raise SystemExit(
                "SF screen: python -m harness.elo --ckpt "
                f"{ckpt} --out-prefix exp287_policy"
            )
        if not a.vs_chessbot:
            raise SystemExit("eval needs --vs-chessbot or --sf")
        eval_vs_chessbot(cfg, ckpt, device, out)


if __name__ == "__main__":
    main()
