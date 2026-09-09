#!/usr/bin/env python3
"""GPU preflight for the 270M squares64 model. Do this before the overnight.

Measures training throughput, peak memory, inference latency, and a bounded
LR stability probe. 99M LRs are not assumed to transfer.

Writes recommend.batch_size / accum_steps so effective batch stays 64.
Enables activation checkpointing only if bs=64 OOMs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from chess_squares64 import (
    DEFAULT_270M_SQUARES64_CONFIG,
    EXPECTED_270M_PARAMS,
    average_recurrent_grads,
    build_squares64,
    count_parameters,
)


def fake_batch(bs: int, device: torch.device) -> dict:
    return {
        "fused_ids": torch.randint(0, 13, (bs, 64), device=device),
        "turn": torch.randint(0, 2, (bs,), device=device),
        "castling": torch.randint(0, 16, (bs,), device=device),
        "ep_file": torch.randint(0, 9, (bs,), device=device),
    }


def peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def make_model(device: torch.device, *, checkpoint: bool):
    cfg = DEFAULT_270M_SQUARES64_CONFIG
    if checkpoint:
        from dataclasses import replace
        cfg = replace(cfg, gradient_checkpointing=True)
    return build_squares64(cfg).to(device)


def try_step(model, opt, bs: int, device, *, checkpoint: bool) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    x = fake_batch(bs, device)
    t0 = time.perf_counter()
    out = model(x)
    loss = out["policy_logits"].float().pow(2).mean() + 0.15 * out["value_logits"].float().pow(2).mean()
    loss.backward()
    average_recurrent_grads(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "ok": True,
        "ms": round(dt * 1000, 1),
        "pos_s": round(bs / max(dt, 1e-6), 2),
        "peak_mb": round(peak_mb(), 1),
        "loss": float(loss.detach().cpu()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/exp270_squares64_pretrain/bench.json")
    ap.add_argument("--mix", default="outputs/exp270_mix_v1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = make_model(device, checkpoint=False)
    n = count_parameters(model)
    if n != EXPECTED_270M_PARAMS:
        raise SystemExit(f"params {n} != {EXPECTED_270M_PARAMS}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    report: dict = {
        "params": n,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "vram_gb": (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if device.type == "cuda" else None
        ),
        "steps": {},
        "recommend": {
            "batch_size": 64,
            "accum_steps": 1,
            "grad_checkpoint": False,
            "muon_lr": 0.012,
            "adam_lr": 0.00018,
            "note": "Pilot LRs are width-scaled guesses. Replace after this probe.",
        },
    }

    target_eff = 64
    chosen = None
    for ckpt in (False, True):
        for bs in (64, 32, 16, 8):
            if target_eff % bs != 0:
                continue
            key = f"bs{bs}_ckpt{int(ckpt)}"
            try:
                if bool(model.config.gradient_checkpointing) != ckpt:
                    model = make_model(device, checkpoint=ckpt)
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    model.train()
                try_step(model, opt, bs, device, checkpoint=ckpt)
                stats = try_step(model, opt, bs, device, checkpoint=ckpt)
            except RuntimeError as exc:
                report["steps"][key] = {"ok": False, "error": str(exc)[:300]}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            report["steps"][key] = stats
            if chosen is None and stats["ok"]:
                chosen = (bs, ckpt, stats)
            print(key, stats, flush=True)
        if chosen:
            break

    if chosen is None:
        raise SystemExit("270M does not fit this GPU even at bs=8 + checkpoint")

    bs, ckpt, stats = chosen
    rec = report["recommend"]
    rec["batch_size"] = bs
    rec["accum_steps"] = target_eff // bs
    rec["grad_checkpoint"] = ckpt
    rec["effective_batch"] = rec["batch_size"] * rec["accum_steps"]
    rec["measured_pos_s"] = stats["pos_s"]
    rec["measured_peak_mb"] = stats["peak_mb"]

    # Inference latency (eval, no grad), bs=1 and bs=64-or-fit
    model.eval()
    lat = {}
    with torch.no_grad():
        for ibs in (1, min(64, bs)):
            x = fake_batch(ibs, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat[str(ibs)] = round((time.perf_counter() - t0) * 1000, 2)
    report["inference_latency_ms"] = lat

    # Bounded LR stability: 8 steps at 3 muon-equivalent Adam LRs (no Polar here).
    # Flag exploding/NaN. Overnight still uses Polar-NorMuon.
    model.train()
    probe = {}
    for lr in (3e-4, 1.8e-4, 8e-5):
        m2 = make_model(device, checkpoint=ckpt)
        o2 = torch.optim.AdamW(m2.parameters(), lr=lr)
        losses = []
        ok = True
        try:
            for _ in range(8):
                x = fake_batch(bs, device)
                out = m2(x)
                loss = out["policy_logits"].float().pow(2).mean()
                if not torch.isfinite(loss):
                    ok = False
                    break
                loss.backward()
                average_recurrent_grads(m2)
                o2.step()
                o2.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
        except RuntimeError as exc:
            ok = False
            losses.append(str(exc)[:200])
        probe[str(lr)] = {"ok": ok, "losses": losses}
        del m2, o2
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["lr_probe_adam"] = probe
    finite = [k for k, v in probe.items() if v["ok"]]
    rec["lr_probe_stable"] = finite
    rec["hours_note"] = (
        "Estimate positions ≈ pos_s * 3600 * train_hours. "
        "First night is a learning curve, not a beat-the-99M test."
    )
    if stats["pos_s"]:
        rec["est_positions_10h"] = int(stats["pos_s"] * 3600 * 10.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["recommend"], indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
