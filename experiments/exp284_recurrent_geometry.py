#!/usr/bin/env python3
"""Prepare or train a checkpoint-preserving 99M recurrent geometry model."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
import chess
import torch
from chess_squares64 import upgrade_with_geometry, count_parameters
from exp283_recurrent_depth import (
    model_from, fingerprint, save_model, depth_schedule, load_data, subset,
    exclude_overlap, train_arm, sweep,
)


def validate(config):
    t = config["train"]
    depth_schedule(t["steps"], t["train_depths"], t["seed"])
    if t["arm"] not in {"fixed", "variable"}:
        raise ValueError("arm must be fixed or variable")
    if min(t["batch_size"], t["save_every"], config["data"]["eval_rows"]) < 1:
        raise ValueError("Batch, save interval and eval rows must be positive")
    if min(t["lr"], t["geometry_lr"]) <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if not 0 <= t["warmup"] < t["steps"] or not 0 <= t["min_lr_fraction"] <= 1:
        raise ValueError("Invalid LR schedule")
    if not 0 <= t["soft_alpha"] <= 1 or t["value_weight"] < 0:
        raise ValueError("Invalid loss coefficients")
    if not config["eval_depths"] or min(config["eval_depths"]) < 1:
        raise ValueError("Invalid evaluation depths")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "train"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/exp284_recurrent_geometry_99m.json")
    p.add_argument("--attention", choices=["none", "gab", "shaw", "both"])
    p.add_argument("--arm", choices=["fixed", "variable"])
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if a.attention:
        cfg["model"]["geometry_attention"] = a.attention
    if a.arm:
        cfg["train"]["arm"] = a.arm
    validate(cfg)
    out = a.out or ROOT / cfg["output"] / (cfg["model"]["geometry_attention"] + "_" + cfg["train"]["arm"] + "_" + a.mode)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be new or empty")
    torch.manual_seed(cfg["train"]["seed"])
    device = torch.device(a.device)
    source = ROOT / cfg["checkpoint"]
    original = model_from(source, device)
    original.eval()
    upgraded = upgrade_with_geometry(original, **cfg["model"]).eval()
    boards = [chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")]
    x = original.encoder.prepare_batch(boards, device)
    with torch.no_grad():
        before, after = original(x), upgraded(x)
        differences = {}
        for key in ("policy_logits", "value_logits"):
            torch.testing.assert_close(before[key], after[key], rtol=2e-4, atol=2e-4)
            differences[key] = (before[key] - after[key]).abs().max().item()
    del original, before, after
    manifest = dict(config=cfg, mode=a.mode, source_sha256=fingerprint(source),
                    device=str(device), torch_version=torch.__version__,
                    parameters=count_parameters(upgraded), initial_max_abs_error=differences,
                    note="No automatic promotion. Evaluation cache may overlap historical pretraining.")
    data, evaluation = None, None
    if a.mode == "train":
        full_eval = load_data(ROOT / cfg["data"]["eval_cache"])
        data, removed = exclude_overlap(load_data(ROOT / cfg["data"]["train_cache"]), full_eval)
        rng = torch.Generator().manual_seed(cfg["train"]["seed"])
        indices = torch.randperm(len(full_eval["turn"]), generator=rng)[:cfg["data"]["eval_rows"]]
        evaluation = subset(full_eval, indices)
        manifest.update(eval_indices=indices.tolist(), removed_overlap=removed,
                        train_rows=len(data["turn"]),
                        train_sha256=fingerprint(ROOT / cfg["data"]["train_cache"]),
                        eval_sha256=fingerprint(ROOT / cfg["data"]["eval_cache"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save_model(upgraded, out / "init.pt", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    if a.mode == "train":
        t = SimpleNamespace(**cfg["train"])
        schedule = depth_schedule(t.steps, t.train_depths, t.seed)
        if t.arm == "fixed":
            schedule = [3] * t.steps
        train_arm(upgraded, data, schedule, t, out)
        result = sweep(upgraded, evaluation, cfg["eval_depths"], device, t.batch_size)
        (out / "sweep.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
