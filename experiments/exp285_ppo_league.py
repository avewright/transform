#!/usr/bin/env python3
"""Persistent PPO + frozen-reference KL + supervised replay + historical league.

See docs/exp285_ppo_league.md. Smoke output is not Elo evidence.

Train: python experiments/exp285_ppo_league.py --config configs/exp285_ppo_99m.json
Smoke: python experiments/exp285_ppo_league.py --smoke --out outputs/exp285_smoke
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import torch
from chess_inference import load_checkpoint
from chess_squares64 import Squares64RecurrentConfig, build_squares64
from rl_selfplay.config import OPENINGS
from rl_selfplay.ppo import PPOConfig, check_model, collect_rollouts, make_optimizer, ppo_update
from autoresearch_8gb.pipeline import attach_static_targets


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload, path):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def model_payload(model):
    return dict(arch="squares64", config=model.config.to_dict(),
                model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()})


def checkpoint(model, optimizer, iteration, reference_path, league, config, rng, py_rng):
    return dict(model_payload(model), optimizer_state_dict=optimizer.state_dict(),
                iteration=iteration, reference_path=str(reference_path), league=[str(p) for p in league],
                run_config=config, sampler_rng=rng.get_state(), python_rng=py_rng.getstate(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_training(payload, optimizer, rng, py_rng):
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng.set_state(payload["sampler_rng"])
    py_rng.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


def load_replay(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = ("board_array", "turn", "castling", "ep_square", "move_idx",
                "cp", "mate", "soft_indices", "soft_probs")
    n = len(data["turn"])
    if not n or any(k not in data or len(data[k]) != n for k in required):
        raise ValueError(f"Empty or malformed replay cache: {path}")
    keep = torch.ones(n, dtype=torch.bool)
    if "split" in data:
        keep &= data["split"].reshape(-1) == 0
    if "policy_mask" in data:
        keep &= data["policy_mask"].reshape(-1).bool()
    if not keep.any():
        raise ValueError("Replay has no valid training rows")
    data = {k: v[keep] if torch.is_tensor(v) and v.ndim and len(v) == n else v for k, v in data.items()}
    return attach_static_targets(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp285_ppo_99m.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny randomly initialized CPU/GAB pipeline test; not Elo evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    run = json.loads(args.config.read_text())
    cfg = PPOConfig(**run["ppo"])
    cfg.validate()
    if run["iterations"] < 1 or run["league_size"] < 1 or not 0 <= run["reference_opponent_fraction"] <= 1:
        raise ValueError("Invalid league/run configuration")
    if args.smoke:
        cfg = replace(cfg, games_per_iteration=4, rollout_batch_size=4, ply_cap=12,
                      epochs=1, minibatch_size=8, microbatch_size=2, replay_weight=0., depths=(2, 3, 4))
        run.update(iterations=2, smoke=True, ppo=asdict(cfg))
    out = args.out or ROOT / run["output"]
    if not args.resume and out.exists() and any(out.iterdir()):
        parser.error("Use a new output directory, or --resume for an existing PPO run")
    if args.resume and not (out / "latest.pt").exists():
        parser.error("No latest.pt to resume")
    device = torch.device("cpu" if args.smoke else args.device)
    torch.manual_seed(run["seed"])
    rng = torch.Generator().manual_seed(run["seed"])
    py_rng = random.Random(run["seed"])
    replay = None
    if cfg.replay_weight:
        path = ROOT / run["replay_cache"]
        run["replay_sha256"] = sha256(path)
        replay = load_replay(path)
    payload = None
    if args.resume:
        payload = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        if payload["run_config"] != run:
            raise ValueError("Resume requires identical run configuration and replay hash")
        model = load_checkpoint(out / "latest.pt", device)
        reference_path = Path(payload["reference_path"])
        league = [Path(p) for p in payload["league"]]
        start = payload["iteration"]
    else:
        if args.smoke:
            torch.set_num_threads(1)
            model = build_squares64(Squares64RecurrentConfig(encoder_dim=16, hidden_dim=32,
                num_heads=4, prefix_layers=1, recurrent_layers=1, suffix_layers=1,
                policy_head_dim=16, value_hidden=16, dropout=.1, geometry_attention="gab",
                gab_d1=4, gab_d2=8, gab_d3=4)).to(device)
        else:
            model = load_checkpoint(ROOT / run["checkpoint"], device)
        check_model(model)
        out.mkdir(parents=True, exist_ok=True)
        reference_path = (out / "reference.pt").resolve()
        atomic_save(model_payload(model), reference_path)
        league, start = [], 0
    n_params = check_model(model)
    reference = load_checkpoint(reference_path, device).eval()
    reference.requires_grad_(False)
    optimizer = make_optimizer(model, cfg)
    if payload is not None:
        restore_training(payload, optimizer, rng, py_rng)
    else:
        manifest = dict(run_config=run, device=str(device), parameters=n_params,
                        reference_sha256=sha256(reference_path), torch_version=torch.__version__,
                        source_sha256=None if args.smoke else sha256(ROOT / run["checkpoint"]))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        atomic_save(checkpoint(model, optimizer, 0, reference_path, league, run, rng, py_rng), out / "latest.pt")
    print(f"PPO actor: {n_params:,} parameters; device={device}; iterations={start}..{run['iterations']}", flush=True)
    for iteration in range(start + 1, run["iterations"] + 1):
        t0 = time.monotonic()
        # STOP is checked at an iteration boundary: no partial/stale rollout reuse.
        if (out / "STOP").exists():
            print("STOP found; resume from latest completed iteration", flush=True)
            break
        selected = (reference_path if not league or py_rng.random() < run["reference_opponent_fraction"]
                    else py_rng.choice(league))
        opponent = reference if selected == reference_path else load_checkpoint(selected, device).eval()
        rows, games = collect_rollouts(model, opponent, cfg, OPENINGS, device, rng, py_rng)
        if opponent is not reference:
            del opponent
        metrics = ppo_update(model, reference, optimizer, rows, cfg, device, rng, replay)
        # Model snapshots bound the historical league; optimizer lives in latest.pt.
        snapshot = (out / f"actor_{iteration:06d}.pt").resolve()
        atomic_save(model_payload(model), snapshot)
        league.append(snapshot)
        obsolete = league[:-run["league_size"]]
        league = league[-run["league_size"]:]
        atomic_save(checkpoint(model, optimizer, iteration, reference_path, league, run, rng, py_rng), out / "latest.pt")
        # Prune only this run's generated snapshots, after checkpoint commit.
        for path in obsolete:
            if path.parent == out.resolve() and path.name.startswith("actor_"):
                path.unlink(missing_ok=True)
        stats = dict(iteration=iteration, opponent=str(selected), elapsed_s=time.monotonic() - t0,
                     wins=sum(g["reward"] == 1 for g in games),
                     draws=sum(g["reward"] == 0 for g in games),
                     losses=sum(g["reward"] == -1 for g in games),
                     truncated=sum(g["reward"] is None for g in games), **metrics)
        (out / f"games_{iteration:06d}.json").write_text(json.dumps(games, indent=2))
        (out / f"update_{iteration:06d}.json").write_text(json.dumps(stats, indent=2))
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({k: v for k, v in stats.items() if k != "minibatches"}) + "\n")
        print(json.dumps({k: v for k, v in stats.items() if k != "minibatches"}), flush=True)
    print(f"Saved {out / 'latest.pt'}; no automatic promotion", flush=True)


if __name__ == "__main__":
    main()
