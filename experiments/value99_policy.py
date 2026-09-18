#!/usr/bin/env python3
"""Train a ChessBot-shaped policy head on ChessFENS, starting from a Value99 trunk."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
import sys
import time

os.environ.setdefault('PYTHONUNBUFFERED', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch._dynamo

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from chess_chessbot import CHESSBOT_VOCAB_SIZE, CHESSFENS_POLICY_SIZE
from chess_value99_policy import PolicyConfig, build_value99_policy, load_value99_trunk
from data_loader import _maybe_load_hf_token_from_env
from experiments.exp287_chessbot_99m import collate_rows, pad_policy, policy_losses, prefetch, take_rows
from experiments.value99_pretrain import atomic, build_optimizer
from polar_normuon import unwrap_compiled
from scripts.prepare_value99_data import REPO as CHESSFENS_REPO, REVISION as CHESSFENS_REVISION


def emit(out: Path, row: dict) -> None:
    row = dict(time=time.time(), **row)
    print(json.dumps(row), flush=True)
    with (out / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    (out / 'status.json').write_text(json.dumps(row, indent=2))


def save(model, optimizer, out: Path, step: int, seen: int, extra: dict) -> None:
    core = unwrap_compiled(model)
    payload = dict(
        arch='value99_policy',
        config=core.config.__dict__,
        model={k: v.detach().cpu() for k, v in core.state_dict().items()},
        optimizer=optimizer.state_dict(),
        step=step,
        seen=seen,
        **extra,
    )
    atomic(payload, out / 'latest.pt')
    if extra.get('snapshot'):
        atomic(
            dict(arch='value99_policy', config=core.config.__dict__,
                 model={k: v.detach().cpu() for k, v in core.state_dict().items()}, step=step),
            out / f'step_{step:06}.pt',
        )


def stream_policy_rows(repo: str, revision: str, seed: int):
    _maybe_load_hf_token_from_env()
    files = sorted(x for x in HfApi().list_repo_files(repo, repo_type='dataset', revision=revision) if x.endswith('.parquet'))
    if not files:
        raise ValueError(f'No parquet files in {repo}@{revision}')
    random.Random(seed).shuffle(files)
    fs = HfFileSystem()
    for name in files:
        uri = f'datasets/{repo}@{revision}/{name}'
        with fs.open(uri, 'rb', block_size=1 << 20) as handle:
            pf = pq.ParquetFile(handle)
            names = set(pf.schema_arrow.names)
            if 'fen' not in names or 'policy' not in names:
                continue
            cols = [c for c in ('fen', 'policy', 'wdl') if c in names]
            for batch in pf.iter_batches(batch_size=256, columns=cols):
                for row in batch.to_pylist():
                    policy = row.get('policy')
                    if not row.get('fen') or policy is None:
                        continue
                    if len(policy) not in (CHESSFENS_POLICY_SIZE, CHESSBOT_VOCAB_SIZE):
                        continue
                    if row.get('wdl') is None:
                        row['wdl'] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
                    yield row


def iter_batches(stream, batch: int, hflip_p: float, rng: torch.Generator):
    while True:
        rows = take_rows(stream, batch)
        if len(rows) < batch:
            return
        yield collate_rows(rows, hflip_p, rng)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/value99_policy.json')
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/value99_policy')
    p.add_argument('--device', default='cuda')
    p.add_argument('--init', type=Path, help='Value99 checkpoint whose trunk is copied')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    out = a.out
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise ValueError('Output already exists')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.manual_seed(int(cfg.get('seed', 294)))
    rng = torch.Generator().manual_seed(int(cfg.get('seed', 294)))
    model = build_value99_policy(PolicyConfig(**cfg['model'])).to(device)
    trunk = None
    init_path = a.init or (ROOT / cfg['init'] if cfg.get('init') else None)
    if init_path:
        ckpt = torch.load(init_path, map_location='cpu', weights_only=False)
        trunk = load_value99_trunk(model, ckpt)
    n = sum(p.numel() for p in model.parameters())
    optimizer, opt_info = build_optimizer(model, cfg)
    compiled = bool(cfg.get('torch_compile')) and device.type == 'cuda'
    if compiled:
        torch._dynamo.config.cache_size_limit = max(int(getattr(torch._dynamo.config, 'cache_size_limit', 8)), 128)
        model = torch.compile(model, dynamic=False)
    step0 = 0
    seen = 0
    if a.resume:
        ckpt = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        unwrap_compiled(model).load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step0 = int(ckpt['step'])
        seen = int(ckpt.get('seen') or 0)
    emit(out, dict(
        stage='loaded', parameters=n, device=str(device), vocab=CHESSBOT_VOCAB_SIZE,
        chessfens_policy=CHESSFENS_POLICY_SIZE, optimizer=opt_info, torch_compile=compiled,
        init=str(init_path) if init_path else None, trunk_tensors=len(trunk['loaded']) if trunk else 0,
        target='ChessFENS LC0 policy → ChessBot 1929 logits',
    ))
    stream = stream_policy_rows(cfg.get('dataset', CHESSFENS_REPO), cfg.get('revision', CHESSFENS_REVISION), int(cfg.get('seed', 294)))
    batches = prefetch(iter_batches(stream, int(cfg['batch']), float(cfg.get('hflip_p', 0.5)), rng))
    started = time.monotonic()
    total = int(cfg['steps'])
    for step in range(step0 + 1, total + 1):
        if (out / 'STOP').exists():
            save(model, optimizer, out, step - 1, seen, {})
            emit(out, dict(stage='stopped', step=step - 1, examples=seen))
            return
        factor = min(step / max(int(cfg.get('warmup', 1)), 1), 1.0)
        if step > cfg.get('warmup', 0) and cfg.get('cosine_decay'):
            progress = (step - cfg['warmup']) / max(total - cfg['warmup'], 1)
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * factor
        planes, policy, *_ = next(batches)
        planes = planes.to(device, non_blocking=True)
        policy = pad_policy(policy.to(device, non_blocking=True))
        out_m = model(planes)
        loss, hard, valid = policy_losses(out_m['policy_logits'], policy, float(cfg.get('soft_alpha', 0.85)))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite policy loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        seen += int(planes.size(0))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if step == 1 or step % int(cfg.get('log_every', 20)) == 0:
            emit(out, dict(
                stage='train', step=step, examples=seen, loss=float(loss.detach()),
                hard=float(hard.detach()), valid=int(valid.sum()),
                grad_norm=float(norm), lr={('muon' if g.get('use_muon') else 'adam'): float(g['lr']) for g in optimizer.param_groups},
                step_seconds=time.monotonic() - started if step == 1 else None,
                positions_per_s=seen / max(time.monotonic() - started, 1e-6),
                peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None,
            ))
        if step % int(cfg.get('save_every', 1000)) == 0 or step == total:
            save(model, optimizer, out, step, seen, dict(snapshot=True))
            emit(out, dict(stage='checkpoint', step=step, examples=seen))
    emit(out, dict(stage='complete', steps=total, examples=seen))


if __name__ == '__main__':
    main()
