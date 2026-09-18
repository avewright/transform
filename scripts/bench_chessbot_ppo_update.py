#!/usr/bin/env python3
"""Time PPO update variants on a saved rollout + checkpoint.

  python3 scripts/bench_chessbot_ppo_update.py --steps 8
"""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def peak_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1 << 20)


def load_pack(args):
    import sys
    sys.path.insert(0, str(ROOT))
    from rl_selfplay.chessbot_ppo import load_original

    device = torch.device(args.device)
    src = ROOT / 'outputs/chessbot_rl_source'
    published = load_original(src, device, unrolls=1)
    published.model.requires_grad_(False)
    actor0 = load_original(src, device, unrolls=2, gate_extra=True)
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    actor0.model.load_state_dict(state['actor'])
    rows = torch.load(args.rollout, map_location='cpu', weights_only=False)
    if isinstance(rows, dict) and 'rows' in rows:
        rows = rows['rows']
    bank = json.loads(args.bank.read_text())
    anchors = bank['anchor']
    if args.max_rows:
        rows = rows[: args.max_rows]
    return published, actor0, state, rows, anchors, device


def clone_pair(actor0, device, lr):
    actor = actor0.clone()
    control = actor0.clone()
    actor.model.to(device)
    control.model.to(device)
    opt = torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.)
    copt = torch.optim.AdamW([p for p in control.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.)
    return actor, control, opt, copt


def run_variant(name, cfg, actor0, published, rows, anchors, device, lr, seed):
    from rl_selfplay.chessbot_ppo import update

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    actor, control, opt, copt = clone_pair(actor0, device, lr)
    rec = []
    t0 = time.perf_counter()
    update(actor, control, published, opt, copt, deepcopy(rows), anchors, cfg, device,
           torch.Generator().manual_seed(seed), rec.append)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    first, last = rec[0], rec[-1]
    return dict(
        name=name, steps=len(rec), seconds=elapsed, sec_per_update=elapsed / max(len(rec), 1),
        peak_mb=peak_mb(),
        first=dict(policy=first['policy'], value=first['value'], reference_kl=first['reference_kl'],
                   behavior_kl=first['behavior_kl'], grad_norm=first['grad_norm']),
        last=dict(policy=last['policy'], value=last['value'], reference_kl=last['reference_kl'],
                  behavior_kl=last['behavior_kl'], grad_norm=last['grad_norm']),
        cfg={k: cfg[k] for k in cfg if k in (
            'microbatch', 'audit_microbatch', 'audit_cheap_size', 'audit_full_every', 'cache_rollout')},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', type=Path, default=ROOT / 'outputs/chessbot_rl_n2/latest.pt')
    p.add_argument('--rollout', type=Path, default=ROOT / 'outputs/chessbot_rl_n2/last_rollout.pt')
    p.add_argument('--bank', type=Path, default=ROOT / 'outputs/chessbot_rl_data/bank.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--max-rows', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--out', type=Path, default=ROOT / 'outputs/chessbot_rl_n2/update_bench.json')
    args = p.parse_args()
    published, actor0, state, rows, anchors, device = load_pack(args)
    print(json.dumps({'stage': 'loaded', 'iteration': state.get('iteration'),
                      'rows': len(rows), 'anchors': len(anchors)}), flush=True)
    base = dict(epochs=1, minibatch=256, microbatch=8, anchor_batch=128, audit_size=2048,
                audit_cheap_size=2048, audit_full_every=1, audit_microbatch=8,
                cache_rollout=False, prep_chunk=256)
    # Limit to `steps` minibatches by slicing rows to steps * 256
    rows = rows[: args.steps * 256]
    variants = [
        ('A_baseline_mb8', dict(base)),
        ('B_cache_mb8_audit128', {**base, 'cache_rollout': True, 'audit_microbatch': 128}),
        ('C_cache_mb32_audit128', {**base, 'cache_rollout': True, 'microbatch': 32,
                                   'audit_microbatch': 128}),
        ('D_cache_mb32_cheap256', {**base, 'cache_rollout': True, 'microbatch': 32,
                                   'audit_microbatch': 128, 'audit_cheap_size': 256,
                                   'audit_full_every': 8}),
        ('E_cache_mb64_cheap256', {**base, 'cache_rollout': True, 'microbatch': 64,
                                   'audit_microbatch': 128, 'audit_cheap_size': 256,
                                   'audit_full_every': 8}),
    ]
    results = []
    for name, cfg in variants:
        print(json.dumps({'stage': 'start', 'name': name}), flush=True)
        row = run_variant(name, cfg, actor0, published, rows, anchors, device, args.lr, 291)
        results.append(row)
        print(json.dumps({'stage': 'done', **{k: row[k] for k in row if k not in ('first', 'last')},
                          'first': row['first']}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint=str(args.ckpt), rollout=str(args.rollout),
                   iteration=state.get('iteration'), rows=len(rows), variants=results)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({'stage': 'wrote', 'out': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
