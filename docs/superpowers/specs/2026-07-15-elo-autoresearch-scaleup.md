# Scale-up card: promote 8GB autoresearch champions to A40 / 400M

When `outputs/autoresearch_8gb/champion.json` beats the lab baseline:

## Replay on A40 (same ~25M class, longer budget)

1. Copy `champion.json` → note `config.model` + `config.train`.
2. Train longer (e.g. 8–20k steps) with the same mix weights on A40:
   - Raise `batch_size` until ~80% VRAM (often 512–1024 for deep-small).
   - Keep NorMuon/AdamW choice from the champion.
3. Re-run `elo_eval_latest.py` with full bracket (games-per-opening-per-color=2).

## Transfer ideas to 400M meta (when VRAM returns)

| Lab finding | 400M action |
|-------------|-------------|
| Safer soft mix wins Elo | Apply `soft_frac` / `soft_alpha` / `deep_mix` to FT from FT3b-class ckpt |
| Meta+Shaw > rel_bias | Prefer meta attention; keep `use_rel_bias=false` |
| NorMuon > AdamW | Keep NorMuon LRs; don’t switch mid-FT |
| Compile helps pos/s | Enable `torch.compile` on A40 after one stable step |
| Soft-heavy mix hurts Elo | Cap soft_alpha; Elo-gate checkpoints (never soft_loss) |

## Do not

- Crown a 400M run because soft holdout improved  
- Blindly copy 8GB batch sizes to 400M  
- Skip Elo confirmation after scale-up  

## Dual high-Elo + Syzygy on A40 (~45GB)

Trial id: `dual_highelo_a40` (no grad-checkpoint, batch probe from 768, `max_vram_gb=40`, `torch.compile`, Elo every 500).

```bash
# laptop → pod (mix ~120MB)
scp outputs/autoresearch_8gb/highelo_puzzle_syzygy_mix.pt USER@POD:~/transform/outputs/autoresearch_8gb/

# on pod
tmux new -s dual 'bash scripts/run_dual_highelo_a40.sh'
# knobs: STEPS=12000 TRAIN_MINUTES=480 BATCH via search_space probe
```

Outputs: `outputs/autoresearch_8gb_a40/trials/dual_highelo_a40/`  
(`latest.pt`, `elo_gauntlet.jsonl`, `train.log`)

Harness entrypoint: `experiments/exp194_autoresearch_8gb.py`  
Spec: `docs/superpowers/specs/2026-07-15-elo-autoresearch-8gb-design.md`
