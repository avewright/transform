# SF19 overnight run — September 8, 2026

The remote job runs in tmux session `overnight-sf19` on `runpod-transform`.
It starts after run2 reaches its existing 4,000-step limit. The new controller
allows 12 hours total: up to 10 hours training (including periodic checks),
then up to 2 hours final evaluation. It does not terminate the rented pod.

```bash
ssh runpod-transform
tmux attach -t overnight-sf19
```

Remote output directory: `/root/transform/outputs/sf19_ft/overnight_20260908`.

## Training

- Continue from run2's final live checkpoint; preserve it as `source_full.pt`.
- Keep optimizer/RNG state, reset the step counter for an explicit new schedule.
- Polar-NorMuon peak LR 0.0007; AdamW auxiliary peak LR 0.00001.
- 100-step warmup, cosine decay, batch 64, compiled model and optimizer.
- At most 120,000 steps, subject to the wall-time limit and validation guards.
- 75% SF19, 20% broad Lichess/MultiPV replay, 5% Syzygy.
- Frozen holdouts from all sources and safe horizontal flips are excluded from
  every training pool. `data_audit.json` records row counts and target audits.
- Eight low-priority CPU workers generate up to 500k additional SF19 positions.
  READY shards are deduplicated and holdout-filtered before attachment at checks.
- Actual pod limits: 15.3 CPU cores, roughly 71 GB RAM, 16 GB GPU memory.

## Validation and recovery

- SF19 and Syzygy validation every 250 steps; checkpoint every 250 steps.
- Separate Lichess retention check, plus all validation sources, every 4,000 steps.
- Stop after three consecutive validation points exceed the source policy CE by
  5% or WDL CE by 20%. Nonfinite loss or a 20-minute log stall also stops training.
- One validation-regression recovery attempt from a preserved safe checkpoint
  at half the learning rate; repeated regression ends training and starts evaluation.
- Latest/full, validation-screened, retention-checked and Elo-screened checkpoints
  are kept separately. Selection by validation alone does not promote a champion.
- A hard wall-time watchdog terminates subprocesses owned by the controller.
- Data generation stops before final evaluation.

## Playing-strength checks and Hugging Face

Up to three periodic gauntlets, at 16k-step intervals. Fixed 8,000-node Stockfish,
1750/1900 settings, paired colors and openings; no opening book or Syzygy assistance.
The final comparison evaluates the original HF initialization, run1 step 1500,
run2's final source, and the screened overnight candidate on 64 games each.
These are benchmark scores, not an official Elo rating.

A candidate is uploaded only when complete final results show at least a
15-percentage-point score gain against each preserved baseline, with a positive
95% lower bound from a paired opening-cluster bootstrap, and no validation-guard
failure on SF19, Lichess or Syzygy. Small-sample screening still has uncertainty.

A qualifying checkpoint and evaluation evidence are uploaded to the separate
private repo `avewright/chess-transformer-100m-overnight_20260908`. The existing
published model is never overwritten. `promotion.json` records a pass, failure,
or upload error. A failed gate leaves all checkpoints local.

## Logs and controls

- `controller.log`: orchestration, periodic evaluation, stopping/recovery decisions.
- `train.log`, `stdout.log`: model configuration, loss, validation, checkpoints.
- `resources.jsonl`: GPU use, GPU memory, temperature/power, load and free disk.
- `generation.stdout.log`, `ingestion.jsonl`: generation and shard verification.
- `periodic_checks.json`, `candidate_validation.json`, `elo_results.json`.
- `finished.json`, `failure.json`, or `deadline_reached.json`: terminal status.

To request a graceful stop of the overnight controller's training:

```bash
touch /root/transform/outputs/sf19_ft/overnight_20260908/HALT
```

The controller checks HALT between training checks. It can take longer to react
while evaluating a checkpoint. Do not use the trainer's STOP file to permanently
stop the controller; it also uses that file for scheduled validation pauses.

`latest.pt` and `best_safe.pt` are full resumable training checkpoints.
`eval_swa.pt`, when available, is evaluation-only. A controller failure retains
these artifacts; it does not silently start another 12-hour billing window.
