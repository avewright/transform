# Export Summary 2026-03-31

## Selected Model

- Hugging Face model repo: `avewright/chess-transformer-200m-latest`
- Selected remote revision for republishing as `latest_model.pt`: `c01d9434fbcd1a5e501f2f82ff88e89134c30f29`
- Selected checkpoint status:
  - `epoch=7`
  - `train_steps=2742`
  - `dataset_records=107851`
  - `last_eval.loss=1.3680388554930687`
  - `last_eval.acc=0.40478515625`
  - `last_eval.top3=0.69970703125`
- Reason selected:
  - quick Elo test `outputs/elo_eval_paused_exp084_current.json` estimated `1646`
  - later quick Elo test `outputs/elo_eval_paused_exp084_current2.json` estimated `1600`
  - this earlier checkpoint was the stronger tested model

## Frozen Dataset

- Hugging Face dataset repo target: `avewright/exp085-parallel-multipv-harvest`
- Local dataset source: `outputs/exp085_parallel_multipv_harvest`
- Frozen after stopping live jobs and checkpointing SQLite WAL
- Canonical frozen counts from JSONL line counts:
  - `dataset_files=44`
  - `dataset_records=224191`
  - `dataset_bytes=551330268`
  - `last_shard=positions_000044.jsonl`
  - `last_shard_records=2724`
- Final status snapshot:
  - `written_positions=217718`
  - `exact_positions_seen=224181`
  - `records_per_min=553.31`
- Note:
  - JSONL line counts are treated as canonical for export because they represent the durable training corpus on disk

## Runtime State At Freeze

- Training process: stopped
- Harvester process: stopped
- `tmux` sessions: stopped
- Local resumable trainer checkpoint retained at `outputs/exp084_old_model_on_exp083/checkpoints/latest.pt`

## Artifacts Published

- Model repo:
  - `latest_model.pt`
  - `status.json`
  - `selected_elo_model_step2742.pt`
  - `selected_elo_model_step2742_status.json`
  - `selected_elo_model_step2742_summary.json`
  - Elo logs/json for both quick tests
- Dataset repo:
  - `README.md`
  - `manifest.json`
  - `status.json`
  - `exp085.log`
  - `stdout.log`
  - `seen_positions.sqlite`
  - `dataset/positions_*.jsonl`

## GitHub Push

- Git remote: `origin https://github.com/avewright/transform.git`
- Branch: `main`
- Repository push includes:
  - code changes
  - architecture doc updates
  - export summary and dataset card
