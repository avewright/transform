# Hugging Face Datasets

This file summarizes the Hugging Face dataset repos referenced by this codebase.

It is based on the dataset IDs, comments, and pipeline scripts currently present in the repo. Where a count is only mentioned in repo notes or code comments, it is marked as approximate.

## At a Glance

| Repo | Status | Approx scale | Main role |
|---|---|---:|---|
| `avewright/chess-positions` | core historical dataset | ~47.5K train + 2.5K test | early and mid-stage supervised training baseline |
| `avewright/chess-positions-lichess-sf` | core large-scale dataset | ~832M positions, 3275 source parquets | streaming large-scale all-phase training |
| `avewright/chess-positions-sf-200k` | auxiliary generated dataset | ~190K train + 10K test | synthetic SF-labeled data expansion |
| `avewright/exp085-parallel-multipv-harvest` | frozen experiment export | 224,191 records | MultiPV harvest snapshot used for later training/export |
| `avewright/chess-dataset-production-1968` | legacy dataset | ~475K positions | old game-play dataset used by early experiments |

## 1. `avewright/chess-positions`

Small, curated supervised dataset used as the main baseline across many experiments before the move to large-scale streaming.

- Purpose: supervised move prediction on realistic positions with extra metadata
- Typical usage: `hf_data.py`, `exp050`-`exp056`, `exp065`, `exp078`-`exp081`
- Approx size from repo notes: `47.5K` train and `2.5K` test
- Why it matters: this is the dataset behind many of the repo's baseline accuracy numbers in the README

### Schema

Documented in `build_dataset.py`:

- `fen`
- `best_move`
- `eval_type`
- `eval_value`
- `wdl_win`
- `wdl_draw`
- `wdl_loss`
- `phase`
- `num_legal`
- `source`
- `game_id`
- `top_moves`
- `ply`

### Repo references

- `build_dataset.py`
- `hf_data.py`
- `experiments/exp055_joint_policy_value.py`
- `experiments/exp065_quick.py`

## 2. `avewright/chess-positions-lichess-sf`

This is the main large-scale dataset for the current 200M-model direction.

- Purpose: train on massively more positions, including middlegames and endgames
- Source: `Lichess/chess-position-evaluations`
- Approx scale from repo notes: `~832M` positions across `3275` source parquet shards
- Why it matters: this is the dataset used by the streaming loaders and the current `exp101` / `exp102` direction

### Pipeline

Primary creation path in this repo:

- `process_lichess_parquets.py`
- `prepare_hf_dataset.py`

The repo notes describe a shard-first upload path to avoid local Arrow materialization during upload.

### Schema

Documented in `prepare_hf_dataset.py`:

- `fen`
- `best_move`
- `eval_type`
- `eval_value`
- `wdl_win`
- `wdl_draw`
- `wdl_loss`
- `phase`
- `num_legal`
- `source`
- `game_id`
- `top_moves`
- `ply`
- `depth`

### Layout notes

The repo's data loader expects two styles of parquet layout:

- `train-src...` / `test-src...` source-sharded files
- `train-xxxxx-of-xxxxx.parquet` main split files

Relevant code:

- `data_loader.py`
- `prepare_hf_dataset.py`
- `process_lichess_parquets.py`

### Repo references

- `data_loader.py`
- `cache_lichess_data.py`
- `auto_push.py`
- `experiments/exp071_extended_training.py`
- `experiments/exp072_data_scale.py`
- `experiments/exp074_resume_200m.py`
- `experiments/exp075_ddp_4gpu.py`
- `experiments/exp076_continue_v2.py`
- `experiments/exp083_pretrain_4xa40.py`
- `experiments/exp101_hf_scale_training.py`
- `experiments/exp102_auxiliary_losses.py`

## 3. `avewright/chess-positions-sf-200k`

Synthetic Stockfish-labeled dataset uploaded from locally generated JSONL.

- Purpose: supplement the smaller curated dataset with more supervised positions
- Upload script: `upload_dataset_hf.py`
- Approx size from repo comments: `190K` train and `10K` test
- Split policy in upload script: `95%` train / `5%` test

### Notes

`upload_dataset_hf.py` says this dataset comes from:

- `outputs/exp059_data_scaling/generated_200k.jsonl`

The upload script flattens:

- `wdl` into `wdl_win`, `wdl_draw`, `wdl_loss`
- `top_moves` into `top_moves_json`

This means the published schema here is not identical to `avewright/chess-positions`.

### Repo references

- `upload_dataset_hf.py`
- `experiments/exp065_quick.py`

## 4. `avewright/exp085-parallel-multipv-harvest`

Frozen dataset export from the `exp085` harvester.

- Purpose: preserve a harvested MultiPV training corpus as a resumable/exportable dataset
- Local source directory: `outputs/exp085_parallel_multipv_harvest`
- Export summary date in repo: `2026-03-31`
- Canonical record count in repo docs: `224,191`

### Snapshot details from repo

- Shards: `44`
- Records: `224191`
- Final shard: `positions_000044.jsonl`
- Artifact type: JSONL shards plus manifest/status/log files

### Included files

From `HF_DATASET_EXP085_README.md` and `EXPORT_SUMMARY_2026-03-31.md`:

- `dataset/positions_*.jsonl`
- `manifest.json`
- `status.json`
- `exp085.log`
- `stdout.log`
- `seen_positions.sqlite`

### Repo references

- `HF_DATASET_EXP085_README.md`
- `EXPORT_SUMMARY_2026-03-31.md`
- `experiments/exp084_old_model_on_exp083.py`

## 5. `avewright/chess-dataset-production-1968`

Legacy dataset used by many earlier experiments.

- Purpose: early large supervised dataset before the newer `chess-positions` family
- Approx scale from repo comments: `~475K` positions
- Label style: game-play moves rather than newer SF-first supervision
- Important limitation from repo notes: older move mapping and less complete game-state metadata

### Notes

The repo describes this dataset as:

- using an older move mapping
- lacking castling / en passant metadata
- carrying game-level winner/value-style labels rather than the later position-centric schema

It is still important because many older experiments depend on it.

### Repo references

- `experiments/exp013_hf_dataset_scale.py`
- `experiments/exp014_full_hf_1epoch.py`
- many `exp0xx` scripts from the earlier transformer/Qwen phase

## Scripted Target Repos Mentioned In This Repo

These repo IDs appear in tooling, but this repo snapshot does not document them as clearly as the five datasets above.

### `avewright/chess-positions-lichess-sf-v2-full-dedup-rowkey`

Canonical deduped export target defined in:

- `build_hf_dataset_v2.py`
- `build_hf_dataset_v3.py`

Intended role:

- globally deduped rebuild of `avewright/chess-positions-lichess-sf`
- deterministic split reassignment
- cleaner documented parquet export

### `avewright/chess-positions-sf-labeled`

Default repo target in:

- `generate_massive.py`

Intended role:

- large generated Stockfish-labeled dataset from the parallel generation pipeline

## Practical Summary

If you only remember the important split:

- `avewright/chess-positions` = small, curated baseline dataset
- `avewright/chess-positions-lichess-sf` = main large-scale streaming dataset
- `avewright/chess-positions-sf-200k` = synthetic supplement
- `avewright/exp085-parallel-multipv-harvest` = frozen experimental export
- `avewright/chess-dataset-production-1968` = older legacy dataset used by early experiments
