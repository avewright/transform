---
pretty_name: exp085 Parallel MultiPV Harvest
task_categories:
  - text-generation
language:
  - en
license: mit
size_categories:
  - 100K<n<1M
---

# exp085 Parallel MultiPV Harvest

This dataset is a frozen export of the `exp085_parallel_multipv_harvest.py` data harvester.

## Snapshot

- Export date: `2026-03-31`
- Shards: `44`
- Records: `224191`
- Size: `551330268` bytes of JSONL shard data
- Final shard: `positions_000044.jsonl`
- Final shard records: `2724`

## Included Files

- `dataset/positions_*.jsonl`
- `manifest.json`
- `status.json`
- `exp085.log`
- `stdout.log`
- `seen_positions.sqlite`

## Notes

- The JSONL line count is treated as the canonical record count for this export.
- `status.json` counters and SQLite dedupe counts may differ slightly from the shard line count because they were written on different update boundaries during harvesting.
- This export was frozen after stopping the live harvester and checkpointing SQLite WAL state.

## Provenance

- Harvester script: `experiments/exp085_parallel_multipv_harvest.py`
- Stockfish-backed MultiPV labeling
- Deduplication via SQLite position index
