# Stockfish 19 WDL rating dataset

Official **Stockfish 19** `UCI_ShowWDL` labels. This is the engine's fishtest-LTC
self-play win/draw/loss model (evaluation + remaining material). It is not a
FIDE or Lichess rating, and it is not `data_loader.compute_wdl`'s sigmoid.

The project's sigmoid at +27cp is ~56% White win. SF19 official WDL at the
same start-like score is ~5% win / 94% draw. Train value on the official
triple if you want the rating the engine actually reports.

Do not mix this out-dir with MultiPV-8 soft shards.

## Contract

- Teacher: pinned Stockfish 19, full strength, `Threads=1`, `UCI_ShowWDL=true`
- Search: **single PV**, default 25k nodes
- `wdl`: White-absolute `[P(White wins), P(draw), P(White loses)]`
- `wdl_raw`: same triple as UCI per-mille integers (sum 1000)
- `wdl_source`: `1` official UCI, `2` terminal (mate / draw). Never sigmoid
- Missing WDL drops the row
- Unique 4-field keys (board, side, castling, ep)
- `n_pieces` stored (the WDL model is material-dependent)
- Best move kept as `n_soft=1` so shards still stack with the soft cache schema
- `split=1` is a 5% holdout

## Commands

From the repository root:

```bash
export MOVE_VOCAB_VERSION=compact
export STOCKFISH_PATH="${STOCKFISH_PATH:-$HOME/.local/bin/stockfish-19}"

# Official WDL vs project sigmoid on a few boards
.venv/bin/python -u scripts/sf19_wdl_dataset.py bench --out-dir outputs/sf19_wdl/bench

# Pipeline check
.venv/bin/python -u scripts/sf19_wdl_dataset.py smoke --out-dir outputs/sf19_wdl/smoke

# ECO harvest toward 1M unique positions
.venv/bin/python -u scripts/sf19_wdl_dataset.py generate --go --mode eco \
  --out-dir outputs/sf19_wdl/eco --target 1000000 --workers 16

# Relabel existing unique boards (no self-play)
.venv/bin/python -u scripts/sf19_wdl_dataset.py generate --go --mode relabel \
  --out-dir outputs/sf19_wdl/relabel --seed-caches outputs/organized_chess_v1/sf19_train.pt
```

```bash
# Upload the finished pack
.venv/bin/python -u scripts/sf19_wdl_dataset.py push \
  --out-dir outputs/sf19_wdl/eco --repo avewright/local-wdl
```

`scripts/run_sf19_wdl.sh` is the long ECO job. Resume is the same command and
the same out-dir. A changed node budget or binary needs a new directory.

Promote nothing from this dataset automatically. It is value supervision.
