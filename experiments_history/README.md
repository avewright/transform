# Move-History Experiments

This folder is a separate experiment track for the hypothesis that:

- move sequence is a better training interface than static board state
- a model can infer the board implicitly from history
- temporal game context may teach deeper strategic structure than board-only input

## Scope

Use this folder for:

- move-history-only models
- board-plus-history hybrids
- latent board reconstruction from move sequence
- legal-mask-only autoregressive policies
- search-trace distillation into sequence models

Keep the main [`experiments`](C:\Users\AWright\OneDrive - Kahua, Inc\Projects\transform\experiments) folder focused on the current board-centric line. Use this folder for the parallel "history-first" research thread.

## Baseline

The first runnable baseline here is:

- [`run_exp001_move_history_baseline.py`](C:\Users\AWright\OneDrive - Kahua, Inc\Projects\transform\experiments_history\run_exp001_move_history_baseline.py)

That launches the existing move-history experiment from:

- [`exp160_move_history_transformer.py`](C:\Users\AWright\OneDrive - Kahua, Inc\Projects\transform\experiments\exp160_move_history_transformer.py)

## Suggested next experiments

1. Add a value head to the history-only transformer.
2. Compare `history-only` vs `board-only` vs `board+history` with matched parameter count.
3. Add board-reconstruction auxiliary loss.
4. Add search-teacher soft targets on top of legal masking.
5. Test piece-token or action-token embeddings instead of raw UCI-only tokens.

## Example usage

```powershell
python experiments_history/run_exp001_move_history_baseline.py `
  --train-pgn outputs/lichess_sf_games.pgn `
  --output-path outputs/exp_history_001/best.pt `
  --train-max-games 100 `
  --epochs 4
```
