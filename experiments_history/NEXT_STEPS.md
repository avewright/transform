# Next Steps

## Immediate goal

Establish a clean history-only baseline under a separate experiment track, then iterate without mixing it into the board-centric experiment stack.

## Priority ablations

1. `history-only` baseline
2. `history-only + value head`
3. `history-only + board reconstruction auxiliary loss`
4. `board + history`
5. `history-only + soft search targets`

## Metrics to watch

- top-1 next-move accuracy
- top-3 next-move accuracy
- legal-masked loss
- value calibration if added
- same-board / different-history consistency
- tactical slice accuracy
- opening and middlegame split accuracy

## Key question

Does move history teach useful latent structure beyond what a board-only model learns, or does it mostly spend capacity reconstructing state?
