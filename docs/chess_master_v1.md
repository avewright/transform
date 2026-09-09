# chess_master_v1

Versioned master chess dataset. Positions and labels are separate tables so a later mix can ask for “unseen middlegames with deep SF19 soft targets” or “verified defensive mistakes in endgames” without converting the sources again.

Do not upload. Do not start training from this pipeline.

## Layout

```
outputs/chess_master_v1/
  inventory.json              pinned sources
  input_manifest.json
  positions/mix-*.parquet
  annotations/mix-*.parquet
  membership/mix-*.parquet
  quarantine/*.parquet
  samples/cross_source.json   review sample
  quality_report.json
  recipes/                    (repo: chess_master/recipes/)
```

`position_id` joins the three tables. One position can have many annotations. Membership records pool inclusion and splits without collapsing labels.

## Tables

### Positions — one row per canonical board

Identity is the 4-field position: board, side to move, castling, en passant. Halfmove, fullmove, and repetition history are stored when present and otherwise `null` with `rule_state_available=false`. Missing clocks are **unknown**, not zero.

| Field | Notes |
|---|---|
| `position_id` | SHA-256 of the 4-field key (hex16) |
| `legacy_hash` | Existing uint64 used by holdout manifests |
| `equivalence_key` | `hflip64:min(hash,flip)` only when castling is empty; otherwise `exact:{position_id}` |
| `fen_4` / `fen_6` | `fen_6` is null when clocks are unknown |
| `phase` / `phase_method` | `non_king_20_10` (opening ≥20, mid ≥10, else end). Source phase is kept on the annotation |
| material / check / legality | Recomputed from the board. `is_puzzle` is **not** a position category |

Color-swap and castling-sensitive symmetries are not equivalences.

### Annotations — many teacher labels per position

| Dimension | Fields |
|---|---|
| Type | `engine_policy` / `puzzle_solution` / `tablebase` / `model_prediction` |
| Engine | `engine`, `engine_version`, `network` (null/`unknown` if absent) |
| Search | requested vs achieved nodes, depth, `depth_is_sentinel`, MultiPV, bounds, flags |
| Policy | UCI list, compact1968 indices, **probabilities**, tau, transform name |
| Value | CP, mate, WDL, perspective, `value_valid`, `value_eligible` |
| Puzzle | id, game id, rating, themes, full `Moves` line, setup FEN, solver FEN |
| Tablebase | 5-valued WDL, DTZ, piece count, rule/probe (unknown until verified) |
| Model error | checkpoint, predicted move, in-PV flag, tag, drop_cp, `regret_status` |
| Audit | original CP/mate/perspective plus `source_fields_json` |

`search_record_id` groups MultiPV entries from one search. Do not mix candidates across records.

### Membership — exposure and splits

`included_in_pool` is not `actually_sampled`. Both `actually_sampled` and `checkpoint_lineage` are `unknown` unless a training log says otherwise. Do not label rows unseen.

Game ids are namespaced before grouping: `sf19-game:{id}`, `lichess-game:{GameId}`. Lichess MultiPV and Syzygy have no reliable game id and fall back to `pos:{legacy_hash}`.

## Label contracts

1. **Perspective.** SF19 stored CP/mate/WDL are White-absolute. Lichess perspective is unverified. Syzygy WDL is side-to-move `{-2..2}`. Original perspective is always kept.
2. **Moves.** Canonical UCI is stored. Compact-vocab indices go through adapter `compact1968`.
3. **Probabilities are not logits.** Record the transform (`softmax_stm_rank_score_tau`, `onehot_solution`, `softmax_tb_wdl_mapped_cp`, or `source_probabilities_unknown_transform`).
4. **MultiPV.** Only candidates from one `search_record_id`. Bounds and incomplete iterations stay on the row.
5. **Puzzles.** Source FEN is before the opponent setup move. Apply that move, then take the solver target. Keep the full line. Policy only — no invented CP/WDL.
6. **Syzygy.** DTZ is not mate. `mate_is_dtz_proxy=1` when the pack stored a DTZ-derived mate. Value supervision stays off until conversion is verified. `label_depth=999` is a sentinel.
7. **Lichess.** `mate==0` and `|cp|>=90000` is a mate sentinel. Value stays masked. Depth `>=128` is a sentinel.
8. **Regret.** A model move missing from MultiPV has `regret_status=unknown_off_pv`, not an assumed penalty.
9. **Quality.** Recipes must set `quality_relaxation: never`. Shortfalls are reported, not filled by dropping gates.

## Inventory (pinned)

| Source | Revision | License | Local reuse |
|---|---|---|---|
| `avewright/chess-soft-sf19` | `68cef6c9ba62c62f904f25305f1a7489dab825e0` | MIT | organized_chess_v1 shards |
| `avewright/chess-soft-multipv-lichess` | `291acea2a2ce6fc2aba4b289110d1920cf969afe` | MIT | `outputs/hf_soft/multipv_lichess_soft.pt` |
| `avewright/chess-soft-syzygy` | `3889985c482837aa3082e182d8f123e005e2c0d4` | MIT | `outputs/hf_soft/syzygy_soft.pt` |
| `Lichess/chess-puzzles` | `479ea9bc9f681385f5adb23fa27a96c2dc8ae599` | CC0-1.0 | mix used `train-00002-of-00003` |
| `avewright/chess-soft-100m-swa-mistakes` | `69b75e12ab2867c8b9a63d385065dfe646cf86e1` | MIT | no local harvest dir; HF pack only |

`outputs/organized_chess_v1` is **frozen**. Treat it as a membership pool, not a source to rewrite.

Unknown on purpose: SF19 NNUE hash, Lichess engine/nodes/tau, Syzygy 50-move rule verification, SWA checkpoint file hash unless the overnight SWA file is present, halfmove/repetition on almost every row.

## Recipes

`chess_master/recipes/pilot_45_35_15_5.json` replays the 45 / 35 / 15 / 5 pilot from `organized_chess_v1` membership. Overlap policy is `exclusive_source_bucket`: a puzzle-endgame with SF19 is one weighted row.

```
MOVE_VOCAB_VERSION=compact python scripts/build_chess_master.py build
MOVE_VOCAB_VERSION=compact python scripts/build_chess_master.py build --skip-mix
MOVE_VOCAB_VERSION=compact python scripts/build_chess_master.py export --recipe pilot_45_35_15_5
```

Trainer export writes `soft_cache.pt` + `deep_cache.pt` with the existing source ids (Lichess=1, Syzygy=2, SF19=4, Puzzle=5).
