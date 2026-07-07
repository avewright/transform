# `critical_moves.py` Current Behavior

This file is not a generic "critical move detector." In its current form, it is a **data-generation script** that:

1. Loads a trained chess model.
2. Plays games where the model is always White and Stockfish is always Black.
3. At each White turn, searches several candidate model moves recursively.
4. Chooses the best searched move.
5. Writes training records and search traces to JSONL files.

The main entry point is [`main()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L538).

## High-Level Flow

### 1. Setup

At startup, the script defines:

- repo-relative paths for Stockfish, the default checkpoint, and the output directory in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L35)
- a small list of preset openings in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L41)
- terminal / draw score constants and default leaf-eval depth in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L50)

`main()` then:

- parses CLI args
- creates a timestamped run directory under `outputs/deep_search/run_YYYYMMDD_HHMMSS`
- opens Stockfish
- loads the model via [`load_model()` in `play.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/play.py#L203)
- runs one or more games with `play_and_explore_game()`

## How Move Selection Works

### White turns: model candidates

On White's move, the script calls `get_model_candidates()` in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L86).

That function:

- encodes the board with [`encode_board()` in `play.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/play.py#L110)
- runs the model forward
- takes `policy_logits`
- masks out illegal moves using [`legal_move_mask()` in `move_vocab.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/move_vocab.py#L58)
- softmaxes the legal logits
- returns the top `k` legal candidate moves with:
  - `uci`
  - `policy_prob`
  - `rank`

The move vocabulary is a fixed UCI-index mapping in [`move_vocab.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/move_vocab.py#L44), and indices are converted back to `chess.Move` with [`index_to_move()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/move_vocab.py#L53).

### Black turns: Stockfish candidates

On Black's move inside search, the script calls `get_sf_candidates()` in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L146).

That function:

- asks Stockfish for `multipv = sf_branch_k`
- extracts the first move from each PV
- stores the evaluation as `engine_cp_white`
- sorts moves ascending, because the score is from White's perspective and Black wants lower scores

Outside the recursive search, actual game play for Black is simpler: `play_and_explore_game()` just asks Stockfish for one move with `engine.play(...)`.

## The Recursive Search

The core search is `search()` in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L214).

It behaves like a small alpha-beta minimax:

- White nodes maximize score.
- Black nodes minimize score.
- a cache memoizes `(fen, depth, model_branch_k, sf_branch_k)`
- alpha-beta pruning cuts branches early

### Important detail: depth only drops on White moves

When White explores a move, the recursive call uses `depth - 1`.

When Black explores a move, the recursive call uses `depth` unchanged.

So in practice, `search_depth` means roughly:

- "how many future White decisions should be explored"
- not "how many total plies"

That asymmetry is important to understanding the current behavior.

### Leaf handling

The search stops when:

- the board is terminal
- the ply cap is reached
- `depth <= 0`

If `depth <= 0`, it does **not** immediately run a static evaluation. Instead it calls `greedy_rollout()` in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L184).

`greedy_rollout()`:

- keeps playing greedily for White using the model's top-1 move
- keeps playing greedy Stockfish replies for Black
- stops on terminal state or ply cap
- if terminal, returns `score_terminal()`
- otherwise runs a deeper Stockfish eval with `engine_eval_white()`

So the leaf score is:

- true game outcome if the rollout finishes the game
- otherwise Stockfish centipawns from White's perspective

## What Happens At Each White Root Position

For the actual move played in the game, White does **root exploration** with `explore_root_position()` in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L329).

For each top model candidate at the root, it:

1. pushes the candidate move
2. runs recursive `search()`
3. records the returned score, PV, and leaf reason

Then it sorts candidates by score descending and builds one training record with fields such as:

- `fen`
- `best_move`
- `best_cp`
- `value_target`
- `move_values`
- `searched_only`
- `searched_move_count`

This is the main reason the script exists: it creates a labeled root position where only the searched candidate set is scored.

## How A Full Game Runs

`play_and_explore_game()` in [`critical_moves.py`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L421) runs the game loop:

- apply the selected opening moves first
- while the game is not over:
  - if White to move:
    - explore the root position
    - append one training record
    - append per-candidate trace rows
    - play the highest-scoring White move
  - if Black to move:
    - let Stockfish play one move directly

At the end it returns a game summary with result, plies, move list, and search stats.

## Output Files

Each run creates a timestamped folder under `outputs/deep_search`.

The important files are:

- `search.log`: human-readable progress and summary
- `training_records.jsonl`: one JSON object per White root position
- `search_traces.jsonl`: one JSON object per candidate move explored at each root
- `summary.json`: run config plus per-game summaries

These paths are created in [`main()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/transform/critical_moves.py#L538).

## What This File Is Actually Optimizing For

The current script is best understood as:

- a **search-augmented self-play data generator**
- not a full engine
- not MCTS
- not a balanced two-sided learned search

White is the learned policy plus recursive search.
Black is Stockfish.
The generated labels are meant to be better than plain top-1 policy labels because they reflect a searched subset of candidate moves.

## Practical CLI Parameters

The main knobs are:

- `--search-depth`: number of future White decisions to search
- `--branch-k`: number of model candidate moves to consider at White nodes
- `--sf-branch-k`: number of Stockfish MultiPV replies to branch over at Black nodes
- `--sf-depth`: Stockfish search depth for Black move generation
- `--leaf-eval-depth`: Stockfish depth for non-terminal leaf evaluation
- `--opening`: index into the hardcoded `OPENINGS` list
- `--num-games`: number of games to generate in one run

## One Current Limitation To Be Aware Of

The file name suggests "critical moves," but the implementation currently does **root search labeling** rather than explicit criticality detection.

In other words, it answers:

- "Which of the searched White moves looks best after recursive search?"

More than:

- "Which move is uniquely critical or high leverage?"
