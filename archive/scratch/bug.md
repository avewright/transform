# Bug Audit

Audit date: 2026-04-03

Scope:
- active `exp110` / `exp111` training and pipeline code
- value-based search and value-head interpretation paths
- nearby evaluation and reporting code that can mislead experiment decisions

Line numbers below are current as of this audit and may drift as files change.

## High Severity

### 1. WDL convention is inconsistent across the repo

Evidence:
- `data_loader.py:432-448` defines canonical soft WDL as `[win, draw, loss]`.
- `experiments/exp104_policy_guided_search.py:147-149` still scores value as `wdl[2] - wdl[0]`.
- `experiments/exp097_alphabeta_search.py:205-206` still scores value as `wdl[2] - wdl[0]`.
- `experiments/exp103_gumbel_search.py:122-128` still assumes `[loss, draw, win]`.
- `play.py:150-152` still reports `loss=wdl[0], win=wdl[2]`.
- `play_gui.py:468-472` still comments and flips WDL using the old convention.

Impact:
- value-based search can rank moves backwards
- displayed model evals can be wrong, which makes manual debugging unreliable
- different experiments are no longer comparable because they do not agree on what `value_logits[0]` means

Possible solution:
- define one repo-wide convention and enforce it everywhere
- recommended: match `compute_wdl()` and use `[win, draw, loss]`
- add a small shared helper module with functions like:
  - `wdl_probs_to_value_stm(wdl_probs)`
  - `wdl_probs_to_expected_score_stm(wdl_probs)`
  - `wdl_probs_to_display_dict(wdl_probs, perspective=...)`
- replace all direct `wdl[...]` indexing in search / UI / eval code with helper calls

### 2. `exp110_search.py` still reranks candidates with the wrong child value

Evidence:
- `experiments/exp110_search.py:141-150` treats child `value_probs[:, 0]` as "our win probability" after pushing our move.
- after we push a move, the child node is from the opponent's turn
- under canonical `[win, draw, loss]`, index 0 is the opponent's win probability, not ours

Impact:
- the reranker can prefer moves that improve the opponent's position
- any Elo conclusions from `exp110_search.py` are suspect

Possible solution:
- compute child value from the parent mover's perspective explicitly
- if the child WDL is from opponent-to-move perspective and the convention is `[win, draw, loss]`:
  - our win probability is `child_wdl[2]`
  - our draw probability is `child_wdl[1]`
- centralize this logic in a helper instead of repeating comments and index math

### 3. `exp110` / `exp111` training can ingest inverted hard value labels

Evidence:
- the current training scripts use `value_target` directly:
  - `experiments/exp110_diverse_training.py:279` and `experiments/exp110_diverse_training.py:429`
  - `experiments/exp111_conservative_continuation.py:185` and `experiments/exp111_conservative_continuation.py:292`
- several active dataset generators define hard targets as `2=win, 1=draw, 0=loss`:
  - `experiments/exp085_parallel_multipv_harvest.py:174-179`
  - `experiments/exp110_diverse_multipv_harvest.py:118-123`
  - `experiments/exp110_puzzle_harvest.py:79-84`
  - `experiments/exp110_tablebase.py:348-354`
  - `experiments/exp110_syzygy.py:265-271`
  - `experiments/exp110_weakness_harvest.py:224-230`
- the canonical soft WDL path in `data_loader.py:440-448` uses `0=win, 1=draw, 2=loss`
- recent search fixes in `exp094` / `exp112` also assume index 0 is win

Impact:
- continuation training can push the value head in the opposite direction from the baseline checkpoint
- this can silently degrade search and distort value-loss metrics even when policy accuracy looks stable

Possible solution:
- standardize hard labels to the same convention as `compute_wdl()`
- recommended migration: `0=win, 1=draw, 2=loss`
- add a dataset manifest field such as `value_target_convention`
- assert the convention on load before training starts
- add one smoke test that checks a clearly winning FEN produces the expected class index after dataset generation

### 4. Batch wraparound cursor is broken in `exp110` and `exp111`

Evidence:
- `experiments/exp110_diverse_training.py:422-425`
- `experiments/exp111_conservative_continuation.py:286-288`
- both files do:
  - extend a short batch with records from the start of the dataset
  - then set `cursor = BATCH_SIZE - len(batch)` after `batch` has already been extended to full length
- that means `cursor` becomes `0` instead of the intended remainder count
- `experiments/exp084_old_model_on_exp083.py:520-523` shows the correct pattern using a separate `needed` variable

Impact:
- the first records are repeated too often when an epoch wraps
- sampling is biased
- dataset coverage per epoch is not what the script claims

Possible solution:
- compute `needed = BATCH_SIZE - len(batch)` before extending
- set `cursor = needed` after extending
- copy the `exp084` implementation pattern into `exp110` and `exp111`

## Medium Severity

### 5. `exp110_pipeline.sh` can fail log capture and reports completion too early

Evidence:
- `exp110_pipeline.sh:63-70` tees into `outputs/exp110_weakness_harvest/weakness.log`
- `exp110_pipeline.sh:77` tees into `outputs/exp110c_weakness_training/exp110c.log`
- those directories are currently missing in the workspace
- `exp110_pipeline.sh:51-54` prints `FULL PIPELINE COMPLETE!` before phases 5-7 begin

Impact:
- `tee` can fail to open the log path
- later phases can run without logs or with misleading shell status
- operators may think the pipeline is done when important phases have not started

Possible solution:
- `mkdir -p` every output directory before the first `tee`
- switch to `set -euo pipefail`
- use the `log()` helper consistently
- only print completion after the final phase and final eval finish

### 6. `exp110_pipeline.sh` treats "not running" as "finished successfully"

Evidence:
- `exp110_pipeline.sh:13-18` only waits for `pgrep -f "exp110_diverse_training"`
- if training never started or crashed early, the pipeline moves directly to evaluation as long as an older `best_model.pt` exists

Impact:
- stale checkpoints can be evaluated and promoted as if they came from the current run
- failed runs can look successful from the pipeline logs

Possible solution:
- have training write an explicit completion artifact or `status.json` field on success
- make the pipeline wait for both:
  - process exit
  - successful completion marker with a fresh timestamp

### 7. `exp110b` / `exp110c` evaluation silently drops failing batches

Evidence:
- `experiments/exp110b_syzygy_training.py:330-343`
- `experiments/exp110c_weakness_training.py:335-348`
- both use bare `except Exception: pass` inside evaluation loops

Impact:
- evaluation can skip bad batches without telling the operator
- metrics can look healthy while being computed on only a subset of data

Possible solution:
- log the first exception with a traceback
- count dropped batches and report that count in the metrics
- fail evaluation if the dropped-batch rate crosses a small threshold

## Lower Severity

### 8. WDL reporting is still mislabeled in some user-facing paths

Evidence:
- `play.py:150-152` reports `loss=wdl_probs[0]` and `win=wdl_probs[2]`
- `experiments/exp094_search_eval.py:178`, `experiments/exp094_search_eval.py:262`, and `experiments/exp094_search_eval.py:279` still build debug info with the old label names

Impact:
- human inspection of the model's evaluation can be backwards even when the move choice is correct
- this makes bug triage slower because logs look plausible but mean the opposite

Possible solution:
- update every WDL info dict to match the chosen convention
- route all WDL-to-dict formatting through one helper

## Recommended Fix Order

1. Standardize WDL conventions repo-wide and patch all value-based search paths.
2. Fix hard `value_target` generation and add a load-time convention assertion.
3. Fix the `exp110` / `exp111` cursor wraparound bug.
4. Harden the pipeline with explicit output dirs and success markers.
5. Remove silent eval exception swallowing and make dropped-batch counts visible.
