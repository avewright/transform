# Experiment 283: recurrent depth from the pretrained 99M

Hypothesis: variable-depth continuation training makes additional recurrent
passes useful for search-free chess. Preserve encoding, attention, parameter
count, and source checkpoint. Test GAB/Shaw separately after this experiment.

The model runs prefix → shared bank × N → suffix/head. A runtime override
changes N without changing checkpoint keys or the default three-pass behavior.
No learned stopping mechanism yet: first establish a useful depth/strength curve.

## Arms and controls

- Untouched 99M: inference at 1, 2, 3, 4, 6, 8 passes.
- Fixed continuation: three passes per training batch.
- Variable continuation: shuffled cycles of 2, 3, 4 passes.

Both continuation arms start from the same checkpoint and use identical sampled
row sequences, seeds, batch sizes, optimizer updates, and total block-position
evaluations. Wall time can differ. Dropout random draws differ because depths
differ. This is compute matching by transformer block count, not measured FLOPs.
Variable 6/8-pass inference is extrapolation beyond its training depths.

Both arms use fresh AdamW at 1e-5, weight decay .01, clipping 1, and the existing
recurrent gradient averaging rule with the actual sampled depth. This is a
controlled pilot, not a reproduction of the incumbent's Polar-NorMuon run.
Loss: .45 hard policy CE + .55 engine soft CE + .15 valid-row WDL CE.
No augmentation, KD teacher, intermediate losses, extra heads, or automatic
promotion. Saved models are inference/weights-only checkpoints, not full
optimizer resumes. A new nonempty output directory is refused.

## Run

From the repository root, using the project Python environment:

```bash
export MOVE_VOCAB_VERSION=compact
.venv/bin/python experiments/exp283_recurrent_depth.py sweep \
  --ckpt outputs/hf_100m_squares64/latest.pt \
  --eval-cache outputs/organized_chess_v1/sf19_eval.pt \
  --out outputs/exp283/baseline --eval-rows 512 --batch-size 8

# Existing local mixed cache: 450k SF19 + 350k Lichess + 150k puzzle rows.
# Both arms see this same frozen cache; value eligibility follows its masks.
.venv/bin/python experiments/exp283_recurrent_depth.py train \
  --ckpt outputs/hf_100m_squares64/latest.pt \
  --train-cache outputs/organized_chess_v1/soft_cache.pt \
  --eval-cache outputs/organized_chess_v1/sf19_eval.pt \
  --arm fixed --steps 1200 --out outputs/exp283/fixed

.venv/bin/python experiments/exp283_recurrent_depth.py train \
  --ckpt outputs/hf_100m_squares64/latest.pt \
  --train-cache outputs/organized_chess_v1/soft_cache.pt \
  --eval-cache outputs/organized_chess_v1/sf19_eval.pt \
  --arm variable --steps 1200 --out outputs/exp283/variable
```

The script removes training overlap against the full supplied evaluation cache,
before selecting evaluation rows. This does not prove the pretrained model never
saw those boards; historical training provenance requires a separate audit.
Terminal positions are rejected from the diagnostic sweep. Metrics are legal
top-1 agreement, full-vocabulary hard CE, paired rescues/regressions relative to
three passes, and batched forward throughput. These are not Elo or engine regret.

Checkpoint/data SHA256s, selected evaluation indices, arguments and torch version
are recorded. Use an explicit --ckpt for a newer incumbent; the local default
is a September 5 snapshot, not a claim that it is the latest remote champion.

## Elo confirmation

Add --export-depths to export standalone depth-specific inference checkpoints
(approximately 396 MB each for FP32 99M). They load through the existing harness:

```bash
.venv/bin/python -m harness.elo \
  --ckpt outputs/exp283/baseline/loops_3.pt --mode policy \
  --device mps --elos 1900 2050 2200 --no-stop-after-bracket \
  --ply-cap 400 --out-prefix exp283_baseline_3
```

Pin STOCKFISH_PATH and use identical protocols across finalists. Establish a new
long-cap baseline rather than comparing directly to previous 160-ply results.
Report PLY_CAP outcomes separately; the harness scores them as draws. Screen
with the existing openings, then confirm using fresh paired openings and at
least 200 games, extending when uncertainty remains. Freeze depth/checkpoint
selection before that confirmation. Evaluate a second seed for a promising arm.

Advance only if variable training improves actual games at a useful inference
budget versus fixed continuation. A drop in CE alone is not a win. Adaptive
halting, GAB, and encoding changes are subsequent, separately controlled tests.
