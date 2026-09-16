# Recurrent Geometry 99M — exp284

New warm-start architecture: existing 64-square fused encoder and FiLM context
→ four prefix blocks → seven shared geometry blocks repeated N times → four
suffix blocks → existing spatial policy and WDL heads. Width 736, eight heads,
and FFN ratio four are inherited from the pretrained 99M. No weight expansion.

Default: GAB inside the recurrent bank only. The board-conditioned bias is
recomputed from each block's normalized hidden input, every pass. The same
block's attention and GAB weights are shared across passes. Configurable modes:

- `none`: original attention, for a control.
- `gab`: dynamic mixtures of learned 64×64 templates (d1=16, d2=64, d3=32).
- `shaw`: 225 displacement-tied Q/K/V vectors per head. Scores use
  `(q_i + aQ_ij) dot (k_j + aK_ij) / sqrt(head_dim)` and outputs include aV.
- `both`: add GAB to Shaw attention scores; the two mechanisms train together.

Shaw is a 2D Shaw-style Q/K/V variant combined with incumbent QK normalization,
not a claim of exact reproduction of a published Chessformer. GAB uses SDPA;
Shaw explicitly materializes attention, so measure its actual speed/memory.
`geometry_scope: all` also upgrades prefix/suffix, as a separate ablation.

Existing tensor names and shapes are preserved. Only new geometry contributions
start at zero. GAB's template projection learns first; upstream GAB layers receive
gradients once templates depart from zero. No extra zero gate blocks this path.
Old checkpoints retain `geometry_attention=none`. New checkpoints carry complete
geometry settings and load through `chess_inference` and the existing Elo harness.

## Training configuration

`configs/exp284_recurrent_geometry_99m.json` is executable configuration:

- Pretrained local September 5 99M; change `checkpoint` for another incumbent.
- 3,000 updates, batch eight, shuffled complete cycles of 2/3/4 passes.
- New AdamW: backbone LR 1e-5, geometry LR 1e-4, weight decay .01.
- 100-step warmup, cosine decay to 10% of each initial LR.
- .45 hard CE + .55 soft-policy CE + .15 valid-row WDL CE.
- Existing gradient averaging uses actual pass count, followed by clipping at 1.
- Local 950k mixed rows (450k SF19, 350k Lichess, 150k puzzles).
- Exclude full eval-cache board overlap before selecting diagnostic rows.
- Evaluate 1/2/3/4/6/8 passes; six/eight are extrapolation, not training depths.

This is a conservative FP32 continuation pilot, not a sufficient budget claim.
No learned halting, intermediate losses, history changes, auto-promotion, uploads,
or optimizer resume. Saved checkpoints are weights-only; do not present restart
from them as an exact training continuation. Both fixed and variable controls
share the same training recipe; geometry overhead is additional to block count.

## Commands (repository root)

```bash
export MOVE_VOCAB_VERSION=compact
# Load the real checkpoint, verify initial predictions, export init.pt/manifest.
.venv/bin/python experiments/exp284_recurrent_geometry.py prepare

# Explicitly launch the pilot.
.venv/bin/python experiments/exp284_recurrent_geometry.py train

# Architecture ablations; separate output directories selected automatically.
.venv/bin/python experiments/exp284_recurrent_geometry.py train --attention shaw
.venv/bin/python experiments/exp284_recurrent_geometry.py train --attention both

# Fixed/variable attention controls for the four-arm design.
.venv/bin/python experiments/exp284_recurrent_geometry.py train --attention none --arm fixed
.venv/bin/python experiments/exp284_recurrent_geometry.py train --attention none --arm variable
.venv/bin/python experiments/exp284_recurrent_geometry.py train --attention gab --arm fixed
```

Choose `--device cuda|mps|cpu` as appropriate. `--out` must be new or empty.
Preparation initializes model modules with the configured seed and records a
checkpoint SHA256, configuration, parameter count and initial prediction error.
Training additionally records data hashes and selected evaluation rows.

Promote only after fresh paired-opening games against the same pinned engine and
protocol, with confidence intervals and ply-cap outcomes reported separately.
Teacher agreement and reduced CE do not establish Elo gains. Historical
pretraining overlap with the diagnostic cache remains unverified.
