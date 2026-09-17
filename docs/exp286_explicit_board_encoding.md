# Experiment 286: explicit rule features on the pretrained 99M

Status: design only; no training launched. This is an additive encoding test,
not a reproduction of ChessBot or a replacement of the pretrained encoder.

## Question

Does exposing rule state directly at each square improve search-free strength
over matched continuation of our existing fused-piece/FiLM encoder? Separate
easier access to existing information from adding previously missing information.

Use the original 98,971,224-parameter, three-pass squares64 checkpoint as the
starting point. The current candidate is outputs/hf_100m_squares64/latest.pt;
resolve its availability and record its SHA256 before execution. Do not assume
it is the latest remote champion. Every arm starts from the same frozen bytes.

## Architecture and arms

Keep piece embeddings, square embeddings, all FiLM modules, 256-dimensional
encoder, 736-wide trunk, three recurrent passes, spatial policy head, and WDL
head. Add a bias-free projection into the encoder immediately before its final
LayerNorm, after existing FiLM transforms:

    h_new[s] = LayerNorm(h_after_film[s] + W_rules @ rules[s])

Initialize W_rules to zero. All arms therefore produce the original predictions
at initialization. Do not add a second zero-initialized gate, which would block
learning. Keep all original weights trainable.

| Arm | New encoder inputs | What it tests |
| --- | --- | --- |
| A | None; matched continuation | Effect of more training alone |
| B | Six channels: side to move, four individual castling flags, spatial EP target | Explicit access to information already available through FiLM |
| C | B plus normalized halfmove clock and clock-known flag | Adding trustworthy rule-state information |

B adds 6 × 256 = 1,536 parameters; C adds 8 × 256 = 2,048.
Clock-known is an experimental data-integrity feature, not part of ChessBot.
Side to move is white=1/black=0; rights are separate WK/WQ/BK/BQ flags, each
broadcast to all squares. EP is 1 only on the stored target square. Clock is
min(halfmove_clock, 100)/100 and known=1 when faithfully preserved from source;
otherwise both clock channels are zero. Live inference always has a known clock.
Use our absolute a1=0 square order throughout, not ChessBot's flipped rank order.

B versus A is the primary representation test. C versus B is an information test;
a C win must not be presented as evidence that planes beat embeddings.
If C wins, follow with clock-only versus A to check whether the extra six channels
were necessary. Whole-encoder replacement and square-specific multiplicative
gating are later experiments; they would change additional factors here.

## Data gate before running C

The existing exp283 cache loader requires board_array, turn, castling, ep_square
and policy targets, but does not require a halfmove clock. Its reconstructed
boards cannot establish that original clocks survived. Inspect source schemas,
cache construction, actual clock coverage, and engine-label provenance first.

- Never treat python-chess's default zero clock as a recovered source clock.
- Never infer a clock from piece placement, and never attach randomized clocks
  to existing targets. A changed clock can change the correct value or move.
- Verify Stockfish labels were generated from the same clock-bearing FEN. If
  clocks were stripped before labeling, regenerate affected labels or exclude
  those rows from the clock experiment.
- If trustworthy clocks are unavailable, run A/B and explicitly defer C. B is
  independently useful; do not fabricate a complete three-arm result.
- For A/B/C comparisons, use exactly the same rows, labels and row order for
  all three arms. If C needs a rebuilt dataset, rerun its controls on that data.

Start from the local organized_chess_v1 mixed cache used by exp283, subject to
the audit above. Freeze data hashes, source counts, policy/value eligibility,
and split manifests. Remove evaluation overlap before sampling training rows.
Group splits by board state (pieces, turn, castling, EP), ignoring clock, so clock
variants cannot leak across splits. Prefer source-game grouping where IDs exist.
Historical pretraining overlap remains unknown unless separately audited.

Report coverage for EP positions, each castling state, and verified clock buckets
0–19, 20–59, 60–79, 80–99, and >=100. Source-missing clocks form their own bucket.
Do not oversample rare cases differently across arms. An empty slice is no evidence.

## Training budget

First run preparation and 100-update smoke checks. Then use 10,000 updates at
effective batch 64: 640,000 position presentations per arm. Microbatch 8 with
accumulation 8 is the initial memory-conservative implementation. Benchmark
throughput on the actual machine and report estimated hours before a full run;
this design is not authorization to rent hardware.

All arms use fresh AdamW, original weights LR 1e-5, new projection LR 1e-4,
weight decay .01, 500-update warmup, cosine decay to 10% of peak, and clipping 1.
Apply existing three-pass recurrent-gradient averaging once per optimizer update,
after accumulation and before clipping. Keep dropout at .05. Train the full model.

Loss is .45 hard policy CE + .55 existing soft-policy CE + .15 valid-row WDL CE,
with identical masks and reductions across arms. No label recipe change, teacher
KD, augmentation, geometry additions, history features, PPO, or variable depth.

Precompute identical sample indices for each seed. Reset RNG after constructing
each arm so new module initialization does not shift dropout randomness. Initial
seed 286; repeat baseline and selected candidate with seed 1286 before claiming
a reproducible improvement. Record exposure, unique rows seen, updates, wall
time, peak memory, and inference latency. Equal exposure is the primary match;
report actual extra compute rather than claiming exact FLOP matching.

Save resumable checkpoints including optimizer, scheduler, RNG and sampler
position. Pre-register step 10,000 as the comparison checkpoint; diagnostics at
2,000/5,000/10,000 do not authorize picking a lucky intermediate checkpoint.

## Validation and evaluation

Required correctness checks before training:

1. Zero projection preserves original logits within device-appropriate numerical
   tolerance on real boards, including castling, EP, and promotions.
2. Every channel is correctly encoded for both colors; verify hflip mappings if
   augmentation is later introduced. No augmentation in this experiment.
3. New weights receive nonzero gradients on appropriate examples.
4. Training and live inference construct identical features from identical FENs.
5. Save/load round-trip preserves outputs; old checkpoints still load unchanged.

Diagnostics: legal top-1, hard CE, soft CE, valid-row WDL CE, and paired rescues
versus regressions, overall and by phase/rule-feature slice. Treat these as
diagnostics, not strength proof. Clock-sensitive paired positions require fresh
engine labels for each clock; changing clocks alone is only an encoding test.

Use greedy legal policy, no search/book/tablebases. Pin Stockfish 19 binary/hash,
threads=1, hash=32MB, 50ms per move, and use a 400-ply cap. Reevaluate the untouched
source checkpoint and matched control with that cap; do not compare directly
against old 160-ply Elo estimates. Isolate engine CPU resources across matches.

Screen each final checkpoint at SF 1900/2050/2200 with the existing eight openings,
both colors, two repeats: 96 games per arm. Use the aggregate game score to choose
at most one candidate for confirmation; break exact ties in favor of simpler B.
No candidate beating A means no automatic expansion of the experiment budget.

Freeze the selected candidate before fresh confirmation. Play 200 held-out opening
pairs (400 games) directly against its matched-continuation A control, then repeat
with the second training seed. Use the same held-out openings for both seeds and
never tune on those games. Also run both seeds' candidates/controls against a
common pinned SF ladder to check opponent-specific gains.

Report W/D/L, score, opening-pair bootstrap 95% confidence intervals, and relative
Elo with uncertainty. Bootstrap whole opening clusters when combining seeds, not
individual games. Report capped games separately and finish any capped games in
a predeclared extension to 800 plies; show sensitivity rather than allowing an
arbitrary cap to create an apparent win. If unresolved caps change the decision,
the result is inconclusive. This does not estimate FIDE or Lichess Elo.

Practical target: >=30 relative Elo over A (about 54.3% direct-match score), a
combined paired 95% interval above zero, positive direction in both training
seeds, and no material regression against the common SF ladder. At 400 games per
seed the interval may still be wide: an inconclusive result remains inconclusive,
with no repeated peeking until significance. Any larger confirmation is a newly
fixed follow-up budget. Never auto-promote from a diagnostic loss improvement.

## Interpretation and next action

- B wins: explicit rule access helps; retain the residual adapter and replicate.
- Only C wins: missing clock information is the leading explanation; run clock-only.
- Neither wins: this pilot does not support prioritizing rule encoding. Test the
  ChessBot-style global square-pair policy head next, under matched supervision.
  That follow-up is exp287: ChessBot architecture at 99M size/depth on ChessFENS.
- Loss improves but games do not: no playing-strength win.
- All arms regress: inspect continuation/forgetting before judging encoding.

This cannot attribute ChessBot's overall advantage to encoding, establish an
optimal representation from scratch, or show that a small null result proves
equivalence. It tests a low-disruption, directly actionable upgrade to our model.

Implementation deliverables when this design is executed: optional rules-adapter
config and encoder, audited feature/cache path, inference integration, matched
training runner, focused correctness tests, and a results manifest. No executable
runner or completed result is implied by this design document.
