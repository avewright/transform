# Value99: scratch scalar-value training on a 20GB CUDA GPU

Status: pipeline prepared; **training has not been launched**. A 1M-position local preparation finished before the request to defer downloads. No further data preparation is running. Its location is `outputs/value99_data_v1`; nothing automatically transfers it to a rental GPU.

## Model and move selection

98,920,577 parameters, randomly initialized. No ChessBot weights are loaded.

| Component | Value99 |
| --- | --- |
| Transformer | 24 independent pre-LN blocks, width 576, 8 heads |
| Attention | Q/K RMS normalization, learned 2D relative square bias, PyTorch SDPA |
| Feed-forward | SwiGLU, inner width 1,584 |
| Board tokens | 64 squares, canonicalized so side to move is White |
| Square input | 12 piece channels, 4 castling rights, en-passant square, halfmove clock: 18 features |
| Position | Learned square embedding plus relative attention bias |
| Value decoder | Each square 576→128; flatten 64 squares; 8,192→128→1 |
| Output | One scalar `V = 2 sigmoid(logit) - 1`, in [−1,+1], side-to-move perspective |

ChessBot has 10 transformer layers; this uses 24 separately learned layers. Recurrent weight sharing is not part of this first experiment. Depth, normalization and the new decoder are hypotheses to test, not evidence of stronger chess.

The target is `P(win) − P(loss)`. Soft binary cross entropy trains the logit against `P(win) + 0.5 P(draw)`. This predicts expected game score, not centipawns; a draw and an uncertain 50/50 win/loss prediction both map to zero. The network receives no move history, so repetition must be handled by the playing environment. Fullmove number is omitted; halfmove clock is clipped at 100.

For each legal move, build its successor board and compute **`move_score = −V(successor)`**. Pick the largest score. `score_legal_moves` batches successors and resolves terminal outcomes. This is one-ply evaluation with no opponent-reply search; evaluating every successor costs many more network evaluations than one policy forward pass.

## Data plan and quality limits

Primary candidate: [Maxlegrec/ChessFENS](https://huggingface.co/datasets/Maxlegrec/ChessFENS), about 732M rows. Revision is pinned in the preparation script. Read only `fen` and `wdl` columns from remote Parquet; skip the much larger policy column. WDL order is source-documented side-to-move win/draw/loss. Mirroring Black-to-move positions does not negate this target.

The preparation pipeline checks complete six-field FENs, standard-chess validity, finite nonnegative normalized WDL, and excludes terminal boards. Canonical position hashes deduplicate rows and permanently assign 99% train / 0.5% validation / 0.5% test. The split ignores clocks to keep identical arrangements with different clocks together; deduplication also retains only the first such arrangement. That reduces clock diversity and should be revisited for a dedicated fifty-move-rule dataset.

**Scale is established; uniform teacher quality is not.** This source lacks per-row search visits/depth and game IDs. Splits are position-disjoint, not proven game-disjoint. Nearby positions from the same game may cross splits. The default pilot takes prefixes from shuffled source shards, not a uniform random sample of every row. The pilot had 989,940 training, 5,066 validation and 4,994 test rows after filtering 1,106,921 source rows. All accepted source rows were White to move. Its coarse phase proxy classified 19.5% opening, 39.0% middlegame, 41.5% low-material positions; it is not a balanced playing distribution.

Other sources worth assessing before a long run:

- [Raw LCZero training data](https://storage.lczero.org/files/training_data/) has richer teacher metadata; [format documentation](https://lczero.org/dev/wiki/training-data-format-versions/) describes search values and visits. A separate decoder and provenance/effort filters would be required.
- [Lichess position evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) can supply searched Stockfish targets, but depth varies, centipawns need calibrated conversion and some records lack complete rule state. No importer is implemented here.
- [Pawitt/zero-evaluator](https://huggingface.co/datasets/Pawitt/zero-evaluator) is useful for an auxiliary comparison, but its 2M LCZero subset is not 2M searched value labels, and the Stockfish-zero labels are static depth-zero evaluations.

Before committing to tens of millions of examples, independently re-score a stratified held-out sample with fixed-budget Stockfish and test one-ply move quality. Those audits are next steps, not completed validation. Lower value loss alone is insufficient to establish Elo improvement.

## CUDA setup — no download occurs until the preparation command

Use a CUDA-compatible PyTorch installation for the GPU rental, then install dependencies from `configs/value99_requirements.txt`. BF16 requires a supported GPU; VRAM capacity alone does not establish BF16 support. For an older CUDA card, explicitly change `precision` to `fp32` and repeat the memory check.

Run the synthetic preflight first. It constructs the full model and optimizer, executes three updates, reports peak allocated/reserved VRAM, and accesses no dataset or pretrained weights:

```bash
python scripts/benchmark_value99.py
```

Defaults: BF16 compute with FP32 weights and AdamW states, gradient checkpointing, microbatch 8, effective batch 128 through accumulation. These are conservative starting settings for 20GB, **not a measured fit guarantee**. If the memory check fails, try `--microbatch 4` or `2`, then put the successful setting in the training config. Leave room for CUDA workspaces and validation. The preflight's optimizer update per microbatch measures memory, not full accumulated-step throughput.

**Only when ready to fetch data**, run this separately, preferably on the training machine:

```bash
python scripts/prepare_value99_data.py --rows 1000000 --out outputs/value99_data_v1
```

The output directory must be empty. Preparation finishes its manifest before training can begin. Larger preparations can raise `--rows` and `--per-source-shard`; e.g. 50M accepted rows requires scanning more than 50M source rows. The preparer keeps its deduplication hashes in host memory; plan host RAM separately from GPU VRAM. It is a bounded preparation job, not a fault-tolerant distributed ingestion system.

Then start the pilot explicitly:

```bash
python experiments/value99_pretrain.py --device cuda
```

Default pilot: 8,192 optimizer steps × 128 positions = 1,048,576 presentations. This is a pipeline/learning sanity check, not a finished 99M model. Edit `data`, `steps` and schedule in the config for later-scale runs, using a new output directory. Nothing launches automatically.

Training uses disk-backed arrays, holds the shuffled index order in host RAM, and streams batches to CUDA. The first load verifies shard checksums and builds a local `.mmap` cache, costing about 75 bytes per example in addition to compressed shards. Later loads trust this derived cache; remove `.mmap` to rebuild/reverify it. Position hashes and original NPZs remain in the prepared data. One training process per data cache is expected. At very large scale, random disk access and the global index shuffle can become bottlenecks.

The trainer logs training loss, validation value error by phase, positions/second and peak GPU allocation. It checkpoints weights, AdamW, RNG state, shuffle/cursor and data/source/config hashes every 128 steps. Validation uses a fixed 2,048-position subset; the test split is untouched. Resume requires identical config/source/data and run device:

```bash
python experiments/value99_pretrain.py --device cuda --resume
```

Create `outputs/value99_pretrain_v1/STOP` for a clean stop at the next optimizer-step boundary. Remove it before resuming. CUDA performance and memory have not been measured on this Mac. CPU unit tests cover architecture size, canonical encoding, label direction/split consistency, backward/checkpointing, scalar range, one-ply negation/checkmate and disk-backed loading.
