# Chess Transformer

Encoder-only transformer for chess: learned board embeddings, spatial policy head, Stockfish-supervised labels.

## Architecture

```
Board → LearnedBoardEncoder → [CLS] + 67 tokens → Transformer → Spatial policy + WDL value
```

| Component | File |
|-----------|------|
| Model | `chess_model.py`, `chess_transformer_factory.py` |
| Board encoding | `chess_features.py`, `move_vocab.py` |
| Training data | `data.py`, `data_loader.py` |
| Config | `config.py` |
| Train / eval / play | `train.py`, `evaluate.py`, `selfplay.py`, `play.py`, `uci_engine.py` |

Active experiments start at **exp052** (`experiments/`). Older runs live in `archive/experiments/`.

## Setup

```bash
pip install -e .
pip install git+https://github.com/KellerJordan/Muon   # exp182 pretrain
pip install stockfish   # optional, for labeling / eval / RL
```

### RunPod (A100 / Linux)

```bash
git clone https://github.com/avewright/transform.git
cd transform
bash scripts/runpod_setup.sh

# Pretrain 705M (A100 preset)
python experiments/exp182_pretrain_700m.py --go --a100 --smoke
python experiments/exp182_pretrain_700m.py --go --a100 --resume

# Expert-iteration self-play
python experiments/exp183_selfplay.py --preset a100 --go --mode sf
```

Checkpoints save under `outputs/` (gitignored). Upload to Hugging Face or S3 before stopping the pod.

Legacy: `bash setup.sh` still works for older experiment paths.

## Quick start

```bash
python -u experiments/exp052_head_comparison_v2.py   # small spatial model
python -u experiments/exp053_scaled_spatial.py       # medium (512d/8L)
python -u experiments/exp055_joint_policy_value.py   # policy + value
python -u experiments/exp054_search_baseline.py      # vs Stockfish
```

## Max Elo harness

Champion metric is **greedy policy Elo only** (no opening book, no Syzygy), frozen in [`harness/protocol.json`](harness/protocol.json). Train may screen on `top1`; **never promote on soft_loss**. Seed weights: `avewright/chess-transformer-437m-ft3h` → `outputs/hf_437m_ft3h_hub/best_model.pt`.

```bash
# Pure-policy Elo (promotion protocol)
python -m harness.elo --ckpt outputs/hf_437m_ft3h_hub/best_model.pt --mode policy

# Soft FT → Elo → promote (writes outputs/champion/)
python -m harness.loop --name ft3j --soft-cache outputs/hf_soft_mix/soft_cache.pt \
  --init outputs/champion/champion.pt

# Inspect champion
python -m harness.promote --show

# Path-to-2500 stages (p0/p2-ft/p3 wired to harness)
bash scripts/run_path_2500.sh p0
bash scripts/run_path_2500.sh p2-ft
```

Pin Stockfish with `STOCKFISH_PATH`. MCTS Elo is report-only until policy clears ~2000.

## Layout

```
├── harness/                    # max-Elo train/eval/promote
├── chess_model.py              # encoder, spatial head, ChessTransformerV2
├── chess_transformer_factory.py
├── chess_features.py / move_vocab.py / config.py
├── data.py / data_loader.py
├── train.py / evaluate.py / selfplay.py / play.py
├── chess_inference.py          # load checkpoint + greedy move
├── rl_selfplay/                # MCTS expert-iteration loop
├── experiments/                # exp052+ (current)
├── scripts/runpod_setup.sh     # RunPod / A100 quick setup
├── archive/                    # legacy Qwen path, scratch, early exps
└── scripts/compact_repo.ps1    # one-shot tidy (moves, never deletes)
```

`syzygy/` tablebases and `data/*.jsonl` labels stay local (gitignored) — download or regenerate as needed. Checkpoints live under `outputs/` (also gitignored).

Legacy Qwen backbone + text self-play code is under `archive/legacy/` if you need it.
