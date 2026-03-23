# Chess-Transformer

Training a chess engine from scratch using a **chess-native encoder-only transformer** with learned board embeddings, a factorized spatial policy head, and Stockfish-supervised labels. The model learns move priors from positions; search + value head turn priors into gameplay.

## Current Architecture

```
Board State (chess.Board)
        │
   LearnedBoardEncoder (256d)
        │  piece_embed(7 types) × color_proj(3 linear)
        │  + square_embed(64) + context tokens (turn, castling, ep)
        │  → 67 tokens × 256d
        ▼
   [CLS] token (learned) prepended → 68 tokens
        │
   Linear Projection (256 → 512)
        │
   Encoder-Only Transformer (8 layers, 8 heads, bidirectional)
        │           norm_first=True, GELU, 512d hidden
        │
   ┌────┴────┐
Spatial       WDL Value
Policy Head    Head
(from×to×promo) (→ W/D/L)
```

**Key components:**
- **LearnedBoardEncoder** (`chess_model.py`): Per-square embeddings with color projections + game-state context tokens. ~223K params
- **SpatialPolicyHead**: Factorized from-square × to-square × promotion scoring using per-square hidden states + CLS context. ~1M params
- **Move vocabulary** (`move_vocab.py`): 5504 possible UCI moves with legal-move masking at inference
- **ChessTransformerV2**: Full model (encoder + transformer + heads). ~26M params (Medium config)

> **Legacy:** An older Qwen3-0.6B frozen-backbone path exists in `chess_model.py` (ChessModel). The chess-native transformer consistently outperforms it and is the active research direction.

## Results

| Experiment | Model | Data | Top-1 Acc | Top-3 | Notes |
|-----------|-------|------|-----------|-------|-------|
| exp052 flat | Small 256d/6L | 47.5K HF | 11.3% ± 0.2% | 28.8% | Flat head baseline, 3 seeds |
| **exp052 spatial** | Small 256d/6L | 47.5K HF | **30.3% ± 0.2%** | 52.5% | Spatial head, CLS token, 3 seeds |
| **exp053** | Medium 512d/8L | 47.5K HF | **35.3%** | 58.6% | Scaled spatial, 2 seeds |
| exp055 | Medium + joint value | 47.5K HF | 35.1% | 58.1% | Value head trained (80% WDL) |
| exp046 | 8L transformer | 209K Lichess 2200+ | 37.1% | 62.0% | Top-player data from scratch |

100% legal move rate throughout (via legal-move masking).

### Search / Gameplay Results

| Strategy | vs SF d1 | vs SF d2 | vs SF d3 | Notes |
|----------|----------|----------|----------|-------|
| Policy argmax | W0/D3/L5 (18.8%) | W0/D0/L8 (0%) | W0/D1/L7 (6.2%) | Baseline |
| **Value rerank k5** | **W0/D6/L2 (37.5%)** | W0/D1/L7 (6.2%) | W0/D0/L8 (0%) | **Best strategy** |
| Alpha-beta 2-ply k5 | W0/D1/L7 (6.2%) | W0/D1/L7 (6.2%) | W0/D1/L7 (6.2%) | Uniform but weak |

**Key findings:**
- Spatial policy head is **2.7x better** than flat head with fewer params
- Medium model (26M) outperforms Small (6M) by ~5% absolute
- **Value reranking doubles gameplay score** at SF d1 (37.5% vs 18.8%)
- Jointly-trained WDL value head outperforms SF-calibrated value head for search
- Deeper search (2-ply) hurts because the value head is too noisy for minimax
- The bottleneck is now data volume (47.5K is too few for 26M params)

**Active direction:** Scale training data to 200K+ positions for the Medium spatial model, then retest search.

## Setup

### RunPod / Linux GPU
```bash
git clone https://github.com/avewright/transform.git
cd transform
bash setup.sh
```

### Windows (local)
```bash
pip install -e .
pip install stockfish
# Download Stockfish binary to stockfish/stockfish/
```

## Quick Start

```bash
# Train spatial policy model (Small, ~10 min):
python -u experiments/exp052_head_comparison_v2.py

# Train Medium spatial model (~30 min):
python -u experiments/exp053_scaled_spatial.py

# Joint policy+value training:
python -u experiments/exp055_joint_policy_value.py

# Search baseline (play games vs Stockfish):
python -u experiments/exp054_search_baseline.py
```

## Project Structure

```
├── chess_model.py          # ChessModel, LearnedBoardEncoder, BoardEncoder (CNN)
├── chess_features.py       # Board → tensor conversion (token IDs + feature planes)
├── move_vocab.py           # 5504 UCI move vocabulary, legal masking
├── config.py               # All configuration dataclasses
├── model.py                # Qwen model loading
├── data.py                 # PGN parsing, board encodings, Stockfish labeling
├── constrained.py          # Trie-based constrained decoding (text mode)
├── selfplay.py             # Self-play game loop, move generation
├── evaluate.py             # Evaluation utilities
├── randopt.py              # Random optimization (perturbation search)
├── attnres.py              # Block Attention Residuals
├── train.py                # CLI entry point (selfplay / randopt modes)
├── setup.sh                # RunPod / Linux GPU setup
├── experiments/            # Individual experiment scripts (exp001-017)
└── .github/instructions/   # Agent instructions
```

## Research Log

### Phase 1: Text-based self-play (exp001-006)
- Qwen3-0.6B + grid_compact encoding + material adjudication
- σ=0.01 perturbations consistently lose to champion — too small to find improving directions

### Phase 2: Embedding-based architecture (exp008-012)
- **exp008**: CNN encoder → Qwen backbone → policy head works. Untrained model picks legal opening moves
- **exp009**: Learned embedding encoder matches CNN with 21x fewer params (223K vs 4.8M)
- **exp010**: Unfreezing backbone doesn't help — data volume is the bottleneck
- **exp012b**: Stockfish depth-10 labels. 5K positions → 14.2% accuracy, top3 31%. Needs more data + compute

### Phase 3: Supervision quality & data scaling (exp_av, exp013)
- **exp_av_comparison_v2**: Fair A/B on 5K random positions — policy CE vs action-value Q(s,a) = TIE at 8.8%
- **exp_av_real_games**: Same comparison on 3K real game positions — TIE at 23.4-23.6%
- **Key insight**: Position quality (real games vs random play) dominates loss function choice. 15pp gap from data source, 0pp from AV signal
- **exp013**: 50K HF game-play positions with policy CE → 25% accuracy (best result)

### Next Steps
1. **Train value head jointly** (exp055) — WDL + soft policy targets from Stockfish
2. **Build search** (exp054) — top-k policy + value reranking + MCTS
3. **Beat Stockfish depth 1** — first realistic gameplay milestone
4. **Scale data** — combine HF + Lichess + SF-synthetic for 1M+ diverse positions
5. **Deeper search** — alpha-beta with iterative deepening once value head is calibrated

## References

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): Base transformer backbone
- [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, 2026)
- [Neural Thickets / RandOpt](https://arxiv.org/abs/2603.12228) (Gan & Isola, MIT, 2026)
