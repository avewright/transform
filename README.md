# Chess-Transformer

Training a chess model by feeding **learned board embeddings** directly into a pretrained LLM backbone (Qwen3-0.6B) via `inputs_embeds`, bypassing text tokenization entirely. The LLM functions as a deep feature mixer over structured chess representations.

## Architecture

```
Board State (chess.Board)
        │
   LearnedBoardEncoder
        │  piece_embed(7 types) × color_proj(3 linear) + square_embed(64) + context tokens
        │  → 67 tokens × 256d
        ▼
   Linear Projection (256 → 1024)
        │
   Qwen3-0.6B Backbone (frozen, 28 layers, via inputs_embeds)
        │
   ┌────┴────┐
Policy Head  Value Head
(→ 5504 moves) (→ W/D/L)
```

**Key components:**
- **LearnedBoardEncoder** (`chess_model.py`): Per-square embeddings with color projections + game-state context tokens (turn, castling, en passant). 223K params
- **Move vocabulary** (`move_vocab.py`): 5504 possible UCI moves with legal-move masking at inference
- **Board features** (`chess_features.py`): Board → token ID conversion for the learned encoder
- **ChessModel** (`chess_model.py`): Full model tying encoder → backbone → heads. 7.4M trainable / 603M total

## Results So Far

| Experiment | Data | Accuracy | Notes |
|-----------|------|----------|-------|
| exp008 | 200 self-distilled | 24% | CNN encoder PoC, 100% legal |
| exp009 | 500 self-distilled | 46% (learned) vs 50% (CNN) | Learned encoder matches CNN with 21x fewer params |
| exp010 | 500 self-distilled | 48% (all variants) | Unfreezing backbone doesn't help — data is bottleneck |
| exp012b | 5000 Stockfish d10 | 14.2% (top3: 31%) | Stockfish labels are harder. Gets checkmated by SF d3 |
| exp_av_v2 | 5K random + SF all-move | 8.8% (top3: 21.8%) | AV vs policy CE = TIE on random positions |
| exp_av_real | 3K HF games + SF all-move | 23.6% (top3: 48.8%) | AV vs policy = TIE; real games >>  random positions |
| exp013 | 50K HF game-play | 25.0% (top3: 45%) | Policy CE, best result so far |

100% legal move rate throughout (via legal-move masking).

**Key finding:** Position quality (real games vs random play) matters far more than loss function design (policy CE vs action-value Q). Real game positions give ~23% accuracy at 3K while random positions give ~9% at 5K.

**Next milestone:** Scale real-game Stockfish-labeled positions to 50K+ and test soft-target/KL formulations.

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
# Run the Stockfish-supervised training experiment:
python experiments/exp012b_quick_stockfish.py

# Self-play evolution (legacy text mode):
python train.py selfplay --generations 10 --games 4
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
1. **Scale real-game data to 50K+** with Stockfish best-move labels on HF game positions
2. **Soft targets**: Test top-k move probabilities or KL divergence instead of hard best-move CE
3. **Chess-native transformer**: Build small encoder-only model (12-16 layers, 384-768d) trained from scratch on chess tokens
4. **LoRA fine-tuning (exp015)**: Unfreeze backbone attention via low-rank adaptation — test after data scaling saturates
5. **MCTS search (exp014)**: Use policy+value heads for tree search — test after value quality improves
6. **Goal: Beat Stockfish** at progressively higher depth levels

## References

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): Base transformer backbone
- [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, 2026)
- [Neural Thickets / RandOpt](https://arxiv.org/abs/2603.12228) (Gan & Isola, MIT, 2026)
