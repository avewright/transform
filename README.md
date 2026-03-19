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

100% legal move rate throughout (via legal-move masking).

**Next milestone:** Scale to 50K-100K Stockfish-labeled positions with longer training on GPU (RunPod).

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

### Phase 3: Advanced training & search (exp013-017, in progress)
- **exp013**: Action-value Q(s,a) training — label ALL legal moves per position with Stockfish, ~30× more gradient signal per position
- **exp014**: Monte Carlo Tree Search at inference — use policy head as prior + value head for leaf evaluation
- **exp015**: LoRA fine-tuning of Qwen backbone attention layers (rank=16, alpha=32) to unlock backbone adaptation
- **exp016**: Enriched board encoder with attack maps, pawn structure, material balance, game phase, mobility (71 tokens vs 67)
- **exp017**: Data scaling law measurement — fit power law at 1K/5K/10K/20K and extrapolate to 100K-1M

### Next Steps
1. **Scale data to 100K+ Stockfish-labeled positions** on RunPod GPU — scaling law (exp017) suggests this is the single biggest lever
2. **Action-value training (exp013)**: Train on all legal move evaluations per position for denser gradient signal
3. **MCTS search (exp014)**: Use policy+value heads for tree search at inference to improve playing strength beyond raw accuracy
4. **LoRA fine-tuning (exp015)**: Unfreeze backbone attention via low-rank adaptation — likely matters more at larger data volumes
5. **Rich features (exp016)**: Attack maps, pawn structure, material balance give the encoder more chess knowledge
6. **Knowledge distillation**: Use Stockfish top-3 moves as soft targets (KL divergence loss) instead of hard best-move labels
7. **Curriculum learning**: Train on progressively harder positions (simple endgames → complex middlegames)
8. **Hybrid architecture**: Combine learned embeddings with lightweight CNN features for both global and local pattern recognition
9. **Goal: Beat Stockfish** at progressively higher depth levels

## References

- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): Base transformer backbone
- [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, 2026)
- [Neural Thickets / RandOpt](https://arxiv.org/abs/2603.12228) (Gan & Isola, MIT, 2026)
