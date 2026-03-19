"""Central configuration for the chess-transformer project."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Pretrained model to use as base."""
    name_or_path: str = "Qwen/Qwen3-0.6B"
    torch_dtype: str = "bfloat16"
    max_seq_len: int = 256  # chess positions are short


@dataclass
class AttnResConfig:
    """Attention Residuals settings."""
    enabled: bool = True
    block_size: int = 4  # layers per block for Block AttnRes
    head_dim: int = 64   # dimension for the per-layer pseudo-query


@dataclass
class RandOptConfig:
    """RandOpt post-training settings."""
    n_perturbations: int = 5000   # N random weight perturbations to sample
    top_k: int = 16               # select top K for ensemble
    noise_std: float = 0.01       # std of Gaussian perturbation
    eval_batch_size: int = 64     # batch size for evaluation of each perturbation
    eval_positions: int = 1024    # number of chess positions to evaluate per perturbation
    seed: int = 42
    # Which parameters to perturb (regex patterns on param names)
    perturb_patterns: list[str] = field(default_factory=lambda: [
        r".*mlp\..*weight",       # MLP weights
        r".*self_attn\..*weight",  # attention weights
    ])
    # Layers to skip perturbation (e.g. embedding, lm_head)
    skip_patterns: list[str] = field(default_factory=lambda: [
        r".*embed.*",
        r".*lm_head.*",
        r".*layernorm.*",
        r".*norm.*",
    ])


@dataclass
class DataConfig:
    """Chess data pipeline settings."""
    # Path to a PGN file or directory of PGNs, or HuggingFace dataset name
    source: str = "lichess"  # "lichess", "pgn", or HF dataset name
    pgn_path: str | None = None
    max_positions: int = 500_000
    stockfish_path: str | None = None  # path to stockfish binary for labeling
    stockfish_depth: int = 12
    stockfish_threads: int = 4
    train_split: float = 0.9
    cache_dir: str = "data/cache"
    board_encoding: str = "fen"  # "fen", "grid", "grid_compact", "squares"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    attnres: AttnResConfig = field(default_factory=AttnResConfig)
    randopt: RandOptConfig = field(default_factory=RandOptConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "outputs"
    device: str = "cuda"
    log_every: int = 100

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


@dataclass
class SelfPlayConfig:
    """Self-play evolutionary training settings."""
    generations: int = 100           # number of evolution steps
    games_per_matchup: int = 10      # games per pair (half as white, half as black)
    max_moves: int = 150             # max moves per game before draw
    adjudicate_material: int = 5     # material advantage threshold to adjudicate win (0=disabled)
    noise_std: float = 0.01          # Gaussian noise std for perturbations
    noise_decay: float = 0.999       # multiply noise_std each generation
    noise_floor: float = 0.001       # minimum noise_std
    temperature: float = 0.8         # sampling temperature for move generation
    max_retries: int = 5             # illegal move retries before random fallback
    constrained_decoding: bool = True # use trie-constrained logits to guarantee legal moves
    population_size: int = 4         # candidates per generation (tournament mode)
    challengers_per_gen: int = 1     # challengers sampled per generation (1v1 mode)
    elite_keep: int = 1              # elites carried over unchanged
    mode: str = "1v1"                # "1v1" or "tournament"
    seed: int = 42
    save_every: int = 10             # save checkpoint every N generations
    log_games: bool = False          # log full PGN of played games
    board_encoding: str = "fen"      # "fen", "grid", "grid_compact", "squares"
    # Which parameters to perturb (regex on param names)
    perturb_patterns: list[str] = field(default_factory=lambda: [
        r".*mlp\..*weight",
        r".*self_attn\..*weight",
    ])
    skip_patterns: list[str] = field(default_factory=lambda: [
        r".*embed.*",
        r".*lm_head.*",
        r".*layernorm.*",
        r".*norm.*",
    ])
