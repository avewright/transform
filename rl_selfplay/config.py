"""Self-play / expert-iteration presets for 8GB laptop and A100 80GB."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SelfPlayConfig:
    """One expert-iteration cycle: generate MCTS games, then fine-tune."""

    # Generation
    mode: str = "self"  # "self" | "sf" | "prior"
    n_games: int = 8
    mcts_sims: int = 32
    mcts_batch_size: int = 4
    mcts_c_puct: float = 2.5
    visit_temp: float = 1.0
    ply_cap: int = 200
    sf_elo: int = 1500  # ignored when sf_full_strength=True
    sf_full_strength: bool = False  # True = no UCI_LimitStrength
    sf_depth: int | None = None  # if set, Limit(depth=...) instead of time
    sf_move_time: float = 0.05
    root_noise_frac: float = 0.25
    record_sf_moves: bool = False  # save SF positions for supervised CE

    # Training
    train_epochs: int = 1
    train_batch_size: int = 8
    train_lr: float = 1e-5
    value_weight: float = 0.5
    grad_clip: float = 0.5
    mix_sf_frac: float = 0.0  # fraction of batches from external supervised shards
    sf_shard_dir: str | None = None
    # Within collected positions: fraction of batches that use SF hard moves
    # (rest use soft MCTS visit targets). Only applies when SF records exist.
    mix_sf_move_frac: float = 0.5

    # Dataset
    dataset_dir: str | None = None  # cumulative position dataset root

    # Loop
    iterations: int = 1
    games_per_iter: int | None = None  # overrides n_games each iter if set
    eval_games: int = 4
    eval_sims: int = 16

    # Paths
    checkpoint: str | None = None
    prior_checkpoint: str | None = None
    output_dir: str = "outputs/rl_selfplay"

    # Hardware
    use_fp16: bool = True
    use_bf16: bool = False
    num_workers: int = 1  # reserved for future parallel game workers

    def to_dict(self) -> dict:
        return asdict(self)


OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4"],
    ["e2e4", "c7c5", "g1f3"],
    ["d2d4", "g8f6", "c2c4"],
]


def laptop_8gb_config(**overrides) -> SelfPlayConfig:
    """RTX 4060 8GB: low sims, small batches, short games."""
    cfg = SelfPlayConfig(
        n_games=4,
        mcts_sims=32,
        mcts_batch_size=4,
        train_batch_size=8,
        ply_cap=150,
        eval_games=2,
        eval_sims=16,
        use_fp16=True,
        output_dir="outputs/rl_selfplay_8gb",
    )
    return replace(cfg, **overrides) if overrides else cfg


def a100_80gb_config(**overrides) -> SelfPlayConfig:
    """A100 80GB: high sims, large MCTS batches, more games per iter."""
    cfg = SelfPlayConfig(
        n_games=64,
        mcts_sims=200,
        mcts_batch_size=32,
        train_batch_size=64,
        ply_cap=300,
        eval_games=20,
        eval_sims=100,
        iterations=10,
        games_per_iter=64,
        use_fp16=False,
        use_bf16=True,
        mix_sf_frac=0.25,
        sf_shard_dir=str(ROOT / "outputs" / "exp139_massive_train" / "shards"),
        output_dir="outputs/rl_selfplay_a100",
    )
    return replace(cfg, **overrides) if overrides else cfg


def a40_45gb_config(**overrides) -> SelfPlayConfig:
    """A40 45GB: full-strength SF (low depth) + soft MCTS; fill VRAM."""
    cfg = SelfPlayConfig(
        n_games=32,
        mcts_sims=200,
        mcts_batch_size=192,
        train_batch_size=192,
        ply_cap=200,
        eval_games=8,
        eval_sims=100,
        iterations=40,
        games_per_iter=32,
        sf_full_strength=True,
        sf_depth=8,              # full strength, shallow for speed
        sf_elo=3200,             # unused when full strength
        record_sf_moves=True,    # keep SF moves in dataset + train CE
        visit_temp=1.0,          # soft MCTS visit distribution
        root_noise_frac=0.25,
        mix_sf_move_frac=0.5,    # half batches SF hard, half soft MCTS
        mix_sf_frac=0.0,
        sf_shard_dir=None,
        dataset_dir="outputs/rl_selfplay_a40_soft/dataset",
        use_fp16=False,
        use_bf16=True,
        output_dir="outputs/rl_selfplay_a40_soft",
    )
    return replace(cfg, **overrides) if overrides else cfg
