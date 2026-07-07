"""AlphaZero-style expert iteration: MCTS self-play → train on visit distributions."""

from rl_selfplay.config import SelfPlayConfig, laptop_8gb_config, a100_80gb_config

__all__ = ["SelfPlayConfig", "laptop_8gb_config", "a100_80gb_config"]
