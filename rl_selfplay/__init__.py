"""AlphaZero-style expert iteration: MCTS or search-free self-play → train."""

from rl_selfplay.config import (
    SelfPlayConfig,
    a100_80gb_config,
    laptop_8gb_config,
    searchfree_99m_config,
)

__all__ = [
    "SelfPlayConfig",
    "laptop_8gb_config",
    "a100_80gb_config",
    "searchfree_99m_config",
]
