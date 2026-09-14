"""Frozen 99M expert bank + a small trainable router.

Experts stay eval()/requires_grad_(False). The router is the only trained
module. Inference is Switch-style: route to one expert, run that forward.

Expert ids (stable):
  0 incumbent  avewright/chess-transformer-100m-squares64
  1 puzzle     avewright/puzzle-model
  2 endgame    avewright/endgame-model
  3 opening    avewright/opening-model
  4 middlegame avewright/middlegame-model
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64, count_parameters

TRUNK_HIDDEN_DIM = DEFAULT_100M_SQUARES64_CONFIG.hidden_dim

EXPERTS: tuple[tuple[str, str], ...] = (
    ("incumbent", "avewright/chess-transformer-100m-squares64"),
    ("puzzle", "avewright/puzzle-model"),
    ("endgame", "avewright/endgame-model"),
    ("opening", "avewright/opening-model"),
    ("middlegame", "avewright/middlegame-model"),
)
N_EXPERTS = len(EXPERTS)
EXPERT_NAMES = [n for n, _ in EXPERTS]
EXPERT_REPOS = {n: r for n, r in EXPERTS}


def n_pieces_from_fused(fused_ids: torch.Tensor) -> torch.Tensor:
    """fused empty token is 0. Returns (B,) int16."""
    return (fused_ids != 0).sum(dim=1).to(torch.int16)


def phase_prior(n_pcs: torch.Tensor) -> torch.Tensor:
    """Heuristic expert id from piece count. Incumbent/puzzle never win the prior."""
    out = torch.full_like(n_pcs, 4, dtype=torch.long)  # middlegame
    out = torch.where(n_pcs >= 26, torch.full_like(out, 3), out)  # opening
    out = torch.where(n_pcs <= 13, torch.full_like(out, 2), out)  # endgame
    return out


def phase_label_to_expert(phase: torch.Tensor) -> torch.Tensor:
    """Dataset phase 0/1/2 → opening/middlegame/endgame expert ids."""
    table = torch.tensor([3, 4, 2], device=phase.device, dtype=torch.long)
    return table[phase.long().clamp(0, 2)]


class ExpertRouter(nn.Module):
    """MLP on a frozen 99M trunk's global_hidden. No second encoder."""

    def __init__(self, hidden_dim: int | None = None, mlp_hidden: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim or TRUNK_HIDDEN_DIM)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, N_EXPERTS),
        )

    def forward(self, global_hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(global_hidden)


@dataclass
class RoutedForward:
    policy_logits: torch.Tensor
    value_logits: torch.Tensor
    gate_logits: torch.Tensor
    expert_id: torch.Tensor
    expert_name: str
    stem_policy_logits: torch.Tensor | None = None
    stem_value_logits: torch.Tensor | None = None

    def __getitem__(self, key: str):
        return getattr(self, key)


def slice_board_input(board_input: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v.index_select(0, idx) if torch.is_tensor(v) else v for k, v in board_input.items()}


class FrozenExpertMoE(nn.Module):
    """Five resident 99M experts + router. `freeze=True` is inference / router-only."""

    def __init__(self, router: nn.Module, experts: list[nn.Module], *, freeze: bool = True):
        super().__init__()
        if len(experts) != N_EXPERTS:
            raise ValueError(f"need {N_EXPERTS} experts, got {len(experts)}")
        self.router = router
        packed = [freeze_expert(e) for e in experts] if freeze else list(experts)
        self.experts = nn.ModuleList(packed)
        if not freeze:
            self.unfreeze()

    def unfreeze(self) -> "FrozenExpertMoE":
        self.train()
        for p in self.parameters():
            p.requires_grad_(True)
        return self

    def encode_and_route(self, board_input: dict[str, torch.Tensor]):
        """One 99M stem encode (expert 0), then the router head."""
        stem_out = self.experts[0](board_input)
        logits = self.router(stem_out["global_hidden"])
        return logits, logits.argmax(dim=-1), stem_out

    def route_ids(self, board_input: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits, ids, _ = self.encode_and_route(board_input)
        return logits, ids

    def forward(self, board_input: dict[str, torch.Tensor]) -> RoutedForward:
        gate, ids, stem_out = self.encode_and_route(board_input)
        policy = stem_out["policy_logits"].clone()
        value = stem_out["value_logits"].clone()
        for e in range(1, N_EXPERTS):
            idx = (ids == e).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            out = self.experts[e](slice_board_input(board_input, idx))
            policy.index_copy_(0, idx, out["policy_logits"])
            value.index_copy_(0, idx, out["value_logits"])
        names = {int(i) for i in ids.tolist()}
        name = EXPERT_NAMES[next(iter(names))] if len(names) == 1 else "mixed"
        return RoutedForward(
            policy_logits=policy,
            value_logits=value,
            gate_logits=gate,
            expert_id=ids,
            expert_name=name,
            stem_policy_logits=stem_out["policy_logits"],
            stem_value_logits=stem_out["value_logits"],
        )


def freeze_expert(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_router() -> ExpertRouter:
    return ExpertRouter()


def is_moe_router_ckpt(ckpt: dict) -> bool:
    if not isinstance(ckpt, dict):
        return False
    if ckpt.get("arch") in ("frozen_moe_router", "chess_moe_full"):
        return True
    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict):
        return False
    return "experts" in ckpt and any(str(k).startswith("mlp.") for k in state)


def load_moe_pipeline(
    router_ckpt: str | Path,
    device: torch.device | str | None = None,
    *,
    expert_root: str | Path | None = None,
    ckpt: dict | None = None,
) -> FrozenExpertMoE:
    """Load the frozen 5-expert bank plus the trained router head."""
    from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64

    path = Path(router_ckpt)
    if ckpt is None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not is_moe_router_ckpt(ckpt):
        raise ValueError(f"not a MoE router checkpoint: {path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    router = build_router()
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    router.load_state_dict(state)
    root = Path(expert_root) if expert_root else Path(__file__).resolve().parent / "outputs" / "hf_models" / "experts"
    experts: list[nn.Module] = []
    saved = ckpt.get("expert_state_dicts")
    for i, (name, repo) in enumerate(EXPERTS):
        model = build_squares64(DEFAULT_100M_SQUARES64_CONFIG)
        if saved and i < len(saved) and saved[i] is not None:
            model.load_state_dict(saved[i], strict=False)
        else:
            ck = root / name / "latest.pt"
            if not ck.exists():
                raise FileNotFoundError(f"missing expert {name} at {ck} (hf://{repo})")
            load_expert_weights(model, str(ck))
        experts.append(model.to(device))
    moe = FrozenExpertMoE(router.to(device), experts, freeze=True)
    moe.eval()
    return moe


def router_param_count() -> int:
    return count_parameters(build_router())


def load_expert_weights(model: nn.Module, ckpt: dict | str) -> nn.Module:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent
    if str(root / "scripts") not in sys.path:
        sys.path.insert(0, str(root / "scripts"))
    from autoresearch_8gb.pipeline import load_model_state

    if isinstance(ckpt, str):
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    else:
        blob = ckpt
    state = load_model_state(blob if isinstance(blob, dict) else {"model_state_dict": blob})
    model.load_state_dict(state, strict=False)
    return freeze_expert(model)


def best_expert_from_ce(ce: torch.Tensor) -> torch.Tensor:
    """ce: (B, E) → (B,) argmin."""
    return ce.argmin(dim=-1)


def robust_best_expert(ce: torch.Tensor, puzzle_margin: float = 0.5) -> torch.Tensor:
    """Argmin, but puzzle only if it beats the runner-up by `puzzle_margin`.

    Puzzle CE is ~0.23 when it wins and ~4+ when it doesn't. Thin argmin
    wins teach the router to guess puzzle; a miss is catastrophic.
    """
    best = ce.argmin(dim=-1)
    filled = ce.clone()
    filled.scatter_(1, best.unsqueeze(1), float(ce.max()) + 1e6)
    second = filled.argmin(dim=-1)
    puzzle_id = EXPERT_NAMES.index("puzzle")
    is_puz = best == puzzle_id
    margin = ce.gather(1, second.unsqueeze(1)).squeeze(1) - ce.gather(1, best.unsqueeze(1)).squeeze(1)
    return torch.where(is_puz & (margin < puzzle_margin), second, best)


def soft_expert_targets(ce: torch.Tensor, tau: float = 0.4) -> torch.Tensor:
    """Lower CE → higher probability. (B, E)."""
    return F.softmax(-ce / max(float(tau), 1e-3), dim=-1)


def soft_target_loss(gate_logits: torch.Tensor, expert_ce: torch.Tensor, tau: float = 0.4) -> torch.Tensor:
    """CE(softmax(router) || softmax(-expert_ce / tau))."""
    targets = soft_expert_targets(expert_ce, tau)
    return -(targets * F.log_softmax(gate_logits, dim=-1)).sum(dim=-1).mean()


def expected_ce_loss(gate_logits: torch.Tensor, expert_ce: torch.Tensor) -> torch.Tensor:
    """E_gate[CE] = softmax(router) · expert_ce. Experts frozen; only gates train."""
    gates = gate_logits.softmax(dim=-1)
    return (gates * expert_ce).sum(dim=-1).mean()
