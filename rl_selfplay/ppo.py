"""On-policy PPO for recurrent chess policies, with outcome GAE and reference KL.

An actor transition spans its move and the opponent's reply. Rewards and GAE
stay in the actor's frame; do not negate between those two plies. The WDL head
is White-absolute, so values and bootstraps are flipped when the actor is Black.
Dropout stays disabled for collection AND gradient updates. The recorded loop
count, temperature and legal support define each behavior-policy probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from collections import Counter

import chess
import torch
from torch import nn
from torch.nn import functional as F

from chess_squares64 import Squares64RecurrentTransformer
from move_vocab import VOCAB_SIZE, index_to_move, legal_move_mask
from rl_selfplay.searchfree import encode_boards


@dataclass
class PPOConfig:
    games_per_iteration: int = 32
    rollout_batch_size: int = 8
    ply_cap: int = 400
    depths: tuple[int, ...] = (3,)
    temperature: float = 0.8
    opponent_temperature: float = 0.8
    epochs: int = 2
    minibatch_size: int = 32
    microbatch_size: int = 8
    learning_rate: float = 2e-6
    clip_ratio: float = 0.15
    value_clip: float = 0.2
    value_weight: float = 0.5
    reference_kl_weight: float = 0.02
    entropy_weight: float = 0.001
    target_kl: float = 0.02
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95
    replay_weight: float = 0.25
    replay_batch_size: int = 8
    weight_decay: float = 0.01

    def validate(self):
        for name in ("games_per_iteration", "rollout_batch_size", "ply_cap", "epochs",
                     "minibatch_size", "microbatch_size", "replay_batch_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.games_per_iteration % 2 or self.rollout_batch_size % 2:
            raise ValueError("Game counts and rollout batches must be even for paired colors")
        if not self.depths or any(type(d) is not int or d < 1 for d in self.depths):
            raise ValueError("depths must contain positive integers")
        for name in ("temperature", "learning_rate", "target_kl", "max_grad_norm"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.opponent_temperature < 0:
            raise ValueError("opponent_temperature must be nonnegative")
        if not 0 < self.clip_ratio < 1 or not 0 <= self.value_clip <= 2:
            raise ValueError("Invalid PPO clipping")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid GAE parameters")
        for name in ("value_weight", "reference_kl_weight", "entropy_weight", "replay_weight", "weight_decay"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


def check_model(model):
    if not isinstance(model, Squares64RecurrentTransformer) or model.config.n_value_classes != 3:
        raise ValueError("PPO requires a squares64 recurrent model with WDL head")
    n = sum(p.numel() for p in model.parameters())
    if n >= 1_000_000_000 or VOCAB_SIZE != 1968:
        raise ValueError("Require fewer than 1B parameters and compact move vocabulary")
    return n


def masked_log_probs(logits, mask, temperature=1.0):
    if temperature <= 0 or not mask.any(-1).all():
        raise ValueError("Positive temperature and nonempty legal support required")
    if not torch.isfinite(logits[mask]).all():
        raise FloatingPointError("Nonfinite legal policy logits")
    return F.log_softmax((logits.float() / temperature).masked_fill(~mask, torch.finfo(torch.float32).min), dim=-1)


def forward_kl(logp, reference_logp):
    """KL(current || fixed reference), over the SAME legal move support."""
    return (logp.exp() * (logp - reference_logp)).sum(-1)


def wdl_value(logits):
    """White-absolute value: P(White wins) - P(White loses)."""
    p = logits.float().softmax(-1)
    return p[:, 0] - p[:, 2]


def actor_value(white_value, actor_white):
    """Put a White-absolute WDL value into the actor's win/loss frame."""
    if torch.is_tensor(actor_white):
        value = white_value if torch.is_tensor(white_value) else torch.as_tensor(white_value)
        sign = actor_white.to(device=value.device, dtype=value.dtype) * 2 - 1
        return value * sign
    return white_value if actor_white else -white_value


def gae(values, terminal_reward, bootstrap, gamma=1., lam=0.95):
    """terminal_reward=None denotes a truncation, which must bootstrap."""
    advantages = [0.] * len(values)
    advantage = 0.
    next_value = 0. if terminal_reward is not None else bootstrap
    for i in reversed(range(len(values))):
        reward = terminal_reward if i == len(values) - 1 and terminal_reward is not None else 0.
        delta = reward + gamma * next_value - values[i]
        advantage = delta + gamma * lam * advantage
        advantages[i] = advantage
        next_value = values[i]
    return advantages, [a + v for a, v in zip(advantages, values)]


def stack_inputs(rows, device):
    return {k: torch.stack([r["input"][k] for r in rows]).to(device) for k in rows[0]["input"]}


@torch.no_grad()
def decisions(model, boards, device, depth, temperature, rng):
    model.eval()
    inputs = encode_boards(model, boards, device)
    out = model(inputs, recurrent_unrolls=depth)
    masks = torch.stack([legal_move_mask(b) for b in boards]).to(device)
    logp = masked_log_probs(out["policy_logits"], masks, temperature if temperature > 0 else 1.)
    # CPU generator makes sampling independent of accelerator RNG state.
    actions = (logp.argmax(-1).cpu() if temperature == 0 else
               torch.multinomial(logp.exp().cpu(), 1, generator=rng).flatten())
    values = wdl_value(out["value_logits"])
    records = []
    for i, action in enumerate(actions.tolist()):
        records.append(dict(input={k: v[i].detach().cpu() for k, v in inputs.items()},
                            mask=masks[i].cpu(), action=action, depth=depth,
                            old_logp=float(logp[i, action]), old_value=float(values[i])))
    return records


def terminal_reward(board, actor_white):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.
    return 1. if outcome.winner == actor_white else -1.


def collect_rollouts(model, opponent, cfg, openings, device, rng, py_rng, log=print):
    """One frozen neural opponent per collection; model is fixed throughout.

    At a cap on the opponent's turn, allow its reply before bootstrapping at
    the actor's next decision. Thus a game can exceed ply_cap by one ply.
    Every opening and depth is paired across colors. Keep complete histories
    in chess.Board during play; store already-encoded inputs for learning.
    """
    cfg.validate()
    check_model(model)
    model.eval()
    opponent.eval()
    if not openings:
        raise ValueError("Need training openings")
    rows, games = [], []
    for offset in range(0, cfg.games_per_iteration, cfg.rollout_batch_size):
        size = min(cfg.rollout_batch_size, cfg.games_per_iteration - offset)
        boards, depths, starts = [], [], []
        for _ in range(size // 2):
            opening = py_rng.choice(openings)
            board = chess.Board()
            for uci in opening:
                board.push_uci(uci)  # Illegal openings fail loudly.
            if board.is_game_over(claim_draw=True) or len(board.move_stack) >= cfg.ply_cap:
                raise ValueError("Training opening is terminal or beyond ply cap")
            depth = py_rng.choice(cfg.depths)
            boards.extend([board.copy(stack=True), board.copy(stack=True)])
            depths.extend([depth, depth])
            starts.extend([list(opening), list(opening)])
        trajectories = [[] for _ in boards]
        live = set(range(size))
        while live:
            # Finalize real outcomes, or bootstrap truncated actor decisions.
            for i in sorted(live):
                board = boards[i]
                actor_white = i % 2 == 0
                reward = terminal_reward(board, actor_white)
                truncated = reward is None and len(board.move_stack) >= cfg.ply_cap and board.turn == actor_white
                if reward is None and not truncated:
                    continue
                bootstrap = 0.
                if truncated and trajectories[i]:
                    with torch.no_grad():
                        inputs = encode_boards(model, [board], device)
                        white_v = float(wdl_value(model(inputs, recurrent_unrolls=depths[i])["value_logits"])[0])
                    bootstrap = actor_value(white_v, actor_white)
                adv, ret = gae([r["old_value"] for r in trajectories[i]], reward, bootstrap,
                               cfg.gamma, cfg.gae_lambda)
                for row, a, target in zip(trajectories[i], adv, ret):
                    row.update(advantage=a, return_target=target)
                    rows.append(row)
                outcome = board.outcome(claim_draw=True)
                games.append(dict(game_id=offset + i, opening=starts[i], actor_white=actor_white,
                                  depth=depths[i], reward=reward, bootstrap=bootstrap,
                                  termination=outcome.termination.name if outcome else "TRUNCATED",
                                  plies=len(board.move_stack), moves=[m.uci() for m in board.move_stack],
                                  final_fen=board.fen()))
                live.remove(i)
            actor_ids = [i for i in sorted(live) if boards[i].turn == (i % 2 == 0)]
            opponent_ids = [i for i in sorted(live) if i not in actor_ids]
            for depth in sorted(set(depths[i] for i in actor_ids)):
                ids = [i for i in actor_ids if depths[i] == depth]
                sampled = decisions(model, [boards[i] for i in ids], device, depth, cfg.temperature, rng)
                for i, row in zip(ids, sampled):
                    actor_white = i % 2 == 0
                    row["old_value"] = actor_value(row["old_value"], actor_white)
                    row["actor_white"] = actor_white
                    trajectories[i].append(row)
                    boards[i].push(index_to_move(row["action"]))
            if opponent_ids:
                sampled = decisions(opponent, [boards[i] for i in opponent_ids], device,
                                    opponent.config.recurrent_unrolls, cfg.opponent_temperature, rng)
                for i, row in zip(opponent_ids, sampled):
                    boards[i].push(index_to_move(row["action"]))
        log(f"collected {offset + size}/{cfg.games_per_iteration} games; {len(rows)} actor decisions")
    if not rows:
        raise ValueError("Collection produced no actor decisions")
    return rows, games


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def clipped_policy_loss(new_logp, old_logp, advantages, epsilon):
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    loss = -torch.minimum(ratio * advantages, ratio.clamp(1 - epsilon, 1 + epsilon) * advantages).mean()
    approx_kl = ((ratio - 1) - log_ratio).mean()
    return loss, approx_kl, ((ratio - 1).abs() > epsilon).float().mean()


def grouped_microbatches(indices, rows, size):
    for depth in sorted({rows[i]["depth"] for i in indices}):
        group = [i for i in indices if rows[i]["depth"] == depth]
        for start in range(0, len(group), size):
            yield group[start:start + size], depth


def replay_loss(model, cache, cfg, device, rng):
    # Import lazily: this is a supervised auxiliary, never off-policy PPO data.
    from scripts.autoresearch_8gb.pipeline import prepare_soft_batch, soft_policy_loss
    ids = torch.randint(len(cache["turn"]), (cfg.replay_batch_size,), generator=rng)
    inputs, hard, _, si, sp = prepare_soft_batch(cache, ids, device)
    logits = model(inputs, recurrent_unrolls=model.config.recurrent_unrolls)["policy_logits"]
    return .45 * F.cross_entropy(logits, hard) + .55 * soft_policy_loss(logits, si, sp)


def ppo_update(model, reference, optimizer, rows, cfg, device, rng, replay=None):
    """Use each fresh rollout for at most cfg.epochs, then discard it.

    Eval mode intentionally leaves gradients enabled while disabling dropout.
    Standard gradients sum across unrolls; no depth-dependent rescaling is applied
    to a mixed-depth PPO minibatch. One global clip follows accumulation.
    """
    cfg.validate()
    if not rows:
        raise ValueError("PPO needs fresh transitions")
    if cfg.replay_weight and replay is None:
        raise ValueError("Replay weight > 0 requires a supervised cache")
    model.eval()
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    advantages = torch.tensor([r["advantage"] for r in rows])
    # Preserve negative-only/positive-only constant batches (centering erases them).
    std = advantages.std(unbiased=False)
    if std > 1e-8:
        advantages = (advantages - advantages.mean()) / (std + 1e-8)
    logs = []
    stopped = False
    for epoch in range(cfg.epochs):
        permutation = torch.randperm(len(rows), generator=rng).tolist()
        for start in range(0, len(rows), cfg.minibatch_size):
            ids = permutation[start:start + cfg.minibatch_size]
            optimizer.zero_grad(set_to_none=True)
            stats = Counter()
            for group, depth in grouped_microbatches(ids, rows, cfg.microbatch_size):
                batch = [rows[i] for i in group]
                inputs = stack_inputs(batch, device)
                mask = torch.stack([r["mask"] for r in batch]).to(device)
                out = model(inputs, recurrent_unrolls=depth)
                logp = masked_log_probs(out["policy_logits"], mask, cfg.temperature)
                actions = torch.tensor([r["action"] for r in batch], device=device)
                taken = logp.gather(1, actions[:, None]).squeeze(1)
                old_logp = torch.tensor([r["old_logp"] for r in batch], device=device)
                policy, kl, clipfrac = clipped_policy_loss(taken, old_logp, advantages[group].to(device), cfg.clip_ratio)
                if not torch.isfinite(kl) or kl > 1.5 * cfg.target_kl:
                    stopped = True
                    break  # Discard this entire minibatch's accumulated gradients.
                value = actor_value(wdl_value(out["value_logits"]),
                                   torch.tensor([bool(r.get("actor_white", True)) for r in batch], device=device))
                target = torch.tensor([r["return_target"] for r in batch], device=device)
                old_value = torch.tensor([r["old_value"] for r in batch], device=device)
                clipped = old_value + (value - old_value).clamp(-cfg.value_clip, cfg.value_clip)
                vloss = .5 * torch.maximum((value - target).square(), (clipped - target).square()).mean()
                with torch.no_grad():
                    ref = reference(inputs, recurrent_unrolls=depth)
                    ref_logp = masked_log_probs(ref["policy_logits"], mask, cfg.temperature)
                anchor = forward_kl(logp, ref_logp).mean()
                entropy = -(logp.exp() * logp).sum(-1).mean()
                loss = policy + cfg.value_weight * vloss + cfg.reference_kl_weight * anchor - cfg.entropy_weight * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO objective")
                weight = len(group) / len(ids)
                (weight * loss).backward()
                for key, val in dict(policy=policy, value=vloss, reference_kl=anchor,
                                     entropy=entropy, behavior_kl=kl, clip_fraction=clipfrac).items():
                    stats[key] += weight * float(val.detach())
            if stopped:
                optimizer.zero_grad(set_to_none=True)
                break
            if cfg.replay_weight:
                supervised = replay_loss(model, replay, cfg, device, rng)
                (cfg.replay_weight * supervised).backward()
                stats["replay_loss"] = float(supervised.detach())
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            logs.append(dict(stats, epoch=epoch, grad_norm=float(norm), decisions=len(ids)))
        if stopped:
            break
    return dict(updates=len(logs), early_stop_kl=stopped, minibatches=logs,
                rollout_decisions=len(rows), advantage_mean=float(advantages.mean()))
