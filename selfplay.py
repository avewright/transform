"""Self-play evolutionary training via competitive perturbation selection.

The model plays itself: two differently-perturbed copies of the base model
play chess against each other. The perturbation whose side wins (or plays
better) is kept; the loser is discarded. The winner becomes the new
baseline, and the process repeats.

This is essentially (μ+λ)-ES with game outcomes as the fitness signal,
combined with the RandOpt insight that good perturbations are dense
around pretrained weights.

Loop:
    1. Current champion weights = θ
    2. Sample two Gaussian perturbations: θ_A = θ + ε_A, θ_B = θ + ε_B
    3. Play N games: θ_A as White vs θ_B as Black, then swap colors
    4. Score: wins=1, draws=0.5, losses=0
    5. The perturbation with the higher score becomes the new champion
    6. Commit winner's noise into the base weights → θ' = θ + ε_winner
    7. Repeat from θ'
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config, RandOptConfig, SelfPlayConfig
from constrained import build_move_trie, build_token_text_map, make_legal_move_processor
from data import fen_to_prompt
from randopt import (
    apply_perturbation,
    get_perturbable_params,
    remove_perturbation,
    sample_noise,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class GameResult:
    """Result of a single self-play game."""
    white_name: str
    black_name: str
    outcome: str            # "1-0", "0-1", "1/2-1/2"
    termination: str        # "checkmate", "stalemate", "draw_*", "max_moves", "illegal"
    num_moves: int
    pgn_moves: list[str]    # SAN moves for the game
    white_illegal: int = 0  # count of illegal move attempts by white
    black_illegal: int = 0  # count of illegal move attempts by black


@dataclass
class Candidate:
    """A candidate in the evolutionary population."""
    name: str
    noise: dict[str, torch.Tensor] | None  # None = base model (no perturbation)
    score: float = 0.0
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    illegal_moves: int = 0


# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_move(
    model: nn.Module,
    tokenizer,
    board: chess.Board,
    temperature: float = 0.8,
    max_retries: int = 5,
    constrained: bool = True,
    token_texts: dict[int, str] | None = None,
    board_encoding: str = "fen",
) -> tuple[chess.Move | None, int]:
    """Generate a chess move from the model for the current board position.

    If constrained=True (default), uses trie-based logit masking so the model
    can ONLY produce tokens that form a legal UCI move. Illegal moves become
    impossible at the token level — no retries needed.

    If constrained=False, falls back to retry + random legal move on failure.

    Returns (move, num_retries). With constrained=True, retries is always 0.
    """
    fen = board.fen()
    prompt = fen_to_prompt(fen, encoding=board_encoding)

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    # ── Constrained decoding path ──────────────────────────────
    if constrained:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0

        processor = make_legal_move_processor(board, tokenizer, prompt_length, token_texts=token_texts)

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=6,  # UCI moves are 4-5 chars, ~2-3 tokens
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_k=50,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            logits_processor=processor,
        )

        new_ids = gen_ids[:, prompt_length:]
        text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        uci_str = text.replace(" ", "").lower()[:5].rstrip()

        # The trie guarantees this is a legal move, but validate anyway
        for length in [5, 4]:
            candidate = uci_str[:length]
            try:
                move = chess.Move.from_uci(candidate)
                if move in board.legal_moves:
                    return move, 0
            except (ValueError, chess.InvalidMoveError):
                pass

        # Trie constraint failed (edge case) — fall back to random
        return random.choice(legal_moves), 0

    # ── Unconstrained path (retry + fallback) ──────────────────
    retries = 0
    for attempt in range(max_retries):
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_k=50,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        new_ids = gen_ids[:, prompt_length:]
        text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

        uci_str = text.replace(" ", "").lower()[:5].rstrip()
        for length in [5, 4]:
            candidate = uci_str[:length]
            try:
                move = chess.Move.from_uci(candidate)
                if move in board.legal_moves:
                    return move, retries
            except (ValueError, chess.InvalidMoveError):
                pass

        retries += 1

    # All retries exhausted — fall back to random legal move
    legal = list(board.legal_moves)
    if legal:
        return random.choice(legal), retries
    return None, retries


# ---------------------------------------------------------------------------
# Self-play game
# ---------------------------------------------------------------------------

def play_game(
    model: nn.Module,
    tokenizer,
    white_noise: dict[str, torch.Tensor] | None,
    black_noise: dict[str, torch.Tensor] | None,
    white_name: str = "white",
    black_name: str = "black",
    max_moves: int = 150,
    temperature: float = 0.8,
    max_retries: int = 5,
    constrained: bool = True,
    token_texts: dict[int, str] | None = None,
    board_encoding: str = "fen",
    adjudicate_material: int = 0,
) -> GameResult:
    """Play a single game between two perturbations of the same model.

    Since we only have one model in memory, we swap perturbations in and out
    on each move. This is slower but uses minimal memory.
    """
    board = chess.Board()
    pgn_moves: list[str] = []
    white_illegal = 0
    black_illegal = 0

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        is_white_turn = board.turn == chess.WHITE
        noise = white_noise if is_white_turn else black_noise

        # Apply this side's perturbation
        if noise is not None:
            apply_perturbation(model, noise)

        move, retries = generate_move(
            model, tokenizer, board, temperature, max_retries,
            constrained=constrained,
            token_texts=token_texts,
            board_encoding=board_encoding,
        )

        # Remove perturbation immediately
        if noise is not None:
            remove_perturbation(model, noise)

        if is_white_turn:
            white_illegal += retries
        else:
            black_illegal += retries

        if move is None:
            # No legal moves — shouldn't happen if board isn't game over
            break

        pgn_moves.append(board.san(move))
        board.push(move)

    # Determine outcome
    outcome, termination = _classify_game_end(
        board, move_num >= max_moves - 1, adjudicate_material=adjudicate_material,
    )

    return GameResult(
        white_name=white_name,
        black_name=black_name,
        outcome=outcome,
        termination=termination,
        num_moves=len(pgn_moves),
        pgn_moves=pgn_moves,
        white_illegal=white_illegal,
        black_illegal=black_illegal,
    )


_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def _material_balance(board: chess.Board) -> int:
    """Return material advantage for White (positive) or Black (negative)."""
    bal = 0
    for sq, piece in board.piece_map().items():
        val = _PIECE_VALUES.get(piece.piece_type, 0)
        bal += val if piece.color == chess.WHITE else -val
    return bal


def _classify_game_end(
    board: chess.Board, hit_max_moves: bool, adjudicate_material: int = 0,
) -> tuple[str, str]:
    """Classify how and why the game ended.

    If adjudicate_material > 0 and the game hit max_moves, the side with
    a material advantage >= threshold wins instead of drawing.
    """
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return "0-1", "checkmate"
        else:
            return "1-0", "checkmate"
    elif board.is_stalemate():
        return "1/2-1/2", "stalemate"
    elif board.is_insufficient_material():
        return "1/2-1/2", "insufficient_material"
    elif board.can_claim_fifty_moves():
        return "1/2-1/2", "fifty_moves"
    elif board.is_repetition(3):
        return "1/2-1/2", "repetition"
    elif hit_max_moves:
        if adjudicate_material > 0:
            bal = _material_balance(board)
            if bal >= adjudicate_material:
                return "1-0", "adjudication"
            elif bal <= -adjudicate_material:
                return "0-1", "adjudication"
        return "1/2-1/2", "max_moves"
    else:
        return "1/2-1/2", "unknown"


# ---------------------------------------------------------------------------
# Match: multiple games between two candidates
# ---------------------------------------------------------------------------

def play_match(
    model: nn.Module,
    tokenizer,
    candidate_a: Candidate,
    candidate_b: Candidate,
    games: int = 10,
    max_moves: int = 150,
    temperature: float = 0.8,
    max_retries: int = 5,
    log_games: bool = False,
    constrained: bool = True,
    token_texts: dict[int, str] | None = None,
    board_encoding: str = "fen",
    adjudicate_material: int = 0,
) -> tuple[float, float, list[GameResult]]:
    """Play a match of N games between two candidates, alternating colors.

    Returns (score_a, score_b, game_results).
    Scores: win=1, draw=0.5, loss=0.
    """
    score_a = 0.0
    score_b = 0.0
    results = []

    for g in range(games):
        # Alternate who plays white
        if g % 2 == 0:
            w_noise, b_noise = candidate_a.noise, candidate_b.noise
            w_name, b_name = candidate_a.name, candidate_b.name
        else:
            w_noise, b_noise = candidate_b.noise, candidate_a.noise
            w_name, b_name = candidate_b.name, candidate_a.name

        result = play_game(
            model, tokenizer,
            white_noise=w_noise,
            black_noise=b_noise,
            white_name=w_name,
            black_name=b_name,
            max_moves=max_moves,
            temperature=temperature,
            max_retries=max_retries,
            constrained=constrained,
            token_texts=token_texts,
            board_encoding=board_encoding,
            adjudicate_material=adjudicate_material,
        )
        results.append(result)

        # Accumulate scores
        if result.outcome == "1-0":
            # White won
            if g % 2 == 0:
                score_a += 1.0
            else:
                score_b += 1.0
        elif result.outcome == "0-1":
            # Black won
            if g % 2 == 0:
                score_b += 1.0
            else:
                score_a += 1.0
        else:
            score_a += 0.5
            score_b += 0.5

        # Track illegal moves for the candidates themselves
        if g % 2 == 0:
            candidate_a.illegal_moves += result.white_illegal
            candidate_b.illegal_moves += result.black_illegal
        else:
            candidate_b.illegal_moves += result.white_illegal
            candidate_a.illegal_moves += result.black_illegal

        if log_games:
            outcome_str = result.outcome
            term_str = result.termination
            moves_str = " ".join(
                f"{i//2+1}.{' ' if i%2==0 else '...'}{m}"
                for i, m in enumerate(result.pgn_moves)
            )
            print(f"  Game {g+1}: {w_name} vs {b_name} → {outcome_str} ({term_str}, {result.num_moves} moves)")

    return score_a, score_b, results


# ---------------------------------------------------------------------------
# Evolutionary self-play loop
# ---------------------------------------------------------------------------

def selfplay_evolve(
    model: nn.Module,
    tokenizer,
    cfg: SelfPlayConfig,
    randopt_cfg: RandOptConfig,
    output_dir: str | Path = "outputs/selfplay",
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Run the self-play evolutionary loop.

    Each generation:
      1. From current model weights θ, sample `population_size` perturbations
      2. Run a round-robin tournament among the candidates
      3. The top `elite_keep` candidates survive, their noise is committed
         into θ (the base weights are permanently shifted)
      4. Noise std optionally decays each generation (annealing)

    Args:
        model: The base pretrained model (weights will be modified in-place).
        tokenizer: The model's tokenizer.
        cfg: Self-play config.
        randopt_cfg: Which params to perturb (uses perturb/skip patterns).
        output_dir: Where to save checkpoints.
        device: Device.

    Returns:
        The evolved model (same object, weights updated in-place).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device) if isinstance(device, str) else device
    dtype = next(model.parameters()).dtype

    # Identify perturbable parameters
    perturbable = get_perturbable_params(model, cfg.perturb_patterns, cfg.skip_patterns)
    param_shapes = {name: p.shape for name, p in perturbable.items()}
    num_params = sum(p.numel() for p in perturbable.values())
    print(f"Self-play evolution: {len(perturbable)} tensors, {num_params:,} perturbable params")

    # Pre-build token map for constrained decoding (expensive, do once)
    token_texts = None
    if cfg.constrained_decoding:
        print("  Building token-text map for constrained decoding...")
        token_texts = build_token_text_map(tokenizer)
        print(f"  Token map: {len(token_texts)} usable tokens")

    # RNG
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    noise_std = cfg.noise_std
    history: list[dict] = []
    cumulative_noise: dict[str, torch.Tensor] = {}  # track total noise applied

    for gen in range(cfg.generations):
        gen_start = time.time()
        print(f"\n{'='*60}")
        print(f" Generation {gen+1}/{cfg.generations}  |  σ={noise_std:.6f}")
        print(f"{'='*60}")

        # --- 1. Create population ---
        candidates: list[Candidate] = []

        # Elite: the current model (no additional noise)
        candidates.append(Candidate(name="elite", noise=None))

        # Challengers: random perturbations
        for i in range(cfg.population_size - 1):
            noise = sample_noise(param_shapes, noise_std, device, dtype, generator)
            candidates.append(Candidate(name=f"challenger_{i}", noise=noise))

        # --- 2. Round-robin tournament ---
        print(f"  Tournament: {len(candidates)} candidates, "
              f"{cfg.games_per_matchup} games/matchup")

        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                score_i, score_j, games = play_match(
                    model, tokenizer,
                    candidates[i], candidates[j],
                    games=cfg.games_per_matchup,
                    max_moves=cfg.max_moves,
                    temperature=cfg.temperature,
                    max_retries=cfg.max_retries,
                    log_games=cfg.log_games,
                    constrained=cfg.constrained_decoding,
                    token_texts=token_texts,
                    board_encoding=cfg.board_encoding,
                    adjudicate_material=cfg.adjudicate_material,
                )
                candidates[i].score += score_i
                candidates[j].score += score_j
                candidates[i].games_played += cfg.games_per_matchup
                candidates[j].games_played += cfg.games_per_matchup

                # Tally W/D/L
                for g_idx, game in enumerate(games):
                    i_is_white = g_idx % 2 == 0
                    if game.outcome == "1-0":
                        if i_is_white:
                            candidates[i].wins += 1
                            candidates[j].losses += 1
                        else:
                            candidates[j].wins += 1
                            candidates[i].losses += 1
                    elif game.outcome == "0-1":
                        if i_is_white:
                            candidates[i].losses += 1
                            candidates[j].wins += 1
                        else:
                            candidates[j].losses += 1
                            candidates[i].wins += 1
                    else:
                        candidates[i].draws += 1
                        candidates[j].draws += 1

        # --- 3. Rank and select ---
        candidates.sort(key=lambda c: c.score, reverse=True)

        print(f"\n  Results:")
        for c in candidates:
            pct = c.score / max(c.games_played, 1) * 100
            print(f"    {c.name:20s}  score={c.score:.1f}/{c.games_played}  "
                  f"({pct:.0f}%)  W/D/L={c.wins}/{c.draws}/{c.losses}  "
                  f"illegal={c.illegal_moves}")

        winner = candidates[0]
        print(f"\n  Winner: {winner.name} (score={winner.score:.1f})")

        # --- 4. Commit winner's noise into base weights ---
        if winner.noise is not None:
            print(f"  Committing {winner.name}'s noise into base weights")
            apply_perturbation(model, winner.noise)

            # Track cumulative noise
            for name, noise in winner.noise.items():
                if name in cumulative_noise:
                    cumulative_noise[name] = cumulative_noise[name] + noise
                else:
                    cumulative_noise[name] = noise.clone()
        else:
            print(f"  Elite held — no weight update this generation")

        # --- 5. Decay noise ---
        noise_std = max(noise_std * cfg.noise_decay, cfg.noise_floor)

        # --- 6. Log ---
        gen_elapsed = time.time() - gen_start
        gen_record = {
            "generation": gen + 1,
            "winner": winner.name,
            "winner_score": winner.score,
            "noise_std": noise_std,
            "elapsed_s": gen_elapsed,
            "standings": [
                {
                    "name": c.name, "score": c.score,
                    "wins": c.wins, "draws": c.draws, "losses": c.losses,
                    "illegal": c.illegal_moves,
                }
                for c in candidates
            ],
        }
        history.append(gen_record)

        # --- 7. Checkpoint ---
        if (gen + 1) % cfg.save_every == 0:
            _save_checkpoint(model, cumulative_noise, history, output_dir, gen + 1)

    # Final save
    _save_checkpoint(model, cumulative_noise, history, output_dir, cfg.generations)

    print(f"\nEvolution complete. {cfg.generations} generations.")
    total_weight_shift = sum(n.norm().item() for n in cumulative_noise.values())
    print(f"Total weight shift (L2 norm of cumulative noise): {total_weight_shift:.4f}")

    return model


def _save_checkpoint(
    model: nn.Module,
    cumulative_noise: dict[str, torch.Tensor],
    history: list[dict],
    output_dir: Path,
    generation: int,
):
    """Save a checkpoint of the evolved model."""
    ckpt_dir = output_dir / f"gen_{generation:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save cumulative noise (so we can replay from base model)
    torch.save(
        {name: t.cpu() for name, t in cumulative_noise.items()},
        ckpt_dir / "cumulative_noise.pt",
    )

    # Save full model weights
    torch.save(model.state_dict(), ckpt_dir / "model_state_dict.pt")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Checkpoint saved: {ckpt_dir}")


# ---------------------------------------------------------------------------
# Quick 1v1 mode — simplest version (no population, just A vs B)
# ---------------------------------------------------------------------------

def selfplay_1v1(
    model: nn.Module,
    tokenizer,
    cfg: SelfPlayConfig,
    output_dir: str | Path = "outputs/selfplay",
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Simplified self-play: each generation is a 1v1 between the current
    champion (base weights) and a single challenger (base + noise).

    Faster per generation than tournament mode, more generations needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device) if isinstance(device, str) else device
    dtype = next(model.parameters()).dtype

    perturbable = get_perturbable_params(model, cfg.perturb_patterns, cfg.skip_patterns)
    param_shapes = {name: p.shape for name, p in perturbable.items()}

    # Pre-build token map for constrained decoding (expensive, do once)
    token_texts = None
    if cfg.constrained_decoding:
        print("  Building token-text map for constrained decoding...")
        token_texts = build_token_text_map(tokenizer)
        print(f"  Token map: {len(token_texts)} usable tokens")

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    noise_std = cfg.noise_std
    history = []
    cumulative_noise: dict[str, torch.Tensor] = {}
    champion_wins_streak = 0

    for gen in range(cfg.generations):
        gen_start = time.time()
        n_challengers = cfg.challengers_per_gen

        best_noise = None
        best_margin = 0.0  # score_chall - score_champ; must be > 0 to beat champion
        best_label = "champion"
        best_champ_cand = None
        best_chall_cand = None

        for c_idx in range(n_challengers):
            noise = sample_noise(param_shapes, noise_std, device, dtype, generator)

            champion = Candidate(name="champion", noise=None)
            chall_name = f"challenger_{c_idx}" if n_challengers > 1 else "challenger"
            challenger = Candidate(name=chall_name, noise=noise)

            score_champ, score_chall, games = play_match(
                model, tokenizer, champion, challenger,
                games=cfg.games_per_matchup,
                max_moves=cfg.max_moves,
                temperature=cfg.temperature,
                max_retries=cfg.max_retries,
                log_games=cfg.log_games,
                constrained=cfg.constrained_decoding,
                token_texts=token_texts,
                board_encoding=cfg.board_encoding,
                adjudicate_material=cfg.adjudicate_material,
            )

            # Count outcomes for display
            for g_idx, game in enumerate(games):
                champ_is_white = g_idx % 2 == 0
                if game.outcome == "1-0":
                    if champ_is_white:
                        champion.wins += 1; challenger.losses += 1
                    else:
                        challenger.wins += 1; champion.losses += 1
                elif game.outcome == "0-1":
                    if champ_is_white:
                        champion.losses += 1; challenger.wins += 1
                    else:
                        challenger.losses += 1; champion.wins += 1
                else:
                    champion.draws += 1; challenger.draws += 1

            margin = score_chall - score_champ
            if n_challengers > 1:
                print(f"  {chall_name}: champ={score_champ:.1f} chall={score_chall:.1f} margin={margin:+.1f}")

            if margin > best_margin:
                best_margin = margin
                # Free previous best noise
                if best_noise is not None:
                    del best_noise
                best_noise = noise
                best_label = chall_name
                best_champ_cand = champion
                best_chall_cand = challenger
            else:
                del noise

        elapsed = time.time() - gen_start

        # Use the best challenger's results for display
        if best_champ_cand is None:
            # All challengers tied exactly — create dummy display
            best_champ_cand = Candidate(name="champion", noise=None)
            best_chall_cand = Candidate(name="challenger", noise=None)

        challenger_won = best_margin > 0
        label = f"CHALLENGER ({best_label})" if challenger_won else (
            "TIE/CHAMPION" if best_margin == 0 else "CHAMPION"
        )

        print(
            f"Gen {gen+1:4d} | σ={noise_std:.5f} | "
            f"best_margin={best_margin:+.1f} → {label} | "
            f"W/D/L(ch)={best_champ_cand.wins}/{best_champ_cand.draws}/{best_champ_cand.losses} "
            f"W/D/L(cl)={best_chall_cand.wins}/{best_chall_cand.draws}/{best_chall_cand.losses} | "
            f"tested={n_challengers} | {elapsed:.1f}s"
        )

        if challenger_won and best_noise is not None:
            # Commit best challenger's noise into base weights
            apply_perturbation(model, best_noise)
            for name, n in best_noise.items():
                if name in cumulative_noise:
                    cumulative_noise[name] = cumulative_noise[name] + n
                else:
                    cumulative_noise[name] = n.clone()
            champion_wins_streak = 0
        else:
            champion_wins_streak += 1

        # Clean up
        if best_noise is not None:
            del best_noise
            best_noise = None

        # Anneal noise
        noise_std = max(noise_std * cfg.noise_decay, cfg.noise_floor)

        # If champion keeps winning, briefly increase noise to escape plateau
        if champion_wins_streak >= 10:
            noise_std = min(noise_std * 2.0, cfg.noise_std)
            print(f"  ↑ Noise boosted to {noise_std:.5f} (champion streak={champion_wins_streak})")
            champion_wins_streak = 0

        history.append({
            "generation": gen + 1,
            "winner": best_label if challenger_won else "champion",
            "best_margin": best_margin,
            "challengers_tested": n_challengers,
            "noise_std": noise_std,
            "elapsed_s": elapsed,
        })

        if (gen + 1) % cfg.save_every == 0:
            _save_checkpoint(model, cumulative_noise, history, output_dir, gen + 1)

    _save_checkpoint(model, cumulative_noise, history, output_dir, cfg.generations)

    updates = sum(1 for h in history if h["winner"] == "challenger")
    print(f"\nSelf-play complete: {updates}/{cfg.generations} challenger wins committed")

    return model
