"""Generate self-play / expert-iteration training positions via MCTS."""

from __future__ import annotations

import chess
import chess.engine
import torch

from move_vocab import move_to_index
from opening_book import get_book_move
from rl_selfplay.config import OPENINGS, SelfPlayConfig
from rl_selfplay.utils import game_result, resolve_stockfish, should_adjudicate
from uci_engine import MCTSSearch, SyzygyProbe


def extract_visit_distribution(root, visit_temp: float = 1.0) -> tuple[dict[int, float], float] | None:
    """Return ({move_idx: prob}, root_q) from an MCTS root node."""
    if root is None or not root.children:
        return None
    total_visits = sum(c.visit_count for c in root.children.values())
    if total_visits <= 0:
        return None

    visit_dist: dict[int, float] = {}
    for move, child in root.children.items():
        if child.visit_count > 0:
            visit_dist[move_to_index(move)] = float(child.visit_count)

    if visit_temp != 1.0:
        peak = max(visit_dist.values())
        visit_dist = {
            k: (v / peak) ** (1.0 / visit_temp) for k, v in visit_dist.items()
        }
    total = sum(visit_dist.values())
    visit_dist = {k: v / total for k, v in visit_dist.items()}
    return visit_dist, root.q_value()


def _play_opening(board: chess.Board, opening: list[str]) -> None:
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)


def _mcts_move(mcts: MCTSSearch, board: chess.Board, sims: int) -> tuple[chess.Move, dict | None]:
    tb = mcts.syzygy.get_move(board)
    if tb is not None:
        mcts.new_game()
        return tb, None
    book = get_book_move(board)
    if book is not None:
        mcts.new_game()
        return book, None

    move, _info = mcts.search(board, max_sims=sims)
    record = None
    extracted = extract_visit_distribution(mcts.root, visit_temp=mcts.policy_temp)
    if extracted is not None:
        visit_dist, root_q = extracted
        record = {
            "fen": board.fen(),
            "visit_dist": visit_dist,
            "root_q": root_q,
            "chosen_move": move_to_index(move),
        }
    mcts.new_game()
    return move, record


def play_self_game(
    mcts: MCTSSearch,
    cfg: SelfPlayConfig,
    game_id: int,
    log_fn=print,
) -> tuple[list[dict], float]:
    """Model vs model MCTS self-play. Records all MCTS positions."""
    board = chess.Board()
    _play_opening(board, OPENINGS[game_id % len(OPENINGS)])
    mcts.new_game()
    positions: list[dict] = []

    while len(board.move_stack) < cfg.ply_cap:
        if should_adjudicate(board, len(board.move_stack)):
            break
        if board.is_game_over(claim_draw=True):
            break

        move, record = _mcts_move(mcts, board, cfg.mcts_sims)
        if record is not None:
            record["game_id"] = game_id
            positions.append(record)
        board.push(move)

    result = 0.5
    outcome = board.outcome(claim_draw=True)
    if outcome and outcome.winner is not None:
        result = 1.0 if outcome.winner == chess.WHITE else 0.0
    log_fn(
        f"  self game {game_id + 1}: {len(board.move_stack)} ply, "
        f"{len(positions)} positions, result={result:.1f}"
    )
    return positions, result


def play_sf_game(
    mcts: MCTSSearch,
    cfg: SelfPlayConfig,
    engine: chess.engine.SimpleEngine,
    game_id: int,
    model_color: chess.Color,
    log_fn=print,
) -> tuple[list[dict], float]:
    """Model (MCTS) vs Stockfish. Records model moves only."""
    board = chess.Board()
    _play_opening(board, OPENINGS[game_id % len(OPENINGS)])
    mcts.new_game()
    positions: list[dict] = []

    while len(board.move_stack) < cfg.ply_cap:
        if should_adjudicate(board, len(board.move_stack)):
            break
        if board.is_game_over(claim_draw=True):
            break

        if board.turn == model_color:
            move, record = _mcts_move(mcts, board, cfg.mcts_sims)
            if record is not None:
                record["game_id"] = game_id
                positions.append(record)
            board.push(move)
        else:
            sf_move = engine.play(
                board, chess.engine.Limit(time=cfg.sf_move_time),
            ).move
            if sf_move not in board.legal_moves:
                sf_move = next(iter(board.legal_moves))
            board.push(sf_move)

    result = game_result(board, model_color)
    log_fn(
        f"  sf game {game_id + 1}: {'W' if model_color == chess.WHITE else 'B'} "
        f"{len(board.move_stack)} ply, {len(positions)} pos, result={result:.1f}"
    )
    return positions, result


def play_prior_game(
    mcts: MCTSSearch,
    prior_mcts: MCTSSearch,
    cfg: SelfPlayConfig,
    game_id: int,
    log_fn=print,
) -> tuple[list[dict], float]:
    """Current model vs frozen prior checkpoint. Records current-model moves."""
    board = chess.Board()
    _play_opening(board, OPENINGS[game_id % len(OPENINGS)])
    mcts.new_game()
    prior_mcts.new_game()
    positions: list[dict] = []
    current_is_white = game_id % 2 == 0

    while len(board.move_stack) < cfg.ply_cap:
        if should_adjudicate(board, len(board.move_stack)):
            break
        if board.is_game_over(claim_draw=True):
            break

        is_current = (board.turn == chess.WHITE) == current_is_white
        active = mcts if is_current else prior_mcts
        move, record = _mcts_move(active, board, cfg.mcts_sims)
        if is_current and record is not None:
            record["game_id"] = game_id
            positions.append(record)
        board.push(move)

    result = game_result(board, chess.WHITE if current_is_white else chess.BLACK)
    log_fn(
        f"  prior game {game_id + 1}: {len(board.move_stack)} ply, "
        f"{len(positions)} pos, result={result:.1f}"
    )
    return positions, result


def build_mcts(model, device: torch.device, cfg: SelfPlayConfig) -> MCTSSearch:
    return MCTSSearch(
        model, device, SyzygyProbe(),
        c_puct=cfg.mcts_c_puct,
        batch_size=cfg.mcts_batch_size,
        fpu_reduction=0.25,
        root_noise_alpha=0.3,
        root_noise_frac=cfg.root_noise_frac,
        use_fp16=cfg.use_fp16 and device.type == "cuda",
        policy_temp=cfg.visit_temp,
        use_transpositions=True,
    )


@torch.no_grad()
def generate_positions(
    model,
    device: torch.device,
    cfg: SelfPlayConfig,
    n_games: int | None = None,
    prior_model=None,
    log_fn=print,
) -> tuple[list[dict], list[float]]:
    """Run games and return (positions, per-game results)."""
    n = n_games if n_games is not None else cfg.n_games
    mcts = build_mcts(model, device, cfg)
    all_positions: list[dict] = []
    results: list[float] = []

    if cfg.mode == "sf":
        sf_path = resolve_stockfish()
        engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": cfg.sf_elo,
            "Threads": 1,
        })
        try:
            for i in range(n):
                color = chess.WHITE if i % 2 == 0 else chess.BLACK
                pos, res = play_sf_game(mcts, cfg, engine, i, color, log_fn)
                all_positions.extend(pos)
                results.append(res)
        finally:
            engine.quit()
        return all_positions, results

    if cfg.mode == "prior":
        if prior_model is None:
            raise ValueError("prior mode requires prior_model")
        prior_mcts = build_mcts(prior_model, device, cfg)
        for i in range(n):
            pos, res = play_prior_game(mcts, prior_mcts, cfg, i, log_fn)
            all_positions.extend(pos)
            results.append(res)
        return all_positions, results

    for i in range(n):
        pos, res = play_self_game(mcts, cfg, i, log_fn)
        all_positions.extend(pos)
        results.append(res)
    return all_positions, results
