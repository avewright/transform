"""Search-free self-play: sample the policy, train on the winner's moves."""

from __future__ import annotations

import chess
import torch
import torch.nn.functional as F

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import index_to_move, legal_move_mask, move_to_index
from rl_selfplay.config import OPENINGS, SelfPlayConfig


def encode_boards(model, boards: list[chess.Board], device: torch.device) -> dict[str, torch.Tensor]:
    """Squares64 prepare_batch when present; fused tokens otherwise."""
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "prepare_batch"):
        return encoder.prepare_batch(boards, device)
    batch = batch_boards_to_fused_token_ids(boards, device)
    if getattr(getattr(model, "config", None), "use_history", False):
        from chess_history import batch_history_features

        batch.update(batch_history_features(boards, device))
    return batch


def _record(ply: dict, game_id: int, root_q: float, source: str) -> dict:
    idx = int(ply["chosen_move"])
    return {
        "fen": ply["fen"],
        "chosen_move": idx,
        "visit_dist": {idx: 1.0},
        "root_q": root_q,
        "source": source,
        "game_id": game_id,
        "stm_white": bool(ply["stm_white"]),
    }


def student_score(result: float, student_white: bool) -> float:
    if result == 0.5:
        return 0.5
    white_won = result > 0.5
    return 1.0 if bool(student_white) == white_won else 0.0


def filter_student_win_positions(
    trajectory: list[dict],
    result: float,
    game_id: int,
    student_white: bool,
) -> list[dict]:
    """Keep the student's moves only if the student won. Draws / losses → []."""
    if student_score(result, student_white) != 1.0:
        return []
    return filter_winner_positions(trajectory, result, game_id)


def filter_winner_positions(
    trajectory: list[dict],
    result: float,
    game_id: int,
) -> list[dict]:
    """Keep the winner's (fen, move) pairs. Draws → []."""
    if result == 0.5:
        return []
    white_won = result > 0.5
    return [
        _record(ply, game_id, 1.0, "winner")
        for ply in trajectory
        if bool(ply["stm_white"]) == white_won
    ]


def all_played_positions(
    trajectory: list[dict],
    result: float,
    game_id: int,
) -> list[dict]:
    """Every ply, labeled with STM outcome Q in {-1, 0, +1}."""
    q_white = 0.0 if result == 0.5 else (1.0 if result > 0.5 else -1.0)
    out: list[dict] = []
    for ply in trajectory:
        stm_q = q_white if ply["stm_white"] else -q_white
        source = "winner" if stm_q > 0 else "played"
        out.append(_record(ply, game_id, stm_q, source))
    return out


def _play_opening(board: chess.Board, opening: list[str]) -> None:
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)


def _legal_fallback(board: chess.Board) -> tuple[chess.Move, int]:
    move = next(iter(board.legal_moves))
    return move, move_to_index(move)


@torch.no_grad()
def sample_policy_moves(
    model,
    boards: list[chess.Board],
    device: torch.device,
    temperature: float = 0.7,
) -> list[tuple[chess.Move, int]]:
    """Legal-masked policy sample (T>0) or argmax (T<=0)."""
    if not boards:
        return []
    batch = encode_boards(model, boards, device)
    use_amp = device.type == "cuda"
    with torch.amp.autocast("cuda", enabled=use_amp):
        out = model(batch)
    logits = out["policy_logits"].float()

    picked: list[tuple[chess.Move, int]] = []
    for i, board in enumerate(boards):
        lg = logits[i].clone()
        mask = legal_move_mask(board).to(device)
        if not bool(mask.any()):
            picked.append(_legal_fallback(board))
            continue
        lg = lg.masked_fill(~mask, float("-inf"))
        if temperature <= 0:
            idx = int(lg.argmax().item())
        else:
            probs = F.softmax(lg / temperature, dim=-1)
            if not torch.isfinite(probs).any() or float(probs.sum()) <= 0:
                idx = int(lg.argmax().item())
            else:
                idx = int(torch.multinomial(probs, 1).item())
        move = index_to_move(idx)
        if move not in board.legal_moves:
            move, idx = _legal_fallback(board)
        picked.append((move, idx))
    return picked


def _game_result(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def _alive(board: chess.Board, ply_cap: int) -> bool:
    if len(board.move_stack) >= ply_cap:
        return False
    return not board.is_game_over(claim_draw=True)


@torch.no_grad()
def play_searchfree_games(
    model,
    device: torch.device,
    cfg: SelfPlayConfig,
    game_ids: list[int],
    log_fn=print,
) -> tuple[list[dict], list[float]]:
    """Batched model-vs-model games. No search, book, or tablebase."""
    model.eval()
    boards = [chess.Board() for _ in game_ids]
    for board, gid in zip(boards, game_ids):
        _play_opening(board, OPENINGS[gid % len(OPENINGS)])
    trajectories: list[list[dict]] = [[] for _ in game_ids]
    live = [i for i, b in enumerate(boards) if _alive(b, cfg.ply_cap)]

    while live:
        live_boards = [boards[i] for i in live]
        moves = sample_policy_moves(model, live_boards, device, cfg.sample_temp)
        nxt: list[int] = []
        for i, (move, idx) in zip(live, moves):
            board = boards[i]
            trajectories[i].append({
                "fen": board.fen(),
                "chosen_move": idx,
                "stm_white": board.turn == chess.WHITE,
            })
            board.push(move)
            if _alive(board, cfg.ply_cap):
                nxt.append(i)
        live = nxt

    all_positions: list[dict] = []
    results: list[float] = []
    for i, gid in enumerate(game_ids):
        result = _game_result(boards[i])
        results.append(result)
        if cfg.winner_only:
            kept = filter_winner_positions(trajectories[i], result, gid)
        else:
            kept = all_played_positions(trajectories[i], result, gid)
        all_positions.extend(kept)
        if result > 0.5:
            label = "1-0"
        elif result < 0.5:
            label = "0-1"
        else:
            label = "1/2-1/2"
        log_fn(
            f"  self game {gid + 1}: {len(boards[i].move_stack)} ply, "
            f"{len(kept)} winner-moves, {label}"
        )
    return all_positions, results


@torch.no_grad()
def play_vs_incumbent_games(
    student,
    incumbent,
    device: torch.device,
    cfg: SelfPlayConfig,
    game_ids: list[int],
    log_fn=print,
) -> tuple[list[dict], list[float]]:
    """Student (sample_temp) vs frozen incumbent (incumbent_temp). Paired colors."""
    student.eval()
    incumbent.eval()
    boards = [chess.Board() for _ in game_ids]
    student_white = [gid % 2 == 0 for gid in game_ids]
    for board, gid in zip(boards, game_ids):
        _play_opening(board, OPENINGS[gid % len(OPENINGS)])
    trajectories: list[list[dict]] = [[] for _ in game_ids]
    live = [i for i, b in enumerate(boards) if _alive(b, cfg.ply_cap)]

    while live:
        stu_ids = [i for i in live if (boards[i].turn == chess.WHITE) == student_white[i]]
        inc_ids = [i for i in live if i not in stu_ids]
        moves: dict[int, tuple[chess.Move, int]] = {}
        if stu_ids:
            for i, picked in zip(
                stu_ids,
                sample_policy_moves(student, [boards[i] for i in stu_ids], device, cfg.sample_temp),
            ):
                moves[i] = picked
        if inc_ids:
            for i, picked in zip(
                inc_ids,
                sample_policy_moves(
                    incumbent, [boards[i] for i in inc_ids], device, cfg.incumbent_temp,
                ),
            ):
                moves[i] = picked
        nxt: list[int] = []
        for i in live:
            move, idx = moves[i]
            board = boards[i]
            is_student = (board.turn == chess.WHITE) == student_white[i]
            if is_student:
                trajectories[i].append({
                    "fen": board.fen(),
                    "chosen_move": idx,
                    "stm_white": board.turn == chess.WHITE,
                })
            board.push(move)
            if _alive(board, cfg.ply_cap):
                nxt.append(i)
        live = nxt

    all_positions: list[dict] = []
    scores: list[float] = []
    for i, gid in enumerate(game_ids):
        result = _game_result(boards[i])
        score = student_score(result, student_white[i])
        scores.append(score)
        kept = filter_student_win_positions(
            trajectories[i], result, gid, student_white[i],
        )
        all_positions.extend(kept)
        color = "W" if student_white[i] else "B"
        if score > 0.5:
            label = "student_win"
        elif score < 0.5:
            label = "incumbent_win"
        else:
            label = "draw"
        log_fn(
            f"  vs-inc game {gid + 1}: student={color} {len(boards[i].move_stack)} ply, "
            f"{len(kept)} train-moves, {label}"
        )
    return all_positions, scores


def generate_searchfree(
    model,
    device: torch.device,
    cfg: SelfPlayConfig,
    n_games: int | None = None,
    opponent=None,
    log_fn=print,
) -> tuple[list[dict], list[float]]:
    n = n_games if n_games is not None else cfg.n_games
    batch = max(1, int(cfg.play_batch_size))
    all_positions: list[dict] = []
    results: list[float] = []
    if cfg.vs_incumbent:
        if opponent is None:
            raise ValueError("vs_incumbent requires a frozen opponent")
        log_fn(
            f"  student T={cfg.sample_temp} vs frozen incumbent T={cfg.incumbent_temp}  "
            f"n={n} play_bs={batch}  train=student wins only"
        )
        play = lambda ids: play_vs_incumbent_games(
            model, opponent, device, cfg, ids, log_fn=log_fn,
        )
    else:
        log_fn(
            f"  search-free self-play  n={n} temp={cfg.sample_temp} "
            f"winner_only={cfg.winner_only} play_bs={batch}"
        )
        play = lambda ids: play_searchfree_games(model, device, cfg, ids, log_fn=log_fn)
    for start in range(0, n, batch):
        ids = list(range(start, min(start + batch, n)))
        pos, res = play(ids)
        all_positions.extend(pos)
        results.extend(res)
    if cfg.vs_incumbent:
        wins = sum(1 for r in results if r > 0.5)
        losses = sum(1 for r in results if r < 0.5)
        draws = len(results) - wins - losses
        score = sum(results) / max(1, len(results))
        log_fn(
            f"  student W/D/L={wins}/{draws}/{losses} score={score:.3f}  "
            f"train_positions={len(all_positions)}"
        )
    else:
        wins = sum(1 for r in results if r > 0.5)
        losses = sum(1 for r in results if r < 0.5)
        draws = len(results) - wins - losses
        log_fn(
            f"  W/D/L={wins}/{draws}/{losses}  winner_positions={len(all_positions)}"
        )
    return all_positions, results
