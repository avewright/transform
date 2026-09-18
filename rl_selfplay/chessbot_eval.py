"""Paired-game evaluation contract for ChessBot RL.

score_bounds remain unfinished-game pessimism/optimism. Statistical uncertainty
is a separate interval over color-swapped opening pairs (Fishtest-style units).
PPO loss is never a strength signal.
"""
from __future__ import annotations

import math
import os
from pathlib import Path


VERDICTS = ('not_evaluated', 'inconclusive', 'stronger', 'weaker')
PENT_KEYS = (0.0, 0.5, 1.0, 1.5, 2.0)
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 24: 2.064,
    30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def game_score(reward):
    if reward is None:
        return None
    return (float(reward) + 1.0) / 2.0


def t_crit_95(df: int) -> float:
    if df <= 0:
        return float('nan')
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    for k in keys:
        if k >= df:
            return _T95[k]
    return 1.96


def pair_units(match):
    """One unit per opening: mean of the two color-reversed game scores."""
    by = {}
    for game in match.get('games') or []:
        by.setdefault(tuple(game['opening']), {})[bool(game['color'])] = game.get('reward')
    units, unfinished = [], 0
    pent = {k: 0 for k in PENT_KEYS}
    for sides in by.values():
        if True not in sides or False not in sides:
            continue
        a, b = sides[True], sides[False]
        if a is None or b is None:
            unfinished += 1
            continue
        sa, sb = game_score(a), game_score(b)
        pair = sa + sb
        units.append(pair / 2.0)
        key = min(PENT_KEYS, key=lambda k: abs(k - pair))
        pent[key] += 1
    return units, pent, unfinished


def paired_interval(units, z=None):
    n = len(units)
    if n == 0:
        return None, [None, None], None
    mean = sum(units) / n
    if n == 1:
        return mean, [mean, mean], 0.0
    var = sum((x - mean) ** 2 for x in units) / (n - 1)
    se = math.sqrt(var / n)
    crit = t_crit_95(n - 1) if z is None else float(z)
    lo, hi = mean - crit * se, mean + crit * se
    return mean, [max(0.0, lo), min(1.0, hi)], se


def verdict(ci):
    if not ci or ci[0] is None or ci[1] is None:
        return 'not_evaluated'
    lo, hi = ci
    if lo > 0.5:
        return 'stronger'
    if hi < 0.5:
        return 'weaker'
    return 'inconclusive'


def pair_stats(match):
    """Score each opening pair. Color-splits are ties; both-color wins are the signal."""
    by = {}
    for game in match.get('games') or []:
        by.setdefault(tuple(game['opening']), {})[bool(game['color'])] = game.get('reward')
    plus = minus = tie = 0
    for sides in by.values():
        if True not in sides or False not in sides or None in sides.values():
            continue
        total = sides[True] + sides[False]
        if total > 0:
            plus += 1
        elif total < 0:
            minus += 1
        else:
            tie += 1
    n = plus + minus + tie
    return dict(pairs=n, plus_pairs=plus, minus_pairs=minus, tied_pairs=tie,
                pair_score=(plus - minus) / n if n else 0.)


def paired_eval(match, *, kind='development', question='stronger', opponent=None):
    stats = pair_stats(match)
    units, pent, unfinished = pair_units(match)
    mean, ci, se = paired_interval(units)
    wdl = match if 'wins' in match else {}
    bounds = match.get('score_bounds')
    return dict(
        **stats,
        wins=wdl.get('wins', sum(g.get('reward') == 1 for g in match.get('games') or [])),
        draws=wdl.get('draws', sum(g.get('reward') == 0 for g in match.get('games') or [])),
        losses=wdl.get('losses', sum(g.get('reward') == -1 for g in match.get('games') or [])),
        unknown=wdl.get('unknown', sum(g.get('reward') is None for g in match.get('games') or [])),
        n=wdl.get('n', len(match.get('games') or [])),
        score=mean,
        paired_ci_95=ci,
        paired_se=se,
        pentanomial={str(k): pent[k] for k in PENT_KEYS},
        unfinished_pairs=unfinished,
        verdict=verdict(ci),
        kind=kind,
        question=question,
        opponent=opponent,
        score_bounds=bounds,
        note='paired_ci_95 is a t-interval over opening pairs. score_bounds only cover unfinished games.',
    )


def previous_from_league(league, exclude=()):
    skip = {Path(p).resolve() for p in exclude}
    for path in reversed(list(league or ())):
        resolved = Path(path).resolve()
        if resolved in skip or not resolved.exists():
            continue
        return str(resolved)
    return None


def screen_roles(*, incumbent, control, previous, recurrent):
    roles = ['original']
    if incumbent:
        roles.append('incumbent')
    if control:
        roles.append('control')
    if previous:
        roles.append('previous')
    if recurrent:
        roles.append('self_n1')
    return roles


def question_for(label):
    if label == 'control':
        return 'rl'
    if label == 'self_n1':
        return 'recurrence'
    return 'stronger'


def stockfish_path():
    for cand in (os.environ.get('STOCKFISH_PATH'), str(Path.home() / '.local/bin/stockfish-19'),
                 str(Path.home() / '.local/bin/stockfish')):
        if cand and Path(cand).exists():
            return cand
    return None


def sf_prefer_n2(rows, nodes=8000, limit=16):
    """Fixed-budget SF: +1 if n2 move scores better than n1, -1 if worse."""
    path = stockfish_path()
    if not path or not rows:
        return dict(available=bool(path), n=0, n2_better=0, n2_worse=0, equal=0)
    import chess
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci(path)
    better = worse = equal = 0
    try:
        limit_cfg = chess.engine.Limit(nodes=int(nodes))
        for row in rows[:limit]:
            board = chess.Board(row['fen'])
            try:
                m1 = chess.Move.from_uci(row['n1'])
                m2 = chess.Move.from_uci(row['n2'])
            except ValueError:
                continue
            if m1 not in board.legal_moves or m2 not in board.legal_moves:
                continue
            info = engine.analyse(board, limit_cfg, root_moves=[m1, m2], multipv=2)
            scores = {}
            for item in info if isinstance(info, list) else [info]:
                mv = item.get('pv', [None])[0]
                sc = item.get('score')
                if mv is None or sc is None:
                    continue
                scores[mv.uci()] = sc.white().score(mate_score=100000)
            a, b = scores.get(row['n1']), scores.get(row['n2'])
            if a is None or b is None:
                continue
            stm = 1 if board.turn == chess.WHITE else -1
            delta = stm * (b - a)
            if delta > 0:
                better += 1
            elif delta < 0:
                worse += 1
            else:
                equal += 1
    finally:
        engine.quit()
    return dict(available=True, n=better + worse + equal, n2_better=better, n2_worse=worse, equal=equal)
