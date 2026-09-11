"""Lichess ECO opening loader and ECO game-spec construction."""
from pathlib import Path

import chess

from scripts.lichess_openings import (
    HF_CARD_N,
    fen4,
    load_openings,
    openings_summary,
    prefix_fens,
    start_positions,
)
from scripts.sf19_soft_dataset import (
    ECO_EPSILONS,
    build_eco_game_spec,
    pick_play_move,
)

FIXTURE = Path(__file__).parent / "fixtures" / "openings_sample.tsv"
FULL_DIR = Path(__file__).resolve().parents[1] / "data" / "lichess_openings"


def test_parse_fixture_skips_nothing_but_marks_terminal():
    openings = load_openings(FIXTURE)
    assert len(openings) == 4
    by_name = {o.name: o for o in openings}
    amar = by_name["Amar Opening"]
    assert amar.eco == "A00"
    assert amar.uci == ("g1h3",)
    assert fen4(amar.fen) == "rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq -"
    assert amar.terminal is False
    mate = by_name["Barnes Opening: Fool's Mate"]
    assert mate.terminal is True
    assert chess.Board(mate.fen).is_checkmate()


def test_start_positions_drop_terminals_and_can_include_prefixes():
    openings = load_openings(FIXTURE)
    leaves = start_positions(openings, include_prefixes=False)
    assert {s["name"] for s in leaves} == {"Amar Opening", "King's Pawn Game", "Sicilian Defense"}
    assert all(s["kind"] == "leaf" for s in leaves)
    with_pref = start_positions(openings, include_prefixes=True)
    keys = {s["key"] for s in with_pref}
    # e4 is a shared prefix of C20 and B20
    e4 = chess.Board()
    e4.push_uci("e2e4")
    assert fen4(e4.fen()) in keys
    assert not any(chess.Board(s["fen"]).is_game_over(claim_draw=True) for s in with_pref)
    assert len(with_pref) > len(leaves)


def test_prefix_fens_follow_uci():
    fens = prefix_fens(("e2e4", "e7e5"))
    assert len(fens) == 2
    b = chess.Board(fens[-1])
    assert b.piece_at(chess.E4) is not None
    assert b.piece_at(chess.E5) is not None


def test_eco_specs_cycle_starts_and_vary_epsilon():
    openings = load_openings(FIXTURE)
    starts = start_positions(openings, include_prefixes=False)
    specs = [build_eco_game_spec(i, starts, seed=19, holdout_frac=0.0) for i in range(12)]
    assert {s["start_fen"] for s in specs} == {s["fen"] for s in starts}
    assert {s["epsilon"] for s in specs} == set(ECO_EPSILONS)
    assert {s["book_noise"] for s in specs} == {0, 1, 2, 3, 4}
    assert all(s["opening"] == [] for s in specs)
    assert all(s["split"] == 0 for s in specs)


def test_pick_play_move_wild_ignores_best():
    board = chess.Board()
    parsed = {"items": [{"uci": "e2e4"}], "probs": [1.0], "terminal": False}
    # seed chosen so the first wild roll fires
    mv = pick_play_move(parsed, board, __import__("random").Random(1), epsilon=0.0, wild=1.0)
    assert mv in board.legal_moves


def test_full_dataset_meets_hf_card_when_present():
    if not (FULL_DIR / "a.tsv").exists():
        return
    openings = load_openings(FULL_DIR)
    starts = start_positions(openings, include_prefixes=True)
    summary = openings_summary(openings, starts)
    assert summary["n_unique"] >= HF_CARD_N
    assert summary["n_terminal"] >= 1
    assert len(starts) >= HF_CARD_N
    assert all(v > 0 for v in summary["volumes"].values())
    assert not any(chess.Board(s["fen"]).is_game_over(claim_draw=True) for s in starts[:50])
