import os

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import chess
import torch

from chess_chessbot import CHESSBOT_UCI_TO_IDX, move_to_policy_index
from chess_chessbot_recurrent import RecurrentChessBot, RecurrentSplit, build_empty_chessbot, identity_errors
from experiments.exp290_chessbot_local_mix import (
    BUCKETS,
    ENDGAME_SOURCE_KEYS,
    HOLDOUT_COUNT,
    HOLDOUT_START,
    REVISIONS,
    bucket_take,
    compact_to_chessbot_idx,
    depth_schedule,
    iter_bucket,
    iter_even_mix,
    iter_round_robin,
    keep_parquet_row,
    lr_factor,
    policy_losses,
    row_in_holdout,
    run_val,
    soft_to_policy,
    stack_val_rows,
    val_cache_ok,
    white_wdl_to_chessbot,
)
from move_vocab import UCI_TO_IDX


def test_dedicated_phase_repos():
    assert BUCKETS == ("puzzles", "endgame", "middlegame", "opening", "soft")
    assert REVISIONS["opening"][0] == "avewright/lichess-opening-bestline"
    assert REVISIONS["middlegame"][0] == "avewright/lichess-middlegame-bestline"
    assert REVISIONS["endgame"][0] == "avewright/lichess-endgame-bestline"
    assert REVISIONS["endgame_sf19"][0] == "avewright/endgame-dataset"
    assert ENDGAME_SOURCE_KEYS == ("endgame", "endgame_sf19", "endgame_syzygy")
    assert "chess-soft-multipv-lichess" not in {v[0] for v in REVISIONS.values()}


def test_holdout_window_excluded_from_train():
    start, end = HOLDOUT_START, HOLDOUT_START + HOLDOUT_COUNT
    train = [i for i in range(end + 8) if keep_parquet_row(0, i, holdout=False)]
    hold = [i for i in range(end + 8) if keep_parquet_row(0, i, holdout=True)]
    assert hold == list(range(start, end))
    assert set(train).isdisjoint(hold)
    assert 0 in train and start - 1 in train and end in train
    assert not row_in_holdout(1, start)
    assert all(keep_parquet_row(1, i, holdout=False) for i in range(start, end))
    assert not any(keep_parquet_row(1, i, holdout=True) for i in range(start, end))


def test_endgame_round_robin_reaches_all_sources():
    def src(name, n=3):
        for i in range(n):
            yield {"src": name, "i": i}

    out = [r["src"] for r in iter_round_robin([src("endgame"), src("endgame_sf19"), src("endgame_syzygy")])]
    assert out == [
        "endgame", "endgame_sf19", "endgame_syzygy",
        "endgame", "endgame_sf19", "endgame_syzygy",
        "endgame", "endgame_sf19", "endgame_syzygy",
    ]


def test_iter_bucket_endgame_does_not_starve_later_sources(monkeypatch):
    def fake_engine_row(rec, *, value_valid):
        pol = torch.full((1929,), -1.0)
        pol[0] = 1.0
        return {
            "fen": chess.Board().fen(),
            "policy": pol,
            "wdl": torch.tensor([0.0, 1.0, 0.0]),
            "value_valid": value_valid,
        }

    def fake_rows(repo, revision):
        while True:
            yield {"repo": repo}

    monkeypatch.setattr("experiments.exp290_chessbot_local_mix.engine_row", fake_engine_row)
    monkeypatch.setattr("experiments.exp290_chessbot_local_mix.sf19_ok", lambda rec: True)
    monkeypatch.setattr("experiments.exp290_chessbot_local_mix.syzygy_ok", lambda rec: True)
    stream = iter_bucket("endgame", rows_fn=fake_rows)
    rows = [next(stream) for _ in range(6)]
    assert [r["endgame_source"] for r in rows] == [
        "endgame", "endgame_sf19", "endgame_syzygy",
        "endgame", "endgame_sf19", "endgame_syzygy",
    ]
    assert {r["endgame_source"] for r in rows} == set(ENDGAME_SOURCE_KEYS)


def test_val_cache_rejects_leaky_v1():
    cache = {
        "n_per_bucket": 48,
        "buckets": list(BUCKETS),
        "fen": ["x"],
    }
    assert not val_cache_ok(cache, 48)


def test_compact_to_chessbot_maps_castle_and_knight_promo():
    assert compact_to_chessbot_idx(UCI_TO_IDX["e2e4"]) == CHESSBOT_UCI_TO_IDX["e2e4"]
    assert compact_to_chessbot_idx(UCI_TO_IDX["e1h1"]) == CHESSBOT_UCI_TO_IDX["e1g1"]
    knight = next(u for u in UCI_TO_IDX if u.endswith("n"))
    assert compact_to_chessbot_idx(UCI_TO_IDX[knight]) == CHESSBOT_UCI_TO_IDX[knight[:-1]]


def test_white_wdl_flips_to_chessbot_order():
    out = white_wdl_to_chessbot([0.7, 0.2, 0.1])
    torch.testing.assert_close(out, torch.tensor([0.1, 0.2, 0.7]))


def test_soft_policy_drops_unmapped_mass():
    e2e4 = UCI_TO_IDX["e2e4"]
    d2d4 = UCI_TO_IDX["d2d4"]
    pol = soft_to_policy(e2e4, [e2e4, d2d4, -1], [0.6, 0.4, 0.0])
    assert pol is not None
    assert abs(float(pol[CHESSBOT_UCI_TO_IDX["e2e4"]]) - 0.6) < 1e-6
    assert abs(float(pol[CHESSBOT_UCI_TO_IDX["d2d4"]]) - 0.4) < 1e-6
    assert abs(float(pol.clamp(min=0).sum()) - 1.0) < 1e-6


def test_slow_warmup_starts_near_zero():
    assert lr_factor(0, 1000, 8000, True, 0.1) == 0.0
    assert abs(lr_factor(4000, 156250, 8000, True, 0.1) - 0.5) < 1e-9
    assert lr_factor(8000, 156250, 8000, True, 0.1) == 1.0
    assert lr_factor(156250, 156250, 8000, True, 0.1) == 0.1


def test_deeper_unroll_schedule():
    sched = depth_schedule(12, [2, 3], 290)
    assert set(sched) == {2, 3}
    assert sched.count(2) == 6
    assert sched.count(3) == 6


def test_tiny_wrap_n1_matches_published_loop():
    torch.manual_seed(290)
    base = build_empty_chessbot(num_layers=4, d_model=32, d_ff=64, num_heads=4)
    model = RecurrentChessBot(base, default_unrolls=2, split=RecurrentSplit(1, 2, 1))
    planes = torch.zeros(2, 64, 19)
    planes[:, :, 12] = 1.0
    err = identity_errors(model, planes)
    assert max(err.values()) < 1e-5
    from chess_chessbot_recurrent import depth_identity_errors
    assert max(depth_identity_errors(model, planes, 2).values()) < 1e-5
    assert float(model.alpha) == 0
    assert model.effective_depth(1) == 4
    assert model.effective_depth(2) == 6
    assert model.effective_depth(3) == 8


def test_even_mix_uses_one_row_per_bucket(monkeypatch):
    def fake_bucket(name):
        while True:
            pol = torch.full((1929,), -1.0)
            pol[move_to_policy_index(chess.Move.from_uci("e2e4"))] = 1.0
            yield {
                "fen": chess.Board().fen(),
                "policy": pol,
                "wdl": torch.tensor([0.0, 1.0, 0.0]),
                "value_valid": name == "soft",
                "bucket": name,
            }

    monkeypatch.setattr("experiments.exp290_chessbot_local_mix.iter_bucket", fake_bucket)
    take = bucket_take(5, 290)
    assert take == {name: 1 for name in BUCKETS}
    take32 = bucket_take(32, 290)
    assert sum(take32.values()) == 32
    assert set(take32) == set(BUCKETS)
    planes, policy, wdl, valid = next(iter_even_mix(5, seed=290, prefetch=False))
    assert planes.shape[0] == 5
    assert int(valid.sum()) == 1
    assert policy.shape == (5, 1929)


def test_run_val_on_tiny_wrap():
    torch.manual_seed(290)
    base = build_empty_chessbot(num_layers=4, d_model=32, d_ff=64, num_heads=4)
    model = RecurrentChessBot(base, default_unrolls=2, split=RecurrentSplit(1, 2, 1))
    pol = torch.full((1929,), -1.0)
    pol[move_to_policy_index(chess.Move.from_uci("e2e4"))] = 1.0
    cache = stack_val_rows([{
        "fen": chess.Board().fen(),
        "policy": pol,
        "wdl": torch.tensor([0.0, 1.0, 0.0]),
        "value_valid": False,
        "bucket": "opening",
    }])
    out = run_val(model, model, cache, torch.device("cpu"), [1, 2, 3], batch_size=1)
    assert out["n"] == 1
    assert "1" in out["by_unrolls"] and "2" in out["by_unrolls"] and "3" in out["by_unrolls"]
    assert out["by_unrolls"]["1"]["depth"] == 4
    assert out["by_unrolls"]["2"]["depth"] == 6
    assert out["by_unrolls"]["3"]["depth"] == 8


def test_policy_loss_finite():
    logits = torch.zeros(2, 1929)
    target = torch.full((2, 1929), -1.0)
    target[:, 10] = 1.0
    loss, hard, valid = policy_losses(logits, target, 0.55)
    assert torch.isfinite(loss)
    assert bool(valid.all())
    assert hard > 0
