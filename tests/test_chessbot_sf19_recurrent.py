import json
from pathlib import Path

import chess
import numpy as np
import torch

from chess_chessbot import CHESSBOT_UCI_TO_IDX, CHESSBOT_VOCAB_SIZE, expand_chessbot_policy
from chess_chessbot_recurrent import RecurrentChessBot, RecurrentSplit, build_empty_chessbot, depth_identity_errors
from experiments.chessbot_sf19_recurrent import (
    VALUE_SOURCES,
    collate,
    freeze_trunk,
    game_reward,
    keep_row,
    load_openings,
    lr_factor,
    policy_losses,
    synthetic_rows,
    train,
    value_losses,
)
from rl_selfplay.chessbot_eval import paired_eval
from scripts.chessbot_sf19_monitor import parse_events


def test_expand_sparse_multipv_stays_minus_one():
    e2e4 = CHESSBOT_UCI_TO_IDX["e2e4"]
    d2d4 = CHESSBOT_UCI_TO_IDX["d2d4"]
    pol = expand_chessbot_policy([e2e4, d2d4, -1, -1], [0.7, 0.3, 0.0, 0.0])
    assert pol.shape == (CHESSBOT_VOCAB_SIZE,)
    assert pol.dtype == np.float32
    assert abs(float(pol.clip(min=0).sum()) - 1.0) < 1e-5
    assert abs(float(pol[e2e4]) - 0.7) < 1e-6
    assert abs(float(pol[d2d4]) - 0.3) < 1e-6
    assert float((pol < 0).mean()) > 0.99


def test_keep_row_honors_split():
    row = {"fen": "x", "split": 0}
    assert keep_row(row, 0)
    assert not keep_row(row, 1)
    assert not keep_row({"fen": "", "split": 0}, 0)
    assert not keep_row({"fen": "x", "split": 1}, 0)


def test_collate_does_not_remap_wdl():
    rows = synthetic_rows(2)
    rows[1]["wdl_source"] = 0
    planes, policy, hard, wdl, valid = collate(rows)
    assert planes.shape == (2, 64, 19)
    assert policy.shape == (2, CHESSBOT_VOCAB_SIZE)
    assert int(hard[0]) == CHESSBOT_UCI_TO_IDX["e2e4"]
    torch.testing.assert_close(wdl[0], torch.tensor([0.12, 0.56, 0.32]))
    assert bool(valid[0]) and not bool(valid[1])
    assert set(VALUE_SOURCES) == {1, 3}


def test_hard_ce_uses_hard_idx_not_argmax_of_mass():
    logits = torch.zeros(1, CHESSBOT_VOCAB_SIZE)
    e2e4 = CHESSBOT_UCI_TO_IDX["e2e4"]
    d2d4 = CHESSBOT_UCI_TO_IDX["d2d4"]
    logits[0, d2d4] = 8.0
    policy = torch.full((1, CHESSBOT_VOCAB_SIZE), -1.0)
    policy[0, e2e4] = 0.4
    policy[0, d2d4] = 0.6
    _, h_e2e4, _, _ = policy_losses(logits, policy, torch.tensor([e2e4]), 0.0)
    _, h_d2d4, _, _ = policy_losses(logits, policy, torch.tensor([d2d4]), 0.0)
    assert float(h_e2e4) > float(h_d2d4)


def test_value_loss_skipped_when_policy_only():
    logits = {
        "value_logits": torch.zeros(2, 3, requires_grad=True),
        "value_logits_q": torch.zeros(2, 3, requires_grad=True),
    }
    wdl = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    loss, hard, soft = value_losses(logits, wdl, torch.tensor([False, False]), 0.15)
    assert float(loss) == 0.0 and float(hard) == 0.0 and float(soft) == 0.0
    used, _, _ = value_losses(logits, wdl, torch.tensor([True, False]), 0.15)
    used.backward()
    assert logits["value_logits"].grad is not None


def test_low_lr_warmup():
    assert lr_factor(0, 122500, 200, True, 0.1) == 0.0
    assert abs(lr_factor(100, 122500, 200, True, 0.1) - 0.5) < 1e-9
    assert lr_factor(200, 122500, 200, True, 0.1) == 1.0
    assert lr_factor(122500, 122500, 200, True, 0.1) == 0.1


def test_gated_n3_is_identity_on_tiny_wrap():
    torch.manual_seed(296)
    base = build_empty_chessbot(num_layers=4, d_model=32, d_ff=64, num_heads=4)
    model = RecurrentChessBot(base, default_unrolls=3, split=RecurrentSplit(1, 2, 1), gate_extra=True)
    planes = torch.zeros(2, 64, 19)
    planes[:, :, 12] = 1.0
    assert float(model.alpha) == 0
    assert max(depth_identity_errors(model, planes, 3).values()) < 1e-5
    assert model.effective_depth(3) == 8


def test_game_reward_and_paired_match_event():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert game_reward(board, True, 400) == 0
    mate = chess.Board("8/8/8/8/8/6K1/7Q/7k b - - 0 1")
    assert game_reward(mate, True, 400) == 1
    assert game_reward(mate, False, 400) == -1
    live = chess.Board()
    assert game_reward(live, True, 400) is None
    assert game_reward(live, True, 0) is None
    match = dict(games=[
        dict(opening=["e2e4"], color=True, reward=1),
        dict(opening=["e2e4"], color=False, reward=1),
    ], wins=2, draws=0, losses=0, unknown=0, n=2, score_bounds=[1.0, 1.0])
    ev = paired_eval(match, opponent="original")
    assert ev["verdict"] == "stronger"
    assert ev["score"] == 1.0


def test_monitor_parses_match_events(tmp_path):
    log = tmp_path / "events.jsonl"
    log.write_text(json.dumps({
        "stage": "match", "step": 1000, "score": 0.53, "wins": 9, "draws": 16,
        "losses": 7, "verdict": "inconclusive", "paired_ci_95": [0.41, 0.65],
        "n": 32, "gate": 0.0002,
    }) + "\n")
    d = parse_events(log, 122500)
    assert d["last_match"]["verdict"] == "inconclusive"
    assert d["last_match"]["wins"] == 9


def test_load_openings_caps_pairs():
    rows = load_openings(None, 4)
    assert len(rows) == 4
    assert all(isinstance(o, list) and o for o in rows)


def test_freeze_trunk_only_gate_moves():
    torch.manual_seed(297)
    base = build_empty_chessbot(num_layers=4, d_model=32, d_ff=64, num_heads=4)
    model = RecurrentChessBot(base, default_unrolls=3, split=RecurrentSplit(1, 2, 1), gate_extra=True)
    report = freeze_trunk(model)
    assert report["trainable"] == ["alpha"]
    assert model.alpha.requires_grad
    assert all(not p.requires_grad for n, p in model.named_parameters() if n != "alpha")


def test_synthetic_frozen_trunk(tmp_path):
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/chessbot_sf19_n3_gate.json").read_text())
    cfg.update(steps=2, batch=2, val_size=2, val_every=2, save_every=2, log_every=1)
    train(cfg, tmp_path, torch.device("cpu"), resume=False, synthetic=True)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    loaded = next(e for e in events if e["stage"] == "loaded")
    assert loaded["freeze_trunk"]["trainable"] == ["alpha"]


def test_synthetic_two_steps(tmp_path):
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/chessbot_sf19_n3.json").read_text())
    cfg.update(steps=2, batch=2, val_size=2, val_every=2, save_every=2, log_every=1, optimizer="adamw", lr=1e-4)
    train(cfg, tmp_path, torch.device("cpu"), resume=False, synthetic=True)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e["stage"] == "loaded" for e in events)
    assert any(e["stage"] == "train" for e in events)
    assert (tmp_path / "latest.pt").exists()
