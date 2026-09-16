import os
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")

import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import chess
import pytest
import torch

from chess_squares64 import Squares64RecurrentConfig, build_squares64
from rl_selfplay.ppo import (
    PPOConfig, check_model, masked_log_probs, forward_kl, gae, decisions,
    collect_rollouts, make_optimizer, ppo_update, stack_inputs, clipped_policy_loss,
    terminal_reward, wdl_value, actor_value,
)
from experiments.exp285_ppo_league import checkpoint, restore_training, atomic_save, load_replay
from scripts.eval_stockfish_full import paired_score_summary, full_strength_options
from autoresearch_8gb.pipeline import attach_static_targets, board_to_cache_row, stack_rows


def tiny():
    torch.set_num_threads(1)
    torch.manual_seed(285)
    return build_squares64(Squares64RecurrentConfig(encoder_dim=16, hidden_dim=32,
        num_heads=4, prefix_layers=1, recurrent_layers=1, suffix_layers=1,
        policy_head_dim=16, value_hidden=16, dropout=.5, geometry_attention="gab",
        gab_d1=4, gab_d2=8, gab_d3=4, zero_init_out_proj=False))


def test_white_absolute_wdl_flips_for_black_actor():
    logits = torch.tensor([[4., 0., -4.], [-4., 0., 4.]])
    white = wdl_value(logits)
    assert white[0].item() > 0.9 and white[1].item() < -0.9
    flipped = actor_value(white, torch.tensor([True, False]))
    assert flipped[0].item() == pytest.approx(white[0].item())
    assert flipped[1].item() == pytest.approx(-white[1].item())
    assert actor_value(0.8, False) == pytest.approx(-0.8)


def test_gae_same_player_perspective_and_truncation():
    a, returns = gae([.2, -.1], 1., 999., gamma=1, lam=1)
    assert a == pytest.approx([.8, 1.1])
    assert returns == pytest.approx([1., 1.])
    _, returns = gae([.2, -.1], -1., 999., gamma=1, lam=1)
    assert returns == pytest.approx([-1., -1.])
    _, returns = gae([.2, -.1], None, .7, gamma=1, lam=1)
    assert returns == pytest.approx([.7, .7])
    _, returns = gae([.2], 0., .9)
    assert returns == pytest.approx([0.])


def test_terminal_outcome_signs_and_repetition():
    board = chess.Board()
    for u in "f2f3 e7e5 g2g4 d8h4".split():
        board.push_uci(u)
    assert terminal_reward(board, True) == -1
    assert terminal_reward(board, False) == 1
    board = chess.Board()
    for u in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split():
        board.push_uci(u)
    assert terminal_reward(board, True) == 0
    assert terminal_reward(chess.Board(), True) is None


def test_legal_mask_and_correct_forward_kl():
    mask = torch.tensor([[True, True, False]])
    x = torch.tensor([[1., 2., 1000.]], requires_grad=True)
    logp = masked_log_probs(x, mask, .8)
    ref = masked_log_probs(torch.tensor([[2., 0., 0.]]), mask, .8)
    assert logp.exp()[0, 2] == 0
    expected = (logp.exp()[0, :2] * (logp[0, :2] - ref[0, :2])).sum()
    torch.testing.assert_close(forward_kl(logp, ref)[0], expected)
    (-logp[0, 0] + forward_kl(logp, ref).mean()).backward()
    assert x.grad[0, 2] == 0 and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        masked_log_probs(x, torch.zeros_like(mask))


def test_clipping_has_no_gradient_past_bound_for_positive_advantage():
    new = torch.tensor([1.], requires_grad=True)
    loss, _, fraction = clipped_policy_loss(new, torch.tensor([0.]), torch.tensor([1.]), .15)
    loss.backward()
    assert new.grad.item() == 0 and fraction.item() == 1


def test_recorded_probability_matches_update_with_dropout_and_depth():
    model = tiny().train()
    rng = torch.Generator().manual_seed(10)
    board = chess.Board()
    row = decisions(model, [board], torch.device("cpu"), 4, .8, rng)[0]
    out = model(stack_inputs([row], "cpu"), recurrent_unrolls=row["depth"])
    logp = masked_log_probs(out["policy_logits"], row["mask"][None], .8)
    assert float(logp[0, row["action"]].detach()) == pytest.approx(row["old_logp"], abs=1e-7)
    assert not model.training


def test_paired_collection_and_update_keeps_reference_frozen(tmp_path):
    model = tiny()
    reference = copy.deepcopy(model).eval()
    ref_before = {k: v.clone() for k, v in reference.state_dict().items()}
    cfg = PPOConfig(games_per_iteration=4, rollout_batch_size=4, ply_cap=6,
                    depths=(2, 3, 4), epochs=1, minibatch_size=8, microbatch_size=2,
                    replay_weight=0, learning_rate=1e-4)
    rng, py_rng = torch.Generator().manual_seed(285), random.Random(285)
    rows, games = collect_rollouts(model, reference, cfg, [[], ["e2e4", "e7e5"]], torch.device("cpu"), rng, py_rng, log=lambda _: None)
    assert len(games) == 4 and rows
    by_id = {g["game_id"]: g for g in games}
    for i in (0, 2):
        assert by_id[i]["opening"] == by_id[i + 1]["opening"]
        assert by_id[i]["depth"] == by_id[i + 1]["depth"]
    assert all(g["reward"] is None for g in games)
    assert all(g["plies"] in (6, 7) for g in games)
    assert any(r["actor_white"] for r in rows) and any(not r["actor_white"] for r in rows)
    model.eval()
    with torch.no_grad():
        for row in rows:
            white_v = float(wdl_value(model(stack_inputs([row], "cpu"),
                                            recurrent_unrolls=row["depth"])["value_logits"])[0])
            assert row["old_value"] == pytest.approx(actor_value(white_v, row["actor_white"]), abs=1e-5)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = make_optimizer(model, cfg)
    result = ppo_update(model, reference, optimizer, rows, cfg, torch.device("cpu"), rng)
    assert result["updates"] > 0
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in ref_before.items())
    assert all(p.grad is None for p in reference.parameters())
    # Full optimizer and sampler state round-trip, not a weights-only restart.
    payload = checkpoint(model, optimizer, 1, tmp_path / "ref.pt", [], {}, rng, py_rng)
    path = tmp_path / "latest.pt"
    atomic_save(payload, path)
    loaded = torch.load(path, weights_only=False)
    other = make_optimizer(model, cfg)
    rng2, py2 = torch.Generator(), random.Random()
    restore_training(loaded, other, rng2, py2)
    assert other.state_dict()["state"]
    assert torch.equal(torch.rand(5, generator=rng), torch.rand(5, generator=rng2))
    assert py_rng.random() == py2.random()


def test_kl_stop_discards_minibatch():
    model = tiny().eval()
    ref = copy.deepcopy(model)
    cfg = PPOConfig(epochs=1, replay_weight=0)
    rng = torch.Generator().manual_seed(10)
    row = decisions(model, [chess.Board()], torch.device("cpu"), 3, .8, rng)[0]
    row.update(old_logp=row["old_logp"] - 5, advantage=1., return_target=1.)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = ppo_update(model, ref, make_optimizer(model, cfg), [row], cfg, torch.device("cpu"), rng)
    assert result["early_stop_kl"] and result["updates"] == 0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_replay_cache_filters_and_aux_update(tmp_path):
    board = chess.Board()
    train = attach_static_targets(stack_rows([board_to_cache_row(board, chess.Move.from_uci("e2e4"))]))
    hold = attach_static_targets(stack_rows([board_to_cache_row(board, chess.Move.from_uci("d2d4"))]))
    hold["split"] = torch.ones(1, dtype=torch.int8)
    train["split"] = torch.zeros(1, dtype=torch.int8)
    train["policy_mask"] = torch.ones(1, dtype=torch.int8)
    hold["policy_mask"] = torch.ones(1, dtype=torch.int8)
    mixed = {k: torch.cat([train[k], hold[k]], 0) if torch.is_tensor(train[k]) else train[k]
             for k in train}
    path = tmp_path / "replay.pt"
    torch.save(mixed, path)
    replay = load_replay(path)
    assert len(replay["turn"]) == 1
    assert int(replay["move_idx"][0]) != int(hold["move_idx"][0])
    bad = tmp_path / "bad.pt"
    torch.save({"turn": torch.zeros(0)}, bad)
    with pytest.raises(ValueError, match="malformed|Empty"):
        load_replay(bad)
    model = tiny()
    reference = copy.deepcopy(model).eval()
    row = decisions(model, [chess.Board()], torch.device("cpu"), 3, .8, torch.Generator().manual_seed(1))[0]
    row.update(advantage=1., return_target=1., actor_white=True)
    cfg = PPOConfig(epochs=1, replay_weight=0.25, replay_batch_size=1, target_kl=10.,
                    learning_rate=1e-4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = ppo_update(model, reference, make_optimizer(model, cfg), [row], cfg,
                        torch.device("cpu"), torch.Generator().manual_seed(2), replay)
    assert result["updates"] == 1 and "replay_loss" in result["minibatches"][0]
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_config_and_evaluation_protocol():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/exp285_ppo_99m.json").read_text())
    PPOConfig(**cfg["ppo"]).validate()
    with pytest.raises(ValueError):
        replace(PPOConfig(), games_per_iteration=3).validate()
    proto = json.loads((root / "configs/stockfish_full_policy.json").read_text())
    for opening in proto["openings"]:
        board = chess.Board()
        for move in opening:
            board.push_uci(move)
    class Engine:
        options = {"UCI_LimitStrength": None, "Skill Level": None, "SyzygyPath": None}
        def configure(self, values):
            assert "UCI_Elo" not in values
    options = full_strength_options(Engine())
    assert options["UCI_LimitStrength"] is False and options["Skill Level"] == 20
    result = paired_score_summary([dict(pair=0, score=1.), dict(pair=0, score=None)])
    assert result["draws"] == 0 and result["unfinished"] == 1
    assert result["score_bounds"] == [.5, 1.]
