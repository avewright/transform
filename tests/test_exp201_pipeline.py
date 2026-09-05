#!/usr/bin/env python3
"""CPU tests for exp201 training-pipeline correctness.

Does not touch the live GPU run or overwrite outputs/exp201_recurrent_64/*.pt.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import chess
import torch
import torch.nn.functional as F

from autoresearch_8gb.pipeline import (
    apply_membership,
    attach_static_targets,
    audit_soft_targets,
    board_to_cache_row,
    classify_checkpoint,
    cheap_eval_losses,
    collect_rng_state,
    concat_soft_tables,
    exposure_report,
    filter_disjoint,
    ingest_ready_bonus_shards,
    list_attached_shards,
    list_ready_shards,
    load_model_state,
    lr_scale,
    make_val_membership,
    muon_update_scale_note,
    pick_mix_source,
    policy_soft_temp_weight,
    position_hashes,
    session_throughput,
    prepare_soft_batch,
    restore_rng_state,
    save_training_checkpoint,
    soften_policy_targets,
    soft_policy_loss,
    soft_temp_policy_loss,
    stack_rows,
)
from data_loader import CASTLING_MAP, hflip_board_array, hflip_ep_square, hflip_move_idx
from move_vocab import IDX_TO_UCI, move_to_index, legal_move_mask


def _device():
    return torch.device("cpu")


def test_soft_temp_padding_does_not_change_loss_or_grad():
    torch.manual_seed(0)
    B, V, K = 5, 32, 4
    logits = torch.randn(B, V, requires_grad=True)
    idx = torch.tensor([
        [3, 7, -1, -1],
        [1, -1, -1, -1],
        [4, 5, 9, -1],
        [-1, -1, -1, -1],
        [2, 8, 11, 13],
    ], dtype=torch.int64)
    prob = torch.tensor([
        [0.6, 0.4, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.3, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.4, 0.3, 0.2, 0.1],
    ])
    # padded clone with extra dummy slots that old code would treat as 1e-8
    idx_pad = torch.cat([idx, torch.full((B, 4), -1)], dim=1)
    prob_pad = torch.cat([prob, torch.zeros(B, 4)], dim=1)

    loss_a = soft_temp_policy_loss(logits, idx, prob, temperature=4.0)
    g_a = torch.autograd.grad(loss_a, logits, retain_graph=True)[0]
    loss_b = soft_temp_policy_loss(logits, idx_pad, prob_pad, temperature=4.0)
    g_b = torch.autograd.grad(loss_b, logits)[0]
    assert torch.isfinite(loss_a)
    assert abs(float(loss_a - loss_b)) < 1e-6
    assert torch.allclose(g_a, g_b, atol=1e-6)

    # empty-row batch must stay finite
    empty_i = torch.full((2, 4), -1)
    empty_p = torch.zeros(2, 4)
    loss_e = soft_temp_policy_loss(torch.randn(2, V, requires_grad=True), empty_i, empty_p, 4.0)
    assert torch.isfinite(loss_e)
    assert float(loss_e) == 0.0

    p_t = soften_policy_targets(idx_pad, prob_pad, 4.0)
    assert torch.all(p_t[idx_pad < 0] == 0)
    # old bug: each pad slot contributed ~0.01 before normalize at T=4
    old = prob_pad.clamp_min(1e-8).pow(0.25)
    old = old / old.sum(dim=-1, keepdim=True)
    # padded mass in the old target should be clearly nonzero on empty-ish rows
    assert float(old[0, 2:].sum()) > 1e-3
    assert float(p_t[0, 2:].sum()) == 0.0


def test_hflip_skips_castling_and_maps_ep_moves():
    # 1) start position: castling rights live → must not flip even at p=1
    start = chess.Board()
    e4 = chess.Move.from_uci("e2e4")
    start_row = board_to_cache_row(start, e4)
    assert int(start_row["castling"]) != 0
    data = stack_rows([start_row])
    bi, hard, _w, si, _sp = prepare_soft_batch(data, torch.tensor([0]), _device(), hflip_p=1.0)
    assert torch.equal(bi["fused_ids"][0].cpu(), data["board_array"][0].long())
    assert int(hard[0]) == int(start_row["move_idx"])
    assert int(si[0, 0]) == int(start_row["soft_indices"][0])
    assert int(bi["castling"][0]) == int(start_row["castling"])

    # 2) EP position, no castling: 1.e4 a6 2.e5 d5  (ep d6), White plays exd6
    b = chess.Board()
    for u in ("e2e4", "a7a6", "e4e5", "d7d5"):
        b.push_uci(u)
    assert b.ep_square == chess.D6
    b.castling_rights = 0
    cap = chess.Move.from_uci("e5d6")
    assert cap in b.legal_moves
    row = board_to_cache_row(b, cap)
    assert int(row["castling"]) == 0
    data = stack_rows([row])
    bi, hard, _w, si, _sp = prepare_soft_batch(data, torch.tensor([0]), _device(), hflip_p=1.0)

    flipped = chess.Board()
    for u in ("d2d4", "h7h6", "d4d5", "e7e5"):
        flipped.push_uci(u)
    flipped.castling_rights = 0
    assert flipped.ep_square == chess.E6
    want = chess.Move.from_uci("d5e6")
    assert want in flipped.legal_moves
    want_idx = move_to_index(want)
    assert int(hard[0]) == want_idx
    assert int(si[0, 0]) == want_idx
    assert int(bi["ep_file"][0]) == (chess.E6 % 8) + 1
    # mapped move is legal on the flipped board
    mask = legal_move_mask(flipped)
    assert bool(mask[int(hard[0])])

    # 3) mixed batch: castle row + ep row, p=1 → only ep row flips
    mixed = stack_rows([start_row, row])
    bi, hard, _w, si, _sp = prepare_soft_batch(mixed, torch.tensor([0, 1]), _device(), hflip_p=1.0)
    assert int(hard[0]) == int(start_row["move_idx"])
    assert int(hard[1]) == want_idx
    assert int(bi["castling"][0]) == int(start_row["castling"])
    assert int(bi["castling"][1]) == 0


def test_hflip_hard_and_soft_indices_match_table():
    b = chess.Board(None)
    b.set_piece_at(chess.E4, chess.Piece.from_symbol("N"))
    b.set_piece_at(chess.E1, chess.Piece.from_symbol("K"))
    b.set_piece_at(chess.E8, chess.Piece.from_symbol("k"))
    b.turn = chess.WHITE
    b.castling_rights = 0
    mv = chess.Move.from_uci("e4f6")
    row = board_to_cache_row(b, mv)
    data = stack_rows([row])
    _bi, hard, _w, si, _sp = prepare_soft_batch(data, torch.tensor([0]), _device(), hflip_p=1.0)
    expect = int(hflip_move_idx(row["move_idx"].view(1))[0])
    assert int(hard[0]) == expect
    assert int(si[0, 0]) == expect
    assert IDX_TO_UCI[expect] == "d4c6"


def test_val_membership_blocks_flip_equivalents():
    start = board_to_cache_row(chess.Board(), chess.Move.from_uci("e2e4"))
    b = chess.Board()
    for u in ("e2e4", "a7a6", "e4e5", "d7d5"):
        b.push_uci(u)
    b.castling_rights = 0
    ep_row = board_to_cache_row(b, chess.Move.from_uci("e5d6"))
    # include the explicit flip of the ep row as a "different" sample
    flipped_ba = hflip_board_array(ep_row["board_array"].unsqueeze(0))[0]
    flipped_ep = hflip_ep_square(ep_row["ep_square"].unsqueeze(0))[0]
    flip_row = dict(ep_row)
    flip_row["board_array"] = flipped_ba
    flip_row["ep_square"] = flipped_ep.to(ep_row["ep_square"].dtype)
    data = stack_rows([start, ep_row, flip_row, start])
    man = make_val_membership(data, n_hold=1, seed=0, source="test")
    # force the ep row into val by constructing a tiny set
    from autoresearch_8gb.pipeline import position_hashes
    hs = position_hashes(data)
    man["hashes"] = [int(hs[1])]
    man["blocked_hashes"] = [int(hs[1]), int(hs[2])]
    train_idx, val_idx = apply_membership(data, man)
    assert 1 in val_idx.tolist()
    assert 1 not in train_idx.tolist()
    assert 2 not in train_idx.tolist()  # flip equivalent blocked


def test_filter_disjoint_drops_internal_and_prior():
    b_e4 = chess.Board()
    b_e4.push_uci("e2e4")
    b_d4 = chess.Board()
    b_d4.push_uci("d2d4")
    e4 = board_to_cache_row(b_e4, chess.Move.from_uci("e7e5"))
    d4 = board_to_cache_row(b_d4, chess.Move.from_uci("d7d5"))
    out, hs, rep = filter_disjoint(stack_rows([e4, e4, d4]), None)
    assert rep["n_in"] == 3
    assert rep["n_out"] == 2
    assert rep["internal_dups"] == 1
    assert rep["vs_seen"] == 0
    assert int(out["board_array"].shape[0]) == 2
    assert int(hs.size) == 2

    seen = position_hashes(stack_rows([e4]))
    out2, hs2, rep2 = filter_disjoint(stack_rows([e4, d4]), seen)
    assert rep2["n_out"] == 1
    assert rep2["vs_seen"] == 1
    assert int(hs2.size) == 1


def test_list_ready_shards_ignores_attached():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ready = td / "shard_014"
        attached = td / "shard_008"
        both = td / "shard_019"
        for p in (ready, attached, both):
            p.mkdir()
            (p / "soft_cache.pt").write_bytes(b"x")
        (ready / "READY").write_text("n=1\n")
        (attached / "ATTACHED").write_text("step=1\n")
        (both / "READY").write_text("n=1\n")
        (both / "ATTACHED").write_text("step=1\n")
        assert list_ready_shards(td) == [ready, both]
        assert list_attached_shards(td) == [attached]


def test_audit_and_exposure():
    rows = [board_to_cache_row(chess.Board(), chess.Move.from_uci("e2e4")) for _ in range(8)]
    data = stack_rows(rows)
    rep = audit_soft_targets(data)
    assert rep["ok"]
    exp = exposure_report(
        shallow_n=10_498_000, deep_n=399_000, deep_mix_frac=0.4,
        shallow_seen=0, deep_seen=0,
    )
    # ~17.5× as specified
    assert 17.0 < exp["deep_vs_shallow_odds"] < 18.0


def test_checkpoint_roundtrip_and_weights_only_label():
    from chess_squares64 import Squares64RecurrentConfig, build_squares64

    cfg = Squares64RecurrentConfig(
        encoder_dim=32, hidden_dim=32, num_heads=4, ffn_ratio=2,
        dropout=0.0, prefix_layers=1, recurrent_layers=1, recurrent_unrolls=2,
        suffix_layers=1, policy_head_dim=32, value_hidden=32,
    )
    model = build_squares64(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        live = td / "latest.pt"
        ev = td / "eval_swa.pt"
        kg = td / "known_good.pt"
        save_training_checkpoint(
            path=live, model=model, optimizer=opt, step=7, positions=128,
            model_cfg=cfg.to_dict(), train_cfg={"warmup": 0, "min_lr_frac": 1.0, "muon_lr": 0.001, "adam_lr": 1e-5},
            trial_id="tiny", arch="squares64", n_params=sum(p.numel() for p in model.parameters()),
            sampler_rng=None, manifest={"hashes": [1]}, swa_state=None, swa_n=0,
            resume_kind="full", status="mid", eval_path=ev, known_good_path=kg,
        )
        ck = torch.load(live, map_location="cpu", weights_only=False)
        assert classify_checkpoint(ck) == "full"
        assert ck["model_state_dict"]
        assert ck["optimizer_state_dict"]
        assert ck["steps"] == 7
        assert kg.exists()
        assert not ev.exists()  # no SWA yet

        old = {"model_state_dict": ck["model_state_dict"], "steps": 32500, "swa_n": 0}
        assert classify_checkpoint(old) == "weights_only"
        model2 = build_squares64(cfg)
        model2.load_state_dict(load_model_state(ck), strict=True)
        for a, b in zip(model.parameters(), model2.parameters()):
            assert torch.equal(a, b)


def _tiny_step(model, opt, batch, avg_fn):
    opt.zero_grad(set_to_none=True)
    out = model(batch)
    loss = out["policy_logits"].float().pow(2).mean() + out["value_logits"].float().pow(2).mean()
    loss.backward()
    avg_fn(model)
    opt.step()
    return float(loss.detach())


def test_resume_matches_uninterrupted():
    from chess_squares64 import Squares64RecurrentConfig, average_recurrent_grads, build_squares64

    def make():
        cfg = Squares64RecurrentConfig(
            encoder_dim=32, hidden_dim=32, num_heads=4, ffn_ratio=2,
            dropout=0.0, prefix_layers=1, recurrent_layers=1, recurrent_unrolls=2,
            suffix_layers=1, policy_head_dim=32, value_hidden=32,
        )
        m = build_squares64(cfg)
        o = torch.optim.AdamW(m.parameters(), lr=3e-3)
        return cfg, m, o

    torch.manual_seed(123)
    cfg, m_a, o_a = make()
    torch.manual_seed(123)
    _cfg, m_b, o_b = make()
    for pa, pb in zip(m_a.parameters(), m_b.parameters()):
        assert torch.equal(pa, pb)

    rng = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(4):
        B = 3
        batches.append({
            "fused_ids": torch.randint(0, 13, (B, 64), generator=rng),
            "turn": torch.zeros(B, dtype=torch.long),
            "castling": torch.zeros(B, dtype=torch.long),
            "ep_file": torch.zeros(B, dtype=torch.long),
        })

    # uninterrupted 4 steps
    torch.manual_seed(7)
    losses_a = [_tiny_step(m_a, o_a, b, average_recurrent_grads) for b in batches]

    # 2 steps, save, reload, 2 steps
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        losses_b = []
        for b in batches[:2]:
            losses_b.append(_tiny_step(m_b, o_b, b, average_recurrent_grads))
        rng_snap = collect_rng_state()
        save_training_checkpoint(
            path=td / "latest.pt", model=m_b, optimizer=o_b, step=2, positions=6,
            model_cfg=cfg.to_dict(), train_cfg={"warmup": 0, "min_lr_frac": 1.0},
            trial_id="tiny", arch="squares64", n_params=1,
            sampler_rng=None, manifest=None, swa_state=None, swa_n=0,
            resume_kind="full", status="mid",
        )
        ck = torch.load(td / "latest.pt", map_location="cpu", weights_only=False)
        _, m_c, o_c = make()
        m_c.load_state_dict(load_model_state(ck), strict=True)
        o_c.load_state_dict(ck["optimizer_state_dict"])
        restore_rng_state(rng_snap)
        for b in batches[2:]:
            losses_b.append(_tiny_step(m_c, o_c, b, average_recurrent_grads))
        for pa, pc in zip(m_a.parameters(), m_c.parameters()):
            assert torch.allclose(pa, pc, atol=1e-6, rtol=1e-5), (pa - pc).abs().max()
    assert all(abs(a - b) < 1e-6 for a, b in zip(losses_a, losses_b))


def test_recurrent_grad_avg_does_not_scale_muon_like_adam():
    """1/unroll on grads is not 1/unroll on Polar or RMS-Adam updates.

    Polar is scale-invariant. Adam m/sqrt(v) is also ~scale-invariant for a
    stationary grad. SGD and the *pre-clip* global grad norm do scale.
    """
    from polar_normuon import _polar_express_impl, adam_update

    torch.manual_seed(0)
    G = torch.randn(16, 16)
    u_full = _polar_express_impl(G).float()
    u_avg = _polar_express_impl(G / 3).float()
    muon_ratio = float(u_avg.norm() / u_full.norm().clamp_min(1e-12))
    assert 0.85 < muon_ratio < 1.15, muon_ratio

    g = G[0].clone()
    a_full = adam_update(g.clone(), torch.zeros_like(g), torch.zeros_like(g), 1, (0.9, 0.95), 1e-10)
    a_avg = adam_update(g.clone() / 3, torch.zeros_like(g), torch.zeros_like(g), 1, (0.9, 0.95), 1e-10)
    adam_ratio = float(a_avg.norm() / a_full.norm().clamp_min(1e-12))
    assert 0.85 < adam_ratio < 1.15, adam_ratio

    sgd_ratio = float((g / 3).norm() / g.norm())
    assert abs(sgd_ratio - 1 / 3) < 1e-6

    p = torch.nn.Parameter(torch.randn(8, 8))
    p.grad = G[:8, :8].clone()
    pre = torch.nn.utils.clip_grad_norm_([p], 1e9)
    p.grad = G[:8, :8].clone() / 3
    post = torch.nn.utils.clip_grad_norm_([p], 1e9)
    assert abs(float(post / pre) - 1 / 3) < 1e-5
    assert "does NOT divide" in muon_update_scale_note()


def test_lr_scale_constant_continuation():
    assert lr_scale(100, warmup=0, max_steps=46488, min_lr_frac=1.0) == 1.0


def test_pick_mix_and_bonus_soft_temp_override():
    assert pick_mix_source(0.07, 0.08, 0.20, has_bonus=True, has_deep=True) == "bonus"
    assert pick_mix_source(0.15, 0.08, 0.20, has_bonus=True, has_deep=True) == "deep"
    assert pick_mix_source(0.50, 0.08, 0.20, has_bonus=True, has_deep=True) == "shallow"
    assert pick_mix_source(0.07, 0.08, 0.20, has_bonus=False, has_deep=True) == "deep"
    n = 20_000
    draws = [i / n for i in range(n)]
    counts = {"bonus": 0, "deep": 0, "shallow": 0}
    for d in draws:
        counts[pick_mix_source(d, 0.08, 0.20, has_bonus=True, has_deep=True)] += 1
    assert abs(counts["bonus"] / n - 0.08) < 0.002
    assert abs(counts["deep"] / n - 0.20) < 0.002
    assert abs(counts["shallow"] / n - 0.72) < 0.002
    assert policy_soft_temp_weight(False, 0.4, 0.0) == 0.4
    assert policy_soft_temp_weight(True, 0.4, 0.0) == 0.0
    assert policy_soft_temp_weight(True, 0.4, None) == 0.4
    assert session_throughput(192 * 25, 10.0) == 480.0
    # resume-cumulative 20M / 10s must not be used
    assert session_throughput(192 * 25, 10.0) < 1_000_000


def test_ingest_ready_bonus_shards_drops_seen_and_marks_attached():
    b_e4 = chess.Board()
    b_e4.push_uci("e2e4")
    b_d4 = chess.Board()
    b_d4.push_uci("d2d4")
    e4 = board_to_cache_row(b_e4, chess.Move.from_uci("e7e5"))
    d4 = board_to_cache_row(b_d4, chess.Move.from_uci("d7d5"))
    seen = position_hashes(stack_rows([e4]))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sh = td / "shard_000000"
        sh.mkdir()
        torch.save(stack_rows([e4, d4]), sh / "soft_cache.pt")
        (sh / "READY").write_text("n=2\n")
        chunks, new_seen, reports = ingest_ready_bonus_shards(td, seen)
        assert len(chunks) == 1
        assert int(chunks[0]["board_array"].shape[0]) == 1
        assert reports[0]["n_out"] == 1
        assert reports[0]["vs_seen"] == 1
        assert not (sh / "READY").exists()
        assert (sh / "ATTACHED").exists()
        merged = concat_soft_tables([stack_rows([e4]), chunks[0]])
        assert int(merged["board_array"].shape[0]) == 2
        # second ingest is a no-op
        chunks2, _, reports2 = ingest_ready_bonus_shards(td, new_seen)
        assert chunks2 == []
        assert reports2 == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        raise SystemExit(f"{len(failed)} failed")
    print(f"{len(tests)} passed")
