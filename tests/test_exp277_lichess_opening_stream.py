"""CPU tests for exp277 opening inbox flush + frozen-val absorb."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import torch

from build_lichess_evals_soft_cache import pieces_fen, write_inbox_shard  # noqa: E402
from exp277_lichess_opening_stream import (  # noqa: E402
    MIN_PCS,
    absorb_ready,
    ckpt_step,
    inbox_cache_shards,
    pack_initial,
    reload_train_capped,
)
from push_lichess_opening_bestline_hf import convert_shard, inbox_shards, readme_text  # noqa: E402
from upload_exp277_hf import BLOCKED, DEFAULT_REPO, refuse_blocked  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def _acc(fen: str, uci: str) -> dict:
    return {fen: (40, 8000, 200, uci)}


def test_start_fen_is_opening():
    assert pieces_fen(START_FEN) == 32
    assert pieces_fen(START_FEN) >= MIN_PCS


def test_push_convert_shard_roundtrip():
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="op_push_"))
    inbox = root / "inbox"
    sh = write_inbox_shard(inbox, {START_FEN: (40, 8000, 200, "e2e4")}, one_hot=True, tau=120.0)
    dest = root / "out.parquet"
    n = convert_shard(sh, dest)
    assert n == 1
    assert dest.exists()
    assert inbox_shards(inbox) == [sh]
    card = readme_text("avewright/lichess-opening-bestline", 1, 1)
    assert "26 or more" in card
    shutil.rmtree(root)


def test_upload_refuses_incumbent():
    assert DEFAULT_REPO == "avewright/opening-model"
    assert "avewright/chess-transformer-100m-squares64" in BLOCKED
    assert "avewright/endgame-model" in BLOCKED
    try:
        refuse_blocked("avewright/puzzle-model")
        raise AssertionError("should refuse")
    except SystemExit:
        pass


def test_ckpt_step_reads_steps():
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="exp277_ckpt_"))
    p = d / "latest.pt"
    torch.save({"steps": 200, "status": "trained"}, p)
    assert ckpt_step(p) == 200
    torch.save({"step": 7}, p)
    assert ckpt_step(p) == 7


def test_flush_writes_ready_onehot():
    import shutil
    import tempfile

    inbox = Path(tempfile.mkdtemp(prefix="exp277_inbox_"))
    sh = write_inbox_shard(inbox, _acc(START_FEN, "e2e4"), one_hot=True, tau=120.0)
    assert (sh / "READY").exists()
    t = torch.load(sh / "soft_cache.pt", map_location="cpu", weights_only=False)
    assert int(t["turn"].shape[0]) == 1
    assert abs(float(t["soft_probs"][0, 0]) - 1.0) < 1e-6
    assert int((t["soft_indices"][0] >= 0).sum()) == 1
    shutil.rmtree(inbox)


def _walk_opening(n: int) -> list[tuple[str, str]]:
    import chess

    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    stack = [chess.Board()]
    while stack and len(out) < n:
        board = stack.pop()
        if sum(1 for _ in board.piece_map()) < MIN_PCS:
            continue
        fen = board.fen()
        if fen in seen:
            continue
        seen.add(fen)
        quiet = [m for m in board.legal_moves if not board.is_capture(m)]
        legal = quiet or list(board.legal_moves)
        if not legal:
            continue
        out.append((fen, legal[0].uci()))
        for mv in legal[:4]:
            nxt = board.copy(stack=False)
            nxt.push(mv)
            stack.append(nxt)
    return out


def test_pack_and_absorb_no_val_leak():
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="exp277_"))
    inbox = root / "inbox"
    out = root / "out"
    rows = _walk_opening(40)
    assert len(rows) >= 30
    assert all(pieces_fen(fen) >= MIN_PCS for fen, _ in rows)
    for fen, uci in rows[:25]:
        write_inbox_shard(inbox, _acc(fen, uci), one_hot=True, tau=120.0)
    pack_initial(out, inbox, val_n=2)
    val_n = int(torch.load(out / "lichess_eval.pt", map_location="cpu", weights_only=False)["turn"].shape[0])
    assert val_n == 2
    for fen, uci in rows[25:]:
        write_inbox_shard(inbox, _acc(fen, uci), one_hot=True, tau=120.0)
    before = int(torch.load(out / "lichess_train.pt", map_location="cpu", weights_only=False)["turn"].shape[0])
    absorb_ready(out, inbox)
    after = int(torch.load(out / "lichess_train.pt", map_location="cpu", weights_only=False)["turn"].shape[0])
    val_n2 = int(torch.load(out / "lichess_eval.pt", map_location="cpu", weights_only=False)["turn"].shape[0])
    assert val_n2 == val_n
    assert after >= before
    assert inbox_cache_shards(inbox)
    reload_train_capped(out, inbox, cap=3)
    n = int(torch.load(out / "lichess_train.pt", map_location="cpu", weights_only=False)["turn"].shape[0])
    assert 1 <= n <= 3
    shutil.rmtree(root)
