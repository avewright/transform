"""CPU tests for exp276 inbox flush + frozen-val absorb."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import torch

from build_lichess_evals_soft_cache import write_inbox_shard  # noqa: E402
from exp276_lichess_endgame_stream import (  # noqa: E402
    absorb_ready,
    ckpt_step,
    inbox_cache_shards,
    pack_initial,
    reload_train_capped,
)
from push_lichess_endgame_bestline_hf import convert_shard, inbox_shards, readme_text  # noqa: E402
from upload_exp276_hf import BLOCKED, DEFAULT_REPO, refuse_blocked  # noqa: E402


def _acc(fen: str, uci: str) -> dict:
    return {fen: (40, 8000, 200, uci)}


def test_push_convert_shard_roundtrip():
    import shutil
    import tempfile

    from build_lichess_evals_soft_cache import write_inbox_shard

    root = Path(tempfile.mkdtemp(prefix="eg_push_"))
    inbox = root / "inbox"
    fen = "6k1/4Rppp/8/8/8/8/5PPP/6K1 w - -"
    sh = write_inbox_shard(inbox, {fen: (40, 8000, 200, "e7e8")}, one_hot=True, tau=120.0)
    dest = root / "out.parquet"
    n = convert_shard(sh, dest)
    assert n == 1
    assert dest.exists()
    assert inbox_shards(inbox) == [sh]
    card = readme_text("avewright/lichess-endgame-bestline", 1, 1)
    assert "fewer than 14" in card
    shutil.rmtree(root)


def test_upload_refuses_incumbent():
    assert DEFAULT_REPO == "avewright/endgame-model"
    assert "avewright/chess-transformer-100m-squares64" in BLOCKED
    try:
        refuse_blocked("avewright/puzzle-model")
        raise AssertionError("should refuse")
    except SystemExit:
        pass


def test_ckpt_step_reads_steps(tmp_path=None):
    import tempfile

    d = Path(tempfile.mkdtemp(prefix="exp276_ckpt_"))
    p = d / "latest.pt"
    torch.save({"steps": 200, "status": "trained"}, p)
    assert ckpt_step(p) == 200
    torch.save({"step": 7}, p)
    assert ckpt_step(p) == 7


def test_flush_writes_ready_onehot(tmp_path=None):
    inbox = Path(tmp_path) if tmp_path else ROOT / "outputs" / "_tmp_exp276_test_inbox"
    if tmp_path is None:
        if inbox.exists():
            import shutil
            shutil.rmtree(inbox)
        inbox.mkdir(parents=True)
    fen = "6k1/4Rppp/8/8/8/8/5PPP/6K1 w - -"
    sh = write_inbox_shard(inbox, _acc(fen, "e7e8"), one_hot=True, tau=120.0)
    assert (sh / "READY").exists()
    t = torch.load(sh / "soft_cache.pt", map_location="cpu", weights_only=False)
    assert int(t["turn"].shape[0]) == 1
    assert abs(float(t["soft_probs"][0, 0]) - 1.0) < 1e-6
    assert int((t["soft_indices"][0] >= 0).sum()) == 1
    if tmp_path is None:
        import shutil
        shutil.rmtree(inbox)


def _walk_fens(n: int) -> list[tuple[str, str]]:
    import chess

    board = chess.Board("8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -")
    out: list[tuple[str, str]] = []
    while len(out) < n and not board.is_game_over():
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = legal[0]
        out.append((board.fen(), mv.uci()))
        board.push(mv)
    return out


def test_pack_and_absorb_no_val_leak():
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="exp276_"))
    inbox = root / "inbox"
    out = root / "out"
    rows = _walk_fens(40)
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
