"""Official SF19 WDL rating labels: perspective, no sigmoid fallback, resume."""
import chess
import numpy as np
import torch

from data_loader import compute_wdl
from scripts.sf19_soft_dataset import parse_multipv
from scripts.sf19_wdl_dataset import (
    WDL_SOURCE_TERMINAL,
    WDL_SOURCE_UCI,
    WdlConfig,
    attach_official_wdl,
    label_wdl_row,
    parse_uci_wdl,
    sigmoid_wdl_white,
    stack_wdl_rows,
    terminal_wdl,
    wdl_run_fingerprint,
)


class _Rel:
    def __init__(self, wins, draws, losses):
        self.wins = wins
        self.draws = draws
        self.losses = losses


class _Wdl:
    def __init__(self, wins, draws, losses):
        self.relative = _Rel(wins, draws, losses)


class _Cp:
    def __init__(self, cp):
        self._cp = cp

    def pov(self, _t):
        return self

    def is_mate(self):
        return False

    def score(self, mate_score=None):
        return self._cp


def test_uci_wdl_flips_for_black_to_move():
    raw, probs = parse_uci_wdl(_Wdl(5, 935, 60), turn_black=True)
    assert list(raw) == [60, 935, 5]
    assert abs(float(probs.sum()) - 1.0) < 1e-6
    assert abs(float(probs[0]) - 0.060) < 1e-6
    raw_w, _ = parse_uci_wdl(_Wdl(52, 942, 6), turn_black=False)
    assert list(raw_w) == [52, 942, 6]


def test_missing_or_empty_wdl_is_rejected():
    assert parse_uci_wdl(None, turn_black=False) is None
    assert parse_uci_wdl(_Wdl(0, 0, 0), turn_black=False) is None


def test_terminal_draw_is_exact_not_sigmoid_cp0():
    board = chess.Board("k7/P7/K7/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    raw, probs = terminal_wdl(board)
    assert list(raw) == [0, 1000, 0]
    assert list(probs) == [0.0, 1.0, 0.0]
    sig = compute_wdl(torch.tensor([0]), torch.tensor([0]))[0].numpy()
    assert not np.allclose(sig, [0.0, 1.0, 0.0])
    row = label_wdl_row(board, {"terminal": True}, nodes_budget=0, tau=120.0)
    assert int(row["wdl_source"]) == WDL_SOURCE_TERMINAL
    assert int(row["policy_mask"]) == 0
    assert list(row["wdl"]) == [0.0, 1.0, 0.0]


def test_checkmate_is_white_absolute():
    board = chess.Board("8/8/8/8/8/5K2/6Q1/7k b - - 0 1")
    assert board.is_checkmate()
    row = label_wdl_row(board, {"terminal": True}, nodes_budget=0, tau=120.0)
    assert list(row["wdl"]) == [1.0, 0.0, 0.0]
    assert int(row["mate"]) == 1


def test_label_drops_row_without_official_wdl():
    board = chess.Board()
    parsed = parse_multipv(
        [{"pv": [chess.Move.from_uci("e2e4")], "score": _Cp(27), "depth": 8, "multipv": 1, "nodes": 100}],
        board, k=1, tau=120.0,
    )
    assert parsed is not None
    assert parsed["items"][0].get("wdl") is None
    assert label_wdl_row(board, parsed, nodes_budget=100, tau=120.0) is None


def test_official_wdl_is_not_the_project_sigmoid():
    official = np.array([52, 942, 6], dtype=np.float32) / 1000.0
    sig = sigmoid_wdl_white(27, 0)
    assert float(np.abs(official - sig).sum()) > 0.5
    assert official[1] > 0.9
    assert sig[1] < 0.6


def test_attach_official_wdl_and_stack():
    board = chess.Board()
    infos = [{
        "pv": [chess.Move.from_uci("e2e4")],
        "score": _Cp(27),
        "depth": 10,
        "multipv": 1,
        "nodes": 5000,
        "wdl": _Wdl(52, 942, 6),
    }]
    parsed = parse_multipv(infos, board, k=1, tau=120.0)
    parsed = attach_official_wdl(parsed, infos, board)
    row = label_wdl_row(board, parsed, nodes_budget=5000, tau=120.0)
    assert int(row["wdl_source"]) == WDL_SOURCE_UCI
    assert list(row["wdl_raw"]) == [52, 942, 6]
    assert abs(float(row["wdl"].sum()) - 1.0) < 1e-5
    assert int(row["n_pieces"]) == 32
    assert int(row["n_soft"]) == 1
    data = stack_wdl_rows([row])
    assert data["wdl_raw"].shape == (1, 3)
    assert int(data["wdl_source"][0]) == WDL_SOURCE_UCI
    assert torch.allclose(data["wdl"][0], torch.tensor(row["wdl"]))


def test_wdl_parquet_keeps_raw_source():
    import pyarrow.parquet as pq
    from scripts.sf19_wdl_dataset import wdl_chunk_table

    board = chess.Board()
    infos = [{
        "pv": [chess.Move.from_uci("e2e4")],
        "score": _Cp(27),
        "depth": 10,
        "multipv": 1,
        "nodes": 5000,
        "wdl": _Wdl(52, 942, 6),
    }]
    parsed = attach_official_wdl(parse_multipv(infos, board, k=1, tau=120.0), infos, board)
    row = label_wdl_row(board, parsed, nodes_budget=5000, tau=120.0)
    data = stack_wdl_rows([row])
    table = wdl_chunk_table(data, "t", 0, 1)
    assert list(table.column("wdl_raw")[0].as_py()) == [52, 942, 6]
    assert int(table.column("wdl_source")[0].as_py()) == WDL_SOURCE_UCI


def test_local_wdl_readme_renders():
    from scripts.sf19_wdl_dataset import _local_wdl_readme
    text = _local_wdl_readme(
        "avewright/local-wdl",
        1000,
        {"n_starts": 10, "volumes": {"A": 1}},
        {"config": {"nodes": 25000, "play_nodes": 2000}, "run": {"binary_sha256": "abc"}},
        {"accepted": 1000, "rejected": {"analyze_fail": 0, "no_wdl": 0}},
    )
    assert "pretty_name: Local WDL" in text
    assert "from datasets import load_dataset" in text
    assert "analyze_fail=0" in text
    assert "White-absolute" in text


def test_wdl_fingerprint_changes_with_nodes():
    fp = {"uci_name": "Stockfish 19", "binary_sha256": "abc", "eval_file": "nn.nnue"}
    a = wdl_run_fingerprint(fp, WdlConfig(nodes=25_000))
    b = wdl_run_fingerprint(fp, WdlConfig(nodes=50_000))
    assert a["wdl_required"] == 1
    assert a["multipv"] == 1
    assert a["fingerprint_id"] != b["fingerprint_id"]


def test_wdl_shard_cursor_starts_after_hf(tmp_path):
    from scripts.sf19_wdl_dataset import next_wdl_shard_dir
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (tmp_path / "shard_cursor").write_text("200")
    first = next_wdl_shard_dir(inbox)
    assert first.name == "shard_000200"
    first.mkdir()
    assert next_wdl_shard_dir(inbox).name == "shard_000201"


def test_resume_rejects_other_fingerprint(tmp_path):
    from scripts.sf19_soft_dataset import assert_resume_compatible
    (tmp_path / "teacher.json").write_text('{"fingerprint_id": "wdl-aaa"}')
    try:
        assert_resume_compatible(tmp_path, {"fingerprint_id": "wdl-bbb"})
    except SystemExit as exc:
        assert "fingerprint" in str(exc).lower()
    else:
        raise AssertionError("expected SystemExit")
    assert_resume_compatible(tmp_path, {"fingerprint_id": "wdl-aaa"})
