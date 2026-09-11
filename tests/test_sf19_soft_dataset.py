"""SF19 soft-target scoring, encoding, and resume contracts."""
import json
import random

import chess
import numpy as np
import torch

from scripts.sf19_soft_dataset import (
    SOFT_K,
    encode_board,
    label_to_row,
    mate_rank_score,
    parse_multipv,
    softmax_from_scores,
    stm_rank_score,
    to_white_abs,
)


def test_softmax_stable_and_normalized():
    p = softmax_from_scores([1200.0, 1200.0, -400.0], 120.0)
    assert abs(sum(p) - 1) < 1e-6
    assert p[0] == p[1]
    assert p[0] > p[2]


def test_mate_rank_prefers_faster_wins_and_slower_losses():
    assert mate_rank_score(1) > mate_rank_score(8)
    assert mate_rank_score(-8) > mate_rank_score(-1)
    assert stm_rank_score(0, 1) > stm_rank_score(9000, 0)
    assert stm_rank_score(0, -1) < stm_rank_score(-9000, 0)


def test_white_abs_flips_for_black():
    assert to_white_abs(80, 0, False) == (80, 0)
    assert to_white_abs(80, 0, True) == (-80, 0)
    assert to_white_abs(0, 3, True) == (0, -3)


def test_encode_start_and_ep_castling():
    board = chess.Board()
    arr, turn, castling, ep = encode_board(board)
    assert turn == 0
    assert castling == 15
    assert ep == -1
    assert int(arr[chess.E2]) == 1
    # python-chess omits EP when no legal capture exists; legal EP must survive.
    fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    arr2, turn2, _, ep2 = encode_board(chess.Board(fen))
    assert turn2 == 0
    assert ep2 == chess.D6


def test_parse_skips_bound_scores_and_pads():
    board = chess.Board()
    e2e4 = chess.Move.from_uci("e2e4")
    d2d4 = chess.Move.from_uci("d2d4")

    class Score:
        def __init__(self, cp):
            self._cp = cp

        def pov(self, _turn):
            return self

        def is_mate(self):
            return False

        def score(self, mate_score=None):
            return self._cp

    infos = [
        {"pv": [e2e4], "score": Score(40), "depth": 12, "nodes": 1000},
        {"pv": [d2d4], "score": Score(30), "depth": 12, "upperbound": True},
    ]
    parsed = parse_multipv(infos, board, k=1, tau=120.0)
    assert parsed["bound_skipped"] == 1
    assert parsed["items"][0]["uci"] == "e2e4"
    row = label_to_row(board, parsed, tau=120.0, nodes_budget=1000)
    assert row["soft_indices"][1] == -1
    assert row["soft_probs"][1] == 0
    assert row["policy_mask"] == 1
    assert row["soft_indices"].shape == (SOFT_K,)


def test_terminal_masks_policy():
    board = chess.Board("8/8/8/8/8/5K2/6Q1/7k b - - 0 1")
    assert board.is_checkmate()
    row = label_to_row(board, {"terminal": True}, tau=120.0, nodes_budget=0)
    assert int(row["policy_mask"]) == 0
    assert int(row["move_idx"]) == -1
    assert int(row["mate"]) == 1  # White delivered mate; stored White-absolute


def test_inbox_state_counts_ready_shards(tmp_path):
    from scripts.sf19_soft_dataset import inbox_state, write_shard, stack_rows

    inbox = tmp_path / "inbox"
    board = chess.Board()
    e2e4 = chess.Move.from_uci("e2e4")

    class Score:
        def __init__(self, cp):
            self._cp = cp
        def pov(self, _t):
            return self
        def is_mate(self):
            return False
        def score(self, mate_score=None):
            return self._cp

    parsed = parse_multipv(
        [{"pv": [e2e4], "score": Score(30), "depth": 8, "nodes": 100}],
        board, k=1, tau=120.0,
    )
    row = label_to_row(board, parsed, tau=120.0, nodes_budget=100)
    row["game_id"] = np.int64(7)
    row["ply"] = np.int16(2)
    row["split"] = np.int8(0)
    write_shard(stack_rows([row]), inbox / "shard_000000", {"n": 1, "nodes": 10000, "multipv": 4, "tau": 120})
    n, next_game = inbox_state(inbox)
    assert n == 1
    assert next_game == 8


def test_probs_sum_and_legal_indices():
    from move_vocab import UCI_TO_IDX
    board = chess.Board()
    e2e4 = chess.Move.from_uci("e2e4")
    d2d4 = chess.Move.from_uci("d2d4")

    class Score:
        def __init__(self, cp):
            self._cp = cp
        def pov(self, _t):
            return self
        def is_mate(self):
            return False
        def score(self, mate_score=None):
            return self._cp

    parsed = parse_multipv(
        [
            {"pv": [e2e4], "score": Score(30), "depth": 10},
            {"pv": [d2d4], "score": Score(20), "depth": 10},
        ],
        board, k=2, tau=120.0,
    )
    row = label_to_row(board, parsed, tau=120.0, nodes_budget=10)
    assert abs(float(row["soft_probs"].sum()) - 1) < 1e-5
    assert int(row["soft_indices"][0]) == UCI_TO_IDX["e2e4"]
    assert int(row["cp"]) == 30  # white to move, white-abs == stm


class _Cp:
    def __init__(self, cp):
        self._cp = cp
    def pov(self, _t):
        return self
    def is_mate(self):
        return False
    def score(self, mate_score=None):
        return self._cp


def test_parse_keeps_last_complete_iteration_not_newer_partial():
    from scripts.sf19_soft_dataset import last_complete_iteration
    board = chess.Board()
    moves = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1f3", "c2c4")]
    infos = []
    for i, mv in enumerate(moves, 1):
        infos.append({"pv": [mv], "score": _Cp(40 - i), "depth": 10, "multipv": i, "nodes": 800})
    infos.append({"pv": [moves[0]], "score": _Cp(90), "depth": 11, "multipv": 1, "nodes": 1000})
    infos.append({"pv": [moves[1]], "score": _Cp(80), "depth": 11, "multipv": 2, "nodes": 1000})
    parsed = parse_multipv(infos, board, k=4, tau=120.0)
    assert parsed["depth"] == 10
    assert parsed["complete_iteration"] is True
    assert {it["uci"] for it in parsed["items"]} == {m.uci() for m in moves}
    assert parsed["items"][0]["stm_cp"] != 90
    selected = last_complete_iteration(infos, board, k=4)
    assert selected is not None
    assert selected[1] == 10


def test_union_kl_penalizes_missing_moves_and_regret_uses_ref_of_pick():
    from scripts.sf19_soft_dataset import ref_move_regret, union_kl_and_coverage
    from move_vocab import UCI_TO_IDX

    def _row(pairs, wdl=(0.4, 0.4, 0.2)):
        si = np.full(8, -1, dtype=np.int64)
        sp = np.zeros(8, dtype=np.float32)
        sc = np.zeros(8, dtype=np.int32)
        sm = np.zeros(8, dtype=np.int32)
        for i, (uci, pr, cp) in enumerate(pairs):
            si[i] = UCI_TO_IDX[uci]
            sp[i] = pr
            sc[i] = cp
        return {
            "soft_indices": si, "soft_probs": sp, "soft_cps": sc, "soft_mates": sm,
            "move_idx": si[0], "wdl": np.array(wdl, dtype=np.float32),
        }

    ref = _row([("e2e4", 0.7, 40), ("d2d4", 0.3, 20)])
    cand = _row([("e2e4", 1.0, 15)])
    kl_missing, ref_in_q, _ = union_kl_and_coverage(ref, cand)
    cand_full = _row([("e2e4", 0.7, 40), ("d2d4", 0.3, 20)])
    kl_full, ref_in_full, _ = union_kl_and_coverage(ref, cand_full)
    assert kl_missing > kl_full
    assert ref_in_q < ref_in_full
    cand_d4 = _row([("d2d4", 1.0, 20)])
    regret, missing = ref_move_regret(ref, cand_d4)
    assert missing is False
    assert regret == 20  # ref best 40 minus ref(d4)=20


def test_seen_db_is_disk_backed_without_full_mem_set(tmp_path):
    from scripts.sf19_soft_dataset import SeenDB, compact_key_bytes
    db = SeenDB(tmp_path / "seen.sqlite")
    assert not hasattr(db, "mem")
    keys = [compact_key_bytes(np.zeros(64, dtype=np.int8), 0, 15, -1),
            compact_key_bytes(np.ones(64, dtype=np.int8), 1, 0, 12)]
    db.add_many(keys)
    assert db.has(keys[0])
    assert db.has_many(keys) == set(keys)
    ro = SeenDB(tmp_path / "seen.sqlite", readonly=True)
    assert ro.has(keys[1])
    assert len(ro) == 2


def test_resume_rejects_fingerprint_mismatch(tmp_path):
    from scripts.sf19_soft_dataset import GenConfig, assert_resume_compatible
    (tmp_path / "teacher.json").write_text(
        json.dumps({"fingerprint_id": "aaa", "run": {"fingerprint_id": "aaa"}}),
        encoding="utf-8",
    )
    try:
        assert_resume_compatible(tmp_path, {"fingerprint_id": "bbb"})
    except SystemExit as e:
        assert "fingerprint" in str(e).lower()
    else:
        raise AssertionError("expected SystemExit")
    assert_resume_compatible(tmp_path, {"fingerprint_id": "aaa"})


def test_sf19_parquet_round_trip_keeps_wdl_scores_and_split(tmp_path):
    import pyarrow.parquet as pq
    from scripts.export_soft_caches_to_hf import sf19_chunk_table, sf19_table_to_cache
    from scripts.sf19_soft_dataset import stack_rows

    board = chess.Board()
    parsed = parse_multipv(
        [
            {"pv": [chess.Move.from_uci("e2e4")], "score": _Cp(30), "depth": 8, "multipv": 1, "nodes": 100},
            {"pv": [chess.Move.from_uci("d2d4")], "score": _Cp(20), "depth": 8, "multipv": 2, "nodes": 100},
        ],
        board, k=2, tau=120.0,
    )
    row = label_to_row(board, parsed, tau=120.0, nodes_budget=10000)
    row["game_id"] = np.int64(3)
    row["ply"] = np.int16(6)
    row["split"] = np.int8(1)
    data = stack_rows([row])
    table = sf19_chunk_table(data, "t", 0, 1)
    dest = tmp_path / "t.parquet"
    pq.write_table(table, dest)
    back = sf19_table_to_cache(pq.read_table(dest))
    assert back["game_id"][0] == 3
    assert back["split"][0] == 1
    assert back["policy_mask"][0] == 1
    assert torch.allclose(back["wdl"], data["wdl"])
    assert torch.equal(back["soft_cps"], data["soft_cps"])
    assert torch.equal(back["soft_indices"], data["soft_indices"])
    assert int(back["origin"][0]) == 0
    assert int(row["flags"]) & 0 == 0


def test_flags_mark_shallow_and_decided():
    from scripts.sf19_soft_dataset import FLAG_DECIDED, FLAG_SHALLOW, compute_flags
    shallow = {"label_depth": 4, "policy_mask": 1, "mate": 0, "cp": 10, "bound_skipped": 0}
    assert compute_flags(shallow, {"bound_skipped": 0}) & FLAG_SHALLOW
    decided = {"label_depth": 12, "policy_mask": 1, "mate": 0, "cp": 900, "bound_skipped": 0}
    assert compute_flags(decided) & FLAG_DECIDED


def test_should_push_rows_on_50k_landmarks():
    from scripts.sf19_soft_dataset import should_push_rows
    assert should_push_rows(140_000, 135_000, 50_000, False) is False
    assert should_push_rows(149_999, 135_000, 50_000, False) is False
    assert should_push_rows(150_000, 135_000, 50_000, False) is True
    assert should_push_rows(200_000, 150_000, 50_000, False) is True
    assert should_push_rows(136_000, 135_000, 50_000, True) is True
    assert should_push_rows(135_000, 135_000, 50_000, True) is False


def test_eval_bucket_and_existing_sample_skips_adjacent(tmp_path):
    from scripts.sf19_soft_prod import eval_bucket, sample_existing_specs
    from scripts.sf19_soft_dataset import stack_rows
    assert eval_bucket(-200, 0) == "losing"
    assert eval_bucket(20, 0) == "equal"
    board = chess.Board()
    parsed = parse_multipv(
        [{"pv": [chess.Move.from_uci("e2e4")], "score": _Cp(30), "depth": 10, "multipv": 1, "nodes": 50}],
        board, k=1, tau=120.0,
    )
    row = label_to_row(board, parsed, tau=120.0, nodes_budget=100)
    data = stack_rows([row, row, row])
    cache = tmp_path / "c.pt"
    torch.save(data, cache)
    specs = sample_existing_specs([cache], 2, random.Random(0), stride=2)
    assert len(specs) <= 2
