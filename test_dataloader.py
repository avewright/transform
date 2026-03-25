"""Quick validation: does avewright/chess-positions-lichess-sf work with the dataloader?"""

import time
import chess
import torch
from datasets import load_dataset
from move_vocab import UCI_TO_IDX, IDX_TO_UCI, VOCAB_SIZE
from chess_features import NUM_FUSED_TOKENS, NUM_CASTLING_STATES, NUM_EP_STATES
from data_loader import (
    load_training_data, board_array_to_fused, board_array_to_baseline,
    ep_square_to_file, compute_wdl, get_batch_input
)

def test_raw_hf():
    print("=" * 60)
    print("TEST 1: Raw HF streaming - check schema & values")
    print("=" * 60)
    ds = load_dataset("avewright/chess-positions-lichess-sf", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if i >= 99:
            break

    print(f"Loaded {len(rows)} rows")
    print(f"Schema keys: {sorted(rows[0].keys())}")
    r = rows[0]
    print(f"\nSample row:")
    print(f"  fen:        {r['fen']}")
    print(f"  best_move:  {r['best_move']}")
    print(f"  eval_type:  {r['eval_type']}")
    print(f"  eval_value: {r['eval_value']}")
    print(f"  depth:      {r['depth']}")
    print(f"  phase:      {r['phase']}")
    print(f"  num_legal:  {r['num_legal']}")
    print(f"  wdl:        ({r['wdl_win']:.3f}, {r['wdl_draw']:.3f}, {r['wdl_loss']:.3f})")

    errors = []
    for i, r in enumerate(rows):
        # FEN should parse
        try:
            board = chess.Board(r["fen"])
        except Exception as e:
            errors.append(f"Row {i}: bad FEN: {e}")
            continue
        # best_move should be in vocab and legal
        best_uci = r["best_move"]
        if best_uci not in UCI_TO_IDX:
            errors.append(f"Row {i}: best_move '{best_uci}' not in move vocab")
            continue
        move = chess.Move.from_uci(best_uci)
        if move not in board.legal_moves:
            errors.append(f"Row {i}: move {best_uci} not legal in position")
        # eval_type
        if r["eval_type"] not in ("cp", "mate"):
            errors.append(f"Row {i}: bad eval_type={r['eval_type']}")
        # depth > 0
        if r["depth"] <= 0:
            errors.append(f"Row {i}: depth={r['depth']} <= 0")

    if errors:
        print(f"\n*** {len(errors)} ERRORS ***")
        for e in errors[:10]:
            print(f"  {e}")
    else:
        print(f"\nAll 100 rows PASS raw schema & value checks!")
    return len(errors) == 0


def test_load_training_data():
    print("\n" + "=" * 60)
    print("TEST 2: load_training_data() with small sample")
    print("=" * 60)
    t0 = time.time()
    train_tensors, eval_data, eval_tensors = load_training_data(
        n_train=500, n_eval=50, encoder_type="both", seed=42
    )
    elapsed = time.time() - t0
    print(f"Loaded in {elapsed:.1f}s")

    print(f"\ntrain_tensors keys: {sorted(train_tensors.keys())}")
    for k, v in sorted(train_tensors.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min().item()}, {v.max().item()}]")

    print(f"\neval_tensors keys: {sorted(eval_tensors.keys())}")
    for k, v in sorted(eval_tensors.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min().item()}, {v.max().item()}]")

    print(f"\neval_data: {len(eval_data)} items")
    if eval_data:
        e = eval_data[0]
        print(f"  sample: fen={e['board'].fen()[:40]}...")
        print(f"  move={e['move'].uci()}, wdl={e['wdl']}, phase={e['phase']}")

    # Validate tensor constraints
    print("\nValidating tensor constraints...")
    asserts = []

    fi = train_tensors["fused_ids"]
    if fi.min() < 0 or fi.max() >= NUM_FUSED_TOKENS:
        asserts.append(f"fused_ids range [{fi.min()}, {fi.max()}] outside [0, {NUM_FUSED_TOKENS})")

    t = train_tensors["turn"]
    if t.min() < 0 or t.max() > 1:
        asserts.append(f"turn range [{t.min()}, {t.max()}] outside [0,1]")

    c = train_tensors["castling"]
    if c.min() < 0 or c.max() >= NUM_CASTLING_STATES:
        asserts.append(f"castling range [{c.min()}, {c.max()}] outside [0,{NUM_CASTLING_STATES})")

    ep = train_tensors["ep_file"]
    if ep.min() < 0 or ep.max() >= NUM_EP_STATES:
        asserts.append(f"ep_file range [{ep.min()}, {ep.max()}] outside [0,{NUM_EP_STATES})")

    mi = train_tensors["move_idx"]
    if mi.min() < 0 or mi.max() >= VOCAB_SIZE:
        asserts.append(f"move_idx range [{mi.min()}, {mi.max()}] outside [0,{VOCAB_SIZE})")

    wdl = train_tensors["wdl"]
    wdl_sum = wdl.sum(dim=1)
    if (wdl_sum - 1.0).abs().max() > 0.01:
        asserts.append(f"wdl rows dont sum to 1.0, max dev={(wdl_sum - 1.0).abs().max():.4f}")

    bp = train_tensors["baseline_piece_ids"]
    bc = train_tensors["baseline_color_ids"]
    if bp.min() < 0 or bp.max() >= 7:
        asserts.append(f"baseline_piece_ids range [{bp.min()}, {bp.max()}] outside [0,7)")
    if bc.min() < 0 or bc.max() >= 3:
        asserts.append(f"baseline_color_ids range [{bc.min()}, {bc.max()}] outside [0,3)")

    if asserts:
        print(f"*** {len(asserts)} FAILURES ***")
        for a in asserts:
            print(f"  {a}")
    else:
        print("All tensor constraints PASS!")
    return len(asserts) == 0, train_tensors


def test_batch_input(train_tensors):
    print("\n" + "=" * 60)
    print("TEST 3: get_batch_input smoke test")
    print("=" * 60)
    device = torch.device("cpu")
    indices = torch.arange(min(16, train_tensors["fused_ids"].shape[0]))

    batch = get_batch_input(train_tensors, indices, "fused", device)
    print(f"Batch keys (fused): {sorted(batch.keys())}")
    for k, v in sorted(batch.items()):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    batch_bl = get_batch_input(train_tensors, indices, "baseline", device)
    print(f"Batch keys (baseline): {sorted(batch_bl.keys())}")
    for k, v in sorted(batch_bl.items()):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    print("get_batch_input() works for both encoder types!")
    return True


if __name__ == "__main__":
    ok1 = test_raw_hf()
    ok2, train_tensors = test_load_training_data()
    ok3 = test_batch_input(train_tensors)

    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
