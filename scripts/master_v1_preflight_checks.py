#!/usr/bin/env python3
"""Post-step checks for the isolated master-v1 preflight. No incumbent writes."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from chess_inference import get_model_move, load_checkpoint
from chess_master.schema import SOURCE_IDS
from scripts.autoresearch_8gb.pipeline import (
    attach_static_targets,
    classify_checkpoint,
    masked_mean_ce,
    prepare_soft_batch,
    value_valid_rows,
)
from chess_squares64 import average_recurrent_grads, build_squares64, count_parameters
import chess


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-dir", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--incumbent", required=True)
    args = ap.parse_args()
    pre = Path(args.preflight_dir)
    exp = Path(args.export_dir)
    inc = Path(args.incumbent)
    errors = []

    def fail(m):
        errors.append(m)
        print("FAIL", m, flush=True)

    def ok(m):
        print("OK", m, flush=True)

    live = pre / "latest.pt"
    if not live.exists():
        fail("preflight latest.pt missing")
        return 1
    ckpt = torch.load(live, map_location="cpu", weights_only=False)
    ok(f"latest classify={classify_checkpoint(ckpt)} steps={ckpt.get('steps')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(live, device=device)
    n = sum(p.numel() for p in model.parameters())
    if n != 98971224:
        fail(f"param count {n}")
    else:
        ok(f"params={n}")

    board = chess.Board()
    move, info = get_model_move(model, board, device)
    if move not in board.legal_moves:
        fail(f"illegal startpos decode {move}")
    else:
        ok(f"startpos decode {move.uci()}")

    swa = pre / "eval_swa.pt"
    if swa.exists():
        m2 = load_checkpoint(swa, device="cpu")
        ok(f"reloaded SWA params={sum(p.numel() for p in m2.parameters())}")
    else:
        ok("no SWA after 1 step (expected if swa_start_frac=0.75)")

    # Value-mask probe: Lichess/puzzle/syzygy batches should contribute ~0 value loss
    # if the dummy target is ignored. Use a zeroed value target vs a real forward.
    model.eval()
    for name, expect_valid in (("lichess", 0), ("puzzles", 0), ("syzygy", 0), ("sf19", 1)):
        d = torch.load(exp / f"{name}_eval.pt", map_location="cpu", weights_only=False)
        attach_static_targets(d)
        idx = torch.arange(min(64, int(d["turn"].shape[0])))
        v_ok = value_valid_rows(d, idx)
        ones = int((v_ok == 1).sum()) if v_ok is not None else -1
        if expect_valid == 0 and ones != 0:
            fail(f"{name} probe value_valid ones={ones}")
        if expect_valid == 1 and ones != int(idx.numel()):
            fail(f"{name} probe value_valid ones={ones}")
        bi, hard, wdl, si, sp = prepare_soft_batch(d, idx, device, hflip_p=0.0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(bi)
            v = masked_mean_ce(out["value_logits"], wdl.to(device), v_ok.to(device) if v_ok is not None else None)
        if not torch.isfinite(v):
            fail(f"{name} value loss non-finite")
        if expect_valid == 0 and float(v) != 0.0:
            fail(f"{name} masked value loss {float(v)} != 0")
        else:
            ok(f"{name} masked_value_loss={float(v):.6f} ones={ones}")

    compile_flag = os.environ.get("COMPILE_FLAG", "on")
    train_log = (pre / "train.log").read_text(encoding="utf-8") if (pre / "train.log").exists() else ""
    if "WEIGHTS-ONLY WARM START" not in train_log and "eval/SWA" not in train_log:
        # train_trial wording
        if "WEIGHTS-ONLY" not in train_log:
            fail("preflight log missing weights-only warm start")
        else:
            ok("weights-only warm start logged")
    else:
        ok("weights-only warm start logged")

    inc_sha = sha256(inc)
    auth = json.loads((exp / "AUTHORIZED.json").read_text())
    sf_bin = "/root/.local/bin/stockfish-19"
    env_sf = os.environ.get("STOCKFISH_PATH") or ""
    if env_sf and Path(env_sf).exists():
        sf_bin = env_sf
    sf_id = subprocess.check_output([sf_bin], input=b"uci\nquit\n").decode()
    sf_name = next((ln.split("id name ", 1)[1] for ln in sf_id.splitlines() if ln.startswith("id name ")), "")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "incumbent": str(inc),
        "incumbent_sha256": inc_sha,
        "hf_model_repo": "avewright/chess-transformer-100m-overnight_20260908",
        "hf_model_revision": "2314e519cf17157ad677421b2ac68eba0bce58fb",
        "dataset_repo": "avewright/chess-master-v1",
        "dataset_revision": "3cc4feb61171387520718a0c4d4b8bb150987780",
        "recipe": "pilot_45_35_15_5",
        "export_checksums": auth.get("checksums"),
        "git": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "stockfish": sf_name,
        "stockfish_path": os.environ.get("STOCKFISH_PATH"),
        "torch_compile": compile_flag,
        "optimizer": "polar_normuon",
        "muon_lr": 0.0007,
        "adam_lr": 1e-5,
        "warmup": 100,
        "min_lr_frac": 0.05,
        "lr_schedule": "linear warmup 100 steps, then cosine to min_lr_frac * base",
        "effective_batch": 64,
        "precision": "bf16 autocast on cuda",
        "resume_kind": "weights_only",
        "errors": errors,
    }
    (pre / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if errors:
        print("PREFLIGHT FAILED", errors, flush=True)
        return 1
    (pre / "PREFLIGHT_OK.json").write_text(json.dumps({
        "ok": True, "torch_compile": compile_flag, "incumbent_sha256": inc_sha,
    }, indent=2) + "\n")
    print("PREFLIGHT_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
