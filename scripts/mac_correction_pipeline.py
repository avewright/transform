#!/usr/bin/env python3
"""Bounded, resumable inference-and-correction loop for the Mac.

Phases:
  sample   Stratified 100k from the frozen organized mix (no holdouts).
  scan     Incumbent greedy-legal policy on MPS. Writes every sampled row.
  rank     Informative disagreements, plus uncertain and control strata.
  verify   Deeper SF19 on a 5–10k queue. Dual root-move eval. No invented regret.
  assemble Freeze a unique correction bucket. 10–15% sampling is a mix weight,
           not a reason to pad with OK or replace Lichess/puzzles/Syzygy.
  rescan   New checkpoint on the same 100k. Keep remaining weaknesses only.

Does not mutate outputs/organized_chess_v1. Does not start training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

from build_organized_chess_mix import (
    PHASE_FRAC,
    SOURCE_LICHESS,
    SOURCE_PUZZLE,
    SOURCE_SF19,
    SOURCE_SYZYGY,
    canonical_hashes,
    json_write,
    squeeze_scalars,
)
from build_organized_correction_variant import (
    SOURCE_MISTAKE,
    ensure_value_mask,
    link_or_copy,
    replace_sf19,
)
from chess_inference import load_checkpoint
from data_loader import board_array_to_fused, compute_wdl, ep_square_to_file
from harvest_hf100m_bulk import _write_shard
from harvest_swa_mistakes import (
    I_TO_TAG,
    TAG_TO_I,
    _pick_legal,
    board_array_to_fen,
    classify_lapse,
    to_stm,
)
from move_vocab import VOCAB_SIZE, index_to_move

BASELINE = ROOT / "outputs/organized_chess_v1"
DEFAULT_CKPT = ROOT / "outputs/sf19_ft/overnight_20260908/eval_swa.pt"
DEFAULT_OUT = ROOT / "outputs/mac_correction_v1"
SUBSTANTIAL = frozenset({
    TAG_TO_I["inaccuracy"], TAG_TO_I["blunder"],
    TAG_TO_I["conversion"], TAG_TO_I["major"],
})
SOURCE_FILES = (
    ("sf19", SOURCE_SF19, "sf19_train.pt"),
    ("lichess", SOURCE_LICHESS, "lichess_train.pt"),
    ("puzzles", SOURCE_PUZZLE, "puzzles_train.pt"),
    ("syzygy", SOURCE_SYZYGY, "syzygy_train.pt"),
)
SOURCE_QUOTA = {
    SOURCE_SF19: 0.45,
    SOURCE_LICHESS: 0.35,
    SOURCE_PUZZLE: 0.15,
    SOURCE_SYZYGY: 0.05,
}
EVAL_BUCKETS = ("losing", "equal", "winning", "mate_win", "mate_lose", "masked", "sentinel")
REGRET_UNKNOWN = 0
REGRET_KNOWN = 1
UNCERTAIN_LO, UNCERTAIN_HI = 0.12, 0.45
VERIFY_PROTOCOL = {
    "engine": "Stockfish 19",
    "nodes_budget": 100_000,
    "multipv": 8,
    "tau": 120.0,
    "threads": 1,
    "hash_mb": 64,
    "watchdog_s": 45.0,
    "teacher_eval": "root_moves=[teacher] at nodes_budget",
    "model_eval": "root_moves=[model] at nodes_budget",
    "multipv_refresh": "analyze_board MultiPV-8 for soft pack and in-PV check",
    "unknown_preverify": (
        "If the model move is outside the stored MultiPV list, regret is unknown. "
        "drop_cp stays 0 until explicit SF evaluation. No invented penalty."
    ),
}


def log(msg: str, path: Path | None = None) -> None:
    print(msg, flush=True)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pt(path: Path) -> dict:
    return squeeze_scalars(torch.load(path, map_location="cpu", weights_only=False))


def blocked_hashes(baseline: Path) -> np.ndarray:
    chunks = []
    man = baseline / "blocked_manifest.json"
    if man.exists():
        raw = json.loads(man.read_text()).get("blocked_hashes") or []
        if raw:
            chunks.append(np.asarray(raw, dtype=np.uint64))
    extra = ROOT / "outputs/sf19_ft/overnight_20260908"
    for name in ("val_manifest_soft.json", "val_manifest_deep.json", "val_manifest_replay.json"):
        p = extra / name
        if not p.exists():
            continue
        raw = json.loads(p.read_text()).get("blocked_hashes") or json.loads(p.read_text()).get("hashes") or []
        if raw:
            chunks.append(np.asarray(raw, dtype=np.uint64))
    if not chunks:
        raise SystemExit(f"no blocked hashes at {man}")
    return np.unique(np.concatenate(chunks))


def eval_bucket(cp: int, mate: int, turn: int, value_valid: int, source: int) -> str:
    if int(source) in (SOURCE_PUZZLE, SOURCE_SYZYGY) or int(value_valid) == 0:
        if abs(int(cp)) >= 90_000:
            return "sentinel"
        if int(source) in (SOURCE_PUZZLE, SOURCE_SYZYGY):
            return "masked"
        if int(value_valid) == 0 and int(source) == SOURCE_LICHESS:
            return "sentinel" if abs(int(cp)) >= 90_000 else "masked"
        return "masked"
    stm_cp, stm_mate = to_stm(int(cp), int(mate), int(turn), white_abs=True)
    if stm_mate > 0:
        return "mate_win"
    if stm_mate < 0:
        return "mate_lose"
    if stm_cp <= -150:
        return "losing"
    if stm_cp >= 150:
        return "winning"
    return "equal"


def teacher_p_and_in_pv(soft_i: np.ndarray, soft_p: np.ndarray, model_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    match = soft_i == model_idx.reshape(-1, 1)
    in_pv = match.any(axis=1)
    slot = np.argmax(match, axis=1)
    teacher_p = np.where(in_pv, soft_p[np.arange(soft_i.shape[0]), slot], 0.0).astype(np.float32)
    return teacher_p, in_pv.astype(np.int8)


def rank_score(confidence: np.ndarray, teacher_p: np.ndarray, regret_status: np.ndarray, drop_cp: np.ndarray) -> np.ndarray:
    """Confident low-teacher-p first. Known eval loss scales it. Unknown is not faked."""
    miss = 1.0 - np.clip(teacher_p.astype(np.float64), 0.0, 1.0)
    base = np.clip(confidence.astype(np.float64), 0.0, 1.0) * miss
    out = base.copy()
    known = regret_status == REGRET_KNOWN
    substantial = known & (drop_cp >= 75)
    harmless = known & (drop_cp < 75)
    out[substantial] = base[substantial] * (1.0 + np.minimum(drop_cp[substantial].astype(np.float64), 800.0) / 250.0)
    out[harmless] *= 0.15
    out[regret_status == REGRET_UNKNOWN] = base[regret_status == REGRET_UNKNOWN] * 0.85
    return out.astype(np.float32)


def repeat_exposure(*, unique_n: int, mix_frac: float, steps: int = 8000, batch: int = 64) -> dict:
    draws = steps * batch * float(mix_frac)
    return {
        "unique_n": int(unique_n),
        "mix_frac": float(mix_frac),
        "steps": int(steps),
        "batch": int(batch),
        "expected_draws": draws,
        "repeat_exposure": (draws / unique_n) if unique_n else None,
    }


def remaining_weaknesses(
    *,
    pos_hash: np.ndarray,
    new_pred: np.ndarray,
    teacher: np.ndarray,
    old_pred: np.ndarray | None,
    verified_tag: np.ndarray | None,
    prev_fixed: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    n = pos_hash.shape[0]
    agree = new_pred == teacher
    was_sub = np.zeros(n, dtype=bool)
    if verified_tag is not None:
        was_sub = np.isin(verified_tag, np.fromiter(SUBSTANTIAL, dtype=np.int8))
    fixed = was_sub & agree
    still = was_sub & ~agree
    new_wrong = ~agree
    if old_pred is not None:
        newly = new_wrong & (old_pred == teacher)
    else:
        newly = new_wrong
    focus = still | newly
    if prev_fixed is not None and prev_fixed.size:
        focus &= ~np.isin(pos_hash, prev_fixed)
    report = {
        "fixed": int(fixed.sum()),
        "still_wrong_substantial": int(still.sum()),
        "newly_wrong": int(newly.sum()),
        "focus": int(focus.sum()),
    }
    return focus, report


def _stratum_take(labels: np.ndarray, n_take: int, rng: np.random.RandomState, weights: dict[str, float]) -> np.ndarray:
    picked: list[np.ndarray] = []
    remain = int(n_take)
    keys = [k for k in weights if (labels == k).any()]
    if not keys:
        return np.zeros(0, dtype=np.int64)
    wsum = sum(weights[k] for k in keys) or 1.0
    for k in keys:
        idx = np.flatnonzero(labels == k)
        want = min(len(idx), int(round(n_take * weights[k] / wsum)))
        want = min(want, remain)
        if want <= 0:
            continue
        take = rng.choice(idx, size=want, replace=False)
        picked.append(take)
        remain -= want
    have = np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)
    if remain > 0:
        leftover = np.setdiff1d(np.arange(labels.shape[0]), have, assume_unique=False)
        if leftover.size:
            extra = rng.choice(leftover, size=min(remain, leftover.size), replace=False)
            have = np.concatenate([have, extra]) if have.size else extra
    if have.size > n_take:
        have = rng.choice(have, size=n_take, replace=False)
    return np.sort(have.astype(np.int64))


def sample_source(data: dict, n_take: int, blocked: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    hs = canonical_hashes(data)
    keep = ~np.isin(hs, blocked)
    idx = np.flatnonzero(keep)
    if idx.size < n_take:
        raise RuntimeError(f"only {idx.size} unblocked rows, need {n_take}")
    phase = data["phase"].numpy()[idx]
    cp = data["cp"].numpy()[idx]
    mate = data["mate"].numpy()[idx]
    turn = data["turn"].numpy()[idx]
    src = data["source"].numpy()[idx]
    vv = data["value_valid"].numpy()[idx] if "value_valid" in data else np.ones(idx.size, dtype=np.int8)
    buckets = np.array([
        eval_bucket(int(cp[i]), int(mate[i]), int(turn[i]), int(vv[i]), int(src[i]))
        for i in range(idx.size)
    ], dtype=object)
    strata = np.array([f"{int(phase[i])}:{buckets[i]}" for i in range(idx.size)], dtype=object)
    weights = {}
    for ph, pf in PHASE_FRAC.items():
        present = {buckets[i] for i in range(idx.size) if int(phase[i]) == ph}
        ev = [b for b in ("losing", "equal", "winning") if b in present]
        if not ev:
            ev = [b for b in EVAL_BUCKETS if b in present]
        share = pf / max(len(ev), 1)
        for b in ev:
            weights[f"{ph}:{b}"] = share
    local = _stratum_take(strata, n_take, rng, weights)
    return idx[local]


def _take_idx(data: dict, idx: np.ndarray) -> dict:
    n = int(data["turn"].shape[0])
    out = {}
    for k, v in data.items():
        if torch.is_tensor(v) and v.ndim and v.shape[0] == n:
            out[k] = v[idx].contiguous()
    return squeeze_scalars(out)


def build_sample(args) -> dict:
    out = Path(args.out_dir)
    sample_path = out / "scan" / "sample.pt"
    if sample_path.exists() and not args.force:
        data = load_pt(sample_path)
        log(f"sample exists n={int(data['turn'].shape[0])} {sample_path}", out / "pipeline.log")
        return {"n": int(data["turn"].shape[0]), "path": str(sample_path), "resumed": True}
    baseline = Path(args.baseline)
    if not (baseline / "FROZEN.json").exists():
        raise SystemExit(f"baseline is not frozen: {baseline / 'FROZEN.json'}")
    blocked = blocked_hashes(baseline)
    rng = np.random.RandomState(args.seed)
    n_total = int(args.scan_n)
    chunks = []
    report = {"blocked": int(blocked.size), "sources": {}, "phases": {}, "eval_buckets": {}}
    for name, source_id, fname in SOURCE_FILES:
        want = int(round(n_total * SOURCE_QUOTA[source_id]))
        data = load_pt(baseline / fname)
        if "value_valid" not in data:
            data["value_valid"] = torch.zeros(int(data["turn"].shape[0]), dtype=torch.int8)
        take = sample_source(data, want, blocked, rng)
        part = _take_idx(data, take)
        part["source"] = torch.full((want,), source_id, dtype=torch.int8)
        chunks.append(part)
        report["sources"][name] = want
        del data
        log(f"sample {name} {want}", out / "pipeline.log")
    keys = [k for k in chunks[0] if all(k in c for c in chunks)]
    sample = squeeze_scalars({k: torch.cat([c[k] for c in chunks], dim=0) for k in keys})
    hs = canonical_hashes(sample)
    sample["pos_hash"] = torch.from_numpy(hs.astype(np.int64))
    if int(sample["turn"].shape[0]) != n_total:
        # rounding; trim or the last source absorbed the residue
        pass
    (out / "scan").mkdir(parents=True, exist_ok=True)
    torch.save(sample, sample_path)
    phase, _ = np.unique(sample["phase"].numpy(), return_counts=True)
    report["n"] = int(sample["turn"].shape[0])
    report["phases"] = {int(k): int((sample["phase"] == k).sum()) for k in range(3)}
    buckets = [
        eval_bucket(
            int(sample["cp"][i]), int(sample["mate"][i]), int(sample["turn"][i]),
            int(sample["value_valid"][i]) if "value_valid" in sample else 1,
            int(sample["source"][i]),
        )
        for i in range(int(sample["turn"].shape[0]))
    ]
    report["eval_buckets"] = {k: int(sum(b == k for b in buckets)) for k in EVAL_BUCKETS}
    report["unique"] = int(np.unique(hs).size)
    json_write(out / "scan" / "sample_report.json", report)
    log(f"SAMPLE n={report['n']} unique={report['unique']}", out / "pipeline.log")
    return report


@torch.inference_mode()
def predict_legal_conf(model, fused, turn, castling, ep_file, ba, turn_np, cast_np, ep_np, device, micro: int):
    preds, confs, ents = [], [], []
    n = fused.shape[0]
    use_amp = device.type == "cuda"
    for i in range(0, n, micro):
        sl = slice(i, i + micro)
        inp = {
            "fused_ids": fused[sl].to(device, non_blocking=device.type == "cuda"),
            "turn": turn[sl].to(device, non_blocking=device.type == "cuda"),
            "castling": castling[sl].to(device, non_blocking=device.type == "cuda"),
            "ep_file": ep_file[sl].to(device, non_blocking=device.type == "cuda"),
        }
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            logits = model(inp)["policy_logits"].float()
        probs = torch.softmax(logits, dim=-1)
        topk = logits.topk(16, dim=-1).indices.cpu().numpy()
        pred = _pick_legal(topk, ba[sl], turn_np[sl], cast_np[sl], ep_np[sl])
        gathered = probs.gather(1, torch.from_numpy(pred).to(probs.device).view(-1, 1)).squeeze(1)
        ent = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
        preds.append(pred)
        confs.append(gathered.cpu().numpy().astype(np.float32))
        ents.append(ent.cpu().numpy().astype(np.float32))
    return np.concatenate(preds), np.concatenate(confs), np.concatenate(ents)


def _annotate_batch(pred, conf, ent, teacher, soft_i, soft_p) -> dict:
    teacher_p, in_pv = teacher_p_and_in_pv(soft_i, soft_p, pred)
    disagree = pred != teacher
    regret = np.full(pred.shape[0], REGRET_UNKNOWN, dtype=np.int8)
    regret[~disagree] = REGRET_KNOWN
    # In-PV without stored MultiPV scores still has no eval drop. Do not invent one.
    drop = np.zeros(pred.shape[0], dtype=np.int32)
    teacher_ce = -np.log(np.clip(teacher_p, 1e-8, 1.0)).astype(np.float32)
    return {
        "model_move_idx": pred.astype(np.int64),
        "confidence": conf.astype(np.float32),
        "entropy": ent.astype(np.float32),
        "teacher_p": teacher_p,
        "model_in_pv": in_pv,
        "regret_status": regret,
        "drop_cp": drop,
        "teacher_ce": teacher_ce,
        "disagree": disagree.astype(np.int8),
    }


def run_scan(args) -> dict:
    out = Path(args.out_dir)
    log_path = out / "pipeline.log"
    sample_path = out / "scan" / "sample.pt"
    if not sample_path.exists():
        build_sample(args)
    sample = load_pt(sample_path)
    n = int(sample["turn"].shape[0])
    ckpt = Path(args.ckpt)
    ckpt_hash = file_sha256(ckpt)
    cursor_path = out / "scan" / "cursor.json"
    chunk_dir = out / "scan" / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    next_i = 0
    if cursor_path.exists() and not args.force:
        next_i = int(json.loads(cursor_path.read_text()).get("next") or 0)
        if next_i >= n and (out / "scan" / "scan.pt").exists():
            log(f"scan already complete n={n}", log_path)
            return json.loads((out / "scan" / "scan_report.json").read_text())
    device = pick_device()
    log(f"scan device={device} ckpt={ckpt} sha256={ckpt_hash[:12]} n={n} resume={next_i}", log_path)
    model = load_checkpoint(ckpt, device=device)
    ba = sample["board_array"].numpy()
    turn64 = sample["turn"].numpy().astype(np.int64)
    cast64 = sample["castling"].numpy().astype(np.int64)
    ep64 = sample["ep_square"].numpy().astype(np.int64)
    teacher = sample["move_idx"].numpy().astype(np.int64)
    soft_i = sample["soft_indices"].numpy().astype(np.int64)
    soft_p = sample["soft_probs"].numpy().astype(np.float32)
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True
        log("scan stop requested", log_path)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    t0 = time.time()
    chunk = int(args.scan_chunk)
    micro = int(args.micro_batch)
    while next_i < n and not stop:
        sl = slice(next_i, min(n, next_i + chunk))
        fused = board_array_to_fused(sample["board_array"][sl])
        pred, conf, ent = predict_legal_conf(
            model, fused,
            sample["turn"][sl].long(),
            sample["castling"][sl].long(),
            ep_square_to_file(sample["ep_square"][sl]).long(),
            ba[sl], turn64[sl], cast64[sl], ep64[sl],
            device, micro,
        )
        extra = _annotate_batch(pred, conf, ent, teacher[sl], soft_i[sl], soft_p[sl])
        packed = {k: v[sl].contiguous() if torch.is_tensor(v) and v.shape[0] == n else v for k, v in sample.items()}
        for k, v in extra.items():
            packed[k] = torch.from_numpy(v)
        packed["ckpt_hash8"] = torch.full((sl.stop - sl.start,), int(ckpt_hash[:16], 16) & ((1 << 63) - 1), dtype=torch.int64)
        idx = next_i // chunk
        torch.save(squeeze_scalars(packed), chunk_dir / f"chunk_{idx:05d}.pt")
        next_i = sl.stop
        dt = max(time.time() - t0, 1e-6)
        log(f"  scan {next_i:,}/{n:,} {next_i / dt:.1f} pos/s disagree={int(extra['disagree'].sum())}", log_path)
        json_write(cursor_path, {"next": next_i, "ckpt_sha256": ckpt_hash, "device": str(device)})
    if next_i < n:
        return {"status": "paused", "next": next_i, "n": n}
    chunks = [load_pt(p) for p in sorted(chunk_dir.glob("chunk_*.pt"))]
    keys = [k for k in chunks[0] if all(k in c and torch.is_tensor(c[k]) for c in chunks)]
    scan = squeeze_scalars({k: torch.cat([c[k] for c in chunks], dim=0) for k in keys})
    torch.save(scan, out / "scan" / "scan.pt")
    report = {
        "status": "complete",
        "n": int(scan["turn"].shape[0]),
        "ckpt": str(ckpt),
        "ckpt_sha256": ckpt_hash,
        "device": str(device),
        "disagree": int(scan["disagree"].sum()),
        "unknown_regret": int((scan["regret_status"] == REGRET_UNKNOWN).sum()),
        "in_pv": int(scan["model_in_pv"].sum()),
        "mean_confidence": float(scan["confidence"].mean()),
        "secs": time.time() - t0,
    }
    json_write(out / "scan" / "scan_report.json", report)
    log(f"SCAN {json.dumps({k: report[k] for k in ('n', 'disagree', 'unknown_regret')})}", log_path)
    return report


def select_verify_queue(
    scan: dict,
    *,
    n_verify: int,
    rng: np.random.RandomState,
    informative_frac: float = 0.70,
    unknown_frac: float = 0.15,
    uncertain_frac: float = 0.08,
    control_frac: float = 0.07,
) -> tuple[np.ndarray, dict]:
    n = int(scan["turn"].shape[0])
    conf = scan["confidence"].numpy()
    teacher_p = scan["teacher_p"].numpy()
    regret = scan["regret_status"].numpy()
    drop = scan["drop_cp"].numpy()
    disagree = scan["disagree"].numpy().astype(bool)
    scores = rank_score(conf, teacher_p, regret, drop)
    order = np.argsort(-scores)
    picked = np.zeros(n, dtype=bool)
    want_info = int(round(n_verify * informative_frac))
    info_pool = [i for i in order if disagree[i]]
    info = np.asarray(info_pool[:want_info], dtype=np.int64)
    picked[info] = True
    want_unk = int(round(n_verify * unknown_frac))
    unk_pool = np.flatnonzero((regret == REGRET_UNKNOWN) & disagree & ~picked)
    unk_pool = unk_pool[np.argsort(-scores[unk_pool])]
    unk = unk_pool[:want_unk]
    picked[unk] = True
    want_unc = int(round(n_verify * uncertain_frac))
    unc_pool = np.flatnonzero((conf >= UNCERTAIN_LO) & (conf <= UNCERTAIN_HI) & ~picked)
    if unc_pool.size:
        unc = rng.choice(unc_pool, size=min(want_unc, unc_pool.size), replace=False)
    else:
        unc = np.zeros(0, dtype=np.int64)
    picked[unc] = True
    remain = n_verify - int(picked.sum())
    ctrl_pool = np.flatnonzero(~picked)
    ctrl = rng.choice(ctrl_pool, size=min(max(remain, 0), ctrl_pool.size), replace=False) if ctrl_pool.size else np.zeros(0, dtype=np.int64)
    idx = np.concatenate([info, unk, unc, ctrl])
    idx = np.unique(idx)
    if idx.size > n_verify:
        # keep all informative/unknown, trim control then uncertain
        keep = np.zeros(n, dtype=bool)
        keep[info] = True
        keep[unk] = True
        extra = np.concatenate([unc, ctrl])
        room = n_verify - int(keep.sum())
        if room > 0 and extra.size:
            keep[extra[:room]] = True
        idx = np.flatnonzero(keep)
    report = {
        "n": int(idx.size),
        "informative": int(info.size),
        "unknown_regret": int(unk.size),
        "uncertain": int(unc.size),
        "control": int(ctrl.size),
        "mean_rank_score": float(scores[idx].mean()) if idx.size else 0.0,
        "note": "Rank is confidence × (1 - teacher_p), scaled by verified drop when known. Not CE-only.",
    }
    return idx.astype(np.int64), report


def run_rank(args) -> dict:
    out = Path(args.out_dir)
    scan = load_pt(out / "scan" / "scan.pt")
    rng = np.random.RandomState(args.seed + 1)
    n_verify = min(int(args.verify_n), int(scan["turn"].shape[0]))
    n_verify = max(5_000, min(10_000, n_verify)) if int(scan["turn"].shape[0]) >= 5_000 else n_verify
    idx, report = select_verify_queue(scan, n_verify=n_verify, rng=rng)
    queue = _take_idx(scan, idx)
    scores = rank_score(
        queue["confidence"].numpy(), queue["teacher_p"].numpy(),
        queue["regret_status"].numpy(), queue["drop_cp"].numpy(),
    )
    queue["rank_score"] = torch.from_numpy(scores)
    (out / "rank").mkdir(parents=True, exist_ok=True)
    torch.save(squeeze_scalars(queue), out / "rank" / "verify_queue.pt")
    json_write(out / "rank" / "rank_report.json", report)
    log(f"RANK {json.dumps(report)}", out / "pipeline.log")
    return report


def analyse_root(engine, board, move, *, nodes: int, watchdog_s: float) -> dict:
    import chess.engine

    limit = chess.engine.Limit(nodes=max(1, nodes), time=max(0.5, watchdog_s))
    info = engine.analyse(board, limit, root_moves=[move])
    if isinstance(info, list):
        info = info[0]
    sc = info.get("score")
    pv = info.get("pv") or []
    if sc is None:
        return {"cp_stm": 0, "mate_stm": 0, "depth": 0, "nodes": 0, "pv": []}
    pov = sc.pov(board.turn)
    if pov.is_mate():
        cp, mate = 0, int(pov.mate() or 0)
    else:
        mate = 0
        raw = pov.score(mate_score=None)
        cp = int(raw or 0)
    return {
        "cp_stm": cp,
        "mate_stm": mate,
        "depth": int(info.get("depth") or 0),
        "nodes": int(info.get("nodes") or 0),
        "pv": [m.uci() for m in pv],
    }


def verify_one(engine, item: dict, *, nodes: int, multipv: int, tau: float, watchdog_s: float) -> dict | None:
    import chess

    from sf19_soft_dataset import analyze_board

    board = chess.Board(item["fen"])
    try:
        teacher_mv = index_to_move(int(item["teacher_idx"]))
        model_mv = index_to_move(int(item["model_idx"]))
    except Exception:
        return None
    if teacher_mv not in board.legal_moves or model_mv not in board.legal_moves:
        return None
    row = analyze_board(engine, board, nodes=nodes, multipv=multipv, tau=tau, watchdog_s=watchdog_s)
    if row is None:
        return None
    teacher = analyse_root(engine, board, teacher_mv, nodes=nodes, watchdog_s=watchdog_s)
    model = analyse_root(engine, board, model_mv, nodes=nodes, watchdog_s=watchdog_s)
    soft_i = np.asarray(row["soft_indices"]).reshape(-1)
    in_pv = int(int(item["model_idx"]) in set(int(x) for x in soft_i if int(x) >= 0))
    info = classify_lapse(
        best_cp=teacher["cp_stm"], best_mate=teacher["mate_stm"],
        model_cp=model["cp_stm"], model_mate=model["mate_stm"],
        model_in_pv=bool(in_pv),
    )
    drop = info["drop_cp"]
    tag = info["tag"]
    return {
        **{k: row[k] for k in (
            "board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate",
            "soft_indices", "soft_probs", "soft_cps", "soft_mates", "label_depth",
            "phase", "wdl", "nodes", "nodes_budget",
        ) if k in row},
        "source": np.int8(SOURCE_MISTAKE),
        "pos_hash": np.int64(item["pos_hash"]),
        "model_move_idx": np.int64(item["model_idx"]),
        "teacher_move_idx": np.int64(item["teacher_idx"]),
        "tag": np.int8(TAG_TO_I.get(tag, TAG_TO_I["disagree"])),
        "drop_cp": np.int32(int(drop or 0)),
        "model_in_pv": np.int8(in_pv),
        "needs_sf": np.int8(0),
        "regret_status": np.int8(REGRET_KNOWN),
        "teacher_cp_stm": np.int32(teacher["cp_stm"]),
        "teacher_mate_stm": np.int32(teacher["mate_stm"]),
        "teacher_depth": np.int16(teacher["depth"]),
        "teacher_nodes": np.int32(teacher["nodes"]),
        "model_cp_stm": np.int32(model["cp_stm"]),
        "model_mate_stm": np.int32(model["mate_stm"]),
        "model_depth": np.int16(model["depth"]),
        "model_nodes": np.int32(model["nodes"]),
        "orig_cp": np.int32(item["orig_cp"]),
        "orig_depth": np.int16(item["orig_depth"]),
        "kind": tag,
        "teacher_pv": teacher["pv"],
        "model_pv": model["pv"],
        "nodes_budget": np.int32(nodes),
    }


def _verify_worker(sf: str, nodes: int, hash_mb: int, watchdog_s: float, job_q, res_q) -> None:
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(sf)
    engine.configure({"Threads": 1, "Hash": int(hash_mb), "UCI_ShowWDL": True})
    while True:
        item = job_q.get()
        if item is None:
            break
        try:
            row = verify_one(
                engine, item, nodes=int(nodes),
                multipv=8, tau=120.0, watchdog_s=float(watchdog_s),
            )
        except Exception:
            row = None
        res_q.put((item["i"], row))
    engine.quit()


def run_verify(args) -> dict:
    from multiprocessing import Process, Queue

    from sf19_soft_dataset import resolve_sf

    out = Path(args.out_dir)
    queue = load_pt(out / "rank" / "verify_queue.pt")
    n = int(queue["turn"].shape[0])
    dest = out / "verify"
    dest.mkdir(parents=True, exist_ok=True)
    json_write(dest / "protocol.json", {**VERIFY_PROTOCOL, "nodes_budget": int(args.nodes)})
    state_path = dest / "cursor.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "done": [], "n_ok": 0, "tags": {}, "harmless": 0, "substantial": 0,
    }
    if args.force:
        state = {"done": [], "n_ok": 0, "tags": {}, "harmless": 0, "substantial": 0}
    done = set(int(x) for x in state.get("done") or [])
    if len(done) >= n:
        log("verify already complete", out / "pipeline.log")
        return json.loads((dest / "verify_report.json").read_text()) if (dest / "verify_report.json").exists() else state
    sf = resolve_sf()
    log(f"verify sf={sf} nodes={args.nodes} workers={args.workers} done={len(done)}/{n}", out / "pipeline.log")
    ba = queue["board_array"].numpy()
    items = []
    for i in range(n):
        if i in done:
            continue
        items.append({
            "i": i,
            "fen": board_array_to_fen(ba[i], queue["turn"][i], queue["castling"][i], queue["ep_square"][i]),
            "teacher_idx": int(queue["move_idx"][i]),
            "model_idx": int(queue["model_move_idx"][i]),
            "pos_hash": int(queue["pos_hash"][i]),
            "orig_cp": int(queue["cp"][i]),
            "orig_depth": int(queue["label_depth"][i]),
        })
    job_q: Queue = Queue()
    res_q: Queue = Queue()
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    procs = [
        Process(
            target=_verify_worker,
            args=(
                sf, int(args.nodes), int(VERIFY_PROTOCOL["hash_mb"]),
                float(VERIFY_PROTOCOL["watchdog_s"]), job_q, res_q,
            ),
            daemon=True,
        )
        for _ in range(int(args.workers))
    ]
    for p in procs:
        p.start()
    pending: list[dict] = []
    lines_path = dest / "lines.jsonl"
    shard_i = 0
    existing = sorted((dest / "inbox").glob("shard_*")) if (dest / "inbox").exists() else []
    if existing:
        shard_i = int(existing[-1].name.split("_", 1)[1]) + 1
    t0 = time.time()
    inflight = 0
    window = max(int(args.workers) * 2, 4)
    cursor = 0

    def flush(force: bool = False) -> None:
        nonlocal pending, shard_i
        if not pending or (len(pending) < 256 and not force):
            return
        take = pending[:256] if not force else pending
        pending = pending[len(take):]
        skip = {"teacher_pv", "model_pv", "kind"}
        keys = [k for k in take[0] if k not in skip]
        packed = {k: torch.from_numpy(np.stack([np.asarray(r[k]) for r in take])) for k in keys}
        _write_shard(packed, dest, shard_i)
        shard_i += 1

    while cursor < len(items) or inflight:
        while inflight < window and cursor < len(items) and not stop:
            job_q.put(items[cursor])
            cursor += 1
            inflight += 1
        if inflight == 0:
            break
        i, row = res_q.get()
        inflight -= 1
        state.setdefault("done", [])
        if i not in done:
            state["done"].append(int(i))
            done.add(int(i))
        if row is None:
            json_write(state_path, state)
            continue
        state["n_ok"] = int(state.get("n_ok") or 0) + 1
        tag = I_TO_TAG.get(int(row["tag"]), str(int(row["tag"])))
        state.setdefault("tags", {})
        state["tags"][tag] = int(state["tags"].get(tag, 0)) + 1
        if int(row["tag"]) in SUBSTANTIAL:
            state["substantial"] = int(state.get("substantial") or 0) + 1
        else:
            state["harmless"] = int(state.get("harmless") or 0) + 1
        pending.append(row)
        with lines_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "pos_hash": int(row["pos_hash"]),
                "teacher_pv": row["teacher_pv"],
                "model_pv": row["model_pv"],
                "teacher_cp_stm": int(row["teacher_cp_stm"]),
                "model_cp_stm": int(row["model_cp_stm"]),
                "teacher_depth": int(row["teacher_depth"]),
                "model_depth": int(row["model_depth"]),
                "teacher_nodes": int(row["teacher_nodes"]),
                "model_nodes": int(row["model_nodes"]),
                "nodes_budget": int(args.nodes),
                "tag": tag,
                "drop_cp": int(row["drop_cp"]),
            }) + "\n")
        if state["n_ok"] % 25 == 0:
            rate = state["n_ok"] / max(time.time() - t0, 1e-6)
            log(f"verify {state['n_ok']:,}/{n:,} {rate:.2f}/s last={tag} drop={int(row['drop_cp'])}", out / "pipeline.log")
        flush(False)
        json_write(state_path, state)
        if stop:
            break
    flush(True)
    for _ in procs:
        job_q.put(None)
    for p in procs:
        p.join(timeout=30)
    report = {
        **state,
        "protocol": {**VERIFY_PROTOCOL, "nodes_budget": int(args.nodes)},
        "secs": time.time() - t0,
        "status": "complete" if len(done) >= n else "paused",
    }
    json_write(dest / "verify_report.json", report)
    log(f"VERIFY {json.dumps({k: report.get(k) for k in ('n_ok', 'substantial', 'harmless', 'tags', 'status')})}", out / "pipeline.log")
    return report


def _load_verified(out: Path) -> dict | None:
    shards = sorted((out / "verify" / "inbox").glob("shard_*/soft_cache.pt"))
    if not shards:
        return None
    chunks = [load_pt(p) for p in shards]
    keys = [k for k in chunks[0] if all(k in c for c in chunks)]
    return squeeze_scalars({k: torch.cat([c[k] for c in chunks], dim=0) for k in keys})


def run_assemble(args) -> dict:
    out = Path(args.out_dir)
    bucket_dir = out / "bucket"
    if (bucket_dir / "FROZEN.json").exists() and not args.force:
        raise SystemExit(f"correction bucket is frozen at {bucket_dir}; delete FROZEN.json to rebuild")
    verified = _load_verified(out)
    if verified is None:
        raise SystemExit("no verified shards yet")
    tags = verified["tag"].numpy()
    keep = np.isin(tags, np.fromiter(SUBSTANTIAL, dtype=np.int8))
    if int(keep.sum()) == 0:
        raise SystemExit("no substantial verified corrections")
    corr = _take_idx(verified, np.flatnonzero(keep))
    hs = corr["pos_hash"].numpy().astype(np.uint64) if "pos_hash" in corr else canonical_hashes(corr)
    _, uniq_i = np.unique(hs, return_index=True)
    corr = _take_idx(corr, np.sort(uniq_i))
    n_u = int(corr["turn"].shape[0])
    corr["source"] = torch.full((n_u,), SOURCE_MISTAKE, dtype=torch.int8)
    if "wdl" not in corr:
        corr["wdl"] = compute_wdl(corr["cp"], corr["mate"])
    corr["value_valid"] = torch.ones(n_u, dtype=torch.int8)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    torch.save(squeeze_scalars(corr), bucket_dir / "corrections.pt")
    mix_frac = float(args.correction_frac)
    coverage = {
        "unique_n": n_u,
        "padded_with_ok": False,
        "tags": {I_TO_TAG[int(k)]: int((corr["tag"] == k).sum()) for k in np.unique(corr["tag"].numpy())},
        "mix_frac_requested": mix_frac,
        "repeat_exposure": {
            f"{int(p * 100)}pct": repeat_exposure(unique_n=n_u, mix_frac=p)
            for p in (0.10, mix_frac, 0.15)
        },
        "drawn_if_swapped_once": n_u / 1_000_000.0,
        "trainer": {
            "soft_cache": "outputs/organized_chess_v1/soft_cache.pt",
            "bonus_cache": str((bucket_dir / "corrections.pt").relative_to(ROOT)),
            "bonus_mix_frac": mix_frac,
            "deep_mix_frac": 0.05,
            "note": (
                "Corrections replace part of the ordinary stream via bonus_mix_frac, "
                "not by rewriting Lichess/puzzles/Syzygy. Unique rows are not tiled on disk."
            ),
        },
    }
    baseline = Path(args.baseline)
    sf19 = ensure_value_mask(load_pt(baseline / "sf19_train.pt"), 1)
    n_keep = int(sf19["turn"].shape[0]) - n_u
    mixed = replace_sf19(sf19, corr, n_keep, args.seed)
    lich = ensure_value_mask(load_pt(baseline / "lichess_train.pt"), 0)
    puz = ensure_value_mask(load_pt(baseline / "puzzles_train.pt"), 0)
    keys = [k for k in ("board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate",
                        "soft_indices", "soft_probs", "label_depth", "phase", "source", "wdl", "value_valid")
            if k in mixed and k in lich and k in puz]
    soft = squeeze_scalars({k: torch.cat([mixed[k], lich[k], puz[k]], dim=0) for k in keys})
    var = bucket_dir / "variant"
    var.mkdir(parents=True, exist_ok=True)
    torch.save(soft, var / "soft_cache.pt")
    link_or_copy(baseline / "deep_cache.pt", var / "deep_cache.pt")
    for name in ("sf19", "lichess", "puzzles", "syzygy"):
        link_or_copy(baseline / f"{name}_eval.pt", var / f"{name}_eval.pt")
    coverage["variant"] = {
        "soft_cache": str((var / "soft_cache.pt").relative_to(ROOT)),
        "sf19": int((soft["source"] == SOURCE_SF19).sum()),
        "corrections": int((soft["source"] == SOURCE_MISTAKE).sum()),
    }
    json_write(bucket_dir / "coverage.json", coverage)
    frozen = {
        "role": "correction_bucket",
        "frozen_at": utc_now(),
        "unique_n": n_u,
        "padded_with_ok": False,
        "sha256": file_sha256(bucket_dir / "corrections.pt"),
        "do_not_overwrite": True,
        "note": "Frozen during training comparison. Rescan after that round; do not remine fixed errors into this file.",
    }
    json_write(bucket_dir / "FROZEN.json", frozen)
    json_write(bucket_dir / "manifest.json", {"status": "frozen", **coverage, "frozen": frozen})
    log(f"ASSEMBLE unique={n_u} tags={coverage['tags']}", out / "pipeline.log")
    return coverage


def run_rescan(args) -> dict:
    out = Path(args.out_dir)
    sample_path = out / "scan" / "sample.pt"
    if not sample_path.exists():
        raise SystemExit("need the original 100k sample")
    old_scan = load_pt(out / "scan" / "scan.pt") if (out / "scan" / "scan.pt").exists() else None
    # Scan into a sibling dir, then join.
    rescan_dir = out / "rescan"
    rescan_dir.mkdir(parents=True, exist_ok=True)
    saved_out = args.out_dir
    args.out_dir = str(rescan_dir)
    (rescan_dir / "scan").mkdir(parents=True, exist_ok=True)
    link_or_copy(sample_path, rescan_dir / "scan" / "sample.pt")
    scan_rep = run_scan(args)
    args.out_dir = saved_out
    new_scan = load_pt(rescan_dir / "scan" / "scan.pt")
    verified = _load_verified(out)
    tag_map = {}
    if verified is not None and "pos_hash" in verified:
        for h, t in zip(verified["pos_hash"].numpy(), verified["tag"].numpy()):
            tag_map[int(h)] = int(t)
    hs = new_scan["pos_hash"].numpy()
    vtag = np.array([tag_map.get(int(h), -1) for h in hs], dtype=np.int8)
    prev_fixed = np.zeros(0, dtype=np.int64)
    if (out / "rescan" / "fixed_hashes.npy").exists():
        prev_fixed = np.load(out / "rescan" / "fixed_hashes.npy")
    old_pred = old_scan["model_move_idx"].numpy() if old_scan is not None else None
    # align old pred by hash if lengths match sample order (same sample.pt)
    focus, report = remaining_weaknesses(
        pos_hash=hs,
        new_pred=new_scan["model_move_idx"].numpy(),
        teacher=new_scan["move_idx"].numpy(),
        old_pred=old_pred,
        verified_tag=vtag,
        prev_fixed=prev_fixed,
    )
    fixed_now = (vtag >= 0) & np.isin(vtag, np.fromiter(SUBSTANTIAL, dtype=np.int8)) & (
        new_scan["model_move_idx"].numpy() == new_scan["move_idx"].numpy()
    )
    fixed_hashes = np.unique(np.concatenate([prev_fixed, hs[fixed_now].astype(np.int64)])) if fixed_now.any() or prev_fixed.size else prev_fixed
    np.save(rescan_dir / "fixed_hashes.npy", fixed_hashes)
    focus_idx = np.flatnonzero(focus)
    if focus_idx.size:
        torch.save(_take_idx(new_scan, focus_idx), rescan_dir / "focus_queue.pt")
    report.update(scan=scan_rep, ckpt=str(args.ckpt), focus_n=int(focus_idx.size), fixed_registry=int(fixed_hashes.size))
    json_write(rescan_dir / "rescan_report.json", report)
    log(f"RESCAN {json.dumps(report)}", out / "pipeline.log")
    return report


def self_test() -> None:
    pred = np.array([5, 9, 1], dtype=np.int64)
    teacher = np.array([1, 1, 1], dtype=np.int64)
    soft_i = np.array([[5, 1, -1, -1, -1, -1, -1, -1], [2, 1, -1, -1, -1, -1, -1, -1], [1, 2, -1, -1, -1, -1, -1, -1]], dtype=np.int64)
    soft_p = np.array([[0.1, 0.9, 0, 0, 0, 0, 0, 0], [0.4, 0.6, 0, 0, 0, 0, 0, 0], [0.8, 0.2, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    p, in_pv = teacher_p_and_in_pv(soft_i, soft_p, pred)
    assert abs(float(p[0]) - 0.1) < 1e-6 and in_pv[0] == 1
    assert float(p[1]) == 0.0 and in_pv[1] == 0
    assert in_pv[2] == 1
    extra = _annotate_batch(pred, np.array([0.9, 0.8, 0.7], np.float32), np.zeros(3, np.float32), teacher, soft_i, soft_p)
    assert int(extra["regret_status"][1]) == REGRET_UNKNOWN
    assert int(extra["drop_cp"][1]) == 0
    assert int(extra["regret_status"][2]) == REGRET_KNOWN
    scores = rank_score(
        np.array([0.9, 0.2, 0.9, 0.9], np.float32),
        np.array([0.05, 0.05, 0.05, 0.80], np.float32),
        np.array([REGRET_KNOWN, REGRET_KNOWN, REGRET_UNKNOWN, REGRET_KNOWN], np.int8),
        np.array([200, 200, 0, 0], np.int32),
    )
    assert scores[0] > scores[1], "confident substantial beats low-confidence"
    assert scores[0] > scores[3], "not CE/disagreement alone: high teacher_p ranks lower"
    assert scores[2] > 0 and scores[2] < scores[0]
    cov = repeat_exposure(unique_n=2000, mix_frac=0.12)
    assert abs(cov["repeat_exposure"] - (8000 * 64 * 0.12 / 2000)) < 1e-6
    focus, rep = remaining_weaknesses(
        pos_hash=np.array([1, 2, 3], dtype=np.int64),
        new_pred=np.array([1, 9, 3]),
        teacher=np.array([1, 2, 3]),
        old_pred=np.array([9, 9, 3]),
        verified_tag=np.array([TAG_TO_I["blunder"], TAG_TO_I["major"], TAG_TO_I["ok"]], dtype=np.int8),
        prev_fixed=None,
    )
    assert rep["fixed"] == 1 and focus[1] and not focus[0]
    print("self-test ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="sample + scan + rank, then verify if Stockfish is present")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--scan-n", type=int, default=100_000)
    ap.add_argument("--scan-chunk", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--verify-n", type=int, default=7500)
    ap.add_argument("--nodes", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--correction-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if VOCAB_SIZE != 1968:
        raise SystemExit(f"need compact vocab, got {VOCAB_SIZE}")
    args.out_dir = str(Path(args.out_dir).resolve())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.assemble:
        run_assemble(args)
        return
    if args.rescan:
        run_rescan(args)
        return
    if args.verify:
        run_verify(args)
        return
    if args.rank:
        run_rank(args)
        return
    if args.scan:
        build_sample(args)
        run_scan(args)
        return
    if args.sample:
        build_sample(args)
        return
    if not args.go:
        raise SystemExit("pass --go or a phase flag")
    build_sample(args)
    scan = run_scan(args)
    if scan.get("status") == "paused":
        return
    run_rank(args)
    try:
        from sf19_soft_dataset import resolve_sf
        resolve_sf()
    except FileNotFoundError:
        log("Stockfish 19 not found; scan+rank done, verify skipped", Path(args.out_dir) / "pipeline.log")
        return
    run_verify(args)


if __name__ == "__main__":
    main()
