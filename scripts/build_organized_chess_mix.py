#!/usr/bin/env python3
"""Build a versioned, CPU-only searchless mix from existing labels.

Hypothesis (pilot, not a proven optimum):
  45% SF19 soft  |  35% deep Lichess MultiPV  |  15% puzzles  |  5% Syzygy

Puzzles are policy-only. Lichess and Syzygy values stay masked until their
value provenance is trustworthy. SF19 keeps genuine teacher WDL.

Interrupted builds resume completed sources. A complete directory is not
overwritten unless --force is set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "experiments")]

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download

from autoresearch_8gb.pipeline import attach_static_targets, hflip_cache_slice
from build_hf_elo_mix import position_hashes
from data_loader import CASTLING_MAP, _fast_parse_fen
from exp193_puzzle_soft_harvest import puzzle_to_record
from move_vocab import UCI_TO_IDX, VOCAB_SIZE, index_to_move

SOURCE_LICHESS = 1
SOURCE_SYZYGY = 2
SOURCE_SF19 = 4
SOURCE_PUZZLE = 5
SOURCES = {
    "sf19": ("avewright/chess-soft-sf19", 0.45, SOURCE_SF19),
    "lichess": ("avewright/chess-soft-multipv-lichess", 0.35, SOURCE_LICHESS),
    "puzzles": ("Lichess/chess-puzzles", 0.15, SOURCE_PUZZLE),
    "syzygy": ("avewright/chess-soft-syzygy", 0.05, SOURCE_SYZYGY),
}
CORE = (
    "board_array", "turn", "castling", "ep_square", "move_idx", "cp", "mate",
    "soft_indices", "soft_probs", "label_depth", "phase",
)
DTYPES = dict(
    board_array=torch.int8, turn=torch.int8, castling=torch.int8,
    ep_square=torch.int8, move_idx=torch.int64, cp=torch.int32,
    mate=torch.int32, soft_indices=torch.int64, soft_probs=torch.float32,
    label_depth=torch.int16, phase=torch.int8,
)
PHASE_FRAC = {0: 0.22, 1: 0.50, 2: 0.28}
RATING_EDGES = (1200, 1600, 2000, 2400)
SF19_MIN_BUDGET = 100_000
SF19_MIN_DEPTH = 12
LICHESS_MIN_DEPTH = 22
LICHESS_MAX_DEPTH = 127
DUMMY_WDL = torch.tensor([0.0, 1.0, 0.0])
LIMITATIONS = [
    "New holdouts are dataset-disjoint, not proven unseen by existing checkpoints.",
    "Legacy Lichess values are masked: mate is always 0 and |cp|>=90000 is a mate sentinel, not a White-absolute WDL.",
    "Syzygy tb_wdl is STM {-2..2} and dtz is distance-to-zero, not mate. Value supervision stays off.",
    "SF19 nodes_budget is the search cap; label_depth and nodes are achieved. Train rows require budget>=100k and depth>=12.",
    "SF19 data/shard_000000.parquet is the upstream eval split (split=1) and is never trained on.",
    "Puzzles are one-hot policy after the opponent setup move; full solution lines and GameId stay in provenance.",
    "Phase and rating quotas are first-pass preferences. A second fill pass is used if a source would otherwise miss its row target.",
]


def load_hf_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'").strip('"')
            break


def fingerprint(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_write(p: Path, d) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, indent=2) + "\n")
    tmp.replace(p)


def CASTLE_FEN(c: int) -> str:
    return "".join(ch for bit, ch in ((8, "K"), (4, "Q"), (2, "k"), (1, "q")) if c & bit) or "-"


def encode_board(board: chess.Board):
    arr = np.zeros(64, dtype=np.int8)
    turn, castling, ep = _fast_parse_fen(board.fen(), arr)
    return arr, int(turn), int(castling), int(ep if ep is not None else -1)


def reconstruct_board(d: dict, i: int) -> chess.Board | None:
    b = chess.Board(None)
    for sq, v in enumerate(d["board_array"][i].tolist()):
        if v:
            b.set_piece_at(sq, chess.Piece((v - 1) % 6 + 1, v <= 6))
    b.turn = int(d["turn"][i]) == 0
    b.set_castling_fen(CASTLE_FEN(int(d["castling"][i])))
    ep = int(d["ep_square"][i])
    b.ep_square = ep if ep > 0 else None
    if not b.is_valid() or b.is_game_over():
        return None
    return b


def policy_ok(d: dict, i: int) -> bool:
    si, sp = d["soft_indices"][i], d["soft_probs"][i]
    mid = int(d["move_idx"][i])
    if not (0 <= mid < VOCAB_SIZE):
        return False
    if not torch.isfinite(sp).all() or (sp < 0).any():
        return False
    if abs(float(sp.sum()) - 1) > 0.001:
        return False
    active = si[sp > 0]
    if active.numel() == 0:
        return False
    if int((active < 0).sum()) or int((active >= VOCAB_SIZE).sum()):
        return False
    vals = active.tolist()
    return len(set(vals)) == len(vals)


def legal_row(d: dict, i: int) -> bool:
    if not policy_ok(d, i):
        return False
    b = reconstruct_board(d, i)
    if b is None:
        return False
    si, sp = d["soft_indices"][i], d["soft_probs"][i]
    try:
        moves = [index_to_move(int(d["move_idx"][i]))]
        moves.extend(index_to_move(int(x)) for x in si[sp > 0].tolist())
    except (IndexError, ValueError, KeyError):
        return False
    return all(mv in b.legal_moves for mv in moves)


def value_ok(wdl: torch.Tensor) -> bool:
    w = wdl.float().reshape(-1)
    if w.numel() != 3 or not torch.isfinite(w).all() or (w < 0).any():
        return False
    return abs(float(w.sum()) - 1) <= 0.001


def syzygy_meta_ok(tb_wdl, dtz, mate, *, reject_dtz_mate: bool = False) -> tuple[bool, str]:
    try:
        w = int(tb_wdl)
        z = int(dtz)
        m = int(mate)
    except (TypeError, ValueError):
        return False, "meta_unreadable"
    if w not in (-2, -1, 0, 1, 2):
        return False, "tb_wdl_range"
    if abs(z) > 1000:
        return False, "dtz_range"
    if reject_dtz_mate and m != 0:
        return False, "dtz_as_mate"
    return True, "ok"


def pack_puzzle(puzzle: dict) -> tuple[dict | None, dict]:
    rec = puzzle_to_record(puzzle, 600, 3500)
    if rec is None:
        return None, {}
    board = chess.Board(rec["fen"])
    arr, turn, castling, ep = encode_board(board)
    mid = UCI_TO_IDX.get(rec["best_move"], -1)
    if mid < 0:
        return None, {}
    si = torch.full((8,), -1, dtype=torch.int64)
    sp = torch.zeros(8, dtype=torch.float32)
    si[0] = mid
    sp[0] = 1.0
    row = {
        "board_array": torch.from_numpy(arr.copy()),
        "turn": torch.tensor(turn, dtype=torch.int8),
        "castling": torch.tensor(castling, dtype=torch.int8),
        "ep_square": torch.tensor(ep if ep > 0 else -1, dtype=torch.int8),
        "move_idx": torch.tensor(mid, dtype=torch.int64),
        "cp": torch.tensor(0, dtype=torch.int32),
        "mate": torch.tensor(0, dtype=torch.int32),
        "soft_indices": si,
        "soft_probs": sp,
        "label_depth": torch.tensor(0, dtype=torch.int16),
        "phase": torch.tensor({"opening": 0, "middlegame": 1, "endgame": 2}.get(rec.get("phase"), 1), dtype=torch.int8),
        "source": torch.tensor(SOURCE_PUZZLE, dtype=torch.int8),
        "value_valid": torch.tensor(0, dtype=torch.int8),
    }
    meta = {
        "PuzzleId": puzzle.get("PuzzleId"),
        "GameId": puzzle.get("GameId"),
        "Moves": puzzle.get("Moves"),
        "Rating": int(puzzle.get("Rating") or 0),
        "Themes": puzzle.get("Themes"),
        "FEN": puzzle.get("FEN"),
        "solver_fen": rec["fen"],
        "best_move": rec["best_move"],
    }
    return row, meta


def canonical_hashes(d: dict) -> np.ndarray:
    h = position_hashes(d)
    idx = torch.where(d["castling"] == 0)[0]
    if len(idx):
        h[idx.numpy()] = np.minimum(h[idx.numpy()], position_hashes(hflip_cache_slice(d, idx)))
    return h


def group_key(name: str, meta: dict, pos_hash: int) -> str:
    if name == "puzzles":
        gid = meta.get("GameId") or meta.get("PuzzleId")
        return f"puzzle-game:{gid}" if gid else f"pos:{pos_hash}"
    if name == "sf19":
        gid = meta.get("game_id", -1)
        if int(gid) >= 0:
            return f"sf19-game:{int(gid)}"
    return f"pos:{pos_hash}"


def is_eval_group(group: str, eval_pct: int = 2) -> bool:
    return int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "little") % 100 < eval_pct


def packed(table) -> dict:
    names = set(table.column_names)
    return {k: torch.tensor(table[k].to_pylist(), dtype=dt) for k, dt in DTYPES.items() if k in names}


def stack_rows(rows: list[dict]) -> dict:
    keys = list(rows[0].keys())
    out = {}
    for k in keys:
        first = rows[0][k]
        if torch.is_tensor(first):
            out[k] = torch.stack([r[k] for r in rows], dim=0)
        else:
            out[k] = torch.from_numpy(np.stack([np.asarray(r[k]) for r in rows]))
    return out


def collect_blocked(out: Path) -> tuple[set[int], list[str]]:
    blocked: set[int] = set()
    used = []
    for p in sorted((ROOT / "outputs").rglob("val_manifest*.json")):
        if out in p.parents:
            continue
        raw = json.loads(p.read_text())
        blocked.update(int(x) for x in raw.get("blocked_hashes", raw.get("hashes", [])))
        used.append(str(p.relative_to(ROOT)))
    for p in sorted((ROOT / "outputs").rglob("blocked_manifest.json")):
        if out in p.parents:
            continue
        raw = json.loads(p.read_text())
        blocked.update(int(x) for x in raw.get("blocked_hashes", []))
        used.append(str(p.relative_to(ROOT)))
    return blocked, used


def load_progress(out: Path) -> dict:
    p = out / "progress.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"completed": [], "seen": []}


def save_progress(out: Path, completed: list[str], seen: set[int]) -> None:
    json_write(out / "progress.json", {"completed": completed, "seen": sorted(seen)})


def source_done(out: Path, name: str) -> bool:
    return (out / f"{name}_train.pt").exists() and (out / f"{name}_eval.pt").exists()


def load_seen_from_shards(out: Path, names: list[str]) -> set[int]:
    seen: set[int] = set()
    for name in names:
        for split in ("train", "eval"):
            p = out / f"{name}_{split}.pt"
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            seen.update(int(x) for x in canonical_hashes(d).tolist())
    return seen


def rating_bucket(rating: int) -> int:
    return sum(rating >= x for x in RATING_EDGES)


def batches(name, files, report, rng, local_lichess: Path, local_syzygy: Path, revision: str, repo: str):
    recorded = {x.get("file") or x.get("local") for x in report["inputs"]}
    if name == "lichess" and local_lichess.exists():
        if str(local_lichess) not in recorded:
            report["inputs"].append({"local": str(local_lichess), "sha256": fingerprint(local_lichess)})
        d = torch.load(local_lichess, map_location="cpu", weights_only=False)
        n = len(d["turn"])
        order = list(range(0, n, 8192))
        rng.shuffle(order)
        for start in order:
            sl = {k: v[start:start + 8192] for k, v in d.items() if k in CORE}
            yield sl, {}, str(local_lichess), start
        return
    if name == "syzygy" and local_syzygy.exists():
        if str(local_syzygy) not in recorded:
            report["inputs"].append({"local": str(local_syzygy), "sha256": fingerprint(local_syzygy)})
        d = torch.load(local_syzygy, map_location="cpu", weights_only=False)
        n = len(d["turn"])
        order = list(range(0, n, 8192))
        rng.shuffle(order)
        for start in order:
            sl = {k: v[start:start + 8192] for k, v in d.items() if k in CORE}
            extra = {}
            for key in ("wdl", "dtz", "mate"):
                if key in d:
                    extra[key] = d[key][start:start + 8192]
            yield sl, extra, str(local_syzygy), start
        return
    for fn in files:
        if name == "sf19" and fn.endswith("data/shard_000000.parquet"):
            report["skipped_upstream_eval_files"] = 1
            continue
        p = hf_hub_download(repo, fn, repo_type="dataset", revision=revision)
        if fn not in recorded:
            report["inputs"].append({"file": fn, "sha256": fingerprint(p)})
        offset = 0
        for b in pq.ParquetFile(p).iter_batches(batch_size=8192):
            t = b.to_pydict()
            if name == "puzzles":
                recs, meta = [], []
                for j in range(b.num_rows):
                    puzzle = {k: t[k][j] for k in t}
                    row, info = pack_puzzle(puzzle)
                    if row is None:
                        continue
                    recs.append(row)
                    meta.append((j, info))
                if recs:
                    yield stack_rows(recs), {"puzzle": meta}, fn, offset
            else:
                yield packed(b), t, fn, offset
            offset += b.num_rows


SCALAR_KEYS = {
    "turn", "castling", "ep_square", "move_idx", "cp", "mate",
    "label_depth", "phase", "source", "value_valid", "ep_file",
}


def _np(t) -> np.ndarray:
    return np.ascontiguousarray(np.squeeze(torch.as_tensor(t).detach().cpu().numpy()))


def squeeze_scalars(data: dict) -> dict:
    for k in SCALAR_KEYS:
        if k in data and torch.is_tensor(data[k]) and data[k].ndim > 1:
            data[k] = data[k].reshape(data[k].shape[0])
    return data


def take_row(d: dict, i: int, name: str, source_id: int, sm: dict) -> dict:
    row = {k: _np(d[k][i]) for k in CORE}
    row["source"] = np.int8(source_id)
    if name == "sf19":
        wdl = torch.as_tensor(sm["wdl"], dtype=torch.float32)
        row["wdl"] = _np(wdl)
        row["value_valid"] = np.int8(1 if value_ok(wdl) else 0)
    else:
        row["wdl"] = _np(DUMMY_WDL)
        row["value_valid"] = np.int8(0)
    return row


def source_meta(name: str, meta: dict, i: int) -> dict:
    if name == "puzzles":
        return meta["puzzle"][i][1]
    out = {}
    if name == "sf19":
        for k in ("split", "policy_mask", "nodes_budget", "nodes", "game_id", "wdl"):
            if k in meta:
                out[k] = meta[k][i]
    if name == "syzygy":
        for k in ("wdl", "dtz", "mate"):
            if k in meta:
                v = meta[k][i]
                out["tb_wdl" if k == "wdl" else k] = int(v) if not torch.is_tensor(v) else int(v)
    return out


def qualify(name: str, d: dict, i: int, meta: dict, sm: dict) -> str | None:
    if name == "sf19":
        if int(sm.get("split", 0)) != 0:
            return "upstream_holdout"
        if int(sm.get("policy_mask", 1)) != 1:
            return "teacher_quality"
        budget = int(sm.get("nodes_budget", 0))
        depth = int(d["label_depth"][i])
        if budget < SF19_MIN_BUDGET or depth < SF19_MIN_DEPTH:
            return "teacher_quality"
        if "wdl" in sm and not value_ok(torch.as_tensor(sm["wdl"], dtype=torch.float32)):
            return "invalid_value"
    if name == "lichess":
        depth = int(d["label_depth"][i])
        if not (LICHESS_MIN_DEPTH <= depth <= LICHESS_MAX_DEPTH):
            return "depth_or_sentinel"
    if name == "syzygy":
        ok, reason = syzygy_meta_ok(sm.get("tb_wdl", 0), sm.get("dtz", 0), sm.get("mate", 0))
        if not ok:
            return reason
    if not legal_row(d, i):
        return "invalid_position_or_targets"
    return None


def build(args) -> dict:
    torch.set_num_threads(2)
    load_hf_token()
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    rng = random.Random(args.seed)
    api = HfApi()

    if out.exists() and (out / "FROZEN.json").exists():
        raise SystemExit(f"Refusing to touch frozen mix at {out}; delete FROZEN.json to unfreeze")
    if out.exists() and (out / "manifest.json").exists():
        prev = json.loads((out / "manifest.json").read_text())
        if prev.get("status") in ("complete", "frozen") and not args.force:
            raise SystemExit(f"Refusing to overwrite {prev.get('status')} mix at {out}; pass --force")
        if prev.get("status") == "complete" and args.force:
            for p in out.iterdir():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)

    blocked, block_files = collect_blocked(out)
    progress = load_progress(out)
    completed = [n for n in progress.get("completed", []) if source_done(out, n)]
    seen = load_seen_from_shards(out, completed)
    seen.update(int(x) for x in progress.get("seen", []))
    all_eval_blocks = set(blocked)

    manifest = {
        "seed": args.seed,
        "requested_train_rows": args.rows,
        "eval_rows_per_source": args.eval_rows,
        "sources": {},
        "status": "building",
        "vocab": "compact1968",
        "mix": {k: v[1] for k, v in SOURCES.items()},
        "limitations": LIMITATIONS,
        "existing_block_manifests": block_files,
        "existing_blocked_count": len(blocked),
        "value_supervision": {
            "sf19": "white_absolute_wdl",
            "lichess": "masked",
            "puzzles": "policy_only",
            "syzygy": "masked_pending_tb_wdl_conversion",
        },
    }
    if (out / "manifest.json").exists():
        old = json.loads((out / "manifest.json").read_text())
        if old.get("status") == "building":
            manifest["sources"] = {k: v for k, v in old.get("sources", {}).items() if k in completed}

    json_write(out / "manifest.json", manifest)
    source_train, source_eval = {}, {}

    for name in completed:
        source_train[name] = torch.load(out / f"{name}_train.pt", map_location="cpu", weights_only=False)
        source_eval[name] = torch.load(out / f"{name}_eval.pt", map_location="cpu", weights_only=False)
        print(f"resume skip {name} train={len(source_train[name]['turn'])} eval={len(source_eval[name]['turn'])}", flush=True)

    local_lichess = ROOT / "outputs/hf_soft/multipv_lichess_soft.pt"
    local_syzygy = ROOT / "outputs/hf_soft/syzygy_soft.pt"

    for name, (repo, share, source_id) in SOURCES.items():
        if name in completed:
            continue
        target = int(round(args.rows * share))
        info = api.dataset_info(repo)
        files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
        rng.shuffle(files)
        report = {
            "repo": repo, "revision": info.sha, "inputs": [], "rejected": {},
            "train": 0, "eval": 0, "nodes_budget_vs_depth": {},
        }
        manifest["sources"][name] = report
        train, val = [], []
        groups: dict[str, bool] = {}
        phase_counts: dict[int, int] = {}
        rating_counts: dict[int, int] = {}

        def reject(reason: str) -> None:
            report["rejected"][reason] = report["rejected"].get(reason, 0) + 1

        prov_path = out / f"{name}_provenance.jsonl"
        provenance = prov_path.open("w")

        def consume(enforce_quota: bool) -> None:
            for d, meta, fn, offset in batches(
                name, files, report, rng, local_lichess, local_syzygy, info.sha, repo
            ):
                d["ep_square"] = torch.where(d["ep_square"] <= 0, -1, d["ep_square"]).to(torch.int8)
                h = position_hashes(d)
                canon = canonical_hashes(d)
                order = list(range(len(h)))
                rng.shuffle(order)
                for i in order:
                    if len(train) >= target and len(val) >= args.eval_rows:
                        return
                    key = int(canon[i])
                    if key in seen or key in blocked or int(h[i]) in blocked:
                        reject("duplicate_or_blocked")
                        continue
                    sm = source_meta(name, meta, i)
                    reason = qualify(name, d, i, meta, sm)
                    if reason:
                        reject(reason)
                        continue
                    group = group_key(name, sm, key)
                    if group not in groups:
                        groups[group] = is_eval_group(group, 2)
                    is_eval = groups[group]
                    dest = val if is_eval else train
                    cap = args.eval_rows if is_eval else target
                    if len(dest) >= cap:
                        reject("split_full")
                        continue
                    ph = int(d["phase"][i])
                    if enforce_quota and (not is_eval) and name in ("sf19", "lichess"):
                        if phase_counts.get(ph, 0) >= max(1, round(target * PHASE_FRAC.get(ph, 0))):
                            reject("phase_quota")
                            continue
                    if enforce_quota and (not is_eval) and name == "puzzles":
                        rating = int(sm.get("Rating") or 0)
                        bucket = rating_bucket(rating)
                        if rating_counts.get(bucket, 0) >= max(1, target // 5):
                            reject("rating_quota")
                            continue
                    row = take_row(d, i, name, source_id, sm)
                    if name == "sf19" and int(row["value_valid"]) != 1:
                        reject("invalid_value")
                        continue
                    dest.append(row)
                    seen.add(key)
                    if is_eval:
                        all_eval_blocks.update([key, int(h[i])])
                        if int(d["castling"][i]) == 0:
                            all_eval_blocks.add(int(position_hashes(hflip_cache_slice(d, torch.tensor([i])))[0]))
                    else:
                        phase_counts[ph] = phase_counts.get(ph, 0) + 1
                        if name == "puzzles":
                            rating_counts[rating_bucket(int(sm.get("Rating") or 0))] = (
                                rating_counts.get(rating_bucket(int(sm.get("Rating") or 0)), 0) + 1
                            )
                    rec = {
                        "split": "eval" if is_eval else "train",
                        "row": len(dest) - 1,
                        "input": fn,
                        "input_row": offset + (meta["puzzle"][i][0] if name == "puzzles" else i),
                        "group": group,
                        "canonical_hash": key,
                    }
                    if name == "puzzles":
                        rec["puzzle"] = sm
                    if name == "sf19":
                        rec["nodes_budget"] = int(sm.get("nodes_budget", 0))
                        rec["nodes_achieved"] = int(sm.get("nodes", 0))
                        rec["label_depth"] = int(d["label_depth"][i])
                        rec["game_id"] = int(sm.get("game_id", -1))
                    if name == "syzygy":
                        rec["tb_wdl"] = int(sm.get("tb_wdl", 0))
                        rec["dtz"] = int(sm.get("dtz", 0))
                        rec["exported_mate_ignored"] = int(sm.get("mate", 0))
                    provenance.write(json.dumps(rec) + "\n")
                print(
                    name, "train", len(train), "eval", len(val),
                    "rejected", sum(report["rejected"].values()),
                    "quota" if enforce_quota else "fill",
                    flush=True,
                )

        consume(True)
        if len(train) < target or len(val) < args.eval_rows:
            report["quota_relaxed"] = True
            print(name, "relaxing phase/rating quotas to fill shortfall", flush=True)
            consume(False)
        provenance.close()
        if len(train) < target or len(val) < args.eval_rows:
            json_write(out / "manifest.json", manifest)
            raise RuntimeError(
                f"{name}: insufficient qualifying rows {len(train)}/{target}, eval {len(val)}/{args.eval_rows}"
            )
        source_train[name] = squeeze_scalars(attach_static_targets(stack_rows(train)))
        source_eval[name] = squeeze_scalars(attach_static_targets(stack_rows(val)))
        torch.save(source_train[name], out / f"{name}_train.pt")
        torch.save(source_eval[name], out / f"{name}_eval.pt")
        n_tr = len(train)
        report.update(
            train=n_tr, eval=len(val),
            phase_train=phase_counts, rating_buckets_train=rating_counts,
            actual_share=n_tr / args.rows,
            value_valid_train=int(source_train[name]["value_valid"].sum()),
        )
        completed.append(name)
        save_progress(out, completed, seen)
        json_write(out / "manifest.json", manifest)

    soft_names = ("sf19", "lichess", "puzzles")
    keys = [k for k in source_train["sf19"] if all(k in source_train[n] for n in soft_names)]
    soft = squeeze_scalars({k: torch.cat([source_train[n][k] for n in soft_names]) for k in keys})
    torch.save(soft, out / "soft_cache.pt")
    torch.save(squeeze_scalars(source_train["syzygy"]), out / "deep_cache.pt")
    json_write(out / "blocked_manifest.json", {
        "blocked_hashes": sorted(all_eval_blocks),
        "note": "Includes prior holdouts plus this mix's eval rows and safe flips. Not an unseen-by-old-checkpoints claim.",
    })
    actual = {n: int(source_train[n]["turn"].shape[0]) for n in SOURCES}
    total = sum(actual.values())
    manifest.update({
        "status": "complete",
        "actual_train_rows": total,
        "actual_proportions": {n: actual[n] / total for n in SOURCES},
        "actual_counts": actual,
        "artifacts": {p.name: fingerprint(p) for p in sorted(out.glob("*.pt"))},
        "trainer": {
            "deep_mix_frac": 0.05,
            "bonus_mix_frac": 0,
            "soft_cache": "soft_cache.pt",
            "deep_cache": "deep_cache.pt",
        },
    })
    json_write(out / "manifest.json", manifest)
    print("COMPLETE", out, flush=True)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/organized_chess_v1")
    p.add_argument("--rows", type=int, default=1_000_000)
    p.add_argument("--eval-rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--force", action="store_true", help="Replace a complete mix directory")
    build(p.parse_args())
