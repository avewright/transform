#!/usr/bin/env python3
"""exp279: train a router over frozen 99M specialists.

Experts (frozen): incumbent, puzzle, endgame, opening, middlegame.
Next-move rows come from HF (bestline + MultiPV + puzzles). Score each
row with every expert, then train only the router:

    L = softmax(router) · expert_ce

  MOVE_VOCAB_VERSION=compact python experiments/exp279_moe_router.py --pull
  MOVE_VOCAB_VERSION=compact python experiments/exp279_moe_router.py --score
  MOVE_VOCAB_VERSION=compact python experiments/exp279_moe_router.py --go
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_moe import (  # noqa: E402
    EXPERTS,
    EXPERT_NAMES,
    N_EXPERTS,
    FrozenExpertMoE,
    best_expert_from_ce,
    build_router,
    expected_ce_loss,
    freeze_expert,
    phase_prior,
    robust_best_expert,
    router_param_count,
    soft_expert_targets,
    soft_target_loss,
    TRUNK_HIDDEN_DIM,
)
from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG, build_squares64
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "exp279_moe_router"
HF_DIR = ROOT / "outputs" / "hf_models" / "experts"


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit("Expected compact vocab 1968. Export MOVE_VOCAB_VERSION=compact.")


def expert_local_dir(name: str) -> Path:
    return HF_DIR / name


def pull_experts(*, skip_missing: bool = False) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download
    from upload_exp201_hf import load_hf_token

    token = load_hf_token()
    found: dict[str, Path] = {}
    local_incumbent = ROOT / "outputs" / "hf_models" / "99m" / "latest.pt"
    for name, repo in EXPERTS:
        dest = expert_local_dir(name)
        dest.mkdir(parents=True, exist_ok=True)
        ckpt = dest / "latest.pt"
        try:
            if name == "incumbent" and local_incumbent.exists() and not ckpt.exists():
                shutil.copy2(local_incumbent, ckpt)
                print(f"copied local incumbent -> {ckpt}", flush=True)
            else:
                ckpt = Path(hf_hub_download(repo, "latest.pt", local_dir=str(dest), token=token))
                try:
                    hf_hub_download(repo, "model_config.json", local_dir=str(dest), token=token)
                except Exception:
                    pass
                print(f"pulled {repo} -> {ckpt}", flush=True)
            found[name] = ckpt
        except Exception as e:
            msg = f"missing {repo}: {e}"
            if skip_missing:
                print(f"skip {msg}", flush=True)
                continue
            raise SystemExit(msg)
    return found


def load_frozen_expert(ckpt: Path, device: torch.device):
    model = build_squares64(DEFAULT_100M_SQUARES64_CONFIG)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    from autoresearch_8gb.pipeline import load_model_state

    state = load_model_state(blob if isinstance(blob, dict) else {"model_state_dict": blob})
    model.load_state_dict(state, strict=False)
    return freeze_expert(model.to(device))


ROW_KEYS = ("board_array", "turn", "castling", "ep_square", "move_idx")
HF_NEXTMOVE = (
    ("opening", "avewright/lichess-opening-bestline"),
    ("middlegame", "avewright/lichess-middlegame-bestline"),
    ("endgame", "avewright/lichess-endgame-bestline"),
    ("general", "avewright/chess-soft-multipv-lichess"),
    ("puzzles", "Lichess/chess-puzzles"),
)
LICHESS_INBOXES = (
    ("opening", ROOT / "outputs" / "lichess_opening_bestline" / "inbox"),
    ("middlegame", ROOT / "outputs" / "lichess_middlegame_bestline" / "inbox"),
    ("endgame", ROOT / "outputs" / "lichess_endgame_bestline" / "inbox"),
)
HOLD_OUTS = (
    ("opening_val", ROOT / "outputs" / "exp277_lichess_opening_stream" / "lichess_eval.pt"),
    ("middlegame_val", ROOT / "outputs" / "exp278_lichess_middlegame_stream" / "lichess_eval.pt"),
    ("endgame_val", ROOT / "outputs" / "exp276_lichess_endgame_stream" / "lichess_eval.pt"),
)


def _inbox_shards(inbox: Path) -> list[Path]:
    found: list[Path] = []
    for sh in sorted(inbox.glob("shard_*")):
        if (sh / "soft_cache.pt").exists() and (
            (sh / "READY").exists() or (sh / "ATTACHED").exists()
        ):
            found.append(sh)
    return found


def sample_lichess_inbox(inbox: Path, n_take: int, rng: torch.Generator) -> dict | None:
    """Uniform-over-shards then row subsample. Avoids loading the full 50M pack."""
    shards = _inbox_shards(inbox)
    if not shards or n_take <= 0:
        return None
    chunks: list[dict] = []
    left = int(n_take)
    order = torch.randperm(len(shards), generator=rng).tolist()
    keys = ROW_KEYS
    for i in order:
        if left <= 0:
            break
        data = torch.load(shards[i] / "soft_cache.pt", map_location="cpu", weights_only=False)
        n = int(data["move_idx"].shape[0])
        take = min(left, n)
        if take < n:
            idx = torch.randperm(n, generator=rng)[:take]
            data = {k: data[k][idx] for k in ROW_KEYS if k in data}
        else:
            data = {k: data[k] for k in ROW_KEYS if k in data}
        chunks.append(data)
        left -= take
    if not chunks:
        return None
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}


def _row_keys(data: dict) -> list[str]:
    n = int(data["move_idx"].shape[0])
    return [k for k, v in data.items() if torch.is_tensor(v) and int(v.shape[0]) == n]


def attach_onehot(data: dict) -> dict:
    """prepare_soft_batch needs soft_* and cp/mate. Next-move labels are one-hot."""
    n = int(data["move_idx"].shape[0])
    data = dict(data)
    mid = data["move_idx"].long()
    si = torch.full((n, 8), -1, dtype=torch.int64)
    sp = torch.zeros(n, 8, dtype=torch.float32)
    si[:, 0] = mid
    sp[:, 0] = 1.0
    data["soft_indices"] = si
    data["soft_probs"] = sp
    if "cp" not in data:
        data["cp"] = torch.zeros(n, dtype=torch.int32)
    if "mate" not in data:
        data["mate"] = torch.zeros(n, dtype=torch.int32)
    return data


def _with_holdout(data: dict, holdout: bool) -> dict:
    data = attach_onehot(data)
    n = int(data["move_idx"].shape[0])
    data["is_holdout"] = torch.full((n,), holdout, dtype=torch.bool)
    return data


def _split_holdout(data: dict, n_val: int, rng: torch.Generator) -> tuple[dict, dict]:
    n = int(data["move_idx"].shape[0])
    n_val = min(int(n_val), max(0, n - 1))
    perm = torch.randperm(n, generator=rng)
    keys = _row_keys(data)
    val = {k: data[k][perm[:n_val]] for k in keys}
    train = {k: data[k][perm[n_val:]] for k in keys}
    return train, val


def parquet_to_rows(path: Path) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(ROW_KEYS))
    out: dict[str, torch.Tensor] = {}
    for name in ROW_KEYS:
        col = table.column(name)
        try:
            arr = col.to_numpy(zero_copy_only=False)
        except Exception:
            arr = None
        if arr is None or getattr(arr, "dtype", None) == object or name == "board_array":
            arr = np.asarray(col.to_pylist())
            if name == "board_array":
                arr = np.stack([np.asarray(x, dtype=np.int8) for x in arr])
        if name == "board_array":
            out[name] = torch.from_numpy(np.ascontiguousarray(arr).astype(np.int8))
        elif name == "move_idx":
            out[name] = torch.from_numpy(np.ascontiguousarray(arr).reshape(-1).astype(np.int64))
        else:
            out[name] = torch.from_numpy(np.ascontiguousarray(arr).reshape(-1).astype(np.int8))
    return out


def sample_hf_parquet(repo: str, n_take: int, rng: torch.Generator, token: str | None) -> dict | None:
    from huggingface_hub import HfApi, hf_hub_download

    if n_take <= 0:
        return None
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    if not files:
        print(f"skip {repo}: no parquet", flush=True)
        return None
    order = torch.randperm(len(files), generator=rng).tolist()
    chunks: list[dict] = []
    left = int(n_take)
    for i in order:
        if left <= 0:
            break
        local = Path(hf_hub_download(repo, files[i], repo_type="dataset", token=token))
        data = parquet_to_rows(local)
        n = int(data["move_idx"].shape[0])
        if n == 0:
            continue
        take = min(left, n)
        if take < n:
            idx = torch.randperm(n, generator=rng)[:take]
            data = {k: data[k][idx] for k in ROW_KEYS}
        chunks.append(data)
        left -= take
        print(f"  {repo} {files[i]} +{take:,} left={left:,}", flush=True)
    if not chunks:
        return None
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in ROW_KEYS}


def sample_hf_puzzles(n_take: int, seed: int) -> dict | None:
    if n_take <= 0:
        return None
    from datasets import load_dataset
    from exp273_puzzle_finetune import _stack_rows, play_puzzle

    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    try:
        ds = ds.shuffle(seed=int(seed), buffer_size=10_000)
    except Exception:
        pass
    rows: list = []
    skipped = 0
    for rec in ds:
        got = play_puzzle(rec)
        if not got:
            skipped += 1
            continue
        rows.extend(got)
        if len(rows) >= n_take:
            break
        if len(rows) % 20000 < 8:
            print(f"  puzzles {len(rows):,}/{n_take:,} skipped={skipped}", flush=True)
    stacked = _stack_rows(rows[:n_take], 0)
    if stacked is None:
        return None
    print(f"  Lichess/chess-puzzles n={int(stacked['move_idx'].shape[0]):,} skipped={skipped}", flush=True)
    return {k: stacked[k] for k in ROW_KEYS}


def default_score_tables(*, rows_per_source: int, seed: int = 279, val_n: int = 4096) -> list[tuple[str, dict]]:
    from upload_exp201_hf import load_hf_token

    token = load_hf_token()
    rng = torch.Generator().manual_seed(int(seed))
    out: list[tuple[str, dict]] = []
    for name, path in HOLD_OUTS:
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=False)
            out.append((name, _with_holdout({k: data[k] for k in ROW_KEYS if k in data}, True)))
        else:
            print(f"skip missing holdout {name} {path}", flush=True)
    for name, repo in HF_NEXTMOVE:
        print(f"sample HF {name} {repo} n={rows_per_source:,}", flush=True)
        try:
            if name == "puzzles":
                sampled = sample_hf_puzzles(int(rows_per_source) + int(val_n), seed)
            else:
                sampled = sample_hf_parquet(repo, int(rows_per_source) + int(val_n), rng, token)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        if sampled is None or int(sampled["move_idx"].shape[0]) == 0:
            print(f"skip empty {name}", flush=True)
            continue
        train, val = _split_holdout(sampled, val_n, rng)
        if int(train["move_idx"].shape[0]) > 0:
            out.append((name, _with_holdout(train, False)))
        if int(val["move_idx"].shape[0]) > 0:
            out.append((f"{name}_val", _with_holdout(val, True)))
        print(f"  kept train={int(train['move_idx'].shape[0]):,} val={int(val['move_idx'].shape[0]):,}", flush=True)
    return out


@torch.no_grad()
def score_cache(model, data: dict, device: torch.device, *, micro: int = 256) -> torch.Tensor:
    from autoresearch_8gb.pipeline import prepare_soft_batch

    n = int(data["move_idx"].shape[0])
    ce = torch.empty(n, dtype=torch.float32)
    for start in range(0, n, micro):
        idx = torch.arange(start, min(start + micro, n))
        bi, hard, *_ = prepare_soft_batch(data, idx, device, hflip_p=0.0)
        logits = model(bi)["policy_logits"].float()
        ce[start : start + idx.numel()] = F.cross_entropy(logits, hard, reduction="none").cpu()
    return ce


def score_all(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables = default_score_tables(
        rows_per_source=int(args.rows_per_source),
        seed=int(args.seed),
        val_n=int(args.val_n),
    )
    if not tables:
        raise SystemExit("no HF next-move tables downloaded")
    ckpts = {n: expert_local_dir(n) / "latest.pt" for n, _ in EXPERTS}
    missing = [n for n, p in ckpts.items() if not p.exists()]
    if missing:
        raise SystemExit(f"pull experts first; missing {missing}")

    OUT.mkdir(parents=True, exist_ok=True)
    parts: list[dict] = []
    for src_name, data in tables:
        n = int(data["move_idx"].shape[0])
        print(f"score source={src_name} n={n:,}", flush=True)
        ces = torch.zeros(n, N_EXPERTS, dtype=torch.float32)
        for e, (ename, _) in enumerate(EXPERTS):
            print(f"  expert {ename}", flush=True)
            model = load_frozen_expert(ckpts[ename], device)
            ces[:, e] = score_cache(model, data, device, micro=int(args.microbatch))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rec = dict(data)
        rec["expert_ce"] = ces
        rec["best_expert"] = best_expert_from_ce(ces)
        parts.append(rec)
        oracle = float(ces.min(dim=1).values.mean())
        print(
            f"  oracle_ce={oracle:.4f} win_share="
            f"{[round(float((rec['best_expert']==i).float().mean()), 3) for i in range(N_EXPERTS)]}",
            flush=True,
        )

    keys = [
        "board_array", "turn", "castling", "ep_square", "move_idx",
        "soft_indices", "soft_probs", "cp", "mate",
        "expert_ce", "best_expert", "is_holdout",
    ]
    table = {k: torch.cat([p[k] for p in parts], dim=0) for k in keys}
    dest = OUT / "router_labels.pt"
    torch.save(table, dest)
    n = int(table["best_expert"].shape[0])
    hist = {EXPERT_NAMES[i]: int((table["best_expert"] == i).sum()) for i in range(N_EXPERTS)}
    report = {
        "n": n,
        "best_expert_hist": hist,
        "sources": [s for s, _ in tables],
        "oracle_ce": float(table["expert_ce"].min(dim=1).values.mean()),
    }
    (OUT / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SCORE", json.dumps(report, indent=2), flush=True)
    return dest


# Layout of router_labels.pt from the 8k/1k score pass (holdouts + HF_NEXTMOVE).
_LABEL_LAYOUT_61504: tuple[tuple[str, int, bool], ...] = (
    ("opening_holdout", 8192, True),
    ("middlegame_holdout", 8192, True),
    ("opening", 8000, False),
    ("opening", 1024, True),
    ("middlegame", 8000, False),
    ("middlegame", 1024, True),
    ("endgame", 8000, False),
    ("endgame", 1024, True),
    ("general", 8000, False),
    ("general", 1024, True),
    ("puzzles", 8000, False),
    ("puzzles", 1024, True),
)


# Dataset → frozen expert. "general" is the mixed-game set, so incumbent.
SOURCE_TO_EXPERT = {
    "opening": "opening",
    "middlegame": "middlegame",
    "endgame": "endgame",
    "general": "incumbent",
    "puzzles": "puzzle",
}
SOURCE_NAMES = ("opening", "middlegame", "endgame", "general", "puzzles")


def source_row_labels(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(source_id 0..4, expert_id) for the 61,504-row cache."""
    if n != 61504:
        raise ValueError(f"source labels need the 61504-row cache, n={n}")
    src = torch.empty(n, dtype=torch.long)
    expert = torch.empty(n, dtype=torch.long)
    pos = 0
    for name, size, _hold in _LABEL_LAYOUT_61504:
        key = name.replace("_holdout", "")
        src[pos : pos + size] = SOURCE_NAMES.index(key)
        expert[pos : pos + size] = EXPERT_NAMES.index(SOURCE_TO_EXPERT[key])
        pos += size
    return src, expert


def attach_trunk_hidden(data: dict, device: torch.device, *, micro: int = 256) -> dict:
    """Cache incumbent 99M global_hidden. Router is a head on this encode."""
    n = int(data["move_idx"].shape[0])
    cache = OUT / f"trunk_hidden_n{n}.pt"
    if "trunk_hidden" in data and int(data["trunk_hidden"].shape[0]) == n:
        return data
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        hid = blob["trunk_hidden"] if isinstance(blob, dict) else blob
        if torch.is_tensor(hid) and int(hid.shape[0]) == n and int(hid.shape[-1]) == TRUNK_HIDDEN_DIM:
            data = dict(data)
            data["trunk_hidden"] = hid
            print(f"loaded trunk_hidden {tuple(hid.shape)} from {cache}", flush=True)
            return data
    print(f"encode incumbent trunk n={n:,} dim={TRUNK_HIDDEN_DIM}", flush=True)
    ckpt = expert_local_dir("incumbent") / "latest.pt"
    if not ckpt.exists():
        raise SystemExit(f"missing incumbent trunk {ckpt}")
    model = load_frozen_expert(ckpt, device)
    from autoresearch_8gb.pipeline import prepare_soft_batch

    hid = torch.empty(n, TRUNK_HIDDEN_DIM, dtype=torch.float32)
    for start in range(0, n, micro):
        idx = torch.arange(start, min(start + micro, n))
        bi, *_ = prepare_soft_batch(data, idx, device, hflip_p=0.0)
        hid[start : start + idx.numel()] = model(bi)["global_hidden"].float().cpu()
        if start == 0 or (start + idx.numel()) % 4096 == 0 or start + idx.numel() == n:
            print(f"  trunk {start + idx.numel():,}/{n:,}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.save({"trunk_hidden": hid, "stem": "incumbent", "n": n}, cache)
    data = dict(data)
    data["trunk_hidden"] = hid
    print(f"wrote {cache}", flush=True)
    return data


def _tag_cls_rows(data: dict, name: str, holdout: bool) -> dict:
    data = attach_onehot({k: data[k] for k in ROW_KEYS if k in data})
    n = int(data["move_idx"].shape[0])
    data["is_holdout"] = torch.full((n,), holdout, dtype=torch.bool)
    data["source_id"] = torch.full((n,), SOURCE_NAMES.index(name), dtype=torch.long)
    data["source_expert"] = torch.full(
        (n,), EXPERT_NAMES.index(SOURCE_TO_EXPERT[name]), dtype=torch.long
    )
    return data


def sample_specialist_rows(
    name: str, n_take: int, rng: torch.Generator, token: str | None, seed: int
) -> dict | None:
    repo = dict(HF_NEXTMOVE)[name]
    if name != "puzzles":
        inbox = dict(LICHESS_INBOXES).get(name)
        if inbox is not None:
            got = sample_lichess_inbox(inbox, n_take, rng)
            if got is not None and int(got["move_idx"].shape[0]) > 0:
                print(f"  inbox {name} n={int(got['move_idx'].shape[0]):,}", flush=True)
                return got
        return sample_hf_parquet(repo, n_take, rng, token)
    return sample_hf_puzzles(n_take, seed)


def build_source_pack(rows_per_source: int, val_n: int, seed: int) -> dict:
    from upload_exp201_hf import load_hf_token

    token = load_hf_token()
    rng = torch.Generator().manual_seed(int(seed))
    parts: list[dict] = []
    for name in ("opening", "middlegame", "endgame", "puzzles"):
        need = int(rows_per_source) + int(val_n)
        print(f"pack source={name} n={need:,}", flush=True)
        sampled = sample_specialist_rows(name, need, rng, token, seed)
        if sampled is None or int(sampled["move_idx"].shape[0]) == 0:
            raise SystemExit(f"failed to sample {name}")
        train, val = _split_holdout(sampled, val_n, rng)
        parts.append(_tag_cls_rows(train, name, False))
        if int(val["move_idx"].shape[0]) > 0:
            parts.append(_tag_cls_rows(val, name, True))
        print(
            f"  {name} train={int(train['move_idx'].shape[0]):,} val={int(val['move_idx'].shape[0]):,}",
            flush=True,
        )
    keys = [k for k, v in parts[0].items() if torch.is_tensor(v)]
    out = {k: torch.cat([p[k] for p in parts], dim=0) for k in keys}
    dest = OUT / "source_pack.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, dest)
    print(f"wrote {dest} n={int(out['move_idx'].shape[0]):,}", flush=True)
    return out


def load_or_build_source_pack(args: argparse.Namespace) -> dict:
    rows = int(args.pack_rows)
    val_n = int(args.val_n)
    dest = OUT / "source_pack.pt"
    if dest.exists() and not getattr(args, "rebuild_pack", False):
        raw = torch.load(dest, map_location="cpu", weights_only=False)
        n = int(raw["move_idx"].shape[0])
        if n >= 4 * rows:
            print(f"loaded {dest} n={n:,}", flush=True)
            return raw
        print(f"rebuild pack: have {n:,} want >= {4 * rows:,}", flush=True)
    return build_source_pack(rows, val_n, int(args.seed))


def attach_source_route_targets(data: dict) -> dict:
    n = int(data["move_idx"].shape[0])
    if "source_id" in data and "source_expert" in data:
        return data
    src, expert = source_row_labels(n)
    data = dict(data)
    data["source_id"] = src
    data["source_expert"] = expert
    return data


def label_source_slices(n: int) -> dict[str, tuple[slice, slice]]:
    """source -> (train_slice, val_slice) for the known 61,504-row cache."""
    if n != 61504:
        return {}
    out: dict[str, list[slice | None]] = {}
    pos = 0
    for name, size, hold in _LABEL_LAYOUT_61504:
        sl = slice(pos, pos + size)
        pos += size
        rec = out.setdefault(name.replace("_holdout", ""), [None, None])
        rec[1 if hold else 0] = sl
    return {k: (tr, va) for k, (tr, va) in out.items() if tr is not None or va is not None}


def rows_for_sources(n: int, names: list[str]) -> torch.Tensor:
    """All rows (train + val + holdout) whose source is in `names`."""
    if n != 61504:
        raise SystemExit(f"source subset needs the 61504-row cache (n={n})")
    keep = set(names)
    parts: list[torch.Tensor] = []
    pos = 0
    for name, size, _hold in _LABEL_LAYOUT_61504:
        key = name.replace("_holdout", "")
        if key in keep:
            parts.append(torch.arange(pos, pos + size))
        pos += size
    if not parts:
        raise SystemExit(f"no rows for sources {names}")
    return torch.cat(parts)


def parse_subset_names(subset: str) -> list[str] | None:
    if subset in ("", "all"):
        return None
    if subset in ("specialists", "nongeneral"):
        return ["opening", "middlegame", "endgame", "puzzles"]
    names = [s.strip() for s in subset.split(",") if s.strip()]
    unknown = [s for s in names if s not in SOURCE_NAMES]
    if unknown:
        raise SystemExit(f"unknown subset {unknown}; want {SOURCE_NAMES} or specialists")
    return names


def subset_label_rows(data: dict, subset: str) -> dict:
    n = int(data["move_idx"].shape[0])
    names = parse_subset_names(subset)
    if names is None:
        return data
    idx = rows_for_sources(n, names)
    return {k: v[idx] if torch.is_tensor(v) and int(v.shape[0]) == n else v for k, v in data.items()}


def train_router(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack_rows = int(getattr(args, "pack_rows", 0) or 0)
    if args.objective == "source" and pack_rows > 0:
        data = load_or_build_source_pack(args)
    else:
        labels = OUT / "router_labels.pt"
        if not labels.exists():
            raise SystemExit("run --score first")
        raw = torch.load(labels, map_location="cpu", weights_only=False)
        if getattr(args, "objective", "soft") == "source":
            raw = attach_source_route_targets(raw)
        data = subset_label_rows(raw, args.subset)
    if "expert_ce" in data:
        data["best_expert"] = robust_best_expert(data["expert_ce"])
    if args.objective == "source":
        if "source_expert" not in data:
            raise SystemExit("source objective needs source_expert labels")
        data["best_expert"] = data["source_expert"]
    n = int(data["best_expert"].shape[0])
    hold = data.get("is_holdout")
    if hold is not None and bool(hold.any()) and bool((~hold).any()):
        val_idx = hold.nonzero(as_tuple=False).squeeze(1)
        train_idx = (~hold).nonzero(as_tuple=False).squeeze(1)
    else:
        perm = torch.randperm(n)
        n_val = min(8192, max(256, n // 10))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
    overfit_n = int(getattr(args, "overfit_n", 0) or 0)
    if overfit_n > 0:
        rng = torch.Generator().manual_seed(int(args.seed))
        perm = train_idx[torch.randperm(int(train_idx.numel()), generator=rng)]
        train_idx = perm[: min(overfit_n, int(perm.numel()))]
        val_idx = train_idx
    from autoresearch_8gb.pipeline import prepare_soft_batch

    stem_ckpt = expert_local_dir("incumbent") / "latest.pt"
    if not stem_ckpt.exists():
        raise SystemExit(f"missing incumbent trunk {stem_ckpt}")
    stem = load_frozen_expert(stem_ckpt, device)
    print(f"frozen stem={stem_ckpt} (no grad)", flush=True)

    def _encode(idx: torch.Tensor, *, hflip: float = 0.0) -> torch.Tensor:
        bi, *_ = prepare_soft_batch(data, idx, device, hflip_p=hflip)
        with torch.no_grad():
            return stem(bi)["global_hidden"]

    router = build_router().to(device)
    start_step = 0
    resume_path = Path(args.router_ckpt) if getattr(args, "router_ckpt", "") else None
    if getattr(args, "resume", False) and resume_path is None:
        resume_path = OUT / "router.pt"
    if resume_path and resume_path.exists():
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        try:
            router.load_state_dict(blob["model_state_dict"])
            start_step = int(blob.get("steps", 0))
            print(f"resumed {resume_path} steps={start_step}", flush=True)
        except RuntimeError as e:
            print(f"skip incompatible resume ({e})", flush=True)
    opt = torch.optim.AdamW(router.parameters(), lr=float(args.lr), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(args.max_steps), 1), eta_min=1e-5)
    tau = float(getattr(args, "tau", 0.4))
    balance_w = float(getattr(args, "balance", 0.05))
    hflip_p = 0.5 if args.objective == "source" else 0.0
    smooth = 0.1 if args.objective == "source" else 0.0
    print(
        f"router params={router_param_count():,} subset={args.subset} "
        f"objective={args.objective} train={len(train_idx):,} val={len(val_idx):,} "
        f"tau={tau} balance={balance_w} overfit={overfit_n} hflip={hflip_p} "
        f"smooth={smooth} lr={args.lr}",
        flush=True,
    )
    if args.objective == "source":
        print(f"source→expert {SOURCE_TO_EXPERT}", flush=True)
        for split, idx in (("train", train_idx), ("val", val_idx)):
            hist = {SOURCE_NAMES[i]: int((data["source_id"][idx] == i).sum()) for i in range(5)}
            if "expert_ce" in data:
                g = data["source_expert"][idx]
                ceiling = float(data["expert_ce"][idx].gather(1, g.unsqueeze(1)).mean())
                print(f"  {split} source_ceiling_ce={ceiling:.4f} n={hist}", flush=True)
            else:
                print(f"  {split} n={hist}", flush=True)

    def _eval(idx: torch.Tensor) -> dict:
        router.eval()
        n = loss_sum = route_sum = prior_sum = oracle_sum = acc_sum = cls_sum = 0.0
        expert_sum = [0.0] * N_EXPERTS
        route_hist = [0.0] * N_EXPERTS
        phase_n = [0.0, 0.0, 0.0]
        phase_route = [0.0, 0.0, 0.0]
        src_n = [0.0] * 5
        src_acc = [0.0] * 5
        src_route = [0.0] * 5
        with torch.no_grad():
            for start in range(0, int(idx.numel()), 512):
                take = idx[start : start + 512]
                logits = router(_encode(take, hflip=0.0))
                gold = data["best_expert"][take].to(device)
                ce = data["expert_ce"][take].to(device) if "expert_ce" in data else None
                w = float(take.numel())
                n += w
                cls_sum += float(F.cross_entropy(logits, gold)) * w
                pick = logits.argmax(-1)
                if ce is not None:
                    loss_sum += float(expected_ce_loss(logits, ce)) * w
                    routed = ce.gather(1, pick.unsqueeze(1)).squeeze(1)
                    route_sum += float(routed.mean()) * w
                    prior = phase_prior(data["board_array"][take].ne(0).sum(1).to(device))
                    prior_sum += float(ce.gather(1, prior.unsqueeze(1)).mean()) * w
                    oracle_sum += float(ce.min(dim=1).values.mean()) * w
                    for e in range(N_EXPERTS):
                        expert_sum[e] += float(ce[:, e].mean()) * w
                else:
                    routed = torch.zeros(int(take.numel()), device=device)
                acc_sum += float((pick == gold).float().mean()) * w
                pcs = data["board_array"][take].ne(0).sum(1).to(device)
                for e in range(N_EXPERTS):
                    route_hist[e] += float((pick == e).sum())
                if "source_id" in data:
                    sid = data["source_id"][take]
                    for s in range(5):
                        sm = sid == s
                        if bool(sm.any()):
                            src_n[s] += float(sm.sum())
                            src_acc[s] += float((pick[sm] == gold[sm]).sum())
                            if ce is not None:
                                src_route[s] += float(routed[sm].sum())
                buckets = (pcs >= 26, (pcs >= 14) & (pcs < 26), pcs <= 13)
                for i, mask in enumerate(buckets):
                    if bool(mask.any()):
                        phase_n[i] += float(mask.sum())
                        phase_route[i] += float(routed[mask].sum())
        router.train()
        return {
            "exp_ce": (cls_sum if args.objective == "source" else loss_sum) / max(n, 1),
            "cls_ce": cls_sum / max(n, 1),
            "route_ce": route_sum / max(n, 1),
            "prior_ce": prior_sum / max(n, 1),
            "oracle_ce": oracle_sum / max(n, 1),
            "best_acc": acc_sum / max(n, 1),
            "expert_ce": [s / max(n, 1) for s in expert_sum],
            "route_hist": [h / max(n, 1) for h in route_hist],
            "phase_route": [phase_route[i] / max(phase_n[i], 1) for i in range(3)],
            "src_acc": [src_acc[i] / max(src_n[i], 1) for i in range(5)],
            "src_route": [src_route[i] / max(src_n[i], 1) for i in range(5)],
        }

    def _save(path: Path, step: int, extra: dict | None = None) -> None:
        blob = {
            "arch": "frozen_moe_router",
            "vocab_version": "compact",
            "model_state_dict": router.state_dict(),
            "experts": EXPERTS,
            "steps": step,
        }
        if extra:
            blob.update(extra)
        torch.save(blob, path)

    # Winner-stratified batches: each specialist's wins are seen, not a usage quota.
    gold_tr = data["best_expert"][train_idx]
    buckets = [train_idx[gold_tr == e] for e in range(N_EXPERTS)]
    buckets = [b for b in buckets if int(b.numel()) > 0]

    steps = int(args.max_steps)
    bs = int(args.batch_size)
    log_path = OUT / "train.log"
    OUT.mkdir(parents=True, exist_ok=True)
    best_route = float("inf")
    best_cls = float("inf")
    for step in range(1, steps + 1):
        parts = []
        left = bs
        for i, bkt in enumerate(buckets):
            take_n = left // (len(buckets) - i)
            parts.append(bkt[torch.randint(0, int(bkt.numel()), (take_n,))])
            left -= take_n
        pick = torch.cat(parts)
        logits = router(_encode(pick, hflip=hflip_p))
        if args.objective == "source":
            loss = F.cross_entropy(
                logits, data["source_expert"][pick].to(device), label_smoothing=smooth
            )
        else:
            ce = data["expert_ce"][pick].to(device)
            soft = soft_target_loss(logits, ce, tau=tau)
            exp_ce = expected_ce_loss(logits, ce)
            gates = logits.softmax(-1)
            want = soft_expert_targets(ce, tau).mean(0)
            balance = F.kl_div(gates.mean(0).clamp_min(1e-8).log(), want, reduction="batchmean")
            loss = soft + 0.25 * exp_ce + balance_w * balance
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            ev = _eval(val_idx)
            probe_n = min(8192, int(train_idx.numel()))
            probe = train_idx[torch.randperm(int(train_idx.numel()))[:probe_n]]
            tr = _eval(probe) if overfit_n > 0 or probe_n > 0 else ev
            now = datetime.now().strftime("%H:%M:%S")
            fixed = " ".join(f"{EXPERT_NAMES[i][:3]}={ev['expert_ce'][i]:.4f}" for i in range(N_EXPERTS))
            share = " ".join(f"r{EXPERT_NAMES[i][:3]}={ev['route_hist'][i]:.2f}" for i in range(N_EXPERTS))
            sacc = " ".join(f"{SOURCE_NAMES[i][:3]}={ev['src_acc'][i]:.2f}" for i in range(5))
            line = (
                f"[{now}] step {step}/{steps} | loss={float(loss):.4f} "
                f"| val_exp_ce={ev['exp_ce']:.4f} val_route_ce={ev['route_ce']:.4f} "
                f"prior_ce={ev['prior_ce']:.4f} oracle_ce={ev['oracle_ce']:.4f} "
                f"trn_route_ce={tr['route_ce']:.4f} trn_acc={tr['best_acc']:.3f} "
                f"trn_cls={tr['cls_ce']:.4f} val_cls_ce={ev['cls_ce']:.4f} "
                f"o_rt={ev['phase_route'][0]:.3f} m_rt={ev['phase_route'][1]:.3f} "
                f"e_rt={ev['phase_route'][2]:.3f} best_acc={ev['best_acc']:.3f} "
                f"| {fixed} | {share} | sacc {sacc} | 0 pos/s"
            )
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if ev["route_ce"] < best_route:
                best_route = ev["route_ce"]
            if ev["cls_ce"] < best_cls:
                best_cls = ev["cls_ce"]
                _save(
                    OUT / "router_best.pt",
                    start_step + step,
                    {"val_route_ce": ev["route_ce"], "val_cls_ce": best_cls, "val": ev},
                )
                print(f"  best val_cls_ce={best_cls:.4f} val_route_ce={ev['route_ce']:.4f} step={step}", flush=True)
        if step % 200 == 0:
            _save(OUT / "router.pt", start_step + step)
    _save(OUT / "router.pt", start_step + steps)
    print(
        f"wrote {OUT / 'router.pt'} best_val_cls_ce={best_cls:.4f} best_val_route_ce={best_route:.4f}",
        flush=True,
    )


def _greedy_move(model, board, device: torch.device):
    from chess_features import board_to_fused_token_ids
    from move_vocab import index_to_move, legal_move_mask

    bi = board_to_fused_token_ids(board)
    bi = {
        "fused_ids": bi["fused_ids"].unsqueeze(0).to(device),
        "turn": bi["turn"].to(device),
        "castling": bi["castling"].to(device),
        "ep_file": bi["ep_file"].to(device),
    }
    out = model(bi)
    logits = out.policy_logits[0].detach().cpu() if hasattr(out, "policy_logits") else out["policy_logits"][0].detach().cpu()
    logits = logits.masked_fill(~legal_move_mask(board), -1e9)
    mv = index_to_move(int(logits.argmax()))
    if mv not in board.legal_moves:
        for alt in board.legal_moves:
            return alt
    return mv


def play_router_probe(args: argparse.Namespace) -> dict:
    """Greedy MoE vs incumbent. Tests real dispatch, not just cached CE."""
    import chess

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.router_ckpt) if getattr(args, "router_ckpt", None) else OUT / "router_best.pt"
    if not ckpt.exists():
        ckpt = OUT / "router.pt"
    if not ckpt.exists():
        raise SystemExit("train the router first")
    router = build_router()
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    router.load_state_dict(blob["model_state_dict"])
    router.to(device).eval()
    experts = []
    for name, _ in EXPERTS:
        experts.append(load_frozen_expert(expert_local_dir(name) / "latest.pt", device))
    moe = FrozenExpertMoE(router, experts).to(device)
    incumbent = experts[0]
    openings = [[], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["e2e4", "c7c5"]]
    ply_cap = 80
    w = d = l = 0
    hist = {n: 0 for n in EXPERT_NAMES}
    games = 0
    for opening in openings:
        for moe_white in (True, False):
            board = chess.Board()
            for uci in opening:
                board.push_uci(uci)
            while not board.is_game_over(claim_draw=True) and board.ply() < ply_cap:
                if board.turn == chess.WHITE:
                    use_moe = moe_white
                else:
                    use_moe = not moe_white
                if use_moe:
                    mv = _greedy_move(moe, board, device)
                    with torch.no_grad():
                        from chess_features import board_to_fused_token_ids

                        bi = board_to_fused_token_ids(board)
                        bi = {
                            "fused_ids": bi["fused_ids"].unsqueeze(0).to(device),
                            "turn": bi["turn"].to(device),
                            "castling": bi["castling"].to(device),
                            "ep_file": bi["ep_file"].to(device),
                        }
                        eid = int(moe.route_ids(bi)[1].item())
                        hist[EXPERT_NAMES[eid]] += 1
                else:
                    mv = _greedy_move(incumbent, board, device)
                board.push(mv)
            games += 1
            if board.is_checkmate():
                winner_white = not board.turn
                moe_won = winner_white == moe_white
                if moe_won:
                    w += 1
                else:
                    l += 1
            else:
                d += 1
            print(f"  game moe_white={moe_white} opening={opening} ply={board.ply()} result={board.result(claim_draw=True)}", flush=True)
    report = {"games": games, "w": w, "d": d, "l": l, "score": (w + 0.5 * d) / max(games, 1), "route_hist": hist, "ckpt": str(ckpt)}
    (OUT / "play.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PLAY", json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--play", action="store_true", help="greedy MoE vs incumbent after train")
    ap.add_argument("--router-ckpt", default="")
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--rows-per-source", type=int, default=200_000)
    ap.add_argument("--val-n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=279)
    ap.add_argument("--microbatch", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=2_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", action="store_true", help="continue from outputs/exp279_moe_router/router.pt")
    ap.add_argument("--objective", default="soft", choices=("soft", "source"))
    ap.add_argument("--pack-rows", type=int, default=0, help="if >0, sample this many boards per specialist source")
    ap.add_argument("--rebuild-pack", action="store_true")
    ap.add_argument(
        "--subset",
        default="all",
        help="all | specialists | opening,middlegame,endgame,puzzles | one source",
    )
    ap.add_argument("--overfit-n", type=int, default=0, help="if >0, train+eval on a fixed train subset")
    ap.add_argument("--tau", type=float, default=0.4, help="soft-target temperature")
    ap.add_argument("--balance", type=float, default=0.05, help="KL(batch usage || batch soft targets)")
    args = ap.parse_args()
    _assert_compact()
    print(f"exp279 MoE router experts={EXPERT_NAMES} router_params={router_param_count():,}", flush=True)
    if args.pull:
        pull_experts(skip_missing=args.skip_missing)
    if args.score:
        score_all(args)
    if args.go:
        train_router(args)
    if args.play:
        play_router_probe(args)
    if not (args.pull or args.score or args.go or args.play):
        print("pass --pull / --score / --go / --play", flush=True)


if __name__ == "__main__":
    main()
