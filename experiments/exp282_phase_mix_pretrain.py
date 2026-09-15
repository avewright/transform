#!/usr/bin/env python3
"""exp282: continue-pretrain the frozen 99M on a 4-way phase mix.

20% opening + 20% middlegame + 20% endgame + 20% puzzles. The leftover
20% is split across those same four sources (25/25/25/25) so the mix
is complete. Does not write the public 99M incumbent.

  MOVE_VOCAB_VERSION=compact python experiments/exp282_phase_mix_pretrain.py --pack
  MOVE_VOCAB_VERSION=compact python experiments/exp282_phase_mix_pretrain.py --go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chess_squares64 import DEFAULT_100M_SQUARES64_CONFIG
from exp273_puzzle_finetune import (
    SOURCE_PUZZLE,
    _pack_record_batch,
    _stack_rows,
    _worker_init,
    cat_tables,
    drop_eval_overlap,
    pull_99m,
)
from exp275_endgame_finetune import split_of
from move_vocab import VOCAB_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "exp282_phase_mix_pretrain"
TEACHER_REPO = "avewright/chess-transformer-100m-squares64"
INCUMBENT_REPO = "avewright/chess-transformer-100m-squares64"
EXPECTED_99M_PARAMS = 98_971_224
SPLIT_SEED = 282
TRAIN_PCT = 80
VAL_N = 2_048
DEFAULT_TRAIN_N = 15_000_000  # ~1 epoch at bs=297 / 50k steps

SOURCE_OPENING = 7
SOURCE_MID = 8
SOURCE_END = 6

NAMED_FRACS = {
    "opening": 0.20,
    "middlegame": 0.20,
    "endgame": 0.20,
    "puzzles": 0.20,
}
SOURCES = {
    "opening": {
        "repo": "avewright/lichess-opening-bestline",
        "source": SOURCE_OPENING,
        "phase": 0,
        "kind": "bestline",
    },
    "middlegame": {
        "repo": "avewright/lichess-middlegame-bestline",
        "source": SOURCE_MID,
        "phase": 1,
        "kind": "bestline",
    },
    "endgame": {
        "repo": "avewright/lichess-endgame-bestline",
        "source": SOURCE_END,
        "phase": 2,
        "kind": "bestline",
    },
    "puzzles": {
        "repo": "Lichess/chess-puzzles",
        "source": SOURCE_PUZZLE,
        "phase": None,
        "kind": "puzzles",
    },
}
CORE = (
    "board_array", "turn", "castling", "ep_square",
    "move_idx", "cp", "mate", "soft_indices", "soft_probs",
)
OPTIONAL = ("label_depth", "phase", "source", "n_pieces", "wdl", "dtz", "ply")
PACK_KEYS = CORE + ("source", "value_valid", "label_depth", "phase", "split")


def _assert_compact() -> None:
    if VOCAB_SIZE != 1968:
        raise SystemExit("Expected compact vocab 1968. Export MOVE_VOCAB_VERSION=compact.")


def refuse_incumbent(repo: str) -> None:
    if repo.strip() == INCUMBENT_REPO:
        raise SystemExit(f"refusing to write the public incumbent {repo}")


def resolve_fracs(named: dict[str, float] | None = None) -> dict[str, float]:
    """20% each of four sources leaves 20%; spread that leftover evenly."""
    named = dict(named or NAMED_FRACS)
    total = float(sum(named.values()))
    if total <= 0:
        raise ValueError("fracs must sum > 0")
    leftover = max(0.0, 1.0 - total)
    extra = leftover / len(named)
    out = {k: float(v) + extra for k, v in named.items()}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def write_init_ckpt(src: Path, dest: Path) -> Path:
    from autoresearch_8gb.pipeline import load_model_state

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = load_model_state(ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt})
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "config": ckpt.get("config") if isinstance(ckpt, dict) else None,
            "eval_only": True,
            "steps": 0,
            "note": "99M weights-only warm start for exp282 phase-mix continue-pretrain",
            "source": str(src),
        },
        dest,
    )
    print(f"init ckpt {dest} keys={len(state)}", flush=True)
    return dest


def n_rows(table: dict) -> int:
    return int(table["board_array"].shape[0])


def slice_table(table: dict, idx: torch.Tensor) -> dict:
    n = n_rows(table)
    return {
        k: (v[idx] if torch.is_tensor(v) and int(v.shape[0]) == n else v)
        for k, v in table.items()
    }


def tag_source(table: dict, source_id: int, phase: int | None) -> dict:
    table = dict(table)
    n = n_rows(table)
    table["source"] = torch.full((n,), int(source_id), dtype=torch.int8)
    table["value_valid"] = torch.zeros(n, dtype=torch.int8) if source_id == SOURCE_PUZZLE else torch.ones(n, dtype=torch.int8)
    if "phase" not in table or table["phase"] is None:
        if phase is None:
            table["phase"] = torch.ones(n, dtype=torch.int8)
        else:
            table["phase"] = torch.full((n,), int(phase), dtype=torch.int8)
    if "label_depth" not in table:
        table["label_depth"] = torch.zeros(n, dtype=torch.int16)
    table["ep_square"] = torch.where(table["ep_square"] <= 0, -1, table["ep_square"]).to(torch.int8)
    return table


def _arrow_to_tensor(col, dtype: torch.dtype) -> torch.Tensor:
    import pyarrow as pa

    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    t = arr.type
    if pa.types.is_fixed_size_list(t):
        vals = np.asarray(arr.values.to_numpy())
        return torch.from_numpy(np.ascontiguousarray(vals.reshape(len(arr), t.list_size))).to(dtype)
    if pa.types.is_list(t):
        stacked = np.stack([np.asarray(x.as_py()) for x in arr])
        return torch.from_numpy(np.ascontiguousarray(stacked)).to(dtype)
    np_arr = np.asarray(arr.to_numpy())
    ten = torch.from_numpy(np.ascontiguousarray(np_arr)).to(dtype)
    if ten.ndim == 0:
        return ten.view(1)
    if ten.ndim > 1 and ten.shape[-1] == 1:
        return ten.reshape(-1)
    if ten.ndim > 1:
        return ten
    return ten.view(-1)


def parquet_to_cache(path: Path) -> dict:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    out: dict[str, torch.Tensor] = {}
    dtypes = {
        "board_array": torch.int8,
        "turn": torch.int8,
        "castling": torch.int8,
        "ep_square": torch.int8,
        "move_idx": torch.int64,
        "cp": torch.int32,
        "mate": torch.int32,
        "soft_indices": torch.int64,
        "soft_probs": torch.float32,
        "label_depth": torch.int16,
        "phase": torch.int8,
        "source": torch.int8,
        "n_pieces": torch.int8,
        "wdl": torch.int8,
        "dtz": torch.int16,
        "ply": torch.int16,
    }
    for k in CORE:
        if k not in table.column_names:
            raise SystemExit(f"{path}: missing {k}; have {table.column_names}")
        out[k] = _arrow_to_tensor(table[k], dtypes[k])
    for k in OPTIONAL:
        if k not in table.column_names:
            continue
        out[k] = _arrow_to_tensor(table[k], dtypes[k])
    return out


def list_parquet(repo: str) -> list[str]:
    from huggingface_hub import HfApi

    files = [
        s.rfilename
        for s in HfApi().dataset_info(repo).siblings
        if s.rfilename.endswith(".parquet")
    ]
    files.sort()
    return files


def load_bestline(name: str, spec: dict, need: int, rng: np.random.Generator) -> dict:
    from huggingface_hub import hf_hub_download

    files = list_parquet(spec["repo"])
    order = rng.permutation(len(files))
    chunks: list[dict] = []
    got = 0
    print(f"load {name} repo={spec['repo']} files={len(files)} need={need:,}", flush=True)
    for i in order:
        if got >= need:
            break
        fn = files[int(i)]
        local = Path(hf_hub_download(spec["repo"], fn, repo_type="dataset"))
        part = tag_source(parquet_to_cache(local), spec["source"], spec["phase"])
        chunks.append(part)
        got += n_rows(part)
        print(f"  {name} {fn} +{n_rows(part):,} got={got:,}", flush=True)
    if not chunks:
        raise SystemExit(f"{name}: no parquet rows from {spec['repo']}")
    return cat_tables(chunks) if len(chunks) > 1 else chunks[0]


def load_puzzles(need: int, *, workers: int = 16) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    from huggingface_hub import hf_hub_download

    from exp273_puzzle_finetune import PUZZLE_REPO, PUZZLE_REVISION

    import pyarrow.parquet as pq

    files = list_parquet(PUZZLE_REPO)
    files.sort()
    print(f"pack puzzles need={need:,} files={len(files)}", flush=True)
    parts: list[dict] = []
    scanned = skipped = 0
    pending: list[dict] = []
    chunk = 1024
    n_workers = max(1, int(workers))
    pool = (
        ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init)
        if n_workers > 1
        else None
    )

    def flush(batch: list[dict]) -> None:
        nonlocal skipped, scanned
        if not batch:
            return
        chunks = [batch[i:i + chunk] for i in range(0, len(batch), chunk)]
        payloads = [(c, 0, 4000, SPLIT_SEED, 100) for c in chunks]
        if pool is None:
            results = [_pack_record_batch(p) for p in payloads]
        else:
            results = list(pool.map(_pack_record_batch, payloads, chunksize=1))
        rows: list[tuple] = []
        for t, e, sk, n in results:
            rows.extend(t)
            rows.extend(e)
            skipped += sk
            scanned += n
        stacked = _stack_rows(rows, 0)
        if stacked is not None:
            parts.append(tag_source(stacked, SOURCE_PUZZLE, None))
        print(
            f"  puzzles scanned={scanned:,} rows={sum(n_rows(p) for p in parts):,} skipped={skipped:,}",
            flush=True,
        )

    try:
        for fname in files:
            if sum(n_rows(p) for p in parts) >= need:
                break
            local = Path(hf_hub_download(PUZZLE_REPO, fname, repo_type="dataset", revision=PUZZLE_REVISION))
            pf = pq.ParquetFile(local)
            cols = ["PuzzleId", "FEN", "Moves", "Rating"]
            for batch in pf.iter_batches(batch_size=8192, columns=cols):
                if sum(n_rows(p) for p in parts) >= need:
                    break
                d = batch.to_pydict()
                for i in range(len(d["FEN"])):
                    if sum(n_rows(p) for p in parts) + len(pending) >= need * 2 and parts:
                        break
                    pending.append({
                        "PuzzleId": d["PuzzleId"][i],
                        "FEN": d["FEN"][i],
                        "Moves": d["Moves"][i],
                        "Rating": d["Rating"][i],
                    })
                    if len(pending) >= chunk * n_workers:
                        flush(pending)
                        pending = []
                        if sum(n_rows(p) for p in parts) >= need:
                            break
            print(f"  file {fname} rows={sum(n_rows(p) for p in parts):,}", flush=True)
        flush(pending)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    if not parts:
        raise SystemExit("puzzle pack produced no rows")
    return cat_tables(parts) if len(parts) > 1 else parts[0]


def hash_split(table: dict, *, seed: int, train_pct: int) -> tuple[dict, dict]:
    from autoresearch_8gb.pipeline import position_hashes

    hs = position_hashes(table)
    mask = np.fromiter(
        (split_of(str(int(h)), seed=seed, train_pct=train_pct) == 0 for h in hs.tolist()),
        dtype=np.bool_,
        count=int(hs.shape[0]),
    )
    train_idx = torch.from_numpy(np.flatnonzero(mask))
    eval_idx = torch.from_numpy(np.flatnonzero(~mask))
    return slice_table(table, train_idx), slice_table(table, eval_idx)


def take_n(table: dict, n: int, rng: np.random.Generator) -> dict:
    n0 = n_rows(table)
    n = min(int(n), n0)
    if n <= 0:
        raise SystemExit("empty take_n")
    if n == n0:
        return table
    idx = torch.from_numpy(rng.choice(n0, size=n, replace=False))
    return slice_table(table, idx)


def assemble_mix(
    parts: dict[str, dict],
    *,
    train_n: int,
    eval_n: int,
    seed: int = SPLIT_SEED,
    named_fracs: dict[str, float] | None = None,
) -> tuple[dict, dict, dict]:
    fracs = resolve_fracs(named_fracs)
    rng = np.random.default_rng(seed)
    trains: list[dict] = []
    evals: list[dict] = []
    report_sources: dict[str, dict] = {}
    for name, table in parts.items():
        if name not in fracs:
            raise SystemExit(f"unexpected source {name}")
        tagged = tag_source(table, SOURCES[name]["source"], SOURCES[name]["phase"])
        tr, ev = hash_split(tagged, seed=seed, train_pct=TRAIN_PCT)
        want_tr = max(1, int(round(train_n * fracs[name])))
        want_ev = max(1, int(eval_n))
        tr = take_n(tr, want_tr, rng)
        ev = take_n(ev, want_ev, rng)
        trains.append(tr)
        evals.append(ev)
        report_sources[name] = {
            "frac": fracs[name],
            "named_frac": float((named_fracs or NAMED_FRACS)[name]),
            "train_n": n_rows(tr),
            "eval_n": n_rows(ev),
            "source_id": int(SOURCES[name]["source"]),
            "repo": SOURCES[name]["repo"],
        }
    train = cat_tables(trains)
    ev = cat_tables(evals)
    train, n_overlap = drop_eval_overlap(train, ev)
    perm = torch.from_numpy(rng.permutation(n_rows(train)))
    train = slice_table(train, perm)
    report = {
        "fracs": fracs,
        "named_fracs": dict(named_fracs or NAMED_FRACS),
        "leftover_rule": "spread leftover mass evenly across the four named sources",
        "train_n": n_rows(train),
        "eval_n": n_rows(ev),
        "overlap_dropped": int(n_overlap),
        "sources": report_sources,
        "split_seed": seed,
        "train_pct": TRAIN_PCT,
    }
    return train, ev, report


def _try_hf_token() -> None:
    try:
        from upload_exp201_hf import load_hf_token

        load_hf_token()
    except SystemExit:
        pass


def pack_mix(out: Path, *, train_n: int, eval_n: int, workers: int, seed: int = SPLIT_SEED) -> dict:
    _assert_compact()
    _try_hf_token()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fracs = resolve_fracs()
    parts: dict[str, dict] = {}
    for name, spec in SOURCES.items():
        need = int(round(train_n * fracs[name])) + int(eval_n) + 2_048
        need = int(need / (TRAIN_PCT / 100.0)) + 4_096
        raw_path = out / f"{name}_raw.pt"
        if raw_path.exists():
            loaded = torch.load(raw_path, map_location="cpu", weights_only=False)
            if n_rows(loaded) >= need:
                parts[name] = loaded
                print(f"reuse {name} {raw_path} n={n_rows(loaded):,}", flush=True)
                continue
        if spec["kind"] == "puzzles":
            parts[name] = load_puzzles(need, workers=workers)
        else:
            parts[name] = load_bestline(name, spec, need, rng)
        torch.save(parts[name], raw_path)
        print(f"source {name} loaded={n_rows(parts[name]):,} wrote {raw_path}", flush=True)
    train, ev, report = assemble_mix(parts, train_n=train_n, eval_n=eval_n, seed=seed)
    train_path = out / "mix_train.pt"
    eval_path = out / "mix_eval.pt"
    torch.save(train, train_path)
    torch.save(ev, eval_path)
    report["train_path"] = str(train_path)
    report["eval_path"] = str(eval_path)
    (out / "mix_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PACK", json.dumps(report, indent=2), flush=True)
    return report


def trial_config() -> dict:
    model = DEFAULT_100M_SQUARES64_CONFIG.to_dict()
    model["gradient_checkpointing"] = False
    return {
        "id": "exp282_phase_mix_pretrain",
        "arch": "squares64",
        "desc": "99M continue-pretrain on 25/25/25/25 opening/mid/end/puzzle mix.",
        "init": {"repo": TEACHER_REPO, "params": EXPECTED_99M_PARAMS},
        "data": {
            "named_fracs": NAMED_FRACS,
            "fracs": resolve_fracs(),
            "split_seed": SPLIT_SEED,
        },
        "model": model,
        "train": {
            "batch_size": 528,
            "min_batch_size": 32,
            "max_batch_size": 528,
            "accum_steps": 1,
            "soft_frac": 1.0,
            "soft_alpha": 0.0,
            "soft_temp": 0.0,
            "soft_temp_weight": 0.0,
            "deep_mix_frac": 0.0,
            "bonus_mix_frac": 0.0,
            "quality_mix_frac": 0.0,
            "puzzle_mix_frac": 0.0,
            "use_swa": False,
            "hflip_p": 0.5,
            "value_weight": 0.15,
            "min_depth": 12,
            "optimizer": "polar_normuon",
            "compile_polar": True,
            "force_lr": True,
            "muon_lr": 0.002,
            "adam_lr": 3e-5,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "warmup": 80,
            "min_lr_frac": 0.1,
            "torch_compile": True,
            "compile_mode": "default",
            "grad_checkpoint": False,
            "fill_vram": False,
            "max_vram_gb": 22.0,
            "save_every_steps": 250,
            "keep_step_every": 500,
            "keep_last_ckpts": 4,
            "val_every_steps": 250,
            "val_eval_n": VAL_N,
            "elo_every_steps": 0,
        },
    }


def pick_resume(out: Path, init_path: Path, resume_arg: str | None) -> Path:
    if resume_arg:
        p = Path(resume_arg)
        if not p.exists():
            raise SystemExit(f"resume ckpt missing: {p}")
        return p
    latest = out / "latest.pt"
    if latest.exists():
        return latest
    return init_path


def train(args: argparse.Namespace) -> dict:
    from autoresearch_8gb.train_trial import train_trial

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    refuse_incumbent(str(args.hf_repo or ""))
    ckpt_src = Path(args.checkpoint) if args.checkpoint else ROOT / "outputs" / "hf_models" / "99m" / "latest.pt"
    if not ckpt_src.exists():
        ckpt_src = pull_99m(ckpt_src.parent)
    init_path = out / "init.pt"
    if not init_path.exists() or args.refresh_init:
        write_init_ckpt(ckpt_src, init_path)

    train_cache = out / "mix_train.pt"
    eval_cache = out / "mix_eval.pt"
    if not train_cache.exists() or not eval_cache.exists() or args.repack:
        pack_mix(out, train_n=args.train_n, eval_n=args.eval_n, workers=args.workers)
    if not train_cache.exists() or not eval_cache.exists():
        raise SystemExit(f"missing mix caches {train_cache} {eval_cache}")

    trial = trial_config()
    cfg = trial["train"]
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
        cfg["max_batch_size"] = int(args.batch_size)
        cfg["fill_vram"] = False
    cfg["external_eval"] = {"mix": str(eval_cache.resolve())}

    if args.one_epoch:
        n_train = int(json.loads((out / "mix_report.json").read_text())["train_n"])
        bs = int(args.batch_size or cfg["batch_size"])
        args.max_steps = max(1, (n_train + bs - 1) // bs)
        print(f"one_epoch n={n_train:,} bs={bs} max_steps={args.max_steps}", flush=True)

    resume_ckpt = pick_resume(out, init_path, args.resume)
    print(f"resume_ckpt={resume_ckpt}", flush=True)
    result = train_trial(
        trial,
        out,
        soft_cache=train_cache,
        deep_cache=None,
        max_steps=args.max_steps,
        max_minutes=args.train_minutes,
        smoke=False,
        resume_ckpt=resume_ckpt,
    )
    (out / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--repack", action="store_true")
    ap.add_argument("--refresh-init", action="store_true")
    ap.add_argument("--output-dir", default=str(OUT_DIR))
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--train-n", type=int, default=DEFAULT_TRAIN_N)
    ap.add_argument("--eval-n", type=int, default=VAL_N)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=50_000)
    ap.add_argument("--one-epoch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-minutes", type=float, default=10_080)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--hf-repo", default="")
    args = ap.parse_args()
    _assert_compact()
    if args.pack and not args.go:
        pack_mix(Path(args.output_dir), train_n=args.train_n, eval_n=args.eval_n, workers=args.workers)
        return
    if args.go:
        train(args)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
