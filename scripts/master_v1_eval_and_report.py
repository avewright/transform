#!/usr/bin/env python3
"""128-game SF2050/2200 screen + retain/replace recommendation. Does not promote."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_searchless_gauntlet import _summarize, paired_opening_bootstrap


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_elo(ckpt: Path, prefix: str, out_json: Path) -> Path:
    if out_json.exists():
        print(f"reuse {out_json}", flush=True)
        return out_json
    env = os.environ.copy()
    env["MOVE_VOCAB_VERSION"] = "compact"
    env["STOCKFISH_PATH"] = env.get("STOCKFISH_PATH", "/root/.local/bin/stockfish-19")
    cmd = [
        str(ROOT / ".venv/bin/python"), "-u", "-m", "harness.elo",
        "--ckpt", str(ckpt),
        "--out-prefix", prefix,
        "--mode", "policy",
        "--no-book", "--no-syzygy",
        "--nodes", "8000",
        "--games-per-opening-per-color", "4",
        "--no-stop-after-bracket",
        "--elos", "2050", "2200",
    ]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    produced = ROOT / "outputs" / f"elo_eval_{prefix}.json"
    if produced.exists() and produced.resolve() != out_json.resolve():
        out_json.write_bytes(produced.read_bytes())
    elif produced.exists():
        return produced
    return out_json if out_json.exists() else produced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--incumbent", required=True)
    args = ap.parse_args()
    base = Path(args.base)
    inc = Path(args.incumbent)
    ev = base / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    train = base / "train"
    candidate = train / "eval_swa.pt"
    kind = "eval_swa"
    if not candidate.exists():
        candidate = train / "latest.pt"
        kind = "live"
    if not candidate.exists():
        print("FATAL no candidate checkpoint", flush=True)
        return 1

    inc_sha = sha256(inc)
    cand_sha = sha256(candidate)
    cached = ROOT / "outputs/sf19_ft/overnight_20260908/evaluation/elo_reference_2050_2200.json"
    reuse = False
    if cached.exists():
        raw = json.loads(cached.read_text())
        proto = raw.get("protocol") or raw.get("config") or {}
        reuse = (
            proto.get("mode") == "policy"
            and proto.get("book") is False
            and proto.get("syzygy") is False
            and int(proto.get("nodes") or 0) == 8000
            and proto.get("elos") == [2050, 2200]
            and int(proto.get("games_per_opening_per_color") or 0) == 4
            and len(raw.get("games") or []) == 128
        )
        # Cached path is the same file we hashed; protocol matches but hash is not stored.
        print(f"cached incumbent protocol_ok={reuse} (hash not embedded; will still re-run incumbent if flag false)", flush=True)

    inc_json = ev / "elo_incumbent_n8000_r4.json"
    if reuse:
        inc_json.write_text(cached.read_text())
        print("reused overnight reference for incumbent (same protocol; same file path family)", flush=True)
    else:
        run_elo(inc, "master_v1_incumbent_n8000_r4", inc_json)

    cand_json = ev / "elo_candidate_n8000_r4.json"
    run_elo(candidate, f"master_v1_candidate_{kind}_n8000_r4", cand_json)

    inc_raw = json.loads(inc_json.read_text()) if inc_json.exists() else json.loads(
        (ROOT / "outputs/elo_eval_master_v1_incumbent_n8000_r4.json").read_text()
    )
    cand_path = cand_json if cand_json.exists() else ROOT / f"outputs/elo_eval_master_v1_candidate_{kind}_n8000_r4.json"
    cand_raw = json.loads(Path(cand_path).read_text())

    inc_sum = _summarize(inc_raw)
    cand_sum = _summarize(cand_raw)
    diff = paired_opening_bootstrap(cand_raw.get("games") or [], inc_raw.get("games") or [])
    replace = bool(diff and diff.get("a_better"))
    rec = "replace_incumbent" if replace else "retain_incumbent"
    reason = (
        "candidate paired-opening CI entirely above 0 vs incumbent"
        if replace else
        "no evidence of improvement; retain overnight SWA"
    )
    auth = json.loads((base / "export/AUTHORIZED.json").read_text()) if (base / "export/AUTHORIZED.json").exists() else {}
    summary = json.loads((train / "train_summary.json").read_text()) if (train / "train_summary.json").exists() else {}
    report = {
        "recommendation": rec,
        "reason": reason,
        "do_not_overwrite_incumbent": True,
        "do_not_publish": True,
        "starting_checkpoint": str(inc),
        "starting_sha256": inc_sha,
        "candidate": str(candidate),
        "candidate_kind": kind,
        "candidate_sha256": cand_sha,
        "dataset": {
            "repo": "avewright/chess-master-v1",
            "revision": "3cc4feb61171387520718a0c4d4b8bb150987780",
            "recipe": "pilot_45_35_15_5",
            "counts": auth.get("counts"),
            "sampled_target": {"sf19": 0.45, "lichess": 0.35, "puzzles": 0.15, "syzygy": 0.05},
            "deep_mix_frac": 0.05,
            "bonus_mix_frac": 0,
        },
        "train_summary": {
            "steps": summary.get("steps") or summary.get("step"),
            "status": summary.get("status"),
            "raw_keys": list(summary)[:20] if summary else [],
        },
        "incumbent_elo_provenance": {
            "label": "cached_results" if reuse else "freshly_played",
            "replayed_this_arm": not reuse,
            "source": str(cached) if reuse else "harness.elo this arm",
            "engine": (inc_sum.get("protocol") or {}).get("sf_version"),
            "sf_path": (inc_sum.get("protocol") or {}).get("sf_path"),
            "nodes": (inc_sum.get("protocol") or {}).get("nodes"),
            "openings": (inc_sum.get("protocol") or {}).get("openings"),
            "inference": "policy greedy argmax temperature=0, compact vocab, no book, no Syzygy",
        },
        "incumbent_elo": inc_sum,
        "candidate_elo": cand_sum,
        "paired_opening_candidate_minus_incumbent": diff,
        "source_holdouts_note": (
            "sf19/lichess/puzzles/syzygy eval sets are membership holdouts from "
            "organized_chess_v1. They are not proven unseen by older checkpoints."
        ),
        "overnight_blocked_hashes": "not recovered as a list on this pod; membership export already excludes organized eval",
    }
    dest = ev / "recommendation.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("WROTE", dest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
