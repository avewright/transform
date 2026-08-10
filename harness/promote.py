#!/usr/bin/env python3
"""Elo-gated champion promotion (pure-policy protocol only)."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.common import CHAMPION_DIR, DEFAULT_SEED_CKPT, ROOT, link_or_copy, load_protocol


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_ok(data: dict[str, Any], *, require_pure: bool = True) -> tuple[bool, str]:
    proto = data.get("protocol") or data.get("config") or {}
    mode = data.get("mode") or proto.get("mode") or "policy"
    if mode != "policy":
        return False, f"mode={mode!r} (need policy)"
    book = proto.get("book", True)
    syzygy = proto.get("syzygy", True)
    # also check config keys from harness
    if "book" in (data.get("config") or {}):
        book = data["config"]["book"]
    if "syzygy" in (data.get("config") or {}):
        syzygy = data["config"]["syzygy"]
    if require_pure and book:
        return False, "book=true (pure policy required)"
    if require_pure and syzygy:
        return False, "syzygy=true (pure policy required)"
    return True, "ok"


def elo_of(data: dict[str, Any]) -> float | None:
    est = data.get("estimate") or {}
    elo = est.get("estimated_elo")
    if elo is None:
        elo = data.get("estimated_elo")
    if elo is None:
        return None
    try:
        return float(elo)
    except (TypeError, ValueError):
        return None


def secondary_of(data: dict[str, Any]) -> float:
    """Tie-break: total games then mean score at highest level."""
    summaries = data.get("summaries") or []
    games = sum(s.get("games", 0) for s in summaries)
    if not summaries:
        return float(games)
    last = max(summaries, key=lambda s: s.get("sf_elo", 0))
    return games + float(last.get("score", 0.0))


def load_champion() -> dict[str, Any] | None:
    path = CHAMPION_DIR / "CHAMPION.json"
    if not path.exists():
        return None
    return load_json(path)


def write_champion(
    *,
    ckpt: Path,
    elo_json: Path,
    elo: float,
    meta: dict[str, Any] | None = None,
) -> Path:
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHAMPION_DIR / "champion.pt"
    link_or_copy(ckpt, dest)
    # also keep a concrete copy name for cloud sync friendliness
    copy_path = CHAMPION_DIR / "champion_copy.pt"
    if not copy_path.exists() or copy_path.resolve() != ckpt.resolve():
        try:
            if copy_path.exists() or copy_path.is_symlink():
                copy_path.unlink()
            shutil.copy2(ckpt, copy_path)
        except Exception:
            pass

    data = load_json(elo_json)
    payload = {
        "ckpt": str(dest),
        "source_ckpt": str(ckpt.resolve()),
        "elo": elo,
        "elo_json": str(elo_json.resolve()),
        "protocol": data.get("protocol"),
        "estimate": data.get("estimate"),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
    }
    out = CHAMPION_DIR / "CHAMPION.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def should_promote(
    candidate_elo: float,
    champ_elo: float | None,
    *,
    noise: int,
    candidate_secondary: float = 0.0,
    champ_secondary: float = 0.0,
) -> tuple[bool, str]:
    if champ_elo is None:
        return True, "no existing champion"
    delta = candidate_elo - champ_elo
    if delta >= noise:
        return True, f"+{delta:.0f} Elo (>= {noise} band)"
    if abs(delta) < noise and candidate_secondary > champ_secondary:
        return True, f"within ±{noise} but better secondary ({candidate_secondary:.1f}>{champ_secondary:.1f})"
    return False, f"delta={delta:.0f} (need +{noise} or secondary win within band)"


def promote_from_elo_json(
    elo_json: Path,
    *,
    ckpt: Path | None = None,
    noise: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    protocol = load_protocol()
    noise = int(noise if noise is not None else protocol.get("noise_band", 100))
    data = load_json(elo_json)
    ok, reason = protocol_ok(data, require_pure=True)
    if not ok and not force:
        raise SystemExit(f"refuse promote: {reason}")

    elo = elo_of(data)
    if elo is None:
        raise SystemExit("candidate JSON missing estimated_elo")

    ckpt = Path(ckpt or data["checkpoint"]).resolve()
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    champ = load_champion()
    champ_elo = float(champ["elo"]) if champ and champ.get("elo") is not None else None
    cand_sec = secondary_of(data)
    champ_sec = 0.0
    if champ and champ.get("elo_json"):
        try:
            champ_sec = secondary_of(load_json(Path(champ["elo_json"])))
        except Exception:
            champ_sec = 0.0

    yes, why = should_promote(elo, champ_elo, noise=noise, candidate_secondary=cand_sec, champ_secondary=champ_sec)
    result = {
        "promote": yes or force,
        "reason": "forced" if force and not yes else why,
        "candidate_elo": elo,
        "champion_elo": champ_elo,
        "noise": noise,
        "ckpt": str(ckpt),
        "elo_json": str(elo_json),
    }
    if dry_run:
        return result
    if result["promote"]:
        path = write_champion(ckpt=ckpt, elo_json=elo_json, elo=elo, meta={"reason": result["reason"]})
        result["champion_json"] = str(path)
    return result


def seed_champion(ckpt: Path, elo_json: Path, *, force: bool = True) -> dict[str, Any]:
    return promote_from_elo_json(elo_json, ckpt=ckpt, force=force)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Promote Elo champion (pure policy only)")
    ap.add_argument("--candidate", required=False, help="Path to elo_eval_*.json")
    ap.add_argument("--ckpt", default=None, help="Override checkpoint path")
    ap.add_argument("--noise", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show", action="store_true", help="Print current champion")
    ap.add_argument(
        "--seed",
        action="store_true",
        help=f"Seed from DEFAULT_SEED_CKPT ({DEFAULT_SEED_CKPT}) using --candidate JSON",
    )
    args = ap.parse_args(argv)

    if args.show:
        champ = load_champion()
        print(json.dumps(champ or {"error": "no champion"}, indent=2))
        return 0 if champ else 1

    if not args.candidate:
        ap.error("--candidate JSON required (or use --show)")

    ckpt = Path(args.ckpt) if args.ckpt else None
    if args.seed:
        ckpt = Path(args.ckpt or DEFAULT_SEED_CKPT)

    result = promote_from_elo_json(
        Path(args.candidate),
        ckpt=ckpt,
        noise=args.noise,
        force=args.force or args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("promote") or args.dry_run else 2


if __name__ == "__main__":
    raise SystemExit(main())
