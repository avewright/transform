"""Propose next autoresearch trials by mutating the Elo champion.

Appends new trial stubs into search_space.json (inherits + overrides) and
returns their ids. Deterministic given journal + generation index.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "autoresearch_8gb"))
SPACE_PATH = ROOT / "scripts" / "autoresearch_8gb" / "search_space.json"
OUT_ROOT = ROOT / "outputs" / "autoresearch_8gb"

# Single-flag model flips relative to parent
MODEL_FLIPS: list[tuple[str, dict[str, Any]]] = [
    ("tog_gab", {"use_gab": True}),
    ("no_gab", {"use_gab": False}),
    ("tog_qk", {"use_qk_norm": True}),
    ("no_qk", {"use_qk_norm": False}),
    ("tog_zero", {"zero_init_out_proj": True}),
    ("no_zero", {"zero_init_out_proj": False}),
    ("tog_meta", {"use_meta_attention": True, "use_rel_bias": False}),
    ("no_meta", {"use_meta_attention": False}),
    ("tog_relbias", {"use_rel_bias": True}),
    ("no_relbias", {"use_rel_bias": False}),
    ("tog_shaw", {"use_shaw_on_pos": True}),
    ("no_shaw", {"use_shaw_on_pos": False}),
    ("tog_swiglu", {"use_swiglu": True}),
    ("gelu", {"use_swiglu": False}),
    ("drop0", {"dropout": 0.0}),
    ("drop10", {"dropout": 0.1}),
]

TRAIN_MUTATIONS: list[tuple[str, dict[str, Any]]] = [
    ("softT4", {"soft_temp": 4.0, "soft_temp_weight": 0.5}),
    ("softT4h", {"soft_temp": 4.0, "soft_temp_weight": 0.8, "soft_alpha": 0.7}),
    ("swa", {"use_swa": True, "swa_start_frac": 0.75}),
    ("muon_hot", {"muon_lr": 0.04}),
    ("muon_cool", {"muon_lr": 0.01}),
    ("val_heavy", {"value_weight": 1.0}),
    ("val_light", {"value_weight": 0.05}),
    ("ls05", {"label_smoothing": 0.05}),
    ("warm800", {"warmup": 800}),
    ("soft_a60", {"soft_alpha": 0.6}),
    ("soft_a30", {"soft_alpha": 0.3}),
    ("polar", {"optimizer": "polar_normuon"}),
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_done(journal_path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not journal_path.exists():
        return latest
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "done" and row.get("id") and row.get("elo_estimate") is not None:
            latest[row["id"]] = row
    return latest


def _champion_id(out_root: Path, latest: dict[str, dict]) -> str | None:
    champ_path = out_root / "champion.json"
    if champ_path.exists():
        c = _load_json(champ_path)
        if c.get("id"):
            return c["id"]
    if not latest:
        return None
    return max(latest.values(), key=lambda r: float(r.get("elo_estimate") or -1e9))["id"]


def _parent_exists(space: dict, parent_id: str) -> bool:
    return any(t.get("id") == parent_id for t in space["trials"])


def propose(
    *,
    space_path: Path = SPACE_PATH,
    out_root: Path = OUT_ROOT,
    n: int = 6,
    generation: int | None = None,
) -> list[str]:
    space = _load_json(space_path)
    existing = {t["id"] for t in space["trials"]}
    latest = _latest_done(out_root / "trials.jsonl")
    parent = _champion_id(out_root, latest) or space.get("baseline_id", "baseline_deep_small")
    if not _parent_exists(space, parent):
        parent = space.get("baseline_id", "baseline_deep_small")

    gen = generation
    if gen is None:
        # Count prior mut_* trials
        gen = sum(1 for tid in existing if tid.startswith("mut_")) // max(n, 1)

    # Rank mutations: prefer flips that differ from parent's known config
    parent_cfg = None
    for t in space["trials"]:
        if t["id"] == parent:
            parent_cfg = t
            break
    parent_model: dict[str, Any] = {}
    parent_train: dict[str, Any] = {}
    if parent_cfg:
        try:
            from train_trial import resolve_trial_config  # type: ignore
            resolved = resolve_trial_config(parent_cfg, space)
            parent_model = dict(resolved["model"])
            parent_train = dict(resolved["train"])
        except Exception:
            parent_model = dict(parent_cfg.get("model") or {})
            parent_train = dict(parent_cfg.get("train") or {})
            parent_model.update(parent_cfg.get("model_overrides") or {})
            parent_train.update(parent_cfg.get("train_overrides") or {})

    candidates: list[tuple[str, str, dict, dict]] = []
    for tag, ov in MODEL_FLIPS:
        # skip no-ops
        noop = all(parent_model.get(k) == v for k, v in ov.items()) if parent_model else False
        if noop:
            continue
        tid = f"mut_g{gen}_{tag}_from_{parent}"[:80]
        candidates.append((tid, f"mut gen{gen} {tag} from {parent}", ov, {}))
    for tag, ov in TRAIN_MUTATIONS:
        noop = all(parent_train.get(k) == v for k, v in ov.items()) if parent_train else False
        if noop:
            continue
        tid = f"mut_g{gen}_{tag}_from_{parent}"[:80]
        candidates.append((tid, f"mut gen{gen} {tag} from {parent}", {}, ov))

    # Also combo: GAB+softT4+SWA from parent if not already stacked
    combo_tid = f"mut_g{gen}_stack_from_{parent}"[:80]
    candidates.insert(0, (
        combo_tid,
        f"mut gen{gen} stack GAB+QK+softT4+SWA from {parent}",
        {"use_gab": True, "use_qk_norm": True},
        {"soft_temp": 4.0, "soft_temp_weight": 0.5, "use_swa": True, "swa_start_frac": 0.75, "soft_frac": 1.0},
    ))

    added: list[str] = []
    for tid, desc, mov, tov in candidates:
        if tid in existing:
            continue
        trial = {
            "id": tid,
            "desc": desc,
            "inherits": parent,
            "model_overrides": mov,
            "train_overrides": {**{"soft_frac": 1.0}, **tov},
        }
        space["trials"].append(trial)
        existing.add(tid)
        added.append(tid)
        if len(added) >= n:
            break

    if added:
        space_path.write_text(json.dumps(space, indent=2) + "\n", encoding="utf-8")
        queue_path = out_root / "dynamic_queue.json"
        prev = _load_json(queue_path) if queue_path.exists() else {"pending": []}
        pending = list(prev.get("pending") or [])
        for tid in added:
            if tid not in pending:
                pending.append(tid)
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(
            json.dumps({
                "updated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "parent": parent,
                "generation": gen,
                "pending": pending,
                "added": added,
            }, indent=2),
            encoding="utf-8",
        )
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--generation", type=int, default=None)
    ap.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    args = ap.parse_args()
    added = propose(out_root=Path(args.out_root), n=args.n, generation=args.generation)
    print(json.dumps({"added": added, "n": len(added)}))


if __name__ == "__main__":
    main()
