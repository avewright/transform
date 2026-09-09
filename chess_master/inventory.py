"""Pin source revisions, checksums, schemas, licenses, and unknown metadata."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chess_master.io_util import ROOT, fingerprint, json_write, load_hf_token
from chess_master.schema import DATASET_VERSION, UNKNOWN

HF_SOURCES = {
    "sf19": {
        "repo": "avewright/chess-soft-sf19",
        "kind": "engine_policy",
        "license": "mit",
        "role": "Stockfish 19 MultiPV soft targets",
    },
    "lichess_hf": {
        "repo": "avewright/chess-soft-multipv-lichess",
        "kind": "engine_policy",
        "license": "mit",
        "role": "Deep Lichess MultiPV soft pack",
    },
    "syzygy_hf": {
        "repo": "avewright/chess-soft-syzygy",
        "kind": "tablebase",
        "license": "mit",
        "role": "Syzygy-derived soft pack",
    },
    "puzzles": {
        "repo": "Lichess/chess-puzzles",
        "kind": "puzzle_solution",
        "license": "cc0-1.0",
        "role": "Lichess puzzles",
    },
    "swa_mistakes": {
        "repo": "avewright/chess-soft-100m-swa-mistakes",
        "kind": "model_prediction",
        "license": "mit",
        "role": "Overnight SWA mistake scan + verified corrections",
    },
}

LOCAL_SOURCES = {
    "organized_chess_v1": ROOT / "outputs/organized_chess_v1",
    "organized_chess_v1_corr15": ROOT / "outputs/organized_chess_v1_corr15",
    "lichess_local": ROOT / "outputs/hf_soft/multipv_lichess_soft.pt",
    "syzygy_local": ROOT / "outputs/hf_soft/syzygy_soft.pt",
    "swa_local": ROOT / "outputs/swa_mistakes",
    "mac_correction": ROOT / "outputs/mac_correction_v1",
}

KNOWN_MISSING = {
    "sf19": [
        "NNUE network hash is unknown",
        "halfmove/fullmove/repetition history dropped on FEN-only rows",
        "opening ECO/name unknown",
    ],
    "lichess": [
        "engine version unknown",
        "network unknown",
        "nodes requested/achieved unknown",
        "MultiPV requested vs achieved unknown",
        "tau/transform unknown; stored values are probabilities, not logits",
        "cp/mate perspective unverified; |cp|>=90000 is a mate sentinel",
        "game_id unknown; splits use position hash",
        "value supervision stays masked",
    ],
    "puzzles": [
        "no engine evaluation; do not invent CP/WDL",
        "source FEN is before the opponent setup move",
    ],
    "syzygy": [
        "rule convention (DTZ 50-move) not independently verified on this pack",
        "label_depth=999 is a sentinel, not search depth",
        "exported mate may be a DTZ proxy and must not be used as mate distance",
        "value supervision stays masked until conversion is verified",
        "game identifiers are synthetic",
    ],
    "swa_mistakes": [
        "verified_246k.parquet dropped tag/drop_cp/model_move; those live on scan shards",
        "off-PV model moves have unknown regret, not an assumed penalty",
        "checkpoint hash unknown unless the local overnight SWA file is present",
    ],
}


def _hf_info(repo: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(repo)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    card = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    license_ = UNKNOWN
    if isinstance(card, dict):
        license_ = card.get("license") or UNKNOWN
    elif card is not None and getattr(card, "license", None):
        license_ = card.license
    return {
        "repo": repo,
        "revision": info.sha,
        "n_parquet": len(files),
        "parquet_files_head": files[:12],
        "license": license_,
        "available": True,
    }


def _local_info(path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "available": False}
    info: dict[str, Any] = {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "available": True,
        "is_dir": path.is_dir(),
        "bytes": path.stat().st_size if path.is_file() else None,
    }
    if path.is_file() and hash_file:
        info["sha256"] = fingerprint(path)
    if path.is_dir():
        info["entries"] = sorted(p.name for p in path.iterdir())[:40]
        man = path / "manifest.json"
        if man.exists():
            info["manifest"] = str(man.relative_to(ROOT))
            info["manifest_sha256"] = fingerprint(man)
    return info


def _mix_pins(mix: Path) -> dict[str, Any]:
    import json

    if not (mix / "manifest.json").exists():
        return {"available": False, "path": str(mix)}
    man = json.loads((mix / "manifest.json").read_text())
    frozen = mix / "FROZEN.json"
    return {
        "available": True,
        "path": str(mix.relative_to(ROOT)),
        "status": man.get("status"),
        "seed": man.get("seed"),
        "counts": man.get("actual_counts"),
        "value_supervision": man.get("value_supervision"),
        "limitations": man.get("limitations"),
        "source_revisions": {k: v.get("revision") for k, v in man.get("sources", {}).items()},
        "source_inputs": {k: v.get("inputs") for k, v in man.get("sources", {}).items()},
        "frozen": frozen.exists(),
        "frozen_sha256": fingerprint(frozen) if frozen.exists() else None,
        "blocked_manifest": "blocked_manifest.json" if (mix / "blocked_manifest.json").exists() else None,
        "do_not_overwrite": True,
    }


def _training_manifests() -> list[dict[str, Any]]:
    out = []
    for p in sorted((ROOT / "outputs").rglob("val_manifest*.json")):
        out.append({
            "path": str(p.relative_to(ROOT)),
            "sha256": fingerprint(p),
            "role": "prior_holdout",
            "exposure_certainty": "pool_member_or_holdout_hashes_only",
        })
    for p in sorted((ROOT / "outputs").rglob("blocked_manifest.json")):
        out.append({
            "path": str(p.relative_to(ROOT)),
            "sha256": fingerprint(p),
            "role": "blocked_hashes",
            "note": "Includes eval rows and safe flips. Not an unseen-by-old-checkpoints claim.",
        })
    return out


def build_inventory(out_dir: Path) -> dict[str, Any]:
    load_hf_token()
    hf: dict[str, Any] = {}
    for name, spec in HF_SOURCES.items():
        try:
            info = _hf_info(spec["repo"])
            info.update({k: spec[k] for k in ("kind", "role")})
            if info.get("license") in (None, ""):
                info["license"] = spec.get("license") or UNKNOWN
            hf[name] = info
        except Exception as exc:
            hf[name] = {"repo": spec["repo"], "available": False, "error": f"{type(exc).__name__}: {exc}"}

    local = {
        "organized_chess_v1": _mix_pins(LOCAL_SOURCES["organized_chess_v1"]),
        "organized_chess_v1_corr15": _mix_pins(LOCAL_SOURCES["organized_chess_v1_corr15"]),
        "lichess_local": _local_info(LOCAL_SOURCES["lichess_local"], hash_file=False),
        "syzygy_local": _local_info(LOCAL_SOURCES["syzygy_local"], hash_file=True),
        "swa_local": _local_info(LOCAL_SOURCES["swa_local"]),
        "mac_correction": _local_info(LOCAL_SOURCES["mac_correction"]),
    }
    # Known checksums from the frozen mix — do not re-hash the 1.4GB Lichess cache.
    lich = local["lichess_local"]
    if lich.get("available"):
        lich["sha256_from_organized_v1"] = "9b9387d260bac2b11db598ee01a3e695af00974d0714ad441857da341cd9b490"
        lich["hash_status"] = "recorded_in_mix_manifest"
    if not LOCAL_SOURCES["swa_local"].exists():
        local["swa_local"] = {
            "path": "outputs/swa_mistakes",
            "available": False,
            "note": "Recovered analyses are the HF pack, not a local harvest dir",
        }

    report = {
        "dataset_version": DATASET_VERSION,
        "hf": hf,
        "local": local,
        "training_manifests": _training_manifests(),
        "missing_metadata": KNOWN_MISSING,
        "label_conventions": {
            "sf19_value": "white_absolute_wdl",
            "sf19_policy": "stm_softmax_tau_120_over_complete_multipv",
            "lichess_value": "masked_unverified",
            "lichess_policy": "source_probabilities_unknown_transform",
            "puzzles": "policy_only_after_opponent_setup_move",
            "syzygy_value": "masked_pending_tb_wdl_conversion",
            "syzygy_dtz": "distance_to_zero_not_mate",
            "probabilities": "not_logits",
            "off_pv_regret": "unknown",
        },
        "notes": [
            "Absent metadata is unknown, never zero or verified.",
            "organized_chess_v1 is frozen and must not be overwritten.",
            "Included-in-pool is not the same as actually-sampled during training.",
        ],
    }
    json_write(out_dir / "inventory.json", report)
    return report
