#!/usr/bin/env python3
"""Verify overnight resume pairing and cached Elo provenance. Read-only."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch

from scripts.autoresearch_8gb.pipeline import classify_checkpoint, load_model_state
from scripts.autoresearch_8gb.train_trial import build_polar_normuon_optimizer
from chess_squares64 import build_squares64, count_parameters

INC_DIR = ROOT / "outputs/sf19_ft/overnight_20260908"
SWA = INC_DIR / "eval_swa.pt"
RESUME = INC_DIR / "training_resume.pt"
REF = INC_DIR / "evaluation/elo_reference_2050_2200.json"
FROZEN = ROOT / "outputs/overnight_corr15/FROZEN.json"
def _resolve_sf() -> Path:
    for raw in (
        os.environ.get("STOCKFISH_PATH", ""),
        "/root/.local/bin/stockfish-19",
        str(Path.home() / ".local/bin/stockfish-19"),
    ):
        if raw and Path(raw).exists():
            return Path(raw)
    return Path("/root/.local/bin/stockfish-19")


SF = _resolve_sf()
SCREEN_OPENINGS = {
    "startpos",
    "e2e4 e7e5",
    "d2d4 d7d5",
    "e2e4 c7c5",
    "d2d4 g8f6",
    "e2e4 e7e6",
    "c2c4 e7e5",
    "g1f3 d7d5",
}
SWA_SHA = "ea3255e60ff981800c11cdb960e7b0a45575f9d83a4984a99d8e54619fa3e7f4"
RESUME_SHA = "69aa7c8e5f4cf4c002a092e7cce37f3df795446924823d56f8d2b097f1ce0bdc"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sf_identity(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    proc = subprocess.run(
        [str(path)],
        input="uci\nquit\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    name = ""
    for line in proc.stdout.splitlines():
        if line.startswith("id name "):
            name = line[len("id name ") :]
            break
    return {
        "exists": True,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "uci_name": name,
        "matches_recorded_version": name == "Stockfish 19",
    }


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/heldout_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []

    def fail(m: str) -> None:
        errors.append(m)
        print("FAIL", m, flush=True)

    def warn(m: str) -> None:
        warnings.append(m)
        print("WARN", m, flush=True)

    def ok(m: str) -> None:
        print("OK", m, flush=True)

    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else {}
    swa_sha = sha256(SWA)
    resume_sha = sha256(RESUME)
    if swa_sha != SWA_SHA:
        fail(f"eval_swa sha {swa_sha} != {SWA_SHA}")
    else:
        ok(f"eval_swa sha {swa_sha[:12]}")
    if resume_sha != RESUME_SHA:
        fail(f"training_resume sha {resume_sha} != {RESUME_SHA}")
    else:
        ok(f"training_resume sha {resume_sha[:12]}")
    if frozen.get("incumbent", {}).get("sha256") and frozen["incumbent"]["sha256"] != swa_sha:
        fail("FROZEN incumbent sha mismatch")
    if frozen.get("training_resume", {}).get("sha256") and frozen["training_resume"]["sha256"] != resume_sha:
        fail("FROZEN resume sha mismatch")

    swa = torch.load(SWA, map_location="cpu", weights_only=False)
    resume = torch.load(RESUME, map_location="cpu", weights_only=False)
    swa_kind = classify_checkpoint(swa)
    resume_kind = classify_checkpoint(resume)
    if swa.get("eval_only") is not True:
        fail(f"eval_swa eval_only={swa.get('eval_only')}")
    else:
        ok("eval_swa is eval_only")
    if swa.get("optimizer_state_dict"):
        fail("eval_swa unexpectedly has optimizer")
    if resume_kind != "full":
        fail(f"training_resume classify={resume_kind}")
    else:
        ok("training_resume classify=full")
    if resume.get("eval_only"):
        fail("training_resume marked eval_only")
    if resume.get("optimizer_state_dict") is None:
        fail("training_resume missing optimizer")

    steps_swa = int(swa.get("steps") or -1)
    steps_res = int(resume.get("steps") or resume.get("global_step") or -1)
    if steps_swa != steps_res:
        fail(f"step mismatch swa={steps_swa} resume={steps_res}")
    else:
        ok(f"shared steps={steps_res}")
    if swa.get("arch") != resume.get("arch"):
        fail(f"arch mismatch {swa.get('arch')} vs {resume.get('arch')}")
    if swa.get("vocab_version", swa.get("vocab")) not in (None, "compact"):
        fail(f"swa vocab {swa.get('vocab_version')}")
    if resume.get("vocab_version", resume.get("vocab")) not in (None, "compact"):
        fail(f"resume vocab {resume.get('vocab_version')}")

    swa_sd = load_model_state(swa)
    res_sd = load_model_state(resume)
    if set(swa_sd) != set(res_sd):
        fail(f"model key mismatch extra={set(res_sd)-set(swa_sd)} missing={set(swa_sd)-set(res_sd)}")
    else:
        ok(f"model keys match n={len(res_sd)}")
    n_diff = sum(1 for k in res_sd if not torch.equal(res_sd[k].cpu(), swa_sd[k].cpu()))
    if n_diff == 0:
        fail("live and SWA weights are identical; they should not be")
    else:
        ok(f"live vs SWA differ on {n_diff}/{len(res_sd)} tensors (expected)")

    model = build_squares64(resume.get("config") or swa.get("config") or {})
    try:
        model.load_state_dict(res_sd, strict=True)
        ok(f"live weights load strict params={count_parameters(model)}")
    except Exception as e:
        fail(f"live weights load failed: {e}")
        model = None

    opt_ok = False
    opt_info: dict = {}
    if model is not None:
        train_cfg = resume.get("train") or {}
        opt_name = train_cfg.get("optimizer", "polar_normuon")
        muon_lr = float(train_cfg.get("muon_lr", 0.0007))
        adam_lr = float(train_cfg.get("adam_lr", 1e-5))
        wd = float(train_cfg.get("weight_decay", 0.01))
        opt_info = {
            "optimizer": opt_name,
            "muon_lr": muon_lr,
            "adam_lr": adam_lr,
            "weight_decay": wd,
            "mix_hint": {
                "deep_mix_frac": train_cfg.get("deep_mix_frac"),
                "bonus_mix_frac": train_cfg.get("bonus_mix_frac"),
            },
        }
        try:
            optimizer, muon_n, adam_n = build_polar_normuon_optimizer(
                model, muon_lr, adam_lr, wd, compile_polar=False
            )
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            opt_ok = True
            n_state = len(resume["optimizer_state_dict"].get("state") or {})
            opt_info.update({"muon_params": muon_n, "adam_params": adam_n, "opt_state_entries": n_state})
            ok(f"optimizer state loads onto freshly built PolarNorMuon ({muon_n/1e6:.1f}M+{adam_n/1e6:.1f}M)")
        except Exception as e:
            fail(f"optimizer does not load onto model from this checkpoint: {e}")

    if model is not None:
        try:
            swa_model = build_squares64(swa.get("config") or {})
            swa_model.load_state_dict(swa_sd, strict=True)
            opt2, _, _ = build_polar_normuon_optimizer(swa_model, 0.0007, 1e-5, 0.01, compile_polar=False)
            try:
                opt2.load_state_dict(resume["optimizer_state_dict"])
                warn("optimizer also loads onto SWA weights — grouping matches, but do not continue from this pairing")
            except Exception:
                ok("optimizer refuses SWA weights (or grouping differs); do not pair them anyway")
        except Exception as e:
            warn(f"SWA+optimizer pairing probe skipped: {e}")

    engine = sf_identity(SF)
    if not engine.get("exists"):
        fail(f"stockfish missing {SF}")
    elif not engine.get("matches_recorded_version"):
        fail(f"uci name {engine.get('uci_name')!r} != Stockfish 19")
    else:
        ok(f"engine {engine['uci_name']} {engine['path']}")

    ref = json.loads(REF.read_text())
    proto = ref.get("protocol") or {}
    games = ref.get("games") or []
    provenance = {
        "label": "cached_results",
        "replayed": False,
        "source": str(REF),
        "checkpoint": ref.get("checkpoint"),
        "checkpoint_exists": Path(ref.get("checkpoint") or "").exists(),
        "checkpoint_sha256": swa_sha,
        "n_games": len(games),
        "protocol": proto,
        "engine_now": engine,
        "engine_recorded": {"sf_path": proto.get("sf_path"), "sf_version": proto.get("sf_version")},
        "path_match": Path(proto.get("sf_path") or "").resolve() == SF.resolve() if proto.get("sf_path") else False,
        "version_match": proto.get("sf_version") == "Stockfish 19" and engine.get("uci_name") == "Stockfish 19",
        "protocol_ok": (
            proto.get("mode") == "policy"
            and proto.get("book") is False
            and proto.get("syzygy") is False
            and int(proto.get("nodes") or 0) == 8000
            and proto.get("elos") == [2050, 2200]
            and int(proto.get("games_per_opening_per_color") or 0) == 4
            and proto.get("vocab") == "compact"
            and int(proto.get("threads") or 0) == 1
            and int(proto.get("hash") or 0) == 32
            and set(proto.get("openings") or []) == SCREEN_OPENINGS
            and len(games) == 128
        ),
    }
    if not provenance["checkpoint_exists"]:
        fail("cached Elo checkpoint path missing")
    if Path(ref.get("checkpoint") or "").resolve() != SWA.resolve():
        fail(f"cached Elo checkpoint {ref.get('checkpoint')} != {SWA}")
    if not provenance["path_match"]:
        fail(f"recorded sf_path {proto.get('sf_path')} != {SF}")
    if not provenance["version_match"]:
        fail("recorded sf_version does not match live UCI name")
    if not provenance["protocol_ok"]:
        fail("cached Elo protocol does not match the 128-game 2050/2200 screen")
    if not errors:
        ok("cached incumbent Elo provenance matches current engine and protocol")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "eval_swa": {
            "path": str(SWA),
            "sha256": swa_sha,
            "eval_only": bool(swa.get("eval_only")),
            "steps": steps_swa,
            "swa_n": swa.get("swa_n"),
            "classify": swa_kind,
            "has_optimizer": swa.get("optimizer_state_dict") is not None,
        },
        "training_resume": {
            "path": str(RESUME),
            "sha256": resume_sha,
            "eval_only": bool(resume.get("eval_only")),
            "steps": steps_res,
            "swa_n": resume.get("swa_n"),
            "classify": resume_kind,
            "resume_kind": resume.get("resume_kind"),
            "has_optimizer": resume.get("optimizer_state_dict") is not None,
            "optimizer_loads_on_own_weights": opt_ok,
            "n_keys_differ_from_swa": n_diff,
            "train": opt_info,
            "note": "Live weights + optimizer at the same step as SWA. Continuation from this file is a different experiment from SWA warm starts. Never load this optimizer into eval_swa.pt.",
        },
        "cached_elo_provenance": provenance,
        "recommendation": {
            "reuse_cached_8opening_screen": bool(provenance["protocol_ok"] and provenance["version_match"] and not errors),
            "reuse_cached_on_heldout_v1": False,
            "continue_from": str(RESUME) if opt_ok else None,
            "do_not_pair_optimizer_with_swa": True,
        },
    }
    dest = out_dir / "provenance_and_resume.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", dest, flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
