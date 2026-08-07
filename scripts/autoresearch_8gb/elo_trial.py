"""Run elo_eval_latest on a trial checkpoint; return estimate dict."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def run_elo_trial(
    ckpt_path: Path | str,
    out_prefix: str,
    *,
    model_config: Path | str | None = None,
    movetime: float = 0.05,
    games_per_opening_per_color: int | None = None,
    elos: list[int] | None = None,
    stop_after_bracket: bool = True,
    smoke: bool = False,
) -> dict[str, Any]:
    ckpt_path = Path(ckpt_path)
    # F0.3: denser gauntlet so we escape the 1320 floor noise.
    # Stockfish 18 UCI_Elo min is 1320 — lower levels raise EngineError.
    elos = elos or ([1320, 1450] if smoke else [1320, 1450, 1600, 1750, 1900, 2050, 2200])
    if games_per_opening_per_color is None:
        games_per_opening_per_color = 1 if smoke else 2
    if smoke:
        games_per_opening_per_color = 1
        # Keep smoke short: few levels, still real games
        elos = elos[:2]

    cmd = [
        sys.executable, "-u", str(ROOT / "elo_eval_latest.py"),
        str(ckpt_path), out_prefix,
        "--movetime", str(movetime),
        "--games-per-opening-per-color", str(games_per_opening_per_color),
        "--elos", *[str(e) for e in elos],
    ]
    if stop_after_bracket:
        cmd.append("--stop-after-bracket")
    if model_config is not None:
        cmd.extend(["--model-config", str(model_config)])

    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env.setdefault("MOVE_VOCAB_VERSION", "compact")
    env.setdefault("PYTHONUNBUFFERED", "1")

    proc = subprocess.run(
        cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    json_path = ROOT / "outputs" / f"elo_eval_{out_prefix}.json"
    result: dict[str, Any] = {
        "rc": proc.returncode,
        "json_path": str(json_path) if json_path.exists() else None,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        est = data.get("estimate") or {}
        result["estimate"] = est
        elo = est.get("estimated_elo")
        if elo is None:
            elo = est.get("elo")
        if elo is None and est.get("lower_bound") is not None and est.get("upper_bound") is not None:
            try:
                elo = 0.5 * (float(est["lower_bound"]) + float(est["upper_bound"]))
            except (TypeError, ValueError):
                elo = None
        result["elo"] = elo
        result["summaries"] = data.get("summaries")
    else:
        result["elo"] = None
        result["estimate"] = None
    return result
