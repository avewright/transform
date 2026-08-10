"""Shared helpers for the max-Elo harness."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path(__file__).resolve().parent / "protocol.json"
CHAMPION_DIR = ROOT / "outputs" / "champion"
DEFAULT_SEED_CKPT = ROOT / "outputs" / "hf_437m_ft3h_hub" / "best_model.pt"


def load_protocol(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else PROTOCOL_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def pick_device(explicit: str | None = None):
    import torch

    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_compact_vocab(checkpoint: Path | str | None = None) -> str:
    """Force MOVE_VOCAB_VERSION before move_vocab is imported.

    Must be called before any `import move_vocab` / factory import that pulls vocab.
    """
    wanted = "compact"
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            import torch

            try:
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                vv = ck.get("vocab_version")
                if vv and vv != "compact":
                    raise RuntimeError(
                        f"Checkpoint vocab_version={vv!r} but harness requires compact"
                    )
                if vv:
                    wanted = vv
            except RuntimeError:
                raise
            except Exception:
                pass
    existing = os.environ.get("MOVE_VOCAB_VERSION")
    if existing and existing != wanted:
        raise RuntimeError(
            f"MOVE_VOCAB_VERSION={existing!r} conflicts with required {wanted!r}"
        )
    os.environ["MOVE_VOCAB_VERSION"] = wanted
    return wanted


def resolve_stockfish() -> Path:
    configured = os.environ.get("STOCKFISH_PATH")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    which = shutil.which("stockfish")
    if which:
        candidates.append(Path(which))
    candidates.extend(
        [
            ROOT / "stockfish" / "stockfish-native-arm64",
            ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
            Path("/opt/homebrew/bin/stockfish"),
            Path("/usr/local/bin/stockfish"),
            Path("/usr/games/stockfish"),
            Path("/usr/bin/stockfish"),
        ]
    )
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Stockfish not found. Set STOCKFISH_PATH or install stockfish on PATH."
    )


def stockfish_version(sf_path: Path | str) -> str:
    """Return UCI `id name` line (best-effort)."""
    try:
        proc = subprocess.run(
            [str(sf_path)],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in (proc.stdout or "").splitlines():
            if line.startswith("id name "):
                return line[len("id name ") :].strip()
    except Exception as e:
        return f"unknown ({e})"
    return "unknown"


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return None


def opening_name(opening: list[str]) -> str:
    return "startpos" if not opening else " ".join(opening)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def ensure_best_aliases(out_dir: Path) -> Path | None:
    """Ensure best_model.pt → best.pt for legacy scripts. Returns best path if present."""
    best = out_dir / "best.pt"
    if not best.exists():
        # some older trees only have best_model.pt
        alt = out_dir / "best_model.pt"
        if alt.exists() and not alt.is_symlink():
            link_or_copy(alt, best)
        else:
            return None
    alias = out_dir / "best_model.pt"
    if not alias.exists() or alias.is_symlink() or alias.resolve() != best.resolve():
        link_or_copy(best, alias)
    return best
