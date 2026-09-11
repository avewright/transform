"""Load the Lichess / Encyclopaedia of Chess Openings dataset.

Source of truth is `data/lichess_openings/{a-e}.tsv` from
https://github.com/lichess-org/chess-openings (eco, name, pgn).
The HF card still says 3,704; current master is a bit larger.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "lichess_openings"
VOLUMES = ("a", "b", "c", "d", "e")
# HF card (2026-05-20). Live master may exceed this.
HF_CARD_N = 3704
TSV_URL = "https://raw.githubusercontent.com/lichess-org/chess-openings/master/{volume}.tsv"


def fen4(fen: str) -> str:
    return " ".join(fen.split()[:4])


@dataclass(frozen=True)
class Opening:
    eco: str
    name: str
    pgn: str
    uci: tuple[str, ...]
    fen: str
    terminal: bool
    volume: str

    @property
    def key(self) -> str:
        return fen4(self.fen)

    @property
    def ply(self) -> int:
        return len(self.uci)


def parse_opening_pgn(pgn: str) -> tuple[tuple[str, ...], chess.Board]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError(f"unreadable opening pgn: {pgn!r}")
    board = game.board()
    ucis: list[str] = []
    for mv in game.mainline_moves():
        if mv not in board.legal_moves:
            raise ValueError(f"illegal opening move {mv.uci()} in {pgn!r}")
        board.push(mv)
        ucis.append(mv.uci())
    return tuple(ucis), board


def parse_tsv_text(text: str, *, volume: str = "") -> list[Opening]:
    lines = text.splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    if header[:3] != ["eco", "name", "pgn"]:
        raise ValueError(f"unexpected openings header: {header!r}")
    out: list[Opening] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        eco, name, pgn = line.split("\t", 2)
        ucis, board = parse_opening_pgn(pgn)
        vol = volume or (eco[:1] if eco else "")
        out.append(Opening(
            eco=eco,
            name=name,
            pgn=pgn,
            uci=ucis,
            fen=board.fen(),
            terminal=board.is_game_over(claim_draw=True),
            volume=vol.upper(),
        ))
    return out


def load_openings(source: Path | None = None) -> list[Opening]:
    """Load every volume. `source` may be a directory of TSVs or one TSV."""
    src = Path(source) if source is not None else DEFAULT_DIR
    if src.is_file():
        return parse_tsv_text(src.read_text(encoding="utf-8"), volume=src.stem)
    if not src.is_dir():
        raise FileNotFoundError(f"lichess openings dir missing: {src}")
    rows: list[Opening] = []
    for vol in VOLUMES:
        path = src / f"{vol}.tsv"
        if not path.exists():
            continue
        rows.extend(parse_tsv_text(path.read_text(encoding="utf-8"), volume=vol))
    if not rows:
        raise FileNotFoundError(f"no a-e.tsv files in {src}")
    return rows


def prefix_fens(uci: tuple[str, ...] | list[str], *, include_start: bool = False) -> list[str]:
    board = chess.Board()
    out: list[str] = []
    if include_start:
        out.append(board.fen())
    for u in uci:
        mv = chess.Move.from_uci(u)
        if mv not in board.legal_moves:
            break
        board.push(mv)
        out.append(board.fen())
    return out


def start_positions(
    openings: list[Opening],
    *,
    include_prefixes: bool = True,
    include_startpos: bool = False,
    skip_terminal: bool = True,
) -> list[dict]:
    """Unique playable starts: ECO leaves, optionally every prefix on the line."""
    seen: set[str] = set()
    starts: list[dict] = []

    def add(fen: str, opening: Opening, kind: str) -> None:
        board = chess.Board(fen)
        if skip_terminal and board.is_game_over(claim_draw=True):
            return
        key = fen4(fen)
        if key in seen:
            return
        seen.add(key)
        starts.append({
            "fen": board.fen(),
            "key": key,
            "eco": opening.eco,
            "name": opening.name,
            "volume": opening.volume,
            "kind": kind,
            "book_ply": board.ply(),
        })

    if include_startpos:
        dummy = Opening(eco="", name="Start Position", pgn="", uci=(), fen=chess.Board().fen(),
                        terminal=False, volume="")
        add(dummy.fen, dummy, "startpos")
    for opening in openings:
        if include_prefixes:
            for fen in prefix_fens(opening.uci, include_start=False):
                kind = "leaf" if fen4(fen) == opening.key else "prefix"
                add(fen, opening, kind)
        else:
            add(opening.fen, opening, "leaf")
    return starts


def openings_summary(openings: list[Opening], starts: list[dict] | None = None) -> dict:
    keys = {o.key for o in openings}
    return {
        "n_rows": len(openings),
        "n_unique": len(keys),
        "n_terminal": sum(1 for o in openings if o.terminal),
        "n_eco": len({o.eco for o in openings}),
        "volumes": {v: sum(1 for o in openings if o.volume == v) for v in "ABCDE"},
        "hf_card_n": HF_CARD_N,
        "n_starts": len(starts) if starts is not None else None,
        "median_ply": sorted(o.ply for o in openings)[len(openings) // 2] if openings else 0,
        "max_ply": max((o.ply for o in openings), default=0),
    }
