#!/usr/bin/env python3
"""Mine positions where the champion blunders vs Stockfish and render them.

A blunder is a legal greedy policy move that drops >= 150cp versus SF's best
(or misses/allows mate). Writes PNG boards + an HTML gallery.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MOVE_VOCAB_VERSION", "compact")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import chess
import chess.engine
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from chess_features import batch_boards_to_fused_token_ids  # noqa: E402
from chess_inference import load_checkpoint  # noqa: E402
from harness.common import resolve_stockfish  # noqa: E402
from move_vocab import index_to_move, legal_move_mask  # noqa: E402

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
COORD = (90, 70, 50)
CAPTION_BG = (28, 28, 30)
SF_COLOR = (46, 160, 90)
MODEL_COLOR = (220, 50, 47)
SQ = 72
MARGIN = 28
PHASE_NAME = {0: "endgame", 1: "middlegame", 2: "opening"}
UNICODE_PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def fen6(fen4: str) -> str:
    return fen4 if fen4.count(" ") >= 5 else f"{fen4} 0 1"


def load_font(size: int, *, prefer_chess: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if prefer_chess:
        candidates.extend(
            [
                "/System/Library/Fonts/Apple Symbols.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def square_xy(square: int, origin: int) -> tuple[float, float]:
    file_i = chess.square_file(square)
    rank_i = chess.square_rank(square)
    x = origin + (file_i + 0.5) * SQ
    y = origin + (7 - rank_i + 0.5) * SQ
    return x, y


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color, width: int = 9) -> None:
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length < 8:
        return
    ux, uy = dx / length, dy / length
    # stop short of the destination piece
    x2 -= ux * (SQ * 0.28)
    y2 -= uy * (SQ * 0.28)
    x1 += ux * (SQ * 0.18)
    y1 += uy * (SQ * 0.18)
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    hx, hy = -uy, ux
    head = 18
    p1 = (x2, y2)
    p2 = (x2 - ux * head + hx * 8, y2 - uy * head + hy * 8)
    p3 = (x2 - ux * head - hx * 8, y2 - uy * head - hy * 8)
    draw.polygon([p1, p2, p3], fill=color)


def render_board(board: chess.Board, rec: dict, path: Path) -> None:
    origin = MARGIN
    board_px = 8 * SQ
    caption_h = 118
    w = origin * 2 + board_px
    h = origin + board_px + caption_h
    img = Image.new("RGB", (w, h), (36, 36, 38))
    draw = ImageDraw.Draw(img)
    piece_font = load_font(50, prefer_chess=True)
    letter_font = load_font(42)
    coord_font = load_font(16)
    title_font = load_font(20)
    body_font = load_font(17)
    small_font = load_font(13)

    draw.rectangle((0, origin + board_px, w, h), fill=CAPTION_BG)

    for rank in range(8):
        for file_i in range(8):
            x0 = origin + file_i * SQ
            y0 = origin + (7 - rank) * SQ
            dark = (file_i + rank) % 2 == 0
            draw.rectangle((x0, y0, x0 + SQ, y0 + SQ), fill=DARK if dark else LIGHT)

    # highlight from/to squares
    for move, color in (
        (chess.Move.from_uci(rec["sf_uci"]), (46, 160, 90, 70)),
        (chess.Move.from_uci(rec["model_uci"]), (220, 50, 47, 70)),
    ):
        for sq in (move.from_square, move.to_square):
            fx = origin + chess.square_file(sq) * SQ
            fy = origin + (7 - chess.square_rank(sq)) * SQ
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rectangle((fx, fy, fx + SQ, fy + SQ), fill=color)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

    for sq, piece in board.piece_map().items():
        cx, cy = square_xy(sq, origin)
        glyph = UNICODE_PIECES[piece.symbol()]
        fill = (20, 20, 20) if piece.color == chess.WHITE else (15, 15, 15)
        # white pieces: white fill + dark outline via double draw
        if piece.color == chess.WHITE:
            fill = (250, 250, 250)
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.text((cx + ox, cy + oy), glyph, font=piece_font, fill=(40, 40, 40), anchor="mm")
        try:
            draw.text((cx, cy + 2), glyph, font=piece_font, fill=fill, anchor="mm")
        except Exception:
            draw.text((cx, cy), piece.symbol().upper(), font=letter_font, fill=fill, anchor="mm")

    arrows = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ad = ImageDraw.Draw(arrows)
    sf_m = chess.Move.from_uci(rec["sf_uci"])
    model_m = chess.Move.from_uci(rec["model_uci"])
    draw_arrow(ad, square_xy(sf_m.from_square, origin), square_xy(sf_m.to_square, origin), (*SF_COLOR, 220), 10)
    draw_arrow(ad, square_xy(model_m.from_square, origin), square_xy(model_m.to_square, origin), (*MODEL_COLOR, 220), 10)
    img = Image.alpha_composite(img.convert("RGBA"), arrows).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i in range(8):
        draw.text((origin - 10, origin + (7 - i) * SQ + SQ / 2), str(i + 1), font=coord_font, fill=(200, 196, 188), anchor="mm")
        draw.text((origin + i * SQ + SQ / 2, origin + board_px + 12), chr(ord("a") + i), font=coord_font, fill=(200, 196, 188), anchor="mm")

    side = "White" if rec["turn"] == 0 else "Black"
    title = f"#{rec['idx']:03d}  {side} to move  ·  {rec['phase_name']}  ·  drop {rec['drop_label']}"
    sf_line = f"Stockfish  {rec['sf_san']}   ({rec['sf_eval']})"
    model_line = f"Model      {rec['model_san']}   ({rec['model_eval']})"
    y = origin + board_px + 28
    draw.text((origin, y), title, font=title_font, fill=(245, 245, 247))
    draw.text((origin, y + 28), sf_line, font=body_font, fill=SF_COLOR)
    draw.text((origin, y + 50), model_line, font=body_font, fill=MODEL_COLOR)
    draw.text((origin, y + 76), rec["fen"], font=small_font, fill=(160, 160, 165))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG", optimize=True)


def sample_labeled_fens(n: int, seed: int) -> list[dict]:
    pos_dir = ROOT / "outputs/chess_master_v1/positions"
    ann_dir = ROOT / "outputs/chess_master_v1/annotations"
    shards = sorted(pos_dir.glob("mix-*.parquet"))
    if not shards:
        raise SystemExit(f"no mix parquets under {pos_dir}")
    rng = np.random.default_rng(seed)
    per = max(400, n // max(len(shards), 1) + 80)
    picked: list[dict] = []
    for pos_path in shards:
        ann_path = ann_dir / pos_path.name
        if not ann_path.exists():
            continue
        pos = pq.read_table(
            pos_path,
            columns=["position_id", "fen_4", "phase", "turn", "in_check", "piece_count"],
        )
        ann = pq.read_table(
            ann_path,
            columns=["position_id", "best_uci", "source_name", "annotation_type"],
        )
        keep = pc.and_(
            pc.equal(ann["annotation_type"], "engine_policy"),
            pc.is_valid(ann["best_uci"]),
        )
        if "sf19" in set(ann["source_name"].to_pylist()):
            keep = pc.and_(keep, pc.equal(ann["source_name"], "sf19"))
        ann = ann.filter(keep)
        joined = pos.join(ann, keys="position_id", join_type="inner")
        if joined.num_rows == 0:
            continue
        take = min(per, joined.num_rows)
        idx = rng.choice(joined.num_rows, size=take, replace=False)
        idx.sort()
        sl = joined.take(pa.array(idx))
        fens = sl["fen_4"].to_pylist()
        bests = sl["best_uci"].to_pylist()
        phases = sl["phase"].to_pylist()
        turns = sl["turn"].to_pylist()
        checks = sl["in_check"].to_pylist()
        pieces = sl["piece_count"].to_pylist()
        for fen, best, phase, turn, check, pc_n in zip(fens, bests, phases, turns, checks, pieces):
            if not fen or not best:
                continue
            picked.append(
                {
                    "fen": fen6(fen),
                    "teacher_uci": best,
                    "phase": int(phase) if phase is not None else 1,
                    "turn": int(turn),
                    "in_check": bool(check),
                    "piece_count": int(pc_n or 0),
                }
            )
        if len(picked) >= n * 2:
            break
    rng.shuffle(picked)
    # de-dupe by fen
    seen = set()
    uniq = []
    for row in picked:
        if row["fen"] in seen:
            continue
        seen.add(row["fen"])
        uniq.append(row)
        if len(uniq) >= n:
            break
    return uniq


@torch.no_grad()
def predict_moves(model, boards: list[chess.Board], device) -> list[chess.Move]:
    inp = batch_boards_to_fused_token_ids(boards, device)
    logits = model(inp)["policy_logits"].float()
    out = []
    for i, board in enumerate(boards):
        mask = legal_move_mask(board).to(device)
        row = logits[i]
        row = row.masked_fill(~mask, float("-inf"))
        out.append(index_to_move(int(row.argmax().item())))
    return out


def score_cp(score: chess.engine.PovScore) -> tuple[int, int]:
    if score.is_mate():
        mate = int(score.mate())
        return (10_000 - abs(mate) * 10) * (1 if mate > 0 else -1), mate
    return int(score.score(mate_score=10_000)), 0


def fmt_eval(cp: int, mate: int) -> str:
    if mate:
        return f"#{mate}"
    return f"{cp / 100:+.2f}"


def fmt_drop(cp: int, kind: str) -> str:
    if kind == "missed_mate":
        return "missed mate"
    if kind == "allowed_mate":
        return "allowed mate"
    return f"{cp}cp"


def classify(sf_cp: int, sf_mate: int, model_cp: int, model_mate: int) -> tuple[bool, int, str]:
    if sf_mate > 0 and model_mate <= 0:
        return True, 10_000, "missed_mate"
    if model_mate < 0 and sf_mate >= 0:
        return True, 10_000, "allowed_mate"
    drop = sf_cp - model_cp
    return drop >= 150, drop, "cp"


def analyse_pair(
    engine, board: chess.Board, model_move: chess.Move, depth: int, movetime: float
) -> dict | None:
    limit = chess.engine.Limit(depth=depth, time=movetime)
    infos = engine.analyse(
        board,
        limit,
        multipv=3,
        info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
    )
    if isinstance(infos, dict):
        infos = [infos]
    if not infos or "pv" not in infos[0] or not infos[0]["pv"]:
        return None
    best = infos[0]["pv"][0]
    sf_cp, sf_mate = score_cp(infos[0]["score"].pov(board.turn))
    model_cp = model_mate = None
    for info in infos:
        pv = info.get("pv") or []
        if pv and pv[0] == model_move:
            model_cp, model_mate = score_cp(info["score"].pov(board.turn))
            break
    if model_cp is None:
        try:
            info = engine.analyse(
                board,
                limit,
                root_moves=[model_move],
                info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
            )
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            return None
        if "score" not in info:
            return None
        model_cp, model_mate = score_cp(info["score"].pov(board.turn))
    is_blunder, drop, kind = classify(sf_cp, sf_mate, model_cp, model_mate)
    return {
        "sf_move": best,
        "sf_cp": sf_cp,
        "sf_mate": sf_mate,
        "model_cp": model_cp,
        "model_mate": model_mate,
        "drop": drop,
        "kind": kind,
        "is_blunder": is_blunder,
    }


def pick_diverse(rows: list[dict], k: int) -> list[dict]:
    buckets = defaultdict(list)
    for r in rows:
        drop = r["drop"]
        if r["kind"] != "cp":
            dbin = "mate"
        elif drop >= 600:
            dbin = "major"
        elif drop >= 300:
            dbin = "blunder"
        else:
            dbin = "sharp"
        key = (r["phase"], r["turn"], dbin)
        buckets[key].append(r)
    for key in buckets:
        buckets[key].sort(key=lambda x: -x["drop"])
    keys = list(buckets)
    random.Random(0).shuffle(keys)
    chosen, used_fen = [], set()
    while len(chosen) < k and any(buckets[k_] for k_ in keys):
        for key in keys:
            if len(chosen) >= k:
                break
            while buckets[key]:
                rec = buckets[key].pop(0)
                if rec["fen"] in used_fen:
                    continue
                used_fen.add(rec["fen"])
                chosen.append(rec)
                break
    chosen.sort(key=lambda x: (-(x["kind"] != "cp"), -x["drop"]))
    return chosen[:k]


def write_html(rows: list[dict], out_dir: Path, ckpt: str) -> None:
    cards = []
    for r in rows:
        cards.append(
            f"""<figure>
  <img src="boards/{r['idx']:03d}.png" alt="blunder {r['idx']}"/>
  <figcaption>
    <b>#{r['idx']:03d}</b> {html.escape(r['side'])} · {html.escape(r['phase_name'])} · {html.escape(r['drop_label'])}<br/>
    <span class="sf">SF {html.escape(r['sf_san'])} ({html.escape(r['sf_eval'])})</span>
    <span class="md">Model {html.escape(r['model_san'])} ({html.escape(r['model_eval'])})</span>
  </figcaption>
</figure>"""
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8"/>
<title>Champion blunders vs Stockfish</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #111; color: #eee; margin: 24px; }}
  h1 {{ font-size: 22px; font-weight: 600; }}
  p {{ color: #bbb; max-width: 80ch; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 22px; }}
  figure {{ margin: 0; background: #1c1c1e; border-radius: 10px; overflow: hidden; }}
  img {{ width: 100%; display: block; }}
  figcaption {{ padding: 10px 12px 14px; font-size: 13px; line-height: 1.45; }}
  .sf {{ color: #2ea05a; margin-right: 12px; }}
  .md {{ color: #dc322f; }}
</style>
<h1>100 champion blunders vs Stockfish</h1>
<p>Greedy policy of <code>{html.escape(ckpt)}</code> vs Stockfish 19. Kept when the model move drops ≥150cp or misses/allows mate. Green arrow = Stockfish, red arrow = model.</p>
<div class="grid">
{chr(10).join(cards)}
</div>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs/champion/champion.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/blunder_gallery"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pool", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--movetime", type=float, default=0.12, help="SF seconds per search")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = Path(args.out)
    boards_dir = out_dir / "boards"
    boards_dir.mkdir(parents=True, exist_ok=True)

    log(f"sampling {args.pool} labeled positions")
    rows = sample_labeled_fens(args.pool, args.seed)
    log(f"sampled {len(rows)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"loading {args.checkpoint} on {device}")
    model = load_checkpoint(args.checkpoint, device)
    model.eval()

    disagreements: list[dict] = []
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        boards = []
        keep_chunk = []
        for rec in chunk:
            board = chess.Board(rec["fen"])
            if board.is_game_over() or board.legal_moves.count() == 0:
                continue
            boards.append(board)
            keep_chunk.append(rec)
        if not boards:
            continue
        preds = predict_moves(model, boards, device)
        for rec, board, pred in zip(keep_chunk, boards, preds):
            teacher = rec["teacher_uci"]
            if pred.uci() == teacher:
                continue
            rec = dict(rec)
            rec["board"] = board
            rec["model_move"] = pred
            disagreements.append(rec)
        if (start // args.batch_size) % 20 == 0:
            log(f"  scored {min(start + args.batch_size, len(rows))}/{len(rows)} disagree={len(disagreements)}")
    random.Random(args.seed).shuffle(disagreements)

    log(
        f"disagreements {len(disagreements)} — SF depth<={args.depth} "
        f"movetime={args.movetime}s"
    )
    sf_path = str(resolve_stockfish())

    def start_engine():
        eng = chess.engine.SimpleEngine.popen_uci(sf_path)
        eng.configure({"Threads": 2, "Hash": 64})
        return eng

    engine = start_engine()
    blunders: list[dict] = []
    try:
        for i, rec in enumerate(disagreements):
            board: chess.Board = rec["board"]
            model_move: chess.Move = rec["model_move"]
            if model_move not in board.legal_moves:
                continue
            try:
                result = analyse_pair(engine, board, model_move, args.depth, args.movetime)
            except (chess.engine.EngineError, chess.engine.EngineTerminatedError, TimeoutError) as exc:
                log(f"  sf skip/restart: {type(exc).__name__}: {exc}")
                try:
                    engine.quit()
                except Exception:
                    pass
                engine = start_engine()
                continue
            if not result or not result["is_blunder"]:
                if i and i % 25 == 0:
                    log(f"  analysed {i}/{len(disagreements)} blunders={len(blunders)}")
                continue
            sf_move = result["sf_move"]
            if sf_move == model_move:
                continue
            try:
                sf_san = board.san(sf_move)
                model_san = board.san(model_move)
            except ValueError:
                sf_san, model_san = sf_move.uci(), model_move.uci()
            blunders.append(
                {
                    "fen": board.fen(),
                    "turn": 0 if board.turn == chess.WHITE else 1,
                    "side": "White" if board.turn == chess.WHITE else "Black",
                    "phase": rec["phase"],
                    "phase_name": PHASE_NAME.get(rec["phase"], "middlegame"),
                    "in_check": board.is_check(),
                    "piece_count": rec["piece_count"],
                    "sf_uci": sf_move.uci(),
                    "model_uci": model_move.uci(),
                    "sf_san": sf_san,
                    "model_san": model_san,
                    "sf_eval": fmt_eval(result["sf_cp"], result["sf_mate"]),
                    "model_eval": fmt_eval(result["model_cp"], result["model_mate"]),
                    "drop": int(result["drop"]),
                    "kind": result["kind"],
                    "drop_label": fmt_drop(result["drop"], result["kind"]),
                    "teacher_uci": rec["teacher_uci"],
                }
            )
            if i and i % 25 == 0:
                log(f"  analysed {i}/{len(disagreements)} blunders={len(blunders)}")
            # keep extra so diversity pick has room
            if len(blunders) >= args.n * 3:
                log(f"  enough candidates ({len(blunders)}), stopping analysis")
                break
    finally:
        engine.quit()

    log(f"raw blunders {len(blunders)}")
    if len(blunders) < args.n:
        log(f"warning: only {len(blunders)} blunders found, rendering all")
        chosen = blunders
    else:
        chosen = pick_diverse(blunders, args.n)
    for i, rec in enumerate(chosen, 1):
        rec["idx"] = i

    log(f"rendering {len(chosen)} boards -> {boards_dir}")
    for rec in chosen:
        board = chess.Board(rec["fen"])
        render_board(board, rec, boards_dir / f"{rec['idx']:03d}.png")

    dump = [{k: v for k, v in rec.items() if k != "board"} for rec in chosen]
    (out_dir / "blunders.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
    write_html(chosen, out_dir, args.checkpoint)
    log(f"done {len(chosen)} -> {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
