#!/usr/bin/env python3
"""Parse official Stockfish fishtest self-play PGNs into a training cache.

Each PGN is Stockfish-vs-Stockfish with per-move Stockfish eval comments.
We extract (position -> move_played, SF_centipawn, SF_depth, game_result) and
emit a soft/hard cache in the pipeline's tensor format:
  board_array(fused 64), turn, castling, ep_square, move_idx, cp, mate, result,
  soft_indices(most moves) + soft_probs.

Usage:
  python data_convert_fishtest.py --pgns /tmp/fishtest/*.pgn.gz --out outputs/fishtest_cache.pt --limit 20000
"""
from __future__ import annotations
import argparse, glob, gzip, io, json, os, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import chess, chess.pgn, torch
from move_vocab import UCI_TO_IDX, move_to_index

CE_RE = re.compile(r'([+-]?\d+\.?\d*)/(\d+)')


def result_to_class(res: str, turn: chess.Color) -> float:
    """Return Q in [-1,1] for the side-to-move."""
    if res == '1-0':
        return 1.0 if turn == chess.WHITE else -1.0
    if res == '0-1':
        return -1.0 if turn == chess.WHITE else 1.0
    return 0.0


def parse_game(path, out_rows, max_depth=0, limit=None):
    with gzip.open(path, 'rb') as f:
        text = f.read().decode('utf-8', errors='replace')
    pgn_io = io.StringIO(text)
    n_games = 0
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        result = game.headers.get('Result', '*')
        board = game.board()
        for node in game.mainline():
            mv = node.move
            if mv is None or mv.uci() not in UCI_TO_IDX:
                if mv:
                    board.push(mv)
                continue
            fen = board.fen()
            m_idx = UCI_TO_IDX[mv.uci()]
            cp = None; depth = None
            m = CE_RE.search(node.comment or '')
            if m:
                cp = int(float(m.group(1)) * 100)
                depth = int(m.group(2))
            turn = board.turn
            q = result_to_class(result, turn)
            out_rows.append({
                "fen": fen, "move_idx": m_idx,
                "cp": cp, "depth": depth, "q": q,
            })
            board.push(mv)
            if limit and len(out_rows) >= limit:
                return n_games
        n_games += 1
    return n_games


def to_cache(rows):
    """rows -> tensors. board_array via fused_ids from chess.Board."""
    ba = torch.zeros(len(rows), 64, dtype=torch.int8)
    turn = torch.zeros(len(rows), dtype=torch.int8)
    castling = torch.zeros(len(rows), dtype=torch.int8)
    ep = torch.zeros(len(rows), dtype=torch.int8) - 1
    move_idx = torch.zeros(len(rows), dtype=torch.int64)
    cp = torch.zeros(len(rows), dtype=torch.int32)
    q = torch.zeros(len(rows), dtype=torch.float16)
    # fused tokens: 0=empty, 1-6=white P/N/B/R/Q/K, 7-12=black.
    fused_map = {(sym, True): i for i, sym in enumerate(['P','N','B','R','Q','K'], start=1)}
    fused_map.update({(sym, False): i for i, sym in enumerate(['P','N','B','R','Q','K'], start=7)})
    cmap = {chess.PAWN:'P',chess.KNIGHT:'N',chess.BISHOP:'B',chess.ROOK:'R',chess.QUEEN:'Q',chess.KING:'K'}
    for i, r in enumerate(rows):
        b = chess.Board(r['fen'])
        pid = torch.zeros(64, dtype=torch.int64)
        fid = torch.zeros(64, dtype=torch.int8)
        for sq, p in b.piece_map().items():
            k = cmap[p.piece_type]
            color = p.color
            fid[sq] = fused_map[(k, color)]
        ba[i] = fid
        turn[i] = 0 if b.turn == chess.WHITE else 1
        c = 0
        if b.has_kingside_castling_rights(chess.WHITE): c |= 1
        if b.has_queenside_castling_rights(chess.WHITE): c |= 2
        if b.has_kingside_castling_rights(chess.BLACK): c |= 4
        if b.has_queenside_castling_rights(chess.BLACK): c |= 8
        castling[i] = c
        if b.ep_square is not None: ep[i] = b.ep_square
        move_idx[i] = r['move_idx']
        cp[i] = r['cp'] if r['cp'] is not None else 0
        q[i] = r['q']
    return {
        "board_array": ba, "turn": turn, "castling": castling, "ep_square": ep,
        "move_idx": move_idx, "cp": cp, "result_q": q,
        "soft_indices": move_idx.unsqueeze(1), "soft_probs": torch.ones(len(rows),1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgns", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--min-depth", type=int, default=0)
    ap.add_argument("--tactical-only", action="store_true",
                    help="keep only positions with |cp|>150 (tactically decisive)")
    args = ap.parse_args()
    rows = []
    for pat in args.pgns:
        for path in sorted(glob.glob(pat)):
            try:
                parse_game(path, rows, max_depth=args.min_depth, limit=args.limit)
            except Exception as e:
                print("skip", path, str(e)[:50])
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break
    # filter by depth / tactical
    before = len(rows)
    if args.min_depth:
        rows = [r for r in rows if r['depth'] is not None and r['depth'] >= args.min_depth]
    if args.tactical_only:
        rows = [r for r in rows if r['cp'] is not None and abs(r['cp']) >= 150]
    print(f"parsed {len(rows)}/{before} rows (min_depth={args.min_depth}, tactical_only={args.tactical_only})")
    if not rows:
        print("no rows; abort"); return
    cache = to_cache(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    print("saved", args.out, "rows", len(rows))
    # quick stats
    import collections
    ph = collections.Counter()
    for r in rows:
        b = chess.Board(r['fen'])
        ph[('open' if len(b.piece_map())>=28 else 'mid' if len(b.piece_map())>=16 else 'end')]+=1
    print("phase counts:", dict(ph))


if __name__ == "__main__":
    main()
