#!/usr/bin/env python3
"""Hybrid UCI engine: Stockfish alpha-beta search + meta-attention transformer.

Goal: "combine both" — keep Stockfish's world-class search, but let a learned
meta-attention transformer (with a latent-search/refine head) *slightly affect
the move targets* at the root by blending its policy signal into Stockfish's
MultiPV root scores.

Mechanism per position:
  1. Ask Stockfish for the top-N root moves (MultiPV) with centipawn scores.
  2. Run the transformer on the same position -> move policy (softmax probs).
  3. Convert SF centipawns to a distribution, then blend:
        combined(move) = POLICY_WEIGHT * sf_p[move]
                       + (1 - POLICY_WEIGHT) * tf_p[move]
     with optional temperature (attention sharpen) on the transformer side.
  4. Play the top blended move.

Config via UCI options or CLI flags so the blend is tunable and measurable
against the plain native Stockfish baseline.

Usage:
  python hybrid_uci.py --stockfish stockfish/stockfish-native-arm64 \
      --checkpoint outputs/exp195_meta_latent_search/latest.pt \
      --multipv 8 --policy-weight 0.6 --temp 1.0
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import threading
from pathlib import Path

import chess
import chess.engine
import chess.polyglot
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chess_inference import load_checkpoint  # noqa: E402
from chess_features import batch_boards_to_fused_token_ids  # noqa: E402
from move_vocab import IDX_TO_UCI, legal_move_mask  # noqa: E402

DEFAULT_CKPT = ROOT / "outputs" / "exp195_meta_latent_search" / "latest.pt"


class HybridEngine:
    """UCI engine that blends Stockfish root eval with a transformer policy."""

    def __init__(
        self,
        sf_binary: str,
        checkpoint: str,
        multipv: int = 8,
        policy_weight: float = 0.6,
        temp: float = 1.0,
        sf_threads: int = 4,
        sf_hash: int = 64,
        sf_time: float = 2.0,
        device: str | None = None,
    ):
        self.sf_binary = sf_binary
        self.multipv = max(2, multipv)
        self.policy_weight = float(policy_weight)
        self.temp = float(temp)
        self.sf_threads = sf_threads
        self.sf_hash = sf_hash
        self.sf_time = float(sf_time)

        self.dev = torch.device(
            device
            or "cpu"  # default CPU: moving 437M params to MPS costs ~200s one-time
        )
        self.model = load_checkpoint(checkpoint, self.dev)
        self.model.eval()

        self._sf = None
        self._lock = threading.Lock()

    # ── Stockfish subprocess ────────────────────────────────────────────────
    def _start_sf(self):
        eng = chess.engine.SimpleEngine.popen_uci(str(self.sf_binary))
        eng.configure({
            "Threads": self.sf_threads,
            "Hash": self.sf_hash,
        })
        return eng

    def _get_root_analysis(self, board: chess.Board) -> list[dict]:
        """Return Stockfish root MultiPV: [{'uci','cp','mate'|None}, ...].

        Uses a time limit (not fixed depth) so per-move cost stays bounded
        and the engine is playable.
        """
        info = self.engine.analyse(
            board,
            chess.engine.Limit(time=min(self.sf_time, 5.0)),
            multipv=self.multipv,
        )
        moves = []
        for pv in info:
            if not pv.get("pv"):
                continue
            move = pv["pv"][0]
            score = pv.get("score")
            if score is None:
                continue
            cp = score.white().score(mate_score=32000)
            moves.append({"uci": move.uci(), "mv": move, "cp": cp})
        # keep only legal moves (SF only returns legal at root anyway)
        return moves

    # ── transformer policy ──────────────────────────────────────────────────
    @torch.no_grad()
    def _tf_policy(self, board: chess.Board) -> tuple[dict[str, float], dict]:
        """Return {uci: prob} over legal moves and WDL meta-info."""
        board_input = batch_boards_to_fused_token_ids([board], self.dev)
        with torch.amp.autocast("cuda", enabled=self.dev.type == "cuda"):
            result = self.model(board_input)
        logits = result["policy_logits"][0].float()
        mask = legal_move_mask(board).to(self.dev)
        logits[~mask] = float("-inf")
        logits = logits / max(self.temp, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        top_ids = probs.topk(self.multipv).indices.tolist()
        policy = {IDX_TO_UCI[i]: probs[i].item() for i in top_ids}

        val = result["value_logits"][0].float()
        if val.shape[-1] == 3:
            wdl = torch.softmax(val, dim=-1).tolist()
            meta = {"wdl_win": wdl[0], "wdl_draw": wdl[1], "wdl_loss": wdl[2]}
        else:
            n = val.shape[-1]
            centers = torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=val.device)
            wp = (torch.softmax(val, dim=-1) * centers).sum().item()
            meta = {"win_pct": wp}
        return policy, meta

    # ── blend ───────────────────────────────────────────────────────────────
    def combine(self, board: chess.Board) -> tuple[chess.Move, dict]:
        """Return (move, info) that blends SF root scores with policy."""
        sf_moves = self._get_root_analysis(board)
        if not sf_moves:
            return chess.Move.null(), {"note": "empty sf list"}

        tf_policy, meta = self._tf_policy(board)

        # Convert SF centipawns -> probability. Higher cp = better for the
        # side to move. Use a softmax over a normalized score with a scale
        # equal to the spread of the root candidates.
        cps = [m["cp"] for m in sf_moves]
        spread = max(1.0, (max(cps) - min(cps)))
        # normalized to [0,~1], best candidate -> highest logit
        sf_logits_t = torch.tensor(
            [(m["cp"] - min(cps)) / (spread + 1e-6) for m in sf_moves],
            dtype=torch.float32,
        )
        sf_p = torch.softmax(sf_logits_t, dim=-1).tolist()

        combined = {}
        for idx, m in enumerate(sf_moves):
            tf_p = tf_policy.get(m["uci"], 0.0)
            score = (self.policy_weight * sf_p[idx]
                     + (1.0 - self.policy_weight) * tf_p)
            combined[m["uci"]] = (score, m["cp"], tf_p)

        best_uci = max(combined, key=lambda k: combined[k][0])
        move = chess.Move.from_uci(best_uci)

        top = sorted(combined.items(), key=lambda kv: kv[1][0], reverse=True)[:5]
        info = {
            "best_cp": combined[best_uci][1],
            "blend": [
                {"uci": u, "score": round(s, 4),
                 "cp": c, "tf_p": round(t, 4)}
                for u, (s, c, t) in top
            ],
            "tf_meta": meta,
        }
        return move, info

    # ── UCI loop (minimal, enough for cutechess-cli / manual) ───────────────
    @property
    def engine(self):
        if self._sf is None:
            self._sf = self._start_sf()
        return self._sf

    def run(self):
        board = chess.Board()
        out = sys.stdout
        print(f"id name Hybrid-SF+TF(pw={self.policy_weight},t={self.temp})")
        print("id author transform", flush=True)
        print("option name MultiPV type spin default {} min 1 max 40".format(self.multipv))
        print("uok", flush=True)
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip()
            parts = cmd.split()
            if not parts:
                continue
            c = parts[0].lower()
            if c == "uci":
                print("uciok", flush=True)
            elif c == "isready":
                print("readyok", flush=True)
            elif c == "ucinewgame":
                if self._sf:
                    self._sf.quit()
                    self._sf = None
            elif c == "setoption":
                # name X value Y  (ignore for now)
                pass
            elif c == "position":
                self._set_position(parts[1:], board)
            elif c == "go":
                move, info = self.combine(board)
                self._announce(move, info)
            elif c == "quit":
                try:
                    self._sf.quit()
                except Exception:
                    pass
                try:
                    self._sf.close()
                except Exception:
                    pass
                break

    def _set_position(self, rest, board):
        board.reset()
        if rest and rest[0] == "startpos":
            rest = rest[1:]
        if rest and rest[0] == "fen":
            # fen f1 ... moves ...
            fen_parts = rest[1:7]
            board.set_fen(" ".join(fen_parts))
            rest = rest[7:]
        if rest and rest[0] == "moves":
            for m in rest[1:]:
                board.push_uci(m)

    def _announce(self, move, info):
        print(f"info string blend={info.get('blend')} tf={info.get('tf_meta')}", flush=True)
        print(f"bestmove {move.uci()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True,
                    help="path to (native) Stockfish binary")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--policy-weight", type=float, default=0.6,
                    help="0=pure transformer top-k, 1=pure Stockfish")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="temperature on the transformer policy")
    ap.add_argument("--sf-time", type=float, default=2.0,
                    help="Stockfish multiPV analysis time per move (s)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--play", nargs="?", default=None, const="e2e4",
                    help="optional: send a single 'go' for testing")
    args = ap.parse_args()

    eng = HybridEngine(
        sf_binary=args.stockfish,
        checkpoint=args.checkpoint,
        multipv=args.multipv,
        policy_weight=args.policy_weight,
        temp=args.temp,
        sf_threads=args.threads,
        sf_hash=args.hash,
        sf_time=args.sf_time,
        device=args.device,
    )
    eng.run()


if __name__ == "__main__":
    main()
