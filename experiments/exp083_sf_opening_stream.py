"""exp083: Stream Stockfish opening games into a persistent soft-target dataset.

Pipeline:
  1. Play Stockfish depth 5 vs Stockfish depth 4 from a fixed opening book.
  2. For every visited position, score every legal move with Stockfish depth 5.
  3. Persist the labeled position to JSONL for future training.
  4. Feed new records into a GPU trainer while generation continues on CPU.

This is intended to run continuously and can be resumed from checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue

import chess
import chess.engine
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import ChessTransformerConfig, build_model, count_parameters
from move_vocab import UCI_TO_IDX, VOCAB_SIZE

OUTPUT_DIR = Path("outputs/exp083_sf_opening_stream")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
DATASET_DIR = OUTPUT_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "positions.jsonl"
GAMES_PATH = DATASET_DIR / "games.jsonl"
STATUS_PATH = OUTPUT_DIR / "status.json"
LOG_PATH = OUTPUT_DIR / "exp083.log"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
DEFAULT_CONFIG_PATH = Path("configs/chess_transformer_16l_p256_no_pos.json")
DEFAULT_INIT_CHECKPOINT = Path("outputs/hf/chess-transformer-200m-latest/best_model.pt")
STOCKFISH_PATH = Path("stockfish/stockfish/stockfish-windows-x86-64-avx2.exe")

TRAIN_BATCH = 4
TRAIN_ACCUM = 16
TRAIN_LR = 1e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
SOFT_TARGET_TAU = 120.0
HARD_CE_WEIGHT = 0.25
VALUE_LOSS_WEIGHT = 0.10
MIN_BUFFER_TO_TRAIN = 128
REPLAY_BUFFER_MAX = 50000
SAVE_INTERVAL = 200
LOG_INTERVAL = 25
STATUS_INTERVAL_SEC = 15.0

SF_STRONG_DEPTH = 5
SF_WEAK_DEPTH = 4
SF_LABEL_DEPTH = 5
SF_THREADS = 1
SF_HASH_MB = 64
MAX_GAME_PLIES = 180
QUEUE_MAXSIZE = 1024
RANDOM_POSITION_MIN_PLIES = 8
RANDOM_POSITION_MAX_PLIES = 80
RANDOM_MULTI_PV = 4
RANDOM_MOVE_TAU = 75.0

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

OPENINGS = [
    {"name": "startpos", "moves": []},
    {"name": "king_pawn_game", "moves": ["e2e4", "e7e5"]},
    {"name": "sicilian_defense", "moves": ["e2e4", "c7c5"]},
    {"name": "french_defense", "moves": ["e2e4", "e7e6"]},
    {"name": "caro_kann", "moves": ["e2e4", "c7c6"]},
    {"name": "pirc_defense", "moves": ["e2e4", "d7d6"]},
    {"name": "modern_defense", "moves": ["e2e4", "g7g6"]},
    {"name": "alekhine_defense", "moves": ["e2e4", "g8f6"]},
    {"name": "scandinavian_defense", "moves": ["e2e4", "d7d5"]},
    {"name": "owen_defense", "moves": ["e2e4", "b7b6"]},
    {"name": "nimzowitsch_defense", "moves": ["e2e4", "b8c6"]},
    {"name": "queen_pawn_game", "moves": ["d2d4", "d7d5"]},
    {"name": "queens_gambit_declined", "moves": ["d2d4", "d7d5", "c2c4", "e7e6"]},
    {"name": "slav_defense", "moves": ["d2d4", "d7d5", "c2c4", "c7c6"]},
    {"name": "queens_gambit_accepted", "moves": ["d2d4", "d7d5", "c2c4", "d5c4"]},
    {"name": "kings_indian_defense", "moves": ["d2d4", "g8f6", "c2c4", "g7g6"]},
    {"name": "grunfeld_defense", "moves": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"]},
    {"name": "nimzo_indian_defense", "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]},
    {"name": "queens_indian_defense", "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"]},
    {"name": "benoni_defense", "moves": ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6"]},
    {"name": "benko_gambit", "moves": ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5"]},
    {"name": "dutch_defense", "moves": ["d2d4", "f7f5"]},
    {"name": "bird_opening", "moves": ["f2f4", "d7d5"]},
    {"name": "english_symmetrical", "moves": ["c2c4", "c7c5"]},
    {"name": "english_reversed_sicilian", "moves": ["c2c4", "e7e5"]},
    {"name": "reti_opening", "moves": ["g1f3", "d7d5", "c2c4"]},
    {"name": "catalan_setup", "moves": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5", "g2g3"]},
    {"name": "london_system", "moves": ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4"]},
    {"name": "colle_system", "moves": ["d2d4", "d7d5", "g1f3", "g8f6", "e2e3"]},
    {"name": "trompowsky_attack", "moves": ["d2d4", "g8f6", "c1g5"]},
    {"name": "torre_attack", "moves": ["d2d4", "g8f6", "g1f3", "e7e6", "c1g5"]},
    {"name": "veresov_attack", "moves": ["d2d4", "g8f6", "b1c3", "d7d5"]},
    {"name": "stonewall_attack", "moves": ["d2d4", "d7d5", "e2e3", "g8f6", "f2f4"]},
    {"name": "vienna_game", "moves": ["e2e4", "e7e5", "b1c3"]},
    {"name": "kings_gambit", "moves": ["e2e4", "e7e5", "f2f4"]},
    {"name": "bishops_opening", "moves": ["e2e4", "e7e5", "f1c4"]},
    {"name": "italian_game", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]},
    {"name": "ruy_lopez", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]},
    {"name": "scotch_game", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"]},
    {"name": "petroff_defense", "moves": ["e2e4", "e7e5", "g1f3", "g8f6"]},
    {"name": "philidor_defense", "moves": ["e2e4", "e7e5", "g1f3", "d7d6"]},
    {"name": "four_knights_game", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"]},
    {"name": "ponziani_opening", "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "c2c3"]},
    {"name": "sicilian_najdorf_setup", "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]},
    {"name": "sicilian_dragon_setup", "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"]},
    {"name": "sicilian_classical_setup", "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "b8c6"]},
    {"name": "french_advance", "moves": ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"]},
    {"name": "caro_advance", "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5"]},
    {"name": "caro_exchange", "moves": ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5"]},
    {"name": "scandinavian_mainline", "moves": ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"]},
]
OPENING_PHASE_GAMES = len(OPENINGS) * 2

LOG_FILE = None


def log(message: str) -> None:
    timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(timestamped + "\n")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def score_to_cp(score_obj: chess.engine.PovScore, pov_color: bool) -> tuple[int, str]:
    pov = score_obj.pov(pov_color)
    if pov.is_mate():
        mate = pov.mate()
        if mate is None:
            return 0, "cp"
        sign = 1 if mate > 0 else -1
        return sign * (100000 - min(abs(mate), 1000)), "mate"
    cp = pov.score(mate_score=100000)
    return int(cp if cp is not None else 0), "cp"


def cp_to_value_class(cp: int) -> int:
    if cp > 100:
        return 2
    if cp < -100:
        return 0
    return 1


def move_values_to_sparse_soft_targets(move_values: list[dict]) -> list[dict]:
    cps = torch.tensor([item["cp"] for item in move_values], dtype=torch.float32)
    probs = F.softmax(cps / SOFT_TARGET_TAU, dim=0).tolist()
    return [
        {"uci": item["uci"], "prob": float(prob), "cp": item["cp"], "eval_type": item["eval_type"]}
        for item, prob in zip(move_values, probs)
    ]


def sparse_soft_targets_to_dense(records: list[dict]) -> torch.Tensor:
    dense = torch.zeros(len(records), VOCAB_SIZE, dtype=torch.float32)
    for row_idx, record in enumerate(records):
        for target in record["soft_targets"]:
            dense[row_idx, UCI_TO_IDX[target["uci"]]] = float(target["prob"])
    return dense


def label_position(board: chess.Board, engine: chess.engine.SimpleEngine) -> dict:
    move_values = []
    for move in board.legal_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=SF_LABEL_DEPTH))
        board.pop()
        cp, eval_type = score_to_cp(info["score"], board.turn)
        move_values.append({"uci": move.uci(), "cp": cp, "eval_type": eval_type})
    move_values.sort(key=lambda item: item["cp"], reverse=True)
    best_cp = move_values[0]["cp"]
    return {
        "best_move": move_values[0]["uci"],
        "best_cp": best_cp,
        "value_target": cp_to_value_class(best_cp),
        "move_values": move_values,
        "soft_targets": move_values_to_sparse_soft_targets(move_values),
        "num_legal": len(move_values),
    }


def create_board_from_opening(opening_moves: list[str]) -> chess.Board:
    board = chess.Board()
    for uci in opening_moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal opening move {uci} for sequence {opening_moves}")
        board.push(move)
    return board


def load_recent_records(path: Path, max_records: int) -> list[dict]:
    if not path.exists():
        return []
    recent = deque(maxlen=max_records)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recent.append(json.loads(line))
    return list(recent)


def load_model_weights_flexible(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    clean_state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in clean_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)
    missing = [key for key in model_state.keys() if key not in matched]
    model.load_state_dict(matched, strict=False)
    return {"loaded": len(matched), "skipped": skipped, "missing": missing}


def build_manifest(args: argparse.Namespace, model_config: ChessTransformerConfig, init_report: dict) -> dict:
    return {
        "started_at": utcnow_iso(),
        "experiment": "exp083_sf_opening_stream",
        "paths": {
            "dataset": str(DATASET_PATH),
            "games": str(GAMES_PATH),
            "checkpoints": str(CHECKPOINT_DIR),
            "model_config": str(args.model_config),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        },
        "model": model_config.to_dict(),
        "init_report": init_report,
        "training": {
            "batch": TRAIN_BATCH,
            "accum": TRAIN_ACCUM,
            "lr": TRAIN_LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "hard_ce_weight": HARD_CE_WEIGHT,
            "value_loss_weight": VALUE_LOSS_WEIGHT,
            "min_buffer_to_train": MIN_BUFFER_TO_TRAIN,
            "replay_buffer_max": REPLAY_BUFFER_MAX,
        },
        "stockfish": {
            "strong_depth": SF_STRONG_DEPTH,
            "weak_depth": SF_WEAK_DEPTH,
            "label_depth": SF_LABEL_DEPTH,
            "threads": SF_THREADS,
            "hash_mb": SF_HASH_MB,
        },
        "generation": {
            "openings": len(OPENINGS),
            "switch_colors_per_opening": True,
            "opening_phase_games": OPENING_PHASE_GAMES,
            "post_opening_mode": "random_positions",
            "max_game_plies": MAX_GAME_PLIES,
            "max_games": args.max_games if args.max_games > 0 else None,
            "max_train_steps": args.max_train_steps if args.max_train_steps > 0 else None,
        },
    }


def save_status(state: dict) -> None:
    atomic_write_json(STATUS_PATH, state)


class GeneratorState:
    def __init__(self, opening_index: int = 0, reverse_colors: bool = False):
        self.opening_index = opening_index
        self.reverse_colors = reverse_colors
        self.games_generated = 0
        self.positions_generated = 0
        self.random_positions_generated = 0
        self.lock = threading.Lock()

    def mode(self) -> str:
        return "openings" if self.games_generated < OPENING_PHASE_GAMES else "random_positions"

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "mode": self.mode(),
                "opening_index": self.opening_index,
                "reverse_colors": self.reverse_colors,
                "games_generated": self.games_generated,
                "positions_generated": self.positions_generated,
                "random_positions_generated": self.random_positions_generated,
            }

    def advance_after_game(self) -> None:
        with self.lock:
            self.games_generated += 1
            if self.games_generated < OPENING_PHASE_GAMES and self.reverse_colors:
                self.opening_index = (self.opening_index + 1) % len(OPENINGS)
            self.reverse_colors = not self.reverse_colors

    def add_positions(self, count: int) -> None:
        with self.lock:
            self.positions_generated += count

    def add_random_position(self) -> None:
        with self.lock:
            self.positions_generated += 1
            self.random_positions_generated += 1


def create_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB})
    return engine


def play_engine_move(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> chess.Move:
    result = engine.play(board, chess.engine.Limit(depth=depth))
    move = result.move
    if move not in board.legal_moves:
        return next(iter(board.legal_moves))
    return move


def sample_engine_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
    multipv: int = RANDOM_MULTI_PV,
    tau: float = RANDOM_MOVE_TAU,
) -> chess.Move:
    k = min(multipv, board.legal_moves.count())
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    if not isinstance(info, list):
        info = [info]
    candidates = []
    for pv_info in info:
        pv = pv_info.get("pv", [])
        if not pv:
            continue
        move = pv[0]
        if move not in board.legal_moves:
            continue
        cp, _ = score_to_cp(pv_info["score"], board.turn)
        candidates.append((move, cp))
    if not candidates:
        return play_engine_move(engine, board, depth)
    cps = torch.tensor([cp for _, cp in candidates], dtype=torch.float32)
    probs = F.softmax(cps / tau, dim=0)
    idx = torch.multinomial(probs, num_samples=1).item()
    return candidates[idx][0]


def generate_random_position(
    strong_engine: chess.engine.SimpleEngine,
    weak_engine: chess.engine.SimpleEngine,
) -> tuple[chess.Board, dict]:
    board = chess.Board()
    target_plies = random.randint(RANDOM_POSITION_MIN_PLIES, RANDOM_POSITION_MAX_PLIES)
    strong_white = bool(random.getrandbits(1))
    move_history = []

    while len(board.move_stack) < target_plies and not board.is_game_over(claim_draw=True):
        actor_is_strong = (board.turn == chess.WHITE and strong_white) or (board.turn == chess.BLACK and not strong_white)
        actor_depth = SF_STRONG_DEPTH if actor_is_strong else SF_WEAK_DEPTH
        actor_engine = strong_engine if actor_is_strong else weak_engine
        move = sample_engine_move(actor_engine, board, actor_depth)
        move_history.append(move.uci())
        board.push(move)

    metadata = {
        "target_plies": target_plies,
        "actual_plies": len(board.move_stack),
        "strong_white": strong_white,
        "history": move_history,
    }
    return board, metadata


def generator_worker(
    out_queue: Queue,
    generator_state: GeneratorState,
    stop_event: threading.Event,
    done_event: threading.Event,
    max_games: int,
) -> None:
    strong_engine = create_engine()
    weak_engine = create_engine()
    label_engine = create_engine()
    try:
        while not stop_event.is_set():
            snapshot = generator_state.snapshot()
            if max_games > 0 and snapshot["games_generated"] >= max_games:
                break

            if snapshot["mode"] == "openings":
                opening = OPENINGS[snapshot["opening_index"]]
                strong_white = not snapshot["reverse_colors"]
                board = create_board_from_opening(opening["moves"])
                game_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_g{snapshot['games_generated'] + 1:05d}"
                local_positions = 0
                start_fen = board.fen()
                while not stop_event.is_set() and not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_GAME_PLIES:
                    label = label_position(board, label_engine)
                    actor_is_strong = (board.turn == chess.WHITE and strong_white) or (board.turn == chess.BLACK and not strong_white)
                    actor_depth = SF_STRONG_DEPTH if actor_is_strong else SF_WEAK_DEPTH
                    actor_label = "sf5" if actor_is_strong else "sf4"
                    actor_engine = strong_engine if actor_is_strong else weak_engine
                    played_move = play_engine_move(actor_engine, board, actor_depth)
                    record = {
                        "source": "sf5_vs_sf4_openings_v1",
                        "created_at": utcnow_iso(),
                        "game_id": game_id,
                        "generation_mode": "openings",
                        "opening_name": opening["name"],
                        "opening_index": snapshot["opening_index"],
                        "opening_moves": opening["moves"],
                        "strong_white": strong_white,
                        "ply": len(board.move_stack),
                        "fen": board.fen(),
                        "side_to_move": "white" if board.turn == chess.WHITE else "black",
                        "played_move": played_move.uci(),
                        "played_by": actor_label,
                        "played_depth": actor_depth,
                        "label_depth": SF_LABEL_DEPTH,
                        "best_move": label["best_move"],
                        "best_cp": label["best_cp"],
                        "value_target": label["value_target"],
                        "num_legal": label["num_legal"],
                        "move_values": label["move_values"],
                        "soft_targets": label["soft_targets"],
                    }
                    append_jsonl(DATASET_PATH, record)
                    while not stop_event.is_set():
                        try:
                            out_queue.put(record, timeout=1.0)
                            break
                        except Full:
                            continue
                    board.push(played_move)
                    local_positions += 1

                outcome = board.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    result = "1/2-1/2"
                    winner = None
                else:
                    result = board.result(claim_draw=True)
                    winner = "white" if outcome.winner == chess.WHITE else "black"
                append_jsonl(
                    GAMES_PATH,
                    {
                        "created_at": utcnow_iso(),
                        "game_id": game_id,
                        "generation_mode": "openings",
                        "opening_name": opening["name"],
                        "opening_index": snapshot["opening_index"],
                        "opening_moves": opening["moves"],
                        "strong_white": strong_white,
                        "start_fen": start_fen,
                        "final_fen": board.fen(),
                        "result": result,
                        "winner": winner,
                        "plies": len(board.move_stack),
                        "positions_labeled": local_positions,
                    },
                )
                generator_state.add_positions(local_positions)
                generator_state.advance_after_game()
                post = generator_state.snapshot()
                log(
                    f"generated game {post['games_generated']}: opening={opening['name']} "
                    f"strong_white={strong_white} result={result} plies={len(board.move_stack)} "
                    f"positions={local_positions}"
                )
                continue

            board, random_meta = generate_random_position(strong_engine, weak_engine)
            label = label_position(board, label_engine)
            random_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_r{snapshot['random_positions_generated'] + 1:07d}"
            record = {
                "source": "sf_random_positions_v1",
                "created_at": utcnow_iso(),
                "game_id": random_id,
                "generation_mode": "random_positions",
                "opening_name": None,
                "opening_index": None,
                "opening_moves": [],
                "strong_white": random_meta["strong_white"],
                "ply": len(board.move_stack),
                "fen": board.fen(),
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "played_move": None,
                "played_by": "mixed_randomized_sf",
                "played_depth": None,
                "label_depth": SF_LABEL_DEPTH,
                "best_move": label["best_move"],
                "best_cp": label["best_cp"],
                "value_target": label["value_target"],
                "num_legal": label["num_legal"],
                "move_values": label["move_values"],
                "soft_targets": label["soft_targets"],
                "random_start": {
                    "target_plies": random_meta["target_plies"],
                    "actual_plies": random_meta["actual_plies"],
                    "move_history": random_meta["history"],
                },
            }
            append_jsonl(DATASET_PATH, record)
            append_jsonl(
                GAMES_PATH,
                {
                    "created_at": utcnow_iso(),
                    "game_id": random_id,
                    "generation_mode": "random_positions",
                    "strong_white": random_meta["strong_white"],
                    "plies": random_meta["actual_plies"],
                    "positions_labeled": 1,
                    "move_history": random_meta["history"],
                    "final_fen": board.fen(),
                },
            )
            while not stop_event.is_set():
                try:
                    out_queue.put(record, timeout=1.0)
                    break
                except Full:
                    continue
            generator_state.add_random_position()
            post = generator_state.snapshot()
            if post["random_positions_generated"] % 10 == 0:
                log(
                    f"generated random positions={post['random_positions_generated']} "
                    f"total_positions={post['positions_generated']} last_ply={len(board.move_stack)}"
                )
    finally:
        done_event.set()
        strong_engine.quit()
        weak_engine.quit()
        label_engine.quit()


def sample_batch(records: list[dict], batch_size: int) -> list[dict]:
    if len(records) >= batch_size:
        return random.sample(records, batch_size)
    return [random.choice(records) for _ in range(batch_size)]


def train_optimizer_step(
    model: torch.nn.Module,
    optimizer: AdamW,
    scaler: GradScaler,
    replay_buffer: list[dict],
) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = ce_sum = kl_sum = val_sum = 0.0

    for _ in range(TRAIN_ACCUM):
        batch = sample_batch(replay_buffer, TRAIN_BATCH)
        boards = [chess.Board(item["fen"]) for item in batch]
        best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
        value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.long, device=DEVICE)
        soft_targets = sparse_soft_targets_to_dense(batch).to(DEVICE)
        board_input = batch_boards_to_fused_token_ids(boards, DEVICE)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
            out = model(board_input)
            logits = out["policy_logits"]
            value_logits = out["value_logits"]
            hard_ce = F.cross_entropy(logits, best_moves)
            kl = F.kl_div(F.log_softmax(logits, dim=-1), soft_targets, reduction="batchmean")
            value_loss = F.cross_entropy(value_logits, value_targets)
            total_loss = ((1.0 - HARD_CE_WEIGHT) * kl + HARD_CE_WEIGHT * hard_ce + VALUE_LOSS_WEIGHT * value_loss) / TRAIN_ACCUM

        scaler.scale(total_loss).backward()
        loss_sum += total_loss.item() * TRAIN_ACCUM
        ce_sum += hard_ce.item()
        kl_sum += kl.item()
        val_sum += value_loss.item()

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss_sum,
        "ce": ce_sum / TRAIN_ACCUM,
        "kl": kl_sum / TRAIN_ACCUM,
        "value": val_sum / TRAIN_ACCUM,
        "grad_norm": float(grad_norm),
    }


def drain_queue(queue: Queue, replay_buffer: deque, limit: int = 256) -> int:
    drained = 0
    while drained < limit:
        try:
            replay_buffer.append(queue.get_nowait())
            drained += 1
        except Empty:
            break
    return drained


def main() -> None:
    global LOG_FILE
    global DEVICE
    global AMP_ENABLED
    global SF_STRONG_DEPTH
    global SF_WEAK_DEPTH
    global SF_LABEL_DEPTH
    global MIN_BUFFER_TO_TRAIN
    global MAX_GAME_PLIES

    parser = argparse.ArgumentParser(description="Stream Stockfish opening games into concurrent training.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--device", type=str, default=str(DEVICE))
    parser.add_argument("--max-games", type=int, default=0, help="0 means run indefinitely.")
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means run indefinitely.")
    parser.add_argument("--strong-depth", type=int, default=SF_STRONG_DEPTH)
    parser.add_argument("--weak-depth", type=int, default=SF_WEAK_DEPTH)
    parser.add_argument("--label-depth", type=int, default=SF_LABEL_DEPTH)
    parser.add_argument("--min-buffer-to-train", type=int, default=MIN_BUFFER_TO_TRAIN)
    parser.add_argument("--max-game-plies", type=int, default=MAX_GAME_PLIES)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DEVICE = torch.device(args.device)
    AMP_ENABLED = DEVICE.type == "cuda"
    SF_STRONG_DEPTH = args.strong_depth
    SF_WEAK_DEPTH = args.weak_depth
    SF_LABEL_DEPTH = args.label_depth
    MIN_BUFFER_TO_TRAIN = args.min_buffer_to_train
    MAX_GAME_PLIES = args.max_game_plies

    for path in (OUTPUT_DIR, CHECKPOINT_DIR, DATASET_DIR):
        path.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_PATH

    device = DEVICE
    model_config = ChessTransformerConfig.from_json(args.model_config)
    model = build_model(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
    replay_buffer = deque(load_recent_records(DATASET_PATH, REPLAY_BUFFER_MAX), maxlen=REPLAY_BUFFER_MAX)
    train_steps = 0
    resume_loaded = False
    init_report = {"loaded": 0, "skipped": [], "missing": []}
    generator_state = GeneratorState()

    latest_ckpt = CHECKPOINT_DIR / "latest.pt"
    if latest_ckpt.exists():
        state = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler_state = state.get("scaler_state_dict", {})
        if scaler.is_enabled() and scaler_state:
            scaler.load_state_dict(scaler_state)
        train_steps = int(state.get("train_steps", 0))
        snapshot = state.get("generator_state", {})
        generator_state = GeneratorState(
            opening_index=int(snapshot.get("opening_index", 0)),
            reverse_colors=bool(snapshot.get("reverse_colors", False)),
        )
        generator_state.games_generated = int(snapshot.get("games_generated", 0))
        generator_state.positions_generated = int(snapshot.get("positions_generated", 0))
        generator_state.random_positions_generated = int(snapshot.get("random_positions_generated", 0))
        resume_loaded = True
        log(f"resumed checkpoint from {latest_ckpt} at train_step={train_steps}")
    elif args.init_checkpoint and args.init_checkpoint.exists():
        init_report = load_model_weights_flexible(model, args.init_checkpoint)
        log(
            f"loaded {init_report['loaded']} compatible tensors from {args.init_checkpoint}; "
            f"skipped={len(init_report['skipped'])} missing={len(init_report['missing'])}"
        )

    manifest = build_manifest(args, model_config, init_report if not resume_loaded else {"resume": str(latest_ckpt)})
    atomic_write_json(MANIFEST_PATH, manifest)

    log("=" * 72)
    log("exp083: Stockfish opening stream -> persistent soft-target dataset -> GPU training")
    log("=" * 72)
    log(f"device={device}")
    if device.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(device)} vram_gb={torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}")
    log(f"model_config={args.model_config}")
    log(f"params={count_parameters(model):,}")
    log(f"dataset_records_in_memory={len(replay_buffer)}")
    log(f"openings={len(OPENINGS)}")

    stream_queue: Queue = Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()
    done_event = threading.Event()
    worker = threading.Thread(
        target=generator_worker,
        args=(stream_queue, generator_state, stop_event, done_event, args.max_games),
        daemon=True,
    )
    worker.start()

    last_status_write = 0.0
    last_log_train = train_steps
    try:
        while True:
            drained = drain_queue(stream_queue, replay_buffer)
            if drained:
                snapshot = generator_state.snapshot()
                log(
                    f"ingested {drained} new records; buffer={len(replay_buffer)} "
                    f"games={snapshot['games_generated']} positions={snapshot['positions_generated']}"
                )

            if len(replay_buffer) >= MIN_BUFFER_TO_TRAIN:
                metrics = train_optimizer_step(model, optimizer, scaler, list(replay_buffer))
                train_steps += 1
                if train_steps - last_log_train >= LOG_INTERVAL:
                    last_log_train = train_steps
                    log(
                        f"train_step={train_steps} loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} "
                        f"kl={metrics['kl']:.4f} value={metrics['value']:.4f} gnorm={metrics['grad_norm']:.2f} "
                        f"buffer={len(replay_buffer)}"
                    )
                if train_steps % SAVE_INTERVAL == 0:
                    checkpoint_state = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "train_steps": train_steps,
                        "generator_state": generator_state.snapshot(),
                        "model_config": model_config.to_dict(),
                    }
                    save_checkpoint(latest_ckpt, checkpoint_state)
                    save_checkpoint(CHECKPOINT_DIR / f"step_{train_steps}.pt", checkpoint_state)
                    log(f"saved checkpoint at step {train_steps}")
            else:
                if done_event.is_set() and stream_queue.empty():
                    break
                time.sleep(0.25)

            if args.max_train_steps > 0 and train_steps >= args.max_train_steps:
                break

            now = time.time()
            if now - last_status_write >= STATUS_INTERVAL_SEC:
                last_status_write = now
                snapshot = generator_state.snapshot()
                save_status(
                    {
                        "updated_at": utcnow_iso(),
                        "train_steps": train_steps,
                        "buffer_size": len(replay_buffer),
                        "queue_size": stream_queue.qsize(),
                        "generator": snapshot,
                        "resume_loaded": resume_loaded,
                        "dataset_path": str(DATASET_PATH),
                        "games_path": str(GAMES_PATH),
                    }
                )
    except KeyboardInterrupt:
        log("keyboard interrupt received; shutting down")
    finally:
        stop_event.set()
        worker.join(timeout=10.0)
        checkpoint_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_steps": train_steps,
            "generator_state": generator_state.snapshot(),
            "model_config": model_config.to_dict(),
        }
        save_checkpoint(latest_ckpt, checkpoint_state)
        save_status(
            {
                "updated_at": utcnow_iso(),
                "train_steps": train_steps,
                "buffer_size": len(replay_buffer),
                "queue_size": stream_queue.qsize(),
                "generator": generator_state.snapshot(),
                "stopped": True,
                "dataset_path": str(DATASET_PATH),
                "games_path": str(GAMES_PATH),
            }
        )
        log(f"final checkpoint saved to {latest_ckpt}")


if __name__ == "__main__":
    main()
