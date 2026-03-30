"""exp082: Online Stockfish game loop with full legal-move soft labels.

Loop:
  1. Play full games against Stockfish.
  2. For every model position from those games, score every legal move with Stockfish.
  3. Convert the legal-move scores into soft policy targets.
  4. Train on those targets plus a replay buffer from prior cycles.
  5. Save checkpoints, cycle logs, and labeled positions, then repeat.

Designed for 8GB VRAM:
  - ChessTransformer200M
  - batch 4, accum 16
  - online labeling is CPU-bound; training remains GPU-light
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_model import FusedBoardEncoder
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, UCI_TO_IDX, index_to_move, legal_move_mask

OUTPUT_DIR = Path("outputs/exp082_sf_game_softloop")
CHECKPOINT_PATH = Path("outputs/hf/chess-transformer-200m-latest/best_model.pt")
STOCKFISH_PATH = Path("stockfish/stockfish/stockfish-windows-x86-64-avx2.exe")

TRAIN_BATCH = 4
TRAIN_ACCUM = 16
TRAIN_STEPS_PER_CYCLE = 150
TRAIN_LR = 3e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
SOFT_TARGET_TAU = 120.0
HARD_CE_WEIGHT = 0.30
VALUE_LOSS_WEIGHT = 0.10

GAMES_PER_CYCLE = 2
MAX_GAME_PLIES = 160
SF_PLAY_ELO = 1600
SF_PLAY_TIME = 0.05
SF_LABEL_DEPTH = 10
SF_THREADS = 1
SF_HASH_MB = 64

REPLAY_BUFFER_MAX = 12000
MAX_CYCLES = 1000000
OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
]

ENCODER_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 16
NUM_HEADS = 16
FFN_RATIO = 4
DROPOUT = 0.1
POLICY_HEAD_DIM = 512
VALUE_HIDDEN = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
LOG_FILE = None


def log(msg: str):
    print(msg, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _build_move_square_indices():
    from_sqs, to_sqs, promo_types = [], [], []
    promo_map = {"q": 1, "r": 2, "b": 3, "n": 4}
    for i in range(VOCAB_SIZE):
        uci = IDX_TO_UCI[i]
        from_sqs.append(chess.parse_square(uci[:2]))
        to_sqs.append(chess.parse_square(uci[2:4]))
        promo_types.append(promo_map.get(uci[4:5], 0))
    return (
        torch.tensor(from_sqs, dtype=torch.long),
        torch.tensor(to_sqs, dtype=torch.long),
        torch.tensor(promo_types, dtype=torch.long),
    )


class SpatialPolicyHead(nn.Module):
    def __init__(self, hidden_size, n_ctx_tokens=4, head_dim=512):
        super().__init__()
        self.n_ctx = n_ctx_tokens
        self.from_proj = nn.Linear(hidden_size, head_dim)
        self.to_proj = nn.Linear(hidden_size, head_dim)
        self.global_proj = nn.Linear(hidden_size, head_dim)
        self.promo_embed = nn.Embedding(5, head_dim)
        self.score_proj = nn.Linear(head_dim, 1)
        from_sqs, to_sqs, promo_types = _build_move_square_indices()
        self.register_buffer("from_sqs", from_sqs)
        self.register_buffer("to_sqs", to_sqs)
        self.register_buffer("promo_types", promo_types)

    def forward(self, hidden_states, cls_hidden):
        sq_hidden = hidden_states[:, self.n_ctx:self.n_ctx + 64, :]
        from_feats = sq_hidden[:, self.from_sqs, :]
        to_feats = sq_hidden[:, self.to_sqs, :]
        combined = (
            self.from_proj(from_feats) * self.to_proj(to_feats)
            + self.global_proj(cls_hidden).unsqueeze(1)
            + self.promo_embed(self.promo_types).unsqueeze(0)
        )
        return self.score_proj(F.relu(combined)).squeeze(-1)


class ChessTransformer200M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FusedBoardEncoder(embed_dim=ENCODER_DIM)
        self.input_proj = nn.Linear(ENCODER_DIM, HIDDEN_DIM)
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 68, HIDDEN_DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM * FFN_RATIO,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.policy_head = SpatialPolicyHead(HIDDEN_DIM, n_ctx_tokens=4, head_dim=POLICY_HEAD_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 3),
        )

    def forward(self, board_input):
        tokens = self.encoder(board_input)
        hidden = self.input_proj(tokens)
        bsz = hidden.shape[0]
        hidden = torch.cat([self.cls_token.expand(bsz, -1, -1), hidden], dim=1) + self.pos_embed
        hidden = self.norm(self.transformer(hidden))
        cls_hidden = hidden[:, 0, :]
        return {
            "policy_logits": self.policy_head(hidden, cls_hidden),
            "value_logits": self.value_head(cls_hidden),
        }


def load_model(path, device):
    model = ChessTransformer200M()
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def get_model_move(model, board, temperature=0.0):
    out = model(batch_boards_to_fused_token_ids([board], DEVICE))
    logits = out["policy_logits"][0].float()
    logits = logits.masked_fill(~legal_move_mask(board).to(DEVICE), float("-inf"))
    if temperature <= 0:
        idx = logits.argmax().item()
    else:
        idx = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item()
    move = index_to_move(idx)
    if move not in board.legal_moves:
        move = next(iter(board.legal_moves))
    return move


def score_to_cp(score_obj, pov_color):
    pov = score_obj.pov(pov_color)
    if pov.is_mate():
        mate = pov.mate()
        if mate is None:
            return 0, "cp"
        sign = 1 if mate > 0 else -1
        return sign * (100000 - min(abs(mate), 1000)), "mate"
    cp = pov.score(mate_score=100000)
    return int(cp if cp is not None else 0), "cp"


def cp_to_value_class(cp):
    if cp > 100:
        return 2
    if cp < -100:
        return 0
    return 1


def move_values_to_soft_target(move_values):
    cps = torch.tensor([mv["cp"] for mv in move_values], dtype=torch.float32)
    probs = F.softmax(cps / SOFT_TARGET_TAU, dim=0)
    target = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
    for mv, prob in zip(move_values, probs.tolist()):
        target[UCI_TO_IDX[mv["uci"]]] = prob
    return target


def label_position(board, engine):
    move_values = []
    for move in board.legal_moves:
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=SF_LABEL_DEPTH))
        board.pop()
        cp, eval_type = score_to_cp(info["score"], board.turn)
        move_values.append({"uci": move.uci(), "cp": cp, "eval_type": eval_type})
    move_values.sort(key=lambda mv: mv["cp"], reverse=True)
    best_cp = move_values[0]["cp"]
    return {
        "fen": board.fen(),
        "best_move": move_values[0]["uci"],
        "best_cp": best_cp,
        "value_target": cp_to_value_class(best_cp),
        "move_values": move_values,
    }


def play_game_vs_sf(model, engine, model_white, opening):
    board = chess.Board()
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)

    model_positions = []
    model_color = chess.WHITE if model_white else chess.BLACK
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_GAME_PLIES:
        if board.turn == model_color:
            model_positions.append(board.fen())
            move = get_model_move(model, board, temperature=0.0)
        else:
            move = engine.play(board, chess.engine.Limit(time=SF_PLAY_TIME)).move
        if move not in board.legal_moves:
            move = next(iter(board.legal_moves))
        board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        score = 0.5
    elif outcome.winner == model_color:
        score = 1.0
    else:
        score = 0.0
    return {
        "score": score,
        "result": board.result(claim_draw=True),
        "plies": len(board.move_stack),
        "positions": model_positions,
        "final_fen": board.fen(),
    }


def sample_training_batch(current_data, replay_data):
    batch = []
    replay_count = min(len(replay_data), TRAIN_BATCH // 2)
    current_count = TRAIN_BATCH - replay_count
    if current_data:
        batch.extend(random.sample(current_data, min(current_count, len(current_data))))
    if replay_count > 0:
        batch.extend(random.sample(replay_data, replay_count))
    while len(batch) < TRAIN_BATCH and current_data:
        batch.append(random.choice(current_data))
    random.shuffle(batch)
    return batch


def train_cycle(model, optimizer, current_data, replay_buffer):
    model.train()
    optimizer.zero_grad()
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    steps_done = 0
    micro = 0

    for _ in range(TRAIN_STEPS_PER_CYCLE * TRAIN_ACCUM):
        batch = sample_training_batch(current_data, replay_buffer)
        boards = [chess.Board(item["fen"]) for item in batch]
        targets = [UCI_TO_IDX[item["best_move"]] for item in batch]
        soft_targets = [move_values_to_soft_target(item["move_values"]) for item in batch]
        value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.long, device=DEVICE)

        out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
        logits = out["policy_logits"]
        value_logits = out["value_logits"]
        hard_ce = F.cross_entropy(logits, torch.tensor(targets, dtype=torch.long, device=DEVICE))
        soft_batch = torch.stack(soft_targets).to(DEVICE)
        kl = F.kl_div(F.log_softmax(logits, dim=-1), soft_batch, reduction="batchmean")
        value_loss = F.cross_entropy(value_logits, value_targets)
        loss = ((1.0 - HARD_CE_WEIGHT) * kl + HARD_CE_WEIGHT * hard_ce + VALUE_LOSS_WEIGHT * value_loss) / TRAIN_ACCUM
        loss.backward()

        loss_sum += loss.item() * TRAIN_ACCUM
        ce_sum += hard_ce.item()
        kl_sum += kl.item()
        val_sum += value_loss.item()
        micro += 1

        if micro >= TRAIN_ACCUM:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            micro = 0
            steps_done += 1
            if steps_done % 25 == 0:
                log(
                    f"    train step {steps_done}/{TRAIN_STEPS_PER_CYCLE}: "
                    f"loss={loss_sum / steps_done:.4f} ce={ce_sum / steps_done:.4f} "
                    f"kl={kl_sum / steps_done:.4f} val={val_sum / steps_done:.4f} gnorm={float(grad_norm):.2f}"
                )
            if steps_done >= TRAIN_STEPS_PER_CYCLE:
                break

    return {
        "steps": steps_done,
        "loss": loss_sum / max(steps_done, 1),
        "ce": ce_sum / max(steps_done, 1),
        "kl": kl_sum / max(steps_done, 1),
        "value": val_sum / max(steps_done, 1),
    }


def save_buffer(path, buffer_data):
    with open(path, "w", encoding="utf-8") as f:
        for item in buffer_data:
            f.write(json.dumps(item) + "\n")


def load_buffer(path):
    if not path.exists():
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_checkpoint(model, optimizer, cycle, best_score, history):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cycle": cycle,
        "best_score": best_score,
        "history": history,
    }
    torch.save(ckpt, OUTPUT_DIR / "latest.pt")


def main():
    global LOG_FILE
    random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = OUTPUT_DIR / "exp082.log"
    buffer_path = OUTPUT_DIR / "replay_buffer.jsonl"
    history_path = OUTPUT_DIR / "history.jsonl"

    log("=" * 72)
    log("exp082: Online SF game soft-label loop")
    log("=" * 72)
    log(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = load_model(CHECKPOINT_PATH, DEVICE)
    optimizer = AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=WEIGHT_DECAY)
    history = []
    best_score = -1.0
    start_cycle = 1

    latest = OUTPUT_DIR / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state.get("history", [])
        best_score = state.get("best_score", -1.0)
        start_cycle = int(state.get("cycle", 0)) + 1
        log(f"Resuming from cycle {start_cycle - 1}")

    replay_buffer = load_buffer(buffer_path)
    log(f"Replay buffer size: {len(replay_buffer)}")

    play_engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    play_engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB, "UCI_LimitStrength": True, "UCI_Elo": SF_PLAY_ELO})
    analysis_engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    analysis_engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB})

    try:
        for cycle in range(start_cycle, MAX_CYCLES + 1):
            log("\n" + "=" * 72)
            log(f"Cycle {cycle}")
            log("=" * 72)
            t0 = time.time()

            games = []
            all_fens = []
            for game_idx in range(GAMES_PER_CYCLE):
                opening = OPENINGS[(cycle + game_idx - 1) % len(OPENINGS)]
                model_white = game_idx % 2 == 0
                game = play_game_vs_sf(model, play_engine, model_white, opening)
                games.append(game)
                all_fens.extend(game["positions"])
                log(
                    f"  game {game_idx + 1}/{GAMES_PER_CYCLE}: score={game['score']:.1f} "
                    f"result={game['result']} plies={game['plies']} positions={len(game['positions'])}"
                )

            deduped_fens = list(dict.fromkeys(all_fens))
            log(f"  Collected {len(all_fens)} model positions, {len(deduped_fens)} unique")

            labeled = []
            for idx, fen in enumerate(deduped_fens, 1):
                labeled.append(label_position(chess.Board(fen), analysis_engine))
                if idx % 10 == 0 or idx == len(deduped_fens):
                    log(f"    labeled {idx}/{len(deduped_fens)} positions")

            cycle_json = OUTPUT_DIR / f"cycle_{cycle:04d}_labels.json"
            with open(cycle_json, "w", encoding="utf-8") as f:
                json.dump(labeled, f)

            replay_buffer.extend(labeled)
            if len(replay_buffer) > REPLAY_BUFFER_MAX:
                replay_buffer = replay_buffer[-REPLAY_BUFFER_MAX:]
            save_buffer(buffer_path, replay_buffer)

            log(f"  Training on {len(labeled)} current + {len(replay_buffer)} replay-buffer positions")
            train_stats = train_cycle(model, optimizer, labeled, replay_buffer[:-len(labeled)] if len(replay_buffer) > len(labeled) else [])

            avg_score = sum(g["score"] for g in games) / max(len(games), 1)
            cycle_summary = {
                "cycle": cycle,
                "avg_score": avg_score,
                "games": [{"score": g["score"], "result": g["result"], "plies": g["plies"]} for g in games],
                "positions_total": len(all_fens),
                "positions_unique": len(deduped_fens),
                "labeled_positions": len(labeled),
                "replay_buffer_size": len(replay_buffer),
                "train": train_stats,
                "elapsed_s": round(time.time() - t0, 1),
            }
            history.append(cycle_summary)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cycle_summary) + "\n")

            log(
                f"  cycle summary: score={avg_score:.3f} labeled={len(labeled)} "
                f"train_loss={train_stats['loss']:.4f} elapsed={cycle_summary['elapsed_s']:.1f}s"
            )

            if avg_score > best_score:
                best_score = avg_score
                torch.save({"model_state_dict": model.state_dict()}, OUTPUT_DIR / "best_model.pt")
                log(f"  New best model saved (avg_score={best_score:.3f})")
            save_checkpoint(model, optimizer, cycle, best_score, history)
            torch.save({"model_state_dict": model.state_dict()}, OUTPUT_DIR / "latest_model.pt")
    finally:
        play_engine.quit()
        analysis_engine.quit()


if __name__ == "__main__":
    main()
