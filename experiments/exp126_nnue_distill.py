"""exp126: NNUE distillation — Train a fast eval network from the 204M transformer.

Hypothesis: Distilling the transformer's policy+value knowledge into a 0.4M NNUE
network enables 18-60x faster NN evaluation → drastically more MCTS sims/second
→ higher ELO at fixed wall-clock time per move.

Procedure:
  1. Generate positions from diverse sources (opening + middlegame + endgame)
  2. Run teacher (204M transformer) on each position to get soft targets:
     - WDL value distribution (temperature-scaled)
     - Policy distribution over legal moves (temperature-scaled)
  3. Train NNUE student to match both distributions via KL divergence
  4. Evaluate: compare NNUE-MCTS vs Transformer-MCTS at matched wall-clock time

Expected outcome:
  If NNUE retains ~80% of teacher's policy quality, the 18x speedup means
  it can run 1800 sims in the same time as transformer runs 100 sims.
  At 100 sims the transformer achieves ~2091 ELO → NNUE at 1800 sims should
  exceed that even with somewhat weaker per-evaluation quality.

Hardware: RTX 4060 Laptop (8GB VRAM) — both teacher and student fit simultaneously.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids, batch_boards_to_planes
from chess_transformer_factory import build_model
from move_vocab import legal_move_mask
from nnue_model import (NNUEModel, NNUEDistiller, batch_boards_to_halfka_sparse,
                        count_parameters)

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_PATH = None


def log(msg):
    print(msg, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")


def find_checkpoint():
    candidates = [
        ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt",
        ROOT / "outputs" / "hf" / "chess-transformer-200m-latest" / "best_model.pt",
        ROOT / "outputs" / "hf_checkpoint" / "best_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download("avewright/chess-transformer-200m-latest",
                               "best_model.pt")
    except Exception:
        pass
    raise FileNotFoundError("Checkpoint not found")


def generate_random_positions(n, min_ply=4, max_ply=200):
    """Generate diverse positions by playing random games."""
    positions = []
    while len(positions) < n:
        board = chess.Board()
        target_ply = np.random.randint(min_ply, max_ply + 1)
        for _ in range(target_ply):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(np.random.choice(legal))
        if not board.is_game_over() and board.legal_moves.count() > 0:
            positions.append(board.copy())
    return positions[:n]


def generate_sf_game_positions(sf_path, n, sf_elo=1500, time_per_move=0.01):
    """Generate positions from Stockfish self-play games."""
    positions = []
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    while len(positions) < n:
        board = chess.Board()
        game_positions = []
        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 300:
            result = engine.play(board, chess.engine.Limit(time=time_per_move))
            board.push(result.move)
            if len(board.move_stack) >= 4 and not board.is_game_over():
                game_positions.append(board.copy())
        # Sample ~10 positions per game
        if game_positions:
            sample_n = min(10, len(game_positions))
            sampled = np.random.choice(len(game_positions), sample_n, replace=False)
            for idx in sampled:
                positions.append(game_positions[idx])
        if len(positions) % 100 < 11:
            log(f"  Generated {len(positions)}/{n} positions...")

    engine.quit()
    return positions[:n]


def load_existing_positions(max_positions=50000):
    """Load positions from existing harvest datasets."""
    positions = []

    # Check for parquet/jsonl files in outputs
    harvest_dirs = [
        ROOT / "outputs" / "exp085_parallel_multipv_harvest",
        ROOT / "outputs" / "exp087_full_legal_harvest",
        ROOT / "outputs" / "exp095_endgame_harvest",
        ROOT / "outputs" / "exp099_middlegame_harvest",
    ]

    for d in harvest_dirs:
        dataset_dir = d / "dataset"
        if not dataset_dir.exists():
            dataset_dir = d
        # Try JSONL files first (more likely to exist)
        for f in sorted(dataset_dir.glob("*.jsonl")):
            try:
                import json as _json
                with open(f, "r") as fh:
                    for line in fh:
                        try:
                            row = _json.loads(line)
                            fen = row.get("fen")
                            if fen:
                                board = chess.Board(fen)
                                if not board.is_game_over() and board.legal_moves.count() > 0:
                                    positions.append(board)
                        except Exception:
                            pass
                        if len(positions) >= max_positions:
                            return positions
            except Exception:
                pass
        # Try parquet files
        for f in sorted(dataset_dir.glob("*.parquet")):
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(str(f), columns=["fen"])
                for fen in table["fen"].to_pylist():
                    try:
                        board = chess.Board(fen)
                        if not board.is_game_over() and board.legal_moves.count() > 0:
                            positions.append(board)
                    except Exception:
                        pass
                    if len(positions) >= max_positions:
                        return positions
            except Exception:
                pass

    return positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num-positions", type=int, default=50000,
                    help="Number of positions for distillation")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--accumulator-size", type=int, default=512)
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: 5000 positions, 2 epochs")
    args = ap.parse_args()

    if args.quick:
        args.num_positions = 5000
        args.epochs = 2

    global LOG_PATH
    out_dir = ROOT / "outputs" / "exp126_nnue_distill"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = out_dir / "training.log"
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    # Load teacher
    ckpt_path = args.checkpoint or find_checkpoint()
    log(f"Loading teacher from {ckpt_path}...")
    teacher = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    teacher.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in state.items()})
    teacher = teacher.to(DEVICE)
    teacher.eval()
    log(f"Teacher loaded: 204M params")

    # Create student
    student = NNUEModel(
        accumulator_size=args.accumulator_size,
        hidden1=32, hidden2=32, policy_channels=32,
    ).to(DEVICE)
    log(f"Student created: {count_parameters(student)/1e6:.2f}M params")

    # Generate or load training positions
    log(f"Loading positions...")
    positions = load_existing_positions(max_positions=args.num_positions)
    if len(positions) < args.num_positions:
        log(f"  Loaded {len(positions)} from existing data. "
            f"Generating {args.num_positions - len(positions)} random...")
        extra = generate_random_positions(
            args.num_positions - len(positions), min_ply=4, max_ply=200)
        positions.extend(extra)
    log(f"Total positions: {len(positions)}")

    # Split train/val
    np.random.shuffle(positions)
    val_size = min(2000, len(positions) // 10)
    val_positions = positions[:val_size]
    train_positions = positions[val_size:]
    log(f"Train: {len(train_positions)}, Val: {val_size}")

    # Create distiller
    distiller = NNUEDistiller(
        teacher, student, DEVICE,
        lr=args.lr,
        value_weight=args.value_weight,
        policy_weight=args.policy_weight,
        temperature=args.temperature,
    )

    # Training loop
    best_val_loss = float('inf')
    batch_size = args.batch_size
    config_info = {
        "num_positions": len(positions),
        "batch_size": batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "temperature": args.temperature,
        "accumulator_size": args.accumulator_size,
        "student_params": count_parameters(student),
    }

    log(f"\n{'=' * 60}")
    log(f"NNUE Distillation Training")
    log(f"{'=' * 60}")
    log(f"Config: {json.dumps(config_info, indent=2)}")

    for epoch in range(args.epochs):
        # Shuffle training data
        np.random.shuffle(train_positions)
        student.train()

        epoch_losses = {"total": 0, "value": 0, "policy": 0}
        n_batches = 0

        for start in range(0, len(train_positions), batch_size):
            batch = train_positions[start:start + batch_size]
            if len(batch) < 2:
                continue

            losses = distiller.train_step(batch)
            for k in epoch_losses:
                epoch_losses[k] += losses[k]
            n_batches += 1

            if n_batches % 50 == 0:
                avg_total = epoch_losses["total"] / n_batches
                avg_val = epoch_losses["value"] / n_batches
                avg_pol = epoch_losses["policy"] / n_batches
                log(f"  E{epoch + 1} batch {n_batches}: "
                    f"loss={avg_total:.4f} (v={avg_val:.4f} p={avg_pol:.4f})")

        # Validation
        student.eval()
        val_losses = {"total": 0, "value": 0, "policy": 0}
        val_batches = 0
        with torch.no_grad():
            for start in range(0, len(val_positions), batch_size):
                batch = val_positions[start:start + batch_size]
                if len(batch) < 2:
                    continue

                # Get teacher targets
                value_targets, policy_targets = distiller.generate_targets(batch)

                # Student predictions
                halfka = batch_boards_to_halfka_sparse(batch, DEVICE)
                planes = batch_boards_to_planes(batch).to(DEVICE)
                student_out = student(halfka, planes)

                T = args.temperature
                s_val_log = F.log_softmax(
                    student_out["value_logits"] / T, dim=-1)
                v_loss = F.kl_div(s_val_log, value_targets,
                                  reduction='batchmean') * (T * T)

                s_pol_log = F.log_softmax(
                    student_out["policy_logits"] / T, dim=-1)
                p_loss = F.kl_div(s_pol_log, policy_targets,
                                  reduction='batchmean') * (T * T)

                total = args.value_weight * v_loss + args.policy_weight * p_loss
                val_losses["total"] += total.item()
                val_losses["value"] += v_loss.item()
                val_losses["policy"] += p_loss.item()
                val_batches += 1

        avg_val_loss = val_losses["total"] / max(1, val_batches)
        avg_val_v = val_losses["value"] / max(1, val_batches)
        avg_val_p = val_losses["policy"] / max(1, val_batches)

        log(f"\n  Epoch {epoch + 1}/{args.epochs}: "
            f"val_loss={avg_val_loss:.4f} (v={avg_val_v:.4f} p={avg_val_p:.4f})")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": student.state_dict(),
                "config": config_info,
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
            }, out_dir / "best_nnue.pt")
            log(f"  ** New best! Saved to {out_dir / 'best_nnue.pt'}")

        # Save latest
        torch.save({
            "model_state_dict": student.state_dict(),
            "config": config_info,
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
        }, out_dir / "latest_nnue.pt")

    log(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    log(f"Model saved to {out_dir}")


if __name__ == "__main__":
    main()
