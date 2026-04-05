"""exp147: Expert Iteration — MCTS visit distributions as improved policy targets.

Core idea from AlphaZero: MCTS search acts as a policy improvement operator.
The visit distribution after MCTS search is a BETTER policy than the raw network.
Training on visit distributions bootstraps the model to play at MCTS-level without search.

Loop:
  1. Play games using MCTS (c_puct=2.5, 100 sims, FP16)
  2. At each model move, record {board, visit_distribution, value}
  3. Fine-tune on visit distributions (soft cross-entropy) + value (WDL KL)
  4. Test ELO
  5. Repeat

Key design choices:
  - Soft policy targets: normalize visit counts to probabilities
  - Temperature: apply temperature before softmax (higher T → more entropy, regularization)
  - Conservative LR: 1e-5, warmup, cosine decay (avoid catastrophic forgetting)
  - Value target: MCTS root Q-value converted to WDL (not game outcome — too noisy)
  - Mix with original data (50% expert iteration + 50% original SF labels)

Phase 1 (this script): Generate 50 self-play games, extract ~5K positions, fine-tune 1 epoch
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from chess_transformer_factory import build_model
from move_vocab import VOCAB_SIZE, move_to_index, legal_move_mask
from opening_book import get_book_move
from uci_engine import MCTSSearch, SyzygyProbe

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = None


def log(msg):
    print(msg, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a") as f:
            f.write(msg + "\n")


def resolve_sf():
    for p in [
        Path(os.environ.get("STOCKFISH_PATH", "")),
        Path(shutil.which("stockfish") or ""),
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        ROOT / "stockfish" / "stockfish" / "stockfish-ubuntu-x86-64-avx2",
    ]:
        if p and p.exists() and p.is_file():
            return p
    raise FileNotFoundError("Stockfish not found")


OPENINGS = [
    [],
    ["e2e4", "e7e5"],
    ["d2d4", "d7d5"],
    ["e2e4", "c7c5"],
    ["d2d4", "g8f6"],
    ["e2e4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4"],
    ["e2e4", "c7c5", "g1f3"],
    ["d2d4", "g8f6", "c2c4"],
]


# ── Phase 1: Generate expert iteration data ──

def generate_expert_data(model, mcts, sf_elo, n_games, sims,
                         visit_temp=1.0, ply_cap=300):
    """Play games vs Stockfish, recording MCTS visit distributions.

    Returns list of position dicts with:
        - board: chess.Board (position before model's move)
        - visit_dist: dict {move_idx: probability} from MCTS visit counts
        - root_q: float, MCTS root Q-value (side-to-move perspective)
        - chosen_move: the move actually played
    """
    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    positions = []
    game_results = []

    for game_i in range(n_games):
        board = chess.Board()
        op = OPENINGS[game_i % len(OPENINGS)]
        for uci in op:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)

        mcts.new_game()
        model_color = chess.WHITE if game_i % 2 == 0 else chess.BLACK
        game_positions = []

        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
            if board.turn == model_color:
                # Check tablebase and opening book first
                tb = mcts.syzygy.get_move(board)
                if tb is not None:
                    board.push(tb)
                    mcts.new_game()
                    continue
                bm = get_book_move(board)
                if bm is not None:
                    board.push(bm)
                    mcts.new_game()
                    continue

                # Run MCTS search
                move, info = mcts.search(board, max_sims=sims)

                # Extract visit distribution from MCTS root
                root = mcts.root
                if root and root.children:
                    total_visits = sum(c.visit_count for c in root.children.values())
                    if total_visits > 0:
                        # Compute visit probability distribution
                        visit_dist = {}
                        for m_key, child in root.children.items():
                            if child.visit_count > 0:
                                idx = move_to_index(m_key)
                                # Apply temperature
                                visit_dist[idx] = child.visit_count

                        # Apply temperature and normalize
                        if visit_temp != 1.0:
                            max_v = max(visit_dist.values())
                            visit_dist = {k: (v / max_v) ** (1.0 / visit_temp)
                                          for k, v in visit_dist.items()}
                        total = sum(visit_dist.values())
                        visit_dist = {k: v / total for k, v in visit_dist.items()}

                        # Root Q-value (side-to-move perspective)
                        root_q = root.q_value()

                        # Store position data
                        game_positions.append({
                            "fen": board.fen(),
                            "visit_dist": visit_dist,
                            "root_q": root_q,
                            "chosen_move": move_to_index(move),
                        })

                board.push(move)
                mcts.new_game()
            else:
                sf_move = engine.play(board, chess.engine.Limit(time=0.05)).move
                if sf_move not in board.legal_moves:
                    sf_move = next(iter(board.legal_moves))
                board.push(sf_move)

        # Game result
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            result = 0.5
        elif outcome.winner == model_color:
            result = 1.0
        else:
            result = 0.0

        positions.extend(game_positions)
        game_results.append(result)
        score = sum(game_results) / len(game_results)
        log(f"  Game {game_i+1}/{n_games}: {'W' if model_color == chess.WHITE else 'B'} "
            f"{'WIN' if result == 1 else ('DRAW' if result == 0.5 else 'LOSS')} "
            f"({len(board.move_stack)}ply) | {len(game_positions)} positions | "
            f"total={len(positions)} | score={score:.3f}")

    engine.quit()
    return positions, game_results


# ── Phase 2: Train on expert data ──

def q_to_wdl(q, sharpness=4.0):
    """Convert Q-value (side-to-move) to WDL probability.

    Q is in [-1, 1]. Use a sigmoid-based model:
    - q > 0 → more W, q < 0 → more L, q ≈ 0 → more D
    """
    # Scale q by sharpness, then convert to WDL
    w = 1.0 / (1.0 + math.exp(-sharpness * q))
    l = 1.0 - w
    # Draw probability from uncertainty
    d = max(0, 0.5 - abs(w - 0.5)) * 2
    w = w * (1 - d * 0.5)
    l = l * (1 - d * 0.5)
    total = w + d + l
    return [w / total, d / total, l / total]


def train_on_expert_data(model, positions, lr=1e-5, batch_size=32,
                         epochs=1, value_weight=0.5, original_data_frac=0.0,
                         original_shard_path=None):
    """Fine-tune model on MCTS visit distributions.

    Loss = KL(visit_dist || model_policy) + value_weight * KL(wdl_target || model_wdl)
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Convert positions to tensors
    # We need FEN → token_ids (fused encoder input)
    from chess_features import board_to_fused_token_ids

    # Build training data
    train_data = []
    for pos in positions:
        board = chess.Board(pos["fen"])
        token_ids = board_to_fused_token_ids(board)  # dict of tensors
        visit_target = torch.zeros(VOCAB_SIZE)
        for idx, prob in pos["visit_dist"].items():
            visit_target[idx] = prob
        wdl_target = torch.tensor(q_to_wdl(pos["root_q"]))
        train_data.append({
            "token_ids": token_ids,
            "visit_target": visit_target,
            "wdl_target": wdl_target,
            "fen": pos["fen"],
        })

    log(f"  Training on {len(train_data)} positions, {epochs} epochs, "
        f"lr={lr}, batch={batch_size}")

    total_loss_sum = 0.0
    n_batches = 0

    for epoch in range(epochs):
        # Shuffle
        indices = list(range(len(train_data)))
        import random
        random.shuffle(indices)

        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_batches = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch = [train_data[i] for i in batch_idx]
            if not batch:
                continue

            # Stack token_ids
            boards = [chess.Board(b["fen"]) for b in batch]
            inp = batch_boards_to_fused_token_ids(boards, DEVICE)

            visit_targets = torch.stack([b["visit_target"] for b in batch]).to(DEVICE)
            wdl_targets = torch.stack([b["wdl_target"] for b in batch]).to(DEVICE)

            out = model(inp)

            # Policy loss: KL divergence of visit distribution
            policy_logits = out["policy_logits"]

            # Mask illegal moves
            for i, board in enumerate(boards):
                mask = legal_move_mask(board).to(DEVICE)
                policy_logits[i][~mask] = -1e9  # Finite value, not -inf

            log_probs = F.log_softmax(policy_logits, dim=-1)
            # Manual cross-entropy: avoid 0 * -inf = NaN from F.kl_div
            # KL(target || model) ≈ -sum(target * log(model)) + const
            # Only sum where visit_targets > 0
            nonzero = visit_targets > 0
            ce_terms = torch.where(nonzero, visit_targets * log_probs,
                                   torch.zeros_like(log_probs))
            policy_loss = -ce_terms.sum(dim=-1).mean()

            # Value loss: KL divergence of WDL
            value_logits = out["value_logits"]
            value_log_probs = F.log_softmax(value_logits, dim=-1)
            value_loss = F.kl_div(value_log_probs, wdl_targets, reduction="batchmean")

            total_loss = policy_loss + value_weight * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_batches += 1
            n_batches += 1

            if epoch_batches % 20 == 0:
                avg_p = epoch_policy_loss / epoch_batches
                avg_v = epoch_value_loss / epoch_batches
                log(f"    E{epoch+1} batch {epoch_batches}: "
                    f"loss={epoch_loss/epoch_batches:.4f} "
                    f"(p={avg_p:.4f} v={avg_v:.4f})")

        avg_loss = epoch_loss / max(1, epoch_batches)
        avg_p = epoch_policy_loss / max(1, epoch_batches)
        avg_v = epoch_value_loss / max(1, epoch_batches)
        log(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
            f"(p={avg_p:.4f} v={avg_v:.4f})")
        total_loss_sum += epoch_loss

    model.eval()
    return total_loss_sum / max(1, n_batches)


# ── Phase 3: Evaluation ──

def evaluate_elo(model, sf_elo, n_games, sims, label):
    """Quick ELO evaluation."""
    from experiments.exp146_cpuct_sweep import play_game, wilson_ci, elo_diff

    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    syzygy = SyzygyProbe()
    mcts = MCTSSearch(
        model, DEVICE, syzygy,
        c_puct=2.5, batch_size=8,
        fpu_reduction=0.25,
        root_noise_alpha=0.3, root_noise_frac=0.0,
        use_fp16=True,
    )

    results = []
    total = 0.0

    log(f"\n{'='*60}")
    log(f"EVAL: {label} vs SF{sf_elo} ({n_games}g, {sims} sims)")
    log(f"{'='*60}")

    for i in range(n_games):
        op = OPENINGS[i % len(OPENINGS)]
        mc = chess.WHITE if i % 2 == 0 else chess.BLACK

        board = chess.Board()
        for uci in op:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)

        mcts.new_game()
        t0 = time.time()

        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 300:
            if board.turn == mc:
                tb = mcts.syzygy.get_move(board)
                if tb is not None:
                    board.push(tb)
                    mcts.new_game()
                    continue
                bm = get_book_move(board)
                if bm is not None:
                    board.push(bm)
                    mcts.new_game()
                    continue
                move, info = mcts.search(board, max_sims=sims)
                board.push(move)
                mcts.new_game()
            else:
                sf_move = engine.play(board, chess.engine.Limit(time=0.05)).move
                if sf_move not in board.legal_moves:
                    sf_move = next(iter(board.legal_moves))
                board.push(sf_move)

        el = time.time() - t0
        o = board.outcome(claim_draw=True)
        if o is None or o.winner is None:
            sc = 0.5
        elif o.winner == mc:
            sc = 1.0
        else:
            sc = 0.0

        results.append(sc)
        total += sc
        w = sum(1 for x in results if x == 1.0)
        d = sum(1 for x in results if x == 0.5)
        l = sum(1 for x in results if x == 0.0)
        score = total / len(results)
        ci = wilson_ci(total, len(results))
        rs = "WIN" if sc == 1 else ("DRAW" if sc == 0.5 else "LOSS")
        log(f"  G{i+1:>3}/{n_games}: {'W' if mc == chess.WHITE else 'B'} {rs} "
            f"({len(board.move_stack)}ply {el:.0f}s) | "
            f"{score:.3f} ({w}W-{d}D-{l}L)")

    engine.quit()
    sc = total / n_games
    w = sum(1 for x in results if x == 1.0)
    d = sum(1 for x in results if x == 0.5)
    l = sum(1 for x in results if x == 0.0)
    ci = wilson_ci(total, n_games)
    ed = elo_diff(sc)
    log(f"\n  FINAL: {sc:.3f} ({w}W-{d}D-{l}L) CI=[{ci[0]:.3f},{ci[1]:.3f}] "
        f"ELO~{sf_elo + ed:.0f}")
    return {"score": sc, "w": w, "d": d, "l": l, "est_elo": round(sf_elo + ed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--gen-games", type=int, default=50,
                    help="Games to play for data generation")
    ap.add_argument("--gen-sims", type=int, default=100,
                    help="MCTS sims per move during generation")
    ap.add_argument("--train-epochs", type=int, default=1)
    ap.add_argument("--train-lr", type=float, default=1e-5)
    ap.add_argument("--train-batch", type=int, default=32)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--eval-sims", type=int, default=100)
    ap.add_argument("--quick", action="store_true",
                    help="Quick test: 20 gen games, 8 eval games")
    ap.add_argument("--iterations", type=int, default=1,
                    help="Number of expert iteration loops")
    ap.add_argument("--visit-temp", type=float, default=1.0,
                    help="Temperature for visit count normalization")
    args = ap.parse_args()

    if args.quick:
        args.gen_games = 20
        args.eval_games = 8

    global LOG_PATH
    out_dir = ROOT / "outputs" / "exp147_expert_iter"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = out_dir / "training.log"
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    syzygy = SyzygyProbe()
    log(f"Device: {DEVICE}")
    log(f"Syzygy: {'available' if syzygy.available else 'not found'}")

    # Load model
    ckpt_path = args.checkpoint or str(
        ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt")
    log(f"Loading model from {ckpt_path}...")
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()
    log("Model loaded (204M params)")

    all_iter_results = []

    for iteration in range(args.iterations):
        log(f"\n{'#'*60}")
        log(f"EXPERT ITERATION {iteration+1}/{args.iterations}")
        log(f"{'#'*60}")

        # Phase 1: Generate expert data (or load cached)
        pos_path = out_dir / f"positions_iter{iteration+1}.json"
        if pos_path.exists() and iteration == 0:
            log(f"\n--- Phase 1: Loading cached positions from {pos_path} ---")
            with open(pos_path) as f:
                serializable = json.load(f)
            positions = []
            for p in serializable:
                positions.append({
                    "fen": p["fen"],
                    "visit_dist": {int(k): v for k, v in p["visit_dist"].items()},
                    "root_q": p["root_q"],
                    "chosen_move": p["chosen_move"],
                })
            gen_score = 0.0  # Unknown from cache
            log(f"  Loaded {len(positions)} positions")
        else:
            log(f"\n--- Phase 1: Generate {args.gen_games} games "
                f"({args.gen_sims} sims/move) vs SF{args.sf_elo} ---")

            mcts = MCTSSearch(
                model, DEVICE, syzygy,
                c_puct=2.5, batch_size=8,
                fpu_reduction=0.25,
                root_noise_alpha=0.3, root_noise_frac=0.25,
                use_fp16=True,
            )

            t0 = time.time()
            positions, game_results = generate_expert_data(
                model, mcts, args.sf_elo, args.gen_games,
                args.gen_sims, visit_temp=args.visit_temp)
            gen_time = time.time() - t0

            gen_score = sum(game_results) / len(game_results)
            log(f"\n  Generated {len(positions)} positions from {args.gen_games} games "
                f"in {gen_time:.0f}s")
            log(f"  Generation score: {gen_score:.3f}")

            # Save positions
            serializable = []
            for p in positions:
                serializable.append({
                    "fen": p["fen"],
                    "visit_dist": {str(k): v for k, v in p["visit_dist"].items()},
                    "root_q": p["root_q"],
                    "chosen_move": p["chosen_move"],
                })
            with open(pos_path, "w") as f:
                json.dump(serializable, f)
            log(f"  Saved to {pos_path}")

        # Phase 2: Train
        log(f"\n--- Phase 2: Train on {len(positions)} expert positions ---")
        t0 = time.time()
        avg_loss = train_on_expert_data(
            model, positions,
            lr=args.train_lr,
            batch_size=args.train_batch,
            epochs=args.train_epochs,
        )
        train_time = time.time() - t0
        log(f"  Training complete in {train_time:.0f}s, avg_loss={avg_loss:.4f}")

        # Save checkpoint
        ckpt_path_out = out_dir / f"model_iter{iteration+1}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "iteration": iteration + 1,
            "gen_score": gen_score,
            "avg_loss": avg_loss,
            "n_positions": len(positions),
        }, ckpt_path_out)
        log(f"  Saved checkpoint to {ckpt_path_out}")

        # Phase 3: Evaluate
        log(f"\n--- Phase 3: Evaluate ({args.eval_games} games, "
            f"{args.eval_sims} sims) ---")
        model.eval()
        eval_result = evaluate_elo(
            model, args.sf_elo, args.eval_games, args.eval_sims,
            label=f"iter{iteration+1}")

        iter_result = {
            "iteration": iteration + 1,
            "gen_games": args.gen_games,
            "gen_positions": len(positions),
            "gen_score": gen_score,
            "train_loss": avg_loss,
            "eval_result": eval_result,
        }
        all_iter_results.append(iter_result)

        log(f"\n  Iteration {iteration+1} ELO: ~{eval_result['est_elo']}")

    # Final summary
    log(f"\n{'='*60}")
    log("EXPERT ITERATION SUMMARY")
    log(f"{'='*60}")
    log(f"{'Iter':>4} {'GenScore':>9} {'Positions':>10} {'TrainLoss':>10} "
        f"{'ELO':>6} {'Score':>7}")
    log("-" * 60)
    for r in all_iter_results:
        log(f"{r['iteration']:>4} {r['gen_score']:>9.3f} "
            f"{r['gen_positions']:>10} {r['train_loss']:>10.4f} "
            f"{r['eval_result']['est_elo']:>6} "
            f"{r['eval_result']['score']:>7.3f}")

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_iter_results, f, indent=2)
    log(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
