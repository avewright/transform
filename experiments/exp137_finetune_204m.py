"""exp137: Fine-tune existing 204M checkpoint on 500K lichess-sf positions.

Strategy: Load existing best_model.pt (trained on 224K exp085 data),
continue training on 500K lichess-sf positions with lower LR.
"""
import argparse
import gc
import json
import math
import os
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_transformer_factory import build_model, count_parameters
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move
from data_loader import load_training_data, get_batch_input, get_eval_batch_input, compute_wdl, compute_phase
from uci_engine import MCTSSearch, SyzygyProbe

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparams — conservative for fine-tuning
BATCH_SIZE = 16
ACCUM_STEPS = 4  # eff_batch = 64
LR = 5e-5  # lower LR for fine-tuning (was 2e-4 for exp085)
WARMUP_FRAC = 0.01
MIN_LR_FRAC = 0.1
GRAD_CLIP = 0.3
WEIGHT_DECAY = 0.01
VALUE_WEIGHT = 0.5
SEED = 42

LOG_INTERVAL = 10
EVAL_INTERVAL = 50
SAVE_INTERVAL = 50

OPENINGS = [
    [], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["e2e4", "c7c5"],
    ["d2d4", "g8f6"], ["e2e4", "e7e6"], ["c2c4", "e7e5"], ["g1f3", "d7d5"],
]

SHUTDOWN = False
def _sig(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    print(f"\n[SIGNAL] Shutdown...", flush=True)
signal.signal(signal.SIGINT, _sig)

LOG_FILE = None
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


def find_checkpoint():
    """Find best existing 204M checkpoint."""
    candidates = [
        ROOT / "outputs" / "hf" / "chess-transformer-200m-latest" / "best_model.pt",
        ROOT / "outputs" / "hf_checkpoint" / "best_model.pt",
        ROOT / "outputs" / "exp100_diverse_training" / "best_model.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # HF cache
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download("avewright/chess-transformer-200m-latest", "best_model.pt"))
    except Exception:
        pass
    raise FileNotFoundError("No 204M checkpoint found")


def evaluate(model, eval_data, eval_tensors, device, batch_size=64):
    model.eval()
    correct = top3_correct = total = val_correct = val_total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            chunk = eval_data[i:i + batch_size]
            n = len(chunk)
            batch_input = get_eval_batch_input(eval_tensors, slice(i, i + n), "fused", device)
            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
            logits = result["policy_logits"].float()
            wdl_logits = result["value_logits"].float()
            for j, d in enumerate(chunk):
                l = logits[j].clone()
                mask = legal_move_mask(d["board"]).to(device)
                l[~mask] = float("-inf")
                pred_idx = l.argmax().item()
                true_idx = move_to_index(d["move"])
                if pred_idx == true_idx:
                    correct += 1
                topk = l.topk(min(3, l.shape[0])).indices.tolist()
                if true_idx in topk:
                    top3_correct += 1
                total += 1
                pred_class = wdl_logits[j].argmax().item()
                true_wdl = d["wdl"]
                true_class = max(range(3), key=lambda k: true_wdl[k])
                if pred_class == true_class:
                    val_correct += 1
                val_total += 1
    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "top3": top3_correct / max(total, 1),
        "value_acc": val_correct / max(val_total, 1),
        "n": total,
    }


def wilson_ci(s, n, z=1.96):
    if n <= 0: return 0.0, 1.0
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0, c - m), min(1, c + m)

def elo_diff(score):
    if score <= 0: return -400
    if score >= 1: return 400
    return -400 * math.log10(1 / score - 1)

def resolve_sf():
    for p in [
        Path(os.environ.get("STOCKFISH_PATH", "")),
        Path(shutil.which("stockfish") or ""),
        ROOT / "stockfish" / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
    ]:
        if p and p.exists() and p.is_file():
            return p
    raise FileNotFoundError("Stockfish not found")

def play_game(engine, model, mcts, sf_elo, model_color, opening, sims=100, ply_cap=300):
    board = chess.Board()
    for uci in opening:
        m = chess.Move.from_uci(uci)
        if m in board.legal_moves:
            board.push(m)
    mcts.new_game()
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < ply_cap:
        if board.turn == model_color:
            tb = mcts.syzygy.get_move(board)
            if tb:
                move = tb
            else:
                move, _ = mcts.search(board, max_sims=sims)
            mcts.new_game()
            board.push(move)
        else:
            sf_move = engine.play(board, chess.engine.Limit(time=0.05)).move
            if sf_move not in board.legal_moves:
                sf_move = next(iter(board.legal_moves))
            board.push(sf_move)
    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None: return 0.5
    return 1.0 if o.winner == model_color else 0.0

def run_elo_gauntlet(model, label, sf_elo, n_games, sims=100):
    syzygy = SyzygyProbe()
    mcts = MCTSSearch(model, DEVICE, syzygy, c_puct=2.5, batch_size=8,
                      root_noise_alpha=0.3, root_noise_frac=0.0, use_fp16=True)
    sf = resolve_sf()
    engine = chess.engine.SimpleEngine.popen_uci(str(sf))
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})
    total_score = 0.0
    for gi in range(n_games):
        color = chess.WHITE if gi % 2 == 0 else chess.BLACK
        opening = OPENINGS[gi % len(OPENINGS)]
        score = play_game(engine, model, mcts, sf_elo, color, opening, sims=sims)
        total_score += score
        avg = total_score / (gi + 1)
        lo, hi = wilson_ci(total_score, gi + 1)
        tag = "W" if score == 1.0 else ("D" if score == 0.5 else "L")
        log(f"  G{gi+1}/{n_games}: {tag} | {avg:.3f} [{lo:.3f},{hi:.3f}] ~{sf_elo + elo_diff(avg):.0f}")
    engine.quit()
    return {"score": total_score / n_games, "elo": sf_elo + elo_diff(total_score / n_games),
            "ci": list(wilson_ci(total_score, n_games)), "n": n_games}


def main():
    global LOG_FILE, SHUTDOWN

    ap = argparse.ArgumentParser(description="exp137: Fine-tune 204M on more data")
    ap.add_argument("--n-train", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--elo", action="store_true")
    ap.add_argument("--elo-games", type=int, default=16)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--eval-only", action="store_true", help="Just evaluate existing checkpoint")
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "exp137_finetune_204m"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(out_dir / "exp137.log", "w")

    log("=" * 60)
    log("exp137: Fine-tune 204M on lichess-sf data")
    log(f"  device: {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f}GB)")

    # Load checkpoint
    ckpt_path = find_checkpoint()
    log(f"  Checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    
    # Build model (default 204M config)
    model = build_model(None)
    n_params = count_parameters(model)
    log(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Load weights
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    log("  Loaded pretrained weights")

    # Load data
    log("Loading training data...")
    train_tensors, eval_data, eval_tensors = load_training_data(
        n_train=args.n_train, n_eval=2500, encoder_type="fused",
        min_depth=15, seed=SEED,
    )
    n_train = train_tensors["move_idx"].shape[0]
    log(f"  {n_train:,} train, {len(eval_data):,} eval")

    # Baseline eval
    base_metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  BASELINE: acc={base_metrics['accuracy']:.4f} top3={base_metrics['top3']:.4f} "
        f"val={base_metrics['value_acc']:.4f}")

    if args.eval_only:
        if args.elo:
            model.eval()
            elo_result = run_elo_gauntlet(model, "base", args.sf_elo, args.elo_games)
            log(f"  BASE ELO: score={elo_result['score']:.3f} elo={elo_result['elo']:.0f}")
        LOG_FILE.close()
        return

    # Training
    model.train()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda') if DEVICE.type == "cuda" else None

    eff_batch = BATCH_SIZE * ACCUM_STEPS
    steps_per_epoch = n_train // eff_batch
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * WARMUP_FRAC)
    log(f"  Batch: {BATCH_SIZE}x{ACCUM_STEPS}={eff_batch}, LR: {LR}")
    log(f"  {steps_per_epoch:,} steps/epoch, {total_steps:,} total, warmup={warmup_steps}")

    optimizer.zero_grad()
    global_step = 0
    accum_count = 0
    running_ce = running_val = running_gnorm = 0.0
    log_count = 0
    best_acc = base_metrics['accuracy']
    nan_count = 0
    t_start = time.time()

    log("  Training...")
    for epoch in range(args.epochs):
        perm = torch.randperm(n_train)
        for batch_start in range(0, n_train, BATCH_SIZE):
            if SHUTDOWN or (args.max_steps and global_step >= args.max_steps):
                break
            batch_end = min(batch_start + BATCH_SIZE, n_train)
            if batch_end - batch_start < BATCH_SIZE:
                break
            idx = perm[batch_start:batch_end]

            batch_input = get_batch_input(train_tensors, idx, "fused", DEVICE)
            move_targets = train_tensors["move_idx"][idx].to(DEVICE)
            wdl_targets = train_tensors["wdl"][idx].to(DEVICE)

            with autocast('cuda', dtype=torch.float16):
                result = model(batch_input)
                ce_loss = F.cross_entropy(result["policy_logits"], move_targets)
                value_loss = F.cross_entropy(result["value_logits"], wdl_targets)
                total_loss = ce_loss + VALUE_WEIGHT * value_loss
                scaled_loss = total_loss / ACCUM_STEPS

            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                nan_count += 1
                if nan_count > 20:
                    log(f"  ERROR: Too many NaN ({nan_count})")
                    break
                optimizer.zero_grad()
                accum_count = 0
                continue
            nan_count = 0

            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_count += 1
            running_ce += ce_loss.item()
            running_val += value_loss.item()

            if accum_count >= ACCUM_STEPS:
                # LR schedule
                progress = global_step / max(total_steps, 1)
                if global_step < warmup_steps:
                    lr_now = LR * (global_step + 1) / max(warmup_steps, 1)
                else:
                    p = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                    lr_now = LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * p)))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

                if scaler:
                    scaler.unscale_(optimizer)
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                running_gnorm += gnorm if not (math.isnan(gnorm) or math.isinf(gnorm)) else 0.0
                log_count += 1
                global_step += 1
                accum_count = 0

                if global_step % LOG_INTERVAL == 0 and log_count > 0:
                    avg_ce = running_ce / (log_count * ACCUM_STEPS)
                    avg_val = running_val / (log_count * ACCUM_STEPS)
                    avg_gn = running_gnorm / log_count
                    elapsed = time.time() - t_start
                    pos_s = (global_step * eff_batch) / max(elapsed, 1)
                    log(f"  step={global_step:,} e{epoch} ce={avg_ce:.4f} val={avg_val:.4f} "
                        f"gn={avg_gn:.2f} lr={lr_now:.2e} {pos_s:.0f}pos/s")
                    running_ce = running_val = running_gnorm = 0.0
                    log_count = 0

                if global_step % EVAL_INTERVAL == 0:
                    metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
                    log(f"  EVAL step={global_step}: acc={metrics['accuracy']:.4f} "
                        f"top3={metrics['top3']:.4f} val={metrics['value_acc']:.4f}")
                    if metrics['accuracy'] > best_acc:
                        best_acc = metrics['accuracy']
                        torch.save({"model_state_dict": model.state_dict()},
                                   out_dir / "best_model.pt")
                        log(f"  NEW BEST: {best_acc:.4f}")

                if global_step % SAVE_INTERVAL == 0:
                    torch.save({"model_state_dict": model.state_dict()},
                               out_dir / "latest_model.pt")

        if SHUTDOWN or (args.max_steps and global_step >= args.max_steps):
            break
        log(f"  Epoch {epoch} done")

    elapsed = time.time() - t_start
    total_pos = global_step * eff_batch
    log(f"\n  Done: {global_step:,} steps, {total_pos:,} pos in {elapsed/60:.1f}min ({total_pos/max(elapsed,1):.0f} pos/s)")

    final_metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  FINAL: acc={final_metrics['accuracy']:.4f} top3={final_metrics['top3']:.4f} "
        f"val={final_metrics['value_acc']:.4f}")
    torch.save({"model_state_dict": model.state_dict()}, out_dir / "final_model.pt")

    # Compare
    log(f"\n  IMPROVEMENT: acc {base_metrics['accuracy']:.4f} → {final_metrics['accuracy']:.4f} "
        f"({final_metrics['accuracy'] - base_metrics['accuracy']:+.4f})")

    if args.elo:
        log(f"\n  ELO gauntlet ({args.elo_games}g vs SF{args.sf_elo})...")
        best_path = out_dir / "best_model.pt"
        if best_path.exists():
            ckpt2 = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt2.get("model_state_dict", ckpt2))
        model.to(DEVICE).eval()
        elo_result = run_elo_gauntlet(model, "FT", args.sf_elo, args.elo_games)
        log(f"  ELO: score={elo_result['score']:.3f} elo={elo_result['elo']:.0f} ci={elo_result['ci']}")

    results = {
        "base_metrics": base_metrics, "final_metrics": final_metrics,
        "best_acc": best_acc, "global_step": global_step, "total_pos": total_pos,
    }
    with open(out_dir / "exp137_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if LOG_FILE:
        LOG_FILE.close()


if __name__ == "__main__":
    main()
