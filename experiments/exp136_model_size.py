"""exp136: Smaller model on cached Lichess data.

ROOT CAUSE: The 204M model trained on 224K positions has 910 params per example.
This experiment trains SMALLER models on cached HF data (up to 1M positions).

Strategy: Use load_training_data() to cache positions locally as .pt, then
train in a simple random-batch loop. No streaming complexity.

Configs:
  A: 4L/256d/4H   (~3.5M params) — tiny
  B: 8L/512d/8H   (~25.9M params) — AlphaZero-scale
  C: 16L/512d/8H  (~51.2M params) — deeper

Hardware: RTX 4060 8GB, AMP FP16, gradient accumulation
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

from chess_transformer_factory import (
    build_model, ChessTransformerConfig, count_parameters,
)
from move_vocab import VOCAB_SIZE, IDX_TO_UCI, move_to_index, legal_move_mask, index_to_move
from data_loader import (
    load_training_data, get_batch_input, get_eval_batch_input,
    compute_wdl, compute_phase,
)
from uci_engine import MCTSSearch, SyzygyProbe

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Configs ──
MODEL_CONFIGS = {
    "A": ChessTransformerConfig(
        encoder_dim=256, hidden_dim=256, num_layers=4, num_heads=4,
        ffn_ratio=4, dropout=0.1, policy_head_dim=256, value_hidden=256,
    ),
    "B": ChessTransformerConfig(
        encoder_dim=256, hidden_dim=512, num_layers=8, num_heads=8,
        ffn_ratio=4, dropout=0.1, policy_head_dim=256, value_hidden=256,
    ),
    "C": ChessTransformerConfig(
        encoder_dim=256, hidden_dim=512, num_layers=16, num_heads=8,
        ffn_ratio=4, dropout=0.1, policy_head_dim=256, value_hidden=256,
    ),
}

# ── Training Configs (tuned for RTX 4060 8GB — profiled VRAM limits) ──
TRAIN_CONFIGS = {
    "A": {"batch_size": 128, "accum_steps": 2, "lr": 3e-4},   # 691 pos/s, 5.2GB
    "B": {"batch_size": 64,  "accum_steps": 4, "lr": 2e-4},   # 316 pos/s, 4.6GB
    "C": {"batch_size": 64,  "accum_steps": 8, "lr": 1e-4},   # ~200 pos/s est
}

WARMUP_FRAC = 0.02
MIN_LR_FRAC = 0.05
GRAD_CLIP = 0.5
WEIGHT_DECAY = 0.01
VALUE_WEIGHT = 0.5
SEED = 42

EMA_DECAY = 0.999
EMA_START_STEP = 100

LOG_INTERVAL = 10
EVAL_INTERVAL = 100
SAVE_INTERVAL = 100

# ELO eval
OPENINGS = [
    [], ["e2e4", "e7e5"], ["d2d4", "d7d5"], ["e2e4", "c7c5"],
    ["d2d4", "g8f6"], ["e2e4", "e7e6"], ["c2c4", "e7e5"], ["g1f3", "d7d5"],
]

SHUTDOWN_REQUESTED = False
def _signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SIGNAL] Shutdown requested...", flush=True)
signal.signal(signal.SIGINT, _signal_handler)

LOG_FILE = None
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        LOG_FILE.write(line + "\n")
        LOG_FILE.flush()


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr_frac):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_model_only(path, model, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pt.tmp')
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, tmp)
    os.replace(str(tmp), str(path))


# ── Evaluation ──
def evaluate(model, eval_data, eval_tensors, device, batch_size=128):
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


# ── ELO Gauntlet ──
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
        log(f"  [{label}] G{gi+1}/{n_games}: {tag} | {avg:.3f} [{lo:.3f},{hi:.3f}] ~{sf_elo + elo_diff(avg):.0f}")
    engine.quit()
    return {"score": total_score / n_games, "elo": sf_elo + elo_diff(total_score / n_games),
            "ci": list(wilson_ci(total_score, n_games)), "n": n_games}


# ── Training ──
def train_model(config_label, model_config, train_config, train_tensors,
                eval_data, eval_tensors, out_dir, epochs=1,
                max_steps=None, run_elo=False, sf_elo=1900, n_elo_games=16):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = False

    log(f"\n{'='*60}")
    log(f"Config {config_label}: {model_config.num_layers}L/{model_config.hidden_dim}d/{model_config.num_heads}H")

    model = build_model(model_config)
    n_params = count_parameters(model)
    log(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    model.to(DEVICE).train()

    bs = train_config["batch_size"]
    accum = train_config["accum_steps"]
    lr = train_config["lr"]
    eff_batch = bs * accum
    log(f"  Batch: {bs}x{accum}={eff_batch}, LR: {lr}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler('cuda') if DEVICE.type == "cuda" else None
    ema = EMAModel(model, decay=EMA_DECAY)

    n_train = train_tensors["move_idx"].shape[0]
    steps_per_epoch = n_train // eff_batch
    total_steps = steps_per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    warmup_steps = int(total_steps * WARMUP_FRAC)
    log(f"  Data: {n_train:,} pos, {steps_per_epoch:,} steps/epoch, "
        f"{total_steps:,} total, warmup={warmup_steps}")

    init_metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  Init: acc={init_metrics['accuracy']:.4f}, top3={init_metrics['top3']:.4f}, "
        f"val={init_metrics['value_acc']:.4f}")

    config_out = out_dir / config_label
    config_out.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad()
    global_step = 0
    accum_count = 0
    running_ce = running_val = running_gnorm = 0.0
    log_count = 0
    best_acc = init_metrics['accuracy']
    nan_count = 0
    t_start = time.time()

    log("  Training...")
    for epoch in range(epochs):
        perm = torch.randperm(n_train)

        for batch_start in range(0, n_train, bs):
            if SHUTDOWN_REQUESTED or (max_steps and global_step >= max_steps):
                break

            batch_end = min(batch_start + bs, n_train)
            if batch_end - batch_start < bs:
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
                scaled_loss = total_loss / accum

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

            if accum_count >= accum:
                lr_now = cosine_lr(global_step, total_steps, warmup_steps, lr, MIN_LR_FRAC)
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

                if global_step >= EMA_START_STEP:
                    ema.update(model)

                running_gnorm += gnorm if not (math.isnan(gnorm) or math.isinf(gnorm)) else 0.0
                log_count += 1
                global_step += 1
                accum_count = 0

                if global_step % LOG_INTERVAL == 0 and log_count > 0:
                    avg_ce = running_ce / (log_count * accum)
                    avg_val = running_val / (log_count * accum)
                    avg_gn = running_gnorm / log_count
                    elapsed = time.time() - t_start
                    pos_s = (global_step * eff_batch) / max(elapsed, 1)
                    log(f"  [{config_label}] step={global_step:,} e{epoch} "
                        f"ce={avg_ce:.4f} val={avg_val:.4f} gn={avg_gn:.2f} "
                        f"lr={lr_now:.2e} {pos_s:.0f}pos/s")
                    running_ce = running_val = running_gnorm = 0.0
                    log_count = 0

                if global_step % EVAL_INTERVAL == 0:
                    metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
                    log(f"  [{config_label}] EVAL step={global_step}: "
                        f"acc={metrics['accuracy']:.4f} top3={metrics['top3']:.4f} "
                        f"val={metrics['value_acc']:.4f}")
                    if metrics['accuracy'] > best_acc:
                        best_acc = metrics['accuracy']
                        ema.apply_shadow(model)
                        save_model_only(config_out / "best_model.pt", model, model_config)
                        ema.restore(model)
                        log(f"  [{config_label}] NEW BEST: {best_acc:.4f}")

                if global_step % SAVE_INTERVAL == 0:
                    save_model_only(config_out / "latest_model.pt", model, model_config)

        if SHUTDOWN_REQUESTED or (max_steps and global_step >= max_steps):
            break
        log(f"  [{config_label}] Epoch {epoch} done")

    elapsed = time.time() - t_start
    total_pos = global_step * eff_batch
    log(f"\n  [{config_label}] Done: {global_step:,} steps, {total_pos:,} pos in {elapsed/60:.1f}min "
        f"({total_pos/max(elapsed,1):.0f} pos/s)")

    if global_step >= EMA_START_STEP:
        ema.apply_shadow(model)

    final_metrics = evaluate(model, eval_data, eval_tensors, DEVICE)
    log(f"  [{config_label}] FINAL: acc={final_metrics['accuracy']:.4f} "
        f"top3={final_metrics['top3']:.4f} val={final_metrics['value_acc']:.4f}")
    save_model_only(config_out / "final_model.pt", model, model_config)
    if global_step >= EMA_START_STEP:
        ema.restore(model)

    elo_result = None
    if run_elo:
        log(f"\n  [{config_label}] ELO gauntlet ({n_elo_games}g vs SF{sf_elo})...")
        best_path = config_out / "best_model.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.to(DEVICE).eval()
        elo_result = run_elo_gauntlet(model, config_label, sf_elo, n_elo_games)
        log(f"  [{config_label}] ELO: score={elo_result['score']:.3f} "
            f"elo={elo_result['elo']:.0f} ci={elo_result['ci']}")

    del model, optimizer, scaler, ema
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "config": config_label, "model_config": model_config.to_dict(),
        "n_params": n_params, "global_step": global_step,
        "total_positions": total_pos, "elapsed_min": elapsed / 60,
        "init_metrics": init_metrics, "final_metrics": final_metrics,
        "best_acc": best_acc, "elo": elo_result,
    }


def main():
    global LOG_FILE

    ap = argparse.ArgumentParser(description="exp136: Model size ablation")
    ap.add_argument("--config", type=str, default="B")
    ap.add_argument("--n-train", type=int, default=500_000,
                    help="Number of training positions to load from HF")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--elo", action="store_true")
    ap.add_argument("--elo-games", type=int, default=16)
    ap.add_argument("--sf-elo", type=int, default=1900)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "exp136_model_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = open(out_dir / "exp136.log", "w")

    log("=" * 60)
    log(f"exp136: Model Size vs Data Scale")
    log(f"  configs: {args.config}, n_train: {args.n_train:,}, epochs: {args.epochs}")
    log(f"  elo: {args.elo} ({args.elo_games}g vs SF{args.sf_elo})")
    log(f"  device: {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"  GPU: {props.name} ({props.total_memory / 1e9:.1f}GB)")

    log("Loading training data from HF (will cache locally)...")
    train_tensors, eval_data, eval_tensors = load_training_data(
        n_train=args.n_train,
        n_eval=2500,
        encoder_type="fused",
        min_depth=15,
        seed=args.seed,
    )
    n_train = train_tensors["move_idx"].shape[0]
    log(f"  Loaded: {n_train:,} train, {len(eval_data):,} eval positions")

    configs_to_run = ["A", "B", "C"] if args.config.lower() == "all" else \
        [c.strip().upper() for c in args.config.split(",")]

    all_results = []
    for cfg_label in configs_to_run:
        if cfg_label not in MODEL_CONFIGS:
            continue
        result = train_model(
            cfg_label, MODEL_CONFIGS[cfg_label], TRAIN_CONFIGS[cfg_label],
            train_tensors, eval_data, eval_tensors, out_dir,
            epochs=args.epochs, max_steps=args.max_steps,
            run_elo=args.elo, sf_elo=args.sf_elo, n_elo_games=args.elo_games,
        )
        all_results.append(result)

    log("\n" + "=" * 60)
    log("SUMMARY:")
    log(f"{'Cfg':<5} {'Params':<10} {'Steps':<8} {'BestAcc':<10} {'FinalAcc':<10} {'Top3':<8} {'ELO':<8}")
    log("-" * 70)
    for r in all_results:
        elo_str = f"{r['elo']['elo']:.0f}" if r['elo'] else "N/A"
        log(f"{r['config']:<5} {r['n_params']/1e6:<10.1f} {r['global_step']:<8} "
            f"{r['best_acc']:<10.4f} {r['final_metrics']['accuracy']:<10.4f} "
            f"{r['final_metrics']['top3']:<8.4f} {elo_str:<8}")

    with open(out_dir / "exp136_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    if LOG_FILE:
        LOG_FILE.close()


if __name__ == "__main__":
    main()
