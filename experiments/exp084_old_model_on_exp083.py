"""exp084: Fine-tune the pre-architecture-change 200M model on the exp083 dataset snapshot.

Uses the original 16-layer / 512-head-dim / pos_embed architecture and trains on
the persistent JSONL corpus produced by exp083.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_features import batch_boards_to_fused_token_ids
from move_vocab import UCI_TO_IDX, VOCAB_SIZE, legal_move_mask
from play import ChessTransformer200M

OUTPUT_DIR = Path("outputs/exp084_old_model_on_exp083")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_PATH = OUTPUT_DIR / "exp084.log"
CONFIG_PATH = OUTPUT_DIR / "config.json"
STATUS_PATH = OUTPUT_DIR / "status.json"
LATEST_PATH = CHECKPOINT_DIR / "latest.pt"
BEST_STATE_PATH = CHECKPOINT_DIR / "best.pt"
LATEST_MODEL_PATH = OUTPUT_DIR / "latest_model.pt"
BEST_PATH = OUTPUT_DIR / "best_model.pt"

DEFAULT_DATASET_PATH = Path("outputs/exp083_sf_opening_stream/dataset/positions.jsonl")
INIT_CHECKPOINT_CANDIDATES = [
    Path("outputs/exp082_sf_game_softloop/latest_model.pt"),
    Path("outputs/exp082_sf_game_softloop/best_model.pt"),
    Path("outputs/hf/chess-transformer-200m-latest/best_model.pt"),
]

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

BATCH_SIZE = 4
ACCUM_STEPS = 16
EPOCHS = 1
LR = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
HARD_CE_WEIGHT = 0.25
VALUE_LOSS_WEIGHT = 0.10
TEACHER_TEMP = 1.0
SOFT_TOP_K = 0
KL_CONF_SCALE = 0.0
KL_CONF_MIN = 0.10
KL_CONF_MAX = 1.00
EVAL_FRACTION = 0.05
MAX_EVAL_RECORDS = 2048
LOG_INTERVAL = 25
SAVE_INTERVAL = 200
UPLOAD_INTERVAL_SEC = 300
UPLOAD_TO_HF = True
SAVE_CHECKPOINTS = True
SAVE_FINAL_ONLY = False
SAVE_WEIGHTS_ONLY_CHECKPOINTS = False
HF_REPO_ID = os.environ.get("EXP084_HF_REPO", "avewright/chess-transformer-200m-latest")
HF_PATH_PREFIX = os.environ.get("EXP084_HF_PATH_PREFIX", "").strip().strip("/")
HF_DATASET_REPO = os.environ.get("EXP084_HF_DATASET_REPO", "avewright/exp085-parallel-multipv-harvest")
HF_DATASET_GLOB = "dataset/positions_*.jsonl"


def _hf_repo_path(filename: str) -> str:
    return f"{HF_PATH_PREFIX}/{filename}" if HF_PATH_PREFIX else filename


HF_LATEST_MODEL_PATH = _hf_repo_path("latest_model.pt")
HF_BEST_MODEL_PATH = _hf_repo_path("best_model.pt")
HF_STATUS_PATH = _hf_repo_path("status.json")
HF_CONFIG_PATH = _hf_repo_path("config.json")
HF_LOG_PATH = _hf_repo_path("exp084.log")
HF_RUN_START_MODEL_PATH = _hf_repo_path("run_start_model.pt")
HF_RUN_START_STATUS_PATH = _hf_repo_path("run_start_status.json")

LOG_FILE = None


def log(message: str) -> None:
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(stamped + "\n")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save(payload, tmp, _use_new_zipfile_serialization=False)
    os.replace(tmp, path)


def save_model_weights(path: Path, model: ChessTransformer200M) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    torch.save({"model_state_dict": _model_state_dict_cpu(model)}, tmp)
    os.replace(tmp, path)


def _model_state_dict_cpu(model: ChessTransformer200M) -> dict:
    return {
        k.replace("_orig_mod.", ""): v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def build_resume_checkpoint(
    *,
    model: ChessTransformer200M,
    optimizer: AdamW,
    scaler: GradScaler,
    state: dict,
    eval_fens: set[str],
) -> dict:
    payload = {
        "model_state_dict": _model_state_dict_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if AMP_ENABLED else None,
        "epoch": int(state["epoch"]),
        "train_steps": int(state["train_steps"]),
        "best_eval_loss": float(state["best_eval_loss"]),
        "last_eval": state.get("last_eval"),
        "last_upload_time": float(state.get("last_upload_time", time.time())),
        "dataset_records": int(state.get("dataset_records", 0)),
        "eval_fens": sorted(eval_fens),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def _hf_token_local() -> str | None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("HF_TOKEN")


def _resolve_init_checkpoint(prefer: str = "latest") -> Path | None:
    for candidate in INIT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    remote_paths = [HF_BEST_MODEL_PATH, HF_LATEST_MODEL_PATH] if prefer == "best" else [HF_LATEST_MODEL_PATH, HF_BEST_MODEL_PATH]
    for remote_path in remote_paths:
        downloaded = _hf_download_to_temp(remote_path)
        if downloaded is not None:
            return downloaded
    return None


def _hf_download_to_temp(path_in_repo: str) -> Path | None:
    if not UPLOAD_TO_HF:
        return None
    token = _hf_token_local()
    if not token:
        return None
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=path_in_repo,
            repo_type="model",
            token=token,
        )
        return Path(downloaded)
    except Exception:
        return None


def _upload_file_to_hf(local_path: Path, path_in_repo: str) -> None:
    if not UPLOAD_TO_HF:
        return
    token = _hf_token_local()
    if not token:
        log("[hf] no HF_TOKEN found; skipping upload")
        return
    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)
        try:
            create_repo(HF_REPO_ID, exist_ok=True, repo_type="model", token=token)
        except Exception:
            pass

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
    except Exception as exc:
        log(f"[hf] upload failed for {path_in_repo}: {exc}")


def _write_model_weights_tmp(model: ChessTransformer200M) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix="exp084_", suffix=".pt")
    os.close(fd)
    tmp_path = Path(tmp_name)
    state_dict = _model_state_dict_cpu(model)
    torch.save({"model_state_dict": state_dict}, tmp_path)
    return tmp_path


def _upload_snapshot(model: ChessTransformer200M, status_payload: dict, is_best: bool) -> None:
    weights_path = _write_model_weights_tmp(model)
    try:
        _upload_file_to_hf(weights_path, HF_LATEST_MODEL_PATH)
        if is_best:
            _upload_file_to_hf(weights_path, HF_BEST_MODEL_PATH)
    finally:
        if weights_path.exists():
            weights_path.unlink()

    _upload_file_to_hf(STATUS_PATH, HF_STATUS_PATH)
    _upload_file_to_hf(CONFIG_PATH, HF_CONFIG_PATH)
    _upload_file_to_hf(LOG_PATH, HF_LOG_PATH)


def _upload_run_start_snapshot(init_checkpoint: Path, status_payload: dict) -> None:
    if not UPLOAD_TO_HF:
        return
    _upload_file_to_hf(init_checkpoint, HF_RUN_START_MODEL_PATH)
    run_start_status = dict(status_payload)
    run_start_status["backup_kind"] = "run_start"
    fd, tmp_name = tempfile.mkstemp(prefix="exp084_run_start_", suffix=".json")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        atomic_write_json(tmp_path, run_start_status)
        _upload_file_to_hf(tmp_path, HF_RUN_START_STATUS_PATH)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def build_status_payload(
    *,
    state: dict,
    init_checkpoint: Path,
    last_eval: dict | None,
    done: bool = False,
) -> dict:
    return {
        "updated_at": utcnow_iso(),
        "epoch": state["epoch"],
        "train_steps": state["train_steps"],
        "dataset_records": state["dataset_records"],
        "best_eval_loss": state["best_eval_loss"],
        "last_eval": last_eval,
        "done": done,
        "init_checkpoint": str(init_checkpoint),
        "hf_repo_id": HF_REPO_ID if UPLOAD_TO_HF else None,
        "hf_latest_model_path": HF_LATEST_MODEL_PATH if UPLOAD_TO_HF else None,
        "hf_best_model_path": HF_BEST_MODEL_PATH if UPLOAD_TO_HF else None,
    }


def load_jsonl_dataset(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Allow training off a live-growing dataset shard without dying on a partial final line.
                        log(f"skipping malformed jsonl row path={path} line={line_no}")
    return records


def stable_train_eval_split(records: list[dict]) -> tuple[list[dict], list[dict]]:
    if not records:
        return [], []

    eval_target = min(MAX_EVAL_RECORDS, max(512, int(len(records) * EVAL_FRACTION)))
    keyed: list[tuple[int, dict]] = []
    for item in records:
        fen = item["fen"]
        digest = hashlib.blake2b(fen.encode("utf-8"), digest_size=8).digest()
        keyed.append((int.from_bytes(digest, "big"), item))
    keyed.sort(key=lambda pair: pair[0])
    eval_records = [item for _, item in keyed[:eval_target]]
    train_records = [item for _, item in keyed[eval_target:]]
    return train_records, eval_records


def train_split_with_fixed_eval(records: list[dict], eval_fens: set[str]) -> list[dict]:
    return [item for item in records if item["fen"] not in eval_fens]


def _temperature_scale_probs(targets: list[dict], teacher_temp: float) -> list[float]:
    probs = [max(float(target["prob"]), 1e-12) for target in targets]
    if teacher_temp <= 0:
        raise ValueError("teacher_temp must be > 0")
    if abs(teacher_temp - 1.0) < 1e-9:
        total = sum(probs)
        return [prob / total for prob in probs]
    scaled = [prob ** (1.0 / teacher_temp) for prob in probs]
    total = sum(scaled)
    if total <= 0:
        return [1.0 / len(targets)] * len(targets)
    return [prob / total for prob in scaled]


def _select_soft_targets(item: dict, soft_top_k: int) -> list[dict]:
    targets = item["soft_targets"]
    if soft_top_k > 0:
        return targets[:soft_top_k]
    return targets


def sparse_soft_targets_to_dense(
    batch: list[dict],
    teacher_temp: float = TEACHER_TEMP,
    soft_top_k: int = SOFT_TOP_K,
) -> torch.Tensor:
    dense = torch.zeros(len(batch), VOCAB_SIZE, dtype=torch.float32)
    for row_idx, item in enumerate(batch):
        selected_targets = _select_soft_targets(item, soft_top_k)
        scaled_probs = _temperature_scale_probs(selected_targets, teacher_temp)
        for target, scaled_prob in zip(selected_targets, scaled_probs):
            dense[row_idx, UCI_TO_IDX[target["uci"]]] = float(scaled_prob)
    return dense


def batch_kl_confidence_weights(
    batch: list[dict],
    *,
    kl_conf_scale: float,
    kl_conf_min: float,
    kl_conf_max: float,
) -> torch.Tensor:
    if kl_conf_scale <= 0:
        return torch.ones(len(batch), dtype=torch.float32)

    weights = []
    for item in batch:
        cp_gap = max(float(item.get("cp_gap_top1_top2", 0.0)), 0.0)
        conf = cp_gap / kl_conf_scale
        conf = max(kl_conf_min, min(kl_conf_max, conf))
        weights.append(conf)
    return torch.tensor(weights, dtype=torch.float32)


def compute_policy_losses(
    logits: torch.Tensor,
    best_moves: torch.Tensor,
    soft_targets: torch.Tensor,
    kl_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hard_ce = F.cross_entropy(logits, best_moves)
    log_probs = F.log_softmax(logits, dim=-1)
    kl_per_sample = F.kl_div(log_probs, soft_targets, reduction="none").sum(dim=-1)
    kl = (kl_per_sample * kl_weights).mean()
    return hard_ce, kl


def load_model(checkpoint_path: Path, device: torch.device) -> ChessTransformer200M:
    model = ChessTransformer200M()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    return model


def sample_batch(records: list[dict], batch_size: int) -> list[dict]:
    if len(records) >= batch_size:
        return random.sample(records, batch_size)
    return [random.choice(records) for _ in range(batch_size)]


def evaluate(
    model: ChessTransformer200M,
    eval_records: list[dict],
    *,
    teacher_temp: float = TEACHER_TEMP,
    hard_ce_weight: float = HARD_CE_WEIGHT,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
    soft_top_k: int = SOFT_TOP_K,
    kl_conf_scale: float = KL_CONF_SCALE,
    kl_conf_min: float = KL_CONF_MIN,
    kl_conf_max: float = KL_CONF_MAX,
) -> dict:
    model.eval()
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    total = correct = top3 = 0

    with torch.no_grad():
        for idx in range(0, len(eval_records), BATCH_SIZE):
            batch = eval_records[idx:idx + BATCH_SIZE]
            boards = [chess.Board(item["fen"]) for item in batch]
            best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
            value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.long, device=DEVICE)
            soft_targets = sparse_soft_targets_to_dense(batch, teacher_temp=teacher_temp, soft_top_k=soft_top_k).to(DEVICE)
            kl_weights = batch_kl_confidence_weights(
                batch,
                kl_conf_scale=kl_conf_scale,
                kl_conf_min=kl_conf_min,
                kl_conf_max=kl_conf_max,
            ).to(DEVICE)
            out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
            logits = out["policy_logits"].float()
            value_logits = out["value_logits"].float()

            hard_ce, kl = compute_policy_losses(logits, best_moves, soft_targets, kl_weights)
            value_loss = F.cross_entropy(value_logits, value_targets)
            total_loss = (1.0 - hard_ce_weight) * kl + hard_ce_weight * hard_ce + value_loss_weight * value_loss

            loss_sum += total_loss.item() * len(batch)
            ce_sum += hard_ce.item() * len(batch)
            kl_sum += kl.item() * len(batch)
            val_sum += value_loss.item() * len(batch)

            for row_idx, board in enumerate(boards):
                masked = logits[row_idx].clone()
                mask = legal_move_mask(board).to(DEVICE)
                masked[~mask] = float("-inf")
                pred_idx = masked.argmax().item()
                true_idx = best_moves[row_idx].item()
                if pred_idx == true_idx:
                    correct += 1
                topk = masked.topk(min(3, int(mask.sum().item()))).indices.tolist()
                if true_idx in topk:
                    top3 += 1
                total += 1

    return {
        "loss": loss_sum / max(total, 1),
        "ce": ce_sum / max(total, 1),
        "kl": kl_sum / max(total, 1),
        "value": val_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "top3": top3 / max(total, 1),
        "n": total,
    }


def train_epoch(
    model: ChessTransformer200M,
    optimizer: AdamW,
    scaler: GradScaler,
    train_records: list[dict],
    eval_records: list[dict],
    eval_fens: set[str],
    state: dict,
    init_checkpoint: Path,
    teacher_temp: float,
    hard_ce_weight: float,
    value_loss_weight: float,
    soft_top_k: int,
    kl_conf_scale: float,
    kl_conf_min: float,
    kl_conf_max: float,
) -> dict:
    model.train()
    random.shuffle(train_records)
    steps_per_epoch = math.ceil(len(train_records) / (BATCH_SIZE * ACCUM_STEPS))
    cursor = 0
    loss_sum = ce_sum = kl_sum = val_sum = 0.0
    best_eval = state.get("best_eval_loss", float("inf"))
    last_upload_time = float(state.get("last_upload_time", time.time()))
    last_eval_metrics = state.get("last_eval")

    for step_idx in range(steps_per_epoch):
        optimizer.zero_grad(set_to_none=True)
        step_loss = step_ce = step_kl = step_val = 0.0

        for _ in range(ACCUM_STEPS):
            batch = train_records[cursor:cursor + BATCH_SIZE]
            cursor += BATCH_SIZE
            if len(batch) < BATCH_SIZE:
                needed = BATCH_SIZE - len(batch)
                batch = batch + train_records[:needed]
                cursor = needed

            boards = [chess.Board(item["fen"]) for item in batch]
            best_moves = torch.tensor([UCI_TO_IDX[item["best_move"]] for item in batch], dtype=torch.long, device=DEVICE)
            value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.long, device=DEVICE)
            soft_targets = sparse_soft_targets_to_dense(batch, teacher_temp=teacher_temp, soft_top_k=soft_top_k).to(DEVICE)
            kl_weights = batch_kl_confidence_weights(
                batch,
                kl_conf_scale=kl_conf_scale,
                kl_conf_min=kl_conf_min,
                kl_conf_max=kl_conf_max,
            ).to(DEVICE)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED):
                out = model(batch_boards_to_fused_token_ids(boards, DEVICE))
                logits = out["policy_logits"]
                value_logits = out["value_logits"]
                hard_ce, kl = compute_policy_losses(logits, best_moves, soft_targets, kl_weights)
                value_loss = F.cross_entropy(value_logits, value_targets)
                total_loss = ((1.0 - hard_ce_weight) * kl + hard_ce_weight * hard_ce + value_loss_weight * value_loss) / ACCUM_STEPS

            scaler.scale(total_loss).backward()
            step_loss += total_loss.item() * ACCUM_STEPS
            step_ce += hard_ce.item()
            step_kl += kl.item()
            step_val += value_loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        state["train_steps"] += 1
        loss_sum += step_loss
        ce_sum += step_ce / ACCUM_STEPS
        kl_sum += step_kl / ACCUM_STEPS
        val_sum += step_val / ACCUM_STEPS

        if state["train_steps"] % LOG_INTERVAL == 0:
            done_steps = step_idx + 1
            log(
                f"train_step={state['train_steps']} epoch={state['epoch']} "
                f"loss={loss_sum / done_steps:.4f} ce={ce_sum / done_steps:.4f} "
                f"kl={kl_sum / done_steps:.4f} value={val_sum / done_steps:.4f} "
                f"gnorm={float(grad_norm):.2f}"
            )

        now = time.time()
        if UPLOAD_TO_HF and now - last_upload_time >= UPLOAD_INTERVAL_SEC:
            state["best_eval_loss"] = best_eval
            state["last_eval"] = last_eval_metrics
            if SAVE_CHECKPOINTS:
                if SAVE_WEIGHTS_ONLY_CHECKPOINTS:
                    save_model_weights(LATEST_MODEL_PATH, model)
                else:
                    save_checkpoint(
                        LATEST_PATH,
                        build_resume_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            state=state,
                            eval_fens=eval_fens,
                        ),
                    )
            heartbeat_status = build_status_payload(
                state=state,
                init_checkpoint=init_checkpoint,
                last_eval=last_eval_metrics,
            )
            atomic_write_json(STATUS_PATH, heartbeat_status)
            log(f"[hf] heartbeat upload at step={state['train_steps']}")
            _upload_snapshot(model, heartbeat_status, is_best=False)
            last_upload_time = time.time()
            state["last_upload_time"] = last_upload_time

        if state["train_steps"] % SAVE_INTERVAL == 0 or step_idx == steps_per_epoch - 1:
            eval_metrics = evaluate(
                model,
                eval_records,
                teacher_temp=teacher_temp,
                hard_ce_weight=hard_ce_weight,
                value_loss_weight=value_loss_weight,
                soft_top_k=soft_top_k,
                kl_conf_scale=kl_conf_scale,
                kl_conf_min=kl_conf_min,
                kl_conf_max=kl_conf_max,
            )
            log(
                f"eval step={state['train_steps']} loss={eval_metrics['loss']:.4f} "
                f"ce={eval_metrics['ce']:.4f} kl={eval_metrics['kl']:.4f} value={eval_metrics['value']:.4f} "
                f"acc={eval_metrics['acc']:.4f} top3={eval_metrics['top3']:.4f} n={eval_metrics['n']}"
            )
            is_best = eval_metrics["loss"] < best_eval
            if is_best:
                best_eval = eval_metrics["loss"]
            last_eval_metrics = eval_metrics
            state["best_eval_loss"] = best_eval
            state["last_eval"] = eval_metrics
            resume_payload = build_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                state=state,
                eval_fens=eval_fens,
            )
            if SAVE_CHECKPOINTS:
                if SAVE_WEIGHTS_ONLY_CHECKPOINTS:
                    save_model_weights(LATEST_MODEL_PATH, model)
                    if is_best:
                        save_model_weights(BEST_PATH, model)
                else:
                    save_checkpoint(LATEST_PATH, resume_payload)
                    if is_best:
                        save_checkpoint(BEST_STATE_PATH, resume_payload)
            status_payload = build_status_payload(
                state=state,
                init_checkpoint=init_checkpoint,
                last_eval=eval_metrics,
            )
            atomic_write_json(STATUS_PATH, status_payload)
            _upload_snapshot(model, status_payload, is_best=is_best)
            last_upload_time = time.time()
            state["last_upload_time"] = last_upload_time

    state["best_eval_loss"] = best_eval
    state["last_eval"] = last_eval_metrics
    state["last_upload_time"] = last_upload_time
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp084 trainer for JSONL soft-target datasets")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--dataset-glob", type=str, default=None)
    parser.add_argument("--hf-dataset-repo", type=str, default=None)
    parser.add_argument("--hf-dataset-glob", type=str, default=HF_DATASET_GLOB)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--upload-interval-sec", type=int, default=UPLOAD_INTERVAL_SEC)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--teacher-temp", type=float, default=TEACHER_TEMP)
    parser.add_argument("--hard-ce-weight", type=float, default=HARD_CE_WEIGHT)
    parser.add_argument("--value-loss-weight", type=float, default=VALUE_LOSS_WEIGHT)
    parser.add_argument("--soft-top-k", type=int, default=SOFT_TOP_K)
    parser.add_argument("--kl-conf-scale", type=float, default=KL_CONF_SCALE)
    parser.add_argument("--kl-conf-min", type=float, default=KL_CONF_MIN)
    parser.add_argument("--kl-conf-max", type=float, default=KL_CONF_MAX)
    parser.add_argument("--resume-from", choices=["latest", "best"], default="latest")
    parser.add_argument("--no-upload-to-hf", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--save-final-only", action="store_true")
    parser.add_argument("--save-weights-only-checkpoints", action="store_true")
    parser.add_argument("--reload-dataset-each-epoch", action="store_true")
    return parser.parse_args()


def _download_hf_dataset_snapshot(repo_id: str, dataset_glob: str) -> list[Path]:
    token = _hf_token_local()
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[dataset_glob],
            token=token,
        )
    )
    return sorted(path for path in snapshot_dir.glob(dataset_glob) if path.is_file())


def resolve_dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.dataset_path is not None:
        paths = [args.dataset_path]
    elif args.dataset_glob:
        paths = sorted(Path().glob(args.dataset_glob))
    elif args.hf_dataset_repo:
        paths = _download_hf_dataset_snapshot(args.hf_dataset_repo, args.hf_dataset_glob)
    elif DEFAULT_DATASET_PATH.exists():
        paths = [DEFAULT_DATASET_PATH]
    else:
        paths = sorted((Path("outputs/exp085_parallel_multipv_harvest/dataset")).glob("positions_*.jsonl"))

    paths = [path for path in paths if path.exists() and path.is_file()]
    if not paths:
        raise FileNotFoundError("No dataset files found for exp084.")
    return paths


def main() -> None:
    global OUTPUT_DIR, CHECKPOINT_DIR, LOG_PATH, CONFIG_PATH, STATUS_PATH, LATEST_PATH, BEST_STATE_PATH, LATEST_MODEL_PATH, BEST_PATH
    global LOG_FILE, UPLOAD_INTERVAL_SEC, UPLOAD_TO_HF, SAVE_CHECKPOINTS, SAVE_FINAL_ONLY, SAVE_WEIGHTS_ONLY_CHECKPOINTS
    args = parse_args()
    UPLOAD_INTERVAL_SEC = args.upload_interval_sec
    UPLOAD_TO_HF = UPLOAD_TO_HF and not args.no_upload_to_hf
    SAVE_FINAL_ONLY = args.save_final_only
    SAVE_WEIGHTS_ONLY_CHECKPOINTS = args.save_weights_only_checkpoints
    SAVE_CHECKPOINTS = not args.no_save_checkpoints and not SAVE_FINAL_ONLY
    OUTPUT_DIR = args.output_dir
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_PATH = OUTPUT_DIR / "exp084.log"
    CONFIG_PATH = OUTPUT_DIR / "config.json"
    STATUS_PATH = OUTPUT_DIR / "status.json"
    LATEST_PATH = CHECKPOINT_DIR / "latest.pt"
    BEST_STATE_PATH = CHECKPOINT_DIR / "best.pt"
    LATEST_MODEL_PATH = OUTPUT_DIR / "latest_model.pt"
    BEST_PATH = OUTPUT_DIR / "best_model.pt"
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_PATH

    dataset_paths = resolve_dataset_paths(args)
    records = load_jsonl_dataset(dataset_paths)
    train_records, eval_records = stable_train_eval_split(records)
    eval_fens = {item["fen"] for item in eval_records}

    init_checkpoint = args.init_checkpoint.resolve() if args.init_checkpoint is not None else _resolve_init_checkpoint(args.resume_from)
    if init_checkpoint is None:
        raise FileNotFoundError("No initial checkpoint found locally or on Hugging Face for exp084.")
    model = load_model(init_checkpoint, DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device="cuda", enabled=AMP_ENABLED)

    state = {
        "epoch": 1,
        "train_steps": 0,
        "best_eval_loss": float("inf"),
        "dataset_records": len(records),
        "last_eval": None,
        "last_upload_time": time.time(),
    }

    if LATEST_PATH.exists():
        ckpt = torch.load(LATEST_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if AMP_ENABLED and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        state["epoch"] = int(ckpt.get("epoch", 1))
        state["train_steps"] = int(ckpt.get("train_steps", 0))
        state["best_eval_loss"] = float(ckpt.get("best_eval_loss", float("inf")))
        state["last_eval"] = ckpt.get("last_eval")
        state["last_upload_time"] = float(ckpt.get("last_upload_time", time.time()))
        state["dataset_records"] = int(ckpt.get("dataset_records", len(records)))
        if ckpt.get("eval_fens"):
            eval_fens = set(ckpt["eval_fens"])
            eval_records = [item for item in records if item["fen"] in eval_fens]
            train_records = train_split_with_fixed_eval(records, eval_fens)
        if ckpt.get("python_random_state") is not None:
            random.setstate(ckpt["python_random_state"])
        if ckpt.get("torch_random_state") is not None:
            torch.random.set_rng_state(ckpt["torch_random_state"])
        if torch.cuda.is_available() and ckpt.get("cuda_random_state_all") is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_random_state_all"])
        log(f"resumed from {LATEST_PATH} at epoch={state['epoch']} step={state['train_steps']}")
    else:
        remote_model = _hf_download_to_temp(HF_BEST_MODEL_PATH if args.resume_from == "best" else HF_LATEST_MODEL_PATH)
        remote_status = _hf_download_to_temp(HF_STATUS_PATH)
        if remote_model is not None:
            remote_ckpt = torch.load(remote_model, map_location="cpu", weights_only=False)
            remote_state = remote_ckpt.get("model_state_dict", remote_ckpt)
            model.load_state_dict(remote_state)
            if remote_status is not None:
                try:
                    status = json.loads(remote_status.read_text(encoding="utf-8"))
                    state["epoch"] = int(status.get("epoch", 1))
                    state["train_steps"] = int(status.get("train_steps", 0))
                    state["best_eval_loss"] = float(status.get("best_eval_loss", float("inf")))
                    state["last_eval"] = status.get("last_eval")
                except Exception as exc:
                    log(f"[hf] failed to parse remote status: {exc}")
            log(
                f"resumed weights from hf://{HF_REPO_ID}/"
                f"{HF_BEST_MODEL_PATH if args.resume_from == 'best' else HF_LATEST_MODEL_PATH} "
                f"at epoch={state['epoch']} step={state['train_steps']}"
            )

    atomic_write_json(
        CONFIG_PATH,
        {
            "started_at": utcnow_iso(),
            "init_checkpoint": str(init_checkpoint),
            "dataset_path": str(dataset_paths[0]) if len(dataset_paths) == 1 else None,
            "dataset_files": [str(path) for path in dataset_paths],
            "dataset_records": len(records),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "fixed_eval_fens": len(eval_fens),
            "batch_size": BATCH_SIZE,
            "accum_steps": ACCUM_STEPS,
            "effective_batch": BATCH_SIZE * ACCUM_STEPS,
            "epochs": args.epochs,
            "lr": args.lr,
            "teacher_temp": args.teacher_temp,
            "soft_top_k": args.soft_top_k,
            "kl_conf_scale": args.kl_conf_scale,
            "kl_conf_min": args.kl_conf_min,
            "kl_conf_max": args.kl_conf_max,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "hard_ce_weight": args.hard_ce_weight,
            "value_loss_weight": args.value_loss_weight,
            "resume_from": args.resume_from,
            "save_checkpoints": SAVE_CHECKPOINTS,
            "save_final_only": SAVE_FINAL_ONLY,
            "save_weights_only_checkpoints": SAVE_WEIGHTS_ONLY_CHECKPOINTS,
            "reload_dataset_each_epoch": args.reload_dataset_each_epoch,
            "hf_dataset_repo": args.hf_dataset_repo,
            "hf_dataset_glob": args.hf_dataset_glob if args.hf_dataset_repo else None,
            "device": str(DEVICE),
            "upload_to_hf": UPLOAD_TO_HF,
            "hf_repo_id": HF_REPO_ID if UPLOAD_TO_HF else None,
            "hf_path_prefix": HF_PATH_PREFIX if UPLOAD_TO_HF else None,
        },
    )

    log("=" * 72)
    log("exp084: old-architecture model fine-tune on exp083 dataset snapshot")
    log("=" * 72)
    log(f"device={DEVICE}")
    if DEVICE.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)} vram_gb={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
    log(f"dataset_records={len(records)} train={len(train_records)} eval={len(eval_records)} files={len(dataset_paths)}")
    log(f"effective_batch={BATCH_SIZE * ACCUM_STEPS} epochs={args.epochs}")
    log(
        f"lr={args.lr} teacher_temp={args.teacher_temp} hard_ce_weight={args.hard_ce_weight} "
        f"soft_top_k={args.soft_top_k} kl_conf_scale={args.kl_conf_scale} resume_from={args.resume_from} "
        f"reload_dataset_each_epoch={args.reload_dataset_each_epoch}"
    )
    log(f"init_checkpoint={init_checkpoint}")
    if UPLOAD_TO_HF:
        log(f"hf_checkpoint_target={HF_REPO_ID}/{HF_PATH_PREFIX}")
    if args.hf_dataset_repo:
        log(f"hf_dataset_source={args.hf_dataset_repo}/{args.hf_dataset_glob}")
    if not SAVE_CHECKPOINTS:
        log("local_checkpoints=disabled")
    if SAVE_FINAL_ONLY:
        log("save_strategy=final_only")
    if SAVE_WEIGHTS_ONLY_CHECKPOINTS:
        log("save_strategy=weights_only")

    initial_eval = evaluate(
        model,
        eval_records,
        teacher_temp=args.teacher_temp,
        hard_ce_weight=args.hard_ce_weight,
        value_loss_weight=args.value_loss_weight,
        soft_top_k=args.soft_top_k,
        kl_conf_scale=args.kl_conf_scale,
        kl_conf_min=args.kl_conf_min,
        kl_conf_max=args.kl_conf_max,
    )
    log(
        f"initial_eval loss={initial_eval['loss']:.4f} ce={initial_eval['ce']:.4f} "
        f"kl={initial_eval['kl']:.4f} value={initial_eval['value']:.4f} "
        f"acc={initial_eval['acc']:.4f} top3={initial_eval['top3']:.4f} n={initial_eval['n']}"
    )
    state["best_eval_loss"] = min(state["best_eval_loss"], initial_eval["loss"])
    state["last_eval"] = initial_eval
    if SAVE_CHECKPOINTS:
        if SAVE_WEIGHTS_ONLY_CHECKPOINTS:
            save_model_weights(LATEST_MODEL_PATH, model)
        else:
            save_checkpoint(
                LATEST_PATH,
                build_resume_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    state=state,
                    eval_fens=eval_fens,
                ),
            )
    atomic_write_json(
        STATUS_PATH,
        build_status_payload(
            state=state,
            init_checkpoint=init_checkpoint,
            last_eval=initial_eval,
        ),
    )
    _upload_run_start_snapshot(
        init_checkpoint,
        build_status_payload(
            state=state,
            init_checkpoint=init_checkpoint,
            last_eval=initial_eval,
        ),
    )

    while state["epoch"] <= args.epochs:
        if args.reload_dataset_each_epoch and state["epoch"] > 1:
            dataset_paths = resolve_dataset_paths(args)
            records = load_jsonl_dataset(dataset_paths)
            eval_records = [item for item in records if item["fen"] in eval_fens]
            train_records = train_split_with_fixed_eval(records, eval_fens)
            state["dataset_records"] = len(records)
            log(
                f"reloaded_dataset epoch={state['epoch']} records={len(records)} "
                f"train={len(train_records)} eval={len(eval_records)} files={len(dataset_paths)}"
            )
        train_epoch(
            model,
            optimizer,
            scaler,
            train_records,
            eval_records,
            eval_fens,
            state,
            init_checkpoint,
            teacher_temp=args.teacher_temp,
            hard_ce_weight=args.hard_ce_weight,
            value_loss_weight=args.value_loss_weight,
            soft_top_k=args.soft_top_k,
            kl_conf_scale=args.kl_conf_scale,
            kl_conf_min=args.kl_conf_min,
            kl_conf_max=args.kl_conf_max,
        )
        state["epoch"] += 1

    final_eval = evaluate(
        model,
        eval_records,
        teacher_temp=args.teacher_temp,
        hard_ce_weight=args.hard_ce_weight,
        value_loss_weight=args.value_loss_weight,
        soft_top_k=args.soft_top_k,
        kl_conf_scale=args.kl_conf_scale,
        kl_conf_min=args.kl_conf_min,
        kl_conf_max=args.kl_conf_max,
    )
    log(
        f"final_eval loss={final_eval['loss']:.4f} ce={final_eval['ce']:.4f} "
        f"kl={final_eval['kl']:.4f} value={final_eval['value']:.4f} "
        f"acc={final_eval['acc']:.4f} top3={final_eval['top3']:.4f} n={final_eval['n']}"
    )
    final_is_best = final_eval["loss"] <= state["best_eval_loss"]
    state["best_eval_loss"] = min(state["best_eval_loss"], final_eval["loss"])
    state["last_eval"] = final_eval
    final_resume_payload = build_resume_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        state=state,
        eval_fens=eval_fens,
    )
    if SAVE_CHECKPOINTS:
        if SAVE_WEIGHTS_ONLY_CHECKPOINTS:
            save_model_weights(LATEST_MODEL_PATH, model)
            if final_is_best:
                save_model_weights(BEST_PATH, model)
        else:
            save_checkpoint(LATEST_PATH, final_resume_payload)
            if final_is_best:
                save_checkpoint(BEST_STATE_PATH, final_resume_payload)
    elif SAVE_FINAL_ONLY:
        save_checkpoint(LATEST_PATH, final_resume_payload)
    final_status = build_status_payload(
        state=state,
        init_checkpoint=init_checkpoint,
        last_eval=final_eval,
        done=True,
    )
    atomic_write_json(STATUS_PATH, final_status)
    _upload_snapshot(model, final_status, is_best=final_is_best)
    log("done")


if __name__ == "__main__":
    main()
