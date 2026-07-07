#!/usr/bin/env python3
"""Persistent health monitor for exp076 training.

Run in tmux: tmux new -s monitor 'python3 monitor_exp076.py'

Checks every 2 minutes:
  - GPU temp, power, utilization, memory
  - Training process alive
  - Training progress (step, loss, throughput)
  - Disk space
  - Alerts on anomalies (NaN, crash, GPU overheat, low throughput)

Writes a compact health log to outputs/exp076_continue_v2/health.log
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("outputs/exp076_continue_v2")
HEALTH_LOG = OUTPUT_DIR / "health.log"
TRAIN_LOG_JSON = OUTPUT_DIR / "training_log.json"
CHECK_INTERVAL = 120  # seconds

# Thresholds
GPU_TEMP_WARN = 80       # °C
GPU_TEMP_CRITICAL = 88   # °C
MIN_THROUGHPUT = 100      # pos/s (after warmup)
MAX_NAN_COUNT = 10
WARMUP_STEPS = 500       # don't alert on low throughput before this


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    with open(HEALTH_LOG, "a") as f:
        f.write(line + "\n")


def get_gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "temp": int(parts[0]),
            "power_w": float(parts[1]),
            "util_pct": int(parts[2]),
            "mem_used_mb": int(parts[3]),
            "mem_total_mb": int(parts[4]),
        }
    except Exception as e:
        return {"error": str(e)}


def get_gpu_processes():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            text=True, timeout=10
        ).strip()
        return len(out.splitlines()) if out else 0
    except Exception:
        return -1


def get_training_state():
    try:
        with open(TRAIN_LOG_JSON) as f:
            log_data = json.load(f)
        if not log_data:
            return None

        last = log_data[-1]
        trains = [e for e in log_data if e.get("type") == "train"]
        evals = [e for e in log_data if e.get("type") == "eval"]

        result = {
            "last_step": last.get("step", 0),
            "last_type": last.get("type", "?"),
            "positions_seen": last.get("positions_seen", 0),
            "total_entries": len(log_data),
        }

        if trains:
            lt = trains[-1]
            result["last_train_step"] = lt["step"]
            result["policy_loss"] = lt.get("policy_loss")
            result["value_loss"] = lt.get("value_loss")
            result["throughput"] = lt.get("throughput")
            result["lr"] = lt.get("lr")
            result["grad_norm"] = lt.get("grad_norm")
            result["peak_mem_gb"] = lt.get("peak_mem_gb")

        if evals:
            le = evals[-1]
            result["last_eval_step"] = le["step"]
            result["accuracy"] = le.get("accuracy")
            result["top3_accuracy"] = le.get("top3_accuracy")
            result["value_accuracy"] = le.get("value_accuracy")
            result["mean_sf_rank"] = le.get("mean_sf_rank")

        return result
    except Exception as e:
        return {"error": str(e)}


def get_disk_space():
    try:
        out = subprocess.check_output(
            ["df", "-h", "/root"],
            text=True, timeout=10
        )
        lines = out.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            return {"total": parts[1], "used": parts[2], "avail": parts[3], "use_pct": parts[4]}
    except Exception:
        pass
    return {}


def check_for_crashes():
    crash_log = OUTPUT_DIR / "crash_log.txt"
    if crash_log.exists():
        return crash_log.read_text()[-200:]
    return None


def main():
    log("=" * 60)
    log("HEALTH MONITOR STARTED for exp076_continue_v2")
    log(f"Check interval: {CHECK_INTERVAL}s")
    log("=" * 60)

    checks = 0
    last_step = -1
    stall_count = 0

    while True:
        checks += 1
        gpu = get_gpu_stats()
        gpu_procs = get_gpu_processes()
        state = get_training_state()
        disk = get_disk_space()
        crash = check_for_crashes()

        # Build status line
        if "error" not in gpu:
            gpu_str = (f"GPU: {gpu['temp']}°C {gpu['power_w']:.0f}W "
                      f"{gpu['util_pct']}% {gpu['mem_used_mb']}MB/{gpu['mem_total_mb']}MB")
        else:
            gpu_str = f"GPU: ERROR {gpu['error']}"

        if state and "error" not in state:
            step = state.get("last_train_step", state.get("last_step", 0))
            pos = state.get("positions_seen", 0)
            pl = state.get("policy_loss")
            tp = state.get("throughput")
            acc = state.get("accuracy")

            train_str = f"Step {step:,} | {pos:,} pos"
            if pl is not None:
                train_str += f" | pl={pl:.4f}"
            if tp is not None:
                train_str += f" | {tp}/s"
            if acc is not None:
                train_str += f" | acc={acc:.1%}"
        else:
            step = 0
            train_str = "Training: no data"

        disk_str = f"Disk: {disk.get('use_pct', '?')} used ({disk.get('avail', '?')} free)"

        # Normal log
        log(f"#{checks} | {gpu_str} | {train_str} | GPU procs={gpu_procs} | {disk_str}")

        # ── Alerts ──

        # GPU temperature
        if "error" not in gpu:
            if gpu["temp"] >= GPU_TEMP_CRITICAL:
                log(f"CRITICAL: GPU temperature {gpu['temp']}°C >= {GPU_TEMP_CRITICAL}°C!", "ALERT")
            elif gpu["temp"] >= GPU_TEMP_WARN:
                log(f"WARNING: GPU temperature {gpu['temp']}°C >= {GPU_TEMP_WARN}°C", "WARN")

        # No GPU processes (training died)
        if gpu_procs == 0:
            log("WARNING: No GPU processes found! Training may have crashed.", "ALERT")
            log("Watchdog should auto-restart. Checking in next cycle...", "WARN")

        # Stall detection
        if state and "error" not in state:
            current_step = state.get("last_train_step", state.get("last_step", 0))
            if current_step == last_step and last_step > 0:
                stall_count += 1
                if stall_count >= 3:
                    log(f"STALL: Training stuck at step {current_step} for {stall_count} checks!", "ALERT")
            else:
                stall_count = 0
            last_step = current_step

            # Low throughput after warmup
            tp = state.get("throughput")
            if tp is not None and current_step > WARMUP_STEPS and tp < MIN_THROUGHPUT:
                log(f"WARNING: Low throughput {tp}/s (expected >= {MIN_THROUGHPUT})", "WARN")

        # Crash log
        if crash:
            log(f"CRASH DETECTED: {crash}", "ALERT")

        # NaN in train log
        train_log_path = OUTPUT_DIR / "train.log"
        if train_log_path.exists():
            try:
                tail = subprocess.check_output(
                    ["tail", "-50", str(train_log_path)],
                    text=True, timeout=5
                )
                nan_lines = [l for l in tail.splitlines() if "NaN" in l or "nan" in l.lower()]
                if nan_lines:
                    log(f"NaN detected in train.log: {nan_lines[-1].strip()}", "WARN")
            except Exception:
                pass

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
