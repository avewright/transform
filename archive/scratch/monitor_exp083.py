#!/usr/bin/env python3
"""Health monitor for exp083 training run.

Polls worker status files, metrics, GPU stats, and prints a live dashboard.
Detects stalls, NaN events, memory pressure, and thermal throttling.

Usage:
    python monitor_exp083.py              # continuous monitoring
    python monitor_exp083.py --once       # single snapshot
    python monitor_exp083.py --interval 10  # custom poll interval
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("outputs/exp083_pretrain_4xa40")
STATUS_DIR = OUTPUT_DIR / "status"
METRICS_DIR = OUTPUT_DIR / "metrics"
NUM_GPUS = 4

# Health thresholds
STALL_THRESHOLD_S = 120   # worker stalled if no status update in 2 min
TEMP_WARN_C = 80          # GPU temp warning
TEMP_CRIT_C = 85          # GPU temp critical
MEM_WARN_PCT = 95         # GPU memory warning


def get_gpu_info():
    """Get GPU stats via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,temperature.gpu,memory.used,memory.total,"
             "utilization.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                gpus.append({
                    "id": int(parts[0]),
                    "name": parts[1],
                    "temp": int(parts[2]),
                    "mem_used": int(parts[3]),
                    "mem_total": int(parts[4]),
                    "util": int(parts[5]),
                    "power": float(parts[6]),
                    "power_limit": float(parts[7]),
                })
        return gpus
    except Exception:
        return []


def read_worker_status(worker_id):
    """Read worker status JSON."""
    path = STATUS_DIR / f"worker{worker_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_last_metrics(worker_id, n=5):
    """Read last N metrics records from JSONL."""
    path = METRICS_DIR / f"worker{worker_id}_metrics.jsonl"
    if not path.exists():
        return []
    try:
        lines = path.read_text().strip().split("\n")
        return [json.loads(l) for l in lines[-n:]]
    except Exception:
        return []


def count_events(worker_id, event_type=None):
    """Count events in worker event log."""
    path = STATUS_DIR / f"worker{worker_id}.events.jsonl"
    if not path.exists():
        return 0
    try:
        count = 0
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            if event_type is None:
                count += 1
            else:
                rec = json.loads(line)
                if rec.get("event") == event_type:
                    count += 1
        return count
    except Exception:
        return 0


def read_training_log():
    """Read training_log.json for eval results."""
    path = OUTPUT_DIR / "training_log.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def check_health(statuses, gpus):
    """Return list of health warnings."""
    warnings = []
    now = datetime.utcnow()

    for i in range(NUM_GPUS):
        s = statuses[i]
        if s is None:
            warnings.append(f"  ⚠ W{i}: No status file (not started?)")
            continue

        # Stall detection
        try:
            ts = datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
            age_s = (now - ts).total_seconds()
            if age_s > STALL_THRESHOLD_S and s.get("state") == "training":
                warnings.append(f"  🔴 W{i}: STALLED — no update in {age_s:.0f}s")
        except Exception:
            pass

        # Fatal error
        if s.get("state") == "fatal_error":
            warnings.append(f"  🔴 W{i}: FATAL ERROR — {s.get('error', '?')}")

        # NaN events
        nan_count = count_events(i, "nan_loss")
        if nan_count > 0:
            warnings.append(f"  ⚠ W{i}: {nan_count} NaN loss events")

    for g in gpus:
        if g["temp"] >= TEMP_CRIT_C:
            warnings.append(f"  🔴 GPU {g['id']}: CRITICAL TEMP {g['temp']}°C")
        elif g["temp"] >= TEMP_WARN_C:
            warnings.append(f"  ⚠ GPU {g['id']}: HIGH TEMP {g['temp']}°C")

        mem_pct = g["mem_used"] / g["mem_total"] * 100
        if mem_pct >= MEM_WARN_PCT:
            warnings.append(f"  ⚠ GPU {g['id']}: MEM {mem_pct:.0f}% "
                            f"({g['mem_used']}/{g['mem_total']} MiB)")

    return warnings


def print_dashboard(once=False):
    """Print a single dashboard snapshot."""
    os.system("clear" if not once else ":")

    print(f"{'='*78}")
    print(f" EXP083 HEALTH MONITOR — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*78}")

    # GPU hardware status
    gpus = get_gpu_info()
    if gpus:
        print(f"\n  GPU HARDWARE:")
        print(f"  {'ID':>2} {'Name':>10} {'Temp':>5} {'Mem':>12} {'Util':>5} {'Power':>12}")
        print(f"  {'—'*52}")
        for g in gpus:
            mem_str = f"{g['mem_used']}/{g['mem_total']}M"
            pwr_str = f"{g['power']:.0f}/{g['power_limit']:.0f}W"
            print(f"  {g['id']:>2} {g['name'][-8:]:>10} {g['temp']:>4}°C "
                  f"{mem_str:>12} {g['util']:>4}% {pwr_str:>12}")

    # Worker status
    statuses = [read_worker_status(i) for i in range(NUM_GPUS)]
    print(f"\n  WORKERS:")
    print(f"  {'W':>2} {'State':>12} {'Step':>8} {'Positions':>14} "
          f"{'Best':>6} {'pos/s':>7} {'ETA':>6} {'pl':>7} {'vl':>7}")
    print(f"  {'—'*78}")

    total_throughput = 0
    for i in range(NUM_GPUS):
        s = statuses[i]
        if s is None:
            print(f"  {i:>2} {'—':>12}")
            continue

        state = s.get("state", "?")[:12]
        step = s.get("global_step", 0)
        pos = s.get("positions_seen", 0)
        best = s.get("best_acc", 0.0)
        tp = s.get("throughput_pos_s", 0)
        eta = s.get("eta_h", 0)
        pl = s.get("ema_policy_loss", 0)
        vl = s.get("ema_value_loss", 0)
        total_throughput += tp

        print(f"  {i:>2} {state:>12} {step:>8,} {pos:>14,} "
              f"{best:>5.1%} {tp:>6,} {eta:>5.1f}h "
              f"{pl:>7.4f} {vl:>7.4f}")

    if total_throughput > 0:
        print(f"\n  Combined throughput: {total_throughput:,} pos/s")

    # Latest eval results
    tlog = read_training_log()
    evals = [r for r in tlog if r.get("type") == "eval"]
    if evals:
        print(f"\n  LAST 3 EVALS:")
        for ev in evals[-3:]:
            print(f"    step={ev['step']:>7,} "
                  f"acc={ev.get('accuracy', 0):.1%} "
                  f"top3={ev.get('top3_accuracy', 0):.1%} "
                  f"sf_rank={ev.get('mean_sf_rank', 0):.1f} "
                  f"val={ev.get('value_accuracy', 0):.1%}")

    # Health checks
    warnings = check_health(statuses, gpus)
    if warnings:
        print(f"\n  HEALTH WARNINGS:")
        for w in warnings:
            print(w)
    else:
        print(f"\n  ✅ All systems healthy")

    # Disk usage
    try:
        ckpt_size = sum(
            f.stat().st_size for f in OUTPUT_DIR.rglob("*.pt") if f.is_file()
        ) / 1e9
        log_size = sum(
            f.stat().st_size for f in OUTPUT_DIR.rglob("*.jsonl") if f.is_file()
        ) / 1e6
        print(f"\n  DISK: checkpoints={ckpt_size:.1f}GB, logs={log_size:.1f}MB")
    except Exception:
        pass

    print(f"\n{'='*78}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Single snapshot")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between polls")
    args = parser.parse_args()

    if not OUTPUT_DIR.exists():
        print(f"Output directory not found: {OUTPUT_DIR}")
        print("Training may not have started yet. Waiting...")

    if args.once:
        print_dashboard(once=True)
        return

    while True:
        try:
            print_dashboard()
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break


if __name__ == "__main__":
    main()
