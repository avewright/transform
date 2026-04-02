#!/usr/bin/env python3
"""Quick status dashboard for all running experiments."""
import json
import pathlib
import time

def safe_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

# Harvest
s = safe_json("outputs/exp087_full_legal_harvest/status.json")
if s and "stats" in s:
    hw = s["stats"]["written_positions"]
    hr = s["stats"]["records_per_min"]
    remain = (100000 - hw) / max(hr, 1)
    stopped = s.get("stop_requested", False)
    label = "DONE" if hw >= 100000 or stopped else f"~{remain:.0f}min left"
    print(f"HARVEST: {hw}/100000 ({hr:.0f}/min) {label}")

# Training exp092
s2 = safe_json("outputs/exp092_full_legal_confkl_top8/status.json")
if s2 and s2.get("last_eval"):
    step = s2["train_steps"]
    done = s2["done"]
    loss = s2["last_eval"]["loss"]
    acc = s2["last_eval"]["acc"]
    top3 = s2["last_eval"].get("top3", 0)
    print(f"EXP092:  step={step} done={done} loss={loss:.4f} acc={acc:.2%} top3={top3:.2%}")

# Relabel
d8_dir = pathlib.Path("outputs/exp087_relabeled_d8/dataset")
count = 0
if d8_dir.exists():
    for p in sorted(d8_dir.glob("positions_*.jsonl")):
        with open(p) as f:
            count += sum(1 for _ in f)
print(f"RELABEL: {count} positions relabeled (d8)")

# Training exp093 (if started)
s3 = safe_json("outputs/exp093_ema_curriculum_d8/status.json")
if s3 and s3.get("train_steps") is not None:
    step = s3["train_steps"]
    done = s3.get("done", False)
    if "eval_live" in s3:
        la = s3["eval_live"]["acc"]
        ea = s3["eval_ema"]["acc"]
        print(f"EXP093:  step={step} done={done} live_acc={la:.2%} ema_acc={ea:.2%}")
    else:
        print(f"EXP093:  step={step} done={done}")
