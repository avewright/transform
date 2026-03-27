#!/usr/bin/env python3
"""Quick status check for exp073."""
import subprocess, os, time

r = subprocess.run(['pgrep', '-c', '-f', 'exp073'], capture_output=True, text=True)
print(f'Process count: {r.stdout.strip()}')

log = '/root/transform/outputs/exp073_200m_full_epoch/train.log'
size = os.path.getsize(log)
age = time.time() - os.path.getmtime(log)
print(f'Log: {size} bytes, modified {age:.0f}s ago')

with open(log) as f:
    lines = [l.strip() for l in f if l.strip() and 'Downloading' not in l and '[A' not in l]
    for l in lines[-15:]:
        print(f'  {l}')
