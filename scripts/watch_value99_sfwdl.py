#!/usr/bin/env python3
"""Eval new Value99 snapshots on the Stockfish WDL holdout.

Leaves the trainer process alone. Runs only when a new step_*.pt appears.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def snapshots(run):
    return sorted(run.glob('step_*.pt'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'outputs/value99_50m')
    p.add_argument('--data',type=Path,default=ROOT/'outputs/value99_data_sfwdl')
    p.add_argument('--device',default='cuda')
    p.add_argument('--poll',type=float,default=30)
    a=p.parse_args()
    done_path=a.run/'teacher_done.json'
    done=set(json.loads(done_path.read_text())) if done_path.exists() else set()
    log=a.run/'events.jsonl'
    while True:
        for path in snapshots(a.run):
            key=path.name
            if key in done:continue
            cmd=[sys.executable,'-u',str(ROOT/'scripts/eval_value99_sfwdl.py'),
                 '--checkpoint',str(path),'--data',str(a.data),'--device',a.device,
                 '--append-log',str(log),'--out',str(a.run/f'teacher_{key}.json')]
            print(f'eval {path}',flush=True)
            rc=subprocess.call(cmd)
            if rc==0:
                done.add(key);done_path.write_text(json.dumps(sorted(done),indent=2))
            else:
                print(f'eval failed rc={rc} {path}',flush=True)
        time.sleep(a.poll)


if __name__=='__main__':main()
