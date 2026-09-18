#!/usr/bin/env python3
"""Rebuild the stratified validation mix from an already-prepared dataset."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from value99_valmix import build_valmix,write_valmix,PHASE_NAMES,OUTCOME_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--rows',type=int,default=8192)
    p.add_argument('--seed',type=int,default=294)
    a=p.parse_args()
    manifest=json.loads((a.data/'manifest.json').read_text())
    targets=[];phases=[];black=[]
    for shard in manifest['shards']:
        if shard['split']!='validation':continue
        with np.load(a.data/shard['file']) as data:
            targets.append(data['target']);phases.append(data['phase'])
            black.append(data['black'] if 'black' in data.files else np.zeros(len(data['target']),dtype=np.uint8))
    if not targets:raise SystemExit('No validation shards')
    mix=build_valmix(np.concatenate(targets),np.concatenate(phases),a.rows,a.seed,black=np.concatenate(black))
    meta=write_valmix(a.data,mix)
    print(json.dumps({'filled':meta['filled'],'shortfall':meta['shortfall'],
                      'n':meta['n'],'black_source':meta['black_source'],
                      'phases':PHASE_NAMES,'outcomes':OUTCOME_NAMES},indent=2))
    if any(meta['shortfall'].values()):
        raise SystemExit('Validation mix has shortfall; pool cannot support the requested prior.')


if __name__=='__main__':main()
