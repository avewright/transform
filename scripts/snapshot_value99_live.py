#!/usr/bin/env python3
"""Freeze flushed train shards and collect a disjoint validation mix.

Does not write into the live 50M preparation directory.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi,HfFileSystem
from scripts.prepare_value99_data import REPO,REVISION,convert
from value99_valmix import build_valmix,cells_needed,outcome_bucket,write_valmix,PHASE_NAMES,OUTCOME_NAMES

SRC=ROOT/'outputs'/'value99_data_50m'
DST=ROOT/'outputs'/'value99_data_live'
VALMIX_ROWS=8192
SEED=294


def main():
    ckpt=json.loads((SRC/'checkpoint.json').read_text())
    train=[s for s in ckpt['shards'] if s['split']=='train']
    if not train:raise SystemExit('No flushed train shards yet')
    DST.mkdir(parents=True,exist_ok=True)
    written=[]
    for shard in train:
        src=SRC/shard['file'];dst=DST/shard['file']
        if dst.exists() or dst.is_symlink():dst.unlink()
        os.link(src,dst)
        written.append(shard)
    need=cells_needed(VALMIX_ROWS)
    have={(p,o):0 for p in range(3) for o in range(3)}
    buf=[];val_written=[];idx=0
    files=sorted(x for x in HfApi().list_repo_files(REPO,repo_type='dataset',revision=REVISION) if x.endswith('.parquet'))
    fs=HfFileSystem()
    def flush():
        nonlocal idx,buf
        if not buf:return
        name=f'validation_{idx:05}.npz';path=DST/name
        np.savez_compressed(path,packed=np.stack([r['packed'] for r in buf]),
            target=np.asarray([r['target'] for r in buf],dtype=np.float32),
            hashes=np.asarray([r['hash'] for r in buf],dtype=np.uint64),
            phase=np.asarray([r['phase'] for r in buf],dtype=np.uint8),
            black=np.asarray([r['black'] for r in buf],dtype=np.uint8))
        val_written.append(dict(file=name,split='validation',rows=len(buf),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        idx+=1;buf=[]
    for name in files:
        if all(have[c]>=need[c] for c in need):break
        uri=f'datasets/{REPO}@{REVISION}/{name}'
        with fs.open(uri,'rb',block_size=1<<20) as f:
            pf=pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=4096,columns=['fen','wdl']):
                for row in batch.to_pylist():
                    if all(have[c]>=need[c] for c in need):break
                    parsed,reason=convert(row)
                    if reason or parsed['split']!='validation':continue
                    cell=(parsed['phase'],outcome_bucket(parsed['target']))
                    if have[cell]>=need[cell]:continue
                    buf.append(parsed);have[cell]+=1
                    if len(buf)>=50000:flush()
                if all(have[c]>=need[c] for c in need):break
    flush()
    val_targets=[];val_phases=[];val_black=[]
    for shard in val_written:
        with np.load(DST/shard['file']) as data:
            val_targets.append(data['target']);val_phases.append(data['phase']);val_black.append(data['black'])
    if not val_targets:raise SystemExit('Failed to collect validation rows')
    mix=build_valmix(np.concatenate(val_targets),np.concatenate(val_phases),VALMIX_ROWS,SEED,
                     black=np.concatenate(val_black))
    if any(mix['shortfall'].values()):
        raise SystemExit(f'Val mix shortfall {mix["shortfall"]}')
    mix_meta=write_valmix(DST,mix)
    shards=written+val_written
    counts=dict(train=sum(s['rows'] for s in written),validation=sum(s['rows'] for s in val_written),test=0)
    manifest=dict(repo=REPO,revision=REVISION,columns=['fen','wdl'],seed=SEED,
        rows_requested=counts['train']+counts['validation'],accepted=counts['train']+counts['validation'],
        counts=counts,valmix_rows=VALMIX_ROWS,valmix=mix_meta,shards=shards,
        snapshot_of=str(SRC),note='Live snapshot of flushed 50M train shards plus a fresh hash-val mix. Test split omitted.',
        target={'name':'side_to_move_Pwin_minus_Ploss'},
        split='canonical-fen4 blake2b modulo1000; train from live 50M flush, val collected separately')
    (DST/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(train=counts['train'],validation=counts['validation'],valmix=mix_meta['filled'],
                          shortfall=mix_meta['shortfall']),indent=2))


if __name__=='__main__':main()
