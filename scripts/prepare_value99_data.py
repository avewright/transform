#!/usr/bin/env python3
"""Project ChessFENS FEN/WDL columns into compact value-only training shards.

Pinned remote Parquet, column projection, bounded preparation, permanent
canonical board-state split. Never downloads or trains on policy vectors.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi,HfFileSystem
from chess_value99 import canonical_board,pack_board

REPO='Maxlegrec/ChessFENS'
REVISION='0d8d4e6bbda49d42e84c3272be026701659a457c'


def convert(row):
    fen=row.get('fen','');wdl=np.asarray(row.get('wdl',[]),dtype=np.float64)
    if len(fen.split())!=6:return None,'missing_rule_state'
    if wdl.shape!=(3,) or not np.isfinite(wdl).all() or (wdl<0).any() or abs(wdl.sum()-1)>.001:
        return None,'invalid_wdl'
    try:b=chess.Board(fen)
    except ValueError:return None,'invalid_fen'
    if not b.is_valid():return None,'nonstandard_or_invalid'
    if b.is_game_over(claim_draw=False):return None,'terminal'
    c=canonical_board(b)
    key=' '.join(c.fen(en_passant='legal').split()[:4])
    h=int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8).digest(),'little')
    # Source is documented side-to-move W,D,L. Canonical mirroring does NOT negate it.
    wdl/=wdl.sum();target=float(wdl[0]-wdl[2])
    split='validation' if h%1000<5 else ('test' if h%1000<10 else 'train')
    pieces=len(c.piece_map());nonpawn=sum(len(c.pieces(p,color)) for p in [2,3,4,5] for color in [False,True])
    phase=2 if nonpawn<=4 else (0 if pieces>=28 else 1)
    return dict(packed=pack_board(b),target=target,hash=h,split=split,phase=phase,
                black=not b.turn),None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,default=1000000)
    p.add_argument('--per-source-shard',type=int,default=100000)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/value99_data_v1')
    p.add_argument('--seed',type=int,default=294)
    a=p.parse_args();out=a.out
    if out.exists() and any(out.iterdir()):raise ValueError('Refusing nonempty preparation directory')
    out.mkdir(parents=True,exist_ok=True)
    files=sorted(x for x in HfApi().list_repo_files(REPO,repo_type='dataset',revision=REVISION) if x.endswith('.parquet'))
    random.Random(a.seed).shuffle(files);fs=HfFileSystem();seen=set();counts=Counter();phases=Counter();consumed=[];written=[]
    buffers={s:[] for s in ['train','validation','test']};indices=Counter();started=time.time()
    def emit(stage):
        status=dict(stage=stage,time=time.time(),accepted=len(seen),counts=dict(counts),phases=dict(phases),
                    files=len(consumed),elapsed_s=time.time()-started)
        print(json.dumps(status),flush=True);(out/'status.json').write_text(json.dumps(status,indent=2))
    def flush(split):
        rows=buffers[split]
        if not rows:return
        name=f'{split}_{indices[split]:05}.npz';path=out/name
        np.savez_compressed(path,packed=np.stack([r['packed'] for r in rows]),
            target=np.asarray([r['target'] for r in rows],dtype=np.float32),
            hashes=np.asarray([r['hash'] for r in rows],dtype=np.uint64),
            phase=np.asarray([r['phase'] for r in rows],dtype=np.uint8))
        written.append(dict(file=name,split=split,rows=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        indices[split]+=1;buffers[split]=[]
    for name in files:
        count=0
        uri=f'datasets/{REPO}@{REVISION}/{name}'
        with fs.open(uri,'rb',block_size=1<<20) as f:
            pf=pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=4096,columns=['fen','wdl']):
                for row in batch.to_pylist():
                    if count>=a.per_source_shard or len(seen)>=a.rows:break
                    count+=1;counts['raw']+=1
                    parsed,reason=convert(row)
                    if reason:counts[reason]+=1;continue
                    if parsed['hash'] in seen:counts['duplicate']+=1;continue
                    seen.add(parsed['hash']);split=parsed['split'];buffers[split].append(parsed)
                    counts[split]+=1;counts['black_source']+=int(parsed['black']);phases[str(parsed['phase'])]+=1
                    if len(buffers[split])>=50000:flush(split)
                if count>=a.per_source_shard or len(seen)>=a.rows:break
        consumed.append(dict(file=name,raw_rows=count,source_rows=pf.metadata.num_rows));emit('preparing')
        if len(seen)>=a.rows:break
    for split in buffers:flush(split)
    if counts['validation']<128 or counts['train']<1000:raise ValueError('Insufficient valid data')
    manifest=dict(repo=REPO,revision=REVISION,columns=['fen','wdl'],source_shards=len(files),
        seed=a.seed,counts=dict(counts),phases=dict(phases),source_files=consumed,shards=written,
        target='side_to_move_Pwin_minus_Ploss',split='canonical-fen4 blake2b modulo1000: val<5,test<10,train>=10',
        limitations=['No game IDs: position-disjoint, not proven game-disjoint.',
                    'No per-row search depth/visits: teacher quality cannot be filtered by search effort.',
                    'Initial preparation samples shard prefixes across shuffled files; not uniform over all 732M.',
                    'History absent; fullmove index is not used; rule clock clipped at 100.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));emit('complete')


if __name__=='__main__':main()
