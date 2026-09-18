#!/usr/bin/env python3
"""Project ChessFENS FEN/WDL columns into compact value-only training shards.

Pinned remote Parquet, column projection, bounded preparation, permanent
canonical board-state split. Never downloads or trains on policy vectors.
Resumable: rerun the same command on a partial output directory.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
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
from value99_valmix import (build_valmix,cell_counts,cells_needed,outcome_bucket,
                            write_valmix,PHASE_NAMES,OUTCOME_NAMES)

REPO='Maxlegrec/ChessFENS'
REVISION='0d8d4e6bbda49d42e84c3272be026701659a457c'
HASH_DTYPE=np.uint64
FLUSH_ROWS=50000
TARGET_EDGES=np.linspace(-1,1,21)


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


def per_source_budget(rows,n_files,cap):
    """Spread reads across shards; raise the cap if it cannot reach --rows."""
    if n_files<1:raise ValueError('No source parquet files')
    need=max(1,math.ceil(rows*1.25/n_files))
    return max(cap,need)


def target_histogram(values):
    if not values:return {'bins':TARGET_EDGES.tolist(),'counts':[0]*(len(TARGET_EDGES)-1),'mean':None,'std':None}
    hist=np.histogram(np.asarray(values,dtype=np.float64),bins=TARGET_EDGES)[0]
    arr=np.asarray(values,dtype=np.float64)
    return {'bins':TARGET_EDGES.tolist(),'counts':hist.tolist(),'mean':float(arr.mean()),'std':float(arr.std())}


class HashSeen:
    """O(1) membership with a disk-backed uint64 log so resume stays bounded."""
    def __init__(self,path,buffer=65536):
        self.path=path;self.seen=set();self.handle=None;self.pending=[];self.buffer=buffer

    def open(self,rebuild=False):
        if rebuild and self.path.exists():
            raw=np.fromfile(self.path,dtype=HASH_DTYPE)
            self.seen=set(int(x) for x in raw)
        self.handle=self.path.open('ab')
        return len(self.seen)

    def add(self,value):
        if value in self.seen:return False
        self.seen.add(value);self.pending.append(value)
        if len(self.pending)>=self.buffer:self.flush()
        return True

    def flush(self):
        if self.handle and self.pending:
            self.handle.write(np.asarray(self.pending,dtype=HASH_DTYPE).tobytes())
            self.handle.flush();self.pending.clear()

    def close(self):
        self.flush()
        if self.handle:self.handle.close();self.handle=None

    def __len__(self):
        return len(self.seen)

    def __contains__(self,value):
        return value in self.seen


def load_partial(out):
    ckpt=out/'checkpoint.json'
    if not ckpt.exists():return None
    state=json.loads(ckpt.read_text())
    if state.get('stage')=='complete' or (out/'manifest.json').exists():
        raise ValueError(f'Preparation already complete in {out}')
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,default=1000000)
    p.add_argument('--per-source-shard',type=int,default=100000)
    p.add_argument('--valmix-rows',type=int,default=8192)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/value99_data_v1')
    p.add_argument('--seed',type=int,default=294)
    a=p.parse_args();out=a.out
    if a.rows<1000 or a.per_source_shard<1 or a.valmix_rows<128:
        raise ValueError('rows, per-source-shard and valmix-rows must be positive')
    partial=load_partial(out) if out.exists() else None
    if out.exists() and any(out.iterdir()) and partial is None and not (out/'checkpoint.json').exists():
        raise ValueError('Refusing nonempty preparation directory without checkpoint.json')
    out.mkdir(parents=True,exist_ok=True)
    files=sorted(x for x in HfApi().list_repo_files(REPO,repo_type='dataset',revision=REVISION) if x.endswith('.parquet'))
    random.Random(a.seed).shuffle(files)
    budget=per_source_budget(a.rows,len(files),a.per_source_shard)
    fs=HfFileSystem()
    seen=HashSeen(out/'seen_hashes.u64')
    counts=Counter();phases=Counter();consumed=[];written=[]
    target_counts=np.zeros(len(TARGET_EDGES)-1,dtype=np.int64)
    target_sum=0.;target_sumsq=0.;target_n=0
    buffers={s:[] for s in ['train','validation','test']};indices=Counter();started=time.time()
    done_files=set()
    prep_stage='train'
    if partial:
        if partial.get('revision')!=REVISION or partial.get('seed')!=a.seed or partial.get('rows')!=a.rows:
            raise ValueError('Resume requires identical revision/seed/--rows')
        if int(partial.get('valmix_rows',a.valmix_rows))!=a.valmix_rows:
            raise ValueError('Resume requires identical --valmix-rows')
        counts.update(partial.get('counts',{}));phases.update(partial.get('phases',{}))
        consumed=list(partial.get('source_files',[]));written=list(partial.get('shards',[]))
        done_files={x['file'] for x in consumed}
        target_counts=np.asarray(partial.get('target_counts',target_counts),dtype=np.int64)
        target_sum=float(partial.get('target_sum',0));target_sumsq=float(partial.get('target_sumsq',0))
        target_n=int(partial.get('target_n',0))
        prep_stage=partial.get('prep_stage','train')
        for shard in written:
            split=shard['split'];name=shard['file']
            indices[split]=max(indices[split],int(name.split('_')[1].split('.')[0])+1)
        seen.open(rebuild=True)
        if len(seen)!=int(partial.get('accepted',len(seen))):
            raise ValueError(f'Hash log size {len(seen)} != checkpoint accepted {partial.get("accepted")}')
        started=time.time()-float(partial.get('elapsed_s',0))
    else:
        seen.open(rebuild=False)
    def emit(stage):
        status=dict(stage=stage,time=time.time(),accepted=len(seen),counts=dict(counts),phases=dict(phases),
                    files=len(consumed),elapsed_s=time.time()-started,per_source_shard=budget,
                    source_shards=len(files),rows_requested=a.rows)
        print(json.dumps(status),flush=True);(out/'status.json').write_text(json.dumps(status,indent=2))
    def checkpoint():
        seen.flush()
        state=dict(stage='preparing',repo=REPO,revision=REVISION,seed=a.seed,rows=a.rows,
                   valmix_rows=a.valmix_rows,accepted=len(seen),counts=dict(counts),phases=dict(phases),
                   source_files=consumed,shards=written,
                   target_counts=target_counts.tolist(),target_sum=target_sum,target_sumsq=target_sumsq,
                   target_n=target_n,elapsed_s=time.time()-started,per_source_shard=budget,
                   prep_stage=prep_stage)
        tmp=out/'checkpoint.json.tmp';tmp.write_text(json.dumps(state));tmp.replace(out/'checkpoint.json')
    def flush(split):
        rows=buffers[split]
        if not rows:return
        name=f'{split}_{indices[split]:05}.npz';path=out/name
        packed=np.stack([r['packed'] for r in rows])
        np.savez_compressed(path,packed=packed,
            target=np.asarray([r['target'] for r in rows],dtype=np.float32),
            hashes=np.asarray([r['hash'] for r in rows],dtype=np.uint64),
            phase=np.asarray([r['phase'] for r in rows],dtype=np.uint8),
            black=np.asarray([r['black'] for r in rows],dtype=np.uint8))
        written.append(dict(file=name,split=split,rows=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        indices[split]+=1;buffers[split]=[]
    emit('preparing' if partial else 'start')
    for name in files:
        if len(seen)>=a.rows:break
        if name in done_files:continue
        count=0
        uri=f'datasets/{REPO}@{REVISION}/{name}'
        with fs.open(uri,'rb',block_size=1<<20) as f:
            pf=pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=4096,columns=['fen','wdl']):
                for row in batch.to_pylist():
                    if count>=budget or len(seen)>=a.rows:break
                    count+=1;counts['raw']+=1
                    parsed,reason=convert(row)
                    if reason:counts[reason]+=1;continue
                    if parsed['hash'] in seen:counts['duplicate']+=1;continue
                    seen.add(parsed['hash']);split=parsed['split'];buffers[split].append(parsed)
                    counts[split]+=1;counts['black_source']+=int(parsed['black']);phases[str(parsed['phase'])]+=1
                    target_n+=1;target_sum+=parsed['target'];target_sumsq+=parsed['target']**2
                    target_counts[int(np.clip(np.digitize(parsed['target'],TARGET_EDGES,right=True)-1,0,len(target_counts)-1))]+=1
                    if len(buffers[split])>=FLUSH_ROWS:flush(split)
                if count>=budget or len(seen)>=a.rows:break
            source_rows=pf.metadata.num_rows
        consumed.append(dict(file=name,raw_rows=count,source_rows=source_rows));checkpoint();emit('preparing')
    def val_pool_counts():
        have=cell_counts([],[])
        for shard in written:
            if shard['split']!='validation':continue
            with np.load(out/shard['file']) as data:
                for cell,n in cell_counts(data['target'],data['phase']).items():have[cell]+=n
        if buffers['validation']:
            for cell,n in cell_counts([r['target'] for r in buffers['validation']],
                                      [r['phase'] for r in buffers['validation']]).items():
                have[cell]+=n
        return have
    need=cells_needed(a.valmix_rows)
    short={cell:max(0,need[cell]-val_pool_counts().get(cell,0)) for cell in need}
    accepted=len(seen)
    if accepted>=a.rows and any(short.values()):
        prep_stage='valmix';emit('valmix_backfill')
        consumed_at={item['file']:item['raw_rows'] for item in consumed}
        for name in files:
            if not any(short.values()):break
            skip=consumed_at.get(name,0);count=0
            uri=f'datasets/{REPO}@{REVISION}/{name}'
            with fs.open(uri,'rb',block_size=1<<20) as f:
                pf=pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=4096,columns=['fen','wdl']):
                    for row in batch.to_pylist():
                        if not any(short.values()):break
                        count+=1
                        if count<=skip:continue
                        counts['raw']+=1
                        parsed,reason=convert(row)
                        if reason:counts[reason]+=1;continue
                        if parsed['hash'] in seen:counts['duplicate']+=1;continue
                        if parsed['split']!='validation':continue
                        cell=(parsed['phase'],outcome_bucket(parsed['target']))
                        if short.get(cell,0)<=0:continue
                        seen.add(parsed['hash']);buffers['validation'].append(parsed)
                        counts['validation']+=1;counts['black_source']+=int(parsed['black'])
                        phases[str(parsed['phase'])]+=1;counts['valmix_backfill']+=1
                        short[cell]-=1
                        if len(buffers['validation'])>=FLUSH_ROWS:flush('validation')
                    if not any(short.values()):break
                source_rows=pf.metadata.num_rows
            consumed_at[name]=max(skip,count)
            found=next((item for item in consumed if item['file']==name),None)
            record=dict(file=name,raw_rows=consumed_at[name],source_rows=source_rows)
            if found:found.update(record)
            else:consumed.append(record)
            checkpoint();emit('valmix_backfill')
    for split in buffers:flush(split)
    checkpoint()
    accepted=len(seen)
    if accepted<a.rows:
        raise ValueError(f'Requested {a.rows} unique accepted rows, got {accepted}. Not treating this as a complete dataset.')
    if counts['validation']<128 or counts['train']<1000:raise ValueError('Insufficient valid data')
    val_targets=[];val_phases=[];val_black=[];all_targets=[]
    for shard in written:
        with np.load(out/shard['file']) as data:
            all_targets.append(data['target'])
            if shard['split']=='validation':
                val_targets.append(data['target']);val_phases.append(data['phase'])
                val_black.append(data['black'] if 'black' in data.files else np.zeros(len(data['target']),dtype=np.uint8))
    hist=target_histogram(np.concatenate(all_targets) if all_targets else [])
    if not val_targets:raise ValueError('No validation rows for the mix')
    mix=build_valmix(np.concatenate(val_targets),np.concatenate(val_phases),a.valmix_rows,a.seed,
                     black=np.concatenate(val_black))
    mix_meta=write_valmix(out,mix)
    pool={f'{PHASE_NAMES[p]}_{OUTCOME_NAMES[o]}':int(c) for (p,o),c in val_pool_counts().items()}
    if any(mix['shortfall'].values()):
        missing={k:v for k,v in mix['shortfall'].items() if v}
        raise ValueError(f'Validation mix shortfall {missing}; pool={pool}. Not treating this as a complete dataset.')
    manifest=dict(repo=REPO,revision=REVISION,columns=['fen','wdl'],source_shards=len(files),
        seed=a.seed,rows_requested=a.rows,accepted=accepted,per_source_shard=budget,
        valmix_rows=a.valmix_rows,valmix=mix_meta,valmix_pool=pool,
        counts=dict(counts),phases=dict(phases),source_files=consumed,shards=written,
        target=hist | {'name':'side_to_move_Pwin_minus_Ploss'},
        split='canonical-fen4 blake2b modulo1000: val<5,test<10,train>=10; val scored with stratified mix',
        limitations=['No game IDs: position-disjoint, not proven game-disjoint.',
                    'No per-row search depth/visits: teacher quality cannot be filtered by search effort.',
                    'Preparation samples bounded prefixes across shuffled files; not uniform over every row in 732M.',
                    'History absent; fullmove index is not used; rule clock clipped at 100.',
                    'Validation mix is a diagnostic prior (25/45/30 phase, 30/40/30 outcome), not a playing book.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    seen.close()
    (out/'checkpoint.json').write_text(json.dumps(dict(stage='complete',accepted=accepted,rows=a.rows)))
    emit('complete')


if __name__=='__main__':main()
