#!/usr/bin/env python3
"""Pack official SF19 WDL train rows (split=0) for Value99 mix.

Holdout split=1 stays out. Incremental: rerun or --watch to pick up new
HuggingFace shards without rewriting existing npz files.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from data_loader import _maybe_load_hf_token_from_env
from experiments.value99_pretrain import digest
from value99_sfwdl import REPO,convert_train_row

FLUSH=50000


def source_key(kind,src):
    path=Path(src)
    if kind=='parquet' or path.suffix=='.parquet' or str(src).startswith('parquet:'):
        return f'parquet:{path.name}'
    if str(src).startswith('pt:'):
        return str(src)
    return f'pt:{path.parent.name}'


def normalize_done(done):
    out=set()
    for item in done:
        text=str(item)
        if text.startswith('parquet:') or text.endswith('.parquet'):
            out.add(f'parquet:{Path(text.split(":",1)[-1]).name}')
        elif text.startswith('pt:'):
            out.add(text)
        else:
            out.add(f'pt:{Path(text).parent.name}')
    return out


def write_manifest(out,repo,written,counts,started,note=None):
    if not written:return None
    manifest=dict(repo=repo,target='side_to_move_Pwin_minus_Ploss',
                  split='source split=0 only; holdout excluded',
                  train_rows=sum(s['rows'] for s in written),counts=counts,
                  shards=written,elapsed_s=time.time()-started,
                  note=note or 'Sprinkle into Value99. Do not replace ChessFENS.')
    tmp=out/'manifest.json.tmp'
    tmp.write_text(json.dumps(manifest,indent=2))
    tmp.replace(out/'manifest.json')
    return manifest


def flush(rows,out,written):
    if not rows:return written
    idx=len(written)
    path=out/f'train_{idx:05d}.npz'
    np.savez(path,packed=np.stack([r['packed'] for r in rows]),
             target=np.asarray([r['target'] for r in rows],np.float32),
             phase=np.asarray([r['phase'] for r in rows],np.uint8))
    written.append({'file':path.name,'rows':len(rows),'split':'train','sha256':digest(path)})
    rows.clear()
    return written


def accept(rec,rows,counts):
    converted,reason=convert_train_row(rec)
    if converted is None:
        counts[reason]=counts.get(reason,0)+1;return
    rows.append(converted);counts['train']=counts.get('train',0)+1


def pull_cache(repo,refresh):
    _maybe_load_hf_token_from_env()
    kwargs=dict(repo_id=repo,repo_type='dataset',allow_patterns=['data/*.parquet'])
    if not refresh:
        try:
            return Path(snapshot_download(**kwargs,local_files_only=True))
        except Exception:
            pass
    return Path(snapshot_download(**kwargs))


def pack_once(out,inbox,repo,refresh,started):
    state_path=out/'prepare.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'done':[],'shards':[],'counts':{}}
    done=normalize_done(state['done']);written=list(state['shards']);counts=dict(state.get('counts') or {})
    rows=[]
    cache=pull_cache(repo,refresh)
    sources=[]
    remote=set()
    for path in sorted(cache.glob('data/shard_*.parquet')):
        sources.append(('parquet',str(path)));remote.add(path.name)
    if inbox.exists():
        for sh in sorted(p for p in inbox.glob('shard_*') if (p/'READY').exists() and (p/'soft_cache.pt').exists()):
            if f'{sh.name}.parquet' in remote:
                continue
            sources.append(('pt',str(sh/'soft_cache.pt')))
    new_sources=0
    for kind,src in sources:
        key=source_key(kind,src)
        if key in done:continue
        new_sources+=1
        if kind=='parquet':
            table=pq.read_table(src,columns=['board_array','turn','castling','ep_square','wdl','wdl_source','split'])
            cols={n:table.column(n).to_pylist() for n in table.column_names}
            for i in range(len(cols['split'])):
                accept({k:cols[k][i] for k in cols},rows,counts)
                if len(rows)>=FLUSH:
                    written=flush(rows,out,written)
                    write_manifest(out,repo,written,counts,started)
        else:
            data=torch.load(src,map_location='cpu',weights_only=False)
            n=int(data['move_idx'].shape[0])
            for i in range(n):
                rec=dict(board_array=data['board_array'][i].numpy(),turn=int(data['turn'][i]),
                         castling=int(data['castling'][i]),ep_square=int(data['ep_square'][i]),
                         wdl=data['wdl'][i].numpy(),wdl_source=int(data['wdl_source'][i]),
                         split=int(data['split'][i]))
                accept(rec,rows,counts)
                if len(rows)>=FLUSH:
                    written=flush(rows,out,written)
                    write_manifest(out,repo,written,counts,started)
            del data
        done.add(key)
        state=dict(done=sorted(done),shards=written,counts=counts)
        state_path.write_text(json.dumps(state))
        print(json.dumps({'src':Path(src).name,'key':key,'train':counts.get('train',0),
                          'holdout':counts.get('holdout',0),'remote':len(remote)}),flush=True)
    written=flush(rows,out,written)
    if written:
        write_manifest(out,repo,written,counts,started)
        state_path.write_text(json.dumps(dict(done=sorted(done),shards=written,counts=counts)))
    return dict(train_rows=sum(s['rows'] for s in written),counts=counts,
                shards=len(written),sources=len(sources),new_sources=new_sources,
                remote=len(remote),refresh=refresh)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/value99_data_sfwdl_train')
    p.add_argument('--inbox',type=Path,default=ROOT/'outputs/sf19_wdl/eco/inbox')
    p.add_argument('--repo',default=REPO)
    p.add_argument('--refresh',action='store_true',
                   help='Re-query HuggingFace instead of trusting the local snapshot')
    p.add_argument('--watch',action='store_true',help='Loop: pull new HF shards, then sleep')
    p.add_argument('--every',type=float,default=1800,help='Seconds between HF refreshes')
    a=p.parse_args();out=a.out
    out.mkdir(parents=True,exist_ok=True)
    lock_fd=os.open(out/'prepare.lock',os.O_CREAT|os.O_RDWR)
    fcntl.flock(lock_fd,fcntl.LOCK_EX)
    started=time.time()
    try:
        while True:
            refresh=a.refresh or a.watch
            result=pack_once(out,a.inbox,a.repo,refresh,started)
            if not result['train_rows'] and not a.watch:
                raise ValueError('No SF WDL train rows')
            print(json.dumps(result,indent=2),flush=True)
            if not a.watch:break
            print(json.dumps({'stage':'sleep','every':a.every,'train_rows':result['train_rows']}),flush=True)
            time.sleep(max(30.,float(a.every)))
    finally:
        fcntl.flock(lock_fd,fcntl.LOCK_UN)
        os.close(lock_fd)


if __name__=='__main__':main()
