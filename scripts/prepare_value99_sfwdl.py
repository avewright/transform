#!/usr/bin/env python3
"""Pack avewright/local-wdl holdout as a Value99 Stockfish teacher suite.

Uses split=1 only. Does not touch the ChessFENS training shards.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from experiments.value99_pretrain import digest
from value99_sfwdl import REPO,REVISION,convert_row
from value99_valmix import build_valmix,write_valmix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/value99_data_sfwdl')
    p.add_argument('--eval-rows',type=int,default=8192)
    p.add_argument('--seed',type=int,default=294)
    p.add_argument('--revision',default=REVISION)
    a=p.parse_args();out=a.out
    if out.exists() and (out/'manifest.json').exists():
        raise ValueError(f'Already prepared: {out}')
    out.mkdir(parents=True,exist_ok=True)
    cache=snapshot_download(repo_id=REPO,repo_type='dataset',revision=a.revision,
                            allow_patterns=['data/*.parquet','teacher.json','manifest.json','README.md'])
    cache=Path(cache)
    files=sorted(cache.glob('data/shard_*.parquet'))
    if not files:raise ValueError('No local-wdl parquet shards')
    rows=[];counts={};started=time.time()
    for path in files:
        table=pq.read_table(path,columns=['board_array','turn','castling','ep_square','wdl',
                                          'wdl_source','split','nodes','label_depth','n_pieces','game_id'])
        cols={name:table.column(name).to_pylist() for name in table.column_names}
        n=len(cols['split'])
        for i in range(n):
            if int(cols['split'][i])!=1:
                counts['train_skipped']=counts.get('train_skipped',0)+1;continue
            rec={k:cols[k][i] for k in cols}
            converted,reason=convert_row(rec)
            if converted is None:
                counts[reason]=counts.get(reason,0)+1;continue
            rows.append(converted);counts['holdout']=counts.get('holdout',0)+1
    if len(rows)<a.eval_rows:
        raise ValueError(f'Holdout too small: {len(rows)} accepted, need {a.eval_rows}')
    packed=np.stack([r['packed'] for r in rows]);target=np.asarray([r['target'] for r in rows],np.float32)
    phase=np.asarray([r['phase'] for r in rows],np.uint8);black=np.asarray([r['black'] for r in rows],np.uint8)
    extra={k:np.asarray([r[k] for r in rows]) for k in ('nodes','label_depth','n_pieces','game_id','white_score')}
    shard=out/'teacher_holdout.npz'
    np.savez(shard,packed=packed,target=target,phase=phase,black=black,**extra)
    mix=build_valmix(target,phase,a.eval_rows,a.seed,black=black)
    write_valmix(out,mix)
    np.save(out/'packed.npy',packed[mix['indices']])
    np.save(out/'target.npy',target[mix['indices']])
    np.save(out/'phase.npy',phase[mix['indices']])
    np.save(out/'black.npy',black[mix['indices']])
    manifest=dict(repo=REPO,revision=a.revision,target='side_to_move_Pwin_minus_Ploss',
                  perspective='local-wdl is white-absolute; converted to STM',
                  split='source split=1 holdout only',
                  holdout_rows=len(rows),eval_rows=int(mix['n']),seed=a.seed,
                  counts=counts,valmix={k:mix[k] for k in ('filled','shortfall','black_source','n')},
                  shards=[{'file':shard.name,'rows':len(rows),'split':'teacher','sha256':digest(shard)}],
                  elapsed_s=time.time()-started,
                  note='Teacher eval only. Not mixed into ChessFENS training.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ('holdout_rows','eval_rows','counts','valmix','elapsed_s')},indent=2))


if __name__=='__main__':main()
