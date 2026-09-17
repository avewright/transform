#!/usr/bin/env python3
"""From-scratch 99M scalar value pretraining on audited, compact ChessFENS data."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from chess_value99 import ValueConfig,ValueTransformer


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def atomic(payload,path):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def load_split(root,manifest,split):
    # Disk-backed arrays keep hundreds of millions of positions out of host RAM.
    shards=[s for s in manifest['shards'] if s['split']==split]
    if not shards:raise ValueError(f'Missing {split}')
    key=hashlib.sha256(json.dumps(shards,sort_keys=True).encode()).hexdigest()[:16]
    cache=root/'.mmap'/f'{split}_{key}';cache.mkdir(parents=True,exist_ok=True)
    specs={'packed':(np.uint8,(sum(s['rows'] for s in shards),70)),
           'target':(np.float32,(sum(s['rows'] for s in shards),)),
           'phase':(np.uint8,(sum(s['rows'] for s in shards),))}
    if not (cache/'READY').exists():
        arrays={k:np.lib.format.open_memmap(cache/f'{k}.npy',mode='w+',dtype=d,shape=shape)
                for k,(d,shape) in specs.items()}
        offset=0
        for shard in shards:
            path=root/shard['file']
            if digest(path)!=shard['sha256']:raise ValueError(f'Corrupt data shard {path}')
            with np.load(path) as data:
                for k,array in arrays.items():array[offset:offset+shard['rows']]=data[k]
            offset+=shard['rows']
        for array in arrays.values():array.flush()
        (cache/'READY').touch()
    return tuple(torch.from_numpy(np.load(cache/f'{k}.npy',mmap_mode='c')) for k in specs)


def precision_context(device,precision):
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()



@torch.no_grad()
def evaluate(model,data,device,microbatch,precision="fp32"):
    model.eval();x,y,phase=data;pred=[]
    for start in range(0,len(y),microbatch):
        with precision_context(device,precision):
            pred.append(model(x[start:start+microbatch].to(device)).float().cpu())
    pred=torch.cat(pred);mse=(pred-y).square();mae=(pred-y).abs();decisive=y.abs()>.2
    result=dict(n=len(y),mse=float(mse.mean()),mae=float(mae.mean()),
        decisive_direction=float(((pred[decisive]>0)==(y[decisive]>0)).float().mean()) if decisive.any() else None,
        mean_prediction=float(pred.mean()),mean_target=float(y.mean()),
        phase_mse={str(p):float(mse[phase==p].mean()) for p in [0,1,2] if (phase==p).any()})
    model.train();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/value99_pretrain.json')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/value99_pretrain_v1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text());out=a.out
    if out.exists() and any(out.iterdir()) and not a.resume:raise ValueError('Output already exists')
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device)
    precision=cfg.get('precision','bf16')
    if precision not in ['bf16','fp32']:raise ValueError('precision must be bf16 or fp32')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; training has not started')
    if precision=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('BF16 requires a supported CUDA GPU. Set precision=fp32 explicitly for other hardware.')
    if cfg['batch']<1 or cfg['microbatch']<1:raise ValueError('Batch sizes must be positive')
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=torch.Generator().manual_seed(cfg['seed'])
    def emit(d):
        d=dict(time=time.time(),**d);print(json.dumps(d),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        (out/'status.json').write_text(json.dumps(d,indent=2))
    data_root=ROOT/cfg['data'];path=data_root/'manifest.json'
    if not path.exists():raise ValueError('Data preparation must finish before training')
    manifest=json.loads(path.read_text());train_x,train_y,_=load_split(data_root,manifest,'train')
    val=load_split(data_root,manifest,'validation');val_indices=torch.randperm(len(val[1]),generator=rng)[:cfg['val_rows']]
    val=tuple(t[val_indices] for t in val)
    model=ValueTransformer(ValueConfig(**cfg['model'])).to(device)
    n=sum(p.numel() for p in model.parameters())
    if n!=98920577:raise ValueError(f'Expected 98,920,577 parameters, got {n}')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
    run_manifest=dict(config=cfg,parameters=n,data_manifest_sha256=digest(path),device=str(device),
        torch_version=torch.__version__,initialization='random; no pretrained checkpoint',
        target='side-to-move W-L; scalar output',train_rows=len(train_y),validation_rows=len(val[1]),
        benchmark=a.benchmark,source_sha256={str(p):digest(p) for p in [ROOT/'chess_value99.py',Path(__file__)]})
    step0=0;epoch=0;position=0;order=torch.randperm(len(train_y),generator=rng);seen=0
    if a.resume:
        old=json.loads((out/'manifest.json').read_text())
        if old!=run_manifest:raise ValueError('Resume requires identical source/config/data')
        ckpt=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        step0=ckpt['step'];epoch=ckpt['epoch'];position=ckpt['position'];order=ckpt['order'];seen=ckpt['seen']
        rng.set_state(ckpt['rng']);torch.set_rng_state(ckpt['torch_rng'])
        if device.type=='mps':torch.mps.set_rng_state(ckpt['device_rng'])
        elif device.type=='cuda':torch.cuda.set_rng_state_all(ckpt['device_rng'])
    else:(out/'manifest.json').write_text(json.dumps(run_manifest,indent=2))
    emit(dict(stage='loaded',parameters=n,train_rows=len(train_y),device=str(device)))
    def save(step):
        atomic(dict(arch='value99',config=asdict(model.config),model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),step=step,epoch=epoch,position=position,order=order,seen=seen,
            rng=rng.get_state(),torch_rng=torch.get_rng_state(),
            device_rng=torch.mps.get_rng_state() if device.type=='mps' else torch.cuda.get_rng_state_all() if device.type=='cuda' else None),out/'latest.pt')
    emit(dict(stage='initial_validation',**evaluate(model,val,device,cfg['microbatch'],precision)))
    if not a.benchmark and not a.resume:save(0)
    times=[];started=time.monotonic();total=5 if a.benchmark else cfg['steps']
    for step in range(step0+1,total+1):
        if (out/'STOP').exists():save(step-1);emit(dict(stage='stopped',step=step-1));return
        t=time.monotonic();ids=[]
        while len(ids)<cfg['batch']:
            take=min(cfg['batch']-len(ids),len(order)-position)
            ids.extend(order[position:position+take].tolist());position+=take
            if position==len(order):epoch+=1;order=torch.randperm(len(train_y),generator=rng);position=0
        factor=min(step/cfg['warmup'],1.)
        if step>cfg['warmup']:
            progress=(step-cfg['warmup'])/max(cfg['steps']-cfg['warmup'],1)
            factor=.1+.9*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
        optimizer.zero_grad(set_to_none=True);loss_sum=0.;mse_sum=0.
        for start in range(0,len(ids),cfg['microbatch']):
            ix=ids[start:start+cfg['microbatch']];x=train_x[ix].to(device);y=train_y[ix].to(device)
            with precision_context(device,precision):
                logit=model(x,return_logit=True)
                # Soft target represents expected score, not draw classification.
                loss=F.binary_cross_entropy_with_logits(logit,(y+1)/2)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            (loss*len(ix)/len(ids)).backward();loss_sum+=float(loss.detach())*len(ix)/len(ids)
            mse_sum+=float(((2*logit.detach().sigmoid()-1)-y).square().mean())*len(ix)/len(ids)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();seen+=len(ids)
        if device.type=='mps':torch.mps.synchronize()
        elif device.type=='cuda':torch.cuda.synchronize()
        duration=time.monotonic()-t;times.append(duration)
        if step==1 or step%cfg['log_every']==0 or a.benchmark:
            emit(dict(stage='train',step=step,examples=seen,epoch=epoch,loss=loss_sum,mse=mse_sum,
                grad_norm=float(norm),lr=optimizer.param_groups[0]['lr'],step_seconds=duration,
                peak_vram_gb=torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else None,
                positions_per_s=cfg['batch']/np.mean(times[-20:]),elapsed_s=time.monotonic()-started))
        if not a.benchmark and (step%cfg['save_every']==0 or step==total):
            save(step);emit(dict(stage='validation',step=step,**evaluate(model,val,device,cfg['microbatch'],precision)))
    emit(dict(stage='benchmark_complete' if a.benchmark else 'complete',steps=total,
        steady_positions_per_s=cfg['batch']/np.mean(times[1:] if len(times)>1 else times),
        note='Value fitting is not Elo; no automatic promotion'))


if __name__=='__main__':main()
