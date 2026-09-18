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
import torch._dynamo
from torch.nn import functional as F
from chess_value99 import ValueConfig,ValueTransformer
from polar_normuon import SingleDeviceNorMuonPolarWithAuxAdam,unwrap_compiled
from value99_valmix import PHASE_NAMES,OUTCOME_NAMES,build_valmix,load_valmix

ADAM_HINTS=('embed','norm','bn','rel_bias','ma_gating','gating')


def build_optimizer(model,cfg):
    name=cfg.get('optimizer','adamw');wd=float(cfg.get('weight_decay',.01))
    if name=='polar_normuon':
        muon,adam,muon_n,adam_n=[],[],0,0
        for n,p in model.named_parameters():
            if not p.requires_grad:continue
            if any(h in n for h in ADAM_HINTS) or p.ndim<2:
                adam.append(p);adam_n+=p.numel()
            else:
                muon.append(p);muon_n+=p.numel()
        if not muon:raise ValueError('Polar-NorMuon found no 2D trunk parameters')
        opt=SingleDeviceNorMuonPolarWithAuxAdam([
            dict(params=muon,use_muon=True,lr=float(cfg.get('muon_lr',.02)),weight_decay=wd,momentum=.95,beta2=.95),
            dict(params=adam,use_muon=False,lr=float(cfg.get('adam_lr',3e-4)),betas=(.9,.95),weight_decay=wd),
        ],cautious_wd=True,compile_polar=True)
        for group in opt.param_groups:group['initial_lr']=group['lr']
        return opt,dict(name=name,muon=muon_n,adam=adam_n)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.get('lr',2e-4),weight_decay=wd)
    for group in opt.param_groups:group['initial_lr']=group['lr']
    return opt,dict(name='adamw')


def group_lrs(optimizer):
    return {('muon' if g.get('use_muon') else 'adam'):float(g['lr']) for g in optimizer.param_groups}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def atomic(payload,path):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def resume_compatible(old,new):
    """Allow adding an SF WDL sprinkle to a live ChessFENS run."""
    if old==new:return True
    skip={'config','source_sha256','sfwdl_rows'}
    if {k:old[k] for k in old if k not in skip}!={k:new[k] for k in new if k not in skip}:
        return False
    ocfg={k:v for k,v in old.get('config',{}).items() if not str(k).startswith('sfwdl')}
    ncfg={k:v for k,v in new.get('config',{}).items() if not str(k).startswith('sfwdl')}
    return ocfg==ncfg


def take_ids(order,position,n,rng,length):
    ids=[];wraps=0
    while len(ids)<n:
        take=min(n-len(ids),len(order)-position)
        ids.extend(order[position:position+take].tolist());position+=take
        if position==len(order):
            order=torch.randperm(length,generator=rng);position=0;wraps+=1
    return ids,order,position,wraps


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


def sf_stamp(root):
    path=Path(root)/'manifest.json'
    if not path.exists():return None
    st=path.stat()
    return (int(st.st_mtime_ns),int(st.st_size))


def should_reload_sf(stamp,last_stamp,now,last_reload,reload_s,have_rows):
    if stamp is None or stamp==last_stamp:return False
    if have_rows and (now-last_reload)<reload_s:return False
    return True


def mix_n(batch,mix,mix_n_cfg,rows):
    n=int(mix_n_cfg or 0)
    if not n and mix and rows:
        n=int(round(batch*float(mix)))
    if rows==0:n=0
    return max(0,min(n,batch-1))


def precision_context(device,precision):
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()



@torch.no_grad()
def evaluate(model,data,device,microbatch,precision="fp32"):
    model.eval();x,y,phase=data;pred=[]
    for start in range(0,len(y),microbatch):
        with precision_context(device,precision):
            pred.append(model(x[start:start+microbatch].to(device)).float().cpu())
    pred=torch.cat(pred);mse=(pred-y).square();mae=(pred-y).abs();decisive=y.abs()>=0.2
    outcome=torch.zeros(len(y),dtype=torch.long)
    outcome[y.abs()>=0.2]=1;outcome[y.abs()>=0.6]=2
    stratum_mse={};stratum_n={}
    for p,pname in enumerate(PHASE_NAMES):
        for o,oname in enumerate(OUTCOME_NAMES):
            mask=(phase==p)&(outcome==o)
            if mask.any():
                key=f'{pname}_{oname}';stratum_mse[key]=float(mse[mask].mean());stratum_n[key]=int(mask.sum())
    result=dict(n=len(y),mse=float(mse.mean()),mae=float(mae.mean()),
        decisive_direction=float(((pred[decisive]>0)==(y[decisive]>0)).float().mean()) if decisive.any() else None,
        mean_prediction=float(pred.mean()),mean_target=float(y.mean()),
        phase_mse={PHASE_NAMES[p]:float(mse[phase==p].mean()) for p in [0,1,2] if (phase==p).any()},
        outcome_mse={OUTCOME_NAMES[o]:float(mse[outcome==o].mean()) for o in range(3) if (outcome==o).any()},
        stratum_mse=stratum_mse,stratum_n=stratum_n)
    model.train();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/value99_pretrain.json')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/value99_pretrain_v1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--init',type=Path,help='Load model weights only; optimizer and step reset')
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
    val=load_split(data_root,manifest,'validation')
    sf_root=ROOT/cfg['sfwdl_data'] if cfg.get('sfwdl_data') else None
    sf_x=sf_y=None;sf_rows=0;sf_loaded=None
    if sf_root is not None and (sf_root/'manifest.json').exists():
        sf_manifest=json.loads((sf_root/'manifest.json').read_text())
        sf_x,sf_y,_=load_split(sf_root,sf_manifest,'train');sf_rows=len(sf_y)
        sf_loaded=sf_stamp(sf_root)
    n_sf=mix_n(cfg['batch'],cfg.get('sfwdl_mix'),cfg.get('sfwdl_mix_n'),sf_rows)
    n_cf=cfg['batch']-n_sf
    sf_reload_s=float(cfg.get('sfwdl_reload_s') or 1800);sf_reloaded=time.monotonic()
    saved=load_valmix(data_root)
    if saved is not None and int(saved['n'])==int(cfg['val_rows']) and int(saved['indices'].max())<len(val[1]):
        mix=saved;val_indices=np.ascontiguousarray(saved['indices'])
    else:
        mix=build_valmix(val[1].numpy(),val[2].numpy(),cfg['val_rows'],cfg['seed'])
        val_indices=mix['indices']
    val=tuple(t[torch.from_numpy(val_indices.copy())] for t in val)
    raw=ValueTransformer(ValueConfig(**cfg['model'])).to(device)
    n=sum(p.numel() for p in raw.parameters())
    if n!=98920577:raise ValueError(f'Expected 98,920,577 parameters, got {n}')
    init_note='random; no pretrained checkpoint'
    if a.init:
        ckpt=torch.load(a.init,map_location='cpu',weights_only=False)
        raw.load_state_dict(ckpt['model'])
        init_note=f"weights from {a.init}; step {int(ckpt.get('step',-1))}; optimizer reset"
    optimizer,opt_info=build_optimizer(raw,cfg)
    compiled=bool(cfg.get('torch_compile')) and device.type=='cuda'
    if compiled:
        torch._dynamo.config.cache_size_limit=max(int(getattr(torch._dynamo.config,'cache_size_limit',8)),128)
        model=torch.compile(raw,dynamic=False)
    else:
        model=raw
    run_manifest=dict(config=cfg,parameters=n,data_manifest_sha256=digest(path),device=str(device),
        torch_version=torch.__version__,initialization=init_note,
        optimizer=opt_info,torch_compile=compiled,
        target='side-to-move W-L; scalar output',train_rows=len(train_y),validation_rows=len(val[1]),
        sfwdl_rows=sf_rows,
        benchmark=a.benchmark,source_sha256={str(p):digest(p) for p in [ROOT/'chess_value99.py',Path(__file__)]})
    step0=0;epoch=0;position=0;order=torch.randperm(len(train_y),generator=rng);seen=0
    sf_position=0;sf_order=torch.randperm(sf_rows,generator=rng) if sf_rows else None
    if a.resume:
        old=json.loads((out/'manifest.json').read_text())
        if old!=run_manifest:
            if not resume_compatible(old,run_manifest):
                raise ValueError('Resume requires identical source/config/data')
            (out/'manifest.json').write_text(json.dumps(run_manifest,indent=2))
        ckpt=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        unwrap_compiled(model).load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        step0=ckpt['step'];epoch=ckpt['epoch'];position=ckpt['position'];order=ckpt['order'];seen=ckpt['seen']
        rng.set_state(ckpt['rng']);torch.set_rng_state(ckpt['torch_rng'])
        if ckpt.get('sf_order') is not None and sf_rows:
            sf_order=ckpt['sf_order'];sf_position=int(ckpt.get('sf_position') or 0)
            if len(sf_order)!=sf_rows:
                sf_order=torch.randperm(sf_rows,generator=rng);sf_position=0
        if device.type=='mps':torch.mps.set_rng_state(ckpt['device_rng'])
        elif device.type=='cuda':torch.cuda.set_rng_state_all(ckpt['device_rng'])
    else:(out/'manifest.json').write_text(json.dumps(run_manifest,indent=2))
    emit(dict(stage='loaded',parameters=n,train_rows=len(train_y),device=str(device),
             validation_rows=len(val[1]),optimizer=opt_info,torch_compile=compiled,
             sfwdl_rows=sf_rows,sfwdl_mix_n=n_sf,
             valmix={k:mix[k] for k in ('n','filled','shortfall','black_source') if k in mix}))
    def save(step,snapshot=False):
        core=unwrap_compiled(model)
        payload=dict(arch='value99',config=asdict(core.config),model={k:v.detach().cpu() for k,v in core.state_dict().items()},
            optimizer=optimizer.state_dict(),step=step,epoch=epoch,position=position,order=order,seen=seen,
            sf_position=sf_position,sf_order=sf_order,
            rng=rng.get_state(),torch_rng=torch.get_rng_state(),
            device_rng=torch.mps.get_rng_state() if device.type=='mps' else torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        atomic(payload,out/'latest.pt')
        if snapshot:
            atomic(dict(arch='value99',config=asdict(core.config),
                model={k:v.detach().cpu() for k,v in core.state_dict().items()},step=step),
                out/f'step_{step:06}.pt')
    emit(dict(stage='initial_validation',**evaluate(model,val,device,cfg['microbatch'],precision)))
    if not a.benchmark and not a.resume:save(0,snapshot=True)
    times=[];started=time.monotonic();total=5 if a.benchmark else cfg['steps']
    for step in range(step0+1,total+1):
        if (out/'STOP').exists():save(step-1);emit(dict(stage='stopped',step=step-1));return
        if sf_root is not None and should_reload_sf(sf_stamp(sf_root),sf_loaded,time.monotonic(),sf_reloaded,sf_reload_s,sf_rows>0):
            try:
                man=json.loads((sf_root/'manifest.json').read_text())
                nxt_x,nxt_y,_=load_split(sf_root,man,'train')
                nxt_rows=len(nxt_y)
            except Exception as exc:
                emit(dict(stage='sfwdl_reload_failed',step=step-1,error=type(exc).__name__))
            else:
                sf_x,sf_y,sf_rows=nxt_x,nxt_y,nxt_rows
                sf_loaded=sf_stamp(sf_root);sf_reloaded=time.monotonic()
                n_sf=mix_n(cfg['batch'],cfg.get('sfwdl_mix'),cfg.get('sfwdl_mix_n'),sf_rows)
                n_cf=cfg['batch']-n_sf
                if sf_rows:
                    if sf_order is None or len(sf_order)!=sf_rows:
                        sf_order=torch.randperm(sf_rows,generator=rng);sf_position=0
                else:
                    sf_order=None;sf_position=0
                emit(dict(stage='sfwdl_reload',step=step-1,sfwdl_rows=sf_rows,sfwdl_mix_n=n_sf))
        t=time.monotonic()
        cf_ids,order,position,wraps=take_ids(order,position,n_cf,rng,len(train_y))
        epoch+=wraps
        xs=[];ys=[]
        if cf_ids:
            xs.append(train_x[cf_ids]);ys.append(train_y[cf_ids])
        if n_sf:
            sf_ids,sf_order,sf_position,_=take_ids(sf_order,sf_position,n_sf,rng,sf_rows)
            xs.append(sf_x[sf_ids]);ys.append(sf_y[sf_ids])
        x_step=torch.cat(xs);y_step=torch.cat(ys)
        factor=min(step/cfg['warmup'],1.)
        if step>cfg['warmup']:
            progress=(step-cfg['warmup'])/max(cfg['steps']-cfg['warmup'],1)
            factor=.1+.9*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
        optimizer.zero_grad(set_to_none=True);loss_sum=0.;mse_sum=0.
        for start in range(0,len(y_step),cfg['microbatch']):
            x=x_step[start:start+cfg['microbatch']].to(device);y=y_step[start:start+cfg['microbatch']].to(device)
            with precision_context(device,precision):
                logit=model(x,return_logit=True)
                # Soft target represents expected score, not draw classification.
                loss=F.binary_cross_entropy_with_logits(logit,(y+1)/2)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            (loss*len(y)/len(y_step)).backward();loss_sum+=float(loss.detach())*len(y)/len(y_step)
            mse_sum+=float(((2*logit.detach().sigmoid()-1)-y).square().mean())*len(y)/len(y_step)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();seen+=len(y_step)
        if device.type=='mps':torch.mps.synchronize()
        elif device.type=='cuda':torch.cuda.synchronize()
        duration=time.monotonic()-t;times.append(duration)
        if step==1 or step%cfg['log_every']==0 or a.benchmark:
            emit(dict(stage='train',step=step,examples=seen,epoch=epoch,loss=loss_sum,mse=mse_sum,
                sfwdl_n=n_sf,
                grad_norm=float(norm),lr=group_lrs(optimizer),step_seconds=duration,
                peak_vram_gb=torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else None,
                positions_per_s=cfg['batch']/np.mean(times[-20:]),elapsed_s=time.monotonic()-started))
        if not a.benchmark and (step%cfg['save_every']==0 or step==total):
            save(step,snapshot=True);emit(dict(stage='validation',step=step,**evaluate(model,val,device,cfg['microbatch'],precision)))
    emit(dict(stage='benchmark_complete' if a.benchmark else 'complete',steps=total,
        steady_positions_per_s=cfg['batch']/np.mean(times[1:] if len(times)>1 else times),
        note='Value fitting is not Elo; no automatic promotion'))


if __name__=='__main__':main()
