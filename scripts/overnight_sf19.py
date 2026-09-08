#!/usr/bin/env python3
"""Bounded SF19 overnight continuation. Never promotes or uploads weights."""
import argparse, gc, json, math, os, re, shutil, signal, subprocess, sys, time, threading
from pathlib import Path
os.environ.setdefault('MOVE_VOCAB_VERSION','compact')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('PYTHONUNBUFFERED','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from scripts.autoresearch_8gb.pipeline import (position_hashes, make_val_membership, apply_membership,
    concat_soft_tables, attach_static_targets, cheap_eval_losses, audit_soft_targets)

def write(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)
def log(s): print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def load(p): return torch.load(p,map_location='cpu',weights_only=False)
def sliced(d, idx):
    n=len(d['board_array']); idx=torch.as_tensor(idx,dtype=torch.long)
    return {k:v[idx] for k,v in d.items() if torch.is_tensor(v) and v.ndim and len(v)==n}
def save(d,p):
    tmp=Path(str(p)+'.tmp'); torch.save(d,tmp); tmp.replace(p)

def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    paths={'soft':ROOT/'outputs/sf19_ft/soft_cache.pt',
           'replay':ROOT/'outputs/hf_elo_mix/soft_cache.pt',
           'deep':ROOT/'outputs/hf_elo_mix/deep_cache.pt'}
    ds={k:load(p) for k,p in paths.items()}
    manifests={}
    for k,d in ds.items():
        old=ROOT/f'outputs/sf19_ft/run2/val_manifest_{k}.json'
        manifests[k]=json.loads(old.read_text()) if old.exists() else make_val_membership(d,n_hold=2000,seed=203 if k=='replay' else 202,source=k)
    assert manifests['soft']['method']=='saved_split_v1'
    blocked=np.unique(np.concatenate([np.asarray(m['blocked_hashes'],dtype=np.uint64) for m in manifests.values()]))
    report={'sources':{k:str(v) for k,v in paths.items()},'blocked_hashes':len(blocked),'rows':{},'shards':[]}
    # Freeze validation tensors before removing cross-source overlaps.
    for k,d in ds.items():
        _,vi=apply_membership(d,manifests[k]); assert len(vi)>0
        save(sliced(d,vi[:2000]),out/f'eval_{k}.pt')
    chunks=[ds['soft']]
    for p in sorted((ROOT/'outputs/sf19_soft/expand1/inbox').glob('shard_*/soft_cache.pt')):
        if not (p.parent/'READY').exists(): continue
        d=load(p); assert 'split' in d and not d['split'].any()
        chunks.append(d); report['shards'].append(str(p))
    ds['soft']=concat_soft_tables(chunks); del chunks,d
    for k,d in ds.items():
        hs=position_hashes(d).astype(np.uint64); _,first=np.unique(hs,return_index=True)
        unique=np.zeros(len(hs),dtype=bool); unique[first]=True
        if k=='replay':
            keep=unique & ~np.isin(hs,blocked)
        else:
            ownval=np.isin(hs,np.asarray(manifests[k]['hashes'],dtype=np.uint64))
            keep=unique & (~np.isin(hs,blocked) | ownval)
            manifests[k]['blocked_hashes']=[int(x) for x in blocked]
            manifests[k]['n_blocked']=len(blocked)
            write(out/f'val_manifest_{k}.json',manifests[k])
        cleaned=sliced(d,np.flatnonzero(keep)); attach_static_targets(cleaned)
        audit=audit_soft_targets(cleaned,max_rows=len(cleaned['board_array']))
        assert audit['ok'],(k,audit)
        if k=='replay': ti=torch.arange(len(cleaned['board_array']))
        else:
            ti,vi=apply_membership(cleaned,manifests[k])
            save(sliced(cleaned,vi[:2000]),out/f'eval_{k}.pt')
        assert not np.isin(position_hashes(cleaned)[ti.numpy()],blocked).any(),k
        report['rows'][k]={'input':len(hs),'saved':len(cleaned['board_array']),'train':len(ti),'audit':audit}
        save(cleaned,out/f'{k}_cache.pt'); del cleaned
    write(out/'data_audit.json',report)
    log('Data prepared: '+json.dumps(report['rows']))

def evaluate(ckpt,out):
    from chess_inference import load_checkpoint
    model=load_checkpoint(ckpt,'cuda'); result={}
    for name in ('soft','replay','deep'):
        d=load(out/f'eval_{name}.pt'); attach_static_targets(d)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            result[name]=cheap_eval_losses(model,d,torch.arange(len(d['board_array'])),torch.device('cuda'),microbatch=64)
        assert all(math.isfinite(v) for v in result[name].values())
    del model,d; gc.collect(); torch.cuda.empty_cache()
    return result

def ingest_generated(out):
    manifest_path=out/'extra_manifest.json'
    man=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'shards':[],'originals':[]}
    inbox=out/'generation/inbox'; fresh=[p for p in sorted(inbox.glob('shard_*/soft_cache.pt'))
        if (p.parent/'READY').exists() and str(p) not in man['originals']]
    if not fresh:
        if not manifest_path.exists(): write(manifest_path,man)
        return
    blocked=np.asarray(json.loads((out/'val_manifest_soft.json').read_text())['blocked_hashes'],dtype=np.uint64)
    prior=[]
    for path in [out/'soft_cache.pt',out/'replay_cache.pt',out/'deep_cache.pt',*[Path(p) for p in man['shards']]]:
        d=load(path); prior.append(position_hashes(d)); del d
    seen=np.unique(np.concatenate(prior)); del prior
    safe_dir=out/'generated_verified'; safe_dir.mkdir(exist_ok=True)
    for path in fresh:
        d=load(path); assert 'split' in d and not d['split'].any()
        hs=position_hashes(d); _,first=np.unique(hs,return_index=True)
        keep=np.zeros(len(hs),dtype=bool); keep[first]=True
        keep &= ~np.isin(hs,blocked) & ~np.isin(hs,seen)
        clean=sliced(d,np.flatnonzero(keep)); audit=audit_soft_targets(clean,max_rows=len(clean['board_array']))
        assert audit['ok']
        man['originals'].append(str(path))
        if len(clean['board_array']):
            dest=safe_dir/(path.parent.name+'.pt'); save(clean,dest); man['shards'].append(str(dest))
            seen=np.unique(np.concatenate([seen,hs[keep]]))
        with open(out/'ingestion.jsonl','a') as f:
            f.write(json.dumps({'time':time.time(),'source':str(path),'input':len(hs),'accepted':int(keep.sum()),'audit':audit})+'\n')
    write(manifest_path,man); log(f'Verified new-data manifest: {len(man["shards"])} shards')

def start_generation(out):
    cmd=['nice','-n','10',sys.executable,'-u','scripts/sf19_soft_dataset.py','generate','--go',
         '--mode','selfplay','--out-dir',str(out/'generation'),'--target','500000','--workers','8',
         '--nodes','100000','--play-nodes','4000','--multipv','8','--tau','120','--epsilon','0.25',
         '--ply-stride','3','--ply-skip-open','4','--ply-cap','140','--book-noise','4','--hash-mb','32',
         '--shard-size','20000','--holdout-frac','0','--game-start','2000000','--seed','1908',
         '--seed-fens-n','2048','--seed-caches',str(out/'soft_cache.pt'),
         '--exclude-caches',str(out/'soft_cache.pt'),str(out/'replay_cache.pt'),str(out/'deep_cache.pt')]
    f=open(out/'generation.stdout.log','a')
    p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    ACTIVE.append(p); write(out/'generation_command.json',{'command':cmd,'pid':p.pid}); return p

def stop_generation(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGINT)
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait()

def guard_bad(metrics,baseline):
    return (not all(math.isfinite(v) for v in metrics.values()) or
            metrics.get('hard_ce',0)>baseline['hard_ce']*1.05 or
            metrics.get('soft_ce',0)>baseline['soft_ce']*1.05 or
            metrics.get('wdl_ce',0)>baseline['wdl_ce']*1.20)

def stop_child(p,out):
    (out/'STOP').touch()
    try: p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGINT)
        try: p.wait(timeout=60)
        except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGTERM); p.wait(timeout=30)

ACTIVE=[]
def start_watchdog(out,deadline):
    def watch():
        while time.time()<deadline:
            try:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                row={'time':time.time(),'gpu_util_memory_temp_power':gpu,'load':os.getloadavg(),
                     'disk_free_gb':shutil.disk_usage(out).free/1e9}
                with open(out/'resources.jsonl','a') as f: f.write(json.dumps(row)+'\n')
            except Exception as e: log('Resource logger: '+str(e))
            time.sleep(min(60,max(0,deadline-time.time())))
        (out/'STOP').touch()
        write(out/'deadline_reached.json',{'time':time.time(),'note':'12-hour wall limit; stopping owned subprocesses'})
        for p in ACTIVE:
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        time.sleep(5)
        for p in ACTIVE:
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError: pass
        os._exit(0)
    threading.Thread(target=watch,daemon=True).start()

def sf_screen(path,name,out,timeout, repeats=1):
    prefix=f'{out.name}_{name}_n8000_r{repeats}'
    cmd=[sys.executable,'-u','-m','harness.elo','--ckpt',str(path),'--out-prefix',prefix,
        '--mode','policy','--no-book','--no-syzygy','--nodes','8000',
        '--games-per-opening-per-color',str(repeats),'--no-stop-after-bracket','--elos','1750','1900']
    dest=ROOT/f'outputs/elo_eval_{prefix}.json'
    with open(out/f'elo_{name}_r{repeats}.stdout.log','a') as f:
        ep=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        ACTIVE.append(ep)
        try: rc=ep.wait(timeout=max(30,timeout))
        except subprocess.TimeoutExpired:
            os.killpg(ep.pid,signal.SIGTERM)
            try: ep.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(ep.pid,signal.SIGKILL); ep.wait()
            rc=-15
    result={'returncode':rc,'result':str(dest)}
    if rc==0 and dest.exists():
        d=json.loads(dest.read_text()); games=d.get('games',[])
        result['n_games']=len(games)
        result['score']=sum(g['score'] for g in games)/max(1,len(games))
        result['estimate']=d.get('estimate')
    log('Elo screen '+name+': '+json.dumps(result))
    return result

def clear_win(candidate,baseline):
    # Paired opening-cluster bootstrap: repeated games from one opening are not independent.
    ca=json.loads(Path(candidate['result']).read_text())
    ba=json.loads(Path(baseline['result']).read_text())
    def grouped(d):
        result={}
        for g in d['games']:
            opening=g['opening']; key=tuple(opening.split()) if isinstance(opening,str) else tuple(opening); result.setdefault(key,[]).append(g['score'])
        return {k:float(np.mean(v)) for k,v in result.items()}
    c,b=grouped(ca),grouped(ba)
    if set(c)!=set(b) or len(c)<8 or min(len(ca['games']),len(ba['games']))<64:
        return {'passed':False,'reason':'incomplete matched evaluation'}
    dif=np.array([c[k]-b[k] for k in sorted(c)])
    rng=np.random.default_rng(1908)
    boot=rng.choice(dif,(10000,len(dif)),replace=True).mean(axis=1)
    low=float(np.quantile(boot,.025)); delta=float(dif.mean())
    return {'passed':delta>=.15 and low>0,'score_gain':delta,'opening_bootstrap_lower95':low}

def maybe_upload(chosen,candidates,results,validations,baseline,out):
    evidence={'promoted':False,'candidate':chosen,'comparisons':{}}
    if chosen in ('source','run1_1500') or chosen not in results:
        evidence['reason']='No new evaluated candidate'; write(out/'promotion.json',evidence); return
    for name in ('original','source','run1_1500'):
        if name not in results or results[name]['returncode']!=0 or results[chosen]['returncode']!=0:
            evidence['reason']='Incomplete Elo results'; write(out/'promotion.json',evidence); return
        evidence['comparisons'][name]=clear_win(results[chosen],results[name])
    v=validations[chosen]
    evidence['validation_passed']=all(not guard_bad(v[n],baseline[n]) for n in ('soft','replay','deep'))
    if not evidence['validation_passed'] or not all(x['passed'] for x in evidence['comparisons'].values()):
        evidence['reason']='Did not clear conservative Elo and retention gates'; write(out/'promotion.json',evidence); return
    # User authorized HF upload on a great Elo checkpoint. Use a separate private repo;
    # do not overwrite the existing published model or its latest checkpoint.
    try:
        from huggingface_hub import HfApi
        from data_loader import _hf_token
        api=HfApi(token=_hf_token())
        repo='avewright/chess-transformer-100m-'+out.name
        stage=out/'hf_candidate'; stage.mkdir(exist_ok=True)
        ck=load(candidates[chosen])
        payload={k:ck[k] for k in ('model_state_dict','config','arch','vocab','n_params','steps') if k in ck}
        payload['eval_only']=True
        save(payload,stage/'best_model.pt')
        save(ck,stage/'training_resume.pt') if not ck.get('eval_only') else None
        write(stage/'evaluation.json',{'gate':evidence,'results':results,'validation':validations})
        for name,item in results.items(): shutil.copy2(item['result'],stage/f'elo_{name}.json')
        for filename in ('launch.json','data_audit.json','training_exit.json','baseline_validation.json'):
            shutil.copy2(out/filename,stage/filename)
        (stage/'README.md').write_text('# SF19 overnight candidate\n\n100M squares64 chess policy. '
            'Passed a conservative fixed-node Stockfish screening comparison against three preserved baselines. '
            'These are local benchmark results, not an official playing rating.\n\n'
            'See evaluation.json for the opening-cluster bootstrap and validation checks. '
            'best_model.pt contains evaluation weights; training_resume.pt, when present, is the resumable live checkpoint.\n')
        api.create_repo(repo_id=repo,private=True,exist_ok=True)
        api.upload_folder(repo_id=repo,folder_path=str(stage),commit_message='SF19 overnight candidate with matched Elo evidence')
        evidence.update(promoted=True,repo=repo,original_repo_unchanged=True)
    except Exception as e: evidence['upload_error']=str(e)
    write(out/'promotion.json',evidence)

def main(args):
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.prepare_only: prepare(out); return
    assert not (out/'started.json').exists(),'This job already started; use its checkpoints for explicit recovery.'
    start=time.time(); deadline=start+args.hours*3600; train_end=deadline-7200
    start_watchdog(out,deadline)
    write(out/'started.json',{'start':start,'deadline':deadline,'train_end':train_end,'hours':args.hours})
    if not (out/'data_audit.json').exists(): prepare(out)
    source=Path(args.source).resolve()
    shutil.copy2(source,out/'source_full.pt')
    baseline=evaluate(out/'source_full.pt',out); write(out/'baseline_validation.json',baseline)
    ck=load(out/'source_full.pt')
    assert not ck.get('eval_only')
    ck['steps']=0; ck['global_step']=0; ck['positions']=0
    ck.pop('swa_state',None); ck.pop('swa_state_dict',None); ck['swa_n']=0
    ck['overnight_source']=str(source)
    save(ck,out/'init.pt'); del ck; gc.collect()
    ingest_generated(out)
    generator=start_generation(out)
    command=[sys.executable,'-u','experiments/exp201_recurrent_64.py','--go','--skip-mix',
        '--resume',str(out/'init.pt'),'--output-dir',str(out),
        '--attach-from-manifest',str(out/'extra_manifest.json'),
        '--soft-cache',str(out/'soft_cache.pt'),'--deep-cache',str(out/'deep_cache.pt'),
        '--bonus-cache',str(out/'replay_cache.pt'),'--deep-mix-frac','0.05','--bonus-mix-frac','0.20',
        '--optimizer','polar_normuon','--torch-compile','--compile-polar','--force-lr',
        '--muon-lr','0.0007','--adam-lr','0.00001','--warmup','100','--batch-size','64',
        '--max-steps','120000','--train-minutes',str(max(1,(train_end-time.time())/60)),
        '--val-every','250','--val-eval-n','2000','--save-every','250','--elo-every','0']
    write(out/'launch.json',{'command':command,'mix':{'sf19':.75,'replay':.20,'syzygy':.05},
        'guard':'Stop after 3 consecutive validations >5% worse policy CE or >20% worse WDL CE than source; nonfinite/stall stop immediately.',
        'evaluation':'Up to 3 periodic fixed-node SF screens; final matched comparison with original/run1/run2. Gated upload to separate private HF repo; existing champion untouched.'})
    log('Training launch: '+' '.join(command))
    best=baseline['soft']['hard_ce']; bad={'soft':0,'deep':0}; processed=0; reason='completed'; last_step=0
    next_check=4000; periodic={}; gauntlets=0; safe=out/'init.pt'; rollback_used=False
    shutil.copy2(out/'init.pt',out/'best_safe.pt')
    with open(out/'stdout.log','a') as stdout:
        p=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True); ACTIVE.append(p)
        write(out/'process.json',{'controller_pid':os.getpid(),'trainer_pid':p.pid})
        while p.poll() is None:
            time.sleep(10)
            f=out/'train.log'; lines=f.read_text().splitlines() if f.exists() else []
            stop=False
            for line in lines[processed:]:
                m=re.search(r'step (\d+)/120000',line)
                if m: last_step=int(m[1])
                if 'NON-FINITE' in line: reason='nonfinite'; stop=True
                m=re.search(r'val/(soft|deep) (.*)',line)
                if not m: continue
                name=m[1]; metrics={k:float(v) for k,v in re.findall(r'(\w+)=([-+\w.]+)',m[2])}
                bad[name]=bad[name]+1 if guard_bad(metrics,baseline[name]) else 0
                if name=='soft' and metrics['hard_ce']<best and all(math.isfinite(v) for v in metrics.values()):
                    latest=out/'latest.pt'
                    if latest.exists():
                        shutil.copy2(latest,out/'best_validation.pt'); best=metrics['hard_ce']
                        write(out/'best_validation.json',{'step':last_step,'metrics':metrics,'selection':'validation screen, not Elo promotion'})
                if bad[name]>=3: reason=f'{name}_validation_regression'; stop=True
            processed=len(lines)
            if time.time()>=train_end: reason='training_wall_budget'; stop=True
            if f.exists() and time.time()-f.stat().st_mtime>1200: reason='training_stalled'; stop=True
            if (out/'HALT').exists(): reason='user_halt'; stop=True
            if stop:
                log('Stopping training: '+reason); stop_child(p,out)
                if 'validation_regression' in reason and not rollback_used and time.time()<train_end-1800:
                    rollback_used=True; bad={'soft':0,'deep':0}
                    command[command.index('--resume')+1]=str(out/'best_safe.pt')
                    command[command.index('--muon-lr')+1]='0.00035'
                    command[command.index('--adam-lr')+1]='0.000005'
                    log('One recovery attempt from best_safe.pt at half learning rate')
                    write(out/'rollback.json',{'reason':reason,'command':command,'time':time.time()})
                    p=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True); ACTIVE.append(p)
                    next_check=max(next_check,last_step+4000); reason='completed'; continue
                break
            if last_step>=next_check and time.time()<train_end-1200:
                stop_child(p,out)
                snapshot=out/f'check_{last_step:06d}.pt'; shutil.copy2(out/'latest.pt',snapshot)
                metrics=evaluate(snapshot,out)
                periodic[str(last_step)]={'validation':metrics}
                valid=all(not guard_bad(metrics[n],baseline[n]) for n in ('soft','replay','deep'))
                if valid:
                    shutil.copy2(snapshot,out/'best_safe.pt')
                else:
                    log('Periodic retention check failed; preserving earlier safe weights')
                if next_check%16000==0 and gauntlets<3 and time.time()<train_end-1800:
                    periodic[str(last_step)]['elo']=sf_screen(snapshot,f'step{last_step}',out,min(1500,train_end-time.time()-600))
                    elo=periodic[str(last_step)]['elo']
                    prior=json.loads((out/'best_elo.json').read_text()) if (out/'best_elo.json').exists() else {'score':-1}
                    if valid and elo.get('n_games',0)>=32 and elo.get('score',-1)>prior['score']:
                        shutil.copy2(snapshot,out/'best_elo.pt'); write(out/'best_elo.json',elo)
                    gauntlets+=1
                write(out/'periodic_checks.json',periodic)
                ingest_generated(out)
                next_check+=4000
                command[command.index('--resume')+1]=str(out/'latest.pt')
                if not valid and not rollback_used:
                    rollback_used=True; command[command.index('--resume')+1]=str(out/'best_safe.pt')
                    command[command.index('--muon-lr')+1]='0.00035'
                    command[command.index('--adam-lr')+1]='0.000005'
                    write(out/'rollback.json',{'reason':'periodic_retention','command':command,'time':time.time()})
                elif not valid:
                    reason='repeated_retention_regression'; break
                p=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True); ACTIVE.append(p)
                write(out/'process.json',{'controller_pid':os.getpid(),'trainer_pid':p.pid})
        rc=p.wait()
    stop_generation(generator)
    write(out/'training_exit.json',{'returncode':rc,'reason':reason,'step':last_step})
    if reason=='user_halt':
        write(out/'finished.json',{'reason':reason,'elapsed_hours':(time.time()-start)/3600}); return
    log(f'Training ended rc={rc} reason={reason}; validating candidates')
    candidates={'original':ROOT/'outputs/sf19_ft/init.pt','source':out/'source_full.pt','run1_1500':ROOT/'outputs/sf19_ft/run1/step_001500.pt'}
    for name in ('best_validation','best_safe','best_elo','latest','eval_swa'):
        path=out/f'{name}.pt'
        if path.exists(): candidates[name]=path
    validations={}
    for name,path in candidates.items():
        if time.time()>deadline-600: break
        try: validations[name]=evaluate(path,out)
        except Exception as e: validations[name]={'error':str(e)}
        write(out/'candidate_validation.json',validations)
    eligible=[n for n,v in validations.items() if 'soft' in v and n not in ('original','source','run1_1500') and all(not guard_bad(v[k],baseline[k]) for k in ('soft','replay','deep'))]
    chosen='best_elo' if 'best_elo' in eligible else (min(eligible,key=lambda n:validations[n]['soft']['hard_ce']) if eligible else 'source')
    # Compare with both preserved baselines. This is a screen, never automatic promotion.
    final_names=list(dict.fromkeys(['original','run1_1500','source',chosen]))
    results={}
    for i,name in enumerate(final_names):
        remaining=deadline-time.time()-300
        if remaining<120: break
        results[name]=sf_screen(candidates[name],name,out,remaining/(len(final_names)-i),repeats=2)
        write(out/'elo_results.json',results)
    if time.time()<deadline-120:
        maybe_upload(chosen,candidates,results,validations,baseline,out)
    write(out/'finished.json',{'reason':reason,'training_returncode':rc,'elapsed_hours':(time.time()-start)/3600,
        'screen_candidate':chosen,'elo_results':results,'promotion_report':str(out/'promotion.json')})
    log('Overnight job finished. Checkpoints retained; see promotion.json for the gated HF upload result.')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--source')
    ap.add_argument('--hours',type=float,default=12); ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    if not args.prepare_only and not args.source: ap.error('--source required')
    try:
        main(args)
    except BaseException as exc:
        out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
        write(out/'failure.json',{'error':repr(exc),'time':time.time()})
        for child in ACTIVE:
            if child.poll() is None:
                try: os.killpg(child.pid,signal.SIGINT)
                except ProcessLookupError: pass
        for child in ACTIVE:
            if child.poll() is None:
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait()
        raise
