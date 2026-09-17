#!/usr/bin/env python3
"""Mac-native original ChessBot PPO. See docs/chessbot_rl_mac.md for deviations."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import chess
import torch
from rl_selfplay.chessbot_ppo import load_original,collect,update,diagnostics,sample,greedy_match,pair_stats

SOURCE=ROOT/'outputs/chessbot_rl_source'
DATA=ROOT/'outputs/chessbot_rl_data'
OPENINGS=[[],['e2e4','e7e5'],['d2d4','d7d5'],['e2e4','c7c5'],
          ['d2d4','g8f6'],['e2e4','e7e6'],['c2c4','e7e5'],['g1f3','d7d5']]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def atomic(payload,path):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)


def weights(adapter):
    return {k:v.detach().cpu() for k,v in adapter.model.state_dict().items()}


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'bank.json').exists():return
    groups={0:[],1:[],2:[]};seen=set();counts=Counter();sources=[]
    # Original six-field FENs only. Packed caches without rule state are excluded.
    for path in sorted((ROOT/'outputs/exp193_tactical/dataset').glob('positions_*.jsonl')):
        sources.append(dict(path=str(path),sha256=digest(path)))
        with path.open() as f:
            for line in f:
                r=json.loads(line);fen=r.get('fen','')
                if len(fen.split())!=6:counts['missing_rule_state']+=1;continue
                try:
                    b=chess.Board(fen)
                    if not b.is_valid() or b.is_game_over(claim_draw=True):continue
                except ValueError:continue
                key=' '.join(b.fen().split()[:4])
                if key in seen:continue
                seen.add(key)
                nonpawn=sum(len(b.pieces(p,c)) for p in [chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN] for c in [0,1])
                phase=2 if nonpawn<=4 else (0 if b.fullmove_number<=12 else 1)
                if len(groups[phase])<12000:groups[phase].append(b.fen())
                counts['eligible']+=1
    rng=random.Random(291);anchor=[];validation=[]
    for phase,fs in groups.items():
        rng.shuffle(fs)
        nval=min(128,len(fs)//5)
        validation+=fs[:nval];anchor+=fs[nval:nval+4096]
    if len(anchor)<512 or len(validation)<96:raise ValueError('Insufficient full-FEN anchors')
    bank=dict(anchor=anchor,validation=validation,sources=sources,counts=dict(counts),
        phase_available={k:len(v) for k,v in groups.items()},
        note='Mac pilot bank from tactical development sources; full clocks retained. Not 100k balanced bank.')
    (DATA/'bank.json').write_text(json.dumps(bank,indent=2))
    print(json.dumps({k:v for k,v in bank.items() if k not in ['anchor','validation','sources']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','benchmark','train','eval'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/chessbot_rl_mac.json')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/chessbot_rl_mac')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--openings',type=Path)
    p.add_argument('--incumbent',type=Path,help='Frozen previous incumbent weights for greedy screens')
    p.add_argument('--eval-every',type=int,default=1,help='Greedy original/incumbent screen every N iterations')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='mps')
    args=p.parse_args()
    if args.mode=='prepare':prepare();return
    cfg=json.loads(args.config.read_text());torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);rng=torch.Generator().manual_seed(cfg['seed']);pr=random.Random(cfg['seed'])
    device=torch.device(args.device);out=args.out
    if args.mode=='train' and not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError('Output exists: use new path or --resume')
    out.mkdir(parents=True,exist_ok=True)
    def emit(d):
        d=dict(time=time.time(),**d)
        print(json.dumps(d),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        (out/'status.json').write_text(json.dumps(d,indent=2))
    actor=load_original(SOURCE,device)
    reference=actor.clone();reference.model.requires_grad_(False)
    bank=json.loads((DATA/'bank.json').read_text())
    manifest=dict(config=cfg,device=str(device),parameters=sum(p.numel() for p in actor.model.parameters()),
        source_hashes={p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()},
        anchor_sha256=digest(DATA/'bank.json'),mode=args.mode,torch_version=torch.__version__,
        init_checkpoint=str(args.checkpoint) if args.checkpoint else None,
        deviations=['Complete paired game batches can exceed decision threshold.',
                    'Pilot anchor bank is smaller and tactical-source biased.',
                    'No strength promotion; confirmation evaluation remains separate.'])
    if args.mode=='train':
        if args.resume:
            old=json.loads((out/'manifest.json').read_text())
            if old!=manifest:raise ValueError('Resume manifest mismatch')
        else:(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    elif not (out/'manifest.json').exists():
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    emit(dict(stage='loaded',parameters=manifest['parameters'],device=str(device)))
    if args.mode=='benchmark':
        settings={**cfg,'decisions':1,'min_games':64}
        t=time.monotonic()
        rows,games=collect(actor,{'original':(reference,1.)},settings,OPENINGS,device,rng,pr,emit)
        seconds=time.monotonic()-t
        (out/'games.json').write_text(json.dumps(games,indent=2))
        atomic(rows,out/'rollout.pt')
        emit(dict(stage='benchmark_complete',seconds=seconds,games=len(games),decisions=len(rows),
            decisions_per_s=len(rows)/seconds,truncated=sum(g['truncated'] for g in games)))
        return
    def ensure_openings():
        path=out/'development_openings.json'
        if args.openings and args.openings.exists():
            if args.openings.resolve()!=path.resolve():path.write_text(args.openings.read_text())
            return json.loads(path.read_text())
        if path.exists():return json.loads(path.read_text())
        emit(dict(stage='opening_sample_start',n=cfg['eval_pairs']))
        opening_rng=torch.Generator().manual_seed(cfg.get('eval_opening_seed',9291))
        existing=set();dev=[];training_keys=set();attempts=0
        for seq in OPENINGS:
            b=chess.Board()
            for u in seq:b.push_uci(u)
            training_keys.add(' '.join(b.fen().split()[:4]))
        ply_cycle=[4,6,8,10]
        limit=cfg['eval_pairs']*64
        while len(dev)<cfg['eval_pairs']:
            attempts+=1
            if attempts>limit:
                raise ValueError(f'Could not sample {cfg["eval_pairs"]} distinct development openings')
            b=chess.Board()
            use_uniform=attempts%5==0
            for _ in range(ply_cycle[len(dev)%4]):
                if b.is_game_over(claim_draw=True):break
                if use_uniform:b.push(pr.choice(list(b.legal_moves)))
                else:
                    move,_=sample(reference,[b],device,opening_rng,temperature=1.)[0];b.push(move)
            key=' '.join(b.fen().split()[:4])
            if key in training_keys or key in existing or b.is_game_over(claim_draw=True):continue
            existing.add(key);dev.append([m.uci() for m in b.move_stack])
            if len(dev)%8==0:emit(dict(stage='opening_sample',have=len(dev),attempts=attempts))
        path.write_text(json.dumps(dev,indent=2))
        emit(dict(stage='opening_sample_complete',n=len(dev),attempts=attempts))
        return dev
    def frozen(path):
        m=reference.clone();m.model.load_state_dict(torch.load(path,map_location=device,weights_only=True));m.model.requires_grad_(False)
        return m
    def load_weights(adapter,payload):
        adapter.model.load_state_dict(payload['actor'] if isinstance(payload,dict) and 'actor' in payload else payload)
    def screen_opponents(league=()):
        named=[('original',reference)]
        inc=args.incumbent or out/'incumbent.pt'
        seen=set()
        if inc.exists() and (args.checkpoint or args.incumbent):
            named.append(('incumbent',frozen(inc)));seen.add(inc.resolve())
        for path in reversed(list(league)):
            resolved=Path(path).resolve()
            if resolved in seen or not resolved.exists():continue
            named.append(('previous',frozen(resolved)));break
        return named
    def screen(tag,league=()):
        openings=ensure_openings()
        results={}
        for label,opponent in screen_opponents(league):
            emit(dict(stage='evaluation_start',tag=tag,opponent=label,pairs=len(openings)))
            t=time.monotonic()
            match=greedy_match(actor,opponent,openings,device)
            (out/f'eval_{tag}_{label}.json').write_text(json.dumps(match,indent=2))
            summary=dict({k:v for k,v in match.items() if k!='games'},**pair_stats(match),elapsed_s=time.monotonic()-t)
            results[label]=summary
            emit(dict(stage='development_evaluation',tag=tag,opponent=label,**summary))
        (out/'eval_index.jsonl').open('a').write(json.dumps(dict(time=time.time(),tag=tag,results=results))+'\n')
        return results
    if args.mode=='eval':
        ckpt=args.checkpoint or out/'latest.pt'
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        load_weights(actor,state)
        tag=f"{int(state['iteration']):03}" if isinstance(state,dict) and 'iteration' in state else 'ad_hoc'
        emit(dict(stage='evaluation_start',checkpoint=str(ckpt),tag=tag))
        screen(tag)
        return
    control=actor.clone()
    opt=torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad],lr=cfg.get('lr',1e-6),weight_decay=0.)
    copt=torch.optim.AdamW([p for p in control.model.parameters() if p.requires_grad],lr=cfg.get('lr',1e-6),weight_decay=0.)
    league=[];start=0;drift_count=0;regress_count=0
    if args.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        actor.model.load_state_dict(state['actor']);control.model.load_state_dict(state['control'])
        opt.load_state_dict(state['optimizer']);copt.load_state_dict(state['control_optimizer'])
        rng.set_state(state['rng']);pr.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        if device.type=='mps':torch.mps.set_rng_state(state['mps_rng'])
        league=state['league'];start=state['iteration'];drift_count=state['drift_count']
        regress_count=state.get('regress_count',0)
    elif args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        actor.model.load_state_dict(state['actor'] if isinstance(state,dict) and 'actor' in state else state)
        if isinstance(state,dict) and 'control' in state:
            control.model.load_state_dict(state['control'])
        else:
            control.model.load_state_dict(actor.model.state_dict())
        incumbent=(out/'incumbent.pt').resolve();atomic(weights(actor),incumbent);league=[str(incumbent)]
        emit(dict(stage='init_incumbent',checkpoint=str(args.checkpoint),league=league))
    def save(iteration):
        atomic(dict(actor=weights(actor),control=weights(control),optimizer=opt.state_dict(),
            control_optimizer=copt.state_dict(),iteration=iteration,league=league,
            drift_count=drift_count,regress_count=regress_count,rng=rng.get_state(),python_rng=pr.getstate(),
            torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if device.type=='mps' else None,
            config=cfg),out/'latest.pt')
    if not args.resume:
        save(0)
        emit(dict(stage='baseline',**diagnostics(actor,reference,bank['validation'],device,cfg['microbatch'])))
    if cfg.get('eval_pairs'):ensure_openings()
    for iteration in range(start+1,cfg['iterations']+1):
        if (out/'STOP').exists():emit(dict(stage='stopped',iteration=iteration-1));return
        emit(dict(stage='iteration_start',iteration=iteration))
        t=time.monotonic();current=actor.clone();current.model.requires_grad_(False)
        opponents={'original':(reference,.4 if league else .8),'current':(current,.2)}
        for i,path in enumerate(league):
            opponents[f'history_{i}']=(frozen(path),.4/len(league))
        rows,games=collect(actor,opponents,cfg,OPENINGS,device,rng,pr,emit)
        del opponents,current
        (out/f'games_{iteration:03}.json').write_text(json.dumps(games,indent=2))
        atomic(rows,out/'last_rollout.pt')
        trunc=sum(g['truncated'] for g in games)/len(games)
        if trunc>.05:emit(dict(stage='guard_stop',reason='truncations >5%',iteration=iteration,rate=trunc));return
        result=update(actor,control,reference,opt,copt,rows,bank['anchor'],cfg,device,rng,emit)
        diag=diagnostics(actor,reference,bank['validation'],device,cfg['microbatch'])
        cdiag=diagnostics(control,reference,bank['validation'],device,cfg['microbatch'])
        drift_count=drift_count+1 if diag['reference_kl']>.1 else 0
        if iteration%5==0:
            path=(out/f'actor_{iteration:03}.pt').resolve();atomic(weights(actor),path);league=(league+[str(path)])[-4:]
            atomic(weights(control),out/f'control_{iteration:03}.pt')
        save(iteration)
        stats=dict(stage='iteration_complete',iteration=iteration,elapsed_s=time.monotonic()-t,
            wins=sum(g['reward']==1 for g in games),draws=sum(g['reward']==0 for g in games),
            losses=sum(g['reward']==-1 for g in games),truncated=sum(g['truncated'] for g in games),
            games=len(games),decisions=len(rows),diagnostics=diag,control_diagnostics=cdiag,
            updates=result['updates'],kl_stopped=result['kl_stopped'])
        (out/f'update_{iteration:03}.json').write_text(json.dumps(result,indent=2));emit(stats)
        if cfg.get('eval_pairs') and (iteration%max(args.eval_every,1)==0 or iteration==cfg['iterations']):
            results=screen(f'{iteration:03}',league)
            if 'incumbent' in results:
                regress_count=regress_count+1 if results['incumbent']['score_bounds'][1]<.5 else 0
                save(iteration)
                if regress_count>=2:
                    emit(dict(stage='guard_stop',reason='greedy incumbent regression',
                              iteration=iteration,score_bounds=results['incumbent']['score_bounds']))
                    return
        if drift_count>=2:emit(dict(stage='guard_stop',reason='reference drift',iteration=iteration));return
    emit(dict(stage='complete',iteration=cfg['iterations'],note='No Elo or promotion claim'))


if __name__=='__main__':main()
