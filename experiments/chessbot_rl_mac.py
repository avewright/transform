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
from rl_selfplay.chessbot_eval import paired_eval,previous_from_league,question_for,sf_prefer_n2
from rl_selfplay.chessbot_ppo import (
    load_original,collect,update,diagnostics,sample,greedy_match,
    compare_policies,supervised_distill,AtUnrolls,recurrence_diagnostics,inspect_boards,
)

SOURCE=ROOT/'outputs/chessbot_rl_source'
DATA=ROOT/'outputs/chessbot_rl_data'
CHESSBOT_REPO='Maxlegrec/ChessBot'
CHESSFENS_REPO='Maxlegrec/ChessFENS'
CHESSFENS_REV='0d8d4e6bbda49d42e84c3272be026701659a457c'
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


def pin_source():
    SOURCE.mkdir(parents=True,exist_ok=True)
    needed=['modeling_chessbot.py','config.json','model.safetensors']
    if all((SOURCE/name).exists() for name in needed):
        return
    from huggingface_hub import HfApi,snapshot_download
    import shutil
    local=Path(snapshot_download(CHESSBOT_REPO))
    sha=HfApi().model_info(CHESSBOT_REPO).sha
    for name in needed:
        src=local/name
        if not src.exists():
            raise FileNotFoundError(f'{CHESSBOT_REPO} missing {name}')
        shutil.copy2(src,SOURCE/name)
    (SOURCE/'revision.json').write_text(json.dumps(dict(repo=CHESSBOT_REPO,sha=sha,files=needed),indent=2))


def _keep_fen(fen,groups,seen,counts,cap=12000):
    if len(fen.split())!=6:
        counts['missing_rule_state']+=1
        return
    try:
        b=chess.Board(fen)
        if not b.is_valid() or b.is_game_over(claim_draw=True):
            return
    except ValueError:
        return
    key=' '.join(b.fen().split()[:4])
    if key in seen:
        return
    seen.add(key)
    nonpawn=sum(len(b.pieces(p,c)) for p in [chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN] for c in [0,1])
    phase=2 if nonpawn<=4 else (0 if b.fullmove_number<=12 else 1)
    if len(groups[phase])<cap:
        groups[phase].append(b.fen())
    counts['eligible']+=1


def harvest_chessfens(groups,seen,counts):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi,HfFileSystem
    files=sorted(x for x in HfApi().list_repo_files(CHESSFENS_REPO,repo_type='dataset',revision=CHESSFENS_REV) if x.endswith('.parquet'))
    fs=HfFileSystem()
    target=8192
    name=files[0]
    print(json.dumps({'stage':'anchor_file','file':name}),flush=True)
    with fs.open(f'datasets/{CHESSFENS_REPO}@{CHESSFENS_REV}/{name}','rb',block_size=1<<20) as handle:
        pf=pq.ParquetFile(handle)
        for batch in pf.iter_batches(batch_size=512,columns=['fen']):
            for rec in batch.to_pylist():
                _keep_fen(rec.get('fen') or '',groups,seen,counts,cap=4096)
            kept=sum(len(v) for v in groups.values())
            if kept>=target:
                break
    print(json.dumps({'stage':'anchor_have','have':{k:len(v) for k,v in groups.items()}}),flush=True)


def prepare():
    pin_source()
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'bank.json').exists():return
    groups={0:[],1:[],2:[]};seen=set();counts=Counter();sources=[]
    # Original six-field FENs only. Packed caches without rule state are excluded.
    tactical=sorted((ROOT/'outputs/exp193_tactical/dataset').glob('positions_*.jsonl'))
    for path in tactical:
        sources.append(dict(path=str(path),sha256=digest(path)))
        with path.open() as f:
            for line in f:
                _keep_fen(json.loads(line).get('fen',''),groups,seen,counts)
    if any(len(groups[p])<512 for p in groups):
        sources.append(dict(path=f'{CHESSFENS_REPO}@{CHESSFENS_REV}',sha256=CHESSFENS_REV))
        harvest_chessfens(groups,seen,counts)
    rng=random.Random(291);anchor=[];validation=[]
    for phase,fs in groups.items():
        rng.shuffle(fs)
        nval=min(128,len(fs)//5)
        validation+=fs[:nval];anchor+=fs[nval:nval+4096]
    if len(anchor)<512 or len(validation)<96:raise ValueError('Insufficient full-FEN anchors')
    bank=dict(anchor=anchor,validation=validation,sources=sources,counts=dict(counts),
        phase_available={k:len(v) for k,v in groups.items()},
        note='Anchor FENs from exp193 tactical jsonl when present, else ChessFENS. Full clocks retained. Not 100k balanced bank.')
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
    unrolls=int(cfg.get('unrolls',1))
    published=load_original(SOURCE,device,unrolls=1)
    published.model.requires_grad_(False)
    actor=load_original(SOURCE,device,unrolls=unrolls,gate_extra=True)
    reference=published
    bank=json.loads((DATA/'bank.json').read_text())
    nparam=sum(p.numel() for p in actor.model.parameters())
    depth=actor.model.effective_depth() if hasattr(actor.model,'effective_depth') else 10
    gate0=float(torch.tanh(actor.model.alpha)) if hasattr(actor.model,'alpha') else None
    manifest=dict(config=cfg,device=str(device),parameters=nparam,unrolls=unrolls,
        effective_depth=depth,gate_extra=True,gate0=gate0,
        source_hashes={p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()},
        anchor_sha256=digest(DATA/'bank.json'),mode=args.mode,torch_version=torch.__version__,
        init_checkpoint=str(args.checkpoint) if args.checkpoint else None,
        deviations=['Complete paired game batches can exceed decision threshold.',
                    'Pilot anchor bank is smaller and tactical-source biased.',
                    'No strength promotion; confirmation evaluation remains separate.',
                    f'Recurrent wrap N={unrolls} with zero-init tanh(alpha) extra-pass gate.',
                    'Frozen reference and original opponent are untouched one-pass ChessBot.'])
    if args.mode=='train':
        if args.resume:
            old=json.loads((out/'manifest.json').read_text())
            if old!=manifest:raise ValueError('Resume manifest mismatch')
        else:(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    elif not (out/'manifest.json').exists():
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    emit(dict(stage='loaded',parameters=manifest['parameters'],device=str(device),
             unrolls=unrolls,effective_depth=depth,gate=gate0))
    if hasattr(actor.model,'published_forward'):
        from chess_chessbot import fens_to_planes
        from chess_chessbot_recurrent import depth_identity_errors,identity_errors
        planes=fens_to_planes([chess.STARTING_FEN],device)
        err=identity_errors(actor.model,planes)
        depth=depth_identity_errors(actor.model,planes,unrolls)
        emit(dict(stage='identity_n1',**err))
        emit(dict(stage='identity_train_depth',unrolls=unrolls,**depth))
        if max(err.values())>1e-3 or max(depth.values())>1e-3:
            raise RuntimeError(f'Identity failed n1={err} depth={depth}')
    published_cmp=compare_policies(actor,published,bank['validation'],device,cfg.get('microbatch',8))
    emit(dict(stage='published_parity',**published_cmp))
    if published_cmp['legal_agreement']<0.99 or published_cmp['policy_kl']>0.05:
        raise RuntimeError(f'Train-depth actor is not published ChessBot: {published_cmp}')
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
    def sample_openings(path,n,seed,exclude_keys=()):
        if path.exists():return json.loads(path.read_text())
        emit(dict(stage='opening_sample_start',n=n,path=str(path)))
        opening_rng=torch.Generator().manual_seed(seed)
        existing=set();dev=[];training_keys=set(exclude_keys);attempts=0
        for seq in OPENINGS:
            b=chess.Board()
            for u in seq:b.push_uci(u)
            training_keys.add(' '.join(b.fen().split()[:4]))
        ply_cycle=[4,6,8,10]
        limit=n*64
        while len(dev)<n:
            attempts+=1
            if attempts>limit:
                raise ValueError(f'Could not sample {n} distinct openings')
            b=chess.Board()
            use_uniform=attempts%5==0
            for _ in range(ply_cycle[len(dev)%4]):
                if b.is_game_over(claim_draw=True):break
                if use_uniform:b.push(pr.choice(list(b.legal_moves)))
                else:
                    move,_=sample(published,[b],device,opening_rng,temperature=1.)[0];b.push(move)
            key=' '.join(b.fen().split()[:4])
            if key in training_keys or key in existing or b.is_game_over(claim_draw=True):continue
            existing.add(key);dev.append([m.uci() for m in b.move_stack])
            if len(dev)%8==0:emit(dict(stage='opening_sample',have=len(dev),attempts=attempts))
        path.write_text(json.dumps(dev,indent=2))
        emit(dict(stage='opening_sample_complete',n=len(dev),attempts=attempts,path=str(path)))
        return dev
    def ensure_openings():
        path=out/'development_openings.json'
        if args.openings and args.openings.exists():
            if args.openings.resolve()!=path.resolve():path.write_text(args.openings.read_text())
            return json.loads(path.read_text())
        return sample_openings(path,cfg['eval_pairs'],cfg.get('eval_opening_seed',9291))
    def ensure_confirm_openings():
        n=int(cfg.get('confirm_pairs',96))
        path=out/'confirmation_openings.json'
        held=set()
        for seq in ensure_openings():
            b=chess.Board()
            for u in seq:b.push_uci(u)
            held.add(' '.join(b.fen().split()[:4]))
        return sample_openings(path,n,cfg.get('confirm_opening_seed',39291),held)
    def frozen(path):
        m=actor.clone();m.model.load_state_dict(torch.load(path,map_location=device,weights_only=True));m.model.requires_grad_(False)
        return m
    def load_weights(adapter,payload):
        adapter.model.load_state_dict(payload['actor'] if isinstance(payload,dict) and 'actor' in payload else payload)
    def incumbent_path():
        return Path(args.incumbent).resolve() if args.incumbent else (out/'incumbent.pt').resolve()
    def init_incumbent():
        path=incumbent_path()
        if not path.exists():
            atomic(weights(actor),path)
            emit(dict(stage='init_incumbent',source='published',path=str(path)))
        return path
    def screen_opponents(*,control=None,previous=None):
        named=[('original',published)]
        inc=incumbent_path()
        if inc.exists():
            named.append(('incumbent',frozen(inc)))
        if control is not None:
            c=control.clone();c.model.requires_grad_(False)
            named.append(('control',c))
        if previous:
            named.append(('previous',frozen(previous)))
        if hasattr(actor.model,'default_unrolls') and actor.unrolls>1:
            named.append(('self_n1',AtUnrolls(actor,1)))
        return named
    def play_match(tag,label,opponent,openings,kind):
        emit(dict(stage='evaluation_start',tag=tag,opponent=label,pairs=len(openings),kind=kind))
        t=time.monotonic()
        match=greedy_match(actor,opponent,openings,device)
        (out/f'eval_{tag}_{label}.json').write_text(json.dumps(match,indent=2))
        summary=paired_eval(match,kind=kind,question=question_for(label),opponent=label)
        summary['elapsed_s']=time.monotonic()-t
        summary['score_bounds']=match.get('score_bounds')
        emit(dict(stage='development_evaluation' if kind=='development' else 'confirmation_evaluation',
                  tag=tag,**{k:v for k,v in summary.items() if k!='note'}))
        return summary
    def write_inspect(tag):
        fens=[chess.STARTING_FEN]
        bank_val=bank.get('validation') or []
        fens+=bank_val[:4]
        inc=frozen(incumbent_path()) if incumbent_path().exists() else None
        rows=inspect_boards(published,inc,actor,fens,device)
        payload=dict(tag=tag,boards=rows)
        (out/f'inspect_{tag}.json').write_text(json.dumps(payload,indent=2))
        (out/'inspect_latest.json').write_text(json.dumps(payload,indent=2))
        return rows
    def screen(tag,league=(),control=None,previous=None,kind='development'):
        openings=ensure_openings() if kind=='development' else ensure_confirm_openings()
        results={}
        for label,opponent in screen_opponents(control=control,previous=previous):
            if kind=='confirmation' and label!='incumbent':
                continue
            results[label]=play_match(tag,label,opponent,openings,kind)
        write_inspect(tag)
        (out/'eval_index.jsonl').open('a').write(json.dumps(dict(time=time.time(),tag=tag,kind=kind,results=results))+'\n')
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
    if not args.resume:
        init_incumbent()
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
        init_incumbent()
        emit(dict(stage='warm_start',checkpoint=str(args.checkpoint)))
    def save(iteration):
        atomic(dict(actor=weights(actor),control=weights(control),optimizer=opt.state_dict(),
            control_optimizer=copt.state_dict(),iteration=iteration,league=league,
            drift_count=drift_count,regress_count=regress_count,rng=rng.get_state(),python_rng=pr.getstate(),
            torch_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if device.type=='mps' else None,
            config=cfg),out/'latest.pt')
    if not args.resume:
        save(0)
        emit(dict(stage='baseline',**diagnostics(actor,published,bank['validation'],device,cfg['microbatch'])))
    if cfg.get('eval_pairs'):ensure_openings()
    if not args.resume and int(cfg.get('sl_warmup',0))>0:
        emit(dict(stage='supervised_start',steps=int(cfg['sl_warmup'])))
        supervised_distill(actor,published,bank['anchor'],int(cfg['sl_warmup']),
                           int(cfg.get('sl_batch',cfg.get('minibatch',32))),opt,device,rng,emit)
        emit(dict(stage='supervised_parity',**compare_policies(actor,published,bank['validation'],device,cfg.get('microbatch',8))))
        save(0)
    if not args.resume and cfg.get('eval_pairs'):
        screen('000',control=control)
    for iteration in range(start+1,cfg['iterations']+1):
        if (out/'STOP').exists():emit(dict(stage='stopped',iteration=iteration-1));return
        emit(dict(stage='iteration_start',iteration=iteration))
        t=time.monotonic();current=actor.clone();current.model.requires_grad_(False)
        opponents={'original':(published,.4 if league else .8),'current':(current,.2)}
        for i,path in enumerate(league):
            opponents[f'history_{i}']=(frozen(path),.4/len(league))
        rows,games=collect(actor,opponents,cfg,OPENINGS,device,rng,pr,emit)
        del opponents,current
        (out/f'games_{iteration:03}.json').write_text(json.dumps(games,indent=2))
        atomic(rows,out/'last_rollout.pt')
        trunc=sum(g['truncated'] for g in games)/len(games)
        if trunc>.05:emit(dict(stage='guard_stop',reason='truncations >5%',iteration=iteration,rate=trunc));return
        result=update(actor,control,published,opt,copt,rows,bank['anchor'],cfg,device,rng,emit)
        diag=diagnostics(actor,published,bank['validation'],device,cfg['microbatch'])
        cdiag=diagnostics(control,published,bank['validation'],device,cfg['microbatch'])
        gate=float(torch.tanh(actor.model.alpha)) if hasattr(actor.model,'alpha') else None
        recur=recurrence_diagnostics(actor,bank['validation'][:64],device,cfg.get('microbatch',8)) if hasattr(actor.model,'extra_gate') else {}
        if recur.get('changed'):
            recur['stockfish']=sf_prefer_n2(recur['changed'],nodes=int(cfg.get('sf_nodes',8000)))
        drift_count=drift_count+1 if diag['reference_kl']>.1 else 0
        previous=previous_from_league(league)
        if iteration%5==0:
            path=(out/f'actor_{iteration:03}.pt').resolve();atomic(weights(actor),path);league=(league+[str(path)])[-4:]
            atomic(weights(control),out/f'control_{iteration:03}.pt')
        save(iteration)
        stats=dict(stage='iteration_complete',iteration=iteration,elapsed_s=time.monotonic()-t,
            wins=sum(g['reward']==1 for g in games),draws=sum(g['reward']==0 for g in games),
            losses=sum(g['reward']==-1 for g in games),truncated=sum(g['truncated'] for g in games),
            games=len(games),decisions=len(rows),diagnostics=diag,control_diagnostics=cdiag,
            updates=result['updates'],kl_stopped=result['kl_stopped'],gate=gate,
            recurrence={k:v for k,v in recur.items() if k!='changed'})
        (out/f'update_{iteration:03}.json').write_text(json.dumps(result,indent=2));emit(stats)
        if recur:emit(dict(stage='recurrence',iteration=iteration,**{k:v for k,v in recur.items() if k!='changed'}))
        if cfg.get('eval_pairs'):
            results=screen(f'{iteration:03}',control=control,previous=previous)
            inc=results.get('incumbent')
            if inc:
                regress_count=regress_count+1 if inc.get('verdict')=='weaker' else 0
                save(iteration)
                if regress_count>=2:
                    emit(dict(stage='guard_stop',reason='greedy incumbent regression',
                              iteration=iteration,verdict=inc['verdict'],paired_ci_95=inc.get('paired_ci_95')))
                    return
                if inc.get('verdict')=='stronger' and int(cfg.get('confirm_pairs',96))>0:
                    confirm=screen(f'{iteration:03}c',control=None,previous=None,kind='confirmation')
                    cres=confirm.get('incumbent')
                    if cres and cres.get('verdict')=='stronger':
                        atomic(weights(actor),incumbent_path())
                        emit(dict(stage='promoted',iteration=iteration,path=str(incumbent_path()),
                                  development=inc,confirmation=cres))
                    else:
                        emit(dict(stage='promotion_rejected',iteration=iteration,
                                  development=inc.get('verdict'),confirmation=None if not cres else cres.get('verdict')))
        if drift_count>=2:emit(dict(stage='guard_stop',reason='reference drift',iteration=iteration));return
    emit(dict(stage='complete',iteration=cfg['iterations'],note='No Elo or promotion claim'))


if __name__=='__main__':main()
