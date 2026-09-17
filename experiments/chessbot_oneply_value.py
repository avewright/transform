#!/usr/bin/env python3
"""Value-only sibling-ranking pilot: all legal children, no reply search at play."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import chess
import chess.engine
import numpy as np
import torch
from torch.nn import functional as F
from rl_selfplay.chessbot_ppo import load_original,sample,reward


def board_key(board):
    return ' '.join(board.fen(en_passant='legal').split()[:4])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for s in iter(lambda:f.read(1<<20),b''):h.update(s)
    return h.hexdigest()


def atomic(payload,path):
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def terminal_wdl(board):
    """Absolute [black win, draw, white win], respecting real move history."""
    out=board.outcome(claim_draw=True)
    if out is None:return None
    result=[0.,0.,0.];result[1 if out.winner is None else (2 if out.winner else 0)]=1.
    return result


def utility(probabilities,color):
    return (probabilities[...,2]-probabilities[...,0])*(1 if color else -1)


@torch.no_grad()
def trunk(adapter,fens,device):
    """Original trunk without unused policy/value computations."""
    model=adapter.model
    x=torch.from_numpy(np.stack([adapter.module.fen_to_tensor(f) for f in fens])).to(device)
    x=x.reshape(len(fens),64,19)
    x=model.ma_gating(model.layernorm1(F.gelu(model.linear1(x))))
    pos=model.positional(x)
    for layer in model.layers:x=layer(x,pos)
    return x


@torch.no_grad()
def oneply_moves(adapter,head,boards,device,batch=64):
    """Score EVERY legal child from the parent's player, including terminal moves."""
    scores=[];moves=[];pending=[];fens=[]
    for parent,b in enumerate(boards):
        legal=sorted(b.legal_moves,key=lambda m:m.uci())
        if not legal:raise ValueError('Cannot select move on terminal board')
        moves.append(legal);scores.append([None]*len(legal))
        for j,m in enumerate(legal):
            b.push(m)
            exact=terminal_wdl(b)
            if exact is not None:scores[parent][j]=float(utility(torch.tensor(exact),not b.turn))
            else:pending.append((parent,j));fens.append(b.fen())
            b.pop()
    for start in range(0,len(fens),batch):
        h=trunk(adapter,fens[start:start+batch],device)
        probs=head(h).float().softmax(-1).cpu()
        for k,(parent,j) in enumerate(pending[start:start+batch]):
            scores[parent][j]=float(utility(probs[k],boards[parent].turn))
    chosen=[ms[max(range(len(ms)),key=lambda j:ss[j])] for ms,ss in zip(moves,scores)]
    return chosen,scores


def choose_parents(cfg,out):
    """Split by recorded game first, then exclude any overlapping parent/child state."""
    rng=random.Random(cfg['seed']);items=[];source_hashes={};seen_games=set()
    for raw in cfg['game_sources']:
        path=ROOT/raw;source_hashes[raw]=sha(path)
        for g in json.loads(path.read_text()):
            moves=g['moves'];gid=hashlib.sha256(' '.join(moves).encode()).hexdigest()
            if gid in seen_games:continue
            seen_games.add(gid)
            # One parent per game; no game straddles train/validation.
            candidates=list(range(12,min(len(moves),180),4));rng.shuffle(candidates)
            for ply in candidates:
                b=chess.Board()
                try:
                    for u in moves[:ply]:b.push_uci(u)
                except ValueError:continue
                if b.is_game_over(claim_draw=True) or b.legal_moves.count()<2:continue
                keys={board_key(b)}
                for m in list(b.legal_moves):
                    b.push(m);keys.add(board_key(b));b.pop()
                items.append(dict(game_id=gid,moves=moves[:ply],fen=b.fen(),keys=keys));break
    rng.shuffle(items);val=[];train=[];blocked=set()
    for row in items:
        if len(val)<cfg['val_parents']:
            if row['keys']&blocked:continue
            val.append(row);blocked.update(row['keys'])
    val_ids={r['game_id'] for r in val};train_keys=set()
    for row in items:
        if row['game_id'] in val_ids or row['keys']&blocked or board_key(chess.Board(row['fen'])) in train_keys:continue
        train.append(row);train_keys.add(board_key(chess.Board(row['fen'])))
        if len(train)==cfg['train_parents']:break
    if len(train)<cfg['train_parents'] or len(val)<cfg['val_parents']:raise ValueError('Insufficient disjoint games')
    records=[]
    for split,rows in [('train',train),('validation',val)]:
        for row in rows:records.append({k:v for k,v in row.items() if k!='keys'}|{'split':split})
    (out/'parents.json').write_text(json.dumps(records,indent=2))
    (out/'source_games.json').write_text(json.dumps(source_hashes,indent=2))
    return records


def label_partition(rows,cfg):
    engine=chess.engine.SimpleEngine.popen_uci(cfg['stockfish'])
    engine.configure({'Threads':1,'Hash':32,'UCI_ShowWDL':True,'UCI_LimitStrength':False})
    results=[]
    try:
        for row in rows:
            b=chess.Board()
            for u in row['moves']:b.push_uci(u)
            children=[]
            for move in sorted(b.legal_moves,key=lambda m:m.uci()):
                b.push(move);exact=terminal_wdl(b)
                if exact is not None:
                    wdl=exact;cp=None;depth=0;nodes=0
                else:
                    info=engine.analyse(b,chess.engine.Limit(nodes=cfg['teacher_nodes']),game=object())
                    if 'wdl' not in info:raise ValueError('Stockfish did not supply WDL')
                    w=info['wdl'].white();wdl=[w.losses/1000,w.draws/1000,w.wins/1000]
                    cp=info['score'].white().score(mate_score=100000)
                    depth=info.get('depth');nodes=info.get('nodes')
                children.append(dict(move=move.uci(),fen=b.fen(),wdl=wdl,terminal=exact is not None,
                                     cp_white=cp,depth=depth,nodes=nodes))
                b.pop()
            results.append(row|{'color':b.turn,'children':children})
    finally:engine.quit()
    return results


def label(rows,cfg,out,emit):
    # Small ordered batches give visible progress and restartable artifacts.
    complete=[]
    with ThreadPoolExecutor(max_workers=cfg['label_workers']) as pool:
        for start in range(0,len(rows),16):
            group=rows[start:start+16]
            futures=[pool.submit(label_partition,group[i::cfg['label_workers']],cfg) for i in range(cfg['label_workers'])]
            labeled=[r for f in futures for r in f.result()]
            lookup={r['game_id']:r for r in labeled}
            complete.extend(lookup[r['game_id']] for r in group)
            (out/'labels.partial.json').write_text(json.dumps(complete))
            emit(dict(stage='labeling',parents=len(complete),total=len(rows),
                      children=sum(len(r['children']) for r in complete)))
    (out/'labels.json').write_text(json.dumps(complete));return complete


@torch.no_grad()
def cache_features(adapter,rows,device,out,batch,emit):
    fs=[];groups=[];wdl=[];term=[]
    for row in rows:
        begin=len(fs)
        for c in row['children']:fs.append(c['fen']);wdl.append(c['wdl']);term.append(c['terminal'])
        groups.append(dict(start=begin,end=len(fs),color=row['color'],split=row['split'],game_id=row['game_id']))
    features=[];original=[]
    for start in range(0,len(fs),batch):
        h=trunk(adapter,fs[start:start+batch],device)
        original.append(adapter.model.value_head_q(h).float().softmax(-1).cpu())
        features.append(h.half().cpu())
        if start%(batch*16)==0:emit(dict(stage='features',children=start,total=len(fs)))
    data=dict(features=torch.cat(features),original=torch.cat(original),wdl=torch.tensor(wdl),
              terminal=torch.tensor(term,dtype=torch.bool),groups=groups)
    atomic(data,out/'features.pt');return data


def group_scores(head,data,g,device):
    sl=slice(g['start'],g['end']);p=head(data['features'][sl].to(device).float()).softmax(-1)
    target=data['wdl'][sl].to(device);terminal=data['terminal'][sl].to(device)
    # Exact terminal recognition at inference is also represented in ranking loss.
    p=torch.where(terminal[:,None],target,p)
    return p,target,terminal


@torch.no_grad()
def validate(head,data,device,split='validation'):
    regrets=[];hits=[];ce=[]
    for g in data['groups']:
        if g['split']!=split:continue
        p,target,term=group_scores(head,data,g,device)
        score=utility(p,g['color']);truth=utility(target,g['color']);best=truth.max();choice=score.argmax()
        regrets.append(float(best-truth[choice]));hits.append(float(best-truth[choice]<=.01))
        if (~term).any():ce.append(float(-(target[~term]*p[~term].clamp_min(1e-8).log()).sum(-1).mean()))
    return dict(n=len(regrets),mean_expected_score_regret=sum(regrets)/len(regrets),
                within_001_of_best=sum(hits)/len(hits),wdl_ce=sum(ce)/max(len(ce),1))


def train_head(adapter,data,cfg,device,out,emit):
    head=copy.deepcopy(adapter.model.value_head_q).to(device)
    head.requires_grad_(True);head.train(False)
    opt=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
    groups=[g for g in data['groups'] if g['split']=='train'];rng=random.Random(cfg['seed'])
    baseline=validate(head,data,device);emit(dict(stage='head_baseline',**baseline))
    (out/'head_baseline.json').write_text(json.dumps(baseline,indent=2))
    for epoch in range(1,cfg['epochs']+1):
        rng.shuffle(groups);losses=[]
        for start in range(0,len(groups),cfg['parent_batch']):
            batch=groups[start:start+cfg['parent_batch']];opt.zero_grad(set_to_none=True)
            for g in batch:
                p,target,terminal=group_scores(head,data,g,device)
                score=utility(p,g['color']);truth=utility(target,g['color'])
                target_rank=(truth/cfg['rank_temperature']).softmax(-1)
                ranking=-(target_rank*(score/cfg['rank_temperature']).log_softmax(-1)).sum()
                valid=~terminal
                if valid.any():
                    lp=p[valid].clamp_min(1e-8).log()
                    wdl_loss=-(target[valid]*lp).sum(-1).mean()
                    ref=data['original'][g['start']:g['end']].to(device)[valid]
                    anchor=(p[valid]*(lp-ref.clamp_min(1e-8).log())).sum(-1).mean()
                else:wdl_loss=ranking*0;anchor=ranking*0
                loss=wdl_loss+cfg['ranking_weight']*ranking+cfg['anchor_weight']*anchor
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head objective')
                (loss/len(batch)).backward();losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
        metrics=validate(head,data,device)
        atomic(dict(head_state_dict={k:v.cpu() for k,v in head.state_dict().items()},epoch=epoch,
                    config=cfg,validation=metrics),out/'value_head_latest.pt')
        emit(dict(stage='head_epoch',epoch=epoch,loss=sum(losses)/len(losses),**metrics))
    return head


@torch.no_grad()
def match(adapter,head,openings,device,cfg,emit,tag):
    """One-ply candidate vs unchanged greedy policy. Both colors; no cap-as-draw."""
    games=[];time_candidate=0.;candidate_moves=0;begin=time.monotonic()
    for pair,opening in enumerate(openings):
        boards=[]
        for color in [True,False]:
            b=chess.Board()
            for u in opening:b.push_uci(u)
            boards.append(b)
        live=[0,1]
        while live:
            for i in live[:]:
                r=reward(boards[i],i==0)
                if r is not None or boards[i].ply()>=cfg['ply_cap']:
                    games.append(dict(pair=pair,opening=opening,color=i==0,reward=r,
                        unknown=r is None,plies=boards[i].ply(),moves=[m.uci() for m in boards[i].move_stack]))
                    live.remove(i)
            ours=[i for i in live if boards[i].turn==(i==0)];theirs=[i for i in live if i not in ours]
            if ours:
                t=time.monotonic();chosen,_=oneply_moves(adapter,head,[boards[i] for i in ours],device,cfg['inference_batch'])
                time_candidate+=time.monotonic()-t;candidate_moves+=len(ours)
                for i,m in zip(ours,chosen):boards[i].push(m)
            if theirs:
                chosen=sample(adapter,[boards[i] for i in theirs],device,None,temperature=0)
                for i,(m,_) in zip(theirs,chosen):boards[i].push(m)
        emit(dict(stage='games',arm=tag,pairs=pair+1,total_pairs=len(openings),
                  wins=sum(g['reward']==1 for g in games),draws=sum(g['reward']==0 for g in games),
                  losses=sum(g['reward']==-1 for g in games),unknown=sum(g['unknown'] for g in games)))
    w=sum(g['reward']==1 for g in games);d=sum(g['reward']==0 for g in games);l=sum(g['reward']==-1 for g in games);u=sum(g['unknown'] for g in games)
    return dict(wins=w,draws=d,losses=l,unknown=u,n=len(games),score_bounds=[(w+.5*d)/len(games),(w+.5*d+u)/len(games)],
                mean_candidate_ms=1000*time_candidate/max(candidate_moves,1),elapsed_s=time.monotonic()-begin,games=games)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/chessbot_oneply_value.json')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/chessbot_oneply_value')
    parser.add_argument('--device',default='mps')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=args.out
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True);device=torch.device(args.device);torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    def emit(d):
        d=dict(time=time.time(),**d);print(json.dumps(d),flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        (out/'status.json').write_text(json.dumps(d,indent=2))
    adapter=load_original(ROOT/'outputs/chessbot_rl_source',device);adapter.model.requires_grad_(False)
    source=ROOT/'outputs/chessbot_rl_source/model.safetensors'
    manifest=dict(config=cfg,source_sha256=sha(source),stockfish_sha256=sha(cfg['stockfish']),
                  device=str(device),torch_version=torch.__version__,trainable='value_head_q only',
                  parameters=sum(p.numel() for p in adapter.model.value_head_q.parameters()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));emit(dict(stage='loaded',**manifest))
    openings=json.loads((ROOT/cfg['openings']).read_text())[:cfg['eval_pairs']]
    (out/'openings.json').write_text(json.dumps(openings,indent=2))
    baseline=match(adapter,adapter.model.value_head_q,openings,device,cfg,emit,'original_value')
    (out/'eval_original_value.json').write_text(json.dumps(baseline,indent=2))
    emit(dict(stage='baseline_complete',**{k:v for k,v in baseline.items() if k!='games'}))
    parents=choose_parents(cfg,out);emit(dict(stage='parents',n=len(parents)))
    labels=label(parents,cfg,out,emit)
    data=cache_features(adapter,labels,device,out,cfg['inference_batch'],emit)
    trained=train_head(adapter,data,cfg,device,out,emit)
    # No original parameters were optimized; verify against source to catch mistakes.
    from safetensors.torch import load_file
    original=load_file(str(source))
    if any(not torch.equal(v.cpu(),original[k]) for k,v in adapter.model.state_dict().items()):
        raise AssertionError('Original model changed')
    result=match(adapter,trained,openings,device,cfg,emit,'trained_value')
    (out/'eval_trained_value.json').write_text(json.dumps(result,indent=2))
    emit(dict(stage='complete',**{k:v for k,v in result.items() if k!='games'},note='Small development screen; no promotion'))


if __name__=='__main__':main()
