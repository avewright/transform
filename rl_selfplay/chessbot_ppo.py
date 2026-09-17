"""Original ChessBot adapter and complete-game PPO collection for the Mac pilot.

Published source is pinned by file hashes in the run manifest. No square64 or
recurrent model assumptions are used. All probabilities are legal and FP32.
"""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import chess
import torch
from torch.nn import functional as F


def load_original(source, device):
    from safetensors.torch import load_file
    source = Path(source)
    name = 'chessbot_rl_pinned'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source / 'modeling_chessbot.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    module.ChessBotModel.all_tied_weights_keys = {}
    model = module.ChessBotModel(module.ChessBotConfig(**json.loads((source / 'config.json').read_text())))
    model.load_state_dict(load_file(str(source / 'model.safetensors')), strict=True)
    for p in model.value_head.parameters():
        p.requires_grad_(False)
    return Adapter(model.to(device).eval(), module)


class Adapter:
    def __init__(self, model, module):
        self.model, self.module = model, module
        self.index = {u: i for i, u in enumerate(module.policy_index)}
        if len(self.index) != len(module.policy_index):
            raise ValueError('Duplicate policy vocabulary entries')

    def clone(self):
        return Adapter(copy.deepcopy(self.model), self.module)

    def mapping(self, board):
        mapping = {}
        for move in board.legal_moves:
            uci = move.uci()
            key = uci[:-1] if move.promotion == chess.KNIGHT else uci
            idx = self.index[key]
            if idx in mapping:
                raise ValueError('Legal moves collide in policy vocabulary')
            mapping[idx] = move
        return mapping

    def tensors(self, fens, device):
        x = torch.from_numpy(np.stack([self.module.fen_to_tensor(f) for f in fens])).to(device)
        mask = torch.zeros(len(fens), len(self.index), dtype=torch.bool)
        maps = [self.mapping(chess.Board(f)) for f in fens]
        for i, mapping in enumerate(maps):
            mask[i, list(mapping)] = True
        return x.unsqueeze(1), mask.to(device), maps

    def forward(self, fens, device):
        x, mask, maps = self.tensors(fens, device)
        out = self.model(x)
        return out.last_hidden_state[:, 0].float(), out.hidden_states[1][:, 0].float(), mask, maps


def log_probs(logits, mask, temperature=.8):
    if not mask.any(-1).all() or not torch.isfinite(logits[mask]).all():
        raise ValueError('Invalid legal probabilities')
    return F.log_softmax((logits / temperature).masked_fill(~mask, -1e9), -1)


def kl(p, q):
    return (p.exp() * (p - q)).sum(-1)


def actor_value(q, colors):
    p = q.softmax(-1)
    white = p[:, 2] - p[:, 0]
    return white * (torch.as_tensor(colors, device=q.device, dtype=q.dtype) * 2 - 1)


def reward(board, color):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0. if outcome.winner is None else (1. if outcome.winner == color else -1.)


def finish(rows, terminal, bootstrap, lam=.95):
    a = 0.
    v_next = bootstrap if terminal is None else 0.
    for i in reversed(range(len(rows))):
        r = terminal if i == len(rows)-1 and terminal is not None else 0.
        a = r + v_next - rows[i]['value'] + lam * a
        rows[i]['advantage'] = a
        rows[i]['target'] = a + rows[i]['value']
        v_next = rows[i]['value']


@torch.no_grad()
def sample(adapter, boards, device, rng, temperature=.8):
    fens = [b.fen() for b in boards]
    logits, q, mask, maps = adapter.forward(fens, device)
    lp = log_probs(logits, mask, temperature or 1.)
    actions = lp.argmax(-1).cpu() if temperature == 0 else torch.multinomial(lp.exp().cpu(), 1, generator=rng)[:, 0]
    values = actor_value(q, [b.turn for b in boards]).cpu().tolist()
    return [(maps[i][a], dict(fen=fens[i], action=a, color=boards[i].turn,
             old_logp=float(lp[i,a]), value=values[i])) for i,a in enumerate(actions.tolist())]


@torch.no_grad()
def collect(actor, opponents, cfg, openings, device, rng, py_rng, emit):
    """Complete paired games; threshold can overshoot by one batch, never censor.

    Each trajectory spans actor decisions (actor move plus opponent reply).
    Cap truncations bootstrap at the next actor turn and are not draws.
    """
    all_rows, games = [], []
    batch = cfg['environments']
    if batch % 2:
        raise ValueError('Need even environment count')
    start_time = time.monotonic()
    while len(all_rows) < cfg['decisions'] or len(games) < cfg.get('min_games', 0):
        envs = []
        for _ in range(batch // 2):
            opening = py_rng.choice(openings)
            identity = py_rng.choices(list(opponents), weights=[v[1] for v in opponents.values()])[0]
            for color in [chess.WHITE, chess.BLACK]:
                b = chess.Board()
                for u in opening:
                    b.push_uci(u)
                if b.is_game_over(claim_draw=True):
                    raise ValueError('Terminal opening')
                envs.append(dict(board=b, color=color, opponent=identity, rows=[], opening=opening))
        live = list(range(batch))
        tick = 0
        while live:
            for i in live[:]:
                e = envs[i]; b = e['board']; result = reward(b, e['color'])
                capped = result is None and b.ply() >= cfg['ply_cap'] and b.turn == e['color']
                if result is None and not capped:
                    continue
                boot = 0.
                if capped:
                    _, q, _, _ = actor.forward([b.fen()], device)
                    boot = float(actor_value(q, [e['color']])[0])
                finish(e['rows'], result, boot)
                all_rows.extend(e['rows'])
                games.append(dict(color=e['color'], opponent=e['opponent'], reward=result,
                    truncated=capped, bootstrap=boot, opening=e['opening'], plies=b.ply(),
                    termination=b.outcome(claim_draw=True).termination.name if result is not None else 'TRUNCATED',
                    moves=[m.uci() for m in b.move_stack]))
                live.remove(i)
            # Capture the turn partitions before pushing any move.
            actor_ids = [i for i in live if envs[i]['board'].turn == envs[i]['color']]
            opp_ids = [i for i in live if i not in actor_ids]
            if actor_ids:
                samples = sample(actor, [envs[i]['board'] for i in actor_ids], device, rng)
                for i,(move,row) in zip(actor_ids,samples):
                    envs[i]['rows'].append(row); envs[i]['board'].push(move)
            for identity in opponents:
                ids = [i for i in opp_ids if envs[i]['opponent'] == identity]
                if ids:
                    samples = sample(opponents[identity][0], [envs[i]['board'] for i in ids], device, rng)
                    for i,(move,_) in zip(ids,samples):
                        envs[i]['board'].push(move)
            tick += 1
            if tick % 32 == 0:
                emit(dict(stage='collect', finished_games=len(games), live=len(live),
                    decisions=len(all_rows)+sum(len(envs[i]['rows']) for i in live),
                    elapsed_s=time.monotonic()-start_time))
        emit(dict(stage='batch_complete', games=len(games), decisions=len(all_rows),
                  elapsed_s=time.monotonic()-start_time))
    return all_rows, games


@torch.no_grad()
def diagnostics(actor, reference, fens, device, batch=8):
    total_kl, agrees, n = 0.,0,0
    for start in range(0,len(fens),batch):
        fs = fens[start:start+batch]
        p,q,m,_ = actor.forward(fs,device); r,rq,_,_ = reference.forward(fs,device)
        lp,lr = log_probs(p,m,1.), log_probs(r,m,1.)
        total_kl += float(kl(lp,lr).sum()); agrees += int((lp.argmax(-1)==lr.argmax(-1)).sum()); n+=len(fs)
    return dict(reference_kl=total_kl/n, legal_agreement=agrees/n, n=n)


@torch.no_grad()
def greedy_match(actor,opponent,openings,device,ply_cap=800):
    """Development-only paired games; unresolved outcomes remain unknown."""
    games=[]
    for start in range(0,len(openings),8):
        boards=[];colors=[];origins=[]
        for opening in openings[start:start+8]:
            for color in [True,False]:
                b=chess.Board()
                for u in opening:b.push_uci(u)
                boards.append(b);colors.append(color);origins.append(opening)
        live=list(range(len(boards)))
        while live:
            for i in live[:]:
                r=reward(boards[i],colors[i])
                if r is not None or boards[i].ply()>=ply_cap:
                    games.append(dict(opening=origins[i],color=colors[i],reward=r,
                        truncated=r is None,moves=[m.uci() for m in boards[i].move_stack]))
                    live.remove(i)
            actor_ids=[i for i in live if boards[i].turn==colors[i]]
            opponent_ids=[i for i in live if i not in actor_ids]
            for model,ids in [(actor,actor_ids),(opponent,opponent_ids)]:
                if ids:
                    for i,(move,_) in zip(ids,sample(model,[boards[i] for i in ids],device,None,temperature=0)):
                        boards[i].push(move)
    w=sum(g['reward']==1 for g in games);d=sum(g['reward']==0 for g in games)
    l=sum(g['reward']==-1 for g in games);u=sum(g['reward'] is None for g in games)
    return dict(wins=w,draws=d,losses=l,unknown=u,n=len(games),
                score_bounds=[(w+.5*d)/len(games),(w+.5*d+u)/len(games)],games=games)


def pair_stats(match):
    """Score each opening pair. Color-splits are ties; both-color wins are the signal."""
    by={}
    for g in match['games']:
        by.setdefault(tuple(g['opening']),{})[bool(g['color'])]=g['reward']
    plus=minus=tie=0
    for sides in by.values():
        if True not in sides or False not in sides or None in sides.values():
            continue
        s=sides[True]+sides[False]
        if s>0:plus+=1
        elif s<0:minus+=1
        else:tie+=1
    n=plus+minus+tie
    return dict(pairs=n,plus_pairs=plus,minus_pairs=minus,tied_pairs=tie,
                pair_score=(plus-minus)/n if n else 0.)


def update(actor, control, reference, opt, control_opt, rows, anchors, cfg, device, rng, emit):
    adv = torch.tensor([r['advantage'] for r in rows])
    if adv.std(unbiased=False)>1e-8:
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    # Keep a fixed audit sample for post-update likelihood-ratio checks.
    audit_ids = torch.randperm(len(rows),generator=rng)[:cfg['audit_size']].tolist()
    records=[]; stopped=False
    for epoch in range(cfg['epochs']):
        order=torch.randperm(len(rows),generator=rng).tolist()
        for start in range(0,len(rows),cfg['minibatch']):
            ids=order[start:start+cfg['minibatch']]
            opt.zero_grad(set_to_none=True);control_opt.zero_grad(set_to_none=True)
            stats=dict(policy=0.,value=0.,reference_kl=0.,entropy=0.,clip_fraction=0.)
            for j in range(0,len(ids),cfg['microbatch']):
                group=ids[j:j+cfg['microbatch']];rs=[rows[k] for k in group];fs=[r['fen'] for r in rs]
                p,q,mask,_=actor.forward(fs,device);lp=log_probs(p,mask)
                with torch.no_grad():
                    rp,rq,_,_=reference.forward(fs,device);rlp=log_probs(rp,mask)
                actions=torch.tensor([r['action'] for r in rs],device=device)
                old=torch.tensor([r['old_logp'] for r in rs],device=device)
                ratio=(lp.gather(1,actions[:,None])[:,0]-old).exp()
                a=adv[group].to(device)
                pol=-torch.minimum(ratio*a,ratio.clamp(.9,1.1)*a).mean()
                value=actor_value(q,[r['color'] for r in rs])
                vl=F.mse_loss(value,torch.tensor([r['target'] for r in rs],device=device))
                anchor=kl(lp,rlp).mean();entropy=-(lp.exp()*lp).sum(-1).mean()
                loss=pol+.5*vl+.05*anchor-.001*entropy
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                weight=len(group)/len(ids);(weight*loss).backward()
                cp,_,_,_=control.forward(fs,device)
                (weight*.05*kl(log_probs(cp,mask),rlp).mean()).backward()
                for k,v in [('policy',pol),('value',vl),('reference_kl',anchor),('entropy',entropy),
                            ('clip_fraction',((ratio-1).abs()>.1).float().mean())]:stats[k]+=weight*float(v.detach())
            anchor_ids=torch.randint(len(anchors),(cfg['anchor_batch'],),generator=rng).tolist()
            for j in range(0,len(anchor_ids),cfg['microbatch']):
                fs=[anchors[k] for k in anchor_ids[j:j+cfg['microbatch']]]
                with torch.no_grad():
                    rp,rq,mask,_=reference.forward(fs,device);rlp=log_probs(rp,mask);target=rq.softmax(-1)
                for model in [actor,control]:
                    p,q,_,_=model.forward(fs,device)
                    loss=.05*kl(log_probs(p,mask),rlp).mean()+.1*(-target*q.log_softmax(-1)).sum(-1).mean()
                    (len(fs)/len(anchor_ids)*loss).backward()
            norm=torch.nn.utils.clip_grad_norm_(actor.model.parameters(),.5,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(control.model.parameters(),.5,error_if_nonfinite=True)
            opt.step();control_opt.step()
            # Nonnegative sampled-action KL estimator; audit after applying the update.
            audit=[]
            with torch.no_grad():
                for j in range(0,len(audit_ids),cfg['microbatch']):
                    rs=[rows[k] for k in audit_ids[j:j+cfg['microbatch']]]
                    p,_,m,_=actor.forward([r['fen'] for r in rs],device);lp=log_probs(p,m)
                    acts=torch.tensor([r['action'] for r in rs],device=device)
                    d=lp.gather(1,acts[:,None])[:,0]-torch.tensor([r['old_logp'] for r in rs],device=device)
                    audit.extend(((d.exp()-1)-d).cpu().tolist())
            stats.update(epoch=epoch,grad_norm=float(norm),behavior_kl=sum(audit)/len(audit),update=len(records)+1)
            records.append(stats);emit(dict(stage='update',**stats))
            if stats['behavior_kl']>.01:
                stopped=True;break
        if stopped:break
    return dict(updates=len(records),kl_stopped=stopped,minibatches=records)
