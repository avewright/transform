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
from rl_selfplay.chessbot_eval import pair_stats


def load_original(source, device, unrolls=1, gate_extra=True):
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
    unrolls = int(unrolls)
    if unrolls < 1:
        raise ValueError('unrolls must be >= 1')
    if unrolls > 1:
        from chess_chessbot_recurrent import RecurrentChessBot
        model = RecurrentChessBot(model, default_unrolls=unrolls, gate_extra=gate_extra)
    return Adapter(model.to(device).eval(), module, unrolls=unrolls)


class Adapter:
    def __init__(self, model, module, unrolls=1):
        self.model, self.module, self.unrolls = model, module, int(unrolls)
        self.index = {u: i for i, u in enumerate(module.policy_index)}
        if len(self.index) != len(module.policy_index):
            raise ValueError('Duplicate policy vocabulary entries')

    def clone(self):
        return Adapter(copy.deepcopy(self.model), self.module, self.unrolls)

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

    def forward_tensors(self, x, mask, unrolls=None):
        n = self.unrolls if unrolls is None else int(unrolls)
        if hasattr(self.model, 'default_unrolls'):
            out = self.model(x, recurrent_unrolls=n)
        else:
            if unrolls is not None and n != 1:
                raise ValueError('one-pass ChessBot cannot unroll')
            out = self.model(x)
        policy, value = _policy_value(out)
        return policy, value, mask

    def forward(self, fens, device, unrolls=None):
        x, mask, maps = self.tensors(fens, device)
        policy, value, mask = self.forward_tensors(x, mask, unrolls=unrolls)
        return policy, value, mask, maps


class AtUnrolls:
    """Same weights, fixed unroll count. Used for 1-pass vs 2-pass matches."""

    def __init__(self, adapter, unrolls):
        self.adapter = adapter
        self.unrolls = int(unrolls)

    def forward(self, fens, device, unrolls=None):
        return self.adapter.forward(fens, device, unrolls=self.unrolls if unrolls is None else unrolls)


def _policy_value(out):
    if isinstance(out, dict) and 'policy_logits' in out:
        return out['policy_logits'].float(), out['value_logits_q'].float()
    if hasattr(out, 'last_hidden_state') and hasattr(out, 'hidden_states'):
        return out.last_hidden_state[:, 0].float(), out.hidden_states[1][:, 0].float()
    raise TypeError(f'Unrecognized ChessBot output {type(out)}')


def compare_policies(actor, teacher, fens, device, batch=32):
    """Legal-argmax agreement and mean policy KL vs a frozen teacher."""
    agree = total = kl_sum = 0.0
    for start in range(0, len(fens), batch):
        fs = fens[start:start + batch]
        p, _, mask, _ = actor.forward(fs, device)
        with torch.no_grad():
            tp, _, tmask, _ = teacher.forward(fs, device)
        if not torch.equal(mask, tmask):
            raise ValueError('Legal masks differ')
        lp, lq = log_probs(p, mask, 1.), log_probs(tp, mask, 1.)
        agree += float((lp.argmax(-1) == lq.argmax(-1)).sum())
        total += len(fs)
        kl_sum += float(kl(lp, lq).sum())
    n = max(total, 1)
    return dict(legal_agreement=agree / n, policy_kl=kl_sum / n, n=int(total))


def supervised_distill(actor, teacher, fens, steps, batch, opt, device, rng, emit):
    """Match published one-pass policy + Q-WDL on stored FENs."""
    if steps <= 0 or not fens:
        return dict(steps=0)
    actor.model.train()
    records = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(fens), (min(batch, len(fens)),), generator=rng).tolist()
        fs = [fens[i] for i in idx]
        with torch.no_grad():
            tp, tq, mask, _ = teacher.forward(fs, device)
            tlp = log_probs(tp, mask, 1.)
            target = tq.softmax(-1)
        p, q, _, _ = actor.forward(fs, device)
        loss = kl(log_probs(p, mask, 1.), tlp).mean() + .1 * (-target * q.log_softmax(-1)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite supervised warmup loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.model.parameters(), .5, error_if_nonfinite=True)
        opt.step()
        gate = float(torch.tanh(actor.model.alpha)) if hasattr(actor.model, 'alpha') else None
        row = dict(stage='supervised', step=step, loss=float(loss.detach()),
                   grad_norm=float(norm), gate=gate)
        records.append(row)
        emit(row)
    actor.model.eval()
    return dict(steps=steps, last=records[-1] if records else None)


def _average_bank_grads(adapter):
    model = adapter.model
    unrolls = getattr(adapter, 'unrolls', 1)
    if unrolls <= 1 or not hasattr(model, 'recurrent_parameters'):
        return
    from chess_chessbot_recurrent import average_recurrent_grads
    average_recurrent_grads(model, unrolls)


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


def pack_fens(adapter, fens, device, chunk=256):
    """Board tensors + legal masks, built once."""
    if not fens:
        raise ValueError('No FENs to pack')
    xs, masks = [], []
    for start in range(0, len(fens), chunk):
        x, mask, _ = adapter.tensors(fens[start:start + chunk], device)
        xs.append(x)
        masks.append(mask)
    return torch.cat(xs), torch.cat(masks)


@torch.no_grad()
def frozen_outputs(adapter, x, mask, device, temperature=.8, chunk=128):
    """Cached policy / Q / legal log-probs for a frozen net."""
    ps, qs, lps = [], [], []
    for start in range(0, x.size(0), chunk):
        xb, mb = x[start:start + chunk], mask[start:start + chunk]
        p, q, _ = adapter.forward_tensors(xb, mb)
        ps.append(p)
        qs.append(q)
        lps.append(log_probs(p, mb, temperature))
    return torch.cat(ps), torch.cat(qs), torch.cat(lps)


def pack_rollout(adapter, rows, device, chunk=256):
    x, mask = pack_fens(adapter, [r['fen'] for r in rows], device, chunk)
    return dict(
        x=x, mask=mask,
        action=torch.tensor([r['action'] for r in rows], device=device),
        old_logp=torch.tensor([r['old_logp'] for r in rows], device=device, dtype=torch.float32),
        color=torch.tensor([r['color'] for r in rows], device=device),
        target=torch.tensor([r['target'] for r in rows], device=device, dtype=torch.float32),
    )


def should_full_audit(step, epoch_end, every):
    every = max(int(every), 1)
    return bool(epoch_end) or (step % every == 0)


def update(actor, control, reference, opt, control_opt, rows, anchors, cfg, device, rng, emit):
    adv = torch.tensor([r['advantage'] for r in rows], device=device, dtype=torch.float32)
    if adv.std(unbiased=False)>1e-8:
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    cache = bool(cfg.get('cache_rollout', True))
    micro = int(cfg.get('microbatch', 8))
    audit_mb = int(cfg.get('audit_microbatch', max(micro, 128)))
    prep = int(cfg.get('prep_chunk', 256))
    if cache:
        pack = pack_rollout(actor, rows, device, prep)
        _, _, ref_lp = frozen_outputs(reference, pack['x'], pack['mask'], device, .8, audit_mb)
        ax, am = pack_fens(actor, anchors, device, prep)
        _, aq, alp = frozen_outputs(reference, ax, am, device, 1., audit_mb)
        atarget = aq.softmax(-1)
    audit_ids = torch.randperm(len(rows),generator=rng)[:cfg['audit_size']].tolist()
    cheap_n = min(int(cfg.get('audit_cheap_size', 256)), len(audit_ids))
    cheap_ids = audit_ids[:cheap_n]
    full_every = int(cfg.get('audit_full_every', 8))
    records=[]; stopped=False
    n_mb = max((len(rows) + cfg['minibatch'] - 1) // cfg['minibatch'], 1)
    for epoch in range(cfg['epochs']):
        order=torch.randperm(len(rows),generator=rng).tolist()
        for mb_i, start in enumerate(range(0,len(rows),cfg['minibatch'])):
            ids=order[start:start+cfg['minibatch']]
            opt.zero_grad(set_to_none=True);control_opt.zero_grad(set_to_none=True)
            stats=dict(policy=0.,value=0.,reference_kl=0.,entropy=0.,clip_fraction=0.)
            for j in range(0,len(ids),micro):
                group=ids[j:j+micro]
                if cache:
                    xb, mb = pack['x'][group], pack['mask'][group]
                    p,q,_=actor.forward_tensors(xb, mb)
                    lp=log_probs(p, mb)
                    rlp=ref_lp[group]
                    actions, old = pack['action'][group], pack['old_logp'][group]
                    colors, tgt = pack['color'][group], pack['target'][group]
                else:
                    rs=[rows[k] for k in group];fs=[r['fen'] for r in rs]
                    p,q,mb,_=actor.forward(fs,device);lp=log_probs(p,mb)
                    with torch.no_grad():
                        rp,_,_,_=reference.forward(fs,device);rlp=log_probs(rp,mb)
                    actions=torch.tensor([r['action'] for r in rs],device=device)
                    old=torch.tensor([r['old_logp'] for r in rs],device=device)
                    colors=[r['color'] for r in rs]
                    tgt=torch.tensor([r['target'] for r in rs],device=device)
                ratio=(lp.gather(1,actions[:,None])[:,0]-old).exp()
                a=adv[group]
                pol=-torch.minimum(ratio*a,ratio.clamp(.9,1.1)*a).mean()
                value=actor_value(q, colors)
                vl=F.mse_loss(value, tgt)
                anchor=kl(lp,rlp).mean();entropy=-(lp.exp()*lp).sum(-1).mean()
                loss=pol+.5*vl+.05*anchor-.001*entropy
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                weight=len(group)/len(ids);(weight*loss).backward()
                if cache:
                    cp,_,_=control.forward_tensors(xb, mb)
                    (weight*.05*kl(log_probs(cp,mb),rlp).mean()).backward()
                else:
                    cp,_,_,_=control.forward(fs,device)
                    (weight*.05*kl(log_probs(cp,mb),rlp).mean()).backward()
                for k,v in [('policy',pol),('value',vl),('reference_kl',anchor),('entropy',entropy),
                            ('clip_fraction',((ratio-1).abs()>.1).float().mean())]:stats[k]+=weight*float(v.detach())
            anchor_ids=torch.randint(len(anchors),(cfg['anchor_batch'],),generator=rng).tolist()
            for j in range(0,len(anchor_ids),micro):
                g=anchor_ids[j:j+micro]
                if cache:
                    xb, mb, rlp, target = ax[g], am[g], alp[g], atarget[g]
                    for model in [actor,control]:
                        p,q,_=model.forward_tensors(xb, mb)
                        loss=.05*kl(log_probs(p,mb),rlp).mean()+.1*(-target*q.log_softmax(-1)).sum(-1).mean()
                        (len(g)/len(anchor_ids)*loss).backward()
                else:
                    fs=[anchors[k] for k in g]
                    with torch.no_grad():
                        rp,rq,mb,_=reference.forward(fs,device);rlp=log_probs(rp,mb,1.);target=rq.softmax(-1)
                    for model in [actor,control]:
                        p,q,_,_=model.forward(fs,device)
                        loss=.05*kl(log_probs(p,mb),rlp).mean()+.1*(-target*q.log_softmax(-1)).sum(-1).mean()
                        (len(fs)/len(anchor_ids)*loss).backward()
            _average_bank_grads(actor);_average_bank_grads(control)
            if hasattr(actor.model, 'alpha') and actor.model.alpha.grad is not None:
                stats['gate'] = float(torch.tanh(actor.model.alpha).detach())
                stats['gate_grad'] = float(actor.model.alpha.grad.detach().reshape(-1)[0])
            norm=torch.nn.utils.clip_grad_norm_(actor.model.parameters(),.5,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(control.model.parameters(),.5,error_if_nonfinite=True)
            opt.step();control_opt.step()
            step=len(records)+1
            epoch_end = mb_i + 1 == n_mb
            use_full = should_full_audit(step, epoch_end, full_every)
            probe = audit_ids if use_full else cheap_ids
            audit=[]
            with torch.no_grad():
                for j in range(0,len(probe),audit_mb if cache else micro):
                    g=probe[j:j+ (audit_mb if cache else micro)]
                    if cache:
                        p,_,mb=actor.forward_tensors(pack['x'][g], pack['mask'][g])
                        lp=log_probs(p, mb)
                        d=lp.gather(1,pack['action'][g][:,None])[:,0]-pack['old_logp'][g]
                    else:
                        rs=[rows[k] for k in g]
                        p,_,m,_=actor.forward([r['fen'] for r in rs],device);lp=log_probs(p,m)
                        acts=torch.tensor([r['action'] for r in rs],device=device)
                        d=lp.gather(1,acts[:,None])[:,0]-torch.tensor([r['old_logp'] for r in rs],device=device)
                    audit.extend(((d.exp()-1)-d).detach().cpu().tolist())
            stats.update(epoch=epoch,grad_norm=float(norm),behavior_kl=sum(audit)/len(audit),
                         update=step,audit_n=len(probe),audit_full=use_full)
            records.append(stats);emit(dict(stage='update',**stats))
            if stats['behavior_kl']>.01:
                stopped=True;break
        if stopped:break
    return dict(updates=len(records),kl_stopped=stopped,minibatches=records)


@torch.no_grad()
def top_moves(adapter, fen, device, k=5, unrolls=None):
    p, q, mask, maps = adapter.forward([fen], device, unrolls=unrolls)
    lp = log_probs(p, mask, 1.)[0]
    board = chess.Board(fen)
    value = float(actor_value(q, [board.turn])[0])
    wdl = q[0].softmax(-1).tolist()
    legal = int(mask[0].sum())
    idx = torch.topk(lp.exp(), min(k, legal)).indices.tolist()
    moves = []
    for i in idx:
        mv = maps[0].get(i)
        moves.append(dict(uci=None if mv is None else mv.uci(), p=float(lp.exp()[i]), idx=int(i)))
    return dict(fen=fen, value=value, wdl=wdl, top=moves,
                preferred=None if not moves else moves[0]['uci'])


@torch.no_grad()
def recurrence_diagnostics(adapter, fens, device, batch=16):
    model = adapter.model
    if not hasattr(model, 'extra_gate') or not fens:
        return dict(gate=None, extra_hidden_rms=0., move_changes=0, n=0, changed=[],
                    latency_n1_ms=None, latency_n2_ms=None)
    gate = float(model.extra_gate())
    deltas, changed = [], []
    t1 = t2 = 0.
    n_lat = min(8, len(fens))
    for start in range(0, len(fens), batch):
        fs = fens[start:start + batch]
        if start < n_lat:
            t0 = time.perf_counter()
        p1, _, mask, maps = adapter.forward(fs, device, unrolls=1)
        if start < n_lat:
            t1 += time.perf_counter() - t0
            t0 = time.perf_counter()
        p2, _, _, _ = adapter.forward(fs, device, unrolls=adapter.unrolls)
        if start < n_lat:
            t2 += time.perf_counter() - t0
        a1 = log_probs(p1, mask, 1.).argmax(-1)
        a2 = log_probs(p2, mask, 1.).argmax(-1)
        for i, fen in enumerate(fs):
            if int(a1[i]) == int(a2[i]):
                continue
            m1, m2 = maps[i].get(int(a1[i])), maps[i].get(int(a2[i]))
            changed.append(dict(fen=fen,
                                n1=None if m1 is None else m1.uci(),
                                n2=None if m2 is None else m2.uci()))
        x, _, _ = adapter.tensors(fs, device)
        h, pos, _, _ = model._encode(x)
        for layer in model.base.layers[:model.split.prefix]:
            h = layer(h, pos)
        h = model._bank(h, pos)
        nxt = model._bank(h, pos)
        deltas.append(float((nxt - h).pow(2).mean().sqrt()))
    n = len(fens)
    lat_n = max(min(n_lat, n), 1)
    return dict(
        gate=gate, extra_hidden_rms=sum(deltas) / max(len(deltas), 1),
        move_changes=len(changed), n=n, changed=changed[:32],
        change_rate=len(changed) / n,
        latency_n1_ms=1000. * t1 / lat_n, latency_n2_ms=1000. * t2 / lat_n,
        extra_latency_ms=1000. * (t2 - t1) / lat_n,
    )


@torch.no_grad()
def inspect_boards(published, incumbent, candidate, fens, device):
    rows = []
    for fen in fens:
        pub = top_moves(published, fen, device, unrolls=1)
        inc = top_moves(incumbent, fen, device) if incumbent is not None else None
        n1 = top_moves(candidate, fen, device, unrolls=1) if hasattr(candidate.model, 'default_unrolls') else pub
        n2 = top_moves(candidate, fen, device)
        pref = {
            'published': pub['preferred'],
            'incumbent': None if inc is None else inc['preferred'],
            'n1': n1['preferred'],
            'n2': n2['preferred'],
        }
        rows.append(dict(
            fen=fen, published=pub, incumbent=inc, n1=n1, n2=n2, preferred=pref,
            n2_changed_from_published=pref['n2'] != pref['published'],
            n2_changed_from_n1=pref['n2'] != pref['n1'],
        ))
    return rows
