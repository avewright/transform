"""Stratified validation mix for Value99.

The hash split stays 99/0.5/0.5 and never moves a train row into val.
This module only rebalances *which validation rows* are scored so the
loop is not dominated by the source prior (the ChessFENS pilot was
endgame-heavy and almost all White-to-move).

Phase is the existing coarse proxy: opening / middlegame / low-material.
Outcome is |P(win)-P(loss)|: drawish <0.2, moderate <0.6, decisive otherwise.
A playing-like mix uses more middlegames than endgames, and keeps enough
drawish and decisive positions that a single easy slice cannot hide error.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np

PHASE_NAMES=('opening','middlegame','endgame')
OUTCOME_NAMES=('drawish','moderate','decisive')
# Playing-like diagnostic prior, not the ChessFENS dump prior.
PHASE_WEIGHTS=(0.25,0.45,0.30)
OUTCOME_WEIGHTS=(0.30,0.40,0.30)
DRAWISH=0.2
DECISIVE=0.6


def outcome_bucket(target):
    magnitude=abs(float(target))
    if magnitude<DRAWISH:return 0
    if magnitude<DECISIVE:return 1
    return 2


def cell_quotas(n,phase_weights=PHASE_WEIGHTS,outcome_weights=OUTCOME_WEIGHTS):
    if n<1:raise ValueError('val mix size must be positive')
    raw=[(p,o,phase_weights[p]*outcome_weights[o]*n)
         for p in range(3) for o in range(3)]
    quotas={(p,o):int(math.floor(weight)) for p,o,weight in raw}
    leftover=n-sum(quotas.values())
    for p,o,_ in sorted(raw,key=lambda item:item[2]-math.floor(item[2]),reverse=True)[:leftover]:
        quotas[(p,o)]+=1
    return quotas


def _take(candidates,k,rng,black=None):
    if k<=0 or not candidates:return []
    order=candidates[:]
    rng.shuffle(order)
    if black is None or k>=len(order):return order[:k]
    white=[i for i in order if not black[i]]
    dark=[i for i in order if black[i]]
    want_black=min(len(dark),k//2)
    want_white=min(len(white),k-want_black)
    want_black=min(len(dark),k-want_white)
    picked=white[:want_white]+dark[:want_black]
    if len(picked)<k:
        used=set(picked)
        picked.extend(i for i in order if i not in used)
    return picked[:k]


def build_valmix(targets,phases,n,seed=294,black=None,phase_weights=PHASE_WEIGHTS,
                 outcome_weights=OUTCOME_WEIGHTS):
    """Return a deterministic stratified subset of validation indices."""
    targets=np.asarray(targets);phases=np.asarray(phases,dtype=np.int64)
    if len(targets)!=len(phases):raise ValueError('target/phase length mismatch')
    if black is not None:
        black=np.asarray(black).astype(bool)
        if len(black)!=len(targets):raise ValueError('black length mismatch')
    quotas=cell_quotas(min(n,len(targets)),phase_weights,outcome_weights)
    buckets=defaultdict(list)
    for i,(target,phase) in enumerate(zip(targets.tolist(),phases.tolist())):
        if phase not in (0,1,2):continue
        buckets[(int(phase),outcome_bucket(target))].append(i)
    rng=np.random.RandomState(seed)
    chosen=[];filled={};shortfall={}
    for cell,need in quotas.items():
        picked=_take(buckets[cell],need,rng,black)
        chosen.extend(picked);filled[cell]=len(picked)
        shortfall[cell]=max(0,need-len(picked))
        taken=set(picked)
        buckets[cell]=[i for i in buckets[cell] if i not in taken]
    missing=min(n,len(targets))-len(chosen)
    if missing:
        # Same phase first, then any remaining validation row. Never invent rows.
        leftovers=[]
        for phase in range(3):
            for outcome in range(3):
                leftovers.extend(buckets[(phase,outcome)])
        chosen.extend(_take(leftovers,missing,rng,black))
    chosen=np.asarray(chosen[:min(n,len(targets))],dtype=np.int64)
    counts=defaultdict(int)
    for i in chosen.tolist():
        counts[(int(phases[i]),outcome_bucket(targets[i]))]+=1
    return dict(indices=chosen,quotas={f'{PHASE_NAMES[p]}_{OUTCOME_NAMES[o]}':q for (p,o),q in quotas.items()},
                filled={f'{PHASE_NAMES[p]}_{OUTCOME_NAMES[o]}':int(counts[(p,o)]) for p in range(3) for o in range(3)},
                shortfall={f'{PHASE_NAMES[p]}_{OUTCOME_NAMES[o]}':int(shortfall[(p,o)]) for p in range(3) for o in range(3)},
                n=int(len(chosen)),seed=int(seed),
                phase_weights=list(phase_weights),outcome_weights=list(outcome_weights),
                black_source=int(black[chosen].sum()) if black is not None else None)


def cell_counts(targets,phases):
    counts=defaultdict(int)
    for target,phase in zip(np.asarray(targets).tolist(),np.asarray(phases).tolist()):
        if phase in (0,1,2):counts[(int(phase),outcome_bucket(target))]+=1
    return counts


def cells_needed(n,margin=1.5):
    return {cell:max(1,int(math.ceil(q*margin))) for cell,q in cell_quotas(n).items()}


def write_valmix(path,mix):
    path=Path(path)
    np.save(path/'valmix_indices.npy',mix['indices'])
    meta={k:v for k,v in mix.items() if k!='indices'}
    meta['index_file']='valmix_indices.npy'
    (path/'valmix.json').write_text(json.dumps(meta,indent=2))
    return meta


def load_valmix(path):
    path=Path(path)
    if not (path/'valmix.json').exists() or not (path/'valmix_indices.npy').exists():
        return None
    meta=json.loads((path/'valmix.json').read_text())
    meta['indices']=np.load(path/'valmix_indices.npy')
    return meta
