#!/usr/bin/env python3
"""Score a fixed one-ply suite. Value loss is not Elo."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import torch
from chess_value99 import ValueConfig,ValueTransformer,score_legal_moves

SUITE=(
    dict(id='mate_kq_k',fen='7k/5Q2/6K1/8/8/8/8/8 w - - 0 1',best='f7g7'),
    dict(id='scholar',fen='r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4',best='h5f7'),
    dict(id='hanging_queen',fen='rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',best=None),
    dict(id='start',fen=chess.STARTING_FEN,best=None),
)


def load_model(path,device):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    model=ValueTransformer(ValueConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model']);model.eval()
    return model,int(ckpt.get('step',-1))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu')
    model,step=load_model(a.checkpoint,device)
    rows=[]
    for item in SUITE:
        board=chess.Board(item['fen'])
        scores=score_legal_moves(model,board,device=str(device))
        ranked=sorted(scores,key=scores.get,reverse=True)
        top=ranked[0]
        rows.append(dict(id=item['id'],fen=item['fen'],best=item['best'],choice=top,
                         correct=(item['best'] is None or top==item['best']),
                         top_score=scores[top],
                         top5=[{'uci':u,'score':scores[u]} for u in ranked[:5]]))
    report=dict(step=step,checkpoint=str(a.checkpoint),
                known_best_accuracy=sum(r['correct'] for r in rows if r['best'])/max(1,sum(1 for r in rows if r['best'])),
                positions=rows,note='One-ply argmax is not an Elo estimate.')
    text=json.dumps(report,indent=2)
    print(text)
    if a.out:a.out.write_text(text)


if __name__=='__main__':main()
