#!/usr/bin/env python3
"""Value99 one-ply Elo gauntlet vs Stockfish UCI_Elo.

Move choice: score every legal successor as -V(child) (side-to-move WDL)
and play the argmax. No policy head, no search, no book, no Syzygy.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.engine
import torch
from chess_value99 import ValueConfig,ValueTransformer,score_legal_moves
from harness.common import load_protocol,opening_name,resolve_stockfish,stockfish_version
from harness.elo import estimate_elo,summarize_results,wilson_interval


def load_value99(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    arch=ckpt.get('arch') or ''
    if arch=='value99_policy' or 'vocab_size' in (ckpt.get('config') or {}):
        from chess_value99_policy import build_value99_policy
        model=build_value99_policy(ckpt['config']).to(device)
        model.load_state_dict(ckpt['model']);model.eval()
        return model,int(ckpt.get('step',-1)),'policy'
    model=ValueTransformer(ValueConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model']);model.eval()
    return model,int(ckpt.get('step',-1)),'value'


def choose_move(model,board,device,mode='value',batch_size=64):
    if mode=='policy':
        move,info=model.select_move(board,device,0.0)
        return move,info.get('uci')
    scores=score_legal_moves(model,board,device=str(device),batch_size=batch_size)
    uci=max(scores,key=scores.get)
    return chess.Move.from_uci(uci),float(scores[uci])


def play_game(engine,model,device,sf_elo,model_color,opening,movetime,ply_cap,nodes=0,mode='value'):
    board=chess.Board()
    for uci in opening:
        m=chess.Move.from_uci(uci)
        if m in board.legal_moves:board.push(m)
    last_score=None
    while not board.is_game_over(claim_draw=True) and len(board.move_stack)<ply_cap:
        if board.turn==model_color:
            move,last_score=choose_move(model,board,device,mode)
            source='policy' if mode=='policy' else 'oneply'
        else:
            limit=chess.engine.Limit(nodes=int(nodes)) if int(nodes)>0 else chess.engine.Limit(time=movetime)
            move=engine.play(board,limit).move;source='sf'
        if move not in board.legal_moves:
            move=next(iter(board.legal_moves));source='fallback'
        board.push(move)
    outcome=board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:score=0.5
    elif outcome.winner==model_color:score=1.0
    else:score=0.0
    return dict(sf_elo=sf_elo,model_color='white' if model_color==chess.WHITE else 'black',
                opening=opening,opening_name=opening_name(opening),
                result=board.result(claim_draw=True),score=score,plies=len(board.move_stack),
                termination=outcome.termination.name if outcome else 'PLY_CAP',
                final_fen=board.fen(),last_value=last_score)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'outputs'/'value99_elo')
    p.add_argument('--device',default='cuda')
    p.add_argument('--games-per-opening-per-color',type=int,default=1)
    p.add_argument('--elos',type=int,nargs='+',default=[1320,1450,1600,1750,1900])
    p.add_argument('--movetime',type=float,default=0.05)
    p.add_argument('--ply-cap',type=int,default=160)
    p.add_argument('--nodes',type=int,default=0)
    a=p.parse_args()
    proto=load_protocol();out=a.out;out.mkdir(parents=True,exist_ok=True)
    device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu')
    frozen=out/'ckpt.pt'
    if Path(a.ckpt).resolve()!=frozen.resolve():shutil.copy2(a.ckpt,frozen)
    model,step,mode=load_value99(frozen,device)
    sf=resolve_stockfish();ver=stockfish_version(sf)
    openings=[list(o) for o in proto['openings']]
    log=out/'elo.log';events=out/'elo.jsonl'
    def emit(d):
        d=dict(time=time.time(),**d);print(json.dumps(d),flush=True)
        events.open('a').write(json.dumps(d)+'\n')
        (out/'status.json').write_text(json.dumps(d,indent=2))
    emit(dict(stage='start',step=step,ckpt=str(a.ckpt),device=str(device),
              stockfish=str(sf),sf_version=ver,
              move='argmax legal ChessBot logit' if mode=='policy' else 'argmax -V(successor)',
              mode=mode,elos=a.elos,games_per_opening_per_color=a.games_per_opening_per_color))
    summaries=[];all_games=[];estimate={}
    def dump():
        payload=dict(checkpoint=str(a.ckpt),step=step,device=str(device),
                     mode='value99_policy' if mode=='policy' else 'value99_oneply',
                     summaries=summaries,games=all_games,estimate=estimate)
        (out/'elo.json').write_text(json.dumps(payload,indent=2))
    for elo in a.elos:
        emit(dict(stage='begin',sf_elo=elo))
        engine=chess.engine.SimpleEngine.popen_uci(str(sf))
        engine.configure({'UCI_LimitStrength':True,'UCI_Elo':elo,'Threads':1,'Hash':32})
        results=[]
        try:
            for opening in openings:
                for color in (chess.WHITE,chess.BLACK):
                    for repeat in range(a.games_per_opening_per_color):
                        r=play_game(engine,model,device,elo,color,opening,a.movetime,a.ply_cap,a.nodes,mode)
                        r['repeat_idx']=repeat;results.append(r);all_games.append(r)
                        emit(dict(stage='game',sf_elo=elo,color=r['model_color'],
                                  opening=r['opening_name'],result=r['result'],
                                  score=r['score'],plies=r['plies'],termination=r['termination']))
        finally:
            engine.quit()
        summary=summarize_results(elo,results);summaries.append(summary)
        estimate=estimate_elo(summaries);dump()
        emit(dict(stage='summary',**summary,estimate=estimate))
        if estimate.get('lower_bound') is not None and estimate.get('upper_bound') is not None:
            emit(dict(stage='bracketed',**estimate));break
    emit(dict(stage='done',estimate=estimate))


if __name__=='__main__':main()
